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

namespace ops_hccl {

namespace ccu = ::AscendC::ccu;

namespace {

constexpr uint32_t REMOTE_OUTPUT_VAR = 0;
constexpr uint32_t REMOTE_TOKEN_VAR = 1;
constexpr uint32_t NOTIFY_INDEX = 0;
constexpr uint32_t REDUCE_PHASE_VALUE = 0;

constexpr uint16_t OUTPUT_ADDR_BIT = 1U << 0;
constexpr uint16_t OUTPUT_TOKEN_BIT = 1U << 1;
constexpr uint16_t REDUCE_GO_BIT = 1U << 2;
constexpr uint16_t REDUCE_DONE_BIT = 1U << 3;
constexpr uint16_t RESULT_READY_BIT = 1U << 4;
constexpr uint16_t RESULT_ACK_BIT = 1U << 5;

#define RETURN_IF_CCU_ERROR(call) \
    do { \
        CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

struct TreeAllReduceContext {
    const TreeAllReduceKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable dataSize;
    ccu::Variable dataOffset;
    ccu::Variable phase;
    std::vector<ccu::Variable> remoteOutput;
    std::vector<ccu::Variable> remoteOutputToken;
    ccu::Event event;
};

CcuResult InitResources(TreeAllReduceContext &ctx)
{
    const TreeAllReduceKernelArg *arg = ctx.arg;
    if (arg->channelCount == 0 || arg->channelCount >= MAX_RANK_SIZE ||
        (!arg->isParent && arg->channelCount != 1)) {
        HCCL_ERROR("Invalid tree channel count, rank[%u] channelCount[%u] isParent[%d]",
            arg->rankId, arg->channelCount, arg->isParent);
        return CCU_E_PARA;
    }

    ctx.remoteOutput.resize(arg->channelCount);
    ctx.remoteOutputToken.resize(arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        ctx.remoteOutput[channelIdx]
            = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], REMOTE_OUTPUT_VAR);
        ctx.remoteOutputToken[channelIdx]
            = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], REMOTE_TOKEN_VAR);
    }
    return CCU_SUCCESS;
}

CcuResult LoadTaskArgs(TreeAllReduceContext &ctx)
{
    uint32_t argId = 0;
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.input, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.output, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.dataSize, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.dataOffset, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.phase, argId++));
    return CCU_SUCCESS;
}

CcuResult ExchangeOutputAddress(TreeAllReduceContext &ctx)
{
    const TreeAllReduceKernelArg *arg = ctx.arg;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
            arg->channels[channelIdx], ctx.output, REMOTE_OUTPUT_VAR, NOTIFY_INDEX, OUTPUT_ADDR_BIT));
        RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
            arg->channels[channelIdx], ctx.outputToken, REMOTE_TOKEN_VAR, NOTIFY_INDEX, OUTPUT_TOKEN_BIT));
    }

    constexpr uint16_t addressMask = OUTPUT_ADDR_BIT | OUTPUT_TOKEN_BIT;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(ccu::NotifyWait(arg->channels[channelIdx], NOTIFY_INDEX, addressMask));
    }
    return CCU_SUCCESS;
}

CcuResult RunParentReduce(TreeAllReduceContext &ctx)
{
    const TreeAllReduceKernelArg *arg = ctx.arg;
    // 同一目标地址只允许一个 WriteReduce 在途，保证 FP32 SUM 的顺序稳定。
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(
            ccu::NotifyRecord(arg->channels[channelIdx], NOTIFY_INDEX, REDUCE_GO_BIT));
        RETURN_IF_CCU_ERROR(
            ccu::NotifyWait(arg->channels[channelIdx], NOTIFY_INDEX, REDUCE_DONE_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult RunChildReduce(TreeAllReduceContext &ctx)
{
    const TreeAllReduceKernelArg *arg = ctx.arg;
    RETURN_IF_CCU_ERROR(ccu::NotifyWait(arg->channels[0], NOTIFY_INDEX, REDUCE_GO_BIT));

    // output 在 Host 侧已由 input 初始化；高层 reduce 时它还承载低层聚合结果。
    ccu::LocalAddr localInput;
    localInput.addr = ctx.output;
    localInput.addr += ctx.dataOffset;
    localInput.token = ctx.outputToken;

    ccu::RemoteAddr rootOutput;
    rootOutput.addr = ctx.remoteOutput[0];
    rootOutput.addr += ctx.dataOffset;
    rootOutput.token = ctx.remoteOutputToken[0];

    RETURN_IF_CCU_ERROR(ccu::WriteReduce(arg->channels[0], rootOutput, localInput, ctx.dataSize,
        arg->dataType, arg->reduceOp, ctx.event, 1));
    RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, 1));
    RETURN_IF_CCU_ERROR(ccu::NotifyRecord(arg->channels[0], NOTIFY_INDEX, REDUCE_DONE_BIT));
    return CCU_SUCCESS;
}

CcuResult RunParentBroadcast(TreeAllReduceContext &ctx)
{
    const TreeAllReduceKernelArg *arg = ctx.arg;
    ccu::LocalAddr localOutput;
    localOutput.addr = ctx.output;
    localOutput.addr += ctx.dataOffset;
    localOutput.token = ctx.outputToken;

    uint16_t eventMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        ccu::RemoteAddr peerOutput;
        peerOutput.addr = ctx.remoteOutput[channelIdx];
        peerOutput.addr += ctx.dataOffset;
        peerOutput.token = ctx.remoteOutputToken[channelIdx];
        RETURN_IF_CCU_ERROR(
            ccu::Write(arg->channels[channelIdx], peerOutput, localOutput, ctx.dataSize, ctx.event, eventBit));
        eventMask = static_cast<uint16_t>(eventMask | eventBit);
    }
    RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, eventMask));

    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(
            ccu::NotifyRecord(arg->channels[channelIdx], NOTIFY_INDEX, RESULT_READY_BIT));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(
            ccu::NotifyWait(arg->channels[channelIdx], NOTIFY_INDEX, RESULT_ACK_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult RunChildBroadcast(TreeAllReduceContext &ctx)
{
    const TreeAllReduceKernelArg *arg = ctx.arg;
    RETURN_IF_CCU_ERROR(ccu::NotifyWait(arg->channels[0], NOTIFY_INDEX, RESULT_READY_BIT));
    RETURN_IF_CCU_ERROR(ccu::NotifyRecord(arg->channels[0], NOTIFY_INDEX, RESULT_ACK_BIT));
    return CCU_SUCCESS;
}

constexpr uint32_t LOCAL_REMOTE_INPUT_VAR = 0;
constexpr uint32_t LOCAL_REMOTE_OUTPUT_VAR = 1;
constexpr uint32_t LOCAL_REMOTE_INPUT_TOKEN_VAR = 2;
constexpr uint32_t LOCAL_REMOTE_OUTPUT_TOKEN_VAR = 3;
constexpr uint32_t LOCAL_NOTIFY_INDEX = 0;
constexpr uint16_t LOCAL_ADDR_BIT = 1U << 0;
constexpr uint16_t LOCAL_TOKEN_BIT = 1U << 1;
constexpr uint16_t LOCAL_DONE_BIT = 1U << 2;
constexpr uint32_t LOCAL_EXCHANGE_INPUT = 0;
constexpr uint32_t LOCAL_REDUCE = 1;
constexpr uint32_t LOCAL_EXCHANGE_OUTPUT = 2;
constexpr uint32_t LOCAL_GATHER = 3;
constexpr uint32_t LOCAL_BARRIER = 4;

struct LocalMeshContext {
    const LocalMeshKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable dataSize;
    ccu::Variable dataOffset;
    ccu::Variable mode;
    std::vector<ccu::Variable> remoteInput;
    std::vector<ccu::Variable> remoteInputToken;
    std::vector<ccu::Variable> remoteOutput;
    std::vector<ccu::Variable> remoteOutputToken;
    ccu::Event event;
};

CcuResult InitLocalResources(LocalMeshContext &ctx)
{
    const LocalMeshKernelArg *arg = ctx.arg;
    if (arg->channelCount == 0 || arg->channelCount >= MAX_RANK_SIZE) {
        HCCL_ERROR("Invalid local Mesh channel count, rank[%u] channelCount[%u]",
            arg->rankId, arg->channelCount);
        return CCU_E_PARA;
    }

    ctx.remoteInput.resize(arg->channelCount);
    ctx.remoteInputToken.resize(arg->channelCount);
    ctx.remoteOutput.resize(arg->channelCount);
    ctx.remoteOutputToken.resize(arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        ctx.remoteInput[channelIdx]
            = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], LOCAL_REMOTE_INPUT_VAR);
        ctx.remoteInputToken[channelIdx] = ccu::GetResByChannel<ccu::Variable>(
            arg->channels[channelIdx], LOCAL_REMOTE_INPUT_TOKEN_VAR);
        ctx.remoteOutput[channelIdx]
            = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], LOCAL_REMOTE_OUTPUT_VAR);
        ctx.remoteOutputToken[channelIdx] = ccu::GetResByChannel<ccu::Variable>(
            arg->channels[channelIdx], LOCAL_REMOTE_OUTPUT_TOKEN_VAR);
    }
    return CCU_SUCCESS;
}

CcuResult LoadLocalTaskArgs(LocalMeshContext &ctx)
{
    uint32_t argId = 0;
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.input, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.output, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.scratch, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.scratchToken, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.dataSize, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.dataOffset, argId++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.mode, argId++));
    return CCU_SUCCESS;
}

CcuResult ExchangeLocalInput(LocalMeshContext &ctx)
{
    const LocalMeshKernelArg *arg = ctx.arg;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(arg->channels[channelIdx],
            ctx.input, LOCAL_REMOTE_INPUT_VAR, LOCAL_NOTIFY_INDEX, LOCAL_ADDR_BIT));
        RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(arg->channels[channelIdx],
            ctx.inputToken, LOCAL_REMOTE_INPUT_TOKEN_VAR, LOCAL_NOTIFY_INDEX, LOCAL_TOKEN_BIT));
    }

    constexpr uint16_t addressMask = LOCAL_ADDR_BIT | LOCAL_TOKEN_BIT;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(
            ccu::NotifyWait(arg->channels[channelIdx], LOCAL_NOTIFY_INDEX, addressMask));
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeLocalOutput(LocalMeshContext &ctx)
{
    const LocalMeshKernelArg *arg = ctx.arg;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(arg->channels[channelIdx],
            ctx.output, LOCAL_REMOTE_OUTPUT_VAR, LOCAL_NOTIFY_INDEX, LOCAL_ADDR_BIT));
        RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(arg->channels[channelIdx],
            ctx.outputToken, LOCAL_REMOTE_OUTPUT_TOKEN_VAR, LOCAL_NOTIFY_INDEX, LOCAL_TOKEN_BIT));
    }

    constexpr uint16_t addressMask = LOCAL_ADDR_BIT | LOCAL_TOKEN_BIT;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(
            ccu::NotifyWait(arg->channels[channelIdx], LOCAL_NOTIFY_INDEX, addressMask));
    }
    return CCU_SUCCESS;
}

CcuResult RunLocalReduceScatter(LocalMeshContext &ctx)
{
    const LocalMeshKernelArg *arg = ctx.arg;
    ccu::LocalAddr reduceOutput;
    reduceOutput.addr = ctx.output;
    reduceOutput.addr += ctx.dataOffset;
    reduceOutput.token = ctx.outputToken;

    std::vector<ccu::LocalAddr> peerScratch(arg->channelCount);
    ccu::LocalAddr nextScratch;
    nextScratch.addr = ctx.scratch;
    nextScratch.token = ctx.scratchToken;

    uint16_t readMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        ccu::RemoteAddr peerInput;
        peerInput.addr = ctx.remoteInput[channelIdx];
        peerInput.addr += ctx.dataOffset;
        peerInput.token = ctx.remoteInputToken[channelIdx];

        peerScratch[channelIdx] = nextScratch;
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        RETURN_IF_CCU_ERROR(ccu::Read(
            arg->channels[channelIdx], peerScratch[channelIdx], peerInput,
            ctx.dataSize, ctx.event, eventBit));
        readMask = static_cast<uint16_t>(readMask | eventBit);
        nextScratch.addr += ctx.dataSize;
    }
    RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, readMask));

    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(ccu::LocalReduce(
            reduceOutput, peerScratch[channelIdx], ctx.dataSize,
            arg->dataType, arg->reduceOp, ctx.event, 1));
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, 1));
    }
    return CCU_SUCCESS;
}

CcuResult RunLocalAllGather(LocalMeshContext &ctx)
{
    const LocalMeshKernelArg *arg = ctx.arg;
    ccu::LocalAddr localResult;
    localResult.addr = ctx.output;
    localResult.addr += ctx.dataOffset;
    localResult.token = ctx.outputToken;

    uint16_t writeMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        ccu::RemoteAddr peerOutput;
        peerOutput.addr = ctx.remoteOutput[channelIdx];
        peerOutput.addr += ctx.dataOffset;
        peerOutput.token = ctx.remoteOutputToken[channelIdx];

        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIdx);
        RETURN_IF_CCU_ERROR(ccu::Write(
            arg->channels[channelIdx], peerOutput, localResult,
            ctx.dataSize, ctx.event, eventBit));
        writeMask = static_cast<uint16_t>(writeMask | eventBit);
    }
    RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, writeMask));
    return CCU_SUCCESS;
}

CcuResult SyncLocalPeers(LocalMeshContext &ctx)
{
    const LocalMeshKernelArg *arg = ctx.arg;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(
            ccu::NotifyRecord(arg->channels[channelIdx], LOCAL_NOTIFY_INDEX, LOCAL_DONE_BIT));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        RETURN_IF_CCU_ERROR(
            ccu::NotifyWait(arg->channels[channelIdx], LOCAL_NOTIFY_INDEX, LOCAL_DONE_BIT));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuTreeAllReduceKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<TreeAllReduceKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize < 2 || kernelArg->rankSize > MAX_RANK_SIZE ||
        kernelArg->rankId >= kernelArg->rankSize) {
        return CCU_E_PARA;
    }

    TreeAllReduceContext ctx;
    ctx.arg = kernelArg;
    RETURN_IF_CCU_ERROR(InitResources(ctx));
    RETURN_IF_CCU_ERROR(LoadTaskArgs(ctx));
    RETURN_IF_CCU_ERROR(ExchangeOutputAddress(ctx));

    CCU_IF(ctx.phase == REDUCE_PHASE_VALUE)
    {
        RETURN_IF_CCU_ERROR(
            kernelArg->isParent ? RunParentReduce(ctx) : RunChildReduce(ctx));
    }
    CCU_ELSE
    {
        RETURN_IF_CCU_ERROR(
            kernelArg->isParent ? RunParentBroadcast(ctx) : RunChildBroadcast(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuLocalMeshKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<LocalMeshKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize < 2 || kernelArg->rankSize > MAX_RANK_SIZE ||
        kernelArg->rankId >= kernelArg->rankSize) {
        return CCU_E_PARA;
    }

    LocalMeshContext ctx;
    ctx.arg = kernelArg;
    RETURN_IF_CCU_ERROR(InitLocalResources(ctx));
    RETURN_IF_CCU_ERROR(LoadLocalTaskArgs(ctx));

    CCU_IF(ctx.mode == LOCAL_EXCHANGE_INPUT)
    {
        RETURN_IF_CCU_ERROR(ExchangeLocalInput(ctx));
    }
    CCU_IF(ctx.mode == LOCAL_REDUCE)
    {
        RETURN_IF_CCU_ERROR(RunLocalReduceScatter(ctx));
    }
    CCU_IF(ctx.mode == LOCAL_EXCHANGE_OUTPUT)
    {
        RETURN_IF_CCU_ERROR(ExchangeLocalOutput(ctx));
    }
    CCU_IF(ctx.mode == LOCAL_GATHER)
    {
        RETURN_IF_CCU_ERROR(RunLocalAllGather(ctx));
    }
    CCU_IF(ctx.mode == LOCAL_BARRIER)
    {
        RETURN_IF_CCU_ERROR(SyncLocalPeers(ctx));
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
