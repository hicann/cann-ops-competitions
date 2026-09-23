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

#include "custom.h"
#include "ccu_kernel.h"

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

constexpr uint32_t BUFFER_XN_ID = 0;
constexpr uint32_t TOKEN_XN_ID = 1;
constexpr uint32_t STAGING_XN_ID = 2;       // 中继：relay 中转缓冲地址
constexpr uint32_t STAGING_TOKEN_XN_ID = 3; // 中继：relay 中转缓冲 token
constexpr uint32_t PRE_SYNC_NOTIFY_IDX = 0;
constexpr uint32_t POST_SYNC_NOTIFY_IDX = 1;
constexpr uint32_t RELAY_NOTIFY_IDX = 2;    // 中继：root 写 payload 就绪通知
constexpr uint16_t BUFFER_READY_MASK = 1U << BUFFER_XN_ID;
constexpr uint16_t TOKEN_READY_MASK = 1U << TOKEN_XN_ID;
constexpr uint16_t STAGING_READY_MASK = 1U << STAGING_XN_ID;
constexpr uint16_t STAGING_TOKEN_READY_MASK = 1U << STAGING_TOKEN_XN_ID;
constexpr uint16_t ALL_READY_MASK = BUFFER_READY_MASK | TOKEN_READY_MASK;
constexpr uint16_t RELAY_ALL_READY_MASK = ALL_READY_MASK | STAGING_READY_MASK | STAGING_TOKEN_READY_MASK;
constexpr uint16_t DONE_MASK = 1U;

#define CCU_CHECK_RET(call) \
    do { \
        const CcuResult ccuRet = static_cast<CcuResult>(call); \
        if (ccuRet != CCU_SUCCESS) { \
            return ccuRet; \
        } \
    } while (0)

// ==================== 直发星形 Kernel（小消息） ====================
struct ScatterContext {
    CcuKernelArgScatter *arg;
    std::vector<ccu::Variable> remoteBuffer;
    std::vector<ccu::Variable> remoteToken;
    std::vector<ccu::Variable> remoteStaging;      // 中继：对端(relay) 中转缓冲地址
    std::vector<ccu::Variable> remoteStagingToken; // 中继：对端(relay) 中转缓冲 token
    std::vector<ccu::Variable> peerOffsets;
    std::vector<ccu::Variable> payloadOffsets;     // 中继 root mesh：每 channel 的转发源偏移(targetRank*chunk)
    ccu::Variable recvBuffer;
    ccu::Variable recvToken;
    ccu::Variable sendBuffer;
    ccu::Variable sendToken;
    ccu::Variable sliceOffset;
    ccu::Variable sliceSize;
    ccu::Variable rootOffset;
    ccu::Variable slice1Size; // 大消息第一片（256MB）
    ccu::Variable slice2Size; // 大消息第二片（chunkBytes - 256MB）
    ccu::Variable stagingBuffer; // 中继中转缓冲地址
    ccu::Variable stagingToken;  // 中继中转缓冲 token
    ccu::Variable targetOffset;  // 中继：要转发的 target chunk 偏移（targetRank*chunkBytes）
    ccu::Variable relayBytes;    // 中继：按拓扑带宽平衡并受 CCL buffer 容量限制的转发量
    ccu::Event writeEvent;
    ccu::Event writeEvent2; // 大消息第二片的写事件
    ccu::Event copyEvent;
    ccu::Event copyEvent2; // 大消息第二片的自留拷贝事件
    ccu::Event payloadEvent;  // 中继 root mesh：payload 写事件
};

static CcuResult InitResources(ScatterContext &ctx)
{
    if (ctx.arg->channelCount == 0 || ctx.arg->channelCount > MAX_RANK_SIZE - 1) {
        return CCU_E_INTERNAL;
    }
    ctx.remoteBuffer.reserve(ctx.arg->channelCount);
    ctx.remoteToken.reserve(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        ctx.remoteBuffer.push_back(ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], BUFFER_XN_ID));
        ctx.remoteToken.push_back(ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult LoadTaskArgs(ScatterContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceOffset, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.rootOffset, argId++));
    ctx.peerOffsets.reserve(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        ctx.peerOffsets.emplace_back();
        CCU_CHECK_RET(ccu::LoadArg(ctx.peerOffsets.back(), argId++));
    }
    return CCU_SUCCESS;
}

static CcuResult PreSyncRoot(ScatterContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], PRE_SYNC_NOTIFY_IDX, ALL_READY_MASK));
    }
    return CCU_SUCCESS;
}

static CcuResult PreSyncNonRoot(ScatterContext &ctx)
{
    if (ctx.arg->channelCount != 1) {
        return CCU_E_INTERNAL;
    }
    CCU_CHECK_RET(ccu::WriteVariableWithNotify(
        ctx.arg->channels[0], ctx.recvBuffer, BUFFER_XN_ID, PRE_SYNC_NOTIFY_IDX, BUFFER_READY_MASK));
    CCU_CHECK_RET(ccu::WriteVariableWithNotify(
        ctx.arg->channels[0], ctx.recvToken, TOKEN_XN_ID, PRE_SYNC_NOTIFY_IDX, TOKEN_READY_MASK));
    return CCU_SUCCESS;
}

static CcuResult SubmitScatter(ScatterContext &ctx, uint16_t &sendMask)
{
    sendMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        ccu::LocalAddr src;
        src.addr = ctx.sendBuffer;
        src.addr += ctx.peerOffsets[channelIdx];
        src.addr += ctx.sliceOffset;
        src.token = ctx.sendToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[channelIdx];
        dst.addr += ctx.sliceOffset;
        dst.token = ctx.remoteToken[channelIdx];

        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.sliceSize, ctx.writeEvent, channelMask));
        sendMask = static_cast<uint16_t>(sendMask | channelMask);
    }
    return CCU_SUCCESS;
}

static CcuResult SubmitRootCopy(ScatterContext &ctx)
{
    if (ctx.arg->doLocalCopy == 0) {
        return CCU_SUCCESS;
    }
    ccu::LocalAddr dst;
    dst.addr = ctx.recvBuffer;
    dst.addr += ctx.sliceOffset;
    dst.token = ctx.recvToken;

    ccu::LocalAddr src;
    src.addr = ctx.sendBuffer;
    src.addr += ctx.rootOffset;
    src.addr += ctx.sliceOffset;
    src.token = ctx.sendToken;

    CCU_CHECK_RET(ccu::LocalCopy(dst, src, ctx.sliceSize, ctx.copyEvent, 1U));
    return CCU_SUCCESS;
}

static CcuResult CompleteRoot(ScatterContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], POST_SYNC_NOTIFY_IDX, DONE_MASK));
    }
    return CCU_SUCCESS;
}

static CcuResult CompleteNonRoot(ScatterContext &ctx)
{
    if (ctx.arg->channelCount != 1) {
        return CCU_E_INTERNAL;
    }
    CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[0], POST_SYNC_NOTIFY_IDX, DONE_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuScatterRootKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankId != kernelArg->rootId || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->rootId >= kernelArg->rankSize) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadTaskArgs(ctx));
    CCU_CHECK_RET(PreSyncRoot(ctx));

    uint16_t sendMask = 0;
    CCU_CHECK_RET(SubmitScatter(ctx, sendMask));
    CCU_CHECK_RET(SubmitRootCopy(ctx));
    if (sendMask != 0) {
        CCU_CHECK_RET(ccu::EventWait(ctx.writeEvent, sendMask));
    }
    if (ctx.arg->doLocalCopy != 0) {
        CCU_CHECK_RET(ccu::EventWait(ctx.copyEvent, 1U));
    }

    CCU_CHECK_RET(CompleteRoot(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuScatterNonRootKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankId == kernelArg->rootId || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->rootId >= kernelArg->rankSize) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadTaskArgs(ctx));
    CCU_CHECK_RET(PreSyncNonRoot(ctx));
    CCU_CHECK_RET(CompleteNonRoot(ctx));
    return CCU_SUCCESS;
}

// ==================== 4×1 小消息专用 Kernel（PUSH 直发星形） ====================
//   1. 自留块本地拷贝先提交（copyEvent 独立事件），与地址交换(NotifyWait)和网络写重叠；
//   2. 逐 channel NotifyWait→Write 交错：任一 peer 的 recvBuf 地址就绪即启动该路写，不等最慢的一路。
// 保留写等待(EventWait)+DONE 通知，保证非 root 返回前数据已落盘（Scatter 语义必需）。
static CcuResult Submit4x1Scatter(ScatterContext &ctx, uint16_t &sendMask)
{
    sendMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        // 该 peer 的 recvBuf 地址/Token 就绪即启动该路写
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], PRE_SYNC_NOTIFY_IDX, ALL_READY_MASK));

        ccu::LocalAddr src;
        src.addr = ctx.sendBuffer;
        src.addr += ctx.peerOffsets[channelIdx];
        src.addr += ctx.sliceOffset;
        src.token = ctx.sendToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[channelIdx];
        dst.addr += ctx.sliceOffset;
        dst.token = ctx.remoteToken[channelIdx];

        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.sliceSize, ctx.writeEvent, channelMask));
        sendMask = static_cast<uint16_t>(sendMask | channelMask);
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatter4x1RootKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankId != kernelArg->rootId || kernelArg->rankSize != 4 || kernelArg->channelCount == 0
        || kernelArg->channelCount >= MAX_RANK_SIZE || kernelArg->rankId >= kernelArg->rankSize
        || kernelArg->rootId >= kernelArg->rankSize) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    // 先只加载自拷贝需要的 7 个参数，尽早提交自拷贝（与地址交换/网络写重叠），再加载写路径的 3 个 peerOffset。
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvBuffer, 0));
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvToken, 1));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendBuffer, 2));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendToken, 3));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceOffset, 4));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSize, 5));
    CCU_CHECK_RET(ccu::LoadArg(ctx.rootOffset, 6));
    CCU_CHECK_RET(SubmitRootCopy(ctx));

    ctx.peerOffsets.reserve(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        ctx.peerOffsets.emplace_back();
        CCU_CHECK_RET(ccu::LoadArg(ctx.peerOffsets.back(), 7 + channelIdx));
    }

    uint16_t sendMask = 0;
    CCU_CHECK_RET(Submit4x1Scatter(ctx, sendMask));
    // 4×1 小消息的激进路径：3 路网络写提交后仍在飞，只等独立的自拷贝完成即退，
    // 不发 DONE（非 root 也不等 DONE），靠框架 stream sync 在 checker 读之前把写刷完。
    if (ctx.arg->doLocalCopy != 0) {
        CCU_CHECK_RET(ccu::EventWait(ctx.copyEvent, 1U));
    }

    return CCU_SUCCESS;
}

CcuResult CcuScatter4x1NonRootKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankId == kernelArg->rootId || kernelArg->rankSize != 4 || kernelArg->channelCount != 1
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->rootId >= kernelArg->rankSize) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;

    // 先只加载发布所需的 recvBuf 地址/Token，立即发布，缩短 root 的 NotifyWait 等待。
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvBuffer, 0));
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvToken, 1));
    CCU_CHECK_RET(PreSyncNonRoot(ctx));

    // 框架校验 LoadArg 数量必须等于 taskArgs 数量（argsNum=8），故其余 6 个不参与非 root 通信的
    // 参数仍需 Load；放在发布之后、等 DONE 之前，与 root 的地址交换/网络写重叠，不拖发布时刻。
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendBuffer, 2));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendToken, 3));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceOffset, 4));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSize, 5));
    CCU_CHECK_RET(ccu::LoadArg(ctx.rootOffset, 6));
    ctx.peerOffsets.resize(1);
    CCU_CHECK_RET(ccu::LoadArg(ctx.peerOffsets[0], 7));

    // 非 root 发布完 recvBuf 地址即退，不等 DONE（root 也不发 DONE）。
    return CCU_SUCCESS;
}

// ==================== 拉模型 Kernel（小消息） ====================
// root 只发布 sendBuf 地址/Token，非 root 主动从 root 读自己的 chunk，规避 root 逐 peer 写的串行通知开销。
static CcuResult PullLoadTaskArgs(ScatterContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSize, argId++));  // chunkBytes
    CCU_CHECK_RET(ccu::LoadArg(ctx.rootOffset, argId++)); // myRank * chunkBytes（本 rank 的 chunk 偏移）
    return CCU_SUCCESS;
}

CcuResult CcuScatterPullRootKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankId != kernelArg->rootId || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->rootId >= kernelArg->rankSize) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    // 先加载全部 6 个参数，先提交自拷贝（LocalCopy DMA）与后面的发布（控制面）重叠，再发布 sendBuf。
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvBuffer, 0));
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvToken, 1));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendBuffer, 2));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendToken, 3));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSize, 4));
    CCU_CHECK_RET(ccu::LoadArg(ctx.rootOffset, 5));
    // 自留块本地拷贝先提交，与发布/non-root 读数据重叠。
    if (ctx.arg->doLocalCopy != 0) {
        ccu::LocalAddr dst;
        dst.addr = ctx.recvBuffer;
        dst.token = ctx.recvToken;
        ccu::LocalAddr src;
        src.addr = ctx.sendBuffer;
        src.addr += ctx.rootOffset;
        src.token = ctx.sendToken;
        CCU_CHECK_RET(ccu::LocalCopy(dst, src, ctx.sliceSize, ctx.copyEvent, 1U));
    }
    // 发布 sendBuf（与自拷贝重叠）
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[channelIdx], ctx.sendBuffer, BUFFER_XN_ID, PRE_SYNC_NOTIFY_IDX, BUFFER_READY_MASK));
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[channelIdx], ctx.sendToken, TOKEN_XN_ID, PRE_SYNC_NOTIFY_IDX, TOKEN_READY_MASK));
    }
    // 等自拷贝完成即退（不等 non-root 的 DONE）。
    if (ctx.arg->doLocalCopy != 0) {
        CCU_CHECK_RET(ccu::EventWait(ctx.copyEvent, 1U));
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatterPullNonRootKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankId == kernelArg->rootId || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->rootId >= kernelArg->rankSize
        || kernelArg->channelCount != 1) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(PullLoadTaskArgs(ctx));

    CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[0], PRE_SYNC_NOTIFY_IDX, ALL_READY_MASK));

    ccu::RemoteAddr src;
    src.addr = ctx.remoteBuffer[0];
    src.addr += ctx.rootOffset;
    src.token = ctx.remoteToken[0];
    ccu::LocalAddr dst;
    dst.addr = ctx.recvBuffer;
    dst.token = ctx.recvToken;
    CCU_CHECK_RET(ccu::Read(ctx.arg->channels[0], dst, src, ctx.sliceSize, ctx.copyEvent, 1U));
    // 读提交后即退，不等读完成也不发 DONE，靠框架 stream sync 在 checker 读之前把读刷完。
    return CCU_SUCCESS;
}

// ==================== 大消息 Kernel（>256MB，分片下放 kernel 内） ====================
// 地址交换（PreSync）与完成通知（DONE）只做一次，两片背靠背下发，省掉每片重复的地址交换/DONE/host 往返。
static CcuResult LoadLargeTaskArgs(ScatterContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.slice1Size, argId++)); // 第一片 = 256MB
    CCU_CHECK_RET(ccu::LoadArg(ctx.slice2Size, argId++)); // 第二片 = chunkBytes - 256MB
    CCU_CHECK_RET(ccu::LoadArg(ctx.rootOffset, argId++)); // root * chunkBytes（自留块）
    ctx.peerOffsets.reserve(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        ctx.peerOffsets.emplace_back();
        CCU_CHECK_RET(ccu::LoadArg(ctx.peerOffsets.back(), argId++));
    }
    return CCU_SUCCESS;
}

static CcuResult SubmitLargeScatter(ScatterContext &ctx, uint16_t &sendMask)
{
    sendMask = 0;
    // 先让所有通道开始第一片传输，再提交第二片，避免单个通道连续两片阻塞其他链路起步。
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        sendMask = static_cast<uint16_t>(sendMask | channelMask);

        ccu::LocalAddr src;
        src.addr = ctx.sendBuffer;
        src.addr += ctx.peerOffsets[channelIdx];
        src.token = ctx.sendToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[channelIdx];
        dst.token = ctx.remoteToken[channelIdx];

        CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.slice1Size, ctx.writeEvent, channelMask));
    }

    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);

        ccu::LocalAddr src;
        src.addr = ctx.sendBuffer;
        src.addr += ctx.peerOffsets[channelIdx];
        src.addr += ctx.slice1Size;
        src.token = ctx.sendToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[channelIdx];
        dst.addr += ctx.slice1Size;
        dst.token = ctx.remoteToken[channelIdx];

        CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.slice2Size, ctx.writeEvent2, channelMask));
    }
    return CCU_SUCCESS;
}

static CcuResult SubmitLargeRootCopy(ScatterContext &ctx)
{
    if (ctx.arg->doLocalCopy == 0) {
        return CCU_SUCCESS;
    }
    ccu::LocalAddr dst;
    dst.addr = ctx.recvBuffer;
    dst.token = ctx.recvToken;

    ccu::LocalAddr src;
    src.addr = ctx.sendBuffer;
    src.addr += ctx.rootOffset;
    src.token = ctx.sendToken;

    CCU_CHECK_RET(ccu::LocalCopy(dst, src, ctx.slice1Size, ctx.copyEvent, 1U));
    dst.addr += ctx.slice1Size;
    src.addr += ctx.slice1Size;
    CCU_CHECK_RET(ccu::LocalCopy(dst, src, ctx.slice2Size, ctx.copyEvent2, 1U));
    return CCU_SUCCESS;
}

CcuResult CcuScatterLargeRootKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankId != kernelArg->rootId || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->rootId >= kernelArg->rankSize) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadLargeTaskArgs(ctx));
    CCU_CHECK_RET(PreSyncRoot(ctx)); // 地址交换只做一次

    uint16_t sendMask = 0;
    CCU_CHECK_RET(SubmitLargeScatter(ctx, sendMask));
    CCU_CHECK_RET(SubmitLargeRootCopy(ctx));
    if (sendMask != 0) {
        // 只等第一片写完成；第二片写提交后仍在飞，等自拷贝完成后即退。
        CCU_CHECK_RET(ccu::EventWait(ctx.writeEvent, sendMask));
    }
    if (ctx.arg->doLocalCopy != 0) {
        CCU_CHECK_RET(ccu::EventWait(ctx.copyEvent, 1U));
        CCU_CHECK_RET(ccu::EventWait(ctx.copyEvent2, 1U));
    }
    // 第二片写仍在飞，无 DONE（非 root 也不等 DONE），靠框架 stream sync 在 checker 读之前把写刷完。
    return CCU_SUCCESS;
}

CcuResult CcuScatterLargeNonRootKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankId == kernelArg->rootId || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->rootId >= kernelArg->rankSize
        || kernelArg->channelCount != 1) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadLargeTaskArgs(ctx));
    CCU_CHECK_RET(PreSyncNonRoot(ctx)); // 发布 recvBuf 一次
    // 不等 DONE（root 也不发 DONE），靠框架 stream sync 在 checker 读之前把写刷完。
    return CCU_SUCCESS;
}

// ==================== 中继 Kernel（2×8 / 4×3 大消息） ====================
// 2×8(root=0)：relay=rank1..7，target=rank8..14，rank15 直收。
// root 把 target 块前 relayBytes 交给 relay 转发（mesh→relay→Clos），自己只直发尾块，摊薄 Clos 上行。
// relay 拆 gather(读 payload→staging) + own(读自己块→recvBuf) 两 kernel，forward(Clos) 在 gather 完成后
// 启动、与 own 读并行（own 是 mesh 上最大的一笔，forward 摊到它后面跑，缩短 relay 关键路径）。
static CcuResult LoadRelayTargetTaskArgs(ScatterContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvBuffer, argId++)); // 0
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvToken, argId++));  // 1
    return CCU_SUCCESS;
}

// relay mesh gather kernel（die1，PULL）：等 root 发布 sendBuf，读 payload（target 前 relayBytes）→ staging。
// payload 落地即可，Clos 转发由 host 线程 notify 串起来（跨 die 先后靠 host 侧保证）。
CcuResult CcuScatterRelayRelayMeshKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr || kernelArg->relayRole != 1 || kernelArg->channelCount != 1
        || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE || kernelArg->rankId >= kernelArg->rankSize) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.stagingBuffer, argId++)); // 0 中转缓冲地址
    CCU_CHECK_RET(ccu::LoadArg(ctx.stagingToken, argId++));  // 1
    CCU_CHECK_RET(ccu::LoadArg(ctx.targetOffset, argId++));  // 2 目标块源偏移 targetRank*chunk
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSize, argId++));     // 3 本片大小
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceOffset, argId++));   // 4 本片在目标块内的偏移

    const uint32_t rootCh = 0;
    CCU_CHECK_RET(ccu::NotifyWait(kernelArg->channels[rootCh], PRE_SYNC_NOTIFY_IDX, ALL_READY_MASK));

    ccu::Variable rootSendBuf = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[rootCh], BUFFER_XN_ID);
    ccu::Variable rootSendToken = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[rootCh], TOKEN_XN_ID);

    // 读 payload 本片（targetOffset + sliceOffset 起 sliceSize 字节）→ staging
    ccu::RemoteAddr srcFwd;
    srcFwd.addr = rootSendBuf;
    srcFwd.addr += ctx.targetOffset;
    srcFwd.addr += ctx.sliceOffset;
    srcFwd.token = rootSendToken;
    ccu::LocalAddr dstFwd;
    dstFwd.addr = ctx.stagingBuffer;
    dstFwd.token = ctx.stagingToken;
    CCU_CHECK_RET(ccu::Read(kernelArg->channels[rootCh], dstFwd, srcFwd, ctx.sliceSize, ctx.copyEvent, 1U));

    // payload 落地即可（跨 die 先后由 host 侧线程 notify 串，不再用 device LocalNotify）
    CCU_CHECK_RET(ccu::EventWait(ctx.copyEvent, 1U));
    return CCU_SUCCESS;
}

// relay mesh own kernel（die1，PULL）：读自己块→recvBuf。与 gather 同 thread/die/channel、在其后 launch，
// gather 已等 PRE_SYNC 保证 root sendBuf 已发布，此处无需重复 NotifyWait。与 forward(Clos) 并行。
CcuResult CcuScatterRelayRelayOwnKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr || kernelArg->relayRole != 1 || kernelArg->channelCount != 1
        || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE || kernelArg->rankId >= kernelArg->rankSize) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvBuffer, argId++)); // 0 自己输出地址
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvToken, argId++));  // 1
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSize, argId++));  // 2 chunkBytes（自己块大小）
    CCU_CHECK_RET(ccu::LoadArg(ctx.rootOffset, argId++)); // 3 自己块源偏移 relayRank*chunk

    const uint32_t rootCh = 0;
    ccu::Variable rootSendBuf = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[rootCh], BUFFER_XN_ID);
    ccu::Variable rootSendToken = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[rootCh], TOKEN_XN_ID);

    // 读自己块 → recvBuf（与 forward 并行）
    ccu::RemoteAddr srcOwn;
    srcOwn.addr = rootSendBuf;
    srcOwn.addr += ctx.rootOffset;
    srcOwn.token = rootSendToken;
    ccu::LocalAddr dstOwn;
    dstOwn.addr = ctx.recvBuffer;
    dstOwn.token = ctx.recvToken;
    CCU_CHECK_RET(ccu::Read(kernelArg->channels[rootCh], dstOwn, srcOwn, ctx.sliceSize, ctx.copyEvent, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.copyEvent, 1U));
    return CCU_SUCCESS;
}

// relay Clos kernel（die0，PUSH）：转发 staging→target recvBuf。跨 die 顺序由 host 侧保证。
CcuResult CcuScatterRelayRelayClosKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr || kernelArg->relayRole != 1 || kernelArg->channelCount != 1
        || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE || kernelArg->rankId >= kernelArg->rankSize) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.stagingBuffer, argId++)); // 0
    CCU_CHECK_RET(ccu::LoadArg(ctx.stagingToken, argId++));  // 1
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSize, argId++));     // 2 本片大小
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceOffset, argId++));   // 3 本片在目标块内的偏移

    const uint32_t targetCh = 0;
    CCU_CHECK_RET(ccu::NotifyWait(kernelArg->channels[targetCh], PRE_SYNC_NOTIFY_IDX, ALL_READY_MASK));

    ccu::LocalAddr srcFwd;
    srcFwd.addr = ctx.stagingBuffer;
    srcFwd.token = ctx.stagingToken;
    ccu::RemoteAddr dstFwd;
    dstFwd.addr = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[targetCh], BUFFER_XN_ID);
    dstFwd.addr += ctx.sliceOffset;
    dstFwd.token = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[targetCh], TOKEN_XN_ID);
    CCU_CHECK_RET(ccu::Write(kernelArg->channels[targetCh], dstFwd, srcFwd, ctx.sliceSize, ctx.copyEvent, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.copyEvent, 1U));
    // 只有最后一片才向 target 发 DONE（doNotify 是注册期 host 侧标志，非切片用 1，切片非末片用 0）。
    if (kernelArg->doNotify != 0) {
        CCU_CHECK_RET(ccu::NotifyRecord(kernelArg->channels[targetCh], POST_SYNC_NOTIFY_IDX, DONE_MASK));
    }
    return CCU_SUCCESS;
}

// target：收 root 尾块(relayBytes..chunk) + relay 前块(0..relayBytes)。
CcuResult CcuScatterRelayTargetKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr || kernelArg->relayRole != 2 || kernelArg->channelCount != 2) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(LoadRelayTargetTaskArgs(ctx));

    // channel0=root, channel1=relay（host 侧约定），都发布 recvBuf，等两边写完。
    for (uint32_t c = 0; c < 2; c++) {
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[c], ctx.recvBuffer, BUFFER_XN_ID,
            PRE_SYNC_NOTIFY_IDX, BUFFER_READY_MASK));
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[c], ctx.recvToken, TOKEN_XN_ID,
            PRE_SYNC_NOTIFY_IDX, TOKEN_READY_MASK));
    }
    CCU_CHECK_RET(ccu::NotifyWait(kernelArg->channels[0], POST_SYNC_NOTIFY_IDX, DONE_MASK)); // root 尾块写完
    CCU_CHECK_RET(ccu::NotifyWait(kernelArg->channels[1], POST_SYNC_NOTIFY_IDX, DONE_MASK)); // relay 前块写完
    return CCU_SUCCESS;
}

// relay 的 root 侧 kernel：mesh 侧写 payload 到 relay 中转缓冲 + 写自己块到 relay recvBuf；Clos 侧直发 target 尾块。单片。
CcuResult CcuScatterRelayRootKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr || kernelArg->rankId != kernelArg->rootId || kernelArg->rankSize == 0
        || kernelArg->rankSize > MAX_RANK_SIZE || kernelArg->rankId >= kernelArg->rankSize) {
        return CCU_E_INTERNAL;
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;
    // 任务参数布局：recvBuf/Token、sendBuf/Token、chunkBytes、root 自留偏移、relayBytes、尾块大小、
    // 每 channel 的 (peerOffset, payloadOffset)。
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.recvToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sendToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSize, argId++));   // chunkBytes
    CCU_CHECK_RET(ccu::LoadArg(ctx.rootOffset, argId++));  // root 自留偏移
    CCU_CHECK_RET(ccu::LoadArg(ctx.relayBytes, argId++));  // 转发量
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceOffset, argId++)); // 尾块大小 = chunkBytes - relayBytes
    ctx.peerOffsets.reserve(ctx.arg->channelCount);
    ctx.payloadOffsets.reserve(ctx.arg->channelCount);
    for (uint32_t c = 0; c < ctx.arg->channelCount; c++) {
        ctx.peerOffsets.emplace_back();
        CCU_CHECK_RET(ccu::LoadArg(ctx.peerOffsets.back(), argId++));
        ctx.payloadOffsets.emplace_back();
        CCU_CHECK_RET(ccu::LoadArg(ctx.payloadOffsets.back(), argId++));
    }

    const bool isRelayGroup = (kernelArg->relayIsMesh != 0);
    if (isRelayGroup) {
        // mesh 侧（PULL）：发布 sendBuf 给 relay，relay 主动读 payload + 自己块；root 只做自留块本地拷贝。
        for (uint32_t c = 0; c < ctx.arg->channelCount; c++) {
            CCU_CHECK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[c], ctx.sendBuffer, BUFFER_XN_ID,
                PRE_SYNC_NOTIFY_IDX, BUFFER_READY_MASK));
            CCU_CHECK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[c], ctx.sendToken, TOKEN_XN_ID,
                PRE_SYNC_NOTIFY_IDX, TOKEN_READY_MASK));
        }
        if (ctx.arg->doLocalCopy != 0) {
            ccu::LocalAddr dst;
            dst.addr = ctx.recvBuffer;
            dst.token = ctx.recvToken;
            ccu::LocalAddr src;
            src.addr = ctx.sendBuffer;
            src.addr += ctx.rootOffset;
            src.token = ctx.sendToken;
            CCU_CHECK_RET(ccu::LocalCopy(dst, src, ctx.sliceSize, ctx.copyEvent, 1U));
            CCU_CHECK_RET(ccu::EventWait(ctx.copyEvent, 1U));
        }
        return CCU_SUCCESS;
    }

    // Clos 侧：被中继 target 写尾块，直收 target 写整块。
    CCU_CHECK_RET(InitResources(ctx));
    uint16_t sendMask = 0;
    for (uint32_t c = 0; c < ctx.arg->channelCount; c++) {
        CCU_CHECK_RET(ccu::NotifyWait(kernelArg->channels[c], PRE_SYNC_NOTIFY_IDX, ALL_READY_MASK));
        const uint16_t channelMask = static_cast<uint16_t>(1U << c);
        sendMask = static_cast<uint16_t>(sendMask | channelMask);

        ccu::LocalAddr src;
        src.addr = ctx.sendBuffer;
        src.addr += ctx.peerOffsets[c];
        src.token = ctx.sendToken;
        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[c];
        dst.token = ctx.remoteToken[c];

        const bool isFullTarget = (kernelArg->relayIsRelayed[c] == 0);
        if (isFullTarget) {
            CCU_CHECK_RET(ccu::Write(kernelArg->channels[c], dst, src, ctx.sliceSize, ctx.writeEvent, channelMask));
        } else {
            src.addr += ctx.relayBytes;
            dst.addr += ctx.relayBytes;
            CCU_CHECK_RET(ccu::Write(kernelArg->channels[c], dst, src, ctx.sliceOffset, ctx.writeEvent, channelMask));
        }
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.writeEvent, sendMask));
    for (uint32_t c = 0; c < ctx.arg->channelCount; c++) {
        CCU_CHECK_RET(ccu::NotifyRecord(kernelArg->channels[c], POST_SYNC_NOTIFY_IDX, DONE_MASK));
    }
    return CCU_SUCCESS;
}

#undef CCU_CHECK_RET

} // namespace ops_hccl
