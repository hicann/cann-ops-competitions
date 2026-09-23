/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
inline void *BytePtr(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
}

inline const void *ConstBytePtr(const void *base, uint64_t offset)
{
    return static_cast<const void *>(static_cast<const uint8_t *>(base) + offset);
}

inline bool ScatterNeedAck(uint64_t chunkIdx)
{
    return chunkIdx >= SCATTER_SLOT_NUM;
}

inline bool ScatterShouldReturnAck(uint64_t chunkIdx, uint64_t chunks)
{
    return chunkIdx + SCATTER_SLOT_NUM < chunks;
}

inline HcclResult WriteAndNotify(
    ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len, uint32_t remoteNotifyIdx)
{
#if SCATTER_USE_FUSED_WRITE
    return static_cast<HcclResult>(HcommWriteWithNotifyOnThread(thread, channel, dst, src, len, remoteNotifyIdx));
#else
    SCATTER_CHK(static_cast<HcclResult>(HcommWriteOnThread(thread, channel, dst, src, len)));
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel, remoteNotifyIdx));
#endif
}

#if SCATTER_USE_START_HANDSHAKE

HcclResult ReleaseSubThreads(const std::vector<ThreadHandle> &threads, uint32_t usedNum)
{
    for (uint32_t i = 1; i < usedNum; i++) {
        SCATTER_CHK(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[0], threads[i], SCATTER_SUB_THREAD_START_NOTIFY_IDX)));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitStartOnSubThread(const std::vector<ThreadHandle> &threads, uint32_t threadIdx)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(threads[threadIdx], SCATTER_SUB_THREAD_START_NOTIFY_IDX, CUSTOM_TIMEOUT));
}
#endif

HcclResult ReportDoneOnSubThread(const std::vector<ThreadHandle> &threads, uint32_t threadIdx)
{
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[threadIdx], threads[0], threadIdx));
}

HcclResult JoinSubThreads(const std::vector<ThreadHandle> &threads, uint32_t usedNum)
{
    for (uint32_t i = 1; i < usedNum; i++) {
        SCATTER_CHK(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], i, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult PushChunk(ThreadHandle thread, const ChannelInfo &ch, uint32_t stream, uint32_t dstRegion, const void *src,
    const ScatterChunkPlan &plan, uint64_t idx, uint64_t slotBytes)
{
    const uint64_t sent = ScatterChunkOffset(plan, idx);
    const uint64_t chunk = ScatterChunkBytes(plan, idx);
    void *dst = BytePtr(ch.remoteCclMem.addr, ScatterSlotOffset(dstRegion, idx, slotBytes));
    if (ScatterNeedAck(idx)) {
        SCATTER_CHK(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, ch.handle, ScatterAckNotifyIdx(stream, idx), CUSTOM_TIMEOUT)));
    }
    return WriteAndNotify(thread, ch.handle, dst, ConstBytePtr(src, sent), chunk, ScatterDataNotifyIdx(stream, idx));
}

HcclResult DrainToUser(ThreadHandle thread, const ChannelInfo &ch, uint32_t stream, uint32_t region,
    const void *localBuf, void *dst, uint64_t size, uint64_t slotBytes)
{
    if (size == 0) {
        return HCCL_SUCCESS;
    }
    const ScatterChunkPlan plan = ScatterPlanChunks(size, slotBytes);
    for (uint64_t i = 0; i < plan.chunks; i++) {
        const uint64_t got = ScatterChunkOffset(plan, i);
        const uint64_t chunk = ScatterChunkBytes(plan, i);
        SCATTER_CHK(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, ch.handle, ScatterDataNotifyIdx(stream, i), CUSTOM_TIMEOUT)));
        SCATTER_CHK(static_cast<HcclResult>(HcommLocalCopyOnThread(
            thread, BytePtr(dst, got), ConstBytePtr(localBuf, ScatterSlotOffset(region, i, slotBytes)), chunk)));
        if (ScatterShouldReturnAck(i, plan.chunks)) {
            SCATTER_CHK(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, ch.handle, ScatterAckNotifyIdx(stream, i))));
        }
    }
    return HCCL_SUCCESS;
}

#if SCATTER_RELAY_USE_READ

HcclResult ForwardStreamByRead(ThreadHandle thread, const ChannelInfo &inCh, uint32_t inStream,
    const ChannelInfo &outCh, uint32_t outStream, uint64_t size, uint64_t slotBytes)
{
    if (size == 0) {
        return HCCL_SUCCESS;
    }
    const ScatterChunkPlan plan = ScatterPlanChunks(size, slotBytes);
    for (uint64_t i = 0; i < plan.chunks; i++) {
        SCATTER_CHK(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, inCh.handle, ScatterDataNotifyIdx(inStream, i), CUSTOM_TIMEOUT)));
        SCATTER_CHK(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, outCh.handle, ScatterDataNotifyIdx(outStream, i))));
        if (ScatterShouldReturnAck(i, plan.chunks)) {
            SCATTER_CHK(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, outCh.handle, ScatterAckNotifyIdx(outStream, i), CUSTOM_TIMEOUT)));
            SCATTER_CHK(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, inCh.handle, ScatterAckNotifyIdx(inStream, i))));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult DrainByRead(ThreadHandle thread, const ChannelInfo &ch, uint32_t stream, uint32_t remoteRegion, void *dst,
    uint64_t size, uint64_t slotBytes)
{
    if (size == 0) {
        return HCCL_SUCCESS;
    }
    const ScatterChunkPlan plan = ScatterPlanChunks(size, slotBytes);
    for (uint64_t i = 0; i < plan.chunks; i++) {
        const uint64_t got = ScatterChunkOffset(plan, i);
        const uint64_t chunk = ScatterChunkBytes(plan, i);
        SCATTER_CHK(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, ch.handle, ScatterDataNotifyIdx(stream, i), CUSTOM_TIMEOUT)));
        SCATTER_CHK(static_cast<HcclResult>(HcommReadOnThread(thread, ch.handle, BytePtr(dst, got),
            ConstBytePtr(ch.remoteCclMem.addr, ScatterSlotOffset(remoteRegion, i, slotBytes)), chunk)));
        if (ScatterShouldReturnAck(i, plan.chunks)) {
            SCATTER_CHK(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, ch.handle, ScatterAckNotifyIdx(stream, i))));
        }
    }
    return HCCL_SUCCESS;
}
#endif

#if !SCATTER_RELAY_USE_READ

HcclResult ForwardStream(ThreadHandle thread, const ChannelInfo &inCh, uint32_t inStream, uint32_t region,
    const ChannelInfo &outCh, uint32_t outStream, uint32_t outRegion, const void *localBuf, uint64_t size,
    uint64_t slotBytes)
{
    if (size == 0) {
        return HCCL_SUCCESS;
    }
    const ScatterChunkPlan plan = ScatterPlanChunks(size, slotBytes);
    for (uint64_t i = 0; i < plan.chunks; i++) {
        const uint64_t chunk = ScatterChunkBytes(plan, i);
        const void *stage = ConstBytePtr(localBuf, ScatterSlotOffset(region, i, slotBytes));
        void *rdst = BytePtr(outCh.remoteCclMem.addr, ScatterSlotOffset(outRegion, i, slotBytes));
        SCATTER_CHK(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, inCh.handle, ScatterDataNotifyIdx(inStream, i), CUSTOM_TIMEOUT)));
        if (ScatterNeedAck(i)) {
            SCATTER_CHK(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, outCh.handle, ScatterAckNotifyIdx(outStream, i), CUSTOM_TIMEOUT)));
        }
        SCATTER_CHK(WriteAndNotify(thread, outCh.handle, rdst, stage, chunk, ScatterDataNotifyIdx(outStream, i)));
        if (ScatterShouldReturnAck(i, plan.chunks)) {
            SCATTER_CHK(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, inCh.handle, ScatterAckNotifyIdx(inStream, i))));
        }
    }
    return HCCL_SUCCESS;
}
#endif

HcclResult RunRootReadBroadcast(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t shareSize)
{
    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t channelNum = static_cast<uint32_t>(resCtx.channels.size());
    const uint64_t total = shareSize * static_cast<uint64_t>(param.rankSize);

    SCATTER_CHK(static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, resCtx.localBuffer.addr, param.inputPtr, total)));

    const uint32_t dataIdx = ScatterDataNotifyIdx(SCATTER_STREAM_MAIN, 0);
    for (uint32_t chIdx = 0; chIdx < channelNum; chIdx++) {
        SCATTER_CHK(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, resCtx.channels[chIdx].handle, dataIdx)));
    }

    const uint64_t ownBase = static_cast<uint64_t>(param.myRank) * shareSize;
    return static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, param.outputPtr, ConstBytePtr(param.inputPtr, ownBase), shareSize));
}

HcclResult RunNonRootReadBroadcast(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t shareSize)
{
    if (UNLIKELY(resCtx.rootChannelIdx >= resCtx.channels.size())) {
        return HCCL_E_INTERNAL;
    }
    const ChannelInfo &rootCh = resCtx.channels[resCtx.rootChannelIdx];
    const ThreadHandle thread = resCtx.threads[0];

    SCATTER_CHK(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, rootCh.handle, ScatterDataNotifyIdx(SCATTER_STREAM_MAIN, 0), CUSTOM_TIMEOUT)));

    const uint64_t ownBase = static_cast<uint64_t>(param.myRank) * shareSize;
    return static_cast<HcclResult>(HcommReadOnThread(
        thread, rootCh.handle, param.outputPtr, ConstBytePtr(rootCh.remoteCclMem.addr, ownBase), shareSize));
}

constexpr uint64_t SCATTER_MIN_BYTES_PER_THREAD = 1ULL * 1024 * 1024;

constexpr uint64_t SCATTER_SMALL_SHARE_BYTES = 256ULL * 1024;
constexpr uint32_t SCATTER_SMALL_THREAD_NUM = 6;

uint32_t CalcChannelThreadNum(uint32_t threadNum, uint32_t channelNum, uint64_t shareSize)
{
    if (threadNum <= 1 || channelNum == 0) {
        return 1;
    }
    if (shareSize >= SCATTER_SMALL_SHARE_BYTES) {
        return threadNum;
    }
    uint32_t used = (threadNum < SCATTER_SMALL_THREAD_NUM) ? threadNum : SCATTER_SMALL_THREAD_NUM;
    if (used > channelNum) {
        used = channelNum;
    }
    return (used == 0) ? 1 : used;
}

uint32_t CalcOwnCopyParts(uint32_t threadNum, uint64_t dataSize)
{
    const uint64_t maxParts = dataSize / SCATTER_MIN_BYTES_PER_THREAD;
    if (maxParts <= 1 || threadNum <= 1) {
        return 1;
    }
    return (maxParts < threadNum) ? static_cast<uint32_t>(maxParts) : threadNum;
}

HcclResult CopyOwnShareSlice(
    const OpParam &param, ThreadHandle thread, uint32_t threadIdx, uint32_t parts, uint64_t dataSize)
{
    if (parts == 0 || threadIdx >= parts) {
        return HCCL_SUCCESS;
    }
    const uint64_t perPart = dataSize / parts / SCATTER_SLOT_ALIGN * SCATTER_SLOT_ALIGN;
    const uint64_t offset = perPart * threadIdx;
    if (offset >= dataSize) {
        return HCCL_SUCCESS;
    }
    const uint64_t size = (threadIdx + 1 == parts) ? (dataSize - offset) : perPart;
    if (size == 0) {
        return HCCL_SUCCESS;
    }
    const uint64_t ownBase = static_cast<uint64_t>(param.myRank) * dataSize;
    return static_cast<HcclResult>(HcommLocalCopyOnThread(
        thread, BytePtr(param.outputPtr, offset), ConstBytePtr(param.inputPtr, ownBase + offset), size));
}

struct RootChanPlan {
    const void *relaySrc = nullptr;
    const void *mainSrc = nullptr;
    ScatterChunkPlan relayPlan;
    ScatterChunkPlan mainPlan;
    uint32_t steps = 0;
};

void BuildRootChanPlan(const OpParam &param, const ChannelInfo &ch, uint64_t shareSize, uint64_t relayBytes,
    uint64_t directBytes, uint64_t slotBytes, RootChanPlan &plan)
{
    if (relayBytes > 0 && ch.relayForRank != INVALID_VALUE_RANKID) {
        const uint64_t relayBase = static_cast<uint64_t>(ch.relayForRank) * shareSize + directBytes;
        plan.relaySrc = ConstBytePtr(param.inputPtr, relayBase);
        plan.relayPlan = ScatterPlanChunks(relayBytes, slotBytes);
    }
    const uint64_t ownBase = static_cast<uint64_t>(ch.remoteRank) * shareSize;
    const uint64_t sendBytes = (ch.sendFullShare != 0) ? shareSize : directBytes;
    plan.mainSrc = ConstBytePtr(param.inputPtr, ownBase);
    plan.mainPlan = ScatterPlanChunks(sendBytes, slotBytes);
    plan.steps = static_cast<uint32_t>(plan.relayPlan.chunks + plan.mainPlan.chunks);
}

inline uint32_t ChannelThreadIdx(const ChannelInfo &ch, uint32_t chIdx, uint32_t usedNum, bool useAssigned)
{
    if (usedNum == 0) {
        return 0;
    }
    if (useAssigned && ch.threadIdx < usedNum) {
        return ch.threadIdx;
    }
    return chIdx % usedNum;
}

HcclResult EmitRootStep(
    ThreadHandle thread, const ChannelInfo &ch, const RootChanPlan &plan, uint32_t step, uint64_t slotBytes)
{
    if (step < plan.relayPlan.chunks) {
        return PushChunk(
            thread, ch, SCATTER_STREAM_RELAY, SCATTER_REGION_RELAY, plan.relaySrc, plan.relayPlan, step, slotBytes);
    }
    const uint64_t idx = step - plan.relayPlan.chunks;
    return PushChunk(thread, ch, SCATTER_STREAM_MAIN, SCATTER_REGION_MAIN, plan.mainSrc, plan.mainPlan, idx, slotBytes);
}

HcclResult SendOnChannel(const OpParam &param, const AlgResourceCtx &resCtx, const ChannelInfo &ch,
    ThreadHandle thread, uint64_t shareSize, uint64_t relayBytes, uint64_t directBytes)
{
    RootChanPlan plan;
    BuildRootChanPlan(param, ch, shareSize, relayBytes, directBytes, resCtx.slotBytes, plan);
    for (uint32_t step = 0; step < plan.steps; step++) {
        SCATTER_CHK(EmitRootStep(thread, ch, plan, step, resCtx.slotBytes));
    }
    return HCCL_SUCCESS;
}

HcclResult RunRoot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t shareSize, uint64_t relayBytes,
    uint64_t directBytes)
{
    const std::vector<ThreadHandle> &threads = resCtx.threads;
    const uint32_t channelNum = static_cast<uint32_t>(resCtx.channels.size());
    const uint32_t threadNum = static_cast<uint32_t>(threads.size());

    const uint32_t usedNum = CalcChannelThreadNum(threadNum, channelNum, shareSize);
    const bool useAssigned = (usedNum == threadNum);
    const uint32_t ownCopyParts = CalcOwnCopyParts(usedNum, shareSize);
    const uint64_t slotBytes = resCtx.slotBytes;

#if SCATTER_USE_START_HANDSHAKE
    SCATTER_CHK(ReleaseSubThreads(threads, usedNum));
    for (uint32_t threadIdx = 1; threadIdx < usedNum; threadIdx++) {
        SCATTER_CHK(WaitStartOnSubThread(threads, threadIdx));
    }
#endif

#if SCATTER_USE_BREADTH_FIRST
    if (channelNum <= SCATTER_MAX_PLAN_CHANNELS) {
        RootChanPlan plans[SCATTER_MAX_PLAN_CHANNELS];
        uint32_t maxSteps = 0;
        for (uint32_t chIdx = 0; chIdx < channelNum; chIdx++) {
            BuildRootChanPlan(
                param, resCtx.channels[chIdx], shareSize, relayBytes, directBytes, slotBytes, plans[chIdx]);
            if (plans[chIdx].steps > maxSteps) {
                maxSteps = plans[chIdx].steps;
            }
        }

        for (uint32_t step = 0; step < maxSteps; step++) {
            for (uint32_t chIdx = 0; chIdx < channelNum; chIdx++) {
                if (step < plans[chIdx].steps) {
                    const uint32_t tIdx = ChannelThreadIdx(resCtx.channels[chIdx], chIdx, usedNum, useAssigned);
                    SCATTER_CHK(EmitRootStep(threads[tIdx], resCtx.channels[chIdx], plans[chIdx], step, slotBytes));
                }
            }
        }
    } else
#endif
    {
        for (uint32_t chIdx = 0; chIdx < channelNum; chIdx++) {
            const uint32_t tIdx = ChannelThreadIdx(resCtx.channels[chIdx], chIdx, usedNum, useAssigned);
            SCATTER_CHK(SendOnChannel(
                param, resCtx, resCtx.channels[chIdx], threads[tIdx], shareSize, relayBytes, directBytes));
        }
    }

    for (uint32_t threadIdx = 0; threadIdx < usedNum; threadIdx++) {
        SCATTER_CHK(CopyOwnShareSlice(param, threads[threadIdx], threadIdx, ownCopyParts, shareSize));
    }
    for (uint32_t threadIdx = 1; threadIdx < usedNum; threadIdx++) {
        SCATTER_CHK(ReportDoneOnSubThread(threads, threadIdx));
    }
    SCATTER_CHK(JoinSubThreads(threads, usedNum));
    return HCCL_SUCCESS;
}

HcclResult RunRelayRole(const OpParam &param, const AlgResourceCtx &resCtx, const ChannelInfo &rootCh,
    const ChannelInfo &relayCh, ThreadHandle thread, uint64_t relayBytes, uint64_t directBytes)
{
    const uint64_t slotBytes = resCtx.slotBytes;
    if (resCtx.relayRole == SCATTER_ROLE_FORWARD) {
#if SCATTER_RELAY_USE_READ

        return ForwardStreamByRead(
            thread, rootCh, SCATTER_STREAM_RELAY, relayCh, SCATTER_STREAM_MAIN, relayBytes, slotBytes);
#else

        return ForwardStream(thread, rootCh, SCATTER_STREAM_RELAY, SCATTER_REGION_RELAY, relayCh,
            SCATTER_STREAM_MAIN, SCATTER_REGION_RELAY, resCtx.localBuffer.addr, relayBytes, slotBytes);
#endif
    }
#if SCATTER_RELAY_USE_READ

    return DrainByRead(thread, relayCh, SCATTER_STREAM_MAIN, SCATTER_REGION_RELAY,
        BytePtr(param.outputPtr, directBytes), relayBytes, slotBytes);
#else

    return DrainToUser(thread, relayCh, SCATTER_STREAM_MAIN, SCATTER_REGION_RELAY, resCtx.localBuffer.addr,
        BytePtr(param.outputPtr, directBytes), relayBytes, slotBytes);
#endif
}

HcclResult RunNonRoot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t shareSize, uint64_t relayBytes,
    uint64_t directBytes)
{
    const std::vector<ThreadHandle> &threads = resCtx.threads;
    const uint64_t slotBytes = resCtx.slotBytes;

    if (UNLIKELY(resCtx.rootChannelIdx >= resCtx.channels.size())) {
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo &rootCh = resCtx.channels[resCtx.rootChannelIdx];
    const uint64_t fromRootBytes = (resCtx.recvFromRootFull != 0) ? shareSize : directBytes;

    const bool hasRelay = (resCtx.relayRole != SCATTER_ROLE_NONE) && (relayBytes > 0)
        && (resCtx.relayChannelIdx < resCtx.channels.size());
    const ChannelInfo &relayCh = resCtx.channels[hasRelay ? resCtx.relayChannelIdx : resCtx.rootChannelIdx];

    const bool useSubThread = hasRelay && threads.size() >= 2;
    const uint32_t usedThreadNum = useSubThread ? 2U : 1U;
#if SCATTER_USE_START_HANDSHAKE
    SCATTER_CHK(ReleaseSubThreads(threads, usedThreadNum));
    if (useSubThread) {
        SCATTER_CHK(WaitStartOnSubThread(threads, 1));
    }
#endif

    if (useSubThread) {
        SCATTER_CHK(RunRelayRole(param, resCtx, rootCh, relayCh, threads[1], relayBytes, directBytes));
        SCATTER_CHK(ReportDoneOnSubThread(threads, 1));
    }

    SCATTER_CHK(DrainToUser(threads[0], rootCh, SCATTER_STREAM_MAIN, SCATTER_REGION_MAIN, resCtx.localBuffer.addr,
        param.outputPtr, fromRootBytes, slotBytes));

    if (hasRelay && !useSubThread) {
        SCATTER_CHK(RunRelayRole(param, resCtx, rootCh, relayCh, threads[0], relayBytes, directBytes));
    }

    SCATTER_CHK(JoinSubThreads(threads, usedThreadNum));
    return HCCL_SUCCESS;
}

struct RootMergedPlan {
    const void *relaySrc = nullptr;
    const void *mainSrc = nullptr;
    uint64_t relayBytes = 0;
    uint64_t mainBytes = 0;
    uint32_t steps = 0;
};

void BuildRootMergedPlan(const OpParam &param, const ChannelInfo &ch, uint64_t shareSize, uint64_t relayBytes,
    uint64_t directBytes, RootMergedPlan &plan)
{
    if (relayBytes > 0 && ch.relayForRank != INVALID_VALUE_RANKID) {
        const uint64_t relayBase = static_cast<uint64_t>(ch.relayForRank) * shareSize + directBytes;
        plan.relaySrc = ConstBytePtr(param.inputPtr, relayBase);
        plan.relayBytes = relayBytes;
        plan.steps = 1;
    }
    const uint64_t ownBase = static_cast<uint64_t>(ch.remoteRank) * shareSize;
    plan.mainSrc = ConstBytePtr(param.inputPtr, ownBase);
    plan.mainBytes = (ch.sendFullShare != 0) ? shareSize : directBytes;
    plan.steps += 1;
}

HcclResult EmitRootMergedStep(
    ThreadHandle thread, const ChannelInfo &ch, const RootMergedPlan &plan, uint32_t step, uint64_t slotBytes)
{
    if (plan.relayBytes > 0 && step == 0) {
        void *dst = BytePtr(ch.remoteCclMem.addr, ScatterSlotOffset(SCATTER_REGION_RELAY, 0, slotBytes));
        return WriteAndNotify(thread, ch.handle, dst, plan.relaySrc, plan.relayBytes,
            ScatterDataNotifyIdx(SCATTER_STREAM_RELAY, 0));
    }
    void *dst = BytePtr(ch.remoteCclMem.addr, ScatterSlotOffset(SCATTER_REGION_MAIN, 0, slotBytes));
    return WriteAndNotify(thread, ch.handle, dst, plan.mainSrc, plan.mainBytes,
        ScatterDataNotifyIdx(SCATTER_STREAM_MAIN, 0));
}

HcclResult RunRootMerged(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t shareSize, uint64_t relayBytes,
    uint64_t directBytes)
{
    const std::vector<ThreadHandle> &threads = resCtx.threads;
    const uint32_t channelNum = static_cast<uint32_t>(resCtx.channels.size());
    const uint32_t threadNum = static_cast<uint32_t>(threads.size());
    const uint32_t usedNum = CalcChannelThreadNum(threadNum, channelNum, shareSize);
    const bool useAssigned = (usedNum == threadNum);
    const uint32_t ownCopyParts = CalcOwnCopyParts(usedNum, shareSize);
    const uint64_t slotBytes = resCtx.slotBytes;

#if SCATTER_USE_START_HANDSHAKE
    SCATTER_CHK(ReleaseSubThreads(threads, usedNum));
    for (uint32_t threadIdx = 1; threadIdx < usedNum; threadIdx++) {
        SCATTER_CHK(WaitStartOnSubThread(threads, threadIdx));
    }
#endif

    if (channelNum <= SCATTER_MAX_PLAN_CHANNELS) {
        RootMergedPlan plans[SCATTER_MAX_PLAN_CHANNELS];
        uint32_t maxSteps = 0;
        for (uint32_t chIdx = 0; chIdx < channelNum; chIdx++) {
            BuildRootMergedPlan(param, resCtx.channels[chIdx], shareSize, relayBytes, directBytes, plans[chIdx]);
            if (plans[chIdx].steps > maxSteps) {
                maxSteps = plans[chIdx].steps;
            }
        }

        for (uint32_t step = 0; step < maxSteps; step++) {
            for (uint32_t chIdx = 0; chIdx < channelNum; chIdx++) {
                if (step < plans[chIdx].steps) {
                    const ChannelInfo &ch = resCtx.channels[chIdx];
                    const uint32_t tIdx = ChannelThreadIdx(ch, chIdx, usedNum, useAssigned);
                    SCATTER_CHK(EmitRootMergedStep(threads[tIdx], ch, plans[chIdx], step, slotBytes));
                }
            }
        }
    } else {
        for (uint32_t chIdx = 0; chIdx < channelNum; chIdx++) {
            RootMergedPlan plan;
            BuildRootMergedPlan(param, resCtx.channels[chIdx], shareSize, relayBytes, directBytes, plan);
            const uint32_t tIdx = ChannelThreadIdx(resCtx.channels[chIdx], chIdx, usedNum, useAssigned);
            for (uint32_t step = 0; step < plan.steps; step++) {
                SCATTER_CHK(EmitRootMergedStep(threads[tIdx], resCtx.channels[chIdx], plan, step, slotBytes));
            }
        }
    }

    for (uint32_t threadIdx = 0; threadIdx < usedNum; threadIdx++) {
        SCATTER_CHK(CopyOwnShareSlice(param, threads[threadIdx], threadIdx, ownCopyParts, shareSize));
    }
    for (uint32_t threadIdx = 1; threadIdx < usedNum; threadIdx++) {
        SCATTER_CHK(ReportDoneOnSubThread(threads, threadIdx));
    }
    SCATTER_CHK(JoinSubThreads(threads, usedNum));
    return HCCL_SUCCESS;
}

HcclResult RunNonRootMerged(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t shareSize,
    uint64_t relayBytes, uint64_t directBytes)
{
    const std::vector<ThreadHandle> &threads = resCtx.threads;
    const uint64_t slotBytes = resCtx.slotBytes;

    if (UNLIKELY(resCtx.rootChannelIdx >= resCtx.channels.size())) {
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo &rootCh = resCtx.channels[resCtx.rootChannelIdx];
    const uint64_t fromRootBytes = (resCtx.recvFromRootFull != 0) ? shareSize : directBytes;
    const uint32_t dataIdx = ScatterDataNotifyIdx(SCATTER_STREAM_MAIN, 0);
    const uint32_t relayDataIdx = ScatterDataNotifyIdx(SCATTER_STREAM_RELAY, 0);
    const void *localBuf = resCtx.localBuffer.addr;
    const void *mainStage = ConstBytePtr(localBuf, ScatterSlotOffset(SCATTER_REGION_MAIN, 0, slotBytes));
    const void *relayStage = ConstBytePtr(localBuf, ScatterSlotOffset(SCATTER_REGION_RELAY, 0, slotBytes));

    const bool hasRelay = (resCtx.relayRole != SCATTER_ROLE_NONE) && (relayBytes > 0)
        && (resCtx.relayChannelIdx < resCtx.channels.size());
    const bool useSubThread = hasRelay && threads.size() >= 2;
    const uint32_t usedThreadNum = useSubThread ? 2U : 1U;

#if SCATTER_USE_START_HANDSHAKE
    SCATTER_CHK(ReleaseSubThreads(threads, usedThreadNum));
    if (useSubThread) {
        SCATTER_CHK(WaitStartOnSubThread(threads, 1));
    }
#endif

    if (hasRelay && resCtx.relayRole == SCATTER_ROLE_RECV_RELAY) {
        const ChannelInfo &relayCh = resCtx.channels[resCtx.relayChannelIdx];
        const ThreadHandle relayThread = useSubThread ? threads[1] : threads[0];
        SCATTER_CHK(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(relayThread, relayCh.handle, dataIdx, CUSTOM_TIMEOUT)));
        SCATTER_CHK(static_cast<HcclResult>(HcommLocalCopyOnThread(
            relayThread, BytePtr(param.outputPtr, directBytes), relayStage, relayBytes)));
        if (useSubThread) {
            SCATTER_CHK(ReportDoneOnSubThread(threads, 1));
        }
    }

    if (hasRelay && resCtx.relayRole == SCATTER_ROLE_FORWARD) {
        SCATTER_CHK(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(threads[0], rootCh.handle, relayDataIdx, CUSTOM_TIMEOUT)));
    }
    SCATTER_CHK(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(threads[0], rootCh.handle, dataIdx, CUSTOM_TIMEOUT)));

    if (hasRelay && resCtx.relayRole == SCATTER_ROLE_FORWARD) {
        const ChannelInfo &relayCh = resCtx.channels[resCtx.relayChannelIdx];
        void *rdst = BytePtr(relayCh.remoteCclMem.addr, ScatterSlotOffset(SCATTER_REGION_RELAY, 0, slotBytes));
        if (useSubThread) {
            SCATTER_CHK(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(threads[0], threads[1], SCATTER_RELAY_GO_NOTIFY_IDX)));
            SCATTER_CHK(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(threads[1], SCATTER_RELAY_GO_NOTIFY_IDX, CUSTOM_TIMEOUT)));
            SCATTER_CHK(WriteAndNotify(threads[1], relayCh.handle, rdst, relayStage, relayBytes, dataIdx));
            SCATTER_CHK(ReportDoneOnSubThread(threads, 1));
        } else {
            SCATTER_CHK(WriteAndNotify(threads[0], relayCh.handle, rdst, relayStage, relayBytes, dataIdx));
        }
    }

    SCATTER_CHK(static_cast<HcclResult>(
        HcommLocalCopyOnThread(threads[0], param.outputPtr, mainStage, fromRootBytes)));

    SCATTER_CHK(JoinSubThreads(threads, usedThreadNum));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (UNLIKELY(resCtx.threads.empty() || resCtx.slotBytes == 0)) {
        return HCCL_E_INTERNAL;
    }

    const uint32_t dataTypeSize = ScatterDataTypeSize(param.dataType);
    if (UNLIKELY(dataTypeSize == 0)) {
        return HCCL_E_NOT_SUPPORT;
    }

    const uint64_t shareSize = param.count * dataTypeSize;
    const std::vector<ThreadHandle> &threads = resCtx.threads;

    if (param.rankSize <= 1) {
        if (shareSize != 0) {
            SCATTER_CHK(static_cast<HcclResult>(
                HcommLocalCopyOnThread(threads[0], param.outputPtr, param.inputPtr, shareSize)));
        }
        return HCCL_SUCCESS;
    }

    if (shareSize == 0) {
        return HCCL_SUCCESS;
    }

    if (ScatterUseReadBroadcast(shareSize, param.rankSize, resCtx.slotBytes)) {
        return (param.myRank == param.root) ? RunRootReadBroadcast(param, resCtx, shareSize)
                                            : RunNonRootReadBroadcast(param, resCtx, shareSize);
    }

    const uint64_t relayBytes = ScatterRelayBytes(shareSize);
    const uint64_t directBytes = shareSize - relayBytes;

    const bool merged = ScatterUseMerged(shareSize, resCtx.slotBytes);

    if (param.myRank != param.root) {
        return merged ? RunNonRootMerged(param, resCtx, shareSize, relayBytes, directBytes)
                      : RunNonRoot(param, resCtx, shareSize, relayBytes, directBytes);
    }
    return merged ? RunRootMerged(param, resCtx, shareSize, relayBytes, directBytes)
                  : RunRoot(param, resCtx, shareSize, relayBytes, directBytes);
}
} // namespace ops_hccl
