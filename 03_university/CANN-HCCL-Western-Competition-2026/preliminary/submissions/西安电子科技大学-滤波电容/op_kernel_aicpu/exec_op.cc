/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <limits>

#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
const ChannelInfo *FindChannel(const AlgResourceCtx &ctx, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : ctx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

void *Offset(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
}

HcclResult NotifyRecord(ThreadHandle thread, const ChannelInfo *channel, uint32_t notifyIdx)
{
    if (channel == nullptr) {
        return HCCL_E_INTERNAL;
    }
    return static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel->handle, notifyIdx));
}

HcclResult NotifyWait(ThreadHandle thread, const ChannelInfo *channel, uint32_t notifyIdx)
{
    if (channel == nullptr) {
        return HCCL_E_INTERNAL;
    }
    return static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel->handle, notifyIdx, CUSTOM_TIMEOUT));
}

HcclResult LocalCopy(ThreadHandle thread, void *dst, const void *src, uint64_t len)
{
    return static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, len));
}

HcclResult LocalReduce(ThreadHandle thread, void *dst, const void *src, uint64_t count)
{
    return static_cast<HcclResult>(HcommLocalReduceOnThread(thread, dst, src, count,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
}

HcclResult ReadReduce(ThreadHandle thread, const ChannelInfo *channel, void *dst, const void *src,
    uint64_t count)
{
    if (channel == nullptr) {
        return HCCL_E_INTERNAL;
    }
    return static_cast<HcclResult>(HcommReadReduceOnThread(thread, channel->handle, dst, src,
        count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
}

HcclResult Read(ThreadHandle thread, const ChannelInfo *channel, void *dst, const void *src, uint64_t len)
{
    if (channel == nullptr) {
        return HCCL_E_INTERNAL;
    }
    return static_cast<HcclResult>(HcommReadOnThread(thread, channel->handle, dst, src, len));
}

HcclResult Write(ThreadHandle thread, const ChannelInfo *channel, void *dst, const void *src, uint64_t len)
{
    if (channel == nullptr) {
        return HCCL_E_INTERNAL;
    }
    return static_cast<HcclResult>(HcommWriteOnThread(thread, channel->handle, dst, src, len));
}

HcclResult StartParallelThreads(const std::vector<ThreadHandle> &threads)
{
    if (threads.size() != AR_PARALLEL_THREAD_NUM) {
        return HCCL_E_INTERNAL;
    }
    for (uint32_t index = 1; index < AR_PARALLEL_THREAD_NUM; ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[0], threads[index], 0)));
    }
    for (uint32_t index = 1; index < AR_PARALLEL_THREAD_NUM; ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[index], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult JoinParallelThreads(const std::vector<ThreadHandle> &threads)
{
    if (threads.size() != AR_PARALLEL_THREAD_NUM) {
        return HCCL_E_INTERNAL;
    }
    for (uint32_t index = 1; index < AR_PARALLEL_THREAD_NUM; ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[0], index - 1U, CUSTOM_TIMEOUT)));
    }
    for (uint32_t index = 1; index < AR_PARALLEL_THREAD_NUM; ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[index], threads[0], index - 1U)));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadNotifyRecord(ThreadHandle thread, ThreadHandle dstThread, uint32_t notifyIdx)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(thread, dstThread, notifyIdx));
}

HcclResult ThreadNotifyWait(ThreadHandle thread, uint32_t notifyIdx)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(thread, notifyIdx, CUSTOM_TIMEOUT));
}

uint64_t ShardCount(uint64_t totalCount, uint32_t lane)
{
    const uint64_t base = totalCount / AR_RANKS_PER_SERVER;
    const uint64_t remainder = totalCount % AR_RANKS_PER_SERVER;
    return base + (lane < remainder ? 1U : 0U);
}

uint64_t ShardOffsetElements(uint64_t totalCount, uint32_t lane)
{
    const uint64_t base = totalCount / AR_RANKS_PER_SERVER;
    const uint64_t remainder = totalCount % AR_RANKS_PER_SERVER;
    return base * lane + (lane < remainder ? lane : remainder);
}

HcclResult StripedChunkBarrier(const AlgResourceCtx &resCtx, ThreadHandle thread,
    uint32_t server, uint32_t lane)
{
    for (uint32_t step = 1; step < AR_RANKS_PER_SERVER; step <<= 1U) {
        const uint32_t childLane = lane + step;
        if (lane % (step << 1U) == 0 && childLane < AR_RANKS_PER_SERVER) {
            const uint32_t childRank = resCtx.rankOfServerLane[server][childLane];
            CHK_RET(NotifyWait(thread, FindChannel(resCtx, childRank), AR_NOTIFY_FINAL));
        }
    }

    const ChannelInfo *parent = nullptr;
    if (lane != 0) {
        const uint32_t parentLane = lane - (lane & (~lane + 1U));
        const uint32_t parentRank = resCtx.rankOfServerLane[server][parentLane];
        parent = FindChannel(resCtx, parentRank);
        CHK_RET(NotifyRecord(thread, parent, AR_NOTIFY_FINAL));
        CHK_RET(NotifyWait(thread, parent, AR_NOTIFY_BROADCAST));
    } else {
        const uint32_t otherLeader = resCtx.rankOfServerLane[server ^ 1U][0];
        const ChannelInfo *interServer = FindChannel(resCtx, otherLeader);
        if (server != 0) {
            CHK_RET(NotifyRecord(thread, interServer, AR_NOTIFY_FINAL));
            CHK_RET(NotifyWait(thread, interServer, AR_NOTIFY_BROADCAST));
        } else {
            CHK_RET(NotifyWait(thread, interServer, AR_NOTIFY_FINAL));
            CHK_RET(NotifyRecord(thread, interServer, AR_NOTIFY_BROADCAST));
        }
    }

    for (uint32_t step = AR_RANKS_PER_SERVER >> 1U; step != 0; step >>= 1U) {
        const uint32_t childLane = lane + step;
        if (lane % (step << 1U) == 0 && childLane < AR_RANKS_PER_SERVER) {
            const uint32_t childRank = resCtx.rankOfServerLane[server][childLane];
            CHK_RET(NotifyRecord(thread, FindChannel(resCtx, childRank), AR_NOTIFY_BROADCAST));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecTreeChunk(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread,
    uint32_t server, uint32_t lane, uint64_t processedBytes, uint64_t currentBytes,
    uint64_t currentCount, void *reduction, void *output)
{
    CHK_RET(LocalCopy(thread, reduction, Offset(param.inputPtr, processedBytes), currentBytes));

    for (uint32_t step = 1; step < AR_RANKS_PER_SERVER; step <<= 1U) {
        const uint32_t childLane = lane + step;
        if (lane % (step << 1U) == 0 && childLane < AR_RANKS_PER_SERVER) {
            const uint32_t childRank = resCtx.rankOfServerLane[server][childLane];
            const ChannelInfo *child = FindChannel(resCtx, childRank);
            CHK_RET(NotifyWait(thread, child, AR_NOTIFY_READY));
            CHK_RET(ReadReduce(thread, child, reduction,
                Offset(child->remoteCclMem.addr, resCtx.reductionOffset), currentCount));
            CHK_RET(NotifyRecord(thread, child, AR_NOTIFY_CONSUMED));
        }
    }

    const ChannelInfo *parent = nullptr;
    if (lane != 0) {
        const uint32_t parentLane = lane - (lane & (~lane + 1U));
        const uint32_t parentRank = resCtx.rankOfServerLane[server][parentLane];
        parent = FindChannel(resCtx, parentRank);
        CHK_RET(NotifyRecord(thread, parent, AR_NOTIFY_READY));
        CHK_RET(NotifyWait(thread, parent, AR_NOTIFY_CONSUMED));
    }

    const uint32_t leader = resCtx.leaderRank[server];
    if (param.myRank == leader) {
        const uint32_t otherLeader = resCtx.leaderRank[server ^ 1U];
        const ChannelInfo *interServer = FindChannel(resCtx, otherLeader);
        if (leader != resCtx.rootLeader) {
            CHK_RET(NotifyRecord(thread, interServer, AR_NOTIFY_READY));
            CHK_RET(NotifyWait(thread, interServer, AR_NOTIFY_FINAL));
        } else {
            CHK_RET(NotifyWait(thread, interServer, AR_NOTIFY_READY));
            CHK_RET(ReadReduce(thread, interServer, reduction,
                Offset(interServer->remoteCclMem.addr, resCtx.reductionOffset), currentCount));
            CHK_RET(LocalCopy(thread, output, reduction, currentBytes));
            CHK_RET(Write(thread, interServer,
                Offset(interServer->remoteCclMem.addr, resCtx.outputOffset), reduction, currentBytes));
            CHK_RET(NotifyRecord(thread, interServer, AR_NOTIFY_FINAL));
        }
    }

    if (lane != 0) {
        CHK_RET(NotifyWait(thread, parent, AR_NOTIFY_BROADCAST));
    }
    for (uint32_t step = AR_RANKS_PER_SERVER >> 1U; step != 0; step >>= 1U) {
        const uint32_t childLane = lane + step;
        if (lane % (step << 1U) == 0 && childLane < AR_RANKS_PER_SERVER) {
            const uint32_t childRank = resCtx.rankOfServerLane[server][childLane];
            const ChannelInfo *child = FindChannel(resCtx, childRank);
            CHK_RET(Write(thread, child,
                Offset(child->remoteCclMem.addr, resCtx.outputOffset), output, currentBytes));
            CHK_RET(NotifyRecord(thread, child, AR_NOTIFY_BROADCAST));
        }
    }

    CHK_RET(LocalCopy(thread, Offset(param.outputPtr, processedBytes), output, currentBytes));

    for (uint32_t step = 1; step < AR_RANKS_PER_SERVER; step <<= 1U) {
        const uint32_t childLane = lane + step;
        if (lane % (step << 1U) == 0 && childLane < AR_RANKS_PER_SERVER) {
            const uint32_t childRank = resCtx.rankOfServerLane[server][childLane];
            CHK_RET(NotifyWait(thread, FindChannel(resCtx, childRank), AR_NOTIFY_FINAL));
        }
    }
    if (lane != 0) {
        CHK_RET(NotifyRecord(thread, parent, AR_NOTIFY_FINAL));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecStripedChunk(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread,
    uint32_t server, uint32_t lane, uint64_t processedBytes, uint64_t currentBytes,
    uint64_t currentCount, void *reduction, void *output)
{
    CHK_RET(LocalCopy(thread, reduction, Offset(param.inputPtr, processedBytes), currentBytes));

    for (uint32_t mask = AR_RANKS_PER_SERVER >> 1U; mask != 0; mask >>= 1U) {
        const uint32_t peerLane = lane ^ mask;
        const uint32_t peerRank = resCtx.rankOfServerLane[server][peerLane];
        const ChannelInfo *peer = FindChannel(resCtx, peerRank);
        CHK_RET(NotifyRecord(thread, peer, AR_NOTIFY_READY));
        CHK_RET(NotifyWait(thread, peer, AR_NOTIFY_READY));

        const uint32_t keepStartLane = (lane / mask) * mask;
        const uint64_t keepOffsetElements = ShardOffsetElements(currentCount, keepStartLane);
        const uint64_t keepEndElements = ShardOffsetElements(currentCount, keepStartLane + mask);
        const uint64_t keepCount = keepEndElements - keepOffsetElements;
        if (keepCount != 0) {
            const uint64_t keepOffsetBytes = keepOffsetElements * AR_FLOAT_BYTES;
            CHK_RET(ReadReduce(thread, peer, Offset(reduction, keepOffsetBytes),
                Offset(peer->remoteCclMem.addr, resCtx.reductionOffset + keepOffsetBytes), keepCount));
        }
    }

    const uint64_t ownCount = ShardCount(currentCount, lane);
    const uint64_t ownOffsetBytes = ShardOffsetElements(currentCount, lane) * AR_FLOAT_BYTES;
    const uint64_t ownBytes = ownCount * AR_FLOAT_BYTES;
    void *ownReduction = Offset(reduction, ownOffsetBytes);
    const uint32_t otherRank = resCtx.rankOfServerLane[server ^ 1U][lane];
    const ChannelInfo *interServer = FindChannel(resCtx, otherRank);
    if (server != 0) {
        CHK_RET(NotifyRecord(thread, interServer, AR_NOTIFY_READY));
        CHK_RET(NotifyWait(thread, interServer, AR_NOTIFY_FINAL));
    } else {
        CHK_RET(NotifyWait(thread, interServer, AR_NOTIFY_READY));
        if (ownCount != 0) {
            CHK_RET(ReadReduce(thread, interServer, ownReduction,
                Offset(interServer->remoteCclMem.addr, resCtx.reductionOffset + ownOffsetBytes), ownCount));
            CHK_RET(LocalCopy(thread, Offset(output, ownOffsetBytes), ownReduction, ownBytes));
            CHK_RET(Write(thread, interServer,
                Offset(interServer->remoteCclMem.addr, resCtx.outputOffset + ownOffsetBytes),
                ownReduction, ownBytes));
        }
        CHK_RET(NotifyRecord(thread, interServer, AR_NOTIFY_FINAL));
    }

    for (uint32_t mask = 1; mask < AR_RANKS_PER_SERVER; mask <<= 1U) {
        const uint32_t peerLane = lane ^ mask;
        const uint32_t peerRank = resCtx.rankOfServerLane[server][peerLane];
        const ChannelInfo *peer = FindChannel(resCtx, peerRank);
        CHK_RET(NotifyRecord(thread, peer, AR_NOTIFY_CONSUMED));
        CHK_RET(NotifyWait(thread, peer, AR_NOTIFY_CONSUMED));

        const uint32_t peerBlockStartLane = (peerLane / mask) * mask;
        const uint64_t peerOffsetElements = ShardOffsetElements(currentCount, peerBlockStartLane);
        const uint64_t peerEndElements = ShardOffsetElements(currentCount, peerBlockStartLane + mask);
        const uint64_t peerCount = peerEndElements - peerOffsetElements;
        if (peerCount != 0) {
            const uint64_t peerOffsetBytes = peerOffsetElements * AR_FLOAT_BYTES;
            const uint64_t peerBytes = peerCount * AR_FLOAT_BYTES;
            CHK_RET(Read(thread, peer, Offset(output, peerOffsetBytes),
                Offset(peer->remoteCclMem.addr, resCtx.outputOffset + peerOffsetBytes), peerBytes));
        }
    }
    CHK_RET(LocalCopy(thread, Offset(param.outputPtr, processedBytes), output, currentBytes));
    return StripedChunkBarrier(resCtx, thread, server, lane);
}

__attribute__((noinline)) HcclResult ExecParallelChunk(const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint32_t server, uint32_t lane, uint64_t processedBytes, uint64_t currentBytes,
    uint64_t currentCount, void *input, void *staging, void *output)
{
    const ThreadHandle mainThread = resCtx.threads[0];
    const uint64_t ownCount = ShardCount(currentCount, lane);
    const uint64_t ownOffsetBytes = ShardOffsetElements(currentCount, lane) * AR_FLOAT_BYTES;
    const uint64_t ownBytes = ownCount * AR_FLOAT_BYTES;
    const uint64_t stagingSlotBytes = resCtx.chunkCapacity / AR_RANKS_PER_SERVER;
    void *ownOutput = Offset(output, ownOffsetBytes);

    CHK_RET(LocalCopy(mainThread, input, Offset(param.inputPtr, processedBytes), currentBytes));

    CHK_RET(StartParallelThreads(resCtx.threads));
    uint32_t workerIndex = 1;
    for (uint32_t peerLane = 0; peerLane < AR_RANKS_PER_SERVER; ++peerLane) {
        if (peerLane == lane) {
            continue;
        }
        const uint32_t peerRank = resCtx.rankOfServerLane[server][peerLane];
        const ChannelInfo *peer = FindChannel(resCtx, peerRank);
        const ThreadHandle worker = resCtx.threads[workerIndex];
        CHK_RET(NotifyRecord(worker, peer, AR_NOTIFY_READY));
        CHK_RET(NotifyWait(worker, peer, AR_NOTIFY_READY));
        if (ownBytes != 0) {
            CHK_RET(Read(worker, peer, Offset(staging, (workerIndex - 1U) * stagingSlotBytes),
                Offset(peer->remoteCclMem.addr, resCtx.inputOffset + ownOffsetBytes), ownBytes));
        }
        CHK_RET(NotifyRecord(worker, peer, AR_NOTIFY_CONSUMED));
        CHK_RET(NotifyWait(worker, peer, AR_NOTIFY_CONSUMED));
        ++workerIndex;
    }

    ThreadHandle thread1 = resCtx.threads[1];
    ThreadHandle thread2 = resCtx.threads[2];
    ThreadHandle thread3 = resCtx.threads[3];
    ThreadHandle thread4 = resCtx.threads[4];
    ThreadHandle thread5 = resCtx.threads[5];
    ThreadHandle thread6 = resCtx.threads[6];
    ThreadHandle thread7 = resCtx.threads[7];
    if (ownCount != 0) {
        CHK_RET(LocalReduce(thread1, ownOutput, Offset(staging, 0), ownCount));
    }
    CHK_RET(ThreadNotifyRecord(thread3, thread2, 1));
    CHK_RET(ThreadNotifyWait(thread2, 1));
    if (ownCount != 0) {
        CHK_RET(LocalReduce(thread2, Offset(staging, stagingSlotBytes),
            Offset(staging, 2U * stagingSlotBytes), ownCount));
    }
    CHK_RET(ThreadNotifyRecord(thread5, thread4, 1));
    CHK_RET(ThreadNotifyWait(thread4, 1));
    if (ownCount != 0) {
        CHK_RET(LocalReduce(thread4, Offset(staging, 3U * stagingSlotBytes),
            Offset(staging, 4U * stagingSlotBytes), ownCount));
    }
    CHK_RET(ThreadNotifyRecord(thread7, thread6, 1));
    CHK_RET(ThreadNotifyWait(thread6, 1));
    if (ownCount != 0) {
        CHK_RET(LocalReduce(thread6, Offset(staging, 5U * stagingSlotBytes),
            Offset(staging, 6U * stagingSlotBytes), ownCount));
    }

    CHK_RET(ThreadNotifyRecord(thread2, thread1, 1));
    CHK_RET(ThreadNotifyWait(thread1, 1));
    if (ownCount != 0) {
        CHK_RET(LocalReduce(thread1, ownOutput, Offset(staging, stagingSlotBytes), ownCount));
    }
    CHK_RET(ThreadNotifyRecord(thread6, thread4, 2));
    CHK_RET(ThreadNotifyWait(thread4, 2));
    if (ownCount != 0) {
        CHK_RET(LocalReduce(thread4, Offset(staging, 3U * stagingSlotBytes),
            Offset(staging, 5U * stagingSlotBytes), ownCount));
    }

    CHK_RET(ThreadNotifyRecord(thread4, thread1, 2));
    CHK_RET(ThreadNotifyWait(thread1, 2));
    if (ownCount != 0) {
        CHK_RET(LocalReduce(thread1, ownOutput, Offset(staging, 3U * stagingSlotBytes), ownCount));
    }
    CHK_RET(ThreadNotifyRecord(thread1, mainThread, 0));
    CHK_RET(ThreadNotifyWait(mainThread, 0));

    const uint32_t otherRank = resCtx.rankOfServerLane[server ^ 1U][lane];
    const ChannelInfo *interServer = FindChannel(resCtx, otherRank);
    if (server != 0) {
        CHK_RET(NotifyRecord(mainThread, interServer, AR_NOTIFY_READY));
        CHK_RET(NotifyWait(mainThread, interServer, AR_NOTIFY_FINAL));
    } else {
        CHK_RET(NotifyWait(mainThread, interServer, AR_NOTIFY_READY));
        if (ownCount != 0) {
            CHK_RET(ReadReduce(mainThread, interServer, ownOutput,
                Offset(interServer->remoteCclMem.addr, resCtx.outputOffset + ownOffsetBytes),
                ownCount));
            CHK_RET(Write(mainThread, interServer,
                Offset(interServer->remoteCclMem.addr, resCtx.outputOffset + ownOffsetBytes),
                ownOutput, ownBytes));
        }
        CHK_RET(NotifyRecord(mainThread, interServer, AR_NOTIFY_FINAL));
    }

    CHK_RET(StartParallelThreads(resCtx.threads));
    workerIndex = 1;
    for (uint32_t peerLane = 0; peerLane < AR_RANKS_PER_SERVER; ++peerLane) {
        if (peerLane == lane) {
            continue;
        }
        const uint32_t peerRank = resCtx.rankOfServerLane[server][peerLane];
        const ChannelInfo *peer = FindChannel(resCtx, peerRank);
        const ThreadHandle worker = resCtx.threads[workerIndex];
        CHK_RET(NotifyRecord(worker, peer, AR_NOTIFY_GATHER));
        CHK_RET(NotifyWait(worker, peer, AR_NOTIFY_GATHER));
        const uint64_t peerCount = ShardCount(currentCount, peerLane);
        if (peerCount != 0) {
            const uint64_t peerOffsetBytes = ShardOffsetElements(currentCount, peerLane) *
                AR_FLOAT_BYTES;
            const uint64_t peerBytes = peerCount * AR_FLOAT_BYTES;
            CHK_RET(Read(worker, peer, Offset(output, peerOffsetBytes),
                Offset(peer->remoteCclMem.addr, resCtx.outputOffset + peerOffsetBytes), peerBytes));
        }
        ++workerIndex;
    }
    CHK_RET(JoinParallelThreads(resCtx.threads));
    CHK_RET(LocalCopy(mainThread, Offset(param.outputPtr, processedBytes), output, currentBytes));
    return StripedChunkBarrier(resCtx, mainThread, server, lane);
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize != AR_RANKS || param.myRank >= AR_RANKS ||
        param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM ||
        param.inputPtr == nullptr || param.outputPtr == nullptr || resCtx.localBuffer.addr == nullptr ||
        param.count > std::numeric_limits<uint64_t>::max() / AR_FLOAT_BYTES ||
        resCtx.chunkCapacity < AR_FLOAT_BYTES ||
        resCtx.chunkCapacity > std::numeric_limits<uint64_t>::max() / 3U ||
        resCtx.inputOffset != 0) {
        return HCCL_E_NOT_SUPPORT;
    }

    const uint64_t totalDataBytes = param.count * AR_FLOAT_BYTES;
    const bool useStriped = totalDataBytes >= AR_STRIPED_THRESHOLD_BYTES;
    const bool useParallel = totalDataBytes >= AR_PARALLEL_THRESHOLD_BYTES;
    uint64_t requiredBufferBytes = 0;
    if (useParallel) {
        const uint64_t stagingBytes = (resCtx.chunkCapacity / AR_RANKS_PER_SERVER) *
            AR_PARALLEL_PEER_NUM;
        if (resCtx.chunkCapacity % AR_STRIPE_ALIGNMENT_BYTES != 0 ||
            resCtx.reductionOffset != resCtx.chunkCapacity ||
            resCtx.outputOffset != 0 ||
            resCtx.threads.size() != AR_PARALLEL_THREAD_NUM) {
            return HCCL_E_NOT_SUPPORT;
        }
        requiredBufferBytes = resCtx.reductionOffset + stagingBytes;
    } else {
        if (resCtx.reductionOffset != 0 || resCtx.outputOffset != resCtx.chunkCapacity ||
            resCtx.threads.size() != 1U) {
            return HCCL_E_NOT_SUPPORT;
        }
        requiredBufferBytes = resCtx.outputOffset + resCtx.chunkCapacity;
    }
    if (requiredBufferBytes > resCtx.localBuffer.size ||
        requiredBufferBytes > resCtx.cclBufferBytes) {
        return HCCL_E_NOT_SUPPORT;
    }

    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteCclMem.addr == nullptr ||
            requiredBufferBytes > channel.remoteCclMem.size) {
            return HCCL_E_NOT_SUPPORT;
        }
    }

    ThreadHandle thread = resCtx.aicpuThread;
    void *input = Offset(resCtx.localBuffer.addr, resCtx.inputOffset);
    void *reduction = Offset(resCtx.localBuffer.addr, resCtx.reductionOffset);
    void *output = Offset(resCtx.localBuffer.addr, resCtx.outputOffset);

    const uint32_t server = resCtx.serverOfRank[param.myRank];
    if (server >= AR_SERVERS || resCtx.leaderRank[server] >= AR_RANKS ||
        resCtx.rootLeader != resCtx.leaderRank[0]) {
        return HCCL_E_NOT_SUPPORT;
    }

    const uint32_t lane = resCtx.laneOfRank[param.myRank];
    if (lane >= AR_RANKS_PER_SERVER || resCtx.rankOfServerLane[server][lane] != param.myRank) {
        return HCCL_E_NOT_SUPPORT;
    }
    uint64_t processedBytes = 0;
    while (processedBytes < totalDataBytes) {
        const uint64_t remainingBytes = totalDataBytes - processedBytes;
        const uint64_t currentBytes = remainingBytes < resCtx.chunkCapacity ?
            remainingBytes : resCtx.chunkCapacity;
        const uint64_t currentCount = currentBytes / AR_FLOAT_BYTES;
        if (useParallel) {
            CHK_RET(ExecParallelChunk(param, resCtx, server, lane, processedBytes,
                currentBytes, currentCount, input, reduction, output));
        } else if (useStriped) {
            CHK_RET(ExecStripedChunk(param, resCtx, thread, server, lane, processedBytes,
                currentBytes, currentCount, reduction, output));
        } else {
            CHK_RET(ExecTreeChunk(param, resCtx, thread, server, lane, processedBytes,
                currentBytes, currentCount, reduction, output));
        }
        processedBytes += currentBytes;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
