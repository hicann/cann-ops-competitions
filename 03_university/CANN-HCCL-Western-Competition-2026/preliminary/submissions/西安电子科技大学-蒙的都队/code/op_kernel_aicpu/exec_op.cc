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
#include <cstdint>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_PRIMITIVE_BYTES = 256ULL * 1024 * 1024;

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const auto &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

HcclResult Record(ThreadHandle thread, const ChannelInfo &channel, uint32_t notifyIndex)
{
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyIndex));
}

HcclResult Wait(ThreadHandle thread, const ChannelInfo &channel, uint32_t notifyIndex)
{
    return static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, notifyIndex, CUSTOM_TIMEOUT));
}

uint64_t ChunkCount(uint64_t count, uint32_t chunkIndex, uint32_t chunkNum)
{
    return count / chunkNum + (chunkIndex < count % chunkNum ? 1 : 0);
}

uint64_t ChunkOffset(uint64_t count, uint32_t chunkIndex, uint32_t chunkNum)
{
    return chunkIndex * (count / chunkNum) + std::min<uint64_t>(chunkIndex, count % chunkNum);
}

HcclResult ExchangeSegment(const OpParam &param, const AlgResourceCtx &resCtx, const ChannelInfo &prevChannel,
    const ChannelInfo &nextChannel, const void *sendAddr, uint64_t sendCount, void *recvAddr, uint64_t recvCount,
    bool reduce)
{
    ThreadHandle thread = resCtx.aicpuThread;

    // The receiver first advertises that its single-slot CCL buffer is free.
    CHK_RET(Record(thread, prevChannel, NOTIFY_IDX_ACK));
    CHK_RET(Wait(thread, nextChannel, NOTIFY_IDX_ACK));
    if (sendCount > 0) {
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
            thread, nextChannel.handle, nextChannel.remoteCclMem.addr, sendAddr, sendCount * sizeof(float))));
    }
    CHK_RET(Record(thread, nextChannel, NOTIFY_IDX_DATA_SIGNAL));

    CHK_RET(Wait(thread, prevChannel, NOTIFY_IDX_DATA_SIGNAL));
    if (reduce && recvCount > 0) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, recvAddr, resCtx.localBuffer.addr,
            recvCount, static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
    } else if (recvCount > 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, recvAddr, resCtx.localBuffer.addr, recvCount * sizeof(float))));
    }
    CHK_RET(Record(thread, prevChannel, NOTIFY_IDX_ACK));
    CHK_RET(Wait(thread, nextChannel, NOTIFY_IDX_ACK));
    return HCCL_SUCCESS;
}

HcclResult RingPhase(const OpParam &param, const AlgResourceCtx &resCtx, bool reduceScatter,
    uint64_t maxSegmentCount)
{
    const uint32_t localRank = param.myRank % resCtx.ranksPerServer;
    const uint32_t serverBase = (param.myRank / resCtx.ranksPerServer) * resCtx.ranksPerServer;
    const uint32_t prevRank = serverBase + (localRank + resCtx.ranksPerServer - 1) % resCtx.ranksPerServer;
    const uint32_t nextRank = serverBase + (localRank + 1) % resCtx.ranksPerServer;
    const ChannelInfo *prevChannel = FindChannel(resCtx, prevRank);
    const ChannelInfo *nextChannel = FindChannel(resCtx, nextRank);
    CHK_PRT_RET(prevChannel == nullptr || nextChannel == nullptr,
        HCCL_ERROR("Ring channels not found for rank[%u]", param.myRank), HCCL_E_INTERNAL);

    for (uint32_t step = 0; step < resCtx.ranksPerServer - 1; ++step) {
        uint32_t sendChunk = 0;
        uint32_t recvChunk = 0;
        if (reduceScatter) {
            sendChunk = (localRank + resCtx.ranksPerServer - step) % resCtx.ranksPerServer;
            recvChunk = (localRank + resCtx.ranksPerServer - step - 1) % resCtx.ranksPerServer;
        } else {
            sendChunk = (localRank + 1 + resCtx.ranksPerServer - step) % resCtx.ranksPerServer;
            recvChunk = (localRank + resCtx.ranksPerServer - step) % resCtx.ranksPerServer;
        }

        const uint64_t sendCount = ChunkCount(param.count, sendChunk, resCtx.ranksPerServer);
        const uint64_t recvCount = ChunkCount(param.count, recvChunk, resCtx.ranksPerServer);
        const uint64_t sendOffset = ChunkOffset(param.count, sendChunk, resCtx.ranksPerServer);
        const uint64_t recvOffset = ChunkOffset(param.count, recvChunk, resCtx.ranksPerServer);
        const uint64_t maxChunkCount = (param.count + resCtx.ranksPerServer - 1) / resCtx.ranksPerServer;
        const uint64_t segmentNum = (maxChunkCount + maxSegmentCount - 1) / maxSegmentCount;

        for (uint64_t segment = 0; segment < segmentNum; ++segment) {
            const uint64_t segmentOffset = segment * maxSegmentCount;
            const uint64_t sendSegmentCount =
                segmentOffset < sendCount ? std::min(maxSegmentCount, sendCount - segmentOffset) : 0;
            const uint64_t recvSegmentCount =
                segmentOffset < recvCount ? std::min(maxSegmentCount, recvCount - segmentOffset) : 0;
            const uint64_t sendElementOffset = sendOffset + segment * maxSegmentCount;
            const uint64_t recvElementOffset = recvOffset + segment * maxSegmentCount;
            const void *sendAddr = static_cast<const uint8_t *>(param.outputPtr) + sendElementOffset * sizeof(float);
            void *recvAddr = static_cast<uint8_t *>(param.outputPtr) + recvElementOffset * sizeof(float);
            CHK_RET(ExchangeSegment(param, resCtx, *prevChannel, *nextChannel, sendAddr, sendSegmentCount,
                recvAddr, recvSegmentCount, reduceScatter));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceAcrossServers(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t ownedChunk,
    uint64_t maxSegmentCount)
{
    const uint32_t localRank = param.myRank % resCtx.ranksPerServer;
    const uint32_t lowRank = localRank;
    const uint32_t highRank = localRank + resCtx.ranksPerServer;
    const uint32_t peerRank = param.myRank == lowRank ? highRank : lowRank;
    const ChannelInfo *channel = FindChannel(resCtx, peerRank);
    CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Cross-server channel to rank[%u] not found", peerRank),
        HCCL_E_INTERNAL);

    ThreadHandle thread = resCtx.aicpuThread;
    const uint64_t count = ChunkCount(param.count, ownedChunk, resCtx.ranksPerServer);
    const uint64_t offset = ChunkOffset(param.count, ownedChunk, resCtx.ranksPerServer);
    for (uint64_t segmentOffset = 0; segmentOffset < count; segmentOffset += maxSegmentCount) {
        const uint64_t segmentCount = std::min(maxSegmentCount, count - segmentOffset);
        const uint64_t bytes = segmentCount * sizeof(float);
        void *chunkAddr = static_cast<uint8_t *>(param.outputPtr) + (offset + segmentOffset) * sizeof(float);
        if (param.myRank == lowRank) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, resCtx.localBuffer.addr, chunkAddr, bytes)));
            CHK_RET(Record(thread, *channel, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(Wait(thread, *channel, NOTIFY_IDX_ACK));
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, chunkAddr, resCtx.localBuffer.addr, bytes)));
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                thread, channel->handle, channel->remoteCclMem.addr, resCtx.localBuffer.addr, bytes)));
            CHK_RET(Record(thread, *channel, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(Wait(thread, *channel, NOTIFY_IDX_ACK));
        } else {
            CHK_RET(Wait(thread, *channel, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, channel->handle,
                channel->remoteCclMem.addr, chunkAddr, segmentCount, static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
            CHK_RET(Record(thread, *channel, NOTIFY_IDX_ACK));
            CHK_RET(Wait(thread, *channel, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, chunkAddr, resCtx.localBuffer.addr, bytes)));
            CHK_RET(Record(thread, *channel, NOTIFY_IDX_ACK));
        }
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.ranksPerServer == 0 || param.rankSize != 2 * resCtx.ranksPerServer,
        HCCL_ERROR("Invalid topology: rankSize[%u], ranksPerServer[%u]", param.rankSize, resCtx.ranksPerServer),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only FP32 SUM is supported"), HCCL_E_INTERNAL);

    const uint64_t maxSegmentBytes = std::min(MAX_PRIMITIVE_BYTES, resCtx.localBuffer.size);
    const uint64_t maxSegmentCount = maxSegmentBytes / sizeof(float);
    CHK_PRT_RET(maxSegmentCount == 0, HCCL_ERROR("HCCL buffer is too small"), HCCL_E_INTERNAL);

    // recvBuf is the stable accumulator throughout the ring and also supports in-place AllReduce.
    for (uint64_t offset = 0; offset < param.count; offset += maxSegmentCount) {
        const uint64_t segmentCount = std::min(maxSegmentCount, param.count - offset);
        const uint64_t bytes = segmentCount * sizeof(float);
        const void *src = static_cast<const uint8_t *>(param.inputPtr) + offset * sizeof(float);
        void *dst = static_cast<uint8_t *>(param.outputPtr) + offset * sizeof(float);
        if (src != dst) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.aicpuThread, dst, src, bytes)));
        }
    }

    CHK_RET(RingPhase(param, resCtx, true, maxSegmentCount));
    const uint32_t localRank = param.myRank % resCtx.ranksPerServer;
    const uint32_t ownedChunk = (localRank + 1) % resCtx.ranksPerServer;
    CHK_RET(ReduceAcrossServers(param, resCtx, ownedChunk, maxSegmentCount));
    CHK_RET(RingPhase(param, resCtx, false, maxSegmentCount));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
