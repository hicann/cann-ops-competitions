/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom.h"
#include "log.h"
#include "exec_op.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace ops_hccl {
namespace {
constexpr uint64_t FLOAT32_BYTES = sizeof(float);

void *AddBytes(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
}

HcclResult StartWorkers(const std::vector<ThreadHandle> &threads, uint32_t workerCount)
{
    CHK_PRT_RET(workerCount == 0 || workerCount > threads.size(),
        HCCL_ERROR("Invalid worker count[%u], thread count[%zu]", workerCount, threads.size()), HCCL_E_PARA);
    for (uint32_t worker = 1; worker < workerCount; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            threads[0], threads[worker], THREAD_START_NOTIFY_IDX)));
    }
    for (uint32_t worker = 1; worker < workerCount; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            threads[worker], THREAD_START_NOTIFY_IDX, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult FinishWorkers(const std::vector<ThreadHandle> &threads, uint32_t workerCount)
{
    CHK_PRT_RET(workerCount == 0 || workerCount > threads.size(),
        HCCL_ERROR("Invalid worker count[%u], thread count[%zu]", workerCount, threads.size()), HCCL_E_PARA);
    for (uint32_t worker = 1; worker < workerCount; ++worker) {
        uint32_t notifyIdx = MAIN_THREAD_FINISH_NOTIFY_BASE + worker - 1;
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            threads[0], notifyIdx, CUSTOM_TIMEOUT)));
    }
    for (uint32_t worker = 1; worker < workerCount; ++worker) {
        uint32_t notifyIdx = MAIN_THREAD_FINISH_NOTIFY_BASE + worker - 1;
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            threads[worker], threads[0], notifyIdx)));
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateResources(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.inputPtr == nullptr || param.outputPtr == nullptr,
        HCCL_ERROR("Input or output pointer is null"), HCCL_E_PTR);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 ||
        param.reduceType != HcclReduceOp::HCCL_REDUCE_SUM,
        HCCL_ERROR("Only float32 sum is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank topology: rank[%u], rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(resCtx.threads.empty() || resCtx.aicpuThread != resCtx.threads[0],
        HCCL_ERROR("Invalid AICPU thread resources"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0,
        HCCL_ERROR("Invalid local CCL buffer"), HCCL_E_INTERNAL);

    uint32_t expectedPeers = param.rankSize - 1;
    uint32_t expectedThreads = std::max(1U, expectedPeers);
    CHK_PRT_RET(resCtx.threads.size() != expectedThreads || resCtx.channels.size() != expectedPeers,
        HCCL_ERROR("Resource count mismatch: threads[%zu], channels[%zu], peers[%u]",
            resCtx.threads.size(), resCtx.channels.size(), expectedPeers),
        HCCL_E_INTERNAL);

    uint32_t channelIdx = 0;
    for (uint32_t expectedRemoteRank = 0; expectedRemoteRank < param.rankSize; ++expectedRemoteRank) {
        if (expectedRemoteRank == param.myRank) {
            continue;
        }
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        CHK_PRT_RET(channel.remoteRank != expectedRemoteRank,
            HCCL_ERROR("Unexpected remote rank[%u], expected[%u]", channel.remoteRank, expectedRemoteRank),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.notifyNum < CHANNEL_NOTIFY_NUM || channel.remoteCclMem.addr == nullptr ||
            channel.remoteCclMem.size == 0,
            HCCL_ERROR("Invalid Channel resource for rank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
        ++channelIdx;
    }
    return HCCL_SUCCESS;
}


const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

HcclResult PairHandshake(ThreadHandle thread, const ChannelInfo &channel, uint32_t notifyIdx)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, channel.handle, notifyIdx)));
    return static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, notifyIdx, CUSTOM_TIMEOUT));
}

void InitWriteDesc(HcommBatchTransferDesc &desc, void *dst, const void *src, uint64_t len)
{
    desc = HcommBatchTransferDesc{};
    desc.transType = HCOMM_TRANSFER_TYPE_WRITE;
    desc.transferInfo.write.len = len;
    desc.transferInfo.write.dst = dst;
    desc.transferInfo.write.src = const_cast<void *>(src);
}

void InitWriteReduceDesc(HcommBatchTransferDesc &desc, void *dst, const void *src, uint64_t count)
{
    desc = HcommBatchTransferDesc{};
    desc.transType = HCOMM_TRANSFER_TYPE_WRITE_REDUCE;
    desc.transferInfo.reduce.count = count;
    desc.transferInfo.reduce.dst = dst;
    desc.transferInfo.reduce.src = const_cast<void *>(src);
    desc.transferInfo.reduce.reduceOp = HCOMM_REDUCE_SUM;
    desc.transferInfo.reduce.dataType = HCOMM_DATA_TYPE_FP32;
}

void InitReadReduceDesc(HcommBatchTransferDesc &desc, void *dst, const void *src, uint64_t count)
{
    desc = HcommBatchTransferDesc{};
    desc.transType = HCOMM_TRANSFER_TYPE_READ_REDUCE;
    desc.transferInfo.reduce.count = count;
    desc.transferInfo.reduce.dst = dst;
    desc.transferInfo.reduce.src = const_cast<void *>(src);
    desc.transferInfo.reduce.reduceOp = HCOMM_REDUCE_SUM;
    desc.transferInfo.reduce.dataType = HCOMM_DATA_TYPE_FP32;
}

void InitNotifyRecordDesc(HcommBatchTransferDesc &desc, uint32_t notifyIdx)
{
    desc = HcommBatchTransferDesc{};
    desc.transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    desc.transferInfo.notifyRecord.notifyIdx = notifyIdx;
}

HcclResult SubmitWriteAndNotify(ThreadHandle thread, const ChannelInfo &channel,
    void *dst, const void *src, uint64_t len, uint32_t notifyIdx)
{
    HcommBatchTransferDesc descs[2] = {};
    InitWriteDesc(descs[0], dst, src, len);
    InitNotifyRecordDesc(descs[1], notifyIdx);
    HcclResult ret = static_cast<HcclResult>(
        HcommBatchTransferOnThread(thread, channel.handle, descs, 2));
    if (ret != HCCL_E_NOT_SUPPORT) {
        return ret;
    }

    CHK_RET(static_cast<HcclResult>(
        HcommWriteOnThread(thread, channel.handle, dst, src, len)));
    return static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyIdx));
}

HcclResult SubmitWriteReduceAndNotify(ThreadHandle thread, const ChannelInfo &channel,
    void *dst, const void *src, uint64_t count, uint32_t notifyIdx)
{
    HcommBatchTransferDesc descs[2] = {};
    InitWriteReduceDesc(descs[0], dst, src, count);
    InitNotifyRecordDesc(descs[1], notifyIdx);
    HcclResult ret = static_cast<HcclResult>(
        HcommBatchTransferOnThread(thread, channel.handle, descs, 2));
    if (ret != HCCL_E_NOT_SUPPORT) {
        return ret;
    }

    CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(
        thread, channel.handle, dst, src, count,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyIdx));
}

HcclResult SubmitReadReduceAndNotify(ThreadHandle thread, const ChannelInfo &channel,
    void *dst, const void *src, uint64_t count, uint32_t notifyIdx)
{
    HcommBatchTransferDesc descs[2] = {};
    InitReadReduceDesc(descs[0], dst, src, count);
    InitNotifyRecordDesc(descs[1], notifyIdx);
    HcclResult ret = static_cast<HcclResult>(
        HcommBatchTransferOnThread(thread, channel.handle, descs, 2));
    if (ret != HCCL_E_NOT_SUPPORT) {
        return ret;
    }

    CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(
        thread, channel.handle, dst, src, count,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyIdx));
}

// Safe fallback for all non-special shapes.  The aggressive paths below are
// guarded by exact rank/size predicates; functional cases retain V14's
// verified full-mesh write exchange and local tree.
HcclResult ExchangeChunkFull(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t outputBytes, uint64_t processedBytes, uint64_t chunkBytes, uint64_t slotStride)
{
    const uint64_t sendSlotCount = param.rankSize - 1;
    const uint64_t receiveBaseOffset = sendSlotCount * slotStride;
    void *localContribution = AddBytes(resCtx.localBuffer.addr,
        receiveBaseOffset + static_cast<uint64_t>(param.myRank) * slotStride);
    const void *localInput = AddBytes(param.inputPtr,
        static_cast<uint64_t>(param.myRank) * outputBytes + processedBytes);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[0], localContribution, localInput, chunkBytes)));
    const uint32_t workerCount = static_cast<uint32_t>(resCtx.channels.size());
    CHK_RET(StartWorkers(resCtx.threads, workerCount));
    for (uint32_t idx = 0; idx < workerCount; ++idx) {
        const ChannelInfo &channel = resCtx.channels[idx];
        ThreadHandle thread = resCtx.threads[idx];
        void *staging = AddBytes(resCtx.localBuffer.addr, static_cast<uint64_t>(idx) * slotStride);
        const void *source = AddBytes(param.inputPtr,
            static_cast<uint64_t>(channel.remoteRank) * outputBytes + processedBytes);
        void *remoteDestination = AddBytes(channel.remoteCclMem.addr,
            receiveBaseOffset + static_cast<uint64_t>(param.myRank) * slotStride);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, staging, source, chunkBytes)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            thread, channel.handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
            thread, channel.handle, remoteDestination, staging, chunkBytes)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return FinishWorkers(resCtx.threads, workerCount);
}

/*
 * Four-stage in-place recursive-halving ReduceScatter for the exact 512KiB
 * case.  At every stage a rank reduces into the half it keeps, while its
 * peer reads the disjoint half this rank discards.  One readiness handshake
 * therefore suffices between stages; only the final stage needs a completion
 * handshake before the operator may release the CCL source buffer.
 */
HcclResult ReduceSmallRecursiveHalving16(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t outputBytes)
{
    CHK_PRT_RET(param.rankSize != V15_RANKS || outputBytes != V6_READ_SMALL_OUTPUT_BYTES ||
        outputBytes * param.rankSize != V15_SMALL_INPUT_BYTES,
        HCCL_ERROR("Invalid recursive-halving small shape"), HCCL_E_PARA);
    const uint64_t totalBytes = outputBytes * param.rankSize;
    CHK_PRT_RET(totalBytes > resCtx.localBuffer.size,
        HCCL_ERROR("Small recursive-halving buffer exceeds CCL memory"), HCCL_E_INTERNAL);

    ThreadHandle thread = resCtx.threads[0];
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        thread, resCtx.localBuffer.addr, param.inputPtr, totalBytes)));

    uint64_t currentOffset = 0;
    uint64_t currentBytes = totalBytes;
    for (uint32_t mask = V15_SERVER_RANKS; mask != 0; mask >>= 1) {
        const uint32_t peerRank = param.myRank ^ mask;
        const ChannelInfo *peer = FindChannel(resCtx, peerRank);
        CHK_PRT_RET(peer == nullptr, HCCL_ERROR("Missing recursive-halving peer"), HCCL_E_INTERNAL);
        const uint64_t halfBytes = currentBytes / 2;
        const bool keepUpper = (param.myRank & mask) != 0;
        const uint64_t keepOffset = keepUpper ? halfBytes : 0;

        CHK_RET(PairHandshake(thread, *peer, NOTIFY_IDX_ACK));
        void *localKeep = AddBytes(resCtx.localBuffer.addr, currentOffset + keepOffset);
        const void *remoteSource = AddBytes(peer->remoteCclMem.addr, currentOffset + keepOffset);
        if (mask == 1) {
            // The final ReadReduce and its source-lifetime completion Record
            // share one data-plane submission.  The peer Wait is retained, so
            // neither rank can release CCL memory while the opposite read is
            // still in flight.
            CHK_RET(SubmitReadReduceAndNotify(
                thread, *peer, localKeep, remoteSource,
                halfBytes / FLOAT32_BYTES, NOTIFY_IDX_DATA_SIGNAL));
        } else {
            CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(
                thread, peer->handle, localKeep, remoteSource, halfBytes / FLOAT32_BYTES,
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
        // The next stage's readiness handshake proves completion of every
        // previous stage.  The last stage has no successor, so close it here.
        if (mask == 1) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, peer->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }

        currentOffset += keepOffset;
        currentBytes = halfBytes;
    }
    return static_cast<HcclResult>(HcommLocalCopyOnThread(
        thread, param.outputPtr, AddBytes(resCtx.localBuffer.addr, currentOffset), outputBytes));
}

uint64_t V19BufferBase(uint32_t bufferIndex, uint64_t slotStride)
{
    return static_cast<uint64_t>(bufferIndex) * V19_SLOTS_PER_BUFFER * slotStride;
}

uint32_t V22TreeRootIndex(uint32_t bufferIndex)
{
    return bufferIndex % 2;
}

uint32_t V22TreePair23Index(uint32_t bufferIndex)
{
    return 2 + bufferIndex % 2;
}

uint32_t V22TreePair45Index(uint32_t bufferIndex)
{
    return 4 + bufferIndex % 2;
}

uint32_t V22TreePair01Notify(uint32_t bufferIndex)
{
    return V22_TREE_ROOT_NOTIFY_BASE +
        (bufferIndex / 2) * V22_TREE_ROOT_NOTIFY_STRIDE;
}

uint32_t V22TreeLevel2Notify(uint32_t bufferIndex)
{
    return V22TreePair01Notify(bufferIndex) + 1;
}

uint32_t V22TreeLevel3Notify(uint32_t bufferIndex)
{
    return V22TreePair01Notify(bufferIndex) + 2;
}

uint32_t V22TreeSelfNotify(uint32_t bufferIndex)
{
    return V22TreePair01Notify(bufferIndex) + 3;
}

uint32_t V22TreePair23Notify(uint32_t bufferIndex)
{
    return V22_TREE_PAIR23_NOTIFY_BASE + bufferIndex / 2;
}

uint32_t V22TreePair45Notify(uint32_t bufferIndex)
{
    return V22_TREE_PAIR45_NOTIFY_BASE +
        (bufferIndex / 2) * V22_TREE_PAIR45_NOTIFY_STRIDE;
}

uint32_t V22TreeLeaf6Notify(uint32_t bufferIndex)
{
    return V22TreePair45Notify(bufferIndex) + 1;
}

HcclResult ScheduleV19Pack(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t outputBytes, uint64_t processedBytes, uint64_t chunkBytes,
    uint64_t slotStride, uint32_t bufferIndex, bool waitReuse)
{
    const uint32_t localIndex = param.myRank % V15_SERVER_RANKS;
    const uint32_t localBase = param.myRank - localIndex;
    const uint64_t bufferBase = V19BufferBase(bufferIndex, slotStride);
    const ThreadHandle crossThread = resCtx.threads[V19_CROSS_THREAD_INDEX];

    if (waitReuse) {
        for (uint32_t p = 0; p < V19_PACK_THREAD_COUNT; ++p) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[V19_PACK_THREAD_BASE + p],
                V22_BUFFER_REUSE_NOTIFY_BASE + bufferIndex, CUSTOM_TIMEOUT)));
        }
    }

    // Only the eight local-server destination blocks need CCL initialization.
    // The other server's blocks remain in user input and are consumed directly
    // by WriteReduce in ScheduleV19Cross.
    uint32_t task = 0;
    for (uint32_t d = 0; d < V15_SERVER_RANKS; ++d) {
        ThreadHandle thread = resCtx.threads[V19_PACK_THREAD_BASE + task % V19_PACK_THREAD_COUNT];
        void *accumulator = AddBytes(resCtx.localBuffer.addr,
            bufferBase + static_cast<uint64_t>(V19_ACCUMULATOR_SLOT_BASE + d) * slotStride);
        const void *source = AddBytes(param.inputPtr,
            static_cast<uint64_t>(localBase + d) * outputBytes + processedBytes);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            thread, accumulator, source, chunkBytes)));
        ++task;
    }
    CHK_PRT_RET(task != V19_PACK_SLOT_COUNT,
        HCCL_ERROR("V24 initializer task count mismatch"), HCCL_E_INTERNAL);

    const uint32_t pairNotify = V22_PACK_PAIR_NOTIFY_BASE + bufferIndex;
    const uint32_t level2Notify = V22_PACK_LEVEL2_NOTIFY_BASE + bufferIndex;
    const uint32_t level3Notify = V22_PACK_LEVEL3_NOTIFY_BASE + bufferIndex;
    const ThreadHandle pack0 = resCtx.threads[V19_PACK_THREAD_BASE];
    const ThreadHandle pack2 = resCtx.threads[V19_PACK_THREAD_BASE + 2];
    const ThreadHandle pack4 = resCtx.threads[V19_PACK_THREAD_BASE + 4];

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[V19_PACK_THREAD_BASE + 1], pack0, pairNotify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[V19_PACK_THREAD_BASE + 3], pack2, pairNotify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[V19_PACK_THREAD_BASE + 5], pack4, pairNotify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[V19_PACK_THREAD_BASE + 6], pack4, level2Notify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        pack0, pairNotify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        pack2, pairNotify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        pack4, pairNotify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        pack4, level2Notify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        pack2, pack0, level2Notify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        pack4, pack0, level3Notify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        pack0, level2Notify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        pack0, level3Notify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        pack0, crossThread, V22_PACK_ROOT_NOTIFY_BASE + bufferIndex)));
    return HCCL_SUCCESS;
}

HcclResult ScheduleV19Cross(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t processedBytes, uint64_t chunkBytes,
    uint64_t slotStride, uint32_t bufferIndex)
{
    const uint32_t localIndex = param.myRank % V15_SERVER_RANKS;
    const uint32_t localBase = param.myRank - localIndex;
    const uint32_t remoteBase = V15_SERVER_RANKS - localBase;
    const uint64_t outputBytes = param.count * FLOAT32_BYTES;
    const uint32_t mateRank = param.myRank ^ V15_SERVER_RANKS;
    const uint64_t bufferBase = V19BufferBase(bufferIndex, slotStride);
    const uint64_t elementCount = chunkBytes / FLOAT32_BYTES;
    const ThreadHandle crossThread = resCtx.threads[V19_CROSS_THREAD_INDEX];
    const ChannelInfo *mate = FindChannel(resCtx, mateRank);
    CHK_PRT_RET(mate == nullptr, HCCL_ERROR("Missing V19 cross-server mate"), HCCL_E_INTERNAL);

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        crossThread, V22_PACK_ROOT_NOTIFY_BASE + bufferIndex, CUSTOM_TIMEOUT)));

    CHK_RET(PairHandshake(crossThread, *mate, NOTIFY_IDX_ACK));

    // Both mate ranks initialize their own-server accumulators, then push the
    // opposite-server input slices directly into the mate's accumulators.
    // A distinct channel notify releases every completed destination without
    // a CCL publish slot, a ReadReduce landing task, or a per-block round trip.
    // V26 submits each WriteReduce/Record pair as a two-descriptor batch.
    // Keeping exactly one completion Record per BatchTransfer is required by
    // CheckerV3's Record/Wait expansion.  It also preserves V24's
    // per-destination wavefront while halving cross-stream AICPU API calls.
    // The self destination remains last so the seventh layer-0 write overlaps
    // the eighth layer-1 WriteReduce.
    for (uint32_t offset = 1; offset <= V15_SERVER_RANKS; ++offset) {
        const uint32_t d = (localIndex + offset) % V15_SERVER_RANKS;
        void *remoteDestination = AddBytes(mate->remoteCclMem.addr,
            bufferBase + static_cast<uint64_t>(V19_ACCUMULATOR_SLOT_BASE + d) * slotStride);
        const void *source = AddBytes(param.inputPtr,
            static_cast<uint64_t>(remoteBase + d) * outputBytes + processedBytes);
        const uint32_t dataNotify = V24_CROSS_DATA_NOTIFY_BASE +
            bufferIndex * V24_CROSS_DATA_NOTIFY_PER_BUFFER + d;
        CHK_RET(SubmitWriteReduceAndNotify(
            crossThread, *mate, remoteDestination, source, elementCount,
            dataNotify));
    }

    const uint32_t selfDataNotify = V24_CROSS_DATA_NOTIFY_BASE +
        bufferIndex * V24_CROSS_DATA_NOTIFY_PER_BUFFER + localIndex;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        crossThread, mate->handle, selfDataNotify, CUSTOM_TIMEOUT)));
    void *selfAccumulator = AddBytes(resCtx.localBuffer.addr,
        bufferBase + static_cast<uint64_t>(V19_ACCUMULATOR_SLOT_BASE + localIndex) * slotStride);
    void *output = AddBytes(param.outputPtr, processedBytes);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        crossThread, output, selfAccumulator, chunkBytes)));
    const uint32_t treeRoot = V22TreeRootIndex(bufferIndex);
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        crossThread, resCtx.threads[treeRoot], V22TreeSelfNotify(bufferIndex))));
    return HCCL_SUCCESS;
}

HcclResult ScheduleV21ArrivalSeven(const AlgResourceCtx &resCtx, void *output,
    uint64_t bufferBase, uint64_t slotStride, uint64_t chunkBytes, uint32_t bufferIndex)
{
    const uint64_t elementCount = chunkBytes / FLOAT32_BYTES;
    auto slot = [&](uint32_t index) -> void * {
        return AddBytes(resCtx.localBuffer.addr,
            bufferBase + static_cast<uint64_t>(V19_INCOMING_SLOT_BASE + index) * slotStride);
    };
    const uint32_t rootIndex = V22TreeRootIndex(bufferIndex);
    const uint32_t pair23Index = V22TreePair23Index(bufferIndex);
    const uint32_t pair45Index = V22TreePair45Index(bufferIndex);
    const uint32_t pair01PeerIndex = 1 - rootIndex;
    const uint32_t pair23PeerIndex = 5 - pair23Index;
    const uint32_t pair45PeerIndex = 9 - pair45Index;
    const uint32_t pair01Notify = V22TreePair01Notify(bufferIndex);
    const uint32_t level2Notify = V22TreeLevel2Notify(bufferIndex);
    const uint32_t level3Notify = V22TreeLevel3Notify(bufferIndex);
    const uint32_t selfNotify = V22TreeSelfNotify(bufferIndex);
    const uint32_t pair23Notify = V22TreePair23Notify(bufferIndex);
    const uint32_t pair45Notify = V22TreePair45Notify(bufferIndex);
    const uint32_t leaf6Notify = V22TreeLeaf6Notify(bufferIndex);
    const ThreadHandle rootThread = resCtx.threads[rootIndex];
    const ThreadHandle pair23Thread = resCtx.threads[pair23Index];
    const ThreadHandle pair45Thread = resCtx.threads[pair45Index];

    // Build the seven incoming leaves immediately as they arrive. Four exact
    // tokens remove V20's 4->6->4 round trip while retaining a balanced
    // three-level incoming tree:
    //
    // level 1: (0,1) (2,3) (4,5)
    // level 2: (0,2) (4,6)
    // level 3: (0,4)
    // final:   (output,self + incoming-tree)
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[pair01PeerIndex], rootThread, pair01Notify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[pair23PeerIndex], pair23Thread, pair23Notify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[pair45PeerIndex], pair45Thread, pair45Notify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[6], pair45Thread, leaf6Notify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        rootThread, pair01Notify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        pair23Thread, pair23Notify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        pair45Thread, pair45Notify, CUSTOM_TIMEOUT)));

    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        rootThread, slot(rootIndex), slot(pair01PeerIndex), elementCount,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        pair23Thread, slot(pair23Index), slot(pair23PeerIndex), elementCount,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        pair45Thread, slot(pair45Index), slot(pair45PeerIndex), elementCount,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        pair23Thread, rootThread, level2Notify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        rootThread, level2Notify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        pair45Thread, leaf6Notify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        rootThread, slot(rootIndex), slot(pair23Index), elementCount,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        pair45Thread, slot(pair45Index), slot(6), elementCount,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        pair45Thread, rootThread, level3Notify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        rootThread, level3Notify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        rootThread, slot(rootIndex), slot(pair45Index), elementCount,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        rootThread, selfNotify, CUSTOM_TIMEOUT)));
    return static_cast<HcclResult>(HcommLocalReduceOnThread(
        rootThread, output, slot(rootIndex), elementCount,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
}

HcclResult ScheduleV19IntraAndReduce(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t processedBytes, uint64_t chunkBytes, uint64_t slotStride,
    uint32_t bufferIndex, bool signalReuse)
{
    const uint32_t localIndex = param.myRank % V15_SERVER_RANKS;
    const uint32_t localBase = param.myRank - localIndex;
    const uint32_t mateRank = param.myRank ^ V15_SERVER_RANKS;
    const ChannelInfo *mate = FindChannel(resCtx, mateRank);
    CHK_PRT_RET(mate == nullptr, HCCL_ERROR("Missing V24 cross-server mate"), HCCL_E_INTERNAL);
    const uint64_t bufferBase = V19BufferBase(bufferIndex, slotStride);
    uint32_t worker = 0;
    for (uint32_t d = 0; d < V15_SERVER_RANKS; ++d) {
        if (d == localIndex) {
            continue;
        }
        const uint32_t peerRank = localBase + d;
        const ChannelInfo *peer = FindChannel(resCtx, peerRank);
        CHK_PRT_RET(peer == nullptr, HCCL_ERROR("Missing V19 intra-server peer"), HCCL_E_INTERNAL);
        ThreadHandle thread = resCtx.threads[worker];
        const uint32_t dataNotify = V24_CROSS_DATA_NOTIFY_BASE +
            bufferIndex * V24_CROSS_DATA_NOTIFY_PER_BUFFER + d;
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, mate->handle, dataNotify, CUSTOM_TIMEOUT)));
        const void *partial = AddBytes(resCtx.localBuffer.addr,
            bufferBase + static_cast<uint64_t>(V19_ACCUMULATOR_SLOT_BASE + d) * slotStride);
        const uint32_t remoteIncomingIndex = localIndex < d ? localIndex : localIndex - 1;
        void *remoteReceive = AddBytes(peer->remoteCclMem.addr,
            bufferBase + static_cast<uint64_t>(V19_INCOMING_SLOT_BASE + remoteIncomingIndex) * slotStride);
        CHK_RET(PairHandshake(thread, *peer, NOTIFY_IDX_ACK));
        CHK_RET(SubmitWriteAndNotify(
            thread, *peer, remoteReceive, partial, chunkBytes,
            NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, peer->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        ++worker;
    }

    void *output = AddBytes(param.outputPtr, processedBytes);
    CHK_RET(ScheduleV21ArrivalSeven(
        resCtx, output, bufferBase, slotStride, chunkBytes, bufferIndex));

    if (signalReuse) {
        const ThreadHandle rootThread = resCtx.threads[V22TreeRootIndex(bufferIndex)];
        for (uint32_t p = 0; p < V19_PACK_THREAD_COUNT; ++p) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                rootThread, resCtx.threads[V19_PACK_THREAD_BASE + p],
                V22_BUFFER_REUSE_NOTIFY_BASE + bufferIndex)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceLargePipelinedV19(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t outputBytes, uint64_t minCclBytes)
{
    CHK_PRT_RET(param.rankSize != V15_RANKS || resCtx.threads.size() < V15_RANKS - 1,
        HCCL_ERROR("V26 batched direct pipeline requires 16 ranks and 15 threads"), HCCL_E_PARA);
    const uint32_t targetStripeCount = outputBytes >= V19_POINT6_OUTPUT_BYTES ?
        V22_POINT6_STRIPE_COUNT : V22_POINT7_STRIPE_COUNT;
    CHK_PRT_RET(targetStripeCount == 0 || targetStripeCount > V22_MAX_BUFFER_COUNT,
        HCCL_ERROR("Invalid V26 stripe count"), HCCL_E_INTERNAL);
    const uint64_t targetStripeBytes =
        ((outputBytes / FLOAT32_BYTES + targetStripeCount - 1) / targetStripeCount) * FLOAT32_BYTES;
    const uint64_t capacityBufferCount =
        minCclBytes / (static_cast<uint64_t>(V19_SLOTS_PER_BUFFER) * targetStripeBytes);
    const uint32_t bufferCount = static_cast<uint32_t>(std::min(
        static_cast<uint64_t>(targetStripeCount),
        std::max(static_cast<uint64_t>(V22_FALLBACK_BUFFER_COUNT), capacityBufferCount)));
    const uint64_t totalSlotCount = static_cast<uint64_t>(bufferCount) * V19_SLOTS_PER_BUFFER;
    const uint64_t slotStride =
        (minCclBytes / totalSlotCount / CCL_SLOT_ALIGNMENT) * CCL_SLOT_ALIGNMENT;
    CHK_PRT_RET(slotStride < FLOAT32_BYTES,
        HCCL_ERROR("V24 direct pipeline slot stride too small"), HCCL_E_INTERNAL);
    const uint64_t capacityStripeCount = (outputBytes + slotStride - 1) / slotStride;
    const uint64_t chunkCount =
        std::max(static_cast<uint64_t>(targetStripeCount), capacityStripeCount);

    // Make every worker reachable from the AICPU main stream before the
    // stripe DAG is submitted.  The data path has its own exact dependencies,
    // but HCOMM also requires this one-time stream activation edge.
    CHK_RET(StartWorkers(resCtx.threads, V15_RANKS - 1));

    uint64_t processedBytes = 0;
    for (uint64_t index = 0; index < chunkCount; ++index) {
        const uint64_t chunksLeft = chunkCount - index;
        const uint64_t remainingElements = (outputBytes - processedBytes) / FLOAT32_BYTES;
        const uint64_t chunkElements = (remainingElements + chunksLeft - 1) / chunksLeft;
        const uint64_t chunkBytes = chunkElements * FLOAT32_BYTES;
        CHK_PRT_RET(chunkBytes == 0 || chunkBytes > slotStride,
            HCCL_ERROR("Invalid V24 direct stripe"), HCCL_E_INTERNAL);
        const uint32_t bufferIndex = static_cast<uint32_t>(index % bufferCount);
        CHK_RET(ScheduleV19Pack(param, resCtx, outputBytes, processedBytes, chunkBytes,
            slotStride, bufferIndex, index >= bufferCount));
        CHK_RET(ScheduleV19Cross(param, resCtx, processedBytes, chunkBytes,
            slotStride, bufferIndex));
        CHK_RET(ScheduleV19IntraAndReduce(param, resCtx, processedBytes, chunkBytes,
            slotStride, bufferIndex, index + bufferCount < chunkCount));
        processedBytes += chunkBytes;
    }
    CHK_PRT_RET(processedBytes != outputBytes,
        HCCL_ERROR("V24 direct stripe plan does not cover output"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceChunkSerial(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t processedBytes, uint64_t chunkBytes, uint64_t slotStride, uint64_t receiveBaseOffset)
{
    const uint64_t chunkCount = chunkBytes / FLOAT32_BYTES;
    for (uint32_t span = 1; span < param.rankSize; span *= 2) {
        for (uint32_t base = 0; base + span < param.rankSize; base += span * 2) {
            void *destination = AddBytes(resCtx.localBuffer.addr,
                receiveBaseOffset + static_cast<uint64_t>(base) * slotStride);
            const void *source = AddBytes(resCtx.localBuffer.addr,
                receiveBaseOffset + static_cast<uint64_t>(base + span) * slotStride);
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[0], destination, source, chunkCount,
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
    }
    void *output = AddBytes(param.outputPtr, processedBytes);
    const void *result = AddBytes(resCtx.localBuffer.addr, receiveBaseOffset);
    return static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[0], output, result, chunkBytes));
}

HcclResult ReduceChunkParallel(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t processedBytes, uint64_t chunkBytes, uint64_t slotStride, uint64_t receiveBaseOffset)
{
    const uint64_t chunkCount = chunkBytes / FLOAT32_BYTES;
    for (uint32_t span = 1; span < param.rankSize; span *= 2) {
        uint32_t taskCount = 0;
        for (uint32_t base = 0; base + span < param.rankSize; base += span * 2) {
            ++taskCount;
        }
        CHK_RET(StartWorkers(resCtx.threads, taskCount));
        uint32_t task = 0;
        for (uint32_t base = 0; base + span < param.rankSize; base += span * 2) {
            void *destination = AddBytes(resCtx.localBuffer.addr,
                receiveBaseOffset + static_cast<uint64_t>(base) * slotStride);
            const void *source = AddBytes(resCtx.localBuffer.addr,
                receiveBaseOffset + static_cast<uint64_t>(base + span) * slotStride);
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[task], destination, source, chunkCount,
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            ++task;
        }
        CHK_RET(FinishWorkers(resCtx.threads, taskCount));
    }
    void *output = AddBytes(param.outputPtr, processedBytes);
    const void *result = AddBytes(resCtx.localBuffer.addr, receiveBaseOffset);
    return static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[0], output, result, chunkBytes));
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_RET(ValidateResources(param, resCtx));
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / FLOAT32_BYTES,
        HCCL_ERROR("Output byte count overflow"), HCCL_E_PARA);
    const uint64_t outputBytes = param.count * FLOAT32_BYTES;
    CHK_PRT_RET(outputBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("Input byte count overflow"), HCCL_E_PARA);

    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(
            resCtx.threads[0], param.outputPtr, param.inputPtr, outputBytes));
    }

    uint64_t minCclBytes = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        minCclBytes = std::min(minCclBytes, channel.remoteCclMem.size);
    }
    const uint64_t fullSendSlotCount = param.rankSize - 1;
    const uint64_t fullSlotCount = static_cast<uint64_t>(param.rankSize) + fullSendSlotCount;
    const uint64_t fullSlotStride =
        (minCclBytes / fullSlotCount / CCL_SLOT_ALIGNMENT) * CCL_SLOT_ALIGNMENT;
    const uint64_t slotStride = fullSlotStride;
    CHK_PRT_RET(slotStride < FLOAT32_BYTES,
        HCCL_ERROR("CCL buffer too small: minBytes[%llu], slots[%llu]",
            static_cast<unsigned long long>(minCclBytes), static_cast<unsigned long long>(fullSlotCount)),
        HCCL_E_INTERNAL);
    if (param.rankSize == V15_RANKS && outputBytes == V6_READ_SMALL_OUTPUT_BYTES) {
        return ReduceSmallRecursiveHalving16(param, resCtx, outputBytes);
    }
    if (param.rankSize == V15_RANKS && outputBytes >= V15_LARGE_OUTPUT_MIN_BYTES) {
        return ReduceLargePipelinedV19(param, resCtx, outputBytes, minCclBytes);
    }
    for (uint64_t processedBytes = 0; processedBytes < outputBytes;) {
        const uint64_t chunkBytes = std::min(slotStride, outputBytes - processedBytes);
        const uint64_t receiveBaseOffset = fullSendSlotCount * slotStride;
        CHK_RET(ExchangeChunkFull(
            param, resCtx, outputBytes, processedBytes, chunkBytes, slotStride));
        if (chunkBytes <= SERIAL_REDUCE_THRESHOLD_BYTES) {
            CHK_RET(ReduceChunkSerial(
                param, resCtx, processedBytes, chunkBytes, slotStride, receiveBaseOffset));
        } else {
            CHK_RET(ReduceChunkParallel(
                param, resCtx, processedBytes, chunkBytes, slotStride, receiveBaseOffset));
        }
        processedBytes += chunkBytes;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
