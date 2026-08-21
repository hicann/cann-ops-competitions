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
#include <vector>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint32_t COMPETITION_RANK_SIZE = 16;
constexpr uint32_t COMPETITION_XOR_STAGE_COUNT = 4;
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint64_t ELEMENT_BYTES = sizeof(float);
constexpr uint64_t CHUNK_ALIGN_COUNT = 4096; // FP32 下为 16 KiB。
constexpr uint32_t DIRECT_PUSH_STRIPE_COUNT = 3;
constexpr uint64_t FIRST_STRIPE_NUMERATOR = 4;
constexpr uint64_t SECOND_STRIPE_NUMERATOR = 11;
constexpr uint64_t STRIPE_DENOMINATOR = 32;
constexpr uint64_t PACKED_FIRST_STRIPE_NUMERATOR = 9;
constexpr uint64_t PACKED_SECOND_STRIPE_NUMERATOR = 13;
constexpr uint64_t PACKED_THIRD_STRIPE_NUMERATOR = 10;
constexpr uint32_t REFERENCE_V2_STRIPE_COUNT = 4;
constexpr uint64_t REFERENCE_V2_STRIPE_NUMERATORS[REFERENCE_V2_STRIPE_COUNT] = {
    4, 8, 10, 10};
constexpr uint32_t LOCAL_WORKER_COUNT = RANKS_PER_SERVER - 1;
constexpr uint32_t CROSS_WORKER_SLOT = LOCAL_WORKER_COUNT;
constexpr uint32_t NHR_READ_BATCH_CAPACITY = COMPETITION_RANK_SIZE / 2;
constexpr uint32_t NHR_BATCH_CAPACITY = NHR_READ_BATCH_CAPACITY + 1;
constexpr uint32_t EAGER_NOTIFY_STRIPE0 = 0;
constexpr uint32_t EAGER_NOTIFY_STRIPE1 = 1;
constexpr uint32_t EAGER_NOTIFY_STRIPE2 = 2;
constexpr uint32_t EAGER_NOTIFY_FINAL_ACK = 3;
constexpr uint32_t PACKED_NOTIFY_STRIPE0 = 0;
constexpr uint32_t PACKED_NOTIFY_STRIPE1 = 1;
constexpr uint32_t PACKED_NOTIFY_REUSE_PERMIT = 2;
constexpr uint32_t PACKED_NOTIFY_STRIPE2 = 3;
constexpr uint32_t REFERENCE_NOTIFY_STRIPE3 = 3;
constexpr uint32_t REDUCE_GROUP_BEGIN[RS_TOTAL_REDUCE_GROUPS + 1] = {
    0, RS_OUTPUT_ROOT_REMOTE_COUNT, 7, 11, 15};

const char *GetAlgorithmName()
{
#if HCCL_RS_CANDIDATE_MODE == RS_CANDIDATE_FLATTEN
    return "integrated-" HCCL_RS_VARIANT_TAG "-persistent-handoff";
#elif HCCL_RS_CANDIDATE_MODE == RS_CANDIDATE_PAIR_MESH
    return "adaptive-pair-mesh";
#else
    return "adaptive-dual-path-3of8";
#endif
}

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const auto &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

uint32_t GetNhrStepNum(uint32_t rankSize)
{
    uint32_t steps = 0;
    for (uint32_t value = rankSize - 1; value != 0; value >>= 1) {
        ++steps;
    }
    return steps;
}

uint64_t GetInputBytes(const OpParam &param)
{
    return param.count * ELEMENT_BYTES * param.rankSize;
}

bool UseNhrPath(const OpParam &param)
{
    return param.rankSize != COMPETITION_RANK_SIZE;
}

bool UseTopologyXorPath(const OpParam &param)
{
    const uint64_t inputBytes = GetInputBytes(param);
#if HCCL_RS_CANDIDATE_MODE != RS_CANDIDATE_FLATTEN
    // 两个分层候选专门针对赛题 2×8 Rank；其它通信域保持 NHR 泛化路径。
    return false;
#else
    return param.rankSize == COMPETITION_RANK_SIZE &&
        inputBytes < RS_NHR_INPUT_LIMIT_BYTES;
#endif
}

bool UseTinySpecialPath(const OpParam &param)
{
    return HCCL_RS_TINY_MODE != RS_TINY_MODE_XOR_HALF_COPY &&
        param.rankSize == COMPETITION_RANK_SIZE && param.count == 1;
}

bool UsePackedStripePath(const OpParam &param)
{
    return param.rankSize == COMPETITION_RANK_SIZE &&
        GetInputBytes(param) > RS_PACKED_STRIPE_INPUT_MIN_BYTES;
}

bool UseReferenceV2SingleOuterPath(const OpParam &param)
{
    const uint64_t inputBytes = GetInputBytes(param);
    return param.rankSize == COMPETITION_RANK_SIZE &&
        inputBytes >= RS_REFERENCE_V2_INPUT_MIN_BYTES &&
        inputBytes <= RS_PACKED_STRIPE_INPUT_MIN_BYTES;
}

uint64_t GetUsableBufferBytes(const AlgResourceCtx &resCtx)
{
    uint64_t usableBytes = resCtx.localBuffer.size;
    for (const auto &channel : resCtx.channels) {
        if (channel.remoteCclMem.addr == nullptr) {
            return 0;
        }
        usableBytes = std::min(usableBytes, channel.remoteCclMem.size);
    }
    return usableBytes;
}

uint64_t AlignChunkCount(uint64_t count)
{
    if (count < CHUNK_ALIGN_COUNT) {
        return count;
    }
    return count / CHUNK_ALIGN_COUNT * CHUNK_ALIGN_COUNT;
}

HcclResult BeginPeerRead(ThreadHandle thread, const ChannelInfo &sendChannel,
    const ChannelInfo &recvChannel)
{
    CHK_PRT_RET(sendChannel.notifyNum <= NOTIFY_IDX_DATA_SIGNAL ||
            recvChannel.notifyNum <= NOTIFY_IDX_DATA_SIGNAL,
        HCCL_ERROR("peer channels require two notify slots"), HCCL_E_INTERNAL);
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, recvChannel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult EndPeerRead(ThreadHandle thread, const ChannelInfo &sendChannel,
    const ChannelInfo &recvChannel)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult StartWorkers(const std::vector<ThreadHandle> &threads,
    const std::vector<uint32_t> &workerSlots)
{
    CHK_PRT_RET(threads.empty(), HCCL_ERROR("no main thread"), HCCL_E_INTERNAL);
    const ThreadHandle mainThread = threads[0];
    for (const uint32_t slot : workerSlots) {
        CHK_PRT_RET(slot + 1 >= threads.size(),
            HCCL_ERROR("worker slot[%u] exceeds thread count[%zu]", slot, threads.size()),
            HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            mainThread, threads[slot + 1], 0)));
    }
    for (const uint32_t slot : workerSlots) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            threads[slot + 1], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult FinishWorkers(const std::vector<ThreadHandle> &threads,
    const std::vector<uint32_t> &workerSlots)
{
    const ThreadHandle mainThread = threads[0];
    for (const uint32_t slot : workerSlots) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            threads[slot + 1], mainThread, slot)));
    }
    for (const uint32_t slot : workerSlots) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            mainThread, slot, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult CopyInputSlices(const OpParam &param, ThreadHandle thread,
    uint8_t *scratch, uint64_t processedCount, uint64_t chunkCount)
{
    const uint64_t chunkBytes = chunkCount * ELEMENT_BYTES;
    if (processedCount == 0 && chunkCount == param.count) {
        // 小包完整输入在内存中连续：把原实现的 16 个 LocalCopy 合并为一个。
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, scratch,
            param.inputPtr, static_cast<uint64_t>(param.rankSize) * chunkBytes)));
        return HCCL_SUCCESS;
    }

    for (uint32_t sliceRank = 0; sliceRank < param.rankSize; ++sliceRank) {
        const uint64_t inputElementOffset =
            static_cast<uint64_t>(sliceRank) * param.count + processedCount;
        void *inputChunk = static_cast<void *>(
            static_cast<uint8_t *>(param.inputPtr) + inputElementOffset * ELEMENT_BYTES);
        void *scratchChunk = static_cast<void *>(
            scratch + static_cast<uint64_t>(sliceRank) * chunkBytes);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            thread, scratchChunk, inputChunk, chunkBytes)));
    }
    return HCCL_SUCCESS;
}

HcclResult RunNhrStep(ThreadHandle thread, const ChannelInfo &sendChannel,
    const ChannelInfo &recvChannel, uint8_t *localBuffer, uint8_t *remoteBuffer,
    uint64_t chunkCount, uint64_t chunkBytes, uint32_t rankSize,
    uint32_t myRank, uint32_t step)
{
    CHK_RET(BeginPeerRead(thread, sendChannel, recvChannel));

    const uint32_t deltaRank = 1U << step;
    const uint32_t sliceStride = deltaRank << 1;
    const uint32_t sliceCount = (rankSize - 1 + deltaRank) / sliceStride;
    uint32_t recvSlice = myRank;
    HcommBatchTransferDesc transferDescs[NHR_BATCH_CAPACITY];
    uint32_t batchSize = 0;
    for (uint32_t index = 0; index < sliceCount; ++index) {
        const uint64_t sliceOffset = static_cast<uint64_t>(recvSlice) * chunkBytes;
        HcommBatchTransferDesc &desc = transferDescs[batchSize++];
        desc = {};
        desc.transType = HCOMM_TRANSFER_TYPE_READ_REDUCE;
        desc.transferInfo.reduce.count = chunkCount;
        desc.transferInfo.reduce.dst = localBuffer + sliceOffset;
        desc.transferInfo.reduce.src = remoteBuffer + sliceOffset;
        desc.transferInfo.reduce.reduceOp = HCOMM_REDUCE_SUM;
        desc.transferInfo.reduce.dataType = HCOMM_DATA_TYPE_FP32;
        const bool isLastSlice = index + 1 == sliceCount;
        if (!isLastSlice) {
            recvSlice = (recvSlice + rankSize - (sliceStride % rankSize)) % rankSize;
        }
        if (isLastSlice) {
            HcommBatchTransferDesc &notifyDesc = transferDescs[batchSize++];
            notifyDesc = {};
            notifyDesc.transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
            notifyDesc.transferInfo.notifyRecord.notifyIdx = NOTIFY_IDX_DATA_SIGNAL;
        }
        if (batchSize == NHR_READ_BATCH_CAPACITY || isLastSlice) {
            CHK_RET(static_cast<HcclResult>(HcommBatchTransferOnThread(thread,
                recvChannel.handle, transferDescs, batchSize)));
            batchSize = 0;
        }
    }

    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RunNhr(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    const ThreadHandle thread = resCtx.aicpuThread;
    uint64_t maxChunkCount = usableBytes /
        (static_cast<uint64_t>(param.rankSize) * ELEMENT_BYTES);
    CHK_PRT_RET(maxChunkCount == 0,
        HCCL_ERROR("HCCL buffer cannot hold NHR scratch"), HCCL_E_INTERNAL);
    const uint32_t stepCount = GetNhrStepNum(param.rankSize);

    for (uint64_t processedCount = 0; processedCount < param.count;) {
        const uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        const uint64_t chunkBytes = chunkCount * ELEMENT_BYTES;
        uint8_t *sendBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        CHK_RET(CopyInputSlices(param, thread, sendBase, processedCount, chunkCount));

        for (uint32_t step = 0; step < stepCount; ++step) {
            const uint32_t deltaRank = 1U << step;
            const uint32_t recvFrom = (param.myRank + deltaRank) % param.rankSize;
            const uint32_t sendTo =
                (param.myRank + param.rankSize - deltaRank) % param.rankSize;
            const ChannelInfo *recvChannel = FindChannel(resCtx, recvFrom);
            const ChannelInfo *sendChannel = FindChannel(resCtx, sendTo);
            CHK_PRT_RET(recvChannel == nullptr || sendChannel == nullptr,
                HCCL_ERROR("NHR step[%u] missing channel", step), HCCL_E_INTERNAL);
            CHK_RET(RunNhrStep(thread, *sendChannel, *recvChannel, sendBase,
                static_cast<uint8_t *>(recvChannel->remoteCclMem.addr),
                chunkCount, chunkBytes, param.rankSize, param.myRank, step));
        }

        void *outputChunk = static_cast<void *>(
            static_cast<uint8_t *>(param.outputPtr) + processedCount * ELEMENT_BYTES);
        void *resultChunk = static_cast<void *>(
            sendBase + static_cast<uint64_t>(param.myRank) * chunkBytes);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            thread, outputChunk, resultChunk, chunkBytes)));
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

HcclResult RunTopologyXorPullReduceFinalCopyOverlap(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    const uint32_t stepCount = GetNhrStepNum(param.rankSize);
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE ||
            stepCount != COMPETITION_XOR_STAGE_COUNT ||
            resCtx.channels.size() != stepCount || resCtx.threads.size() != 1,
        HCCL_ERROR("V81 XOR pull requires 16 ranks, 4 channels and 1 thread"),
        HCCL_E_INTERNAL);

    const ThreadHandle thread = resCtx.aicpuThread;
    const uint64_t scratchSliceCount = static_cast<uint64_t>(param.rankSize);
    uint64_t maxChunkCount = usableBytes / (scratchSliceCount * ELEMENT_BYTES);
    CHK_PRT_RET(maxChunkCount == 0,
        HCCL_ERROR("HCCL buffer cannot hold V81 XOR pull scratch"), HCCL_E_INTERNAL);
    HCCL_INFO("RS_V81_XOR_PULL_FINAL_COPY_OVERLAP rank[%u] scratchSlices[%llu] "
        "maxChunkCount[%llu] maxInputChunkBytes[%llu]", param.myRank,
        static_cast<unsigned long long>(scratchSliceCount),
        static_cast<unsigned long long>(maxChunkCount),
        static_cast<unsigned long long>(maxChunkCount * param.rankSize * ELEMENT_BYTES));

    struct XorPullStage {
        const ChannelInfo *channel = nullptr;
        uint32_t firstSlice = 0;
        uint32_t contiguousSliceCount = 0;
    };
    XorPullStage stages[COMPETITION_XOR_STAGE_COUNT];
    for (uint32_t ordinal = 0; ordinal < stepCount; ++ordinal) {
        const uint32_t dimension = stepCount - 1 - ordinal;
        const uint32_t contiguousSliceCount = 1U << dimension;
        const uint32_t peer = param.myRank ^ contiguousSliceCount;
        const ChannelInfo &channel = resCtx.channels[dimension];
        CHK_PRT_RET(channel.remoteRank != peer,
            HCCL_ERROR("V81 XOR dimension[%u] expected peer[%u], got[%u]",
                dimension, peer, channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.notifyNum <= NOTIFY_IDX_ACK,
            HCCL_ERROR("V81 XOR channel requires notify index[%u]", NOTIFY_IDX_ACK),
            HCCL_E_INTERNAL);
        const uint32_t keepMask =
            (param.rankSize - 1U) & ~((1U << dimension) - 1U);
        XorPullStage &stage = stages[ordinal];
        stage.channel = &channel;
        stage.firstSlice = param.myRank & keepMask;
        stage.contiguousSliceCount = contiguousSliceCount;
    }

    for (uint64_t processedCount = 0; processedCount < param.count;) {
        const uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        const uint64_t chunkBytes = chunkCount * ELEMENT_BYTES;
        uint8_t *sendBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr) +
            processedCount * ELEMENT_BYTES;
        CHK_RET(CopyInputSlices(param, thread, sendBase, processedCount, chunkCount));

        for (uint32_t ordinal = 0; ordinal < stepCount; ++ordinal) {
            const XorPullStage &stage = stages[ordinal];
            const bool finalStage = ordinal + 1 == stepCount;
            const uint64_t contiguousCount =
                chunkCount * static_cast<uint64_t>(stage.contiguousSliceCount);
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                thread, stage.channel->handle, NOTIFY_IDX_ACK)));

            // 最后一轮双方Record以后，把本地部分和复制到最终输出；该复制与
            // 对端READY传播重叠。随后ReadReduce直接写输出，删除循环后的尾部拷贝。
            void *reduceDestination = sendBase +
                static_cast<uint64_t>(stage.firstSlice) * chunkBytes;
            if (finalStage) {
                CHK_PRT_RET(stage.firstSlice != param.myRank ||
                        stage.contiguousSliceCount != 1,
                    HCCL_ERROR("V81 final XOR stage is not the local output slice"),
                    HCCL_E_INTERNAL);
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                    thread, outputChunk, reduceDestination, chunkBytes)));
                reduceDestination = outputChunk;
            }

            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, stage.channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(
                thread, stage.channel->handle, reduceDestination,
                static_cast<uint8_t *>(stage.channel->remoteCclMem.addr) +
                    static_cast<uint64_t>(stage.firstSlice) * chunkBytes,
                contiguousCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

HcclResult RunTopologyXorPullReduceTinyUnrolled(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE || param.count != 1 ||
            resCtx.channels.size() != COMPETITION_XOR_STAGE_COUNT ||
            resCtx.threads.size() != 1 ||
            usableBytes < COMPETITION_RANK_SIZE * ELEMENT_BYTES,
        HCCL_ERROR("V92 tiny XOR requires count1, 16 ranks, 4 channels and 1 thread"),
        HCCL_E_INTERNAL);

    const ThreadHandle thread = resCtx.aicpuThread;
    uint8_t *sendBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);

    // 点5实际输入固定为16个连续FP32。直接提交一次64B拷贝，删除通用
    // chunk循环、min/divide、stage描述符构造、逐stage校验和两条INFO格式化路径。
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        thread, sendBase, param.inputPtr, COMPETITION_RANK_SIZE * ELEMENT_BYTES)));

    const ChannelInfo &xor8 = resCtx.channels[3];
    const uint64_t offset8 = static_cast<uint64_t>(param.myRank & 8U) * ELEMENT_BYTES;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, xor8.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, xor8.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(
        thread, xor8.handle, sendBase + offset8,
        static_cast<uint8_t *>(xor8.remoteCclMem.addr) + offset8,
        8, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));

    const ChannelInfo &xor4 = resCtx.channels[2];
    const uint64_t offset4 = static_cast<uint64_t>(param.myRank & 12U) * ELEMENT_BYTES;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, xor4.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, xor4.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(
        thread, xor4.handle, sendBase + offset4,
        static_cast<uint8_t *>(xor4.remoteCclMem.addr) + offset4,
        4, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));

    const ChannelInfo &xor2 = resCtx.channels[1];
    const uint64_t offset2 = static_cast<uint64_t>(param.myRank & 14U) * ELEMENT_BYTES;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, xor2.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, xor2.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(
        thread, xor2.handle, sendBase + offset2,
        static_cast<uint8_t *>(xor2.remoteCclMem.addr) + offset2,
        2, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));

    const ChannelInfo &xor1 = resCtx.channels[0];
    const uint64_t ownOffset = static_cast<uint64_t>(param.myRank) * ELEMENT_BYTES;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, xor1.handle, NOTIFY_IDX_ACK)));
    // 与V81R1相同：最后READY传播期间复制本地部分，ReadReduce直接写最终输出。
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        thread, output, sendBase + ownOffset, ELEMENT_BYTES)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, xor1.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(
        thread, xor1.handle, output,
        static_cast<uint8_t *>(xor1.remoteCclMem.addr) + ownOffset,
        1, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return HCCL_SUCCESS;
}

HcclResult GetCompetitionChannels(const OpParam &param, const AlgResourceCtx &resCtx,
    std::vector<const ChannelInfo *> &localChannels, const ChannelInfo *&crossChannel)
{
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE,
        HCCL_ERROR("competition hierarchy requires 16 ranks"), HCCL_E_INTERNAL);
    const uint32_t serverBase = (param.myRank / RANKS_PER_SERVER) * RANKS_PER_SERVER;
    for (uint32_t localRank = 0; localRank < RANKS_PER_SERVER; ++localRank) {
        const uint32_t peer = serverBase + localRank;
        if (peer == param.myRank) {
            continue;
        }
        const ChannelInfo *channel = FindChannel(resCtx, peer);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("missing local channel to rank[%u]", peer), HCCL_E_INTERNAL);
        localChannels.push_back(channel);
    }
    const uint32_t counterpart =
        (param.myRank + RANKS_PER_SERVER) % COMPETITION_RANK_SIZE;
    crossChannel = FindChannel(resCtx, counterpart);
    CHK_PRT_RET(crossChannel == nullptr,
        HCCL_ERROR("missing counterpart channel to rank[%u]", counterpart), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunFlatten(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() != param.rankSize,
        HCCL_ERROR("flatten requires p-1 channels and p threads"), HCCL_E_INTERNAL);
    const uint64_t scratchGroups =
        static_cast<uint64_t>(param.rankSize) + param.rankSize - 1;
    uint64_t maxChunkCount = AlignChunkCount(usableBytes / (scratchGroups * ELEMENT_BYTES));
    CHK_PRT_RET(maxChunkCount == 0,
        HCCL_ERROR("HCCL buffer cannot hold flatten scratch"), HCCL_E_INTERNAL);

    std::vector<uint32_t> workerSlots;
    for (uint32_t slot = 0; slot < resCtx.channels.size(); ++slot) {
        workerSlots.push_back(slot);
    }

    const ThreadHandle mainThread = resCtx.threads[0];
    for (uint64_t processedCount = 0; processedCount < param.count;) {
        const uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        const uint64_t chunkBytes = chunkCount * ELEMENT_BYTES;
        uint8_t *sendBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        uint8_t *receiveBase = sendBase + static_cast<uint64_t>(param.rankSize) * chunkBytes;
        CHK_RET(CopyInputSlices(param, mainThread, sendBase, processedCount, chunkCount));

        CHK_RET(StartWorkers(resCtx.threads, workerSlots));
        for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
            const ChannelInfo &channel = resCtx.channels[index];
            const ThreadHandle worker = resCtx.threads[index + 1];
            CHK_RET(BeginPeerRead(worker, channel, channel));
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(worker, channel.handle,
                receiveBase + static_cast<uint64_t>(index) * chunkBytes,
                static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                    static_cast<uint64_t>(param.myRank) * chunkBytes,
                chunkBytes)));
            CHK_RET(EndPeerRead(worker, channel, channel));
        }
        CHK_RET(FinishWorkers(resCtx.threads, workerSlots));

        uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr) +
            processedCount * ELEMENT_BYTES;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
            outputChunk, sendBase + static_cast<uint64_t>(param.myRank) * chunkBytes,
            chunkBytes)));
        // 固定按 remoteRank 升序规约；并行读取从不写同一目的地址。
        for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread,
                outputChunk, receiveBase + static_cast<uint64_t>(index) * chunkBytes,
                chunkCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

struct PackedStripeLayout {
    uint64_t counts[DIRECT_PUSH_STRIPE_COUNT] = {0, 0, 0};
    uint64_t offsets[DIRECT_PUSH_STRIPE_COUNT] = {0, 0, 0};
    uint64_t regionAStrideCount = 0;
    uint64_t regionBStrideCount = 0;
    uint64_t scratchCount = 0;
};

uint32_t GetCompactSourceSlot(uint32_t sourceRank, uint32_t receiverRank)
{
    return sourceRank < receiverRank ? sourceRank : sourceRank - 1;
}

PackedStripeLayout GetPackedStripeLayout(uint64_t chunkCount, uint32_t rankSize)
{
    PackedStripeLayout layout;
    uint64_t firstBoundary =
        chunkCount * PACKED_FIRST_STRIPE_NUMERATOR / STRIPE_DENOMINATOR;
    uint64_t secondBoundary =
        chunkCount * (PACKED_FIRST_STRIPE_NUMERATOR +
            PACKED_SECOND_STRIPE_NUMERATOR) / STRIPE_DENOMINATOR;
    if (firstBoundary >= CHUNK_ALIGN_COUNT) {
        firstBoundary = AlignChunkCount(firstBoundary);
    }
    if (secondBoundary >= CHUNK_ALIGN_COUNT) {
        secondBoundary = AlignChunkCount(secondBoundary);
    }
    layout.counts[0] = firstBoundary;
    layout.counts[1] = secondBoundary - firstBoundary;
    layout.counts[2] = chunkCount - secondBoundary;
    layout.offsets[0] = 0;
    layout.offsets[1] = firstBoundary;
    layout.offsets[2] = secondBoundary;
    layout.regionAStrideCount = std::max(layout.counts[0], layout.counts[2]);
    layout.regionBStrideCount = layout.counts[1];
    layout.scratchCount = static_cast<uint64_t>(rankSize - 1) *
        (layout.regionAStrideCount + layout.regionBStrideCount);
    return layout;
}

uint64_t GetPackedScratchOffsetCount(const PackedStripeLayout &layout,
    uint32_t rankSize, uint32_t compactSlot, uint32_t stripe)
{
    if (stripe == 1) {
        return static_cast<uint64_t>(rankSize - 1) * layout.regionAStrideCount +
            static_cast<uint64_t>(compactSlot) * layout.regionBStrideCount;
    }
    return static_cast<uint64_t>(compactSlot) * layout.regionAStrideCount;
}

uint64_t GetPackedMaxChunkCount(uint64_t usableBytes, uint32_t rankSize)
{
    const uint64_t scratchNumerator = static_cast<uint64_t>(rankSize - 1) *
        (std::max(PACKED_FIRST_STRIPE_NUMERATOR,
            PACKED_THIRD_STRIPE_NUMERATOR) + PACKED_SECOND_STRIPE_NUMERATOR);
    uint64_t maxChunkCount = AlignChunkCount(
        (usableBytes / ELEMENT_BYTES) * STRIPE_DENOMINATOR / scratchNumerator);
    while (maxChunkCount != 0) {
        const PackedStripeLayout layout = GetPackedStripeLayout(maxChunkCount, rankSize);
        if (layout.scratchCount <= usableBytes / ELEMENT_BYTES) {
            return maxChunkCount;
        }
        if (maxChunkCount <= CHUNK_ALIGN_COUNT) {
            return 0;
        }
        maxChunkCount -= CHUNK_ALIGN_COUNT;
    }
    return 0;
}

uint32_t GetReduceGroupBegin(uint32_t group)
{
    return REDUCE_GROUP_BEGIN[group];
}

HcclResult ReducePackedFinalStripeWithWorkerHandoff(const OpParam &param,
    const AlgResourceCtx &resCtx, const PackedStripeLayout &layout,
    uint32_t stripe, uint8_t *outputChunk, uint8_t *receiveBase)
{
    const uint32_t channelCount = static_cast<uint32_t>(resCtx.channels.size());
    CHK_PRT_RET(channelCount != param.rankSize - 1 ||
            resCtx.threads.size() != param.rankSize ||
            RS_DEDICATED_REDUCE_GROUPS == 0 ||
            RS_TOTAL_REDUCE_GROUPS != RS_DEDICATED_REDUCE_GROUPS + 1 ||
            REDUCE_GROUP_BEGIN[RS_TOTAL_REDUCE_GROUPS] != channelCount ||
            stripe >= DIRECT_PUSH_STRIPE_COUNT || layout.counts[stripe] == 0,
        HCCL_ERROR("worker-handoff packed final reduce has invalid resources"),
        HCCL_E_INTERNAL);
    // 最后一条带的15个通信worker仍处于上一轮Start之后。V94不再先做
    // FinishWorkers全局15路屏障、再由main重新Start三个worker。每个非根
    // worker在自己的DATA Wait之后把完成信号直接交给所属组根；组根收到
    // 3个成员后立刻规约。主组的3个worker直接交给main。这样所有规约只
    // 依赖本组DATA就绪，并通过3/4/4/4树传递完成，浮点树与V92完全相同。
    const ThreadHandle mainThread = resCtx.threads[0];
    for (uint32_t index = GetReduceGroupBegin(0);
         index < GetReduceGroupBegin(1); ++index) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[index + 1], mainThread, index)));
    }

    for (uint32_t group = 1; group < RS_TOTAL_REDUCE_GROUPS; ++group) {
        const uint32_t begin = GetReduceGroupBegin(group);
        const uint32_t end = GetReduceGroupBegin(group + 1);
        const ThreadHandle rootWorker = resCtx.threads[begin + 1];
        for (uint32_t index = begin + 1; index < end; ++index) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[index + 1], rootWorker, index - begin - 1)));
        }
    }

    for (uint32_t group = 1; group < RS_TOTAL_REDUCE_GROUPS; ++group) {
        const uint32_t begin = GetReduceGroupBegin(group);
        const uint32_t end = GetReduceGroupBegin(group + 1);
        const ChannelInfo &root = resCtx.channels[begin];
        const uint32_t rootSlot =
            GetCompactSourceSlot(root.remoteRank, param.myRank);
        void *groupRoot = receiveBase + GetPackedScratchOffsetCount(
            layout, param.rankSize, rootSlot, stripe) * ELEMENT_BYTES;
        const ThreadHandle rootWorker = resCtx.threads[begin + 1];
        for (uint32_t index = begin + 1; index < end; ++index) {
            const uint32_t memberNotify = index - begin - 1;
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                rootWorker, memberNotify, CUSTOM_TIMEOUT)));
            const ChannelInfo &source = resCtx.channels[index];
            const uint32_t sourceSlot =
                GetCompactSourceSlot(source.remoteRank, param.myRank);
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                rootWorker, groupRoot,
                receiveBase + GetPackedScratchOffsetCount(
                    layout, param.rankSize, sourceSlot, stripe) * ELEMENT_BYTES,
                layout.counts[stripe], HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
    }

    const uint32_t root1Index = GetReduceGroupBegin(1);
    const uint32_t root2Index = GetReduceGroupBegin(2);
    const uint32_t root3Index = GetReduceGroupBegin(3);
    const ThreadHandle root1Worker = resCtx.threads[root1Index + 1];
    const ThreadHandle root2Worker = resCtx.threads[root2Index + 1];
    const ThreadHandle root3Worker = resCtx.threads[root3Index + 1];
    const ChannelInfo &root1 = resCtx.channels[GetReduceGroupBegin(1)];
    const ChannelInfo &root2 = resCtx.channels[GetReduceGroupBegin(2)];
    const ChannelInfo &root3 = resCtx.channels[GetReduceGroupBegin(3)];
    const uint32_t root1Slot =
        GetCompactSourceSlot(root1.remoteRank, param.myRank);
    const uint32_t root2Slot =
        GetCompactSourceSlot(root2.remoteRank, param.myRank);
    const uint32_t root3Slot =
        GetCompactSourceSlot(root3.remoteRank, param.myRank);
    void *root2Addr = receiveBase + GetPackedScratchOffsetCount(
        layout, param.rankSize, root2Slot, stripe) * ELEMENT_BYTES;
    const void *root3Addr = receiveBase + GetPackedScratchOffsetCount(
        layout, param.rankSize, root3Slot, stripe) * ELEMENT_BYTES;

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        root1Worker, mainThread, root1Index)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        root3Worker, root2Worker, 3)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        root2Worker, 3, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        root2Worker, root2Addr, root3Addr, layout.counts[stripe],
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        root2Worker, mainThread, root2Index)));

    for (uint32_t index = GetReduceGroupBegin(0);
         index < GetReduceGroupBegin(1); ++index) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            mainThread, index, CUSTOM_TIMEOUT)));
        const ChannelInfo &source = resCtx.channels[index];
        const uint32_t sourceSlot =
            GetCompactSourceSlot(source.remoteRank, param.myRank);
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            mainThread, outputChunk + layout.offsets[stripe] * ELEMENT_BYTES,
            receiveBase + GetPackedScratchOffsetCount(
                layout, param.rankSize, sourceSlot, stripe) * ELEMENT_BYTES,
            layout.counts[stripe], HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, root1Index, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        mainThread, outputChunk + layout.offsets[stripe] * ELEMENT_BYTES,
        receiveBase + GetPackedScratchOffsetCount(
            layout, param.rankSize, root1Slot, stripe) * ELEMENT_BYTES,
        layout.counts[stripe], HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, root2Index, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        mainThread, outputChunk + layout.offsets[stripe] * ELEMENT_BYTES,
        root2Addr, layout.counts[stripe],
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return HCCL_SUCCESS;
}

HcclResult ReduceDirectFinalStripeWithWorkerHandoff(const OpParam &param,
    const AlgResourceCtx &resCtx, const uint64_t *stripeCounts,
    const uint64_t *stripeOffsets, uint32_t stripe, uint64_t chunkBytes,
    uint8_t *outputChunk, uint8_t *receiveBase)
{
    const uint32_t channelCount = static_cast<uint32_t>(resCtx.channels.size());
    CHK_PRT_RET(channelCount != param.rankSize - 1 ||
            resCtx.threads.size() != param.rankSize ||
            RS_DEDICATED_REDUCE_GROUPS == 0 ||
            RS_TOTAL_REDUCE_GROUPS != RS_DEDICATED_REDUCE_GROUPS + 1 ||
            REDUCE_GROUP_BEGIN[RS_TOTAL_REDUCE_GROUPS] != channelCount ||
            stripe >= REFERENCE_V2_STRIPE_COUNT || stripeCounts[stripe] == 0,
        HCCL_ERROR("worker-handoff direct final reduce has invalid resources"),
        HCCL_E_INTERNAL);

    const ThreadHandle mainThread = resCtx.threads[0];
    for (uint32_t index = GetReduceGroupBegin(0);
         index < GetReduceGroupBegin(1); ++index) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[index + 1], mainThread, index)));
    }
    for (uint32_t group = 1; group < RS_TOTAL_REDUCE_GROUPS; ++group) {
        const uint32_t begin = GetReduceGroupBegin(group);
        const uint32_t end = GetReduceGroupBegin(group + 1);
        const ThreadHandle rootWorker = resCtx.threads[begin + 1];
        for (uint32_t index = begin + 1; index < end; ++index) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[index + 1], rootWorker, index - begin - 1)));
        }
    }

    for (uint32_t group = 1; group < RS_TOTAL_REDUCE_GROUPS; ++group) {
        const uint32_t begin = GetReduceGroupBegin(group);
        const uint32_t end = GetReduceGroupBegin(group + 1);
        const ChannelInfo &root = resCtx.channels[begin];
        void *groupRoot = receiveBase +
            static_cast<uint64_t>(root.remoteRank) * chunkBytes +
            stripeOffsets[stripe] * ELEMENT_BYTES;
        const ThreadHandle rootWorker = resCtx.threads[begin + 1];
        for (uint32_t index = begin + 1; index < end; ++index) {
            const uint32_t memberNotify = index - begin - 1;
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                rootWorker, memberNotify, CUSTOM_TIMEOUT)));
            const ChannelInfo &source = resCtx.channels[index];
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                rootWorker, groupRoot,
                receiveBase + static_cast<uint64_t>(source.remoteRank) * chunkBytes +
                    stripeOffsets[stripe] * ELEMENT_BYTES,
                stripeCounts[stripe], HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
    }

    const uint32_t root1Index = GetReduceGroupBegin(1);
    const uint32_t root2Index = GetReduceGroupBegin(2);
    const uint32_t root3Index = GetReduceGroupBegin(3);
    const ThreadHandle root1Worker = resCtx.threads[root1Index + 1];
    const ThreadHandle root2Worker = resCtx.threads[root2Index + 1];
    const ThreadHandle root3Worker = resCtx.threads[root3Index + 1];
    const ChannelInfo &root1 = resCtx.channels[root1Index];
    const ChannelInfo &root2 = resCtx.channels[root2Index];
    const ChannelInfo &root3 = resCtx.channels[root3Index];
    void *root2Addr = receiveBase +
        static_cast<uint64_t>(root2.remoteRank) * chunkBytes +
        stripeOffsets[stripe] * ELEMENT_BYTES;
    const void *root3Addr = receiveBase +
        static_cast<uint64_t>(root3.remoteRank) * chunkBytes +
        stripeOffsets[stripe] * ELEMENT_BYTES;

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        root1Worker, mainThread, root1Index)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        root3Worker, root2Worker, 3)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        root2Worker, 3, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        root2Worker, root2Addr, root3Addr, stripeCounts[stripe],
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        root2Worker, mainThread, root2Index)));

    for (uint32_t index = GetReduceGroupBegin(0);
         index < GetReduceGroupBegin(1); ++index) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            mainThread, index, CUSTOM_TIMEOUT)));
        const ChannelInfo &source = resCtx.channels[index];
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            mainThread, outputChunk + stripeOffsets[stripe] * ELEMENT_BYTES,
            receiveBase + static_cast<uint64_t>(source.remoteRank) * chunkBytes +
                stripeOffsets[stripe] * ELEMENT_BYTES,
            stripeCounts[stripe], HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, root1Index, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        mainThread, outputChunk + stripeOffsets[stripe] * ELEMENT_BYTES,
        receiveBase + static_cast<uint64_t>(root1.remoteRank) * chunkBytes +
            stripeOffsets[stripe] * ELEMENT_BYTES,
        stripeCounts[stripe], HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, root2Index, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        mainThread, outputChunk + stripeOffsets[stripe] * ELEMENT_BYTES,
        root2Addr, stripeCounts[stripe], HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunTinyDirectAllPeerPush(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE || param.count != 1 ||
            resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() != param.rankSize ||
            usableBytes < param.rankSize * ELEMENT_BYTES,
        HCCL_ERROR("tiny direct push requires count1, 16 ranks, 15 channels "
            "and 16 threads"), HCCL_E_INTERNAL);

    std::vector<uint32_t> workerSlots;
    for (uint32_t slot = 0; slot < resCtx.channels.size(); ++slot) {
        workerSlots.push_back(slot);
    }
    CHK_RET(StartWorkers(resCtx.threads, workerSlots));

    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *receiveBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        mainThread, output,
        static_cast<uint8_t *>(param.inputPtr) +
            static_cast<uint64_t>(param.myRank) * ELEMENT_BYTES,
        ELEMENT_BYTES)));

    for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
        const ChannelInfo &channel = resCtx.channels[index];
        CHK_PRT_RET(channel.notifyNum <= NOTIFY_IDX_DATA_SIGNAL,
            HCCL_ERROR("tiny direct channel to rank[%u] needs two notify slots",
                channel.remoteRank), HCCL_E_INTERNAL);
        const ThreadHandle worker = resCtx.threads[index + 1];
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
            worker, channel.handle,
            static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                static_cast<uint64_t>(param.myRank) * ELEMENT_BYTES,
            static_cast<uint8_t *>(param.inputPtr) +
                static_cast<uint64_t>(channel.remoteRank) * ELEMENT_BYTES,
            ELEMENT_BYTES, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }

    const uint64_t stripeCounts[REFERENCE_V2_STRIPE_COUNT] = {1, 0, 0, 0};
    const uint64_t stripeOffsets[REFERENCE_V2_STRIPE_COUNT] = {0, 1, 1, 1};
    return ReduceDirectFinalStripeWithWorkerHandoff(
        param, resCtx, stripeCounts, stripeOffsets, 0, ELEMENT_BYTES,
        output, receiveBase);
}

[[maybe_unused]] HcclResult RunTinyHierarchyPush(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    constexpr uint64_t TINY_RECEIVE_PAIR_COUNT =
        RANKS_PER_SERVER * 2;
    constexpr uint64_t TINY_PARTIAL_PAIR_OFFSET =
        TINY_RECEIVE_PAIR_COUNT;
    constexpr uint64_t TINY_CROSS_RECEIVE_OFFSET =
        TINY_PARTIAL_PAIR_OFFSET + 2;
    constexpr uint64_t TINY_SCRATCH_COUNT =
        TINY_CROSS_RECEIVE_OFFSET + 1;
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE || param.count != 1 ||
            resCtx.channels.size() != RANKS_PER_SERVER ||
            resCtx.threads.size() != RANKS_PER_SERVER ||
            usableBytes < TINY_SCRATCH_COUNT * ELEMENT_BYTES,
        HCCL_ERROR("tiny hierarchy requires count1, 16 ranks, 8 channels "
            "and 8 threads"), HCCL_E_INTERNAL);

    std::vector<const ChannelInfo *> localChannels;
    const ChannelInfo *crossChannel = nullptr;
    CHK_RET(GetCompetitionChannels(param, resCtx, localChannels, crossChannel));
    CHK_PRT_RET(localChannels.size() != LOCAL_WORKER_COUNT ||
            crossChannel == nullptr ||
            crossChannel->notifyNum <= NOTIFY_IDX_DATA_SIGNAL,
        HCCL_ERROR("tiny hierarchy channel partition is invalid"),
        HCCL_E_INTERNAL);

    std::vector<uint32_t> workerSlots;
    for (uint32_t slot = 0; slot < LOCAL_WORKER_COUNT; ++slot) {
        workerSlots.push_back(slot);
    }
    CHK_RET(StartWorkers(resCtx.threads, workerSlots));

    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *receiveBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint8_t *partialPair =
        receiveBase + TINY_PARTIAL_PAIR_OFFSET * ELEMENT_BYTES;
    uint8_t *crossReceive =
        receiveBase + TINY_CROSS_RECEIVE_OFFSET * ELEMENT_BYTES;
    const uint32_t sourceLocalId = param.myRank % RANKS_PER_SERVER;

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        mainThread, partialPair,
        static_cast<uint8_t *>(param.inputPtr) +
            static_cast<uint64_t>(sourceLocalId) * ELEMENT_BYTES,
        ELEMENT_BYTES)));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        mainThread, partialPair + ELEMENT_BYTES,
        static_cast<uint8_t *>(param.inputPtr) +
            static_cast<uint64_t>(sourceLocalId + RANKS_PER_SERVER) *
                ELEMENT_BYTES,
        ELEMENT_BYTES)));

    for (uint32_t index = 0; index < localChannels.size(); ++index) {
        const ChannelInfo &channel = *localChannels[index];
        CHK_PRT_RET(channel.notifyNum <= NOTIFY_IDX_DATA_SIGNAL,
            HCCL_ERROR("tiny hierarchy local channel to rank[%u] needs "
                "two notify slots", channel.remoteRank), HCCL_E_INTERNAL);
        const uint32_t targetLocalId =
            channel.remoteRank % RANKS_PER_SERVER;
        const ThreadHandle worker = resCtx.threads[index + 1];
        HcommBatchTransferDesc descs[3] = {};
        descs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
        descs[0].transferInfo.write.len = ELEMENT_BYTES;
        descs[0].transferInfo.write.dst =
            static_cast<uint8_t *>(channel.remoteCclMem.addr) +
            static_cast<uint64_t>(sourceLocalId * 2) * ELEMENT_BYTES;
        descs[0].transferInfo.write.src =
            static_cast<uint8_t *>(param.inputPtr) +
            static_cast<uint64_t>(targetLocalId) * ELEMENT_BYTES;
        descs[1].transType = HCOMM_TRANSFER_TYPE_WRITE;
        descs[1].transferInfo.write.len = ELEMENT_BYTES;
        descs[1].transferInfo.write.dst =
            static_cast<uint8_t *>(channel.remoteCclMem.addr) +
            static_cast<uint64_t>(sourceLocalId * 2 + 1) * ELEMENT_BYTES;
        descs[1].transferInfo.write.src =
            static_cast<uint8_t *>(param.inputPtr) +
            static_cast<uint64_t>(targetLocalId + RANKS_PER_SERVER) *
                ELEMENT_BYTES;
        descs[2].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
        descs[2].transferInfo.notifyRecord.notifyIdx =
            NOTIFY_IDX_DATA_SIGNAL;
        CHK_RET(static_cast<HcclResult>(HcommBatchTransferOnThread(
            worker, channel.handle, descs, 3)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }

#if HCCL_RS_TINY_MODE == RS_TINY_MODE_HIERARCHY_SEQUENTIAL
    for (uint32_t index = 0; index < localChannels.size(); ++index) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[index + 1], mainThread, index)));
    }
    for (uint32_t index = 0; index < localChannels.size(); ++index) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            mainThread, index, CUSTOM_TIMEOUT)));
        const uint32_t sourceSlot =
            localChannels[index]->remoteRank % RANKS_PER_SERVER;
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            mainThread, partialPair,
            receiveBase + static_cast<uint64_t>(sourceSlot * 2) *
                ELEMENT_BYTES,
            2, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
#else
    // 7个本地贡献按3+4分组。前三个worker直接交给main；后四个以
    // worker3为根先合并，再只向main交一个根，固定浮点次序。
    for (uint32_t index = 0; index < 3; ++index) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[index + 1], mainThread, index)));
    }
    const ThreadHandle groupRootWorker = resCtx.threads[4];
    for (uint32_t index = 4; index < LOCAL_WORKER_COUNT; ++index) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[index + 1], groupRootWorker, index - 4)));
    }
    const uint32_t groupRootSlot =
        localChannels[3]->remoteRank % RANKS_PER_SERVER;
    void *groupRootPair =
        receiveBase + static_cast<uint64_t>(groupRootSlot * 2) *
            ELEMENT_BYTES;
    for (uint32_t index = 4; index < LOCAL_WORKER_COUNT; ++index) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            groupRootWorker, index - 4, CUSTOM_TIMEOUT)));
        const uint32_t sourceSlot =
            localChannels[index]->remoteRank % RANKS_PER_SERVER;
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            groupRootWorker, groupRootPair,
            receiveBase + static_cast<uint64_t>(sourceSlot * 2) *
                ELEMENT_BYTES,
            2, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        groupRootWorker, mainThread, 3)));
    for (uint32_t index = 0; index < 3; ++index) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            mainThread, index, CUSTOM_TIMEOUT)));
        const uint32_t sourceSlot =
            localChannels[index]->remoteRank % RANKS_PER_SERVER;
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            mainThread, partialPair,
            receiveBase + static_cast<uint64_t>(sourceSlot * 2) *
                ELEMENT_BYTES,
            2, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, 3, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        mainThread, partialPair, groupRootPair, 2,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
#endif

    const uint32_t serverIndex = param.myRank / RANKS_PER_SERVER;
    const uint32_t ownPartialIndex = serverIndex;
    const uint32_t sendPartialIndex = 1U - serverIndex;
    CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
        mainThread, crossChannel->handle,
        static_cast<uint8_t *>(crossChannel->remoteCclMem.addr) +
            TINY_CROSS_RECEIVE_OFFSET * ELEMENT_BYTES,
        partialPair + static_cast<uint64_t>(sendPartialIndex) * ELEMENT_BYTES,
        ELEMENT_BYTES, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        mainThread, param.outputPtr,
        partialPair + static_cast<uint64_t>(ownPartialIndex) * ELEMENT_BYTES,
        ELEMENT_BYTES)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        mainThread, crossChannel->handle, NOTIFY_IDX_DATA_SIGNAL,
        CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        mainThread, param.outputPtr, crossReceive, 1,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return HCCL_SUCCESS;
}

HcclResult ReduceStripeWithDedicatedWorkers(const OpParam &param,
    const AlgResourceCtx &resCtx, uint8_t *outputStripe,
    uint8_t *const sourceAddrs[COMPETITION_RANK_SIZE - 1],
    uint64_t count, uint32_t channelNotifyIndex, uint32_t stripeOrdinal)
{
    const uint32_t channelCount = static_cast<uint32_t>(resCtx.channels.size());
    CHK_PRT_RET(channelCount != param.rankSize - 1 ||
            resCtx.threads.size() !=
                param.rankSize + RS_DEDICATED_REDUCE_GROUPS ||
            count == 0 || stripeOrdinal >= REFERENCE_V2_STRIPE_COUNT,
        HCCL_ERROR("dedicated stripe reduce has invalid resources"),
        HCCL_E_INTERNAL);

    const ThreadHandle mainThread = resCtx.threads[0];
    const ThreadHandle root1Worker = resCtx.threads[channelCount + 1];
    const ThreadHandle root2Worker = resCtx.threads[channelCount + 2];
    const ThreadHandle root3Worker = resCtx.threads[channelCount + 3];

    const uint32_t group1Begin = GetReduceGroupBegin(1);
    const uint32_t group1End = GetReduceGroupBegin(2);
    for (uint32_t index = group1Begin; index < group1End; ++index) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            root1Worker, resCtx.channels[index].handle, channelNotifyIndex,
            CUSTOM_TIMEOUT)));
    }
    for (uint32_t index = group1Begin + 1; index < group1End; ++index) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            root1Worker, sourceAddrs[group1Begin], sourceAddrs[index],
            count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }

    const uint32_t group2Begin = GetReduceGroupBegin(2);
    const uint32_t group2End = GetReduceGroupBegin(3);
    for (uint32_t index = group2Begin; index < group2End; ++index) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            root2Worker, resCtx.channels[index].handle, channelNotifyIndex,
            CUSTOM_TIMEOUT)));
    }
    for (uint32_t index = group2Begin + 1; index < group2End; ++index) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            root2Worker, sourceAddrs[group2Begin], sourceAddrs[index],
            count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }

    const uint32_t group3Begin = GetReduceGroupBegin(3);
    const uint32_t group3End = GetReduceGroupBegin(4);
    for (uint32_t index = group3Begin; index < group3End; ++index) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            root3Worker, resCtx.channels[index].handle, channelNotifyIndex,
            CUSTOM_TIMEOUT)));
    }
    for (uint32_t index = group3Begin + 1; index < group3End; ++index) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            root3Worker, sourceAddrs[group3Begin], sourceAddrs[index],
            count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }

    // FinishWorkers会让18个worker分别写main notify[0,17]。规约完成信号
    // 必须使用独立区间，否则通信worker可能先完成并与规约root形成两个
    // Record抢同一个寄存器。root2自身的notify[0]同样保留给StartWorkers。
    const uint32_t mainCompletionNotifyBase =
        static_cast<uint32_t>(resCtx.threads.size() - 1);
    const uint32_t root1MainNotify =
        mainCompletionNotifyBase + stripeOrdinal * 2;
    const uint32_t root2MainNotify = root1MainNotify + 1;
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        root1Worker, mainThread, root1MainNotify)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        root3Worker, root2Worker, stripeOrdinal + 1)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        root2Worker, stripeOrdinal + 1, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        root2Worker, sourceAddrs[group2Begin], sourceAddrs[group3Begin],
        count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        root2Worker, mainThread, root2MainNotify)));

    for (uint32_t index = GetReduceGroupBegin(0);
         index < GetReduceGroupBegin(1); ++index) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            mainThread, resCtx.channels[index].handle, channelNotifyIndex,
            CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            mainThread, outputStripe, sourceAddrs[index], count,
            HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, root1MainNotify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        mainThread, outputStripe, sourceAddrs[group1Begin], count,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, root2MainNotify, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        mainThread, outputStripe, sourceAddrs[group2Begin], count,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunDedicatedPackedReducePipeline(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE ||
            resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() !=
                param.rankSize + RS_DEDICATED_REDUCE_GROUPS,
        HCCL_ERROR("dedicated packed pipeline requires 19 threads"),
        HCCL_E_INTERNAL);
    const uint64_t maxChunkCount =
        GetPackedMaxChunkCount(usableBytes, param.rankSize);
    CHK_PRT_RET(maxChunkCount == 0 || param.count > maxChunkCount,
        HCCL_ERROR("dedicated packed pipeline requires one outer"),
        HCCL_E_INTERNAL);
    const PackedStripeLayout layout =
        GetPackedStripeLayout(param.count, param.rankSize);
    CHK_PRT_RET(layout.counts[0] == 0 || layout.counts[1] == 0 ||
            layout.counts[2] == 0 ||
            layout.scratchCount * ELEMENT_BYTES > usableBytes,
        HCCL_ERROR("invalid dedicated packed layout"), HCCL_E_INTERNAL);

    std::vector<uint32_t> workerSlots;
    for (uint32_t slot = 0; slot <
            param.rankSize + RS_DEDICATED_REDUCE_GROUPS - 1; ++slot) {
        workerSlots.push_back(slot);
    }
    CHK_RET(StartWorkers(resCtx.threads, workerSlots));

    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *receiveBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr);
    const void *ownInputChunk = static_cast<const uint8_t *>(param.inputPtr) +
        static_cast<uint64_t>(param.myRank) * param.count * ELEMENT_BYTES;
    for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
        const ChannelInfo &channel = resCtx.channels[index];
        const ThreadHandle worker = resCtx.threads[index + 1];
        CHK_PRT_RET(channel.notifyNum <= PACKED_NOTIFY_STRIPE2,
            HCCL_ERROR("dedicated packed channel needs four notify slots"),
            HCCL_E_INTERNAL);
        const uint32_t remoteSlot =
            GetCompactSourceSlot(param.myRank, channel.remoteRank);
        for (uint32_t stripe = 0; stripe < 2; ++stripe) {
            const uint64_t sourceOffset =
                (static_cast<uint64_t>(channel.remoteRank) * param.count +
                    layout.offsets[stripe]) * ELEMENT_BYTES;
            void *remoteScratch =
                static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                GetPackedScratchOffsetCount(
                    layout, param.rankSize, remoteSlot, stripe) * ELEMENT_BYTES;
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                worker, channel.handle, remoteScratch,
                static_cast<uint8_t *>(param.inputPtr) + sourceOffset,
                layout.counts[stripe] * ELEMENT_BYTES)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                worker, channel.handle,
                stripe == 0 ? PACKED_NOTIFY_STRIPE0 :
                    PACKED_NOTIFY_STRIPE1)));
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            worker, channel.handle, PACKED_NOTIFY_REUSE_PERMIT,
            CUSTOM_TIMEOUT)));
        const uint64_t sourceOffset =
            (static_cast<uint64_t>(channel.remoteRank) * param.count +
                layout.offsets[2]) * ELEMENT_BYTES;
        void *remoteScratch =
            static_cast<uint8_t *>(channel.remoteCclMem.addr) +
            GetPackedScratchOffsetCount(
                layout, param.rankSize, remoteSlot, 2) * ELEMENT_BYTES;
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
            worker, channel.handle, remoteScratch,
            static_cast<uint8_t *>(param.inputPtr) + sourceOffset,
            layout.counts[2] * ELEMENT_BYTES)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            worker, channel.handle, PACKED_NOTIFY_STRIPE2)));
    }

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        mainThread, outputChunk, ownInputChunk, param.count * ELEMENT_BYTES)));
    for (uint32_t stripe = 0; stripe < DIRECT_PUSH_STRIPE_COUNT; ++stripe) {
        uint8_t *sourceAddrs[COMPETITION_RANK_SIZE - 1] = {};
        for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
            const uint32_t localSlot = GetCompactSourceSlot(
                resCtx.channels[index].remoteRank, param.myRank);
            sourceAddrs[index] = receiveBase + GetPackedScratchOffsetCount(
                layout, param.rankSize, localSlot, stripe) * ELEMENT_BYTES;
        }
        const uint32_t notifyIndex =
            stripe == 0 ? PACKED_NOTIFY_STRIPE0 :
                (stripe == 1 ? PACKED_NOTIFY_STRIPE1 :
                    PACKED_NOTIFY_STRIPE2);
        CHK_RET(ReduceStripeWithDedicatedWorkers(
            param, resCtx,
            outputChunk + layout.offsets[stripe] * ELEMENT_BYTES,
            sourceAddrs, layout.counts[stripe], notifyIndex, stripe));
        if (stripe == 0) {
            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(
                        mainThread, channel.handle,
                        PACKED_NOTIFY_REUSE_PERMIT)));
            }
        }
    }
    CHK_RET(FinishWorkers(resCtx.threads, workerSlots));
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunDedicatedReferenceReducePipeline(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE ||
            resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() !=
                param.rankSize + RS_DEDICATED_REDUCE_GROUPS,
        HCCL_ERROR("dedicated reference pipeline requires 19 threads"),
        HCCL_E_INTERNAL);
    const uint64_t scratchGroups = static_cast<uint64_t>(param.rankSize);
    const uint64_t maxChunkCount = AlignChunkCount(
        usableBytes / (scratchGroups * ELEMENT_BYTES));
    CHK_PRT_RET(maxChunkCount == 0 || param.count > maxChunkCount,
        HCCL_ERROR("dedicated reference pipeline requires one outer"),
        HCCL_E_INTERNAL);

    uint64_t stripeCounts[REFERENCE_V2_STRIPE_COUNT] = {0, 0, 0, 0};
    uint64_t stripeOffsets[REFERENCE_V2_STRIPE_COUNT] = {0, 0, 0, 0};
    uint64_t cumulative = 0;
    uint64_t previousBoundary = 0;
    for (uint32_t stripe = 0; stripe < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
        cumulative += REFERENCE_V2_STRIPE_NUMERATORS[stripe];
        uint64_t boundary = stripe + 1 == REFERENCE_V2_STRIPE_COUNT ?
            param.count : param.count * cumulative / STRIPE_DENOMINATOR;
        if (boundary >= CHUNK_ALIGN_COUNT &&
            stripe + 1 != REFERENCE_V2_STRIPE_COUNT) {
            boundary = AlignChunkCount(boundary);
        }
        stripeOffsets[stripe] = previousBoundary;
        stripeCounts[stripe] = boundary - previousBoundary;
        previousBoundary = boundary;
        CHK_PRT_RET(stripeCounts[stripe] == 0,
            HCCL_ERROR("dedicated reference requires non-empty stripes"),
            HCCL_E_INTERNAL);
    }

    std::vector<uint32_t> workerSlots;
    for (uint32_t slot = 0; slot <
            param.rankSize + RS_DEDICATED_REDUCE_GROUPS - 1; ++slot) {
        workerSlots.push_back(slot);
    }
    CHK_RET(StartWorkers(resCtx.threads, workerSlots));

    const ThreadHandle mainThread = resCtx.threads[0];
    const uint64_t chunkBytes = param.count * ELEMENT_BYTES;
    uint8_t *receiveBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr);
    const void *ownInputChunk = static_cast<const uint8_t *>(param.inputPtr) +
        static_cast<uint64_t>(param.myRank) * chunkBytes;
    for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
        const ChannelInfo &channel = resCtx.channels[index];
        const ThreadHandle worker = resCtx.threads[index + 1];
        CHK_PRT_RET(channel.notifyNum <= REFERENCE_NOTIFY_STRIPE3,
            HCCL_ERROR("dedicated reference channel needs four notify slots"),
            HCCL_E_INTERNAL);
        void *remoteSlot = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
            static_cast<uint64_t>(param.myRank) * chunkBytes;
        for (uint32_t stripe = 0; stripe < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
            const uint64_t sourceOffset =
                (static_cast<uint64_t>(channel.remoteRank) * param.count +
                    stripeOffsets[stripe]) * ELEMENT_BYTES;
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                worker, channel.handle,
                static_cast<uint8_t *>(remoteSlot) +
                    stripeOffsets[stripe] * ELEMENT_BYTES,
                static_cast<uint8_t *>(param.inputPtr) + sourceOffset,
                stripeCounts[stripe] * ELEMENT_BYTES)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                worker, channel.handle, EAGER_NOTIFY_STRIPE0 + stripe)));
        }
    }

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        mainThread, outputChunk, ownInputChunk, chunkBytes)));
    for (uint32_t stripe = 0; stripe < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
        uint8_t *sourceAddrs[COMPETITION_RANK_SIZE - 1] = {};
        for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
            sourceAddrs[index] = receiveBase +
                static_cast<uint64_t>(resCtx.channels[index].remoteRank) *
                    chunkBytes +
                stripeOffsets[stripe] * ELEMENT_BYTES;
        }
        CHK_RET(ReduceStripeWithDedicatedWorkers(
            param, resCtx,
            outputChunk + stripeOffsets[stripe] * ELEMENT_BYTES,
            sourceAddrs, stripeCounts[stripe],
            EAGER_NOTIFY_STRIPE0 + stripe, stripe));
    }
    CHK_RET(FinishWorkers(resCtx.threads, workerSlots));
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunPersistentPackedSingleOuterHandoff(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE ||
            resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() != param.rankSize,
        HCCL_ERROR("persistent packed handoff requires competition resources"),
        HCCL_E_INTERNAL);
    const uint64_t maxChunkCount = GetPackedMaxChunkCount(usableBytes, param.rankSize);
    CHK_PRT_RET(maxChunkCount == 0 || param.count > maxChunkCount,
        HCCL_ERROR("persistent packed handoff requires a single outer"),
        HCCL_E_INTERNAL);
    const PackedStripeLayout layout =
        GetPackedStripeLayout(param.count, param.rankSize);
    CHK_PRT_RET(layout.counts[0] == 0 || layout.counts[1] == 0 ||
            layout.counts[2] == 0 ||
            layout.scratchCount * ELEMENT_BYTES > usableBytes,
        HCCL_ERROR("invalid persistent packed layout"), HCCL_E_INTERNAL);

    std::vector<uint32_t> workerSlots;
    for (uint32_t slot = 0; slot < resCtx.channels.size(); ++slot) {
        workerSlots.push_back(slot);
    }
    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *receiveBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr);
    const void *ownInputChunk = static_cast<const uint8_t *>(param.inputPtr) +
        static_cast<uint64_t>(param.myRank) * param.count * ELEMENT_BYTES;

    // 一个worker生命周期覆盖三条带。stripe0/1由main直接消费；main消费
    // stripe0后用独立index2许可覆盖A区。worker记录stripe2后等待对端
    // stripe2到达，再直接进入已经通过VM的3/4/4/4规约handoff。
    CHK_RET(StartWorkers(resCtx.threads, workerSlots));
    for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
        const ChannelInfo &channel = resCtx.channels[index];
        const ThreadHandle worker = resCtx.threads[index + 1];
        CHK_PRT_RET(channel.notifyNum <= PACKED_NOTIFY_STRIPE2,
            HCCL_ERROR("persistent packed channel needs four notify slots"),
            HCCL_E_INTERNAL);
        const uint32_t remoteSlot =
            GetCompactSourceSlot(param.myRank, channel.remoteRank);
        for (uint32_t stripe = 0; stripe < 2; ++stripe) {
            const uint64_t sourceOffset =
                (static_cast<uint64_t>(channel.remoteRank) * param.count +
                    layout.offsets[stripe]) * ELEMENT_BYTES;
            const void *source = static_cast<const uint8_t *>(param.inputPtr) +
                sourceOffset;
            void *remoteScratch = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                GetPackedScratchOffsetCount(
                    layout, param.rankSize, remoteSlot, stripe) * ELEMENT_BYTES;
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                worker, channel.handle, remoteScratch, source,
                layout.counts[stripe] * ELEMENT_BYTES)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                worker, channel.handle,
                stripe == 0 ? PACKED_NOTIFY_STRIPE0 : PACKED_NOTIFY_STRIPE1)));
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            worker, channel.handle, PACKED_NOTIFY_REUSE_PERMIT, CUSTOM_TIMEOUT)));
        const uint64_t sourceOffset =
            (static_cast<uint64_t>(channel.remoteRank) * param.count +
                layout.offsets[2]) * ELEMENT_BYTES;
        const void *source = static_cast<const uint8_t *>(param.inputPtr) +
            sourceOffset;
        void *remoteScratch = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
            GetPackedScratchOffsetCount(
                layout, param.rankSize, remoteSlot, 2) * ELEMENT_BYTES;
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
            worker, channel.handle, remoteScratch, source,
            layout.counts[2] * ELEMENT_BYTES)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            worker, channel.handle, PACKED_NOTIFY_STRIPE2)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            worker, channel.handle, PACKED_NOTIFY_STRIPE2, CUSTOM_TIMEOUT)));
    }

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        mainThread, outputChunk, ownInputChunk, param.count * ELEMENT_BYTES)));
    for (uint32_t stripe = 0; stripe < 2; ++stripe) {
        const uint32_t notifyIndex =
            stripe == 0 ? PACKED_NOTIFY_STRIPE0 : PACKED_NOTIFY_STRIPE1;
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                mainThread, channel.handle, notifyIndex, CUSTOM_TIMEOUT)));
        }
        for (const ChannelInfo &channel : resCtx.channels) {
            const uint32_t localSlot =
                GetCompactSourceSlot(channel.remoteRank, param.myRank);
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                mainThread,
                outputChunk + layout.offsets[stripe] * ELEMENT_BYTES,
                receiveBase + GetPackedScratchOffsetCount(
                    layout, param.rankSize, localSlot, stripe) * ELEMENT_BYTES,
                layout.counts[stripe], HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
        if (stripe == 0) {
            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    mainThread, channel.handle, PACKED_NOTIFY_REUSE_PERMIT)));
            }
        }
    }
    return ReducePackedFinalStripeWithWorkerHandoff(
        param, resCtx, layout, 2, outputChunk, receiveBase);
}

[[maybe_unused]] HcclResult RunPersistentReferenceSingleOuterHandoff(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE ||
            resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() != param.rankSize,
        HCCL_ERROR("persistent reference handoff requires competition resources"),
        HCCL_E_INTERNAL);
    const uint64_t scratchGroups = static_cast<uint64_t>(param.rankSize);
    const uint64_t maxChunkCount = AlignChunkCount(
        usableBytes / (scratchGroups * ELEMENT_BYTES));
    CHK_PRT_RET(maxChunkCount == 0 || param.count > maxChunkCount,
        HCCL_ERROR("persistent reference handoff requires a single outer"),
        HCCL_E_INTERNAL);

    uint64_t stripeCounts[REFERENCE_V2_STRIPE_COUNT] = {0, 0, 0, 0};
    uint64_t stripeOffsets[REFERENCE_V2_STRIPE_COUNT] = {0, 0, 0, 0};
    uint64_t cumulative = 0;
    uint64_t previousBoundary = 0;
    for (uint32_t stripe = 0; stripe < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
        cumulative += REFERENCE_V2_STRIPE_NUMERATORS[stripe];
        uint64_t boundary = stripe + 1 == REFERENCE_V2_STRIPE_COUNT ?
            param.count : param.count * cumulative / STRIPE_DENOMINATOR;
        if (boundary >= CHUNK_ALIGN_COUNT &&
            stripe + 1 != REFERENCE_V2_STRIPE_COUNT) {
            boundary = AlignChunkCount(boundary);
        }
        stripeOffsets[stripe] = previousBoundary;
        stripeCounts[stripe] = boundary - previousBoundary;
        previousBoundary = boundary;
    }
    for (uint32_t stripe = 0; stripe < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
        CHK_PRT_RET(stripeCounts[stripe] == 0,
            HCCL_ERROR("persistent reference path requires four non-empty stripes"),
            HCCL_E_INTERNAL);
    }

    std::vector<uint32_t> workerSlots;
    for (uint32_t slot = 0; slot < resCtx.channels.size(); ++slot) {
        workerSlots.push_back(slot);
    }
    const uint64_t chunkBytes = param.count * ELEMENT_BYTES;
    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *receiveBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr);
    const void *ownInputChunk = static_cast<const uint8_t *>(param.inputPtr) +
        static_cast<uint64_t>(param.myRank) * param.count * ELEMENT_BYTES;

    // 四条带一次排入15个通信worker。前三条由main按remoteRank升序消费；
    // worker在记录最后条带后等待对应入站DATA，随后直接进入handoff。
    CHK_RET(StartWorkers(resCtx.threads, workerSlots));
    for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
        const ChannelInfo &channel = resCtx.channels[index];
        const ThreadHandle worker = resCtx.threads[index + 1];
        CHK_PRT_RET(channel.notifyNum <= REFERENCE_NOTIFY_STRIPE3,
            HCCL_ERROR("persistent reference channel needs four notify slots"),
            HCCL_E_INTERNAL);
        void *remoteSlot = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
            static_cast<uint64_t>(param.myRank) * chunkBytes;
        for (uint32_t stripe = 0; stripe < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
            const uint64_t sourceOffset =
                (static_cast<uint64_t>(channel.remoteRank) * param.count +
                    stripeOffsets[stripe]) * ELEMENT_BYTES;
            const void *source = static_cast<const uint8_t *>(param.inputPtr) +
                sourceOffset;
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                worker, channel.handle,
                static_cast<uint8_t *>(remoteSlot) +
                    stripeOffsets[stripe] * ELEMENT_BYTES,
                source, stripeCounts[stripe] * ELEMENT_BYTES)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                worker, channel.handle, EAGER_NOTIFY_STRIPE0 + stripe)));
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            worker, channel.handle, REFERENCE_NOTIFY_STRIPE3, CUSTOM_TIMEOUT)));
    }

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        mainThread, outputChunk, ownInputChunk, chunkBytes)));
    for (uint32_t stripe = 0; stripe + 1 < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                mainThread, channel.handle, EAGER_NOTIFY_STRIPE0 + stripe,
                CUSTOM_TIMEOUT)));
        }
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                mainThread,
                outputChunk + stripeOffsets[stripe] * ELEMENT_BYTES,
                receiveBase + static_cast<uint64_t>(channel.remoteRank) * chunkBytes +
                    stripeOffsets[stripe] * ELEMENT_BYTES,
                stripeCounts[stripe], HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
    }
    return ReduceDirectFinalStripeWithWorkerHandoff(
        param, resCtx, stripeCounts, stripeOffsets,
        REFERENCE_V2_STRIPE_COUNT - 1, chunkBytes, outputChunk, receiveBase);
}

HcclResult RunPackedStripePush(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE ||
            resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() != param.rankSize,
        HCCL_ERROR("packed-stripe push requires the 16-rank competition domain"),
        HCCL_E_INTERNAL);
    const uint64_t maxChunkCount = GetPackedMaxChunkCount(usableBytes, param.rankSize);
    CHK_PRT_RET(maxChunkCount == 0,
        HCCL_ERROR("HCCL buffer cannot hold packed-stripe scratch"), HCCL_E_INTERNAL);
    const PackedStripeLayout maxLayout =
        GetPackedStripeLayout(maxChunkCount, param.rankSize);
    HCCL_INFO("RS_CANDIDATE[%s] packedScratchBytes[%llu] maxChunkCount[%llu] "
              "maxInputChunkBytes[%llu] packedStripeRatio[9,13,10]",
        GetAlgorithmName(),
        static_cast<unsigned long long>(maxLayout.scratchCount * ELEMENT_BYTES),
        static_cast<unsigned long long>(maxChunkCount),
        static_cast<unsigned long long>(maxChunkCount) * param.rankSize * ELEMENT_BYTES);

    std::vector<uint32_t> workerSlots;
    for (uint32_t slot = 0; slot < resCtx.channels.size(); ++slot) {
        workerSlots.push_back(slot);
    }

    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *receiveBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    for (uint64_t processedCount = 0; processedCount < param.count;) {
        const uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        const uint64_t chunkBytes = chunkCount * ELEMENT_BYTES;
        const PackedStripeLayout layout =
            GetPackedStripeLayout(chunkCount, param.rankSize);
        CHK_PRT_RET(layout.counts[0] == 0 || layout.counts[1] == 0 ||
                layout.counts[2] == 0 ||
                layout.scratchCount * ELEMENT_BYTES > usableBytes,
            HCCL_ERROR("invalid packed-stripe layout"), HCCL_E_INTERNAL);

        uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr) +
            processedCount * ELEMENT_BYTES;
        const uint64_t ownInputOffset =
            (static_cast<uint64_t>(param.myRank) * param.count + processedCount) *
            ELEMENT_BYTES;
        const void *ownInputChunk = static_cast<const uint8_t *>(param.inputPtr) +
            ownInputOffset;

        CHK_RET(StartWorkers(resCtx.threads, workerSlots));
        for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
            const ChannelInfo &channel = resCtx.channels[index];
            const ThreadHandle worker = resCtx.threads[index + 1];
            CHK_PRT_RET(channel.notifyNum <= NOTIFY_IDX_DATA_SIGNAL,
                HCCL_ERROR("packed-stripe channels require two notify slots"),
                HCCL_E_INTERNAL);
            const uint64_t sourceOffset =
                (static_cast<uint64_t>(channel.remoteRank) * param.count +
                    processedCount) * ELEMENT_BYTES;
            const void *source = static_cast<const uint8_t *>(param.inputPtr) +
                sourceOffset;
            const uint32_t remoteSlot =
                GetCompactSourceSlot(param.myRank, channel.remoteRank);
            void *remoteScratch = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                GetPackedScratchOffsetCount(
                    layout, param.rankSize, remoteSlot, 0) * ELEMENT_BYTES;
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                worker, channel.handle, remoteScratch, source,
                layout.counts[0] * ELEMENT_BYTES)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            mainThread, outputChunk, ownInputChunk, chunkBytes)));
        CHK_RET(FinishWorkers(resCtx.threads, workerSlots));

        uint32_t previousStripe = 0;
        for (uint32_t stripe = 1; stripe < DIRECT_PUSH_STRIPE_COUNT; ++stripe) {
            CHK_RET(StartWorkers(resCtx.threads, workerSlots));
            for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
                const ChannelInfo &channel = resCtx.channels[index];
                const ThreadHandle worker = resCtx.threads[index + 1];
                // stripe 2 将覆盖 A 区。Start 位于本 Rank stripe 0 规约之后；
                // 双方先 Record 再 Wait，证明对端也已消费 A，随后才允许覆盖。
                if (stripe == 2) {
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                        worker, channel.handle, NOTIFY_IDX_ACK)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                        worker, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                }
                const uint64_t sourceOffset =
                    (static_cast<uint64_t>(channel.remoteRank) * param.count +
                        processedCount + layout.offsets[stripe]) * ELEMENT_BYTES;
                const void *source = static_cast<const uint8_t *>(param.inputPtr) +
                    sourceOffset;
                const uint32_t remoteSlot =
                    GetCompactSourceSlot(param.myRank, channel.remoteRank);
                void *remoteScratch = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                    GetPackedScratchOffsetCount(
                        layout, param.rankSize, remoteSlot, stripe) * ELEMENT_BYTES;
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                    worker, channel.handle, remoteScratch, source,
                    layout.counts[stripe] * ELEMENT_BYTES)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }
            for (const ChannelInfo &channel : resCtx.channels) {
                const uint32_t localSlot =
                    GetCompactSourceSlot(channel.remoteRank, param.myRank);
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                    mainThread,
                    outputChunk + layout.offsets[previousStripe] * ELEMENT_BYTES,
                    receiveBase + GetPackedScratchOffsetCount(
                        layout, param.rankSize, localSlot, previousStripe) * ELEMENT_BYTES,
                    layout.counts[previousStripe], HCOMM_DATA_TYPE_FP32,
                    HCOMM_REDUCE_SUM)));
            }
            CHK_RET(FinishWorkers(resCtx.threads, workerSlots));
            previousStripe = stripe;
        }
        for (const ChannelInfo &channel : resCtx.channels) {
            const uint32_t localSlot =
                GetCompactSourceSlot(channel.remoteRank, param.myRank);
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                mainThread,
                outputChunk + layout.offsets[previousStripe] * ELEMENT_BYTES,
                receiveBase + GetPackedScratchOffsetCount(
                    layout, param.rankSize, localSlot, previousStripe) * ELEMENT_BYTES,
                layout.counts[previousStripe], HCOMM_DATA_TYPE_FP32,
                HCOMM_REDUCE_SUM)));
        }

        // packed scratch只在下一outer写入时需要覆盖保护。
        if (processedCount + chunkCount < param.count) {
            CHK_RET(StartWorkers(resCtx.threads, workerSlots));
            for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
                const ChannelInfo &channel = resCtx.channels[index];
                const ThreadHandle worker = resCtx.threads[index + 1];
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    worker, channel.handle, NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    worker, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            }
            CHK_RET(FinishWorkers(resCtx.threads, workerSlots));
        }
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

HcclResult RunDirectPush(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() != param.rankSize,
        HCCL_ERROR("direct push requires p-1 channels and p threads"), HCCL_E_INTERNAL);
    const uint64_t scratchGroups = static_cast<uint64_t>(param.rankSize);
    uint64_t maxChunkCount = AlignChunkCount(
        usableBytes / (scratchGroups * ELEMENT_BYTES));
    CHK_PRT_RET(maxChunkCount == 0,
        HCCL_ERROR("HCCL buffer cannot hold direct-push scratch"), HCCL_E_INTERNAL);

    std::vector<uint32_t> workerSlots;
    for (uint32_t slot = 0; slot < resCtx.channels.size(); ++slot) {
        workerSlots.push_back(slot);
    }

    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *receiveBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    for (uint64_t processedCount = 0; processedCount < param.count;) {
        const uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        const uint64_t chunkBytes = chunkCount * ELEMENT_BYTES;
        uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr) +
            processedCount * ELEMENT_BYTES;
        const uint64_t ownInputOffset =
            (static_cast<uint64_t>(param.myRank) * param.count + processedCount) *
            ELEMENT_BYTES;
        const void *ownInputChunk = static_cast<const uint8_t *>(param.inputPtr) +
            ownInputOffset;
        // 参考提交86550-v2已经取得42/1280/917 us真值。其单outer改进为
        // 4/8/10/10四条带；这里完整保留累计边界与对齐规则。
        uint64_t stripeCounts[REFERENCE_V2_STRIPE_COUNT] = {0, 0, 0, 0};
        uint64_t stripeOffsets[REFERENCE_V2_STRIPE_COUNT] = {0, 0, 0, 0};
        uint64_t boundaries[REFERENCE_V2_STRIPE_COUNT] = {0, 0, 0, 0};
        uint64_t cumulative = 0;
        for (uint32_t stripe = 0; stripe < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
            cumulative += REFERENCE_V2_STRIPE_NUMERATORS[stripe];
            uint64_t boundary = stripe + 1 == REFERENCE_V2_STRIPE_COUNT ?
                chunkCount : chunkCount * cumulative / STRIPE_DENOMINATOR;
            if (boundary >= CHUNK_ALIGN_COUNT &&
                stripe + 1 != REFERENCE_V2_STRIPE_COUNT) {
                boundary = AlignChunkCount(boundary);
            }
            boundaries[stripe] = boundary;
        }
        stripeCounts[0] = boundaries[0];
        for (uint32_t stripe = 1; stripe < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
            stripeOffsets[stripe] = boundaries[stripe - 1];
            stripeCounts[stripe] = boundaries[stripe] - boundaries[stripe - 1];
        }
        bool anyEmpty = false;
        for (uint32_t stripe = 0; stripe < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
            anyEmpty = anyEmpty || stripeCounts[stripe] == 0;
        }
        if (anyEmpty) {
            stripeCounts[0] = chunkCount;
            stripeOffsets[0] = 0;
            for (uint32_t stripe = 1; stripe < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
                stripeCounts[stripe] = 0;
                stripeOffsets[stripe] = chunkCount;
            }
        }

        // 单缓冲三条带流水：每个条带写入同一源 Rank 槽位的不同偏移，
        // 因而无需增加 scratch 或在条带之间发送 ACK。DATA 的 Record/Wait
        // 在同一 channel stream 上完整消费后才复用 notify，避免多打。
        CHK_RET(StartWorkers(resCtx.threads, workerSlots));
        for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
            const ChannelInfo &channel = resCtx.channels[index];
            const ThreadHandle worker = resCtx.threads[index + 1];
            CHK_PRT_RET(channel.notifyNum <= NOTIFY_IDX_DATA_SIGNAL,
                HCCL_ERROR("striped direct-push channels require two notify slots"),
                HCCL_E_INTERNAL);
            const uint64_t sourceOffset =
                (static_cast<uint64_t>(channel.remoteRank) * param.count +
                    processedCount) * ELEMENT_BYTES;
            const void *source = static_cast<const uint8_t *>(param.inputPtr) +
                sourceOffset;
            void *remoteSlot = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                static_cast<uint64_t>(param.myRank) * chunkBytes;
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                worker, channel.handle, remoteSlot, source,
                stripeCounts[0] * ELEMENT_BYTES)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }
        // 自身输出片复制与第一条带网络写在不同 stream 上并行下发。
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            mainThread, outputChunk, ownInputChunk, chunkBytes)));
        CHK_RET(FinishWorkers(resCtx.threads, workerSlots));

        uint32_t previousStripe = 0;
        for (uint32_t stripe = 1; stripe < REFERENCE_V2_STRIPE_COUNT; ++stripe) {
            if (stripeCounts[stripe] == 0) {
                continue;
            }
            // 先启动当前条带，再在主 stream 上规约上一条带；FinishWorkers
            // 排在规约之后，形成 communication(stripe) || reduce(stripe-1)。
            CHK_RET(StartWorkers(resCtx.threads, workerSlots));
            for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
                const ChannelInfo &channel = resCtx.channels[index];
                const ThreadHandle worker = resCtx.threads[index + 1];
                const uint64_t sourceOffset =
                    (static_cast<uint64_t>(channel.remoteRank) * param.count +
                        processedCount + stripeOffsets[stripe]) * ELEMENT_BYTES;
                const void *source = static_cast<const uint8_t *>(param.inputPtr) +
                    sourceOffset;
                void *remoteSlot = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                    static_cast<uint64_t>(param.myRank) * chunkBytes +
                    stripeOffsets[stripe] * ELEMENT_BYTES;
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                    worker, channel.handle, remoteSlot, source,
                    stripeCounts[stripe] * ELEMENT_BYTES)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }
            // Channel 按 remoteRank 升序保存；每个元素的浮点规约次序与 V3 一致。
            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                    mainThread,
                    outputChunk + stripeOffsets[previousStripe] * ELEMENT_BYTES,
                    receiveBase + static_cast<uint64_t>(channel.remoteRank) * chunkBytes +
                        stripeOffsets[previousStripe] * ELEMENT_BYTES,
                    stripeCounts[previousStripe], HCOMM_DATA_TYPE_FP32,
                    HCOMM_REDUCE_SUM)));
            }
            CHK_RET(FinishWorkers(resCtx.threads, workerSlots));
            previousStripe = stripe;
        }
        // 最后一条带采用参考代码已由官方VM和平台真值验证的配对规约：
        // 7个worker并行把15个远端槽配成7对，再由主线程规约8个结果。
        const uint32_t channelCount = static_cast<uint32_t>(resCtx.channels.size());
        const uint32_t pairCount = channelCount / 2;
        std::vector<uint32_t> reduceWorkers;
        for (uint32_t pair = 0; pair < pairCount; ++pair) {
            reduceWorkers.push_back(pair);
        }
        if (!reduceWorkers.empty()) {
            CHK_RET(StartWorkers(resCtx.threads, reduceWorkers));
            for (uint32_t pair = 0; pair < pairCount; ++pair) {
                const ChannelInfo &first = resCtx.channels[2 * pair];
                const ChannelInfo &second = resCtx.channels[2 * pair + 1];
                const ThreadHandle worker = resCtx.threads[pair + 1];
                void *destination = receiveBase +
                    static_cast<uint64_t>(first.remoteRank) * chunkBytes +
                    stripeOffsets[previousStripe] * ELEMENT_BYTES;
                const void *source = receiveBase +
                    static_cast<uint64_t>(second.remoteRank) * chunkBytes +
                    stripeOffsets[previousStripe] * ELEMENT_BYTES;
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                    worker, destination, source, stripeCounts[previousStripe],
                    HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            }
            CHK_RET(FinishWorkers(resCtx.threads, reduceWorkers));
        }
        for (uint32_t pair = 0; pair < pairCount; ++pair) {
            const ChannelInfo &first = resCtx.channels[2 * pair];
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                mainThread,
                outputChunk + stripeOffsets[previousStripe] * ELEMENT_BYTES,
                receiveBase + static_cast<uint64_t>(first.remoteRank) * chunkBytes +
                    stripeOffsets[previousStripe] * ELEMENT_BYTES,
                stripeCounts[previousStripe], HCOMM_DATA_TYPE_FP32,
                HCOMM_REDUCE_SUM)));
        }
        if (channelCount % 2 == 1) {
            const ChannelInfo &last = resCtx.channels[channelCount - 1];
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                mainThread,
                outputChunk + stripeOffsets[previousStripe] * ELEMENT_BYTES,
                receiveBase + static_cast<uint64_t>(last.remoteRank) * chunkBytes +
                    stripeOffsets[previousStripe] * ELEMENT_BYTES,
                stripeCounts[previousStripe], HCOMM_DATA_TYPE_FP32,
                HCOMM_REDUCE_SUM)));
        }

        // 只有存在下一outer时才需要确认消费完成，终态没有远端槽位覆盖。
        if (processedCount + chunkCount < param.count) {
            CHK_RET(StartWorkers(resCtx.threads, workerSlots));
            for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
                const ChannelInfo &channel = resCtx.channels[index];
                const ThreadHandle worker = resCtx.threads[index + 1];
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    worker, channel.handle, NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    worker, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            }
            CHK_RET(FinishWorkers(resCtx.threads, workerSlots));
        }
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

HcclResult RunEagerDirectPush(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() != param.rankSize,
        HCCL_ERROR("eager direct push requires p-1 channels and p threads"),
        HCCL_E_INTERNAL);
    const uint64_t scratchGroups = static_cast<uint64_t>(param.rankSize);
    uint64_t maxChunkCount = AlignChunkCount(
        usableBytes / (scratchGroups * ELEMENT_BYTES));
    CHK_PRT_RET(maxChunkCount == 0,
        HCCL_ERROR("HCCL buffer cannot hold eager direct-push scratch"),
        HCCL_E_INTERNAL);

    std::vector<uint32_t> workerSlots;
    for (uint32_t slot = 0; slot < resCtx.channels.size(); ++slot) {
        workerSlots.push_back(slot);
    }

    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *receiveBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    for (uint64_t processedCount = 0; processedCount < param.count;) {
        const uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        const uint64_t chunkBytes = chunkCount * ELEMENT_BYTES;
        uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr) +
            processedCount * ELEMENT_BYTES;
        const uint64_t ownInputOffset =
            (static_cast<uint64_t>(param.myRank) * param.count + processedCount) *
            ELEMENT_BYTES;
        const void *ownInputChunk = static_cast<const uint8_t *>(param.inputPtr) +
            ownInputOffset;

        uint64_t firstBoundary =
            chunkCount * FIRST_STRIPE_NUMERATOR / STRIPE_DENOMINATOR;
        uint64_t secondBoundary =
            chunkCount * (FIRST_STRIPE_NUMERATOR + SECOND_STRIPE_NUMERATOR) /
            STRIPE_DENOMINATOR;
        if (firstBoundary >= CHUNK_ALIGN_COUNT) {
            firstBoundary = AlignChunkCount(firstBoundary);
        }
        if (secondBoundary >= CHUNK_ALIGN_COUNT) {
            secondBoundary = AlignChunkCount(secondBoundary);
        }
        uint64_t stripeCounts[DIRECT_PUSH_STRIPE_COUNT] = {
            firstBoundary, secondBoundary - firstBoundary, chunkCount - secondBoundary};
        uint64_t stripeOffsets[DIRECT_PUSH_STRIPE_COUNT] = {
            0, firstBoundary, secondBoundary};
        if (stripeCounts[0] == 0 || stripeCounts[1] == 0 || stripeCounts[2] == 0) {
            stripeCounts[0] = chunkCount;
            stripeCounts[1] = 0;
            stripeCounts[2] = 0;
            stripeOffsets[1] = chunkCount;
            stripeOffsets[2] = chunkCount;
        }

        // 每个 worker 在一个生命周期内依次排入三条 DATA 写，使用三个互不
        // 混用的远端 notify。主线程按 stripe 等待并规约；网络可以继续推进
        // 后续 stripe。最终 ACK 证明接收端已经消费完 scratch，才允许下一 outer。
        CHK_RET(StartWorkers(resCtx.threads, workerSlots));
        for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
            const ChannelInfo &channel = resCtx.channels[index];
            const ThreadHandle worker = resCtx.threads[index + 1];
            CHK_PRT_RET(channel.notifyNum <= EAGER_NOTIFY_FINAL_ACK,
                HCCL_ERROR("eager direct-push channel needs four notify slots"),
                HCCL_E_INTERNAL);
            void *remoteSlot = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                static_cast<uint64_t>(param.myRank) * chunkBytes;
            for (uint32_t stripe = 0; stripe < DIRECT_PUSH_STRIPE_COUNT; ++stripe) {
                if (stripeCounts[stripe] == 0) {
                    continue;
                }
                const uint64_t sourceOffset =
                    (static_cast<uint64_t>(channel.remoteRank) * param.count +
                        processedCount + stripeOffsets[stripe]) * ELEMENT_BYTES;
                const void *source = static_cast<const uint8_t *>(param.inputPtr) +
                    sourceOffset;
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                    worker, channel.handle,
                    static_cast<uint8_t *>(remoteSlot) +
                        stripeOffsets[stripe] * ELEMENT_BYTES,
                    source, stripeCounts[stripe] * ELEMENT_BYTES)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    worker, channel.handle, EAGER_NOTIFY_STRIPE0 + stripe)));
            }
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, channel.handle, EAGER_NOTIFY_FINAL_ACK, CUSTOM_TIMEOUT)));
        }

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            mainThread, outputChunk, ownInputChunk, chunkBytes)));
        for (uint32_t stripe = 0; stripe < DIRECT_PUSH_STRIPE_COUNT; ++stripe) {
            if (stripeCounts[stripe] == 0) {
                continue;
            }
            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    mainThread, channel.handle, EAGER_NOTIFY_STRIPE0 + stripe,
                    CUSTOM_TIMEOUT)));
            }
            // remoteRank 升序保持 V12 的浮点规约顺序。
            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                    mainThread,
                    outputChunk + stripeOffsets[stripe] * ELEMENT_BYTES,
                    receiveBase + static_cast<uint64_t>(channel.remoteRank) * chunkBytes +
                        stripeOffsets[stripe] * ELEMENT_BYTES,
                    stripeCounts[stripe], HCOMM_DATA_TYPE_FP32,
                    HCOMM_REDUCE_SUM)));
            }
        }
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                mainThread, channel.handle, EAGER_NOTIFY_FINAL_ACK)));
        }
        CHK_RET(FinishWorkers(resCtx.threads, workerSlots));
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunEagerPackedStripePush(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE ||
            resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() != param.rankSize,
        HCCL_ERROR("eager packed push requires the 16-rank competition domain"),
        HCCL_E_INTERNAL);
    const uint64_t maxChunkCount = GetPackedMaxChunkCount(usableBytes, param.rankSize);
    CHK_PRT_RET(maxChunkCount == 0,
        HCCL_ERROR("HCCL buffer cannot hold eager packed-stripe scratch"),
        HCCL_E_INTERNAL);
    const PackedStripeLayout maxLayout =
        GetPackedStripeLayout(maxChunkCount, param.rankSize);
    HCCL_INFO("RS_V35_UNUSED_EAGER_PACKED scratchBytes[%llu] maxChunkCount[%llu] "
              "maxInputChunkBytes[%llu] ratio[9,13,10]",
        static_cast<unsigned long long>(maxLayout.scratchCount * ELEMENT_BYTES),
        static_cast<unsigned long long>(maxChunkCount),
        static_cast<unsigned long long>(maxChunkCount) * param.rankSize * ELEMENT_BYTES);

    std::vector<uint32_t> workerSlots;
    for (uint32_t slot = 0; slot < resCtx.channels.size(); ++slot) {
        workerSlots.push_back(slot);
    }
    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *receiveBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);

    for (uint64_t processedCount = 0; processedCount < param.count;) {
        const uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        const uint64_t chunkBytes = chunkCount * ELEMENT_BYTES;
        const PackedStripeLayout layout =
            GetPackedStripeLayout(chunkCount, param.rankSize);
        CHK_PRT_RET(layout.counts[0] == 0 || layout.counts[1] == 0 ||
                layout.counts[2] == 0 ||
                layout.scratchCount * ELEMENT_BYTES > usableBytes,
            HCCL_ERROR("invalid eager packed-stripe layout"), HCCL_E_INTERNAL);

        uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr) +
            processedCount * ELEMENT_BYTES;
        const uint64_t ownInputOffset =
            (static_cast<uint64_t>(param.myRank) * param.count + processedCount) *
            ELEMENT_BYTES;
        const void *ownInputChunk = static_cast<const uint8_t *>(param.inputPtr) +
            ownInputOffset;

        // worker 先排入 A/B 两区写；A 区复用前等待 receiver main 的 index2
        // 许可，再写 stripe2 并用同一 index2 反向通知数据到达。两个方向的
        // Record/Wait 严格交替，因此没有多打同一 notify。
        CHK_RET(StartWorkers(resCtx.threads, workerSlots));
        for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
            const ChannelInfo &channel = resCtx.channels[index];
            const ThreadHandle worker = resCtx.threads[index + 1];
            CHK_PRT_RET(channel.notifyNum <= EAGER_NOTIFY_FINAL_ACK,
                HCCL_ERROR("eager packed channel needs four notify slots"),
                HCCL_E_INTERNAL);
            const uint32_t remoteSlot =
                GetCompactSourceSlot(param.myRank, channel.remoteRank);
            for (uint32_t stripe = 0; stripe < 2; ++stripe) {
                const uint64_t sourceOffset =
                    (static_cast<uint64_t>(channel.remoteRank) * param.count +
                        processedCount + layout.offsets[stripe]) * ELEMENT_BYTES;
                const void *source = static_cast<const uint8_t *>(param.inputPtr) +
                    sourceOffset;
                void *remoteScratch = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                    GetPackedScratchOffsetCount(
                        layout, param.rankSize, remoteSlot, stripe) * ELEMENT_BYTES;
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                    worker, channel.handle, remoteScratch, source,
                    layout.counts[stripe] * ELEMENT_BYTES)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    worker, channel.handle, EAGER_NOTIFY_STRIPE0 + stripe)));
            }
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, channel.handle, EAGER_NOTIFY_STRIPE2, CUSTOM_TIMEOUT)));
            const uint64_t sourceOffset =
                (static_cast<uint64_t>(channel.remoteRank) * param.count +
                    processedCount + layout.offsets[2]) * ELEMENT_BYTES;
            const void *source = static_cast<const uint8_t *>(param.inputPtr) +
                sourceOffset;
            void *remoteScratch = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                GetPackedScratchOffsetCount(
                    layout, param.rankSize, remoteSlot, 2) * ELEMENT_BYTES;
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                worker, channel.handle, remoteScratch, source,
                layout.counts[2] * ELEMENT_BYTES)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                worker, channel.handle, EAGER_NOTIFY_STRIPE2)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, channel.handle, EAGER_NOTIFY_FINAL_ACK, CUSTOM_TIMEOUT)));
        }

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            mainThread, outputChunk, ownInputChunk, chunkBytes)));
        for (uint32_t stripe = 0; stripe < DIRECT_PUSH_STRIPE_COUNT; ++stripe) {
            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    mainThread, channel.handle, EAGER_NOTIFY_STRIPE0 + stripe,
                    CUSTOM_TIMEOUT)));
            }
            for (const ChannelInfo &channel : resCtx.channels) {
                const uint32_t localSlot =
                    GetCompactSourceSlot(channel.remoteRank, param.myRank);
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                    mainThread,
                    outputChunk + layout.offsets[stripe] * ELEMENT_BYTES,
                    receiveBase + GetPackedScratchOffsetCount(
                        layout, param.rankSize, localSlot, stripe) * ELEMENT_BYTES,
                    layout.counts[stripe], HCOMM_DATA_TYPE_FP32,
                    HCOMM_REDUCE_SUM)));
            }
            if (stripe == 0) {
                for (const ChannelInfo &channel : resCtx.channels) {
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                        mainThread, channel.handle, EAGER_NOTIFY_STRIPE2)));
                }
            }
        }
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                mainThread, channel.handle, EAGER_NOTIFY_FINAL_ACK)));
        }
        CHK_RET(FinishWorkers(resCtx.threads, workerSlots));
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunWavefrontPullLite(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() != RANKS_PER_SERVER,
        HCCL_ERROR("Wavefront Pull-Lite requires 15 channels and 8 threads"),
        HCCL_E_INTERNAL);

    std::vector<const ChannelInfo *> localChannels;
    std::vector<const ChannelInfo *> crossChannels;
    const uint32_t serverIndex = param.myRank / RANKS_PER_SERVER;
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank / RANKS_PER_SERVER == serverIndex) {
            localChannels.push_back(&channel);
        }
    }
    const uint32_t localId = param.myRank % RANKS_PER_SERVER;
    const uint32_t otherServerBase = (1U - serverIndex) * RANKS_PER_SERVER;
    for (uint32_t round = 0; round < RANKS_PER_SERVER; ++round) {
        const uint32_t peer = otherServerBase + (localId ^ round);
        const ChannelInfo *channel = FindChannel(resCtx, peer);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("missing XOR round[%u] channel to rank[%u]", round, peer),
            HCCL_E_INTERNAL);
        crossChannels.push_back(channel);
    }
    CHK_PRT_RET(localChannels.size() != LOCAL_WORKER_COUNT ||
            crossChannels.size() != RANKS_PER_SERVER,
        HCCL_ERROR("Wavefront Pull-Lite peer partition is not 7 Mesh + 8 Clos"),
        HCCL_E_INTERNAL);

    // 16 份连续可读输入 + 16 个按 remoteRank 编号的独立接收槽。
    constexpr uint64_t scratchGroups = 2 * COMPETITION_RANK_SIZE;
    uint64_t maxChunkCount = AlignChunkCount(usableBytes / (scratchGroups * ELEMENT_BYTES));
    CHK_PRT_RET(maxChunkCount == 0,
        HCCL_ERROR("HCCL buffer cannot hold Wavefront Pull-Lite scratch"), HCCL_E_INTERNAL);

    std::vector<uint32_t> localWorkers;
    for (uint32_t slot = 0; slot < LOCAL_WORKER_COUNT; ++slot) {
        localWorkers.push_back(slot);
    }

    const ThreadHandle mainThread = resCtx.threads[0];
    for (uint64_t processedCount = 0; processedCount < param.count;) {
        const uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        const uint64_t chunkBytes = chunkCount * ELEMENT_BYTES;
        uint8_t *sendBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        uint8_t *receiveBase = sendBase + COMPETITION_RANK_SIZE * chunkBytes;
        CHK_RET(CopyInputSlices(param, mainThread, sendBase, processedCount, chunkCount));

        // 7 条 Mesh 链路各由一个 worker 独占；主 thread 同时沿 8 条 Clos 对端推进。
        // round k 使用 localId xor k：K8,8 的每一轮都是完美匹配，8 轮覆盖全部对端，
        // 不产生 V14 升序调度的 15 波次等待；每个对端仍只有一个 channel。
        CHK_RET(StartWorkers(resCtx.threads, localWorkers));
        for (uint32_t index = 0; index < localChannels.size(); ++index) {
            const ChannelInfo &channel = *localChannels[index];
            const ThreadHandle worker = resCtx.threads[index + 1];
            CHK_RET(BeginPeerRead(worker, channel, channel));
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(worker, channel.handle,
                receiveBase + static_cast<uint64_t>(channel.remoteRank) * chunkBytes,
                static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                    static_cast<uint64_t>(param.myRank) * chunkBytes,
                chunkBytes)));
            CHK_RET(EndPeerRead(worker, channel, channel));
        }
        for (const ChannelInfo *channelPtr : crossChannels) {
            const ChannelInfo &channel = *channelPtr;
            CHK_RET(BeginPeerRead(mainThread, channel, channel));
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(mainThread, channel.handle,
                receiveBase + static_cast<uint64_t>(channel.remoteRank) * chunkBytes,
                static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                    static_cast<uint64_t>(param.myRank) * chunkBytes,
                chunkBytes)));
            CHK_RET(EndPeerRead(mainThread, channel, channel));
        }
        CHK_RET(FinishWorkers(resCtx.threads, localWorkers));

        uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr) +
            processedCount * ELEMENT_BYTES;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
            outputChunk,
            sendBase + static_cast<uint64_t>(param.myRank) * chunkBytes,
            chunkBytes)));
        // channels 由 Host 的有序 peerSet 生成，故这里固定按 remoteRank 升序规约。
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread,
                outputChunk,
                receiveBase + static_cast<uint64_t>(channel.remoteRank) * chunkBytes,
                chunkCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunDualPath(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t usableBytes)
{
    CHK_PRT_RET(resCtx.threads.size() != RANKS_PER_SERVER + 1,
        HCCL_ERROR("dual path requires 9 threads"), HCCL_E_INTERNAL);
    std::vector<const ChannelInfo *> localChannels;
    const ChannelInfo *crossChannel = nullptr;
    CHK_RET(GetCompetitionChannels(param, resCtx, localChannels, crossChannel));

    // q=3/8 时 scratch=(23+9q)C=26.375C；按 27C 保守分块。
    constexpr uint64_t scratchGroupsUpperBound = 27;
    uint64_t maxChunkCount = AlignChunkCount(
        usableBytes / (scratchGroupsUpperBound * ELEMENT_BYTES));
    CHK_PRT_RET(maxChunkCount == 0,
        HCCL_ERROR("HCCL buffer cannot hold dual-path scratch"), HCCL_E_INTERNAL);

    const uint32_t serverIndex = param.myRank / RANKS_PER_SERVER;
    const uint32_t localId = param.myRank % RANKS_PER_SERVER;
    const uint32_t ownTargetBase = serverIndex * RANKS_PER_SERVER;
    const uint32_t otherTargetBase = (1U - serverIndex) * RANKS_PER_SERVER;
    const ThreadHandle mainThread = resCtx.threads[0];

    for (uint64_t processedCount = 0; processedCount < param.count;) {
        const uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        // 官方比例会量化到八分位；首轮固定 3/8，以兼顾 6~10 倍 Clos 不确定性。
        const uint64_t meshFirstCount = chunkCount * 3 / 8;
        const uint64_t pairFirstCount = chunkCount - meshFirstCount;
        const uint64_t meshBytes = meshFirstCount * ELEMENT_BYTES;
        const uint64_t pairBytes = pairFirstCount * ELEMENT_BYTES;

        uint8_t *meshSendBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        uint8_t *pairSendBase = meshSendBase + COMPETITION_RANK_SIZE * meshBytes;
        uint8_t *meshReceiveBase = pairSendBase + RANKS_PER_SERVER * pairBytes;
        uint8_t *meshPartialBase = meshReceiveBase + 2 * LOCAL_WORKER_COUNT * meshBytes;
        uint8_t *pairPartialBase = meshPartialBase + 2 * meshBytes;
        uint8_t *pairReceiveBase = pairPartialBase + RANKS_PER_SERVER * pairBytes;

        // q 数据供机内 Mesh 读取；1-q 数据只暴露给同号跨机对端。
        if (meshFirstCount > 0) {
            for (uint32_t target = 0; target < COMPETITION_RANK_SIZE; ++target) {
                const uint64_t inputOffset =
                    (static_cast<uint64_t>(target) * param.count + processedCount) * ELEMENT_BYTES;
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
                    meshSendBase + static_cast<uint64_t>(target) * meshBytes,
                    static_cast<uint8_t *>(param.inputPtr) + inputOffset, meshBytes)));
            }
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
                meshPartialBase,
                meshSendBase + static_cast<uint64_t>(localId) * meshBytes,
                meshBytes)));
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
                meshPartialBase + meshBytes,
                meshSendBase + static_cast<uint64_t>(localId + RANKS_PER_SERVER) * meshBytes,
                meshBytes)));
        }
        if (pairFirstCount > 0) {
            for (uint32_t index = 0; index < RANKS_PER_SERVER; ++index) {
                const uint64_t otherOffset =
                    (static_cast<uint64_t>(otherTargetBase + index) * param.count +
                        processedCount + meshFirstCount) * ELEMENT_BYTES;
                const uint64_t ownOffset =
                    (static_cast<uint64_t>(ownTargetBase + index) * param.count +
                        processedCount + meshFirstCount) * ELEMENT_BYTES;
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
                    pairSendBase + static_cast<uint64_t>(index) * pairBytes,
                    static_cast<uint8_t *>(param.inputPtr) + otherOffset, pairBytes)));
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
                    pairPartialBase + static_cast<uint64_t>(index) * pairBytes,
                    static_cast<uint8_t *>(param.inputPtr) + ownOffset, pairBytes)));
            }
        }

        // 阶段 1：q 走 Mesh，1-q 同时走 Pair。
        std::vector<uint32_t> stageOneWorkers;
        if (meshFirstCount > 0) {
            for (uint32_t slot = 0; slot < LOCAL_WORKER_COUNT; ++slot) {
                stageOneWorkers.push_back(slot);
            }
        }
        if (pairFirstCount > 0) {
            stageOneWorkers.push_back(CROSS_WORKER_SLOT);
        }
        CHK_RET(StartWorkers(resCtx.threads, stageOneWorkers));
        if (meshFirstCount > 0) {
            for (uint32_t index = 0; index < localChannels.size(); ++index) {
                const ChannelInfo &channel = *localChannels[index];
                const ThreadHandle worker = resCtx.threads[index + 1];
                CHK_RET(BeginPeerRead(worker, channel, channel));
                for (uint32_t targetGroup = 0; targetGroup < 2; ++targetGroup) {
                    const uint32_t target = localId + targetGroup * RANKS_PER_SERVER;
                    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(worker,
                        channel.handle,
                        meshReceiveBase +
                            static_cast<uint64_t>(index * 2 + targetGroup) * meshBytes,
                        static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                            static_cast<uint64_t>(target) * meshBytes,
                        meshBytes)));
                }
                CHK_RET(EndPeerRead(worker, channel, channel));
            }
        }
        if (pairFirstCount > 0) {
            const ThreadHandle crossThread = resCtx.threads[CROSS_WORKER_SLOT + 1];
            CHK_RET(BeginPeerRead(crossThread, *crossChannel, *crossChannel));
            for (uint32_t index = 0; index < RANKS_PER_SERVER; ++index) {
                CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(crossThread,
                    crossChannel->handle,
                    pairPartialBase + static_cast<uint64_t>(index) * pairBytes,
                    static_cast<uint8_t *>(crossChannel->remoteCclMem.addr) +
                        COMPETITION_RANK_SIZE * meshBytes +
                        static_cast<uint64_t>(index) * pairBytes,
                    pairFirstCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            }
            CHK_RET(EndPeerRead(crossThread, *crossChannel, *crossChannel));
        }
        CHK_RET(FinishWorkers(resCtx.threads, stageOneWorkers));

        if (meshFirstCount > 0) {
            for (uint32_t index = 0; index < localChannels.size(); ++index) {
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread,
                    meshPartialBase,
                    meshReceiveBase + static_cast<uint64_t>(index * 2) * meshBytes,
                    meshFirstCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread,
                    meshPartialBase + meshBytes,
                    meshReceiveBase + static_cast<uint64_t>(index * 2 + 1) * meshBytes,
                    meshFirstCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            }
        }

        uint8_t *outputChunk = static_cast<uint8_t *>(param.outputPtr) +
            processedCount * ELEMENT_BYTES;
        if (meshFirstCount > 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
                outputChunk, meshPartialBase + static_cast<uint64_t>(serverIndex) * meshBytes,
                meshBytes)));
        }
        if (pairFirstCount > 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
                outputChunk + meshBytes,
                pairPartialBase + static_cast<uint64_t>(localId) * pairBytes,
                pairBytes)));
        }

        // 阶段 2：q 走 Pair，1-q 同时走 Mesh。
        std::vector<uint32_t> stageTwoWorkers;
        if (pairFirstCount > 0) {
            for (uint32_t slot = 0; slot < LOCAL_WORKER_COUNT; ++slot) {
                stageTwoWorkers.push_back(slot);
            }
        }
        if (meshFirstCount > 0) {
            stageTwoWorkers.push_back(CROSS_WORKER_SLOT);
        }
        CHK_RET(StartWorkers(resCtx.threads, stageTwoWorkers));
        if (pairFirstCount > 0) {
            for (uint32_t index = 0; index < localChannels.size(); ++index) {
                const ChannelInfo &channel = *localChannels[index];
                const ThreadHandle worker = resCtx.threads[index + 1];
                CHK_RET(BeginPeerRead(worker, channel, channel));
                const uint64_t remotePairPartialOffset =
                    COMPETITION_RANK_SIZE * meshBytes +
                    RANKS_PER_SERVER * pairBytes +
                    2 * LOCAL_WORKER_COUNT * meshBytes + 2 * meshBytes;
                CHK_RET(static_cast<HcclResult>(HcommReadOnThread(worker,
                    channel.handle,
                    pairReceiveBase + static_cast<uint64_t>(index) * pairBytes,
                    static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                        remotePairPartialOffset + static_cast<uint64_t>(localId) * pairBytes,
                    pairBytes)));
                CHK_RET(EndPeerRead(worker, channel, channel));
            }
        }
        if (meshFirstCount > 0) {
            const ThreadHandle crossThread = resCtx.threads[CROSS_WORKER_SLOT + 1];
            const uint64_t remoteMeshPartialOffset =
                COMPETITION_RANK_SIZE * meshBytes +
                RANKS_PER_SERVER * pairBytes +
                2 * LOCAL_WORKER_COUNT * meshBytes;
            CHK_RET(BeginPeerRead(crossThread, *crossChannel, *crossChannel));
            CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(crossThread,
                crossChannel->handle, outputChunk,
                static_cast<uint8_t *>(crossChannel->remoteCclMem.addr) +
                    remoteMeshPartialOffset + static_cast<uint64_t>(serverIndex) * meshBytes,
                meshFirstCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            CHK_RET(EndPeerRead(crossThread, *crossChannel, *crossChannel));
        }
        CHK_RET(FinishWorkers(resCtx.threads, stageTwoWorkers));

        if (pairFirstCount > 0) {
            for (uint32_t index = 0; index < localChannels.size(); ++index) {
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread,
                    outputChunk + meshBytes,
                    pairReceiveBase + static_cast<uint64_t>(index) * pairBytes,
                    pairFirstCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            }
        }
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("unsupported ReduceScatter type[%d] or op[%d]",
            static_cast<int32_t>(param.dataType), static_cast<int32_t>(param.reduceType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    CHK_PRT_RET(resCtx.threads.empty(),
        HCCL_ERROR("no AICPU thread was allocated"), HCCL_E_INTERNAL);

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.aicpuThread,
            param.outputPtr, param.inputPtr, param.count * ELEMENT_BYTES)));
        return HCCL_SUCCESS;
    }

    const uint64_t usableBytes = GetUsableBufferBytes(resCtx);
    CHK_PRT_RET(usableBytes == 0,
        HCCL_ERROR("invalid local or remote HCCL buffer"), HCCL_E_INTERNAL);
    if (UseTinySpecialPath(param)) {
#if HCCL_RS_TINY_MODE == RS_TINY_MODE_DIRECT_15_WORKER
        return RunTinyDirectAllPeerPush(param, resCtx, usableBytes);
#else
        return RunTinyHierarchyPush(param, resCtx, usableBytes);
#endif
    }
    if (UseTopologyXorPath(param) && param.count == 1) {
        return RunTopologyXorPullReduceTinyUnrolled(param, resCtx, usableBytes);
    }
    HCCL_INFO("RS_CANDIDATE[%s] rank[%u/%u] count[%llu] path[%s] usableBytes[%llu]",
        GetAlgorithmName(), param.myRank, param.rankSize,
        static_cast<unsigned long long>(param.count),
        UseTopologyXorPath(param) ? "xor-pull-final-copy-overlap" :
            (UseNhrPath(param) ? "optimized-nhr" :
                (UsePackedStripePath(param) ? "packed-stripe-pingpong" :
                    (UseReferenceV2SingleOuterPath(param) ?
                        "reference-v2-four-stripe-pair-reduce" :
                        "eager-direct-push-pipeline"))),
        static_cast<unsigned long long>(usableBytes));

    if (UseTopologyXorPath(param)) {
        return RunTopologyXorPullReduceFinalCopyOverlap(param, resCtx, usableBytes);
    }
    if (UseNhrPath(param)) {
        return RunNhr(param, resCtx, usableBytes);
    }
    if (UsePackedStripePath(param)) {
        return RunPackedStripePush(param, resCtx, usableBytes);
    }
    if (UseReferenceV2SingleOuterPath(param)) {
        return RunDirectPush(param, resCtx, usableBytes);
    }
    return RunEagerDirectPush(param, resCtx, usableBytes);
}
} // namespace ops_hccl
