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
#include <vector>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
    constexpr uint32_t EXPECTED_RANK_SIZE = 16;
    constexpr uint32_t RANKS_PER_SERVER = 8;
    constexpr uint32_t LOCAL_STEP_NUM = 3;
    constexpr uint32_t MESH_WORKER_NUM = 7;
    constexpr uint32_t MESH_SCRATCH_SLOT_NUM = 8;
    constexpr uint32_t EXPECTED_PARALLEL_THREAD_NUM = 24;
    constexpr uint32_t MAIN_THREAD_INDEX = 0;
    constexpr uint32_t MESH_MAIN_THREAD_INDEX = 1;
    constexpr uint32_t MESH_WORKER_THREAD_BEGIN = 2;
    constexpr uint32_t INTER_THREAD_INDEX = 9;
    constexpr uint32_t DIRECT_PEER_NUM = EXPECTED_RANK_SIZE - 1;
    constexpr uint32_t DIRECT_WORKER_THREAD_BEGIN = 1;
    constexpr uint32_t DIRECT_NETWORK_THREAD_NUM = EXPECTED_RANK_SIZE;
    constexpr uint32_t DIRECT_REDUCE_THREAD_NUM = EXPECTED_RANK_SIZE / 2;
    constexpr uint32_t DIRECT_REDUCE_THREAD_BEGIN = DIRECT_NETWORK_THREAD_NUM;
    constexpr uint32_t DIRECT_REDUCE_LEVEL_NUM = 4;
    constexpr uint32_t DIRECT_MAX_TILE_NUM = 2;
    constexpr uint32_t DIRECT_MAIN_AUX_NOTIFY_BEGIN = 8;
    constexpr uint32_t DIRECT_MAIN_ROOT_NOTIFY = 23;
    constexpr uint32_t DIRECT_WORKER_TREE_NOTIFY_BEGIN = 8;
    constexpr uint32_t LARGE_TILE_NUM = 2;
    constexpr uint32_t THREAD_NOTIFY_NUM = 26;
    constexpr uint32_t THREAD_START_NOTIFY = THREAD_NOTIFY_NUM - 1;
    constexpr uint32_t MESH_COMPLETE_NOTIFY = 1;
    constexpr uint32_t INTER_COMPLETE_NOTIFY = 2;
    constexpr uint32_t SMALL_MESH_COMPLETE_NOTIFY = MESH_WORKER_NUM;
    constexpr uint32_t SMALL_INTER_COMPLETE_NOTIFY = MESH_WORKER_NUM + 1;
    constexpr uint32_t LARGE_LOCAL_COPY_NOTIFY_BEGIN = LARGE_TILE_NUM * MESH_WORKER_NUM;
    constexpr uint32_t LARGE_INTER_AG_NOTIFY_BEGIN = LARGE_LOCAL_COPY_NOTIFY_BEGIN + LARGE_TILE_NUM;
    constexpr uint32_t SMALL_PARTNER_MASKS[] = {1, 2, 4, 8};
    constexpr uint64_t PERFORMANCE_SMALL_BYTES = 512ULL * 1024;
    constexpr uint64_t PERFORMANCE_LARGE_BYTES = 512ULL * 1024 * 1024;
    constexpr uint64_t PERFORMANCE_UNEVEN_BYTES = 400ULL * 1024 * 1024 + sizeof(float);
    constexpr uint64_t SMALL_MESSAGE_BYTES = 32ULL * 1024 * 1024;
    constexpr uint64_t COMPETITION_CCL_BYTES = 400ULL * 1024 * 1024;
    constexpr uint64_t MAX_TASK_BYTES = 256ULL * 1024 * 1024;
    constexpr uint64_t CCL_ADDRESS_ALIGNMENT = 64;
    constexpr uint64_t SMALL_SHARD_BYTES = PERFORMANCE_SMALL_BYTES / RANKS_PER_SERVER;
    constexpr uint64_t SMALL_SHARD_COUNT = SMALL_SHARD_BYTES / sizeof(float);
    constexpr uint64_t SMALL_LOCAL_PUBLISH_OFFSET = MESH_WORKER_NUM * SMALL_SHARD_BYTES;
    constexpr uint64_t SMALL_INTER_REMOTE_OFFSET = SMALL_LOCAL_PUBLISH_OFFSET + SMALL_SHARD_BYTES;
    constexpr uint64_t SMALL_CCL_BYTES = SMALL_INTER_REMOTE_OFFSET + SMALL_SHARD_BYTES;
    constexpr uint32_t SMALL_DATA_NOTIFY = 0;
    constexpr uint32_t SMALL_CONSUMED_NOTIFY = 1;
    constexpr uint64_t SMALL_RS_SLOT0_OFFSET = 0;
    constexpr uint64_t SMALL_RS_SLOT1_OFFSET = 256ULL * 1024;
    constexpr uint64_t SMALL_RS_SLOT2_OFFSET = 384ULL * 1024;
    constexpr uint64_t SMALL_INTER_SLOT_OFFSET = 448ULL * 1024;
    constexpr uint64_t SMALL_AG_SLOT0_OFFSET = 512ULL * 1024;
    constexpr uint64_t SMALL_AG_SLOT1_OFFSET = 576ULL * 1024;
    constexpr uint64_t SMALL_AG_SLOT2_OFFSET = 704ULL * 1024;
    constexpr uint64_t SMALL_RHD_CCL_BYTES = 960ULL * 1024;
    constexpr uint64_t SMALL_ROOT_SLOT_BYTES = PERFORMANCE_SMALL_BYTES;
    constexpr uint64_t SMALL_ROOT_CCL_BYTES = EXPECTED_RANK_SIZE * SMALL_ROOT_SLOT_BYTES;
    constexpr uint32_t SMALL_ROOT_INPUT_READY_NOTIFY = 0;
    constexpr uint32_t SMALL_ROOT_RESULT_READY_NOTIFY = 1;
    constexpr uint32_t SMALL_ROOT_RESULT_CONSUMED_NOTIFY = 2;
    constexpr uint32_t SMALL_ROOT_TREE_NOTIFY_BEGIN = 0;
    constexpr uint64_t SMALL_NATIVE_RHD_CCL_BYTES = PERFORMANCE_SMALL_BYTES;
    constexpr uint32_t SMALL_NATIVE_RS_READY_NOTIFY = 0;
    constexpr uint32_t SMALL_NATIVE_AG_DATA_NOTIFY = 1;
    constexpr uint32_t SMALL_NATIVE_PARTIAL_READY_NOTIFY = 0;
    constexpr uint32_t SMALL_NATIVE_RESULT_READY_NOTIFY = 1;
    constexpr uint32_t SMALL_NATIVE_RESULT_CONSUMED_NOTIFY = 2;
    constexpr uint32_t SMALL_CROSS_DATA_NOTIFY = 0;
    constexpr uint32_t SMALL_CROSS_MASKS[] = {8, 12, 10, 9};
    constexpr uint64_t SMALL_CROSS_CCL_BYTES = PERFORMANCE_SMALL_BYTES * 4;

    constexpr uint64_t DIRECT_NESTED_SEGMENT0_BYTES = 24ULL * 1024 * 1024;
    constexpr uint64_t DIRECT_NESTED_SEGMENT1_BYTES = 8ULL * 1024 * 1024;
    constexpr uint64_t DIRECT_NESTED_SHARD_BYTES =
        DIRECT_NESTED_SEGMENT0_BYTES + DIRECT_NESTED_SEGMENT1_BYTES;
    constexpr uint64_t DIRECT_NESTED_PUBLISH_OFFSET =
        DIRECT_PEER_NUM * DIRECT_NESTED_SEGMENT0_BYTES;
    constexpr uint64_t DIRECT_NESTED_CCL_BYTES =
        DIRECT_NESTED_PUBLISH_OFFSET + DIRECT_NESTED_SHARD_BYTES;
    constexpr uint32_t DIRECT_NESTED_TREE_NOTIFY_BEGIN = 0;
    constexpr uint32_t DIRECT_NESTED_MAIN_DONE_NOTIFY_BEGIN = 8;
    constexpr uint32_t DIRECT_NESTED_ROOT_DONE_NOTIFY = 23;
    constexpr uint32_t DIRECT_NESTED_PUBLISH_READY_THREAD_NOTIFY = 24;
    constexpr uint32_t DIRECT_OVERLAP_FIRST_TREE_NOTIFY_BEGIN = 1;
    constexpr uint32_t DIRECT_OVERLAP_SECOND_TREE_NOTIFY_BEGIN = 5;
    constexpr uint32_t DIRECT_OVERLAP_AG_DONE_NOTIFY_BEGIN = 8;
    constexpr uint32_t DIRECT_OVERLAP_ROOT_DONE_NOTIFY = 23;
    constexpr uint32_t DIRECT_OVERLAP_AG_START_NOTIFY = 24;

    struct DataRange {
        uint64_t begin = 0;
        uint64_t count = 0;
    };

    struct DirectOverlapReduceState {
        std::array<void *, DIRECT_REDUCE_THREAD_NUM> results{};
        std::array<ThreadHandle, DIRECT_REDUCE_THREAD_NUM> threads{};
    };

    struct LargeConfig {
        uint64_t tileCount = 0;
        uint64_t lastTileCount = 0;
        uint64_t halfSlotStride = 0;
    };

    struct LargeTileLayout {
        DataRange tileRange;
        DataRange serverRange;
        DataRange ownedRange;
        DataRange otherServerRange;
        uint64_t slotBase = 0;
        uint64_t halfBytes = 0;
        uint64_t ownedOffset = 0;
        uint64_t ownedBytes = 0;
        uint64_t scratchStride = 0;
    };

    void *AddOffset(void *base, uint64_t offset)
    {
        return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
    }

    const void *AddOffset(const void *base, uint64_t offset)
    {
        return static_cast<const void *>(static_cast<const uint8_t *>(base) + offset);
    }

    uint64_t RangeBytes(const DataRange &range)
    {
        return range.count * sizeof(float);
    }

    uint64_t PartOffset(uint64_t totalCount, uint32_t partNum, uint32_t partIdx)
    {
        uint64_t baseCount = totalCount / partNum;
        // 与原仓模板一致，余数全部落到最后一个分片。
        return partIdx < partNum ? baseCount * partIdx : totalCount;
    }

    DataRange SplitRange(const DataRange &range, uint32_t partNum, uint32_t partIdx)
    {
        uint64_t beginOffset = PartOffset(range.count, partNum, partIdx);
        uint64_t endOffset = PartOffset(range.count, partNum, partIdx + 1);
        return DataRange{range.begin + beginOffset, endOffset - beginOffset};
    }

    uint64_t MaxPartCount(uint64_t totalCount, uint32_t partNum)
    {
        return totalCount / partNum + totalCount % partNum;
    }

    uint32_t ExcludedSlot(uint32_t value, uint32_t excluded)
    {
        return value < excluded ? value : value - 1;
    }

    uint64_t MeshScratchBytes(uint64_t rangeCount)
    {
        return MESH_SCRATCH_SLOT_NUM * MaxPartCount(rangeCount, RANKS_PER_SERVER) * sizeof(float);
    }

    uint64_t AlignUp(uint64_t value, uint64_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    [[maybe_unused]] bool GetLargeConfig(uint64_t totalBytes, LargeConfig &config)
    {
        if (totalBytes == PERFORMANCE_LARGE_BYTES) {
            config.tileCount = PERFORMANCE_LARGE_BYTES / sizeof(float) / LARGE_TILE_NUM;
            config.lastTileCount = config.tileCount;
            config.halfSlotStride = 128ULL * 1024 * 1024;
            return true;
        }
        if (totalBytes == PERFORMANCE_UNEVEN_BYTES) {
            config.tileCount = (200ULL * 1024 * 1024) / sizeof(float);
            config.lastTileCount = config.tileCount + 1;
            config.halfSlotStride = AlignUp(100ULL * 1024 * 1024 + sizeof(float), CCL_ADDRESS_ALIGNMENT);
            return true;
        }
        return false;
    }

    DataRange LargeTileRange(const LargeConfig &config, uint32_t tileIdx)
    {
        return DataRange{config.tileCount * tileIdx,
            tileIdx + 1 == LARGE_TILE_NUM ? config.lastTileCount : config.tileCount};
    }

    LargeTileLayout GetLargeTileLayout(
        const LargeConfig &config, uint32_t tileIdx, uint32_t serverIdx, uint32_t localRank)
    {
        DataRange tileRange = LargeTileRange(config, tileIdx);
        DataRange serverRange = SplitRange(tileRange, 2, serverIdx);
        DataRange ownedRange = SplitRange(serverRange, RANKS_PER_SERVER, localRank);
        return LargeTileLayout{tileRange, serverRange, ownedRange, SplitRange(tileRange, 2, serverIdx ^ 1),
            config.halfSlotStride * tileIdx, RangeBytes(serverRange),
            (ownedRange.begin - serverRange.begin) * sizeof(float), RangeBytes(ownedRange),
            AlignUp(RangeBytes(ownedRange), CCL_ADDRESS_ALIGNMENT)};
    }

    uint64_t LargeScratchOffset(const LargeTileLayout &layout, uint32_t sourceLocalRank)
    {
        return layout.otherServerRange.begin * sizeof(float) + (sourceLocalRank - 1) * layout.scratchStride;
    }

    uint32_t TileWorkerNotify(uint32_t tileIdx, uint32_t workerIdx)
    {
        return tileIdx * MESH_WORKER_NUM + workerIdx;
    }

    uint32_t WorkerInterReadyNotify(uint32_t tileIdx)
    {
        return tileIdx;
    }

    uint32_t WorkerPublishReadyNotify(uint32_t tileIdx)
    {
        return LARGE_TILE_NUM + tileIdx;
    }

    uint32_t MainLocalCopyDoneNotify(uint32_t tileIdx)
    {
        return LARGE_LOCAL_COPY_NOTIFY_BEGIN + tileIdx;
    }

    uint32_t MainInterAgDoneNotify(uint32_t tileIdx)
    {
        return LARGE_INTER_AG_NOTIFY_BEGIN + tileIdx;
    }

    const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank, ChannelType type)
    {
        uint32_t typeIdx = static_cast<uint32_t>(type);
        uint64_t indexPos = static_cast<uint64_t>(typeIdx) * EXPECTED_RANK_SIZE + remoteRank;
        if (indexPos < resCtx.channelIndices.size()) {
            uint32_t channelIdx = resCtx.channelIndices[indexPos];
            if (channelIdx < resCtx.channels.size()) {
                return &resCtx.channels[channelIdx];
            }
        }
        for (const ChannelInfo &channel : resCtx.channels) {
            if (channel.remoteRank == remoteRank && channel.type == type) {
                return &channel;
            }
        }
        return nullptr;
    }

    HcclResult CheckBufferRange(const CommBuffer &buffer, uint64_t offset, uint64_t bytes, const char *name)
    {
        CHK_PRT_RET(offset > buffer.size || bytes > buffer.size - offset,
            HCCL_ERROR("%s buffer range exceeds capacity, offset[%llu], bytes[%llu], capacity[%llu]", name,
                static_cast<unsigned long long>(offset), static_cast<unsigned long long>(bytes),
                static_cast<unsigned long long>(buffer.size)),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    HcclResult CheckUserRange(uint64_t totalBytes, uint64_t offset, uint64_t bytes, const char *name)
    {
        CHK_PRT_RET(offset > totalBytes || bytes > totalBytes - offset,
            HCCL_ERROR("%s range exceeds user buffer, offset[%llu], bytes[%llu], capacity[%llu]", name,
                static_cast<unsigned long long>(offset), static_cast<unsigned long long>(bytes),
                static_cast<unsigned long long>(totalBytes)),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    HcclResult CopyBytes(ThreadHandle thread, void *dst, const void *src, uint64_t bytes)
    {
        if (dst == src || bytes == 0) {
            return HCCL_SUCCESS;
        }

        uint64_t copiedBytes = 0;
        while (copiedBytes < bytes) {
            uint64_t chunkBytes = std::min(MAX_TASK_BYTES, bytes - copiedBytes);
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, AddOffset(dst, copiedBytes), AddOffset(src, copiedBytes), chunkBytes)));
            copiedBytes += chunkBytes;
        }
        return HCCL_SUCCESS;
    }

    HcclResult ReduceRange(const OpParam &param, ThreadHandle thread, void *dst, const void *src, uint64_t count)
    {
        if (count == 0) {
            return HCCL_SUCCESS;
        }
        return static_cast<HcclResult>(HcommLocalReduceOnThread(thread, dst, src, count,
            static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType)));
    }

    HcclResult StartThread(ThreadHandle source, ThreadHandle target)
    {
        return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(source, target, THREAD_START_NOTIFY));
    }

    HcclResult WaitThreadStart(ThreadHandle thread)
    {
        return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, THREAD_START_NOTIFY, CUSTOM_TIMEOUT));
    }

    HcclResult SignalThreadComplete(ThreadHandle source, ThreadHandle target, uint32_t notifyIdx)
    {
        return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(source, target, notifyIdx));
    }

    HcclResult WaitThreadComplete(ThreadHandle thread, uint32_t notifyIdx)
    {
        return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, notifyIdx, CUSTOM_TIMEOUT));
    }

    HcclResult RecordChannel(ThreadHandle thread, const ChannelInfo &channel, uint32_t notifyIdx)
    {
        return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyIdx));
    }

    HcclResult WaitChannel(ThreadHandle thread, const ChannelInfo &channel, uint32_t notifyIdx)
    {
        return static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, notifyIdx, CUSTOM_TIMEOUT));
    }

    HcclResult BeginChannelOp(ThreadHandle thread, const ChannelInfo &channel)
    {
        CHK_RET(RecordChannel(thread, channel, NOTIFY_IDX_ACK));
        return WaitChannel(thread, channel, NOTIFY_IDX_ACK);
    }

    HcclResult EndChannelOp(ThreadHandle thread, const ChannelInfo &channel)
    {
        CHK_RET(RecordChannel(thread, channel, NOTIFY_IDX_DATA_SIGNAL));
        return WaitChannel(thread, channel, NOTIFY_IDX_DATA_SIGNAL);
    }

    HcclResult BatchWriteSignal(ThreadHandle thread, const ChannelInfo &channel, void *remoteDst,
        const void *localSrc, uint64_t bytes, uint32_t notifyIdx)
    {
        HcommBatchTransferDesc descs[2]{};
        descs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
        descs[0].transferInfo.write.dst = remoteDst;
        descs[0].transferInfo.write.src = const_cast<void *>(localSrc);
        descs[0].transferInfo.write.len = bytes;
        descs[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
        descs[1].transferInfo.notifyRecord.notifyIdx = notifyIdx;
        return static_cast<HcclResult>(HcommBatchTransferOnThread(thread, channel.handle, descs, 2));
    }

    HcclResult CrossServerButterflyAllReduce(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        CHK_RET(CheckBufferRange(
            resCtx.localBuffer, 0, SMALL_CROSS_CCL_BYTES, "Cross Butterfly local scratch"));
        ThreadHandle mainThread = resCtx.threads[MAIN_THREAD_INDEX];
        if (param.inputPtr != param.outputPtr) {
            CHK_RET(CopyBytes(mainThread, param.outputPtr, param.inputPtr, PERFORMANCE_SMALL_BYTES));
        }

        for (uint32_t step = 0; step < sizeof(SMALL_CROSS_MASKS) / sizeof(SMALL_CROSS_MASKS[0]); step++) {
            uint32_t remoteRank = param.myRank ^ SMALL_CROSS_MASKS[step];
            const ChannelInfo *channel = FindChannel(resCtx, remoteRank, ChannelType::NHR);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Cross Butterfly channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
            uint64_t scratchOffset = step * PERFORMANCE_SMALL_BYTES;
            CHK_RET(CheckBufferRange(channel->remoteCclMem, scratchOffset,
                PERFORMANCE_SMALL_BYTES, "Cross Butterfly remote scratch"));

            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(mainThread, channel->handle,
                AddOffset(channel->remoteCclMem.addr, scratchOffset), param.outputPtr,
                PERFORMANCE_SMALL_BYTES)));
            CHK_RET(RecordChannel(mainThread, *channel, SMALL_CROSS_DATA_NOTIFY));
            CHK_RET(WaitChannel(mainThread, *channel, SMALL_CROSS_DATA_NOTIFY));
            CHK_RET(ReduceRange(param, mainThread, param.outputPtr,
                AddOffset(resCtx.localBuffer.addr, scratchOffset), param.count));
        }
        return HCCL_SUCCESS;
    }

    HcclResult ExchangeReduce(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread,
        uint32_t remoteRank, void *sendAddr, uint64_t bytes, void *reduceAddr, uint64_t count, uint64_t cclOffset)
    {
        ChannelType type = remoteRank / RANKS_PER_SERVER == param.myRank / RANKS_PER_SERVER
                               ? ChannelType::MESH
                               : ChannelType::NHR;
        const ChannelInfo *channel = FindChannel(resCtx, remoteRank, type);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
        CHK_RET(CheckBufferRange(channel->remoteCclMem, cclOffset, bytes, "Remote CCL"));
        CHK_RET(CheckBufferRange(resCtx.localBuffer, cclOffset, bytes, "Local CCL"));

        CHK_RET(RecordChannel(thread, *channel, NOTIFY_IDX_ACK));
        CHK_RET(WaitChannel(thread, *channel, NOTIFY_IDX_ACK));
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
            thread, channel->handle, AddOffset(channel->remoteCclMem.addr, cclOffset), sendAddr, bytes)));
        CHK_RET(RecordChannel(thread, *channel, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(WaitChannel(thread, *channel, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(ReduceRange(param, thread, reduceAddr, AddOffset(resCtx.localBuffer.addr, cclOffset), count));
        return HCCL_SUCCESS;
    }

    HcclResult RecursiveDoublingAllReduce(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread,
        const void *inputAddr, void *outputAddr, uint64_t count)
    {
        uint64_t totalBytes = count * sizeof(float);
        CHK_RET(CopyBytes(thread, outputAddr, inputAddr, totalBytes));
        uint32_t step = 0;
        for (uint32_t mask : SMALL_PARTNER_MASKS) {
            uint64_t cclOffset = totalBytes * step;
            CHK_RET(ExchangeReduce(
                param, resCtx, thread, param.myRank ^ mask, outputAddr, totalBytes, outputAddr, count, cclOffset));
            step++;
        }
        return HCCL_SUCCESS;
    }

    [[maybe_unused]] HcclResult SmallNativeRhdAllReduce(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        uint64_t totalBytes = param.count * sizeof(float);
        CHK_PRT_RET(totalBytes != PERFORMANCE_SMALL_BYTES,
            HCCL_ERROR("Invalid native RHD bytes[%llu]", static_cast<unsigned long long>(totalBytes)),
            HCCL_E_INTERNAL);
        CHK_RET(CheckBufferRange(
            resCtx.localBuffer, 0, SMALL_NATIVE_RHD_CCL_BYTES, "Native RHD local CCL"));
        CHK_RET(CheckUserRange(totalBytes, 0, totalBytes, "Native RHD input/output"));

        ThreadHandle thread = resCtx.threads[MAIN_THREAD_INDEX];
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        uint32_t serverBase = param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
        CHK_RET(CopyBytes(thread, resCtx.localBuffer.addr, param.inputPtr, totalBytes));

        DataRange ownedRange{0, param.count};
        for (uint32_t step = 0; step < LOCAL_STEP_NUM; step++) {
            uint32_t mask = 1U << step;
            uint32_t peerRank = serverBase + (localRank ^ mask);
            const ChannelInfo *channel = FindChannel(resCtx, peerRank, ChannelType::MESH);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Native RHD RS channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);

            DataRange lowRange = SplitRange(ownedRange, 2, 0);
            DataRange highRange = SplitRange(ownedRange, 2, 1);
            DataRange keepRange = (localRank & mask) == 0 ? lowRange : highRange;
            uint64_t keepOffset = keepRange.begin * sizeof(float);
            uint64_t keepBytes = RangeBytes(keepRange);
            CHK_RET(CheckBufferRange(
                resCtx.localBuffer, keepOffset, keepBytes, "Native RHD RS local keep range"));
            CHK_RET(CheckBufferRange(
                channel->remoteCclMem, keepOffset, keepBytes, "Native RHD RS remote keep range"));

            CHK_RET(RecordChannel(thread, *channel, SMALL_NATIVE_RS_READY_NOTIFY));
            CHK_RET(WaitChannel(thread, *channel, SMALL_NATIVE_RS_READY_NOTIFY));
            CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, channel->handle,
                AddOffset(resCtx.localBuffer.addr, keepOffset),
                AddOffset(channel->remoteCclMem.addr, keepOffset), keepRange.count,
                static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
            ownedRange = keepRange;
        }

        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t interRank = param.myRank ^ RANKS_PER_SERVER;
        const ChannelInfo *interChannel = FindChannel(resCtx, interRank, ChannelType::NHR);
        CHK_PRT_RET(interChannel == nullptr,
            HCCL_ERROR("Native RHD inter channel to rank[%u] not found", interRank), HCCL_E_INTERNAL);
        uint64_t ownedOffset = ownedRange.begin * sizeof(float);
        uint64_t ownedBytes = RangeBytes(ownedRange);
        CHK_RET(CheckBufferRange(
            resCtx.localBuffer, ownedOffset, ownedBytes, "Native RHD inter local range"));
        CHK_RET(CheckBufferRange(
            interChannel->remoteCclMem, ownedOffset, ownedBytes, "Native RHD inter remote range"));

        if (serverIdx == 1) {
            CHK_RET(RecordChannel(thread, *interChannel, SMALL_NATIVE_PARTIAL_READY_NOTIFY));
            CHK_RET(WaitChannel(thread, *interChannel, SMALL_NATIVE_RESULT_READY_NOTIFY));
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, interChannel->handle,
                AddOffset(resCtx.localBuffer.addr, ownedOffset),
                AddOffset(interChannel->remoteCclMem.addr, ownedOffset), ownedBytes)));
            CHK_RET(RecordChannel(thread, *interChannel, SMALL_NATIVE_RESULT_CONSUMED_NOTIFY));
        } else {
            CHK_RET(WaitChannel(thread, *interChannel, SMALL_NATIVE_PARTIAL_READY_NOTIFY));
            CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, interChannel->handle,
                AddOffset(resCtx.localBuffer.addr, ownedOffset),
                AddOffset(interChannel->remoteCclMem.addr, ownedOffset), ownedRange.count,
                static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
            CHK_RET(RecordChannel(thread, *interChannel, SMALL_NATIVE_RESULT_READY_NOTIFY));
            CHK_RET(WaitChannel(thread, *interChannel, SMALL_NATIVE_RESULT_CONSUMED_NOTIFY));
        }

        for (uint32_t step = LOCAL_STEP_NUM; step > 0; step--) {
            uint32_t mask = 1U << (step - 1);
            uint32_t peerRank = serverBase + (localRank ^ mask);
            const ChannelInfo *channel = FindChannel(resCtx, peerRank, ChannelType::MESH);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Native RHD AG channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);

            uint64_t sendOffset = ownedRange.begin * sizeof(float);
            uint64_t sendBytes = RangeBytes(ownedRange);
            uint64_t missingBegin = (localRank & mask) == 0
                                        ? ownedRange.begin + ownedRange.count
                                        : ownedRange.begin - ownedRange.count;
            CHK_RET(CheckBufferRange(
                resCtx.localBuffer, sendOffset, sendBytes, "Native RHD AG local owned range"));
            CHK_RET(CheckBufferRange(
                channel->remoteCclMem, sendOffset, sendBytes, "Native RHD AG remote missing range"));
            CHK_RET(CheckBufferRange(resCtx.localBuffer, missingBegin * sizeof(float), sendBytes,
                "Native RHD AG local missing range"));

            CHK_RET(BatchWriteSignal(thread, *channel,
                AddOffset(channel->remoteCclMem.addr, sendOffset),
                AddOffset(resCtx.localBuffer.addr, sendOffset), sendBytes,
                SMALL_NATIVE_AG_DATA_NOTIFY));
            CHK_RET(WaitChannel(thread, *channel, SMALL_NATIVE_AG_DATA_NOTIFY));
            ownedRange = DataRange{
                std::min(ownedRange.begin, missingBegin), ownedRange.count * 2};
        }

        CHK_PRT_RET(ownedRange.begin != 0 || RangeBytes(ownedRange) != totalBytes,
            HCCL_ERROR("Native RHD final range is incomplete"), HCCL_E_INTERNAL);
        return CopyBytes(thread, param.outputPtr, resCtx.localBuffer.addr, totalBytes);
    }

    uint64_t SmallRootSlotOffset(uint32_t sourceRank)
    {
        return sourceRank * SMALL_ROOT_SLOT_BYTES;
    }

    ThreadHandle SmallRootPairThread(const AlgResourceCtx &resCtx, uint32_t pairIdx)
    {
        return pairIdx == 0
                   ? resCtx.threads[MAIN_THREAD_INDEX]
                   : resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + pairIdx - 1];
    }

    [[maybe_unused]] HcclResult SmallRootStarAllReduce(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        constexpr uint32_t rootRank = 0;
        uint64_t totalBytes = param.count * sizeof(float);
        CHK_PRT_RET(totalBytes != SMALL_ROOT_SLOT_BYTES,
            HCCL_ERROR("Invalid root-star bytes[%llu]", static_cast<unsigned long long>(totalBytes)),
            HCCL_E_INTERNAL);

        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        if (param.myRank != rootRank) {
            ChannelType channelType = param.myRank / RANKS_PER_SERVER == rootRank / RANKS_PER_SERVER
                                          ? ChannelType::MESH
                                          : ChannelType::NHR;
            const ChannelInfo *channel = FindChannel(
                resCtx, rootRank, channelType);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Small root-star channel to rank[0] not found"), HCCL_E_INTERNAL);
            uint64_t remoteOffset = SmallRootSlotOffset(param.myRank);
            CHK_RET(CheckBufferRange(
                channel->remoteCclMem, remoteOffset, totalBytes, "Small root-star remote input slot"));
            CHK_RET(CheckBufferRange(
                channel->remoteCclMem, SmallRootSlotOffset(rootRank), totalBytes,
                "Small root-star remote result slot"));
            CHK_RET(CheckUserRange(totalBytes, 0, totalBytes, "Small root-star input"));
            CHK_RET(CheckUserRange(totalBytes, 0, totalBytes, "Small root-star output"));

            CHK_RET(BatchWriteSignal(coordinator, *channel,
                AddOffset(channel->remoteCclMem.addr, remoteOffset), param.inputPtr, totalBytes,
                SMALL_ROOT_INPUT_READY_NOTIFY));
            CHK_RET(WaitChannel(coordinator, *channel, SMALL_ROOT_RESULT_READY_NOTIFY));
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(coordinator, channel->handle,
                param.outputPtr, AddOffset(channel->remoteCclMem.addr, SmallRootSlotOffset(rootRank)),
                totalBytes)));
            return RecordChannel(coordinator, *channel, SMALL_ROOT_RESULT_CONSUMED_NOTIFY);
        }

        CHK_RET(CheckBufferRange(resCtx.localBuffer, 0, SMALL_ROOT_CCL_BYTES, "Small root-star local CCL"));
        constexpr uint32_t firstLayerPairNum = EXPECTED_RANK_SIZE / 2;
        for (uint32_t pairIdx = 1; pairIdx < firstLayerPairNum; pairIdx++) {
            CHK_RET(StartThread(coordinator, SmallRootPairThread(resCtx, pairIdx)));
        }

        for (uint32_t pairIdx = 0; pairIdx < firstLayerPairNum; pairIdx++) {
            ThreadHandle pairThread = SmallRootPairThread(resCtx, pairIdx);
            if (pairIdx != 0) {
                CHK_RET(WaitThreadStart(pairThread));
            }
            uint32_t lowSource = pairIdx * 2;
            uint32_t highSource = lowSource + 1;
            if (lowSource == rootRank) {
                CHK_RET(CopyBytes(pairThread, resCtx.localBuffer.addr, param.inputPtr, totalBytes));
            } else {
                ChannelType lowType = lowSource < RANKS_PER_SERVER ? ChannelType::MESH : ChannelType::NHR;
                const ChannelInfo *lowChannel = FindChannel(resCtx, lowSource, lowType);
                CHK_PRT_RET(lowChannel == nullptr,
                    HCCL_ERROR("Small root-star channel to rank[%u] not found", lowSource),
                    HCCL_E_INTERNAL);
                CHK_RET(WaitChannel(pairThread, *lowChannel, SMALL_ROOT_INPUT_READY_NOTIFY));
            }
            ChannelType highType = highSource < RANKS_PER_SERVER ? ChannelType::MESH : ChannelType::NHR;
            const ChannelInfo *highChannel = FindChannel(resCtx, highSource, highType);
            CHK_PRT_RET(highChannel == nullptr,
                HCCL_ERROR("Small root-star channel to rank[%u] not found", highSource), HCCL_E_INTERNAL);
            CHK_RET(WaitChannel(pairThread, *highChannel, SMALL_ROOT_INPUT_READY_NOTIFY));
            CHK_RET(ReduceRange(param, pairThread,
                AddOffset(resCtx.localBuffer.addr, SmallRootSlotOffset(lowSource)),
                AddOffset(resCtx.localBuffer.addr, SmallRootSlotOffset(highSource)), param.count));
            if (pairIdx != 0) {
                CHK_RET(SignalThreadComplete(pairThread, coordinator,
                    SMALL_ROOT_TREE_NOTIFY_BEGIN + pairIdx - 1));
            }
        }

        for (uint32_t pairIdx = 1; pairIdx < firstLayerPairNum; pairIdx++) {
            CHK_RET(WaitThreadComplete(
                coordinator, SMALL_ROOT_TREE_NOTIFY_BEGIN + pairIdx - 1));
        }
        for (uint32_t level = 1; level < DIRECT_REDUCE_LEVEL_NUM; level++) {
            uint32_t groupSize = 1U << (level + 1);
            uint32_t groupNum = EXPECTED_RANK_SIZE / groupSize;
            for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
                uint32_t lowSource = groupIdx * groupSize;
                uint32_t highSource = lowSource + groupSize / 2;
                CHK_RET(ReduceRange(param, coordinator,
                    AddOffset(resCtx.localBuffer.addr, SmallRootSlotOffset(lowSource)),
                    AddOffset(resCtx.localBuffer.addr, SmallRootSlotOffset(highSource)), param.count));
            }
        }
        CHK_RET(CopyBytes(coordinator, param.outputPtr, resCtx.localBuffer.addr, totalBytes));

        for (uint32_t sourceRank = 1; sourceRank < EXPECTED_RANK_SIZE; sourceRank++) {
            ChannelType channelType = sourceRank / RANKS_PER_SERVER == rootRank / RANKS_PER_SERVER
                                          ? ChannelType::MESH
                                          : ChannelType::NHR;
            const ChannelInfo *channel = FindChannel(resCtx, sourceRank, channelType);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Small root-star channel to rank[%u] not found", sourceRank), HCCL_E_INTERNAL);
            CHK_RET(RecordChannel(coordinator, *channel, SMALL_ROOT_RESULT_READY_NOTIFY));
        }
        for (uint32_t sourceRank = 1; sourceRank < EXPECTED_RANK_SIZE; sourceRank++) {
            ChannelType channelType = sourceRank / RANKS_PER_SERVER == rootRank / RANKS_PER_SERVER
                                          ? ChannelType::MESH
                                          : ChannelType::NHR;
            const ChannelInfo *channel = FindChannel(resCtx, sourceRank, channelType);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Small root-star channel to rank[%u] not found", sourceRank), HCCL_E_INTERNAL);
            CHK_RET(WaitChannel(coordinator, *channel, SMALL_ROOT_RESULT_CONSUMED_NOTIFY));
        }
        return HCCL_SUCCESS;
    }

    uint64_t SmallRsSlotOffset(uint32_t step)
    {
        constexpr uint64_t offsets[LOCAL_STEP_NUM] = {
            SMALL_RS_SLOT0_OFFSET, SMALL_RS_SLOT1_OFFSET, SMALL_RS_SLOT2_OFFSET};
        return offsets[step];
    }

    uint64_t SmallAgSlotOffset(uint32_t reverseStep)
    {
        constexpr uint64_t offsets[LOCAL_STEP_NUM] = {
            SMALL_AG_SLOT0_OFFSET, SMALL_AG_SLOT1_OFFSET, SMALL_AG_SLOT2_OFFSET};
        return offsets[reverseStep];
    }

    [[maybe_unused]] HcclResult StaticSlotRhdAllReduce(
        const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread)
    {
        CHK_RET(CheckBufferRange(resCtx.localBuffer, 0, SMALL_RHD_CCL_BYTES, "Small static-slot RHD CCL"));
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        uint32_t serverBase = param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
        DataRange ownedRange{0, param.count};

        for (uint32_t step = 0; step < LOCAL_STEP_NUM; step++) {
            uint32_t mask = 1U << step;
            uint32_t peerRank = serverBase + (localRank ^ mask);
            const ChannelInfo *channel = FindChannel(resCtx, peerRank, ChannelType::MESH);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Small RHD mesh channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);

            DataRange lowRange = SplitRange(ownedRange, 2, 0);
            DataRange highRange = SplitRange(ownedRange, 2, 1);
            DataRange keepRange = (localRank & mask) == 0 ? lowRange : highRange;
            DataRange sendRange = (localRank & mask) == 0 ? highRange : lowRange;
            uint64_t slotOffset = SmallRsSlotOffset(step);
            uint64_t slotBytes = RangeBytes(keepRange);
            const void *sendBase = step == 0 ? param.inputPtr : param.outputPtr;
            CHK_RET(CheckUserRange(PERFORMANCE_SMALL_BYTES, sendRange.begin * sizeof(float),
                RangeBytes(sendRange), "Small RHD RS input"));
            CHK_RET(CheckBufferRange(resCtx.localBuffer, slotOffset, slotBytes, "Small RHD RS local slot"));
            CHK_RET(CheckBufferRange(channel->remoteCclMem, slotOffset, RangeBytes(sendRange),
                "Small RHD RS remote slot"));

            CHK_RET(BatchWriteSignal(thread, *channel, AddOffset(channel->remoteCclMem.addr, slotOffset),
                AddOffset(sendBase, sendRange.begin * sizeof(float)), RangeBytes(sendRange), SMALL_DATA_NOTIFY));
            CHK_RET(WaitChannel(thread, *channel, SMALL_DATA_NOTIFY));
            void *keepAddr = AddOffset(param.outputPtr, keepRange.begin * sizeof(float));
            if (step == 0) {
                CHK_RET(CopyBytes(thread, keepAddr,
                    AddOffset(param.inputPtr, keepRange.begin * sizeof(float)), RangeBytes(keepRange)));
            }
            CHK_RET(ReduceRange(param, thread, keepAddr,
                AddOffset(resCtx.localBuffer.addr, slotOffset), keepRange.count));
            ownedRange = keepRange;
        }

        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t interRank = param.myRank ^ RANKS_PER_SERVER;
        const ChannelInfo *interChannel = FindChannel(resCtx, interRank, ChannelType::NHR);
        CHK_PRT_RET(interChannel == nullptr,
            HCCL_ERROR("Small RHD inter channel to rank[%u] not found", interRank), HCCL_E_INTERNAL);
        uint64_t ownedBytes = RangeBytes(ownedRange);
        CHK_RET(CheckBufferRange(
            resCtx.localBuffer, SMALL_INTER_SLOT_OFFSET, ownedBytes, "Small RHD inter local slot"));
        CHK_RET(CheckBufferRange(
            interChannel->remoteCclMem, SMALL_INTER_SLOT_OFFSET, ownedBytes, "Small RHD inter remote slot"));
        CHK_RET(BatchWriteSignal(thread, *interChannel,
            AddOffset(interChannel->remoteCclMem.addr, SMALL_INTER_SLOT_OFFSET),
            AddOffset(param.outputPtr, ownedRange.begin * sizeof(float)), ownedBytes, SMALL_DATA_NOTIFY));
        CHK_RET(WaitChannel(thread, *interChannel, SMALL_DATA_NOTIFY));
        void *ownedAddr = AddOffset(param.outputPtr, ownedRange.begin * sizeof(float));
        void *remotePartial = AddOffset(resCtx.localBuffer.addr, SMALL_INTER_SLOT_OFFSET);
        if (serverIdx == 0) {
            CHK_RET(ReduceRange(param, thread, ownedAddr, remotePartial, ownedRange.count));
        } else {
            CHK_RET(ReduceRange(param, thread, remotePartial, ownedAddr, ownedRange.count));
            CHK_RET(CopyBytes(thread, ownedAddr, remotePartial, ownedBytes));
        }
        CHK_RET(RecordChannel(thread, *interChannel, SMALL_CONSUMED_NOTIFY));
        CHK_RET(WaitChannel(thread, *interChannel, SMALL_CONSUMED_NOTIFY));

        uint32_t reverseStep = 0;
        for (uint32_t step = LOCAL_STEP_NUM; step > 0; step--, reverseStep++) {
            uint32_t mask = 1U << (step - 1);
            uint32_t peerRank = serverBase + (localRank ^ mask);
            const ChannelInfo *channel = FindChannel(resCtx, peerRank, ChannelType::MESH);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Small RHD mesh channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);

            uint64_t expandedBegin =
                (localRank & mask) == 0 ? ownedRange.begin : ownedRange.begin - ownedRange.count;
            DataRange recvRange{expandedBegin, ownedRange.count};
            if ((localRank & mask) == 0) {
                recvRange.begin += ownedRange.count;
            }
            uint64_t slotOffset = SmallAgSlotOffset(reverseStep);
            uint64_t transferBytes = RangeBytes(ownedRange);
            CHK_RET(CheckBufferRange(
                resCtx.localBuffer, slotOffset, transferBytes, "Small RHD AG local slot"));
            CHK_RET(CheckBufferRange(
                channel->remoteCclMem, slotOffset, transferBytes, "Small RHD AG remote slot"));
            CHK_RET(CheckUserRange(PERFORMANCE_SMALL_BYTES, ownedRange.begin * sizeof(float),
                transferBytes, "Small RHD AG input"));
            CHK_RET(CheckUserRange(PERFORMANCE_SMALL_BYTES, recvRange.begin * sizeof(float),
                transferBytes, "Small RHD AG output"));
            CHK_RET(BatchWriteSignal(thread, *channel,
                AddOffset(channel->remoteCclMem.addr, slotOffset),
                AddOffset(param.outputPtr, ownedRange.begin * sizeof(float)), transferBytes, SMALL_DATA_NOTIFY));
            CHK_RET(WaitChannel(thread, *channel, SMALL_DATA_NOTIFY));
            CHK_RET(CopyBytes(thread, AddOffset(param.outputPtr, recvRange.begin * sizeof(float)),
                AddOffset(resCtx.localBuffer.addr, slotOffset), transferBytes));
            ownedRange = DataRange{
                std::min(ownedRange.begin, recvRange.begin), ownedRange.count + recvRange.count};
        }

        // 三轮超立方屏障传播“所有框内 RHD 槽均已消费”，允许下次算子安全复用静态槽。
        for (uint32_t mask = 1; mask < RANKS_PER_SERVER; mask <<= 1) {
            uint32_t peerRank = serverBase + (localRank ^ mask);
            const ChannelInfo *channel = FindChannel(resCtx, peerRank, ChannelType::MESH);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Small RHD barrier channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);
            CHK_RET(RecordChannel(thread, *channel, SMALL_CONSUMED_NOTIFY));
            CHK_RET(WaitChannel(thread, *channel, SMALL_CONSUMED_NOTIFY));
        }
        return HCCL_SUCCESS;
    }

    uint64_t SmallFinalCclOffset(uint32_t serverIdx)
    {
        return serverIdx == 0 ? SMALL_LOCAL_PUBLISH_OFFSET : SMALL_INTER_REMOTE_OFFSET;
    }

    HcclResult ValidateSmallLayout(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        CHK_RET(CheckBufferRange(resCtx.localBuffer, 0, SMALL_CCL_BYTES, "Small 576KiB local CCL"));
        for (uint32_t slotIdx = 0; slotIdx < MESH_WORKER_NUM; slotIdx++) {
            CHK_RET(CheckBufferRange(resCtx.localBuffer, slotIdx * SMALL_SHARD_BYTES, SMALL_SHARD_BYTES,
                "Small remote source slot"));
        }
        CHK_RET(CheckBufferRange(resCtx.localBuffer, SMALL_LOCAL_PUBLISH_OFFSET, SMALL_SHARD_BYTES,
            "Small local publish"));
        CHK_RET(CheckBufferRange(resCtx.localBuffer, SMALL_INTER_REMOTE_OFFSET, SMALL_SHARD_BYTES,
            "Small inter remote"));
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        CHK_RET(CheckUserRange(PERFORMANCE_SMALL_BYTES, localRank * SMALL_SHARD_BYTES, SMALL_SHARD_BYTES,
            "Small output owned shard"));
        return HCCL_SUCCESS;
    }

    HcclResult RunSmallMeshWorker(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t workerIdx)
    {
        ThreadHandle worker = resCtx.threads[MESH_WORKER_THREAD_BEGIN + workerIdx];
        CHK_RET(WaitThreadStart(worker));
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t serverBase = serverIdx * RANKS_PER_SERVER;
        uint32_t remoteLocalRank = workerIdx < localRank ? workerIdx : workerIdx + 1;
        uint32_t remoteRank = serverBase + remoteLocalRank;
        const ChannelInfo *channel = FindChannel(resCtx, remoteRank, ChannelType::MESH);
        CHK_PRT_RET(
            channel == nullptr, HCCL_ERROR("Small mesh channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);

        uint64_t remoteSourceOffset = ExcludedSlot(localRank, remoteLocalRank) * SMALL_SHARD_BYTES;
        CHK_RET(CheckBufferRange(
            channel->remoteCclMem, remoteSourceOffset, SMALL_SHARD_BYTES, "Small mesh RS remote source"));
        CHK_RET(CheckUserRange(PERFORMANCE_SMALL_BYTES, remoteLocalRank * SMALL_SHARD_BYTES, SMALL_SHARD_BYTES,
            "Small mesh RS input shard"));
        CHK_RET(BeginChannelOp(worker, *channel));
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(worker, channel->handle,
            AddOffset(channel->remoteCclMem.addr, remoteSourceOffset),
            AddOffset(param.inputPtr, remoteLocalRank * SMALL_SHARD_BYTES), SMALL_SHARD_BYTES)));
        CHK_RET(EndChannelOp(worker, *channel));
        CHK_RET(SignalThreadComplete(worker, resCtx.threads[MAIN_THREAD_INDEX], workerIdx));

        CHK_RET(WaitThreadComplete(worker, 0));
        uint64_t remoteFinalOffset = SmallFinalCclOffset(serverIdx);
        CHK_RET(CheckBufferRange(
            channel->remoteCclMem, remoteFinalOffset, SMALL_SHARD_BYTES, "Small mesh AG remote final"));
        CHK_RET(CheckUserRange(PERFORMANCE_SMALL_BYTES, remoteLocalRank * SMALL_SHARD_BYTES, SMALL_SHARD_BYTES,
            "Small mesh AG output shard"));
        CHK_RET(BeginChannelOp(worker, *channel));
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(worker, channel->handle,
            AddOffset(param.outputPtr, remoteLocalRank * SMALL_SHARD_BYTES),
            AddOffset(channel->remoteCclMem.addr, remoteFinalOffset), SMALL_SHARD_BYTES)));
        CHK_RET(EndChannelOp(worker, *channel));
        return SignalThreadComplete(worker, resCtx.threads[MESH_MAIN_THREAD_INDEX], workerIdx);
    }

    HcclResult RunSmallMeshMain(const AlgResourceCtx &resCtx)
    {
        ThreadHandle meshMain = resCtx.threads[MESH_MAIN_THREAD_INDEX];
        CHK_RET(WaitThreadStart(meshMain));
        for (uint32_t workerIdx = 0; workerIdx < MESH_WORKER_NUM; workerIdx++) {
            CHK_RET(StartThread(meshMain, resCtx.threads[MESH_WORKER_THREAD_BEGIN + workerIdx]));
        }
        for (uint32_t workerIdx = 0; workerIdx < MESH_WORKER_NUM; workerIdx++) {
            CHK_RET(WaitThreadComplete(meshMain, workerIdx));
        }
        return SignalThreadComplete(meshMain, resCtx.threads[MAIN_THREAD_INDEX], SMALL_MESH_COMPLETE_NOTIFY);
    }

    HcclResult RunSmallLocalReduce(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        ThreadHandle reduceThread = resCtx.threads[MAIN_THREAD_INDEX];
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        for (uint32_t workerIdx = 0; workerIdx < MESH_WORKER_NUM; workerIdx++) {
            CHK_RET(WaitThreadComplete(reduceThread, workerIdx));
        }

        void *publishAddr = AddOffset(resCtx.localBuffer.addr, SMALL_LOCAL_PUBLISH_OFFSET);
        for (uint32_t sourceLocalRank = 0; sourceLocalRank < RANKS_PER_SERVER; sourceLocalRank++) {
            const void *sourceAddr = nullptr;
            if (sourceLocalRank == localRank) {
                sourceAddr = AddOffset(param.inputPtr, localRank * SMALL_SHARD_BYTES);
            } else {
                uint64_t sourceOffset = ExcludedSlot(sourceLocalRank, localRank) * SMALL_SHARD_BYTES;
                sourceAddr = AddOffset(resCtx.localBuffer.addr, sourceOffset);
            }
            if (sourceLocalRank == 0) {
                CHK_RET(CopyBytes(reduceThread, publishAddr, sourceAddr, SMALL_SHARD_BYTES));
            } else {
                CHK_RET(ReduceRange(param, reduceThread, publishAddr, sourceAddr, SMALL_SHARD_COUNT));
            }
        }
        return SignalThreadComplete(reduceThread, resCtx.threads[INTER_THREAD_INDEX], 0);
    }

    HcclResult RunSmallInter(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        ThreadHandle interThread = resCtx.threads[INTER_THREAD_INDEX];
        CHK_RET(WaitThreadStart(interThread));
        CHK_RET(WaitThreadComplete(interThread, 0));
        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        uint32_t remoteRank = param.myRank ^ RANKS_PER_SERVER;
        const ChannelInfo *channel = FindChannel(resCtx, remoteRank, ChannelType::NHR);
        CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Small inter channel to rank[%u] not found", remoteRank),
            HCCL_E_INTERNAL);
        CHK_RET(CheckBufferRange(channel->remoteCclMem, SMALL_LOCAL_PUBLISH_OFFSET, SMALL_SHARD_BYTES,
            "Small inter remote publish"));
        CHK_RET(BeginChannelOp(interThread, *channel));
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(interThread, channel->handle,
            AddOffset(resCtx.localBuffer.addr, SMALL_INTER_REMOTE_OFFSET),
            AddOffset(channel->remoteCclMem.addr, SMALL_LOCAL_PUBLISH_OFFSET), SMALL_SHARD_BYTES)));
        CHK_RET(EndChannelOp(interThread, *channel));

        void *lowAddr = AddOffset(resCtx.localBuffer.addr, SMALL_LOCAL_PUBLISH_OFFSET);
        void *highAddr = AddOffset(resCtx.localBuffer.addr, SMALL_INTER_REMOTE_OFFSET);
        if (serverIdx == 0) {
            CHK_RET(ReduceRange(param, interThread, lowAddr, highAddr, SMALL_SHARD_COUNT));
        } else {
            CHK_RET(ReduceRange(param, interThread, highAddr, lowAddr, SMALL_SHARD_COUNT));
        }
        for (uint32_t workerIdx = 0; workerIdx < MESH_WORKER_NUM; workerIdx++) {
            CHK_RET(SignalThreadComplete(interThread,
                resCtx.threads[MESH_WORKER_THREAD_BEGIN + workerIdx], 0));
        }
        CHK_RET(CopyBytes(interThread, AddOffset(param.outputPtr, localRank * SMALL_SHARD_BYTES),
            AddOffset(resCtx.localBuffer.addr, SmallFinalCclOffset(serverIdx)), SMALL_SHARD_BYTES));
        return SignalThreadComplete(interThread, resCtx.threads[MAIN_THREAD_INDEX], SMALL_INTER_COMPLETE_NOTIFY);
    }

    [[maybe_unused]] HcclResult DirectSmallAllReduce(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        CHK_RET(ValidateSmallLayout(param, resCtx));
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        CHK_RET(StartThread(coordinator, resCtx.threads[MESH_MAIN_THREAD_INDEX]));
        CHK_RET(StartThread(coordinator, resCtx.threads[INTER_THREAD_INDEX]));
        CHK_RET(RunSmallMeshMain(resCtx));
        for (uint32_t workerIdx = 0; workerIdx < MESH_WORKER_NUM; workerIdx++) {
            CHK_RET(RunSmallMeshWorker(param, resCtx, workerIdx));
        }
        CHK_RET(RunSmallInter(param, resCtx));
        CHK_RET(RunSmallLocalReduce(param, resCtx));
        CHK_RET(WaitThreadComplete(coordinator, SMALL_MESH_COMPLETE_NOTIFY));
        CHK_RET(WaitThreadComplete(coordinator, SMALL_INTER_COMPLETE_NOTIFY));
        return HCCL_SUCCESS;
    }

    HcclResult MeshReduceScatter(const OpParam &param, const AlgResourceCtx &resCtx, const DataRange &range,
        const void *inputBase, void *outputBase, uint64_t cclOffset, bool waitStart, bool signalComplete)
    {
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        ThreadHandle meshMain = resCtx.threads[MESH_MAIN_THREAD_INDEX];
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        uint32_t serverBase = param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
        uint64_t slotBytes = MaxPartCount(range.count, RANKS_PER_SERVER) * sizeof(float);
        uint64_t scratchBytes = MESH_SCRATCH_SLOT_NUM * slotBytes;
        CHK_RET(CheckBufferRange(resCtx.localBuffer, cclOffset, scratchBytes, "Mesh local CCL"));
        if (waitStart) {
            CHK_RET(WaitThreadStart(meshMain));
        }

        uint32_t workerIdx = 0;
        for (uint32_t remoteLocalRank = 0; remoteLocalRank < RANKS_PER_SERVER; remoteLocalRank++) {
            if (remoteLocalRank == localRank) {
                continue;
            }
            uint32_t remoteRank = serverBase + remoteLocalRank;
            const ChannelInfo *channel = FindChannel(resCtx, remoteRank, ChannelType::MESH);
            CHK_PRT_RET(
                channel == nullptr, HCCL_ERROR("Mesh channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);

            ThreadHandle worker = resCtx.threads[MESH_WORKER_THREAD_BEGIN + workerIdx];
            DataRange sendRange = SplitRange(range, RANKS_PER_SERVER, remoteLocalRank);
            uint64_t sendOffset = sendRange.begin * sizeof(float);
            uint64_t remoteSlot = ExcludedSlot(localRank, remoteLocalRank);
            uint64_t remoteOffset = cclOffset + remoteSlot * slotBytes;
            CHK_RET(CheckBufferRange(channel->remoteCclMem, remoteOffset, RangeBytes(sendRange), "Mesh remote CCL"));

            CHK_RET(StartThread(meshMain, worker));
            CHK_RET(WaitThreadStart(worker));
            CHK_RET(RecordChannel(worker, *channel, NOTIFY_IDX_ACK));
            CHK_RET(WaitChannel(worker, *channel, NOTIFY_IDX_ACK));
            CHK_RET(static_cast<HcclResult>(
                HcommWriteOnThread(worker, channel->handle, AddOffset(channel->remoteCclMem.addr, remoteOffset),
                    AddOffset(inputBase, sendOffset), RangeBytes(sendRange))));
            CHK_RET(RecordChannel(worker, *channel, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(WaitChannel(worker, *channel, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(SignalThreadComplete(worker, meshMain, workerIdx + 1));
            workerIdx++;
        }

        for (uint32_t idx = 0; idx < MESH_WORKER_NUM; idx++) {
            CHK_RET(WaitThreadComplete(meshMain, idx + 1));
        }

        DataRange ownedRange = SplitRange(range, RANKS_PER_SERVER, localRank);
        uint64_t ownedOffset = ownedRange.begin * sizeof(float);
        void *dst = AddOffset(outputBase, ownedOffset);
        const void *localSrc = AddOffset(inputBase, ownedOffset);
        if (inputBase == outputBase) {
            // 原地归约前保留本 Rank 数据，避免固定归约顺序覆盖输入。
            void *localScratch = AddOffset(resCtx.localBuffer.addr, cclOffset + MESH_WORKER_NUM * slotBytes);
            CHK_RET(CopyBytes(meshMain, localScratch, localSrc, RangeBytes(ownedRange)));
            localSrc = localScratch;
        }
        for (uint32_t sourceLocalRank = 0; sourceLocalRank < RANKS_PER_SERVER; sourceLocalRank++) {
            const void *src = nullptr;
            if (sourceLocalRank == localRank) {
                src = localSrc;
            } else {
                uint64_t localSlot = ExcludedSlot(sourceLocalRank, localRank);
                src = AddOffset(resCtx.localBuffer.addr, cclOffset + localSlot * slotBytes);
            }
            if (sourceLocalRank == 0) {
                CHK_RET(CopyBytes(meshMain, dst, src, RangeBytes(ownedRange)));
            } else {
                CHK_RET(ReduceRange(param, meshMain, dst, src, ownedRange.count));
            }
        }

        return signalComplete ? SignalThreadComplete(meshMain, coordinator, MESH_COMPLETE_NOTIFY) : HCCL_SUCCESS;
    }

    HcclResult MeshAllGather(const OpParam &param, const AlgResourceCtx &resCtx, const DataRange &range,
        void *outputBase, uint64_t cclOffset, bool waitStart, bool signalComplete)
    {
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        ThreadHandle meshMain = resCtx.threads[MESH_MAIN_THREAD_INDEX];
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        uint32_t serverBase = param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
        uint64_t slotBytes = MaxPartCount(range.count, RANKS_PER_SERVER) * sizeof(float);
        uint64_t scratchBytes = MESH_SCRATCH_SLOT_NUM * slotBytes;
        CHK_RET(CheckBufferRange(resCtx.localBuffer, cclOffset, scratchBytes, "Mesh local CCL"));
        if (waitStart) {
            CHK_RET(WaitThreadStart(meshMain));
        }

        DataRange sendRange = SplitRange(range, RANKS_PER_SERVER, localRank);
        uint64_t sendOffset = sendRange.begin * sizeof(float);
        // 先发布本 Rank 分片，对端随后直接读入最终输出区。
        CHK_RET(CopyBytes(meshMain, AddOffset(resCtx.localBuffer.addr, cclOffset),
            AddOffset(outputBase, sendOffset), RangeBytes(sendRange)));
        uint32_t workerIdx = 0;
        for (uint32_t remoteLocalRank = 0; remoteLocalRank < RANKS_PER_SERVER; remoteLocalRank++) {
            if (remoteLocalRank == localRank) {
                continue;
            }
            uint32_t remoteRank = serverBase + remoteLocalRank;
            const ChannelInfo *channel = FindChannel(resCtx, remoteRank, ChannelType::MESH);
            CHK_PRT_RET(
                channel == nullptr, HCCL_ERROR("Mesh channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);

            ThreadHandle worker = resCtx.threads[MESH_WORKER_THREAD_BEGIN + workerIdx];
            DataRange recvRange = SplitRange(range, RANKS_PER_SERVER, remoteLocalRank);
            CHK_RET(CheckBufferRange(channel->remoteCclMem, cclOffset, RangeBytes(recvRange), "Mesh remote CCL"));

            CHK_RET(StartThread(meshMain, worker));
            CHK_RET(WaitThreadStart(worker));
            CHK_RET(RecordChannel(worker, *channel, NOTIFY_IDX_ACK));
            CHK_RET(WaitChannel(worker, *channel, NOTIFY_IDX_ACK));
            CHK_RET(static_cast<HcclResult>(
                HcommReadOnThread(worker, channel->handle, AddOffset(outputBase, recvRange.begin * sizeof(float)),
                    AddOffset(channel->remoteCclMem.addr, cclOffset), RangeBytes(recvRange))));
            CHK_RET(RecordChannel(worker, *channel, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(WaitChannel(worker, *channel, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(SignalThreadComplete(worker, meshMain, workerIdx + 1));
            workerIdx++;
        }

        for (uint32_t idx = 0; idx < MESH_WORKER_NUM; idx++) {
            CHK_RET(WaitThreadComplete(meshMain, idx + 1));
        }

        return signalComplete ? SignalThreadComplete(meshMain, coordinator, MESH_COMPLETE_NOTIFY) : HCCL_SUCCESS;
    }

    HcclResult InterReduceScatter(const OpParam &param, const AlgResourceCtx &resCtx, const DataRange &range,
        const void *inputBase, void *outputBase, uint64_t cclOffset, bool waitStart, bool signalComplete)
    {
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        ThreadHandle interThread = resCtx.threads[INTER_THREAD_INDEX];
        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t remoteRank = param.myRank ^ RANKS_PER_SERVER;
        const ChannelInfo *channel = FindChannel(resCtx, remoteRank, ChannelType::NHR);
        CHK_PRT_RET(
            channel == nullptr, HCCL_ERROR("Inter-server channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);

        DataRange keepRange = SplitRange(range, 2, serverIdx);
        DataRange sendRange = SplitRange(range, 2, serverIdx ^ 1);
        uint64_t maxBytes = MaxPartCount(range.count, 2) * sizeof(float);
        CHK_RET(CheckBufferRange(resCtx.localBuffer, cclOffset, maxBytes, "Inter local CCL"));
        CHK_RET(CheckBufferRange(channel->remoteCclMem, cclOffset, RangeBytes(sendRange), "Inter remote CCL"));
        if (waitStart) {
            CHK_RET(WaitThreadStart(interThread));
        }

        if (serverIdx == 0) {
            CHK_RET(CopyBytes(interThread, AddOffset(resCtx.localBuffer.addr, cclOffset),
                AddOffset(inputBase, keepRange.begin * sizeof(float)), RangeBytes(keepRange)));
        }

        CHK_RET(RecordChannel(interThread, *channel, NOTIFY_IDX_ACK));
        CHK_RET(WaitChannel(interThread, *channel, NOTIFY_IDX_ACK));
        if (serverIdx == 0) {
            CHK_RET(static_cast<HcclResult>(
                HcommWriteOnThread(interThread, channel->handle, AddOffset(channel->remoteCclMem.addr, cclOffset),
                    AddOffset(inputBase, sendRange.begin * sizeof(float)), RangeBytes(sendRange))));
        } else {
            CHK_RET(static_cast<HcclResult>(
                HcommWriteReduceOnThread(interThread, channel->handle, AddOffset(channel->remoteCclMem.addr, cclOffset),
                    AddOffset(inputBase, sendRange.begin * sizeof(float)), sendRange.count,
                    static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
        }
        CHK_RET(RecordChannel(interThread, *channel, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(WaitChannel(interThread, *channel, NOTIFY_IDX_DATA_SIGNAL));

        if (serverIdx == 1) {
            CHK_RET(ReduceRange(param, interThread, AddOffset(resCtx.localBuffer.addr, cclOffset),
                AddOffset(inputBase, keepRange.begin * sizeof(float)), keepRange.count));
        }
        CHK_RET(CopyBytes(interThread, AddOffset(outputBase, keepRange.begin * sizeof(float)),
            AddOffset(resCtx.localBuffer.addr, cclOffset), RangeBytes(keepRange)));
        return signalComplete ? SignalThreadComplete(interThread, coordinator, INTER_COMPLETE_NOTIFY) : HCCL_SUCCESS;
    }

    HcclResult InterAllGather(const OpParam &param, const AlgResourceCtx &resCtx, const DataRange &range,
        void *outputBase, uint64_t cclOffset, bool waitStart, bool signalComplete)
    {
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        ThreadHandle interThread = resCtx.threads[INTER_THREAD_INDEX];
        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t remoteRank = param.myRank ^ RANKS_PER_SERVER;
        const ChannelInfo *channel = FindChannel(resCtx, remoteRank, ChannelType::NHR);
        CHK_PRT_RET(
            channel == nullptr, HCCL_ERROR("Inter-server channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);

        DataRange sendRange = SplitRange(range, 2, serverIdx);
        DataRange recvRange = SplitRange(range, 2, serverIdx ^ 1);
        uint64_t maxBytes = MaxPartCount(range.count, 2) * sizeof(float);
        CHK_RET(CheckBufferRange(resCtx.localBuffer, cclOffset, maxBytes, "Inter local CCL"));
        CHK_RET(CheckBufferRange(channel->remoteCclMem, cclOffset, RangeBytes(sendRange), "Inter remote CCL"));
        if (waitStart) {
            CHK_RET(WaitThreadStart(interThread));
        }

        CHK_RET(RecordChannel(interThread, *channel, NOTIFY_IDX_ACK));
        CHK_RET(WaitChannel(interThread, *channel, NOTIFY_IDX_ACK));
        CHK_RET(static_cast<HcclResult>(
            HcommWriteOnThread(interThread, channel->handle, AddOffset(channel->remoteCclMem.addr, cclOffset),
                AddOffset(outputBase, sendRange.begin * sizeof(float)), RangeBytes(sendRange))));
        CHK_RET(RecordChannel(interThread, *channel, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(WaitChannel(interThread, *channel, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(CopyBytes(interThread, AddOffset(outputBase, recvRange.begin * sizeof(float)),
            AddOffset(resCtx.localBuffer.addr, cclOffset), RangeBytes(recvRange)));
        return signalComplete ? SignalThreadComplete(interThread, coordinator, INTER_COMPLETE_NOTIFY) : HCCL_SUCCESS;
    }

    HcclResult ValidateLargeLayout(
        const OpParam &param, const AlgResourceCtx &resCtx, const LargeConfig &config)
    {
        uint64_t totalBytes = param.count * sizeof(float);
        CHK_RET(CheckBufferRange(
            resCtx.localBuffer, 0, LARGE_TILE_NUM * config.halfSlotStride, "Large two-half-slot CCL"));
        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        for (uint32_t tileIdx = 0; tileIdx < LARGE_TILE_NUM; tileIdx++) {
            LargeTileLayout layout = GetLargeTileLayout(config, tileIdx, serverIdx, localRank);
            CHK_PRT_RET(layout.slotBase % CCL_ADDRESS_ALIGNMENT != 0
                            || layout.halfBytes > config.halfSlotStride
                            || layout.ownedOffset > layout.halfBytes
                            || layout.ownedBytes > layout.halfBytes - layout.ownedOffset,
                HCCL_ERROR("Invalid large layout, tile[%u], slot[%llu], half[%llu], ownedOffset[%llu], owned[%llu]",
                    tileIdx, static_cast<unsigned long long>(layout.slotBase),
                    static_cast<unsigned long long>(layout.halfBytes),
                    static_cast<unsigned long long>(layout.ownedOffset),
                    static_cast<unsigned long long>(layout.ownedBytes)),
                HCCL_E_INTERNAL);
            CHK_RET(CheckBufferRange(
                resCtx.localBuffer, layout.slotBase, layout.halfBytes, "Large local CCL half"));
            CHK_RET(CheckBufferRange(resCtx.localBuffer, layout.slotBase + layout.ownedOffset,
                layout.ownedBytes, "Large local CCL owned shard"));
            CHK_RET(CheckUserRange(
                totalBytes, layout.ownedRange.begin * sizeof(float), layout.ownedBytes, "Large output owned shard"));
            CHK_RET(CheckUserRange(totalBytes, layout.otherServerRange.begin * sizeof(float),
                RangeBytes(layout.otherServerRange), "Large output scratch half"));

            uint64_t scratchHalfBytes = RangeBytes(layout.otherServerRange);
            for (uint32_t sourceLocalRank = 1; sourceLocalRank < RANKS_PER_SERVER; sourceLocalRank++) {
                uint64_t scratchRelative = (sourceLocalRank - 1) * layout.scratchStride;
                CHK_PRT_RET(scratchRelative > scratchHalfBytes
                                || layout.ownedBytes > scratchHalfBytes - scratchRelative,
                    HCCL_ERROR("Large scratch exceeds opposite half, tile[%u], server[%u], localRank[%u], "
                               "source[%u], offset[%llu], bytes[%llu], half[%llu]",
                        tileIdx, serverIdx, localRank, sourceLocalRank,
                        static_cast<unsigned long long>(scratchRelative),
                        static_cast<unsigned long long>(layout.ownedBytes),
                        static_cast<unsigned long long>(scratchHalfBytes)),
                    HCCL_E_INTERNAL);
                CHK_RET(CheckUserRange(totalBytes, LargeScratchOffset(layout, sourceLocalRank),
                    layout.ownedBytes, "Large output source scratch"));
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult LargeInterReduceScatter(const OpParam &param, const AlgResourceCtx &resCtx,
        const LargeConfig &config, uint32_t tileIdx)
    {
        ThreadHandle interThread = resCtx.threads[INTER_THREAD_INDEX];
        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t remoteRank = param.myRank ^ RANKS_PER_SERVER;
        const ChannelInfo *channel = FindChannel(resCtx, remoteRank, ChannelType::NHR);
        CHK_PRT_RET(
            channel == nullptr, HCCL_ERROR("Large inter channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);

        LargeTileLayout layout = GetLargeTileLayout(config, tileIdx, serverIdx, param.myRank % RANKS_PER_SERVER);
        DataRange keepRange = layout.serverRange;
        DataRange sendRange = layout.otherServerRange;
        CHK_RET(CheckBufferRange(
            resCtx.localBuffer, layout.slotBase, RangeBytes(keepRange), "Large inter RS local"));
        CHK_RET(CheckBufferRange(
            channel->remoteCclMem, layout.slotBase, RangeBytes(sendRange), "Large inter RS remote"));

        // The low-server contribution is always the first FP32 operand.
        if (serverIdx == 0) {
            CHK_RET(CopyBytes(interThread, AddOffset(resCtx.localBuffer.addr, layout.slotBase),
                AddOffset(param.inputPtr, keepRange.begin * sizeof(float)), RangeBytes(keepRange)));
        }
        CHK_RET(BeginChannelOp(interThread, *channel));
        if (serverIdx == 0) {
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(interThread, channel->handle,
                AddOffset(channel->remoteCclMem.addr, layout.slotBase),
                AddOffset(param.inputPtr, sendRange.begin * sizeof(float)), RangeBytes(sendRange))));
        } else {
            CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(interThread, channel->handle,
                AddOffset(channel->remoteCclMem.addr, layout.slotBase),
                AddOffset(param.inputPtr, sendRange.begin * sizeof(float)), sendRange.count,
                static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
        }
        CHK_RET(EndChannelOp(interThread, *channel));
        if (serverIdx == 1) {
            CHK_RET(ReduceRange(param, interThread, AddOffset(resCtx.localBuffer.addr, layout.slotBase),
                AddOffset(param.inputPtr, keepRange.begin * sizeof(float)), keepRange.count));
        }

        for (uint32_t workerIdx = 0; workerIdx < MESH_WORKER_NUM; workerIdx++) {
            CHK_RET(SignalThreadComplete(interThread, resCtx.threads[MESH_WORKER_THREAD_BEGIN + workerIdx],
                WorkerInterReadyNotify(tileIdx)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult LargeInterAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
        const LargeConfig &config, uint32_t tileIdx)
    {
        ThreadHandle interThread = resCtx.threads[INTER_THREAD_INDEX];
        CHK_RET(WaitThreadComplete(interThread, tileIdx));

        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t remoteRank = param.myRank ^ RANKS_PER_SERVER;
        const ChannelInfo *channel = FindChannel(resCtx, remoteRank, ChannelType::NHR);
        CHK_PRT_RET(
            channel == nullptr, HCCL_ERROR("Large inter channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
        LargeTileLayout layout = GetLargeTileLayout(config, tileIdx, serverIdx, param.myRank % RANKS_PER_SERVER);
        DataRange recvRange = layout.otherServerRange;
        CHK_RET(CheckBufferRange(
            channel->remoteCclMem, layout.slotBase, RangeBytes(recvRange), "Large inter AG remote"));
        CHK_RET(CheckUserRange(param.count * sizeof(float), recvRange.begin * sizeof(float),
            RangeBytes(recvRange), "Large inter AG output"));

        CHK_RET(BeginChannelOp(interThread, *channel));
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(interThread, channel->handle,
            AddOffset(param.outputPtr, recvRange.begin * sizeof(float)),
            AddOffset(channel->remoteCclMem.addr, layout.slotBase), RangeBytes(recvRange))));
        CHK_RET(EndChannelOp(interThread, *channel));
        return SignalThreadComplete(interThread, resCtx.threads[MAIN_THREAD_INDEX], MainInterAgDoneNotify(tileIdx));
    }

    HcclResult RunLargeInter(
        const OpParam &param, const AlgResourceCtx &resCtx, const LargeConfig &config)
    {
        ThreadHandle interThread = resCtx.threads[INTER_THREAD_INDEX];
        CHK_RET(WaitThreadStart(interThread));
        for (uint32_t tileIdx = 0; tileIdx < LARGE_TILE_NUM; tileIdx++) {
            CHK_RET(LargeInterReduceScatter(param, resCtx, config, tileIdx));
        }
        for (uint32_t tileIdx = 0; tileIdx < LARGE_TILE_NUM; tileIdx++) {
            CHK_RET(LargeInterAllGather(param, resCtx, config, tileIdx));
        }
        return HCCL_SUCCESS;
    }

    HcclResult LargeMeshReduceScatter(const OpParam &param, const AlgResourceCtx &resCtx,
        const LargeConfig &config, uint32_t workerIdx, uint32_t tileIdx)
    {
        ThreadHandle worker = resCtx.threads[MESH_WORKER_THREAD_BEGIN + workerIdx];
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t remoteLocalRank = workerIdx < localRank ? workerIdx : workerIdx + 1;
        uint32_t remoteRank = serverIdx * RANKS_PER_SERVER + remoteLocalRank;
        const ChannelInfo *channel = FindChannel(resCtx, remoteRank, ChannelType::MESH);
        CHK_PRT_RET(
            channel == nullptr, HCCL_ERROR("Large mesh channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);

        CHK_RET(WaitThreadComplete(worker, WorkerInterReadyNotify(tileIdx)));
        LargeTileLayout layout = GetLargeTileLayout(config, tileIdx, serverIdx, localRank);
        uint64_t remoteOffset = layout.slotBase + layout.ownedOffset;
        CHK_RET(CheckBufferRange(
            channel->remoteCclMem, remoteOffset, layout.ownedBytes, "Large mesh RS remote owned shard"));

        uint64_t outputOffset = 0;
        if (remoteLocalRank == 0) {
            outputOffset = layout.ownedRange.begin * sizeof(float);
        } else {
            outputOffset = LargeScratchOffset(layout, remoteLocalRank);
        }
        CHK_RET(CheckUserRange(
            param.count * sizeof(float), outputOffset, layout.ownedBytes, "Large mesh RS output scratch"));
        CHK_RET(BeginChannelOp(worker, *channel));
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(worker, channel->handle,
            AddOffset(param.outputPtr, outputOffset), AddOffset(channel->remoteCclMem.addr, remoteOffset),
            layout.ownedBytes)));
        CHK_RET(EndChannelOp(worker, *channel));
        return SignalThreadComplete(
            worker, resCtx.threads[MAIN_THREAD_INDEX], TileWorkerNotify(tileIdx, workerIdx));
    }

    HcclResult LargeMeshAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
        const LargeConfig &config, uint32_t workerIdx, uint32_t tileIdx)
    {
        ThreadHandle worker = resCtx.threads[MESH_WORKER_THREAD_BEGIN + workerIdx];
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t remoteLocalRank = workerIdx < localRank ? workerIdx : workerIdx + 1;
        uint32_t remoteRank = serverIdx * RANKS_PER_SERVER + remoteLocalRank;
        const ChannelInfo *channel = FindChannel(resCtx, remoteRank, ChannelType::MESH);
        CHK_PRT_RET(
            channel == nullptr, HCCL_ERROR("Large mesh channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);

        CHK_RET(WaitThreadComplete(worker, WorkerPublishReadyNotify(tileIdx)));
        LargeTileLayout layout = GetLargeTileLayout(config, tileIdx, serverIdx, localRank);
        uint64_t remoteOffset = layout.slotBase + layout.ownedOffset;
        uint64_t outputOffset = layout.ownedRange.begin * sizeof(float);
        CHK_RET(CheckBufferRange(
            channel->remoteCclMem, remoteOffset, layout.ownedBytes, "Large mesh AG remote shard"));
        CHK_RET(CheckUserRange(
            param.count * sizeof(float), outputOffset, layout.ownedBytes, "Large mesh AG output owned"));

        CHK_RET(BeginChannelOp(worker, *channel));
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(worker, channel->handle,
            AddOffset(channel->remoteCclMem.addr, remoteOffset), AddOffset(param.outputPtr, outputOffset),
            layout.ownedBytes)));
        CHK_RET(EndChannelOp(worker, *channel));
        return SignalThreadComplete(
            worker, resCtx.threads[MESH_MAIN_THREAD_INDEX], TileWorkerNotify(tileIdx, workerIdx));
    }

    HcclResult RunLargeMeshWorker(const OpParam &param, const AlgResourceCtx &resCtx,
        const LargeConfig &config, uint32_t workerIdx)
    {
        ThreadHandle worker = resCtx.threads[MESH_WORKER_THREAD_BEGIN + workerIdx];
        CHK_RET(WaitThreadStart(worker));
        for (uint32_t tileIdx = 0; tileIdx < LARGE_TILE_NUM; tileIdx++) {
            CHK_RET(LargeMeshReduceScatter(param, resCtx, config, workerIdx, tileIdx));
        }
        for (uint32_t tileIdx = 0; tileIdx < LARGE_TILE_NUM; tileIdx++) {
            CHK_RET(LargeMeshAllGather(param, resCtx, config, workerIdx, tileIdx));
        }
        return HCCL_SUCCESS;
    }

    HcclResult RunLargeLocalReduce(
        const OpParam &param, const AlgResourceCtx &resCtx, const LargeConfig &config)
    {
        ThreadHandle reduceThread = resCtx.threads[MAIN_THREAD_INDEX];
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        for (uint32_t tileIdx = 0; tileIdx < LARGE_TILE_NUM; tileIdx++) {
            for (uint32_t workerIdx = 0; workerIdx < MESH_WORKER_NUM; workerIdx++) {
                CHK_RET(WaitThreadComplete(reduceThread, TileWorkerNotify(tileIdx, workerIdx)));
            }

            LargeTileLayout layout = GetLargeTileLayout(config, tileIdx, serverIdx, localRank);
            uint64_t outputOffset = layout.ownedRange.begin * sizeof(float);
            void *outputAddr = AddOffset(param.outputPtr, outputOffset);
            const void *localSource =
                AddOffset(resCtx.localBuffer.addr, layout.slotBase + layout.ownedOffset);
            if (localRank == 0) {
                CHK_RET(CopyBytes(reduceThread, outputAddr, localSource, layout.ownedBytes));
            }
            for (uint32_t sourceLocalRank = 1; sourceLocalRank < RANKS_PER_SERVER; sourceLocalRank++) {
                const void *sourceAddr = sourceLocalRank == localRank
                                             ? localSource
                                             : AddOffset(param.outputPtr, LargeScratchOffset(layout, sourceLocalRank));
                CHK_RET(ReduceRange(param, reduceThread, outputAddr, sourceAddr, layout.ownedRange.count));
            }

            CHK_RET(CopyBytes(reduceThread,
                AddOffset(resCtx.localBuffer.addr, layout.slotBase + layout.ownedOffset), outputAddr,
                layout.ownedBytes));
            for (uint32_t workerIdx = 0; workerIdx < MESH_WORKER_NUM; workerIdx++) {
                CHK_RET(SignalThreadComplete(reduceThread,
                    resCtx.threads[MESH_WORKER_THREAD_BEGIN + workerIdx], WorkerPublishReadyNotify(tileIdx)));
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult LargeLocalCopy(const OpParam &param, const AlgResourceCtx &resCtx,
        const LargeConfig &config, uint32_t tileIdx)
    {
        ThreadHandle meshMain = resCtx.threads[MESH_MAIN_THREAD_INDEX];
        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        LargeTileLayout layout = GetLargeTileLayout(config, tileIdx, serverIdx, localRank);
        uint64_t serverOutputOffset = layout.serverRange.begin * sizeof(float);
        uint64_t prefixBytes = layout.ownedOffset;
        uint64_t suffixOffset = layout.ownedOffset + layout.ownedBytes;
        uint64_t suffixBytes = layout.halfBytes - suffixOffset;
        CHK_RET(CopyBytes(meshMain, AddOffset(param.outputPtr, serverOutputOffset),
            AddOffset(resCtx.localBuffer.addr, layout.slotBase), prefixBytes));
        CHK_RET(CopyBytes(meshMain, AddOffset(param.outputPtr, serverOutputOffset + suffixOffset),
            AddOffset(resCtx.localBuffer.addr, layout.slotBase + suffixOffset), suffixBytes));
        return HCCL_SUCCESS;
    }

    HcclResult RunLargeMeshMain(
        const OpParam &param, const AlgResourceCtx &resCtx, const LargeConfig &config)
    {
        ThreadHandle meshMain = resCtx.threads[MESH_MAIN_THREAD_INDEX];
        CHK_RET(WaitThreadStart(meshMain));
        for (uint32_t workerIdx = 0; workerIdx < MESH_WORKER_NUM; workerIdx++) {
            CHK_RET(StartThread(meshMain, resCtx.threads[MESH_WORKER_THREAD_BEGIN + workerIdx]));
        }
        for (uint32_t tileIdx = 0; tileIdx < LARGE_TILE_NUM; tileIdx++) {
            for (uint32_t workerIdx = 0; workerIdx < MESH_WORKER_NUM; workerIdx++) {
                CHK_RET(WaitThreadComplete(meshMain, TileWorkerNotify(tileIdx, workerIdx)));
            }
            CHK_RET(SignalThreadComplete(meshMain, resCtx.threads[INTER_THREAD_INDEX], tileIdx));
            CHK_RET(LargeLocalCopy(param, resCtx, config, tileIdx));
            CHK_RET(SignalThreadComplete(
                meshMain, resCtx.threads[MAIN_THREAD_INDEX], MainLocalCopyDoneNotify(tileIdx)));
        }
        return HCCL_SUCCESS;
    }

    [[maybe_unused]] HcclResult LargeAllReduce(
        const OpParam &param, const AlgResourceCtx &resCtx, const LargeConfig &config)
    {
        CHK_PRT_RET(param.inputPtr == param.outputPtr,
            HCCL_ERROR("Large two-half-slot path requires out-of-place buffers"), HCCL_E_INTERNAL);
        CHK_RET(ValidateLargeLayout(param, resCtx, config));
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        CHK_RET(StartThread(coordinator, resCtx.threads[MESH_MAIN_THREAD_INDEX]));
        CHK_RET(StartThread(coordinator, resCtx.threads[INTER_THREAD_INDEX]));
        CHK_RET(RunLargeMeshMain(param, resCtx, config));
        CHK_RET(RunLargeInter(param, resCtx, config));
        for (uint32_t workerIdx = 0; workerIdx < MESH_WORKER_NUM; workerIdx++) {
            CHK_RET(RunLargeMeshWorker(param, resCtx, config, workerIdx));
        }
        CHK_RET(RunLargeLocalReduce(param, resCtx, config));
        for (uint32_t tileIdx = 0; tileIdx < LARGE_TILE_NUM; tileIdx++) {
            CHK_RET(WaitThreadComplete(coordinator, MainLocalCopyDoneNotify(tileIdx)));
            CHK_RET(WaitThreadComplete(coordinator, MainInterAgDoneNotify(tileIdx)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult StartParallelStage(const AlgResourceCtx &resCtx)
    {
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        CHK_RET(StartThread(coordinator, resCtx.threads[MESH_MAIN_THREAD_INDEX]));
        CHK_RET(StartThread(coordinator, resCtx.threads[INTER_THREAD_INDEX]));
        return HCCL_SUCCESS;
    }

    HcclResult FinishParallelStage(const AlgResourceCtx &resCtx)
    {
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        CHK_RET(WaitThreadComplete(coordinator, MESH_COMPLETE_NOTIFY));
        CHK_RET(WaitThreadComplete(coordinator, INTER_COMPLETE_NOTIFY));
        return HCCL_SUCCESS;
    }

    HcclResult ParallelRsagAllReduce(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        DataRange totalRange{0, param.count};
        DataRange plane0 = SplitRange(totalRange, 2, 0);
        DataRange plane1 = SplitRange(totalRange, 2, 1);
        uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;

        uint64_t meshReserve = MeshScratchBytes(plane0.count);
        CHK_RET(CheckBufferRange(
            resCtx.localBuffer, 0, meshReserve + MaxPartCount(plane1.count, 2) * sizeof(float), "Stage 1 CCL"));
        CHK_RET(StartParallelStage(resCtx));
        CHK_RET(MeshReduceScatter(param, resCtx, plane0, param.inputPtr, param.outputPtr, 0, true, true));
        CHK_RET(InterReduceScatter(param, resCtx, plane1, param.inputPtr, param.outputPtr, meshReserve, true, true));
        CHK_RET(FinishParallelStage(resCtx));

        DataRange plane0Local = SplitRange(plane0, RANKS_PER_SERVER, localRank);
        DataRange plane1Server = SplitRange(plane1, 2, serverIdx);
        uint64_t maxPlane1ServerCount = MaxPartCount(plane1.count, 2);
        meshReserve = MeshScratchBytes(maxPlane1ServerCount);
        CHK_RET(CheckBufferRange(resCtx.localBuffer, 0,
            meshReserve + MaxPartCount(MaxPartCount(plane0.count, RANKS_PER_SERVER), 2) * sizeof(float),
            "Stage 2 CCL"));
        CHK_RET(StartParallelStage(resCtx));
        // 中间两阶段按平面连续执行，避免全局等待。
        CHK_RET(MeshReduceScatter(param, resCtx, plane1Server, param.outputPtr, param.outputPtr, 0, true, false));
        CHK_RET(InterReduceScatter(
            param, resCtx, plane0Local, param.outputPtr, param.outputPtr, meshReserve, true, false));
        CHK_RET(MeshAllGather(param, resCtx, plane1Server, param.outputPtr, 0, false, true));
        CHK_RET(InterAllGather(param, resCtx, plane0Local, param.outputPtr, meshReserve, false, true));
        CHK_RET(FinishParallelStage(resCtx));

        meshReserve = MeshScratchBytes(plane0.count);
        CHK_RET(CheckBufferRange(
            resCtx.localBuffer, 0, meshReserve + MaxPartCount(plane1.count, 2) * sizeof(float), "Stage 4 CCL"));
        CHK_RET(StartParallelStage(resCtx));
        CHK_RET(MeshAllGather(param, resCtx, plane0, param.outputPtr, 0, true, true));
        CHK_RET(InterAllGather(param, resCtx, plane1, param.outputPtr, meshReserve, true, true));
        CHK_RET(FinishParallelStage(resCtx));
        return HCCL_SUCCESS;
    }

    struct DirectConfig {
        uint32_t tileNum = 0;
        uint64_t slotStride = 0;
    };

    constexpr uint32_t DIRECT_RS0_DATA_NOTIFY = 0;
    constexpr uint32_t DIRECT_SLOT_RELEASE_NOTIFY = 1;
    constexpr uint32_t DIRECT_RS1_DATA_NOTIFY = 2;
    constexpr uint32_t DIRECT_AG0_READY_NOTIFY = 3;
    constexpr uint32_t DIRECT_AG0_DONE_NOTIFY = 4;
    constexpr uint32_t DIRECT_AG1_READY_NOTIFY = 5;
    constexpr uint32_t DIRECT_AG1_DONE_NOTIFY = 6;

    HcclResult GetDirectConfig(const OpParam &param, const AlgResourceCtx &resCtx, DirectConfig &config)
    {
        uint64_t usableCclBytes = std::min(resCtx.localBuffer.size, COMPETITION_CCL_BYTES);
        for (uint32_t tileNum = 1; tileNum <= DIRECT_MAX_TILE_NUM; tileNum++) {
            uint64_t maxOwnedCount = 0;
            DataRange totalRange{0, param.count};
            for (uint32_t tileIdx = 0; tileIdx < tileNum; tileIdx++) {
                DataRange tileRange = SplitRange(totalRange, tileNum, tileIdx);
                for (uint32_t ownerRank = 0; ownerRank < EXPECTED_RANK_SIZE; ownerRank++) {
                    maxOwnedCount = std::max(
                        maxOwnedCount, SplitRange(tileRange, EXPECTED_RANK_SIZE, ownerRank).count);
                }
            }
            uint64_t slotStride = AlignUp(maxOwnedCount * sizeof(float), CCL_ADDRESS_ALIGNMENT);
            if (slotStride <= usableCclBytes / DIRECT_PEER_NUM) {
                config.tileNum = tileNum;
                config.slotStride = slotStride;
                return HCCL_SUCCESS;
            }
        }
        HCCL_ERROR("Direct allreduce cannot fit CCL buffer, count[%llu], capacity[%llu]",
            static_cast<unsigned long long>(param.count),
            static_cast<unsigned long long>(resCtx.localBuffer.size));
        return HCCL_E_NOT_SUPPORT;
    }

    DataRange DirectTileRange(const OpParam &param, const DirectConfig &config, uint32_t tileIdx)
    {
        return SplitRange(DataRange{0, param.count}, config.tileNum, tileIdx);
    }

    DataRange DirectOwnedRange(
        const OpParam &param, const DirectConfig &config, uint32_t tileIdx, uint32_t ownerRank)
    {
        return SplitRange(DirectTileRange(param, config, tileIdx), EXPECTED_RANK_SIZE, ownerRank);
    }

    uint32_t DirectPeerRank(uint32_t myRank, uint32_t workerIdx)
    {
        return workerIdx < myRank ? workerIdx : workerIdx + 1;
    }

    uint32_t DirectWorkerIndex(uint32_t myRank, uint32_t peerRank)
    {
        return peerRank < myRank ? peerRank : peerRank - 1;
    }

    uint64_t DirectSlotOffset(uint32_t sourceRank, uint32_t ownerRank, uint64_t slotStride)
    {
        return ExcludedSlot(sourceRank, ownerRank) * slotStride;
    }

    uint32_t DirectPublishSource(uint32_t ownerRank)
    {
        return ownerRank == 0 ? 1 : 0;
    }

    uint32_t DirectMainPeerNotify(uint32_t workerIdx)
    {
        return 1 + workerIdx;
    }

    uint32_t DirectMainAuxNotify(uint32_t workerIdx)
    {
        return DIRECT_MAIN_AUX_NOTIFY_BEGIN + workerIdx;
    }

    uint32_t DirectTreeReadyNotify(uint32_t level, bool targetIsMain)
    {
        return (targetIsMain ? DIRECT_MAIN_AUX_NOTIFY_BEGIN : DIRECT_WORKER_TREE_NOTIFY_BEGIN) + level;
    }

    ChannelType DirectChannelType(uint32_t myRank, uint32_t remoteRank)
    {
        return myRank / RANKS_PER_SERVER == remoteRank / RANKS_PER_SERVER
                   ? ChannelType::MESH
                   : ChannelType::NHR;
    }

    DataRange DirectNestedOwnedRange(const OpParam &param, uint32_t ownerRank)
    {
        return SplitRange(DataRange{0, param.count}, EXPECTED_RANK_SIZE, ownerRank);
    }

    DataRange DirectNestedSegmentRange(const OpParam &param, uint32_t ownerRank, uint32_t segmentIdx)
    {
        DataRange ownedRange = DirectNestedOwnedRange(param, ownerRank);
        uint64_t segment0Count = DIRECT_NESTED_SEGMENT0_BYTES / sizeof(float);
        if (segmentIdx == 0) {
            return DataRange{ownedRange.begin, segment0Count};
        }
        return DataRange{ownedRange.begin + segment0Count,
            DIRECT_NESTED_SEGMENT1_BYTES / sizeof(float)};
    }

    uint32_t DirectNestedPhysicalHighSource(uint32_t ownerRank, uint32_t compactSlotIdx)
    {
        uint32_t requiredHighIdx = compactSlotIdx / 3;
        uint32_t currentHighIdx = 0;
        for (uint32_t highSource = 1; highSource < EXPECTED_RANK_SIZE; highSource += 2) {
            if (highSource == ownerRank) {
                continue;
            }
            if (currentHighIdx == requiredHighIdx) {
                return highSource;
            }
            currentHighIdx++;
        }
        return INVALID_VALUE_RANKID;
    }

    uint64_t DirectNestedSegmentOffset(uint32_t ownerRank, uint32_t sourceRank, uint32_t segmentIdx)
    {
        if (sourceRank == 0) {
            return DIRECT_NESTED_PUBLISH_OFFSET
                   + (segmentIdx == 0 ? 0 : DIRECT_NESTED_SEGMENT0_BYTES);
        }
        if (segmentIdx == 0) {
            return DirectSlotOffset(sourceRank, ownerRank, DIRECT_NESTED_SEGMENT0_BYTES);
        }
        uint32_t compactSlotIdx = ExcludedSlot(sourceRank, ownerRank);
        uint32_t physicalHighSource = DirectNestedPhysicalHighSource(ownerRank, compactSlotIdx);
        return DirectSlotOffset(physicalHighSource, ownerRank, DIRECT_NESTED_SEGMENT0_BYTES)
               + compactSlotIdx % 3 * DIRECT_NESTED_SEGMENT1_BYTES;
    }

    bool DirectNestedUsesOwnerLeafCarrier(const OpParam &param)
    {
        return param.myRank != 0 && (param.myRank & 1U) == 0;
    }

    uint32_t DirectNestedOwnerLeafCarrier(const OpParam &param)
    {
        return DirectNestedUsesOwnerLeafCarrier(param) ? param.myRank + 1 : INVALID_VALUE_RANKID;
    }

    ThreadHandle DirectNestedProducerThread(
        const OpParam &param, const AlgResourceCtx &resCtx, uint32_t sourceRank)
    {
        if (sourceRank == param.myRank) {
            return resCtx.threads[MAIN_THREAD_INDEX];
        }
        return resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + DirectWorkerIndex(param.myRank, sourceRank)];
    }

    void *DirectNestedTreeAddress(const OpParam &param, const AlgResourceCtx &resCtx,
        uint32_t segmentIdx, uint32_t sourceRank)
    {
        if (sourceRank == param.myRank && DirectNestedUsesOwnerLeafCarrier(param)) {
            return AddOffset(resCtx.localBuffer.addr,
                DirectNestedSegmentOffset(param.myRank, DirectNestedOwnerLeafCarrier(param), segmentIdx));
        }
        if (sourceRank == 0) {
            return AddOffset(resCtx.localBuffer.addr,
                DirectNestedSegmentOffset(param.myRank, sourceRank, segmentIdx));
        }
        if (sourceRank == param.myRank) {
            DataRange segmentRange = DirectNestedSegmentRange(param, param.myRank, segmentIdx);
            return AddOffset(param.outputPtr, segmentRange.begin * sizeof(float));
        }
        return AddOffset(resCtx.localBuffer.addr,
            DirectNestedSegmentOffset(param.myRank, sourceRank, segmentIdx));
    }

    HcclResult ValidateDirectNestedLayout(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        uint64_t totalBytes = param.count * sizeof(float);
        CHK_PRT_RET(totalBytes != PERFORMANCE_LARGE_BYTES,
            HCCL_ERROR("Invalid nested Direct bytes[%llu]", static_cast<unsigned long long>(totalBytes)),
            HCCL_E_INTERNAL);
        CHK_RET(CheckBufferRange(
            resCtx.localBuffer, 0, DIRECT_NESTED_CCL_BYTES, "Nested Direct 392MiB local CCL"));
        CHK_RET(CheckBufferRange(resCtx.localBuffer, DIRECT_NESTED_PUBLISH_OFFSET,
            DIRECT_NESTED_SHARD_BYTES, "Nested Direct publish"));

        for (uint32_t ownerRank = 0; ownerRank < EXPECTED_RANK_SIZE; ownerRank++) {
            DataRange ownedRange = DirectNestedOwnedRange(param, ownerRank);
            CHK_PRT_RET(RangeBytes(ownedRange) != DIRECT_NESTED_SHARD_BYTES,
                HCCL_ERROR("Invalid nested Direct owner[%u] bytes[%llu]", ownerRank,
                    static_cast<unsigned long long>(RangeBytes(ownedRange))), HCCL_E_INTERNAL);
            CHK_RET(CheckUserRange(totalBytes, ownedRange.begin * sizeof(float),
                RangeBytes(ownedRange), "Nested Direct owned range"));
        }

        for (uint32_t sourceRank = 0; sourceRank < EXPECTED_RANK_SIZE; sourceRank++) {
            if (sourceRank == param.myRank) {
                continue;
            }
            for (uint32_t segmentIdx = 0; segmentIdx < 2; segmentIdx++) {
                uint64_t bytes = segmentIdx == 0
                                     ? DIRECT_NESTED_SEGMENT0_BYTES
                                     : DIRECT_NESTED_SEGMENT1_BYTES;
                uint64_t offset = DirectNestedSegmentOffset(param.myRank, sourceRank, segmentIdx);
                CHK_RET(CheckBufferRange(
                    resCtx.localBuffer, offset, bytes, "Nested Direct local source segment"));
                if (segmentIdx == 1 && sourceRank != 0) {
                    uint32_t compactSlotIdx = ExcludedSlot(sourceRank, param.myRank);
                    uint32_t physicalHighSource =
                        DirectNestedPhysicalHighSource(param.myRank, compactSlotIdx);
                    CHK_PRT_RET(physicalHighSource >= EXPECTED_RANK_SIZE
                                    || physicalHighSource == param.myRank
                                    || (physicalHighSource & 1U) == 0,
                        HCCL_ERROR("Invalid nested Direct compact mapping owner[%u], source[%u], high[%u]",
                            param.myRank, sourceRank, physicalHighSource), HCCL_E_INTERNAL);
                    uint64_t physicalBegin = DirectSlotOffset(
                        physicalHighSource, param.myRank, DIRECT_NESTED_SEGMENT0_BYTES);
                    CHK_PRT_RET(offset < physicalBegin
                                    || offset + bytes > physicalBegin + DIRECT_NESTED_SEGMENT0_BYTES,
                        HCCL_ERROR("Nested Direct compact slot escapes released region"), HCCL_E_INTERNAL);
                }
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult StartDirectOverlapThreads(const AlgResourceCtx &resCtx)
    {
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        for (uint32_t threadIdx = 1; threadIdx < EXPECTED_PARALLEL_THREAD_NUM; threadIdx++) {
            CHK_RET(StartThread(coordinator, resCtx.threads[threadIdx]));
            CHK_RET(WaitThreadStart(resCtx.threads[threadIdx]));
        }
        return HCCL_SUCCESS;
    }

    HcclResult QueueDirectOverlapSegmentWrite(
        const OpParam &param, const AlgResourceCtx &resCtx, uint32_t ownerRank, uint32_t segmentIdx)
    {
        ThreadHandle worker = resCtx.threads[ownerRank];
        const ChannelInfo *channel = FindChannel(
            resCtx, ownerRank, DirectChannelType(param.myRank, ownerRank));
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Overlap Direct RS channel to rank[%u] not found", ownerRank), HCCL_E_INTERNAL);
        if (segmentIdx == 1 && param.myRank != 0) {
            CHK_RET(WaitChannel(worker, *channel, DIRECT_SLOT_RELEASE_NOTIFY));
        }

        DataRange sendRange = DirectNestedSegmentRange(param, ownerRank, segmentIdx);
        uint64_t sendOffset = sendRange.begin * sizeof(float);
        uint64_t remoteOffset = DirectNestedSegmentOffset(ownerRank, param.myRank, segmentIdx);
        uint32_t dataNotify = segmentIdx == 0 ? DIRECT_RS0_DATA_NOTIFY : DIRECT_RS1_DATA_NOTIFY;
        CHK_RET(CheckUserRange(param.count * sizeof(float), sendOffset,
            RangeBytes(sendRange), "Overlap Direct RS input"));
        CHK_RET(CheckBufferRange(channel->remoteCclMem, remoteOffset,
            RangeBytes(sendRange), "Overlap Direct RS remote CCL"));
        return BatchWriteSignal(worker, *channel,
            AddOffset(channel->remoteCclMem.addr, remoteOffset),
            AddOffset(param.inputPtr, sendOffset), RangeBytes(sendRange), dataNotify);
    }

    HcclResult QueueDirectOverlapPairLayer(const OpParam &param, const AlgResourceCtx &resCtx,
        const std::array<const void *, EXPECTED_RANK_SIZE> &sources, void *ownedOutput,
        uint64_t count, uint32_t dataNotify, bool releaseSecondSlots, DirectOverlapReduceState &state)
    {
        for (uint32_t pairIdx = 0; pairIdx < DIRECT_REDUCE_THREAD_NUM; pairIdx++) {
            uint32_t lowSource = pairIdx * 2;
            uint32_t highSource = lowSource + 1;
            ThreadHandle pairThread = resCtx.threads[DIRECT_REDUCE_THREAD_BEGIN + pairIdx];
            state.threads[pairIdx] = pairThread;

            if (lowSource != param.myRank) {
                const ChannelInfo *channel = FindChannel(
                    resCtx, lowSource, DirectChannelType(param.myRank, lowSource));
                CHK_PRT_RET(channel == nullptr,
                    HCCL_ERROR("Overlap Direct low channel to rank[%u] not found", lowSource), HCCL_E_INTERNAL);
                CHK_RET(WaitChannel(pairThread, *channel, dataNotify));
            }
            if (highSource != param.myRank) {
                const ChannelInfo *channel = FindChannel(
                    resCtx, highSource, DirectChannelType(param.myRank, highSource));
                CHK_PRT_RET(channel == nullptr,
                    HCCL_ERROR("Overlap Direct high channel to rank[%u] not found", highSource), HCCL_E_INTERNAL);
                CHK_RET(WaitChannel(pairThread, *channel, dataNotify));
            }

            void *pairResult = nullptr;
            if (lowSource == param.myRank) {
                pairResult = ownedOutput;
                if (pairResult != sources[lowSource]) {
                    CHK_RET(CopyBytes(pairThread, pairResult, sources[lowSource], count * sizeof(float)));
                }
            } else {
                pairResult = const_cast<void *>(sources[lowSource]);
            }
            CHK_RET(ReduceRange(param, pairThread, pairResult, sources[highSource], count));
            state.results[pairIdx] = pairResult;

            if (!releaseSecondSlots) {
                continue;
            }
            for (uint32_t secondSource = 1; secondSource < EXPECTED_RANK_SIZE; secondSource++) {
                if (secondSource == param.myRank) {
                    continue;
                }
                uint32_t compactSlotIdx = ExcludedSlot(secondSource, param.myRank);
                if (DirectNestedPhysicalHighSource(param.myRank, compactSlotIdx) != highSource) {
                    continue;
                }
                const ChannelInfo *channel = FindChannel(
                    resCtx, secondSource, DirectChannelType(param.myRank, secondSource));
                CHK_PRT_RET(channel == nullptr,
                    HCCL_ERROR("Overlap Direct release channel to rank[%u] not found", secondSource),
                    HCCL_E_INTERNAL);
                CHK_RET(RecordChannel(pairThread, *channel, DIRECT_SLOT_RELEASE_NOTIFY));
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult QueueDirectOverlapUpperTree(const OpParam &param, DirectOverlapReduceState &state,
        uint64_t count, uint32_t treeNotifyBegin, void **rootResult, ThreadHandle *rootThread)
    {
        for (uint32_t level = 0, stride = 1; stride < state.results.size(); level++, stride <<= 1) {
            for (uint32_t lowIdx = 0; lowIdx < state.results.size(); lowIdx += stride * 2) {
                uint32_t highIdx = lowIdx + stride;
                uint32_t notifyIdx = treeNotifyBegin + level;
                CHK_RET(SignalThreadComplete(state.threads[highIdx], state.threads[lowIdx], notifyIdx));
                CHK_RET(WaitThreadComplete(state.threads[lowIdx], notifyIdx));
                CHK_RET(ReduceRange(
                    param, state.threads[lowIdx], state.results[lowIdx], state.results[highIdx], count));
            }
        }
        *rootResult = state.results[0];
        *rootThread = state.threads[0];
        return HCCL_SUCCESS;
    }

    HcclResult QueueDirectOverlapAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
        void *rootResult, ThreadHandle rootThread, void *ownedOutput)
    {
        void *publishSource = AddOffset(resCtx.localBuffer.addr, DIRECT_NESTED_PUBLISH_OFFSET);
        if (publishSource != rootResult) {
            CHK_RET(CopyBytes(rootThread, publishSource, rootResult, DIRECT_NESTED_SHARD_BYTES));
        }

        for (uint32_t peerRank = 0; peerRank < EXPECTED_RANK_SIZE; peerRank++) {
            if (peerRank == param.myRank || resCtx.threads[peerRank] == rootThread) {
                continue;
            }
            CHK_RET(SignalThreadComplete(
                rootThread, resCtx.threads[peerRank], DIRECT_OVERLAP_AG_START_NOTIFY));
        }

        for (uint32_t peerRank = 0; peerRank < EXPECTED_RANK_SIZE; peerRank++) {
            if (peerRank == param.myRank) {
                continue;
            }
            ThreadHandle worker = resCtx.threads[peerRank];
            if (worker != rootThread) {
                CHK_RET(WaitThreadComplete(worker, DIRECT_OVERLAP_AG_START_NOTIFY));
            }
            const ChannelInfo *channel = FindChannel(
                resCtx, peerRank, DirectChannelType(param.myRank, peerRank));
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Overlap Direct AG channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);
            DataRange recvRange = DirectNestedOwnedRange(param, peerRank);
            uint64_t recvOffset = recvRange.begin * sizeof(float);
            CHK_RET(CheckUserRange(param.count * sizeof(float), recvOffset,
                RangeBytes(recvRange), "Overlap Direct AG output"));
            CHK_RET(CheckBufferRange(channel->remoteCclMem, DIRECT_NESTED_PUBLISH_OFFSET,
                RangeBytes(recvRange), "Overlap Direct AG remote publish"));
            CHK_RET(RecordChannel(worker, *channel, DIRECT_AG0_READY_NOTIFY));
            CHK_RET(WaitChannel(worker, *channel, DIRECT_AG0_READY_NOTIFY));
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(worker, channel->handle,
                AddOffset(param.outputPtr, recvOffset),
                AddOffset(channel->remoteCclMem.addr, DIRECT_NESTED_PUBLISH_OFFSET),
                RangeBytes(recvRange))));
            CHK_RET(RecordChannel(worker, *channel, DIRECT_AG0_DONE_NOTIFY));
            CHK_RET(WaitChannel(worker, *channel, DIRECT_AG0_DONE_NOTIFY));
            if (worker != resCtx.threads[MAIN_THREAD_INDEX]) {
                uint32_t compactPeer = ExcludedSlot(peerRank, param.myRank);
                CHK_RET(SignalThreadComplete(worker, resCtx.threads[MAIN_THREAD_INDEX],
                    DIRECT_OVERLAP_AG_DONE_NOTIFY_BEGIN + compactPeer));
            }
        }

        if (rootResult != ownedOutput) {
            CHK_RET(CopyBytes(rootThread, ownedOutput, publishSource, DIRECT_NESTED_SHARD_BYTES));
        }
        if (rootThread != resCtx.threads[MAIN_THREAD_INDEX]) {
            CHK_RET(SignalThreadComplete(
                rootThread, resCtx.threads[MAIN_THREAD_INDEX], DIRECT_OVERLAP_ROOT_DONE_NOTIFY));
        }

        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        for (uint32_t peerRank = 0; peerRank < EXPECTED_RANK_SIZE; peerRank++) {
            if (peerRank == param.myRank || resCtx.threads[peerRank] == coordinator) {
                continue;
            }
            uint32_t compactPeer = ExcludedSlot(peerRank, param.myRank);
            CHK_RET(WaitThreadComplete(
                coordinator, DIRECT_OVERLAP_AG_DONE_NOTIFY_BEGIN + compactPeer));
        }
        if (rootThread != coordinator) {
            CHK_RET(WaitThreadComplete(coordinator, DIRECT_OVERLAP_ROOT_DONE_NOTIFY));
        }
        return HCCL_SUCCESS;
    }

    HcclResult DirectNestedOverlapAllReduce(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        CHK_RET(ValidateDirectNestedLayout(param, resCtx));
        CHK_RET(StartDirectOverlapThreads(resCtx));

        for (uint32_t ownerRank = 0; ownerRank < EXPECTED_RANK_SIZE; ownerRank++) {
            if (ownerRank != param.myRank) {
                CHK_RET(QueueDirectOverlapSegmentWrite(param, resCtx, ownerRank, 0));
            }
        }

        std::array<const void *, EXPECTED_RANK_SIZE> firstSources{};
        for (uint32_t sourceRank = 0; sourceRank < EXPECTED_RANK_SIZE; sourceRank++) {
            if (sourceRank == param.myRank) {
                DataRange range = DirectNestedSegmentRange(param, param.myRank, 0);
                firstSources[sourceRank] = AddOffset(param.inputPtr, range.begin * sizeof(float));
            } else {
                firstSources[sourceRank] = AddOffset(resCtx.localBuffer.addr,
                    DirectNestedSegmentOffset(param.myRank, sourceRank, 0));
            }
        }
        DataRange ownedRange = DirectNestedOwnedRange(param, param.myRank);
        void *ownedOutput = AddOffset(param.outputPtr, ownedRange.begin * sizeof(float));
        DirectOverlapReduceState firstState;
        CHK_RET(QueueDirectOverlapPairLayer(param, resCtx, firstSources, ownedOutput,
            DIRECT_NESTED_SEGMENT0_BYTES / sizeof(float), DIRECT_RS0_DATA_NOTIFY, true, firstState));

        for (uint32_t ownerRank = 0; ownerRank < EXPECTED_RANK_SIZE; ownerRank++) {
            if (ownerRank != param.myRank) {
                CHK_RET(QueueDirectOverlapSegmentWrite(param, resCtx, ownerRank, 1));
            }
        }

        void *firstRoot = nullptr;
        ThreadHandle firstRootThread = 0;
        CHK_RET(QueueDirectOverlapUpperTree(param, firstState,
            DIRECT_NESTED_SEGMENT0_BYTES / sizeof(float), DIRECT_OVERLAP_FIRST_TREE_NOTIFY_BEGIN,
            &firstRoot, &firstRootThread));

        std::array<const void *, EXPECTED_RANK_SIZE> secondSources{};
        for (uint32_t sourceRank = 0; sourceRank < EXPECTED_RANK_SIZE; sourceRank++) {
            if (sourceRank == param.myRank) {
                DataRange range = DirectNestedSegmentRange(param, param.myRank, 1);
                secondSources[sourceRank] = AddOffset(param.inputPtr, range.begin * sizeof(float));
            } else {
                secondSources[sourceRank] = AddOffset(resCtx.localBuffer.addr,
                    DirectNestedSegmentOffset(param.myRank, sourceRank, 1));
            }
        }
        DirectOverlapReduceState secondState;
        CHK_RET(QueueDirectOverlapPairLayer(param, resCtx, secondSources,
            AddOffset(ownedOutput, DIRECT_NESTED_SEGMENT0_BYTES),
            DIRECT_NESTED_SEGMENT1_BYTES / sizeof(float), DIRECT_RS1_DATA_NOTIFY, false, secondState));
        void *secondRoot = nullptr;
        ThreadHandle secondRootThread = 0;
        CHK_RET(QueueDirectOverlapUpperTree(param, secondState,
            DIRECT_NESTED_SEGMENT1_BYTES / sizeof(float), DIRECT_OVERLAP_SECOND_TREE_NOTIFY_BEGIN,
            &secondRoot, &secondRootThread));
        CHK_PRT_RET(firstRootThread != secondRootThread,
            HCCL_ERROR("Overlap Direct segment roots use different threads"), HCCL_E_INTERNAL);
        void *expectedSecondRoot = param.myRank == 0
            ? AddOffset(ownedOutput, DIRECT_NESTED_SEGMENT0_BYTES)
            : AddOffset(resCtx.localBuffer.addr,
                DIRECT_NESTED_PUBLISH_OFFSET + DIRECT_NESTED_SEGMENT0_BYTES);
        CHK_PRT_RET(secondRoot != expectedSecondRoot,
            HCCL_ERROR("Overlap Direct second segment has unexpected root"), HCCL_E_INTERNAL);
        void *rootResult = param.myRank == 0 ? ownedOutput : firstRoot;
        return QueueDirectOverlapAllGather(
            param, resCtx, rootResult, secondRootThread, ownedOutput);
    }

    HcclResult QueueDirectNestedSegmentWorker(const OpParam &param, const AlgResourceCtx &resCtx,
        uint32_t segmentIdx, uint32_t workerIdx)
    {
        ThreadHandle worker = resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + workerIdx];
        if (segmentIdx == 0) {
            CHK_RET(WaitThreadStart(worker));
        }
        uint32_t ownerRank = DirectPeerRank(param.myRank, workerIdx);
        const ChannelInfo *channel = FindChannel(
            resCtx, ownerRank, DirectChannelType(param.myRank, ownerRank));
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Nested Direct RS channel to rank[%u] not found", ownerRank), HCCL_E_INTERNAL);
        if (segmentIdx == 1 && param.myRank != 0) {
            CHK_RET(WaitChannel(worker, *channel, DIRECT_SLOT_RELEASE_NOTIFY));
        }

        DataRange sendRange = DirectNestedSegmentRange(param, ownerRank, segmentIdx);
        uint64_t sendOffset = sendRange.begin * sizeof(float);
        uint64_t remoteOffset = DirectNestedSegmentOffset(ownerRank, param.myRank, segmentIdx);
        uint32_t dataNotify = segmentIdx == 0 ? DIRECT_RS0_DATA_NOTIFY : DIRECT_RS1_DATA_NOTIFY;
        CHK_RET(CheckUserRange(param.count * sizeof(float), sendOffset,
            RangeBytes(sendRange), "Nested Direct RS input"));
        CHK_RET(CheckBufferRange(channel->remoteCclMem, remoteOffset,
            RangeBytes(sendRange), "Nested Direct RS remote CCL"));
        CHK_RET(BatchWriteSignal(worker, *channel,
            AddOffset(channel->remoteCclMem.addr, remoteOffset),
            AddOffset(param.inputPtr, sendOffset), RangeBytes(sendRange), dataNotify));
        return WaitChannel(worker, *channel, dataNotify);
    }

    HcclResult QueueDirectNestedReleases(const OpParam &param, const AlgResourceCtx &resCtx,
        ThreadHandle producer, uint32_t releasedHighSource)
    {
        for (uint32_t secondSource = 1; secondSource < EXPECTED_RANK_SIZE; secondSource++) {
            if (secondSource == param.myRank) {
                continue;
            }
            uint32_t compactSlotIdx = ExcludedSlot(secondSource, param.myRank);
            if (DirectNestedPhysicalHighSource(param.myRank, compactSlotIdx) != releasedHighSource) {
                continue;
            }
            const ChannelInfo *channel = FindChannel(
                resCtx, secondSource, DirectChannelType(param.myRank, secondSource));
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Nested Direct release channel to rank[%u] not found", secondSource),
                HCCL_E_INTERNAL);
            CHK_RET(RecordChannel(producer, *channel, DIRECT_SLOT_RELEASE_NOTIFY));
        }
        return HCCL_SUCCESS;
    }

    HcclResult QueueDirectNestedReduceTree(const OpParam &param, const AlgResourceCtx &resCtx,
        uint32_t segmentIdx, bool joinRootToMain)
    {
        DataRange segmentRange = DirectNestedSegmentRange(param, param.myRank, segmentIdx);
        uint64_t segmentBytes = RangeBytes(segmentRange);
        const void *localInput = AddOffset(param.inputPtr, segmentRange.begin * sizeof(float));
        bool usesOwnerLeafCarrier = DirectNestedUsesOwnerLeafCarrier(param);
        uint32_t ownerLeafCarrier = DirectNestedOwnerLeafCarrier(param);

        for (uint32_t level = 0; level < DIRECT_REDUCE_LEVEL_NUM; level++) {
            uint32_t groupSize = 1U << (level + 1);
            uint32_t groupNum = EXPECTED_RANK_SIZE / groupSize;
            for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
                uint32_t lowSource = groupIdx * groupSize;
                uint32_t highSource = lowSource + groupSize / 2;
                ThreadHandle lowProducer = DirectNestedProducerThread(param, resCtx, lowSource);
                ThreadHandle highProducer = DirectNestedProducerThread(param, resCtx, highSource);
                uint32_t readyNotify = DIRECT_NESTED_TREE_NOTIFY_BEGIN
                                       + segmentIdx * DIRECT_REDUCE_LEVEL_NUM + level;
                CHK_RET(SignalThreadComplete(highProducer, lowProducer, readyNotify));
                CHK_RET(WaitThreadComplete(lowProducer, readyNotify));

                void *dst = DirectNestedTreeAddress(param, resCtx, segmentIdx, lowSource);
                const void *src = DirectNestedTreeAddress(param, resCtx, segmentIdx, highSource);
                if (level == 0 && lowSource == param.myRank) {
                    if (usesOwnerLeafCarrier) {
                        src = localInput;
                    } else {
                        CHK_RET(CopyBytes(lowProducer, dst, localInput, segmentBytes));
                    }
                }
                if (level == 0 && highSource == param.myRank) {
                    src = localInput;
                }
                CHK_RET(ReduceRange(param, lowProducer, dst, src, segmentRange.count));
                if (segmentIdx == 0 && level == 0 && highSource != param.myRank) {
                    if (!usesOwnerLeafCarrier || highSource != ownerLeafCarrier) {
                        CHK_RET(QueueDirectNestedReleases(
                            param, resCtx, lowProducer, highSource));
                    }
                }
                if (segmentIdx == 0 && level > 0 && usesOwnerLeafCarrier
                    && highSource == param.myRank) {
                    CHK_RET(QueueDirectNestedReleases(
                        param, resCtx, lowProducer, ownerLeafCarrier));
                }
            }
        }

        if (joinRootToMain && param.myRank != 0) {
            ThreadHandle rootProducer = DirectNestedProducerThread(param, resCtx, 0);
            CHK_RET(SignalThreadComplete(
                rootProducer, resCtx.threads[MAIN_THREAD_INDEX], DIRECT_NESTED_ROOT_DONE_NOTIFY));
            CHK_RET(WaitThreadComplete(
                resCtx.threads[MAIN_THREAD_INDEX], DIRECT_NESTED_ROOT_DONE_NOTIFY));
        }
        return HCCL_SUCCESS;
    }

    HcclResult QueueDirectNestedAllGatherWorker(
        const OpParam &param, const AlgResourceCtx &resCtx, uint32_t workerIdx)
    {
        ThreadHandle worker = resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + workerIdx];
        uint32_t remoteRank = DirectPeerRank(param.myRank, workerIdx);
        const ChannelInfo *channel = FindChannel(
            resCtx, remoteRank, DirectChannelType(param.myRank, remoteRank));
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Nested Direct AG channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
        DataRange recvRange = DirectNestedOwnedRange(param, remoteRank);
        uint64_t recvOffset = recvRange.begin * sizeof(float);
        CHK_RET(CheckUserRange(param.count * sizeof(float), recvOffset,
            RangeBytes(recvRange), "Nested Direct AG output"));
        CHK_RET(CheckBufferRange(channel->remoteCclMem, DIRECT_NESTED_PUBLISH_OFFSET,
            RangeBytes(recvRange), "Nested Direct AG remote publish"));

        CHK_RET(WaitThreadComplete(worker, DIRECT_NESTED_PUBLISH_READY_THREAD_NOTIFY));
        CHK_RET(RecordChannel(worker, *channel, DIRECT_AG0_READY_NOTIFY));
        CHK_RET(WaitChannel(worker, *channel, DIRECT_AG0_READY_NOTIFY));
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(worker, channel->handle,
            AddOffset(param.outputPtr, recvOffset),
            AddOffset(channel->remoteCclMem.addr, DIRECT_NESTED_PUBLISH_OFFSET),
            RangeBytes(recvRange))));
        CHK_RET(RecordChannel(worker, *channel, DIRECT_AG0_DONE_NOTIFY));
        CHK_RET(WaitChannel(worker, *channel, DIRECT_AG0_DONE_NOTIFY));
        return SignalThreadComplete(worker, resCtx.threads[MAIN_THREAD_INDEX],
            DIRECT_NESTED_MAIN_DONE_NOTIFY_BEGIN + workerIdx);
    }

    [[maybe_unused]] HcclResult DirectNestedAllReduce(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        CHK_RET(ValidateDirectNestedLayout(param, resCtx));
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(StartThread(
                coordinator, resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + workerIdx]));
        }
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(QueueDirectNestedSegmentWorker(param, resCtx, 0, workerIdx));
        }
        CHK_RET(QueueDirectNestedReduceTree(param, resCtx, 0, false));
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(QueueDirectNestedSegmentWorker(param, resCtx, 1, workerIdx));
        }
        CHK_RET(QueueDirectNestedReduceTree(param, resCtx, 1, true));

        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(SignalThreadComplete(coordinator,
                resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + workerIdx],
                DIRECT_NESTED_PUBLISH_READY_THREAD_NOTIFY));
        }
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(QueueDirectNestedAllGatherWorker(param, resCtx, workerIdx));
        }
        DataRange ownedRange = DirectNestedOwnedRange(param, param.myRank);
        CHK_RET(CopyBytes(coordinator, AddOffset(param.outputPtr, ownedRange.begin * sizeof(float)),
            AddOffset(resCtx.localBuffer.addr, DIRECT_NESTED_PUBLISH_OFFSET),
            DIRECT_NESTED_SHARD_BYTES));
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(WaitThreadComplete(
                coordinator, DIRECT_NESTED_MAIN_DONE_NOTIFY_BEGIN + workerIdx));
        }
        return HCCL_SUCCESS;
    }

    ThreadHandle DirectProducerThread(
        const OpParam &param, const AlgResourceCtx &resCtx, uint32_t sourceRank)
    {
        if (sourceRank == param.myRank) {
            return resCtx.threads[MAIN_THREAD_INDEX];
        }
        uint32_t workerIdx = DirectWorkerIndex(param.myRank, sourceRank);
        return resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + workerIdx];
    }

    bool DirectUsesOwnerLeafCarrier(const OpParam &param)
    {
        return param.count * sizeof(float) == PERFORMANCE_UNEVEN_BYTES
               && (param.myRank & 1U) == 0;
    }

    uint32_t DirectOwnerLeafCarrier(const OpParam &param)
    {
        return DirectUsesOwnerLeafCarrier(param) ? param.myRank + 1 : INVALID_VALUE_RANKID;
    }

    void *DirectTreeResultAddress(const OpParam &param, const AlgResourceCtx &resCtx,
        const DirectConfig &config, uint32_t tileIdx, uint32_t sourceRank)
    {
        DataRange ownedRange = DirectOwnedRange(param, config, tileIdx, param.myRank);
        if (sourceRank == param.myRank) {
            if (DirectUsesOwnerLeafCarrier(param)) {
                return AddOffset(resCtx.localBuffer.addr,
                    DirectSlotOffset(DirectOwnerLeafCarrier(param), param.myRank, config.slotStride));
            }
            return AddOffset(param.outputPtr, ownedRange.begin * sizeof(float));
        }
        return AddOffset(resCtx.localBuffer.addr,
            DirectSlotOffset(sourceRank, param.myRank, config.slotStride));
    }

    HcclResult QueueDirectReduceScatterWorker(const OpParam &param, const AlgResourceCtx &resCtx,
        const DirectConfig &config, uint32_t tileIdx, uint32_t workerIdx, uint32_t dataNotify,
        bool signalMain, bool waitStart)
    {
        ThreadHandle worker = resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + workerIdx];
        if (waitStart) {
            CHK_RET(WaitThreadStart(worker));
        }
        uint32_t remoteRank = DirectPeerRank(param.myRank, workerIdx);
        const ChannelInfo *channel = FindChannel(
            resCtx, remoteRank, DirectChannelType(param.myRank, remoteRank));
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Direct RS channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);

        DataRange sendRange = DirectOwnedRange(param, config, tileIdx, remoteRank);
        uint64_t sendOffset = sendRange.begin * sizeof(float);
        uint64_t remoteOffset = DirectSlotOffset(param.myRank, remoteRank, config.slotStride);
        CHK_RET(CheckUserRange(
            param.count * sizeof(float), sendOffset, RangeBytes(sendRange), "Direct RS input"));
        CHK_RET(CheckBufferRange(
            channel->remoteCclMem, remoteOffset, RangeBytes(sendRange), "Direct RS remote CCL"));
        CHK_RET(BatchWriteSignal(worker, *channel,
            AddOffset(channel->remoteCclMem.addr, remoteOffset), AddOffset(param.inputPtr, sendOffset),
            RangeBytes(sendRange), dataNotify));
        CHK_RET(WaitChannel(worker, *channel, dataNotify));
        return signalMain
                   ? SignalThreadComplete(
                         worker, resCtx.threads[MAIN_THREAD_INDEX], DirectMainPeerNotify(workerIdx))
                   : HCCL_SUCCESS;
    }

    HcclResult QueueDirectSlotRelease(const OpParam &param, const AlgResourceCtx &resCtx,
        ThreadHandle producer, uint32_t sourceRank)
    {
        const ChannelInfo *channel = FindChannel(
            resCtx, sourceRank, DirectChannelType(param.myRank, sourceRank));
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Direct release channel to rank[%u] not found", sourceRank), HCCL_E_INTERNAL);
        return RecordChannel(producer, *channel, DIRECT_SLOT_RELEASE_NOTIFY);
    }

    HcclResult QueueDirectReduceTree(const OpParam &param, const AlgResourceCtx &resCtx,
        const DirectConfig &config, uint32_t tileIdx, bool releaseTile0Slots)
    {
        DataRange ownedRange = DirectOwnedRange(param, config, tileIdx, param.myRank);
        uint64_t ownedOffset = ownedRange.begin * sizeof(float);
        const void *localInput = AddOffset(param.inputPtr, ownedOffset);
        void *localOutput = AddOffset(param.outputPtr, ownedOffset);
        uint32_t publishSource = DirectPublishSource(param.myRank);

        for (uint32_t level = 0; level < DIRECT_REDUCE_LEVEL_NUM; level++) {
            uint32_t groupSize = 1U << (level + 1);
            uint32_t groupNum = EXPECTED_RANK_SIZE / groupSize;
            for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
                uint32_t lowSource = groupIdx * groupSize;
                uint32_t highSource = lowSource + groupSize / 2;
                ThreadHandle lowProducer = DirectProducerThread(param, resCtx, lowSource);
                ThreadHandle highProducer = DirectProducerThread(param, resCtx, highSource);
                bool lowIsMain = lowSource == param.myRank;
                uint32_t readyNotify = DirectTreeReadyNotify(level, lowIsMain);
                CHK_RET(SignalThreadComplete(highProducer, lowProducer, readyNotify));
                CHK_RET(WaitThreadComplete(lowProducer, readyNotify));

                void *dst = DirectTreeResultAddress(param, resCtx, config, tileIdx, lowSource);
                const void *src = DirectTreeResultAddress(param, resCtx, config, tileIdx, highSource);
                if (level == 0 && lowSource == param.myRank) {
                    if (DirectUsesOwnerLeafCarrier(param)) {
                        src = localInput;
                    } else {
                        CHK_RET(CopyBytes(lowProducer, localOutput, localInput, RangeBytes(ownedRange)));
                        dst = localOutput;
                    }
                }
                if (level == 0 && highSource == param.myRank) {
                    src = localInput;
                }
                CHK_RET(ReduceRange(param, lowProducer, dst, src, ownedRange.count));
                if (releaseTile0Slots && highSource != param.myRank && highSource != publishSource) {
                    CHK_RET(QueueDirectSlotRelease(param, resCtx, lowProducer, highSource));
                }
            }
        }

        if (param.myRank != 0) {
            ThreadHandle rootProducer = DirectProducerThread(param, resCtx, 0);
            CHK_RET(SignalThreadComplete(
                rootProducer, resCtx.threads[MAIN_THREAD_INDEX], DIRECT_MAIN_ROOT_NOTIFY));
            CHK_RET(WaitThreadComplete(
                resCtx.threads[MAIN_THREAD_INDEX], DIRECT_MAIN_ROOT_NOTIFY));
        }
        return HCCL_SUCCESS;
    }

    HcclResult QueueDirectTile1Worker(const OpParam &param, const AlgResourceCtx &resCtx,
        const DirectConfig &config, uint32_t workerIdx)
    {
        ThreadHandle worker = resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + workerIdx];
        uint32_t remoteRank = DirectPeerRank(param.myRank, workerIdx);
        const ChannelInfo *channel = FindChannel(
            resCtx, remoteRank, DirectChannelType(param.myRank, remoteRank));
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Direct Tile1 channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
        CHK_RET(WaitChannel(worker, *channel, DIRECT_SLOT_RELEASE_NOTIFY));
        return QueueDirectReduceScatterWorker(param, resCtx, config, 1, workerIdx,
            DIRECT_RS1_DATA_NOTIFY, true, false);
    }

    HcclResult QueueDirectTile1Workers(const OpParam &param, const AlgResourceCtx &resCtx,
        const DirectConfig &config, bool deferred)
    {
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            uint32_t remoteRank = DirectPeerRank(param.myRank, workerIdx);
            bool pairReleaseIsDeferred =
                param.myRank == DirectPublishSource(remoteRank)
                || remoteRank == DirectPublishSource(param.myRank);
            if (pairReleaseIsDeferred == deferred) {
                CHK_RET(QueueDirectTile1Worker(param, resCtx, config, workerIdx));
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult QueueDirectPublish(const OpParam &param, const AlgResourceCtx &resCtx,
        const DirectConfig &config, uint32_t tileIdx)
    {
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        DataRange ownedRange = DirectOwnedRange(param, config, tileIdx, param.myRank);
        uint64_t ownedOffset = ownedRange.begin * sizeof(float);
        uint64_t ownedBytes = RangeBytes(ownedRange);
        CHK_RET(CheckUserRange(
            param.count * sizeof(float), ownedOffset, ownedBytes, "Direct publish output"));
        CHK_RET(CheckBufferRange(resCtx.localBuffer, 0, ownedBytes, "Direct publish CCL"));
        void *outputAddr = AddOffset(param.outputPtr, ownedOffset);
        if (DirectUsesOwnerLeafCarrier(param)) {
            return CopyBytes(coordinator, outputAddr, resCtx.localBuffer.addr, ownedBytes);
        }
        if (param.myRank == 0) {
            return CopyBytes(coordinator, resCtx.localBuffer.addr, outputAddr, ownedBytes);
        }
        return CopyBytes(coordinator, outputAddr, resCtx.localBuffer.addr, ownedBytes);
    }

    HcclResult QueueDirectPublishReady(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t readyNotify)
    {
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(SignalThreadComplete(coordinator,
                resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + workerIdx], readyNotify));
        }
        return HCCL_SUCCESS;
    }

    HcclResult QueueDirectAllGatherWorker(const OpParam &param, const AlgResourceCtx &resCtx,
        const DirectConfig &config, uint32_t tileIdx, uint32_t workerIdx, uint32_t readyNotify,
        uint32_t doneNotify)
    {
        ThreadHandle worker = resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + workerIdx];
        uint32_t remoteRank = DirectPeerRank(param.myRank, workerIdx);
        const ChannelInfo *channel = FindChannel(
            resCtx, remoteRank, DirectChannelType(param.myRank, remoteRank));
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Direct AG channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
        DataRange recvRange = DirectOwnedRange(param, config, tileIdx, remoteRank);
        uint64_t recvOffset = recvRange.begin * sizeof(float);
        CHK_RET(CheckUserRange(
            param.count * sizeof(float), recvOffset, RangeBytes(recvRange), "Direct AG output"));
        CHK_RET(CheckBufferRange(
            channel->remoteCclMem, 0, RangeBytes(recvRange), "Direct AG remote publish"));

        CHK_RET(WaitThreadComplete(worker, readyNotify));
        CHK_RET(RecordChannel(worker, *channel, readyNotify));
        CHK_RET(WaitChannel(worker, *channel, readyNotify));
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(worker, channel->handle,
            AddOffset(param.outputPtr, recvOffset), channel->remoteCclMem.addr,
            RangeBytes(recvRange))));
        CHK_RET(RecordChannel(worker, *channel, doneNotify));
        CHK_RET(WaitChannel(worker, *channel, doneNotify));
        return SignalThreadComplete(
            worker, resCtx.threads[MAIN_THREAD_INDEX], DirectMainAuxNotify(workerIdx));
    }

    HcclResult QueueDirectAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
        const DirectConfig &config, uint32_t tileIdx, uint32_t readyNotify, uint32_t doneNotify)
    {
        bool overlapOwnedCopy = param.count * sizeof(float) == PERFORMANCE_UNEVEN_BYTES;
        if (!overlapOwnedCopy) {
            CHK_RET(QueueDirectPublish(param, resCtx, config, tileIdx));
        }
        CHK_RET(QueueDirectPublishReady(param, resCtx, readyNotify));
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(QueueDirectAllGatherWorker(
                param, resCtx, config, tileIdx, workerIdx, readyNotify, doneNotify));
        }
        if (overlapOwnedCopy) {
            CHK_RET(QueueDirectPublish(param, resCtx, config, tileIdx));
        }
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(WaitThreadComplete(coordinator, DirectMainAuxNotify(workerIdx)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult DirectSingleTileAllReduce(
        const OpParam &param, const AlgResourceCtx &resCtx, const DirectConfig &config)
    {
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(StartThread(
                coordinator, resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + workerIdx]));
        }
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(QueueDirectReduceScatterWorker(param, resCtx, config, 0, workerIdx,
                DIRECT_RS0_DATA_NOTIFY, false, true));
        }
        CHK_RET(QueueDirectReduceTree(param, resCtx, config, 0, false));
        return QueueDirectAllGather(param, resCtx, config, 0,
            DIRECT_AG0_READY_NOTIFY, DIRECT_AG0_DONE_NOTIFY);
    }

    HcclResult DirectTwoTilePipelineAllReduce(
        const OpParam &param, const AlgResourceCtx &resCtx, const DirectConfig &config)
    {
        ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(StartThread(
                coordinator, resCtx.threads[DIRECT_WORKER_THREAD_BEGIN + workerIdx]));
        }
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(QueueDirectReduceScatterWorker(param, resCtx, config, 0, workerIdx,
                DIRECT_RS0_DATA_NOTIFY, false, true));
        }
        CHK_RET(QueueDirectReduceTree(param, resCtx, config, 0, true));

        // 非发布槽在固定树消费后即可接收 Tile1；每个 Worker 在自身最后一个树节点后持续工作。
        CHK_RET(QueueDirectTile1Workers(param, resCtx, config, false));
        CHK_RET(QueueDirectAllGather(param, resCtx, config, 0,
            DIRECT_AG0_READY_NOTIFY, DIRECT_AG0_DONE_NOTIFY));

        // CCL slot0 是 Tile0 发布区。全部 15 个 AG Read 完成后，才释放与其重叠的最后一条来源槽。
        uint32_t publishSource = DirectPublishSource(param.myRank);
        CHK_RET(QueueDirectSlotRelease(
            param, resCtx, resCtx.threads[MAIN_THREAD_INDEX], publishSource));
        CHK_RET(QueueDirectTile1Workers(param, resCtx, config, true));
        for (uint32_t workerIdx = 0; workerIdx < DIRECT_PEER_NUM; workerIdx++) {
            CHK_RET(WaitThreadComplete(coordinator, DirectMainPeerNotify(workerIdx)));
        }

        CHK_RET(QueueDirectReduceTree(param, resCtx, config, 1, false));
        return QueueDirectAllGather(param, resCtx, config, 1,
            DIRECT_AG1_READY_NOTIFY, DIRECT_AG1_DONE_NOTIFY);
    }

    HcclResult DirectPipelinedAllReduce(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        if (param.count * sizeof(float) == PERFORMANCE_LARGE_BYTES) {
            return DirectNestedOverlapAllReduce(param, resCtx);
        }
        DirectConfig config;
        CHK_RET(GetDirectConfig(param, resCtx, config));
        CHK_RET(CheckBufferRange(resCtx.localBuffer, 0,
            DIRECT_PEER_NUM * config.slotStride, "Direct source slots"));
        for (uint32_t tileIdx = 0; tileIdx < config.tileNum; tileIdx++) {
            DataRange ownedRange = DirectOwnedRange(param, config, tileIdx, param.myRank);
            CHK_RET(CheckUserRange(param.count * sizeof(float), ownedRange.begin * sizeof(float),
                RangeBytes(ownedRange), "Direct owned output"));
        }
        if (config.tileNum == 1) {
            return DirectSingleTileAllReduce(param, resCtx, config);
        }
        CHK_PRT_RET(config.tileNum != 2,
            HCCL_ERROR("Unsupported Direct tile count[%u]", config.tileNum), HCCL_E_INTERNAL);
        return DirectTwoTilePipelineAllReduce(param, resCtx, config);
    }
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only FP32 SUM is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("AICPU thread is missing"), HCCL_E_INTERNAL);
    CHK_PRT_RET(param.rankSize != 1 && param.rankSize != EXPECTED_RANK_SIZE,
        HCCL_ERROR("Only rankSize 1 or 16 is supported, rankSize[%u]", param.rankSize), HCCL_E_NOT_SUPPORT);

    ThreadHandle mainThread = resCtx.threads[MAIN_THREAD_INDEX];
    uint64_t totalBytes = param.count * sizeof(float);
    if (param.rankSize == 1) {
        return CopyBytes(mainThread, param.outputPtr, param.inputPtr, totalBytes);
    }
    CHK_PRT_RET(resCtx.threads.size() != EXPECTED_PARALLEL_THREAD_NUM,
        HCCL_ERROR(
            "Invalid AICPU thread count, expected[%u], actual[%zu]", EXPECTED_PARALLEL_THREAD_NUM,
            resCtx.threads.size()),
        HCCL_E_INTERNAL);

    if (totalBytes == PERFORMANCE_SMALL_BYTES) {
        return CrossServerButterflyAllReduce(param, resCtx);
    }
    if (totalBytes <= SMALL_MESSAGE_BYTES) {
        return RecursiveDoublingAllReduce(param, resCtx, mainThread, param.inputPtr, param.outputPtr, param.count);
    }
    if (totalBytes == PERFORMANCE_LARGE_BYTES || totalBytes == PERFORMANCE_UNEVEN_BYTES) {
        return DirectPipelinedAllReduce(param, resCtx);
    }
    return ParallelRsagAllReduce(param, resCtx);
}
} // namespace ops_hccl