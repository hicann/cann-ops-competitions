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
#include "exec_op.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr uint32_t SCATTER_READY_NOTIFY = 0;
constexpr uint32_t SCATTER_DATA_NOTIFY = 1;
constexpr uint32_t SCATTER_ACK_NOTIFY = 2;
constexpr uint32_t AG_READY_NOTIFY = 3;
constexpr uint32_t AG_ACK_NOTIFY = 4;

using custom_broadcast::RankSlice;
using custom_broadcast::SegmentDesc;
using RankSlices = std::array<RankSlice, custom_broadcast::RANK_SIZE>;
using RoundSegments = std::array<SegmentDesc, custom_broadcast::RANK_SIZE>;

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRankIndex)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRankIndex == remoteRankIndex) {
            return &channel;
        }
    }
    return nullptr;
}

uint32_t FindChannelPosition(const AlgResourceCtx &resCtx, uint32_t remoteRankIndex)
{
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        if (resCtx.channels[i].remoteRankIndex == remoteRankIndex) {
            return i;
        }
    }
    return custom_broadcast::PEER_SIZE;
}

void *OffsetAddr(void *addr, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(addr) + offset);
}

HcclResult CheckChannelBuffer(const ChannelInfo &channel, uint64_t bytes)
{
    CHK_PTR_NULL(channel.remoteCclMem.addr);
    CHK_PRT_RET(bytes > channel.remoteCclMem.size,
        HCCL_ERROR("Remote CCL buffer for rank[%u] is too small: size[%llu], required[%llu]", channel.remoteRank,
            static_cast<unsigned long long>(channel.remoteCclMem.size), static_cast<unsigned long long>(bytes)),
        HCCL_E_MEMORY);
    return HCCL_SUCCESS;
}

HcclResult BatchWriteAndNotify(ThreadHandle thread, const ChannelInfo &channel, void *dst, const void *src,
    uint64_t bytes, uint32_t notifyIdx)
{
    if (bytes == 0) {
        return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyIdx));
    }
    HcommBatchTransferDesc descs[2] = {};
    descs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
    descs[0].transferInfo.write.len = bytes;
    descs[0].transferInfo.write.dst = dst;
    descs[0].transferInfo.write.src = const_cast<void *>(src);
    descs[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    descs[1].transferInfo.notifyRecord.notifyIdx = notifyIdx;
    return static_cast<HcclResult>(HcommBatchTransferOnThread(thread, channel.handle, descs, 2));
}

HcclResult BatchReadAndNotify(ThreadHandle thread, const ChannelInfo &channel, void *dst, const void *src,
    uint64_t bytes, uint32_t notifyIdx)
{
    if (bytes == 0) {
        return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyIdx));
    }
    HcommBatchTransferDesc descs[2] = {};
    descs[0].transType = HCOMM_TRANSFER_TYPE_READ;
    descs[0].transferInfo.read.len = bytes;
    descs[0].transferInfo.read.dst = dst;
    descs[0].transferInfo.read.src = const_cast<void *>(src);
    descs[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    descs[1].transferInfo.notifyRecord.notifyIdx = notifyIdx;
    return static_cast<HcclResult>(HcommBatchTransferOnThread(thread, channel.handle, descs, 2));
}

HcclResult BeginParallel(const std::vector<ThreadHandle> &threads)
{
    CHK_PRT_RET(threads.size() != custom_broadcast::PEER_SIZE,
        HCCL_ERROR("Unexpected thread count[%llu]", static_cast<unsigned long long>(threads.size())),
        HCCL_E_INTERNAL);
    for (uint32_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[i], 0)));
    }
    for (uint32_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[i], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult EndParallel(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], i - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[i], threads[0], i - 1)));
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateResources(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.rankIndex >= custom_broadcast::RANK_SIZE,
        HCCL_ERROR("Invalid logical rank index[%u]", resCtx.rankIndex), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ranks.size() != custom_broadcast::RANK_SIZE,
        HCCL_ERROR("Unexpected rank vector size[%llu]", static_cast<unsigned long long>(resCtx.ranks.size())),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.threads.size() != custom_broadcast::PEER_SIZE,
        HCCL_ERROR("Unexpected thread count[%llu]", static_cast<unsigned long long>(resCtx.threads.size())),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.channels.size() != custom_broadcast::PEER_SIZE,
        HCCL_ERROR("Unexpected channel count[%llu]", static_cast<unsigned long long>(resCtx.channels.size())),
        HCCL_E_INTERNAL);
    CHK_PTR_NULL(resCtx.localBuffer.addr);

    uint32_t expectedRemoteIndex = 0;
    uint32_t intraPeerNum = 0;
    uint32_t interPeerNum = 0;
    for (const ChannelInfo &channel : resCtx.channels) {
        if (expectedRemoteIndex == resCtx.rankIndex) {
            ++expectedRemoteIndex;
        }
        CHK_PRT_RET(channel.remoteRankIndex != expectedRemoteIndex,
            HCCL_ERROR("Unexpected channel index[%u], expected[%u]", channel.remoteRankIndex, expectedRemoteIndex),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.notifyNum < 5,
            HCCL_ERROR("Channel to rank[%u] has insufficient notify resources", channel.remoteRank),
            HCCL_E_INTERNAL);
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        intraPeerNum += static_cast<uint32_t>(channel.netLayer == custom_broadcast::INTRA_NET_LAYER);
        interPeerNum += static_cast<uint32_t>(channel.netLayer == custom_broadcast::INTER_NET_LAYER);
        ++expectedRemoteIndex;
    }
    CHK_PRT_RET(intraPeerNum != custom_broadcast::LOCAL_RANK_SIZE - 1 ||
            interPeerNum != custom_broadcast::LOCAL_RANK_SIZE,
        HCCL_ERROR("Unexpected channel split: intra[%u], inter[%u]", intraPeerNum, interPeerNum), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult GetRootIndex(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t &rootIndex)
{
    auto rootIt = std::find(resCtx.ranks.begin(), resCtx.ranks.end(), param.root);
    CHK_PRT_RET(rootIt == resCtx.ranks.end(),
        HCCL_ERROR("Root rank[%u] is absent from topology", param.root), HCCL_E_PARA);
    rootIndex = static_cast<uint32_t>(std::distance(resCtx.ranks.begin(), rootIt));
    return HCCL_SUCCESS;
}

RankSlices BuildRankSlices(uint64_t totalBytes)
{
    RankSlices slices{};
    for (uint32_t rankIndex = 0; rankIndex < custom_broadcast::RANK_SIZE; ++rankIndex) {
        slices[rankIndex] = custom_broadcast::BuildRankSlice(totalBytes, rankIndex);
    }
    return slices;
}

HcclResult ValidateRankSlices(const RankSlices &slices, uint64_t totalBytes)
{
    uint64_t coveredBytes = 0;
    for (const RankSlice &slice : slices) {
        CHK_PRT_RET(slice.offsetBytes != coveredBytes ||
                slice.validBytes % custom_broadcast::DATA_TYPE_SIZE != 0,
            HCCL_ERROR("Invalid rank slice: offset[%llu], bytes[%llu]",
                static_cast<unsigned long long>(slice.offsetBytes),
                static_cast<unsigned long long>(slice.validBytes)),
            HCCL_E_INTERNAL);
        coveredBytes += slice.validBytes;
    }
    CHK_PRT_RET(coveredBytes != totalBytes,
        HCCL_ERROR("Rank slices cover[%llu], expected[%llu]", static_cast<unsigned long long>(coveredBytes),
            static_cast<unsigned long long>(totalBytes)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

uint64_t MinimumBufferSize(const AlgResourceCtx &resCtx)
{
    uint64_t result = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        result = std::min(result, channel.remoteCclMem.size);
    }
    return result;
}

uint64_t MaximumSliceSize(const RankSlices &slices)
{
    uint64_t result = 0;
    for (const RankSlice &slice : slices) {
        result = std::max(result, slice.validBytes);
    }
    return result;
}

uint64_t SelectTileBytes(const AlgResourceCtx &resCtx, const RankSlices &slices)
{
    const uint64_t maxSliceBytes = MaximumSliceSize(slices);
    const uint64_t minBufferBytes = MinimumBufferSize(resCtx);
    if (maxSliceBytes <= minBufferBytes) {
        return maxSliceBytes;
    }
    return custom_broadcast::AlignDown(minBufferBytes, custom_broadcast::PREFERRED_ALIGN);
}

uint64_t GetRoundCount(const RankSlices &slices, uint64_t tileBytes)
{
    uint64_t result = 0;
    for (const RankSlice &slice : slices) {
        result = std::max(result, custom_broadcast::DivUp(slice.validBytes, tileBytes));
    }
    return result;
}

RoundSegments BuildRoundSegments(const RankSlices &slices, uint64_t round, uint64_t tileBytes)
{
    RoundSegments segments{};
    for (uint32_t rankIndex = 0; rankIndex < custom_broadcast::RANK_SIZE; ++rankIndex) {
        segments[rankIndex] = custom_broadcast::BuildSegment(slices[rankIndex], round, tileBytes);
    }
    return segments;
}

HcclResult DirectScatterRound(const OpParam &param, const AlgResourceCtx &resCtx,
    const RoundSegments &segments, uint32_t rootIndex)
{
    const SegmentDesc &localSegment = segments[resCtx.rankIndex];
    if (resCtx.rankIndex == rootIndex && localSegment.validBytes != 0) {
        const void *src = OffsetAddr(param.inputPtr, localSegment.offsetBytes);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            resCtx.threads[0], resCtx.localBuffer.addr, src, localSegment.validBytes)));
    }

    CHK_RET(BeginParallel(resCtx.threads));
    if (resCtx.rankIndex == rootIndex) {
        for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
            const ChannelInfo &channel = resCtx.channels[i];
            const SegmentDesc &segment = segments[channel.remoteRankIndex];
            const ThreadHandle thread = resCtx.threads[i];
            CHK_RET(CheckChannelBuffer(channel, segment.validBytes));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, SCATTER_READY_NOTIFY, CUSTOM_TIMEOUT)));
            const void *src = OffsetAddr(param.inputPtr, segment.offsetBytes);
            CHK_RET(BatchWriteAndNotify(thread, channel, channel.remoteCclMem.addr, src, segment.validBytes,
                SCATTER_DATA_NOTIFY));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, SCATTER_ACK_NOTIFY, CUSTOM_TIMEOUT)));
        }
    } else {
        const uint32_t rootChannelPosition = FindChannelPosition(resCtx, rootIndex);
        CHK_PRT_RET(rootChannelPosition >= resCtx.channels.size(),
            HCCL_ERROR("Channel to root index[%u] is missing", rootIndex), HCCL_E_NOT_FOUND);
        const ChannelInfo &rootChannel = resCtx.channels[rootChannelPosition];
        const ThreadHandle thread = resCtx.threads[rootChannelPosition];
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, SCATTER_READY_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, rootChannel.handle, SCATTER_DATA_NOTIFY, CUSTOM_TIMEOUT)));
        if (localSegment.validBytes != 0) {
            void *dst = OffsetAddr(param.outputPtr, localSegment.offsetBytes);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                thread, dst, resCtx.localBuffer.addr, localSegment.validBytes)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, SCATTER_ACK_NOTIFY)));
    }
    CHK_RET(EndParallel(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult DirectAllGatherRound(const OpParam &param, const AlgResourceCtx &resCtx,
    const RoundSegments &segments, uint32_t rootIndex)
{
    CHK_RET(BeginParallel(resCtx.threads));
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        const ChannelInfo &channel = resCtx.channels[i];
        const ThreadHandle thread = resCtx.threads[i];
        const SegmentDesc &segment = segments[channel.remoteRankIndex];
        CHK_RET(CheckChannelBuffer(channel, segment.validBytes));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, AG_READY_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, AG_READY_NOTIFY, CUSTOM_TIMEOUT)));
        if (resCtx.rankIndex == rootIndex) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, AG_ACK_NOTIFY)));
        } else {
            void *dst = OffsetAddr(param.outputPtr, segment.offsetBytes);
            CHK_RET(BatchReadAndNotify(thread, channel, dst, channel.remoteCclMem.addr, segment.validBytes,
                AG_ACK_NOTIFY));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, AG_ACK_NOTIFY, CUSTOM_TIMEOUT)));
    }
    CHK_RET(EndParallel(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult RunRootOneShot16(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes)
{
    uint32_t rootIndex = 0;
    CHK_RET(GetRootIndex(param, resCtx, rootIndex));
    CHK_PRT_RET(totalBytes == 0 || totalBytes > MinimumBufferSize(resCtx),
        HCCL_ERROR("OneShot16 message[%llu] exceeds minimum CCL buffer[%llu]",
            static_cast<unsigned long long>(totalBytes),
            static_cast<unsigned long long>(MinimumBufferSize(resCtx))),
        HCCL_E_MEMORY);

    CHK_RET(BeginParallel(resCtx.threads));
    if (resCtx.rankIndex == rootIndex) {
        for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
            const ChannelInfo &channel = resCtx.channels[i];
            const ThreadHandle thread = resCtx.threads[i];
            CHK_RET(CheckChannelBuffer(channel, totalBytes));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, SCATTER_READY_NOTIFY, CUSTOM_TIMEOUT)));
            CHK_RET(BatchWriteAndNotify(thread, channel, channel.remoteCclMem.addr, param.inputPtr, totalBytes,
                SCATTER_DATA_NOTIFY));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, SCATTER_ACK_NOTIFY, CUSTOM_TIMEOUT)));
        }
    } else {
        const uint32_t rootChannelPosition = FindChannelPosition(resCtx, rootIndex);
        CHK_PRT_RET(rootChannelPosition >= resCtx.channels.size(),
            HCCL_ERROR("OneShot16 channel to root index[%u] is missing", rootIndex), HCCL_E_NOT_FOUND);
        const ChannelInfo &rootChannel = resCtx.channels[rootChannelPosition];
        const ThreadHandle thread = resCtx.threads[rootChannelPosition];
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, SCATTER_READY_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, rootChannel.handle, SCATTER_DATA_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, param.outputPtr, resCtx.localBuffer.addr, totalBytes)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, SCATTER_ACK_NOTIFY)));
    }
    CHK_RET(EndParallel(resCtx.threads));
    HCCL_INFO("STOK Broadcast path Root OneShot16: bytes[%llu]", static_cast<unsigned long long>(totalBytes));
    return HCCL_SUCCESS;
}

HcclResult RunDirect16(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes)
{
    uint32_t rootIndex = 0;
    CHK_RET(GetRootIndex(param, resCtx, rootIndex));
    const RankSlices slices = BuildRankSlices(totalBytes);
    CHK_RET(ValidateRankSlices(slices, totalBytes));
    const uint64_t tileBytes = SelectTileBytes(resCtx, slices);
    CHK_PRT_RET(tileBytes < custom_broadcast::DATA_TYPE_SIZE,
        HCCL_ERROR("No feasible Direct-16 tile for minimum CCL buffer[%llu]",
            static_cast<unsigned long long>(MinimumBufferSize(resCtx))),
        HCCL_E_MEMORY);
    const uint64_t roundCount = GetRoundCount(slices, tileBytes);
    CHK_PRT_RET(roundCount == 0, HCCL_ERROR("Direct-16 selected zero rounds"), HCCL_E_INTERNAL);
    HCCL_INFO("STOK Broadcast path Direct-16: bytes[%llu], tile[%llu], rounds[%llu]",
        static_cast<unsigned long long>(totalBytes), static_cast<unsigned long long>(tileBytes),
        static_cast<unsigned long long>(roundCount));

    for (uint64_t round = 0; round < roundCount; ++round) {
        const RoundSegments segments = BuildRoundSegments(slices, round, tileBytes);
        CHK_RET(DirectScatterRound(param, resCtx, segments, rootIndex));
        CHK_RET(DirectAllGatherRound(param, resCtx, segments, rootIndex));
    }
    return HCCL_SUCCESS;
}

bool IsInSpan(uint32_t value, const std::array<uint32_t, 4> &masks, uint32_t maskCount)
{
    const uint32_t combinationCount = 1U << maskCount;
    for (uint32_t combination = 0; combination < combinationCount; ++combination) {
        uint32_t generated = 0;
        for (uint32_t i = 0; i < maskCount; ++i) {
            if ((combination & (1U << i)) != 0) {
                generated ^= masks[i];
            }
        }
        if (generated == value) {
            return true;
        }
    }
    return false;
}

bool CanUseCrossButterfly(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes)
{
    if (totalBytes != custom_broadcast::BUTTERFLY_BYTES ||
        resCtx.localBuffer.size < custom_broadcast::BUTTERFLY_BYTES) {
        return false;
    }
    constexpr std::array<uint32_t, 4> masks = {8, 9, 10, 12};
    for (uint32_t mask : masks) {
        const ChannelInfo *channel = FindChannel(resCtx, resCtx.rankIndex ^ mask);
        if (channel == nullptr || channel->netLayer != custom_broadcast::INTER_NET_LAYER ||
            channel->remoteCclMem.size < custom_broadcast::BUTTERFLY_BYTES) {
            return false;
        }
    }
    uint32_t rootIndex = 0;
    return GetRootIndex(param, resCtx, rootIndex) == HCCL_SUCCESS;
}

HcclResult RunCrossButterfly(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes)
{
    uint32_t rootIndex = 0;
    CHK_RET(GetRootIndex(param, resCtx, rootIndex));
    const ThreadHandle thread = resCtx.threads[0];
    if (resCtx.rankIndex == rootIndex) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, resCtx.localBuffer.addr, param.inputPtr, totalBytes)));
    }

    constexpr std::array<uint32_t, 4> masks = {8, 9, 10, 12};
    const uint32_t relativeRank = resCtx.rankIndex ^ rootIndex;
    for (uint32_t round = 0; round < masks.size(); ++round) {
        const uint32_t peerIndex = resCtx.rankIndex ^ masks[round];
        const ChannelInfo *channel = FindChannel(resCtx, peerIndex);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Butterfly channel to logical rank[%u] is missing", peerIndex), HCCL_E_NOT_FOUND);
        const bool isSender = IsInSpan(relativeRank, masks, round);
        const bool isReceiver = !isSender && IsInSpan(relativeRank ^ masks[round], masks, round);
        if (isSender) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel->handle, SCATTER_READY_NOTIFY, CUSTOM_TIMEOUT)));
            CHK_RET(BatchWriteAndNotify(thread, *channel, channel->remoteCclMem.addr, resCtx.localBuffer.addr,
                totalBytes, SCATTER_DATA_NOTIFY));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel->handle, SCATTER_ACK_NOTIFY, CUSTOM_TIMEOUT)));
        } else if (isReceiver) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel->handle, SCATTER_READY_NOTIFY)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel->handle, SCATTER_DATA_NOTIFY, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel->handle, SCATTER_ACK_NOTIFY)));
        }
    }

    if (resCtx.rankIndex != rootIndex) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, param.outputPtr, resCtx.localBuffer.addr, totalBytes)));
    }
    HCCL_INFO("STOK Broadcast path cross-Clos Butterfly: bytes[%llu]",
        static_cast<unsigned long long>(totalBytes));
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("ExecOp only supports FP32"), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / custom_broadcast::DATA_TYPE_SIZE,
        HCCL_ERROR("Element count overflows byte calculation"), HCCL_E_PARA);
    CHK_RET(ValidateResources(resCtx));

    const uint64_t totalBytes = param.count * custom_broadcast::DATA_TYPE_SIZE;
    if (totalBytes == custom_broadcast::DATA_TYPE_SIZE) {
        return RunRootOneShot16(param, resCtx, totalBytes);
    }
    if (CanUseCrossButterfly(param, resCtx, totalBytes)) {
        return RunCrossButterfly(param, resCtx, totalBytes);
    }
    return RunDirect16(param, resCtx, totalBytes);
}
} // namespace ops_hccl
