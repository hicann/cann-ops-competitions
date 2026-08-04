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
#include <limits>
#include <vector>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_DATA_SIZE = 256ULL * 1024 * 1024;
constexpr uint64_t SMALL_NHR_MAX_OUTPUT_BYTES = 512ULL * 1024;

bool IsPowerOfTwo(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

HcclResult CopyBytesOnThread(ThreadHandle thread, uint8_t *dst, uint8_t *src, uint64_t bytes)
{
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t subBytes = std::min(MAX_DATA_SIZE, bytes - offset);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst + offset, src + offset, subBytes)));
        offset += subBytes;
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateSmallNhrWorkspace(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelByRank, uint64_t totalBytes)
{
    // The first round reads directly from the user input. The HCCL workspace only
    // holds the surviving half plus disjoint receive areas for later rounds:
    // rankSize / 2 + rankSize / 4 + ... + 1 == rankSize - 1 blocks.
    const uint64_t workspaceBlocks = static_cast<uint64_t>(param.rankSize) - 1;
    CHK_PRT_RET(totalBytes > std::numeric_limits<uint64_t>::max() / workspaceBlocks,
        HCCL_ERROR("ExecOp: NHR workspace byte size overflows"), HCCL_E_INTERNAL);
    const uint64_t workspaceBytes = workspaceBlocks * totalBytes;
    CHK_PRT_RET(workspaceBytes > resCtx.localBuffer.size,
        HCCL_ERROR("ExecOp: local HCCL buffer[%llu] is smaller than NHR workspace[%llu]",
            static_cast<unsigned long long>(resCtx.localBuffer.size),
            static_cast<unsigned long long>(workspaceBytes)),
        HCCL_E_INTERNAL);

    for (uint32_t mask = param.rankSize >> 1; mask != 0; mask >>= 1) {
        const uint32_t peer = param.myRank ^ mask;
        const ChannelInfo *channel = channelByRank[peer];
        CHK_PRT_RET(channel == nullptr || channel->remoteCclMem.addr == nullptr ||
                channel->remoteCclMem.size < workspaceBytes,
            HCCL_ERROR("ExecOp: remote HCCL buffer of rank[%u] cannot hold NHR workspace[%llu]", peer,
                static_cast<unsigned long long>(workspaceBytes)),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult ExecSmallNhr(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelByRank, uint64_t totalBytes,
    HcommDataType hcommDataType, HcommReduceOp hcommReduceOp)
{
    ThreadHandle thread = resCtx.aicpuThread;
    auto *input = static_cast<uint8_t *>(param.inputPtr);
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    auto *work = static_cast<uint8_t *>(resCtx.localBuffer.addr);

    // First round: exchange directly from the user input and form the surviving
    // half in the local HCCL workspace. This removes a rankSize * totalBytes copy.
    const uint32_t firstMask = param.rankSize >> 1;
    const uint32_t firstPeer = param.myRank ^ firstMask;
    const uint32_t firstHalfBlocks = param.rankSize >> 1;
    const bool firstKeepLower = (param.myRank & firstMask) == 0;
    const uint32_t firstKeepStart = firstKeepLower ? 0 : firstHalfBlocks;
    const uint32_t firstSendStart = firstKeepLower ? firstHalfBlocks : 0;
    const uint64_t firstExchangeBytes = static_cast<uint64_t>(firstHalfBlocks) * totalBytes;
    const uint64_t firstExchangeCount = static_cast<uint64_t>(firstHalfBlocks) * param.count;
    const ChannelInfo &firstChannel = *channelByRank[firstPeer];

    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, firstChannel.handle,
        firstChannel.remoteCclMem.addr, input + static_cast<uint64_t>(firstSendStart) * totalBytes,
        firstExchangeBytes)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, firstChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, firstChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, work,
        input + static_cast<uint64_t>(firstKeepStart) * totalBytes, firstExchangeCount,
        hcommDataType, hcommReduceOp)));

    uint32_t activeOffset = 0;
    uint32_t activeBlocks = firstHalfBlocks;
    uint64_t scratchOffset = firstExchangeBytes;
    for (uint32_t mask = firstMask >> 1; mask != 0; mask >>= 1) {
        const uint32_t peer = param.myRank ^ mask;
        const ChannelInfo &channel = *channelByRank[peer];
        const uint32_t halfBlocks = activeBlocks >> 1;
        const bool keepLowerHalf = (param.myRank & mask) == 0;
        const uint32_t keepOffset = keepLowerHalf ? activeOffset : activeOffset + halfBlocks;
        const uint32_t sendOffset = keepLowerHalf ? activeOffset + halfBlocks : activeOffset;
        const uint64_t exchangeBytes = static_cast<uint64_t>(halfBlocks) * totalBytes;
        const uint64_t exchangeCount = static_cast<uint64_t>(halfBlocks) * param.count;

        auto *localScratch = work + scratchOffset;
        auto *remoteScratch = static_cast<uint8_t *>(channel.remoteCclMem.addr) + scratchOffset;
        auto *sendAddr = work + static_cast<uint64_t>(sendOffset) * totalBytes;
        auto *keepAddr = work + static_cast<uint64_t>(keepOffset) * totalBytes;

        // The four NHR rounds use different peers and disjoint receive areas. This
        // removes the pre/post ACK pair needed by the single-slot fallback path.
        CHK_RET(static_cast<HcclResult>(
            HcommWriteOnThread(thread, channel.handle, remoteScratch, sendAddr, exchangeBytes)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            thread, keepAddr, localScratch, exchangeCount, hcommDataType, hcommReduceOp)));

        scratchOffset += exchangeBytes;
        activeOffset = keepOffset;
        activeBlocks = halfBlocks;
    }

    CHK_RET(CopyBytesOnThread(thread, output,
        work + static_cast<uint64_t>(activeOffset) * totalBytes, totalBytes));
    return HCCL_SUCCESS;
}

HcclResult ExecDirectFallback(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelByRank, uint64_t totalBytes,
    HcommDataType hcommDataType, HcommReduceOp hcommReduceOp)
{
    ThreadHandle thread = resCtx.aicpuThread;
    auto *input = static_cast<uint8_t *>(param.inputPtr);
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    auto *work = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const uint64_t selfBlockOffset = static_cast<uint64_t>(param.myRank) * totalBytes;

    uint64_t maxChunkBytes = std::min(MAX_DATA_SIZE, resCtx.localBuffer.size);
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        maxChunkBytes = std::min(maxChunkBytes, channel.remoteCclMem.size);
    }
    const uint64_t maxChunkCount = maxChunkBytes / sizeof(float);
    CHK_PRT_RET(maxChunkCount == 0,
        HCCL_ERROR("ExecOp: HCCL buffer cannot hold one element"), HCCL_E_INTERNAL);

    // Process one output slice at a time. The local HCCL buffer starts with the
    // self contribution, then every XOR peer performs an ordered remote reduce
    // into that buffer. This fuses network transfer and reduction while keeping
    // exactly the same deterministic accumulation order as the direct path.
    for (uint64_t countOffset = 0; countOffset < param.count;) {
        const uint64_t subCount = std::min(maxChunkCount, param.count - countOffset);
        const uint64_t subBytes = subCount * sizeof(float);
        const uint64_t byteOffset = countOffset * sizeof(float);
        CHK_RET(CopyBytesOnThread(
            thread, work, input + selfBlockOffset + byteOffset, subBytes));

        for (uint32_t roundId = 1; roundId < param.rankSize; ++roundId) {
            const uint32_t peer = param.myRank ^ roundId;
            const ChannelInfo &channel = *channelByRank[peer];
            const uint64_t sendBlockOffset = static_cast<uint64_t>(peer) * totalBytes;
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                thread, channel.handle, NOTIFY_IDX_ACK)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

            CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, channel.handle,
                channel.remoteCclMem.addr, input + sendBlockOffset + byteOffset, subCount,
                hcommDataType, hcommReduceOp)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }

        CHK_RET(CopyBytesOnThread(thread, output + byteOffset, work, subBytes));
        countOffset += subCount;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(param.rankSize <= 1 || !IsPowerOfTwo(param.rankSize),
        HCCL_ERROR("ExecOp: rank size[%u] must be a power of two greater than one", param.rankSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(param.myRank >= param.rankSize,
        HCCL_ERROR("ExecOp: rank id[%u] is outside rank size[%u]", param.myRank, param.rankSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("ExecOp: unsupported data type[%d] or reduce op[%d]", param.dataType, param.reduceType),
        HCCL_E_INTERNAL);

    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(),
        HCCL_ERROR("ExecOp: data type[%d] has no size mapping", param.dataType), HCCL_E_INTERNAL);
    const uint64_t dataTypeSize = sizeIt->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("ExecOp: element count[%llu] overflows byte size",
            static_cast<unsigned long long>(param.count)),
        HCCL_E_INTERNAL);

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    const uint64_t totalBytes = param.count * dataTypeSize;
    CHK_PRT_RET(totalBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("ExecOp: rank input byte size overflows"), HCCL_E_INTERNAL);

    std::vector<const ChannelInfo *> channelByRank(param.rankSize, nullptr);
    uint32_t channelCount = 0;
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank ||
                channelByRank[channel.remoteRank] != nullptr,
            HCCL_ERROR("ExecOp: invalid or duplicate channel for rank[%u]", channel.remoteRank),
            HCCL_E_INTERNAL);
        channelByRank[channel.remoteRank] = &channel;
        ++channelCount;
    }
    CHK_PRT_RET(channelCount != param.rankSize - 1,
        HCCL_ERROR("ExecOp: expected[%u] peer channels, got[%u]", param.rankSize - 1, channelCount),
        HCCL_E_INTERNAL);

    const HcommDataType hcommDataType = static_cast<HcommDataType>(param.dataType);
    const HcommReduceOp hcommReduceOp = static_cast<HcommReduceOp>(param.reduceType);

    // All ranks select the path from collective parameters only. Avoid using local
    // resource sizes for dispatch, which could make peers enter different algorithms.
    if (totalBytes <= SMALL_NHR_MAX_OUTPUT_BYTES) {
        CHK_RET(ValidateSmallNhrWorkspace(param, resCtx, channelByRank, totalBytes));
        HCCL_INFO("ExecOp: use 4-round small-message NHR path, bytes[%llu]",
            static_cast<unsigned long long>(totalBytes));
        return ExecSmallNhr(param, resCtx, channelByRank, totalBytes, hcommDataType, hcommReduceOp);
    }

    HCCL_INFO("ExecOp: use deterministic direct fallback, bytes[%llu]",
        static_cast<unsigned long long>(totalBytes));
    return ExecDirectFallback(param, resCtx, channelByRank, totalBytes, hcommDataType, hcommReduceOp);
}
} // namespace ops_hccl
