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
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
constexpr uint64_t UB_MAX_DATA_SIZE = 256UL * 1024UL * 1024UL;
constexpr uint64_t SMALL_DATA_SIZE = 512UL * 1024UL;
constexpr uint32_t SMALL_RECURSIVE_DOUBLING_RANK_SIZE = 16;
constexpr uint32_t NOTIFY_IDX_THREAD_START = 0;
constexpr uint32_t NOTIFY_IDX_FINISH_BASE = 1;

static uint32_t GetDataTypeSize(HcclDataType dataType)
{
    constexpr uint32_t DATA_TYPE_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t),
        sizeof(int32_t), 2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t),
        sizeof(uint32_t), 8, 2, 16, 2, 1, 1, 1, 1};
    if (dataType >= HCCL_DATA_TYPE_RESERVED) {
        return 0;
    }
    return DATA_TYPE_SIZE_TABLE[dataType];
}

static HcclResult LocalCopy(
    ThreadHandle thread, void *dstBase, void *srcBase, uint64_t dstOffset, uint64_t srcOffset, uint64_t size)
{
    void *dst = static_cast<void *>(static_cast<uint8_t *>(dstBase) + dstOffset);
    void *src = static_cast<void *>(static_cast<uint8_t *>(srcBase) + srcOffset);
    return static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, size));
}

static HcclResult StartChannelThreads(const AlgResourceCtx &resCtx, uint32_t channelCount)
{
    for (uint32_t i = 0; i < channelCount; i++) {
        ThreadHandle thread = resCtx.threads[i + 1];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[0], thread, NOTIFY_IDX_THREAD_START)));
    }
    for (uint32_t i = 0; i < channelCount; i++) {
        ThreadHandle thread = resCtx.threads[i + 1];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(thread, NOTIFY_IDX_THREAD_START, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

static HcclResult FinishChannelThreads(const AlgResourceCtx &resCtx, uint32_t channelCount)
{
    for (uint32_t i = 0; i < channelCount; i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[0], NOTIFY_IDX_FINISH_BASE + i, CUSTOM_TIMEOUT)));
    }
    for (uint32_t i = 0; i < channelCount; i++) {
        ThreadHandle thread = resCtx.threads[i + 1];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(thread, resCtx.threads[0], NOTIFY_IDX_FINISH_BASE + i)));
    }
    return HCCL_SUCCESS;
}

static HcclResult ReadRemoteSliceToOutput(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t channelIdx,
    uint64_t inputOffset, uint64_t rankDataSize, uint64_t sliceSize)
{
    const ChannelInfo &channel = resCtx.channels[channelIdx];
    ThreadHandle thread = resCtx.threads[channelIdx + 1];
    uint64_t remoteCclOffset = 0;
    uint64_t outputOffset = inputOffset + rankDataSize * channel.remoteRank;
    void *remoteAddr = static_cast<void *>(static_cast<uint8_t *>(channel.remoteCclMem.addr) + remoteCclOffset);
    void *outputAddr = static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + outputOffset);

    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, channel.handle, outputAddr, remoteAddr, sliceSize)));
    return HCCL_SUCCESS;
}

static const ChannelInfo *FindChannelByRemoteRank(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

static HcclResult ExecSmallRecursiveDoubling(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t rankDataSize)
{
    uint32_t channelCount = resCtx.channels.size();
    if (channelCount != param.rankSize - 1) {
        HCCL_ERROR("[ExecSmallRecursiveDoubling] channel resource is insufficient, channelCount[%u], rankSize[%u]",
            channelCount, param.rankSize);
        return HCCL_E_INTERNAL;
    }

    ThreadHandle mainThread = resCtx.threads[0];
    uint64_t localBlockOffset = rankDataSize * param.myRank;
    CHK_RET(LocalCopy(mainThread, resCtx.localBuffer.addr, param.inputPtr, localBlockOffset, 0, rankDataSize));

    for (uint32_t distance = 1; distance < param.rankSize; distance <<= 1) {
        uint32_t partnerRank = param.myRank ^ distance;
        const ChannelInfo *channel = FindChannelByRemoteRank(resCtx, partnerRank);
        if (channel == nullptr) {
            HCCL_ERROR("[ExecSmallRecursiveDoubling] channel is missing, rank[%u], partnerRank[%u]", param.myRank,
                partnerRank);
            return HCCL_E_INTERNAL;
        }

        uint32_t remoteBlockStartRank = partnerRank & ~(distance - 1U);
        uint64_t blockOffset = rankDataSize * remoteBlockStartRank;
        uint64_t blockSize = rankDataSize * distance;
        void *localAddr = static_cast<void *>(static_cast<uint8_t *>(resCtx.localBuffer.addr) + blockOffset);
        void *remoteAddr = static_cast<void *>(static_cast<uint8_t *>(channel->remoteCclMem.addr) + blockOffset);

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(mainThread, channel->handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(mainThread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommReadOnThread(mainThread, channel->handle, localAddr, remoteAddr, blockSize)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(mainThread, channel->handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(mainThread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }

    uint64_t totalDataSize = rankDataSize * param.rankSize;
    return LocalCopy(mainThread, param.outputPtr, resCtx.localBuffer.addr, 0, 0, totalDataSize);
}

static HcclResult ExecMeshLoop(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputOffset,
    uint64_t rankDataSize, uint64_t sliceSize)
{
    uint32_t channelCount = resCtx.channels.size();
    if (resCtx.threads.size() < channelCount + 1) {
        HCCL_ERROR("[ExecMeshLoop] thread resource is insufficient, threadCount[%llu], channelCount[%u]",
            static_cast<unsigned long long>(resCtx.threads.size()), channelCount);
        return HCCL_E_INTERNAL;
    }

    uint64_t localCclOffset = 0;
    CHK_RET(LocalCopy(resCtx.threads[0], resCtx.localBuffer.addr, param.inputPtr, localCclOffset, inputOffset,
        sliceSize));

    CHK_RET(StartChannelThreads(resCtx, channelCount));
    for (uint32_t channelIdx = 0; channelIdx < channelCount; channelIdx++) {
        CHK_RET(ReadRemoteSliceToOutput(param, resCtx, channelIdx, inputOffset, rankDataSize, sliceSize));
    }
    uint64_t localOutputOffset = inputOffset + rankDataSize * param.myRank;
    CHK_RET(LocalCopy(
        resCtx.threads[0], param.outputPtr, param.inputPtr, localOutputOffset, inputOffset, sliceSize));
    CHK_RET(FinishChannelThreads(resCtx, channelCount));
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (resCtx.threads.empty()) {
        HCCL_ERROR("[ExecOp] no AICPU thread resource");
        return HCCL_E_INTERNAL;
    }

    uint32_t dataTypeSize = GetDataTypeSize(param.dataType);
    if (dataTypeSize == 0) {
        HCCL_ERROR("[ExecOp] unsupported data type %d", static_cast<int>(param.dataType));
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    uint64_t rankDataSize = param.count * dataTypeSize;
    if (param.rankSize == 1) {
        CHK_RET(LocalCopy(resCtx.threads[0], param.outputPtr, param.inputPtr, 0, 0, rankDataSize));
        return HCCL_SUCCESS;
    }

    uint64_t cclBufferSize = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        cclBufferSize = std::min(cclBufferSize, channel.remoteCclMem.size);
    }
    uint64_t totalDataSize = rankDataSize * param.rankSize;
    if (param.rankSize == SMALL_RECURSIVE_DOUBLING_RANK_SIZE && rankDataSize <= SMALL_DATA_SIZE &&
        totalDataSize <= cclBufferSize) {
        return ExecSmallRecursiveDoubling(param, resCtx, rankDataSize);
    }

    uint64_t cclBuffBound = cclBufferSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    uint64_t maxDataSizePerLoop = std::min(UB_MAX_DATA_SIZE, cclBuffBound);
    uint64_t maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
    if (maxDataCountPerLoop == 0) {
        HCCL_ERROR(
            "[ExecOp] HCCL buffer is too small, cclBufferSize[%llu]", static_cast<unsigned long long>(cclBufferSize));
        return HCCL_E_INTERNAL;
    }

    uint64_t processedDataCount = 0;
    while (processedDataCount < param.count) {
        uint64_t sliceCount = std::min(maxDataCountPerLoop, param.count - processedDataCount);
        uint64_t sliceSize = sliceCount * dataTypeSize;
        uint64_t inputOffset = processedDataCount * dataTypeSize;
        CHK_RET(ExecMeshLoop(param, resCtx, inputOffset, rankDataSize, sliceSize));
        processedDataCount += sliceCount;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
