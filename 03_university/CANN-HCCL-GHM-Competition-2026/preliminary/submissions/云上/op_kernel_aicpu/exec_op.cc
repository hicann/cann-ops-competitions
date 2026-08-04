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
#include <limits>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
    constexpr uint64_t MAX_DATA_SIZE = 256ULL * 1024 * 1024;
    constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
    constexpr uint32_t NOTIFY_IDX_READY = 0;
    constexpr uint32_t NOTIFY_IDX_DATA = 1;
    constexpr uint32_t NOTIFY_IDX_REDUCE_ACK = 2;

    using ChannelMap = std::unordered_map<uint32_t, const ChannelInfo *>;

    bool IsSupportedReduceOp(HcclReduceOp op)
    {
        return op == HCCL_REDUCE_SUM || op == HCCL_REDUCE_MAX || op == HCCL_REDUCE_MIN;
    }

    HcclResult ValidateTensorShape(const OpParam &param, uint32_t dataTypeSize, uint64_t &blockBytes)
    {
        CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
            HCCL_ERROR("Invalid rank metadata, myRank[%u], rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
        CHK_PRT_RET(dataTypeSize == 0, HCCL_ERROR("Invalid data type size[0]"), HCCL_E_PARA);

        constexpr uint64_t maxValue = std::numeric_limits<uint64_t>::max();
        CHK_PRT_RET(param.count > maxValue / dataTypeSize,
            HCCL_ERROR("Block byte size overflow, count[%llu], dataTypeSize[%u]",
                static_cast<unsigned long long>(param.count), dataTypeSize),
            HCCL_E_PARA);
        blockBytes = param.count * dataTypeSize;
        CHK_PRT_RET(blockBytes != 0 && static_cast<uint64_t>(param.rankSize) > maxValue / blockBytes,
            HCCL_ERROR("Input byte size overflow, rankSize[%u], blockBytes[%llu]", param.rankSize,
                static_cast<unsigned long long>(blockBytes)),
            HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    HcclResult BuildChannelMap(
        const OpParam &param, const AlgResourceCtx &resCtx, ChannelMap &channelMap, uint64_t &globalMinBuffer)
    {
        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0,
            HCCL_ERROR("Invalid local HCCL buffer, addr[%p], size[%llu]", resCtx.localBuffer.addr,
                static_cast<unsigned long long>(resCtx.localBuffer.size)),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.channels.size() != static_cast<size_t>(param.rankSize - 1),
            HCCL_ERROR("Channel count mismatch, actual[%llu], expected[%u]",
                static_cast<unsigned long long>(resCtx.channels.size()), param.rankSize - 1),
            HCCL_E_INTERNAL);

        globalMinBuffer = resCtx.localBuffer.size;
        channelMap.reserve(resCtx.channels.size());
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank,
                HCCL_ERROR("Invalid channel remoteRank[%u], myRank[%u], rankSize[%u]", channel.remoteRank, param.myRank,
                    param.rankSize),
                HCCL_E_INTERNAL);
            CHK_PRT_RET(channel.notifyNum < CHANNEL_NOTIFY_NUM,
                HCCL_ERROR("Insufficient channel Notify, remoteRank[%u], actual[%u], required[%u]", channel.remoteRank,
                    channel.notifyNum, CHANNEL_NOTIFY_NUM),
                HCCL_E_INTERNAL);
            CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size == 0,
                HCCL_ERROR("Invalid remote HCCL buffer, remoteRank[%u], addr[%p], size[%llu]", channel.remoteRank,
                    channel.remoteCclMem.addr, static_cast<unsigned long long>(channel.remoteCclMem.size)),
                HCCL_E_INTERNAL);

            auto insertResult = channelMap.emplace(channel.remoteRank, &channel);
            CHK_PRT_RET(!insertResult.second, HCCL_ERROR("Duplicate channel for remoteRank[%u]", channel.remoteRank),
                HCCL_E_INTERNAL);
            globalMinBuffer = std::min(globalMinBuffer, channel.remoteCclMem.size);
        }
        CHK_PRT_RET(channelMap.size() != static_cast<size_t>(param.rankSize - 1),
            HCCL_ERROR("Peer channel map is incomplete, actual[%llu], expected[%u]",
                static_cast<unsigned long long>(channelMap.size()), param.rankSize - 1),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    HcclResult EnqueueSingleRankCopy(const OpParam &param, ThreadHandle thread, uint32_t dataTypeSize)
    {
        const uint64_t maxDataCountPerLoop = MAX_DATA_SIZE / dataTypeSize;
        CHK_PRT_RET(maxDataCountPerLoop == 0,
            HCCL_ERROR("MAX_DATA_SIZE cannot hold one element, dataTypeSize[%u]", dataTypeSize), HCCL_E_INTERNAL);

        uint64_t chunkOffset = 0;
        while (chunkOffset < param.count) {
            const uint64_t subCount = std::min(maxDataCountPerLoop, param.count - chunkOffset);
            const uint64_t chunkOffsetBytes = chunkOffset * dataTypeSize;
            const uint64_t subBytes = subCount * dataTypeSize;
            void *src = static_cast<uint8_t *>(param.inputPtr) + chunkOffsetBytes;
            void *dst = static_cast<uint8_t *>(param.outputPtr) + chunkOffsetBytes;
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, subBytes)));
            chunkOffset += subCount;
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank metadata, myRank[%u], rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    const auto dataTypeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(dataTypeIt == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type[%d]", static_cast<int32_t>(param.dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(!IsSupportedReduceOp(param.reduceType),
        HCCL_ERROR("Unsupported reduce operation[%d]", static_cast<int32_t>(param.reduceType)), HCCL_E_NOT_SUPPORT);
    const uint32_t dataTypeSize = dataTypeIt->second;
    uint64_t blockBytes = 0;
    CHK_RET(ValidateTensorShape(param, dataTypeSize, blockBytes));

    const ThreadHandle thread = resCtx.aicpuThread;
    if (param.rankSize == 1) {
        return EnqueueSingleRankCopy(param, thread, dataTypeSize);
    }

    ChannelMap channelMap;
    uint64_t globalMinBuffer = 0;
    CHK_RET(BuildChannelMap(param, resCtx, channelMap, globalMinBuffer));

    const uint64_t maxDataCountPerLoop = std::min(MAX_DATA_SIZE / dataTypeSize, globalMinBuffer / 3 / dataTypeSize);
    CHK_PRT_RET(maxDataCountPerLoop == 0,
        HCCL_ERROR("CCL buffer cannot hold accumulator, receive staging and send staging elements, "
                   "minBufferSize[%llu], "
                   "dataTypeSize[%u]",
            static_cast<unsigned long long>(globalMinBuffer), dataTypeSize),
        HCCL_E_INTERNAL);

    const uint64_t slotBytes = maxDataCountPerLoop * dataTypeSize;
    void *accumulatorAddr = resCtx.localBuffer.addr;
    void *stagingAddr = static_cast<uint8_t *>(resCtx.localBuffer.addr) + slotBytes;
    void *sendStagingAddr = static_cast<uint8_t *>(resCtx.localBuffer.addr) + 2 * slotBytes;
    const HcommDataType hcommDataType = static_cast<HcommDataType>(param.dataType);
    const HcommReduceOp hcommReduceOp = static_cast<HcommReduceOp>(param.reduceType);

    uint64_t chunkOffset = 0;
    while (chunkOffset < param.count) {
        const uint64_t subCount = std::min(maxDataCountPerLoop, param.count - chunkOffset);
        const uint64_t chunkOffsetBytes = chunkOffset * dataTypeSize;
        const uint64_t subBytes = subCount * dataTypeSize;

        const uint64_t selfInputOffset = static_cast<uint64_t>(param.myRank) * blockBytes + chunkOffsetBytes;
        void *selfInputAddr = static_cast<uint8_t *>(param.inputPtr) + selfInputOffset;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, accumulatorAddr, selfInputAddr, subBytes)));

        for (uint32_t step = 1; step < param.rankSize; ++step) {
            const uint32_t outgoingRank
                = static_cast<uint32_t>((static_cast<uint64_t>(param.myRank) + step) % param.rankSize);
            const uint32_t incomingRank
                = static_cast<uint32_t>((static_cast<uint64_t>(param.myRank) + param.rankSize - step) % param.rankSize);
            const ChannelInfo &outgoingChannel = *channelMap.find(outgoingRank)->second;
            const ChannelInfo &incomingChannel = *channelMap.find(incomingRank)->second;

            const uint64_t outgoingInputOffset = static_cast<uint64_t>(outgoingRank) * blockBytes + chunkOffsetBytes;
            void *outgoingInputAddr = static_cast<uint8_t *>(param.inputPtr) + outgoingInputOffset;
            void *remoteStagingAddr = static_cast<uint8_t *>(outgoingChannel.remoteCclMem.addr) + slotBytes;

            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, incomingChannel.handle, NOTIFY_IDX_READY)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, outgoingChannel.handle, NOTIFY_IDX_READY, CUSTOM_TIMEOUT)));
            CHK_RET(
                static_cast<HcclResult>(HcommLocalCopyOnThread(thread, sendStagingAddr, outgoingInputAddr, subBytes)));
            CHK_RET(static_cast<HcclResult>(
                HcommWriteOnThread(thread, outgoingChannel.handle, remoteStagingAddr, sendStagingAddr, subBytes)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, outgoingChannel.handle, NOTIFY_IDX_DATA)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, incomingChannel.handle, NOTIFY_IDX_DATA, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                thread, accumulatorAddr, stagingAddr, subCount, hcommDataType, hcommReduceOp)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, incomingChannel.handle, NOTIFY_IDX_REDUCE_ACK)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, outgoingChannel.handle, NOTIFY_IDX_REDUCE_ACK, CUSTOM_TIMEOUT)));
        }

        void *outputAddr = static_cast<uint8_t *>(param.outputPtr) + chunkOffsetBytes;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, outputAddr, accumulatorAddr, subBytes)));
        chunkOffset += subCount;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
