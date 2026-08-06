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

#include <hcomm/hcomm_primitives_expt.h>

#include <algorithm>
#include <limits>

namespace ops_hccl {
namespace {
constexpr uint64_t SMALL_MESSAGE_MAX_BYTES = 512ULL * 1024;

int32_t ReadAndSignal(
    ThreadHandle thread, const ChannelInfo &channel, void *dst, const void *src, uint64_t len)
{
    HcommBatchTransferDesc transfers[2]{};
    transfers[0].transType = HCOMM_TRANSFER_TYPE_READ;
    transfers[0].transferInfo.read.len = len;
    transfers[0].transferInfo.read.dst = dst;
    transfers[0].transferInfo.read.src = const_cast<void *>(src);
    transfers[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    transfers[1].transferInfo.notifyRecord.notifyIdx = NOTIFY_IDX_DATA_SIGNAL;
    return HcommBatchTransferOnThread(thread, channel.handle, transfers, 2);
}

HcclResult ExecSmallAllGather(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputSize)
{
    ThreadHandle mainThread = resCtx.threads[0];
    char *input = static_cast<char *>(param.inputPtr);
    char *output = static_cast<char *>(param.outputPtr);
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);

    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, output, input, inputSize));
    }

    CHK_RET(HcommLocalCopyOnThread(
        mainThread, localBuffer + static_cast<uint64_t>(param.myRank) * inputSize, input, inputSize));

    for (uint64_t mask = 1; mask < param.rankSize; mask <<= 1) {
        const uint32_t partner = param.myRank ^ static_cast<uint32_t>(mask);
        const auto channelIt = std::find_if(resCtx.channels.begin(), resCtx.channels.end(),
            [partner](const ChannelInfo &channel) { return channel.remoteRank == partner; });
        CHK_PRT_RET(channelIt == resCtx.channels.end(), HCCL_ERROR("No channel to partner rank[%u]", partner),
            HCCL_E_INTERNAL);

        const uint32_t blockStart = partner & ~(static_cast<uint32_t>(mask) - 1U);
        const uint64_t blockOffset = static_cast<uint64_t>(blockStart) * inputSize;
        const uint64_t blockSize = mask * inputSize;
        const bool finalStep = (mask << 1) == param.rankSize;

        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channelIt->handle, NOTIFY_IDX_ACK));
        if (finalStep) {
            const uint32_t localBlockStart = param.myRank & ~(static_cast<uint32_t>(mask) - 1U);
            const uint64_t localBlockOffset = static_cast<uint64_t>(localBlockStart) * inputSize;
            CHK_RET(HcommLocalCopyOnThread(mainThread, output + localBlockOffset,
                localBuffer + localBlockOffset, blockSize));
        }
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, channelIt->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        CHK_RET(ReadAndSignal(mainThread, *channelIt,
            (finalStep ? output : localBuffer) + blockOffset,
            static_cast<char *>(channelIt->remoteCclMem.addr) + blockOffset, blockSize));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            mainThread, channelIt->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    }

    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    const auto dataTypeSize = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(dataTypeSize == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", param.dataType), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize->second,
        HCCL_ERROR("Input size overflow, count[%llu]", static_cast<unsigned long long>(param.count)), HCCL_E_PARA);

    const uint64_t inputSize = param.count * dataTypeSize->second;
    if (inputSize == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    CHK_PRT_RET(param.myRank >= param.rankSize, HCCL_ERROR("Invalid rank[%u/%u]", param.myRank, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No communication thread"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.channels.size() + 1 != param.rankSize,
        HCCL_ERROR("Channel count[%zu] does not match rank size[%u]", resCtx.channels.size(), param.rankSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.threads.size() < resCtx.channels.size() + 1,
        HCCL_ERROR("Insufficient communication threads[%zu]", resCtx.threads.size()), HCCL_E_INTERNAL);

    uint64_t chunkCapacity = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank,
            HCCL_ERROR("Invalid remote rank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.notifyNum < 2, HCCL_ERROR("Insufficient channel notify count[%u]", channel.notifyNum),
            HCCL_E_INTERNAL);
        chunkCapacity = std::min(chunkCapacity, channel.remoteCclMem.size);
    }
    CHK_PRT_RET(chunkCapacity == 0, HCCL_ERROR("HCCL buffer has zero capacity"), HCCL_E_INTERNAL);

    ThreadHandle mainThread = resCtx.threads[0];
    char *input = static_cast<char *>(param.inputPtr);
    char *output = static_cast<char *>(param.outputPtr);
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);

    const bool powerOfTwoRanks = (param.rankSize & (param.rankSize - 1U)) == 0;
    if (inputSize <= SMALL_MESSAGE_MAX_BYTES && powerOfTwoRanks &&
        inputSize <= chunkCapacity / param.rankSize) {
        return ExecSmallAllGather(param, resCtx, inputSize);
    }

    for (uint64_t offset = 0; offset < inputSize;) {
        const uint64_t chunkSize = std::min(chunkCapacity, inputSize - offset);

        // 固定地址的 HCCL Buffer 是所有远端 rank 本轮读取的数据源。
        CHK_RET(HcommLocalCopyOnThread(mainThread, localBuffer, input + offset, chunkSize));

        // 启动所有通信 Thread，并为 Checker 建立从主 Thread 到工作 Thread 的任务依赖。
        for (size_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                mainThread, resCtx.threads[channelIdx + 1], 0));
        }
        CHK_RET(HcommLocalCopyOnThread(mainThread,
            output + static_cast<uint64_t>(param.myRank) * inputSize + offset, input + offset, chunkSize));

        for (size_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
            const ChannelInfo &channel = resCtx.channels[channelIdx];
            ThreadHandle worker = resCtx.threads[channelIdx + 1];
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));

            // 双方都完成本地 HCCL Buffer 填充后再开始读取。
            CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel.handle, NOTIFY_IDX_ACK));
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            CHK_RET(ReadAndSignal(worker, channel,
                output + static_cast<uint64_t>(channel.remoteRank) * inputSize + offset,
                static_cast<char *>(channel.remoteCclMem.addr), chunkSize));

            // 确认双方均已读完，主 Thread 才能复用 HCCL Buffer 处理下一分块。
            CHK_RET(HcommChannelNotifyWaitOnThread(
                worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
            CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, channelIdx + 1));
        }

        for (size_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, channelIdx + 1, CUSTOM_TIMEOUT));
        }
        offset += chunkSize;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
