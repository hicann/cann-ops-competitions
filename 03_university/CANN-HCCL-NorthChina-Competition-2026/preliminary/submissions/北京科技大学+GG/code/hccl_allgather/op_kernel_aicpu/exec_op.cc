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

namespace {
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
constexpr uint64_t MAX_DATA_SIZE_PER_LOOP = 256ULL * 1024ULL * 1024ULL;
constexpr uint32_t SERVER_RANK_NUM = 8;
constexpr uint32_t SMALL_THREAD_NUM = 1;
constexpr uint32_t SMALL_CHANNEL_NUM = 4;
constexpr uint64_t SMALL_HIERARCHICAL_THRESHOLD = 512ULL * 1024;

bool UseHierarchicalSmall(uint32_t rankSize, uint64_t dataSize)
{
    return rankSize == 2 * SERVER_RANK_NUM && dataSize <= SMALL_HIERARCHICAL_THRESHOLD;
}

HcclResult ThreadSyncBefore(const ContextVector<ThreadHandle, MAX_CUSTOM_THREAD_NUM> &threads)
{
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[idx], 0)));
    }
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[idx], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadSyncAfter(const ContextVector<ThreadHandle, MAX_CUSTOM_THREAD_NUM> &threads)
{
    // 主线程的 notify 0 留给 Host/AICPU 启动同步，从 1 开始接收从线程完成信号。
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], idx, CUSTOM_TIMEOUT)));
    }
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[idx], threads[0], idx)));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadSyncBeforeLast(
    const ContextVector<ThreadHandle, MAX_CUSTOM_THREAD_NUM> &threads)
{
    const uint32_t masterIdx = static_cast<uint32_t>(threads.size() - 1);
    ThreadHandle master = threads[masterIdx];
    for (uint32_t idx = 0; idx < masterIdx; ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(master, threads[idx], 0)));
    }
    for (uint32_t idx = 0; idx < masterIdx; ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[idx], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadSyncAfterLast(
    const ContextVector<ThreadHandle, MAX_CUSTOM_THREAD_NUM> &threads)
{
    const uint32_t masterIdx = static_cast<uint32_t>(threads.size() - 1);
    ThreadHandle master = threads[masterIdx];
    // Notify 0 on the master is reserved for Host/AICPU synchronization.
    for (uint32_t idx = 0; idx < masterIdx; ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(master, idx + 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t idx = 0; idx < masterIdx; ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[idx], master, idx + 1)));
    }
    return HCCL_SUCCESS;
}

void *OutputSlot(void *output, uint32_t rank, uint64_t dataSize)
{
    return static_cast<void *>(
        static_cast<uint8_t *>(output) + static_cast<uint64_t>(rank) * dataSize);
}

HcclResult ExecHierarchicalSmall(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    if (resCtx.threads.size() != SMALL_THREAD_NUM || resCtx.channels.size() != SMALL_CHANNEL_NUM ||
        resCtx.remoteOutputs.size() != resCtx.channels.size()) {
        HCCL_ERROR("Hierarchical resource mismatch: threads=%zu, channels=%zu, outputs=%zu",
            resCtx.threads.size(), resCtx.channels.size(), resCtx.remoteOutputs.size());
        return HCCL_E_INTERNAL;
    }

    ThreadHandle master = resCtx.threads[0];
    const ChannelInfo &firstChannel = resCtx.channels[0];
    CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
        master, firstChannel.handle,
        OutputSlot(resCtx.remoteOutputs[0], param.myRank, dataSize),
        param.inputPtr, dataSize, NOTIFY_IDX_DATA_SIGNAL)));

    // The first Mesh write reads the original input buffer, so the independent local copy can
    // be issued before waiting for the peer. By the wait completion, both slots of the rank pair
    // are ready for the remaining recursive-doubling rounds.
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        master, OutputSlot(param.outputPtr, param.myRank, dataSize),
        param.inputPtr, dataSize)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        master, firstChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    // Grow the completed pair into 4/8/16-rank contiguous output blocks. xor-2/xor-4 stay on
    // Mesh; xor-8 sends the final eight-rank server block through the wider Clos layer.
    for (uint32_t stage = 1; stage < SMALL_CHANNEL_NUM; ++stage) {
        const uint32_t offset = 1U << stage;
        const uint32_t blockStart = param.myRank & ~(offset - 1U);
        const uint64_t blockSize = static_cast<uint64_t>(offset) * dataSize;
        const ChannelInfo &channel = resCtx.channels[stage];
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
            master, channel.handle,
            OutputSlot(resCtx.remoteOutputs[stage], blockStart, dataSize),
            OutputSlot(param.outputPtr, blockStart, dataSize), blockSize,
            NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            master, channel.handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecRegisteredOutputPushLarge(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    if (resCtx.threads.size() != param.rankSize ||
        resCtx.channels.size() != param.rankSize - 1 ||
        resCtx.remoteOutputs.size() != resCtx.channels.size()) {
        HCCL_ERROR("Registered-output resource mismatch: rankSize=%u, threads=%zu, channels=%zu, outputs=%zu",
            param.rankSize, resCtx.threads.size(), resCtx.channels.size(), resCtx.remoteOutputs.size());
        return HCCL_E_INTERNAL;
    }

    ThreadHandle master = resCtx.threads[resCtx.threads.size() - 1];
    const uint64_t ownOutputOffset = static_cast<uint64_t>(param.myRank) * dataSize;
    void *ownOutput = static_cast<void *>(
        static_cast<uint8_t *>(param.outputPtr) + ownOutputOffset);

    CHK_RET(ThreadSyncBeforeLast(resCtx.threads));
    // All workers are now reachable in the VM dependency graph. The dedicated master performs
    // the local copy while all fifteen workers push this rank's slice to their peers.
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        master, ownOutput, param.inputPtr, dataSize)));

    for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
        const ChannelInfo &channel = resCtx.channels[idx];
        ThreadHandle thread = resCtx.threads[idx];
        void *remoteOwnOutput = static_cast<void *>(static_cast<uint8_t *>(resCtx.remoteOutputs[idx]) +
            ownOutputOffset);

        // The fused primitive guarantees that the peer's DATA notification becomes visible only
        // after this write has completed. Every peer writes a disjoint final-output slot.
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
            thread, channel.handle, remoteOwnOutput, param.inputPtr, dataSize,
            NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }

    CHK_RET(ThreadSyncAfterLast(resCtx.threads));
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.dataType != HCCL_DATA_TYPE_FP32) {
        HCCL_ERROR("Unsupported data type: %d", static_cast<int32_t>(param.dataType));
        return HCCL_E_NOT_SUPPORT;
    }

    if (resCtx.threads.empty()) {
        HCCL_ERROR("No communication thread is available");
        return HCCL_E_INTERNAL;
    }

    const uint64_t dataTypeSize = sizeof(float);
    const uint64_t dataSize = param.count * dataTypeSize;
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize)));
        return HCCL_SUCCESS;
    }

    if (UseHierarchicalSmall(param.rankSize, dataSize)) {
        return ExecHierarchicalSmall(param, resCtx, dataSize);
    }

    if (param.rankSize == 2 * SERVER_RANK_NUM) {
        return ExecRegisteredOutputPushLarge(param, resCtx, dataSize);
    }

    if (resCtx.threads.size() != param.rankSize - 1 || resCtx.channels.size() != param.rankSize - 1) {
        HCCL_ERROR("Resource count mismatch: rankSize=%u, threads=%zu, channels=%zu", param.rankSize,
            resCtx.threads.size(), resCtx.channels.size());
        return HCCL_E_INTERNAL;
    }

    uint64_t usableBufferSize = resCtx.localBuffer.size;
    for (size_t idx = 0; idx < resCtx.channels.size(); ++idx) {
        const ChannelInfo channel = resCtx.channels[idx];
        usableBufferSize = std::min(usableBufferSize, channel.remoteCclMem.size);
    }
    // Pull/read 模式下，每个 rank 只写自己的 CCL Buffer，所有对端只读这同一份数据。
    // 因此无需按 rankSize 预留槽位，可以把整块 Buffer 用作当前轮的源数据区。
    const uint64_t cclBufferBound =
        usableBufferSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    const uint64_t maxDataSizePerLoop = std::min(MAX_DATA_SIZE_PER_LOOP, cclBufferBound);
    if (maxDataSizePerLoop < dataTypeSize) {
        HCCL_ERROR("HCCL buffer is too small: %llu bytes", static_cast<unsigned long long>(usableBufferSize));
        return HCCL_E_INTERNAL;
    }

    const uint64_t maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
    const uint64_t loopCount =
        param.count / maxDataCountPerLoop + static_cast<uint64_t>(param.count % maxDataCountPerLoop != 0);
    uint64_t processedDataCount = 0;

    for (uint64_t loop = 0; loop < loopCount; ++loop) {
        const uint64_t sliceCount = std::min(maxDataCountPerLoop, param.count - processedDataCount);
        const uint64_t sliceSize = sliceCount * dataTypeSize;
        const uint64_t inputOffset = processedDataCount * dataTypeSize;
        void *localSlot = resCtx.localBuffer.addr;
        void *inputSlice = static_cast<void *>(static_cast<uint8_t *>(param.inputPtr) + inputOffset);

        const uint64_t ownOutputOffset = param.myRank * dataSize + inputOffset;
        void *ownOutputSlice = static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + ownOutputOffset);

        // 每轮把本 rank 的输入放入固定通信内存，并直接放到本 rank 的最终输出位置。
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], localSlot, inputSlice, sliceSize)));
        CHK_RET(ThreadSyncBefore(resCtx.threads));
        // The workers can start their channels while the master copies the disjoint own-output slice.
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], ownOutputSlice, inputSlice, sliceSize)));

        for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
            const ChannelInfo &channel = resCtx.channels[idx];
            ThreadHandle thread = resCtx.threads[idx];
            void *remoteSlot = channel.remoteCclMem.addr;
            const uint64_t remoteOutputOffset = channel.remoteRank * dataSize + inputOffset;
            void *remoteOutputSlice =
                static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + remoteOutputOffset);

            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(
                HcommReadOnThread(thread, channel.handle, remoteOutputSlice, remoteSlot, sliceSize)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }

        CHK_RET(ThreadSyncAfter(resCtx.threads));

        processedDataCount += sliceCount;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
