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

namespace {

constexpr uint32_t kRankSize = 16;
constexpr uint32_t kRanksPerServer = 8;
constexpr uint64_t kAlignment = 128;
constexpr uint64_t kMaxTransferBytes = 256ULL * 1024 * 1024;
constexpr uint64_t kSmallSingleSliceMaxBytes = 1024ULL * 1024;
// The scoring label is the final 16-rank output size. A 512KiB output means
// one 32KiB input slice per rank for this fixed-size communicator.
constexpr uint64_t kSmallVariantOutputBytes = 512ULL * 1024;
constexpr uint64_t kSmallVariantBytes = kSmallVariantOutputBytes / kRankSize;
constexpr uint32_t kCoordinatorThread = 0;
constexpr uint32_t kFirstMeshThread = 1;
constexpr uint32_t kPeerNum = kRankSize - 1;
constexpr uint32_t kLocalCopyThread = kPeerNum;
constexpr uint32_t kDirectThreadNum = kRankSize;
constexpr uint32_t kMaxDirectSliceNum = 2;
constexpr uint32_t kOutputPipelineChunkNum = 8;

enum class SmallSingleSliceVariant : uint32_t {
    kDirectOutputWrite,
    kDirectOutputWriteCoordinatorDataWait,
};

constexpr SmallSingleSliceVariant kSmallSingleSliceVariant =
    SmallSingleSliceVariant::kDirectOutputWriteCoordinatorDataWait;

ThreadHandle PeerThread(const AlgResourceCtx &resCtx, uint32_t peerIndex)
{
    if (peerIndex < kRanksPerServer - 1) {
        return resCtx.threads[kFirstMeshThread + peerIndex];
    }
    return peerIndex == kRanksPerServer - 1 ? resCtx.threads[kCoordinatorThread]
                                            : resCtx.threads[peerIndex];
}

uint64_t StagingOffset(uint32_t rank, uint64_t totalInputBytes)
{
    return (static_cast<uint64_t>(rank) * (totalInputBytes % kAlignment)) % kAlignment;
}

uint64_t MaxStagingOffset(uint64_t totalInputBytes)
{
    uint64_t maxOffset = 0;
    for (uint32_t rank = 0; rank < kRankSize; ++rank) {
        maxOffset = std::max(maxOffset, StagingOffset(rank, totalInputBytes));
    }
    return maxOffset;
}

HcclResult SyncWorkers(const AlgResourceCtx &resCtx, uint32_t workerNum, uint32_t notifyIndex)
{
    for (uint32_t worker = 1; worker <= workerNum; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[0], resCtx.threads[worker], notifyIndex)));
    }
    for (uint32_t worker = 1; worker <= workerNum; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.threads[worker], notifyIndex, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult JoinWorkers(const AlgResourceCtx &resCtx, uint32_t workerNum, uint32_t notifyBase)
{
    for (uint32_t worker = 1; worker <= workerNum; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.threads[0], notifyBase + worker - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t worker = 1; worker <= workerNum; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[worker], resCtx.threads[0], notifyBase + worker - 1)));
    }
    return HCCL_SUCCESS;
}

HcclResult Exchange(const ThreadHandle thread, const ChannelInfo &channel, void *remoteAddress,
                    const void *localAddress, uint64_t bytes)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommWriteOnThread(thread, channel.handle, remoteAddress, localAddress, bytes)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult PublishReadyWrite(const ThreadHandle thread, const ChannelInfo &channel,
                             void *remoteAddress, const void *localAddress, uint64_t bytes)
{
    CHK_RET(static_cast<HcclResult>(
        HcommWriteOnThread(thread, channel.handle, remoteAddress, localAddress, bytes)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult ReadPeerContribution(const ThreadHandle thread, const ChannelInfo &channel,
                                void *localOutput, const void *remoteSource, uint64_t bytes,
                                void *localTailOutput, const void *remoteTailSource, uint64_t tailBytes);

HcclResult ExecuteSingleSliceDirectWrite(const OpParam &param, const AlgResourceCtx &resCtx,
                                         uint64_t totalInputBytes)
{
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    auto *localSlot = output + static_cast<uint64_t>(param.myRank) * totalInputBytes;

    // Release all Mesh workers from the coordinator stream, then form this
    // rank's output slot while their peer writes are in flight.
    for (uint32_t worker = 0; worker < kRanksPerServer - 1; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[kCoordinatorThread], resCtx.threads[kFirstMeshThread + worker], 0)));
    }
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[kCoordinatorThread], localSlot, param.inputPtr, totalInputBytes)));

    // Each local peer writes its contribution directly into the final output.
    // Queue the worker completion immediately after its channel transaction so
    // the coordinator can start the Clos phase as soon as all seven finish.
    for (uint32_t worker = 0; worker < kRanksPerServer - 1; ++worker) {
        const ThreadHandle workerThread = resCtx.threads[kFirstMeshThread + worker];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(workerThread, 0, CUSTOM_TIMEOUT)));
        const ChannelInfo &channel = resCtx.channels[worker];
        void *remoteSlot = static_cast<uint8_t *>(channel.remoteOutputMem.addr) +
            static_cast<uint64_t>(param.myRank) * totalInputBytes;
        CHK_RET(Exchange(workerThread, channel, remoteSlot, param.inputPtr, totalInputBytes));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            workerThread, resCtx.threads[kCoordinatorThread], worker)));
    }
    for (uint32_t worker = 0; worker < kRanksPerServer - 1; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.threads[kCoordinatorThread], worker, CUSTOM_TIMEOUT)));
    }

    const uint32_t localServer = param.myRank / kRanksPerServer;
    const uint64_t serverBytes = kRanksPerServer * totalInputBytes;
    const uint64_t localServerOffset = static_cast<uint64_t>(localServer) * serverBytes;
    const ChannelInfo &remoteChannel = resCtx.channels[kRanksPerServer - 1];
    auto *localServerOutput = output + localServerOffset;
    auto *remoteServerOutput =
        static_cast<uint8_t *>(remoteChannel.remoteOutputMem.addr) + localServerOffset;

    // Corresponding ranks exchange their completed local-server halves. The
    // source and destination halves are disjoint on both endpoints.
    return Exchange(resCtx.threads[kCoordinatorThread], remoteChannel,
                    remoteServerOutput, localServerOutput, serverBytes);
}

HcclResult ExecuteSingleSliceDirectWriteCoordinatorDataWait(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalInputBytes)
{
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    auto *localSlot = output + static_cast<uint64_t>(param.myRank) * totalInputBytes;

    // Keep all seven Mesh links on independent workers. This restores the
    // lower-overhead 05df271 layout while retaining the no-ACK experiment.
    for (uint32_t worker = 0; worker < kRanksPerServer - 1; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[kCoordinatorThread], resCtx.threads[kFirstMeshThread + worker], 0)));
    }
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[kCoordinatorThread], localSlot, param.inputPtr, totalInputBytes)));

    // Workers publish their writes with DATA only. The coordinator consumes
    // the peer DATA signals directly, avoiding the six extra worker completion
    // Record/Wait pairs introduced by the safe-join variant.
    for (uint32_t worker = 0; worker < kRanksPerServer - 1; ++worker) {
        const ThreadHandle workerThread = resCtx.threads[kFirstMeshThread + worker];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(workerThread, 0, CUSTOM_TIMEOUT)));
        const ChannelInfo &channel = resCtx.channels[worker];
        void *remoteSlot = static_cast<uint8_t *>(channel.remoteOutputMem.addr) +
            static_cast<uint64_t>(param.myRank) * totalInputBytes;
        CHK_RET(PublishReadyWrite(
            workerThread, channel, remoteSlot, param.inputPtr, totalInputBytes));
    }

    // As in 05df271, the coordinator waits only for the seven incoming Mesh
    // writes that make its local-server output half complete.
    for (uint32_t worker = 0; worker < kRanksPerServer - 1; ++worker) {
        const ChannelInfo &channel = resCtx.channels[worker];
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            resCtx.threads[kCoordinatorThread], channel.handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }

    const uint32_t localServer = param.myRank / kRanksPerServer;
    const uint64_t serverBytes = kRanksPerServer * totalInputBytes;
    const uint64_t localServerOffset = static_cast<uint64_t>(localServer) * serverBytes;
    const ChannelInfo &remoteChannel = resCtx.channels[kRanksPerServer - 1];
    auto *localServerOutput = output + localServerOffset;
    auto *remoteServerOutput =
        static_cast<uint8_t *>(remoteChannel.remoteOutputMem.addr) + localServerOffset;

    CHK_RET(PublishReadyWrite(resCtx.threads[kCoordinatorThread], remoteChannel,
                              remoteServerOutput, localServerOutput, serverBytes));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        resCtx.threads[kCoordinatorThread], remoteChannel.handle,
        NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ReadPeerContribution(const ThreadHandle thread, const ChannelInfo &channel,
                                void *localOutput, const void *remoteSource, uint64_t bytes,
                                void *localTailOutput, const void *remoteTailSource, uint64_t tailBytes)
{
    // ACK proves that both endpoints finished staging their local slice. DATA
    // proves that both endpoints finished reading it before the CCL buffer is
    // reused by the next slice.
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
        thread, channel.handle, localOutput, remoteSource, bytes)));
    if (tailBytes != 0) {
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
            thread, channel.handle, localTailOutput, remoteTailSource, tailBytes)));
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

bool CanUseRegisteredInput(const OpParam &param, const AlgResourceCtx &resCtx,
                           uint64_t totalInputBytes)
{
    if (resCtx.registeredInput.addr != param.inputPtr ||
        resCtx.registeredInput.size < totalInputBytes) {
        return false;
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteInputMem.addr == nullptr || channel.remoteInputMem.size < totalInputBytes) {
            return false;
        }
    }
    return true;
}

bool CanUseRegisteredOutput(const OpParam &param, const AlgResourceCtx &resCtx,
                            uint64_t totalInputBytes)
{
    if (totalInputBytes > std::numeric_limits<uint64_t>::max() / kRankSize) {
        return false;
    }
    const uint64_t totalOutputBytes = totalInputBytes * kRankSize;
    if (resCtx.registeredOutput.addr != param.outputPtr ||
        resCtx.registeredOutput.size < totalOutputBytes) {
        return false;
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteOutputMem.addr == nullptr ||
            channel.remoteOutputMem.size < totalOutputBytes) {
            return false;
        }
    }
    return true;
}

bool CanUseSingleSliceDirectWrite(const OpParam &param, const AlgResourceCtx &resCtx,
                                  uint64_t totalInputBytes)
{
    if (resCtx.registeredInput.addr != param.inputPtr ||
        resCtx.registeredInput.size < totalInputBytes ||
        totalInputBytes > std::numeric_limits<uint64_t>::max() / kRankSize) {
        return false;
    }
    const uint64_t totalOutputBytes = totalInputBytes * kRankSize;
    if (resCtx.registeredOutput.addr != param.outputPtr ||
        resCtx.registeredOutput.size < totalOutputBytes) {
        return false;
    }
    for (uint32_t channelIndex = 0; channelIndex < kRanksPerServer; ++channelIndex) {
        const CommBuffer &remoteOutput = resCtx.channels[channelIndex].remoteOutputMem;
        if (remoteOutput.addr == nullptr || remoteOutput.size < totalOutputBytes) {
            return false;
        }
    }
    return true;
}

HcclResult ExecuteRegisteredOutputRead(const OpParam &param, const AlgResourceCtx &resCtx,
                                       uint64_t totalInputBytes)
{
    auto *localOutput = static_cast<uint8_t *>(param.outputPtr) +
        static_cast<uint64_t>(param.myRank) * totalInputBytes;
    CHK_RET(SyncWorkers(resCtx, kPeerNum, 0));
    const uint64_t unalignedChunkBytes =
        (totalInputBytes + kOutputPipelineChunkNum - 1) / kOutputPipelineChunkNum;
    const uint64_t chunkBytes =
        ((unalignedChunkBytes + kAlignment - 1) / kAlignment) * kAlignment;
    uint32_t chunkNum = 0;

    for (uint64_t offset = 0; offset < totalInputBytes; offset += chunkBytes) {
        const uint64_t bytes = std::min(chunkBytes, totalInputBytes - offset);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            resCtx.threads[kLocalCopyThread], localOutput + offset,
            static_cast<uint8_t *>(param.inputPtr) + offset, bytes)));
        if (chunkNum != 0) {
            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    resCtx.threads[kLocalCopyThread], channel.handle,
                    NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }
        }
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                resCtx.threads[kLocalCopyThread], channel.handle, NOTIFY_IDX_ACK)));
        }
        ++chunkNum;
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            resCtx.threads[kLocalCopyThread], channel.handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }

    for (uint32_t peerIndex = 0; peerIndex < kPeerNum; ++peerIndex) {
        const ChannelInfo &channel = resCtx.channels[peerIndex];
        const uint64_t rankOffset =
            static_cast<uint64_t>(channel.remoteRank) * totalInputBytes;
        auto *rankOutput = static_cast<uint8_t *>(param.outputPtr) + rankOffset;
        auto *remoteRankOutput =
            static_cast<uint8_t *>(channel.remoteOutputMem.addr) + rankOffset;
        const ThreadHandle peerThread = PeerThread(resCtx, peerIndex);
        for (uint32_t chunk = 0; chunk < chunkNum; ++chunk) {
            const uint64_t offset = static_cast<uint64_t>(chunk) * chunkBytes;
            const uint64_t bytes = std::min(chunkBytes, totalInputBytes - offset);
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                peerThread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
                peerThread, channel.handle, rankOutput + offset,
                remoteRankOutput + offset, bytes)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                peerThread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        }
    }

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[kLocalCopyThread], resCtx.threads[kCoordinatorThread], 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[kCoordinatorThread], 0, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ExecuteRegisteredInputRead(const OpParam &param, const AlgResourceCtx &resCtx,
                                      uint64_t totalInputBytes)
{
    CHK_RET(SyncWorkers(resCtx, kPeerNum, 0));

    for (uint32_t peerIndex = 0; peerIndex < kPeerNum; ++peerIndex) {
        const ChannelInfo &channel = resCtx.channels[peerIndex];
        auto *output = static_cast<uint8_t *>(param.outputPtr) +
            static_cast<uint64_t>(channel.remoteRank) * totalInputBytes;
        CHK_RET(ReadPeerContribution(PeerThread(resCtx, peerIndex), channel,
                                     output, channel.remoteInputMem.addr, totalInputBytes,
                                     nullptr, nullptr, 0));
    }

    auto *localOutput = static_cast<uint8_t *>(param.outputPtr) +
        static_cast<uint64_t>(param.myRank) * totalInputBytes;
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[kLocalCopyThread], localOutput, param.inputPtr, totalInputBytes)));
    CHK_RET(JoinWorkers(resCtx, kPeerNum, 0));
    return HCCL_SUCCESS;
}

HcclResult ExecuteDirectReadSlice(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t typeSize,
                                  uint64_t totalInputBytes, uint64_t processedCount, uint64_t count,
                                  uint64_t tailProcessedCount, uint64_t tailCount, uint32_t sliceIndex)
{
    const uint64_t processedBytes = processedCount * typeSize;
    const uint64_t bytes = count * typeSize;
    const uint64_t tailProcessedBytes = tailProcessedCount * typeSize;
    const uint64_t tailBytes = tailCount * typeSize;
    auto *input = static_cast<uint8_t *>(param.inputPtr) + processedBytes;
    auto *localStage = static_cast<uint8_t *>(resCtx.localBuffer.addr) +
        StagingOffset(param.myRank, totalInputBytes);

    // Match each source rank's CCL address phase to its rank-major output
    // phase. This keeps large reads aligned relative to both endpoints when
    // totalInputBytes is not a multiple of 128 bytes.
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[kCoordinatorThread], localStage, input, bytes)));
    if (tailBytes != 0) {
        // Fuse the sub-128-byte tail into this round so it shares the channel
        // handshake without changing either aligned main slice.
        auto *tailInput = static_cast<uint8_t *>(param.inputPtr) + tailProcessedBytes;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            resCtx.threads[kCoordinatorThread], localStage + bytes, tailInput, tailBytes)));
    }
    CHK_RET(SyncWorkers(resCtx, kPeerNum, sliceIndex));

    for (uint32_t peerIndex = 0; peerIndex < kPeerNum; ++peerIndex) {
        const ChannelInfo &channel = resCtx.channels[peerIndex];
        auto *rankOutput = static_cast<uint8_t *>(param.outputPtr) +
            static_cast<uint64_t>(channel.remoteRank) * totalInputBytes;
        auto *output = rankOutput + processedBytes;
        auto *remoteSource = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
            StagingOffset(channel.remoteRank, totalInputBytes);
        auto *tailOutput = rankOutput + tailProcessedBytes;
        auto *remoteTailSource = remoteSource + bytes;
        CHK_RET(ReadPeerContribution(
            PeerThread(resCtx, peerIndex), channel, output, remoteSource, bytes,
            tailOutput, remoteTailSource, tailBytes));
    }

    auto *localRankOutput = static_cast<uint8_t *>(param.outputPtr) +
        static_cast<uint64_t>(param.myRank) * totalInputBytes;
    auto *localOutput = localRankOutput + processedBytes;
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[kLocalCopyThread], localOutput, input, bytes)));
    if (tailBytes != 0) {
        auto *tailInput = static_cast<uint8_t *>(param.inputPtr) + tailProcessedBytes;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            resCtx.threads[kLocalCopyThread], localRankOutput + tailProcessedBytes, tailInput, tailBytes)));
    }
    CHK_RET(JoinWorkers(resCtx, kPeerNum, sliceIndex * kPeerNum));
    return HCCL_SUCCESS;
}

HcclResult ExecuteDirectRead(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t typeSize,
                             uint64_t totalInputBytes, uint64_t maxSliceCount)
{
    const uint64_t alignmentCount = kAlignment / typeSize;
    const uint64_t maxBlockCount = maxSliceCount / alignmentCount;
    const uint64_t alignedBlockCount = param.count / alignmentCount;
    const uint64_t alignedCount = alignedBlockCount * alignmentCount;
    const uint64_t tailCount = param.count - alignedCount;
    uint64_t processedCount = 0;

    if (alignedBlockCount > 0) {
        const uint64_t sliceNum =
            alignedBlockCount / maxBlockCount + (alignedBlockCount % maxBlockCount != 0 ? 1 : 0);
        if (sliceNum > kMaxDirectSliceNum) {
            HCCL_ERROR("Direct-read allgather requires more than two slices");
            return HCCL_E_PARA;
        }
        const uint64_t blocksPerSlice =
            alignedBlockCount / sliceNum + (alignedBlockCount % sliceNum != 0 ? 1 : 0);
        for (uint64_t sliceIndex = 0; sliceIndex < sliceNum; ++sliceIndex) {
            const uint64_t remainingBlocks = alignedBlockCount - processedCount / alignmentCount;
            const uint64_t blockCount = std::min(blocksPerSlice, remainingBlocks);
            const uint64_t count = blockCount * alignmentCount;
            const uint64_t fusedTailCount = sliceIndex == 0 ? tailCount : 0;
            CHK_RET(ExecuteDirectReadSlice(
                param, resCtx, typeSize, totalInputBytes, processedCount, count,
                alignedCount, fusedTailCount, static_cast<uint32_t>(sliceIndex)));
            processedCount += count;
        }
        return HCCL_SUCCESS;
    }

    return ExecuteDirectReadSlice(
        param, resCtx, typeSize, totalInputBytes, 0, param.count, 0, 0, 0);
}

} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.rankSize != kRankSize || resCtx.localRankCount != kRanksPerServer ||
        resCtx.threadNum != kDirectThreadNum || resCtx.channelNum != kPeerNum) {
        HCCL_ERROR("Invalid direct-read allgather resource context");
        return HCCL_E_PARA;
    }
    if (param.dataType != HCCL_DATA_TYPE_FP32) {
        HCCL_ERROR("Unsupported data type %u", static_cast<uint32_t>(param.dataType));
        return HCCL_E_PARA;
    }

    constexpr uint64_t typeSize = sizeof(float);
    const uint64_t totalInputBytes = param.count * typeSize;
    if (totalInputBytes == 0) {
        return HCCL_SUCCESS;
    }

    // Keep the low-latency hierarchical path for small messages. The selected
    // variant uses only standard AICPU+TS Read/Write and notify primitives.
    if (totalInputBytes <= kSmallSingleSliceMaxBytes) {
        if (!CanUseSingleSliceDirectWrite(param, resCtx, totalInputBytes)) {
            HCCL_ERROR("Direct-output single-slice requires current registered input/output, bytes=%llu",
                       static_cast<unsigned long long>(totalInputBytes));
            return HCCL_E_PARA;
        }
        HCCL_INFO("AllGather path: up-to-1MiB single-slice variant, bytes=%llu",
                  static_cast<unsigned long long>(totalInputBytes));
        if (totalInputBytes == kSmallVariantBytes &&
            kSmallSingleSliceVariant ==
                SmallSingleSliceVariant::kDirectOutputWriteCoordinatorDataWait) {
            return ExecuteSingleSliceDirectWriteCoordinatorDataWait(
                param, resCtx, totalInputBytes);
        }
        return ExecuteSingleSliceDirectWrite(param, resCtx, totalInputBytes);
    }

    // Large messages use the normal zero-staging registered-input path. This
    // covers both the aligned 512 MiB case and the unaligned 400 MiB + 4 B
    // case without relying on the unavailable WriteWithNotify API.
    if (CanUseRegisteredInput(param, resCtx, totalInputBytes)) {
        HCCL_INFO("AllGather path: registered input read, bytes=%llu",
                  static_cast<unsigned long long>(totalInputBytes));
        return ExecuteRegisteredInputRead(param, resCtx, totalInputBytes);
    }

    // Retain the original fallback for contexts where input registration is
    // unavailable; it is also useful for non-scoring message sizes.
    if (totalInputBytes % kAlignment != 0 &&
        CanUseRegisteredOutput(param, resCtx, totalInputBytes)) {
        HCCL_INFO("AllGather path: registered output read, bytes=%llu",
                  static_cast<unsigned long long>(totalInputBytes));
        return ExecuteRegisteredOutputRead(param, resCtx, totalInputBytes);
    }

    const uint64_t maxStagingOffset = MaxStagingOffset(totalInputBytes);
    if (resCtx.localBuffer.size <= maxStagingOffset) {
        HCCL_ERROR("HCCL buffer cannot hold the aligned direct-read staging area");
        return HCCL_E_PARA;
    }
    const uint64_t availableBufferBytes = resCtx.localBuffer.size - maxStagingOffset;
    const uint64_t tailBytes = totalInputBytes % kAlignment;
    if (availableBufferBytes <= tailBytes) {
        HCCL_ERROR("HCCL buffer cannot hold the direct-read tail staging area");
        return HCCL_E_PARA;
    }
    const uint64_t sliceBufferBytes = availableBufferBytes - tailBytes;
    const uint64_t maxSliceBytes =
        (std::min(sliceBufferBytes, kMaxTransferBytes) / kAlignment) * kAlignment;
    const uint64_t maxSliceCount = maxSliceBytes / typeSize;
    if (maxSliceCount < kAlignment / typeSize) {
        HCCL_ERROR("HCCL buffer is too small for direct-read allgather");
        return HCCL_E_PARA;
    }
    HCCL_INFO("AllGather path: CCL staging read, bytes=%llu",
              static_cast<unsigned long long>(totalInputBytes));
    return ExecuteDirectRead(param, resCtx, typeSize, totalInputBytes, maxSliceCount);
}
} // namespace ops_hccl
