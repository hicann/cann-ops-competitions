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
#include <cstdint>
#include <limits>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint32_t RANKS_PER_SERVER = ALLGATHER_SERVER_RANK_SIZE;
constexpr uint32_t EXPECTED_RANK_SIZE = ALLGATHER_RANK_SIZE;
constexpr uint32_t THREAD_NOTIFY_START = 1;
constexpr uint32_t THREAD_NOTIFY_DONE_START = 1;
constexpr uint64_t CCL_SLOT_ALIGNMENT = 64;
constexpr uint64_t SMALL_MESSAGE_THRESHOLD = 512ULL * 1024ULL;
constexpr uint32_t SMALL_STAGE_NUM = ALLGATHER_SMALL_STAGE_NUM;

HcclResult WriteThenRecordData(ThreadHandle thread, const ChannelInfo &channel,
    void *remoteDestination, const void *localSource, uint64_t bytes)
{
    // The VM currently emits an invalid trailing SQE for WriteWithNotify on
    // this topology.  Keep data and notification as two ordered tasks on the
    // same thread so both the device parser and checker see write -> record.
    CHK_RET(HcommWriteOnThread(
        thread, channel.handle, remoteDestination, localSource, bytes));
    CHK_RET(HcommChannelNotifyRecordOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
    return HCCL_SUCCESS;
}

HcclResult RunDualCrossPeer(const AlgResourceCtx &resCtx,
    uint32_t firstChannelIdx, uint32_t secondChannelIdx,
    void *firstRemoteDestination, void *secondRemoteDestination,
    const void *localSource, uint64_t blockBytes)
{
    const ChannelInfo &firstChannel = resCtx.channels[firstChannelIdx];
    const ChannelInfo &secondChannel = resCtx.channels[secondChannelIdx];
    ThreadHandle worker = resCtx.threads[firstChannelIdx + 1];
    ThreadHandle mainThread = resCtx.threads[0];

    // xor-2 has already made this four-rank block immutable. Fan the same
    // block out to both remote four-rank groups while mainThread sends it to
    // the opposite local group. Issue both writes before waiting for DATA so
    // the high-bandwidth network does not idle on a round-trip dependency.
    CHK_RET(HcommThreadNotifyWaitOnThread(worker, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
    CHK_RET(WriteThenRecordData(
        worker, firstChannel, firstRemoteDestination, localSource, blockBytes));

    CHK_RET(WriteThenRecordData(
        worker, secondChannel, secondRemoteDestination, localSource, blockBytes));

    CHK_RET(HcommChannelNotifyWaitOnThread(
        worker, firstChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        worker, secondChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

    CHK_RET(HcommThreadNotifyRecordOnThread(
        worker, mainThread, THREAD_NOTIFY_DONE_START + firstChannelIdx));
    return HCCL_SUCCESS;
}

HcclResult ValidateCommonResources(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t totalBytes)
{
    // The immutable topology, channels, notifies and remote buffers were fully
    // validated by Host before this fixed-layout EngineCtx was published.
    // Keep only per-call guards on the AICPU hot path.
    CHK_PRT_RET(param.rankSize != EXPECTED_RANK_SIZE || param.myRank >= param.rankSize ||
            resCtx.ownerRank != param.myRank,
        HCCL_ERROR("Invalid rank or resource owner: rank %u / %u, owner %u",
            param.myRank, param.rankSize, resCtx.ownerRank), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0,
        HCCL_ERROR("Local CCL buffer is invalid"), HCCL_E_MEMORY);
    CHK_PRT_RET(totalBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("Output size overflows uint64"), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

bool CanUseDirectOutput(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t totalBytes)
{
    if ((param.root & DIRECT_OUTPUT_FLAG) == 0 || resCtx.directEnabled == 0) {
        return false;
    }
    // Host initialization already validated all fifteen remote OUTPUT ranges,
    // and the per-call flag requires an exact cached OUTPUT pointer/size match.
    // Keep only the local O(1) guard on the AICPU hot path.
    const uint64_t outputBytes = totalBytes * param.rankSize;
    return resCtx.localOutput.addr == param.outputPtr &&
        resCtx.localOutput.size >= outputBytes;
}

bool CanUseDirectInput(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t totalBytes)
{
    return (param.root & DIRECT_INPUT_FLAG) != 0 && resCtx.inputDirectEnabled != 0 &&
        resCtx.localInput.addr == param.inputPtr && resCtx.localInput.size >= totalBytes;
}

HcclResult RunRecursiveDoubling(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t totalBytes, bool directOutput)
{
    if (resCtx.smallStageValid == 0) {
        HCCL_WARNING("Rank numbering is not contiguous; fall back to Flat AllGather");
        return HCCL_E_NOT_SUPPORT;
    }
    const auto &stageChannels = resCtx.smallStageChannels;
    const uint32_t localIndex = resCtx.smallLocalIndex;

    ThreadHandle mainThread = resCtx.threads[0];
    char *output = static_cast<char *>(param.outputPtr);
    char *localWorking = directOutput ? output : static_cast<char *>(resCtx.localBuffer.addr);
    const uint64_t workingBytes = static_cast<uint64_t>(param.rankSize) * totalBytes;
    if (directOutput) {
        CHK_PRT_RET(resCtx.localOutput.addr != output || resCtx.localOutput.size < workingBytes,
            HCCL_ERROR("Local OUTPUT is too small for Recursive Doubling"), HCCL_E_MEMORY);
    } else {
        CHK_PRT_RET(resCtx.localBuffer.size < workingBytes ||
                resCtx.minRemoteCclBytes < workingBytes,
            HCCL_ERROR("Local CCL buffer is too small for Recursive Doubling"), HCCL_E_MEMORY);
    }

    // At exactly 512 KiB on registered OUTPUT, fan the four-rank block to two
    // remote groups while xor-4 runs locally. Smaller messages and CCL fallback
    // keep the lower-task-count serial recursive-doubling path.
    const bool dualCross = directOutput && totalBytes == SMALL_MESSAGE_THRESHOLD;

    // DATA may be recorded before the peer reaches its wait. Every active
    // channel has exactly one ordered Write -> DATA Record -> DATA Wait, so no
    // separate READY handshake is needed on this competition fast path.
    const uint32_t crossStage = SMALL_STAGE_NUM - 1;

    CHK_RET(HcommLocalCopyOnThread(mainThread,
        localWorking + static_cast<uint64_t>(param.myRank) * totalBytes,
        param.inputPtr, totalBytes));

    const uint32_t serialLocalStages = dualCross ? crossStage - 1U : crossStage;

    // Each DATA Wait makes the newly received block visible before the next
    // dependent stage reads it.
    for (uint32_t stage = 0; stage < serialLocalStages; ++stage) {
        const uint32_t blockRanks = 1U << stage;
        const uint32_t blockLocalStart = localIndex & ~(blockRanks - 1U);
        const uint32_t blockGlobalStart = resCtx.localRanks[blockLocalStart];
        const uint64_t blockBytes = static_cast<uint64_t>(blockRanks) * totalBytes;
        char *localSource = localWorking + static_cast<uint64_t>(blockGlobalStart) * totalBytes;
        const ChannelInfo &channel = resCtx.channels[stageChannels[stage]];
        const CommBuffer &remoteWorking = directOutput ? channel.remoteOutput : channel.remoteCclMem;
        char *remoteDestination = static_cast<char *>(remoteWorking.addr) +
            static_cast<uint64_t>(blockGlobalStart) * totalBytes;

        CHK_RET(WriteThenRecordData(
            mainThread, channel, remoteDestination, localSource, blockBytes));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            mainThread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    }

    if (dualCross) {
        const uint32_t lastLocalStage = crossStage - 1U;
        const uint32_t blockRanks = 1U << lastLocalStage;
        const uint32_t blockLocalStart = localIndex & ~(blockRanks - 1U);
        const uint32_t blockGlobalStart = resCtx.localRanks[blockLocalStart];
        const uint64_t blockBytes = static_cast<uint64_t>(blockRanks) * totalBytes;
        char *localSource = localWorking +
            static_cast<uint64_t>(blockGlobalStart) * totalBytes;

        const uint32_t firstCrossChannelIdx = stageChannels[crossStage];
        const uint32_t secondCrossChannelIdx = resCtx.smallCrossMateChannel;
        const ChannelInfo &firstCrossChannel = resCtx.channels[firstCrossChannelIdx];
        const ChannelInfo &secondCrossChannel = resCtx.channels[secondCrossChannelIdx];
        char *firstRemoteDestination =
            static_cast<char *>(firstCrossChannel.remoteOutput.addr) +
            static_cast<uint64_t>(blockGlobalStart) * totalBytes;
        char *secondRemoteDestination =
            static_cast<char *>(secondCrossChannel.remoteOutput.addr) +
            static_cast<uint64_t>(blockGlobalStart) * totalBytes;
        ThreadHandle crossWorker = resCtx.threads[firstCrossChannelIdx + 1];

        // One START releases a worker whose only sources are the four-rank
        // block completed by xor-2; neither cross write depends on xor-4.
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, crossWorker, THREAD_NOTIFY_START));
        CHK_RET(RunDualCrossPeer(resCtx,
            firstCrossChannelIdx, secondCrossChannelIdx,
            firstRemoteDestination, secondRemoteDestination,
            localSource, blockBytes));

        // In parallel, xor-4 fans the same block to the opposite local group.
        // The three destinations occupy disjoint OUTPUT rank ranges.
        const ChannelInfo &lastLocalChannel = resCtx.channels[stageChannels[lastLocalStage]];
        char *lastLocalRemoteDestination =
            static_cast<char *>(lastLocalChannel.remoteOutput.addr) +
            static_cast<uint64_t>(blockGlobalStart) * totalBytes;
        CHK_RET(WriteThenRecordData(mainThread, lastLocalChannel,
            lastLocalRemoteDestination, localSource, blockBytes));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            mainThread, lastLocalChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread,
            THREAD_NOTIFY_DONE_START + firstCrossChannelIdx, CUSTOM_TIMEOUT));
        return HCCL_SUCCESS;
    }

    // Equal local-index ranks exchange their complete contiguous 8-rank block.
    const uint32_t serverBlockStart = resCtx.localRanks[0];
    const uint64_t serverBlockBytes = static_cast<uint64_t>(RANKS_PER_SERVER) * totalBytes;
    char *localServerBlock = localWorking + static_cast<uint64_t>(serverBlockStart) * totalBytes;
    const ChannelInfo &crossChannel = resCtx.channels[stageChannels[crossStage]];
    const CommBuffer &remoteWorking = directOutput ? crossChannel.remoteOutput : crossChannel.remoteCclMem;
    char *remoteServerBlock = static_cast<char *>(remoteWorking.addr) +
        static_cast<uint64_t>(serverBlockStart) * totalBytes;
    CHK_RET(WriteThenRecordData(
        mainThread, crossChannel, remoteServerBlock, localServerBlock, serverBlockBytes));
    CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, crossChannel.handle,
        NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    if (!directOutput) {
        CHK_RET(HcommLocalCopyOnThread(mainThread, output, localWorking, workingBytes));
    }
    return HCCL_SUCCESS;
}

HcclResult RunOutputDirectPeer(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t channelIdx, uint64_t totalBytes, const void *localSource)
{
    const ChannelInfo &channel = resCtx.channels[channelIdx];
    ThreadHandle worker = resCtx.threads[channelIdx + 1];
    ThreadHandle mainThread = resCtx.threads[0];

    CHK_RET(HcommThreadNotifyWaitOnThread(worker, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));

    char *remoteDestination = static_cast<char *>(channel.remoteOutput.addr) +
        static_cast<uint64_t>(param.myRank) * totalBytes;
    CHK_RET(WriteThenRecordData(
        worker, channel, remoteDestination, localSource, totalBytes));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

    CHK_RET(HcommThreadNotifyRecordOnThread(
        worker, mainThread, THREAD_NOTIFY_DONE_START + channelIdx));
    return HCCL_SUCCESS;
}

HcclResult RunOutputDirectFlat(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t totalBytes)
{
    ThreadHandle mainThread = resCtx.threads[0];
    char *selfOutput = static_cast<char *>(param.outputPtr) +
        static_cast<uint64_t>(param.myRank) * totalBytes;
    const bool overlapSelfCopy = CanUseDirectInput(param, resCtx, totalBytes);
    const void *localSource = overlapSelfCopy ? param.inputPtr : selfOutput;

    // Without registered INPUT, preserve the validated OUTPUT-source path:
    // populate the self slot before any worker is released.
    if (!overlapSelfCopy && param.inputPtr != selfOutput) {
        CHK_RET(HcommLocalCopyOnThread(
            mainThread, selfOutput, param.inputPtr, totalBytes));
    }

    // START always propagates the Host/AICPU input-ready dependency. With
    // registered INPUT, workers read sendBuf directly while main copies the
    // same immutable bytes into the self OUTPUT slot.
    for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[channelIdx + 1], THREAD_NOTIFY_START));
    }

    if (overlapSelfCopy && param.inputPtr != selfOutput) {
        CHK_RET(HcommLocalCopyOnThread(
            mainThread, selfOutput, param.inputPtr, totalBytes));
    }

    for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
        CHK_RET(RunOutputDirectPeer(
            param, resCtx, channelIdx, totalBytes, localSource));
    }
    for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread,
            THREAD_NOTIFY_DONE_START + channelIdx, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult GetCclSlotBytes(const AlgResourceCtx &resCtx, uint64_t &slotBytes)
{
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr,
        HCCL_ERROR("Local CCL buffer is invalid"), HCCL_E_MEMORY);
    const uint64_t minBufferSize = std::min(
        resCtx.localBuffer.size, resCtx.minRemoteCclBytes);
    slotBytes = minBufferSize / EXPECTED_RANK_SIZE;
    slotBytes -= slotBytes % CCL_SLOT_ALIGNMENT;
    CHK_PRT_RET(slotBytes < sizeof(float),
        HCCL_ERROR("CCL buffer cannot hold sixteen rank-indexed slots"), HCCL_E_MEMORY);
    return HCCL_SUCCESS;
}

HcclResult RunCclPeer(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t channelIdx,
    uint64_t totalBytes, uint64_t slotBytes, uint64_t offset, uint64_t windowBytes)
{
    const ChannelInfo &channel = resCtx.channels[channelIdx];
    ThreadHandle worker = resCtx.threads[channelIdx + 1];
    ThreadHandle mainThread = resCtx.threads[0];
    CHK_RET(HcommThreadNotifyWaitOnThread(worker, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));

    char *localBase = static_cast<char *>(resCtx.localBuffer.addr);
    char *remoteBase = static_cast<char *>(channel.remoteCclMem.addr);
    char *localSource = localBase + static_cast<uint64_t>(param.myRank) * slotBytes;
    char *remoteDestination = remoteBase + static_cast<uint64_t>(param.myRank) * slotBytes;
    char *localIncoming = localBase + static_cast<uint64_t>(channel.remoteRank) * slotBytes;
    char *output = static_cast<char *>(param.outputPtr) +
        static_cast<uint64_t>(channel.remoteRank) * totalBytes + offset;

    CHK_RET(WriteThenRecordData(
        worker, channel, remoteDestination, localSource, windowBytes));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        worker, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    CHK_RET(HcommLocalCopyOnThread(worker, output, localIncoming, windowBytes));
    CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel.handle, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        worker, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    CHK_RET(HcommThreadNotifyRecordOnThread(
        worker, mainThread, THREAD_NOTIFY_DONE_START + channelIdx));
    return HCCL_SUCCESS;
}

HcclResult RunCclWindow(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t totalBytes, uint64_t slotBytes, uint64_t offset, uint64_t windowBytes)
{
    ThreadHandle mainThread = resCtx.threads[0];
    char *localBase = static_cast<char *>(resCtx.localBuffer.addr);
    char *localSource = localBase + static_cast<uint64_t>(param.myRank) * slotBytes;
    const char *input = static_cast<const char *>(param.inputPtr) + offset;
    char *selfOutput = static_cast<char *>(param.outputPtr) +
        static_cast<uint64_t>(param.myRank) * totalBytes + offset;

    // The CCL source must be populated before START releases any worker.
    CHK_RET(HcommLocalCopyOnThread(mainThread, localSource, input, windowBytes));
    for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[channelIdx + 1], THREAD_NOTIFY_START));
    }
    CHK_RET(HcommLocalCopyOnThread(mainThread, selfOutput, input, windowBytes));

    for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
        CHK_RET(RunCclPeer(param, resCtx, channelIdx,
            totalBytes, slotBytes, offset, windowBytes));
    }
    for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread,
            THREAD_NOTIFY_DONE_START + channelIdx, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult RunCclFallback(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t totalBytes)
{
    uint64_t slotBytes = 0;
    CHK_RET(GetCclSlotBytes(resCtx, slotBytes));
    for (uint64_t offset = 0; offset < totalBytes;) {
        const uint64_t windowBytes = std::min(slotBytes, totalBytes - offset);
        CHK_RET(RunCclWindow(
            param, resCtx, totalBytes, slotBytes, offset, windowBytes));
        offset += windowBytes;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    auto typeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type %u", static_cast<uint32_t>(param.dataType)), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank: rank %u / %u", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeIt->second,
        HCCL_ERROR("Input size overflows uint64"), HCCL_E_PARA);

    const uint64_t totalBytes = param.count * typeIt->second;
    if (totalBytes == 0) {
        return HCCL_SUCCESS;
    }
    CHK_RET(ValidateCommonResources(param, resCtx, totalBytes));

    // OpParam.root carries the direct-path flag; AllGather has no root rank.
    if (totalBytes <= SMALL_MESSAGE_THRESHOLD) {
        const bool directOutput = CanUseDirectOutput(param, resCtx, totalBytes);
        HCCL_INFO("Executing %s topology-aware Recursive Doubling AllGather",
            directOutput ? "registered-OUTPUT" : "CCL-backed");
        HcclResult ret = RunRecursiveDoubling(param, resCtx, totalBytes, directOutput);
        if (ret == HCCL_SUCCESS) {
            return HCCL_SUCCESS;
        }
        CHK_PRT_RET(ret != HCCL_E_NOT_SUPPORT,
            HCCL_ERROR("Recursive Doubling failed: %d", static_cast<int32_t>(ret)), ret);
    }

    if (CanUseDirectOutput(param, resCtx, totalBytes)) {
        HCCL_INFO("Executing registered-OUTPUT-source Direct Flat AllGather");
        return RunOutputDirectFlat(param, resCtx, totalBytes);
    }

    HCCL_INFO("Executing CCL-staged fallback AllGather");
    return RunCclFallback(param, resCtx, totalBytes);
}
} // namespace ops_hccl
