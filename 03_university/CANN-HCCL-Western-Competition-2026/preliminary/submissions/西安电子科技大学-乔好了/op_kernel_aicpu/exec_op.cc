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

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint32_t ALLREDUCE_STEP_COUNT = 4;
constexpr uint32_t ALLREDUCE_CHANNEL_COUNT = 8;
constexpr uint32_t ALLREDUCE_RANK_SIZE = 16;
constexpr uint32_t LOCAL_RANK_COUNT = 8;
constexpr uint32_t RADIX_THREAD_COUNT = 4;
constexpr uint32_t FULLMESH_THREAD_COUNT = 8;
constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t SMALL_MESSAGE_BYTES = 512ULL * 1024ULL;
constexpr std::array<uint32_t, ALLREDUCE_STEP_COUNT> XOR_CHANNELS = {0, 1, 2, 3};
constexpr std::array<uint32_t, LOCAL_RANK_COUNT - 1> LOCAL_PEER_CHANNELS = {0, 1, 2, 4, 5, 6, 7};
constexpr std::array<uint32_t, ALLREDUCE_CHANNEL_COUNT> CHANNEL_DELTAS = {1, 2, 4, 8, 3, 5, 6, 7};
constexpr uint64_t FULLMESH_MESSAGE_400M4B = 400ULL * 1024ULL * 1024ULL + FP32_BYTES;
constexpr uint64_t FULLMESH_MESSAGE_512M = 512ULL * 1024ULL * 1024ULL;
constexpr uint32_t PIPELINE_RS_SPLIT_WORKERS = LOCAL_RANK_COUNT - 1;
constexpr uint32_t PIPELINE_RS_SPLIT_SLICES = LOCAL_RANK_COUNT;
// Pipeline2 is enabled only for the fixed 512 MiB layout validated against both gates.
constexpr bool ENABLE_FULLMESH_PIPELINE2 = true;

struct NotifyProtocol {
    uint32_t ack = 0;
    uint32_t data = 1;
    uint32_t interReady = 2;
};

constexpr NotifyProtocol NOTIFY = {};

using ChannelMap = ChannelCollection;
using PendingAcks = std::array<bool, ALLREDUCE_CHANNEL_COUNT>;
using LocalFlights = std::array<bool, ALLREDUCE_CHANNEL_COUNT>;
using SegmentBoundaries = std::array<uint64_t, LOCAL_RANK_COUNT + 1>;

enum class AliasGroup : uint8_t {
    NONE,
    XOR_ONE_INTER,
    QUARTER_SCRATCH,
};

struct BufferRegion {
    uint64_t offset = 0;
    uint64_t capacity = 0;
    AliasGroup aliasGroup = AliasGroup::NONE;
};

struct Span {
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct LargeBufferLayout {
    uint64_t maxChunkBytes = 0;
    std::array<BufferRegion, ALLREDUCE_STEP_COUNT> regions = {};
    BufferRegion scratch = {};
};

struct RadixBufferLayout {
    uint64_t maxChunkBytes = 0;
    uint64_t slotBytes = 0;
};

struct FullMeshBufferLayout {
    uint64_t slotBytes = 0;
    uint64_t crossOffset = 0;
    uint64_t scratchOffset = 0;
    uint32_t waveCount = 0;
};

enum class ExecutionPolicy : uint8_t {
    SMALL_XOR,
    LARGE_FULLMESH,
    LARGE_FULLMESH_PIPELINE2,
    LARGE_FAST,
    LARGE_RADIX,
    LARGE_SAFE,
};

struct AllReducePlan {
    uint64_t totalBytes = 0;
    uint64_t xorSlotBytes = 0;
    LargeBufferLayout safe = {};
    LargeBufferLayout fast = {};
    RadixBufferLayout radix = {};
    FullMeshBufferLayout fullMesh = {};
    ExecutionPolicy policy = ExecutionPolicy::LARGE_SAFE;
};

HcclResult ValidateResourceContract(const OpParam &param, const AlgResourceCtx &resource)
{
    CHK_PRT_RET(resource.channels.size() != ALLREDUCE_CHANNEL_COUNT,
        HCCL_ERROR("ExecOp: expected[%u] peer channels, got[%zu]", ALLREDUCE_CHANNEL_COUNT,
            resource.channels.size()), HCCL_E_INTERNAL);
    for (uint32_t index = 0; index < ALLREDUCE_CHANNEL_COUNT; ++index) {
        const uint32_t expectedPeer = param.myRank ^ CHANNEL_DELTAS[index];
        CHK_PRT_RET(resource.channels[index].remoteRank != expectedPeer,
            HCCL_ERROR("ExecOp: channel[%u] peer[%u] expected[%u]", index,
                resource.channels[index].remoteRank, expectedPeer), HCCL_E_INTERNAL);
        CHK_PRT_RET(resource.channels[index].notifyNum <= NOTIFY.interReady,
            HCCL_ERROR("ExecOp: channel[%u] has insufficient notifies", index), HCCL_E_INTERNAL);
        for (uint32_t prior = 0; prior < index; ++prior) {
            CHK_PRT_RET(resource.channels[prior].remoteRank == resource.channels[index].remoteRank,
                HCCL_ERROR("ExecOp: duplicate peer[%u] on channels[%u,%u]",
                    resource.channels[index].remoteRank, prior, index), HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateSmallResourceContract(const OpParam &param, const AlgResourceCtx &resource)
{
    CHK_PRT_RET(param.rankSize != ALLREDUCE_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("ExecOp: invalid 16-rank contract"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(resource.localBuffer.addr == nullptr || resource.minCclBufferBytes < FP32_BYTES,
        HCCL_ERROR("ExecOp: invalid small-message CCL buffer"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resource.threads.size() != ALLREDUCE_CHANNEL_COUNT ||
            resource.channels.size() != ALLREDUCE_CHANNEL_COUNT ||
            resource.aicpuThread == 0 || resource.threads[0] != resource.aicpuThread,
        HCCL_ERROR("ExecOp: invalid small-message resource counts"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

uint64_t GetMinCclBufferBytes(const AlgResourceCtx &resource)
{
    return resource.minCclBufferBytes;
}

uint64_t GetXorSlotBytes(const AlgResourceCtx &resource)
{
    return (GetMinCclBufferBytes(resource) / ALLREDUCE_STEP_COUNT) & ~(FP32_BYTES - 1);
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

LargeBufferLayout GetLargeBufferLayout(const AlgResourceCtx &resource)
{
    const uint64_t bufferElements = GetMinCclBufferBytes(resource) / FP32_BYTES;
    uint64_t chunkElements = bufferElements;
    auto requiredElements = [](uint64_t elements) {
        return CeilDiv(elements, 2) + CeilDiv(elements, 4) + 2 * CeilDiv(elements, 8);
    };
    while (chunkElements > 0 && requiredElements(chunkElements) > bufferElements) {
        --chunkElements;
    }

    const uint64_t halfBytes = CeilDiv(chunkElements, 2) * FP32_BYTES;
    const uint64_t quarterBytes = CeilDiv(chunkElements, 4) * FP32_BYTES;
    const uint64_t eighthBytes = CeilDiv(chunkElements, 8) * FP32_BYTES;
    LargeBufferLayout layout;
    layout.maxChunkBytes = chunkElements * FP32_BYTES;
    layout.regions[2] = {0, halfBytes, AliasGroup::NONE};
    layout.regions[1] = {halfBytes, quarterBytes, AliasGroup::NONE};
    layout.regions[0] = {halfBytes + quarterBytes, eighthBytes, AliasGroup::XOR_ONE_INTER};
    layout.regions[3] = {halfBytes + quarterBytes + eighthBytes, eighthBytes, AliasGroup::NONE};
    layout.scratch = layout.regions[0];
    return layout;
}

LargeBufferLayout GetAggressiveBufferLayout(const AlgResourceCtx &resource)
{
    const uint64_t bufferElements = GetMinCclBufferBytes(resource) / FP32_BYTES;
    uint64_t chunkElements = (bufferElements / 7) * 8 + ((bufferElements % 7) * 8) / 7;
    auto requiredElements = [](uint64_t elements) {
        return CeilDiv(elements, 2) + CeilDiv(elements, 4) + CeilDiv(elements, 8);
    };
    while (chunkElements > 0 && requiredElements(chunkElements) > bufferElements) {
        --chunkElements;
    }

    const uint64_t halfBytes = CeilDiv(chunkElements, 2) * FP32_BYTES;
    const uint64_t quarterBytes = CeilDiv(chunkElements, 4) * FP32_BYTES;
    const uint64_t eighthBytes = CeilDiv(chunkElements, 8) * FP32_BYTES;
    LargeBufferLayout layout;
    layout.maxChunkBytes = chunkElements * FP32_BYTES;
    layout.regions[2] = {0, halfBytes, AliasGroup::NONE};
    layout.regions[1] = {halfBytes, quarterBytes, AliasGroup::QUARTER_SCRATCH};
    layout.regions[0] = {halfBytes + quarterBytes, eighthBytes, AliasGroup::XOR_ONE_INTER};
    layout.regions[3] = layout.regions[0];
    layout.scratch = {layout.regions[1].offset, eighthBytes, AliasGroup::QUARTER_SCRATCH};
    return layout;
}

RadixBufferLayout GetRadixBufferLayout(const AlgResourceCtx &resource)
{
    const uint64_t bufferElements = GetMinCclBufferBytes(resource) / FP32_BYTES;
    uint64_t chunkElements = (bufferElements / 3) * 4 + ((bufferElements % 3) * 4) / 3;
    while (chunkElements > 0 && 3 * CeilDiv(chunkElements, 4) > bufferElements) {
        --chunkElements;
    }
    RadixBufferLayout layout;
    layout.maxChunkBytes = chunkElements * FP32_BYTES;
    layout.slotBytes = CeilDiv(chunkElements, 4) * FP32_BYTES;
    return layout;
}

FullMeshBufferLayout GetFullMeshBufferLayout(const AlgResourceCtx &resource, uint64_t elements,
    bool pipeline)
{
    FullMeshBufferLayout layout;
    layout.slotBytes = CeilDiv(elements, LOCAL_RANK_COUNT) * FP32_BYTES;
    const uint64_t bufferBytes = GetMinCclBufferBytes(resource);
    if (pipeline) {
        const uint64_t localBytes = (LOCAL_RANK_COUNT - 1) * layout.slotBytes;
        layout.crossOffset = localBytes;
        layout.scratchOffset = localBytes + layout.slotBytes;
        if (layout.scratchOffset + layout.slotBytes <= bufferBytes) {
            layout.waveCount = 1;
        }
        return layout;
    }
    layout.crossOffset = 0;
    layout.scratchOffset = 0;
    if ((LOCAL_RANK_COUNT - 1) * layout.slotBytes <= bufferBytes) {
        layout.waveCount = 1;
    } else if (4 * layout.slotBytes <= bufferBytes) {
        layout.waveCount = 2;
    }
    return layout;
}

AllReducePlan BuildAllReducePlan(const OpParam &param, const AlgResourceCtx &resource)
{
    AllReducePlan plan;
    plan.totalBytes = param.count * FP32_BYTES;
    plan.xorSlotBytes = GetXorSlotBytes(resource);
    plan.safe = GetLargeBufferLayout(resource);
    plan.fast = GetAggressiveBufferLayout(resource);
    plan.radix = GetRadixBufferLayout(resource);
    if (plan.totalBytes == FULLMESH_MESSAGE_400M4B) {
        plan.fullMesh = GetFullMeshBufferLayout(resource, param.count, false);
        if (plan.fullMesh.waveCount != 0) {
            plan.policy = ExecutionPolicy::LARGE_FULLMESH;
            return plan;
        }
    } else if (plan.totalBytes == FULLMESH_MESSAGE_512M) {
        const uint64_t maxTileElements = CeilDiv(param.count, 2);
        plan.fullMesh = GetFullMeshBufferLayout(resource, maxTileElements, true);
        if (ENABLE_FULLMESH_PIPELINE2 && plan.fullMesh.waveCount == 1) {
            plan.policy = ExecutionPolicy::LARGE_FULLMESH_PIPELINE2;
            return plan;
        }
        plan.fullMesh = GetFullMeshBufferLayout(resource, param.count, false);
        if (plan.fullMesh.waveCount != 0) {
            plan.policy = ExecutionPolicy::LARGE_FULLMESH;
            return plan;
        }
    }
    if (plan.totalBytes > SMALL_MESSAGE_BYTES && plan.totalBytes <= plan.fast.maxChunkBytes) {
        plan.policy = ExecutionPolicy::LARGE_FAST;
    } else if (plan.totalBytes > plan.fast.maxChunkBytes && plan.totalBytes <= plan.radix.maxChunkBytes) {
        plan.policy = ExecutionPolicy::LARGE_RADIX;
    } else if (plan.totalBytes <= SMALL_MESSAGE_BYTES || plan.safe.maxChunkBytes <= SMALL_MESSAGE_BYTES) {
        plan.policy = ExecutionPolicy::SMALL_XOR;
    }
    return plan;
}

uint32_t RadixSlot(uint32_t sourceDigit, uint32_t ownerDigit)
{
    return sourceDigit < ownerDigit ? sourceDigit : sourceDigit - 1;
}

uint64_t ElementBoundary(uint64_t elements, uint32_t segment)
{
    return (elements / LOCAL_RANK_COUNT) * segment +
        ((elements % LOCAL_RANK_COUNT) * segment) / LOCAL_RANK_COUNT;
}

HcclResult GetChannel(const OpParam &param, const ChannelMap &channels, uint32_t channelIndex,
    const ChannelInfo *&channel)
{
    (void)param;
    CHK_PRT_RET(channelIndex >= channels.size(),
        HCCL_ERROR("ExecOp: invalid channel index[%u]", channelIndex), HCCL_E_INTERNAL);
    channel = &channels[channelIndex];
    return HCCL_SUCCESS;
}

HcclResult GetPeerChannel(const OpParam &param, const ChannelMap &channels, uint32_t peer,
    const ChannelInfo *&channel, uint32_t &channelIndex)
{
    const uint32_t delta = param.myRank ^ peer;
    if (delta == 1) {
        channelIndex = 0;
    } else if (delta == 2) {
        channelIndex = 1;
    } else if (delta == 4) {
        channelIndex = 2;
    } else if (delta == 8) {
        channelIndex = 3;
    } else if (delta == 3) {
        channelIndex = 4;
    } else if (delta == 5) {
        channelIndex = 5;
    } else if (delta == 6) {
        channelIndex = 6;
    } else if (delta == 7) {
        channelIndex = 7;
    } else {
        HCCL_ERROR("ExecOp: unsupported peer delta[%u]", delta);
        return HCCL_E_INTERNAL;
    }
    return GetChannel(param, channels, channelIndex, channel);
}

HcclResult WriteAndWait(const AlgResourceCtx &resource, const ChannelInfo &channel, void *remoteSlot,
    const void *source, uint64_t bytes)
{
    CHK_RET(static_cast<HcclResult>(
        HcommWriteOnThread(resource.aicpuThread, channel.handle, remoteSlot, source, bytes)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.aicpuThread, channel.handle,
        NOTIFY.data)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread, channel.handle,
        NOTIFY.data, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult WriteAndRecordOnWorker(const AlgResourceCtx &resource, uint32_t workerIndex,
    const ChannelInfo &channel, void *remoteSlot, const void *source, uint64_t bytes)
{
    CHK_PRT_RET(workerIndex == 0 || workerIndex >= resource.threads.size(),
        HCCL_ERROR("ExecOp: invalid radix worker index"), HCCL_E_INTERNAL);
    const ThreadHandle worker = resource.threads[workerIndex];
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(resource.aicpuThread, worker, 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommWriteOnThread(worker, channel.handle, remoteSlot, source, bytes)));
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(worker, channel.handle,
        NOTIFY.data));
}

HcclResult WriteTwoAndRecordOnWorker(const AlgResourceCtx &resource, uint32_t workerIndex,
    const ChannelInfo &channel, void *firstRemoteSlot, const void *firstSource, uint64_t firstBytes,
    void *secondRemoteSlot, const void *secondSource, uint64_t secondBytes)
{
    CHK_PRT_RET(workerIndex == 0 || workerIndex >= resource.threads.size(),
        HCCL_ERROR("ExecOp: invalid two-span worker index"), HCCL_E_INTERNAL);
    const ThreadHandle worker = resource.threads[workerIndex];
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(resource.aicpuThread, worker, 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommWriteOnThread(worker, channel.handle, firstRemoteSlot, firstSource, firstBytes)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(worker, channel.handle,
        NOTIFY.data)));
    CHK_RET(static_cast<HcclResult>(
        HcommWriteOnThread(worker, channel.handle, secondRemoteSlot, secondSource, secondBytes)));
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(worker, channel.handle,
        NOTIFY.data));
}

HcclResult WaitData(const AlgResourceCtx &resource, const ChannelInfo &channel)
{
    return static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread, channel.handle,
        NOTIFY.data, CUSTOM_TIMEOUT));
}

HcclResult ReceiveCopyAndAckOnWorker(const AlgResourceCtx &resource, uint32_t workerIndex,
    const ChannelInfo &channel, void *destination, const void *source, uint64_t bytes)
{
    CHK_PRT_RET(workerIndex == 0 || workerIndex >= resource.threads.size(),
        HCCL_ERROR("ExecOp: invalid all-gather worker index"), HCCL_E_INTERNAL);
    const ThreadHandle worker = resource.threads[workerIndex];
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(worker, channel.handle,
        NOTIFY.data, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(worker, destination, source, bytes)));
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(worker, channel.handle,
        NOTIFY.ack));
}

HcclResult WaitPendingAck(const AlgResourceCtx &resource, const ChannelInfo &channel, bool &pending)
{
    if (pending) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread, channel.handle,
            NOTIFY.ack, CUSTOM_TIMEOUT)));
        pending = false;
    }
    return HCCL_SUCCESS;
}

HcclResult RecordAck(const AlgResourceCtx &resource, const ChannelInfo &channel, bool &pending)
{
    CHK_PRT_RET(pending, HCCL_ERROR("ExecOp: ack notify reused before wait"), HCCL_E_INTERNAL);
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.aicpuThread, channel.handle,
        NOTIFY.ack)));
    pending = true;
    return HCCL_SUCCESS;
}

HcclResult WaitAllPendingAcks(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, PendingAcks &pendingAcks)
{
    for (uint32_t channelIndex = 0; channelIndex < ALLREDUCE_CHANNEL_COUNT; ++channelIndex) {
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(param, channels, channelIndex, channel));
        CHK_RET(WaitPendingAck(resource, *channel, pendingAcks[channelIndex]));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitXorPendingAcks(const AlgResourceCtx &resource, const ChannelMap &channels,
    PendingAcks &pendingAcks)
{
    for (const uint32_t channelIndex : XOR_CHANNELS) {
        CHK_RET(WaitPendingAck(resource, channels[channelIndex], pendingAcks[channelIndex]));
    }
    return HCCL_SUCCESS;
}

uint32_t FullMeshSlot(uint32_t sourceRank, uint32_t ownerRank, uint32_t sourceBegin,
    uint32_t sourceEnd)
{
    if (sourceEnd - sourceBegin == LOCAL_RANK_COUNT) {
        return sourceRank < ownerRank ? sourceRank : sourceRank - 1;
    }
    return sourceRank - sourceBegin;
}

HcclResult WaitLocalFlight(const AlgResourceCtx &resource, const ChannelInfo &channel, bool &pending)
{
    if (pending) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread,
            channel.handle, NOTIFY.ack, CUSTOM_TIMEOUT)));
        pending = false;
    }
    return HCCL_SUCCESS;
}

HcclResult DrainLocalFlights(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, LocalFlights &flights)
{
    for (const uint32_t channelIndex : LOCAL_PEER_CHANNELS) {
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(param, channels, channelIndex, channel));
        CHK_RET(WaitLocalFlight(resource, *channel, flights[channelIndex]));
    }
    return HCCL_SUCCESS;
}

HcclResult RecordReceiveAck(const AlgResourceCtx &resource, const ChannelInfo &channel)
{
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.aicpuThread,
        channel.handle, NOTIFY.ack));
}

SegmentBoundaries GetSegmentBoundaries(uint64_t elements)
{
    SegmentBoundaries boundaries = {};
    for (uint32_t index = 0; index <= LOCAL_RANK_COUNT; ++index) {
        boundaries[index] = ElementBoundary(elements, index);
    }
    return boundaries;
}

HcclResult LaunchReduceScatterWave(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, LocalFlights &flights, const FullMeshBufferLayout &layout,
    uint64_t tileOffset, const SegmentBoundaries &boundaries, uint32_t sourceBegin,
    uint32_t sourceEnd)
{
    const uint32_t localRank = param.myRank & (LOCAL_RANK_COUNT - 1);
    if (localRank < sourceBegin || localRank >= sourceEnd) {
        return HCCL_SUCCESS;
    }
    const uint32_t serverBase = param.myRank & ~(LOCAL_RANK_COUNT - 1);
    const auto *input = static_cast<const uint8_t *>(param.inputPtr) + tileOffset;
    uint32_t workerIndex = 1;
    for (uint32_t ownerRank = 0; ownerRank < LOCAL_RANK_COUNT; ++ownerRank) {
        if (ownerRank == localRank) {
            continue;
        }
        const uint32_t peer = serverBase | ownerRank;
        const ChannelInfo *channel = nullptr;
        uint32_t channelIndex = 0;
        CHK_RET(GetPeerChannel(param, channels, peer, channel, channelIndex));
        CHK_RET(WaitLocalFlight(resource, *channel, flights[channelIndex]));
        const uint64_t segmentOffset = boundaries[ownerRank] * FP32_BYTES;
        const uint64_t segmentBytes =
            (boundaries[ownerRank + 1] - boundaries[ownerRank]) * FP32_BYTES;
        const uint32_t slot = FullMeshSlot(localRank, ownerRank, sourceBegin, sourceEnd);
        auto *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) + slot * layout.slotBytes;
        CHK_RET(WriteAndRecordOnWorker(resource, workerIndex, *channel, remoteSlot,
            input + segmentOffset, segmentBytes));
        flights[channelIndex] = true;
        ++workerIndex;
    }
    return HCCL_SUCCESS;
}

HcclResult FinishReduceScatterWave(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, LocalFlights &flights, const FullMeshBufferLayout &layout,
    uint64_t tileOffset, const SegmentBoundaries &boundaries, uint32_t sourceBegin,
    uint32_t sourceEnd, bool &initialized)
{
    const uint32_t localRank = param.myRank & (LOCAL_RANK_COUNT - 1);
    const uint32_t serverBase = param.myRank & ~(LOCAL_RANK_COUNT - 1);
    auto *accumulator = static_cast<uint8_t *>(param.outputPtr) + tileOffset +
        boundaries[localRank] * FP32_BYTES;
    const auto *input = static_cast<const uint8_t *>(param.inputPtr) + tileOffset +
        boundaries[localRank] * FP32_BYTES;
    const uint64_t segmentBytes =
        (boundaries[localRank + 1] - boundaries[localRank]) * FP32_BYTES;
    for (uint32_t sourceRank = sourceBegin; sourceRank < sourceEnd; ++sourceRank) {
        const void *contribution = input;
        const ChannelInfo *channel = nullptr;
        if (sourceRank != localRank) {
            const uint32_t peer = serverBase | sourceRank;
            uint32_t channelIndex = 0;
            CHK_RET(GetPeerChannel(param, channels, peer, channel, channelIndex));
            CHK_RET(WaitData(resource, *channel));
            const uint32_t slot = FullMeshSlot(sourceRank, localRank, sourceBegin, sourceEnd);
            contribution = static_cast<uint8_t *>(resource.localBuffer.addr) + slot * layout.slotBytes;
        }
        if (!initialized) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
                accumulator, contribution, segmentBytes)));
            initialized = true;
        } else {
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
                accumulator, contribution, segmentBytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32,
                HCOMM_REDUCE_SUM)));
        }
        if (channel != nullptr) {
            CHK_RET(RecordReceiveAck(resource, *channel));
        }
    }
    return DrainLocalFlights(param, resource, channels, flights);
}

HcclResult FinishReduceScatterWaveSplit(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, LocalFlights &flights, const FullMeshBufferLayout &layout,
    uint64_t tileOffset, const SegmentBoundaries &boundaries)
{
    const uint32_t localRank = param.myRank & (LOCAL_RANK_COUNT - 1);
    const uint32_t serverBase = param.myRank & ~(LOCAL_RANK_COUNT - 1);
    auto *accumulator = static_cast<uint8_t *>(param.outputPtr) + tileOffset +
        boundaries[localRank] * FP32_BYTES;
    const auto *input = static_cast<const uint8_t *>(param.inputPtr) + tileOffset +
        boundaries[localRank] * FP32_BYTES;
    const uint64_t segmentElements = boundaries[localRank + 1] - boundaries[localRank];

    for (uint32_t sourceRank = 0; sourceRank < LOCAL_RANK_COUNT; ++sourceRank) {
        if (sourceRank == localRank) {
            continue;
        }
        const ChannelInfo *channel = nullptr;
        uint32_t channelIndex = 0;
        CHK_RET(GetPeerChannel(param, channels, serverBase | sourceRank, channel, channelIndex));
        CHK_RET(WaitData(resource, *channel));
    }

    const void *firstContribution = input;
    if (localRank != 0) {
        const uint32_t firstSlot = FullMeshSlot(0, localRank, 0, LOCAL_RANK_COUNT);
        firstContribution = static_cast<uint8_t *>(resource.localBuffer.addr) +
            firstSlot * layout.slotBytes;
    }
    for (uint32_t workerIndex = 1; workerIndex <= PIPELINE_RS_SPLIT_WORKERS; ++workerIndex) {
        const ThreadHandle worker = resource.threads[workerIndex];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resource.aicpuThread, worker, 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT)));
        const uint64_t sliceBegin = segmentElements * workerIndex /
            PIPELINE_RS_SPLIT_SLICES;
        const uint64_t sliceEnd = segmentElements * (workerIndex + 1) /
            PIPELINE_RS_SPLIT_SLICES;
        const uint64_t sliceOffset = sliceBegin * FP32_BYTES;
        const uint64_t sliceElements = sliceEnd - sliceBegin;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(worker,
            accumulator + sliceOffset,
            static_cast<const uint8_t *>(firstContribution) + sliceOffset,
            sliceElements * FP32_BYTES)));
        for (uint32_t sourceRank = 1; sourceRank < LOCAL_RANK_COUNT; ++sourceRank) {
            const uint8_t *contribution = input;
            if (sourceRank != localRank) {
                const uint32_t slot = FullMeshSlot(sourceRank, localRank, 0, LOCAL_RANK_COUNT);
                contribution = static_cast<uint8_t *>(resource.localBuffer.addr) +
                    slot * layout.slotBytes;
            }
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(worker,
                accumulator + sliceOffset, contribution + sliceOffset, sliceElements,
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
    }

    const uint64_t mainSliceElements = segmentElements / PIPELINE_RS_SPLIT_SLICES;
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
        accumulator, firstContribution, mainSliceElements * FP32_BYTES)));
    for (uint32_t sourceRank = 1; sourceRank < LOCAL_RANK_COUNT; ++sourceRank) {
        const uint8_t *contribution = input;
        if (sourceRank != localRank) {
            const uint32_t slot = FullMeshSlot(sourceRank, localRank, 0, LOCAL_RANK_COUNT);
            contribution = static_cast<uint8_t *>(resource.localBuffer.addr) +
                slot * layout.slotBytes;
        }
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
            accumulator, contribution, mainSliceElements, HCOMM_DATA_TYPE_FP32,
            HCOMM_REDUCE_SUM)));
    }

    for (uint32_t workerIndex = PIPELINE_RS_SPLIT_WORKERS; workerIndex > 1; --workerIndex) {
        const ThreadHandle currentWorker = resource.threads[workerIndex];
        const ThreadHandle previousWorker = resource.threads[workerIndex - 1];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(currentWorker, previousWorker, 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(previousWorker, 0, CUSTOM_TIMEOUT)));
    }
    const ThreadHandle worker1 = resource.threads[1];
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(worker1, resource.aicpuThread, 0)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(resource.aicpuThread, 0, CUSTOM_TIMEOUT)));

    for (uint32_t sourceRank = 0; sourceRank < LOCAL_RANK_COUNT; ++sourceRank) {
        if (sourceRank == localRank) {
            continue;
        }
        const ChannelInfo *channel = nullptr;
        uint32_t channelIndex = 0;
        CHK_RET(GetPeerChannel(param, channels, serverBase | sourceRank, channel, channelIndex));
        CHK_RET(RecordReceiveAck(resource, *channel));
    }
    return DrainLocalFlights(param, resource, channels, flights);
}

HcclResult LocalReadyBarrier(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels);

HcclResult ExecReduceScatterTile(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, LocalFlights &flights, const FullMeshBufferLayout &layout,
    uint64_t tileOffset, const SegmentBoundaries &boundaries)
{
    bool initialized = false;
    for (uint32_t wave = 0; wave < layout.waveCount; ++wave) {
        const uint32_t sourceBegin = layout.waveCount == 1 ? 0 : wave * 4;
        const uint32_t sourceEnd = layout.waveCount == 1 ? LOCAL_RANK_COUNT : sourceBegin + 4;
        CHK_RET(LaunchReduceScatterWave(param, resource, channels, flights, layout, tileOffset,
            boundaries, sourceBegin, sourceEnd));
        CHK_RET(FinishReduceScatterWave(param, resource, channels, flights, layout, tileOffset,
            boundaries, sourceBegin, sourceEnd, initialized));
        if (wave + 1 < layout.waveCount) {
            CHK_RET(LocalReadyBarrier(param, resource, channels));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExchangeCrossServerSegment(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, const FullMeshBufferLayout &layout, uint64_t tileOffset,
    const SegmentBoundaries &boundaries, bool requireAck = true)
{
    const uint32_t localRank = param.myRank & (LOCAL_RANK_COUNT - 1);
    auto *accumulator = static_cast<uint8_t *>(param.outputPtr) + tileOffset +
        boundaries[localRank] * FP32_BYTES;
    const uint64_t segmentBytes =
        (boundaries[localRank + 1] - boundaries[localRank]) * FP32_BYTES;
    const ChannelInfo *channel = nullptr;
    CHK_RET(GetChannel(param, channels, 3, channel));
    if (layout.crossOffset == 0) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.aicpuThread,
            channel->handle, NOTIFY.interReady)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread,
            channel->handle, NOTIFY.interReady, CUSTOM_TIMEOUT)));
    }
    auto *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) + layout.crossOffset;
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(resource.aicpuThread, channel->handle,
        remoteSlot, accumulator, segmentBytes)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.aicpuThread,
        channel->handle, NOTIFY.data)));
    auto *localSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + layout.crossOffset;
    CHK_RET(WaitData(resource, *channel));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
        accumulator, localSlot, segmentBytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32,
        HCOMM_REDUCE_SUM)));
    if (!requireAck) {
        return HCCL_SUCCESS;
    }
    CHK_RET(RecordReceiveAck(resource, *channel));
    return static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread,
        channel->handle, NOTIFY.ack, CUSTOM_TIMEOUT));
}

HcclResult ExchangeCrossServerSegmentSplit(const OpParam &param,
    const AlgResourceCtx &resource, const ChannelMap &channels,
    const FullMeshBufferLayout &layout, uint64_t tileOffset,
    const SegmentBoundaries &boundaries, bool requireAck = true)
{
    const uint32_t localRank = param.myRank & (LOCAL_RANK_COUNT - 1);
    auto *accumulator = static_cast<uint8_t *>(param.outputPtr) + tileOffset +
        boundaries[localRank] * FP32_BYTES;
    const uint64_t segmentElements = boundaries[localRank + 1] - boundaries[localRank];
    const uint64_t segmentBytes = segmentElements * FP32_BYTES;
    const ChannelInfo *channel = nullptr;
    CHK_RET(GetChannel(param, channels, 3, channel));
    if (layout.crossOffset == 0) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.aicpuThread,
            channel->handle, NOTIFY.interReady)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread,
            channel->handle, NOTIFY.interReady, CUSTOM_TIMEOUT)));
    }
    auto *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) + layout.crossOffset;
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(resource.aicpuThread, channel->handle,
        remoteSlot, accumulator, segmentBytes)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.aicpuThread,
        channel->handle, NOTIFY.data)));
    CHK_RET(WaitData(resource, *channel));

    auto *localSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + layout.crossOffset;
    for (uint32_t workerIndex = 1; workerIndex <= PIPELINE_RS_SPLIT_WORKERS; ++workerIndex) {
        const ThreadHandle worker = resource.threads[workerIndex];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resource.aicpuThread, worker, 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT)));
        const uint64_t sliceBegin = segmentElements * workerIndex /
            PIPELINE_RS_SPLIT_SLICES;
        const uint64_t sliceEnd = segmentElements * (workerIndex + 1) /
            PIPELINE_RS_SPLIT_SLICES;
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(worker,
            accumulator + sliceBegin * FP32_BYTES,
            localSlot + sliceBegin * FP32_BYTES, sliceEnd - sliceBegin,
            HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    const uint64_t mainSliceElements = segmentElements / PIPELINE_RS_SPLIT_SLICES;
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
        accumulator, localSlot, mainSliceElements, HCOMM_DATA_TYPE_FP32,
        HCOMM_REDUCE_SUM)));
    for (uint32_t workerIndex = PIPELINE_RS_SPLIT_WORKERS; workerIndex > 1; --workerIndex) {
        const ThreadHandle currentWorker = resource.threads[workerIndex];
        const ThreadHandle previousWorker = resource.threads[workerIndex - 1];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(currentWorker, previousWorker, 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(previousWorker, 0, CUSTOM_TIMEOUT)));
    }
    const ThreadHandle worker1 = resource.threads[1];
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(worker1, resource.aicpuThread, 0)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(resource.aicpuThread, 0, CUSTOM_TIMEOUT)));

    if (!requireAck) {
        return HCCL_SUCCESS;
    }
    CHK_RET(RecordReceiveAck(resource, *channel));
    return static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread,
        channel->handle, NOTIFY.ack, CUSTOM_TIMEOUT));
}

HcclResult LocalReadyBarrier(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels)
{
    for (const uint32_t channelIndex : LOCAL_PEER_CHANNELS) {
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(param, channels, channelIndex, channel));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.aicpuThread,
            channel->handle, NOTIFY.interReady)));
    }
    for (const uint32_t channelIndex : LOCAL_PEER_CHANNELS) {
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(param, channels, channelIndex, channel));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread,
            channel->handle, NOTIFY.interReady, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchAllGatherWave(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, LocalFlights &flights, const FullMeshBufferLayout &layout,
    uint64_t tileOffset, const SegmentBoundaries &boundaries, uint32_t sourceBegin,
    uint32_t sourceEnd)
{
    const uint32_t localRank = param.myRank & (LOCAL_RANK_COUNT - 1);
    if (localRank < sourceBegin || localRank >= sourceEnd) {
        return HCCL_SUCCESS;
    }
    const uint32_t serverBase = param.myRank & ~(LOCAL_RANK_COUNT - 1);
    const auto *source = static_cast<const uint8_t *>(param.outputPtr) + tileOffset +
        boundaries[localRank] * FP32_BYTES;
    const uint64_t segmentBytes =
        (boundaries[localRank + 1] - boundaries[localRank]) * FP32_BYTES;
    uint32_t workerIndex = 1;
    for (uint32_t destinationRank = 0; destinationRank < LOCAL_RANK_COUNT; ++destinationRank) {
        if (destinationRank == localRank) {
            continue;
        }
        const uint32_t peer = serverBase | destinationRank;
        const ChannelInfo *channel = nullptr;
        uint32_t channelIndex = 0;
        CHK_RET(GetPeerChannel(param, channels, peer, channel, channelIndex));
        CHK_RET(WaitLocalFlight(resource, *channel, flights[channelIndex]));
        const uint32_t slot = FullMeshSlot(localRank, destinationRank, sourceBegin, sourceEnd);
        auto *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) + slot * layout.slotBytes;
        CHK_RET(WriteAndRecordOnWorker(resource, workerIndex, *channel, remoteSlot, source,
            segmentBytes));
        flights[channelIndex] = true;
        ++workerIndex;
    }
    return HCCL_SUCCESS;
}

HcclResult FinishAllGatherWave(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, LocalFlights &flights, const FullMeshBufferLayout &layout,
    uint64_t tileOffset, const SegmentBoundaries &boundaries, uint32_t sourceBegin,
    uint32_t sourceEnd)
{
    const uint32_t localRank = param.myRank & (LOCAL_RANK_COUNT - 1);
    const uint32_t serverBase = param.myRank & ~(LOCAL_RANK_COUNT - 1);
    auto *output = static_cast<uint8_t *>(param.outputPtr) + tileOffset;
    uint32_t workerIndex = 1;
    for (uint32_t sourceRank = sourceBegin; sourceRank < sourceEnd; ++sourceRank) {
        if (sourceRank == localRank) {
            continue;
        }
        const uint32_t peer = serverBase | sourceRank;
        const ChannelInfo *channel = nullptr;
        uint32_t channelIndex = 0;
        CHK_RET(GetPeerChannel(param, channels, peer, channel, channelIndex));
        const uint32_t slot = FullMeshSlot(sourceRank, localRank, sourceBegin, sourceEnd);
        const void *localSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + slot * layout.slotBytes;
        const uint64_t segmentBytes =
            (boundaries[sourceRank + 1] - boundaries[sourceRank]) * FP32_BYTES;
        CHK_RET(ReceiveCopyAndAckOnWorker(resource, workerIndex, *channel,
            output + boundaries[sourceRank] * FP32_BYTES, localSlot, segmentBytes));
        ++workerIndex;
    }
    return DrainLocalFlights(param, resource, channels, flights);
}

HcclResult ExecAllGatherTile(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, LocalFlights &flights, const FullMeshBufferLayout &layout,
    uint64_t tileOffset, const SegmentBoundaries &boundaries)
{
    for (uint32_t wave = 0; wave < layout.waveCount; ++wave) {
        const uint32_t sourceBegin = layout.waveCount == 1 ? 0 : wave * 4;
        const uint32_t sourceEnd = layout.waveCount == 1 ? LOCAL_RANK_COUNT : sourceBegin + 4;
        CHK_RET(LaunchAllGatherWave(param, resource, channels, flights, layout, tileOffset,
            boundaries, sourceBegin, sourceEnd));
        CHK_RET(FinishAllGatherWave(param, resource, channels, flights, layout, tileOffset,
            boundaries, sourceBegin, sourceEnd));
        if (wave + 1 < layout.waveCount) {
            CHK_RET(LocalReadyBarrier(param, resource, channels));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecLargeFullMesh(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, const FullMeshBufferLayout &layout)
{
    CHK_PRT_RET(resource.threads.size() < FULLMESH_THREAD_COUNT,
        HCCL_ERROR("ExecOp: fullmesh workers are missing"), HCCL_E_INTERNAL);
    LocalFlights flights = {};
    const SegmentBoundaries boundaries = GetSegmentBoundaries(param.count);
    CHK_RET(ExecReduceScatterTile(param, resource, channels, flights, layout, 0, boundaries));
    CHK_RET(ExchangeCrossServerSegment(param, resource, channels, layout, 0, boundaries));
    // The non-pipelined layout aliases the cross-server receive with local slot 0.  No local
    // rank may start AG until all eight ranks have finished their independent XOR8 merge.
    CHK_RET(LocalReadyBarrier(param, resource, channels));
    return ExecAllGatherTile(param, resource, channels, flights, layout, 0, boundaries);
}

HcclResult ExecLargeFullMeshPipeline2(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, const FullMeshBufferLayout &layout)
{
    CHK_PRT_RET(resource.threads.size() < FULLMESH_THREAD_COUNT || layout.waveCount != 1,
        HCCL_ERROR("ExecOp: pipeline2 resource contract is not satisfied"), HCCL_E_INTERNAL);
    const uint64_t tileElements[2] = {param.count / 2, param.count - param.count / 2};
    const uint64_t tileOffsets[2] = {0, tileElements[0] * FP32_BYTES};
    const SegmentBoundaries boundaries[2] = {
        GetSegmentBoundaries(tileElements[0]), GetSegmentBoundaries(tileElements[1])};
    LocalFlights flights = {};

    // Complete local RS(tile0). Every spatial slice preserves source order.
    CHK_RET(LaunchReduceScatterWave(param, resource, channels, flights, layout,
        tileOffsets[0], boundaries[0], 0, LOCAL_RANK_COUNT));
    CHK_RET(FinishReduceScatterWaveSplit(param, resource, channels, flights, layout,
        tileOffsets[0], boundaries[0]));

    // RS(tile1) transfers run on local-peer workers while XOR8 handles tile0.
    CHK_RET(LaunchReduceScatterWave(param, resource, channels, flights, layout,
        tileOffsets[1], boundaries[1], 0, LOCAL_RANK_COUNT));
    CHK_RET(ExchangeCrossServerSegmentSplit(param, resource, channels, layout,
        tileOffsets[0], boundaries[0]));
    CHK_RET(FinishReduceScatterWaveSplit(param, resource, channels, flights, layout,
        tileOffsets[1], boundaries[1]));

    // AG(tile0) runs on local-peer workers while XOR8 handles tile1.
    CHK_RET(LaunchAllGatherWave(param, resource, channels, flights, layout,
        tileOffsets[0], boundaries[0], 0, LOCAL_RANK_COUNT));
    CHK_RET(ExchangeCrossServerSegmentSplit(param, resource, channels, layout,
        tileOffsets[1], boundaries[1], false));
    CHK_RET(FinishAllGatherWave(param, resource, channels, flights, layout,
        tileOffsets[0], boundaries[0], 0, LOCAL_RANK_COUNT));

    return ExecAllGatherTile(param, resource, channels, flights, layout,
        tileOffsets[1], boundaries[1]);
}

// C1: skip redundant copy when sendBuf == recvBuf (in-place AllReduce).
// C3: precompute ElementBoundary boundaries into a local array.
// Opt4: defer RecordAck to after all 4 steps (slots are independent).
HcclResult ExecSmallXorFallback(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels,
    PendingAcks &pendingAcks, uint64_t slotBytes, uint64_t offset, uint64_t bytes)
{
    void *accumulator = static_cast<uint8_t *>(param.outputPtr) + offset;
    const void *input = static_cast<const uint8_t *>(param.inputPtr) + offset;
    if (input != accumulator) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resource.aicpuThread, accumulator, input, bytes)));
    }
    for (const uint32_t channelIndex : XOR_CHANNELS) {
        const ChannelInfo &channel = channels[channelIndex];
        CHK_RET(WaitPendingAck(resource, channel, pendingAcks[channelIndex]));
        void *localSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + channelIndex * slotBytes;
        void *remoteSlot = static_cast<uint8_t *>(channel.remoteCclMem.addr) + channelIndex * slotBytes;
        CHK_RET(WriteAndWait(resource, channel, remoteSlot, accumulator, bytes));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread, accumulator, localSlot,
            bytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    // Opt4: batch ACKs after all 4 steps — slots are independent, no ordering needed.
    for (const uint32_t channelIndex : XOR_CHANNELS) {
        CHK_RET(RecordAck(resource, channels[channelIndex], pendingAcks[channelIndex]));
    }
    return HCCL_SUCCESS;
}

// C3: precompute ElementBoundary boundaries into a local array.
HcclResult ExecHierarchicalChunk(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, PendingAcks &pendingAcks, const LargeBufferLayout &layout, uint64_t offset,
    uint64_t bytes)
{
    const uint64_t elements = bytes / FP32_BYTES;
    uint64_t boundaries[LOCAL_RANK_COUNT + 1];
    for (uint32_t i = 0; i <= LOCAL_RANK_COUNT; ++i) {
        boundaries[i] = ElementBoundary(elements, i);
    }
    auto *accumulator = static_cast<uint8_t *>(param.outputPtr) + offset;
    const auto *input = static_cast<const uint8_t *>(param.inputPtr) + offset;
    const uint32_t localRank = param.myRank & (LOCAL_RANK_COUNT - 1);
    if (input != accumulator) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resource.aicpuThread, accumulator, input, bytes)));
    }

    uint32_t groupStart = 0;
    uint32_t groupEnd = LOCAL_RANK_COUNT;
    for (int32_t channelIndex = 2; channelIndex >= 0; --channelIndex) {
        const uint32_t mask = 1U << channelIndex;
        const uint32_t middle = (groupStart + groupEnd) / 2;
        const bool keepLower = (localRank & mask) == 0;
        const uint32_t keepStart = keepLower ? groupStart : middle;
        const uint32_t keepEnd = keepLower ? middle : groupEnd;
        const uint32_t sendStart = keepLower ? middle : groupStart;
        const uint32_t sendEnd = keepLower ? groupEnd : middle;
        const uint64_t sendOffset = boundaries[sendStart] * FP32_BYTES;
        const uint64_t sendBytes =
            (boundaries[sendEnd] - boundaries[sendStart]) * FP32_BYTES;
        const uint64_t keepOffset = boundaries[keepStart] * FP32_BYTES;
        const uint64_t keepBytes =
            (boundaries[keepEnd] - boundaries[keepStart]) * FP32_BYTES;

        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(param, channels, static_cast<uint32_t>(channelIndex), channel));
        const uint32_t index = static_cast<uint32_t>(channelIndex);
        CHK_RET(WaitPendingAck(resource, *channel, pendingAcks[index]));
        const BufferRegion &region = layout.regions[index];
        CHK_PRT_RET(sendBytes > region.capacity || keepBytes > region.capacity,
            HCCL_ERROR("ExecOp: reduce-scatter transfer exceeds ccl region"), HCCL_E_INTERNAL);
        auto *localSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + region.offset;
        auto *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) + region.offset;
        CHK_RET(WriteAndWait(resource, *channel, remoteSlot, accumulator + sendOffset, sendBytes));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
            accumulator + keepOffset, localSlot, keepBytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32,
            HCOMM_REDUCE_SUM)));
        if (index != 0) {
            CHK_RET(RecordAck(resource, *channel, pendingAcks[index]));
        }
        groupStart = keepStart;
        groupEnd = keepEnd;
    }

    const uint64_t segmentOffset = boundaries[localRank] * FP32_BYTES;
    const uint64_t segmentBytes =
        (boundaries[localRank + 1] - boundaries[localRank]) * FP32_BYTES;
    const ChannelInfo *interChannel = nullptr;
    CHK_RET(GetChannel(param, channels, 3, interChannel));
    CHK_RET(WaitPendingAck(resource, *interChannel, pendingAcks[3]));
    const BufferRegion &interRegion = layout.regions[3];
    CHK_PRT_RET(segmentBytes > interRegion.capacity || segmentBytes > layout.scratch.capacity,
        HCCL_ERROR("ExecOp: cross-server segment exceeds ccl region"), HCCL_E_INTERNAL);
    void *interSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + interRegion.offset;
    void *remoteInterSlot = static_cast<uint8_t *>(interChannel->remoteCclMem.addr) + interRegion.offset;
    CHK_RET(WriteAndWait(resource, *interChannel, remoteInterSlot, accumulator + segmentOffset, segmentBytes));

    // Opt3: Server 0 merges directly into acc (no scratch copy); Server 1 needs scratch.
    void *scratch = static_cast<uint8_t *>(resource.localBuffer.addr) + layout.scratch.offset;
    const bool serverZero = param.myRank < LOCAL_RANK_COUNT;
    if (serverZero) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
            accumulator + segmentOffset, interSlot, segmentBytes / FP32_BYTES,
            HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    } else {
        const void *firstPartial = interSlot;
        const void *secondPartial = static_cast<void *>(accumulator + segmentOffset);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resource.aicpuThread, scratch, firstPartial, segmentBytes)));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread, scratch, secondPartial,
            segmentBytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
            accumulator + segmentOffset, scratch, segmentBytes)));
    }
    // XOR-1 receive storage is the scratch region; release it only after the fixed-order merge.
    const ChannelInfo *xorOneChannel = nullptr;
    CHK_RET(GetChannel(param, channels, 0, xorOneChannel));
    CHK_RET(RecordAck(resource, *xorOneChannel, pendingAcks[0]));

    groupStart = localRank;
    groupEnd = localRank + 1;
    for (uint32_t channelIndex = 0; channelIndex < 3; ++channelIndex) {
        const uint32_t mask = 1U << channelIndex;
        const uint32_t width = groupEnd - groupStart;
        const bool receiveUpper = (localRank & mask) == 0;
        const uint32_t receiveStart = receiveUpper ? groupEnd : groupStart - width;
        const uint32_t receiveEnd = receiveUpper ? groupEnd + width : groupStart;
        const uint64_t sendOffset = boundaries[groupStart] * FP32_BYTES;
        const uint64_t sendBytes =
            (boundaries[groupEnd] - boundaries[groupStart]) * FP32_BYTES;
        const uint64_t receiveOffset = boundaries[receiveStart] * FP32_BYTES;
        const uint64_t receiveBytes =
            (boundaries[receiveEnd] - boundaries[receiveStart]) * FP32_BYTES;

        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(param, channels, channelIndex, channel));
        CHK_RET(WaitPendingAck(resource, *channel, pendingAcks[channelIndex]));
        const BufferRegion &region = layout.regions[channelIndex];
        CHK_PRT_RET(sendBytes > region.capacity || receiveBytes > region.capacity,
            HCCL_ERROR("ExecOp: allgather transfer exceeds ccl region"), HCCL_E_INTERNAL);
        void *localSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + region.offset;
        void *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) + region.offset;
        CHK_RET(WriteAndWait(resource, *channel, remoteSlot, accumulator + sendOffset, sendBytes));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
            accumulator + receiveOffset, localSlot, receiveBytes)));
        groupStart = std::min(groupStart, receiveStart);
        groupEnd = std::max(groupEnd, receiveEnd);
    }
    // These ACKs release the aliased hierarchical layout only after every AllGather read completes.
    for (uint32_t channelIndex = 0; channelIndex < 3; ++channelIndex) {
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(param, channels, channelIndex, channel));
        CHK_RET(RecordAck(resource, *channel, pendingAcks[channelIndex]));
    }
    CHK_RET(RecordAck(resource, *interChannel, pendingAcks[3]));
    return HCCL_SUCCESS;
}

// C3: precompute ElementBoundary boundaries.
HcclResult ExecAggressiveChunk(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelMap &channels, PendingAcks &pendingAcks, const LargeBufferLayout &layout, uint64_t bytes)
{
    CHK_PRT_RET(resource.threads.size() < 2,
        HCCL_ERROR("ExecOp: aggressive pipeline worker is missing"), HCCL_E_INTERNAL);
    const uint64_t elements = bytes / FP32_BYTES;
    uint64_t boundaries[LOCAL_RANK_COUNT + 1];
    for (uint32_t i = 0; i <= LOCAL_RANK_COUNT; ++i) {
        boundaries[i] = ElementBoundary(elements, i);
    }
    auto *accumulator = static_cast<uint8_t *>(param.outputPtr);
    const auto *input = static_cast<const uint8_t *>(param.inputPtr);
    const uint32_t localRank = param.myRank & (LOCAL_RANK_COUNT - 1);

    uint32_t groupStart = 0;
    uint32_t groupEnd = LOCAL_RANK_COUNT;
    for (int32_t channelIndex = 2; channelIndex >= 0; --channelIndex) {
        const uint32_t index = static_cast<uint32_t>(channelIndex);
        const uint32_t mask = 1U << index;
        const uint32_t middle = (groupStart + groupEnd) / 2;
        const bool keepLower = (localRank & mask) == 0;
        const uint32_t keepStart = keepLower ? groupStart : middle;
        const uint32_t keepEnd = keepLower ? middle : groupEnd;
        const uint32_t sendStart = keepLower ? middle : groupStart;
        const uint32_t sendEnd = keepLower ? groupEnd : middle;
        const uint64_t sendOffset = boundaries[sendStart] * FP32_BYTES;
        const uint64_t sendBytes =
            (boundaries[sendEnd] - boundaries[sendStart]) * FP32_BYTES;
        const uint64_t keepOffset = boundaries[keepStart] * FP32_BYTES;
        const uint64_t keepBytes =
            (boundaries[keepEnd] - boundaries[keepStart]) * FP32_BYTES;

        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(param, channels, index, channel));
        CHK_RET(WaitPendingAck(resource, *channel, pendingAcks[index]));
        const BufferRegion &region = layout.regions[index];
        CHK_PRT_RET(sendBytes > region.capacity || keepBytes > region.capacity,
            HCCL_ERROR("ExecOp: aggressive reduce-scatter exceeds ccl region"), HCCL_E_INTERNAL);
        auto *localSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + region.offset;
        void *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) + region.offset;
        if (index == 2 && input != accumulator) {
            // The first exchange sends one half directly from input. Stage only
            // the surviving half on the coordinator while worker 1 sends its peer half.
            CHK_RET(WriteAndRecordOnWorker(resource, 1, *channel,
                remoteSlot, input + sendOffset, sendBytes));
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
                accumulator + keepOffset, input + keepOffset, keepBytes)));
            CHK_RET(WaitData(resource, *channel));
        } else {
            CHK_RET(WriteAndWait(resource, *channel, remoteSlot, accumulator + sendOffset, sendBytes));
        }
        if (index == 1) {
            const uint32_t finalMiddle = (keepStart + keepEnd) / 2;
            const bool finalKeepLower = (localRank & 1U) == 0;
            const uint32_t finalKeepStart = finalKeepLower ? keepStart : finalMiddle;
            const uint32_t finalKeepEnd = finalKeepLower ? finalMiddle : keepEnd;
            const uint32_t finalSendStart = finalKeepLower ? finalMiddle : keepStart;
            const uint32_t finalSendEnd = finalKeepLower ? keepEnd : finalMiddle;
            const uint64_t finalKeepOffset = boundaries[finalKeepStart] * FP32_BYTES;
            const uint64_t finalKeepBytes =
                (boundaries[finalKeepEnd] - boundaries[finalKeepStart]) * FP32_BYTES;
            const uint64_t finalSendOffset = boundaries[finalSendStart] * FP32_BYTES;
            const uint64_t finalSendBytes =
                (boundaries[finalSendEnd] - boundaries[finalSendStart]) * FP32_BYTES;

            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
                accumulator + finalSendOffset, localSlot + finalSendOffset - keepOffset,
                finalSendBytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            const ChannelInfo *finalChannel = nullptr;
            CHK_RET(GetChannel(param, channels, 0, finalChannel));
            CHK_RET(WaitPendingAck(resource, *finalChannel, pendingAcks[0]));
            const BufferRegion &finalRegion = layout.regions[0];
            CHK_PRT_RET(finalSendBytes > finalRegion.capacity || finalKeepBytes > finalRegion.capacity,
                HCCL_ERROR("ExecOp: aggressive fused reduce-scatter exceeds ccl region"), HCCL_E_INTERNAL);
            auto *finalLocalSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + finalRegion.offset;
            auto *finalRemoteSlot = static_cast<uint8_t *>(finalChannel->remoteCclMem.addr) + finalRegion.offset;
            CHK_RET(WriteAndRecordOnWorker(resource, 1, *finalChannel,
                finalRemoteSlot, accumulator + finalSendOffset, finalSendBytes));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
                accumulator + finalKeepOffset, localSlot + finalKeepOffset - keepOffset,
                finalKeepBytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            CHK_RET(WaitData(resource, *finalChannel));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
                accumulator + finalKeepOffset, finalLocalSlot, finalKeepBytes / FP32_BYTES,
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            groupStart = finalKeepStart;
            groupEnd = finalKeepEnd;
            break;
        }
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
            accumulator + keepOffset, localSlot, keepBytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32,
            HCOMM_REDUCE_SUM)));
        if (index == 2) {
            CHK_RET(RecordAck(resource, *channel, pendingAcks[index]));
        }
        groupStart = keepStart;
        groupEnd = keepEnd;
    }

    const uint64_t segmentOffset = boundaries[localRank] * FP32_BYTES;
    const uint64_t segmentBytes =
        (boundaries[localRank + 1] - boundaries[localRank]) * FP32_BYTES;
    const ChannelInfo *interChannel = nullptr;
    CHK_RET(GetChannel(param, channels, 3, interChannel));
    CHK_RET(WaitPendingAck(resource, *interChannel, pendingAcks[3]));
    CHK_PRT_RET(interChannel->notifyNum <= NOTIFY.interReady,
        HCCL_ERROR("ExecOp: aggressive cross-server channel has no ready notify"), HCCL_E_INTERNAL);
    const BufferRegion &interRegion = layout.regions[3];
    CHK_PRT_RET(segmentBytes > interRegion.capacity || segmentBytes > layout.scratch.capacity,
        HCCL_ERROR("ExecOp: aggressive cross-server segment exceeds ccl region"), HCCL_E_INTERNAL);
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.aicpuThread,
        interChannel->handle, NOTIFY.interReady)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread,
        interChannel->handle, NOTIFY.interReady, CUSTOM_TIMEOUT)));
    void *interSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + interRegion.offset;
    void *remoteInterSlot = static_cast<uint8_t *>(interChannel->remoteCclMem.addr) + interRegion.offset;
    CHK_RET(WriteAndWait(resource, *interChannel, remoteInterSlot, accumulator + segmentOffset, segmentBytes));

    // Opt3: Server 0 merges directly into acc (no scratch copy); Server 1 needs scratch.
    void *scratch = static_cast<uint8_t *>(resource.localBuffer.addr) + layout.scratch.offset;
    const bool serverZero = param.myRank < LOCAL_RANK_COUNT;
    if (serverZero) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
            accumulator + segmentOffset, interSlot, segmentBytes / FP32_BYTES,
            HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    } else {
        const void *firstPartial = interSlot;
        const void *secondPartial = static_cast<void *>(accumulator + segmentOffset);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resource.aicpuThread, scratch, firstPartial, segmentBytes)));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread, scratch, secondPartial,
            segmentBytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
            accumulator + segmentOffset, scratch, segmentBytes)));
    }
    for (uint32_t channelIndex = 0; channelIndex < 2; ++channelIndex) {
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(param, channels, channelIndex, channel));
        CHK_RET(RecordAck(resource, *channel, pendingAcks[channelIndex]));
    }

    groupStart = localRank;
    groupEnd = localRank + 1;
    uint8_t *pendingSlot = nullptr;
    uint64_t pendingOffset = 0;
    uint64_t pendingBytes = 0;
    for (uint32_t channelIndex = 0; channelIndex < 3; ++channelIndex) {
        const uint32_t mask = 1U << channelIndex;
        const uint32_t width = groupEnd - groupStart;
        const bool receiveUpper = (localRank & mask) == 0;
        const uint32_t receiveStart = receiveUpper ? groupEnd : groupStart - width;
        const uint32_t receiveEnd = receiveUpper ? groupEnd + width : groupStart;
        const uint64_t sendOffset = boundaries[groupStart] * FP32_BYTES;
        const uint64_t sendBytes =
            (boundaries[groupEnd] - boundaries[groupStart]) * FP32_BYTES;
        const uint64_t receiveOffset = boundaries[receiveStart] * FP32_BYTES;
        const uint64_t receiveBytes =
            (boundaries[receiveEnd] - boundaries[receiveStart]) * FP32_BYTES;

        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(param, channels, channelIndex, channel));
        CHK_RET(WaitPendingAck(resource, *channel, pendingAcks[channelIndex]));
        const BufferRegion &region = layout.regions[channelIndex];
        CHK_PRT_RET(sendBytes > region.capacity || receiveBytes > region.capacity,
            HCCL_ERROR("ExecOp: aggressive allgather exceeds ccl region"), HCCL_E_INTERNAL);
        auto *localSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + region.offset;
        auto *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) + region.offset;
        if (channelIndex == 0) {
            CHK_RET(WriteAndWait(resource, *channel, remoteSlot,
                accumulator + sendOffset, sendBytes));
        } else {
            CHK_PRT_RET(pendingSlot == nullptr || pendingOffset < sendOffset ||
                    pendingOffset + pendingBytes > sendOffset + sendBytes,
                HCCL_ERROR("ExecOp: aggressive allgather pending span is invalid"), HCCL_E_INTERNAL);
            const bool pendingLower = pendingOffset == sendOffset;
            const uint64_t residentOffset = pendingLower ? pendingOffset + pendingBytes : sendOffset;
            const uint64_t residentBytes = sendBytes - pendingBytes;
            CHK_RET(WriteTwoAndRecordOnWorker(resource, 1, *channel,
                remoteSlot + residentOffset - sendOffset, accumulator + residentOffset, residentBytes,
                remoteSlot + pendingOffset - sendOffset, pendingSlot, pendingBytes));
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
                accumulator + pendingOffset, pendingSlot, pendingBytes)));
            CHK_RET(WaitData(resource, *channel));
            CHK_RET(WaitData(resource, *channel));
        }
        if (channelIndex == 2) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
                accumulator + receiveOffset, localSlot, receiveBytes)));
        } else {
            pendingSlot = localSlot;
            pendingOffset = receiveOffset;
            pendingBytes = receiveBytes;
        }
        groupStart = std::min(groupStart, receiveStart);
        groupEnd = std::max(groupEnd, receiveEnd);
    }
    for (uint32_t channelIndex = 0; channelIndex < 3; ++channelIndex) {
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(param, channels, channelIndex, channel));
        CHK_RET(RecordAck(resource, *channel, pendingAcks[channelIndex]));
    }
    CHK_RET(RecordAck(resource, *interChannel, pendingAcks[3]));
    return HCCL_SUCCESS;
}

HcclResult ExecRadixChunk(const OpParam &param, const AlgResourceCtx &resource, const ChannelMap &channels,
    PendingAcks &pendingAcks, const RadixBufferLayout &layout, uint64_t bytes)
{
    CHK_PRT_RET(resource.threads.size() < RADIX_THREAD_COUNT,
        HCCL_ERROR("ExecOp: radix workers are missing"), HCCL_E_INTERNAL);
    const uint64_t elements = bytes / FP32_BYTES;
    auto *accumulator = static_cast<uint8_t *>(param.outputPtr);
    const auto *input = static_cast<const uint8_t *>(param.inputPtr);
    const uint32_t localRank = param.myRank & (LOCAL_RANK_COUNT - 1);
    const uint32_t serverBase = param.myRank & ~(LOCAL_RANK_COUNT - 1);
    const uint32_t radixFourDigit = localRank & 3U;
    const uint32_t radixTwoDigit = localRank >> 2;
    const uint64_t blockOffset = ElementBoundary(elements, 2 * radixFourDigit) * FP32_BYTES;
    const uint64_t blockBytes =
        (ElementBoundary(elements, 2 * radixFourDigit + 2) -
        ElementBoundary(elements, 2 * radixFourDigit)) * FP32_BYTES;

    uint32_t workerIndex = 1;
    for (uint32_t destinationDigit = 0; destinationDigit < 4; ++destinationDigit) {
        if (destinationDigit == radixFourDigit) {
            continue;
        }
        const uint32_t peer = serverBase + radixTwoDigit * 4 + destinationDigit;
        const ChannelInfo *channel = nullptr;
        uint32_t channelIndex = 0;
        CHK_RET(GetPeerChannel(param, channels, peer, channel, channelIndex));
        CHK_RET(WaitPendingAck(resource, *channel, pendingAcks[channelIndex]));
        const uint64_t sendOffset = ElementBoundary(elements, 2 * destinationDigit) * FP32_BYTES;
        const uint64_t sendBytes =
            (ElementBoundary(elements, 2 * destinationDigit + 2) -
            ElementBoundary(elements, 2 * destinationDigit)) * FP32_BYTES;
        CHK_PRT_RET(sendBytes > layout.slotBytes,
            HCCL_ERROR("ExecOp: radix-4 reduce-scatter exceeds ccl slot"), HCCL_E_INTERNAL);
        const uint32_t remoteSlotIndex = RadixSlot(radixFourDigit, destinationDigit);
        void *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) + remoteSlotIndex * layout.slotBytes;
        CHK_RET(WriteAndRecordOnWorker(resource, workerIndex, *channel, remoteSlot,
            input + sendOffset, sendBytes));
        ++workerIndex;
    }
    if (input != accumulator) {
        // Only the locally owned quarter survives reduce-scatter. Copy it while
        // the worker streams send the other three quarters directly from input.
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
            accumulator + blockOffset, input + blockOffset, blockBytes)));
    }
    for (uint32_t sourceDigit = 0; sourceDigit < 4; ++sourceDigit) {
        if (sourceDigit == radixFourDigit) {
            continue;
        }
        const uint32_t peer = serverBase + radixTwoDigit * 4 + sourceDigit;
        const ChannelInfo *channel = nullptr;
        uint32_t channelIndex = 0;
        CHK_RET(GetPeerChannel(param, channels, peer, channel, channelIndex));
        CHK_RET(WaitData(resource, *channel));
    }
    const uint32_t radixTwoPeer = serverBase + (1U - radixTwoDigit) * 4 + radixFourDigit;
    const ChannelInfo *radixTwoChannel = nullptr;
    uint32_t radixTwoChannelIndex = 0;
    CHK_RET(GetPeerChannel(param, channels, radixTwoPeer, radixTwoChannel, radixTwoChannelIndex));
    CHK_RET(WaitPendingAck(resource, *radixTwoChannel, pendingAcks[radixTwoChannelIndex]));
    const uint32_t ownedSegment = 2 * radixFourDigit + radixTwoDigit;
    const uint32_t sentSegment = 2 * radixFourDigit + (1U - radixTwoDigit);
    const uint64_t segmentOffset = ElementBoundary(elements, ownedSegment) * FP32_BYTES;
    const uint64_t segmentBytes =
        (ElementBoundary(elements, ownedSegment + 1) - ElementBoundary(elements, ownedSegment)) * FP32_BYTES;
    const uint64_t sentOffset = ElementBoundary(elements, sentSegment) * FP32_BYTES;
    const uint64_t sentBytes =
        (ElementBoundary(elements, sentSegment + 1) - ElementBoundary(elements, sentSegment)) * FP32_BYTES;
    CHK_PRT_RET(segmentBytes > layout.slotBytes || sentBytes > layout.slotBytes,
        HCCL_ERROR("ExecOp: radix-2 reduce-scatter exceeds ccl slot"), HCCL_E_INTERNAL);
    void *radixReceive = resource.localBuffer.addr;

    // Slot 0 always belongs to the first source in radix order. Reduce both of
    // its halves immediately so the slot can receive the radix-2 exchange.
    const uint32_t firstSourceDigit = radixFourDigit == 0 ? 1 : 0;
    for (uint32_t sourceDigit = 0; sourceDigit < 4; ++sourceDigit) {
        if (sourceDigit == radixFourDigit) {
            continue;
        }
        const uint32_t slotIndex = RadixSlot(sourceDigit, radixFourDigit);
        auto *localSlot = static_cast<uint8_t *>(resource.localBuffer.addr) +
            slotIndex * layout.slotBytes;
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
            accumulator + sentOffset, localSlot + sentOffset - blockOffset,
            sentBytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        if (sourceDigit == firstSourceDigit) {
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
                accumulator + segmentOffset, localSlot + segmentOffset - blockOffset,
                segmentBytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.aicpuThread,
        radixTwoChannel->handle, NOTIFY.interReady)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread,
        radixTwoChannel->handle, NOTIFY.interReady, CUSTOM_TIMEOUT)));
    CHK_RET(WriteAndRecordOnWorker(resource, 1, *radixTwoChannel,
        radixTwoChannel->remoteCclMem.addr, accumulator + sentOffset, sentBytes));

    // Reduce the locally owned half while worker 1 transfers the other half.
    for (uint32_t sourceDigit = 0; sourceDigit < 4; ++sourceDigit) {
        if (sourceDigit == radixFourDigit || sourceDigit == firstSourceDigit) {
            continue;
        }
        const uint32_t slotIndex = RadixSlot(sourceDigit, radixFourDigit);
        auto *localSlot = static_cast<uint8_t *>(resource.localBuffer.addr) +
            slotIndex * layout.slotBytes;
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
            accumulator + segmentOffset, localSlot + segmentOffset - blockOffset,
            segmentBytes / FP32_BYTES, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    CHK_RET(WaitData(resource, *radixTwoChannel));
    if (radixTwoDigit == 0) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
            accumulator + segmentOffset, radixReceive, segmentBytes / FP32_BYTES,
            HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    } else {
        // Preserve the radix digit-0 + digit-1 reduction order in the receive slot.
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
            radixReceive, accumulator + segmentOffset, segmentBytes / FP32_BYTES,
            HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
            accumulator + segmentOffset, radixReceive, segmentBytes)));
    }

    const ChannelInfo *interChannel = nullptr;
    CHK_RET(GetChannel(param, channels, 3, interChannel));
    CHK_RET(WaitPendingAck(resource, *interChannel, pendingAcks[3]));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.aicpuThread,
        interChannel->handle, NOTIFY.interReady)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resource.aicpuThread,
        interChannel->handle, NOTIFY.interReady, CUSTOM_TIMEOUT)));
    CHK_RET(WriteAndWait(resource, *interChannel, interChannel->remoteCclMem.addr,
        accumulator + segmentOffset, segmentBytes));
    const bool serverZero = param.myRank < LOCAL_RANK_COUNT;
    if (serverZero) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
            accumulator + segmentOffset, radixReceive, segmentBytes / FP32_BYTES,
            HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    } else {
        // Preserve the global server-0 + server-1 reduction order in the receive slot.
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.aicpuThread,
            radixReceive, accumulator + segmentOffset, segmentBytes / FP32_BYTES,
            HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
            accumulator + segmentOffset, radixReceive, segmentBytes)));
    }

    CHK_RET(RecordAck(resource, *radixTwoChannel, pendingAcks[radixTwoChannelIndex]));

    CHK_RET(WaitPendingAck(resource, *radixTwoChannel, pendingAcks[radixTwoChannelIndex]));
    void *remoteRadixReceive = radixTwoChannel->remoteCclMem.addr;
    CHK_RET(WriteAndWait(resource, *radixTwoChannel, remoteRadixReceive,
        accumulator + segmentOffset, segmentBytes));
    const uint64_t receivedSegmentOffset = ElementBoundary(elements, sentSegment) * FP32_BYTES;
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
        accumulator + receivedSegmentOffset, radixReceive, sentBytes)));

    for (uint32_t peerDigit = 0; peerDigit < 4; ++peerDigit) {
        if (peerDigit == radixFourDigit) {
            continue;
        }
        const uint32_t peer = serverBase + radixTwoDigit * 4 + peerDigit;
        const ChannelInfo *channel = nullptr;
        uint32_t channelIndex = 0;
        CHK_RET(GetPeerChannel(param, channels, peer, channel, channelIndex));
        CHK_RET(RecordAck(resource, *channel, pendingAcks[channelIndex]));
    }

    workerIndex = 1;
    for (uint32_t destinationDigit = 0; destinationDigit < 4; ++destinationDigit) {
        if (destinationDigit == radixFourDigit) {
            continue;
        }
        const uint32_t peer = serverBase + radixTwoDigit * 4 + destinationDigit;
        const ChannelInfo *channel = nullptr;
        uint32_t channelIndex = 0;
        CHK_RET(GetPeerChannel(param, channels, peer, channel, channelIndex));
        CHK_RET(WaitPendingAck(resource, *channel, pendingAcks[channelIndex]));
        const uint32_t remoteSlotIndex = RadixSlot(radixFourDigit, destinationDigit);
        void *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) + remoteSlotIndex * layout.slotBytes;
        CHK_RET(WriteAndRecordOnWorker(resource, workerIndex, *channel, remoteSlot,
            accumulator + blockOffset, blockBytes));
        ++workerIndex;
    }
    for (uint32_t sourceDigit = 0; sourceDigit < 4; ++sourceDigit) {
        if (sourceDigit == radixFourDigit) {
            continue;
        }
        const uint32_t peer = serverBase + radixTwoDigit * 4 + sourceDigit;
        const ChannelInfo *channel = nullptr;
        uint32_t channelIndex = 0;
        CHK_RET(GetPeerChannel(param, channels, peer, channel, channelIndex));
        CHK_RET(WaitData(resource, *channel));
        const uint32_t localSlotIndex = RadixSlot(sourceDigit, radixFourDigit);
        void *localSlot = static_cast<uint8_t *>(resource.localBuffer.addr) + localSlotIndex * layout.slotBytes;
        const uint64_t receiveOffset = ElementBoundary(elements, 2 * sourceDigit) * FP32_BYTES;
        const uint64_t receiveBytes =
            (ElementBoundary(elements, 2 * sourceDigit + 2) -
            ElementBoundary(elements, 2 * sourceDigit)) * FP32_BYTES;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resource.aicpuThread,
            accumulator + receiveOffset, localSlot, receiveBytes)));
    }
    for (uint32_t peerDigit = 0; peerDigit < 4; ++peerDigit) {
        if (peerDigit == radixFourDigit) {
            continue;
        }
        const uint32_t peer = serverBase + radixTwoDigit * 4 + peerDigit;
        const ChannelInfo *channel = nullptr;
        uint32_t channelIndex = 0;
        CHK_RET(GetPeerChannel(param, channels, peer, channel, channelIndex));
        CHK_RET(RecordAck(resource, *channel, pendingAcks[channelIndex]));
    }
    CHK_RET(RecordAck(resource, *radixTwoChannel, pendingAcks[radixTwoChannelIndex]));
    CHK_RET(RecordAck(resource, *interChannel, pendingAcks[3]));
    return HCCL_SUCCESS;
}
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resource)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    const uint64_t totalBytes = param.count * FP32_BYTES;
    const ChannelMap &channels = resource.channels;
    PendingAcks pendingAcks = {};
    if (totalBytes <= SMALL_MESSAGE_BYTES) {
        CHK_RET(ValidateSmallResourceContract(param, resource));
        const uint64_t xorSlotBytes = GetXorSlotBytes(resource);
        CHK_PRT_RET(xorSlotBytes == 0,
            HCCL_ERROR("ExecOp: ccl buffer cannot hold fp32"), HCCL_E_INTERNAL);
        for (uint64_t offset = 0; offset < totalBytes;) {
            const Span span = {offset, std::min(xorSlotBytes, totalBytes - offset)};
            CHK_RET(ExecSmallXorFallback(param, resource, channels, pendingAcks,
                xorSlotBytes, span.offset, span.bytes));
            offset += span.bytes;
        }
        return WaitXorPendingAcks(resource, channels, pendingAcks);
    }
    CHK_RET(ValidateResourceContract(param, resource));
    const AllReducePlan plan = BuildAllReducePlan(param, resource);
    CHK_PRT_RET(plan.xorSlotBytes == 0 || plan.safe.maxChunkBytes == 0 || plan.fast.maxChunkBytes == 0 ||
            plan.radix.maxChunkBytes == 0,
        HCCL_ERROR("ExecOp: ccl buffer cannot hold fp32"), HCCL_E_INTERNAL);
    if (plan.policy == ExecutionPolicy::LARGE_FULLMESH) {
        CHK_RET(ExecLargeFullMesh(param, resource, channels, plan.fullMesh));
        return WaitAllPendingAcks(param, resource, channels, pendingAcks);
    }
    if (plan.policy == ExecutionPolicy::LARGE_FULLMESH_PIPELINE2) {
        CHK_RET(ExecLargeFullMeshPipeline2(param, resource, channels, plan.fullMesh));
        return WaitAllPendingAcks(param, resource, channels, pendingAcks);
    }
    if (plan.policy == ExecutionPolicy::LARGE_FAST) {
        CHK_RET(ExecAggressiveChunk(param, resource, channels, pendingAcks, plan.fast, plan.totalBytes));
        return WaitAllPendingAcks(param, resource, channels, pendingAcks);
    }
    if (plan.policy == ExecutionPolicy::LARGE_RADIX) {
        CHK_RET(ExecRadixChunk(param, resource, channels, pendingAcks, plan.radix, plan.totalBytes));
        return WaitAllPendingAcks(param, resource, channels, pendingAcks);
    }
    for (uint64_t offset = 0; offset < plan.totalBytes;) {
        const uint64_t remaining = plan.totalBytes - offset;
        if (remaining <= SMALL_MESSAGE_BYTES || plan.policy == ExecutionPolicy::SMALL_XOR) {
            // The XOR and hierarchical layouts alias slots across channels; drain before switching layouts.
            CHK_RET(WaitAllPendingAcks(param, resource, channels, pendingAcks));
            const Span span = {offset, std::min(plan.xorSlotBytes, remaining)};
            CHK_RET(ExecSmallXorFallback(param, resource, channels, pendingAcks,
                plan.xorSlotBytes, span.offset, span.bytes));
            offset += span.bytes;
            continue;
        }
        const Span span = {offset, std::min(plan.safe.maxChunkBytes, remaining)};
        CHK_RET(ExecHierarchicalChunk(param, resource, channels, pendingAcks,
            plan.safe, span.offset, span.bytes));
        offset += span.bytes;
    }
    return WaitAllPendingAcks(param, resource, channels, pendingAcks);
}
}
