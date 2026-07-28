/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include <ccu/ccu_primitives.hpp>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"

namespace ops_hccl {
namespace {
constexpr uint32_t OUTPUT_ADDR_XN_ID = 0;
constexpr uint32_t OUTPUT_TOKEN_XN_ID = 1;
constexpr uint32_t INPUT_ADDR_XN_ID = 2;
constexpr uint32_t INPUT_TOKEN_XN_ID = 3;
constexpr uint32_t SCRATCH_ADDR_XN_ID = 4;
constexpr uint32_t SCRATCH_TOKEN_XN_ID = 5;
constexpr uint32_t RESOURCE_NOTIFY_INDEX = 0;
constexpr uint32_t SYNC_NOTIFY_INDEX = 1;
constexpr uint16_t OUTPUT_ADDR_NOTIFY_BIT = 1U << OUTPUT_ADDR_XN_ID;
constexpr uint16_t OUTPUT_TOKEN_NOTIFY_BIT = 1U << OUTPUT_TOKEN_XN_ID;
constexpr uint16_t INPUT_ADDR_NOTIFY_BIT = 1U << INPUT_ADDR_XN_ID;
constexpr uint16_t INPUT_TOKEN_NOTIFY_BIT = 1U << INPUT_TOKEN_XN_ID;
constexpr uint16_t SCRATCH_ADDR_NOTIFY_BIT = 1U << SCRATCH_ADDR_XN_ID;
constexpr uint16_t SCRATCH_TOKEN_NOTIFY_BIT = 1U << SCRATCH_TOKEN_XN_ID;
constexpr uint16_t RESOURCE_NOTIFY_BITS = OUTPUT_ADDR_NOTIFY_BIT | OUTPUT_TOKEN_NOTIFY_BIT;
constexpr uint16_t INPUT_RESOURCE_NOTIFY_BITS = INPUT_ADDR_NOTIFY_BIT | INPUT_TOKEN_NOTIFY_BIT;
constexpr uint16_t SCRATCH_RESOURCE_NOTIFY_BITS = SCRATCH_ADDR_NOTIFY_BIT | SCRATCH_TOKEN_NOTIFY_BIT;
constexpr uint16_t HIERARCHICAL_RESOURCE_NOTIFY_BITS = RESOURCE_NOTIFY_BITS | INPUT_RESOURCE_NOTIFY_BITS;
constexpr uint16_t RHD_RS_READY_NOTIFY_BIT = 1U << 0;
constexpr uint16_t RHD_RS_DONE_NOTIFY_BIT = 1U << 1;
constexpr uint16_t RHD_AG_READY_NOTIFY_BIT = 1U << 2;
constexpr uint16_t RHD_AG_DONE_NOTIFY_BIT = 1U << 3;
constexpr uint16_t RHD_FOLD_DONE_NOTIFY_BIT = 1U << 4;
constexpr uint16_t RHD_FINAL_READY_NOTIFY_BIT = 1U << 5;
constexpr uint16_t RHD_FINAL_DONE_NOTIFY_BIT = 1U << 6;
constexpr uint16_t HIGH_RADIX_READY_NOTIFY_BIT = 1U << 0;
constexpr uint16_t RING_PRE_SYNC_NOTIFY_BIT = 1U << 5;
constexpr uint16_t RING_RS_DONE_NOTIFY_BIT = 1U << 6;
constexpr uint16_t RING_AG_DONE_NOTIFY_BIT = 1U << 7;
constexpr uint16_t RING_AG_READY_NOTIFY_BIT = 1U << 8;
constexpr uint16_t HIERARCHICAL_RS_STAGE2_READY_NOTIFY_BIT = 1U << 9;
constexpr uint16_t HIERARCHICAL_AG_STAGE1_READY_NOTIFY_BIT = 1U << 10;
constexpr uint16_t HIERARCHICAL_AG_STAGE2_READY_NOTIFY_BIT = 1U << 11;
constexpr uint16_t RADIX4_RS_READY_NOTIFY_BIT = 1U << 12;
constexpr uint16_t TOPOLOGY_STAR_READY_NOTIFY_BIT = 1U << 13;
constexpr uint16_t TOPOLOGY_STAR_RELEASE_NOTIFY_BIT = 1U << 14;
constexpr uint16_t FLAT_OWNER_READY_NOTIFY_BIT = 1U << 15;
constexpr uint16_t FLAT_OWNER_ACK_NOTIFY_BIT = TOPOLOGY_STAR_RELEASE_NOTIFY_BIT;
constexpr uint32_t INVALID_CHANNEL_INDEX = MAX_ALG_CHANNEL_NUM;
constexpr uint32_t TOPOLOGY_PHASE_TASK_ARG_INDEX = 6;
constexpr uint32_t TOPOLOGY_CROSS_SLICE_TASK_ARG_INDEX = 6;
constexpr uint32_t TOPOLOGY_LOCAL_SLICE_TASK_ARG_INDEX = 7;
constexpr uint64_t TOPOLOGY_LOCAL_REDUCE_SELECTOR = 0;
constexpr uint64_t TOPOLOGY_LOCAL_GATHER_SELECTOR = 1;

uint32_t LargestPowerOfTwo(uint32_t value)
{
    uint32_t result = 1;
    while ((result << 1U) <= value) {
        result <<= 1U;
    }
    return result;
}

uint32_t ActiveRankToOriginal(uint32_t activeRank, uint32_t remainder)
{
    return activeRank < remainder ? activeRank * 2U + 1U : activeRank + remainder;
}

uint32_t FindChannelIndex(const AllReduceKernelArg &arg, uint32_t remoteRank)
{
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        if (arg.remoteRanks[idx] == remoteRank) {
            return idx;
        }
    }
    return INVALID_CHANNEL_INDEX;
}

uint64_t GetRangeBytes(const AllReduceKernelArg &arg, uint32_t rangeStart,
    uint32_t rangeCount, uint32_t segmentCount)
{
    const uint32_t rangeEnd = rangeStart + rangeCount;
    const uint64_t endOffset = rangeEnd < segmentCount ? arg.segmentOffsets[rangeEnd] : arg.dataBytes;
    return endOffset - arg.segmentOffsets[rangeStart];
}

uint64_t GetLaneRangeBytes(const AllReduceKernelArg &arg, uint32_t lane,
    uint32_t rangeStart, uint32_t rangeCount)
{
    const uint32_t rangeEnd = rangeStart + rangeCount;
    const uint64_t laneEnd = arg.laneSegmentOffsets[lane][0] + arg.laneBytes[lane];
    const uint64_t endOffset = rangeEnd < arg.rankSize ?
        arg.laneSegmentOffsets[lane][rangeEnd] : laneEnd;
    return endOffset - arg.laneSegmentOffsets[lane][rangeStart];
}

uint32_t RankBitCount(uint32_t rankSize)
{
    uint32_t bits = 0;
    while ((1U << bits) < rankSize) {
        ++bits;
    }
    return bits;
}

uint32_t RotateRankLeftOne(uint32_t rank, uint32_t bits)
{
    const uint32_t mask = (1U << bits) - 1U;
    return ((rank << 1U) & mask) | (rank >> (bits - 1U));
}

uint32_t RotateRankRightOne(uint32_t rank, uint32_t bits)
{
    return (rank >> 1U) | ((rank & 1U) << (bits - 1U));
}

CcuResult ExchangeOutputResources(const AllReduceKernelArg &arg, const ccu::Variable &outputAddr,
    const ccu::Variable &outputToken)
{
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx], outputAddr, OUTPUT_ADDR_XN_ID,
            RESOURCE_NOTIFY_INDEX, OUTPUT_ADDR_NOTIFY_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx], outputToken, OUTPUT_TOKEN_XN_ID,
            RESOURCE_NOTIFY_INDEX, OUTPUT_TOKEN_NOTIFY_BIT));
    }
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        CCU_CHK_RET(ccu::NotifyWait(arg.channels[idx], RESOURCE_NOTIFY_INDEX, RESOURCE_NOTIFY_BITS));
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeHierarchicalResources(const AllReduceKernelArg &arg, const ccu::Variable &inputAddr,
    const ccu::Variable &inputToken, const ccu::Variable &outputAddr, const ccu::Variable &outputToken)
{
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx], outputAddr, OUTPUT_ADDR_XN_ID,
            RESOURCE_NOTIFY_INDEX, OUTPUT_ADDR_NOTIFY_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx], outputToken, OUTPUT_TOKEN_XN_ID,
            RESOURCE_NOTIFY_INDEX, OUTPUT_TOKEN_NOTIFY_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx], inputAddr, INPUT_ADDR_XN_ID,
            RESOURCE_NOTIFY_INDEX, INPUT_ADDR_NOTIFY_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx], inputToken, INPUT_TOKEN_XN_ID,
            RESOURCE_NOTIFY_INDEX, INPUT_TOKEN_NOTIFY_BIT));
    }
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        CCU_CHK_RET(ccu::NotifyWait(arg.channels[idx], RESOURCE_NOTIFY_INDEX, HIERARCHICAL_RESOURCE_NOTIFY_BITS));
    }
    return CCU_SUCCESS;
}

CcuResult PublishSmallDataResources(const AllReduceKernelArg &arg, const ccu::Variable &inputAddr,
    const ccu::Variable &inputToken, const ccu::Variable &scratchAddr,
    const ccu::Variable &scratchToken, bool exchangeScratch)
{
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        const uint32_t peerRank = arg.remoteRanks[idx];
        const bool inputPeer = peerRank / HIGH_RADIX == arg.rankId / HIGH_RADIX;
        const bool scratchPeer = exchangeScratch &&
            peerRank % HIGH_RADIX == arg.rankId % HIGH_RADIX;
        if (!inputPeer && !scratchPeer) {
            return CCU_E_PARA;
        }
        if (inputPeer) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx], inputAddr, INPUT_ADDR_XN_ID,
                RESOURCE_NOTIFY_INDEX, INPUT_ADDR_NOTIFY_BIT));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx], inputToken, INPUT_TOKEN_XN_ID,
                RESOURCE_NOTIFY_INDEX, INPUT_TOKEN_NOTIFY_BIT));
        }
        if (scratchPeer) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx], scratchAddr, SCRATCH_ADDR_XN_ID,
                RESOURCE_NOTIFY_INDEX, SCRATCH_ADDR_NOTIFY_BIT));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx], scratchToken, SCRATCH_TOKEN_XN_ID,
                RESOURCE_NOTIFY_INDEX, SCRATCH_TOKEN_NOTIFY_BIT));
        }
    }
    return CCU_SUCCESS;
}

CcuResult WaitSmallInputResources(const AllReduceKernelArg &arg)
{
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        const uint32_t peerRank = arg.remoteRanks[idx];
        if (peerRank / HIGH_RADIX != arg.rankId / HIGH_RADIX) {
            continue;
        }
        CCU_CHK_RET(ccu::NotifyWait(arg.channels[idx], RESOURCE_NOTIFY_INDEX, INPUT_RESOURCE_NOTIFY_BITS));
    }
    return CCU_SUCCESS;
}

CcuResult WaitSmallScratchResources(const AllReduceKernelArg &arg)
{
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        const uint32_t peerRank = arg.remoteRanks[idx];
        if (peerRank % HIGH_RADIX != arg.rankId % HIGH_RADIX) {
            continue;
        }
        CCU_CHK_RET(ccu::NotifyWait(arg.channels[idx], RESOURCE_NOTIFY_INDEX, SCRATCH_RESOURCE_NOTIFY_BITS));
    }
    return CCU_SUCCESS;
}

CcuResult HierarchicalBarrier(const AllReduceKernelArg &arg, uint32_t groupBase, uint32_t groupStride,
    uint32_t groupSize, uint16_t notifyBit)
{
    for (uint32_t position = 0; position < groupSize; ++position) {
        const uint32_t peerRank = groupBase + position * groupStride;
        if (peerRank == arg.rankId) {
            continue;
        }
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        if (channelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        CCU_CHK_RET(ccu::NotifyRecord(arg.channels[channelIndex], SYNC_NOTIFY_INDEX, notifyBit));
    }
    for (uint32_t position = 0; position < groupSize; ++position) {
        const uint32_t peerRank = groupBase + position * groupStride;
        if (peerRank == arg.rankId) {
            continue;
        }
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        CCU_CHK_RET(ccu::NotifyWait(arg.channels[channelIndex], SYNC_NOTIFY_INDEX, notifyBit));
    }
    return CCU_SUCCESS;
}

CcuResult LocalCopyRanges(const ccu::Variable &inputAddr, const ccu::Variable &inputToken,
    const ccu::Variable &outputAddr, const ccu::Variable &outputToken,
    const uint64_t rangeOffsets[MAX_RANK_SIZE], const uint64_t rangeBytes[MAX_RANK_SIZE],
    uint32_t rangeCount, ccu::Event &dataEvent)
{
    if (rangeCount == 0 || rangeCount * PIPELINE_DEPTH > sizeof(uint16_t) * 8U) {
        return CCU_E_PARA;
    }
    const uint64_t chunkLimit = PIPELINE_CHUNK_BYTES < MAX_DATA_SIZE ?
        PIPELINE_CHUNK_BYTES : MAX_DATA_SIZE;
    uint64_t batchOffsets[MAX_RANK_SIZE] {};
    bool hasPending = true;
    while (hasPending) {
        hasPending = false;
        uint16_t batchMask = 0;
        for (uint32_t range = 0; range < rangeCount; ++range) {
            for (uint32_t lane = 0; lane < PIPELINE_DEPTH; ++lane) {
                const uint64_t chunkOffset = batchOffsets[range] + static_cast<uint64_t>(lane) * chunkLimit;
                if (chunkOffset >= rangeBytes[range]) {
                    continue;
                }
                hasPending = true;
                const uint64_t chunkBytes = rangeBytes[range] - chunkOffset > chunkLimit ?
                    chunkLimit : rangeBytes[range] - chunkOffset;
                ccu::Variable offset;
                ccu::Variable bytes;
                offset = rangeOffsets[range] + chunkOffset;
                bytes = chunkBytes;

                ccu::LocalAddr localSource;
                localSource.addr = inputAddr;
                localSource.addr += offset;
                localSource.token = inputToken;
                ccu::LocalAddr localDestination;
                localDestination.addr = outputAddr;
                localDestination.addr += offset;
                localDestination.token = outputToken;

                const uint32_t eventIndex = range * PIPELINE_DEPTH + lane;
                const uint16_t eventBit = static_cast<uint16_t>(1U << eventIndex);
                CCU_CHK_RET(ccu::LocalCopy(localDestination, localSource, bytes, dataEvent, eventBit));
                batchMask = static_cast<uint16_t>(batchMask | eventBit);
            }
            batchOffsets[range] += chunkLimit * PIPELINE_DEPTH;
        }
        if (batchMask != 0) {
            CCU_CHK_RET(ccu::EventWait(dataEvent, batchMask));
        }
    }
    return CCU_SUCCESS;
}

CcuResult V18LocalCopyRange(const ccu::Variable &inputAddr, const ccu::Variable &inputToken,
    const ccu::Variable &outputAddr, const ccu::Variable &outputToken,
    uint64_t rangeOffset, uint64_t rangeBytes, ccu::Event &dataEvent)
{
    for (uint64_t chunkOffset = 0; chunkOffset < rangeBytes; chunkOffset += MAX_DATA_SIZE) {
        const uint64_t chunkBytes = rangeBytes - chunkOffset > MAX_DATA_SIZE ?
            MAX_DATA_SIZE : rangeBytes - chunkOffset;
        ccu::Variable offset;
        ccu::Variable bytes;
        offset = rangeOffset + chunkOffset;
        bytes = chunkBytes;

        ccu::LocalAddr localSource;
        localSource.addr = inputAddr;
        localSource.addr += offset;
        localSource.token = inputToken;
        ccu::LocalAddr localDestination;
        localDestination.addr = outputAddr;
        localDestination.addr += offset;
        localDestination.token = outputToken;

        CCU_CHK_RET(ccu::LocalCopy(localDestination, localSource, bytes, dataEvent));
        CCU_CHK_RET(ccu::EventWait(dataEvent));
    }
    return CCU_SUCCESS;
}

CcuResult V18PushRange(const AllReduceKernelArg &arg, uint32_t channelIndex,
    const std::vector<ccu::Variable> &remoteOutputs, const std::vector<ccu::Variable> &remoteTokens,
    const ccu::Variable &localAddr, const ccu::Variable &localToken, uint64_t rangeOffset,
    uint64_t rangeBytes, bool reduce, ccu::Event &dataEvent)
{
    if (channelIndex == INVALID_CHANNEL_INDEX) {
        return CCU_E_PARA;
    }
    if (rangeBytes == 0) {
        return CCU_SUCCESS;
    }

    for (uint64_t chunkOffset = 0; chunkOffset < rangeBytes; chunkOffset += MAX_DATA_SIZE) {
        const uint64_t chunkBytes = rangeBytes - chunkOffset > MAX_DATA_SIZE ?
            MAX_DATA_SIZE : rangeBytes - chunkOffset;
        ccu::Variable offset;
        ccu::Variable bytes;
        offset = rangeOffset + chunkOffset;
        bytes = chunkBytes;

        ccu::LocalAddr localSource;
        localSource.addr = localAddr;
        localSource.addr += offset;
        localSource.token = localToken;

        ccu::RemoteAddr remoteDestination;
        remoteDestination.addr = remoteOutputs[channelIndex];
        remoteDestination.addr += offset;
        remoteDestination.token = remoteTokens[channelIndex];

        if (reduce) {
            CCU_CHK_RET(ccu::WriteReduce(arg.channels[channelIndex], remoteDestination, localSource, bytes,
                HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, dataEvent));
        } else {
            CCU_CHK_RET(ccu::Write(arg.channels[channelIndex], remoteDestination, localSource, bytes, dataEvent));
        }
        CCU_CHK_RET(ccu::EventWait(dataEvent));
    }
    return CCU_SUCCESS;
}

CcuResult PipelineLocalCopyRange(const ccu::Variable &inputAddr, const ccu::Variable &inputToken,
    const ccu::Variable &outputAddr, const ccu::Variable &outputToken,
    uint64_t rangeOffset, uint64_t rangeBytes, ccu::Event &dataEvent)
{
    const uint64_t chunkLimit = PIPELINE_CHUNK_BYTES < MAX_DATA_SIZE ?
        PIPELINE_CHUNK_BYTES : MAX_DATA_SIZE;
    for (uint64_t batchOffset = 0; batchOffset < rangeBytes;
        batchOffset += chunkLimit * PIPELINE_DEPTH) {
        uint16_t batchMask = 0;
        for (uint32_t lane = 0; lane < PIPELINE_DEPTH; ++lane) {
            const uint64_t chunkOffset = batchOffset + static_cast<uint64_t>(lane) * chunkLimit;
            if (chunkOffset >= rangeBytes) {
                break;
            }
            const uint64_t chunkBytes = rangeBytes - chunkOffset > chunkLimit ?
                chunkLimit : rangeBytes - chunkOffset;
            ccu::Variable offset;
            ccu::Variable bytes;
            offset = rangeOffset + chunkOffset;
            bytes = chunkBytes;

            ccu::LocalAddr localSource;
            localSource.addr = inputAddr;
            localSource.addr += offset;
            localSource.token = inputToken;
            ccu::LocalAddr localDestination;
            localDestination.addr = outputAddr;
            localDestination.addr += offset;
            localDestination.token = outputToken;

            const uint16_t eventBit = static_cast<uint16_t>(1U << lane);
            CCU_CHK_RET(ccu::LocalCopy(localDestination, localSource, bytes, dataEvent, eventBit));
            batchMask = static_cast<uint16_t>(batchMask | eventBit);
        }
        CCU_CHK_RET(ccu::EventWait(dataEvent, batchMask));
    }
    return CCU_SUCCESS;
}

CcuResult PipelinePushRange(const AllReduceKernelArg &arg, uint32_t channelIndex,
    const std::vector<ccu::Variable> &remoteOutputs, const std::vector<ccu::Variable> &remoteTokens,
    const ccu::Variable &localAddr, const ccu::Variable &localToken, uint64_t rangeOffset,
    uint64_t rangeBytes, bool reduce, ccu::Event &dataEvent)
{
    if (channelIndex == INVALID_CHANNEL_INDEX) {
        return CCU_E_PARA;
    }
    if (rangeBytes == 0) {
        return CCU_SUCCESS;
    }

    const uint64_t chunkLimit = PIPELINE_CHUNK_BYTES < MAX_DATA_SIZE ?
        PIPELINE_CHUNK_BYTES : MAX_DATA_SIZE;
    for (uint64_t batchOffset = 0; batchOffset < rangeBytes;
        batchOffset += chunkLimit * PIPELINE_DEPTH) {
        uint16_t batchMask = 0;
        for (uint32_t lane = 0; lane < PIPELINE_DEPTH; ++lane) {
            const uint64_t chunkOffset = batchOffset + static_cast<uint64_t>(lane) * chunkLimit;
            if (chunkOffset >= rangeBytes) {
                break;
            }
            const uint64_t chunkBytes = rangeBytes - chunkOffset > chunkLimit ?
                chunkLimit : rangeBytes - chunkOffset;
            ccu::Variable offset;
            ccu::Variable bytes;
            offset = rangeOffset + chunkOffset;
            bytes = chunkBytes;

            ccu::LocalAddr localSource;
            localSource.addr = localAddr;
            localSource.addr += offset;
            localSource.token = localToken;

            ccu::RemoteAddr remoteDestination;
            remoteDestination.addr = remoteOutputs[channelIndex];
            remoteDestination.addr += offset;
            remoteDestination.token = remoteTokens[channelIndex];

            const uint16_t eventBit = static_cast<uint16_t>(1U << lane);
            if (reduce) {
                CCU_CHK_RET(ccu::WriteReduce(arg.channels[channelIndex], remoteDestination, localSource, bytes,
                    HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, dataEvent, eventBit));
            } else {
                CCU_CHK_RET(ccu::Write(arg.channels[channelIndex], remoteDestination, localSource, bytes,
                    dataEvent, eventBit));
            }
            batchMask = static_cast<uint16_t>(batchMask | eventBit);
        }
        CCU_CHK_RET(ccu::EventWait(dataEvent, batchMask));
    }
    return CCU_SUCCESS;
}

CcuResult CcuPipelineRecursiveHalvingDoublingKernel(AllReduceKernelArg &arg)
{
    std::vector<ccu::Variable> remoteOutputs(arg.channelCount);
    std::vector<ccu::Variable> remoteTokens(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        remoteOutputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN_ID);
        remoteTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN_ID);
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable unusedScratchAddr;
    ccu::Variable unusedScratchToken;
    uint32_t taskArgIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchToken, taskArgIndex++));

    const uint32_t powerOfTwo = LargestPowerOfTwo(arg.rankSize);
    const uint32_t remainder = arg.rankSize - powerOfTwo;
    ccu::Event dataEvent;
    if (remainder == 0) {
        const uint32_t firstDistance = powerOfTwo >> 1U;
        const uint32_t halfCount = powerOfTwo >> 1U;
        const bool keepLowerHalf = (arg.rankId & firstDistance) == 0;
        const uint32_t keepStart = keepLowerHalf ? 0U : halfCount;
        const uint64_t keepOffset = arg.segmentOffsets[keepStart];
        const uint64_t keepBytes = GetRangeBytes(arg, keepStart, halfCount, powerOfTwo);
        CCU_CHK_RET(PipelineLocalCopyRange(inputAddr, inputToken, outputAddr, outputToken,
            keepOffset, keepBytes, dataEvent));
    } else {
        CCU_CHK_RET(PipelineLocalCopyRange(inputAddr, inputToken, outputAddr, outputToken,
            0, arg.dataBytes, dataEvent));
    }
    CCU_CHK_RET(ExchangeOutputResources(arg, outputAddr, outputToken));

    uint32_t activeRank = 0;
    uint32_t pairChannelIndex = INVALID_CHANNEL_INDEX;

    if (remainder != 0 && arg.rankId < 2U * remainder) {
        const uint32_t pairRank = (arg.rankId & 1U) == 0 ? arg.rankId + 1U : arg.rankId - 1U;
        pairChannelIndex = FindChannelIndex(arg, pairRank);
        if (pairChannelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        const ChannelHandle pairChannel = arg.channels[pairChannelIndex];

        if ((arg.rankId & 1U) == 0) {
            CCU_CHK_RET(PipelinePushRange(arg, pairChannelIndex, remoteOutputs, remoteTokens,
                outputAddr, outputToken, 0, arg.dataBytes, true, dataEvent));
            CCU_CHK_RET(ccu::NotifyRecord(pairChannel, SYNC_NOTIFY_INDEX, RHD_FOLD_DONE_NOTIFY_BIT));
            CCU_CHK_RET(ccu::NotifyRecord(pairChannel, SYNC_NOTIFY_INDEX, RHD_FINAL_READY_NOTIFY_BIT));
            CCU_CHK_RET(ccu::NotifyWait(pairChannel, SYNC_NOTIFY_INDEX, RHD_FINAL_DONE_NOTIFY_BIT));
            return CCU_SUCCESS;
        }

        CCU_CHK_RET(ccu::NotifyWait(pairChannel, SYNC_NOTIFY_INDEX, RHD_FOLD_DONE_NOTIFY_BIT));
        activeRank = arg.rankId / 2U;
    } else {
        activeRank = arg.rankId - remainder;
    }

    uint32_t rangeStart = 0;
    uint32_t rangeCount = powerOfTwo;
    bool firstReduceScatterRound = true;
    for (uint32_t distance = powerOfTwo >> 1U; distance != 0; distance >>= 1U) {
        const uint32_t peerActiveRank = activeRank ^ distance;
        const uint32_t peerRank = ActiveRankToOriginal(peerActiveRank, remainder);
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        if (channelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        const ChannelHandle channel = arg.channels[channelIndex];

        const uint32_t halfCount = rangeCount >> 1U;
        const bool keepLowerHalf = (activeRank & distance) == 0;
        const uint32_t sendStart = keepLowerHalf ? rangeStart + halfCount : rangeStart;
        const uint64_t sendOffset = arg.segmentOffsets[sendStart];
        const uint64_t sendBytes = GetRangeBytes(arg, sendStart, halfCount, powerOfTwo);
        if (!keepLowerHalf) {
            rangeStart += halfCount;
        }
        rangeCount = halfCount;

        CCU_CHK_RET(ccu::NotifyRecord(channel, SYNC_NOTIFY_INDEX, RHD_RS_READY_NOTIFY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(channel, SYNC_NOTIFY_INDEX, RHD_RS_READY_NOTIFY_BIT));
        const ccu::Variable &sendAddr = firstReduceScatterRound && remainder == 0 ? inputAddr : outputAddr;
        const ccu::Variable &sendToken = firstReduceScatterRound && remainder == 0 ? inputToken : outputToken;
        CCU_CHK_RET(PipelinePushRange(arg, channelIndex, remoteOutputs, remoteTokens,
            sendAddr, sendToken, sendOffset, sendBytes, true, dataEvent));
        CCU_CHK_RET(ccu::NotifyRecord(channel, SYNC_NOTIFY_INDEX, RHD_RS_DONE_NOTIFY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(channel, SYNC_NOTIFY_INDEX, RHD_RS_DONE_NOTIFY_BIT));
        firstReduceScatterRound = false;
    }

    rangeStart = activeRank;
    rangeCount = 1;
    for (uint32_t distance = 1; distance < powerOfTwo; distance <<= 1U) {
        const uint32_t peerActiveRank = activeRank ^ distance;
        const uint32_t peerRank = ActiveRankToOriginal(peerActiveRank, remainder);
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        if (channelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        const ChannelHandle channel = arg.channels[channelIndex];
        const uint64_t sendOffset = arg.segmentOffsets[rangeStart];
        const uint64_t sendBytes = GetRangeBytes(arg, rangeStart, rangeCount, powerOfTwo);

        CCU_CHK_RET(ccu::NotifyRecord(channel, SYNC_NOTIFY_INDEX, RHD_AG_READY_NOTIFY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(channel, SYNC_NOTIFY_INDEX, RHD_AG_READY_NOTIFY_BIT));
        CCU_CHK_RET(PipelinePushRange(arg, channelIndex, remoteOutputs, remoteTokens,
            outputAddr, outputToken, sendOffset, sendBytes, false, dataEvent));
        CCU_CHK_RET(ccu::NotifyRecord(channel, SYNC_NOTIFY_INDEX, RHD_AG_DONE_NOTIFY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(channel, SYNC_NOTIFY_INDEX, RHD_AG_DONE_NOTIFY_BIT));

        if ((activeRank & distance) != 0) {
            rangeStart -= rangeCount;
        }
        rangeCount <<= 1U;
    }

    if (pairChannelIndex != INVALID_CHANNEL_INDEX) {
        const ChannelHandle pairChannel = arg.channels[pairChannelIndex];
        CCU_CHK_RET(ccu::NotifyWait(pairChannel, SYNC_NOTIFY_INDEX, RHD_FINAL_READY_NOTIFY_BIT));
        CCU_CHK_RET(PipelinePushRange(arg, pairChannelIndex, remoteOutputs, remoteTokens,
            outputAddr, outputToken, 0, arg.dataBytes, false, dataEvent));
        CCU_CHK_RET(ccu::NotifyRecord(pairChannel, SYNC_NOTIFY_INDEX, RHD_FINAL_DONE_NOTIFY_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult CcuRecursiveHalvingDoublingKernel(AllReduceKernelArg &arg)
{
    std::vector<ccu::Variable> remoteOutputs(arg.channelCount);
    std::vector<ccu::Variable> remoteTokens(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        remoteOutputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN_ID);
        remoteTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN_ID);
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable unusedScratchAddr;
    ccu::Variable unusedScratchToken;
    uint32_t taskArgIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchToken, taskArgIndex++));

    const uint32_t powerOfTwo = LargestPowerOfTwo(arg.rankSize);
    const uint32_t remainder = arg.rankSize - powerOfTwo;
    ccu::Event dataEvent;
    if (remainder == 0) {
        const uint32_t firstDistance = powerOfTwo >> 1U;
        const uint32_t halfCount = powerOfTwo >> 1U;
        const bool keepLowerHalf = (arg.rankId & firstDistance) == 0;
        const uint32_t keepStart = keepLowerHalf ? 0U : halfCount;
        const uint64_t keepOffset = arg.segmentOffsets[keepStart];
        const uint64_t keepBytes = GetRangeBytes(arg, keepStart, halfCount, powerOfTwo);
        CCU_CHK_RET(V18LocalCopyRange(inputAddr, inputToken, outputAddr, outputToken,
            keepOffset, keepBytes, dataEvent));
    } else {
        CCU_CHK_RET(V18LocalCopyRange(inputAddr, inputToken, outputAddr, outputToken,
            0, arg.dataBytes, dataEvent));
    }
    CCU_CHK_RET(ExchangeOutputResources(arg, outputAddr, outputToken));

    uint32_t activeRank = 0;
    uint32_t pairChannelIndex = INVALID_CHANNEL_INDEX;

    if (remainder != 0 && arg.rankId < 2U * remainder) {
        const uint32_t pairRank = (arg.rankId & 1U) == 0 ? arg.rankId + 1U : arg.rankId - 1U;
        pairChannelIndex = FindChannelIndex(arg, pairRank);
        if (pairChannelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        const ChannelHandle pairChannel = arg.channels[pairChannelIndex];

        if ((arg.rankId & 1U) == 0) {
            CCU_CHK_RET(V18PushRange(arg, pairChannelIndex, remoteOutputs, remoteTokens,
                outputAddr, outputToken, 0, arg.dataBytes, true, dataEvent));
            CCU_CHK_RET(ccu::NotifyRecord(pairChannel, SYNC_NOTIFY_INDEX, RHD_FOLD_DONE_NOTIFY_BIT));
            CCU_CHK_RET(ccu::NotifyRecord(pairChannel, SYNC_NOTIFY_INDEX, RHD_FINAL_READY_NOTIFY_BIT));
            CCU_CHK_RET(ccu::NotifyWait(pairChannel, SYNC_NOTIFY_INDEX, RHD_FINAL_DONE_NOTIFY_BIT));
            return CCU_SUCCESS;
        }

        CCU_CHK_RET(ccu::NotifyWait(pairChannel, SYNC_NOTIFY_INDEX, RHD_FOLD_DONE_NOTIFY_BIT));
        activeRank = arg.rankId / 2U;
    } else {
        activeRank = arg.rankId - remainder;
    }

    uint32_t rangeStart = 0;
    uint32_t rangeCount = powerOfTwo;
    bool firstReduceScatterRound = true;
    for (uint32_t distance = powerOfTwo >> 1U; distance != 0; distance >>= 1U) {
        const uint32_t peerActiveRank = activeRank ^ distance;
        const uint32_t peerRank = ActiveRankToOriginal(peerActiveRank, remainder);
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        if (channelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        const ChannelHandle channel = arg.channels[channelIndex];

        const uint32_t halfCount = rangeCount >> 1U;
        const bool keepLowerHalf = (activeRank & distance) == 0;
        const uint32_t sendStart = keepLowerHalf ? rangeStart + halfCount : rangeStart;
        const uint64_t sendOffset = arg.segmentOffsets[sendStart];
        const uint64_t sendBytes = GetRangeBytes(arg, sendStart, halfCount, powerOfTwo);
        if (!keepLowerHalf) {
            rangeStart += halfCount;
        }
        rangeCount = halfCount;

        CCU_CHK_RET(ccu::NotifyRecord(channel, SYNC_NOTIFY_INDEX, RHD_RS_READY_NOTIFY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(channel, SYNC_NOTIFY_INDEX, RHD_RS_READY_NOTIFY_BIT));
        const ccu::Variable &sendAddr = firstReduceScatterRound && remainder == 0 ? inputAddr : outputAddr;
        const ccu::Variable &sendToken = firstReduceScatterRound && remainder == 0 ? inputToken : outputToken;
        CCU_CHK_RET(V18PushRange(arg, channelIndex, remoteOutputs, remoteTokens,
            sendAddr, sendToken, sendOffset, sendBytes, true, dataEvent));
        CCU_CHK_RET(ccu::NotifyRecord(channel, SYNC_NOTIFY_INDEX, RHD_RS_DONE_NOTIFY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(channel, SYNC_NOTIFY_INDEX, RHD_RS_DONE_NOTIFY_BIT));
        firstReduceScatterRound = false;
    }

    rangeStart = activeRank;
    rangeCount = 1;
    for (uint32_t distance = 1; distance < powerOfTwo; distance <<= 1U) {
        const uint32_t peerActiveRank = activeRank ^ distance;
        const uint32_t peerRank = ActiveRankToOriginal(peerActiveRank, remainder);
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        if (channelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        const ChannelHandle channel = arg.channels[channelIndex];
        const uint64_t sendOffset = arg.segmentOffsets[rangeStart];
        const uint64_t sendBytes = GetRangeBytes(arg, rangeStart, rangeCount, powerOfTwo);

        CCU_CHK_RET(ccu::NotifyRecord(channel, SYNC_NOTIFY_INDEX, RHD_AG_READY_NOTIFY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(channel, SYNC_NOTIFY_INDEX, RHD_AG_READY_NOTIFY_BIT));
        CCU_CHK_RET(V18PushRange(arg, channelIndex, remoteOutputs, remoteTokens,
            outputAddr, outputToken, sendOffset, sendBytes, false, dataEvent));
        CCU_CHK_RET(ccu::NotifyRecord(channel, SYNC_NOTIFY_INDEX, RHD_AG_DONE_NOTIFY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(channel, SYNC_NOTIFY_INDEX, RHD_AG_DONE_NOTIFY_BIT));

        if ((activeRank & distance) != 0) {
            rangeStart -= rangeCount;
        }
        rangeCount <<= 1U;
    }

    if (pairChannelIndex != INVALID_CHANNEL_INDEX) {
        const ChannelHandle pairChannel = arg.channels[pairChannelIndex];
        CCU_CHK_RET(ccu::NotifyWait(pairChannel, SYNC_NOTIFY_INDEX, RHD_FINAL_READY_NOTIFY_BIT));
        CCU_CHK_RET(V18PushRange(arg, pairChannelIndex, remoteOutputs, remoteTokens,
            outputAddr, outputToken, 0, arg.dataBytes, false, dataEvent));
        CCU_CHK_RET(ccu::NotifyRecord(pairChannel, SYNC_NOTIFY_INDEX, RHD_FINAL_DONE_NOTIFY_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult CcuTopologyRecursiveHalvingDoublingKernel(AllReduceKernelArg &arg)
{
    if (arg.rankSize != 16U) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> remoteOutputs(arg.channelCount);
    std::vector<ccu::Variable> remoteTokens(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        remoteOutputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN_ID);
        remoteTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN_ID);
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable unusedScratchAddr;
    ccu::Variable unusedScratchToken;
    uint32_t taskArgIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchToken, taskArgIndex++));

    ccu::Event dataEvent;
    if (arg.topologyPhase == TopologyPhase::RHD_LOCAL_REDUCE) {
        constexpr uint32_t ownedSegmentCount = 2U;
        constexpr uint16_t readMask = (1U << (ownedSegmentCount * RHD_LOCAL_PEER_COUNT)) - 1U;
        if (arg.channelCount != RHD_LOCAL_PEER_COUNT) {
            return CCU_E_PARA;
        }

        std::vector<ccu::Variable> remoteInputs(arg.channelCount);
        std::vector<ccu::Variable> remoteInputTokens(arg.channelCount);
        for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
            remoteInputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_ADDR_XN_ID);
            remoteInputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_TOKEN_XN_ID);
        }
        CCU_CHK_RET(ExchangeHierarchicalResources(arg, inputAddr, inputToken, outputAddr, outputToken));

        const uint32_t localPosition = arg.rankId % 8U;
        const uint32_t ownedSegments[ownedSegmentCount] = {localPosition, localPosition + 8U};
        uint64_t segmentOffsets[ownedSegmentCount] {};
        uint64_t stripeOffsets[ownedSegmentCount][RHD_LOCAL_PEER_COUNT] {};
        uint64_t stripeBytes[ownedSegmentCount][RHD_LOCAL_PEER_COUNT] {};
        uint64_t stripeBaseBytes[ownedSegmentCount] {};
        uint32_t stripeRemainders[ownedSegmentCount] {};

        for (uint32_t lane = 0; lane < ownedSegmentCount; ++lane) {
            const uint32_t segment = ownedSegments[lane];
            segmentOffsets[lane] = arg.segmentOffsets[segment];
            const uint64_t segmentBytes = arg.segmentBytes[segment];
            if (segmentBytes == 0 || segmentBytes % sizeof(float) != 0) {
                return CCU_E_PARA;
            }
            CCU_CHK_RET(V18LocalCopyRange(inputAddr, inputToken, outputAddr, outputToken,
                segmentOffsets[lane], segmentBytes, dataEvent));

            const uint64_t segmentElements = segmentBytes / sizeof(float);
            const uint64_t stripeBaseElements = segmentElements / RHD_LOCAL_PEER_COUNT;
            stripeRemainders[lane] = static_cast<uint32_t>(segmentElements % RHD_LOCAL_PEER_COUNT);
            stripeBaseBytes[lane] = stripeBaseElements * sizeof(float);
            if (stripeBaseBytes[lane] == 0 ||
                stripeBaseBytes[lane] + sizeof(float) > MAX_DATA_SIZE) {
                return CCU_E_PARA;
            }

            uint64_t stripeOffset = segmentOffsets[lane];
            for (uint32_t stripe = 0; stripe < RHD_LOCAL_PEER_COUNT; ++stripe) {
                stripeOffsets[lane][stripe] = stripeOffset;
                stripeBytes[lane][stripe] = stripeBaseBytes[lane] +
                    (stripe < stripeRemainders[lane] ? sizeof(float) : 0U);
                stripeOffset += stripeBytes[lane][stripe];
            }
        }

        std::vector<ccu::Variable> stripeIndices(RHD_LOCAL_PEER_COUNT);
        std::vector<std::vector<ccu::Variable>> currentBytes(ownedSegmentCount);
        std::vector<std::vector<ccu::LocalAddr>> localDestinations(ownedSegmentCount);
        std::vector<std::vector<ccu::RemoteAddr>> remoteSources(ownedSegmentCount);
        for (uint32_t lane = 0; lane < ownedSegmentCount; ++lane) {
            currentBytes[lane].resize(RHD_LOCAL_PEER_COUNT);
            localDestinations[lane].resize(RHD_LOCAL_PEER_COUNT);
            remoteSources[lane].resize(RHD_LOCAL_PEER_COUNT);
        }

        for (uint32_t channelIndex = 0; channelIndex < RHD_LOCAL_PEER_COUNT; ++channelIndex) {
            stripeIndices[channelIndex] = channelIndex;
            for (uint32_t lane = 0; lane < ownedSegmentCount; ++lane) {
                ccu::Variable offsetVar;
                offsetVar = stripeOffsets[lane][channelIndex];
                currentBytes[lane][channelIndex] = stripeBytes[lane][channelIndex];

                localDestinations[lane][channelIndex].addr = outputAddr;
                localDestinations[lane][channelIndex].addr += offsetVar;
                localDestinations[lane][channelIndex].token = outputToken;
                remoteSources[lane][channelIndex].addr = remoteInputs[channelIndex];
                remoteSources[lane][channelIndex].addr += offsetVar;
                remoteSources[lane][channelIndex].token = remoteInputTokens[channelIndex];
            }
        }

        ccu::Variable round;
        ccu::Variable one;
        round = 0U;
        one = 1U;
        CCU_WHILE(round != RHD_LOCAL_PEER_COUNT) {
            for (uint32_t channelIndex = 0; channelIndex < RHD_LOCAL_PEER_COUNT; ++channelIndex) {
                for (uint32_t lane = 0; lane < ownedSegmentCount; ++lane) {
                    const uint32_t eventIndex = lane * RHD_LOCAL_PEER_COUNT + channelIndex;
                    const uint16_t eventBit = static_cast<uint16_t>(1U << eventIndex);
                    CCU_CHK_RET(ccu::ReadReduce(arg.channels[channelIndex],
                        localDestinations[lane][channelIndex], remoteSources[lane][channelIndex],
                        currentBytes[lane][channelIndex], HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
                        dataEvent, eventBit));
                }
            }
            CCU_CHK_RET(ccu::EventWait(dataEvent, readMask));

            for (uint32_t channelIndex = 0; channelIndex < RHD_LOCAL_PEER_COUNT; ++channelIndex) {
                for (uint32_t lane = 0; lane < ownedSegmentCount; ++lane) {
                    localDestinations[lane][channelIndex].addr += currentBytes[lane][channelIndex];
                    remoteSources[lane][channelIndex].addr += currentBytes[lane][channelIndex];
                }
                stripeIndices[channelIndex] += one;
                CCU_IF(stripeIndices[channelIndex] == RHD_LOCAL_PEER_COUNT) {
                    stripeIndices[channelIndex] = 0U;
                    for (uint32_t lane = 0; lane < ownedSegmentCount; ++lane) {
                        ccu::Variable segmentOffsetVar;
                        segmentOffsetVar = segmentOffsets[lane];
                        localDestinations[lane][channelIndex].addr = outputAddr;
                        localDestinations[lane][channelIndex].addr += segmentOffsetVar;
                        remoteSources[lane][channelIndex].addr = remoteInputs[channelIndex];
                        remoteSources[lane][channelIndex].addr += segmentOffsetVar;
                    }
                }
                for (uint32_t lane = 0; lane < ownedSegmentCount; ++lane) {
                    if (stripeRemainders[lane] != 0U) {
                        CCU_IF(stripeIndices[channelIndex] == stripeRemainders[lane]) {
                            currentBytes[lane][channelIndex] = stripeBaseBytes[lane];
                        }
                        CCU_IF(stripeIndices[channelIndex] == 0U) {
                            currentBytes[lane][channelIndex] = stripeBaseBytes[lane] + sizeof(float);
                        }
                    }
                }
            }
            round += one;
        }
        return CCU_SUCCESS;
    }

    if (arg.topologyPhase == TopologyPhase::RHD_CROSS_REDUCE) {
        CCU_CHK_RET(ExchangeOutputResources(arg, outputAddr, outputToken));
        const uint32_t channelIndex = FindChannelIndex(arg, arg.rankId ^ 8U);
        if (channelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        const ChannelHandle channel = arg.channels[channelIndex];
        const uint32_t localPosition = arg.rankId % 8U;

        CCU_CHK_RET(ccu::NotifyRecord(channel, SYNC_NOTIFY_INDEX, RHD_RS_READY_NOTIFY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(channel, SYNC_NOTIFY_INDEX, RHD_RS_READY_NOTIFY_BIT));
        if (arg.rankId >= 8U) {
            for (uint32_t lane = 0; lane < 2U; ++lane) {
                const uint32_t segment = localPosition + lane * 8U;
                CCU_CHK_RET(V18PushRange(arg, channelIndex, remoteOutputs, remoteTokens,
                    outputAddr, outputToken, arg.segmentOffsets[segment],
                    arg.segmentBytes[segment], true, dataEvent));
            }
            CCU_CHK_RET(ccu::NotifyRecord(channel, SYNC_NOTIFY_INDEX, RHD_RS_DONE_NOTIFY_BIT));
            CCU_CHK_RET(ccu::NotifyWait(channel, SYNC_NOTIFY_INDEX, RHD_FINAL_DONE_NOTIFY_BIT));
        } else {
            CCU_CHK_RET(ccu::NotifyWait(channel, SYNC_NOTIFY_INDEX, RHD_RS_DONE_NOTIFY_BIT));
            for (uint32_t lane = 0; lane < 2U; ++lane) {
                const uint32_t segment = localPosition + lane * 8U;
                CCU_CHK_RET(V18PushRange(arg, channelIndex, remoteOutputs, remoteTokens,
                    outputAddr, outputToken, arg.segmentOffsets[segment],
                    arg.segmentBytes[segment], false, dataEvent));
            }
            CCU_CHK_RET(ccu::NotifyRecord(channel, SYNC_NOTIFY_INDEX, RHD_FINAL_DONE_NOTIFY_BIT));
        }
        return CCU_SUCCESS;
    }

    if (arg.topologyPhase == TopologyPhase::RHD_LOCAL_GATHER) {
        constexpr uint32_t localPeerCount = 7U;
        constexpr uint32_t segmentsPerPeer = 2U;
        if (arg.channelCount != localPeerCount) {
            return CCU_E_PARA;
        }
        CCU_CHK_RET(ExchangeOutputResources(arg, outputAddr, outputToken));

        uint64_t maxSegmentBytes = 0;
        for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
            const uint32_t peerPosition = arg.remoteRanks[idx] % 8U;
            for (uint32_t lane = 0; lane < segmentsPerPeer; ++lane) {
                const uint32_t segment = peerPosition + lane * 8U;
                if (arg.segmentBytes[segment] > maxSegmentBytes) {
                    maxSegmentBytes = arg.segmentBytes[segment];
                }
            }
        }

        for (uint64_t batchOffset = 0; batchOffset < maxSegmentBytes; batchOffset += MAX_DATA_SIZE) {
            uint16_t readMask = 0;
            for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
                const uint32_t peerPosition = arg.remoteRanks[idx] % 8U;
                for (uint32_t lane = 0; lane < segmentsPerPeer; ++lane) {
                    const uint32_t segment = peerPosition + lane * 8U;
                    const uint64_t segmentBytes = arg.segmentBytes[segment];
                    if (batchOffset >= segmentBytes) {
                        continue;
                    }
                    const uint64_t chunkBytes = segmentBytes - batchOffset > MAX_DATA_SIZE ?
                        MAX_DATA_SIZE : segmentBytes - batchOffset;
                    ccu::Variable offsetVar;
                    ccu::Variable bytesVar;
                    offsetVar = arg.segmentOffsets[segment] + batchOffset;
                    bytesVar = chunkBytes;
                    ccu::LocalAddr localDestination;
                    localDestination.addr = outputAddr;
                    localDestination.addr += offsetVar;
                    localDestination.token = outputToken;
                    ccu::RemoteAddr remoteSource;
                    remoteSource.addr = remoteOutputs[idx];
                    remoteSource.addr += offsetVar;
                    remoteSource.token = remoteTokens[idx];
                    const uint32_t eventIndex = idx * segmentsPerPeer + lane;
                    const uint16_t eventBit = static_cast<uint16_t>(1U << eventIndex);
                    CCU_CHK_RET(ccu::Read(arg.channels[idx], localDestination, remoteSource,
                        bytesVar, dataEvent, eventBit));
                    readMask = static_cast<uint16_t>(readMask | eventBit);
                }
            }
            if (readMask != 0) {
                CCU_CHK_RET(ccu::EventWait(dataEvent, readMask));
            }
        }
        return CCU_SUCCESS;
    }
    return CCU_E_PARA;
}

CcuResult DualPeerBarrier(const AllReduceKernelArg &arg, const uint32_t channelIndices[2],
    uint16_t notifyBit)
{
    if (channelIndices[0] == INVALID_CHANNEL_INDEX || channelIndices[1] == INVALID_CHANNEL_INDEX ||
        channelIndices[0] == channelIndices[1]) {
        return CCU_E_PARA;
    }
    for (uint32_t lane = 0; lane < 2U; ++lane) {
        CCU_CHK_RET(ccu::NotifyRecord(arg.channels[channelIndices[lane]],
            SYNC_NOTIFY_INDEX, notifyBit));
    }
    for (uint32_t lane = 0; lane < 2U; ++lane) {
        CCU_CHK_RET(ccu::NotifyWait(arg.channels[channelIndices[lane]],
            SYNC_NOTIFY_INDEX, notifyBit));
    }
    return CCU_SUCCESS;
}

CcuResult PushDualRanges(const AllReduceKernelArg &arg, const uint32_t channelIndices[2],
    const std::vector<ccu::Variable> &remoteOutputs, const std::vector<ccu::Variable> &remoteTokens,
    const ccu::Variable &localAddr, const ccu::Variable &localToken, const uint64_t rangeOffsets[2],
    const uint64_t rangeBytes[2], bool reduce, ccu::Event &dataEvent)
{
    if (channelIndices[0] == INVALID_CHANNEL_INDEX || channelIndices[1] == INVALID_CHANNEL_INDEX ||
        channelIndices[0] == channelIndices[1]) {
        return CCU_E_PARA;
    }
    const uint64_t chunkLimit = PIPELINE_CHUNK_BYTES < MAX_DATA_SIZE ?
        PIPELINE_CHUNK_BYTES : MAX_DATA_SIZE;
    uint64_t batchOffsets[2] = {0, 0};
    bool hasPending = true;
    while (hasPending) {
        hasPending = false;
        uint16_t eventMask = 0;
        for (uint32_t range = 0; range < 2U; ++range) {
            for (uint32_t lane = 0; lane < PIPELINE_DEPTH; ++lane) {
                const uint64_t chunkOffset = batchOffsets[range] + static_cast<uint64_t>(lane) * chunkLimit;
                if (chunkOffset >= rangeBytes[range]) {
                    continue;
                }
                hasPending = true;
                const uint64_t chunkBytes = rangeBytes[range] - chunkOffset > chunkLimit ?
                    chunkLimit : rangeBytes[range] - chunkOffset;
                ccu::Variable offset;
                ccu::Variable bytes;
                offset = rangeOffsets[range] + chunkOffset;
                bytes = chunkBytes;

                ccu::LocalAddr localSource;
                localSource.addr = localAddr;
                localSource.addr += offset;
                localSource.token = localToken;
                ccu::RemoteAddr remoteDestination;
                remoteDestination.addr = remoteOutputs[channelIndices[range]];
                remoteDestination.addr += offset;
                remoteDestination.token = remoteTokens[channelIndices[range]];

                const uint32_t eventIndex = range * PIPELINE_DEPTH + lane;
                const uint16_t eventBit = static_cast<uint16_t>(1U << eventIndex);
                if (reduce) {
                    CCU_CHK_RET(ccu::WriteReduce(arg.channels[channelIndices[range]], remoteDestination,
                        localSource, bytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, dataEvent, eventBit));
                } else {
                    CCU_CHK_RET(ccu::Write(arg.channels[channelIndices[range]], remoteDestination,
                        localSource, bytes, dataEvent, eventBit));
                }
                eventMask = static_cast<uint16_t>(eventMask | eventBit);
            }
            batchOffsets[range] += chunkLimit * PIPELINE_DEPTH;
        }
        if (eventMask != 0) {
            CCU_CHK_RET(ccu::EventWait(dataEvent, eventMask));
        }
    }
    return CCU_SUCCESS;
}

[[maybe_unused]] CcuResult CcuDualLaneRecursiveHalvingDoublingKernel(AllReduceKernelArg &arg)
{
    if (arg.rankSize != 4U && arg.rankSize != 16U) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> remoteOutputs(arg.channelCount);
    std::vector<ccu::Variable> remoteTokens(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        remoteOutputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN_ID);
        remoteTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN_ID);
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable unusedScratchAddr;
    ccu::Variable unusedScratchToken;
    uint32_t taskArgIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchToken, taskArgIndex++));

    const uint32_t rankBits = RankBitCount(arg.rankSize);
    const uint32_t virtualRanks[2] = {arg.rankId, RotateRankLeftOne(arg.rankId, rankBits)};
    uint32_t rangeStarts[2] = {0, 0};
    uint32_t rangeCounts[2] = {arg.rankSize, arg.rankSize};
    ccu::Event dataEvent;

    const uint32_t firstDistance = arg.rankSize >> 1U;
    uint64_t keepOffsets[MAX_RANK_SIZE] {};
    uint64_t keepBytes[MAX_RANK_SIZE] {};
    for (uint32_t lane = 0; lane < 2U; ++lane) {
        const bool keepLowerHalf = (virtualRanks[lane] & firstDistance) == 0;
        const uint32_t keepStart = keepLowerHalf ? 0U : firstDistance;
        keepOffsets[lane] = arg.laneSegmentOffsets[lane][keepStart];
        keepBytes[lane] = GetLaneRangeBytes(arg, lane, keepStart, firstDistance);
    }
    CCU_CHK_RET(LocalCopyRanges(inputAddr, inputToken, outputAddr, outputToken,
        keepOffsets, keepBytes, 2U, dataEvent));
    CCU_CHK_RET(ExchangeOutputResources(arg, outputAddr, outputToken));

    bool firstReduceScatterRound = true;
    for (uint32_t distance = firstDistance; distance != 0; distance >>= 1U) {
        uint32_t channelIndices[2] = {INVALID_CHANNEL_INDEX, INVALID_CHANNEL_INDEX};
        uint64_t sendOffsets[2] = {0, 0};
        uint64_t sendBytes[2] = {0, 0};
        for (uint32_t lane = 0; lane < 2U; ++lane) {
            const uint32_t peerVirtualRank = virtualRanks[lane] ^ distance;
            const uint32_t peerRank = lane == 0U ? peerVirtualRank :
                RotateRankRightOne(peerVirtualRank, rankBits);
            channelIndices[lane] = FindChannelIndex(arg, peerRank);
            const uint32_t halfCount = rangeCounts[lane] >> 1U;
            const bool keepLowerHalf = (virtualRanks[lane] & distance) == 0;
            const uint32_t sendStart = keepLowerHalf ?
                rangeStarts[lane] + halfCount : rangeStarts[lane];
            sendOffsets[lane] = arg.laneSegmentOffsets[lane][sendStart];
            sendBytes[lane] = GetLaneRangeBytes(arg, lane, sendStart, halfCount);
            if (!keepLowerHalf) {
                rangeStarts[lane] += halfCount;
            }
            rangeCounts[lane] = halfCount;
        }

        CCU_CHK_RET(DualPeerBarrier(arg, channelIndices, RHD_RS_READY_NOTIFY_BIT));
        const ccu::Variable &sendAddr = firstReduceScatterRound ? inputAddr : outputAddr;
        const ccu::Variable &sendToken = firstReduceScatterRound ? inputToken : outputToken;
        CCU_CHK_RET(PushDualRanges(arg, channelIndices, remoteOutputs, remoteTokens,
            sendAddr, sendToken, sendOffsets, sendBytes, true, dataEvent));
        CCU_CHK_RET(DualPeerBarrier(arg, channelIndices, RHD_RS_DONE_NOTIFY_BIT));
        firstReduceScatterRound = false;
    }

    rangeStarts[0] = virtualRanks[0];
    rangeStarts[1] = virtualRanks[1];
    rangeCounts[0] = 1;
    rangeCounts[1] = 1;
    for (uint32_t distance = 1; distance < arg.rankSize; distance <<= 1U) {
        uint32_t channelIndices[2] = {INVALID_CHANNEL_INDEX, INVALID_CHANNEL_INDEX};
        uint64_t sendOffsets[2] = {0, 0};
        uint64_t sendBytes[2] = {0, 0};
        for (uint32_t lane = 0; lane < 2U; ++lane) {
            const uint32_t peerVirtualRank = virtualRanks[lane] ^ distance;
            const uint32_t peerRank = lane == 0U ? peerVirtualRank :
                RotateRankRightOne(peerVirtualRank, rankBits);
            channelIndices[lane] = FindChannelIndex(arg, peerRank);
            sendOffsets[lane] = arg.laneSegmentOffsets[lane][rangeStarts[lane]];
            sendBytes[lane] = GetLaneRangeBytes(arg, lane, rangeStarts[lane], rangeCounts[lane]);
        }

        CCU_CHK_RET(DualPeerBarrier(arg, channelIndices, RHD_AG_READY_NOTIFY_BIT));
        CCU_CHK_RET(PushDualRanges(arg, channelIndices, remoteOutputs, remoteTokens,
            outputAddr, outputToken, sendOffsets, sendBytes, false, dataEvent));
        CCU_CHK_RET(DualPeerBarrier(arg, channelIndices, RHD_AG_DONE_NOTIFY_BIT));

        for (uint32_t lane = 0; lane < 2U; ++lane) {
            if ((virtualRanks[lane] & distance) != 0) {
                rangeStarts[lane] -= rangeCounts[lane];
            }
            rangeCounts[lane] <<= 1U;
        }
    }
    return CCU_SUCCESS;
}

CcuResult HighRadixInputStage(const AllReduceKernelArg &arg, uint32_t groupBase, uint32_t groupStride,
    uint32_t groupSize, const std::vector<ccu::Variable> &remoteInputs,
    const std::vector<ccu::Variable> &remoteTokens, const ccu::Variable &inputAddr,
    const ccu::Variable &inputToken, const ccu::Variable &resultAddr,
    const ccu::Variable &resultToken, uint64_t resultOffset, const ccu::Variable &scratchAddr,
    const ccu::Variable &scratchToken, const ccu::Variable &dataBytes, ccu::Event &dataEvent)
{
    if (groupSize < 2 || groupSize > HIGH_RADIX) {
        return CCU_E_PARA;
    }
    const uint32_t localPosition = (arg.rankId - groupBase) / groupStride;
    uint16_t readMask = 0;
    for (uint32_t position = 0; position < groupSize; ++position) {
        const uint32_t peerRank = groupBase + position * groupStride;
        if (peerRank == arg.rankId) {
            continue;
        }
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        if (channelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        const uint32_t scratchSlot = position < localPosition ? position : position - 1U;
        ccu::Variable slotOffset;
        slotOffset = static_cast<uint64_t>(scratchSlot) * arg.dataBytes;
        ccu::LocalAddr snapshot;
        snapshot.addr = scratchAddr;
        snapshot.addr += slotOffset;
        snapshot.token = scratchToken;
        ccu::RemoteAddr remoteInput;
        remoteInput.addr = remoteInputs[channelIndex];
        remoteInput.token = remoteTokens[channelIndex];
        const uint16_t eventBit = static_cast<uint16_t>(1U << position);
        CCU_CHK_RET(ccu::Read(arg.channels[channelIndex], snapshot, remoteInput,
            dataBytes, dataEvent, eventBit));
        readMask = static_cast<uint16_t>(readMask | eventBit);
    }
    CCU_CHK_RET(ccu::EventWait(dataEvent, readMask));

    ccu::Variable resultOffsetVar;
    resultOffsetVar = resultOffset;
    ccu::LocalAddr result;
    result.addr = resultAddr;
    result.addr += resultOffsetVar;
    result.token = resultToken;
    ccu::LocalAddr firstSource;
    if (localPosition == 0) {
        firstSource.addr = inputAddr;
        firstSource.token = inputToken;
    } else {
        firstSource.addr = scratchAddr;
        firstSource.token = scratchToken;
    }
    CCU_CHK_RET(ccu::LocalCopy(result, firstSource, dataBytes, dataEvent));
    CCU_CHK_RET(ccu::EventWait(dataEvent));
    for (uint32_t position = 1; position < groupSize; ++position) {
        ccu::LocalAddr source;
        if (position == localPosition) {
            source.addr = inputAddr;
            source.token = inputToken;
        } else {
            const uint32_t scratchSlot = position < localPosition ? position : position - 1U;
            ccu::Variable slotOffset;
            slotOffset = static_cast<uint64_t>(scratchSlot) * arg.dataBytes;
            source.addr = scratchAddr;
            source.addr += slotOffset;
            source.token = scratchToken;
        }
        CCU_CHK_RET(ccu::LocalReduce(result, source, dataBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, dataEvent));
        CCU_CHK_RET(ccu::EventWait(dataEvent));
    }
    return CCU_SUCCESS;
}

CcuResult HighRadixImmutableStage(const AllReduceKernelArg &arg, uint32_t groupBase,
    uint32_t groupStride, uint32_t groupSize, const std::vector<ccu::Variable> &remoteScratch,
    const std::vector<ccu::Variable> &remoteScratchTokens, const ccu::Variable &outputAddr,
    const ccu::Variable &outputToken, const ccu::Variable &scratchAddr,
    const ccu::Variable &scratchToken, const ccu::Variable &dataBytes, ccu::Event &dataEvent)
{
    if (groupSize < 2 || groupSize > HIGH_RADIX) {
        return CCU_E_PARA;
    }
    const uint64_t immutableResultOffset = static_cast<uint64_t>(HIGH_RADIX - 1U) * arg.dataBytes;
    ccu::Variable immutableResultOffsetVar;
    immutableResultOffsetVar = immutableResultOffset;

    for (uint32_t position = 0; position < groupSize; ++position) {
        const uint32_t peerRank = groupBase + position * groupStride;
        if (peerRank == arg.rankId) {
            continue;
        }
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        if (channelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        CCU_CHK_RET(ccu::NotifyRecord(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
            HIGH_RADIX_READY_NOTIFY_BIT));
    }
    for (uint32_t position = 0; position < groupSize; ++position) {
        const uint32_t peerRank = groupBase + position * groupStride;
        if (peerRank == arg.rankId) {
            continue;
        }
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        CCU_CHK_RET(ccu::NotifyWait(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
            HIGH_RADIX_READY_NOTIFY_BIT));
    }

    uint16_t readMask = 0;
    uint32_t remoteSlot = 0;
    for (uint32_t position = 0; position < groupSize; ++position) {
        const uint32_t peerRank = groupBase + position * groupStride;
        if (peerRank == arg.rankId) {
            continue;
        }
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        ccu::Variable slotOffset;
        slotOffset = static_cast<uint64_t>(remoteSlot) * arg.dataBytes;
        ccu::LocalAddr snapshot;
        snapshot.addr = scratchAddr;
        snapshot.addr += slotOffset;
        snapshot.token = scratchToken;
        ccu::RemoteAddr remoteResult;
        remoteResult.addr = remoteScratch[channelIndex];
        remoteResult.addr += immutableResultOffsetVar;
        remoteResult.token = remoteScratchTokens[channelIndex];
        const uint16_t eventBit = static_cast<uint16_t>(1U << position);
        CCU_CHK_RET(ccu::Read(arg.channels[channelIndex], snapshot, remoteResult,
            dataBytes, dataEvent, eventBit));
        readMask = static_cast<uint16_t>(readMask | eventBit);
        ++remoteSlot;
    }
    CCU_CHK_RET(ccu::EventWait(dataEvent, readMask));

    ccu::LocalAddr output;
    output.addr = outputAddr;
    output.token = outputToken;
    ccu::LocalAddr ownResult;
    ownResult.addr = scratchAddr;
    ownResult.addr += immutableResultOffsetVar;
    ownResult.token = scratchToken;
    CCU_CHK_RET(ccu::LocalCopy(output, ownResult, dataBytes, dataEvent));
    CCU_CHK_RET(ccu::EventWait(dataEvent));
    for (uint32_t slot = 0; slot + 1U < groupSize; ++slot) {
        ccu::Variable slotOffset;
        slotOffset = static_cast<uint64_t>(slot) * arg.dataBytes;
        ccu::LocalAddr snapshot;
        snapshot.addr = scratchAddr;
        snapshot.addr += slotOffset;
        snapshot.token = scratchToken;
        CCU_CHK_RET(ccu::LocalReduce(output, snapshot, dataBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, dataEvent));
        CCU_CHK_RET(ccu::EventWait(dataEvent));
    }
    return CCU_SUCCESS;
}

CcuResult CcuRecursiveDoublingKernel(AllReduceKernelArg &arg)
{
    std::vector<ccu::Variable> remoteInputs(arg.channelCount);
    std::vector<ccu::Variable> remoteInputTokens(arg.channelCount);
    std::vector<ccu::Variable> remoteScratch(arg.channelCount);
    std::vector<ccu::Variable> remoteScratchTokens(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        remoteInputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_ADDR_XN_ID);
        remoteInputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_TOKEN_XN_ID);
        remoteScratch[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], SCRATCH_ADDR_XN_ID);
        remoteScratchTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], SCRATCH_TOKEN_XN_ID);
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    uint32_t taskArgIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(scratchAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(scratchToken, taskArgIndex++));

    ccu::Variable dataBytes;
    dataBytes = arg.dataBytes;
    ccu::Event dataEvent;
    const bool hasSecondStage = arg.rankSize > HIGH_RADIX;
    CCU_CHK_RET(PublishSmallDataResources(arg, inputAddr, inputToken, scratchAddr, scratchToken,
        hasSecondStage));
    CCU_CHK_RET(WaitSmallInputResources(arg));

    const uint32_t localDigit = arg.rankId % HIGH_RADIX;
    const uint32_t lowGroupBase = arg.rankId - localDigit;
    const uint32_t highGroupSize = arg.rankSize / HIGH_RADIX;
    if (hasSecondStage) {
        const uint64_t immutableResultOffset = static_cast<uint64_t>(HIGH_RADIX - 1U) * arg.dataBytes;
        CCU_CHK_RET(HighRadixInputStage(arg, lowGroupBase, 1, HIGH_RADIX,
            remoteInputs, remoteInputTokens, inputAddr, inputToken, scratchAddr, scratchToken,
            immutableResultOffset, scratchAddr, scratchToken, dataBytes, dataEvent));
        CCU_CHK_RET(WaitSmallScratchResources(arg));
        CCU_CHK_RET(HighRadixImmutableStage(arg, localDigit, HIGH_RADIX, highGroupSize,
            remoteScratch, remoteScratchTokens, outputAddr, outputToken, scratchAddr, scratchToken,
            dataBytes, dataEvent));
    } else {
        CCU_CHK_RET(HighRadixInputStage(arg, lowGroupBase, 1, HIGH_RADIX,
            remoteInputs, remoteInputTokens, inputAddr, inputToken, outputAddr, outputToken,
            0, scratchAddr, scratchToken, dataBytes, dataEvent));
    }
    return CCU_SUCCESS;
}

CcuResult CcuRadix4RsagKernel(AllReduceKernelArg &arg)
{
    if (arg.rankSize != HIGH_RADIX) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> remoteInputs(arg.channelCount);
    std::vector<ccu::Variable> remoteInputTokens(arg.channelCount);
    std::vector<ccu::Variable> remoteOutputs(arg.channelCount);
    std::vector<ccu::Variable> remoteOutputTokens(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        remoteInputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_ADDR_XN_ID);
        remoteInputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_TOKEN_XN_ID);
        remoteOutputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN_ID);
        remoteOutputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN_ID);
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    uint32_t taskArgIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(scratchAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(scratchToken, taskArgIndex++));
    CCU_CHK_RET(ExchangeHierarchicalResources(arg, inputAddr, inputToken, outputAddr, outputToken));

    const uint32_t ownedSegment = arg.rankId;
    const uint64_t segmentOffset = arg.segmentOffsets[ownedSegment];
    const uint64_t segmentBytes = arg.segmentBytes[ownedSegment];
    if (segmentBytes == 0 || segmentBytes > MAX_DATA_SIZE) {
        return CCU_E_PARA;
    }

    ccu::Event dataEvent;
    for (uint64_t batchOffset = 0; batchOffset < segmentBytes; batchOffset += HIERARCHICAL_BATCH_CHUNK_BYTES) {
        const uint64_t batchBytes = segmentBytes - batchOffset > HIERARCHICAL_BATCH_CHUNK_BYTES ?
            HIERARCHICAL_BATCH_CHUNK_BYTES : segmentBytes - batchOffset;
        ccu::Variable dataOffsetVar;
        ccu::Variable batchBytesVar;
        dataOffsetVar = segmentOffset + batchOffset;
        batchBytesVar = batchBytes;

        uint16_t readMask = 0;
        for (uint32_t position = 0; position < HIGH_RADIX; ++position) {
            if (position == arg.rankId) {
                continue;
            }
            const uint32_t channelIndex = FindChannelIndex(arg, position);
            if (channelIndex == INVALID_CHANNEL_INDEX) {
                return CCU_E_PARA;
            }
            const uint32_t scratchSlot = position < arg.rankId ? position : position - 1U;
            ccu::Variable scratchOffset;
            scratchOffset = static_cast<uint64_t>(scratchSlot) * batchBytes;
            ccu::LocalAddr snapshot;
            snapshot.addr = scratchAddr;
            snapshot.addr += scratchOffset;
            snapshot.token = scratchToken;
            ccu::RemoteAddr remoteInput;
            remoteInput.addr = remoteInputs[channelIndex];
            remoteInput.addr += dataOffsetVar;
            remoteInput.token = remoteInputTokens[channelIndex];
            const uint16_t eventBit = static_cast<uint16_t>(1U << position);
            CCU_CHK_RET(ccu::Read(arg.channels[channelIndex], snapshot, remoteInput,
                batchBytesVar, dataEvent, eventBit));
            readMask = static_cast<uint16_t>(readMask | eventBit);
        }
        CCU_CHK_RET(ccu::EventWait(dataEvent, readMask));

        ccu::LocalAddr batchOutput;
        batchOutput.addr = outputAddr;
        batchOutput.addr += dataOffsetVar;
        batchOutput.token = outputToken;
        ccu::LocalAddr firstSource;
        if (arg.rankId == 0) {
            firstSource.addr = inputAddr;
            firstSource.addr += dataOffsetVar;
            firstSource.token = inputToken;
        } else {
            firstSource.addr = scratchAddr;
            firstSource.token = scratchToken;
        }
        CCU_CHK_RET(ccu::LocalCopy(batchOutput, firstSource, batchBytesVar, dataEvent));
        CCU_CHK_RET(ccu::EventWait(dataEvent));
        for (uint32_t position = 1; position < HIGH_RADIX; ++position) {
            ccu::LocalAddr reduceSource;
            if (position == arg.rankId) {
                reduceSource.addr = inputAddr;
                reduceSource.addr += dataOffsetVar;
                reduceSource.token = inputToken;
            } else {
                const uint32_t scratchSlot = position < arg.rankId ? position : position - 1U;
                ccu::Variable scratchOffset;
                scratchOffset = static_cast<uint64_t>(scratchSlot) * batchBytes;
                reduceSource.addr = scratchAddr;
                reduceSource.addr += scratchOffset;
                reduceSource.token = scratchToken;
            }
            CCU_CHK_RET(ccu::LocalReduce(batchOutput, reduceSource, batchBytesVar,
                HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, dataEvent));
            CCU_CHK_RET(ccu::EventWait(dataEvent));
        }
    }

    CCU_CHK_RET(HierarchicalBarrier(arg, 0, 1, HIGH_RADIX, RADIX4_RS_READY_NOTIFY_BIT));
    uint16_t gatherMask = 0;
    for (uint32_t position = 0; position < HIGH_RADIX; ++position) {
        if (position == arg.rankId) {
            continue;
        }
        const uint32_t channelIndex = FindChannelIndex(arg, position);
        if (channelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        const uint64_t gatherOffset = arg.segmentOffsets[position];
        const uint64_t gatherBytes = arg.segmentBytes[position];
        if (gatherBytes == 0 || gatherBytes > MAX_DATA_SIZE) {
            return CCU_E_PARA;
        }
        ccu::Variable gatherOffsetVar;
        ccu::Variable gatherBytesVar;
        gatherOffsetVar = gatherOffset;
        gatherBytesVar = gatherBytes;
        ccu::LocalAddr localDestination;
        localDestination.addr = outputAddr;
        localDestination.addr += gatherOffsetVar;
        localDestination.token = outputToken;
        ccu::RemoteAddr remoteSource;
        remoteSource.addr = remoteOutputs[channelIndex];
        remoteSource.addr += gatherOffsetVar;
        remoteSource.token = remoteOutputTokens[channelIndex];
        const uint16_t eventBit = static_cast<uint16_t>(1U << position);
        CCU_CHK_RET(ccu::Read(arg.channels[channelIndex], localDestination, remoteSource,
            gatherBytesVar, dataEvent, eventBit));
        gatherMask = static_cast<uint16_t>(gatherMask | eventBit);
    }
    CCU_CHK_RET(ccu::EventWait(dataEvent, gatherMask));
    return CCU_SUCCESS;
}

CcuResult V20LocalCopyRange(const ccu::Variable &inputAddr, const ccu::Variable &inputToken,
    const ccu::Variable &outputAddr, const ccu::Variable &outputToken,
    uint64_t rangeOffset, uint64_t rangeBytes, ccu::Event &dataEvent)
{
    for (uint64_t chunkOffset = 0; chunkOffset < rangeBytes; chunkOffset += MAX_DATA_SIZE) {
        const uint64_t chunkBytes = rangeBytes - chunkOffset > MAX_DATA_SIZE ?
            MAX_DATA_SIZE : rangeBytes - chunkOffset;
        ccu::Variable offset;
        ccu::Variable bytes;
        offset = rangeOffset + chunkOffset;
        bytes = chunkBytes;

        ccu::LocalAddr localSource;
        localSource.addr = inputAddr;
        localSource.addr += offset;
        localSource.token = inputToken;
        ccu::LocalAddr localDestination;
        localDestination.addr = outputAddr;
        localDestination.addr += offset;
        localDestination.token = outputToken;

        CCU_CHK_RET(ccu::LocalCopy(localDestination, localSource, bytes, dataEvent));
        CCU_CHK_RET(ccu::EventWait(dataEvent));
    }
    return CCU_SUCCESS;
}

CcuResult AllChannelBarrier(const AllReduceKernelArg &arg, uint16_t notifyBit)
{
    // Rank ordering makes the reverse notification an acknowledgement, so the
    // same notification bits can be safely reused by consecutive barriers.
    for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
        const uint32_t peerRank = arg.remoteRanks[channelIndex];
        if (peerRank >= arg.rankSize || peerRank == arg.rankId) {
            return CCU_E_PARA;
        }
        if (arg.rankId < peerRank) {
            CCU_CHK_RET(ccu::NotifyRecord(
                arg.channels[channelIndex], SYNC_NOTIFY_INDEX, notifyBit));
        }
    }
    for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
        if (arg.rankId > arg.remoteRanks[channelIndex]) {
            CCU_CHK_RET(ccu::NotifyWait(
                arg.channels[channelIndex], SYNC_NOTIFY_INDEX, notifyBit));
        }
    }
    for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
        if (arg.rankId > arg.remoteRanks[channelIndex]) {
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
                FLAT_OWNER_ACK_NOTIFY_BIT));
        }
    }
    for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
        if (arg.rankId < arg.remoteRanks[channelIndex]) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
                FLAT_OWNER_ACK_NOTIFY_BIT));
        }
    }
    return CCU_SUCCESS;
}

void GetFlatOwnerPipelineSlice(const AllReduceKernelArg &arg, uint32_t ownerRank,
    uint32_t sliceIndex, uint64_t &sliceOffset, uint64_t &sliceBytes)
{
    const uint64_t ownerElements = arg.segmentBytes[ownerRank] / sizeof(float);
    const uint64_t baseElements = ownerElements / arg.pipelineDepth;
    const uint64_t remainder = ownerElements % arg.pipelineDepth;
    const uint64_t prefixExtra = sliceIndex < remainder ? sliceIndex : remainder;
    const uint64_t sliceElements = baseElements + (sliceIndex < remainder ? 1ULL : 0ULL);
    sliceOffset = arg.segmentOffsets[ownerRank] +
        (baseElements * sliceIndex + prefixExtra) * sizeof(float);
    sliceBytes = sliceElements * sizeof(float);
}

CcuResult CcuFlatOwnerSlicedPhaseKernel(AllReduceKernelArg &arg, bool initializeOwner,
    bool reducePeers, bool gatherPeers, bool barrierBeforeGather, uint32_t sliceTaskArgIndex)
{
    if ((arg.rankSize != 12U && arg.rankSize != 16U) || arg.pipelineDepth < 2U ||
        arg.pipelineDepth > MAX_PIPELINE_DEPTH || arg.channelCount == 0U ||
        arg.channelCount >= 16U || arg.channelCount >= arg.rankSize) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> remoteInputs(arg.channelCount);
    std::vector<ccu::Variable> remoteInputTokens(arg.channelCount);
    std::vector<ccu::Variable> remoteOutputs(arg.channelCount);
    std::vector<ccu::Variable> remoteOutputTokens(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        remoteInputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_ADDR_XN_ID);
        remoteInputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_TOKEN_XN_ID);
        remoteOutputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN_ID);
        remoteOutputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN_ID);
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable unusedScratchAddr;
    ccu::Variable unusedScratchToken;
    ccu::Variable sliceSelector;
    uint32_t taskArgIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(sliceSelector, sliceTaskArgIndex));
    if (initializeOwner || reducePeers) {
        CCU_IF(sliceSelector == 0U) {
            CCU_CHK_RET(ExchangeHierarchicalResources(arg,
                inputAddr, inputToken, outputAddr, outputToken));
        }
    }

    uint64_t ownerSliceOffsets[MAX_PIPELINE_DEPTH] {};
    uint64_t ownerSliceBytes[MAX_PIPELINE_DEPTH] {};
    for (uint32_t slice = 0; slice < arg.pipelineDepth; ++slice) {
        GetFlatOwnerPipelineSlice(arg, arg.rankId, slice,
            ownerSliceOffsets[slice], ownerSliceBytes[slice]);
        if (ownerSliceBytes[slice] == 0U || ownerSliceBytes[slice] > MAX_DATA_SIZE ||
            ownerSliceBytes[slice] % sizeof(float) != 0U) {
            return CCU_E_PARA;
        }
    }

    ccu::Variable ownerOffset;
    ccu::Variable ownerBytes;
    ownerOffset = ownerSliceOffsets[0];
    ownerBytes = ownerSliceBytes[0];
    for (uint32_t slice = 1; slice < arg.pipelineDepth; ++slice) {
        CCU_IF(sliceSelector == slice) {
            ownerOffset = ownerSliceOffsets[slice];
            ownerBytes = ownerSliceBytes[slice];
        }
    }

    ccu::Event dataEvent;
    if (initializeOwner) {
        ccu::LocalAddr localSource;
        localSource.addr = inputAddr;
        localSource.addr += ownerOffset;
        localSource.token = inputToken;
        ccu::LocalAddr localDestination;
        localDestination.addr = outputAddr;
        localDestination.addr += ownerOffset;
        localDestination.token = outputToken;
        CCU_CHK_RET(ccu::LocalCopy(localDestination, localSource, ownerBytes, dataEvent));
        CCU_CHK_RET(ccu::EventWait(dataEvent));
    }

    if (reducePeers) {
        const uint32_t laneCount = arg.channelCount;
        const uint16_t allLaneMask = static_cast<uint16_t>((1U << laneCount) - 1U);
        uint64_t stripeOffsets[MAX_PIPELINE_DEPTH][MAX_ALG_CHANNEL_NUM] {};
        uint64_t stripeBytes[MAX_PIPELINE_DEPTH][MAX_ALG_CHANNEL_NUM] {};
        for (uint32_t slice = 0; slice < arg.pipelineDepth; ++slice) {
            const uint64_t sliceElements = ownerSliceBytes[slice] / sizeof(float);
            const uint64_t stripeBaseElements = sliceElements / laneCount;
            const uint32_t stripeRemainder = static_cast<uint32_t>(sliceElements % laneCount);
            uint64_t stripeOffset = ownerSliceOffsets[slice];
            for (uint32_t stripe = 0; stripe < laneCount; ++stripe) {
                stripeOffsets[slice][stripe] = stripeOffset;
                stripeBytes[slice][stripe] = (stripeBaseElements +
                    (stripe < stripeRemainder ? 1ULL : 0ULL)) * sizeof(float);
                if (stripeBytes[slice][stripe] == 0U || stripeBytes[slice][stripe] > MAX_DATA_SIZE) {
                    return CCU_E_PARA;
                }
                stripeOffset += stripeBytes[slice][stripe];
            }
        }

        std::vector<ccu::Variable> selectedStripeOffsets(laneCount);
        std::vector<ccu::Variable> selectedStripeBytes(laneCount);
        for (uint32_t stripe = 0; stripe < laneCount; ++stripe) {
            selectedStripeOffsets[stripe] = stripeOffsets[0][stripe];
            selectedStripeBytes[stripe] = stripeBytes[0][stripe];
        }
        for (uint32_t slice = 1; slice < arg.pipelineDepth; ++slice) {
            CCU_IF(sliceSelector == slice) {
                for (uint32_t stripe = 0; stripe < laneCount; ++stripe) {
                    selectedStripeOffsets[stripe] = stripeOffsets[slice][stripe];
                    selectedStripeBytes[stripe] = stripeBytes[slice][stripe];
                }
            }
        }

        std::vector<ccu::Variable> stripeIndices(laneCount);
        std::vector<ccu::Variable> currentBytes(laneCount);
        std::vector<ccu::LocalAddr> localDestinations(laneCount);
        std::vector<ccu::RemoteAddr> remoteSources(laneCount);
        for (uint32_t channelIndex = 0; channelIndex < laneCount; ++channelIndex) {
            stripeIndices[channelIndex] = channelIndex;
            currentBytes[channelIndex] = selectedStripeBytes[channelIndex];
            localDestinations[channelIndex].addr = outputAddr;
            localDestinations[channelIndex].addr += selectedStripeOffsets[channelIndex];
            localDestinations[channelIndex].token = outputToken;
            remoteSources[channelIndex].addr = remoteInputs[channelIndex];
            remoteSources[channelIndex].addr += selectedStripeOffsets[channelIndex];
            remoteSources[channelIndex].token = remoteInputTokens[channelIndex];
        }

        ccu::Variable round;
        ccu::Variable one;
        round = 0U;
        one = 1U;
        CCU_WHILE(round != laneCount) {
            for (uint32_t channelIndex = 0; channelIndex < laneCount; ++channelIndex) {
                const uint16_t eventBit = static_cast<uint16_t>(1U << channelIndex);
                CCU_CHK_RET(ccu::ReadReduce(arg.channels[channelIndex],
                    localDestinations[channelIndex], remoteSources[channelIndex], currentBytes[channelIndex],
                    HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, dataEvent, eventBit));
            }
            CCU_CHK_RET(ccu::EventWait(dataEvent, allLaneMask));

            for (uint32_t channelIndex = 0; channelIndex < laneCount; ++channelIndex) {
                localDestinations[channelIndex].addr += currentBytes[channelIndex];
                remoteSources[channelIndex].addr += currentBytes[channelIndex];
                stripeIndices[channelIndex] += one;
                CCU_IF(stripeIndices[channelIndex] == laneCount) {
                    stripeIndices[channelIndex] = 0U;
                    localDestinations[channelIndex].addr = outputAddr;
                    localDestinations[channelIndex].addr += ownerOffset;
                    remoteSources[channelIndex].addr = remoteInputs[channelIndex];
                    remoteSources[channelIndex].addr += ownerOffset;
                }
                for (uint32_t stripe = 0; stripe < laneCount; ++stripe) {
                    CCU_IF(stripeIndices[channelIndex] == stripe) {
                        currentBytes[channelIndex] = selectedStripeBytes[stripe];
                    }
                }
            }
            round += one;
        }
    }

    if (barrierBeforeGather) {
        CCU_CHK_RET(AllChannelBarrier(arg, FLAT_OWNER_READY_NOTIFY_BIT));
    }

    if (gatherPeers) {
        uint16_t gatherMask = 0;
        for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
            const uint32_t ownerRank = arg.remoteRanks[channelIndex];
            if (ownerRank >= arg.rankSize || ownerRank == arg.rankId) {
                return CCU_E_PARA;
            }
            uint64_t gatherOffsets[MAX_PIPELINE_DEPTH] {};
            uint64_t gatherSizes[MAX_PIPELINE_DEPTH] {};
            for (uint32_t slice = 0; slice < arg.pipelineDepth; ++slice) {
                GetFlatOwnerPipelineSlice(arg, ownerRank, slice,
                    gatherOffsets[slice], gatherSizes[slice]);
                if (gatherSizes[slice] == 0U || gatherSizes[slice] > MAX_DATA_SIZE) {
                    return CCU_E_PARA;
                }
            }
            ccu::Variable gatherOffset;
            ccu::Variable gatherBytes;
            gatherOffset = gatherOffsets[0];
            gatherBytes = gatherSizes[0];
            for (uint32_t slice = 1; slice < arg.pipelineDepth; ++slice) {
                CCU_IF(sliceSelector == slice) {
                    gatherOffset = gatherOffsets[slice];
                    gatherBytes = gatherSizes[slice];
                }
            }
            ccu::LocalAddr localDestination;
            localDestination.addr = outputAddr;
            localDestination.addr += gatherOffset;
            localDestination.token = outputToken;
            ccu::RemoteAddr remoteSource;
            remoteSource.addr = remoteOutputs[channelIndex];
            remoteSource.addr += gatherOffset;
            remoteSource.token = remoteOutputTokens[channelIndex];
            const uint16_t eventBit = static_cast<uint16_t>(1U << channelIndex);
            CCU_CHK_RET(ccu::Read(arg.channels[channelIndex], localDestination, remoteSource,
                gatherBytes, dataEvent, eventBit));
            gatherMask = static_cast<uint16_t>(gatherMask | eventBit);
        }
        CCU_CHK_RET(ccu::EventWait(dataEvent, gatherMask));
    }
    return CCU_SUCCESS;
}

CcuResult CcuFlatOwnerPhaseKernel(AllReduceKernelArg &arg, bool initializeOwner,
    bool reducePeers, bool gatherPeers, bool barrierBeforeGather)
{
    if ((arg.rankSize != 12U && arg.rankSize != 16U) || arg.channelCount == 0U ||
        arg.channelCount >= 16U || arg.channelCount >= arg.rankSize) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> remoteInputs(arg.channelCount);
    std::vector<ccu::Variable> remoteInputTokens(arg.channelCount);
    std::vector<ccu::Variable> remoteOutputs(arg.channelCount);
    std::vector<ccu::Variable> remoteOutputTokens(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        remoteInputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_ADDR_XN_ID);
        remoteInputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_TOKEN_XN_ID);
        remoteOutputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN_ID);
        remoteOutputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN_ID);
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable unusedScratchAddr;
    ccu::Variable unusedScratchToken;
    uint32_t taskArgIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchToken, taskArgIndex++));
    if (initializeOwner || reducePeers) {
        CCU_CHK_RET(ExchangeHierarchicalResources(
            arg, inputAddr, inputToken, outputAddr, outputToken));
    }

    const uint64_t ownerOffset = arg.segmentOffsets[arg.rankId];
    const uint64_t ownerBytes = arg.segmentBytes[arg.rankId];
    if (ownerBytes == 0 || ownerBytes % sizeof(float) != 0) {
        return CCU_E_PARA;
    }

    ccu::Event dataEvent;
    if (initializeOwner) {
        CCU_CHK_RET(V20LocalCopyRange(inputAddr, inputToken, outputAddr, outputToken,
            ownerOffset, ownerBytes, dataEvent));
    }

    if (reducePeers) {
        const uint32_t laneCount = arg.channelCount;
        const uint16_t allLaneMask = static_cast<uint16_t>((1U << laneCount) - 1U);
        const uint64_t ownerElements = ownerBytes / sizeof(float);
        const uint64_t stripeBaseElements = ownerElements / laneCount;
        const uint32_t stripeRemainder = static_cast<uint32_t>(ownerElements % laneCount);
        const uint64_t stripeBaseBytes = stripeBaseElements * sizeof(float);
        if (stripeBaseBytes == 0 || stripeBaseBytes + sizeof(float) > MAX_DATA_SIZE) {
            return CCU_E_PARA;
        }

        uint64_t stripeOffsets[MAX_ALG_CHANNEL_NUM] {};
        uint64_t stripeBytes[MAX_ALG_CHANNEL_NUM] {};
        uint64_t stripeOffset = ownerOffset;
        for (uint32_t stripe = 0; stripe < laneCount; ++stripe) {
            stripeOffsets[stripe] = stripeOffset;
            stripeBytes[stripe] = stripeBaseBytes +
                (stripe < stripeRemainder ? sizeof(float) : 0U);
            stripeOffset += stripeBytes[stripe];
        }

        std::vector<ccu::Variable> stripeIndices(laneCount);
        std::vector<ccu::Variable> currentBytes(laneCount);
        std::vector<ccu::LocalAddr> localDestinations(laneCount);
        std::vector<ccu::RemoteAddr> remoteSources(laneCount);
        for (uint32_t channelIndex = 0; channelIndex < laneCount; ++channelIndex) {
            ccu::Variable offsetVar;
            stripeIndices[channelIndex] = channelIndex;
            offsetVar = stripeOffsets[channelIndex];
            currentBytes[channelIndex] = stripeBytes[channelIndex];
            localDestinations[channelIndex].addr = outputAddr;
            localDestinations[channelIndex].addr += offsetVar;
            localDestinations[channelIndex].token = outputToken;
            remoteSources[channelIndex].addr = remoteInputs[channelIndex];
            remoteSources[channelIndex].addr += offsetVar;
            remoteSources[channelIndex].token = remoteInputTokens[channelIndex];
        }

        ccu::Variable round;
        ccu::Variable one;
        round = 0U;
        one = 1U;
        CCU_WHILE(round != laneCount) {
            for (uint32_t channelIndex = 0; channelIndex < laneCount; ++channelIndex) {
                const uint16_t eventBit = static_cast<uint16_t>(1U << channelIndex);
                CCU_CHK_RET(ccu::ReadReduce(arg.channels[channelIndex],
                    localDestinations[channelIndex], remoteSources[channelIndex], currentBytes[channelIndex],
                    HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, dataEvent, eventBit));
            }
            CCU_CHK_RET(ccu::EventWait(dataEvent, allLaneMask));

            for (uint32_t channelIndex = 0; channelIndex < laneCount; ++channelIndex) {
                localDestinations[channelIndex].addr += currentBytes[channelIndex];
                remoteSources[channelIndex].addr += currentBytes[channelIndex];
                stripeIndices[channelIndex] += one;
                CCU_IF(stripeIndices[channelIndex] == laneCount) {
                    ccu::Variable ownerOffsetVar;
                    ownerOffsetVar = ownerOffset;
                    stripeIndices[channelIndex] = 0U;
                    localDestinations[channelIndex].addr = outputAddr;
                    localDestinations[channelIndex].addr += ownerOffsetVar;
                    remoteSources[channelIndex].addr = remoteInputs[channelIndex];
                    remoteSources[channelIndex].addr += ownerOffsetVar;
                }
                if (stripeRemainder != 0U) {
                    CCU_IF(stripeIndices[channelIndex] == stripeRemainder) {
                        currentBytes[channelIndex] = stripeBaseBytes;
                    }
                    CCU_IF(stripeIndices[channelIndex] == 0U) {
                        currentBytes[channelIndex] = stripeBaseBytes + sizeof(float);
                    }
                }
            }
            round += one;
        }
    }

    if (barrierBeforeGather) {
        CCU_CHK_RET(AllChannelBarrier(arg, FLAT_OWNER_READY_NOTIFY_BIT));
    }

    if (gatherPeers) {
        uint16_t gatherMask = 0;
        for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
            const uint32_t ownerRank = arg.remoteRanks[channelIndex];
            if (ownerRank >= arg.rankSize || ownerRank == arg.rankId) {
                return CCU_E_PARA;
            }
            const uint64_t gatherOffset = arg.segmentOffsets[ownerRank];
            const uint64_t gatherBytes = arg.segmentBytes[ownerRank];
            if (gatherBytes == 0 || gatherBytes > MAX_DATA_SIZE) {
                return CCU_E_PARA;
            }
            ccu::Variable gatherOffsetVar;
            ccu::Variable gatherBytesVar;
            gatherOffsetVar = gatherOffset;
            gatherBytesVar = gatherBytes;
            ccu::LocalAddr localDestination;
            localDestination.addr = outputAddr;
            localDestination.addr += gatherOffsetVar;
            localDestination.token = outputToken;
            ccu::RemoteAddr remoteSource;
            remoteSource.addr = remoteOutputs[channelIndex];
            remoteSource.addr += gatherOffsetVar;
            remoteSource.token = remoteOutputTokens[channelIndex];
            const uint16_t eventBit = static_cast<uint16_t>(1U << channelIndex);
            CCU_CHK_RET(ccu::Read(arg.channels[channelIndex], localDestination, remoteSource,
                gatherBytesVar, dataEvent, eventBit));
            gatherMask = static_cast<uint16_t>(gatherMask | eventBit);
        }
        CCU_CHK_RET(ccu::EventWait(dataEvent, gatherMask));
    }
    return CCU_SUCCESS;
}

CcuResult CcuFullLaneFlatOwnerRsagKernel(AllReduceKernelArg &arg)
{
    if (arg.channelCount != arg.rankSize - 1U) {
        return CCU_E_PARA;
    }
    return CcuFlatOwnerPhaseKernel(arg, true, true, true, true);
}

CcuResult PeerBarrier(const AllReduceKernelArg &arg, uint32_t peerRank, uint16_t notifyBit)
{
    const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
    if (channelIndex == INVALID_CHANNEL_INDEX) {
        return CCU_E_PARA;
    }
    CCU_CHK_RET(ccu::NotifyRecord(arg.channels[channelIndex], SYNC_NOTIFY_INDEX, notifyBit));
    CCU_CHK_RET(ccu::NotifyWait(arg.channels[channelIndex], SYNC_NOTIFY_INDEX, notifyBit));
    return CCU_SUCCESS;
}

CcuResult ReadOutputRange(const AllReduceKernelArg &arg, uint32_t channelIndex,
    const std::vector<ccu::Variable> &remoteOutputs,
    const std::vector<ccu::Variable> &remoteOutputTokens,
    const ccu::Variable &outputAddr, const ccu::Variable &outputToken,
    uint64_t rangeOffset, uint64_t rangeBytes, ccu::Event &dataEvent)
{
    if (channelIndex == INVALID_CHANNEL_INDEX) {
        return CCU_E_PARA;
    }
    for (uint64_t chunkOffset = 0; chunkOffset < rangeBytes; chunkOffset += MAX_DATA_SIZE) {
        const uint64_t chunkBytes = rangeBytes - chunkOffset > MAX_DATA_SIZE ?
            MAX_DATA_SIZE : rangeBytes - chunkOffset;
        ccu::Variable offset;
        ccu::Variable bytes;
        offset = rangeOffset + chunkOffset;
        bytes = chunkBytes;
        ccu::LocalAddr localDestination;
        localDestination.addr = outputAddr;
        localDestination.addr += offset;
        localDestination.token = outputToken;
        ccu::RemoteAddr remoteSource;
        remoteSource.addr = remoteOutputs[channelIndex];
        remoteSource.addr += offset;
        remoteSource.token = remoteOutputTokens[channelIndex];
        CCU_CHK_RET(ccu::Read(arg.channels[channelIndex], localDestination, remoteSource,
            bytes, dataEvent));
        CCU_CHK_RET(ccu::EventWait(dataEvent));
    }
    return CCU_SUCCESS;
}

CcuResult CcuTopologyRadix4x3Kernel(AllReduceKernelArg &arg)
{
    if (arg.rankSize != 12U) {
        return CCU_E_PARA;
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable unusedScratchAddr;
    ccu::Variable unusedScratchToken;
    uint32_t taskArgIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchToken, taskArgIndex++));

    const uint32_t lowPosition = arg.rankId % HIGH_RADIX;
    const uint32_t highPosition = arg.rankId / HIGH_RADIX;
    const uint32_t highGroupCount = arg.rankSize / HIGH_RADIX;
    const uint32_t lowGroupBase = highPosition * HIGH_RADIX;
    ccu::Event dataEvent;

    if (arg.topologyPhase == TopologyPhase::RADIX_LOCAL_REDUCE) {
        std::vector<ccu::Variable> remoteInputs(arg.channelCount);
        std::vector<ccu::Variable> remoteInputTokens(arg.channelCount);
        std::vector<ccu::Variable> remoteOutputs(arg.channelCount);
        std::vector<ccu::Variable> remoteOutputTokens(arg.channelCount);
        for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
            remoteInputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_ADDR_XN_ID);
            remoteInputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_TOKEN_XN_ID);
            remoteOutputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN_ID);
            remoteOutputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN_ID);
        }
        CCU_CHK_RET(ExchangeHierarchicalResources(arg, inputAddr, inputToken, outputAddr, outputToken));

        const uint32_t blockStart = lowPosition * highGroupCount;
        const uint64_t blockOffset = arg.segmentOffsets[blockStart];
        const uint64_t blockBytes = GetRangeBytes(arg, blockStart, highGroupCount, arg.rankSize);
        if (blockBytes == 0) {
            return CCU_E_PARA;
        }
        CCU_CHK_RET(V20LocalCopyRange(inputAddr, inputToken, outputAddr, outputToken,
            blockOffset, blockBytes, dataEvent));

        uint64_t stripeOffsets[3] = {0, 0, 0};
        uint64_t stripeBytes[3] = {0, 0, 0};
        const uint64_t blockElements = blockBytes / sizeof(float);
        const uint64_t stripeBase = blockElements / 3U;
        const uint64_t stripeRemainder = blockElements % 3U;
        uint64_t stripeElementOffset = blockOffset / sizeof(float);
        for (uint32_t stripe = 0; stripe < 3U; ++stripe) {
            const uint64_t stripeElements = stripeBase + (stripe < stripeRemainder ? 1ULL : 0ULL);
            stripeOffsets[stripe] = stripeElementOffset * sizeof(float);
            stripeBytes[stripe] = stripeElements * sizeof(float);
            if (stripeBytes[stripe] == 0 || stripeBytes[stripe] > MAX_DATA_SIZE) {
                return CCU_E_PARA;
            }
            stripeElementOffset += stripeElements;
        }

        for (uint32_t round = 0; round < 3U; ++round) {
            uint16_t readMask = 0;
            uint32_t peerOrdinal = 0;
            for (uint32_t position = 0; position < HIGH_RADIX; ++position) {
                if (position == lowPosition) {
                    continue;
                }
                const uint32_t peerRank = lowGroupBase + position;
                const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
                if (channelIndex == INVALID_CHANNEL_INDEX) {
                    return CCU_E_PARA;
                }
                const uint32_t stripe = (peerOrdinal + round) % 3U;
                ccu::Variable stripeOffsetVar;
                ccu::Variable stripeBytesVar;
                stripeOffsetVar = stripeOffsets[stripe];
                stripeBytesVar = stripeBytes[stripe];
                ccu::LocalAddr localDestination;
                localDestination.addr = outputAddr;
                localDestination.addr += stripeOffsetVar;
                localDestination.token = outputToken;
                ccu::RemoteAddr remoteSource;
                remoteSource.addr = remoteInputs[channelIndex];
                remoteSource.addr += stripeOffsetVar;
                remoteSource.token = remoteInputTokens[channelIndex];
                const uint16_t eventBit = static_cast<uint16_t>(1U << peerOrdinal);
                CCU_CHK_RET(ccu::ReadReduce(arg.channels[channelIndex], localDestination, remoteSource,
                    stripeBytesVar, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, dataEvent, eventBit));
                readMask = static_cast<uint16_t>(readMask | eventBit);
                ++peerOrdinal;
            }
            CCU_CHK_RET(ccu::EventWait(dataEvent, readMask));
        }

        if (arg.rankId < 8U) {
            const uint32_t pairRank = arg.rankId < HIGH_RADIX ?
                arg.rankId + HIGH_RADIX : arg.rankId - HIGH_RADIX;
            const uint32_t channelIndex = FindChannelIndex(arg, pairRank);
            if (channelIndex == INVALID_CHANNEL_INDEX || blockBytes > MAX_DATA_SIZE) {
                return CCU_E_PARA;
            }
            CCU_CHK_RET(PeerBarrier(arg, pairRank, HIERARCHICAL_RS_STAGE2_READY_NOTIFY_BIT));
            if (arg.rankId < HIGH_RADIX) {
                ccu::Variable blockOffsetVar;
                ccu::Variable blockBytesVar;
                blockOffsetVar = blockOffset;
                blockBytesVar = blockBytes;
                ccu::LocalAddr localDestination;
                localDestination.addr = outputAddr;
                localDestination.addr += blockOffsetVar;
                localDestination.token = outputToken;
                ccu::RemoteAddr remoteSource;
                remoteSource.addr = remoteOutputs[channelIndex];
                remoteSource.addr += blockOffsetVar;
                remoteSource.token = remoteOutputTokens[channelIndex];
                CCU_CHK_RET(ccu::ReadReduce(arg.channels[channelIndex], localDestination, remoteSource,
                    blockBytesVar, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, dataEvent));
                CCU_CHK_RET(ccu::EventWait(dataEvent));
            }
            CCU_CHK_RET(PeerBarrier(arg, pairRank, TOPOLOGY_STAR_RELEASE_NOTIFY_BIT));
        }
        return CCU_SUCCESS;
    }

    std::vector<ccu::Variable> remoteOutputs(arg.channelCount);
    std::vector<ccu::Variable> remoteOutputTokens(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        remoteOutputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN_ID);
        remoteOutputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN_ID);
    }
    CCU_CHK_RET(ExchangeOutputResources(arg, outputAddr, outputToken));

    if (arg.topologyPhase == TopologyPhase::RADIX_CROSS_REDUCE_GATHER) {
        if (arg.rankId >= HIGH_RADIX && arg.rankId < 2U * HIGH_RADIX) {
            return CCU_SUCCESS;
        }
        const uint32_t pairRank = arg.rankId < HIGH_RADIX ?
            arg.rankId + 2U * HIGH_RADIX : arg.rankId - 2U * HIGH_RADIX;
        const uint32_t channelIndex = FindChannelIndex(arg, pairRank);
        const uint32_t blockStart = lowPosition * highGroupCount;
        const uint64_t blockOffset = arg.segmentOffsets[blockStart];
        const uint64_t blockBytes = GetRangeBytes(arg, blockStart, highGroupCount, arg.rankSize);
        if (channelIndex == INVALID_CHANNEL_INDEX || blockBytes == 0) {
            return CCU_E_PARA;
        }
        CCU_CHK_RET(PeerBarrier(arg, pairRank, TOPOLOGY_STAR_READY_NOTIFY_BIT));
        if (arg.rankId >= 2U * HIGH_RADIX) {
            CCU_CHK_RET(V18PushRange(arg, channelIndex, remoteOutputs, remoteOutputTokens,
                outputAddr, outputToken, blockOffset, blockBytes, true, dataEvent));
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
                TOPOLOGY_STAR_RELEASE_NOTIFY_BIT));
            CCU_CHK_RET(ReadOutputRange(arg, channelIndex, remoteOutputs, remoteOutputTokens,
                outputAddr, outputToken, blockOffset, blockBytes, dataEvent));
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
                HIERARCHICAL_RS_STAGE2_READY_NOTIFY_BIT));
        } else {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
                TOPOLOGY_STAR_RELEASE_NOTIFY_BIT));
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
                HIERARCHICAL_RS_STAGE2_READY_NOTIFY_BIT));
        }
        return CCU_SUCCESS;
    }

    if (arg.topologyPhase == TopologyPhase::RADIX_LOCAL_GATHER) {
        if (arg.rankId >= HIGH_RADIX && arg.rankId < 2U * HIGH_RADIX) {
            const uint32_t ownerRank = arg.rankId - HIGH_RADIX;
            const uint32_t channelIndex = FindChannelIndex(arg, ownerRank);
            if (channelIndex == INVALID_CHANNEL_INDEX) {
                return CCU_E_PARA;
            }
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
                HIERARCHICAL_AG_STAGE1_READY_NOTIFY_BIT));
            CCU_CHK_RET(ReadOutputRange(arg, channelIndex, remoteOutputs, remoteOutputTokens,
                outputAddr, outputToken, 0, arg.dataBytes, dataEvent));
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
                HIERARCHICAL_AG_STAGE2_READY_NOTIFY_BIT));
            return CCU_SUCCESS;
        }

        const uint32_t ownerGroupBase = arg.rankId < HIGH_RADIX ? 0U : 2U * HIGH_RADIX;
        CCU_CHK_RET(HierarchicalBarrier(arg, ownerGroupBase, 1, HIGH_RADIX,
            HIERARCHICAL_AG_STAGE1_READY_NOTIFY_BIT));
        uint16_t gatherMask = 0;
        for (uint32_t position = 0; position < HIGH_RADIX; ++position) {
            if (position == lowPosition) {
                continue;
            }
            const uint32_t peerRank = ownerGroupBase + position;
            const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
            if (channelIndex == INVALID_CHANNEL_INDEX) {
                return CCU_E_PARA;
            }
            const uint32_t gatherBlockStart = position * highGroupCount;
            const uint64_t gatherBlockOffset = arg.segmentOffsets[gatherBlockStart];
            const uint64_t gatherBlockBytes = GetRangeBytes(arg, gatherBlockStart, highGroupCount, arg.rankSize);
            if (gatherBlockBytes == 0 || gatherBlockBytes > MAX_DATA_SIZE) {
                return CCU_E_PARA;
            }

            ccu::Variable gatherOffsetVar;
            ccu::Variable gatherBytesVar;
            gatherOffsetVar = gatherBlockOffset;
            gatherBytesVar = gatherBlockBytes;
            ccu::LocalAddr localDestination;
            localDestination.addr = outputAddr;
            localDestination.addr += gatherOffsetVar;
            localDestination.token = outputToken;
            ccu::RemoteAddr remoteSource;
            remoteSource.addr = remoteOutputs[channelIndex];
            remoteSource.addr += gatherOffsetVar;
            remoteSource.token = remoteOutputTokens[channelIndex];
            const uint16_t eventBit = static_cast<uint16_t>(1U << position);
            CCU_CHK_RET(ccu::Read(arg.channels[channelIndex], localDestination, remoteSource,
                gatherBytesVar, dataEvent, eventBit));
            gatherMask = static_cast<uint16_t>(gatherMask | eventBit);
        }
        CCU_CHK_RET(ccu::EventWait(dataEvent, gatherMask));

        if (arg.rankId < HIGH_RADIX) {
            const uint32_t followerRank = arg.rankId + HIGH_RADIX;
            const uint32_t channelIndex = FindChannelIndex(arg, followerRank);
            if (channelIndex == INVALID_CHANNEL_INDEX) {
                return CCU_E_PARA;
            }
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
                HIERARCHICAL_AG_STAGE1_READY_NOTIFY_BIT));
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[channelIndex], SYNC_NOTIFY_INDEX,
                HIERARCHICAL_AG_STAGE2_READY_NOTIFY_BIT));
        }
        return CCU_SUCCESS;
    }
    return CCU_E_PARA;
}

CcuResult CcuRadix4x3RsagKernel(AllReduceKernelArg &arg)
{
    if (arg.rankSize != 12U) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> remoteInputs(arg.channelCount);
    std::vector<ccu::Variable> remoteInputTokens(arg.channelCount);
    std::vector<ccu::Variable> remoteOutputs(arg.channelCount);
    std::vector<ccu::Variable> remoteOutputTokens(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        remoteInputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_ADDR_XN_ID);
        remoteInputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_TOKEN_XN_ID);
        remoteOutputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN_ID);
        remoteOutputTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN_ID);
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable unusedScratchAddr;
    ccu::Variable unusedScratchToken;
    uint32_t taskArgIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchToken, taskArgIndex++));
    CCU_CHK_RET(ExchangeHierarchicalResources(arg, inputAddr, inputToken, outputAddr, outputToken));

    const uint32_t lowPosition = arg.rankId % HIGH_RADIX;
    const uint32_t highPosition = arg.rankId / HIGH_RADIX;
    const uint32_t highGroupCount = arg.rankSize / HIGH_RADIX;
    const uint32_t lowGroupBase = highPosition * HIGH_RADIX;
    const uint32_t highGroupBase = lowPosition;
    ccu::Event dataEvent;

    // Stage 1: each rank owns one quarter-block and pulls three striped reductions
    // from immutable peer inputs. Every round uses all three CLOS peers concurrently.
    const uint32_t blockStart = lowPosition * highGroupCount;
    const uint64_t blockOffset = arg.segmentOffsets[blockStart];
    const uint64_t blockBytes = GetRangeBytes(arg, blockStart, highGroupCount, arg.rankSize);
    if (blockBytes == 0) {
        return CCU_E_PARA;
    }
    CCU_CHK_RET(V20LocalCopyRange(inputAddr, inputToken, outputAddr, outputToken,
        blockOffset, blockBytes, dataEvent));

    uint64_t blockStripeOffsets[3] = {0, 0, 0};
    uint64_t blockStripeBytes[3] = {0, 0, 0};
    const uint64_t blockElements = blockBytes / sizeof(float);
    const uint64_t blockStripeBase = blockElements / 3U;
    const uint64_t blockStripeRemainder = blockElements % 3U;
    uint64_t stripeElementOffset = blockOffset / sizeof(float);
    for (uint32_t stripe = 0; stripe < 3U; ++stripe) {
        const uint64_t stripeElements = blockStripeBase + (stripe < blockStripeRemainder ? 1ULL : 0ULL);
        blockStripeOffsets[stripe] = stripeElementOffset * sizeof(float);
        blockStripeBytes[stripe] = stripeElements * sizeof(float);
        if (blockStripeBytes[stripe] == 0 || blockStripeBytes[stripe] > MAX_DATA_SIZE) {
            return CCU_E_PARA;
        }
        stripeElementOffset += stripeElements;
    }

    for (uint32_t round = 0; round < 3U; ++round) {
        uint16_t readMask = 0;
        uint32_t peerOrdinal = 0;
        for (uint32_t position = 0; position < HIGH_RADIX; ++position) {
            if (position == lowPosition) {
                continue;
            }
            const uint32_t peerRank = lowGroupBase + position;
            const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
            if (channelIndex == INVALID_CHANNEL_INDEX) {
                return CCU_E_PARA;
            }
            const uint32_t stripe = (peerOrdinal + round) % 3U;
            ccu::Variable stripeOffsetVar;
            ccu::Variable stripeBytesVar;
            stripeOffsetVar = blockStripeOffsets[stripe];
            stripeBytesVar = blockStripeBytes[stripe];
            ccu::LocalAddr localDestination;
            localDestination.addr = outputAddr;
            localDestination.addr += stripeOffsetVar;
            localDestination.token = outputToken;
            ccu::RemoteAddr remoteSource;
            remoteSource.addr = remoteInputs[channelIndex];
            remoteSource.addr += stripeOffsetVar;
            remoteSource.token = remoteInputTokens[channelIndex];
            const uint16_t eventBit = static_cast<uint16_t>(1U << peerOrdinal);
            CCU_CHK_RET(ccu::ReadReduce(arg.channels[channelIndex], localDestination, remoteSource,
                stripeBytesVar, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, dataEvent, eventBit));
            readMask = static_cast<uint16_t>(readMask | eventBit);
            ++peerOrdinal;
        }
        CCU_CHK_RET(ccu::EventWait(dataEvent, readMask));
    }

    // Stage 2: the three equal low positions reduce one final segment. Each peer
    // modifies only its own segment, so the other two source segments stay immutable.
    CCU_CHK_RET(HierarchicalBarrier(arg, highGroupBase, HIGH_RADIX, highGroupCount,
        HIERARCHICAL_RS_STAGE2_READY_NOTIFY_BIT));
    const uint32_t ownedSegment = lowPosition * highGroupCount + highPosition;
    const uint64_t segmentOffset = arg.segmentOffsets[ownedSegment];
    const uint64_t segmentBytes = arg.segmentBytes[ownedSegment];
    if (segmentBytes == 0) {
        return CCU_E_PARA;
    }
    uint64_t segmentStripeOffsets[2] = {0, 0};
    uint64_t segmentStripeBytes[2] = {0, 0};
    const uint64_t segmentElements = segmentBytes / sizeof(float);
    const uint64_t segmentStripeBase = segmentElements / 2U;
    const uint64_t segmentStripeRemainder = segmentElements % 2U;
    stripeElementOffset = segmentOffset / sizeof(float);
    for (uint32_t stripe = 0; stripe < 2U; ++stripe) {
        const uint64_t stripeElements = segmentStripeBase + (stripe < segmentStripeRemainder ? 1ULL : 0ULL);
        segmentStripeOffsets[stripe] = stripeElementOffset * sizeof(float);
        segmentStripeBytes[stripe] = stripeElements * sizeof(float);
        if (segmentStripeBytes[stripe] == 0 || segmentStripeBytes[stripe] > MAX_DATA_SIZE) {
            return CCU_E_PARA;
        }
        stripeElementOffset += stripeElements;
    }

    for (uint32_t round = 0; round < 2U; ++round) {
        uint16_t readMask = 0;
        uint32_t peerOrdinal = 0;
        for (uint32_t position = 0; position < highGroupCount; ++position) {
            if (position == highPosition) {
                continue;
            }
            const uint32_t peerRank = position * HIGH_RADIX + lowPosition;
            const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
            if (channelIndex == INVALID_CHANNEL_INDEX) {
                return CCU_E_PARA;
            }
            const uint32_t stripe = (peerOrdinal + round) % 2U;
            ccu::Variable stripeOffsetVar;
            ccu::Variable stripeBytesVar;
            stripeOffsetVar = segmentStripeOffsets[stripe];
            stripeBytesVar = segmentStripeBytes[stripe];
            ccu::LocalAddr localDestination;
            localDestination.addr = outputAddr;
            localDestination.addr += stripeOffsetVar;
            localDestination.token = outputToken;
            ccu::RemoteAddr remoteSource;
            remoteSource.addr = remoteOutputs[channelIndex];
            remoteSource.addr += stripeOffsetVar;
            remoteSource.token = remoteOutputTokens[channelIndex];
            const uint16_t eventBit = static_cast<uint16_t>(1U << peerOrdinal);
            CCU_CHK_RET(ccu::ReadReduce(arg.channels[channelIndex], localDestination, remoteSource,
                stripeBytesVar, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, dataEvent, eventBit));
            readMask = static_cast<uint16_t>(readMask | eventBit);
            ++peerOrdinal;
        }
        CCU_CHK_RET(ccu::EventWait(dataEvent, readMask));
    }

    // First all-gather restores the three fully reduced segments in this block.
    CCU_CHK_RET(HierarchicalBarrier(arg, highGroupBase, HIGH_RADIX, highGroupCount,
        HIERARCHICAL_AG_STAGE1_READY_NOTIFY_BIT));
    uint16_t readMask = 0;
    for (uint32_t position = 0; position < highGroupCount; ++position) {
        if (position == highPosition) {
            continue;
        }
        const uint32_t peerRank = position * HIGH_RADIX + lowPosition;
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        if (channelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        const uint32_t gatherSegment = lowPosition * highGroupCount + position;
        const uint64_t gatherOffset = arg.segmentOffsets[gatherSegment];
        const uint64_t gatherBytes = arg.segmentBytes[gatherSegment];
        if (gatherBytes == 0 || gatherBytes > MAX_DATA_SIZE) {
            return CCU_E_PARA;
        }
        ccu::Variable gatherOffsetVar;
        ccu::Variable gatherBytesVar;
        gatherOffsetVar = gatherOffset;
        gatherBytesVar = gatherBytes;
        ccu::LocalAddr localDestination;
        localDestination.addr = outputAddr;
        localDestination.addr += gatherOffsetVar;
        localDestination.token = outputToken;
        ccu::RemoteAddr remoteSource;
        remoteSource.addr = remoteOutputs[channelIndex];
        remoteSource.addr += gatherOffsetVar;
        remoteSource.token = remoteOutputTokens[channelIndex];
        const uint16_t eventBit = static_cast<uint16_t>(1U << position);
        CCU_CHK_RET(ccu::Read(arg.channels[channelIndex], localDestination, remoteSource,
            gatherBytesVar, dataEvent, eventBit));
        readMask = static_cast<uint16_t>(readMask | eventBit);
    }
    CCU_CHK_RET(ccu::EventWait(dataEvent, readMask));

    // Second all-gather restores the other three complete quarter-blocks.
    CCU_CHK_RET(HierarchicalBarrier(arg, lowGroupBase, 1, HIGH_RADIX,
        HIERARCHICAL_AG_STAGE2_READY_NOTIFY_BIT));
    readMask = 0;
    for (uint32_t position = 0; position < HIGH_RADIX; ++position) {
        if (position == lowPosition) {
            continue;
        }
        const uint32_t peerRank = lowGroupBase + position;
        const uint32_t channelIndex = FindChannelIndex(arg, peerRank);
        if (channelIndex == INVALID_CHANNEL_INDEX) {
            return CCU_E_PARA;
        }
        const uint32_t gatherBlockStart = position * highGroupCount;
        const uint64_t gatherBlockOffset = arg.segmentOffsets[gatherBlockStart];
        const uint64_t gatherBlockBytes = GetRangeBytes(arg, gatherBlockStart, highGroupCount, arg.rankSize);
        if (gatherBlockBytes == 0 || gatherBlockBytes > MAX_DATA_SIZE) {
            return CCU_E_PARA;
        }
        ccu::Variable blockOffsetVar;
        ccu::Variable blockBytesVar;
        blockOffsetVar = gatherBlockOffset;
        blockBytesVar = gatherBlockBytes;
        ccu::LocalAddr localDestination;
        localDestination.addr = outputAddr;
        localDestination.addr += blockOffsetVar;
        localDestination.token = outputToken;
        ccu::RemoteAddr remoteSource;
        remoteSource.addr = remoteOutputs[channelIndex];
        remoteSource.addr += blockOffsetVar;
        remoteSource.token = remoteOutputTokens[channelIndex];
        const uint16_t eventBit = static_cast<uint16_t>(1U << position);
        CCU_CHK_RET(ccu::Read(arg.channels[channelIndex], localDestination, remoteSource,
            blockBytesVar, dataEvent, eventBit));
        readMask = static_cast<uint16_t>(readMask | eventBit);
    }
    CCU_CHK_RET(ccu::EventWait(dataEvent, readMask));
    return CCU_SUCCESS;
}

CcuResult CcuRingRsagKernel(AllReduceKernelArg &arg)
{
    const uint32_t previousRank = (arg.rankId + arg.rankSize - 1U) % arg.rankSize;
    const uint32_t nextRank = (arg.rankId + 1U) % arg.rankSize;
    const uint32_t previousChannelIndex = FindChannelIndex(arg, previousRank);
    const uint32_t nextChannelIndex = FindChannelIndex(arg, nextRank);
    if (previousChannelIndex == INVALID_CHANNEL_INDEX || nextChannelIndex == INVALID_CHANNEL_INDEX) {
        return CCU_E_PARA;
    }
    const ChannelHandle previousChannel = arg.channels[previousChannelIndex];
    const ChannelHandle nextChannel = arg.channels[nextChannelIndex];

    std::vector<ccu::Variable> remoteOutputs(arg.channelCount);
    std::vector<ccu::Variable> remoteTokens(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        remoteOutputs[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN_ID);
        remoteTokens[idx] = ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN_ID);
    }

    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable unusedScratchAddr;
    ccu::Variable unusedScratchToken;
    uint32_t taskArgIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchAddr, taskArgIndex++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchToken, taskArgIndex++));

    ccu::Variable segmentOffset;
    ccu::Variable segmentBytes;
    ccu::LocalAddr localSource;
    ccu::LocalAddr localDestination;
    ccu::RemoteAddr remoteDestination;
    ccu::Event dataEvent;

    localSource.token = inputToken;
    localDestination.token = outputToken;
    for (uint64_t chunkOffset = 0; chunkOffset < arg.dataBytes; chunkOffset += MAX_DATA_SIZE) {
        const uint64_t chunkBytes = arg.dataBytes - chunkOffset > MAX_DATA_SIZE ?
            MAX_DATA_SIZE : arg.dataBytes - chunkOffset;
        segmentOffset = chunkOffset;
        segmentBytes = chunkBytes;
        localSource.addr = inputAddr;
        localSource.addr += segmentOffset;
        localDestination.addr = outputAddr;
        localDestination.addr += segmentOffset;
        CCU_CHK_RET(ccu::LocalCopy(localDestination, localSource, segmentBytes, dataEvent));
        CCU_CHK_RET(ccu::EventWait(dataEvent));
    }

    CCU_CHK_RET(ExchangeOutputResources(arg, outputAddr, outputToken));
    localSource.token = outputToken;
    remoteDestination.token = remoteTokens[nextChannelIndex];

    for (uint32_t step = 0; step + 1U < arg.rankSize; ++step) {
        if (step != 0) {
            CCU_CHK_RET(ccu::NotifyRecord(previousChannel, SYNC_NOTIFY_INDEX, RING_PRE_SYNC_NOTIFY_BIT));
            CCU_CHK_RET(ccu::NotifyWait(nextChannel, SYNC_NOTIFY_INDEX, RING_PRE_SYNC_NOTIFY_BIT));
        }

        const uint32_t segment = (arg.rankId + arg.rankSize - step) % arg.rankSize;
        if (arg.segmentBytes[segment] != 0) {
            segmentOffset = arg.segmentOffsets[segment];
            segmentBytes = arg.segmentBytes[segment];
            localSource.addr = outputAddr;
            localSource.addr += segmentOffset;
            remoteDestination.addr = remoteOutputs[nextChannelIndex];
            remoteDestination.addr += segmentOffset;
            CCU_CHK_RET(ccu::WriteReduce(nextChannel, remoteDestination, localSource, segmentBytes,
                HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, dataEvent));
            CCU_CHK_RET(ccu::EventWait(dataEvent));
        }

        CCU_CHK_RET(ccu::NotifyRecord(nextChannel, SYNC_NOTIFY_INDEX, RING_RS_DONE_NOTIFY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(previousChannel, SYNC_NOTIFY_INDEX, RING_RS_DONE_NOTIFY_BIT));
    }

    for (uint32_t step = 0; step + 1U < arg.rankSize; ++step) {
        // A receiver advertises readiness before its predecessor reuses the
        // same AG done bit. This makes every repeated Record follow the peer's
        // consumption of the previous round and prevents CKE-bit coalescing.
        CCU_CHK_RET(ccu::NotifyRecord(previousChannel, SYNC_NOTIFY_INDEX, RING_AG_READY_NOTIFY_BIT));
        CCU_CHK_RET(ccu::NotifyWait(nextChannel, SYNC_NOTIFY_INDEX, RING_AG_READY_NOTIFY_BIT));

        const uint32_t segment = (arg.rankId + 1U + arg.rankSize - step) % arg.rankSize;
        if (arg.segmentBytes[segment] != 0) {
            segmentOffset = arg.segmentOffsets[segment];
            segmentBytes = arg.segmentBytes[segment];
            localSource.addr = outputAddr;
            localSource.addr += segmentOffset;
            remoteDestination.addr = remoteOutputs[nextChannelIndex];
            remoteDestination.addr += segmentOffset;
            CCU_CHK_RET(ccu::Write(nextChannel, remoteDestination, localSource, segmentBytes, dataEvent));
            CCU_CHK_RET(ccu::EventWait(dataEvent));
        }

        // The final All-Gather write has no following round that depends on it.
        if (step + 2U < arg.rankSize) {
            CCU_CHK_RET(ccu::NotifyRecord(nextChannel, SYNC_NOTIFY_INDEX, RING_AG_DONE_NOTIFY_BIT));
            CCU_CHK_RET(ccu::NotifyWait(previousChannel, SYNC_NOTIFY_INDEX, RING_AG_DONE_NOTIFY_BIT));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuSharedLocalTopologyKernel(AllReduceKernelArg &arg)
{
    ccu::Variable phaseSelector;
    CCU_CHK_RET(ccu::LoadArg(phaseSelector, TOPOLOGY_PHASE_TASK_ARG_INDEX));

    if (arg.algorithm == AllReduceAlgorithm::CLOS_FULL_LANE_FLAT_OWNER_RSAG &&
        arg.topologyPhase == TopologyPhase::RADIX_LOCAL_SHARED) {
        if ((arg.rankSize == 12U || arg.rankSize == 16U) &&
            arg.pipelineDepth >= 2U && arg.pipelineDepth <= MAX_PIPELINE_DEPTH) {
            CCU_IF(phaseSelector == TOPOLOGY_LOCAL_REDUCE_SELECTOR) {
                CCU_CHK_RET(CcuFlatOwnerSlicedPhaseKernel(arg, true, true, false, false,
                    TOPOLOGY_LOCAL_SLICE_TASK_ARG_INDEX));
            }
            CCU_IF(phaseSelector == TOPOLOGY_LOCAL_GATHER_SELECTOR) {
                CCU_CHK_RET(CcuFlatOwnerSlicedPhaseKernel(arg, false, false, true, true,
                    TOPOLOGY_LOCAL_SLICE_TASK_ARG_INDEX));
            }
        } else {
            CCU_IF(phaseSelector == TOPOLOGY_LOCAL_REDUCE_SELECTOR) {
                CCU_CHK_RET(CcuFlatOwnerPhaseKernel(arg, true, true, false, false));
            }
            CCU_IF(phaseSelector == TOPOLOGY_LOCAL_GATHER_SELECTOR) {
                CCU_CHK_RET(CcuFlatOwnerPhaseKernel(arg, false, false, true, true));
            }
        }
        return CCU_SUCCESS;
    }

    if (arg.topologyPhase == TopologyPhase::RHD_LOCAL_SHARED) {
        AllReduceKernelArg reduceArg = arg;
        reduceArg.topologyPhase = TopologyPhase::RHD_LOCAL_REDUCE;
        CCU_IF(phaseSelector == TOPOLOGY_LOCAL_REDUCE_SELECTOR) {
            CCU_CHK_RET(CcuTopologyRecursiveHalvingDoublingKernel(reduceArg));
        }

        AllReduceKernelArg gatherArg = arg;
        gatherArg.topologyPhase = TopologyPhase::RHD_LOCAL_GATHER;
        CCU_IF(phaseSelector == TOPOLOGY_LOCAL_GATHER_SELECTOR) {
            CCU_CHK_RET(CcuTopologyRecursiveHalvingDoublingKernel(gatherArg));
        }
        return CCU_SUCCESS;
    }

    if (arg.topologyPhase == TopologyPhase::RADIX_LOCAL_SHARED) {
        AllReduceKernelArg reduceArg = arg;
        reduceArg.topologyPhase = TopologyPhase::RADIX_LOCAL_REDUCE;
        CCU_IF(phaseSelector == TOPOLOGY_LOCAL_REDUCE_SELECTOR) {
            CCU_CHK_RET(CcuTopologyRadix4x3Kernel(reduceArg));
        }

        AllReduceKernelArg gatherArg = arg;
        gatherArg.topologyPhase = TopologyPhase::RADIX_LOCAL_GATHER;
        CCU_IF(phaseSelector == TOPOLOGY_LOCAL_GATHER_SELECTOR) {
            CCU_CHK_RET(CcuTopologyRadix4x3Kernel(gatherArg));
        }
        return CCU_SUCCESS;
    }
    return CCU_E_PARA;
}

} // namespace

CcuResult CcuKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<AllReduceKernelArg *>(kernelArg);
    if (arg == nullptr || arg->rankSize < 2 || arg->rankSize > MAX_RANK_SIZE ||
        arg->rankId >= arg->rankSize || arg->channelCount == 0 ||
        arg->channelCount > MAX_ALG_CHANNEL_NUM || arg->dataBytes == 0) {
        return CCU_E_PARA;
    }

    if (arg->algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_DOUBLING) {
        return CcuRecursiveDoublingKernel(*arg);
    }
    if (arg->algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_HALVING_DOUBLING) {
        if (arg->topologyPhase == TopologyPhase::RHD_LOCAL_SHARED) {
            return CcuSharedLocalTopologyKernel(*arg);
        }
        if (arg->topologyPhase != TopologyPhase::FULL_CLOS) {
            return CcuTopologyRecursiveHalvingDoublingKernel(*arg);
        }
        if (arg->rankSize == 4U) {
            return CcuPipelineRecursiveHalvingDoublingKernel(*arg);
        }
        return CcuRecursiveHalvingDoublingKernel(*arg);
    }
    if (arg->algorithm == AllReduceAlgorithm::CLOS_RADIX4_RSAG) {
        return CcuRadix4RsagKernel(*arg);
    }
    if (arg->algorithm == AllReduceAlgorithm::CLOS_FULL_LANE_FLAT_OWNER_RSAG) {
        if (arg->topologyPhase == TopologyPhase::RADIX_LOCAL_SHARED) {
            return CcuSharedLocalTopologyKernel(*arg);
        }
        if (arg->topologyPhase == TopologyPhase::RADIX_CROSS_REDUCE_GATHER) {
            if ((arg->rankSize == 12U || arg->rankSize == 16U) &&
                arg->pipelineDepth >= 2U && arg->pipelineDepth <= MAX_PIPELINE_DEPTH) {
                return CcuFlatOwnerSlicedPhaseKernel(*arg, false, true, true, true,
                    TOPOLOGY_CROSS_SLICE_TASK_ARG_INDEX);
            }
            return CcuFlatOwnerPhaseKernel(*arg, false, true, true, true);
        }
        if (arg->topologyPhase == TopologyPhase::FULL_CLOS) {
            return CcuFullLaneFlatOwnerRsagKernel(*arg);
        }
        return CCU_E_PARA;
    }
    if (arg->algorithm == AllReduceAlgorithm::CLOS_RADIX4X3_RSAG) {
        if (arg->topologyPhase == TopologyPhase::RADIX_LOCAL_SHARED) {
            return CcuSharedLocalTopologyKernel(*arg);
        }
        if (arg->topologyPhase != TopologyPhase::FULL_CLOS) {
            return CcuTopologyRadix4x3Kernel(*arg);
        }
        return CcuRadix4x3RsagKernel(*arg);
    }
    if (arg->algorithm == AllReduceAlgorithm::CLOS_RING_RSAG) {
        return CcuRingRsagKernel(*arg);
    }
    return CCU_E_PARA;
}
} // namespace ops_hccl
