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
#include <vector>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {
constexpr uint32_t EXPECTED_RANK_SIZE = 16;
constexpr uint32_t EXPECTED_LOCAL_RANK_SIZE = 8;
constexpr uint32_t EXPECTED_CHANNEL_NUM = EXPECTED_RANK_SIZE - 1;
constexpr uint32_t EXPECTED_THREAD_NUM = EXPECTED_RANK_SIZE;
constexpr uint32_t EXPECTED_WORKER_NUM = EXPECTED_THREAD_NUM - 1;
constexpr uint32_t MIN_CHANNEL_NOTIFY_NUM = 4;
constexpr uint32_t PATH_A_READY_NOTIFY = 0;
constexpr uint32_t PATH_A_RESULT_NOTIFY = 1;
constexpr uint32_t FULL_MESH_RS_DONE_NOTIFY = 1;
constexpr uint32_t FULL_MESH_AG_DONE_NOTIFY = 2;
constexpr uint32_t FULL_MESH_REDUCE_READY_NOTIFY = 3;
constexpr uint32_t WORKER_RS_START_NOTIFY = 0;
constexpr uint32_t WORKER_AG_START_NOTIFY = 1;
constexpr uint32_t WORKER_REDUCE_DEPENDENCY_NOTIFY_BASE = 2;
constexpr uint32_t FULL_MESH_REDUCE_STAGE_NUM = 4;
static_assert(WORKER_REDUCE_DEPENDENCY_NOTIFY_BASE + FULL_MESH_REDUCE_STAGE_NUM
    <= HCCL_CUSTOM_THREAD_NOTIFY_NUM);
constexpr uint32_t MAIN_PHASE_DONE_NOTIFY = 0;
constexpr uint32_t DUAL_TREE_LANE_START_NOTIFY = 0;
constexpr uint32_t DUAL_TREE_LANE_DONE_NOTIFY = 1;
constexpr uint32_t DUAL_TREE_LANE0_READY_NOTIFY = 0;
constexpr uint32_t DUAL_TREE_LANE0_RESULT_NOTIFY = 1;
constexpr uint32_t DUAL_TREE_LANE1_READY_NOTIFY = 2;
constexpr uint32_t DUAL_TREE_LANE1_RESULT_NOTIFY = 3;
constexpr uint64_t FP32_BYTES = 4;
constexpr uint64_t DIRECT_MAX_BYTES = 256;
constexpr uint64_t DUAL_ROOT_TREE_BYTES = 512ULL * 1024ULL;
constexpr uint64_t RECURSIVE_DOUBLING_MAX_BYTES = 1ULL * 1024ULL * 1024ULL;
constexpr uint64_t TREE_MAX_BYTES = 4ULL * 1024ULL * 1024ULL;
constexpr uint64_t RECURSIVE_DOUBLING_REGION_NUM = 5;

enum class AlgorithmType {
    DIRECT,
    TREE,
    DUAL_ROOT_TREE,
    RECURSIVE_DOUBLING,
    FULL_MESH_RSAG,
};

struct TopologyLayout {
    std::array<uint32_t, EXPECTED_LOCAL_RANK_SIZE> group0{};
    std::array<uint32_t, EXPECTED_LOCAL_RANK_SIZE> group1{};
    uint32_t localIndex = EXPECTED_LOCAL_RANK_SIZE;
    uint32_t serverIndex = 2;
};

using ChannelTable = std::array<const ChannelInfo *, EXPECTED_RANK_SIZE>;

struct DataRange {
    uint64_t offset = 0;
    uint64_t count = 0;
};

AlgorithmType SelectAlgorithm(uint64_t totalBytes)
{
    if (totalBytes <= DIRECT_MAX_BYTES) {
        return AlgorithmType::DIRECT;
    }
    if (totalBytes == DUAL_ROOT_TREE_BYTES) {
        return AlgorithmType::DUAL_ROOT_TREE;
    }
    if (totalBytes <= RECURSIVE_DOUBLING_MAX_BYTES) {
        return AlgorithmType::RECURSIVE_DOUBLING;
    }
    if (totalBytes <= TREE_MAX_BYTES) {
        return AlgorithmType::TREE;
    }
    return AlgorithmType::FULL_MESH_RSAG;
}

HcclResult ValidateExecOp(
    const OpParam &param, const AlgResourceCtx &resCtx, ChannelTable &channelTable, uint32_t &myRankIndex)
{
    if (param.inputPtr == nullptr || param.outputPtr == nullptr || param.count == 0
        || param.count > std::numeric_limits<uint64_t>::max() / FP32_BYTES) {
        HCCL_ERROR("AllReduce parameters are invalid");
        return HCCL_E_PARA;
    }
    if (param.rankSize != EXPECTED_RANK_SIZE) {
        HCCL_ERROR("AllReduce requires exactly 16 ranks, got [%u]", param.rankSize);
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM) {
        HCCL_ERROR("Only FP32 SUM AllReduce is supported");
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.myRank >= EXPECTED_RANK_SIZE || resCtx.allRanks.size() != EXPECTED_RANK_SIZE
        || resCtx.localRanks.size() != EXPECTED_LOCAL_RANK_SIZE || resCtx.rootRank != 0) {
        HCCL_ERROR("AllReduce topology metadata is invalid for rank [%u]", param.myRank);
        return HCCL_E_INTERNAL;
    }
    const uint32_t localRankBase = (param.myRank / EXPECTED_LOCAL_RANK_SIZE) * EXPECTED_LOCAL_RANK_SIZE;
    for (uint32_t rankIndex = 0; rankIndex < EXPECTED_RANK_SIZE; ++rankIndex) {
        if (resCtx.allRanks[rankIndex] != rankIndex) {
            HCCL_ERROR("AllReduce requires canonical rank indices");
            return HCCL_E_INTERNAL;
        }
    }
    for (uint32_t localIndex = 0; localIndex < EXPECTED_LOCAL_RANK_SIZE; ++localIndex) {
        if (resCtx.localRanks[localIndex] != localRankBase + localIndex) {
            HCCL_ERROR("AllReduce requires contiguous local rank groups");
            return HCCL_E_INTERNAL;
        }
    }
    if (resCtx.threads.size() != EXPECTED_THREAD_NUM || resCtx.aicpuThread != resCtx.threads[0]
        || resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0) {
        HCCL_ERROR("AllReduce local resources are invalid");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.channels.size() != EXPECTED_CHANNEL_NUM) {
        HCCL_ERROR("Expected 15 channels, got [%zu]", resCtx.channels.size());
        return HCCL_E_INTERNAL;
    }

    channelTable.fill(nullptr);
    myRankIndex = param.myRank;
    uint32_t remoteRankMask = 0;
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank >= EXPECTED_RANK_SIZE || channel.remoteRank == param.myRank
            || channel.notifyNum < MIN_CHANNEL_NOTIFY_NUM || channel.handle == 0 || channel.remoteCclMem.addr == nullptr
            || channel.remoteCclMem.size == 0) {
            HCCL_ERROR("Channel metadata for remote rank [%u] is invalid", channel.remoteRank);
            return HCCL_E_INTERNAL;
        }
        const uint32_t remoteRankBit = 1U << channel.remoteRank;
        if ((remoteRankMask & remoteRankBit) != 0) {
            HCCL_ERROR("Channel metadata contains duplicate remote ranks");
            return HCCL_E_INTERNAL;
        }
        remoteRankMask |= remoteRankBit;
        channelTable[channel.remoteRank] = &channel;
    }
    const uint32_t expectedRemoteRankMask = ((1U << EXPECTED_RANK_SIZE) - 1U) & ~(1U << myRankIndex);
    if (remoteRankMask != expectedRemoteRankMask) {
        HCCL_ERROR("Channel metadata does not cover every remote rank");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult BuildTopologyLayout(const OpParam &param, TopologyLayout &layout)
{
    for (uint32_t localIndex = 0; localIndex < EXPECTED_LOCAL_RANK_SIZE; ++localIndex) {
        layout.group0[localIndex] = localIndex;
        layout.group1[localIndex] = EXPECTED_LOCAL_RANK_SIZE + localIndex;
    }
    layout.localIndex = param.myRank % EXPECTED_LOCAL_RANK_SIZE;
    layout.serverIndex = param.myRank / EXPECTED_LOCAL_RANK_SIZE;
    return HCCL_SUCCESS;
}

HcclResult GetChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank, const ChannelInfo *&channel)
{
    channel = resCtx.FindChannel(remoteRank);
    if (channel == nullptr) {
        HCCL_ERROR("Channel to rank [%u] was not found", remoteRank);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult GetChannel(const ChannelTable &channelTable, uint32_t remoteRank, const ChannelInfo *&channel)
{
    if (remoteRank >= EXPECTED_RANK_SIZE || channelTable[remoteRank] == nullptr) {
        HCCL_ERROR("Channel to rank [%u] was not found", remoteRank);
        return HCCL_E_INTERNAL;
    }
    channel = channelTable[remoteRank];
    return HCCL_SUCCESS;
}

DataRange SplitRange(const DataRange &range, uint32_t segmentNum, uint32_t segmentIndex)
{
    const uint64_t baseCount = range.count / segmentNum;
    const uint64_t remainder = range.count % segmentNum;
    const uint64_t segmentCount = baseCount + (segmentIndex < remainder ? 1 : 0);
    const uint64_t segmentOffset
        = range.offset + baseCount * segmentIndex + std::min<uint64_t>(segmentIndex, remainder);
    return DataRange{segmentOffset, segmentCount};
}

HcclResult CompleteWorkerGroup(
    const AlgResourceCtx &resCtx, uint32_t workerIndex, uint32_t workerNum, uint32_t startNotify)
{
    const ThreadHandle workerThread = resCtx.threads[workerIndex + 1];
    const uint32_t firstChild = workerIndex * 2 + 1;
    const uint32_t secondChild = firstChild + 1;
    const uint32_t firstChildNotify = startNotify == 0 ? 1 : 0;
    constexpr uint32_t secondChildNotify = 2;
    if (firstChild < workerNum) {
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, firstChildNotify, CUSTOM_TIMEOUT));
    }
    if (secondChild < workerNum) {
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, secondChildNotify, CUSTOM_TIMEOUT));
    }
    if (workerIndex == 0) {
        CHK_RET(HcommThreadNotifyRecordOnThread(workerThread, resCtx.threads[0], MAIN_PHASE_DONE_NOTIFY));
    } else {
        const uint32_t parentIndex = (workerIndex - 1) / 2;
        const ThreadHandle parentThread = resCtx.threads[parentIndex + 1];
        const uint32_t parentNotify = (workerIndex & 1U) != 0 ? firstChildNotify : secondChildNotify;
        CHK_RET(HcommThreadNotifyRecordOnThread(workerThread, parentThread, parentNotify));
    }
    return HCCL_SUCCESS;
}

HcclResult CompleteWorkerPhase(const AlgResourceCtx &resCtx, uint32_t workerIndex, uint32_t startNotify)
{
    return CompleteWorkerGroup(resCtx, workerIndex, EXPECTED_WORKER_NUM, startNotify);
}

HcclResult RunDirectChunk(const OpParam &param, const AlgResourceCtx &resCtx, const uint8_t *input, uint8_t *output,
    uint8_t *localCcl, uint64_t chunkCount, uint64_t chunkBytes)
{
    const ThreadHandle thread = resCtx.threads[0];
    CHK_RET(HcommLocalCopyOnThread(thread, localCcl, input, chunkBytes));
    if (param.myRank == resCtx.rootRank) {
        for (uint32_t rank : resCtx.allRanks) {
            if (rank == resCtx.rootRank) {
                continue;
            }
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(resCtx, rank, channel));
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, PATH_A_READY_NOTIFY, CUSTOM_TIMEOUT));
            CHK_RET(HcommReadReduceOnThread(thread, channel->handle, localCcl, channel->remoteCclMem.addr, chunkCount,
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        }
        for (uint32_t rank : resCtx.allRanks) {
            if (rank == resCtx.rootRank) {
                continue;
            }
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(resCtx, rank, channel));
            CHK_RET(HcommWriteOnThread(thread, channel->handle, channel->remoteCclMem.addr, localCcl, chunkBytes));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, PATH_A_RESULT_NOTIFY));
        }
    } else {
        const ChannelInfo *rootChannel = nullptr;
        CHK_RET(GetChannel(resCtx, resCtx.rootRank, rootChannel));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, rootChannel->handle, PATH_A_READY_NOTIFY));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, rootChannel->handle, PATH_A_RESULT_NOTIFY, CUSTOM_TIMEOUT));
    }
    CHK_RET(HcommLocalCopyOnThread(thread, output, localCcl, chunkBytes));
    return HCCL_SUCCESS;
}

HcclResult RunTreeChunk(const OpParam &param, const AlgResourceCtx &resCtx, const TopologyLayout &layout,
    const uint8_t *input, uint8_t *output, uint8_t *localCcl, uint64_t chunkCount, uint64_t chunkBytes)
{
    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t localIndex = layout.localIndex;
    const uint32_t group0Leader = layout.group0.front();
    const uint32_t group1Leader = layout.group1.front();

    CHK_RET(HcommLocalCopyOnThread(thread, localCcl, input, chunkBytes));

    for (uint32_t stride = 1; stride < EXPECTED_LOCAL_RANK_SIZE; stride *= 2) {
        const uint32_t groupWidth = stride * 2;
        const uint32_t positionInGroup = localIndex % groupWidth;
        if (positionInGroup == 0) {
            const uint32_t childIndex = localIndex + stride;
            if (childIndex < EXPECTED_LOCAL_RANK_SIZE) {
                const ChannelInfo *childChannel = nullptr;
                CHK_RET(GetChannel(resCtx, resCtx.localRanks[childIndex], childChannel));
                CHK_RET(
                    HcommChannelNotifyWaitOnThread(thread, childChannel->handle, PATH_A_READY_NOTIFY, CUSTOM_TIMEOUT));
                CHK_RET(HcommReadReduceOnThread(thread, childChannel->handle, localCcl, childChannel->remoteCclMem.addr,
                    chunkCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
            }
        } else if (positionInGroup == stride) {
            const uint32_t parentIndex = localIndex - stride;
            const ChannelInfo *parentChannel = nullptr;
            CHK_RET(GetChannel(resCtx, resCtx.localRanks[parentIndex], parentChannel));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, parentChannel->handle, PATH_A_READY_NOTIFY));
            break;
        }
    }

    if (param.myRank == group1Leader) {
        const ChannelInfo *rootChannel = nullptr;
        CHK_RET(GetChannel(resCtx, group0Leader, rootChannel));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, rootChannel->handle, PATH_A_READY_NOTIFY));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, rootChannel->handle, PATH_A_RESULT_NOTIFY, CUSTOM_TIMEOUT));
    } else if (param.myRank == group0Leader) {
        const ChannelInfo *otherLeaderChannel = nullptr;
        CHK_RET(GetChannel(resCtx, group1Leader, otherLeaderChannel));
        CHK_RET(
            HcommChannelNotifyWaitOnThread(thread, otherLeaderChannel->handle, PATH_A_READY_NOTIFY, CUSTOM_TIMEOUT));
        CHK_RET(HcommReadReduceOnThread(thread, otherLeaderChannel->handle, localCcl,
            otherLeaderChannel->remoteCclMem.addr, chunkCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        CHK_RET(HcommWriteOnThread(
            thread, otherLeaderChannel->handle, otherLeaderChannel->remoteCclMem.addr, localCcl, chunkBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, otherLeaderChannel->handle, PATH_A_RESULT_NOTIFY));
    }

    for (uint32_t stride = EXPECTED_LOCAL_RANK_SIZE / 2; stride >= 1; stride /= 2) {
        const uint32_t groupWidth = stride * 2;
        const uint32_t positionInGroup = localIndex % groupWidth;
        if (positionInGroup == 0) {
            const uint32_t childIndex = localIndex + stride;
            if (childIndex < EXPECTED_LOCAL_RANK_SIZE) {
                const ChannelInfo *childChannel = nullptr;
                CHK_RET(GetChannel(resCtx, resCtx.localRanks[childIndex], childChannel));
                CHK_RET(HcommWriteOnThread(
                    thread, childChannel->handle, childChannel->remoteCclMem.addr, localCcl, chunkBytes));
                CHK_RET(HcommChannelNotifyRecordOnThread(thread, childChannel->handle, PATH_A_RESULT_NOTIFY));
            }
        } else if (positionInGroup == stride) {
            const uint32_t parentIndex = localIndex - stride;
            const ChannelInfo *parentChannel = nullptr;
            CHK_RET(GetChannel(resCtx, resCtx.localRanks[parentIndex], parentChannel));
            CHK_RET(
                HcommChannelNotifyWaitOnThread(thread, parentChannel->handle, PATH_A_RESULT_NOTIFY, CUSTOM_TIMEOUT));
        }
    }

    CHK_RET(HcommLocalCopyOnThread(thread, output, localCcl, chunkBytes));
    return HCCL_SUCCESS;
}

HcclResult RunTreeLane(const OpParam &param, const AlgResourceCtx &resCtx, const TopologyLayout &layout,
    const ChannelTable &channelTable, ThreadHandle thread, uint32_t rootIndex, uint32_t readyNotify,
    uint32_t resultNotify, const uint8_t *input, uint8_t *output, uint8_t *localCcl, uint64_t laneOffsetBytes,
    uint64_t crossScratchOffsetBytes, uint64_t laneCount, uint64_t laneBytes)
{
    const uint32_t relativeIndex
        = (layout.localIndex + EXPECTED_LOCAL_RANK_SIZE - rootIndex) % EXPECTED_LOCAL_RANK_SIZE;
    const uint32_t group0Leader = layout.group0[rootIndex];
    const uint32_t group1Leader = layout.group1[rootIndex];

    CHK_RET(HcommLocalCopyOnThread(thread, localCcl, input, laneBytes));

    for (uint32_t stride = 1; stride < EXPECTED_LOCAL_RANK_SIZE; stride *= 2) {
        const uint32_t groupWidth = stride * 2;
        const uint32_t positionInGroup = relativeIndex % groupWidth;
        if (positionInGroup == 0) {
            const uint32_t childRelativeIndex = relativeIndex + stride;
            if (childRelativeIndex < EXPECTED_LOCAL_RANK_SIZE) {
                const uint32_t childIndex = (rootIndex + childRelativeIndex) % EXPECTED_LOCAL_RANK_SIZE;
                const ChannelInfo *childChannel = nullptr;
                CHK_RET(GetChannel(channelTable, resCtx.localRanks[childIndex], childChannel));
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    thread, childChannel->handle, readyNotify, CUSTOM_TIMEOUT));
                CHK_RET(HcommReadReduceOnThread(thread, childChannel->handle, localCcl,
                    static_cast<uint8_t *>(childChannel->remoteCclMem.addr) + laneOffsetBytes, laneCount,
                    HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
            }
        } else if (positionInGroup == stride) {
            const uint32_t parentRelativeIndex = relativeIndex - stride;
            const uint32_t parentIndex = (rootIndex + parentRelativeIndex) % EXPECTED_LOCAL_RANK_SIZE;
            const ChannelInfo *parentChannel = nullptr;
            CHK_RET(GetChannel(channelTable, resCtx.localRanks[parentIndex], parentChannel));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, parentChannel->handle, readyNotify));
            break;
        }
    }

    if (param.myRank == group0Leader || param.myRank == group1Leader) {
        const uint32_t partnerRank = param.myRank == group0Leader ? group1Leader : group0Leader;
        const ChannelInfo *partnerChannel = nullptr;
        CHK_RET(GetChannel(channelTable, partnerRank, partnerChannel));
        uint8_t *const localScratch
            = static_cast<uint8_t *>(resCtx.localBuffer.addr) + crossScratchOffsetBytes;
        uint8_t *const remoteScratch
            = static_cast<uint8_t *>(partnerChannel->remoteCclMem.addr) + crossScratchOffsetBytes;
        CHK_RET(HcommWriteOnThread(thread, partnerChannel->handle, remoteScratch, localCcl, laneBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, partnerChannel->handle, readyNotify));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, partnerChannel->handle, readyNotify, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalReduceOnThread(
            thread, localCcl, localScratch, laneCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
    }

    for (uint32_t stride = EXPECTED_LOCAL_RANK_SIZE / 2; stride >= 1; stride /= 2) {
        const uint32_t groupWidth = stride * 2;
        const uint32_t positionInGroup = relativeIndex % groupWidth;
        if (positionInGroup == 0) {
            const uint32_t childRelativeIndex = relativeIndex + stride;
            if (childRelativeIndex < EXPECTED_LOCAL_RANK_SIZE) {
                const uint32_t childIndex = (rootIndex + childRelativeIndex) % EXPECTED_LOCAL_RANK_SIZE;
                const ChannelInfo *childChannel = nullptr;
                CHK_RET(GetChannel(channelTable, resCtx.localRanks[childIndex], childChannel));
                if ((childRelativeIndex & 1U) == 0) {
                    CHK_RET(HcommWriteOnThread(thread, childChannel->handle,
                        static_cast<uint8_t *>(childChannel->remoteCclMem.addr) + laneOffsetBytes,
                        localCcl, laneBytes));
                }
                CHK_RET(HcommChannelNotifyRecordOnThread(thread, childChannel->handle, resultNotify));
            }
        } else if (positionInGroup == stride) {
            const uint32_t parentRelativeIndex = relativeIndex - stride;
            const uint32_t parentIndex = (rootIndex + parentRelativeIndex) % EXPECTED_LOCAL_RANK_SIZE;
            const ChannelInfo *parentChannel = nullptr;
            CHK_RET(GetChannel(channelTable, resCtx.localRanks[parentIndex], parentChannel));
            CHK_RET(HcommChannelNotifyWaitOnThread(
                thread, parentChannel->handle, resultNotify, CUSTOM_TIMEOUT));
            if ((relativeIndex & 1U) != 0) {
                CHK_RET(HcommReadOnThread(thread, parentChannel->handle, output,
                    static_cast<const uint8_t *>(parentChannel->remoteCclMem.addr) + laneOffsetBytes, laneBytes));
            }
        }
    }

    if ((relativeIndex & 1U) == 0) {
        CHK_RET(HcommLocalCopyOnThread(thread, output, localCcl, laneBytes));
    }
    return HCCL_SUCCESS;
}

HcclResult RunDualRootTreeChunk(const OpParam &param, const AlgResourceCtx &resCtx, const TopologyLayout &layout,
    const ChannelTable &channelTable, uint64_t commonBufferSize, const uint8_t *input, uint8_t *output,
    uint8_t *localCcl, uint64_t chunkCount, uint64_t chunkBytes)
{
    const DataRange fullRange{0, chunkCount};
    const DataRange lane0 = SplitRange(fullRange, 2, 0);
    const DataRange lane1 = SplitRange(fullRange, 2, 1);
    const uint64_t lane0Bytes = lane0.count * FP32_BYTES;
    const uint64_t lane1Bytes = lane1.count * FP32_BYTES;
    const uint64_t requiredCclBytes = chunkBytes * 2;
    if (requiredCclBytes > commonBufferSize) {
        HCCL_ERROR("Dual-tree symmetric exchange requires [%llu] common CCL bytes, only [%llu] are available",
            static_cast<unsigned long long>(requiredCclBytes),
            static_cast<unsigned long long>(commonBufferSize));
        return HCCL_E_INTERNAL;
    }

    CHK_RET(HcommThreadNotifyWaitOnThread(
        resCtx.threads[1], DUAL_TREE_LANE_START_NOTIFY, CUSTOM_TIMEOUT));
    CHK_RET(RunTreeLane(param, resCtx, layout, channelTable, resCtx.threads[1], 1,
        DUAL_TREE_LANE1_READY_NOTIFY, DUAL_TREE_LANE1_RESULT_NOTIFY, input + lane0Bytes, output + lane0Bytes,
        localCcl + lane0Bytes, lane0Bytes, chunkBytes + lane0Bytes, lane1.count, lane1Bytes));
    CHK_RET(HcommThreadNotifyRecordOnThread(
        resCtx.threads[1], resCtx.threads[0], DUAL_TREE_LANE_DONE_NOTIFY));

    CHK_RET(HcommThreadNotifyRecordOnThread(
        resCtx.threads[0], resCtx.threads[1], DUAL_TREE_LANE_START_NOTIFY));
    CHK_RET(RunTreeLane(param, resCtx, layout, channelTable, resCtx.threads[0], 0, DUAL_TREE_LANE0_READY_NOTIFY,
        DUAL_TREE_LANE0_RESULT_NOTIFY, input, output, localCcl, 0, chunkBytes, lane0.count, lane0Bytes));
    CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.threads[0], DUAL_TREE_LANE_DONE_NOTIFY, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult RunRecursiveDoublingChunk(const OpParam &param, const AlgResourceCtx &resCtx, const TopologyLayout &layout,
    const uint8_t *input, uint8_t *output, uint8_t *localCcl, uint64_t chunkCount, uint64_t chunkBytes)
{
    const ThreadHandle thread = resCtx.threads[0];
    CHK_RET(HcommLocalCopyOnThread(thread, localCcl, input, chunkBytes));

    for (uint32_t step = 0; step < 4; ++step) {
        uint32_t partnerRank = 0;
        if (step < 3) {
            partnerRank = resCtx.localRanks[layout.localIndex ^ (1U << step)];
        } else {
            partnerRank = layout.serverIndex == 0 ? layout.group1[layout.localIndex] : layout.group0[layout.localIndex];
        }
        const ChannelInfo *partnerChannel = nullptr;
        CHK_RET(GetChannel(resCtx, partnerRank, partnerChannel));
        uint8_t *const localScratch = localCcl + (step + 1) * chunkBytes;
        uint8_t *const remoteScratch
            = static_cast<uint8_t *>(partnerChannel->remoteCclMem.addr) + (step + 1) * chunkBytes;
        CHK_RET(HcommWriteOnThread(thread, partnerChannel->handle, remoteScratch, localCcl, chunkBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, partnerChannel->handle, PATH_A_READY_NOTIFY));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, partnerChannel->handle, PATH_A_READY_NOTIFY, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalReduceOnThread(
            thread, localCcl, localScratch, chunkCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
    }

    CHK_RET(HcommLocalCopyOnThread(thread, output, localCcl, chunkBytes));
    return HCCL_SUCCESS;
}

HcclResult RunDependencyTreeFullMeshReduce(const OpParam &param, const AlgResourceCtx &resCtx, uint8_t *localCcl,
    uint32_t myRankIndex, uint64_t segmentCount, uint64_t slotBytes)
{
    std::array<uint32_t, EXPECTED_RANK_SIZE> reductionOrder{};
    std::array<ThreadHandle, EXPECTED_RANK_SIZE> producers{};
    uint32_t reductionIndex = 0;
    reductionOrder[reductionIndex++] = myRankIndex;
    for (uint32_t rankIndex = 0; rankIndex < EXPECTED_RANK_SIZE; ++rankIndex) {
        if (rankIndex != myRankIndex) {
            reductionOrder[reductionIndex++] = rankIndex;
        }
    }

    const ThreadHandle mainThread = resCtx.threads[0];
    producers[0] = mainThread;
    for (uint32_t position = 1; position < EXPECTED_RANK_SIZE; ++position) {
        const uint32_t rankIndex = reductionOrder[position];
        const uint32_t workerIndex = rankIndex < myRankIndex ? rankIndex : rankIndex - 1;
        producers[position] = resCtx.threads[workerIndex + 1];
    }

    uint32_t reduceStage = 0;
    for (uint32_t stride = 1; stride < EXPECTED_RANK_SIZE; stride *= 2, ++reduceStage) {
        if (reduceStage >= FULL_MESH_REDUCE_STAGE_NUM) {
            HCCL_ERROR("Full-Mesh reduction has too many dependency stages");
            return HCCL_E_INTERNAL;
        }
        const uint32_t dependencyNotify = WORKER_REDUCE_DEPENDENCY_NOTIFY_BASE + reduceStage;
        for (uint32_t position = 0; position < EXPECTED_RANK_SIZE; position += 2 * stride) {
            const uint32_t sourcePosition = position + stride;
            const uint32_t destinationIndex = reductionOrder[position];
            const uint32_t sourceIndex = reductionOrder[sourcePosition];
            const ThreadHandle leftProducer = producers[position];
            const ThreadHandle rightProducer = producers[sourcePosition];
            const ThreadHandle reduceThread = leftProducer == mainThread ? rightProducer : leftProducer;
            uint8_t *const destination = localCcl + destinationIndex * slotBytes;
            const uint8_t *const source = localCcl + sourceIndex * slotBytes;

            const ThreadHandle dependencyProducer = leftProducer == mainThread ? mainThread : rightProducer;
            CHK_RET(HcommThreadNotifyRecordOnThread(
                dependencyProducer, reduceThread, dependencyNotify));
            CHK_RET(HcommThreadNotifyWaitOnThread(
                reduceThread, dependencyNotify, CUSTOM_TIMEOUT));
            CHK_RET(HcommLocalReduceOnThread(
                reduceThread, destination, source, segmentCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
            producers[position] = reduceThread;
        }
    }

    if (reductionOrder.front() != myRankIndex || resCtx.allRanks[myRankIndex] != param.myRank) {
        HCCL_ERROR("Full-Mesh parallel reduction root is invalid for rank [%u]", param.myRank);
        return HCCL_E_INTERNAL;
    }
    CHK_RET(HcommThreadNotifyRecordOnThread(producers[0], mainThread, MAIN_PHASE_DONE_NOTIFY));
    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, MAIN_PHASE_DONE_NOTIFY, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult RunFullMeshRsagChunk(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelTable &channelTable, uint32_t myRankIndex, const uint8_t *input, uint8_t *output,
    uint8_t *localCcl, uint64_t chunkCount, uint64_t slotBytes, bool hasPreviousFullMeshChunk,
    bool isFinalFullMeshChunk)
{
    if (chunkCount % EXPECTED_RANK_SIZE != 0) {
        HCCL_ERROR("Full-Mesh chunk count [%llu] is not divisible by [%u]",
            static_cast<unsigned long long>(chunkCount), EXPECTED_RANK_SIZE);
        return HCCL_E_PARA;
    }
    const ThreadHandle mainThread = resCtx.threads[0];
    const uint64_t segmentCount = chunkCount / EXPECTED_RANK_SIZE;
    const uint64_t segmentBytes = segmentCount * FP32_BYTES;
    if (slotBytes == 0 || slotBytes > std::numeric_limits<uint64_t>::max() / EXPECTED_RANK_SIZE
        || segmentBytes > slotBytes) {
        HCCL_ERROR("Full-Mesh fixed slot size is invalid");
        return HCCL_E_INTERNAL;
    }
    const uint64_t requiredCclBytes = EXPECTED_RANK_SIZE * slotBytes;
    if (requiredCclBytes > resCtx.localBuffer.size) {
        HCCL_ERROR("Full-Mesh fixed slots require [%llu] CCL bytes, only [%llu] are available",
            static_cast<unsigned long long>(requiredCclBytes),
            static_cast<unsigned long long>(resCtx.localBuffer.size));
        return HCCL_E_INTERNAL;
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        if (requiredCclBytes > channel.remoteCclMem.size) {
            HCCL_ERROR("Remote rank [%u] has insufficient Full-Mesh scratch space", channel.remoteRank);
            return HCCL_E_INTERNAL;
        }
    }

    uint32_t workerIndex = 0;
    for (uint32_t rankIndex = 0; rankIndex < EXPECTED_RANK_SIZE; ++rankIndex) {
        const uint32_t remoteRank = resCtx.allRanks[rankIndex];
        if (remoteRank == param.myRank) {
            continue;
        }
        const ThreadHandle workerThread = resCtx.threads[workerIndex + 1];
        const ChannelInfo *const channel = channelTable[rankIndex];
        const uint64_t packedByteOffset = rankIndex * segmentBytes;

        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, workerThread, WORKER_RS_START_NOTIFY));
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, WORKER_RS_START_NOTIFY, CUSTOM_TIMEOUT));
        uint8_t *const remoteSegment
            = static_cast<uint8_t *>(channel->remoteCclMem.addr) + myRankIndex * slotBytes;
        const uint8_t *const localSegment = input + packedByteOffset;
        CHK_RET(HcommWriteOnThread(
            workerThread, channel->handle, remoteSegment, localSegment, segmentBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, channel->handle, FULL_MESH_RS_DONE_NOTIFY));
        CHK_RET(
            HcommChannelNotifyWaitOnThread(workerThread, channel->handle, FULL_MESH_RS_DONE_NOTIFY, CUSTOM_TIMEOUT));
        ++workerIndex;
    }

    if (hasPreviousFullMeshChunk) {
        for (uint32_t rankIndex = 0; rankIndex < EXPECTED_RANK_SIZE; ++rankIndex) {
            if (rankIndex == myRankIndex) {
                continue;
            }
            const ChannelInfo *const channel = channelTable[rankIndex];
            CHK_RET(HcommChannelNotifyWaitOnThread(
                mainThread, channel->handle, FULL_MESH_AG_DONE_NOTIFY, CUSTOM_TIMEOUT));
        }
    }

    const uint64_t ownedPackedByteOffset = myRankIndex * segmentBytes;
    uint8_t *const ownedSegment = localCcl + myRankIndex * slotBytes;
    const uint8_t *const ownedInput = input + ownedPackedByteOffset;
    CHK_RET(HcommLocalCopyOnThread(mainThread, ownedSegment, ownedInput, segmentBytes));

    CHK_RET(RunDependencyTreeFullMeshReduce(param, resCtx, localCcl, myRankIndex, segmentCount, slotBytes));
    for (uint32_t rankIndex = 0; rankIndex < EXPECTED_RANK_SIZE; ++rankIndex) {
        if (rankIndex == myRankIndex) {
            continue;
        }
        const ChannelInfo *const channel = channelTable[rankIndex];
        CHK_RET(HcommChannelNotifyRecordOnThread(
            mainThread, channel->handle, FULL_MESH_REDUCE_READY_NOTIFY));
    }

    workerIndex = 0;
    for (uint32_t rankIndex = 0; rankIndex < EXPECTED_RANK_SIZE; ++rankIndex) {
        const uint32_t remoteRank = resCtx.allRanks[rankIndex];
        if (remoteRank == param.myRank) {
            continue;
        }
        const ThreadHandle workerThread = resCtx.threads[workerIndex + 1];
        const ChannelInfo *const channel = channelTable[rankIndex];
        const uint64_t packedByteOffset = rankIndex * segmentBytes;

        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, workerThread, WORKER_AG_START_NOTIFY));
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, WORKER_AG_START_NOTIFY, CUSTOM_TIMEOUT));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            workerThread, channel->handle, FULL_MESH_REDUCE_READY_NOTIFY, CUSTOM_TIMEOUT));
        uint8_t *const localOutput = output + packedByteOffset;
        const uint8_t *const remoteResult
            = static_cast<const uint8_t *>(channel->remoteCclMem.addr) + rankIndex * slotBytes;
        CHK_RET(HcommReadOnThread(
            workerThread, channel->handle, localOutput, remoteResult, segmentBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(
            workerThread, channel->handle, FULL_MESH_AG_DONE_NOTIFY));
        if (isFinalFullMeshChunk) {
            CHK_RET(CompleteWorkerPhase(resCtx, workerIndex, WORKER_AG_START_NOTIFY));
        }
        ++workerIndex;
    }

    CHK_RET(HcommLocalCopyOnThread(
        mainThread, output + ownedPackedByteOffset, ownedSegment, segmentBytes));
    if (!isFinalFullMeshChunk) {
        return HCCL_SUCCESS;
    }

    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, MAIN_PHASE_DONE_NOTIFY, CUSTOM_TIMEOUT));
    for (uint32_t rankIndex = 0; rankIndex < EXPECTED_RANK_SIZE; ++rankIndex) {
        if (rankIndex == myRankIndex) {
            continue;
        }
        const ChannelInfo *const channel = channelTable[rankIndex];
        CHK_RET(HcommChannelNotifyWaitOnThread(
            mainThread, channel->handle, FULL_MESH_AG_DONE_NOTIFY, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    ChannelTable channelTable{};
    uint32_t myRankIndex = EXPECTED_RANK_SIZE;
    CHK_RET(ValidateExecOp(param, resCtx, channelTable, myRankIndex));

    TopologyLayout layout;
    CHK_RET(BuildTopologyLayout(param, layout));

    uint64_t commonBufferSize = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        commonBufferSize = std::min(commonBufferSize, channel.remoteCclMem.size);
    }
    commonBufferSize = (commonBufferSize / FP32_BYTES) * FP32_BYTES;
    if (commonBufferSize < FP32_BYTES * 2) {
        HCCL_ERROR("The common CCL buffer is too small");
        return HCCL_E_INTERNAL;
    }

    const uint64_t totalBytes = param.count * FP32_BYTES;
    const AlgorithmType algorithm = SelectAlgorithm(totalBytes);

    uint8_t *const localCcl = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const uint8_t *const input = static_cast<const uint8_t *>(param.inputPtr);
    uint8_t *const output = static_cast<uint8_t *>(param.outputPtr);

    if (algorithm == AlgorithmType::FULL_MESH_RSAG) {
        const uint64_t slotBytes
            = (commonBufferSize / FP32_BYTES / EXPECTED_RANK_SIZE) * FP32_BYTES;
        const uint64_t maxChunkCount = (slotBytes / FP32_BYTES) * EXPECTED_RANK_SIZE;
        if (maxChunkCount == 0) {
            HCCL_ERROR("The common CCL buffer cannot hold one aligned Full-Mesh chunk");
            return HCCL_E_INTERNAL;
        }

        uint64_t elementOffset = 0;
        uint64_t remainingCount = param.count;
        bool hasPreviousFullMeshChunk = false;
        while (remainingCount >= EXPECTED_RANK_SIZE) {
            const uint64_t alignedRemaining
                = (remainingCount / EXPECTED_RANK_SIZE) * EXPECTED_RANK_SIZE;
            const uint64_t chunkCount = std::min(maxChunkCount, alignedRemaining);
            const uint64_t byteOffset = elementOffset * FP32_BYTES;
            const bool isFinalFullMeshChunk = remainingCount - chunkCount < EXPECTED_RANK_SIZE;
            CHK_RET(RunFullMeshRsagChunk(param, resCtx, channelTable, myRankIndex,
                input + byteOffset, output + byteOffset, localCcl, chunkCount, slotBytes,
                hasPreviousFullMeshChunk, isFinalFullMeshChunk));
            elementOffset += chunkCount;
            remainingCount -= chunkCount;
            hasPreviousFullMeshChunk = true;
        }
        if (remainingCount != 0) {
            const uint64_t byteOffset = elementOffset * FP32_BYTES;
            CHK_RET(RunDirectChunk(param, resCtx, input + byteOffset, output + byteOffset, localCcl, remainingCount,
                remainingCount * FP32_BYTES));
        }
        return HCCL_SUCCESS;
    }

    uint64_t maxChunkCount = commonBufferSize / FP32_BYTES;
    if (algorithm == AlgorithmType::RECURSIVE_DOUBLING) {
        maxChunkCount /= RECURSIVE_DOUBLING_REGION_NUM;
    }
    if (maxChunkCount == 0) {
        HCCL_ERROR("No usable FP32 elements fit in the CCL buffer");
        return HCCL_E_INTERNAL;
    }

    const uint64_t numChunks = (param.count - 1) / maxChunkCount + 1;
    const uint64_t baseChunkCount = param.count / numChunks;
    const uint64_t chunkRemainder = param.count % numChunks;
    uint64_t elementOffset = 0;
    for (uint64_t chunkIndex = 0; chunkIndex < numChunks; ++chunkIndex) {
        const uint64_t chunkCount = baseChunkCount + (chunkIndex < chunkRemainder ? 1 : 0);
        const uint64_t chunkBytes = chunkCount * FP32_BYTES;
        const uint64_t byteOffset = elementOffset * FP32_BYTES;
        const uint8_t *const chunkInput = input + byteOffset;
        uint8_t *const chunkOutput = output + byteOffset;

        if (algorithm == AlgorithmType::DIRECT) {
            CHK_RET(RunDirectChunk(param, resCtx, chunkInput, chunkOutput, localCcl, chunkCount, chunkBytes));
        } else if (algorithm == AlgorithmType::TREE) {
            CHK_RET(RunTreeChunk(param, resCtx, layout, chunkInput, chunkOutput, localCcl, chunkCount, chunkBytes));
        } else if (algorithm == AlgorithmType::DUAL_ROOT_TREE) {
            CHK_RET(RunDualRootTreeChunk(param, resCtx, layout, channelTable, commonBufferSize,
                chunkInput, chunkOutput, localCcl, chunkCount, chunkBytes));
        } else if (algorithm == AlgorithmType::RECURSIVE_DOUBLING) {
            CHK_RET(RunRecursiveDoublingChunk(
                param, resCtx, layout, chunkInput, chunkOutput, localCcl, chunkCount, chunkBytes));
        }
        elementOffset += chunkCount;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl