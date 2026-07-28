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
constexpr uint64_t FP32_DATA_TYPE_SIZE = sizeof(float);
constexpr uint32_t REQUIRED_CHANNEL_NOTIFY_NUM = NOTIFY_IDX_DATA_SIGNAL + 1;
constexpr uint32_t EXPECTED_RANK_SIZE = 16;
constexpr uint32_t SERVER_GROUP_SIZE = 8;
constexpr uint32_t LOCAL_PEER_NUM = SERVER_GROUP_SIZE - 1;
constexpr uint32_t REMOTE_PEER_NUM = SERVER_GROUP_SIZE;
constexpr uint32_t GLOBAL_PEER_NUM = EXPECTED_RANK_SIZE - 1;
constexpr uint32_t GLOBAL_WORKER_NUM = GLOBAL_PEER_NUM - 1;
constexpr uint64_t FULL_BLOCK_ALIGNMENT = EXPECTED_RANK_SIZE * FP32_DATA_TYPE_SIZE;
constexpr uint32_t BUFFER_SLOT_NUM = 2;
constexpr uint32_t WORKER_START_NOTIFY_IDX = 0;
constexpr uint32_t MAIN_WORKER_DONE_NOTIFY_BASE = 1;
constexpr uint32_t ROOT_COPY_THREAD_INDEX = GLOBAL_PEER_NUM;
constexpr uint32_t ROOT_COPY_START_NOTIFY_IDX = 0;
constexpr uint32_t MAIN_ROOT_COPY_DONE_NOTIFY_IDX = MAIN_WORKER_DONE_NOTIFY_BASE + GLOBAL_WORKER_NUM;

static_assert(sizeof(float) == 4, "HCCL FP32 must contain four bytes");
static_assert(BROADCAST_AICPU_THREAD_NUM == GLOBAL_PEER_NUM + 1,
    "flat16 dedicated-copy pipeline requires fifteen data threads and one copy thread");
static_assert(BROADCAST_AICPU_NOTIFY_NUM_PER_THREAD > MAIN_ROOT_COPY_DONE_NOTIFY_IDX,
    "main thread requires notify 0 for Host sync, notify 1-14 for workers and notify 15 for root copy");
static_assert(BROADCAST_AICPU_NOTIFY_NUM_PER_THREAD > ROOT_COPY_START_NOTIFY_IDX,
    "the dedicated root-copy thread requires notify 0");

const ChannelInfo *FindChannelByRank(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

struct ScatterAllGatherPlan {
    uint32_t localLayer = INVALID_VALUE_RANKID;
    uint32_t remoteLayer = INVALID_VALUE_RANKID;
    uint32_t localGroupIndex = INVALID_VALUE_RANKID;
    uint32_t pairedRemoteRank = INVALID_VALUE_RANKID;
    bool rootInLocalGroup = false;
    std::vector<uint32_t> localGroup;
    std::vector<uint32_t> remoteGroup;
};

struct SliceLayout {
    uint64_t offsetBytes = 0;
    uint64_t sizeBytes = 0;
};

bool IsBufferRangeValid(const CommBuffer &buffer, uint64_t slotOffset, uint64_t dataOffset, uint64_t dataSize)
{
    if (buffer.addr == nullptr || slotOffset > buffer.size) {
        return false;
    }
    const uint64_t slotRemaining = buffer.size - slotOffset;
    return dataOffset <= slotRemaining && dataSize <= slotRemaining - dataOffset;
}

HcclResult ValidateSortedGroup(
    const char *groupName, const std::vector<uint32_t> &group, uint32_t rankSize)
{
    CHK_PRT_RET(group.size() != SERVER_GROUP_SIZE,
        HCCL_ERROR("[ExecOp] invalid %s size, expected[%u], actual[%llu]", groupName, SERVER_GROUP_SIZE,
            static_cast<unsigned long long>(group.size())),
        HCCL_E_PARA);
    for (uint32_t idx = 0; idx < SERVER_GROUP_SIZE; ++idx) {
        CHK_PRT_RET(group[idx] >= rankSize,
            HCCL_ERROR("[ExecOp] invalid rank[%u] in %s, rankSize[%u]", group[idx], groupName, rankSize),
            HCCL_E_PARA);
        if (idx > 0) {
            CHK_PRT_RET(group[idx] == group[idx - 1],
                HCCL_ERROR("[ExecOp] duplicated rank[%u] in %s", group[idx], groupName), HCCL_E_PARA);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BuildScatterAllGatherPlan(
    const OpParam &param, const AlgResourceCtx &resCtx, ScatterAllGatherPlan &plan)
{
    CHK_PRT_RET(param.rankSize != EXPECTED_RANK_SIZE,
        HCCL_ERROR("[ExecOp] scatter-allgather requires rankSize[%u], actual[%u]", EXPECTED_RANK_SIZE,
            param.rankSize),
        HCCL_E_PARA);

    uint32_t firstLayer = INVALID_VALUE_RANKID;
    uint32_t secondLayer = INVALID_VALUE_RANKID;
    uint32_t firstLayerPeerNum = 0;
    uint32_t secondLayerPeerNum = 0;
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PRT_RET(channel.netLayer == INVALID_VALUE_RANKID,
            HCCL_ERROR("[ExecOp] channel to rank[%u] has no network layer", channel.remoteRank),
            HCCL_E_INTERNAL);
        if (firstLayer == INVALID_VALUE_RANKID || channel.netLayer == firstLayer) {
            firstLayer = channel.netLayer;
            ++firstLayerPeerNum;
        } else if (secondLayer == INVALID_VALUE_RANKID || channel.netLayer == secondLayer) {
            secondLayer = channel.netLayer;
            ++secondLayerPeerNum;
        } else {
            HCCL_ERROR("[ExecOp] more than two selected network layers, layer[%u]", channel.netLayer);
            return HCCL_E_PARA;
        }
    }

    if (firstLayerPeerNum == LOCAL_PEER_NUM && secondLayerPeerNum == REMOTE_PEER_NUM) {
        plan.localLayer = firstLayer;
        plan.remoteLayer = secondLayer;
    } else if (firstLayerPeerNum == REMOTE_PEER_NUM && secondLayerPeerNum == LOCAL_PEER_NUM) {
        plan.localLayer = secondLayer;
        plan.remoteLayer = firstLayer;
    } else {
        HCCL_ERROR("[ExecOp] invalid layer peer counts, firstLayer[%u] peers[%u], secondLayer[%u] peers[%u]",
            firstLayer, firstLayerPeerNum, secondLayer, secondLayerPeerNum);
        return HCCL_E_PARA;
    }

    plan.localGroup.clear();
    plan.remoteGroup.clear();
    plan.localGroup.reserve(SERVER_GROUP_SIZE);
    plan.remoteGroup.reserve(SERVER_GROUP_SIZE);
    plan.localGroup.push_back(param.myRank);
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.netLayer == plan.localLayer) {
            plan.localGroup.push_back(channel.remoteRank);
        } else if (channel.netLayer == plan.remoteLayer) {
            plan.remoteGroup.push_back(channel.remoteRank);
        } else {
            HCCL_ERROR("[ExecOp] channel to rank[%u] uses unexpected layer[%u]", channel.remoteRank,
                channel.netLayer);
            return HCCL_E_PARA;
        }
    }
    std::sort(plan.localGroup.begin(), plan.localGroup.end());
    std::sort(plan.remoteGroup.begin(), plan.remoteGroup.end());
    CHK_RET(ValidateSortedGroup("localGroup", plan.localGroup, param.rankSize));
    CHK_RET(ValidateSortedGroup("remoteGroup", plan.remoteGroup, param.rankSize));

    for (uint32_t localIdx = 0; localIdx < SERVER_GROUP_SIZE; ++localIdx) {
        for (uint32_t remoteIdx = 0; remoteIdx < SERVER_GROUP_SIZE; ++remoteIdx) {
            CHK_PRT_RET(plan.localGroup[localIdx] == plan.remoteGroup[remoteIdx],
                HCCL_ERROR("[ExecOp] rank[%u] appears in both server groups", plan.localGroup[localIdx]),
                HCCL_E_PARA);
        }
        if (plan.localGroup[localIdx] == param.myRank) {
            plan.localGroupIndex = localIdx;
        }
    }
    CHK_PRT_RET(plan.localGroupIndex >= SERVER_GROUP_SIZE,
        HCCL_ERROR("[ExecOp] myRank[%u] is absent from localGroup", param.myRank), HCCL_E_INTERNAL);

    const bool rootInLocalGroup = std::binary_search(plan.localGroup.begin(), plan.localGroup.end(), param.root);
    const bool rootInRemoteGroup = std::binary_search(plan.remoteGroup.begin(), plan.remoteGroup.end(), param.root);
    CHK_PRT_RET(rootInLocalGroup == rootInRemoteGroup,
        HCCL_ERROR("[ExecOp] root[%u] must be present in exactly one server group", param.root), HCCL_E_PARA);
    plan.rootInLocalGroup = rootInLocalGroup;
    plan.pairedRemoteRank = plan.remoteGroup[plan.localGroupIndex];

    const ChannelInfo *pairedChannel = FindChannelByRank(resCtx, plan.pairedRemoteRank);
    CHK_PRT_RET(pairedChannel == nullptr || pairedChannel->netLayer != plan.remoteLayer,
        HCCL_ERROR("[ExecOp] invalid paired channel, myRank[%u], pairedRemoteRank[%u]", param.myRank,
            plan.pairedRemoteRank),
        HCCL_E_NOT_FOUND);
    if (plan.rootInLocalGroup && param.myRank != param.root) {
        const ChannelInfo *rootChannel = FindChannelByRank(resCtx, param.root);
        CHK_PRT_RET(rootChannel == nullptr || rootChannel->netLayer != plan.localLayer,
            HCCL_ERROR("[ExecOp] root[%u] is not reachable through local layer[%u]", param.root, plan.localLayer),
            HCCL_E_NOT_FOUND);
    }
    return HCCL_SUCCESS;
}

HcclResult GetSliceLayout(uint64_t blockBytes, uint32_t sliceIndex, SliceLayout &layout)
{
    CHK_PRT_RET(blockBytes == 0 || blockBytes % FP32_DATA_TYPE_SIZE != 0 || sliceIndex >= EXPECTED_RANK_SIZE,
        HCCL_ERROR("[ExecOp] invalid slice request, blockBytes[%llu], sliceIndex[%u]",
            static_cast<unsigned long long>(blockBytes), sliceIndex),
        HCCL_E_PARA);

    const uint64_t blockElements = blockBytes / FP32_DATA_TYPE_SIZE;
    const uint64_t baseElements = blockElements / EXPECTED_RANK_SIZE;
    const uint64_t extraElements = blockElements % EXPECTED_RANK_SIZE;
    const uint64_t precedingExtra = std::min<uint64_t>(sliceIndex, extraElements);
    const uint64_t offsetElements = baseElements * sliceIndex + precedingExtra;
    const uint64_t sliceElements = baseElements + static_cast<uint64_t>(sliceIndex < extraElements);
    layout.offsetBytes = offsetElements * FP32_DATA_TYPE_SIZE;
    layout.sizeBytes = sliceElements * FP32_DATA_TYPE_SIZE;
    CHK_PRT_RET(layout.offsetBytes > blockBytes || layout.sizeBytes > blockBytes - layout.offsetBytes,
        HCCL_ERROR("[ExecOp] slice exceeds block, blockBytes[%llu], offset[%llu], size[%llu]",
            static_cast<unsigned long long>(blockBytes),
            static_cast<unsigned long long>(layout.offsetBytes),
            static_cast<unsigned long long>(layout.sizeBytes)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

uint64_t GetBalancedBlockBytes(uint64_t remainingBytes, uint64_t blocksLeft, uint64_t blockCapacity)
{
    if (remainingBytes == 0 || blocksLeft == 0 || blockCapacity == 0) {
        return 0;
    }
    const uint64_t targetBytes = (remainingBytes - 1) / blocksLeft + 1;
    const uint64_t alignedBytes =
        ((targetBytes - 1) / FULL_BLOCK_ALIGNMENT + 1) * FULL_BLOCK_ALIGNMENT;
    return std::min(remainingBytes, std::min(blockCapacity, alignedBytes));
}

HcclResult ScheduleRootInputCopy(const AlgResourceCtx &resCtx, const void *inputBlock,
    uint64_t slotOffset, uint64_t blockBytes)
{
    CHK_PTR_NULL(inputBlock);
    CHK_PRT_RET(resCtx.threads.size() != BROADCAST_AICPU_THREAD_NUM,
        HCCL_ERROR("[ExecOp] invalid thread count while scheduling root copy, expected[%u], actual[%llu]",
            BROADCAST_AICPU_THREAD_NUM, static_cast<unsigned long long>(resCtx.threads.size())),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(blockBytes == 0 || !IsBufferRangeValid(resCtx.localBuffer, slotOffset, 0, blockBytes),
        HCCL_ERROR("[ExecOp] root copy exceeds HCCL Buffer, slotOffset[%llu], blockBytes[%llu], bufferSize[%llu]",
            static_cast<unsigned long long>(slotOffset),
            static_cast<unsigned long long>(blockBytes),
            static_cast<unsigned long long>(resCtx.localBuffer.size)),
        HCCL_E_INTERNAL);

    const ThreadHandle copyThread = resCtx.threads[ROOT_COPY_THREAD_INDEX];
    void *localBlockAddr = static_cast<void *>(
        static_cast<uint8_t *>(resCtx.localBuffer.addr) + slotOffset);
    CHK_RET(HcommThreadNotifyRecordOnThread(
        resCtx.aicpuThread, copyThread, ROOT_COPY_START_NOTIFY_IDX));
    CHK_RET(HcommThreadNotifyWaitOnThread(copyThread, ROOT_COPY_START_NOTIFY_IDX, CUSTOM_TIMEOUT));
    CHK_RET(HcommLocalCopyOnThread(copyThread, localBlockAddr, inputBlock, blockBytes));
    CHK_RET(HcommThreadNotifyRecordOnThread(
        copyThread, resCtx.aicpuThread, MAIN_ROOT_COPY_DONE_NOTIFY_IDX));
    return HCCL_SUCCESS;
}

HcclResult WaitRootInputCopy(const AlgResourceCtx &resCtx)
{
    CHK_RET(HcommThreadNotifyWaitOnThread(
        resCtx.aicpuThread, MAIN_ROOT_COPY_DONE_NOTIFY_IDX, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult ScheduleOutputCopy(const AlgResourceCtx &resCtx, void *outputBlock,
    uint64_t slotOffset, uint64_t blockBytes)
{
    CHK_PTR_NULL(outputBlock);
    CHK_PRT_RET(resCtx.threads.size() != BROADCAST_AICPU_THREAD_NUM,
        HCCL_ERROR("[ExecOp] invalid thread count while scheduling output copy, expected[%u], actual[%llu]",
            BROADCAST_AICPU_THREAD_NUM, static_cast<unsigned long long>(resCtx.threads.size())),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(blockBytes == 0 || !IsBufferRangeValid(resCtx.localBuffer, slotOffset, 0, blockBytes),
        HCCL_ERROR("[ExecOp] output copy exceeds HCCL Buffer, slotOffset[%llu], blockBytes[%llu]",
            static_cast<unsigned long long>(slotOffset), static_cast<unsigned long long>(blockBytes)),
        HCCL_E_INTERNAL);

    const ThreadHandle copyThread = resCtx.threads[ROOT_COPY_THREAD_INDEX];
    const void *localBlockAddr = static_cast<const void *>(
        static_cast<const uint8_t *>(resCtx.localBuffer.addr) + slotOffset);
    CHK_RET(HcommThreadNotifyRecordOnThread(
        resCtx.aicpuThread, copyThread, ROOT_COPY_START_NOTIFY_IDX));
    CHK_RET(HcommThreadNotifyWaitOnThread(copyThread, ROOT_COPY_START_NOTIFY_IDX, CUSTOM_TIMEOUT));
    CHK_RET(HcommLocalCopyOnThread(copyThread, outputBlock, localBlockAddr, blockBytes));
    CHK_RET(HcommThreadNotifyRecordOnThread(
        copyThread, resCtx.aicpuThread, MAIN_ROOT_COPY_DONE_NOTIFY_IDX));
    return HCCL_SUCCESS;
}

HcclResult WaitOutputCopy(const AlgResourceCtx &resCtx)
{
    CHK_RET(HcommThreadNotifyWaitOnThread(
        resCtx.aicpuThread, MAIN_ROOT_COPY_DONE_NOTIFY_IDX, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult ScheduleGlobalAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t blockBytes, uint64_t slotOffset, uint32_t rootRank)
{
    CHK_PRT_RET(resCtx.threads.size() != BROADCAST_AICPU_THREAD_NUM,
        HCCL_ERROR("[ExecOp] invalid thread count, expected[%u], actual[%llu]", BROADCAST_AICPU_THREAD_NUM,
            static_cast<unsigned long long>(resCtx.threads.size())),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(param.myRank == rootRank || rootRank >= param.rankSize,
        HCCL_ERROR("[ExecOp] invalid non-root flat16 AllGather invocation, myRank[%u], rootRank[%u]",
            param.myRank, rootRank),
        HCCL_E_INTERNAL);

    // threads[0]负责一个peer，另外14条worker各负责一个peer，15条Channel同时下发。
    for (uint32_t workerIdx = 0; workerIdx < GLOBAL_WORKER_NUM; ++workerIdx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            resCtx.aicpuThread, resCtx.threads[workerIdx + 1], WORKER_START_NOTIFY_IDX));
    }

    uint32_t peerOrdinal = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CHK_PRT_RET(peerOrdinal >= GLOBAL_PEER_NUM,
            HCCL_ERROR("[ExecOp] too many peers while scheduling flat16 AllGather"), HCCL_E_INTERNAL);
        const ThreadHandle dataThread = resCtx.threads[peerOrdinal];
        const ChannelInfo *peerChannel = FindChannelByRank(resCtx, remoteRank);
        CHK_PRT_RET(peerChannel == nullptr,
            HCCL_ERROR("[ExecOp] invalid flat16 AllGather channel to rank[%u]", remoteRank), HCCL_E_NOT_FOUND);

        SliceLayout peerSlice;
        CHK_RET(GetSliceLayout(blockBytes, remoteRank, peerSlice));
        CHK_PRT_RET(!IsBufferRangeValid(
                        resCtx.localBuffer, slotOffset, peerSlice.offsetBytes, peerSlice.sizeBytes) ||
                !IsBufferRangeValid(
                    peerChannel->remoteCclMem, slotOffset, peerSlice.offsetBytes, peerSlice.sizeBytes),
            HCCL_ERROR("[ExecOp] peer slice exceeds HCCL Buffer, peerRank[%u], slotOffset[%llu], "
                       "offset[%llu], size[%llu]",
                remoteRank, static_cast<unsigned long long>(slotOffset),
                static_cast<unsigned long long>(peerSlice.offsetBytes),
                static_cast<unsigned long long>(peerSlice.sizeBytes)),
            HCCL_E_INTERNAL);

        if (peerOrdinal > 0) {
            CHK_RET(HcommThreadNotifyWaitOnThread(dataThread, WORKER_START_NOTIFY_IDX, CUSTOM_TIMEOUT));
        }
        const bool isRootPeer = remoteRank == rootRank;
        // The root's initial READY notification protects its complete block until every
        // non-root sends the final DATA completion. Non-root pairs only exchange READY:
        // the root's final barrier prevents any rank from reusing this block until all
        // incoming reads on all ranks have completed.
        if (!isRootPeer) {
            CHK_RET(HcommChannelNotifyRecordOnThread(dataThread, peerChannel->handle, NOTIFY_IDX_ACK));
            CHK_RET(HcommChannelNotifyWaitOnThread(
                dataThread, peerChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        }
        if (peerSlice.sizeBytes > 0) {
            void *localSliceAddr = static_cast<void *>(
                static_cast<uint8_t *>(resCtx.localBuffer.addr) + slotOffset + peerSlice.offsetBytes);
            const void *remoteSliceAddr = static_cast<const void *>(
                static_cast<const uint8_t *>(peerChannel->remoteCclMem.addr) + slotOffset + peerSlice.offsetBytes);
            CHK_RET(HcommReadOnThread(
                dataThread, peerChannel->handle, localSliceAddr, remoteSliceAddr, peerSlice.sizeBytes));
        }
        if (peerOrdinal > 0) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                dataThread, resCtx.aicpuThread, MAIN_WORKER_DONE_NOTIFY_BASE + peerOrdinal - 1));
        }
        ++peerOrdinal;
    }
    CHK_PRT_RET(peerOrdinal != GLOBAL_PEER_NUM,
        HCCL_ERROR("[ExecOp] global peer count mismatch, expected[%u], actual[%u]", GLOBAL_PEER_NUM, peerOrdinal),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult WaitGlobalAllGather(const AlgResourceCtx &resCtx)
{
    for (uint32_t workerIdx = 0; workerIdx < GLOBAL_WORKER_NUM; ++workerIdx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.aicpuThread, MAIN_WORKER_DONE_NOTIFY_BASE + workerIdx, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateResourcesAndGetChunkCapacity(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t expectedThreadNum, uint64_t &chunkCapacity)
{
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0,
        HCCL_ERROR("[ExecOp] invalid local HCCL buffer, addr[%p], size[%llu]", resCtx.localBuffer.addr,
            static_cast<unsigned long long>(resCtx.localBuffer.size)),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(resCtx.threads.size() != expectedThreadNum ||
            resCtx.aicpuThread != resCtx.threads[0],
        HCCL_ERROR("[ExecOp] invalid AICPU thread resources, expectedThreadNum[%u], actualThreadNum[%llu]",
            expectedThreadNum, static_cast<unsigned long long>(resCtx.threads.size())),
        HCCL_E_INTERNAL);

    const uint64_t expectedChannelNum = static_cast<uint64_t>(param.rankSize - 1);
    CHK_PRT_RET(static_cast<uint64_t>(resCtx.channels.size()) != expectedChannelNum,
        HCCL_ERROR("[ExecOp] channel count mismatch, expected[%llu], actual[%llu]",
            static_cast<unsigned long long>(expectedChannelNum),
            static_cast<unsigned long long>(resCtx.channels.size())),
        HCCL_E_NOT_FOUND);

    chunkCapacity = resCtx.localBuffer.size;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        const ChannelInfo *channel = FindChannelByRank(resCtx, remoteRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("[ExecOp] channel to remoteRank[%u] was not found", remoteRank), HCCL_E_NOT_FOUND);
        CHK_PRT_RET(channel->notifyNum < REQUIRED_CHANNEL_NOTIFY_NUM,
            HCCL_ERROR("[ExecOp] insufficient channel notify count, remoteRank[%u], expected[%u], actual[%u]",
                remoteRank, REQUIRED_CHANNEL_NOTIFY_NUM, channel->notifyNum),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(channel->remoteCclMem.addr == nullptr || channel->remoteCclMem.size == 0,
            HCCL_ERROR("[ExecOp] invalid remote HCCL buffer, remoteRank[%u], addr[%p], size[%llu]", remoteRank,
                channel->remoteCclMem.addr, static_cast<unsigned long long>(channel->remoteCclMem.size)),
            HCCL_E_INTERNAL);
        chunkCapacity = std::min(chunkCapacity, channel->remoteCclMem.size);
    }

    chunkCapacity = chunkCapacity / FP32_DATA_TYPE_SIZE * FP32_DATA_TYPE_SIZE;
    CHK_PRT_RET(chunkCapacity == 0, HCCL_ERROR("[ExecOp] chunk capacity is zero"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[ExecOp] unsupported data type[%d]", static_cast<int32_t>(param.dataType)), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize || param.root >= param.rankSize,
        HCCL_ERROR("[ExecOp] invalid rank information, myRank[%u], root[%u], rankSize[%u]", param.myRank,
            param.root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / FP32_DATA_TYPE_SIZE,
        HCCL_ERROR("[ExecOp] count[%llu] overflows byte size", static_cast<unsigned long long>(param.count)),
        HCCL_E_PARA);

    const uint64_t totalBytes = param.count * FP32_DATA_TYPE_SIZE;
    const bool useScatterAllGather = totalBytes > BROADCAST_SCATTER_ALLGATHER_THRESHOLD;
    const uint32_t expectedThreadNum = useScatterAllGather ? BROADCAST_AICPU_THREAD_NUM : 1;
    uint64_t chunkCapacity = 0;
    CHK_RET(ValidateResourcesAndGetChunkCapacity(param, resCtx, expectedThreadNum, chunkCapacity));

    const uint8_t *inputBase = static_cast<const uint8_t *>(param.inputPtr);
    uint8_t *outputBase = static_cast<uint8_t *>(param.outputPtr);

    // 4B和512KB逐条保留V1星形Pull时序，避免影响已经达到35us的低延迟路径。
    if (!useScatterAllGather) {
        const ChannelInfo *rootChannel = nullptr;
        if (param.myRank != param.root) {
            rootChannel = FindChannelByRank(resCtx, param.root);
            CHK_PRT_RET(rootChannel == nullptr,
                HCCL_ERROR("[ExecOp] channel to root[%u] was not found", param.root), HCCL_E_NOT_FOUND);
        }

        uint64_t processedBytes = 0;
        while (processedBytes < totalBytes) {
            const uint64_t remainingBytes = totalBytes - processedBytes;
            const uint64_t chunkBytes = std::min(chunkCapacity, remainingBytes);
            CHK_PRT_RET(chunkBytes == 0, HCCL_ERROR("[ExecOp] calculated chunk size is zero"), HCCL_E_INTERNAL);

            if (param.myRank == param.root) {
                const void *currentInput = static_cast<const void *>(inputBase + processedBytes);
                CHK_RET(HcommLocalCopyOnThread(
                    resCtx.aicpuThread, resCtx.localBuffer.addr, currentInput, chunkBytes));

                // 先通知所有非 root：当前分块已经写入 root HCCL Buffer。
                for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                    if (remoteRank == param.myRank) {
                        continue;
                    }
                    const ChannelInfo *channel = FindChannelByRank(resCtx, remoteRank);
                    CHK_PRT_RET(channel == nullptr,
                        HCCL_ERROR("[ExecOp] channel to remoteRank[%u] was not found", remoteRank),
                        HCCL_E_NOT_FOUND);
                    CHK_RET(HcommChannelNotifyRecordOnThread(
                        resCtx.aicpuThread, channel->handle, NOTIFY_IDX_ACK));
                }

                // 所有非 root 完成读取后，root 才能覆盖 HCCL Buffer 处理下一分块。
                for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                    if (remoteRank == param.myRank) {
                        continue;
                    }
                    const ChannelInfo *channel = FindChannelByRank(resCtx, remoteRank);
                    CHK_PRT_RET(channel == nullptr,
                        HCCL_ERROR("[ExecOp] channel to remoteRank[%u] was not found", remoteRank),
                        HCCL_E_NOT_FOUND);
                    CHK_RET(HcommChannelNotifyWaitOnThread(
                        resCtx.aicpuThread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
                }
            } else {
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    resCtx.aicpuThread, rootChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
                CHK_RET(HcommReadOnThread(resCtx.aicpuThread, rootChannel->handle, resCtx.localBuffer.addr,
                    rootChannel->remoteCclMem.addr, chunkBytes));
                void *currentOutput = static_cast<void *>(outputBase + processedBytes);
                CHK_RET(HcommLocalCopyOnThread(
                    resCtx.aicpuThread, currentOutput, resCtx.localBuffer.addr, chunkBytes));
                CHK_RET(HcommChannelNotifyRecordOnThread(
                    resCtx.aicpuThread, rootChannel->handle, NOTIFY_IDX_DATA_SIGNAL));
            }

            processedBytes += chunkBytes;
        }
        return HCCL_SUCCESS;
    }

    ScatterAllGatherPlan plan;
    CHK_RET(BuildScatterAllGatherPlan(param, resCtx, plan));

    const bool isRoot = param.myRank == param.root;
    const ChannelInfo *rootChannel = nullptr;
    if (!isRoot) {
        rootChannel = FindChannelByRank(resCtx, param.root);
        CHK_PRT_RET(rootChannel == nullptr,
            HCCL_ERROR("[ExecOp] channel to root[%u] was not found", param.root), HCCL_E_NOT_FOUND);
    }

    const uint64_t slotCapacity = chunkCapacity / BUFFER_SLOT_NUM;
    const uint64_t blockCapacity = slotCapacity / FULL_BLOCK_ALIGNMENT * FULL_BLOCK_ALIGNMENT;
    CHK_PRT_RET(blockCapacity == 0,
        HCCL_ERROR("[ExecOp] HCCL Buffer is too small for two sixteen-slice slots, chunkCapacity[%llu]",
            static_cast<unsigned long long>(chunkCapacity)),
        HCCL_E_INTERNAL);

    const bool useRootCopyPipeline = totalBytes > blockCapacity;
    const uint64_t blockNum = (totalBytes - 1) / blockCapacity + 1;
    uint64_t processedBytes = 0;
    uint64_t blockIndex = 0;
    bool outputCopyPending = false;
    while (processedBytes < totalBytes) {
        const uint64_t remainingBytes = totalBytes - processedBytes;
        const uint64_t blocksLeft = blockNum - blockIndex;
        // Keep the same number of double-buffered rounds, but balance their sizes. This avoids a
        // nearly empty final round and shortens the first synchronous input copy.
        const uint64_t blockBytes = GetBalancedBlockBytes(remainingBytes, blocksLeft, blockCapacity);
        CHK_PRT_RET(blockBytes == 0 || blockBytes % FP32_DATA_TYPE_SIZE != 0,
            HCCL_ERROR("[ExecOp] invalid block size[%llu]", static_cast<unsigned long long>(blockBytes)),
            HCCL_E_INTERNAL);
        const uint32_t slotIndex = static_cast<uint32_t>(blockIndex % BUFFER_SLOT_NUM);
        const uint64_t slotOffset = static_cast<uint64_t>(slotIndex) * blockCapacity;
        const uint64_t nextProcessedBytes = processedBytes + blockBytes;

        // 上一轮预取使用独立copy thread；本轮首次访问该槽前确认预取已经完成。
        if (isRoot && useRootCopyPipeline && blockIndex > 0) {
            CHK_RET(WaitRootInputCopy(resCtx));
        }

        SliceLayout ownSlice;
        CHK_RET(GetSliceLayout(blockBytes, param.myRank, ownSlice));
        CHK_PRT_RET(!IsBufferRangeValid(resCtx.localBuffer, slotOffset, 0, blockBytes) ||
                !IsBufferRangeValid(resCtx.localBuffer, slotOffset, ownSlice.offsetBytes, ownSlice.sizeBytes),
            HCCL_ERROR("[ExecOp] own slice exceeds HCCL Buffer, slotOffset[%llu], offset[%llu], size[%llu]",
                static_cast<unsigned long long>(slotOffset),
                static_cast<unsigned long long>(ownSlice.offsetBytes),
                static_cast<unsigned long long>(ownSlice.sizeBytes)),
            HCCL_E_INTERNAL);

        void *currentOutput = static_cast<void *>(outputBase + processedBytes);
        void *localBlockAddr = static_cast<void *>(
            static_cast<uint8_t *>(resCtx.localBuffer.addr) + slotOffset);
        void *localSliceAddr = static_cast<void *>(
            static_cast<uint8_t *>(resCtx.localBuffer.addr) + slotOffset + ownSlice.offsetBytes);

        if (isRoot) {
            // 首块仍由main thread准备；后续块已由上一轮独立copy thread预取到交替槽。
            if (!useRootCopyPipeline || blockIndex == 0) {
                const void *currentInput = static_cast<const void *>(inputBase + processedBytes);
                CHK_RET(HcommLocalCopyOnThread(
                    resCtx.aicpuThread, localBlockAddr, currentInput, blockBytes));
            }

            if (useRootCopyPipeline && nextProcessedBytes < totalBytes) {
                const uint64_t nextRemainingBytes = totalBytes - nextProcessedBytes;
                const uint64_t nextBlocksLeft = blockNum - blockIndex - 1;
                const uint64_t nextBlockBytes =
                    GetBalancedBlockBytes(nextRemainingBytes, nextBlocksLeft, blockCapacity);
                CHK_PRT_RET(nextBlockBytes == 0,
                    HCCL_ERROR("[ExecOp] calculated next balanced block size is zero"), HCCL_E_INTERNAL);
                const uint32_t nextSlotIndex = (slotIndex + 1) % BUFFER_SLOT_NUM;
                const uint64_t nextSlotOffset = static_cast<uint64_t>(nextSlotIndex) * blockCapacity;
                const void *nextInput = static_cast<const void *>(inputBase + nextProcessedBytes);
                // 独立copy thread预取下一块，不再阻塞当前块的任何Channel。
                CHK_RET(ScheduleRootInputCopy(resCtx, nextInput, nextSlotOffset, nextBlockBytes));
            }

            // Flat16 Scatter：15个非root同时通过各自Channel拉取唯一的约1/16分片。
            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                if (remoteRank == param.myRank) {
                    continue;
                }
                const ChannelInfo *channel = FindChannelByRank(resCtx, remoteRank);
                CHK_PRT_RET(channel == nullptr,
                    HCCL_ERROR("[ExecOp] scatter channel to rank[%u] was not found", remoteRank),
                    HCCL_E_NOT_FOUND);
                CHK_RET(HcommChannelNotifyRecordOnThread(
                    resCtx.aicpuThread, channel->handle, NOTIFY_IDX_ACK));
            }
        } else {
            CHK_RET(HcommChannelNotifyWaitOnThread(
                resCtx.aicpuThread, rootChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            if (ownSlice.sizeBytes > 0) {
                CHK_PRT_RET(!IsBufferRangeValid(
                                rootChannel->remoteCclMem, slotOffset, ownSlice.offsetBytes, ownSlice.sizeBytes),
                    HCCL_ERROR("[ExecOp] root remote slice exceeds HCCL Buffer, slotOffset[%llu], "
                               "offset[%llu], size[%llu]",
                        static_cast<unsigned long long>(slotOffset),
                        static_cast<unsigned long long>(ownSlice.offsetBytes),
                        static_cast<unsigned long long>(ownSlice.sizeBytes)),
                    HCCL_E_INTERNAL);
                const void *rootRemoteSliceAddr = static_cast<const void *>(
                    static_cast<const uint8_t *>(rootChannel->remoteCclMem.addr) + slotOffset + ownSlice.offsetBytes);
                CHK_RET(HcommReadOnThread(resCtx.aicpuThread, rootChannel->handle, localSliceAddr,
                    rootRemoteSliceAddr, ownSlice.sizeBytes));
            }
        }

        if (isRoot) {
            // READY was sent once after the complete root block became stable. Keep both the
            // current root block and its notify generation alive until every non-root has
            // gathered all 16 slices and reports one final completion.
            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                if (remoteRank == param.myRank) {
                    continue;
                }
                const ChannelInfo *channel = FindChannelByRank(resCtx, remoteRank);
                CHK_PRT_RET(channel == nullptr,
                    HCCL_ERROR("[ExecOp] completion channel to rank[%u] was not found", remoteRank),
                    HCCL_E_NOT_FOUND);
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    resCtx.aicpuThread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
            }
        } else {
            // Fifteen channels still transfer in parallel. The root slice is read directly
            // under the initial READY lifetime; the other fourteen peers exchange one READY
            // in each direction. Their per-peer DATA completion is unnecessary because the
            // final root barrier below globally protects every source slice from reuse.
            CHK_RET(ScheduleGlobalAllGather(param, resCtx, blockBytes, slotOffset, param.root));
            CHK_RET(WaitGlobalAllGather(resCtx));
            CHK_RET(HcommChannelNotifyRecordOnThread(
                resCtx.aicpuThread, rootChannel->handle, NOTIFY_IDX_DATA_SIGNAL));

            // Only one output copy is outstanding. The previous copy overlaps this block's
            // communication and is consumed before reusing notify 15 or the HCCL buffer slot.
            if (outputCopyPending) {
                CHK_RET(WaitOutputCopy(resCtx));
            }
            CHK_RET(ScheduleOutputCopy(resCtx, currentOutput, slotOffset, blockBytes));
            outputCopyPending = true;
        }

        processedBytes = nextProcessedBytes;
        ++blockIndex;
    }

    if (outputCopyPending) {
        CHK_RET(WaitOutputCopy(resCtx));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
