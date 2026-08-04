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
#include "exec_op.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace {
constexpr uint32_t PAYLOAD_READY_NOTIFY = 0;
constexpr uint32_t WORKSPACE_FREE_NOTIFY = 1;
constexpr uint32_t LARGE_REDUCE_WORKERS = 4;
constexpr uint32_t INNER_SERVER_LAYER = 0;
constexpr uint32_t BETWEEN_SERVER_LAYER = 1;
constexpr uint32_t OWNER_FANOUT_WORKERS = yga_reduce_scatter::PEER_COUNT;
constexpr uint32_t OWNER_TREE_WORKERS = yga_reduce_scatter::RANKS_PER_SERVER;
constexpr uint32_t TREE_LEVEL_ONE_NOTIFY = 7;
constexpr uint32_t TREE_LEVEL_TWO_NOTIFY = 8;
constexpr uint32_t TREE_LEVEL_THREE_NOTIFY = 9;
constexpr std::array<uint32_t, 3> LOCAL_BARRIER_LANE_XORS = {1U, 2U, 4U};

struct TileView {
    uint64_t offsetBytes = 0;
    uint64_t validBytes = 0;
};

struct WorkerRange {
    uint64_t elementOffset = 0;
    uint64_t elementCount = 0;
};

struct ServerLaneMap {
    std::array<uint32_t, yga_reduce_scatter::RANKS_PER_SERVER> nearby{};
    std::array<uint32_t, yga_reduce_scatter::RANKS_PER_SERVER> opposite{};
    uint32_t lane = INVALID_VALUE_RANKID;
};

struct OwnerInboxPlan {
    uint64_t slotStride = 0;
    uint64_t requiredBytes = 0;
};

const PeerPath *DirectPeer(const AlgResourceCtx &resources, uint32_t peerIndex)
{
    if (peerIndex == resources.selfIndex || peerIndex >= yga_reduce_scatter::RANK_COUNT) {
        return nullptr;
    }
    const uint32_t pathSlot = peerIndex < resources.selfIndex ? peerIndex : peerIndex - 1;
    if (pathSlot >= resources.peerPaths.size() || resources.peerPaths[pathSlot].rankIndex != peerIndex) {
        return nullptr;
    }
    return &resources.peerPaths[pathSlot];
}

bool DescribeServerLanes(const AlgResourceCtx &resources, ServerLaneMap &mapping)
{
    uint32_t nearbyCount = 1;
    uint32_t oppositeCount = 0;
    mapping.nearby[0] = resources.selfIndex;

    for (const PeerPath &path : resources.peerPaths) {
        if (path.fabricLayer == INNER_SERVER_LAYER && nearbyCount < mapping.nearby.size()) {
            mapping.nearby[nearbyCount++] = path.rankIndex;
        } else if (path.fabricLayer == BETWEEN_SERVER_LAYER && oppositeCount < mapping.opposite.size()) {
            mapping.opposite[oppositeCount++] = path.rankIndex;
        } else {
            return false;
        }
    }
    if (nearbyCount != mapping.nearby.size() || oppositeCount != mapping.opposite.size()) {
        return false;
    }

    std::sort(mapping.nearby.begin(), mapping.nearby.end());
    std::sort(mapping.opposite.begin(), mapping.opposite.end());
    const auto self = std::find(mapping.nearby.begin(), mapping.nearby.end(), resources.selfIndex);
    if (self == mapping.nearby.end()) {
        return false;
    }
    mapping.lane = static_cast<uint32_t>(std::distance(mapping.nearby.begin(), self));
    return true;
}

HcclResult StartWorkerGroup(const std::vector<ThreadHandle> &threads, uint32_t activeCount)
{
    CHK_PRT_RET(activeCount == 0 || activeCount > threads.size(),
        HCCL_ERROR(
            "Invalid worker count[%u] for pool[%llu]", activeCount, static_cast<unsigned long long>(threads.size())),
        HCCL_E_INTERNAL);
    for (uint32_t worker = 1; worker < activeCount; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[worker], 0)));
    }
    for (uint32_t worker = 1; worker < activeCount; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[worker], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult FinishWorkerGroup(const std::vector<ThreadHandle> &threads, uint32_t activeCount)
{
    CHK_PRT_RET(activeCount == 0 || activeCount > threads.size(),
        HCCL_ERROR(
            "Invalid worker count[%u] for pool[%llu]", activeCount, static_cast<unsigned long long>(threads.size())),
        HCCL_E_INTERNAL);
    for (uint32_t worker = 1; worker < activeCount; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], worker - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t worker = 1; worker < activeCount; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[worker], threads[0], worker - 1)));
    }
    return HCCL_SUCCESS;
}

uint32_t PackedContributorSlot(uint32_t contributor, uint32_t owner)
{
    return contributor < owner ? contributor : contributor - 1;
}

OwnerInboxPlan BuildOwnerInboxPlan(uint64_t sliceBytes)
{
    OwnerInboxPlan plan;
    plan.slotStride = sliceBytes;
    plan.requiredBytes = static_cast<uint64_t>(yga_reduce_scatter::PEER_COUNT) * sliceBytes;
    return plan;
}

bool CanUseOwnerTree512K(
    const AlgResourceCtx &resources, uint64_t sliceBytes, ServerLaneMap &mapping, OwnerInboxPlan &plan)
{
    if (sliceBytes != yga_reduce_scatter::HALF_MEG_SLICE_BYTES || !DescribeServerLanes(resources, mapping)) {
        return false;
    }

    plan = BuildOwnerInboxPlan(sliceBytes);
    if (resources.workspace.address == nullptr || resources.workspace.bytes < plan.requiredBytes) {
        return false;
    }
    for (const PeerPath &path : resources.peerPaths) {
        if (path.remoteWorkspace.address == nullptr || path.remoteWorkspace.bytes < plan.requiredBytes) {
            return false;
        }
    }
    return true;
}

void *RemoteOwnerLeaf(const AlgResourceCtx &resources, const OwnerInboxPlan &plan, uint32_t contributor)
{
    if (contributor == resources.selfIndex || contributor >= yga_reduce_scatter::RANK_COUNT) {
        return nullptr;
    }
    const uint32_t slot = PackedContributorSlot(contributor, resources.selfIndex);
    return static_cast<uint8_t *>(resources.workspace.address) + static_cast<uint64_t>(slot) * plan.slotStride;
}

const void *OwnerLeaf(const OpParam &param, const AlgResourceCtx &resources, const OwnerInboxPlan &plan,
    uint32_t contributor, uint64_t sliceBytes)
{
    if (contributor == resources.selfIndex) {
        return static_cast<const uint8_t *>(param.inputPtr) + static_cast<uint64_t>(contributor) * sliceBytes;
    }
    return RemoteOwnerLeaf(resources, plan, contributor);
}

HcclResult QueueOwnerSlice(const OpParam &param, const AlgResourceCtx &resources, const PeerPath &path,
    const OwnerInboxPlan &plan, uint64_t sliceBytes, ThreadHandle thread)
{
    CHK_PRT_RET(path.remoteWorkspace.address == nullptr || path.remoteWorkspace.bytes < plan.requiredBytes,
        HCCL_ERROR("Remote owner inbox for rank[%u] is too small", path.rankIndex), HCCL_E_MEMORY);
    const uint32_t slot = PackedContributorSlot(resources.selfIndex, path.rankIndex);
    HcommBatchTransferDesc transfers[2] = {};
    transfers[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
    transfers[0].transferInfo.write.len = sliceBytes;
    transfers[0].transferInfo.write.dst
        = static_cast<uint8_t *>(path.remoteWorkspace.address) + static_cast<uint64_t>(slot) * plan.slotStride;
    transfers[0].transferInfo.write.src
        = static_cast<uint8_t *>(param.inputPtr) + static_cast<uint64_t>(path.rankIndex) * sliceBytes;
    transfers[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    transfers[1].transferInfo.notifyRecord.notifyIdx = PAYLOAD_READY_NOTIFY;
    return static_cast<HcclResult>(HcommBatchTransferOnThread(thread, path.channel, transfers, 2));
}

HcclResult RunOwnerFanout(const OpParam &param, const AlgResourceCtx &resources, const OwnerInboxPlan &plan,
    uint64_t sliceBytes)
{
    CHK_RET(StartWorkerGroup(resources.workerThreads, OWNER_FANOUT_WORKERS));
    for (uint32_t peer = 0; peer < resources.peerPaths.size(); ++peer) {
        CHK_RET(QueueOwnerSlice(
            param, resources, resources.peerPaths[peer], plan, sliceBytes, resources.workerThreads[peer]));
    }
    for (uint32_t peer = 0; peer < resources.peerPaths.size(); ++peer) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resources.workerThreads[peer],
            resources.peerPaths[peer].channel, PAYLOAD_READY_NOTIFY, CUSTOM_TIMEOUT)));
    }
    return FinishWorkerGroup(resources.workerThreads, OWNER_FANOUT_WORKERS);
}

struct LeafPairWork {
    void *accumulator = nullptr;
    const void *source = nullptr;
};

LeafPairWork BuildLeafPair(const OpParam &param, const AlgResourceCtx &resources, const OwnerInboxPlan &plan,
    uint32_t pairIndex, uint64_t sliceBytes)
{
    const uint32_t lowerContributor = pairIndex * 2U;
    const uint32_t upperContributor = lowerContributor + 1U;
    void *lowerSlot = RemoteOwnerLeaf(resources, plan, lowerContributor);
    void *upperSlot = RemoteOwnerLeaf(resources, plan, upperContributor);
    const void *lowerLeaf = OwnerLeaf(param, resources, plan, lowerContributor, sliceBytes);
    const void *upperLeaf = OwnerLeaf(param, resources, plan, upperContributor, sliceBytes);

    LeafPairWork work;
    if (lowerSlot != nullptr) {
        work.accumulator = lowerSlot;
        work.source = upperLeaf;
    } else {
        work.accumulator = upperSlot;
        work.source = lowerLeaf;
    }
    return work;
}

HcclResult JoinTreeBranch(
    const AlgResourceCtx &resources, uint32_t producer, uint32_t consumer, uint32_t notifyIndex)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resources.workerThreads[producer], resources.workerThreads[consumer], notifyIndex)));
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(resources.workerThreads[consumer], notifyIndex, CUSTOM_TIMEOUT));
}

HcclResult RunFixedOwnerTree(const OpParam &param, const AlgResourceCtx &resources, const OwnerInboxPlan &plan,
    uint64_t sliceBytes)
{
    const uint64_t elementCount = sliceBytes / yga_reduce_scatter::FP32_BYTES;
    std::array<void *, OWNER_TREE_WORKERS> roots{};
    std::array<const void *, OWNER_TREE_WORKERS> pairSources{};
    for (uint32_t pair = 0; pair < OWNER_TREE_WORKERS; ++pair) {
        const LeafPairWork work = BuildLeafPair(param, resources, plan, pair, sliceBytes);
        CHK_PRT_RET(work.accumulator == nullptr || work.source == nullptr,
            HCCL_ERROR("Invalid fixed-tree leaf pair[%u]", pair), HCCL_E_INTERNAL);
        roots[pair] = work.accumulator;
        pairSources[pair] = work.source;
    }

    CHK_RET(StartWorkerGroup(resources.workerThreads, OWNER_TREE_WORKERS));
    for (uint32_t pair = 0; pair < OWNER_TREE_WORKERS; ++pair) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resources.workerThreads[pair], roots[pair],
            pairSources[pair], elementCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }

    CHK_RET(JoinTreeBranch(resources, 1, 0, TREE_LEVEL_ONE_NOTIFY));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resources.workerThreads[0], roots[0], roots[1],
        elementCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(JoinTreeBranch(resources, 3, 2, TREE_LEVEL_ONE_NOTIFY));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resources.workerThreads[2], roots[2], roots[3],
        elementCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(JoinTreeBranch(resources, 5, 4, TREE_LEVEL_ONE_NOTIFY));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resources.workerThreads[4], roots[4], roots[5],
        elementCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(JoinTreeBranch(resources, 7, 6, TREE_LEVEL_ONE_NOTIFY));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resources.workerThreads[6], roots[6], roots[7],
        elementCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));

    CHK_RET(JoinTreeBranch(resources, 2, 0, TREE_LEVEL_TWO_NOTIFY));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resources.workerThreads[0], roots[0], roots[2],
        elementCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    CHK_RET(JoinTreeBranch(resources, 6, 4, TREE_LEVEL_TWO_NOTIFY));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resources.workerThreads[4], roots[4], roots[6],
        elementCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));

    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(resources.workerThreads[0], param.outputPtr, roots[0], sliceBytes)));
    CHK_RET(JoinTreeBranch(resources, 4, 0, TREE_LEVEL_THREE_NOTIFY));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resources.workerThreads[0], param.outputPtr, roots[4],
        elementCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return FinishWorkerGroup(resources.workerThreads, OWNER_TREE_WORKERS);
}

HcclResult RunBarrierPair(const AlgResourceCtx &resources, const PeerPath &partner, ThreadHandle controlThread)
{
    if (resources.selfIndex < partner.rankIndex) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(controlThread, partner.channel, WORKSPACE_FREE_NOTIFY)));
        return static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            controlThread, partner.channel, WORKSPACE_FREE_NOTIFY, CUSTOM_TIMEOUT));
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        controlThread, partner.channel, WORKSPACE_FREE_NOTIFY, CUSTOM_TIMEOUT)));
    return static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(controlThread, partner.channel, WORKSPACE_FREE_NOTIFY));
}

HcclResult RunConsumeBarrier(
    const AlgResourceCtx &resources, const ServerLaneMap &mapping, ThreadHandle controlThread)
{
    for (uint32_t laneXor : LOCAL_BARRIER_LANE_XORS) {
        const uint32_t partnerLane = mapping.lane ^ laneXor;
        const PeerPath *partner = DirectPeer(resources, mapping.nearby[partnerLane]);
        CHK_PRT_RET(partner == nullptr || partner->fabricLayer != INNER_SERVER_LAYER,
            HCCL_ERROR("Missing local consume-barrier lane[%u]", partnerLane), HCCL_E_NOT_FOUND);
        CHK_RET(RunBarrierPair(resources, *partner, controlThread));
    }

    const PeerPath *crossPartner = DirectPeer(resources, mapping.opposite[mapping.lane]);
    CHK_PRT_RET(crossPartner == nullptr || crossPartner->fabricLayer != BETWEEN_SERVER_LAYER,
        HCCL_ERROR("Missing cross-server consume-barrier lane[%u]", mapping.lane), HCCL_E_NOT_FOUND);
    return RunBarrierPair(resources, *crossPartner, controlThread);
}

HcclResult ExecuteOwnerTree512K(const OpParam &param, const AlgResourceCtx &resources, const ServerLaneMap &mapping,
    const OwnerInboxPlan &plan, uint64_t sliceBytes)
{
    CHK_RET(RunOwnerFanout(param, resources, plan, sliceBytes));
    CHK_RET(RunFixedOwnerTree(param, resources, plan, sliceBytes));
    return RunConsumeBarrier(resources, mapping, resources.workerThreads[0]);
}

HcclResult CheckResourceLayout(const AlgResourceCtx &resources)
{
    CHK_PRT_RET(resources.selfIndex >= yga_reduce_scatter::RANK_COUNT,
        HCCL_ERROR("Invalid local rank index[%u]", resources.selfIndex), HCCL_E_INTERNAL);
    CHK_PRT_RET(resources.orderedRanks.size() != yga_reduce_scatter::RANK_COUNT,
        HCCL_ERROR(
            "Unexpected topology rank count[%llu]", static_cast<unsigned long long>(resources.orderedRanks.size())),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resources.workerThreads.size() != yga_reduce_scatter::PEER_COUNT,
        HCCL_ERROR(
            "Unexpected worker pool size[%llu]", static_cast<unsigned long long>(resources.workerThreads.size())),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resources.peerPaths.size() != yga_reduce_scatter::PEER_COUNT,
        HCCL_ERROR("Unexpected peer path count[%llu]", static_cast<unsigned long long>(resources.peerPaths.size())),
        HCCL_E_INTERNAL);
    CHK_PTR_NULL(resources.workspace.address);

    uint32_t expectedPeerIndex = 0;
    for (const PeerPath &path : resources.peerPaths) {
        if (expectedPeerIndex == resources.selfIndex) {
            ++expectedPeerIndex;
        }
        CHK_PRT_RET(path.rankIndex != expectedPeerIndex,
            HCCL_ERROR("Peer path order mismatch: got[%u], expected[%u]", path.rankIndex, expectedPeerIndex),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(path.notifyCount < 2, HCCL_ERROR("Peer rank[%u] has insufficient notify resources", path.rankId),
            HCCL_E_INTERNAL);
        CHK_PTR_NULL(path.remoteWorkspace.address);
        ++expectedPeerIndex;
    }
    return HCCL_SUCCESS;
}

WorkerRange SplitElements(uint64_t elementCount, uint32_t worker, uint32_t workerCount)
{
    const uint64_t common = elementCount / workerCount;
    const uint64_t remainder = elementCount % workerCount;
    WorkerRange result;
    result.elementCount = common + static_cast<uint64_t>(worker < remainder);
    result.elementOffset = static_cast<uint64_t>(worker) * common + std::min<uint64_t>(worker, remainder);
    return result;
}

uint64_t SmallestWorkspace(const AlgResourceCtx &resources)
{
    uint64_t minimum = resources.workspace.bytes;
    for (const PeerPath &path : resources.peerPaths) {
        minimum = std::min(minimum, path.remoteWorkspace.bytes);
    }
    return minimum;
}

uint64_t SelectTileSize(uint64_t outputBytes, const AlgResourceCtx &resources)
{
    const uint64_t workspaceBytes = SmallestWorkspace(resources);
    const uint64_t slotCapacity = yga_reduce_scatter::AlignDown(
        workspaceBytes / yga_reduce_scatter::PEER_COUNT, yga_reduce_scatter::DMA_ALIGNMENT);
    if (slotCapacity == 0) {
        return 0;
    }
    if (yga_reduce_scatter::AlignUp(outputBytes, yga_reduce_scatter::DMA_ALIGNMENT) <= slotCapacity) {
        return outputBytes;
    }

    uint64_t tileBytes = std::min(yga_reduce_scatter::TARGET_TILE_BYTES, slotCapacity);
    const uint64_t tailBytes = outputBytes % tileBytes;
    const uint64_t initialRoundCount = yga_reduce_scatter::DivideRoundUp(outputBytes, tileBytes);
    if (tailBytes != 0 && tailBytes < yga_reduce_scatter::SMALL_TAIL_BYTES && initialRoundCount > 1) {
        const uint64_t balancedTile = yga_reduce_scatter::AlignUp(
            yga_reduce_scatter::DivideRoundUp(outputBytes, initialRoundCount - 1), yga_reduce_scatter::DMA_ALIGNMENT);
        if (balancedTile <= slotCapacity) {
            tileBytes = balancedTile;
        }
    }
    return tileBytes;
}

TileView GetTile(uint64_t outputBytes, uint64_t tileBytes, uint64_t round)
{
    TileView tile;
    tile.offsetBytes = round * tileBytes;
    if (tile.offsetBytes < outputBytes) {
        tile.validBytes = std::min(tileBytes, outputBytes - tile.offsetBytes);
    }
    return tile;
}

HcclResult VerifySlotCapacity(uint64_t workspaceBytes, uint64_t slotStride)
{
    CHK_PRT_RET(slotStride == 0 || slotStride > workspaceBytes / yga_reduce_scatter::PEER_COUNT,
        HCCL_ERROR("Workspace[%llu] cannot hold 15 slots of stride[%llu]",
            static_cast<unsigned long long>(workspaceBytes), static_cast<unsigned long long>(slotStride)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult DistributeOwnerSlices(
    const OpParam &param, const AlgResourceCtx &resources, const TileView &tile, uint64_t outputBytes)
{
    const uint64_t slotStride = yga_reduce_scatter::AlignUp(tile.validBytes, yga_reduce_scatter::DMA_ALIGNMENT);
    CHK_RET(StartWorkerGroup(resources.workerThreads, yga_reduce_scatter::PEER_COUNT));
    for (uint32_t i = 0; i < resources.peerPaths.size(); ++i) {
        const PeerPath &path = resources.peerPaths[i];
        CHK_RET(VerifySlotCapacity(path.remoteWorkspace.bytes, slotStride));
        const uint32_t slot = PackedContributorSlot(resources.selfIndex, path.rankIndex);
        void *remoteSlot
            = static_cast<uint8_t *>(path.remoteWorkspace.address) + static_cast<uint64_t>(slot) * slotStride;
        const void *ownerSlice = static_cast<const uint8_t *>(param.inputPtr)
                                 + static_cast<uint64_t>(path.rankIndex) * outputBytes + tile.offsetBytes;
        const ThreadHandle thread = resources.workerThreads[i];
        CHK_RET(
            static_cast<HcclResult>(HcommWriteOnThread(thread, path.channel, remoteSlot, ownerSlice, tile.validBytes)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, path.channel, PAYLOAD_READY_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, path.channel, PAYLOAD_READY_NOTIFY, CUSTOM_TIMEOUT)));
    }
    CHK_RET(FinishWorkerGroup(resources.workerThreads, yga_reduce_scatter::PEER_COUNT));
    return HCCL_SUCCESS;
}

HcclResult ReduceTileInRankOrder(const OpParam &param, const AlgResourceCtx &resources, const TileView &tile,
    uint64_t outputBytes, uint32_t workerCount)
{
    const uint64_t slotStride = yga_reduce_scatter::AlignUp(tile.validBytes, yga_reduce_scatter::DMA_ALIGNMENT);
    CHK_RET(VerifySlotCapacity(resources.workspace.bytes, slotStride));
    const uint64_t tileElements = tile.validBytes / yga_reduce_scatter::FP32_BYTES;
    uint8_t *workspace = static_cast<uint8_t *>(resources.workspace.address);
    const uint8_t *localSlice = static_cast<const uint8_t *>(param.inputPtr)
                                + static_cast<uint64_t>(resources.selfIndex) * outputBytes + tile.offsetBytes;
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr) + tile.offsetBytes;

    CHK_RET(StartWorkerGroup(resources.workerThreads, workerCount));
    for (uint32_t worker = 0; worker < workerCount; ++worker) {
        const WorkerRange range = SplitElements(tileElements, worker, workerCount);
        if (range.elementCount == 0) {
            continue;
        }
        const uint64_t byteOffset = range.elementOffset * yga_reduce_scatter::FP32_BYTES;
        const uint64_t rangeBytes = range.elementCount * yga_reduce_scatter::FP32_BYTES;
        const ThreadHandle thread = resources.workerThreads[worker];
        void *accumulator = nullptr;

        if (resources.selfIndex == 0) {
            accumulator = output + byteOffset;
            const void *rankZeroInput = localSlice + byteOffset;
            if (accumulator != rankZeroInput) {
                CHK_RET(
                    static_cast<HcclResult>(HcommLocalCopyOnThread(thread, accumulator, rankZeroInput, rangeBytes)));
            }
        } else {
            accumulator = workspace + byteOffset;
        }

        for (uint32_t contributor = 1; contributor < yga_reduce_scatter::RANK_COUNT; ++contributor) {
            const void *source = nullptr;
            if (contributor == resources.selfIndex) {
                source = localSlice + byteOffset;
            } else {
                const uint32_t slot = PackedContributorSlot(contributor, resources.selfIndex);
                source = workspace + static_cast<uint64_t>(slot) * slotStride + byteOffset;
            }
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                thread, accumulator, source, range.elementCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }

        if (resources.selfIndex != 0) {
            CHK_RET(
                static_cast<HcclResult>(HcommLocalCopyOnThread(thread, output + byteOffset, accumulator, rangeBytes)));
        }
    }
    CHK_RET(FinishWorkerGroup(resources.workerThreads, workerCount));
    return HCCL_SUCCESS;
}

HcclResult ReleaseRemoteSlots(const AlgResourceCtx &resources)
{
    CHK_RET(StartWorkerGroup(resources.workerThreads, yga_reduce_scatter::PEER_COUNT));
    for (uint32_t i = 0; i < resources.peerPaths.size(); ++i) {
        const PeerPath &path = resources.peerPaths[i];
        const ThreadHandle thread = resources.workerThreads[i];
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, path.channel, WORKSPACE_FREE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, path.channel, WORKSPACE_FREE_NOTIFY, CUSTOM_TIMEOUT)));
    }
    CHK_RET(FinishWorkerGroup(resources.workerThreads, yga_reduce_scatter::PEER_COUNT));
    return HCCL_SUCCESS;
}

HcclResult ExecuteDirectReduceScatter(const OpParam &param, const AlgResourceCtx &resources, uint64_t outputBytes)
{
    const uint64_t tileBytes = SelectTileSize(outputBytes, resources);
    CHK_PRT_RET(tileBytes == 0,
        HCCL_ERROR("No feasible tile for output size[%llu]", static_cast<unsigned long long>(outputBytes)),
        HCCL_E_INTERNAL);
    const uint64_t roundCount = yga_reduce_scatter::DivideRoundUp(outputBytes, tileBytes);
    const uint32_t reduceWorkers = outputBytes <= yga_reduce_scatter::TARGET_TILE_BYTES ? 1U : LARGE_REDUCE_WORKERS;

    for (uint64_t round = 0; round < roundCount; ++round) {
        const TileView tile = GetTile(outputBytes, tileBytes, round);
        CHK_RET(DistributeOwnerSlices(param, resources, tile, outputBytes));
        CHK_RET(ReduceTileInRankOrder(param, resources, tile, outputBytes, reduceWorkers));
        CHK_RET(ReleaseRemoteSlots(resources));
    }
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("ReduceScatter supports FP32 SUM only"), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / yga_reduce_scatter::FP32_BYTES,
        HCCL_ERROR("Output size calculation overflow"), HCCL_E_PARA);
    CHK_RET(CheckResourceLayout(resCtx));

    const uint64_t outputBytes = param.count * yga_reduce_scatter::FP32_BYTES;
    ServerLaneMap laneMapping;
    OwnerInboxPlan inboxPlan;
    if (CanUseOwnerTree512K(resCtx, outputBytes, laneMapping, inboxPlan)) {
        return ExecuteOwnerTree512K(param, resCtx, laneMapping, inboxPlan, outputBytes);
    }
    return ExecuteDirectReduceScatter(param, resCtx, outputBytes);
}
} // namespace ops_hccl
