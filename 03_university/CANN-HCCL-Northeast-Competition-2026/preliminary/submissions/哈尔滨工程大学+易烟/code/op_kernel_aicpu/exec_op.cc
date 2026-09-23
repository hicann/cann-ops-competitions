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

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
struct TilePlan {
    uint64_t bytes = 0;
    uint64_t count = 0;
};

HcclResult GetRecvBytes(const OpParam &param, uint64_t &recvBytes)
{
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize || param.root >= param.rankSize,
        HCCL_ERROR("Invalid Scatter rank information"), HCCL_E_PARA);
    const auto type = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(type == SIZE_TABLE.end(), HCCL_ERROR("Unsupported Scatter data type"), HCCL_E_NOT_SUPPORT);
    const uint64_t typeBytes = type->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeBytes / param.rankSize,
        HCCL_ERROR("Scatter input size overflows uint64_t"), HCCL_E_PARA);
    recvBytes = param.count * typeBytes;
    return HCCL_SUCCESS;
}

HcclResult GetBufferCapacity(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t &capacity)
{
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    CHK_PRT_RET(resCtx.threads.size() != param.rankSize || resCtx.channels.size() != param.rankSize,
        HCCL_ERROR("Scatter resource context does not match rankSize"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.aicpuThread != resCtx.threads[0],
        HCCL_ERROR("Scatter main thread does not match the kernel synchronization thread"), HCCL_E_INTERNAL);

    // 每个 rank 都持有全体 peer 的资源，因而能独立算出相同的最小容量。
    // 不假定 HCCL Buffer 恒为 400MB，也不假定各 rank 的容量相等。
    capacity = resCtx.localBuffer.size;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        const ChannelInfo &channel = resCtx.channels[peer];
        CHK_PRT_RET(channel.remoteRank != peer || channel.notifyNum < SCATTER_CHANNEL_NOTIFY_NUM,
            HCCL_ERROR("Invalid Scatter channel for rank %u", peer), HCCL_E_INTERNAL);
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        capacity = std::min(capacity, channel.remoteCclMem.size);
    }
    CHK_PRT_RET(capacity < sizeof(float), HCCL_ERROR("Scatter communication buffer is too small"), HCCL_E_MEMORY);
    return HCCL_SUCCESS;
}

HcclResult MakeTilePlan(uint64_t recvBytes, uint32_t rankSize, uint64_t capacity, TilePlan &plan)
{
    const uint64_t slotCount = static_cast<uint64_t>(rankSize) * SCATTER_PIPELINE_DEPTH;
    uint64_t maxTileBytes = std::min(SCATTER_MAX_TILE_BYTES, capacity / slotCount);
    const uint64_t alignment = maxTileBytes >= SCATTER_TILE_ALIGNMENT ? SCATTER_TILE_ALIGNMENT : sizeof(float);
    maxTileBytes -= maxTileBytes % alignment;
    CHK_PRT_RET(maxTileBytes == 0, HCCL_ERROR("No space for Scatter pipeline slots"), HCCL_E_MEMORY);

    // 先定块数，再均衡块长，避免 400MB+4B 产生一个单独的 4B 流水尾块。
    // 只对中间块长对齐，最后一块按实际剩余字节搬运，不补读、不补写。
    const uint64_t targetTiles = 1 + (recvBytes - 1) / maxTileBytes;
    const uint64_t balancedBytes = 1 + (recvBytes - 1) / targetTiles;
    plan.bytes = ((balancedBytes + alignment - 1) / alignment) * alignment;
    plan.count = 1 + (recvBytes - 1) / plan.bytes;
    return HCCL_SUCCESS;
}

uint32_t WorkerIndex(uint32_t myRank, uint32_t peer)
{
    // threads[0] 为主线程，其余线程按真实 peer 顺序排列，跳过本 rank。
    return peer < myRank ? peer + 1 : peer;
}

uint32_t ReadyNotify(uint64_t tile)
{
    return SCATTER_READY_NOTIFY_BASE + static_cast<uint32_t>(tile % SCATTER_PIPELINE_DEPTH);
}

uint32_t AckNotify(uint64_t tile)
{
    return SCATTER_ACK_NOTIFY_BASE + static_cast<uint32_t>(tile % SCATTER_PIPELINE_DEPTH);
}

uint64_t RootSlotOffset(uint32_t destination, uint64_t tile, const TilePlan &plan)
{
    const uint64_t slot = static_cast<uint64_t>(destination) * SCATTER_PIPELINE_DEPTH
        + tile % SCATTER_PIPELINE_DEPTH;
    return slot * plan.bytes;
}

HcclResult CopyRootResult(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    const char *source = static_cast<const char *>(param.inputPtr) + static_cast<uint64_t>(param.root) * recvBytes;
    char *destination = static_cast<char *>(param.outputPtr);
    if (source == destination) {
        return HCCL_SUCCESS;
    }

    const uintptr_t sourceAddress = reinterpret_cast<uintptr_t>(source);
    const uintptr_t destinationAddress = reinterpret_cast<uintptr_t>(destination);
    const bool backwards = destinationAddress > sourceAddress;
    const uint64_t distance = backwards ? destinationAddress - sourceAddress : sourceAddress - destinationAddress;
    const bool overlaps = distance < recvBytes;
    uint64_t copyBytes = SCATTER_MAX_TILE_BYTES;
    if (overlaps) {
        copyBytes = std::min(copyBytes, resCtx.localBuffer.size);
        copyBytes -= copyBytes % sizeof(float);
        CHK_PRT_RET(copyBytes == 0, HCCL_ERROR("No scratch space for overlapping Scatter output"), HCCL_E_MEMORY);
    }

    // 必须在所有 peer 读完后调用。root 的 recvBuf 可以等于 sendBuf，不能提前覆盖其他分片。
    // 对自身分片的部分重叠，用已释放的 CCL 槽和 memmove 方向处理，不依赖 LocalCopy 的重叠语义。
    uint64_t remaining = recvBytes;
    while (remaining != 0) {
        const uint64_t bytes = std::min(copyBytes, remaining);
        const uint64_t offset = overlaps && backwards ? remaining - bytes : recvBytes - remaining;
        if (overlaps) {
            CHK_RET(HcommLocalCopyOnThread(resCtx.aicpuThread, resCtx.localBuffer.addr, source + offset, bytes));
            CHK_RET(HcommLocalCopyOnThread(resCtx.aicpuThread, destination + offset, resCtx.localBuffer.addr, bytes));
        } else {
            CHK_RET(HcommLocalCopyOnThread(resCtx.aicpuThread, destination + offset, source + offset, bytes));
        }
        remaining -= bytes;
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterSmall(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    if (param.myRank == param.root) {
        // 一次准备全部小消息，随后各 peer 在自己的 TS 队列上并行读取各自的分片。
        // 先 snapshot，允许 root 的输出与输入重叠。
        CHK_RET(HcommLocalCopyOnThread(mainThread, resCtx.localBuffer.addr, param.inputPtr,
            recvBytes * param.rankSize));
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer != param.root) {
                CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, resCtx.channels[peer].handle, ReadyNotify(0)));
            }
        }
        const char *rootSlice = static_cast<const char *>(resCtx.localBuffer.addr)
            + static_cast<uint64_t>(param.root) * recvBytes;
        CHK_RET(HcommLocalCopyOnThread(mainThread, param.outputPtr, rootSlice, recvBytes));
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer != param.root) {
                CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, resCtx.channels[peer].handle, AckNotify(0),
                    CUSTOM_TIMEOUT));
            }
        }
    } else {
        const ChannelInfo &rootChannel = resCtx.channels[param.root];
        const char *source = static_cast<const char *>(rootChannel.remoteCclMem.addr)
            + static_cast<uint64_t>(param.myRank) * recvBytes;
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, rootChannel.handle, ReadyNotify(0), CUSTOM_TIMEOUT));
        // 最终接收直接落入本次用户输出，省去 CCL 到输出的二次搬运。
        // 远端源仍为已交换的 root CCL；同一 TS 上 Read 完成后再 ACK。
        CHK_RET(HcommReadOnThread(mainThread, rootChannel.handle, param.outputPtr, source, recvBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel.handle, AckNotify(0)));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterSmallTopologyTree(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    const uint32_t relativeRank = param.myRank ^ param.root;
    const uint32_t subtreeBit = relativeRank == 0 ? SCATTER_TOPO_RANKS
        : relativeRank & (~relativeRank + 1U);
    const uint32_t parent = relativeRank == 0 ? param.root : param.myRank ^ subtreeBit;

    if (relativeRank == 0) {
        CHK_RET(HcommLocalCopyOnThread(mainThread, resCtx.localBuffer.addr, param.inputPtr,
            recvBytes * param.rankSize));
    } else {
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, resCtx.channels[parent].handle,
            ReadyNotify(0), CUSTOM_TIMEOUT));
    }

    // root 先放行跨组代表，再放行同组子树。其余边都在各自八卡组内。
    // 先发布全部子 READY，不能逐个发布后立即等待其 ACK。
    for (uint32_t mask = SCATTER_MESH_RANKS; mask != 0; mask >>= 1) {
        if (mask < subtreeBit) {
            const uint32_t child = param.myRank ^ mask;
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, resCtx.channels[child].handle, ReadyNotify(0)));
        }
    }

    if (relativeRank == 0) {
        const char *rootSlice = static_cast<const char *>(resCtx.localBuffer.addr)
            + static_cast<uint64_t>(param.root) * recvBytes;
        CHK_RET(HcommLocalCopyOnThread(mainThread, param.outputPtr, rootSlice, recvBytes));
    } else {
        // 控制来自 parent，数据始终来自实际 root，不能从 parent 的 CCL 读数据。
        const ChannelInfo &rootChannel = resCtx.channels[param.root];
        const char *source = static_cast<const char *>(rootChannel.remoteCclMem.addr)
            + static_cast<uint64_t>(param.myRank) * recvBytes;
        CHK_RET(HcommReadOnThread(mainThread, rootChannel.handle, param.outputPtr, source, recvBytes));
    }

    // 子树 ACK 同时证明其全部后代和自身已经读完 root snapshot。
    for (uint32_t mask = 1; mask < subtreeBit; mask <<= 1) {
        const uint32_t child = param.myRank ^ mask;
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, resCtx.channels[child].handle,
            AckNotify(0), CUSTOM_TIMEOUT));
    }
    if (relativeRank != 0) {
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, resCtx.channels[parent].handle, AckNotify(0)));
    }
    return HCCL_SUCCESS;
}

HcclResult PublishTiles(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t peer,
    uint64_t recvBytes, const TilePlan &plan)
{
    const ThreadHandle worker = resCtx.threads[WorkerIndex(param.myRank, peer)];
    const ChannelHandle channel = resCtx.channels[peer].handle;
    const char *source = static_cast<const char *>(param.inputPtr) + static_cast<uint64_t>(peer) * recvBytes;
    char *scratch = static_cast<char *>(resCtx.localBuffer.addr);

    for (uint64_t tile = 0; tile < plan.count; ++tile) {
        if (tile >= SCATTER_PIPELINE_DEPTH) {
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel, AckNotify(tile), CUSTOM_TIMEOUT));
        }
        const uint64_t offset = tile * plan.bytes;
        const uint64_t bytes = std::min(plan.bytes, recvBytes - offset);
        CHK_RET(HcommLocalCopyOnThread(worker, scratch + RootSlotOffset(peer, tile, plan), source + offset, bytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel, ReadyNotify(tile)));
    }

    // 最后一轮也必须回收 ACK，保证 CCL 和 Notify 可供下一次调用安全复用。
    // 单块只等待一个 ACK；奇数块时从最后实际在途的槽开始，不多等、不漏等。
    const uint64_t firstPending = plan.count > SCATTER_PIPELINE_DEPTH ? plan.count - SCATTER_PIPELINE_DEPTH : 0;
    for (uint64_t tile = firstPending; tile < plan.count; ++tile) {
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel, AckNotify(tile), CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult ReceiveTiles(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes, const TilePlan &plan)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    const ChannelInfo &rootChannel = resCtx.channels[param.root];
    const char *source = static_cast<const char *>(rootChannel.remoteCclMem.addr);
    char *destination = static_cast<char *>(param.outputPtr);

    for (uint64_t tile = 0; tile < plan.count; ++tile) {
        const uint64_t offset = tile * plan.bytes;
        const uint64_t bytes = std::min(plan.bytes, recvBytes - offset);
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, rootChannel.handle, ReadyNotify(tile), CUSTOM_TIMEOUT));
        // 远端源槽和 ACK 协议不变，最终数据直接写入本次输出的对应片段。
        CHK_RET(HcommReadOnThread(mainThread, rootChannel.handle, destination + offset,
            source + RootSlotOffset(param.myRank, tile, plan), bytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel.handle, AckNotify(tile)));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterPipelined(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes, uint64_t capacity)
{
    TilePlan plan;
    CHK_RET(MakeTilePlan(recvBytes, param.rankSize, capacity, plan));
    if (param.myRank != param.root) {
        return ReceiveTiles(param, resCtx, recvBytes, plan);
    }

    const ThreadHandle mainThread = resCtx.aicpuThread;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.root) {
            continue;
        }
        const ThreadHandle worker = resCtx.threads[WorkerIndex(param.myRank, peer)];
        // Checker 仅自动启动主线程。每个活动 worker 的第一个任务都是本地 WAIT。
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, SCATTER_WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, SCATTER_WORKER_START_NOTIFY));
    }

    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.root) {
            continue;
        }
        CHK_RET(PublishTiles(param, resCtx, peer, recvBytes, plan));
        const uint32_t workerIndex = WorkerIndex(param.myRank, peer);
        // DONE 使用主线程独立的通知位 1..rankSize-1，不能与 Host 的 notify[0] 混用。
        CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.threads[workerIndex], mainThread, workerIndex));
    }
    for (uint32_t workerIndex = 1; workerIndex < param.rankSize; ++workerIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, workerIndex, CUSTOM_TIMEOUT));
    }
    return CopyRootResult(param, resCtx, recvBytes);
}

struct MeshClosPlan {
    TilePlan tiles{};
    uint64_t slotBytes = 0;
};

uint64_t RelayBytes(uint64_t tileBytes)
{
    // 按 FP32 元素切分尾块。普通块按 512B 对齐，3/8 和 5/8 均为 64B 的倍数。
    return (tileBytes / sizeof(float) * SCATTER_RELAY_PARTS / SCATTER_SPLIT_PARTS) * sizeof(float);
}

HcclResult MakeMeshClosPlan(uint64_t recvBytes, uint64_t capacity, MeshClosPlan &plan)
{
    // root 为每个真实 rank 保留两个 [转发前缀 | 自身分片] 槽。
    // 16 * 2 * (1 + 3/8) = 44；8MiB 逻辑块最多使用 352MiB CCL。
    constexpr uint64_t capacityFactor = SCATTER_TOPO_RANKS * SCATTER_PIPELINE_DEPTH
        / SCATTER_SPLIT_PARTS * (SCATTER_SPLIT_PARTS + SCATTER_RELAY_PARTS);
    uint64_t maxTileBytes = std::min(SCATTER_MAX_TILE_BYTES, capacity / capacityFactor);
    maxTileBytes -= maxTileBytes % SCATTER_TOPO_ALIGNMENT;
    CHK_PRT_RET(maxTileBytes == 0, HCCL_ERROR("No space for Mesh/Clos Scatter slots"), HCCL_E_MEMORY);

    const uint64_t targetTiles = 1 + (recvBytes - 1) / maxTileBytes;
    const uint64_t balancedBytes = 1 + (recvBytes - 1) / targetTiles;
    plan.tiles.bytes = ((balancedBytes + SCATTER_TOPO_ALIGNMENT - 1) / SCATTER_TOPO_ALIGNMENT)
        * SCATTER_TOPO_ALIGNMENT;
    plan.tiles.count = 1 + (recvBytes - 1) / plan.tiles.bytes;
    plan.slotBytes = plan.tiles.bytes + RelayBytes(plan.tiles.bytes);
    return HCCL_SUCCESS;
}

uint64_t MeshClosTileBytes(uint64_t recvBytes, uint64_t tile, const MeshClosPlan &plan)
{
    return std::min(plan.tiles.bytes, recvBytes - tile * plan.tiles.bytes);
}

uint64_t MeshClosRootSlot(uint32_t peer, uint64_t tile, const MeshClosPlan &plan)
{
    return (static_cast<uint64_t>(peer) * SCATTER_PIPELINE_DEPTH + tile % SCATTER_PIPELINE_DEPTH)
        * plan.slotBytes;
}

uint32_t OppositeMeshRank(uint32_t rank)
{
    // 两组采用固定的逻辑 rank 编号配对，所有 rank 都能从算子参数推导出相同角色。
    // 全 peer Channel 沿用 V1；即使物理编号不同，源片和目的片仍按真实 rank 寻址。
    return rank < SCATTER_MESH_RANKS ? rank + SCATTER_MESH_RANKS : rank - SCATTER_MESH_RANKS;
}

HcclResult PublishMeshClosTiles(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t peer, uint32_t pairedRank, bool toMeshHelper, uint64_t recvBytes, const MeshClosPlan &plan)
{
    const ThreadHandle worker = resCtx.threads[WorkerIndex(param.myRank, peer)];
    const ChannelHandle channel = resCtx.channels[peer].handle;
    const char *input = static_cast<const char *>(param.inputPtr);
    const char *peerInput = input + static_cast<uint64_t>(peer) * recvBytes;
    char *scratch = static_cast<char *>(resCtx.localBuffer.addr);

    for (uint64_t tile = 0; tile < plan.tiles.count; ++tile) {
        if (tile >= SCATTER_PIPELINE_DEPTH) {
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel, AckNotify(tile), CUSTOM_TIMEOUT));
        }
        const uint64_t offset = tile * plan.tiles.bytes;
        const uint64_t bytes = MeshClosTileBytes(recvBytes, tile, plan);
        char *slot = scratch + MeshClosRootSlot(peer, tile, plan);
        if (toMeshHelper) {
            const uint64_t prefixBytes = RelayBytes(bytes);
            if (prefixBytes != 0) {
                const char *relayInput = input + static_cast<uint64_t>(pairedRank) * recvBytes + offset;
                CHK_RET(HcommLocalCopyOnThread(worker, slot, relayInput, prefixBytes));
            }
            // 尾块紧凑打包，不让接收端读取两个有效区域之间的未初始化 padding。
            CHK_RET(HcommLocalCopyOnThread(worker, slot + prefixBytes, peerInput + offset, bytes));
        } else {
            // root 对应位置的远端没有 helper，整块直达；其他远端只从 root 取后 5/8。
            const uint64_t prefixBytes = pairedRank == param.root ? 0 : RelayBytes(bytes);
            CHK_RET(HcommLocalCopyOnThread(worker, slot, peerInput + offset + prefixBytes,
                bytes - prefixBytes));
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel, ReadyNotify(tile)));
    }

    const uint64_t firstPending = plan.tiles.count > SCATTER_PIPELINE_DEPTH
        ? plan.tiles.count - SCATTER_PIPELINE_DEPTH : 0;
    for (uint64_t tile = firstPending; tile < plan.tiles.count; ++tile) {
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel, AckNotify(tile), CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult ReceiveAndRelayMeshTiles(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t remotePeer, uint64_t recvBytes, const MeshClosPlan &plan)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    const ChannelInfo &rootChannel = resCtx.channels[param.root];
    const ChannelHandle remoteChannel = resCtx.channels[remotePeer].handle;
    const char *rootScratch = static_cast<const char *>(rootChannel.remoteCclMem.addr);
    char *localScratch = static_cast<char *>(resCtx.localBuffer.addr);
    char *output = static_cast<char *>(param.outputPtr);
    bool relayPending[SCATTER_PIPELINE_DEPTH] = {};

    for (uint64_t tile = 0; tile < plan.tiles.count; ++tile) {
        const uint32_t slotIndex = static_cast<uint32_t>(tile % SCATTER_PIPELINE_DEPTH);
        // 覆盖本地槽前，先确认远端已取走旧的转发前缀；不以 root ACK 代替此依赖。
        if (relayPending[slotIndex]) {
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, remoteChannel, AckNotify(tile), CUSTOM_TIMEOUT));
            relayPending[slotIndex] = false;
        }
        const uint64_t offset = tile * plan.tiles.bytes;
        const uint64_t bytes = MeshClosTileBytes(recvBytes, tile, plan);
        const uint64_t prefixBytes = RelayBytes(bytes);
        const char *source = rootScratch + MeshClosRootSlot(param.myRank, tile, plan);
        char *slot = localScratch + slotIndex * plan.slotBytes;
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, rootChannel.handle, ReadyNotify(tile), CUSTOM_TIMEOUT));
        if (prefixBytes != 0) {
            CHK_RET(HcommReadOnThread(mainThread, rootChannel.handle, slot, source, prefixBytes));
            // 前缀 Read 的设备任务完成后才发布；远端转发可与下面自身分片的读取并行。
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, remoteChannel, ReadyNotify(tile)));
            relayPending[slotIndex] = true;
        }
        // 自身分片直接输出；上面的转发前缀仍留在本地 CCL，供 partner 读取。
        CHK_RET(HcommReadOnThread(mainThread, rootChannel.handle, output + offset,
            source + prefixBytes, bytes));
        // 两段 Read 完成后可释放 root 源槽；本地前缀槽仍单独等待 partner ACK。
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel.handle, AckNotify(tile)));
    }

    // 短尾块可能没有转发前缀，只排空真正发布过的槽。
    for (uint32_t slot = 0; slot < SCATTER_PIPELINE_DEPTH; ++slot) {
        if (relayPending[slot]) {
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, remoteChannel, AckNotify(slot), CUSTOM_TIMEOUT));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReceiveMeshClosDirect(const OpParam &param, const AlgResourceCtx &resCtx,
    bool hasHelper, uint64_t recvBytes, const MeshClosPlan &plan)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    const ChannelInfo &rootChannel = resCtx.channels[param.root];
    const char *source = static_cast<const char *>(rootChannel.remoteCclMem.addr);
    char *output = static_cast<char *>(param.outputPtr);

    for (uint64_t tile = 0; tile < plan.tiles.count; ++tile) {
        const uint64_t offset = tile * plan.tiles.bytes;
        const uint64_t bytes = MeshClosTileBytes(recvBytes, tile, plan);
        const uint64_t prefixBytes = hasHelper ? RelayBytes(bytes) : 0;
        const uint64_t directBytes = bytes - prefixBytes;
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, rootChannel.handle, ReadyNotify(tile), CUSTOM_TIMEOUT));
        // main 只写后缀；没有 helper 的远端则直接接收完整 tile。
        CHK_RET(HcommReadOnThread(mainThread, rootChannel.handle, output + offset + prefixBytes,
            source + MeshClosRootSlot(param.myRank, tile, plan), directBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel.handle, AckNotify(tile)));
    }
    return HCCL_SUCCESS;
}

HcclResult ReceiveMeshClosRelay(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t helperRank, uint64_t recvBytes, const MeshClosPlan &plan)
{
    const ThreadHandle worker = resCtx.threads[WorkerIndex(param.myRank, helperRank)];
    const ChannelInfo &helperChannel = resCtx.channels[helperRank];
    const char *source = static_cast<const char *>(helperChannel.remoteCclMem.addr);
    char *output = static_cast<char *>(param.outputPtr);

    for (uint64_t tile = 0; tile < plan.tiles.count; ++tile) {
        const uint64_t prefixBytes = RelayBytes(MeshClosTileBytes(recvBytes, tile, plan));
        if (prefixBytes == 0) {
            continue;
        }
        const uint64_t slotIndex = tile % SCATTER_PIPELINE_DEPTH;
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, helperChannel.handle, ReadyNotify(tile), CUSTOM_TIMEOUT));
        // worker 只写当前 tile 的前缀，与 main 的后缀区间互不相交。
        CHK_RET(HcommReadOnThread(worker, helperChannel.handle, output + tile * plan.tiles.bytes,
            source + slotIndex * plan.slotBytes, prefixBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, helperChannel.handle, AckNotify(tile)));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterMeshClos(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvBytes, uint64_t capacity)
{
    // 仅由 ExecOp 的 16-rank 分支进入。保持 Host 与资源布局和 V1 完全一致，
    // 不把物理 Full-Mesh 必须返回 COMM_TOPO_1DMESH 当作任何消息的启动条件。
    MeshClosPlan plan;
    CHK_RET(MakeMeshClosPlan(recvBytes, capacity, plan));
    const ThreadHandle mainThread = resCtx.aicpuThread;

    if (param.myRank == param.root) {
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.root) {
                continue;
            }
            const ThreadHandle worker = resCtx.threads[WorkerIndex(param.myRank, peer)];
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, SCATTER_WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
            CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, SCATTER_WORKER_START_NOTIFY));
        }
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.root) {
                continue;
            }
            const bool toMeshHelper = peer / SCATTER_MESH_RANKS == param.root / SCATTER_MESH_RANKS;
            CHK_RET(PublishMeshClosTiles(param, resCtx, peer, OppositeMeshRank(peer),
                toMeshHelper, recvBytes, plan));
            const uint32_t workerIndex = WorkerIndex(param.myRank, peer);
            CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.threads[workerIndex], mainThread, workerIndex));
        }
        for (uint32_t workerIndex = 1; workerIndex < param.rankSize; ++workerIndex) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, workerIndex, CUSTOM_TIMEOUT));
        }
        // 输入已全部被直达接收端或 helper 读走，保持 V1 的重叠输出处理。
        return CopyRootResult(param, resCtx, recvBytes);
    }

    if (param.myRank / SCATTER_MESH_RANKS == param.root / SCATTER_MESH_RANKS) {
        return ReceiveAndRelayMeshTiles(param, resCtx, OppositeMeshRank(param.myRank), recvBytes, plan);
    }

    const uint32_t helperRank = OppositeMeshRank(param.myRank);
    if (helperRank == param.root) {
        return ReceiveMeshClosDirect(param, resCtx, false, recvBytes, plan);
    }
    const uint32_t workerIndex = WorkerIndex(param.myRank, helperRank);
    const ThreadHandle worker = resCtx.threads[workerIndex];
    // 转发接收独立启动，不依赖直达路径完成，避免 helper 回收槽时形成等待环。
    CHK_RET(HcommThreadNotifyWaitOnThread(worker, SCATTER_WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, SCATTER_WORKER_START_NOTIFY));
    CHK_RET(ReceiveMeshClosRelay(param, resCtx, helperRank, recvBytes, plan));
    CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, workerIndex));
    CHK_RET(ReceiveMeshClosDirect(param, resCtx, true, recvBytes, plan));
    // 两个线程写入互不相交的前缀/后缀，汇合后才通知 Host 本 rank 的输出完成。
    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, workerIndex, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

struct TopologyPushPlan {
    uint64_t prefixBytes = 0;
    uint64_t prefixReserve = 0;
    uint64_t slotBytes = 0;
    TilePlan prefix{};
    TilePlan own{};
    TilePlan suffix{};
    bool resident = false;
};

TilePlan MakeSectionPlan(uint64_t bytes, uint64_t maxTileBytes, uint64_t alignment)
{
    TilePlan plan;
    if (bytes == 0) {
        return plan;
    }
    const uint64_t targetTiles = 1 + (bytes - 1) / maxTileBytes;
    const uint64_t balancedBytes = 1 + (bytes - 1) / targetTiles;
    plan.bytes = ((balancedBytes + alignment - 1) / alignment) * alignment;
    plan.count = 1 + (bytes - 1) / plan.bytes;
    return plan;
}

bool MakeTopologyPushPlan(uint64_t recvBytes, uint64_t capacity, TopologyPushPlan &plan)
{
    // V15 只将 Push 的整片连续前缀改为 4/11，旧 MeshClos 的 RelayBytes 保持 3/8。
    // 先在 FP32 元素上取整，再转字节；余数完整保留在后缀，不补读、不截掉尾部。
    // 商/余数分解避免直接把整个元素数乘以分子。
    const uint64_t elements = recvBytes / sizeof(float);
    const uint64_t prefixElements = (elements / SCATTER_PUSH_SPLIT_PARTS) * SCATTER_PUSH_RELAY_PARTS
        + (elements % SCATTER_PUSH_SPLIT_PARTS) * SCATTER_PUSH_RELAY_PARTS / SCATTER_PUSH_SPLIT_PARTS;
    plan.prefixBytes = prefixElements * sizeof(float);
    plan.prefixReserve = ((plan.prefixBytes + SCATTER_TILE_ALIGNMENT - 1) / SCATTER_TILE_ALIGNMENT)
        * SCATTER_TILE_ALIGNMENT;
    if (plan.prefixReserve > capacity) {
        return false;
    }
    uint64_t maxTileBytes = std::min(SCATTER_MAX_TILE_BYTES,
        (capacity - plan.prefixReserve) / SCATTER_PIPELINE_DEPTH);
    const uint64_t alignment = maxTileBytes >= SCATTER_TILE_ALIGNMENT ? SCATTER_TILE_ALIGNMENT : sizeof(float);
    maxTileBytes -= maxTileBytes % alignment;
    if (maxTileBytes == 0) {
        return false;
    }
    plan.slotBytes = maxTileBytes;
    plan.prefix = MakeSectionPlan(plan.prefixBytes, maxTileBytes, alignment);
    plan.own = MakeSectionPlan(recvBytes, maxTileBytes, alignment);
    plan.suffix = MakeSectionPlan(recvBytes - plan.prefixBytes, maxTileBytes, alignment);
    // V13只改变3--4块自身数据的接收布局；其余情况完整保留V8.1双槽。
    // READY[0..3]各发布一次，前缀仍用4。ACK在反方向，整段完成后仅发一次2。
    // capacity为全rank公共最小容量，所有rank必须据同一条件选择同一协议。
    plan.resident = plan.own.count > SCATTER_PIPELINE_DEPTH && plan.own.count <= 4 &&
        plan.suffix.count <= 4 && recvBytes <= capacity - plan.prefixReserve;
    if (plan.resident) {
        // V16：先锁定 V15 的常驻资格，再合并主段；不能用合并后的两块重判 >2，
        // 否则原本的常驻路径会误退回双槽。此处只改主段，slotBytes 保持原计划。
        uint64_t residentTileBytes = std::min(SCATTER_RESIDENT_MAX_TILE_BYTES,
            std::min(recvBytes, capacity - plan.prefixReserve));
        residentTileBytes -= residentTileBytes % alignment;
        if (residentTileBytes > maxTileBytes) {
            const TilePlan own = MakeSectionPlan(recvBytes, residentTileBytes, alignment);
            const TilePlan suffix = MakeSectionPlan(recvBytes - plan.prefixBytes, residentTileBytes, alignment);
            // READY[0..3] 与前缀 READY[4] 不交叉；合并不增加块数。
            // 若将来改变参数导致计划不满足预算，则继续使用上面的 V15 计划。
            if (own.count <= plan.own.count && suffix.count <= plan.suffix.count &&
                own.bytes <= residentTileBytes && suffix.bytes <= residentTileBytes) {
                plan.own = own;
                plan.suffix = suffix;
            }
        }
        if (plan.prefixBytes != 0 && plan.prefix.count > 1 &&
            plan.prefixBytes <= SCATTER_PREFIX_SINGLE_TRANSFER_BYTES) {
            // V17：原协议已经在整个前缀 Write 完成后才发布一次 READY。
            // 合并数据调用，不改变 READY/ACK 次数、前缀地址和主段边界。
            // 单次只搬运有效字节；prefixReserve 的 padding 不参与传输。
            // root Write 和目标 Read 均使用这一计划，helper 仍只传递一次 READY。
            plan.prefix.bytes = plan.prefixBytes;
            plan.prefix.count = 1;
        }
        const uint64_t suffixBytes = recvBytes - plan.prefixBytes;
        if (plan.suffix.count == 2 && suffixBytes != 0 &&
            suffixBytes <= SCATTER_SUFFIX_SINGLE_TRANSFER_BYTES) {
            // V18：原 resident 已证明整个接收段容量充足；只合并直传后缀。
            // 两端共用 suffix 计划，一次 Write/READY 对应一次 Wait/Copy；
            // 整段 ACK 仍在 Copy 完成后。own、prefix 和回退双槽均不受影响。
            // 使用真实 FP32 字节数，不把对齐 padding 写入用户输出。
            plan.suffix.bytes = suffixBytes;
            plan.suffix.count = 1;
        }
    }
    return true;
}

HcclResult PushTopologyPeer(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t peer, uint64_t recvBytes, const TopologyPushPlan &plan)
{
    const ThreadHandle worker = resCtx.threads[WorkerIndex(param.myRank, peer)];
    const ChannelInfo &peerChannel = resCtx.channels[peer];
    const bool toHelper = peer / SCATTER_MESH_RANKS == param.root / SCATTER_MESH_RANKS;
    const uint32_t pairedRank = OppositeMeshRank(peer);
    const char *input = static_cast<const char *>(param.inputPtr);
    char *remote = static_cast<char *>(peerChannel.remoteCclMem.addr);

    // 逆向 READY[0] 是本次目标许可，与对端尚未被消费的旧 ACK[0/1] 分开。
    CHK_RET(HcommChannelNotifyWaitOnThread(worker, peerChannel.handle, ReadyNotify(0), CUSTOM_TIMEOUT));
    if (toHelper && plan.prefixBytes != 0) {
        const char *source = input + static_cast<uint64_t>(pairedRank) * recvBytes;
        for (uint64_t tile = 0; tile < plan.prefix.count; ++tile) {
            const uint64_t offset = tile * plan.prefix.bytes;
            const uint64_t bytes = std::min(plan.prefix.bytes, plan.prefixBytes - offset);
            CHK_RET(HcommWriteOnThread(worker, peerChannel.handle, remote + offset, source + offset, bytes));
        }
        // 沿用 V7 的普通 Write + Record，避免普通写与 128B 融合 WQE 混用的解析兼容问题。
        // 两步必须在同一 worker、同一 Channel 上顺序下发。
        // 整个前缀写完才发布一次 READY；不能在循环内重复发布同一个通知。
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, peerChannel.handle, SCATTER_RELAY_READY_NOTIFY));
    }

    const bool suffixOnly = !toHelper && pairedRank != param.root;
    const uint64_t sectionOffset = suffixOnly ? plan.prefixBytes : 0;
    const uint64_t sectionBytes = recvBytes - sectionOffset;
    const TilePlan &tiles = suffixOnly ? plan.suffix : plan.own;
    const char *source = input + static_cast<uint64_t>(peer) * recvBytes + sectionOffset;
    char *slots = remote + (toHelper ? plan.prefixReserve : 0);
    for (uint64_t tile = 0; tile < tiles.count; ++tile) {
        if (!plan.resident && tile >= SCATTER_PIPELINE_DEPTH) {
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, peerChannel.handle, AckNotify(tile), CUSTOM_TIMEOUT));
        }
        const uint64_t offset = tile * tiles.bytes;
        const uint64_t bytes = std::min(tiles.bytes, sectionBytes - offset);
        // 常驻布局使用与输出相同的连续偏移，不写padding，不复用先前块。
        char *slot = slots + (plan.resident ? offset : (tile % SCATTER_PIPELINE_DEPTH) * plan.slotBytes);
        CHK_RET(HcommWriteOnThread(worker, peerChannel.handle, slot, source + offset, bytes));
        const uint32_t ready = plan.resident ? static_cast<uint32_t>(tile) : ReadyNotify(tile);
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, peerChannel.handle, ready));
    }
    if (plan.resident) {
        // 接收端Copy完全部块才确认；初始目标许可用反方向0，完成用反方向2。
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, peerChannel.handle, AckNotify(0), CUSTOM_TIMEOUT));
        return HCCL_SUCCESS;
    }
    const uint64_t firstPending = tiles.count > SCATTER_PIPELINE_DEPTH ? tiles.count - SCATTER_PIPELINE_DEPTH : 0;
    for (uint64_t tile = firstPending; tile < tiles.count; ++tile) {
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, peerChannel.handle, AckNotify(tile), CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult ReceiveTopologySection(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t outputOffset, uint64_t localOffset, uint64_t sectionBytes,
    const TilePlan &tiles, const TopologyPushPlan &plan)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    const ChannelHandle rootChannel = resCtx.channels[param.root].handle;
    char *output = static_cast<char *>(param.outputPtr) + outputOffset;
    char *slots = static_cast<char *>(resCtx.localBuffer.addr) + localOffset;
    for (uint64_t tile = 0; tile < tiles.count; ++tile) {
        const uint64_t offset = tile * tiles.bytes;
        const uint64_t bytes = std::min(tiles.bytes, sectionBytes - offset);
        char *slot = slots + (plan.resident ? offset : (tile % SCATTER_PIPELINE_DEPTH) * plan.slotBytes);
        const uint32_t ready = plan.resident ? static_cast<uint32_t>(tile) : ReadyNotify(tile);
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, rootChannel, ready, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(mainThread, output + offset, slot, bytes));
        // ACK 释放 Push 目标槽，必须在 Copy 完成后；不依赖另一块常驻前缀的消费进度。
        if (!plan.resident) {
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel, AckNotify(tile)));
        }
    }
    if (plan.resident) {
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel, AckNotify(0)));
    }
    return HCCL_SUCCESS;
}

HcclResult ReceiveTopologyPrefix(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t helperRank, const TopologyPushPlan &plan)
{
    const ThreadHandle worker = resCtx.threads[WorkerIndex(param.myRank, helperRank)];
    const ChannelInfo &helperChannel = resCtx.channels[helperRank];
    const char *source = static_cast<const char *>(helperChannel.remoteCclMem.addr);
    char *output = static_cast<char *>(param.outputPtr);
    CHK_RET(HcommChannelNotifyWaitOnThread(worker, helperChannel.handle,
        SCATTER_RELAY_READY_NOTIFY, CUSTOM_TIMEOUT));
    for (uint64_t tile = 0; tile < plan.prefix.count; ++tile) {
        const uint64_t offset = tile * plan.prefix.bytes;
        const uint64_t bytes = std::min(plan.prefix.bytes, plan.prefixBytes - offset);
        CHK_RET(HcommReadOnThread(worker, helperChannel.handle, output + offset, source + offset, bytes));
    }
    // 前缀区整次不复用，一次 ACK 证明所有前缀 Read 完成。
    CHK_RET(HcommChannelNotifyRecordOnThread(worker, helperChannel.handle, AckNotify(0)));
    return HCCL_SUCCESS;
}

HcclResult ScatterTopologyPush(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvBytes, const TopologyPushPlan &plan)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    if (param.myRank == param.root) {
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.root) {
                continue;
            }
            const ThreadHandle worker = resCtx.threads[WorkerIndex(param.myRank, peer)];
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, SCATTER_WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
            CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, SCATTER_WORKER_START_NOTIFY));
        }
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.root) {
                continue;
            }
            CHK_RET(PushTopologyPeer(param, resCtx, peer, recvBytes, plan));
            const uint32_t workerIndex = WorkerIndex(param.myRank, peer);
            CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.threads[workerIndex], mainThread, workerIndex));
        }
        for (uint32_t workerIndex = 1; workerIndex < param.rankSize; ++workerIndex) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, workerIndex, CUSTOM_TIMEOUT));
        }
        // 所有输入片的第一跳已完成；二跳只访问 helper 的常驻前缀，不再引用 root 输入。
        return CopyRootResult(param, resCtx, recvBytes);
    }

    const ChannelHandle rootChannel = resCtx.channels[param.root].handle;
    if (param.myRank / SCATTER_MESH_RANKS == param.root / SCATTER_MESH_RANKS) {
        // 先许可整个本地布局，再等待 root；不能把该许可放在 prefix READY 之后。
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel, ReadyNotify(0)));
        const ChannelHandle partner = resCtx.channels[OppositeMeshRank(param.myRank)].handle;
        if (plan.prefixBytes != 0) {
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, rootChannel,
                SCATTER_RELAY_READY_NOTIFY, CUSTOM_TIMEOUT));
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, partner, SCATTER_RELAY_READY_NOTIFY));
        }
        CHK_RET(ReceiveTopologySection(param, resCtx, 0, plan.prefixReserve, recvBytes, plan.own, plan));
        if (plan.prefixBytes != 0) {
            // 仅在本 rank 结束前收二跳 ACK；自身目标双槽从不等待二跳。
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, partner, AckNotify(0), CUSTOM_TIMEOUT));
        }
        return HCCL_SUCCESS;
    }

    const uint32_t helperRank = OppositeMeshRank(param.myRank);
    const bool hasPrefix = helperRank != param.root && plan.prefixBytes != 0;
    const uint32_t workerIndex = WorkerIndex(param.myRank, helperRank);
    if (hasPrefix) {
        const ThreadHandle worker = resCtx.threads[workerIndex];
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, SCATTER_WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, SCATTER_WORKER_START_NOTIFY));
        CHK_RET(ReceiveTopologyPrefix(param, resCtx, helperRank, plan));
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, workerIndex));
    }
    // main 的首次许可先于任何 DATA 等待或 worker JOIN；其首任务仍是模板的 Host WAIT。
    CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel, ReadyNotify(0)));
    const uint64_t outputOffset = helperRank == param.root ? 0 : plan.prefixBytes;
    const TilePlan &tiles = helperRank == param.root ? plan.own : plan.suffix;
    CHK_RET(ReceiveTopologySection(param, resCtx, outputOffset, 0, recvBytes - outputOffset, tiles, plan));
    if (hasPrefix) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, workerIndex, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    uint64_t recvBytes = 0;
    CHK_RET(GetRecvBytes(param, recvBytes));
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(param.outputPtr);
    if (param.myRank == param.root) {
        CHK_PTR_NULL(param.inputPtr);
    }

    uint64_t capacity = 0;
    CHK_RET(GetBufferCapacity(param, resCtx, capacity));
    if (param.rankSize == 1) {
        return CopyRootResult(param, resCtx, recvBytes);
    }
    if (recvBytes <= SCATTER_SMALL_RECV_BYTES && recvBytes * param.rankSize <= capacity) {
        if (param.rankSize == SCATTER_TOPO_RANKS) {
            return ScatterSmallTopologyTree(param, resCtx, recvBytes);
        }
        return ScatterSmall(param, resCtx, recvBytes);
    }
    if (param.rankSize == SCATTER_TOPO_RANKS) {
        TopologyPushPlan plan;
        // capacity 是全 rank CCL 的公共最小值，所有 rank 一致选择 Push 或原 V5 回退。
        if (MakeTopologyPushPlan(recvBytes, capacity, plan)) {
            return ScatterTopologyPush(param, resCtx, recvBytes, plan);
        }
        return ScatterMeshClos(param, resCtx, recvBytes, capacity);
    }
    return ScatterPipelined(param, resCtx, recvBytes, capacity);
}
} // namespace ops_hccl
