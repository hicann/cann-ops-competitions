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

namespace ops_hccl {

constexpr uint64_t MAX_DATA_SIZE = 256ULL * 1024 * 1024;
constexpr uint32_t ROOT_RANK = 0;
constexpr uint32_t SERVER_RANK_SIZE = CUSTOM_SERVER_RANK_SIZE;
constexpr uint32_t SERVER_NUM = 2;
constexpr uint32_t GLOBAL_RANK_SIZE = SERVER_RANK_SIZE * SERVER_NUM;
constexpr uint32_t NOTIFY_IDX_PHASE = 2;
constexpr uint64_t HIERARCHICAL_THRESHOLD = 64ULL * 1024;
constexpr uint32_t WORKER_THREAD_NUM = CUSTOM_INTRA_WORKER_NUM;
constexpr uint64_t PARALLEL_READ_THRESHOLD = 1024ULL * 1024;
// 64KB～4MB使用9a85b32的分片分层路径；更大数据使用6cbb922的FullMesh。
constexpr uint64_t RECURSIVE_DOUBLING_THRESHOLD = 4ULL * 1024 * 1024;
// Official two-dimensional parallel split, quantized to eighths for the
// competition topology: Mesh-first 3/8, Clos-first 5/8.
constexpr uint64_t PARALLEL_SPLIT_DENOMINATOR = 8;
constexpr uint64_t MESH_FIRST_NUMERATOR = 3;
constexpr uint64_t FULLMESH_ALIGNMENT = GLOBAL_RANK_SIZE * PARALLEL_SPLIT_DENOMINATOR;

static HcclResult GetChannel(
    const std::unordered_map<uint32_t, ChannelInfo> &channelMap, uint32_t rank, const ChannelInfo **channel)
{
    auto it = channelMap.find(rank);
    CHK_PRT_RET(it == channelMap.end(), HCCL_ERROR("GetChannel: channel to rank[%u] not found", rank),
        HCCL_E_INTERNAL);
    *channel = &it->second;
    return HCCL_SUCCESS;
}

static HcclResult StartWorkers(const std::vector<ThreadHandle> &threads)
{
    CHK_PRT_RET(threads.size() < WORKER_THREAD_NUM + 1,
        HCCL_ERROR("StartWorkers: insufficient thread count[%zu]", threads.size()), HCCL_E_INTERNAL);
    for (uint32_t worker = 0; worker < WORKER_THREAD_NUM; worker++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[0], threads[worker + 1], 0)));
    }
    for (uint32_t worker = 0; worker < WORKER_THREAD_NUM; worker++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[worker + 1], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

static HcclResult WaitWorkers(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t worker = 0; worker < WORKER_THREAD_NUM; worker++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[0], worker + 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t worker = 0; worker < WORKER_THREAD_NUM; worker++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[worker + 1], threads[0], worker + 1)));
    }
    return HCCL_SUCCESS;
}

// AllReduce: Root-based Reduce + Broadcast
//   - Reduce: 非 root rank 将数据拷入自己的 CCL buffer，root 通过 Read 从远端读取并规约
//     （每个 rank 只写自己的 CCL buffer，root 只读自己的，无内存冲突）
//   - Broadcast: root 将规约结果逐个写到各 rank 的 CCL buffer
static HcclResult ExecRoot(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    uint32_t dataTypeSize = SIZE_TABLE.at(param.dataType);
    ThreadHandle thread = resCtx.aicpuThread;
    void *cclBuf = resCtx.localBuffer.addr;
    uint64_t cclBufSize = resCtx.localBuffer.size;

    // ReadReduce 可直接在 root 的 CCL buffer 中累加，无需单独的远端数据接收槽。
    // 完整 CCL buffer 均可用于当前分片，保持大数据场景较少的循环轮数。
    uint64_t maxDataPerLoop = std::min(MAX_DATA_SIZE, cclBufSize);
    uint64_t maxCountPerLoop = maxDataPerLoop / dataTypeSize;
    CHK_PRT_RET(maxCountPerLoop == 0,
        HCCL_ERROR("ExecOp: ccl buffer too small, maxDataPerLoop[%llu] dataTypeSize[%u]",
            static_cast<unsigned long long>(maxDataPerLoop), dataTypeSize),
        HCCL_E_INTERNAL);

    std::unordered_map<uint32_t, ChannelInfo> channelMap;
    for (const auto &ch : resCtx.channels) {
        channelMap.emplace(ch.remoteRank, ch);
    }

    uint64_t loopCount = param.count / maxCountPerLoop + (param.count % maxCountPerLoop != 0 ? 1 : 0);
    for (uint64_t k = 0; k < loopCount; k++) {
        uint64_t subCount = std::min(maxCountPerLoop, param.count - k * maxCountPerLoop);
        uint64_t subBytes = subCount * dataTypeSize;
        uint64_t sliceOffset = k * maxCountPerLoop * dataTypeSize;

        uint8_t *sendBase = static_cast<uint8_t *>(param.inputPtr) + sliceOffset;
        uint8_t *recvBase = static_cast<uint8_t *>(param.outputPtr) + sliceOffset;

        if (param.myRank == ROOT_RANK) {
            // root 自身数据作为 CCL 累加槽的初始值。
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, cclBuf, sendBase, subBytes)));

            // 依次从每个非 root rank 读取并规约
            for (uint32_t j = 0; j < param.rankSize; j++) {
                if (j == ROOT_RANK) {
                    continue;
                }
                auto it = channelMap.find(j);
                CHK_PRT_RET(it == channelMap.end(),
                    HCCL_ERROR("ExecOp: channel to rank[%u] not found", j), HCCL_E_INTERNAL);
                const ChannelInfo &ch = it->second;

                // 等待 rank j 数据就绪
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, ch.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                // 读取远端 CCL buffer，并直接规约到 root 的 CCL 累加槽。
                CHK_RET(static_cast<HcclResult>(
                    HcommReadReduceOnThread(thread, ch.handle, cclBuf, ch.remoteCclMem.addr, subCount,
                        static_cast<HcommDataType>(param.dataType),
                        static_cast<HcommReduceOp>(param.reduceType))));
                // 通知 rank j 可复用其 CCL buffer
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, ch.handle, NOTIFY_IDX_ACK)));
            }

            // root 输出规约结果。
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, recvBase, cclBuf, subBytes)));

            // 融合 Write + Record，分发结果并通知非 root rank。
            for (uint32_t j = 0; j < param.rankSize; j++) {
                if (j == ROOT_RANK) {
                    continue;
                }
                auto it = channelMap.find(j);
                const ChannelInfo &ch = it->second;

                CHK_RET(static_cast<HcclResult>(
                    HcommWriteWithNotifyOnThread(
                        thread, ch.handle, ch.remoteCclMem.addr, cclBuf, subBytes, NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, ch.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            }
        } else {
            auto it = channelMap.find(ROOT_RANK);
            CHK_PRT_RET(it == channelMap.end(),
                HCCL_ERROR("ExecOp: channel to root rank[%u] not found", ROOT_RANK), HCCL_E_INTERNAL);
            const ChannelInfo &rootCh = it->second;

            // Reduce: 将数据拷入自己的 CCL buffer，通知 root 来读
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, cclBuf, sendBase, subBytes)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, rootCh.handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, rootCh.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

            // Broadcast: 接收结果
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, rootCh.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, recvBase, cclBuf, subBytes)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, rootCh.handle, NOTIFY_IDX_ACK)));
        }
    }

    return HCCL_SUCCESS;
}

// The Four-Template body is aligned to 128 elements.  A 400M+4B payload
// therefore leaves a tiny tail; routing it through the root algorithm adds a
// serial 15-rank fan-in/fan-out.  Recursive doubling completes the same tail
// in four symmetric rounds with a disjoint CCL receive scratch.
static HcclResult ExecTinyRecursiveDoubling(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.rankSize != GLOBAL_RANK_SIZE,
        HCCL_ERROR("ExecTinyRecursiveDoubling: unsupported rankSize[%u]", param.rankSize), HCCL_E_INTERNAL);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    uint32_t dataTypeSize = SIZE_TABLE.at(param.dataType);
    uint64_t bytes = param.count * dataTypeSize;
    CHK_PRT_RET(resCtx.localBuffer.size < bytes * 5,
        HCCL_ERROR("ExecTinyRecursiveDoubling: CCL buffer too small"), HCCL_E_INTERNAL);

    CHK_PRT_RET(resCtx.threads.empty(),
        HCCL_ERROR("ExecTinyRecursiveDoubling: main thread is unavailable"), HCCL_E_INTERNAL);
    // Keep the tail on the Four-Template main queue.  The global body/tail
    // barrier is enqueued on the same queue, so checker sees a strict edge
    // from all large-body CCL accesses to the tiny-tail CCL reuse.
    ThreadHandle thread = resCtx.threads[0];
    uint8_t *work = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    std::unordered_map<uint32_t, ChannelInfo> channelMap;
    for (const auto &channel : resCtx.channels) {
        channelMap.emplace(channel.remoteRank, channel);
    }

    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, work, param.inputPtr, bytes)));
    const uint32_t masks[] = {1, 2, 4, 8};
    uint32_t stage = 0;
    for (uint32_t mask : masks) {
        uint32_t peer = param.myRank ^ mask;
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(channelMap, peer, &channel));
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
            thread, channel->handle,
            static_cast<uint8_t *>(channel->remoteCclMem.addr) + bytes * (stage + 1),
            work, bytes, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel->handle, NOTIFY_IDX_PHASE)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel->handle, NOTIFY_IDX_PHASE, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            thread, work, work + bytes * (stage + 1), param.count,
            static_cast<HcommDataType>(param.dataType),
            static_cast<HcommReduceOp>(param.reduceType))));
        stage++;
    }
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, param.outputPtr, work, bytes)));
    return HCCL_SUCCESS;
}

// 2 Server x 8 Rank hierarchical AllReduce used by the latency-sensitive
// 512KB case: intra-server 1/8 reduce, same-local-rank inter-server merge,
// then an intra-server AllGather.
static HcclResult ExecHierarchical(const OpParam &param, const AlgResourceCtx &resCtx)
{
    uint32_t dataTypeSize = SIZE_TABLE.at(param.dataType);
    uint64_t totalDataBytes = param.count * dataTypeSize;
    bool useParallelRead = totalDataBytes > PARALLEL_READ_THRESHOLD;
    ThreadHandle thread = resCtx.aicpuThread;
    uint8_t *cclBuf = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint64_t maxDataPerLoop = std::min(MAX_DATA_SIZE, resCtx.localBuffer.size);
    uint64_t maxCountPerLoop = maxDataPerLoop / dataTypeSize;
    if (useParallelRead) {
        // 输入占8份，7个独立接收槽再占7份，单轮总占用为15份。
        uint64_t parallelMaxCount = (resCtx.localBuffer.size / dataTypeSize / 15) * SERVER_RANK_SIZE;
        maxCountPerLoop = std::min(maxCountPerLoop, parallelMaxCount);
    }
    CHK_PRT_RET(maxCountPerLoop == 0, HCCL_ERROR("ExecHierarchical: CCL buffer too small"), HCCL_E_INTERNAL);

    std::unordered_map<uint32_t, ChannelInfo> channelMap;
    for (const auto &channel : resCtx.channels) {
        channelMap.emplace(channel.remoteRank, channel);
    }

    uint32_t localRank = param.myRank % SERVER_RANK_SIZE;
    uint32_t serverId = param.myRank / SERVER_RANK_SIZE;
    uint32_t serverBase = serverId * SERVER_RANK_SIZE;
    uint32_t peerRank = (SERVER_NUM - 1 - serverId) * SERVER_RANK_SIZE + localRank;
    const ChannelInfo *peerChannel = nullptr;
    CHK_RET(GetChannel(channelMap, peerRank, &peerChannel));

    std::vector<uint32_t> localPeers;
    for (uint32_t rank = serverBase; rank < serverBase + SERVER_RANK_SIZE; rank++) {
        if (rank != param.myRank) {
            localPeers.push_back(rank);
        }
    }

    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        uint64_t subCount = std::min(maxCountPerLoop, param.count - processedCount);
        uint64_t subBytes = subCount * dataTypeSize;
        uint64_t globalOffset = processedCount * dataTypeSize;
        uint8_t *sendBase = static_cast<uint8_t *>(param.inputPtr) + globalOffset;
        uint8_t *recvBase = static_cast<uint8_t *>(param.outputPtr) + globalOffset;

        uint64_t baseCount = subCount / SERVER_RANK_SIZE;
        uint64_t remainder = subCount % SERVER_RANK_SIZE;
        uint64_t ownerCount = baseCount + static_cast<uint64_t>(localRank < remainder);
        uint64_t ownerOffsetCount = localRank * baseCount + std::min<uint64_t>(localRank, remainder);
        uint64_t ownerOffset = ownerOffsetCount * dataTypeSize;
        uint64_t ownerBytes = ownerCount * dataTypeSize;

        // 所有 rank 先将本轮输入放入本地 CCL buffer，并通知7个机内peer。
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, cclBuf, sendBase, subBytes)));
        for (uint32_t rank = serverBase; rank < serverBase + SERVER_RANK_SIZE; rank++) {
            if (rank == param.myRank) {
                continue;
            }
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(channelMap, rank, &channel));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel->handle, NOTIFY_IDX_ACK)));
        }

        if (useParallelRead) {
            // 主Thread先确认所有peer输入就绪，再启动7条worker并行读取到独立槽。
            for (uint32_t peerIdx = 0; peerIdx < localPeers.size(); peerIdx++) {
                const ChannelInfo *channel = nullptr;
                CHK_RET(GetChannel(channelMap, localPeers[peerIdx], &channel));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            }
            CHK_RET(StartWorkers(resCtx.threads));
            for (uint32_t peerIdx = 0; peerIdx < localPeers.size(); peerIdx++) {
                const ChannelInfo *channel = nullptr;
                CHK_RET(GetChannel(channelMap, localPeers[peerIdx], &channel));
                uint8_t *remoteSrc = static_cast<uint8_t *>(channel->remoteCclMem.addr) + ownerOffset;
                uint8_t *recvSlot = cclBuf + subBytes + peerIdx * ownerBytes;
                CHK_RET(static_cast<HcclResult>(HcommReadOnThread(resCtx.threads[peerIdx + 1], channel->handle,
                    recvSlot, remoteSrc, ownerBytes)));
            }
            CHK_RET(WaitWorkers(resCtx.threads));

            // 通信完成后由主Thread按固定peer顺序规约，保证浮点确定性。
            for (uint32_t peerIdx = 0; peerIdx < localPeers.size(); peerIdx++) {
                uint8_t *recvSlot = cclBuf + subBytes + peerIdx * ownerBytes;
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, cclBuf + ownerOffset, recvSlot,
                    ownerCount, static_cast<HcommDataType>(param.dataType),
                    static_cast<HcommReduceOp>(param.reduceType))));
            }

            // 确认所有rank均完成远端读取后，才允许跨机阶段覆盖CCL中的局部结果。
            for (uint32_t peerIdx = 0; peerIdx < localPeers.size(); peerIdx++) {
                const ChannelInfo *channel = nullptr;
                CHK_RET(GetChannel(channelMap, localPeers[peerIdx], &channel));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL)));
            }
            for (uint32_t peerIdx = 0; peerIdx < localPeers.size(); peerIdx++) {
                const ChannelInfo *channel = nullptr;
                CHK_RET(GetChannel(channelMap, localPeers[peerIdx], &channel));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }
        } else {
            // 小消息保持V9的单Thread固定顺序ReadReduce路径。
            for (uint32_t peerIdx = 0; peerIdx < localPeers.size(); peerIdx++) {
                const ChannelInfo *channel = nullptr;
                CHK_RET(GetChannel(channelMap, localPeers[peerIdx], &channel));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                if (ownerCount != 0) {
                    uint8_t *remoteSrc = static_cast<uint8_t *>(channel->remoteCclMem.addr) + ownerOffset;
                    CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, channel->handle,
                        cclBuf + ownerOffset, remoteSrc, ownerCount, static_cast<HcommDataType>(param.dataType),
                        static_cast<HcommReduceOp>(param.reduceType))));
                }
            }
        }

        // 同 localRank 跨机合并；Server 0 负责最终规约，Server 1 随后读取结果。
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, peerChannel->handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, peerChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        uint8_t *remoteOwner = static_cast<uint8_t *>(peerChannel->remoteCclMem.addr) + ownerOffset;
        if (serverId == 0) {
            if (ownerCount != 0) {
                CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, peerChannel->handle,
                    cclBuf + ownerOffset, remoteOwner, ownerCount, static_cast<HcommDataType>(param.dataType),
                    static_cast<HcommReduceOp>(param.reduceType))));
            }
            CHK_RET(static_cast<HcclResult>(
                HcommWriteWithNotifyOnThread(thread, peerChannel->handle, remoteOwner, cclBuf + ownerOffset,
                    ownerBytes, NOTIFY_IDX_DATA_SIGNAL)));
        } else {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, peerChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }

        // 每个owner将自己的全局分片写给7个机内rank。
        for (uint32_t peerIdx = 0; peerIdx < localPeers.size(); peerIdx++) {
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(channelMap, localPeers[peerIdx], &channel));
            if (ownerBytes != 0) {
                uint8_t *remoteDst = static_cast<uint8_t *>(channel->remoteCclMem.addr) + ownerOffset;
                CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(thread, channel->handle, remoteDst,
                    cclBuf + ownerOffset, ownerBytes, NOTIFY_IDX_PHASE)));
            }
        }

        // 等待其余7个owner写入对应结果分片。
        for (uint32_t shard = 0; shard < SERVER_RANK_SIZE; shard++) {
            uint64_t shardCount = baseCount + static_cast<uint64_t>(shard < remainder);
            uint64_t shardBytes = shardCount * dataTypeSize;
            if (shard == localRank || shardBytes == 0) {
                continue;
            }
            uint32_t ownerRank = serverBase + shard;
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(channelMap, ownerRank, &channel));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, channel->handle, NOTIFY_IDX_PHASE, CUSTOM_TIMEOUT)));
        }
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, recvBase, cclBuf, subBytes)));
        processedCount += subCount;
    }
    return HCCL_SUCCESS;
}

// Official AICPU queue synchronization pattern: main START -> workers, then
// workers DONE -> main.  Notify 0 on each worker is START; main notify [0, 7]
// receives DONE from the seven intra workers and the inter worker.
static HcclResult StartParallelPhase(const std::vector<ThreadHandle> &threads)
{
    CHK_PRT_RET(threads.size() < CUSTOM_TOTAL_THREAD_NUM,
        HCCL_ERROR("StartParallelPhase: thread count[%zu] smaller than required[%u]", threads.size(),
            CUSTOM_TOTAL_THREAD_NUM),
        HCCL_E_INTERNAL);
    ThreadHandle mainThread = threads[0];
    for (uint32_t i = 1; i < CUSTOM_TOTAL_THREAD_NUM; i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(mainThread, threads[i], 0)));
    }
    for (uint32_t i = 1; i < CUSTOM_TOTAL_THREAD_NUM; i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[i], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

static HcclResult FinishParallelPhase(const std::vector<ThreadHandle> &threads)
{
    ThreadHandle mainThread = threads[0];
    // Match the ordering used by official PostSyncInterThreads: enqueue the
    // waits on main first, then enqueue each worker's completion record.
    for (uint32_t i = 1; i < CUSTOM_TOTAL_THREAD_NUM; i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(mainThread, i - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t i = 1; i < CUSTOM_TOTAL_THREAD_NUM; i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[i], mainThread, i - 1)));
    }
    return HCCL_SUCCESS;
}

// Finish the seven intra-server workers first, but only enqueue the inter-server
// completion record.  The main queue can then reduce the disjoint Mesh slots
// while the inter-server transfer is still running.  WaitInterWorker closes the
// phase before any A/B range is reused by the next phase.
static HcclResult FinishIntraAndRecordAllWorkers(const std::vector<ThreadHandle> &threads)
{
    ThreadHandle mainThread = threads[0];
    for (uint32_t i = 1; i < CUSTOM_INTER_THREAD_INDEX; i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(mainThread, i - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t i = 1; i < CUSTOM_TOTAL_THREAD_NUM; i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[i], mainThread, i - 1)));
    }
    return HCCL_SUCCESS;
}

static HcclResult WaitInterWorker(const std::vector<ThreadHandle> &threads)
{
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        threads[0], CUSTOM_INTER_THREAD_INDEX - 1, CUSTOM_TIMEOUT));
}

// Symmetric channel exchanges.  The pre/post handshake makes the remote
// source/destination lifetime explicit to checker V3.
static HcclResult EnqueueWriteExchange(ThreadHandle thread, const ChannelInfo &channel,
    void *remoteDst, void *localSrc, uint64_t bytes)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    if (bytes != 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommWriteOnThread(thread, channel.handle, remoteDst, localSrc, bytes)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

static HcclResult EnqueueReadExchange(ThreadHandle thread, const ChannelInfo &channel,
    void *localDst, void *remoteSrc, uint64_t bytes)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    if (bytes != 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommReadOnThread(thread, channel.handle, localDst, remoteSrc, bytes)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

// Four dissemination rounds form a true 16-rank barrier.  A rank can leave
// round n only after all ranks in its 2^(n+1)-rank group entered the barrier;
// after mask 8 every rank is ordered after every other rank's Phase 4 finish.
static HcclResult EnqueueGlobalBarrier(ThreadHandle thread,
    const std::unordered_map<uint32_t, ChannelInfo> &channelMap, uint32_t myRank)
{
    const uint32_t masks[] = {1, 2, 4, 8};
    for (uint32_t mask : masks) {
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(channelMap, myRank ^ mask, &channel));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel->handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

static HcclResult EnqueueWriteReduceExchange(ThreadHandle thread, const ChannelInfo &channel,
    void *remoteDst, const void *localSrc, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    if (count != 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommWriteReduceOnThread(thread, channel.handle, remoteDst, localSrc, count, dataType, reduceOp)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

// Ascend 950 official-style Push FullMesh Two-Shot + Four-Template AllReduce.
//
// Each loop follows the official topology-weighted split: A is the 3/8
// Mesh-first part and B is the 5/8 Clos-first part:
//   phase 1: A intra Push-Mesh RS || B inter RS
//   phase 2: B intra Push-Mesh RS || A inter RS
//   phase 3: B intra Pull-Mesh AG || A inter AG
//   phase 4: A intra Pull-Mesh AG || B inter AG
//
// Push-Mesh RS writes every sender into a distinct remote slot, so concurrent
// remote writes never overlap.  Pull-Mesh AG writes every owner's result into
// a distinct local slot.  Intra and inter phases use disjoint A/B ranges.
static HcclResult ExecFullMeshFourTemplate(const OpParam &param, const AlgResourceCtx &resCtx)
{
    uint32_t dataTypeSize = SIZE_TABLE.at(param.dataType);
    CHK_PRT_RET(param.rankSize != GLOBAL_RANK_SIZE,
        HCCL_ERROR("ExecFullMeshFourTemplate: unsupported rankSize[%u]", param.rankSize), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.threads.size() < CUSTOM_TOTAL_THREAD_NUM,
        HCCL_ERROR("ExecFullMeshFourTemplate: thread count[%zu] smaller than required[%u]",
            resCtx.threads.size(), CUSTOM_TOTAL_THREAD_NUM),
        HCCL_E_INTERNAL);

    ThreadHandle mainThread = resCtx.threads[0];
    ThreadHandle interThread = resCtx.threads[CUSTOM_INTER_THREAD_INDEX];
    uint8_t *cclBuf = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint64_t cclCount = resCtx.localBuffer.size / dataTypeSize;
    uint64_t alignedCount = (param.count / FULLMESH_ALIGNMENT) * FULLMESH_ALIGNMENT;
    // A=3/8 and B=5/8 must both split into 2 servers x 8 local ranks, hence
    // 16*8=128-count alignment.
    // If the whole aligned body fits in CCL, run one communication round.  This
    // turns 400M+4B into one Four-Template round while 512MB keeps the verified
    // 256MB + 256MB split because it cannot fit in the 400MB CCL buffer.
    uint64_t loopCountLimit = MAX_DATA_SIZE / dataTypeSize;
    if (alignedCount <= cclCount) {
        loopCountLimit = alignedCount;
    }
    uint64_t maxCountPerLoop = std::min(loopCountLimit, cclCount);
    maxCountPerLoop = (maxCountPerLoop / FULLMESH_ALIGNMENT) * FULLMESH_ALIGNMENT;
    CHK_PRT_RET(maxCountPerLoop == 0,
        HCCL_ERROR("ExecFullMeshFourTemplate: CCL buffer too small"), HCCL_E_INTERNAL);

    std::unordered_map<uint32_t, ChannelInfo> channelMap;
    for (const auto &channel : resCtx.channels) {
        channelMap.emplace(channel.remoteRank, channel);
    }

    uint32_t localRank = param.myRank % SERVER_RANK_SIZE;
    uint32_t serverId = param.myRank / SERVER_RANK_SIZE;
    uint32_t otherServerId = SERVER_NUM - 1 - serverId;
    uint32_t serverBase = serverId * SERVER_RANK_SIZE;
    uint32_t peerRank = (SERVER_NUM - 1 - serverId) * SERVER_RANK_SIZE + localRank;
    const ChannelInfo *peerChannel = nullptr;
    CHK_RET(GetChannel(channelMap, peerRank, &peerChannel));
    HcommDataType hcommDataType = static_cast<HcommDataType>(param.dataType);
    HcommReduceOp hcommReduceOp = static_cast<HcommReduceOp>(param.reduceType);

    uint64_t processedCount = 0;
    while (processedCount < alignedCount) {
        uint64_t subCount = std::min(maxCountPerLoop, alignedCount - processedCount);
        uint64_t globalOffset = processedCount * dataTypeSize;
        uint8_t *sendBase = static_cast<uint8_t *>(param.inputPtr) + globalOffset;
        uint8_t *recvBase = static_cast<uint8_t *>(param.outputPtr) + globalOffset;

        const uint64_t aCount = subCount * MESH_FIRST_NUMERATOR / PARALLEL_SPLIT_DENOMINATOR;
        const uint64_t bCount = subCount - aCount;
        const uint64_t aBytes = aCount * dataTypeSize;
        const uint64_t aBase = 0;
        const uint64_t bBase = aBytes;

        // A: intra owner slice (A/8), then global owner slice (A/16).
        const uint64_t aIntraSliceCount = aCount / SERVER_RANK_SIZE;
        const uint64_t aIntraSliceBytes = aIntraSliceCount * dataTypeSize;
        const uint64_t aGlobalSliceCount = aIntraSliceCount / SERVER_NUM;
        const uint64_t aGlobalSliceBytes = aGlobalSliceCount * dataTypeSize;
        const uint64_t aOwnerOffset = aBase + localRank * aIntraSliceBytes;

        // B: server half (B/2), then one local-rank slice inside it (B/16).
        const uint64_t bServerHalfCount = bCount / SERVER_NUM;
        const uint64_t bServerHalfBytes = bServerHalfCount * dataTypeSize;
        const uint64_t bLocalSliceCount = bServerHalfCount / SERVER_RANK_SIZE;
        const uint64_t bLocalSliceBytes = bLocalSliceCount * dataTypeSize;
        const uint64_t bKeepBase = bBase + serverId * bServerHalfBytes;
        const uint64_t bScratchBase = bBase + (SERVER_NUM - 1 - serverId) * bServerHalfBytes;
        const uint64_t bOwnerOffset = bKeepBase + localRank * bLocalSliceBytes;

        // ---------------- phase 1: A intra RS || B inter RS ----------------
        // Own A contribution occupies the own-sender slot.  Remote senders use
        // their own slot, so all seven incoming writes are disjoint.
        CHK_RET(StartParallelPhase(resCtx.threads));

        // Match the official Two-Shot queue layout: the main queue copies its
        // own owner slice while the seven slave queues push peer slices.
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
            cclBuf + aOwnerOffset, sendBase + aOwnerOffset, aIntraSliceBytes)));

        for (uint32_t worker = 0; worker < CUSTOM_INTRA_WORKER_NUM; worker++) {
            uint32_t peerLocalRank = (localRank + worker + 1) % SERVER_RANK_SIZE;
            uint32_t peer = serverBase + peerLocalRank;
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(channelMap, peer, &channel));
            uint8_t *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) +
                aBase + localRank * aIntraSliceBytes;
            uint8_t *localPeerSlice = sendBase + aBase + peerLocalRank * aIntraSliceBytes;
            CHK_RET(EnqueueWriteExchange(resCtx.threads[worker + 1], *channel,
                remoteSlot, localPeerSlice, aIntraSliceBytes));
        }
        // Inter worker initializes only the locally owned B half.  After ACK
        // proves both destinations are ready, each server pushes the other
        // half directly into its peer's owner range with hardware reduction.
        // Compared with ReadReduce this avoids copying the unused B half.
        uint64_t bInterKeepOffset = bKeepBase;
        uint64_t bRemoteKeepOffset = bBase + otherServerId * bServerHalfBytes;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(interThread,
            cclBuf + bInterKeepOffset, sendBase + bInterKeepOffset, bServerHalfBytes)));
        CHK_RET(EnqueueWriteReduceExchange(interThread, *peerChannel,
            static_cast<uint8_t *>(peerChannel->remoteCclMem.addr) + bRemoteKeepOffset,
            sendBase + bRemoteKeepOffset,
            bServerHalfCount, hcommDataType, hcommReduceOp));
        CHK_RET(FinishIntraAndRecordAllWorkers(resCtx.threads));

        // Fold the seven disjoint sender slots into A's owner slot.
        for (uint32_t sender = 0; sender < SERVER_RANK_SIZE; sender++) {
            if (sender == localRank) {
                continue;
            }
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread,
                cclBuf + aOwnerOffset, cclBuf + aBase + sender * aIntraSliceBytes,
                aIntraSliceCount, hcommDataType, hcommReduceOp)));
        }
        CHK_RET(WaitInterWorker(resCtx.threads));

        // ---------------- phase 2: B intra RS || A inter RS ----------------
        CHK_RET(StartParallelPhase(resCtx.threads));
        for (uint32_t worker = 0; worker < CUSTOM_INTRA_WORKER_NUM; worker++) {
            uint32_t peerLocalRank = (localRank + worker + 1) % SERVER_RANK_SIZE;
            uint32_t peer = serverBase + peerLocalRank;
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(channelMap, peer, &channel));
            uint8_t *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) +
                bScratchBase + localRank * bLocalSliceBytes;
            uint8_t *localPeerSlice = cclBuf + bKeepBase + peerLocalRank * bLocalSliceBytes;
            CHK_RET(EnqueueWriteExchange(resCtx.threads[worker + 1], *channel,
                remoteSlot, localPeerSlice, bLocalSliceBytes));
        }
        uint64_t aRemoteKeepOffset = aOwnerOffset + otherServerId * aGlobalSliceBytes;
        CHK_RET(EnqueueWriteReduceExchange(interThread, *peerChannel,
            static_cast<uint8_t *>(peerChannel->remoteCclMem.addr) + aRemoteKeepOffset,
            cclBuf + aRemoteKeepOffset,
            aGlobalSliceCount, hcommDataType, hcommReduceOp));
        CHK_RET(FinishIntraAndRecordAllWorkers(resCtx.threads));

        // B remote contributions live in the currently unused server half.
        for (uint32_t sender = 0; sender < SERVER_RANK_SIZE; sender++) {
            if (sender == localRank) {
                continue;
            }
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread,
                cclBuf + bOwnerOffset, cclBuf + bScratchBase + sender * bLocalSliceBytes,
                bLocalSliceCount, hcommDataType, hcommReduceOp)));
        }
        CHK_RET(WaitInterWorker(resCtx.threads));

        // ---------------- phase 3: B intra AG || A inter AG ----------------
        CHK_RET(StartParallelPhase(resCtx.threads));
        for (uint32_t worker = 0; worker < CUSTOM_INTRA_WORKER_NUM; worker++) {
            uint32_t peerLocalRank = (localRank + worker + 1) % SERVER_RANK_SIZE;
            uint32_t peer = serverBase + peerLocalRank;
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(channelMap, peer, &channel));
            uint64_t peerResultOffset = bKeepBase + peerLocalRank * bLocalSliceBytes;
            CHK_RET(EnqueueReadExchange(resCtx.threads[worker + 1], *channel,
                cclBuf + peerResultOffset,
                static_cast<uint8_t *>(channel->remoteCclMem.addr) + peerResultOffset,
                bLocalSliceBytes));
        }
        uint64_t aMissingOffset = aOwnerOffset + otherServerId * aGlobalSliceBytes;
        CHK_RET(EnqueueReadExchange(interThread, *peerChannel,
            cclBuf + aMissingOffset,
            static_cast<uint8_t *>(peerChannel->remoteCclMem.addr) + aMissingOffset,
            aGlobalSliceBytes));
        // A's locally owned slice is complete after the inter read and can be
        // copied on that same queue.  B must first gather all eight local
        // owners back into CCL: phase 4 exposes this complete server half to
        // the same-local-rank peer.
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(interThread,
            recvBase + aOwnerOffset, cclBuf + aOwnerOffset, aIntraSliceBytes)));
        CHK_RET(FinishParallelPhase(resCtx.threads));

        // ---------------- phase 4: A intra AG || B inter AG ----------------
        CHK_RET(StartParallelPhase(resCtx.threads));
        // The complete local B half must remain in CCL for the peer's Pull-AG,
        // but copying it to the disjoint local Output range need not serialize
        // phase 4.  START is already queued on main, so this Copy overlaps the
        // seven intra Pulls and the inter-server Pull without adding a task.
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
            recvBase + bKeepBase, cclBuf + bKeepBase, bServerHalfBytes)));
        for (uint32_t worker = 0; worker < CUSTOM_INTRA_WORKER_NUM; worker++) {
            uint32_t peerLocalRank = (localRank + worker + 1) % SERVER_RANK_SIZE;
            uint32_t peer = serverBase + peerLocalRank;
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(channelMap, peer, &channel));
            uint64_t peerResultOffset = aBase + peerLocalRank * aIntraSliceBytes;
            CHK_RET(EnqueueReadExchange(resCtx.threads[worker + 1], *channel,
                recvBase + peerResultOffset,
                static_cast<uint8_t *>(channel->remoteCclMem.addr) + peerResultOffset,
                aIntraSliceBytes));
        }
        uint64_t bMissingOffset = bBase + otherServerId * bServerHalfBytes;
        CHK_RET(EnqueueReadExchange(interThread, *peerChannel,
            recvBase + bMissingOffset,
            static_cast<uint8_t *>(peerChannel->remoteCclMem.addr) + bMissingOffset,
            bServerHalfBytes));
        CHK_RET(FinishParallelPhase(resCtx.threads));
        if ((processedCount + subCount == alignedCount) && (param.count != alignedCount)) {
            CHK_RET(EnqueueGlobalBarrier(mainThread, channelMap, param.myRank));
        }
        processedCount += subCount;
    }

    uint64_t tailCount = param.count - alignedCount;
    if (tailCount != 0) {
        OpParam tailParam = param;
        uint64_t tailOffset = alignedCount * dataTypeSize;
        tailParam.inputPtr = static_cast<uint8_t *>(param.inputPtr) + tailOffset;
        tailParam.outputPtr = static_cast<uint8_t *>(param.outputPtr) + tailOffset;
        tailParam.count = tailCount;
        CHK_RET(ExecTinyRecursiveDoubling(tailParam, resCtx));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    uint64_t dataSize = param.count * SIZE_TABLE.at(param.dataType);
    // 4B等极小消息使用低启动开销Root。
    if (dataSize <= HIERARCHICAL_THRESHOLD) {
        return ExecRoot(param, resCtx);
    }
    if (param.rankSize == GLOBAL_RANK_SIZE) {
        // 512KB使用9a85b32真实验证的1/8分片分层路径。
        if (dataSize <= RECURSIVE_DOUBLING_THRESHOLD) {
            return ExecHierarchical(param, resCtx);
        }
        // 512MB / 400MB+4B保留0435c4a真实验证的Pull FullMesh Four-Template主体。
        return ExecFullMeshFourTemplate(param, resCtx);
    }
    // 非竞赛16-rank拓扑回退到通用Root路径，避免误入固定拓扑算法。
    return ExecRoot(param, resCtx);
}
} // namespace ops_hccl