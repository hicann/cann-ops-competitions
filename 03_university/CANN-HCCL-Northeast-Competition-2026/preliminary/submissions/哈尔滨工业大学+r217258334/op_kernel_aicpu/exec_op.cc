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

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint32_t TWO_SERVER_RANK_SIZE = 16;
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint64_t SMALL_SLICE_THRESHOLD = 1024ULL * 1024ULL;
constexpr uint64_t LARGE_ALIGN = 256;
constexpr uint32_t WORKER_START_NOTIFY = 0;
constexpr uint32_t WORKER_PHASE_NOTIFY = 1;
constexpr uint32_t MAIN_WORKER_DONE_NOTIFY = 1;
constexpr uint32_t LARGE_ROOT_THREAD_NUM = 15;
constexpr uint64_t RELAY_RATIO_NUMERATOR = 4;
constexpr uint64_t RELAY_RATIO_DENOMINATOR = 11;

inline uint8_t *AddOffset(void *ptr, uint64_t offset)
{
    return static_cast<uint8_t *>(ptr) + offset;
}

inline const uint8_t *AddOffset(const void *ptr, uint64_t offset)
{
    return static_cast<const uint8_t *>(ptr) + offset;
}

inline uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value / alignment * alignment;
}

const ChannelInfo *GetChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    if (remoteRank >= resCtx.channels.size()) {
        return nullptr;
    }
    const ChannelInfo &channel = resCtx.channels[remoteRank];
    if (channel.remoteRank != remoteRank || channel.remoteCclMem.addr == nullptr) {
        return nullptr;
    }
    return &channel;
}

uint32_t FindRankIndex(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    for (uint32_t idx = 0; idx < ranks.size(); ++idx) {
        if (ranks[idx] == rank) {
            return idx;
        }
    }
    return INVALID_VALUE_RANKID;
}

uint64_t GetMinCclBufferSize(const AlgResourceCtx &resCtx, uint32_t myRank)
{
    uint64_t minSize = resCtx.localBuffer.size;
    for (uint32_t rank = 0; rank < resCtx.channels.size(); ++rank) {
        if (rank == myRank) {
            continue;
        }
        const ChannelInfo *channel = GetChannel(resCtx, rank);
        if (channel != nullptr) {
            minSize = std::min(minSize, channel->remoteCclMem.size);
        }
    }
    return minSize;
}

HcclResult RecordChannel(ThreadHandle thread, const ChannelInfo *channel)
{
    CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Invalid channel for record"), HCCL_E_INTERNAL);
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, NOTIFY_IDX_ACK));
    return HCCL_SUCCESS;
}

HcclResult WaitChannel(ThreadHandle thread, const ChannelInfo *channel)
{
    CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Invalid channel for wait"), HCCL_E_INTERNAL);
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult WaitChannelNotify(ThreadHandle thread, const ChannelInfo *channel, uint32_t notifyIdx)
{
    CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Invalid channel for indexed wait"), HCCL_E_INTERNAL);
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, notifyIdx, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult WriteWithNotify(ThreadHandle thread, const ChannelInfo *channel, void *remoteDst,
    const void *localSrc, uint64_t len, uint32_t notifyIdx)
{
    CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Invalid channel for direct write"), HCCL_E_INTERNAL);
    CHK_PRT_RET(remoteDst == nullptr, HCCL_ERROR("Invalid remote destination for rank[%u]", channel->remoteRank),
        HCCL_E_INTERNAL);
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, channel->handle, remoteDst, localSrc, len, notifyIdx));
    return HCCL_SUCCESS;
}

HcclResult RecordDone(ThreadHandle thread, const ChannelInfo *channel)
{
    CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Invalid channel for done record"), HCCL_E_INTERNAL);
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL));
    return HCCL_SUCCESS;
}

HcclResult WaitDone(ThreadHandle thread, const ChannelInfo *channel)
{
    CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Invalid channel for done wait"), HCCL_E_INTERNAL);
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult StartWorker(ThreadHandle mainThread, ThreadHandle workerThread)
{
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, workerThread, WORKER_START_NOTIFY));
    CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult RunDirectRegion(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t offset, uint64_t len, bool contiguousInput)
{
    if (len == 0) {
        return HCCL_SUCCESS;
    }

    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *localCcl = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    const uint64_t sliceBytes = param.count * sizeof(float);

    if (param.rankSize == 1) {
        CHK_RET(HcommLocalCopyOnThread(mainThread, output + offset, input + offset, len));
        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel = GetChannel(resCtx, param.root);
    if (param.myRank == param.root) {
        if (contiguousInput) {
            // 小包：一次把整个sendBuf搬入HCCL Buffer，15个rank从固定偏移并行直读。
            CHK_RET(HcommLocalCopyOnThread(mainThread, localCcl, input, sliceBytes * param.rankSize));
        } else {
            // 尾块：仅把每个rank的尾部紧凑打包，避免大块非对齐路径污染主算法。
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                CHK_RET(HcommLocalCopyOnThread(mainThread, localCcl + rank * len,
                    input + static_cast<uint64_t>(rank) * sliceBytes + offset, len));
            }
        }

        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank == param.root) {
                continue;
            }
            CHK_RET(RecordChannel(mainThread, GetChannel(resCtx, rank)));
        }

        CHK_RET(HcommLocalCopyOnThread(
            mainThread, output + offset, input + static_cast<uint64_t>(param.root) * sliceBytes + offset, len));

        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank == param.root) {
                continue;
            }
            CHK_RET(WaitDone(mainThread, GetChannel(resCtx, rank)));
        }
        return HCCL_SUCCESS;
    }

    CHK_RET(WaitChannel(mainThread, rootChannel));
    uint64_t remoteOffset = contiguousInput ? static_cast<uint64_t>(param.myRank) * sliceBytes
                                            : static_cast<uint64_t>(param.myRank) * len;
    CHK_RET(HcommReadOnThread(
        mainThread, rootChannel->handle, output + offset, AddOffset(rootChannel->remoteCclMem.addr, remoteOffset), len));
    CHK_RET(RecordDone(mainThread, rootChannel));
    return HCCL_SUCCESS;
}

HcclResult RunRemoteInputSmall(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t sliceBytes)
{
    ThreadHandle mainThread = resCtx.threads[0];
    const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);

    if (param.rankSize == 1) {
        CHK_RET(HcommLocalCopyOnThread(mainThread, output, input, sliceBytes));
        return HCCL_SUCCESS;
    }

    if (param.myRank == param.root) {
        // root输入在进入本算子前已由其stream保证ready。这里仍向15个receiver发ready notify，
        // 用于跨rank stream排序；但不再执行V2的整sendBuf -> HCCL Buffer本地拷贝。
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.root) {
                CHK_RET(RecordChannel(mainThread, GetChannel(resCtx, rank)));
            }
        }
        CHK_RET(HcommLocalCopyOnThread(mainThread, output,
            input + static_cast<uint64_t>(param.root) * sliceBytes, sliceBytes));
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.root) {
                CHK_RET(WaitDone(mainThread, GetChannel(resCtx, rank)));
            }
        }
        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel = GetChannel(resCtx, param.root);
    CHK_PRT_RET(rootChannel == nullptr || rootChannel->hasRemoteUserInput == 0 ||
            rootChannel->remoteUserInput.addr == nullptr ||
            rootChannel->remoteUserInput.size < sliceBytes * param.rankSize,
        HCCL_ERROR("Registered root sendBuf is unavailable on rank[%u]", param.myRank), HCCL_E_INTERNAL);

    // 与官方AICPU Send/Recv排序保持一致：ready -> one-sided read -> done。
    // 唯一变化是remote source由root HCCL Buffer改为HcclCommMemReg暴露的root sendBuf对应slice。
    CHK_RET(WaitChannel(mainThread, rootChannel));
    CHK_RET(HcommReadOnThread(mainThread, rootChannel->handle, output,
        AddOffset(rootChannel->remoteUserInput.addr, static_cast<uint64_t>(param.myRank) * sliceBytes), sliceBytes));
    CHK_RET(RecordDone(mainThread, rootChannel));
    return HCCL_SUCCESS;
}

struct ParallelTopo {
    const std::vector<uint32_t> *rootGroup = nullptr;
    const std::vector<uint32_t> *remoteGroup = nullptr;
    uint32_t myIndex = INVALID_VALUE_RANKID;
    uint32_t rootIndex = INVALID_VALUE_RANKID;
    uint32_t remoteRoot = INVALID_VALUE_RANKID;
    bool onRootServer = false;
};

bool BuildParallelTopo(const OpParam &param, const AlgResourceCtx &resCtx, ParallelTopo &topo)
{
    if (param.rankSize != TWO_SERVER_RANK_SIZE || resCtx.localRanks.size() != SERVER_RANK_SIZE ||
        resCtx.remoteRanks.size() != SERVER_RANK_SIZE) {
        return false;
    }

    uint32_t rootInLocal = FindRankIndex(resCtx.localRanks, param.root);
    if (rootInLocal != INVALID_VALUE_RANKID) {
        topo.rootGroup = &resCtx.localRanks;
        topo.remoteGroup = &resCtx.remoteRanks;
        topo.onRootServer = true;
    } else {
        uint32_t rootInRemote = FindRankIndex(resCtx.remoteRanks, param.root);
        if (rootInRemote == INVALID_VALUE_RANKID) {
            return false;
        }
        topo.rootGroup = &resCtx.remoteRanks;
        topo.remoteGroup = &resCtx.localRanks;
        topo.onRootServer = false;
    }

    topo.rootIndex = FindRankIndex(*topo.rootGroup, param.root);
    topo.remoteRoot = (*topo.remoteGroup)[topo.rootIndex];
    topo.myIndex = topo.onRootServer ? FindRankIndex(*topo.rootGroup, param.myRank)
                                     : FindRankIndex(*topo.remoteGroup, param.myRank);
    return topo.myIndex != INVALID_VALUE_RANKID;
}

uint32_t HighestPowerOfTwo(uint32_t value)
{
    uint32_t power = 1;
    while ((power << 1) <= value) {
        power <<= 1;
    }
    return power;
}

uint32_t RankFromServerLogical(
    const std::vector<uint32_t> &group, uint32_t rootIndex, uint32_t logicalIndex)
{
    return group[(rootIndex + logicalIndex) % SERVER_RANK_SIZE];
}

HcclResult ExecTreeRemoteInputSmall(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t sliceBytes)
{
    ParallelTopo topo;
    if (!BuildParallelTopo(param, resCtx, topo)) {
        return HCCL_E_NOT_SUPPORT;
    }

    const std::vector<uint32_t> &myGroup = topo.onRootServer ? *topo.rootGroup : *topo.remoteGroup;
    uint32_t myGroupIndex = FindRankIndex(myGroup, param.myRank);
    if (myGroupIndex == INVALID_VALUE_RANKID) {
        return HCCL_E_NOT_SUPPORT;
    }

    // 每个Server内部以与root相同的物理下标为逻辑0，构造binomial tree：
    // 0->{1,2,4}, 1->{3,5}, 2->{6}, 3->{7}。
    // root额外先通知remoteRoot，于是跨Server只有一跳ready/done同步；
    // 其余rank拿到ready后仍直接从root注册的sendBuf读取自己的slice。
    const uint32_t myLogical =
        (myGroupIndex + SERVER_RANK_SIZE - topo.rootIndex) % SERVER_RANK_SIZE;

    uint32_t parentRank = INVALID_VALUE_RANKID;
    if (param.myRank != param.root) {
        if (!topo.onRootServer && myLogical == 0) {
            parentRank = param.root;
        } else {
            CHK_PRT_RET(myLogical == 0,
                HCCL_ERROR("Invalid small-tree logical root on rank[%u]", param.myRank), HCCL_E_INTERNAL);
            uint32_t parentLogical = myLogical - HighestPowerOfTwo(myLogical);
            parentRank = RankFromServerLogical(myGroup, topo.rootIndex, parentLogical);
        }
    }

    std::vector<uint32_t> childRanks;
    childRanks.reserve(4);
    if (param.myRank == param.root) {
        // 先启动跨Server分支，隐藏其额外的一跳Clos同步。
        childRanks.push_back(topo.remoteRoot);
    }
    for (uint32_t childLogical = 1; childLogical < SERVER_RANK_SIZE; ++childLogical) {
        if (childLogical - HighestPowerOfTwo(childLogical) == myLogical) {
            childRanks.push_back(RankFromServerLogical(myGroup, topo.rootIndex, childLogical));
        }
    }

    ThreadHandle mainThread = resCtx.threads[0];
    const ChannelInfo *parentChannel =
        parentRank == INVALID_VALUE_RANKID ? nullptr : GetChannel(resCtx, parentRank);

    if (param.myRank != param.root) {
        CHK_PRT_RET(parentChannel == nullptr,
            HCCL_ERROR("Small-tree parent channel is invalid, rank[%u], parent[%u]", param.myRank, parentRank),
            HCCL_E_INTERNAL);
        CHK_RET(WaitChannel(mainThread, parentChannel));
    }

    // ready沿树扩散。深层rank一旦被唤醒就可继续唤醒自己的孩子，
    // 从而把V3由root串行发15次ready变成O(log N)关键路径。
    for (uint32_t childRank : childRanks) {
        CHK_RET(RecordChannel(mainThread, GetChannel(resCtx, childRank)));
    }

    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
    if (param.myRank == param.root) {
        CHK_RET(HcommLocalCopyOnThread(mainThread, output,
            input + static_cast<uint64_t>(param.root) * sliceBytes, sliceBytes));
    } else {
        const ChannelInfo *rootChannel = GetChannel(resCtx, param.root);
        CHK_PRT_RET(rootChannel == nullptr || rootChannel->hasRemoteUserInput == 0 ||
                rootChannel->remoteUserInput.addr == nullptr ||
                rootChannel->remoteUserInput.size < sliceBytes * param.rankSize,
            HCCL_ERROR("Registered root sendBuf is unavailable on small-tree rank[%u]", param.myRank),
            HCCL_E_INTERNAL);

        CHK_RET(HcommReadOnThread(mainThread, rootChannel->handle, output,
            AddOffset(rootChannel->remoteUserInput.addr,
                static_cast<uint64_t>(param.myRank) * sliceBytes),
            sliceBytes));
    }

    // done反向沿同一棵树汇聚。父节点只有在自己的read和整个子树read都完成后才向上record，
    // 所以root只需等待少量直接孩子即可安全释放sendBuf生命周期。
    for (uint32_t childRank : childRanks) {
        CHK_RET(WaitDone(mainThread, GetChannel(resCtx, childRank)));
    }
    if (param.myRank != param.root) {
        CHK_RET(RecordDone(mainThread, parentChannel));
    }
    return HCCL_SUCCESS;
}

uint32_t GetRootPeerThreadIndex(const OpParam &param, const ParallelTopo &topo, uint32_t peerRank)
{
    if (peerRank == topo.remoteRoot) {
        return 0;
    }

    uint32_t threadIdx = 1;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.root || rank == topo.remoteRoot) {
            continue;
        }
        if (rank == peerRank) {
            return threadIdx;
        }
        ++threadIdx;
    }
    return INVALID_VALUE_RANKID;
}

uint32_t GetRootThreadParent(uint32_t threadIdx)
{
    if (threadIdx == 0 || threadIdx >= LARGE_ROOT_THREAD_NUM) {
        return INVALID_VALUE_RANKID;
    }
    return threadIdx - HighestPowerOfTwo(threadIdx);
}

HcclResult StartRootWorkersTree(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.threads.size() < LARGE_ROOT_THREAD_NUM,
        HCCL_ERROR("Balanced direct Scatter needs %u root threads, got[%zu]",
            LARGE_ROOT_THREAD_NUM, resCtx.threads.size()),
        HCCL_E_INTERNAL);

    // V5把root原来的“主thread逐个启动14个worker”改成binomial tree前同步。
    // 每个worker仍先等待一个Thread Notify再下发通信任务，保持AICPU_TS前同步约束；
    // 但主thread只直接启动thread 1/2/4/8，关键路径从14路星型串行控制降为O(logN)。
    for (uint32_t threadIdx = 1; threadIdx < LARGE_ROOT_THREAD_NUM; ++threadIdx) {
        uint32_t parentIdx = GetRootThreadParent(threadIdx);
        CHK_PRT_RET(parentIdx == INVALID_VALUE_RANKID || parentIdx >= resCtx.threads.size(),
            HCCL_ERROR("Invalid root-worker parent, thread[%u], parent[%u]", threadIdx, parentIdx),
            HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyRecordOnThread(
            resCtx.threads[parentIdx], resCtx.threads[threadIdx], WORKER_START_NOTIFY));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.threads[threadIdx], WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult FinishRootWorkersTree(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.threads.size() < LARGE_ROOT_THREAD_NUM,
        HCCL_ERROR("Balanced direct Scatter needs %u root threads, got[%zu]",
            LARGE_ROOT_THREAD_NUM, resCtx.threads.size()),
        HCCL_E_INTERNAL);

    // 后同步沿同一棵树反向汇聚。每个worker先等自己的孩子完成，再向父thread记录完成；
    // 主thread最终只需等待1/2/4/8四个直接子树，而不是逐个等待14个worker。
    for (int32_t threadIdx = static_cast<int32_t>(LARGE_ROOT_THREAD_NUM) - 1; threadIdx >= 1; --threadIdx) {
        for (uint32_t childIdx = static_cast<uint32_t>(threadIdx) + 1;
             childIdx < LARGE_ROOT_THREAD_NUM; ++childIdx) {
            if (GetRootThreadParent(childIdx) == static_cast<uint32_t>(threadIdx)) {
                CHK_RET(HcommThreadNotifyWaitOnThread(
                    resCtx.threads[threadIdx], childIdx, CUSTOM_TIMEOUT));
            }
        }

        uint32_t parentIdx = GetRootThreadParent(static_cast<uint32_t>(threadIdx));
        CHK_PRT_RET(parentIdx == INVALID_VALUE_RANKID || parentIdx >= resCtx.threads.size(),
            HCCL_ERROR("Invalid root-worker finish parent, thread[%d], parent[%u]", threadIdx, parentIdx),
            HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyRecordOnThread(
            resCtx.threads[threadIdx], resCtx.threads[parentIdx], static_cast<uint32_t>(threadIdx)));
    }

    ThreadHandle mainThread = resCtx.threads[0];
    for (uint32_t childIdx = 1; childIdx < LARGE_ROOT_THREAD_NUM; ++childIdx) {
        if (GetRootThreadParent(childIdx) == 0) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, childIdx, CUSTOM_TIMEOUT));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecBalancedDirectLarge(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t sliceBytes)
{
    ParallelTopo topo;
    if (!BuildParallelTopo(param, resCtx, topo)) {
        return HCCL_E_NOT_SUPPORT;
    }

    // 按题面带宽关系配平root的两类出口：
    // Mesh负载 = 1 + a；root Clos负载 = (8 - 7a) / 4。
    // 令二者相等得到 a = 4/11。a部分交给本机7个relay的Clos，剩余7/11由root自己的Clos直写。
    uint64_t relayBytes = (sliceBytes / RELAY_RATIO_DENOMINATOR) * RELAY_RATIO_NUMERATOR;
    relayBytes += ((sliceBytes % RELAY_RATIO_DENOMINATOR) * RELAY_RATIO_NUMERATOR) /
        RELAY_RATIO_DENOMINATOR;
    relayBytes = AlignDown(relayBytes, LARGE_ALIGN);
    if (relayBytes == 0 || relayBytes >= sliceBytes) {
        return HCCL_E_NOT_SUPPORT;
    }
    const uint64_t directBytes = sliceBytes - relayBytes;

    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *localCcl = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);

    if (param.myRank == param.root) {
        CHK_PRT_RET(resCtx.threads.size() < LARGE_ROOT_THREAD_NUM,
            HCCL_ERROR("Root thread count[%zu] is insufficient for balanced direct Scatter", resCtx.threads.size()),
            HCCL_E_INTERNAL);
        CHK_RET(StartRootWorkersTree(resCtx));

        // 7条本机Mesh：先把配对远端rank的4/11写入本机relay的HCCL Buffer，
        // 随后同一channel继续直写该本机rank自己的完整recvBuf。
        // 两次Write分别使用notify 0/1，relay可以在第二次Mesh Write进行时并发跨Clos转发第一段。
        for (uint32_t idx = 0; idx < SERVER_RANK_SIZE; ++idx) {
            uint32_t localRank = (*topo.rootGroup)[idx];
            if (localRank == param.root) {
                continue;
            }

            uint32_t counterpart = (*topo.remoteGroup)[idx];
            uint32_t threadIdx = GetRootPeerThreadIndex(param, topo, localRank);
            CHK_PRT_RET(threadIdx == INVALID_VALUE_RANKID || threadIdx >= resCtx.threads.size(),
                HCCL_ERROR("Cannot map local rank[%u] to a root worker thread", localRank), HCCL_E_INTERNAL);

            const ChannelInfo *localChannel = GetChannel(resCtx, localRank);
            CHK_PRT_RET(localChannel == nullptr || localChannel->remoteCclMem.size < relayBytes,
                HCCL_ERROR("Relay HCCL buffer is too small on rank[%u]", localRank), HCCL_E_INTERNAL);
            CHK_PRT_RET(localChannel->hasRemoteUserOutput == 0 ||
                    localChannel->remoteUserOutput.addr == nullptr ||
                    localChannel->remoteUserOutput.size < sliceBytes,
                HCCL_ERROR("Remote recvBuf is unavailable for local rank[%u]", localRank), HCCL_E_INTERNAL);

            const uint8_t *relaySource = input + static_cast<uint64_t>(counterpart) * sliceBytes;
            CHK_RET(WriteWithNotify(resCtx.threads[threadIdx], localChannel,
                localChannel->remoteCclMem.addr, relaySource, relayBytes, NOTIFY_IDX_ACK));

            const uint8_t *localSource = input + static_cast<uint64_t>(localRank) * sliceBytes;
            CHK_RET(WriteWithNotify(resCtx.threads[threadIdx], localChannel,
                localChannel->remoteUserOutput.addr, localSource, sliceBytes, NOTIFY_IDX_DATA_SIGNAL));
        }

        // root自己的Clos：remoteRoot拿完整slice；其余7个远端rank只拿7/11后半段。
        // 8条跨机channel分别落在独立AICPU_TS thread上，让单卡Clos聚合带宽有机会被吃满。
        for (uint32_t idx = 0; idx < SERVER_RANK_SIZE; ++idx) {
            uint32_t remoteRank = (*topo.remoteGroup)[idx];
            const ChannelInfo *remoteChannel = GetChannel(resCtx, remoteRank);
            CHK_PRT_RET(remoteChannel == nullptr || remoteChannel->hasRemoteUserOutput == 0 ||
                    remoteChannel->remoteUserOutput.addr == nullptr ||
                    remoteChannel->remoteUserOutput.size < sliceBytes,
                HCCL_ERROR("Remote recvBuf is unavailable for remote rank[%u]", remoteRank), HCCL_E_INTERNAL);

            uint32_t threadIdx = GetRootPeerThreadIndex(param, topo, remoteRank);
            CHK_PRT_RET(threadIdx == INVALID_VALUE_RANKID || threadIdx >= resCtx.threads.size(),
                HCCL_ERROR("Cannot map remote rank[%u] to a root thread", remoteRank), HCCL_E_INTERNAL);

            const uint8_t *source = input + static_cast<uint64_t>(remoteRank) * sliceBytes;
            if (remoteRank == topo.remoteRoot) {
                CHK_RET(WriteWithNotify(resCtx.threads[threadIdx], remoteChannel,
                    remoteChannel->remoteUserOutput.addr, source, sliceBytes, NOTIFY_IDX_ACK));
            } else {
                CHK_RET(WriteWithNotify(resCtx.threads[threadIdx], remoteChannel,
                    AddOffset(remoteChannel->remoteUserOutput.addr, relayBytes),
                    source + relayBytes, directBytes, NOTIFY_IDX_ACK));
                }
        }

        // root本地结果无需经过HCCL Buffer。main thread在发起remoteRoot Clos Write后完成本地copy，
        // 再等待其它14条worker thread收尾，保证Host返回前root sendBuf已不再被任何Write读取。
        CHK_RET(HcommLocalCopyOnThread(mainThread, output,
            input + static_cast<uint64_t>(param.root) * sliceBytes, sliceBytes));
        CHK_RET(FinishRootWorkersTree(resCtx));
        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel = GetChannel(resCtx, param.root);
    CHK_PRT_RET(rootChannel == nullptr, HCCL_ERROR("Root channel is unavailable"), HCCL_E_INTERNAL);

    if (topo.onRootServer) {
        // 本机relay收到两次root Write：notify0表示远端A段已落到本地CCL Buffer，
        // notify1表示本rank自己的完整输出已写好。A段在worker上跨Clos直写到配对rank输出。
        CHK_PRT_RET(resCtx.threads.size() < 2,
            HCCL_ERROR("Relay rank needs two AICPU threads"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < relayBytes,
            HCCL_ERROR("Local relay buffer is too small"), HCCL_E_INTERNAL);

        uint32_t counterpart = (*topo.remoteGroup)[topo.myIndex];
        const ChannelInfo *counterpartChannel = GetChannel(resCtx, counterpart);
        CHK_PRT_RET(counterpartChannel == nullptr || counterpartChannel->hasRemoteUserOutput == 0 ||
                counterpartChannel->remoteUserOutput.addr == nullptr ||
                counterpartChannel->remoteUserOutput.size < sliceBytes,
            HCCL_ERROR("Counterpart recvBuf is unavailable for rank[%u]", counterpart), HCCL_E_INTERNAL);

        // V5去掉relay本地的辅助thread：ACK0到达后直接在main thread发起Clos转发，
        // 随后再等待root的第二次Mesh写完成。第二次Mesh写由root侧独立发起，
        // 因而可在本rank执行Clos转发期间并行进行，不需要额外Thread Notify来“等待并行”。
        CHK_RET(WaitChannelNotify(mainThread, rootChannel, NOTIFY_IDX_ACK));
        CHK_RET(WriteWithNotify(mainThread, counterpartChannel,
            counterpartChannel->remoteUserOutput.addr, localCcl, relayBytes, NOTIFY_IDX_ACK));
        CHK_RET(WaitChannelNotify(mainThread, rootChannel, NOTIFY_IDX_DATA_SIGNAL));
        return HCCL_SUCCESS;
    }

    if (param.myRank == topo.remoteRoot) {
        // remoteRoot的整个slice由root自己的Clos直接写入；它不再承担V1中的第二次Mesh扇出。
        CHK_RET(WaitChannelNotify(mainThread, rootChannel, NOTIFY_IDX_ACK));
        return HCCL_SUCCESS;
    }

    // 普通远端rank的两段数据仍由root和配对relay并行写入两个不重叠区域。
    // V5只用main thread顺序消费两个完成Notify：等待动作本身不会串行远端已经发起的数据传输，
    // 因而可删掉worker前/后同步的固定控制开销。
    uint32_t sourceRelay = (*topo.rootGroup)[topo.myIndex];
    const ChannelInfo *relayChannel = GetChannel(resCtx, sourceRelay);
    CHK_PRT_RET(relayChannel == nullptr, HCCL_ERROR("Relay channel is unavailable"), HCCL_E_INTERNAL);

    CHK_RET(WaitChannelNotify(mainThread, rootChannel, NOTIFY_IDX_ACK));
    CHK_RET(WaitChannelNotify(mainThread, relayChannel, NOTIFY_IDX_ACK));
    return HCCL_SUCCESS;
}

HcclResult StageRootHalf(const OpParam &param, ThreadHandle thread, uint8_t *localCcl,
    const std::vector<uint32_t> &rootGroup, const std::vector<uint32_t> &remoteGroup, uint32_t groupSelector,
    uint64_t sliceBytes, uint64_t offset, uint64_t halfBytes)
{
    const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
    const uint64_t bBase = TWO_SERVER_RANK_SIZE * halfBytes;
    const std::vector<uint32_t> &group = (groupSelector == 0) ? rootGroup : remoteGroup;

    for (uint32_t idx = 0; idx < SERVER_RANK_SIZE; ++idx) {
        uint32_t rank = group[idx];
        uint64_t aDstOffset = (static_cast<uint64_t>(idx) * 2 + groupSelector) * halfBytes;
        uint64_t bDstOffset = bBase + (static_cast<uint64_t>(groupSelector) * SERVER_RANK_SIZE + idx) * halfBytes;
        const uint8_t *src = input + static_cast<uint64_t>(rank) * sliceBytes + offset;
        CHK_RET(HcommLocalCopyOnThread(thread, localCcl + aDstOffset, src, halfBytes));
        CHK_RET(HcommLocalCopyOnThread(thread, localCcl + bDstOffset, src + halfBytes, halfBytes));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecParallelLarge(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t sliceBytes)
{
    ParallelTopo topo;
    if (!BuildParallelTopo(param, resCtx, topo) || resCtx.threads.size() < 2) {
        return HCCL_E_NOT_SUPPORT;
    }

    uint64_t minBufferSize = GetMinCclBufferSize(resCtx, param.myRank);
    uint64_t chunkBytes = AlignDown(minBufferSize / TWO_SERVER_RANK_SIZE, LARGE_ALIGN);
    if (chunkBytes < LARGE_ALIGN) {
        return HCCL_E_NOT_SUPPORT;
    }

    // 主路径始终256B对齐并50/50切成Mesh-first与Clos-first两半；最后非对齐尾巴单独走Direct。
    uint64_t alignedBytes = AlignDown(sliceBytes, LARGE_ALIGN);
    uint64_t tailBytes = sliceBytes - alignedBytes;

    HcclResult ret = HCCL_SUCCESS;
    if (alignedBytes > 0) {
        if (param.myRank == param.root) {
            // 两条thread上的任务需要都先排入图；worker先排等待/拷贝，main再排触发与通信。
            CHK_RET(StartWorker(resCtx.threads[0], resCtx.threads[1]));
            // StartWorker已经排入一次worker wait；为避免重复start，这里直接构造worker与main主体。
            // 手工展开root路径，保证两个thread的依赖清晰。
            ThreadHandle mainThread = resCtx.threads[0];
            ThreadHandle workerThread = resCtx.threads[1];
            uint8_t *localCcl = static_cast<uint8_t *>(resCtx.localBuffer.addr);
            uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
            const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);

            // 先把worker整条循环排队。
            for (uint64_t offset = 0; offset < alignedBytes; offset += chunkBytes) {
                uint64_t current = std::min(chunkBytes, alignedBytes - offset);
                uint64_t half = current / 2;
                CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, WORKER_PHASE_NOTIFY, CUSTOM_TIMEOUT));
                CHK_RET(StageRootHalf(param, workerThread, localCcl, *topo.rootGroup, *topo.remoteGroup, 1,
                    sliceBytes, offset, half));
                CHK_RET(HcommThreadNotifyRecordOnThread(workerThread, mainThread, MAIN_WORKER_DONE_NOTIFY));
            }

            for (uint64_t offset = 0; offset < alignedBytes; offset += chunkBytes) {
                uint64_t current = std::min(chunkBytes, alignedBytes - offset);
                uint64_t half = current / 2;
                CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, workerThread, WORKER_PHASE_NOTIFY));
                CHK_RET(StageRootHalf(param, mainThread, localCcl, *topo.rootGroup, *topo.remoteGroup, 0,
                    sliceBytes, offset, half));
                CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, MAIN_WORKER_DONE_NOTIFY, CUSTOM_TIMEOUT));

                for (uint32_t idx = 0; idx < SERVER_RANK_SIZE; ++idx) {
                    uint32_t rank = (*topo.rootGroup)[idx];
                    if (rank != param.root) {
                        CHK_RET(RecordChannel(mainThread, GetChannel(resCtx, rank)));
                    }
                }
                CHK_RET(RecordChannel(mainThread, GetChannel(resCtx, topo.remoteRoot)));
                CHK_RET(HcommLocalCopyOnThread(mainThread, output + offset,
                    input + static_cast<uint64_t>(param.root) * sliceBytes + offset, current));
                for (uint32_t idx = 0; idx < SERVER_RANK_SIZE; ++idx) {
                    uint32_t rank = (*topo.rootGroup)[idx];
                    if (rank != param.root) {
                        CHK_RET(WaitDone(mainThread, GetChannel(resCtx, rank)));
                    }
                }
                CHK_RET(WaitDone(mainThread, GetChannel(resCtx, topo.remoteRoot)));

                for (uint32_t idx = 0; idx < SERVER_RANK_SIZE; ++idx) {
                    uint32_t rank = (*topo.rootGroup)[idx];
                    if (rank != param.root) {
                        CHK_RET(RecordChannel(mainThread, GetChannel(resCtx, rank)));
                    }
                }
                CHK_RET(RecordChannel(mainThread, GetChannel(resCtx, topo.remoteRoot)));
                for (uint32_t idx = 0; idx < SERVER_RANK_SIZE; ++idx) {
                    uint32_t rank = (*topo.rootGroup)[idx];
                    if (rank != param.root) {
                        CHK_RET(WaitDone(mainThread, GetChannel(resCtx, rank)));
                    }
                }
                CHK_RET(WaitDone(mainThread, GetChannel(resCtx, topo.remoteRoot)));
            }
        } else if (topo.onRootServer) {
            // root-server relay: worker跨Clos转发A，main从root收A/B。
            ThreadHandle mainThread = resCtx.threads[0];
            ThreadHandle workerThread = resCtx.threads[1];
            uint8_t *localCcl = static_cast<uint8_t *>(resCtx.localBuffer.addr);
            uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
            const ChannelInfo *rootChannel = GetChannel(resCtx, param.root);
            uint32_t counterpart = (*topo.remoteGroup)[topo.myIndex];
            const ChannelInfo *counterpartChannel = GetChannel(resCtx, counterpart);

            CHK_RET(StartWorker(mainThread, workerThread));
            for (uint64_t offset = 0; offset < alignedBytes; offset += chunkBytes) {
                CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, WORKER_PHASE_NOTIFY, CUSTOM_TIMEOUT));
                CHK_RET(RecordChannel(workerThread, counterpartChannel));
                CHK_RET(WaitDone(workerThread, counterpartChannel));
                CHK_RET(HcommThreadNotifyRecordOnThread(workerThread, mainThread, MAIN_WORKER_DONE_NOTIFY));
            }
            for (uint64_t offset = 0; offset < alignedBytes; offset += chunkBytes) {
                uint64_t current = std::min(chunkBytes, alignedBytes - offset);
                uint64_t half = current / 2;
                uint64_t rootPairOffset = static_cast<uint64_t>(topo.myIndex) * 2 * half;
                uint64_t rootBOffset = TWO_SERVER_RANK_SIZE * half + static_cast<uint64_t>(topo.myIndex) * half;
                CHK_RET(WaitChannel(mainThread, rootChannel));
                CHK_RET(HcommReadOnThread(mainThread, rootChannel->handle, localCcl,
                    AddOffset(rootChannel->remoteCclMem.addr, rootPairOffset), 2 * half));
                CHK_RET(HcommLocalCopyOnThread(mainThread, output + offset, localCcl, half));
                CHK_RET(RecordDone(mainThread, rootChannel));
                CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, workerThread, WORKER_PHASE_NOTIFY));
                CHK_RET(WaitChannel(mainThread, rootChannel));
                CHK_RET(HcommReadOnThread(mainThread, rootChannel->handle, output + offset + half,
                    AddOffset(rootChannel->remoteCclMem.addr, rootBOffset), half));
                CHK_RET(RecordDone(mainThread, rootChannel));
                CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, MAIN_WORKER_DONE_NOTIFY, CUSTOM_TIMEOUT));
            }
        } else if (param.myRank == topo.remoteRoot) {
            ThreadHandle mainThread = resCtx.threads[0];
            ThreadHandle workerThread = resCtx.threads[1];
            uint8_t *localCcl = static_cast<uint8_t *>(resCtx.localBuffer.addr);
            uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
            const ChannelInfo *rootChannel = GetChannel(resCtx, param.root);

            CHK_RET(StartWorker(mainThread, workerThread));
            // worker只使用本Server的7条Mesh channel；root-root Clos channel始终留在main上，避免同一channel跨thread。
            for (uint64_t offset = 0; offset < alignedBytes; offset += chunkBytes) {
                CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, WORKER_PHASE_NOTIFY, CUSTOM_TIMEOUT));
                for (uint32_t idx = 0; idx < SERVER_RANK_SIZE; ++idx) {
                    uint32_t rank = (*topo.remoteGroup)[idx];
                    if (rank != param.myRank) {
                        CHK_RET(RecordChannel(workerThread, GetChannel(resCtx, rank)));
                    }
                }
                for (uint32_t idx = 0; idx < SERVER_RANK_SIZE; ++idx) {
                    uint32_t rank = (*topo.remoteGroup)[idx];
                    if (rank != param.myRank) {
                        CHK_RET(WaitDone(workerThread, GetChannel(resCtx, rank)));
                    }
                }
                CHK_RET(HcommThreadNotifyRecordOnThread(workerThread, mainThread, MAIN_WORKER_DONE_NOTIFY));
            }
            for (uint64_t offset = 0; offset < alignedBytes; offset += chunkBytes) {
                uint64_t current = std::min(chunkBytes, alignedBytes - offset);
                uint64_t half = current / 2;
                uint64_t rootBRemoteOffset = TWO_SERVER_RANK_SIZE * half + SERVER_RANK_SIZE * half;
                uint64_t rootARemoteOffset = (static_cast<uint64_t>(topo.rootIndex) * 2 + 1) * half;

                // Phase 1: 先从root经Clos取整个remote Server的B块。
                CHK_RET(WaitChannel(mainThread, rootChannel));
                CHK_RET(HcommReadOnThread(mainThread, rootChannel->handle, localCcl,
                    AddOffset(rootChannel->remoteCclMem.addr, rootBRemoteOffset), SERVER_RANK_SIZE * half));
                CHK_RET(RecordDone(mainThread, rootChannel));
                CHK_RET(HcommLocalCopyOnThread(
                    mainThread, output + offset + half, localCcl + static_cast<uint64_t>(topo.myIndex) * half, half));

                // B准备好后释放worker在7条Mesh上扇出；main同时在root-root Clos上取自己的A。
                CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, workerThread, WORKER_PHASE_NOTIFY));
                CHK_RET(WaitChannel(mainThread, rootChannel));
                CHK_RET(HcommReadOnThread(mainThread, rootChannel->handle, output + offset,
                    AddOffset(rootChannel->remoteCclMem.addr, rootARemoteOffset), half));
                CHK_RET(RecordDone(mainThread, rootChannel));

                CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, MAIN_WORKER_DONE_NOTIFY, CUSTOM_TIMEOUT));
            }
        } else {
            // remote-server普通rank：main从remoteRoot经Mesh读B，worker从配对relay经Clos读A。
            ThreadHandle mainThread = resCtx.threads[0];
            ThreadHandle workerThread = resCtx.threads[1];
            uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
            const ChannelInfo *remoteRootChannel = GetChannel(resCtx, topo.remoteRoot);
            uint32_t sourceRelay = (*topo.rootGroup)[topo.myIndex];
            const ChannelInfo *relayChannel = GetChannel(resCtx, sourceRelay);

            CHK_RET(StartWorker(mainThread, workerThread));
            for (uint64_t offset = 0; offset < alignedBytes; offset += chunkBytes) {
                uint64_t current = std::min(chunkBytes, alignedBytes - offset);
                uint64_t half = current / 2;
                CHK_RET(WaitChannel(workerThread, relayChannel));
                CHK_RET(HcommReadOnThread(workerThread, relayChannel->handle, output + offset,
                    AddOffset(relayChannel->remoteCclMem.addr, half), half));
                CHK_RET(RecordDone(workerThread, relayChannel));
            }
            CHK_RET(HcommThreadNotifyRecordOnThread(workerThread, mainThread, MAIN_WORKER_DONE_NOTIFY));

            for (uint64_t offset = 0; offset < alignedBytes; offset += chunkBytes) {
                uint64_t current = std::min(chunkBytes, alignedBytes - offset);
                uint64_t half = current / 2;
                CHK_RET(WaitChannel(mainThread, remoteRootChannel));
                CHK_RET(HcommReadOnThread(mainThread, remoteRootChannel->handle, output + offset + half,
                    AddOffset(remoteRootChannel->remoteCclMem.addr, static_cast<uint64_t>(topo.myIndex) * half),
                    half));
                CHK_RET(RecordDone(mainThread, remoteRootChannel));
            }
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, MAIN_WORKER_DONE_NOTIFY, CUSTOM_TIMEOUT));
        }
    }

    if (tailBytes > 0) {
        ret = RunDirectRegion(param, resCtx, alignedBytes, tailBytes, false);
        CHK_RET(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ExecGenericChunked(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t sliceBytes)
{
    ThreadHandle mainThread = resCtx.threads[0];
    uint64_t minBufferSize = GetMinCclBufferSize(resCtx, param.myRank);
    uint64_t slotBytes = AlignDown(minBufferSize / std::max<uint32_t>(1, param.rankSize), 128);
    CHK_PRT_RET(slotBytes == 0, HCCL_ERROR("HCCL buffer is too small for scatter"), HCCL_E_INTERNAL);

    const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCcl = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const ChannelInfo *rootChannel = GetChannel(resCtx, param.root);

    if (param.rankSize == 1) {
        CHK_RET(HcommLocalCopyOnThread(mainThread, output, input, sliceBytes));
        return HCCL_SUCCESS;
    }

    for (uint64_t offset = 0; offset < sliceBytes; offset += slotBytes) {
        uint64_t len = std::min(slotBytes, sliceBytes - offset);
        if (param.myRank == param.root) {
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                CHK_RET(HcommLocalCopyOnThread(mainThread, localCcl + static_cast<uint64_t>(rank) * slotBytes,
                    input + static_cast<uint64_t>(rank) * sliceBytes + offset, len));
            }
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (rank != param.root) {
                    CHK_RET(RecordChannel(mainThread, GetChannel(resCtx, rank)));
                }
            }
            CHK_RET(HcommLocalCopyOnThread(mainThread, output + offset,
                input + static_cast<uint64_t>(param.root) * sliceBytes + offset, len));
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (rank != param.root) {
                    CHK_RET(WaitDone(mainThread, GetChannel(resCtx, rank)));
                }
            }
        } else {
            CHK_RET(WaitChannel(mainThread, rootChannel));
            CHK_RET(HcommReadOnThread(mainThread, rootChannel->handle, output + offset,
                AddOffset(rootChannel->remoteCclMem.addr, static_cast<uint64_t>(param.myRank) * slotBytes), len));
            CHK_RET(RecordDone(mainThread, rootChannel));
        }
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing independent Scatter v5, rank[%u], root[%u], rankSize[%u], count[%lu]", param.myRank,
        param.root, param.rankSize, param.count);

    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No AICPU thread resource"), HCCL_E_INTERNAL);

    const uint64_t sliceBytes = param.count * sizeof(float);
    if (sliceBytes == 0) {
        return HCCL_SUCCESS;
    }

    // V4小包：数据仍直接从注册后的root sendBuf读取，但ready/done不再由root对15个rank做星型串行同步。
    // 改为“跨Server一跳 + 每Server内部binomial tree”扩散/汇聚同步，压缩小包的控制任务关键路径。
    if (sliceBytes <= SMALL_SLICE_THRESHOLD) {
        HcclResult smallRet = ExecTreeRemoteInputSmall(param, resCtx, sliceBytes);
        if (smallRet == HCCL_SUCCESS) {
            return HCCL_SUCCESS;
        }
        if (smallRet != HCCL_E_NOT_SUPPORT) {
            return smallRet;
        }
        return RunRemoteInputSmall(param, resCtx, sliceBytes);
    }

    // V5保留V2的4/11 relay + 7/11数据配平，只瘦身AICPU_TS控制面：root worker树形前/后同步，非root删冗余辅助thread。
    HcclResult ret = ExecBalancedDirectLarge(param, resCtx, sliceBytes);
    if (ret == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }
    if (ret != HCCL_E_NOT_SUPPORT) {
        return ret;
    }

    // 仅当拓扑/尺寸不满足V2时保留V1的双维HCCL-Buffer算法作为功能回退。
    ret = ExecParallelLarge(param, resCtx, sliceBytes);
    if (ret == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }
    if (ret != HCCL_E_NOT_SUPPORT) {
        return ret;
    }
    HCCL_WARNING("Optimized 2x8 paths unavailable; use generic chunked Scatter");
    return ExecGenericChunked(param, resCtx, sliceBytes);
}
} // namespace ops_hccl
