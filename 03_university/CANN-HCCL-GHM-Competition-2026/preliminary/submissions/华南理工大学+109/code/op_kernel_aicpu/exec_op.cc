/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace {

// 按 remoteRank 查找 Channel
const ChannelInfo *FindChannel(const AlgResourceCtx &ctx, uint32_t remoteRank)
{
    for (const auto &ch : ctx.channels) {
        if (ch.remoteRank == remoteRank) {
            return &ch;
        }
    }
    return nullptr;
}

} // namespace

namespace ops_hccl {

// 512 KiB 用例受时延限制。递归折半仅需进行 4 次对端交换，
// 且从高带宽的跨 Server 维度开始。
HcclResult ExecRecursiveHalving(const OpParam &param, const AlgResourceCtx &resCtx)
{
    constexpr uint32_t kRankNum = 16;
    constexpr uint32_t kScratchBlockCount = 8 + 4 + 2 + 1;
    constexpr uint32_t kDataReadyNotify = 0;
    constexpr uint32_t kConsumedNotify = 1;

    CHK_PRT_RET(param.rankSize != kRankNum || resCtx.threads.empty(),
        HCCL_ERROR("Recursive-halving ReduceScatter requires the 16-rank topology"), HCCL_E_INTERNAL);

    const uint64_t blockBytes = param.count * sizeof(float);
    const uint64_t scratchBytes = kScratchBlockCount * blockBytes;
    CHK_PRT_RET(resCtx.localBuffer.size < scratchBytes,
        HCCL_ERROR("HCCL buffer is too small for recursive-halving ReduceScatter"), HCCL_E_INTERNAL);

    ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *sendBuf = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *recvBuf = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *hcclBuf = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint32_t activeStart = 0;
    uint32_t activeBlocks = kRankNum;
    uint64_t scratchOffset = 0;

    // 掩码 8 对应 rank ^ 8，因此走跨 Server 链路；其余掩码使用 Server 内全互连的 3 个维度。
    // 每个阶段保留当前区间中连续的一半数据，无需额外的打包缓冲区。
    for (uint32_t mask = kRankNum / 2; mask != 0; mask >>= 1) {
        const uint32_t peer = param.myRank ^ mask;
        const ChannelInfo *channel = FindChannel(resCtx, peer);
        CHK_PRT_RET(
            channel == nullptr, HCCL_ERROR("Channel to recursive-halving peer %u not found", peer), HCCL_E_INTERNAL);

        const uint32_t halfBlocks = activeBlocks / 2;
        const bool keepUpperHalf = (param.myRank & mask) != 0;
        const uint32_t keepStart = activeStart + (keepUpperHalf ? halfBlocks : 0);
        const uint32_t sendStart = activeStart + (keepUpperHalf ? 0 : halfBlocks);
        const uint64_t transferElems = static_cast<uint64_t>(halfBlocks) * param.count;
        const uint64_t transferBytes = transferElems * sizeof(float);
        uint8_t *keepPtr = sendBuf + static_cast<uint64_t>(keepStart) * blockBytes;
        uint8_t *sendPtr = sendBuf + static_cast<uint64_t>(sendStart) * blockBytes;
        // sendPtr 指向被丢弃的一半数据，后续阶段不会再修改，可直接写出。
        // HCCL 内存仅用于保存每个阶段接收的数据。
        uint8_t *recvSlot = hcclBuf + scratchOffset;
        const uint64_t remoteRecvOffset = scratchOffset;

        int32_t ret = HcommWriteOnThread(mainThread, channel->handle,
            static_cast<uint8_t *>(channel->remoteCclMem.addr) + remoteRecvOffset, sendPtr, transferBytes);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("rank=%u write for recursive-halving peer=%u failed, ret=%d", param.myRank, peer, ret),
            static_cast<HcclResult>(ret));
        ret = HcommChannelNotifyRecordOnThread(mainThread, channel->handle, kDataReadyNotify);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("rank=%u data notify for recursive-halving peer=%u failed, ret=%d", param.myRank, peer, ret),
            static_cast<HcclResult>(ret));
        ret = HcommChannelNotifyWaitOnThread(mainThread, channel->handle, kDataReadyNotify, CUSTOM_TIMEOUT);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("rank=%u data wait for recursive-halving peer=%u failed, ret=%d", param.myRank, peer, ret),
            static_cast<HcclResult>(ret));
        ret = HcommLocalReduceOnThread(
            mainThread, keepPtr, recvSlot, transferElems, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("rank=%u reduce for recursive-halving mask=%u failed, ret=%d", param.myRank, mask, ret),
            static_cast<HcclResult>(ret));
        ret = HcommChannelNotifyRecordOnThread(mainThread, channel->handle, kConsumedNotify);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("rank=%u ACK for recursive-halving peer=%u failed, ret=%d", param.myRank, peer, ret),
            static_cast<HcclResult>(ret));
        ret = HcommChannelNotifyWaitOnThread(mainThread, channel->handle, kConsumedNotify, CUSTOM_TIMEOUT);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("rank=%u ACK wait for recursive-halving peer=%u failed, ret=%d", param.myRank, peer, ret),
            static_cast<HcclResult>(ret));

        activeStart = keepStart;
        activeBlocks = halfBlocks;
        scratchOffset += transferBytes;
    }

    CHK_PRT_RET(activeStart != param.myRank || activeBlocks != 1,
        HCCL_ERROR("Invalid recursive-halving result interval for rank %u", param.myRank), HCCL_E_INTERNAL);
    int32_t ret = HcommLocalCopyOnThread(
        mainThread, recvBuf, sendBuf + static_cast<uint64_t>(activeStart) * blockBytes, blockBytes);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("rank=%u final recursive-halving copy failed, ret=%d", param.myRank, ret),
        static_cast<HcclResult>(ret));
    return HCCL_SUCCESS;
}

// 直接散播归约使用全部物理链路，仅用于大数据块。
// 对 512 KiB 输入而言，额外的 worker 和 channel 任务开销超过其带宽收益。
HcclResult ExecDirect(const OpParam &param, const AlgResourceCtx &resCtx)
{
    constexpr uint32_t kRankNum = 16;
    constexpr uint32_t kPeerNum = kRankNum - 1;
    constexpr uint32_t kDataReadyNotify = 0;
    constexpr uint32_t kConsumedNotify = 1;
    constexpr uint32_t kWorkerStartNotify = 0;
    constexpr uint32_t kWorkerDoneNotifyBase = 1;
    constexpr uint32_t kGroupReleaseNotify = 1;
    constexpr uint32_t kSlotCount = kPeerNum;
    constexpr uint64_t kGuardBytes = 128;

    CHK_PRT_RET(resCtx.threads.size() < kRankNum, HCCL_ERROR("Insufficient worker threads for direct ReduceScatter"),
        HCCL_E_INTERNAL);

    const uint32_t myRank = param.myRank;
    const uint64_t perRankCount = param.count;
    const uint64_t chunkBytes = perRankCount * sizeof(float);
    uint8_t *sendBuf = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *recvBuf = static_cast<uint8_t *>(param.outputPtr);
    ThreadHandle mainThread = resCtx.threads[0];

    std::vector<uint32_t> peers;
    std::vector<const ChannelInfo *> channels;
    peers.reserve(kPeerNum);
    channels.reserve(kPeerNum);
    for (uint32_t rank = 0; rank < kRankNum; ++rank) {
        if (rank == myRank) {
            continue;
        }
        const ChannelInfo *channel = FindChannel(resCtx, rank);
        CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Channel to rank %u not found", rank), HCCL_E_INTERNAL);
        peers.push_back(rank);
        channels.push_back(channel);
    }

    // 4 个跨 Server worker 同时承担归约任务。其在 Clos 上的写入完成较快，
    // 因此可以提前开始本地归约。
    uint32_t groupWorkerIdx[4] = {};
    bool isGroupWorker[kPeerNum] = {};
    uint32_t groupWorkerCount = 0;
    const uint32_t myServer = myRank / 8;
    for (uint32_t peerIdx = 0; peerIdx < kPeerNum && groupWorkerCount < 4; ++peerIdx) {
        if (peers[peerIdx] / 8 != myServer) {
            groupWorkerIdx[groupWorkerCount++] = peerIdx;
            isGroupWorker[peerIdx] = true;
        }
    }
    CHK_PRT_RET(groupWorkerCount != 4, HCCL_ERROR("Insufficient cross-server workers for direct ReduceScatter"),
        HCCL_E_INTERNAL);

    // 将包含本 rank 的分组固定为最终累加器，其部分结果存放在 resultPtr 中，
    // 从而避免每个 rank 都需要将 CCL 部分结果拷贝到结果块。
    uint32_t groupOrder[4] = {myRank / 4, 0, 0, 0};
    uint32_t groupOrderCount = 1;
    for (uint32_t group = 0; group < 4; ++group) {
        if (group != groupOrder[0]) {
            groupOrder[groupOrderCount++] = group;
        }
    }

    uint8_t *hcclBuf = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const uint64_t slotStride = resCtx.localBuffer.size / kSlotCount;
    CHK_PRT_RET(
        slotStride <= kGuardBytes, HCCL_ERROR("HCCL buffer is too small for direct ReduceScatter"), HCCL_E_INTERNAL);
    const uint64_t maxSubBytes = slotStride - kGuardBytes;
    const uint64_t numSubChunks = (chunkBytes + maxSubBytes - 1) / maxSubBytes;
    const uint64_t subChunkFloats = (perRankCount + numSubChunks - 1) / numSubChunks;
    uint8_t *recvSlots = hcclBuf;
    uint8_t *resultPtr = sendBuf + static_cast<uint64_t>(myRank) * chunkBytes;

    // 每个 worker 负责一条目的链路。HcommWrite 直接从 sendBuf 读取数据，
    // 将全部 HCCL 缓冲区容量留给 15 个接收槽位。
    for (uint32_t peerIdx = 0; peerIdx < kPeerNum; ++peerIdx) {
        ThreadHandle worker = resCtx.threads[peerIdx + 1];
        int32_t ret = HcommThreadNotifyRecordOnThread(mainThread, worker, kWorkerStartNotify);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("rank=%u worker=%u start notify failed, ret=%d", myRank, peerIdx, ret),
            static_cast<HcclResult>(ret));
        ret = HcommThreadNotifyWaitOnThread(worker, kWorkerStartNotify, CUSTOM_TIMEOUT);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("rank=%u worker=%u start wait failed, ret=%d", myRank, peerIdx, ret),
            static_cast<HcclResult>(ret));
    }

    for (uint64_t sub = 0; sub < numSubChunks; ++sub) {
        const uint64_t subOffset = sub * subChunkFloats;
        const uint64_t subCount = std::min(subChunkFloats, perRankCount - subOffset);
        const uint64_t subBytes = subCount * sizeof(float);

        for (uint32_t peerIdx = 0; peerIdx < kPeerNum; ++peerIdx) {
            const uint32_t targetRank = peers[peerIdx];
            const ChannelInfo *channel = channels[peerIdx];
            ThreadHandle worker = resCtx.threads[peerIdx + 1];
            uint8_t *inputSubPtr = sendBuf + static_cast<uint64_t>(targetRank) * chunkBytes + subOffset * sizeof(float);
            // 在 targetRank 上，槽位按源 rank 排序，并跳过 targetRank 自身。
            const uint32_t sourceSlot = myRank < targetRank ? myRank : myRank - 1;
            uint8_t *remoteRecvSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr)
                                      + static_cast<uint64_t>(recvSlots - hcclBuf) + sourceSlot * slotStride;

            int32_t ret = HcommWriteOnThread(worker, channel->handle, remoteRecvSlot, inputSubPtr, subBytes);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("rank=%u write to target=%u failed, ret=%d", myRank, targetRank, ret),
                static_cast<HcclResult>(ret));
            ret = HcommChannelNotifyRecordOnThread(worker, channel->handle, kDataReadyNotify);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("rank=%u data notify to target=%u failed, ret=%d", myRank, targetRank, ret),
                static_cast<HcclResult>(ret));
        }

        // 每个分组 worker 在本组源数据到达后立即继续，无需等待全部 15 条链路。
        // 本地源分组直接累加到 resultSubPtr，其他分组以首个远端叶子节点作为累加器，
        // 并在主线程消费前保持该数据不被覆盖。
        uint8_t *resultSubPtr = resultPtr + subOffset * sizeof(float);
        uint8_t *groupPartials[4] = {};
        const ChannelInfo *groupSeedChannels[4] = {};
        for (uint32_t groupPos = 0; groupPos < 4; ++groupPos) {
            const uint32_t sourceGroup = groupOrder[groupPos];
            const uint32_t workerIdx = groupWorkerIdx[groupPos];
            ThreadHandle worker = resCtx.threads[workerIdx + 1];
            if (groupPos == 0) {
                groupPartials[groupPos] = resultSubPtr;
            }

            for (uint32_t sourceRank = sourceGroup * 4; sourceRank < (sourceGroup + 1) * 4; ++sourceRank) {
                uint8_t *leaf = nullptr;
                const ChannelInfo *sourceChannel = nullptr;
                if (sourceRank == myRank) {
                    leaf = resultSubPtr;
                } else {
                    const uint32_t sourceSlot = sourceRank < myRank ? sourceRank : sourceRank - 1;
                    sourceChannel = channels[sourceSlot];
                    leaf = recvSlots + static_cast<uint64_t>(sourceSlot) * slotStride;
                    int32_t ret = HcommChannelNotifyWaitOnThread(
                        worker, sourceChannel->handle, kDataReadyNotify, CUSTOM_TIMEOUT);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("rank=%u group=%u data wait from source=%u failed, ret=%d", myRank, sourceGroup,
                            sourceRank, ret),
                        static_cast<HcclResult>(ret));
                }

                if (groupPos == 0 && sourceRank == myRank) {
                    continue;
                }
                if (groupPos != 0 && sourceRank == sourceGroup * 4) {
                    groupPartials[groupPos] = leaf;
                    groupSeedChannels[groupPos] = sourceChannel;
                    continue;
                }

                int32_t ret = HcommLocalReduceOnThread(
                    worker, groupPartials[groupPos], leaf, subCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "rank=%u group reduce failed, group=%u source=%u ret=%d", myRank, sourceGroup, sourceRank, ret),
                    static_cast<HcclResult>(ret));

                // 此叶子节点在归约后不再被引用，立即释放该源以处理下一个子块。
                if (sourceChannel != nullptr && sub + 1 < numSubChunks) {
                    ret = HcommChannelNotifyRecordOnThread(worker, sourceChannel->handle, kConsumedNotify);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR(
                            "rank=%u group=%u ACK to source=%u failed, ret=%d", myRank, sourceGroup, sourceRank, ret),
                        static_cast<HcclResult>(ret));
                }
            }
            const uint32_t groupDoneNotify = kWorkerDoneNotifyBase + workerIdx;
            int32_t ret = HcommThreadNotifyRecordOnThread(worker, mainThread, groupDoneNotify);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("rank=%u group completion notify failed, group=%u ret=%d", myRank, sourceGroup, ret),
                static_cast<HcclResult>(ret));
            if (sub + 1 < numSubChunks) {
                ret = HcommThreadNotifyWaitOnThread(worker, kGroupReleaseNotify, CUSTOM_TIMEOUT);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("rank=%u group release wait failed, group=%u ret=%d", myRank, sourceGroup, ret),
                    static_cast<HcclResult>(ret));
                if (groupSeedChannels[groupPos] != nullptr) {
                    ret = HcommChannelNotifyRecordOnThread(
                        worker, groupSeedChannels[groupPos]->handle, kConsumedNotify);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("rank=%u group=%u seed ACK failed, ret=%d", myRank, sourceGroup, ret),
                        static_cast<HcclResult>(ret));
                }
            }
        }

        for (uint32_t groupPos = 0; groupPos < 4; ++groupPos) {
            const uint32_t sourceGroup = groupOrder[groupPos];
            const uint32_t workerIdx = groupWorkerIdx[groupPos];
            const uint32_t groupDoneNotify = kWorkerDoneNotifyBase + workerIdx;
            int32_t ret = HcommThreadNotifyWaitOnThread(mainThread, groupDoneNotify, CUSTOM_TIMEOUT);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("rank=%u group completion wait failed, group=%u ret=%d", myRank, sourceGroup, ret),
                static_cast<HcclResult>(ret));
            if (groupPos != 0) {
                ret = HcommLocalReduceOnThread(mainThread, resultSubPtr, groupPartials[groupPos], subCount,
                    HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("rank=%u final group reduce failed, group=%u ret=%d", myRank, sourceGroup, ret),
                    static_cast<HcclResult>(ret));
            }
            if (sub + 1 < numSubChunks) {
                // 本地源分组的下一个子块使用互不重叠的结果区间；远端种子分组
                // 则需在上述最终归约消费种子数据后才释放。
                const ThreadHandle worker = resCtx.threads[workerIdx + 1];
                ret = HcommThreadNotifyRecordOnThread(mainThread, worker, kGroupReleaseNotify);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("rank=%u group release failed, group=%u ret=%d", myRank, sourceGroup, ret),
                    static_cast<HcclResult>(ret));
            }
        }
        int32_t ret = HCCL_SUCCESS;
        if (sub + 1 < numSubChunks) {
            // 每条 channel 在下一次数据就绪 Record 前均需完成确认。
            for (uint32_t peerIdx = 0; peerIdx < kPeerNum; ++peerIdx) {
                ThreadHandle worker = resCtx.threads[peerIdx + 1];
                const ChannelInfo *channel = channels[peerIdx];
                ret = HcommChannelNotifyWaitOnThread(worker, channel->handle, kConsumedNotify, CUSTOM_TIMEOUT);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("rank=%u ACK wait from target=%u failed, ret=%d", myRank, peers[peerIdx], ret),
                    static_cast<HcclResult>(ret));
            }
        }
    }

    for (uint32_t peerIdx = 0; peerIdx < kPeerNum; ++peerIdx) {
        if (isGroupWorker[peerIdx]) {
            // 上一次分组完成等待已同步该 worker，复用其 notify 可避免使用超过平台上限 17 的索引。
            continue;
        }
        ThreadHandle worker = resCtx.threads[peerIdx + 1];
        int32_t ret = HcommThreadNotifyRecordOnThread(worker, mainThread, kWorkerDoneNotifyBase + peerIdx);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("rank=%u worker=%u completion notify failed, ret=%d", myRank, peerIdx, ret),
            static_cast<HcclResult>(ret));
        ret = HcommThreadNotifyWaitOnThread(mainThread, kWorkerDoneNotifyBase + peerIdx, CUSTOM_TIMEOUT);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("rank=%u worker=%u completion wait failed, ret=%d", myRank, peerIdx, ret),
            static_cast<HcclResult>(ret));
    }

    int32_t ret = HcommLocalCopyOnThread(mainThread, recvBuf, resultPtr, chunkBytes);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("rank=%u final copy failed, ret=%d", myRank, ret),
        static_cast<HcclResult>(ret));
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    constexpr uint64_t kSmallInputThresholdBytes = 1ULL << 20;
    const uint64_t perRankBytes = param.count * sizeof(float);
    if (param.rankSize != 0 && perRankBytes <= kSmallInputThresholdBytes / param.rankSize) {
        return ExecRecursiveHalving(param, resCtx);
    }
    return ExecDirect(param, resCtx);
}

} // namespace ops_hccl
