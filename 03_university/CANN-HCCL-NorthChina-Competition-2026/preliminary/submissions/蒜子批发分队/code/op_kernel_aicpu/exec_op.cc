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

#include <algorithm>

namespace ops_hccl {
namespace {
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
constexpr uint64_t SMALL_MESSAGE_MAX_SIZE = 1024 * 1024;
constexpr uint64_t RECURSIVE_DOUBLING_MIN_SIZE = 512 * 1024;
constexpr uint64_t PARALLEL_LOCAL_COPY_MIN_SIZE = 1024 * 1024;
constexpr uint64_t EARLY_LOCAL_COPY_MAX_SIZE = 512ULL * 1024 * 1024;
constexpr uint32_t SMALL_MESSAGE_THREAD_NUM = 8;
constexpr uint32_t RANKS_PER_SERVER = 8;

HcclResult FlatThreadSyncBefore(const std::vector<ThreadHandle> &threads, uint32_t activeThreadNum)
{
    for (uint32_t idx = 1; idx < activeThreadNum; ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(threads[0], threads[idx], 0));
    }
    for (uint32_t idx = 1; idx < activeThreadNum; ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(threads[idx], 0, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult FlatThreadSyncAfter(const std::vector<ThreadHandle> &threads, uint32_t activeThreadNum)
{
    for (uint32_t idx = 1; idx < activeThreadNum; ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(threads[0], idx - 1, CUSTOM_TIMEOUT));
    }
    for (uint32_t idx = 1; idx < activeThreadNum; ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(threads[idx], threads[0], idx - 1));
    }
    return HCCL_SUCCESS;
}

HcclResult TreeThreadSyncBefore(const std::vector<ThreadHandle> &threads, uint32_t activeThreadNum)
{
    // 树形广播主 Thread 的输入就绪依赖，避免所有 Record 串行堆积在 threads[0]。
    for (uint32_t width = 1; width < activeThreadNum; width <<= 1) {
        for (uint32_t src = 0; src < width && src + width < activeThreadNum; ++src) {
            CHK_RET(HcommThreadNotifyRecordOnThread(threads[src], threads[src + width], 0));
        }
        for (uint32_t src = 0; src < width && src + width < activeThreadNum; ++src) {
            CHK_RET(HcommThreadNotifyWaitOnThread(threads[src + width], 0, CUSTOM_TIMEOUT));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TreeThreadSyncAfter(const std::vector<ThreadHandle> &threads, uint32_t activeThreadNum)
{
    // 树形归并全部通信/拷贝完成事件；peer 对应唯一 Notify，允许各子树并行收敛。
    for (uint32_t stride = 1; stride < activeThreadNum; stride <<= 1) {
        for (uint32_t base = 0; base + stride < activeThreadNum; base += stride << 1) {
            const uint32_t peer = base + stride;
            CHK_RET(HcommThreadNotifyWaitOnThread(threads[base], peer - 1, CUSTOM_TIMEOUT));
        }
        for (uint32_t base = 0; base + stride < activeThreadNum; base += stride << 1) {
            const uint32_t peer = base + stride;
            CHK_RET(HcommThreadNotifyRecordOnThread(threads[peer], threads[base], peer - 1));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BatchReadAndNotify(ThreadHandle thread, ChannelHandle channel, void *dst, void *src,
    uint64_t len, uint32_t notifyIdx)
{
    HcommBatchTransferDesc transferDescs[2] = {};
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_READ;
    transferDescs[0].transferInfo.read.len = len;
    transferDescs[0].transferInfo.read.dst = dst;
    transferDescs[0].transferInfo.read.src = src;
    transferDescs[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    transferDescs[1].transferInfo.notifyRecord.notifyIdx = notifyIdx;
    return static_cast<HcclResult>(HcommBatchTransferOnThread(thread, channel, transferDescs, 2));
}

HcclResult BatchWriteTwoSegmentsWithNotify(
    ThreadHandle thread, ChannelHandle channel,
    void *firstDst, void *firstSrc, void *secondDst, void *secondSrc,
    uint64_t len, uint32_t notifyIdx)
{
    HcommBatchTransferDesc transferDescs[2] = {};
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
    transferDescs[0].transferInfo.write.len = len;
    transferDescs[0].transferInfo.write.dst = firstDst;
    transferDescs[0].transferInfo.write.src = firstSrc;
    transferDescs[1].transType = HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY;
    transferDescs[1].transferInfo.writeWithNotify.len = len;
    transferDescs[1].transferInfo.writeWithNotify.dst = secondDst;
    transferDescs[1].transferInfo.writeWithNotify.src = secondSrc;
    transferDescs[1].transferInfo.writeWithNotify.notifyIdx = notifyIdx;
    return static_cast<HcclResult>(
        HcommBatchTransferOnThread(thread, channel, transferDescs, 2));
}

HcclResult StartLocalCopy(ThreadHandle rootThread, ThreadHandle copyThread,
    void *dst, void *src, uint64_t len)
{
    CHK_RET(HcommThreadNotifyRecordOnThread(rootThread, copyThread, 0));
    CHK_RET(HcommThreadNotifyWaitOnThread(copyThread, 0, CUSTOM_TIMEOUT));
    CHK_RET(HcommLocalCopyOnThread(copyThread, dst, src, len));
    return HCCL_SUCCESS;
}

HcclResult InterRankBarrier(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t notifyIdx)
{
    // 本题固定为 16 rank；recursive doubling 在 4 轮后即可传播全局阶段完成状态。
    for (uint32_t mask = 1; mask < param.rankSize; mask <<= 1) {
        const uint32_t partner = param.myRank ^ mask;
        const uint32_t channelIdx = partner < param.myRank ? partner : partner - 1;
        CHK_PRT_RET(channelIdx >= resCtx.channels.size() ||
                resCtx.channels[channelIdx].remoteRank != partner,
            HCCL_ERROR("Barrier channel to rank %u was not found", partner), HCCL_E_NOT_FOUND);
        const ChannelInfo &partnerChannel = resCtx.channels[channelIdx];
        CHK_RET(HcommChannelNotifyRecordOnThread(
            resCtx.threads[0], partnerChannel.handle, notifyIdx));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            resCtx.threads[0], partnerChannel.handle, notifyIdx, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult RunOneShotCclAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // 借鉴 Broadcast ONE_SHOT_SCATTER_ALLGATHER_512K：每个 rank 天然就是
    // 自己输入块的 owner。先在 CCL Buffer 中一次聚齐 16 个 owner 块，
    // 然后只用一个连续 LocalCopy 写入最终输出，避免 16 次分槽拷贝。
    constexpr uint32_t workerNum = SMALL_MESSAGE_THREAD_NUM;
    auto *localCclBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    auto *localOwnerSlot = localCclBuffer + static_cast<uint64_t>(param.myRank) * dataSize;
    const uint64_t packageSize = static_cast<uint64_t>(param.rankSize) * dataSize;

    CHK_RET(HcommLocalCopyOnThread(
        resCtx.threads[0], localOwnerSlot, param.inputPtr, dataSize));
    CHK_RET(TreeThreadSyncBefore(resCtx.threads, workerNum));

    const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
    const uint32_t localServerBase = param.myRank - localRank;
    const uint32_t remoteServerBase = localServerBase ^ RANKS_PER_SERVER;
    for (uint32_t taskIdx = 0; taskIdx < param.rankSize - 1; ++taskIdx) {
        uint32_t remoteRank = 0;
        if (taskIdx < RANKS_PER_SERVER) {
            // 先让 8 条 worker 各自下发一条 Clos，再由前 7 条执行 Mesh。
            remoteRank = remoteServerBase + taskIdx;
        } else {
            uint32_t peerLocalRank = taskIdx - RANKS_PER_SERVER;
            if (peerLocalRank >= localRank) {
                ++peerLocalRank;
            }
            remoteRank = localServerBase + peerLocalRank;
        }
        const uint32_t channelIdx = remoteRank < param.myRank ? remoteRank : remoteRank - 1;
        CHK_PRT_RET(channelIdx >= resCtx.channels.size() ||
                resCtx.channels[channelIdx].remoteRank != remoteRank,
            HCCL_ERROR("One-shot channel to rank %u was not found", remoteRank),
            HCCL_E_NOT_FOUND);
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        ThreadHandle worker = resCtx.threads[taskIdx % workerNum];
        auto *remoteOwnerSlot = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
            static_cast<uint64_t>(param.myRank) * dataSize;

        CHK_RET(HcommWriteWithNotifyOnThread(worker, channel.handle,
            remoteOwnerSlot, localOwnerSlot, dataSize, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    }

    CHK_RET(TreeThreadSyncAfter(resCtx.threads, workerNum));
    CHK_RET(HcommLocalCopyOnThread(
        resCtx.threads[0], param.outputPtr, localCclBuffer, packageSize));
    return HCCL_SUCCESS;
}

HcclResult GetDirectChannelByRank(
    uint32_t myRank, const AlgResourceCtx &resCtx,
    uint32_t remoteRank, const ChannelInfo *&channel)
{
    const uint32_t channelIdx =
        remoteRank < myRank ? remoteRank : remoteRank - 1;
    CHK_PRT_RET(channelIdx >= resCtx.channels.size() ||
            resCtx.channels[channelIdx].remoteRank != remoteRank,
        HCCL_ERROR("Direct channel to rank %u was not found", remoteRank),
        HCCL_E_NOT_FOUND);
    channel = &resCtx.channels[channelIdx];
    return HCCL_SUCCESS;
}

HcclResult GetDirectChannel(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t remoteRank, const ChannelInfo *&channel)
{
    return GetDirectChannelByRank(
        param.myRank, resCtx, remoteRank, channel);
}

[[maybe_unused]] HcclResult RunCrossMinBatchAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // 分层拓扑路径：x1 在本 Server 形成 1MB pair，x8 只交换一次
    // pair；随后 x2/x4/x6 各用一个双段 Batch，把本/远 Server 的
    // 两个非连续 pair 同时扩散。每 rank 的 Clos 流量仅为 1MB。
    ThreadHandle thread = resCtx.threads[0];
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);

    const uint32_t xor1Rank = param.myRank ^ 1U;
    const uint32_t xor2Rank = param.myRank ^ 2U;
    const uint32_t xor4Rank = param.myRank ^ 4U;
    const uint32_t xor6Rank = param.myRank ^ 6U;
    const uint32_t xor8Rank = param.myRank ^ RANKS_PER_SERVER;
    const ChannelInfo *xor1Channel = nullptr;
    const ChannelInfo *xor2Channel = nullptr;
    const ChannelInfo *xor4Channel = nullptr;
    const ChannelInfo *xor6Channel = nullptr;
    const ChannelInfo *xor8Channel = nullptr;
    CHK_RET(GetDirectChannel(param, resCtx, xor1Rank, xor1Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor2Rank, xor2Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor4Rank, xor4Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor6Rank, xor6Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor8Rank, xor8Channel));

    // x1 直接读取 input；本 rank Copy 位于 WN 后，与首轮 Mesh 重叠。
    auto *remoteSelf =
        static_cast<uint8_t *>(xor1Channel->remoteOutput.addr) +
        static_cast<uint64_t>(param.myRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor1Channel->handle, remoteSelf, param.inputPtr,
        dataSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommLocalCopyOnThread(
        thread,
        outputBuffer + static_cast<uint64_t>(param.myRank) * dataSize,
        param.inputPtr, dataSize));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor1Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    const uint32_t localPairStartRank = param.myRank & ~1U;
    const uint32_t remotePairStartRank =
        localPairStartRank ^ RANKS_PER_SERVER;
    const uint64_t pairSize = dataSize * 2U;
    const uint64_t localPairOffset =
        static_cast<uint64_t>(localPairStartRank) * dataSize;
    const uint64_t remotePairOffset =
        static_cast<uint64_t>(remotePairStartRank) * dataSize;
    auto *localPair = outputBuffer + localPairOffset;

    // 只用同编号 x8 Channel 跨 Server 交换 1MB pair。
    auto *remoteLocalPairOnX8 =
        static_cast<uint8_t *>(xor8Channel->remoteOutput.addr) +
        localPairOffset;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor8Channel->handle, remoteLocalPairOnX8, localPair,
        pairSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor8Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    // x8 Wait 后，本 Server pair 与远端同编号 pair 均已就绪。每条 Mesh
    // Batch 的第二段使用 WRITE_WITH_NOTIFY，强序通知同时证明两段写完成。
    auto *remotePair = outputBuffer + remotePairOffset;
    auto *remoteOutputOnX2 =
        static_cast<uint8_t *>(xor2Channel->remoteOutput.addr);
    auto *remoteOutputOnX4 =
        static_cast<uint8_t *>(xor4Channel->remoteOutput.addr);
    auto *remoteOutputOnX6 =
        static_cast<uint8_t *>(xor6Channel->remoteOutput.addr);
    CHK_RET(BatchWriteTwoSegmentsWithNotify(
        thread, xor2Channel->handle,
        remoteOutputOnX2 + localPairOffset, localPair,
        remoteOutputOnX2 + remotePairOffset, remotePair,
        pairSize, NOTIFY_IDX_ACK));
    CHK_RET(BatchWriteTwoSegmentsWithNotify(
        thread, xor4Channel->handle,
        remoteOutputOnX4 + localPairOffset, localPair,
        remoteOutputOnX4 + remotePairOffset, remotePair,
        pairSize, NOTIFY_IDX_ACK));
    CHK_RET(BatchWriteTwoSegmentsWithNotify(
        thread, xor6Channel->handle,
        remoteOutputOnX6 + localPairOffset, localPair,
        remoteOutputOnX6 + remotePairOffset, remotePair,
        pairSize, NOTIFY_IDX_ACK));

    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor2Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor4Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor6Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunPairMeshFanoutClosAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // 题目拓扑中，每个 NPU 的 Clos 总带宽约为单条 Mesh 链路的 8 倍。
    // 先用 x1 形成连续 1MB pair，再通过三条独立 Mesh Channel 将该
    // pair 并行扩散到本 Server 的其余三个 pair；最后只用一个 x8 WQE
    // 交换连续 4MB Server 半区。相比双段 Batch 路径，数据 WQE 从 8
    // 降为 5，且 Case05 不再创建或解析 BatchTransfer 描述符。
    ThreadHandle thread = resCtx.threads[0];
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);

    const uint32_t xor1Rank = param.myRank ^ 1U;
    const uint32_t xor2Rank = param.myRank ^ 2U;
    const uint32_t xor4Rank = param.myRank ^ 4U;
    const uint32_t xor6Rank = param.myRank ^ 6U;
    const uint32_t xor8Rank = param.myRank ^ RANKS_PER_SERVER;
    const ChannelInfo *xor1Channel = nullptr;
    const ChannelInfo *xor2Channel = nullptr;
    const ChannelInfo *xor4Channel = nullptr;
    const ChannelInfo *xor6Channel = nullptr;
    const ChannelInfo *xor8Channel = nullptr;
    CHK_RET(GetDirectChannel(param, resCtx, xor1Rank, xor1Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor2Rank, xor2Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor4Rank, xor4Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor6Rank, xor6Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor8Rank, xor8Channel));

    // x1 直接读取 input；LocalCopy 位于反向通知 Wait 前，与首轮 Mesh
    // 传输重叠。Wait 完成后，本 rank 的连续 1MB pair 已完整就绪。
    auto *remoteSelf =
        static_cast<uint8_t *>(xor1Channel->remoteOutput.addr) +
        static_cast<uint64_t>(param.myRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor1Channel->handle, remoteSelf, param.inputPtr,
        dataSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommLocalCopyOnThread(
        thread,
        outputBuffer + static_cast<uint64_t>(param.myRank) * dataSize,
        param.inputPtr, dataSize));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor1Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    const uint32_t pairStartRank = param.myRank & ~1U;
    const uint64_t pairOffset =
        static_cast<uint64_t>(pairStartRank) * dataSize;
    const uint64_t pairSize = dataSize * 2U;
    auto *localPair = outputBuffer + pairOffset;
    auto *remoteOutputOnX2 =
        static_cast<uint8_t *>(xor2Channel->remoteOutput.addr);
    auto *remoteOutputOnX4 =
        static_cast<uint8_t *>(xor4Channel->remoteOutput.addr);
    auto *remoteOutputOnX6 =
        static_cast<uint8_t *>(xor6Channel->remoteOutput.addr);

    // 三个 WN 全部位于任一 Wait 之前，保证 x2/x4/x6 三条独立 Mesh
    // Channel 都已投递后才可能阻塞。与本地 pair 一起，它们恰好枚举
    // 本 Server 的四个连续 pair。
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor2Channel->handle, remoteOutputOnX2 + pairOffset,
        localPair, pairSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor4Channel->handle, remoteOutputOnX4 + pairOffset,
        localPair, pairSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor6Channel->handle, remoteOutputOnX6 + pairOffset,
        localPair, pairSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor2Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor4Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor6Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    // 本 Server 连续 4MB 半区完整后，通过高带宽 x8 Clos 一次写入对端
    // 相同的绝对 rank 区间。对端本地半区与该目标区间互不重叠。
    const uint32_t serverStartRank =
        param.myRank & ~(RANKS_PER_SERVER - 1U);
    const uint64_t serverOffset =
        static_cast<uint64_t>(serverStartRank) * dataSize;
    const uint64_t serverSize =
        static_cast<uint64_t>(RANKS_PER_SERVER) * dataSize;
    auto *localServerData = outputBuffer + serverOffset;
    auto *remoteServerData =
        static_cast<uint8_t *>(xor8Channel->remoteOutput.addr) +
        serverOffset;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor8Channel->handle, remoteServerData, localServerData,
        serverSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor8Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult FlushCase05Batch(const OpParam &param);

HcclResult RunHostPreparedCase05(
    const Case05ResourcePlan &plan, uint64_t dataSize)
{
    // 首轮直接读取input；本rank Copy位于WN和Wait之间，与512KB x1传输重叠。
    const Case05StageResource &first = plan.stages[0];
    CHK_RET(HcommWriteWithNotifyOnThread(
        plan.thread, first.channel, first.remoteDst,
        first.localSrc, first.len, NOTIFY_IDX_ACK));
    CHK_RET(HcommLocalCopyOnThread(
        plan.thread, plan.localDst, plan.inputPtr, dataSize));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        plan.thread, first.channel, NOTIFY_IDX_ACK,
        CUSTOM_TIMEOUT));

    // 后三轮交换已聚齐的1/2/4MB连续区间。每轮WN/Wait同时完成数据传播
    // 和阶段就绪证明，无需额外全局完成屏障。
    for (uint32_t stageIdx = 1;
         stageIdx < CASE05_PLAN_STAGE_COUNT; ++stageIdx) {
        const Case05StageResource &stage =
            plan.stages[stageIdx];
        CHK_RET(HcommWriteWithNotifyOnThread(
            plan.thread, stage.channel, stage.remoteDst,
            stage.localSrc, stage.len, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            plan.thread, stage.channel,
            NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunPairedSeedMeshBatchAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // 每个 rank 先只与同 local-rank 的另一个 Server peer 交换一次种子，
    // 得到 {self, self^8} 两个512KB块；再通过本 Server七条Mesh Channel
    // 各提交一个双段Batch，把这两个非连续块同时扩散。相比15路直推，
    // 逻辑数据WQE仍为15个，但高层API任务由31降到17，A5 WQEBB约由30
    // 降到23，并把每rank Clos流量由4MB压缩到512KB。
    constexpr uint32_t meshPeerCount = RANKS_PER_SERVER - 1;
    ThreadHandle thread = resCtx.threads[0];
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);
    const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
    const uint32_t localServerBase = param.myRank - localRank;
    const uint32_t crossRank = param.myRank ^ RANKS_PER_SERVER;
    const uint64_t selfOffset =
        static_cast<uint64_t>(param.myRank) * dataSize;
    const uint64_t crossOffset =
        static_cast<uint64_t>(crossRank) * dataSize;

    const ChannelInfo *crossChannel = nullptr;
    CHK_RET(GetDirectChannel(
        param, resCtx, crossRank, crossChannel));

    // 跨Server种子直接读取输入；本rank Copy排在Wait之前与这次Clos重叠。
    auto *remoteSelf =
        static_cast<uint8_t *>(crossChannel->remoteOutput.addr) +
        selfOffset;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, crossChannel->handle, remoteSelf, param.inputPtr,
        dataSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommLocalCopyOnThread(
        thread, outputBuffer + selfOffset, param.inputPtr, dataSize));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, crossChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    // cross Wait后，output[crossRank]已就绪。七个Batch先全部发布再统一
    // Wait；每个Batch的最后一段WRITE_WITH_NOTIFY同时证明前一段普通
    // WRITE已完成。mask=1..7在Server内形成七轮XOR完美匹配。
    auto *crossBlock = outputBuffer + crossOffset;
    const ChannelInfo *meshChannels[meshPeerCount] = {};
    uint32_t meshCount = 0;
    for (uint32_t mask = 1; mask < RANKS_PER_SERVER; ++mask) {
        const uint32_t remoteRank =
            localServerBase + (localRank ^ mask);
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetDirectChannel(
            param, resCtx, remoteRank, channel));
        meshChannels[meshCount++] = channel;
        auto *remoteOutput =
            static_cast<uint8_t *>(channel->remoteOutput.addr);
        CHK_RET(BatchWriteTwoSegmentsWithNotify(
            thread, channel->handle,
            remoteOutput + selfOffset, param.inputPtr,
            remoteOutput + crossOffset, crossBlock,
            dataSize, NOTIFY_IDX_ACK));
    }
    CHK_PRT_RET(meshCount != meshPeerCount,
        HCCL_ERROR("Unexpected paired-seed mesh count %u", meshCount),
        HCCL_E_INTERNAL);

    for (uint32_t idx = 0; idx < meshCount; ++idx) {
        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, meshChannels[idx]->handle, NOTIFY_IDX_ACK,
            CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunFourStageRecursiveDoublingAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // 16 rank的AllGather至少需要4次信息倍增。本路径用x1/x2/x4/x8
    // 四轮连续区间交换达到该下界：只有4个WRITE_WITH_NOTIFY、4个Wait
    // 和1个LocalCopy，共9个算法API任务、4个数据WQE和约8个A5 WQEBB。
    // 最后一轮4MB数据走高聚合带宽Clos，前三轮留在Server内Mesh。
    ThreadHandle thread = resCtx.threads[0];
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);
    const uint64_t selfOffset =
        static_cast<uint64_t>(param.myRank) * dataSize;

    // 第一轮直接读取input，让首个Doorbell不被本地Copy阻塞。
    const uint32_t xor1Rank = param.myRank ^ 1U;
    const ChannelInfo *xor1Channel = nullptr;
    CHK_RET(GetDirectChannel(
        param, resCtx, xor1Rank, xor1Channel));
    auto *remoteSelf =
        static_cast<uint8_t *>(xor1Channel->remoteOutput.addr) +
        selfOffset;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor1Channel->handle, remoteSelf, param.inputPtr,
        dataSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommLocalCopyOnThread(
        thread, outputBuffer + selfOffset, param.inputPtr, dataSize));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor1Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    // x2/x4/x8依次交换已聚齐的2/4/8个连续rank块。每次Wait都是下一轮
    // 唯一的数据就绪依赖；四条Channel各使用一次Notify，不会跨轮复用。
    for (uint32_t mask = 2; mask <= RANKS_PER_SERVER; mask <<= 1) {
        const uint32_t partner = param.myRank ^ mask;
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetDirectChannel(
            param, resCtx, partner, channel));
        const uint32_t groupStartRank =
            param.myRank & ~(mask - 1U);
        const uint64_t groupOffset =
            static_cast<uint64_t>(groupStartRank) * dataSize;
        const uint64_t groupSize =
            static_cast<uint64_t>(mask) * dataSize;
        auto *remoteGroup =
            static_cast<uint8_t *>(channel->remoteOutput.addr) +
            groupOffset;
        CHK_RET(HcommWriteWithNotifyOnThread(
            thread, channel->handle, remoteGroup,
            outputBuffer + groupOffset, groupSize, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunDualRtsqLinkSplitOneWaveAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // 将上一版单RTSQ的15路直接Push按物理链路拆到两条RTSQ：
    // threads[0]负责8条Clos，threads[1]负责7条Mesh和本地Copy。
    // 只保留一次main->worker输入依赖；完成阶段由AICPU Kernel直接
    // Join两条RTSQ，不再构造worker->main Thread Notify。
    constexpr uint32_t closPeerCount = RANKS_PER_SERVER;
    constexpr uint32_t meshPeerCount = RANKS_PER_SERVER - 1;
    constexpr uint32_t workerStartNotifyIdx = 0;
    ThreadHandle closThread = resCtx.threads[0];
    ThreadHandle meshThread = resCtx.threads[1];
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);
    const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
    const uint32_t localServerBase = param.myRank - localRank;
    const uint32_t remoteServerBase =
        localServerBase ^ RANKS_PER_SERVER;
    const uint64_t selfOffset =
        static_cast<uint64_t>(param.myRank) * dataSize;
    const ChannelInfo *closChannels[closPeerCount] = {};
    const ChannelInfo *meshChannels[meshPeerCount] = {};
    uint32_t closCount = 0;
    uint32_t meshCount = 0;

    // 外层Host就绪Wait位于closThread。将该依赖只广播给一条worker；
    // 后续两条RTSQ均可安全读取用户输入。
    CHK_RET(HcommThreadNotifyRecordOnThread(
        closThread, meshThread, workerStartNotifyIdx));
    CHK_RET(HcommThreadNotifyWaitOnThread(
        meshThread, workerStartNotifyIdx, CUSTOM_TIMEOUT));

    // Clos与Mesh分别按XOR mask错峰，每个位置在16卡上都是一对一匹配。
    // 先把两条RTSQ的所有WN完整发布，再在各自RTSQ上等待完成。
    for (uint32_t mask = 0; mask < RANKS_PER_SERVER; ++mask) {
        const uint32_t remoteRank =
            remoteServerBase + (localRank ^ mask);
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetDirectChannel(
            param, resCtx, remoteRank, channel));
        closChannels[closCount++] = channel;
        auto *remoteSelf =
            static_cast<uint8_t *>(channel->remoteOutput.addr) +
            selfOffset;
        CHK_RET(HcommWriteWithNotifyOnThread(
            closThread, channel->handle, remoteSelf, param.inputPtr,
            dataSize, NOTIFY_IDX_ACK));
    }
    for (uint32_t mask = 1; mask < RANKS_PER_SERVER; ++mask) {
        const uint32_t remoteRank =
            localServerBase + (localRank ^ mask);
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetDirectChannel(
            param, resCtx, remoteRank, channel));
        meshChannels[meshCount++] = channel;
        auto *remoteSelf =
            static_cast<uint8_t *>(channel->remoteOutput.addr) +
            selfOffset;
        CHK_RET(HcommWriteWithNotifyOnThread(
            meshThread, channel->handle, remoteSelf, param.inputPtr,
            dataSize, NOTIFY_IDX_ACK));
    }
    CHK_PRT_RET(closCount != closPeerCount || meshCount != meshPeerCount,
        HCCL_ERROR("Unexpected dual-RTSQ peer count clos=%u mesh=%u",
            closCount, meshCount),
        HCCL_E_INTERNAL);

    // Copy放在负载较轻的Mesh RTSQ，并位于其WN与Wait之间。
    CHK_RET(HcommLocalCopyOnThread(
        meshThread, outputBuffer + selfOffset, param.inputPtr, dataSize));
    for (uint32_t idx = 0; idx < closCount; ++idx) {
        CHK_RET(HcommChannelNotifyWaitOnThread(
            closThread, closChannels[idx]->handle, NOTIFY_IDX_ACK,
            CUSTOM_TIMEOUT));
    }
    for (uint32_t idx = 0; idx < meshCount; ++idx) {
        CHK_RET(HcommChannelNotifyWaitOnThread(
            meshThread, meshChannels[idx]->handle, NOTIFY_IDX_ACK,
            CUSTOM_TIMEOUT));
    }

    // ExecOp返回后，AICPU Kernel分别Join meshThread和closThread，
    // 两条RTSQ都完成后Kernel才返回用户Stream。
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunPairFanoutAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // x1 先形成连续 1MB pair；随后把同一 pair 一次扇出到其余
    // 7 个 pair 组。偶数 XOR mask 2..14 加本地 mask 0 恰好枚举
    // 16 rank 的 8 个 pair，数据依赖深度仅为 512KB + 1MB。
    ThreadHandle thread = resCtx.threads[0];
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);

    const uint32_t xor1Rank = param.myRank ^ 1U;
    const uint32_t xor2Rank = param.myRank ^ 2U;
    const uint32_t xor4Rank = param.myRank ^ 4U;
    const uint32_t xor6Rank = param.myRank ^ 6U;
    const uint32_t xor8Rank = param.myRank ^ 8U;
    const uint32_t xor10Rank = param.myRank ^ 10U;
    const uint32_t xor12Rank = param.myRank ^ 12U;
    const uint32_t xor14Rank = param.myRank ^ 14U;
    const ChannelInfo *xor1Channel = nullptr;
    const ChannelInfo *xor2Channel = nullptr;
    const ChannelInfo *xor4Channel = nullptr;
    const ChannelInfo *xor6Channel = nullptr;
    const ChannelInfo *xor8Channel = nullptr;
    const ChannelInfo *xor10Channel = nullptr;
    const ChannelInfo *xor12Channel = nullptr;
    const ChannelInfo *xor14Channel = nullptr;
    CHK_RET(GetDirectChannel(param, resCtx, xor1Rank, xor1Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor2Rank, xor2Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor4Rank, xor4Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor6Rank, xor6Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor8Rank, xor8Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor10Rank, xor10Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor12Rank, xor12Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor14Rank, xor14Channel));

    // 初始发送直接读取 input；本 rank Copy 排在 WN 后，与 x1 传输重叠。
    auto *remoteSelf =
        static_cast<uint8_t *>(xor1Channel->remoteOutput.addr) +
        static_cast<uint64_t>(param.myRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor1Channel->handle, remoteSelf, param.inputPtr,
        dataSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommLocalCopyOnThread(
        thread,
        outputBuffer + static_cast<uint64_t>(param.myRank) * dataSize,
        param.inputPtr, dataSize));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor1Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    const uint32_t pairStartRank = param.myRank & ~1U;
    const uint64_t pairSize = dataSize * 2U;
    auto *pair =
        outputBuffer + static_cast<uint64_t>(pairStartRank) * dataSize;
    auto *remotePairOnX2 =
        static_cast<uint8_t *>(xor2Channel->remoteOutput.addr) +
        static_cast<uint64_t>(pairStartRank) * dataSize;
    auto *remotePairOnX4 =
        static_cast<uint8_t *>(xor4Channel->remoteOutput.addr) +
        static_cast<uint64_t>(pairStartRank) * dataSize;
    auto *remotePairOnX6 =
        static_cast<uint8_t *>(xor6Channel->remoteOutput.addr) +
        static_cast<uint64_t>(pairStartRank) * dataSize;
    auto *remotePairOnX8 =
        static_cast<uint8_t *>(xor8Channel->remoteOutput.addr) +
        static_cast<uint64_t>(pairStartRank) * dataSize;
    auto *remotePairOnX10 =
        static_cast<uint8_t *>(xor10Channel->remoteOutput.addr) +
        static_cast<uint64_t>(pairStartRank) * dataSize;
    auto *remotePairOnX12 =
        static_cast<uint8_t *>(xor12Channel->remoteOutput.addr) +
        static_cast<uint64_t>(pairStartRank) * dataSize;
    auto *remotePairOnX14 =
        static_cast<uint8_t *>(xor14Channel->remoteOutput.addr) +
        static_cast<uint64_t>(pairStartRank) * dataSize;

    // 四条高延迟 Clos 优先敲 Doorbell，随后下发三条 Mesh；七条 WN
    // 全部位于任一 Wait 之前，从单一 RTSQ 暴露最大 Channel 并行度。
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor8Channel->handle, remotePairOnX8, pair,
        pairSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor10Channel->handle, remotePairOnX10, pair,
        pairSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor12Channel->handle, remotePairOnX12, pair,
        pairSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor14Channel->handle, remotePairOnX14, pair,
        pairSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor2Channel->handle, remotePairOnX2, pair,
        pairSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor4Channel->handle, remotePairOnX4, pair,
        pairSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor6Channel->handle, remotePairOnX6, pair,
        pairSize, NOTIFY_IDX_ACK));

    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor8Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor10Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor12Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor14Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor2Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor4Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor6Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunThreeWayQuarterFanoutAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // 固定 2 Server x 8 Rank：先沿 x1/x2 形成连续 4-rank quarter，
    // 再把同一 quarter 同时发送到 x4/x8/x12。三个目标分别补齐
    // 本 Server 异 quarter、远端 Server 同 quarter和远端异 quarter。
    // 最后一级三路 WN 全部先于 Wait 下发，只使用主 RTSQ 和一次 Batch。
    ThreadHandle thread = resCtx.threads[0];
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);

    const uint32_t xor1Rank = param.myRank ^ 1U;
    const uint32_t xor2Rank = param.myRank ^ 2U;
    const uint32_t xor4Rank = param.myRank ^ 4U;
    const uint32_t xor8Rank = param.myRank ^ RANKS_PER_SERVER;
    const uint32_t xor12Rank =
        param.myRank ^ (RANKS_PER_SERVER | 4U);
    const ChannelInfo *xor1Channel = nullptr;
    const ChannelInfo *xor2Channel = nullptr;
    const ChannelInfo *xor4Channel = nullptr;
    const ChannelInfo *xor8Channel = nullptr;
    const ChannelInfo *xor12Channel = nullptr;
    CHK_RET(GetDirectChannel(param, resCtx, xor1Rank, xor1Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor2Rank, xor2Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor4Rank, xor4Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor8Rank, xor8Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor12Rank, xor12Channel));

    // x1 直接读取 input；LocalCopy 排在 WN 后，可与首轮网络传输重叠。
    auto *remoteSelf =
        static_cast<uint8_t *>(xor1Channel->remoteOutput.addr) +
        static_cast<uint64_t>(param.myRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor1Channel->handle, remoteSelf, param.inputPtr,
        dataSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommLocalCopyOnThread(
        thread,
        outputBuffer + static_cast<uint64_t>(param.myRank) * dataSize,
        param.inputPtr, dataSize));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor1Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    // x2 交换连续 pair，得到本 rank 所在的 4-rank / 2MB quarter。
    const uint32_t pairStartRank = param.myRank & ~1U;
    const uint64_t pairSize = dataSize * 2U;
    auto *pair =
        outputBuffer + static_cast<uint64_t>(pairStartRank) * dataSize;
    auto *remotePair =
        static_cast<uint8_t *>(xor2Channel->remoteOutput.addr) +
        static_cast<uint64_t>(pairStartRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor2Channel->handle, remotePair, pair,
        pairSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor2Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
    const uint32_t localServerBase = param.myRank - localRank;
    const uint32_t quarterStartRank =
        localServerBase + (localRank & 4U);
    const uint64_t quarterSize =
        dataSize * (RANKS_PER_SERVER / 2U);
    auto *quarter =
        outputBuffer + static_cast<uint64_t>(quarterStartRank) * dataSize;
    auto *remoteQuarterOnX4 =
        static_cast<uint8_t *>(xor4Channel->remoteOutput.addr) +
        static_cast<uint64_t>(quarterStartRank) * dataSize;
    auto *remoteQuarterOnX8 =
        static_cast<uint8_t *>(xor8Channel->remoteOutput.addr) +
        static_cast<uint64_t>(quarterStartRank) * dataSize;
    auto *remoteQuarterOnX12 =
        static_cast<uint8_t *>(xor12Channel->remoteOutput.addr) +
        static_cast<uint64_t>(quarterStartRank) * dataSize;

    // 三路 2MB 传输互不依赖；先敲响高延迟的 x8/x12 Clos，再下发 x4 Mesh。
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor8Channel->handle, remoteQuarterOnX8, quarter,
        quarterSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor12Channel->handle, remoteQuarterOnX12, quarter,
        quarterSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor4Channel->handle, remoteQuarterOnX4, quarter,
        quarterSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor8Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor12Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor4Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunSingleRtsqDualClosAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // 两条逻辑 lane 全部压入主 RTSQ：own lane 使用 xor1->xor2，
    // opposite lane 通过 bit-swizzle 种子后使用 xor3->xor1。
    // 每阶段先连续下发两条独立 WN 再 Wait，保留 Channel 并行度；
    // 整个算法仅需一次 Batch 提交，不再启动第二 Thread 或最终汇合。
    ThreadHandle thread = resCtx.threads[0];
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);

    const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
    const uint32_t localServerBase = param.myRank - localRank;
    // p=[4,7,6,5,0,3,2,1]是对合，并对所有rank满足
    // p(r^3)=p(r)^1、p(r^1)=p(r)^3。
    const uint32_t lowBit = localRank & 1U;
    const uint32_t swizzledLocalRank =
        ((localRank ^ 4U) & 4U) |
        ((localRank & 2U) ^ (lowBit << 1U)) |
        lowBit;
    const uint32_t seedRank = localServerBase + swizzledLocalRank;
    const uint32_t xor1Rank = param.myRank ^ 1U;
    const uint32_t ownSecondRank = param.myRank ^ 2U;
    const uint32_t oppositeFirstRank = param.myRank ^ 3U;
    const uint32_t ownCrossRank = param.myRank ^ RANKS_PER_SERVER;
    const uint32_t oppositeCrossRank =
        param.myRank ^ (RANKS_PER_SERVER | 2U);
    const ChannelInfo *seedChannel = nullptr;
    const ChannelInfo *xor1Channel = nullptr;
    const ChannelInfo *ownSecondChannel = nullptr;
    const ChannelInfo *oppositeFirstChannel = nullptr;
    const ChannelInfo *ownCrossChannel = nullptr;
    const ChannelInfo *oppositeCrossChannel = nullptr;
    CHK_RET(GetDirectChannel(param, resCtx, seedRank, seedChannel));
    CHK_RET(GetDirectChannel(param, resCtx, xor1Rank, xor1Channel));
    CHK_RET(GetDirectChannel(
        param, resCtx, ownSecondRank, ownSecondChannel));
    CHK_RET(GetDirectChannel(
        param, resCtx, oppositeFirstRank, oppositeFirstChannel));
    CHK_RET(GetDirectChannel(param, resCtx, ownCrossRank, ownCrossChannel));
    CHK_RET(GetDirectChannel(
        param, resCtx, oppositeCrossRank, oppositeCrossChannel));

    // 前两次只发送原始本 rank 块，直接读取用户输入；把 LocalCopy 排在
    // 两条 WN 后，可与首轮 Mesh 重叠且不阻塞首个 Doorbell。
    auto *localBlock =
        outputBuffer + static_cast<uint64_t>(param.myRank) * dataSize;

    // 每个 rank 把本 rank 块发送给 p(rank)；由于 p(p(rank))=rank，
    // seed Channel Wait 后本地获得的正是 p(rank) 对应数据。
    auto *remoteSeedBlock =
        static_cast<uint8_t *>(seedChannel->remoteOutput.addr) +
        static_cast<uint64_t>(param.myRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, seedChannel->handle, remoteSeedBlock,
        param.inputPtr, dataSize, NOTIFY_IDX_DATA_SIGNAL));

    // own xor1 不依赖 seed，提前排在 seed Wait 之前；两条初始 WN
    // 因而能在硬件上并行推进。
    auto *remoteOwnBlock =
        static_cast<uint8_t *>(xor1Channel->remoteOutput.addr) +
        static_cast<uint64_t>(param.myRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor1Channel->handle, remoteOwnBlock,
        param.inputPtr, dataSize, NOTIFY_IDX_ACK));

    // Copy 在同一 RTSQ 上严格位于第二轮 pair 发送之前，保证 self 槽
    // 就绪；同时仍可与前面的两条网络 WN 重叠。
    CHK_RET(HcommLocalCopyOnThread(
        thread, localBlock, param.inputPtr, dataSize));

    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, seedChannel->handle,
        NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

    // opposite xor3 的源块由 seed Wait 保证就绪。
    auto *seedBlock =
        outputBuffer + static_cast<uint64_t>(seedRank) * dataSize;
    auto *remoteOppositeSeedBlock =
        static_cast<uint8_t *>(oppositeFirstChannel->remoteOutput.addr) +
        static_cast<uint64_t>(seedRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, oppositeFirstChannel->handle,
        remoteOppositeSeedBlock,
        seedBlock, dataSize, NOTIFY_IDX_DATA_SIGNAL));

    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor1Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, oppositeFirstChannel->handle,
        NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

    // 第二轮交换两个连续 rank 块：own 走 xor2，opposite 切换到 xor1；
    // 两条 WN 仍先于两个 Wait 下发，并各自得到连续 2MB 半区。
    const uint64_t pairSize = dataSize * 2U;
    const uint32_t ownPairStartRank = param.myRank & ~1U;
    auto *ownPair =
        outputBuffer + static_cast<uint64_t>(ownPairStartRank) * dataSize;
    auto *remoteOwnPair =
        static_cast<uint8_t *>(ownSecondChannel->remoteOutput.addr) +
        static_cast<uint64_t>(ownPairStartRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, ownSecondChannel->handle, remoteOwnPair,
        ownPair, pairSize, NOTIFY_IDX_ACK));

    const uint32_t oppositePairStartRank = seedRank & ~1U;
    auto *oppositePair =
        outputBuffer + static_cast<uint64_t>(oppositePairStartRank) * dataSize;
    auto *remoteOppositePair =
        static_cast<uint8_t *>(xor1Channel->remoteOutput.addr) +
        static_cast<uint64_t>(oppositePairStartRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor1Channel->handle, remoteOppositePair,
        oppositePair, pairSize, NOTIFY_IDX_DATA_SIGNAL));

    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, ownSecondChannel->handle,
        NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor1Channel->handle,
        NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

    const uint32_t ownHalfStartRank =
        localServerBase + (localRank & 4U);
    const uint32_t oppositeHalfStartRank =
        localServerBase + ((localRank ^ 4U) & 4U);
    const uint64_t halfSize = dataSize * (RANKS_PER_SERVER / 2);

    // 两个连续2MB半区无需先汇合，直接在两条Clos Channel上并行交换。
    auto *ownHalf =
        outputBuffer + static_cast<uint64_t>(ownHalfStartRank) * dataSize;
    auto *remoteOwnHalf =
        static_cast<uint8_t *>(ownCrossChannel->remoteOutput.addr) +
        static_cast<uint64_t>(ownHalfStartRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, ownCrossChannel->handle, remoteOwnHalf,
        ownHalf, halfSize, NOTIFY_IDX_ACK));

    auto *oppositeHalf =
        outputBuffer + static_cast<uint64_t>(oppositeHalfStartRank) * dataSize;
    auto *remoteOppositeHalf =
        static_cast<uint8_t *>(oppositeCrossChannel->remoteOutput.addr) +
        static_cast<uint64_t>(oppositeHalfStartRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, oppositeCrossChannel->handle,
        remoteOppositeHalf, oppositeHalf, halfSize,
        NOTIFY_IDX_DATA_SIGNAL));

    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, ownCrossChannel->handle,
        NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, oppositeCrossChannel->handle,
        NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

    return HCCL_SUCCESS;
}

HcclResult FlushCase05Batch(const OpParam &param)
{
    // 提前推进当前主 RTSQ 的 tail，使首批 Mesh/Clos 在 AICPU 继续构造
    // 剩余任务时即可执行；随后重新进入 Batch，最终仍由外层统一 End。
    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.tag)));
    CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.tag)));
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RunQuarterPipelineAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // 单主 Thread 的 4-quarter DAG。Q0/Q1/Q2/Q3 均为连续两个 rank
    // （1MB），通过 x2/x4/x5 的树形 fan-out 形成，再分别走四条 Clos。
    ThreadHandle thread = resCtx.threads[0];
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);

    const uint32_t xor1Rank = param.myRank ^ 1U;
    const uint32_t xor2Rank = param.myRank ^ 2U;
    const uint32_t xor4Rank = param.myRank ^ 4U;
    const uint32_t xor5Rank = param.myRank ^ 5U;
    const uint32_t cross0Rank = param.myRank ^ 8U;
    const uint32_t cross1Rank = param.myRank ^ 9U;
    const uint32_t cross2Rank = param.myRank ^ 10U;
    const uint32_t cross3Rank = param.myRank ^ 11U;

    const ChannelInfo *xor1Channel = nullptr;
    const ChannelInfo *xor2Channel = nullptr;
    const ChannelInfo *xor4Channel = nullptr;
    const ChannelInfo *xor5Channel = nullptr;
    const ChannelInfo *cross0Channel = nullptr;
    const ChannelInfo *cross1Channel = nullptr;
    const ChannelInfo *cross2Channel = nullptr;
    const ChannelInfo *cross3Channel = nullptr;
    CHK_RET(GetDirectChannel(param, resCtx, xor1Rank, xor1Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor2Rank, xor2Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor4Rank, xor4Channel));
    CHK_RET(GetDirectChannel(param, resCtx, xor5Rank, xor5Channel));
    CHK_RET(GetDirectChannel(param, resCtx, cross0Rank, cross0Channel));
    CHK_RET(GetDirectChannel(param, resCtx, cross1Rank, cross1Channel));
    CHK_RET(GetDirectChannel(param, resCtx, cross2Rank, cross2Channel));
    CHK_RET(GetDirectChannel(param, resCtx, cross3Rank, cross3Channel));

    const uint64_t quarterSize = dataSize * 2U;
    const uint32_t quarter0Start = param.myRank & ~1U;
    const uint32_t quarter1Start = (param.myRank ^ 2U) & ~1U;
    const uint32_t quarter2Start = (param.myRank ^ 4U) & ~1U;
    const uint32_t quarter3Start = (param.myRank ^ 6U) & ~1U;
    auto *quarter0 =
        outputBuffer + static_cast<uint64_t>(quarter0Start) * dataSize;
    auto *quarter1 =
        outputBuffer + static_cast<uint64_t>(quarter1Start) * dataSize;
    auto *quarter2 =
        outputBuffer + static_cast<uint64_t>(quarter2Start) * dataSize;
    auto *quarter3 =
        outputBuffer + static_cast<uint64_t>(quarter3Start) * dataSize;

    // x1 先形成 Q0；发送直接读取 input，使本地 Copy 不阻塞首个 Doorbell。
    auto *remoteSelf =
        static_cast<uint8_t *>(xor1Channel->remoteOutput.addr) +
        static_cast<uint64_t>(param.myRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor1Channel->handle, remoteSelf, param.inputPtr,
        dataSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommLocalCopyOnThread(
        thread,
        outputBuffer + static_cast<uint64_t>(param.myRank) * dataSize,
        param.inputPtr, dataSize));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor1Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    // Q0 同时 fan-out 到 x2/x4，并立即经第一条 Clos 发送；三条 Channel
    // 的 WN 都在任何后续 Wait 之前下发，可在硬件上并行推进。
    auto *remoteQuarter0OnX2 =
        static_cast<uint8_t *>(xor2Channel->remoteOutput.addr) +
        static_cast<uint64_t>(quarter0Start) * dataSize;
    auto *remoteQuarter0OnX4 =
        static_cast<uint8_t *>(xor4Channel->remoteOutput.addr) +
        static_cast<uint64_t>(quarter0Start) * dataSize;
    auto *remoteQuarter0OnCross =
        static_cast<uint8_t *>(cross0Channel->remoteOutput.addr) +
        static_cast<uint64_t>(quarter0Start) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor2Channel->handle, remoteQuarter0OnX2, quarter0,
        quarterSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor4Channel->handle, remoteQuarter0OnX4, quarter0,
        quarterSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, cross0Channel->handle, remoteQuarter0OnCross, quarter0,
        quarterSize, NOTIFY_IDX_ACK));

    // 首批任务立即提交；网络运行与下面剩余任务的 AICPU 构图重叠，
    // 仅额外产生一次主 RTSQ tail 更新。
    CHK_RET(FlushCase05Batch(param));

    // x2 到达后得到 Q1。先把 Q1 继续 fan-out 到 x5，再提前走第二条 Clos。
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor2Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    auto *remoteQuarter1OnX5 =
        static_cast<uint8_t *>(xor5Channel->remoteOutput.addr) +
        static_cast<uint64_t>(quarter1Start) * dataSize;
    auto *remoteQuarter1OnCross =
        static_cast<uint8_t *>(cross1Channel->remoteOutput.addr) +
        static_cast<uint64_t>(quarter1Start) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, xor5Channel->handle, remoteQuarter1OnX5, quarter1,
        quarterSize, NOTIFY_IDX_ACK));
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, cross1Channel->handle, remoteQuarter1OnCross, quarter1,
        quarterSize, NOTIFY_IDX_ACK));

    // x4/x5 分别产生 Q2/Q3；每份到达后立即在独立 Clos Channel 上发送。
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor4Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    auto *remoteQuarter2OnCross =
        static_cast<uint8_t *>(cross2Channel->remoteOutput.addr) +
        static_cast<uint64_t>(quarter2Start) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, cross2Channel->handle, remoteQuarter2OnCross, quarter2,
        quarterSize, NOTIFY_IDX_ACK));

    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, xor5Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    auto *remoteQuarter3OnCross =
        static_cast<uint8_t *>(cross3Channel->remoteOutput.addr) +
        static_cast<uint64_t>(quarter3Start) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, cross3Channel->handle, remoteQuarter3OnCross, quarter3,
        quarterSize, NOTIFY_IDX_ACK));

    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, cross0Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, cross1Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, cross2Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, cross3Channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult RunDirectOutputAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // 固定 2 Server x 8 Rank：先跨 Server 交换本 rank 块，再在 Server 内
    // 用两条独立 lane 扩散本 Server/对端 Server 半区。每 rank 仅产生
    // 7 次 WriteWithNotify + 7 次 Channel Wait，并保留一次末尾 Thread 汇合。
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);
    ThreadHandle localLaneThread = resCtx.threads[0];
    ThreadHandle remoteLaneThread = resCtx.threads[1];
    CHK_RET(HcommLocalCopyOnThread(localLaneThread,
        outputBuffer + static_cast<uint64_t>(param.myRank) * dataSize,
        param.inputPtr, dataSize));

    const uint32_t crossServerPartner = param.myRank ^ RANKS_PER_SERVER;
    const uint32_t crossServerChannelIdx = crossServerPartner < param.myRank
        ? crossServerPartner : crossServerPartner - 1;
    CHK_PRT_RET(crossServerChannelIdx >= resCtx.channels.size() ||
            resCtx.channels[crossServerChannelIdx].remoteRank != crossServerPartner,
        HCCL_ERROR("Cross-server channel to rank %u was not found", crossServerPartner),
        HCCL_E_NOT_FOUND);
    const ChannelInfo &crossServerChannel = resCtx.channels[crossServerChannelIdx];
    auto *localBlock = outputBuffer + static_cast<uint64_t>(param.myRank) * dataSize;
    auto *remoteBlock = static_cast<uint8_t *>(crossServerChannel.remoteOutput.addr) +
        static_cast<uint64_t>(param.myRank) * dataSize;
    CHK_RET(HcommWriteWithNotifyOnThread(localLaneThread, crossServerChannel.handle,
        remoteBlock, localBlock, dataSize, NOTIFY_IDX_DATA_SIGNAL));
    CHK_RET(HcommChannelNotifyWaitOnThread(remoteLaneThread, crossServerChannel.handle,
        NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

    const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
    const uint32_t localServerBase = param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
    const uint32_t remoteServerBase = localServerBase ^ RANKS_PER_SERVER;
    for (uint32_t mask = 1; mask < RANKS_PER_SERVER; mask <<= 1) {
        const uint32_t partner = param.myRank ^ mask;
        const uint32_t channelIdx = partner < param.myRank ? partner : partner - 1;
        CHK_PRT_RET(channelIdx >= resCtx.channels.size() ||
                resCtx.channels[channelIdx].remoteRank != partner,
            HCCL_ERROR("Recursive-doubling channel to rank %u was not found", partner),
            HCCL_E_NOT_FOUND);
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        const uint32_t groupOffset = localRank & ~(mask - 1U);
        const uint64_t groupSize = static_cast<uint64_t>(mask) * dataSize;
        const uint32_t localGroupRank = localServerBase + groupOffset;
        const uint32_t remoteGroupRank = remoteServerBase + groupOffset;
        auto *remoteOutput = static_cast<uint8_t *>(channel.remoteOutput.addr);

        CHK_RET(HcommWriteWithNotifyOnThread(localLaneThread, channel.handle,
            remoteOutput + static_cast<uint64_t>(localGroupRank) * dataSize,
            outputBuffer + static_cast<uint64_t>(localGroupRank) * dataSize,
            groupSize, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyWaitOnThread(localLaneThread, channel.handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        CHK_RET(HcommWriteWithNotifyOnThread(remoteLaneThread, channel.handle,
            remoteOutput + static_cast<uint64_t>(remoteGroupRank) * dataSize,
            outputBuffer + static_cast<uint64_t>(remoteGroupRank) * dataSize,
            groupSize, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(remoteLaneThread, channel.handle,
            NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    }
    CHK_RET(HcommThreadNotifyWaitOnThread(localLaneThread, 0, CUSTOM_TIMEOUT));
    CHK_RET(HcommThreadNotifyRecordOnThread(remoteLaneThread, localLaneThread, 0));
    return HCCL_SUCCESS;
}

HcclResult RunPreparedBigDirect(const BigDirectPlan &plan)
{
    CHK_PRT_RET(plan.magic != BIG_DIRECT_PLAN_MAGIC ||
            plan.peerCount != BIG_DIRECT_PEER_COUNT ||
            plan.activeThreadNum != BIG_DIRECT_THREAD_COUNT ||
            plan.readyThreadNum == 0 ||
            plan.readyThreadNum > plan.activeThreadNum ||
            plan.inputPtr == nullptr || plan.localDst == nullptr,
        HCCL_ERROR("Invalid Host-prepared big direct plan"),
        HCCL_E_PARA);

    const ThreadHandle rootThread = plan.threads[0];
    const ThreadHandle copyThread =
        plan.threads[BIG_DIRECT_THREAD_COUNT - 1];
    if (plan.earlyLocalCopy != 0) {
        CHK_RET(StartLocalCopy(
            rootThread, copyThread, plan.localDst,
            plan.inputPtr, plan.dataSize));
    }

    // 保留已验证的大包平铺依赖，只删除每次调用的vector遍历和地址构造。
    for (uint32_t idx = 1; idx < plan.readyThreadNum; ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            rootThread, plan.threads[idx], 0));
    }
    for (uint32_t idx = 1; idx < plan.readyThreadNum; ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            plan.threads[idx], 0, CUSTOM_TIMEOUT));
    }

    for (uint32_t idx = 0; idx < plan.peerCount; ++idx) {
        const BigDirectPeerPlan &peer = plan.peers[idx];
        CHK_RET(HcommChannelNotifyRecordOnThread(
            peer.thread, peer.channel, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            peer.thread, peer.channel, NOTIFY_IDX_DATA_SIGNAL,
            CUSTOM_TIMEOUT));
        CHK_RET(static_cast<HcclResult>(HcommBatchTransferOnThread(
            peer.thread, peer.channel, peer.transfers, 2)));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            peer.thread, peer.channel, NOTIFY_IDX_ACK,
            CUSTOM_TIMEOUT));
    }

    if (plan.earlyLocalCopy == 0) {
        CHK_RET(HcommLocalCopyOnThread(
            copyThread, plan.localDst, plan.inputPtr,
            plan.dataSize));
    }
    for (uint32_t idx = 1; idx < plan.activeThreadNum; ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            rootThread, idx - 1, CUSTOM_TIMEOUT));
    }
    for (uint32_t idx = 1; idx < plan.activeThreadNum; ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            plan.threads[idx], rootThread, idx - 1));
    }
    return HCCL_SUCCESS;
}

HcclResult RunDirectInputAllGather(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);
    const uint32_t channelThreadNum = resCtx.channels.size();
    const bool useSmallMessagePath = dataSize < SMALL_MESSAGE_MAX_SIZE;
    const uint32_t communicationThreadNum = useSmallMessagePath
        ? std::min(channelThreadNum, SMALL_MESSAGE_THREAD_NUM) : channelThreadNum;
    const bool parallelLocalCopy = dataSize >= PARALLEL_LOCAL_COPY_MIN_SIZE
        && resCtx.threads.size() > channelThreadNum;
    const bool earlyLocalCopy = parallelLocalCopy && dataSize < EARLY_LOCAL_COPY_MAX_SIZE;
    const uint32_t activeThreadNum = communicationThreadNum + static_cast<uint32_t>(parallelLocalCopy);
    const uint32_t readyThreadNum = earlyLocalCopy ? communicationThreadNum : activeThreadNum;
    uint8_t *localOutput = outputBuffer + param.myRank * dataSize;
    if (earlyLocalCopy) {
        // 第 16 Thread 仅依赖本 rank 输入就绪，无需等待跨 rank 屏障。
        // 在主 Thread 进入就绪阶段前先启动拷贝，与屏障和远端 Read 重叠。
        CHK_RET(StartLocalCopy(resCtx.threads[0], resCtx.threads[channelThreadNum],
            localOutput, param.inputPtr, dataSize));
    }
    // 性能实测表明：小包的关键路径受任务数主导，大包则由数据传输主导。
    // 512KB 使用 recursive-doubling + 树形本地同步；大包精确回退到已实测最快的
    // 逐对端就绪握手 + 平铺本地同步。
    const bool useRecursiveReadyBarrier = useSmallMessagePath && param.rankSize == 16;
    if (useRecursiveReadyBarrier) {
        CHK_RET(InterRankBarrier(param, resCtx, NOTIFY_IDX_DATA_SIGNAL));
    }
    // 将 Host/主 Thread 上的输入就绪依赖广播到全部通信 Thread，供 checker
    // 和硬件共同保序；大包保留已验证的平铺同步骨架。
    if (useSmallMessagePath) {
        CHK_RET(TreeThreadSyncBefore(resCtx.threads, readyThreadNum));
    } else {
        CHK_RET(FlatThreadSyncBefore(resCtx.threads, readyThreadNum));
    }
    for (uint32_t taskIdx = 0; taskIdx < resCtx.channels.size(); ++taskIdx) {
        const ChannelInfo &channel = resCtx.channels[taskIdx];
        const uint32_t threadIdx = useSmallMessagePath ? taskIdx % communicationThreadNum : taskIdx;
        uint8_t *remoteOutput = outputBuffer + channel.remoteRank * dataSize;

        // 非 16 rank 回退为逐对端就绪握手；本题 16 rank 已由上面的 4 轮全局屏障覆盖。
        if (!useRecursiveReadyBarrier) {
            CHK_RET(HcommChannelNotifyRecordOnThread(
                resCtx.threads[threadIdx], channel.handle, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(HcommChannelNotifyWaitOnThread(
                resCtx.threads[threadIdx], channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        }
        if (useRecursiveReadyBarrier) {
            // 小包在全部本地 Read 归并后使用全局完成屏障，无需15组逐对端 ACK。
            CHK_RET(HcommReadOnThread(resCtx.threads[threadIdx], channel.handle,
                remoteOutput, channel.remoteInput.addr, dataSize));
        } else {
            CHK_RET(BatchReadAndNotify(resCtx.threads[threadIdx], channel.handle,
                remoteOutput, channel.remoteInput.addr, dataSize, NOTIFY_IDX_ACK));
            CHK_RET(HcommChannelNotifyWaitOnThread(
                resCtx.threads[threadIdx], channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        }
    }

    if (!earlyLocalCopy) {
        const ThreadHandle localCopyThread = useSmallMessagePath
            ? resCtx.threads[communicationThreadNum - 1]
            : (parallelLocalCopy ? resCtx.threads[channelThreadNum] : resCtx.threads[0]);
        CHK_RET(HcommLocalCopyOnThread(localCopyThread, localOutput, param.inputPtr, dataSize));
    }
    if (useSmallMessagePath) {
        CHK_RET(TreeThreadSyncAfter(resCtx.threads, activeThreadNum));
    } else {
        CHK_RET(FlatThreadSyncAfter(resCtx.threads, activeThreadNum));
    }
    if (useRecursiveReadyBarrier) {
        // 此屏障证明所有 rank 均已读完全部输入，保证算子返回后用户可安全复用 sendBuf。
        CHK_RET(InterRankBarrier(param, resCtx, NOTIFY_IDX_ACK));
    }
    return HCCL_SUCCESS;
}

HcclResult RunPushAllGather(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataTypeSize,
    uint64_t dataSize, uint64_t maxSliceCount)
{
    const uint64_t loopCount =
        param.count / maxSliceCount + static_cast<uint64_t>(param.count % maxSliceCount != 0);
    uint64_t processedCount = 0;
    auto *localCclBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    auto *inputBuffer = static_cast<uint8_t *>(param.inputPtr);
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);

    for (uint64_t loop = 0; loop < loopCount; ++loop) {
        const uint64_t sliceCount = std::min(maxSliceCount, param.count - processedCount);
        const uint64_t sliceSize = sliceCount * dataTypeSize;
        uint8_t *localSlice = localCclBuffer + sliceSize * param.myRank;
        uint8_t *inputSlice = inputBuffer + processedCount * dataTypeSize;

        CHK_RET(HcommLocalCopyOnThread(resCtx.threads[0], localSlice, inputSlice, sliceSize));
        const uint32_t channelThreadNum = resCtx.channels.size();
        const bool parallelLocalCopy = sliceSize >= PARALLEL_LOCAL_COPY_MIN_SIZE
            && resCtx.threads.size() > channelThreadNum;
        const uint32_t activeThreadNum = channelThreadNum + static_cast<uint32_t>(parallelLocalCopy);
        CHK_RET(TreeThreadSyncBefore(resCtx.threads, activeThreadNum));

        for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
            const ChannelInfo &channel = resCtx.channels[idx];
            uint8_t *remoteSlice =
                static_cast<uint8_t *>(channel.remoteCclMem.addr) + sliceSize * param.myRank;
            uint8_t *receivedSlice = localCclBuffer + sliceSize * channel.remoteRank;
            uint8_t *remoteOutputSlice =
                outputBuffer + channel.remoteRank * dataSize + processedCount * dataTypeSize;

            // 首轮没有 Buffer 复用冲突；后续轮次需等待对端消费完上一轮数据再覆写 CCL Buffer。
            if (loop > 0) {
                CHK_RET(HcommChannelNotifyRecordOnThread(
                    resCtx.threads[idx], channel.handle, NOTIFY_IDX_ACK));
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    resCtx.threads[idx], channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            }

            // 写完成与远端通知融合下发，减少一个独立的 Channel Notify 任务。
            CHK_RET(HcommWriteWithNotifyOnThread(resCtx.threads[idx], channel.handle,
                remoteSlice, localSlice, sliceSize, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(HcommChannelNotifyWaitOnThread(
                resCtx.threads[idx], channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

            // 每条通信 Thread 搬运其对应远端 rank 的结果，避免主 Thread 串行拷贝 16 份数据。
            CHK_RET(HcommLocalCopyOnThread(
                resCtx.threads[idx], remoteOutputSlice, receivedSlice, sliceSize));
        }

        uint8_t *localOutputSlice =
            outputBuffer + param.myRank * dataSize + processedCount * dataTypeSize;
        const ThreadHandle localCopyThread =
            parallelLocalCopy ? resCtx.threads[channelThreadNum] : resCtx.threads[0];
        CHK_RET(HcommLocalCopyOnThread(localCopyThread, localOutputSlice, localSlice, sliceSize));
        CHK_RET(TreeThreadSyncAfter(resCtx.threads, activeThreadNum));
        processedCount += sliceCount;
    }

    return HCCL_SUCCESS;
}

HcclResult RunPullAllGather(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataTypeSize,
    uint64_t dataSize, uint64_t maxSliceCount)
{
    const uint64_t loopCount =
        param.count / maxSliceCount + static_cast<uint64_t>(param.count % maxSliceCount != 0);
    uint64_t processedCount = 0;
    auto *localCclBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    auto *inputBuffer = static_cast<uint8_t *>(param.inputPtr);
    auto *outputBuffer = static_cast<uint8_t *>(param.outputPtr);

    for (uint64_t loop = 0; loop < loopCount; ++loop) {
        const uint64_t sliceCount = std::min(maxSliceCount, param.count - processedCount);
        const uint64_t sliceSize = sliceCount * dataTypeSize;
        uint8_t *inputSlice = inputBuffer + processedCount * dataTypeSize;

        // Pull 模式只在 CCL Buffer 中暂存本 rank 的数据，因此单轮可使用完整 Buffer。
        CHK_RET(HcommLocalCopyOnThread(resCtx.threads[0], localCclBuffer, inputSlice, sliceSize));
        const uint32_t channelThreadNum = resCtx.channels.size();
        const bool parallelLocalCopy = sliceSize >= PARALLEL_LOCAL_COPY_MIN_SIZE
            && resCtx.threads.size() > channelThreadNum;
        const uint32_t activeThreadNum = channelThreadNum + static_cast<uint32_t>(parallelLocalCopy);
        CHK_RET(TreeThreadSyncBefore(resCtx.threads, activeThreadNum));

        for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
            const ChannelInfo &channel = resCtx.channels[idx];
            uint8_t *remoteOutputSlice =
                outputBuffer + channel.remoteRank * dataSize + processedCount * dataTypeSize;

            // 双向发布本轮数据就绪，再从对端 CCL Buffer 直接读取到最终输出。
            CHK_RET(HcommChannelNotifyRecordOnThread(
                resCtx.threads[idx], channel.handle, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(HcommChannelNotifyWaitOnThread(
                resCtx.threads[idx], channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
            // 将远端读取与消费完成通知批量下发；ACK 在 Read 之后保序执行，仅产生一次完成事件。
            CHK_RET(BatchReadAndNotify(resCtx.threads[idx], channel.handle,
                remoteOutputSlice, channel.remoteCclMem.addr, sliceSize, NOTIFY_IDX_ACK));

            // 等待对端的消费确认，保证进入下一轮前所有对端均已读完当前 CCL Buffer。
            CHK_RET(HcommChannelNotifyWaitOnThread(
                resCtx.threads[idx], channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        }

        uint8_t *localOutputSlice =
            outputBuffer + param.myRank * dataSize + processedCount * dataTypeSize;
        const ThreadHandle localCopyThread =
            parallelLocalCopy ? resCtx.threads[channelThreadNum] : resCtx.threads[0];
        CHK_RET(HcommLocalCopyOnThread(localCopyThread, localOutputSlice, localCclBuffer, sliceSize));
        CHK_RET(TreeThreadSyncAfter(resCtx.threads, activeThreadNum));
        processedCount += sliceCount;
    }

    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecCase05(const Case05ResourcePlan &plan)
{
    CHK_PRT_RET(plan.magic != CASE05_RESOURCE_PLAN_MAGIC ||
            plan.rankSize != RANKS_PER_SERVER * 2 ||
            plan.myRank >= plan.rankSize ||
            plan.inputPtr == nullptr || plan.localDst == nullptr,
        HCCL_ERROR("Invalid Host-prepared Case05 resource plan"),
        HCCL_E_PARA);

    return RunHostPreparedCase05(
        plan, RECURSIVE_DOUBLING_MIN_SIZE);
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Unsupported data type %d", static_cast<int32_t>(param.dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No AICPU thread available"), HCCL_E_NOT_FOUND);

    constexpr uint64_t dataTypeSize = sizeof(float);
    const uint64_t dataSize = param.count * dataTypeSize;
    if (param.rankSize == 1) {
        CHK_RET(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize));
        return HCCL_SUCCESS;
    }

    const BigDirectPlan &bigPlan = resCtx.bigDirectPlan;
    if (bigPlan.magic == BIG_DIRECT_PLAN_MAGIC &&
        bigPlan.dataSize == dataSize) {
        return RunPreparedBigDirect(bigPlan);
    }

    CHK_PRT_RET(param.rankSize == 0 || resCtx.channels.size() != param.rankSize - 1 ||
            resCtx.threads.size() < resCtx.channels.size() ||
            resCtx.threads.size() > resCtx.channels.size() + 1,
        HCCL_ERROR("Resource mismatch: rankSize %u, threads %zu, channels %zu", param.rankSize,
            resCtx.threads.size(), resCtx.channels.size()),
        HCCL_E_INTERNAL);

    const bool useOneShotCcl = param.rankSize == 16 &&
        dataSize == RECURSIVE_DOUBLING_MIN_SIZE &&
        resCtx.threads.size() >= SMALL_MESSAGE_THREAD_NUM;
    if (useOneShotCcl) {
        bool hasOneShotBuffers = resCtx.localBuffer.addr != nullptr;
        const uint64_t packageSize = static_cast<uint64_t>(param.rankSize) * dataSize;
        hasOneShotBuffers = hasOneShotBuffers && resCtx.localBuffer.size >= packageSize;
        for (const ChannelInfo &channel : resCtx.channels) {
            if (channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size < packageSize) {
                hasOneShotBuffers = false;
                break;
            }
        }
        if (hasOneShotBuffers) {
            return RunOneShotCclAllGather(param, resCtx, dataSize);
        }
    }

    const bool useOutputRecursiveDoubling = param.rankSize == 16 &&
        dataSize >= RECURSIVE_DOUBLING_MIN_SIZE && dataSize < SMALL_MESSAGE_MAX_SIZE;
    if (useOutputRecursiveDoubling) {
        bool canReadRemoteOutput = true;
        const uint64_t outputSize = static_cast<uint64_t>(param.rankSize) * dataSize;
        for (const ChannelInfo &channel : resCtx.channels) {
            if (channel.remoteOutput.addr == nullptr || channel.remoteOutput.size < outputSize) {
                canReadRemoteOutput = false;
                break;
            }
        }
        if (canReadRemoteOutput) {
            return RunDirectOutputAllGather(param, resCtx, dataSize);
        }
    }

    bool canReadRemoteInput = true;
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteInput.addr == nullptr || channel.remoteInput.size < dataSize) {
            canReadRemoteInput = false;
            break;
        }
    }
    if (canReadRemoteInput) {
        return RunDirectInputAllGather(param, resCtx, dataSize);
    }

    // Push 模式为每个 rank 保留一个接收槽位；Pull 模式只保存本 rank 数据，可使用完整 Buffer。
    uint64_t cclBufferSize = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        cclBufferSize = std::min(cclBufferSize, channel.remoteCclMem.size);
    }
    const uint64_t maxPushSliceSize =
        cclBufferSize / param.rankSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    const uint64_t maxPushSliceCount = maxPushSliceSize / dataTypeSize;
    CHK_PRT_RET(maxPushSliceCount == 0,
        HCCL_ERROR("HCCL Buffer is too small: size %llu, rankSize %u",
            static_cast<unsigned long long>(cclBufferSize), param.rankSize),
        HCCL_E_MEMORY);

    // 小消息保留低握手开销的 Push 路径；无法单槽容纳时切换到大切片、直接落输出的 Pull 路径。
    if (dataSize <= maxPushSliceSize) {
        return RunPushAllGather(param, resCtx, dataTypeSize, dataSize, maxPushSliceCount);
    }

    const uint64_t maxPullSliceSize =
        cclBufferSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    const uint64_t maxPullSliceCount = maxPullSliceSize / dataTypeSize;
    CHK_PRT_RET(maxPullSliceCount == 0,
        HCCL_ERROR("HCCL Buffer is too small for Pull AllGather: size %llu",
            static_cast<unsigned long long>(cclBufferSize)),
        HCCL_E_MEMORY);
    return RunPullAllGather(param, resCtx, dataTypeSize, dataSize, maxPullSliceCount);
}
} // namespace ops_hccl
