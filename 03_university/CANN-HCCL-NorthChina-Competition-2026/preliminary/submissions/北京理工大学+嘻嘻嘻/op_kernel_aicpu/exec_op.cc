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
#include <iterator>
#include <limits>
#include <vector>

namespace {
constexpr uint64_t MAX_CHUNK_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t PARALLEL_MIN_BYTES = 512ULL * 1024ULL;
constexpr uint64_t SHARED_STAGE_MIN_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t SHARED_STAGE_BYTES = 400ULL * 1024ULL * 1024ULL;
constexpr uint32_t HIERARCHICAL_RANK_SIZE = 16;
constexpr uint32_t HIERARCHICAL_GROUP_SIZE = 8;
constexpr uint32_t HIERARCHICAL_PEER_NUM = HIERARCHICAL_RANK_SIZE - 1;
constexpr uint32_t HIERARCHICAL_THREAD_NUM = HIERARCHICAL_RANK_SIZE;
constexpr uint32_t MIRROR_THREAD_INDEX = HIERARCHICAL_GROUP_SIZE;
constexpr uint32_t WORKER_START_NOTIFY_INDEX = 0;
constexpr uint32_t MIRROR_READY_NOTIFY_INDEX = 1;

/**
 * @brief 在单个 Channel 上按 scratch 容量和 256 MiB 上限分片交换连续数据
 * @param thread 承载通信任务的 AICPU thread
 * @param channel 目标 peer 的 Channel 资源
 * @param localBuffer 本端接收远端写入的 HCCL scratch
 * @param source 本端需要发送的连续数据起始地址
 * @param destination 远端数据在本端输出中的连续目标地址
 * @param dataBytes 需要交换的总字节数
 * @return 全部分片任务下发成功返回 HCCL_SUCCESS，否则返回首个错误码
 */
static HcclResult ExchangeChunks(ThreadHandle thread, const ChannelInfo &channel, const CommBuffer &localBuffer,
    const uint8_t *source, uint8_t *destination, uint64_t dataBytes)
{
    uint64_t maxChunkBytes = std::min(MAX_CHUNK_BYTES, localBuffer.size);
    maxChunkBytes = std::min(maxChunkBytes, channel.remoteCclMem.size);
    maxChunkBytes -= maxChunkBytes % sizeof(float);
    CHK_PRT_RET(maxChunkBytes == 0,
        HCCL_ERROR("No aligned transfer space for remote rank %u", channel.remoteRank), HCCL_E_INTERNAL);

    uint64_t processedBytes = 0;
    while (processedBytes < dataBytes) {
        const uint64_t chunkBytes = std::min(maxChunkBytes, dataBytes - processedBytes);
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        CHK_RET(HcommWriteOnThread(
            thread, channel.handle, channel.remoteCclMem.addr, source + processedBytes, chunkBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(
            thread, destination + processedBytes, localBuffer.addr, chunkBytes));
        processedBytes += chunkBytes;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 计算并校验分层并行交换使用的本端 lane、远端 lane 和单片上限
 * @param localBuffer 本端接收远端写入的 HCCL scratch
 * @param channel 目标 peer 的 Channel 资源
 * @param localStride 输出的本端单个 lane 字节跨度
 * @param remoteStride 输出的远端单个 lane 字节跨度
 * @param maxChunkBytes 输出的单次交换最大字节数
 * @return lane 容量有效返回 HCCL_SUCCESS，否则返回对应错误码
 */
static HcclResult GetLaneTransferLimits(const CommBuffer &localBuffer, const ChannelInfo &channel,
    uint64_t &localStride, uint64_t &remoteStride, uint64_t &maxChunkBytes)
{
    localStride = localBuffer.size / HIERARCHICAL_GROUP_SIZE;
    localStride -= localStride % sizeof(float);
    remoteStride = channel.remoteCclMem.size / HIERARCHICAL_GROUP_SIZE;
    remoteStride -= remoteStride % sizeof(float);
    maxChunkBytes = std::min(MAX_CHUNK_BYTES, localStride);
    maxChunkBytes = std::min(maxChunkBytes, remoteStride);
    CHK_PRT_RET(maxChunkBytes == 0,
        HCCL_ERROR("No aligned parallel transfer space for remote rank %u", channel.remoteRank), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

/**
 * @brief 在单个 Channel 的独立收发 lane 上分片交换连续数据
 * @param thread 承载通信任务的 AICPU thread
 * @param channel 目标 peer 的 Channel 资源
 * @param localBuffer 本端接收远端写入的 HCCL scratch
 * @param senderLocalIndex 本 rank 在本 Server 内的索引，用于选择远端写入 lane
 * @param peerLocalIndex 对端 rank 在本 Server 内的索引，用于选择本端读取 lane
 * @param source 本端需要发送的连续数据起始地址
 * @param destination 远端数据在本端输出中的连续目标地址
 * @param dataBytes 需要交换的总字节数
 * @return 全部分片任务下发成功返回 HCCL_SUCCESS，否则返回首个错误码
 */
static HcclResult ExchangeChunksInLane(ThreadHandle thread, const ChannelInfo &channel,
    const CommBuffer &localBuffer, uint32_t senderLocalIndex, uint32_t peerLocalIndex,
    const uint8_t *source, uint8_t *destination, uint64_t dataBytes)
{
    CHK_PRT_RET(senderLocalIndex >= HIERARCHICAL_GROUP_SIZE || peerLocalIndex >= HIERARCHICAL_GROUP_SIZE,
        HCCL_ERROR("Invalid parallel lane index for remote rank %u", channel.remoteRank), HCCL_E_INTERNAL);

    uint64_t localStride = 0;
    uint64_t remoteStride = 0;
    uint64_t maxChunkBytes = 0;
    CHK_RET(GetLaneTransferLimits(localBuffer, channel, localStride, remoteStride, maxChunkBytes));
    uint8_t *localLaneAddr = static_cast<uint8_t *>(localBuffer.addr) + peerLocalIndex * localStride;
    uint8_t *remoteLaneAddr =
        static_cast<uint8_t *>(channel.remoteCclMem.addr) + senderLocalIndex * remoteStride;

    uint64_t processedBytes = 0;
    while (processedBytes < dataBytes) {
        const uint64_t chunkBytes = std::min(maxChunkBytes, dataBytes - processedBytes);
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        CHK_RET(HcommWriteOnThread(thread, channel.handle, remoteLaneAddr, source + processedBytes, chunkBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(thread, destination + processedBytes, localLaneAddr, chunkBytes));
        processedBytes += chunkBytes;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 在主 thread 完成已排任务后启动指定数量的分层并行 worker
 * @param threads 分层并行使用的 thread 列表，首项为主 thread
 * @param workerCount 需要启动的前缀 worker 数量，对应 threads[1] 到 threads[workerCount]
 * @param notifyIndex worker 等待主 thread 启动信号使用的 notify 索引
 * @return 启动同步任务下发成功返回 HCCL_SUCCESS，否则返回首个错误码
 */
static HcclResult StartWorkerThreads(
    const std::vector<ThreadHandle> &threads, uint32_t workerCount, uint32_t notifyIndex)
{
    CHK_PRT_RET(threads.size() != HIERARCHICAL_THREAD_NUM || workerCount == 0 ||
            workerCount >= threads.size() || notifyIndex >= HIERARCHICAL_THREAD_NUM,
        HCCL_ERROR("Invalid thread count for worker start"), HCCL_E_INTERNAL);
    for (uint32_t threadIndex = 1; threadIndex <= workerCount; ++threadIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(threads[0], threads[threadIndex], notifyIndex));
    }
    for (uint32_t threadIndex = 1; threadIndex <= workerCount; ++threadIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(threads[threadIndex], notifyIndex, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 汇聚指定数量的分层并行 worker 完成通知到主 thread
 * @param threads 分层并行使用的 thread 列表，首项为主 thread
 * @param workerCount 需要汇聚的前缀 worker 数量，对应 threads[1] 到 threads[workerCount]
 * @return 完成同步任务下发成功返回 HCCL_SUCCESS，否则返回首个错误码
 */
static HcclResult FinishWorkerThreads(const std::vector<ThreadHandle> &threads, uint32_t workerCount)
{
    CHK_PRT_RET(threads.size() != HIERARCHICAL_THREAD_NUM || workerCount == 0 || workerCount >= threads.size(),
        HCCL_ERROR("Invalid thread count for worker completion"), HCCL_E_INTERNAL);
    for (uint32_t threadIndex = 1; threadIndex <= workerCount; ++threadIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(threads[0], threadIndex, CUSTOM_TIMEOUT));
    }
    for (uint32_t threadIndex = 1; threadIndex <= workerCount; ++threadIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(threads[threadIndex], threads[0], threadIndex));
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 检查全部分层 Channel 是否具备指定 canonical staging 容量
 * @param resCtx 当前算子的算法资源上下文
 * @param requiredBytes canonical staging 需要的总字节数
 * @return 本端和全部远端 HCCL buffer 容量均充足返回 true，否则返回 false
 */
static bool HasCanonicalStageCapacity(const AlgResourceCtx &resCtx, uint64_t requiredBytes)
{
    if (requiredBytes == 0 || resCtx.localBuffer.size < requiredBytes) {
        return false;
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteCclMem.size < requiredBytes) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 检查全部分层 Channel 是否具备完整 400 MiB 共享 staging 容量
 * @param resCtx 当前算子的算法资源上下文
 * @return 本端和全部远端 HCCL buffer 容量均不小于 400 MiB 返回 true，否则返回 false
 */
static bool HasSharedStageCapacity(const AlgResourceCtx &resCtx)
{
    return HasCanonicalStageCapacity(resCtx, SHARED_STAGE_BYTES);
}

/**
 * @brief 在双方 staging 就绪后从指定 peer 的 HCCL buffer 起始位置直读一个分片
 * @param thread 承载当前 peer 直读任务的 worker thread
 * @param channel 当前 peer 的 Channel 资源
 * @param destination 当前分片在本端的目标地址
 * @param chunkBytes 当前分片的字节数
 * @return 直读及前后同步任务下发成功返回 HCCL_SUCCESS，否则返回首个错误码
 */
static HcclResult ReadPeerStageChunk(ThreadHandle thread, const ChannelInfo &channel,
    uint8_t *destination, uint64_t chunkBytes)
{
    CHK_PRT_RET(chunkBytes > MAX_CHUNK_BYTES || chunkBytes > channel.remoteCclMem.size,
        HCCL_ERROR("Remote staging range exceeds buffer for rank %u", channel.remoteRank), HCCL_E_INTERNAL);
    const uint8_t *remoteStage = static_cast<const uint8_t *>(channel.remoteCclMem.addr);
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    CHK_RET(HcommReadOnThread(thread, channel.handle, destination, remoteStage, chunkBytes));
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

/**
 * @brief 使用镜像交换和 XOR 递归倍增完成 2x8 小包分层 AllGather
 * @param param 当前 AllGather 的算子参数
 * @param resCtx 当前算子的算法资源上下文
 * @param channelsByRank 按全局 rank 索引的 Channel 指针列表
 * @param localGroupStart 本 Server 连续 rank 组的起始全局 rank
 * @param remoteGroupStart 对端 Server 连续 rank 组的起始全局 rank
 * @param localIndex 本 rank 在 Server 内的索引
 * @param dataBytes 单个 rank 数据的总字节数
 * @return canonical staging 聚合并复制到输出成功返回 HCCL_SUCCESS，否则返回首个错误码
 */
static HcclResult ExecuteSmallHierarchicalAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelsByRank, uint32_t localGroupStart,
    uint32_t remoteGroupStart, uint32_t localIndex, uint64_t dataBytes)
{
    const ThreadHandle thread = resCtx.threads[0];
    uint8_t *localStage = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    const uint64_t resultBytes = static_cast<uint64_t>(param.rankSize) * dataBytes;

    CHK_RET(HcommLocalCopyOnThread(
        thread, localStage + static_cast<uint64_t>(param.myRank) * dataBytes, input, dataBytes));

    const ChannelInfo &mirrorChannel = *channelsByRank[resCtx.interPeerRank];
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, mirrorChannel.handle, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, mirrorChannel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    const uint64_t mirrorOffset = static_cast<uint64_t>(resCtx.interPeerRank) * dataBytes;
    CHK_RET(HcommReadOnThread(thread, mirrorChannel.handle, localStage + mirrorOffset,
        static_cast<const uint8_t *>(mirrorChannel.remoteCclMem.addr) + mirrorOffset, dataBytes));
    CHK_RET(HcommChannelNotifyRecordOnThread(
        thread, mirrorChannel.handle, NOTIFY_IDX_DATA_SIGNAL));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, mirrorChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

    // 每轮从 XOR partner 读取两个 Server 上已经聚合完成的连续数据段
    for (uint32_t mask = 1; mask < HIERARCHICAL_GROUP_SIZE; mask <<= 1) {
        const uint32_t partnerLocalIndex = localIndex ^ mask;
        const uint32_t partnerRank = localGroupStart + partnerLocalIndex;
        const uint32_t fetchBase = partnerLocalIndex & ~(mask - 1U);
        const uint64_t segmentBytes = static_cast<uint64_t>(mask) * dataBytes;
        const ChannelInfo &channel = *channelsByRank[partnerRank];
        const uint8_t *remoteStage = static_cast<const uint8_t *>(channel.remoteCclMem.addr);

        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        const uint64_t localOffset =
            static_cast<uint64_t>(localGroupStart + fetchBase) * dataBytes;
        CHK_RET(HcommReadOnThread(
            thread, channel.handle, localStage + localOffset, remoteStage + localOffset, segmentBytes));
        const uint64_t remoteOffset =
            static_cast<uint64_t>(remoteGroupStart + fetchBase) * dataBytes;
        CHK_RET(HcommReadOnThread(
            thread, channel.handle, localStage + remoteOffset, remoteStage + remoteOffset, segmentBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    }

    CHK_RET(HcommLocalCopyOnThread(thread, output, localStage, resultBytes));
    return HCCL_SUCCESS;
}

/**
 * @brief 使用十五条独立 peer 链路并行直读完成 2x8 大包分层 AllGather
 * @param param 当前 AllGather 的算子参数
 * @param resCtx 当前算子的算法资源上下文
 * @param channelsByRank 按全局 rank 索引的 Channel 指针列表
 * @param peerRanks 除本 rank 外按全局 XOR 掩码排列的 peer 列表
 * @param dataBytes 单个 rank 数据的总字节数
 * @return 全 peer staging 直读任务下发成功返回 HCCL_SUCCESS，否则返回首个错误码
 */
static HcclResult ExecuteDirectReadHierarchicalAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelsByRank, const std::vector<uint32_t> &peerRanks,
    uint64_t dataBytes)
{
    CHK_PRT_RET(peerRanks.size() != HIERARCHICAL_PEER_NUM,
        HCCL_ERROR("Invalid peer count for direct read"), HCCL_E_INTERNAL);
    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *localStage = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);

    uint64_t processedBytes = 0;
    while (processedBytes < dataBytes) {
        const uint64_t chunkBytes = std::min(MAX_CHUNK_BYTES, dataBytes - processedBytes);
        CHK_RET(HcommLocalCopyOnThread(
            mainThread, localStage, input + processedBytes, chunkBytes));
        CHK_RET(StartWorkerThreads(
            resCtx.threads, HIERARCHICAL_PEER_NUM, WORKER_START_NOTIFY_INDEX));

        // threads[1..15] 分别负责一个按全局 XOR 掩码映射的 peer
        for (size_t peerIndex = 0; peerIndex < peerRanks.size(); ++peerIndex) {
            const uint32_t remoteRank = peerRanks[peerIndex];
            const size_t threadIndex = peerIndex + 1;
            CHK_RET(ReadPeerStageChunk(resCtx.threads[threadIndex], *channelsByRank[remoteRank],
                output + static_cast<uint64_t>(remoteRank) * dataBytes + processedBytes, chunkBytes));
        }

        // 主 thread 仅负责 staging 准备、worker 启动、本 rank 复制和 worker 汇聚
        CHK_RET(HcommLocalCopyOnThread(mainThread,
            output + static_cast<uint64_t>(param.myRank) * dataBytes + processedBytes,
            localStage, chunkBytes));
        CHK_RET(FinishWorkerThreads(
            resCtx.threads, HIERARCHICAL_PEER_NUM));
        processedBytes += chunkBytes;
    }
    return HCCL_SUCCESS;
}
}

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Only FP32 is supported, data type is %d", static_cast<int>(param.dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("AllGather count overflows byte size"), HCCL_E_PARA);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank information, rank %u of %u", param.myRank, param.rankSize), HCCL_E_PARA);
    const bool hierarchical = resCtx.interPeerRank != INVALID_VALUE_RANKID;
    const size_t expectedThreadNum = hierarchical ? HIERARCHICAL_THREAD_NUM : 1;
    CHK_PRT_RET(resCtx.threads.size() != expectedThreadNum || resCtx.aicpuThread == 0 ||
            resCtx.aicpuThread != resCtx.threads[0] ||
            std::find(resCtx.threads.begin(), resCtx.threads.end(), 0) != resCtx.threads.end(),
        HCCL_ERROR("Invalid or inconsistent AICPU communication thread"), HCCL_E_INTERNAL);
    std::vector<ThreadHandle> sortedThreads = resCtx.threads;
    std::sort(sortedThreads.begin(), sortedThreads.end());
    CHK_PRT_RET(std::adjacent_find(sortedThreads.begin(), sortedThreads.end()) != sortedThreads.end(),
        HCCL_ERROR("Duplicate AICPU communication thread"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < sizeof(float),
        HCCL_ERROR("Invalid local HCCL buffer"), HCCL_E_INTERNAL);

    const uint64_t dataBytes = param.count * sizeof(float);
    CHK_PRT_RET(param.rankSize > std::numeric_limits<uint64_t>::max() / dataBytes,
        HCCL_ERROR("AllGather receive size overflows"), HCCL_E_PARA);

    CHK_PRT_RET(hierarchical &&
            (param.rankSize != HIERARCHICAL_RANK_SIZE || resCtx.interPeerRank >= param.rankSize ||
                resCtx.interPeerRank == param.myRank ||
                resCtx.intraGroupStart > HIERARCHICAL_RANK_SIZE - HIERARCHICAL_GROUP_SIZE),
        HCCL_ERROR("Invalid hierarchical AllGather metadata"), HCCL_E_INTERNAL);
    const size_t expectedChannelNum = param.rankSize - 1;
    CHK_PRT_RET(resCtx.channels.size() != expectedChannelNum,
        HCCL_ERROR("Channel count does not match rank size"), HCCL_E_INTERNAL);

    std::vector<const ChannelInfo *> channelsByRank(param.rankSize, nullptr);
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank,
            HCCL_ERROR("Invalid remote rank %u", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channelsByRank[channel.remoteRank] != nullptr,
            HCCL_ERROR("Duplicate Channel for remote rank %u", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.handle == 0 || channel.notifyNum <= NOTIFY_IDX_DATA_SIGNAL,
            HCCL_ERROR("Invalid Channel resource for remote rank %u", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size < sizeof(float),
            HCCL_ERROR("Invalid remote HCCL buffer for rank %u", channel.remoteRank), HCCL_E_INTERNAL);
        channelsByRank[channel.remoteRank] = &channel;
    }
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank != param.myRank) {
            CHK_PRT_RET(channelsByRank[remoteRank] == nullptr,
                HCCL_ERROR("Channel for remote rank %u is missing", remoteRank), HCCL_E_INTERNAL);
        }
    }

    std::vector<uint32_t> intraRanks;
    std::vector<uint32_t> remoteRanks;
    uint32_t localGroupStart = INVALID_VALUE_RANKID;
    uint32_t remoteGroupStart = INVALID_VALUE_RANKID;
    uint32_t localIndex = INVALID_VALUE_RANKID;
    if (hierarchical) {
        CHK_PRT_RET(channelsByRank[resCtx.interPeerRank] == nullptr,
            HCCL_ERROR("Mirror Channel for rank %u is missing", resCtx.interPeerRank), HCCL_E_INTERNAL);
        localGroupStart = resCtx.intraGroupStart;
        const uint32_t localGroupEnd = localGroupStart + HIERARCHICAL_GROUP_SIZE;
        CHK_PRT_RET(param.myRank < localGroupStart || param.myRank >= localGroupEnd,
            HCCL_ERROR("Local rank is outside the hierarchical intra-server group"), HCCL_E_INTERNAL);
        for (uint32_t rank = localGroupStart; rank < localGroupEnd; ++rank) {
            intraRanks.push_back(rank);
        }

        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank < localGroupStart || rank >= localGroupEnd) {
                remoteRanks.push_back(rank);
            }
        }
        bool remoteGroupValid = remoteRanks.size() == HIERARCHICAL_GROUP_SIZE;
        for (uint32_t index = 1; remoteGroupValid && index < HIERARCHICAL_GROUP_SIZE; ++index) {
            remoteGroupValid = remoteRanks[index] == remoteRanks[0] + index;
        }
        CHK_PRT_RET(!remoteGroupValid,
            HCCL_ERROR("Invalid remote-server rank group for hierarchical AllGather"), HCCL_E_INTERNAL);

        localIndex = param.myRank - localGroupStart;
        CHK_PRT_RET(localIndex >= remoteRanks.size() || remoteRanks[localIndex] != resCtx.interPeerRank,
            HCCL_ERROR("Mirror rank does not match local rank index"), HCCL_E_INTERNAL);
        remoteGroupStart = remoteRanks[0];
    }

    const ThreadHandle thread = resCtx.aicpuThread;
    const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    std::vector<uint32_t> intraPeerRanks;
    std::vector<uint32_t> peerRanks;
    if (hierarchical) {
        peerRanks.reserve(HIERARCHICAL_PEER_NUM);
        // 完整 XOR 排列让同一 peer 的双方使用相同 worker，并分散同一批次的远端源
        for (uint32_t mask = 1; mask < param.rankSize; ++mask) {
            peerRanks.push_back(param.myRank ^ mask);
        }
        CHK_PRT_RET(peerRanks.size() != HIERARCHICAL_PEER_NUM,
            HCCL_ERROR("Invalid hierarchical peer count"), HCCL_E_INTERNAL);
        for (uint32_t rank : intraRanks) {
            if (rank != param.myRank) {
                intraPeerRanks.push_back(rank);
            }
        }
        CHK_PRT_RET(intraPeerRanks.size() + 1 != HIERARCHICAL_GROUP_SIZE,
            HCCL_ERROR("Invalid intra-server peer count for parallel AllGather"), HCCL_E_INTERNAL);
    }

    const uint64_t canonicalStageBytes = static_cast<uint64_t>(param.rankSize) * dataBytes;
    const bool smallHierarchical = hierarchical && canonicalStageBytes == PARALLEL_MIN_BYTES &&
        HasCanonicalStageCapacity(resCtx, canonicalStageBytes);
    const bool sharedStageHierarchical = hierarchical && canonicalStageBytes >= SHARED_STAGE_MIN_BYTES &&
        HasSharedStageCapacity(resCtx);
    const bool parallelHierarchical = hierarchical && canonicalStageBytes >= PARALLEL_MIN_BYTES;
    if (parallelHierarchical && !smallHierarchical && !sharedStageHierarchical) {
        for (uint32_t remoteRank : intraPeerRanks) {
            uint64_t localStride = 0;
            uint64_t remoteStride = 0;
            uint64_t maxChunkBytes = 0;
            CHK_RET(GetLaneTransferLimits(resCtx.localBuffer, *channelsByRank[remoteRank],
                localStride, remoteStride, maxChunkBytes));
        }
        uint64_t localStride = 0;
        uint64_t remoteStride = 0;
        uint64_t maxChunkBytes = 0;
        CHK_RET(GetLaneTransferLimits(resCtx.localBuffer, *channelsByRank[resCtx.interPeerRank],
            localStride, remoteStride, maxChunkBytes));
    }

    if (smallHierarchical) {
        HCCL_INFO("Selected small hierarchical AllGather, rank bytes %llu, total bytes %llu",
            static_cast<unsigned long long>(dataBytes),
            static_cast<unsigned long long>(canonicalStageBytes));
        return ExecuteSmallHierarchicalAllGather(param, resCtx, channelsByRank,
            localGroupStart, remoteGroupStart, localIndex, dataBytes);
    }
    if (sharedStageHierarchical) {
        HCCL_INFO("Selected direct read AllGather, rank bytes %llu, total bytes %llu",
            static_cast<unsigned long long>(dataBytes),
            static_cast<unsigned long long>(canonicalStageBytes));
        return ExecuteDirectReadHierarchicalAllGather(
            param, resCtx, channelsByRank, peerRanks, dataBytes);
    }

    uint64_t processedBytes = 0;
    while (processedBytes < dataBytes) {
        const uint64_t chunkBytes = std::min(MAX_CHUNK_BYTES, dataBytes - processedBytes);
        CHK_RET(HcommLocalCopyOnThread(
            thread, output + static_cast<uint64_t>(param.myRank) * dataBytes + processedBytes,
            input + processedBytes, chunkBytes));
        processedBytes += chunkBytes;
    }

    if (parallelHierarchical) {
        CHK_RET(StartWorkerThreads(
            resCtx.threads, HIERARCHICAL_GROUP_SIZE, WORKER_START_NOTIFY_INDEX));

        // 第一轮将本 rank 数据同时发送给七个组内 peer 和镜像 peer
        for (size_t peerIndex = 0; peerIndex < intraPeerRanks.size(); ++peerIndex) {
            const uint32_t remoteRank = intraPeerRanks[peerIndex];
            const uint32_t peerLocalIndex = remoteRank - localGroupStart;
            const uint32_t threadIndex = static_cast<uint32_t>(peerIndex) + 1;
            CHK_RET(ExchangeChunksInLane(resCtx.threads[threadIndex], *channelsByRank[remoteRank],
                resCtx.localBuffer, localIndex, peerLocalIndex, input,
                output + static_cast<uint64_t>(remoteRank) * dataBytes, dataBytes));
        }

        const ThreadHandle mirrorThread = resCtx.threads[MIRROR_THREAD_INDEX];
        const ChannelInfo &mirrorChannel = *channelsByRank[resCtx.interPeerRank];
        CHK_RET(ExchangeChunksInLane(mirrorThread, mirrorChannel, resCtx.localBuffer, localIndex, localIndex,
            input, output + static_cast<uint64_t>(resCtx.interPeerRank) * dataBytes, dataBytes));
        // 镜像数据写入输出后放行七个组内 worker
        for (uint32_t threadIndex = 1; threadIndex < MIRROR_THREAD_INDEX; ++threadIndex) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                mirrorThread, resCtx.threads[threadIndex], MIRROR_READY_NOTIFY_INDEX));
        }

        const uint8_t *mirrorData = output + static_cast<uint64_t>(resCtx.interPeerRank) * dataBytes;
        // 第二轮通过组内 peer 扩散各自收到的镜像数据
        for (size_t peerIndex = 0; peerIndex < intraPeerRanks.size(); ++peerIndex) {
            const uint32_t remoteRank = intraPeerRanks[peerIndex];
            const uint32_t peerLocalIndex = remoteRank - localGroupStart;
            const uint32_t threadIndex = static_cast<uint32_t>(peerIndex) + 1;
            CHK_RET(HcommThreadNotifyWaitOnThread(
                resCtx.threads[threadIndex], MIRROR_READY_NOTIFY_INDEX, CUSTOM_TIMEOUT));
            const uint32_t remoteMirrorRank = remoteRanks[peerLocalIndex];
            CHK_RET(ExchangeChunksInLane(resCtx.threads[threadIndex], *channelsByRank[remoteRank],
                resCtx.localBuffer, localIndex, peerLocalIndex, mirrorData,
                output + static_cast<uint64_t>(remoteMirrorRank) * dataBytes, dataBytes));
        }
        CHK_RET(FinishWorkerThreads(resCtx.threads, HIERARCHICAL_GROUP_SIZE));
    } else if (hierarchical) {
        for (uint32_t remoteRank : intraRanks) {
            if (remoteRank == param.myRank) {
                continue;
            }
            CHK_RET(ExchangeChunks(thread, *channelsByRank[remoteRank], resCtx.localBuffer, input,
                output + static_cast<uint64_t>(remoteRank) * dataBytes, dataBytes));
        }

        const uint64_t aggregateBytes = static_cast<uint64_t>(HIERARCHICAL_GROUP_SIZE) * dataBytes;
        const uint8_t *localGroupData = output + static_cast<uint64_t>(localGroupStart) * dataBytes;
        uint8_t *remoteGroupData = output + static_cast<uint64_t>(remoteGroupStart) * dataBytes;
        CHK_RET(ExchangeChunks(thread, *channelsByRank[resCtx.interPeerRank], resCtx.localBuffer,
            localGroupData, remoteGroupData, aggregateBytes));
    } else {
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
            if (remoteRank == param.myRank) {
                continue;
            }
            CHK_RET(ExchangeChunks(thread, *channelsByRank[remoteRank], resCtx.localBuffer, input,
                output + static_cast<uint64_t>(remoteRank) * dataBytes, dataBytes));
        }
    }

    return HCCL_SUCCESS;
}
} // 命名空间 ops_hccl
