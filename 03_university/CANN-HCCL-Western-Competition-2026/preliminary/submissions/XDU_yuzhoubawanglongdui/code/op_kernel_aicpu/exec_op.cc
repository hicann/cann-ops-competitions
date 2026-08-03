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
// =================================================
// Ring AllReduce 算法实现
//
// 整体分为两个阶段：
//   1. Reduce-Scatter（N-1 步）：数据块在环形拓扑上流转，每步归约
//      接收到的数据，最终每个 Rank 持有一个完整归约后的数据块。
//   2. AllGather（N-1 步）：各 Rank 将手中已归约的数据块沿环广播，
//      最终所有 Rank 拥有全部归约后的完整数据。
//
// 缓冲区布局：
//   workBuf = outputPtr：在输出缓冲区上原地执行 Ring AllReduce，
//      避免 localBuffer 容量不足（HCCL Buffer 默认 400MB，
//      当 totalSize > 400MB 时无法容纳完整数据副本）。
//   recvBuf = localBuffer.addr：HCCL Buffer 仅作为临时接收区，
//      需要的空间不超过 maxChunkSize（≤ totalSize / 2）。
//
// 发送方通过 HcommWriteWithNotifyOnThread 将数据写入远端接收缓冲区
// 并触发通知；接收方等待通知后从本地接收缓冲区读取数据。
// 工作区数据永不被远端写入覆盖，保证浮点归约顺序确定性。
//
// 同步协议（每步三握手）：
//   Sender:   WriteWithNotify(DATA_SIGNAL) → Wait ACK
//   Receiver: Wait DATA_SIGNAL → Reduce/Store → Record ACK
// =================================================

/// HcclDataType → HcommDataType 映射
static HcommDataType ConvertDataType(HcclDataType hcclType)
{
    switch (hcclType) {
        case HCCL_DATA_TYPE_FP32:
            return HCOMM_DATA_TYPE_FP32;
        case HCCL_DATA_TYPE_FP16:
            return HCOMM_DATA_TYPE_FP16;
        case HCCL_DATA_TYPE_INT32:
            return HCOMM_DATA_TYPE_INT32;
        case HCCL_DATA_TYPE_INT8:
            return HCOMM_DATA_TYPE_INT8;
        default:
            return HCOMM_DATA_TYPE_RESERVED;
    }
}

/// HcclReduceOp → HcommReduceOp 映射
static HcommReduceOp ConvertReduceOp(HcclReduceOp hcclOp)
{
    switch (hcclOp) {
        case HCCL_REDUCE_SUM:
            return HCOMM_REDUCE_SUM;
        case HCCL_REDUCE_PROD:
            return HCOMM_REDUCE_PROD;
        case HCCL_REDUCE_MAX:
            return HCOMM_REDUCE_MAX;
        case HCCL_REDUCE_MIN:
            return HCOMM_REDUCE_MIN;
        default:
            return HCOMM_REDUCE_RESERVED;
    }
}

/// 从 channels 中查找指向指定 rank 的 Channel 索引
static int32_t FindChannelByRank(const std::vector<ChannelInfo> &channels, uint32_t targetRank)
{
    for (size_t i = 0; i < channels.size(); i++) {
        if (channels[i].remoteRank == targetRank) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

/// Reduce-Scatter 阶段：计算第 step 步发送的 chunk 索引
static inline uint32_t GetSendChunkIdxReduceScatter(uint32_t myRank, uint32_t rankSize, uint32_t step)
{
    return (myRank + rankSize - step - 1) % rankSize;
}

/// Reduce-Scatter 阶段：计算第 step 步接收并归约的 chunk 索引
static inline uint32_t GetRecvChunkIdxReduceScatter(uint32_t myRank, uint32_t rankSize, uint32_t step)
{
    return (myRank + rankSize - step - 2) % rankSize;
}

/// AllGather 阶段：计算第 step 步发送的 chunk 索引
static inline uint32_t GetSendChunkIdxAllGather(uint32_t myRank, uint32_t rankSize, uint32_t step)
{
    return (myRank + rankSize - step) % rankSize;
}

/// AllGather 阶段：计算第 step 步接收的 chunk 索引
static inline uint32_t GetRecvChunkIdxAllGather(uint32_t myRank, uint32_t rankSize, uint32_t step)
{
    return (myRank + rankSize - step - 1) % rankSize;
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    uint32_t myRank = param.myRank;
    uint32_t rankSize = param.rankSize;

    // 获取数据类型大小
    auto it = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(it == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type: %d", param.dataType), HCCL_E_INTERNAL);
    uint32_t dataSize = it->second;
    uint64_t totalSize = param.count * dataSize;

    // 转换数据类型与归约操作为 HCOMM 枚举
    HcommDataType hcommDataType = ConvertDataType(param.dataType);
    HcommReduceOp hcommReduceOp = ConvertReduceOp(param.reduceType);
    CHK_PRT_RET(hcommDataType == HCOMM_DATA_TYPE_RESERVED, HCCL_ERROR("Unsupported HcclDataType: %d", param.dataType),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(hcommReduceOp == HCOMM_REDUCE_RESERVED, HCCL_ERROR("Unsupported HcclReduceOp: %d", param.reduceType),
        HCCL_E_INTERNAL);

    HCCL_INFO("ExecOp: myRank=%u, rankSize=%u, count=%lu, dataSize=%u", myRank, rankSize, param.count, dataSize);

    ThreadHandle thread = resCtx.aicpuThread;

    // ==============================================
    // 特殊情况：rankSize == 1，无需通信，直接拷贝
    // 必须在 Channel 查找之前处理，因为 rankSize==1 时不创建 Channel
    // ==============================================
    if (rankSize == 1) {
        if (param.inputPtr != param.outputPtr) {
            CHK_RET(HcommLocalCopyOnThread(thread, param.outputPtr, param.inputPtr, totalSize));
        }
        HCCL_INFO("ExecOp: single rank, copy done");
        return HCCL_SUCCESS;
    }

    // ==============================================
    // 查找 prev/next 对应的 Channel
    // ==============================================
    uint32_t prevRank = resCtx.algConfig.prevRank;
    uint32_t nextRank = resCtx.algConfig.nextRank;

    int32_t prevChIdx = FindChannelByRank(resCtx.channels, prevRank);
    int32_t nextChIdx = FindChannelByRank(resCtx.channels, nextRank);
    CHK_PRT_RET(prevChIdx < 0, HCCL_ERROR("Channel to prev rank %u not found", prevRank), HCCL_E_INTERNAL);
    CHK_PRT_RET(nextChIdx < 0, HCCL_ERROR("Channel to next rank %u not found", nextRank), HCCL_E_INTERNAL);

    const ChannelInfo &prevCh = resCtx.channels[prevChIdx];
    const ChannelInfo &nextCh = resCtx.channels[nextChIdx];

    // ==============================================
    // 数据分块与缓冲区布局
    // ==============================================
    // 以元素为单位划分 chunk，确保每个 chunk 边界对齐到 dataSize 的整数倍
    uint64_t chunkCount = param.count / rankSize;
    uint64_t remainderCount = param.count % rankSize;
    uint64_t chunkSize = chunkCount * dataSize;                             // 标准 chunk 字节数
    uint64_t lastChunkSize = (chunkCount + remainderCount) * dataSize;     // 末尾 chunk 字节数

    // 缓冲区布局（避免 localBuffer 溢出，支持 totalSize > cclBufferSize 的场景）：
    //   workBuf = outputPtr：在输出缓冲区上原地执行 AllReduce
    //   recvBuf = localBuffer.addr：HCCL Buffer 仅用于临时接收远端数据
    char *workBuf = static_cast<char *>(param.outputPtr);
    char *recvBuf = static_cast<char *>(resCtx.localBuffer.addr);

    HCCL_INFO("ExecOp: prev=%u, next=%u, chunkCount=%lu, remainderCount=%lu, chunkSize=%lu, lastChunkSize=%lu",
        prevRank, nextRank, chunkCount, remainderCount, chunkSize, lastChunkSize);

    // 将输入数据拷贝到输出缓冲区（作为工作区）；in-place 场景（inputPtr == outputPtr）则跳过
    if (param.inputPtr != param.outputPtr) {
        CHK_RET(HcommLocalCopyOnThread(thread, workBuf, param.inputPtr, totalSize));
    }

    // ==============================================
    // Phase 1: Ring Reduce-Scatter（rankSize - 1 步）
    // ==============================================
    HCCL_INFO("ExecOp: starting Reduce-Scatter phase, steps=%u", rankSize - 1);
    for (uint32_t step = 0; step < rankSize - 1; step++) {
        uint32_t sendIdx = GetSendChunkIdxReduceScatter(myRank, rankSize, step);
        uint32_t recvIdx = GetRecvChunkIdxReduceScatter(myRank, rankSize, step);

        uint64_t sendOffset = sendIdx * chunkSize;
        uint64_t sendLen = (sendIdx == rankSize - 1) ? lastChunkSize : chunkSize;
        uint64_t recvOffset = recvIdx * chunkSize;
        uint64_t recvElemCount = (recvIdx == rankSize - 1) ? (chunkCount + remainderCount) : chunkCount;

        HCCL_DEBUG("ReduceScatter step[%u]: send chunk[%u](%luB) -> rank[%u], "
                   "recv chunk[%u](%lu elems) <- rank[%u]",
            step, sendIdx, sendLen, nextRank, recvIdx, recvElemCount, prevRank);

        // Step A: 将 send chunk 写入 nextRank 的接收区 + 触发 DATA_SIGNAL 通知
        // 注意：数据量极小时某些 chunk 可能为空（sendLen==0），但三维握手协议
        // 要求每步都必须发送通知。HcommChannelNotifyRecordOnThread 在此模拟器
        // 上不被支持（opcode 0），故统一使用 WriteWithNotify，空 chunk 也写 1 字节
        // 占位以确保通知触发。接收端的 recvLen==0 保证不会读取该占位字节。
        CHK_RET(HcommWriteWithNotifyOnThread(thread, nextCh.handle,
            static_cast<char *>(nextCh.remoteCclMem.addr), workBuf + sendOffset,
            (sendLen > 0) ? sendLen : 1,
            NOTIFY_IDX_DATA_SIGNAL));

        // Step B: 等待 prevRank 数据到达本端 recvBuf
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, prevCh.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        // prevRank 已将数据写入 recvBuf，与本地 chunk[recvIdx] 做原地归约 dst=dst OP src
        if (recvElemCount > 0) {
            CHK_RET(HcommLocalReduceOnThread(thread, workBuf + recvOffset, recvBuf, recvElemCount, hcommDataType,
                hcommReduceOp));
        }
        // 通知 prevRank 数据已消费（发送 ACK）
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, prevCh.handle, NOTIFY_IDX_ACK));

        // Step C: 等待 nextRank 确认数据已消费（ACK）
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, nextCh.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    }

    HCCL_INFO("ExecOp: Reduce-Scatter phase completed");

    // ==============================================
    // Phase 2: Ring AllGather（rankSize - 1 步）
    // ==============================================
    HCCL_INFO("ExecOp: starting AllGather phase, steps=%u", rankSize - 1);
    for (uint32_t step = 0; step < rankSize - 1; step++) {
        uint32_t sendIdx = GetSendChunkIdxAllGather(myRank, rankSize, step);
        uint32_t recvIdx = GetRecvChunkIdxAllGather(myRank, rankSize, step);

        uint64_t sendOffset = sendIdx * chunkSize;
        uint64_t sendLen = (sendIdx == rankSize - 1) ? lastChunkSize : chunkSize;
        uint64_t recvOffset = recvIdx * chunkSize;
        uint64_t recvLen = (recvIdx == rankSize - 1) ? lastChunkSize : chunkSize;

        HCCL_DEBUG("AllGather step[%u]: send chunk[%u](%luB) -> rank[%u], "
                   "recv chunk[%u](%luB) <- rank[%u]",
            step, sendIdx, sendLen, nextRank, recvIdx, recvLen, prevRank);

        // Step A: 将已归约 chunk 写入 nextRank 的接收区 + 触发通知
        // 空 chunk（sendLen==0）也写 1 字节占位以确保通知触发。
        // 接收端 recvLen==0 不会读取该占位字节。
        CHK_RET(HcommWriteWithNotifyOnThread(thread, nextCh.handle,
            static_cast<char *>(nextCh.remoteCclMem.addr), workBuf + sendOffset,
            (sendLen > 0) ? sendLen : 1,
            NOTIFY_IDX_DATA_SIGNAL));

        // Step B: 等待 prevRank 数据到达，拷贝到 workBuf 对应 chunk 位置
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, prevCh.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        if (recvLen > 0) {
            CHK_RET(HcommLocalCopyOnThread(thread, workBuf + recvOffset, recvBuf, recvLen));
        }
        // 通知 prevRank 数据已消费
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, prevCh.handle, NOTIFY_IDX_ACK));

        // Step C: 等待 nextRank 确认数据已消费
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, nextCh.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    }

    HCCL_INFO("ExecOp: AllGather phase completed");

    // 算法在 outputPtr 上原地执行，结果已在输出缓冲区中，无需额外拷贝
    HCCL_INFO("ExecOp: AllReduce completed successfully");
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
