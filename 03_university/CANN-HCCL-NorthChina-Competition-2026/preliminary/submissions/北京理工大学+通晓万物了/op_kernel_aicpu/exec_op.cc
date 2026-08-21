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
#include <vector>

namespace ops_hccl {
namespace {
// 对齐限制
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
// 单次循环搬运的数据上限（UB 大小），超过后按 chunk 切分循环处理
constexpr uint64_t UB_MAX_DATA_SIZE = 256ULL * 1024ULL * 1024ULL;
// 每条 channel 上的 3 个 notify 索引定义
constexpr uint32_t NOTIFY_IDX_ACK = 0;          // 对端数据准备完毕，可以开始 DMA 写入
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;  // DMA 写入完成
constexpr uint32_t NOTIFY_IDX_FIN_ACK = 2;      // 预留
constexpr uint32_t CUSTOM_TIMEOUT = 1836;

// 全局前后同步：主thread先通知所有从thread，再等待从thread就绪，确保后续并行通信阶段从thread可用
HcclResult ThreadSyncBefore(const AlgResourceCtx &resCtx)
{
    const uint32_t n = resCtx.threads.size();
    for (uint32_t i = 1; i < n; i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[i], 0)));
    }
    for (uint32_t i = 1; i < n; i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[i], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

// 全局后同步：等待所有从thread完成，再让其向主thread汇报
// 主thread上的slot：slot 0 留给 host/device 同步；从thread使用slot 1..N-1 汇报
HcclResult ThreadSyncAfter(const AlgResourceCtx &resCtx)
{
    const uint32_t n = resCtx.threads.size();
    // 主thread 依次等待每个从thread的完成信号（slot i 与从thread编号 i 对应）
    for (uint32_t i = 1; i < n; i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[0], i, CUSTOM_TIMEOUT)));
    }
    // 从thread 向主thread的第 i 个 slot 汇报
    for (uint32_t i = 1; i < n; i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[i], resCtx.threads[0], i)));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU: AllGather");

    // 获取单个 dataType 的字节数（HCCL_DATA_TYPE_FP32 -> sizeof(float)）
    auto it = SIZE_TABLE.find(param.dataType);
    if (it == SIZE_TABLE.end()) {
        HCCL_ERROR("[ExecOp] Unsupported dataType: %d", param.dataType);
        return HCCL_E_NOT_SUPPORT;
    }
    const uint32_t dataTypeSize = it->second;
    const uint64_t count = param.count;
    const uint64_t dataSize = count * dataTypeSize;

    // 单卡场景：仅做本地搬运
    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.aicpuThread, param.outputPtr, param.inputPtr, dataSize)));
        return HCCL_SUCCESS;
    }

    // 计算单次循环的最大可处理切片大小，受 UB 与 CCL buffer 容量限制
    // CCL buffer 总大小需除以 rankSize（每块rank一个大小为slice的槽）
    const uint32_t cclBuffMultiplier = param.rankSize;
    const uint64_t cclBufferSize = resCtx.localBuffer.size;
    const uint64_t cclBuffBound =
        (cclBufferSize / cclBuffMultiplier / HCCL_MIN_SLICE_ALIGN) * HCCL_MIN_SLICE_ALIGN;
    const uint64_t maxDataSizePerLoop = std::min(UB_MAX_DATA_SIZE, cclBuffBound);
    if (maxDataSizePerLoop == 0) {
        HCCL_ERROR("[ExecOp] CCL buffer too small: cclBufferSize=%lu, rankSize=%u",
                   cclBufferSize, param.rankSize);
        return HCCL_E_INTERNAL;
    }
    const uint64_t maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
    const uint64_t loopCount =
        count / maxDataCountPerLoop + (count % maxDataCountPerLoop != 0 ? 1 : 0);

    void *const cclBuffAddr = resCtx.localBuffer.addr;
    uint64_t processedDataCount = 0;

    // chunks 之间相互独立：每一轮独立的 local copy -> exchange -> local copy
    for (uint64_t loop = 0; loop < loopCount; loop++) {
        const uint64_t sliceCount = std::min(maxDataCountPerLoop, count - loop * maxDataCountPerLoop);
        const uint64_t sliceSize = sliceCount * dataTypeSize;
        const uint64_t inputOffset = processedDataCount * dataTypeSize;

        // ============================================================
        // 微优化：own rank 的数据不再经过 ccl 往返
        //   原: input -> ccl[myRank*sliceSize] -> [channel write src] -> ccl[myRank*sliceSize] -> recvBuf[myRank*dataSize]
        //   改: input -> recvBuf[myRank*dataSize]（一次性到位，channel write 直接读 input）
        //   省下每 chunk 1 次本地 DMA，notify 结构不变。
        // ============================================================
        void *curInputAddr =
            static_cast<void *>(static_cast<uint8_t *>(param.inputPtr) + inputOffset);
        void *curMyRecvBufAddr =
            static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + param.myRank * dataSize + inputOffset);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], curMyRecvBufAddr, curInputAddr, sliceSize)));

        // 线程同步：从thread启动后，开始执行 channel 通信
        CHK_RET(ThreadSyncBefore(resCtx));

        // 全rank写：每条channel把本端数据写入对端cclbuff中本端对应的槽位
        // channel write src 直接读 input（绕过 ccl 槽位），dst 写对端 ccl[myRank*sliceSize]
        for (uint32_t i = 0; i < resCtx.channels.size(); i++) {
            const ChannelHandle &remoteChannelHandle = resCtx.channels[i].handle;
            // 对端cclbuff中预留的位置 = sliceSize * myRank
            void *remoteCclBuffAddr = static_cast<void *>(
                static_cast<uint8_t *>(resCtx.channels[i].remoteCclMem.addr) + sliceSize * param.myRank);
            const ThreadHandle &curThread = resCtx.threads[i];

            // 通知对端数据准备完毕
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(curThread, remoteChannelHandle, NOTIFY_IDX_ACK)));
            // 等待对端确认其数据也已就绪
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(curThread, remoteChannelHandle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            // 把本地 sliceSize 数据直接写入对端 cclbuff（src 用 input，绕过 own ccl 槽）
            CHK_RET(static_cast<HcclResult>(
                HcommWriteOnThread(curThread, remoteChannelHandle, remoteCclBuffAddr, curInputAddr, sliceSize)));
            // 告诉对端数据写入完成，并等待对端写入完成（双向同步）
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(curThread, remoteChannelHandle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(curThread, remoteChannelHandle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }

        // 等待所有 channel 通信完成
        CHK_RET(ThreadSyncAfter(resCtx));

        // 将 cclbuff 中其他 rank 的数据段拷出到 output（own rank 已在 recvBuf 中，跳过）
        const uint64_t baseOutputOffset = processedDataCount * dataTypeSize;
        for (uint32_t rankId = 0; rankId < param.rankSize; rankId++) {
            if (rankId == param.myRank) {
                continue; // own rank 数据已经在 recvBuf[myRank]，跳过 ccl → recvBuf 这次本地拷
            }
            const uint64_t cclBuffOffset = sliceSize * rankId;
            const uint64_t outputOffset = baseOutputOffset + rankId * dataSize;
            void *curSrcCclBuffAddr =
                static_cast<void *>(static_cast<uint8_t *>(cclBuffAddr) + cclBuffOffset);
            void *curOutputAddr =
                static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + outputOffset);
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(resCtx.threads[0], curOutputAddr, curSrcCclBuffAddr, sliceSize)));
        }

        processedDataCount += sliceCount;
    }

    HCCL_INFO("[ExecOp] AllGather done. rank=%u/%u, count=%lu", param.myRank, param.rankSize, param.count);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl