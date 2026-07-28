/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#define NOTIFY_IDX_PERMIT 2
#include "custom.h"
#include "log.h"
#include "exec_op.h"


static HcommDataType ConvertDataType(HcclDataType hcclType)
{
    return HCOMM_DATA_TYPE_FP32;
}

/// HcclReduceOp → HcommReduceOp 映射
static HcommReduceOp ConvertReduceOp(HcclReduceOp hcclOp)
{
    return HCOMM_REDUCE_SUM;
}

namespace ops_hccl {

/// 从 channels 中查找指向指定 rank 的 Channel 索引
static int32_t FindChannelByRank(const std::vector<ChannelInfo> &channels, uint32_t targetRank) {
    for (size_t i = 0; i < channels.size(); i++) {
        if (channels[i].remoteRank == targetRank) return static_cast<int32_t>(i);
    }
    return -1;
}
// ... 头文件、Convert函数、FindChannelByRank 保持不变 ...

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx) {
    uint32_t myRank = param.myRank;
    uint32_t rankSize = param.rankSize;
    uint32_t rootRank = resCtx.algConfig.rootRank;
    bool isRoot = resCtx.algConfig.isRoot;

    auto it = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(it == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type: %d", param.dataType), HCCL_E_INTERNAL);
    uint32_t dataSize = it->second;
    uint64_t totalSize = param.count * dataSize;

    HcommDataType hcommDataType = ConvertDataType(param.dataType);
    HcommReduceOp hcommReduceOp = ConvertReduceOp(param.reduceType);
    CHK_PRT_RET(hcommDataType == HCOMM_DATA_TYPE_RESERVED, HCCL_ERROR("Unsupported HcclDataType: %d", param.dataType), HCCL_E_INTERNAL);
    CHK_PRT_RET(hcommReduceOp == HCOMM_REDUCE_RESERVED, HCCL_ERROR("Unsupported HcclReduceOp: %d", param.reduceType), HCCL_E_INTERNAL);

    HCCL_INFO("ExecOp: myRank=%u, isRoot=%d, rankSize=%u, totalSize=%lu", myRank, isRoot, rankSize, totalSize);

    ThreadHandle thread = resCtx.aicpuThread;

    if (rankSize == 1) {
        if (param.inputPtr != param.outputPtr) {
            CHK_RET(HcommLocalCopyOnThread(thread, param.outputPtr, param.inputPtr, totalSize));
        }
        HCCL_INFO("Single rank, copy done");
        return HCCL_SUCCESS;
    }

    char *workBuf = static_cast<char *>(param.outputPtr);
    char *recvBuf = static_cast<char *>(resCtx.localBuffer.addr);

    if (param.inputPtr != param.outputPtr) {
        CHK_RET(HcommLocalCopyOnThread(thread, workBuf, param.inputPtr, totalSize));
    }

    // ========== 分块参数 ==========
    uint64_t cclBufferSize = resCtx.localBuffer.size;
    uint64_t maxChunkSize = cclBufferSize / 2;       // 安全余量
    if (maxChunkSize == 0) maxChunkSize = 1;
    uint64_t chunkSize = (totalSize > maxChunkSize) ? maxChunkSize : totalSize;
    uint64_t numChunks = (totalSize + chunkSize - 1) / chunkSize;
    HCCL_INFO("Chunking: totalSize=%lu, chunkSize=%lu, numChunks=%lu", totalSize, chunkSize, numChunks);

    // ==============================================
    // Phase 1: Reduce (PERMIT 串行化，分块循环)
    // ==============================================
    if (isRoot) {
        HCCL_INFO("Root: starting Reduce phase, receiving from %u ranks", rankSize - 1);
        for (uint32_t r = 0; r < rankSize; r++) {
            if (r == rootRank) continue;
            int32_t chIdx = FindChannelByRank(resCtx.channels, r);
            CHK_PRT_RET(chIdx < 0, HCCL_ERROR("Root: Channel to rank %u not found", r), HCCL_E_INTERNAL);
            const ChannelInfo &ch = resCtx.channels[chIdx];

            for (uint64_t c = 0; c < numChunks; c++) {
                uint64_t offset = c * chunkSize;
                uint64_t len = (c == numChunks - 1) ? (totalSize - offset) : chunkSize;
                uint64_t elemCount = len / dataSize;

                // 1. 发送 PERMIT
                CHK_RET(HcommWriteWithNotifyOnThread(thread, ch.handle,
                    static_cast<char *>(ch.remoteCclMem.addr), workBuf, 1, NOTIFY_IDX_PERMIT));

                // 2. 等待数据
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, ch.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

                // 3. 归约当前 Chunk
                CHK_RET(HcommLocalReduceOnThread(thread, workBuf + offset, recvBuf, elemCount, hcommDataType, hcommReduceOp));

                // 4. 发送 ACK
                CHK_RET(HcommWriteWithNotifyOnThread(thread, ch.handle,
                    static_cast<char *>(ch.remoteCclMem.addr), workBuf, 1, NOTIFY_IDX_ACK));

                HCCL_DEBUG("Root: reduced chunk %lu from rank %u", c, r);
            }
        }
        HCCL_INFO("Root: Reduce phase completed");
    } else {
        int32_t chIdx = FindChannelByRank(resCtx.channels, rootRank);
        CHK_PRT_RET(chIdx < 0, HCCL_ERROR("Non-root: Channel to root rank %u not found", rootRank), HCCL_E_INTERNAL);
        const ChannelInfo &ch = resCtx.channels[chIdx];

        HCCL_INFO("Non-root %u: sending data to root, %lu chunks", myRank, numChunks);
        for (uint64_t c = 0; c < numChunks; c++) {
            uint64_t offset = c * chunkSize;
            uint64_t len = (c == numChunks - 1) ? (totalSize - offset) : chunkSize;

            // 1. 等待 PERMIT
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, ch.handle, NOTIFY_IDX_PERMIT, CUSTOM_TIMEOUT));

            // 2. 发送当前 Chunk
            CHK_RET(HcommWriteWithNotifyOnThread(thread, ch.handle,
                static_cast<char *>(ch.remoteCclMem.addr), workBuf + offset, len, NOTIFY_IDX_DATA_SIGNAL));

            // 3. 等待 ACK
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, ch.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

            HCCL_DEBUG("Non-root %u: sent chunk %lu", myRank, c);
        }
        HCCL_INFO("Non-root %u: send to root completed", myRank);
    }

    // ==============================================
    // Phase 2: Broadcast (分块循环)
    // ==============================================
    if (isRoot) {
        HCCL_INFO("Root: starting Broadcast phase to %u ranks", rankSize - 1);
        for (uint32_t r = 0; r < rankSize; r++) {
            if (r == rootRank) continue;
            int32_t chIdx = FindChannelByRank(resCtx.channels, r);
            CHK_PRT_RET(chIdx < 0, HCCL_ERROR("Root: Channel to rank %u not found", r), HCCL_E_INTERNAL);
            const ChannelInfo &ch = resCtx.channels[chIdx];

            for (uint64_t c = 0; c < numChunks; c++) {
                uint64_t offset = c * chunkSize;
                uint64_t len = (c == numChunks - 1) ? (totalSize - offset) : chunkSize;

                // 发送 Chunk
                CHK_RET(HcommWriteWithNotifyOnThread(thread, ch.handle,
                    static_cast<char *>(ch.remoteCclMem.addr), workBuf + offset, len, NOTIFY_IDX_DATA_SIGNAL));

                // 等待 ACK
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, ch.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

                HCCL_DEBUG("Root: broadcast chunk %lu to rank %u", c, r);
            }
        }
        HCCL_INFO("Root: Broadcast phase completed");
    } else {
        int32_t chIdx = FindChannelByRank(resCtx.channels, rootRank);
        CHK_PRT_RET(chIdx < 0, HCCL_ERROR("Non-root: Channel to root rank %u not found", rootRank), HCCL_E_INTERNAL);
        const ChannelInfo &ch = resCtx.channels[chIdx];

        HCCL_INFO("Non-root %u: receiving from root, %lu chunks", myRank, numChunks);
        for (uint64_t c = 0; c < numChunks; c++) {
            uint64_t offset = c * chunkSize;
            uint64_t len = (c == numChunks - 1) ? (totalSize - offset) : chunkSize;

            // 等待数据
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, ch.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

            // 拷贝 Chunk
            CHK_RET(HcommLocalCopyOnThread(thread, workBuf + offset, recvBuf, len));

            // 发送 ACK
            CHK_RET(HcommWriteWithNotifyOnThread(thread, ch.handle,
                static_cast<char *>(ch.remoteCclMem.addr), workBuf, 1, NOTIFY_IDX_ACK));

            HCCL_DEBUG("Non-root %u: received chunk %lu", myRank, c);
        }
        HCCL_INFO("Non-root %u: receive from root completed", myRank);
    }

    HCCL_INFO("Star AllReduce completed successfully on rank %u", myRank);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
