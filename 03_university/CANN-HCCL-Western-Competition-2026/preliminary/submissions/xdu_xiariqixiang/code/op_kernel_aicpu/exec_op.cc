/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {

namespace {

// 根据 remoteRank 查找对应的 Channel 信息
const ChannelInfo *FindChannelByRank(const std::vector<ChannelInfo> &channels, uint32_t remoteRank)
{
    for (const auto &ch : channels) {
        if (ch.remoteRank == remoteRank) {
            return &ch;
        }
    }
    return nullptr;
}

// 根据 remoteRank 查找对应的 LinkType
LinkType FindLinkType(const std::vector<LinkInfo> &linkInfos, uint32_t remoteRank)
{
    for (const auto &link : linkInfos) {
        if (link.remoteRank == remoteRank) {
            return link.linkType;
        }
    }
    return LinkType::INTRA_SERVER_MESH; // 默认当作 Intra
}

// 本地归约操作：dst = dst op src（逐元素）
// 目前仅支持 FP32 + SUM，可扩展支持更多类型和操作
HcclResult LocalReduce(void *dst, const void *src, uint64_t count, HcclDataType dataType, HcclReduceOp op)
{
    if (dataType == HCCL_DATA_TYPE_FP32 && op == HCCL_REDUCE_SUM) {
        float *dstPtr = static_cast<float *>(dst);
        const float *srcPtr = static_cast<const float *>(src);
        for (uint64_t i = 0; i < count; i++) {
            dstPtr[i] += srcPtr[i];
        }
        return HCCL_SUCCESS;
    }

    // 其他数据类型 / 归约操作的扩展点
    HCCL_ERROR("Unsupported dataType[%d] or reduceOp[%d]", static_cast<int>(dataType), static_cast<int>(op));
    return HCCL_E_INVALID_ARG;
}

// 获取单个元素字节数
uint32_t GetElementSize(HcclDataType dataType)
{
    auto it = SIZE_TABLE.find(dataType);
    if (it != SIZE_TABLE.end()) {
        return it->second;
    }
    return 0; // 不支持的类型
}

// 初始化 Buffer 分区：将 HCCL Buffer 划分为暂存区
void InitBufferPartition(BufferPartition &bp, const AlgResourceCtx &resCtx,
    uint64_t chunkCount, uint64_t chunkBytes)
{
    char *base = static_cast<char *>(resCtx.localBuffer.addr);
    bp.chunkCount = chunkCount;
    bp.chunkBytes = chunkBytes;
    bp.recvScratch = base;                     // [0, chunkBytes)
    bp.crossScratch = base + chunkBytes;       // [chunkBytes, 2*chunkBytes)
}

} // anonymous namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const RankGroupInfo &rankInfo = resCtx.rankGroupInfo;
    uint32_t myRank = rankInfo.globalRankId;
    uint32_t serverId = rankInfo.serverId;
    uint32_t localRankId = rankInfo.localRankId;
    uint32_t numPerServer = rankInfo.numPerServer; // = 8

    uint64_t count = param.count;
    HcclDataType dataType = param.dataType;
    HcclReduceOp reduceOp = param.reduceType;

    uint32_t elementSize = GetElementSize(dataType);
    if (elementSize == 0) {
        HCCL_ERROR("Unsupported data type: %d", static_cast<int>(dataType));
        return HCCL_E_INVALID_ARG;
    }

    // 将数据按 Server 内 NPU 数量均分为多个 chunk
    uint64_t chunkCount = count / numPerServer;          // 每个 chunk 的元素数
    uint64_t chunkBytes = chunkCount * elementSize;      // 每个 chunk 的字节数
    uint64_t totalBytes = count * elementSize;
    uint64_t remainderCount = count % numPerServer;      // 余数元素（由最后几个 rank 处理）

    HCCL_INFO("ExecOp: myRank=%u, serverId=%u, localRankId=%u, count=%lu, chunkCount=%lu, chunkBytes=%lu",
        myRank, serverId, localRankId, count, chunkCount, chunkBytes);

    // 初始化 Buffer 分区
    BufferPartition bp;
    InitBufferPartition(bp, resCtx, chunkCount, chunkBytes);

    // 校验 HCCL Buffer 大小
    if (resCtx.localBuffer.size < 2 * chunkBytes) {
        HCCL_ERROR("HCCL buffer too small: need %lu bytes, have %lu bytes",
            2 * chunkBytes, resCtx.localBuffer.size);
        return HCCL_E_INVALID_ARG;
    }

    // 获取用于通信的 Thread
    ThreadHandle commThread = resCtx.threads[0];

    // 输出 buffer 中本 rank 负责的 chunk 位置
    char *sendBuf = static_cast<char *>(param.inputPtr);
    char *recvBuf = static_cast<char *>(param.outputPtr);
    char *myChunkOut = recvBuf + localRankId * chunkBytes; // 本 rank 负责的输出 chunk

    // ==============================================
    // Phase 1: Intra-server ReduceScatter
    // ==============================================
    // 目标：将 Server 内所有 rank 的数据按 chunk 归约，每个 rank 负责 1 个 chunk
    // Rank i 负责 chunk i：
    //   1. 初始化 myChunkOut = 本 rank 的 chunk i 数据
    //   2. 向所有其他 rank j 发送本 rank 的 chunk j 数据
    //   3. 从所有其他 rank j 接收 chunk i 数据并累加到 myChunkOut
    // ==============================================
    {
        // 初始化：将本 rank 的 chunk i 拷贝到输出区作为归约初值
        // 处理余数：如果 localRankId < remainderCount，本 rank 的 chunk 多一个元素
        uint64_t myChunkOffset = localRankId * chunkBytes;
        uint64_t myChunkActualCount = chunkCount + (localRankId < remainderCount ? 1 : 0);
        uint64_t myChunkActualBytes = myChunkActualCount * elementSize;
        std::memcpy(myChunkOut, sendBuf + myChunkOffset, myChunkActualBytes);

        HCCL_INFO("Phase1 ReduceScatter: initializing chunk[%u] with local data (%lu elements)",
            localRankId, myChunkActualCount);

        // 遍历所有 Intra-server 对端
        for (const auto &ch : resCtx.channels) {
            LinkType linkType = FindLinkType(resCtx.linkInfos, ch.remoteRank);
            if (linkType != LinkType::INTRA_SERVER_MESH) {
                continue; // 跳过跨 Server 链路，Phase 2 处理
            }

            uint32_t peerRank = ch.remoteRank;
            uint32_t peerLocalRank = peerRank % numPerServer;

            // (a) 接收 peer 对 chunk i 的贡献
            uint64_t recvActualCount = chunkCount + (localRankId < remainderCount ? 1 : 0);
            CHK_RET(HcommRecv(bp.recvScratch, recvActualCount, dataType,
                peerRank, ch.handle, commThread));

            // (b) 发送本 rank 对 chunk j 的贡献
            uint64_t peerChunkOffset = peerLocalRank * chunkBytes;
            uint64_t sendActualCount = chunkCount + (peerLocalRank < remainderCount ? 1 : 0);
            CHK_RET(HcommSend(sendBuf + peerChunkOffset, sendActualCount, dataType,
                peerRank, ch.handle, commThread));

            // (c) 等待接收和发送完成
            CHK_RET(HcommThreadNotifyWaitOnThread(commThread, ch.notifyNum, CUSTOM_TIMEOUT));

            // (d) 本地归约：myChunkOut += recvScratch
            CHK_RET(LocalReduce(myChunkOut, bp.recvScratch, recvActualCount, dataType, reduceOp));
        }
    }

    HCCL_INFO("Phase1 ReduceScatter completed for rank %u", myRank);

    // ==============================================
    // Phase 2: Cross-server AllReduce
    // ==============================================
    // 目标：每个 rank 与对端 Server 中相同 localRank 的 rank 交换已归约的 chunk，再归约
    // Server 0 的 rank i 与 Server 1 的 rank i 交换数据
    // 经过此阶段后，每个 rank 拥有的 chunk 是全局归约后的结果
    // ==============================================
    {
        uint32_t crossPeerRank;
        if (serverId == 0) {
            crossPeerRank = myRank + numPerServer; // Server 1 中相同 localRank 的 rank
        } else {
            crossPeerRank = myRank - numPerServer; // Server 0 中相同 localRank 的 rank
        }

        const ChannelInfo *crossCh = FindChannelByRank(resCtx.channels, crossPeerRank);
        if (crossCh == nullptr) {
            HCCL_ERROR("Cross-server channel not found for peer rank %u", crossPeerRank);
            return HCCL_E_INTERNAL;
        }

        uint64_t myActualCount = chunkCount + (localRankId < remainderCount ? 1 : 0);

        // (a) 接收对端 Server 的已归约 chunk
        CHK_RET(HcommRecv(bp.crossScratch, myActualCount, dataType,
            crossPeerRank, crossCh->handle, commThread));

        // (b) 发送本 Server 的已归约 chunk
        CHK_RET(HcommSend(myChunkOut, myActualCount, dataType,
            crossPeerRank, crossCh->handle, commThread));

        // (c) 等待接收和发送完成
        CHK_RET(HcommThreadNotifyWaitOnThread(commThread, crossCh->notifyNum, CUSTOM_TIMEOUT));

        // (d) 本地归约：myChunkOut += crossScratch
        CHK_RET(LocalReduce(myChunkOut, bp.crossScratch, myActualCount, dataType, reduceOp));

        HCCL_INFO("Phase2 Cross-server: exchanged with rank %u, reduced %lu elements",
            crossPeerRank, myActualCount);
    }

    // ==============================================
    // Phase 3: Intra-server AllGather
    // ==============================================
    // 目标：将每个 rank 的全局归约后的 chunk 广播到 Server 内所有 rank
    // 经过此阶段后，Server 内所有 rank 都拥有完整的全局归约结果
    // ==============================================
    {
        for (const auto &ch : resCtx.channels) {
            LinkType linkType = FindLinkType(resCtx.linkInfos, ch.remoteRank);
            if (linkType != LinkType::INTRA_SERVER_MESH) {
                continue; // 跳过跨 Server 链路
            }

            uint32_t peerRank = ch.remoteRank;
            uint32_t peerLocalRank = peerRank % numPerServer;

            // 接收 peer 的 chunk（放入 recvBuf 的对应位置）
            void *peerChunkOut = recvBuf + peerLocalRank * chunkBytes;
            uint64_t peerActualCount = chunkCount + (peerLocalRank < remainderCount ? 1 : 0);
            CHK_RET(HcommRecv(peerChunkOut, peerActualCount, dataType,
                peerRank, ch.handle, commThread));

            // 发送本 rank 的 chunk 给 peer
            uint64_t myActualCount = chunkCount + (localRankId < remainderCount ? 1 : 0);
            CHK_RET(HcommSend(myChunkOut, myActualCount, dataType,
                peerRank, ch.handle, commThread));

            // 等待接收和发送完成
            CHK_RET(HcommThreadNotifyWaitOnThread(commThread, ch.notifyNum, CUSTOM_TIMEOUT));
        }

        HCCL_INFO("Phase3 AllGather completed for rank %u", myRank);
    }

    // ==============================================
    // 处理余数数据（当 count 不能被 numPerServer 整除时）
    // 余数部分由最后几个 rank 的 chunk 多承载一个元素
    // 已在上述各阶段通过 actualCount 处理完毕
    // 但还需确保 recvBuf 中未被 chunk 覆盖的尾部被正确填充
    // ==============================================
    if (remainderCount > 0) {
        // 余数已由各 rank 在 AllGather 阶段接收，但最后一个完整 chunk 之后
        // 如果有不满一个完整 chunk 的尾部，需要从 sendBuf 复制
        uint64_t coveredBytes = numPerServer * chunkBytes +
            remainderCount * elementSize;
        if (coveredBytes < totalBytes) {
            // 理论上不会进入这里，因为余数已完全分配
            std::memcpy(recvBuf + coveredBytes, sendBuf + coveredBytes,
                totalBytes - coveredBytes);
        }
    }

    HCCL_INFO("ExecOp: AllReduce completed successfully for rank %u", myRank);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
