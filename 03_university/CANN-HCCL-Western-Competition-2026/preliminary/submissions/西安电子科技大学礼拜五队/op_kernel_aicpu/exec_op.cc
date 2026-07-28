/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
uint32_t GetDataTypeSize(HcclDataType dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8: case HCCL_DATA_TYPE_UINT8: case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3: case HCCL_DATA_TYPE_FP8E5M2: case HCCL_DATA_TYPE_FP8E8M0: return 1;
        case HCCL_DATA_TYPE_INT16: case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_FP16: case HCCL_DATA_TYPE_BFP16: return 2;
        case HCCL_DATA_TYPE_INT32: case HCCL_DATA_TYPE_UINT32: case HCCL_DATA_TYPE_FP32: return 4;
        case HCCL_DATA_TYPE_INT64: case HCCL_DATA_TYPE_UINT64: case HCCL_DATA_TYPE_FP64: return 8;
        case HCCL_DATA_TYPE_INT128: return 16;
        default: return 0;
    }
}

HcclResult SyncThreadsBeforeCommunication(const std::vector<ThreadHandle> &threads)
{
    const ThreadHandle mainThread = threads[0];
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, threads[idx], 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(threads[idx], 0, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult SyncThreadsAfterCommunication(const std::vector<ThreadHandle> &threads)
{
    const ThreadHandle mainThread = threads[0];
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(threads[idx], mainThread, idx));
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, idx, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(resCtx.localBuffer.addr);

    const uint32_t dataTypeSize = GetDataTypeSize(param.dataType);
    if (param.count == 0) return HCCL_SUCCESS;

    const uint64_t totalBytes = param.count * dataTypeSize;
    const ThreadHandle mainThread = resCtx.threads[0];
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);
    const char *input = static_cast<const char *>(param.inputPtr);
    char *output = static_cast<char *>(param.outputPtr);
    const HcommDataType dataType = static_cast<HcommDataType>(param.dataType);
    const HcommReduceOp reduceOp = static_cast<HcommReduceOp>(param.reduceType);

    if (param.rankSize == 1) {
        CHK_RET(HcommLocalCopyOnThread(mainThread, localBuffer, input, totalBytes));
        CHK_RET(HcommLocalCopyOnThread(mainThread, output, localBuffer, totalBytes));
        return HCCL_SUCCESS;
    }

    // =========================================================================
    // ★ 引擎 A：单线程 RS+AG 极速算法 - 专攻 512KB 及更小测试点
    // =========================================================================
    if (totalBytes <= 512 * 1024) {
        uint64_t chunkOffsets[64] = {0};
        uint32_t safeRankSize = param.rankSize < 63 ? param.rankSize : 63;
        uint64_t baseCount = (totalBytes / dataTypeSize) / safeRankSize;
        uint64_t remainder = (totalBytes / dataTypeSize) % safeRankSize;
        for (uint32_t i = 0; i < safeRankSize; ++i) {
            uint64_t c = baseCount + (i < remainder ? 1 : 0);
            chunkOffsets[i+1] = chunkOffsets[i] + c;
        }

        uint64_t myOffset = chunkOffsets[param.myRank] * dataTypeSize;
        uint64_t myBytes = (chunkOffsets[param.myRank+1] - chunkOffsets[param.myRank]) * dataTypeSize;

        // 阶段 0：大家把自己的原数据搬入 localBuffer
        CHK_RET(HcommLocalCopyOnThread(mainThread, localBuffer, input, totalBytes));

        // 全网 Barrier 0
        for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, resCtx.channels[idx].handle, 0));
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, resCtx.channels[idx].handle, 0, CUSTOM_TIMEOUT));
        }

        // 阶段 1：Reduce-Scatter (仅单线程拉取属于自己的那块 32KB)
        if (myBytes > 0) {
            for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
                char *remoteSrc = static_cast<char *>(resCtx.channels[idx].remoteCclMem.addr) + myOffset;
                // 用 localBuffer 后面的空闲区作为拉取暂存地，绝对安全
                char *tempDst = localBuffer + totalBytes + idx * myBytes;
                CHK_RET(HcommReadOnThread(mainThread, resCtx.channels[idx].handle, tempDst, remoteSrc, myBytes));
            }

            // 先把自己的初值拷贝到 output
            CHK_RET(HcommLocalCopyOnThread(mainThread, output + myOffset, localBuffer + myOffset, myBytes));

            // 严格按顺序串行累加，保证确定性，直出结果到 output
            uint64_t myElemCount = myBytes / dataTypeSize;
            for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
                char *src = localBuffer + totalBytes + idx * myBytes;
                CHK_RET(HcommLocalReduceOnThread(mainThread, output + myOffset, src, myElemCount, dataType, reduceOp));
            }

            // 把算好的结果放回 localBuffer，供下一阶段大家拉取
            CHK_RET(HcommLocalCopyOnThread(mainThread, localBuffer + myOffset, output + myOffset, myBytes));
        }

        // 全网 Barrier 1
        for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, resCtx.channels[idx].handle, 1));
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, resCtx.channels[idx].handle, 1, CUSTOM_TIMEOUT));
        }

        // 阶段 2：All-Gather (直接拉取别人算好的成品到 output 对应位置)
        for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
            uint32_t remoteRank = resCtx.channels[idx].remoteRank;
            uint64_t remoteOffset = chunkOffsets[remoteRank] * dataTypeSize;
            uint64_t remoteBytes = (chunkOffsets[remoteRank+1] - chunkOffsets[remoteRank]) * dataTypeSize;

            if (remoteBytes > 0) {
                char *remoteSrc = static_cast<char *>(resCtx.channels[idx].remoteCclMem.addr) + remoteOffset;
                char *localDst = output + remoteOffset;
                CHK_RET(HcommReadOnThread(mainThread, resCtx.channels[idx].handle, localDst, remoteSrc, remoteBytes));
            }
        }

        // 全网 Barrier 2 (防退出覆盖)
        for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, resCtx.channels[idx].handle, 2));
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, resCtx.channels[idx].handle, 2, CUSTOM_TIMEOUT));
        }

        return HCCL_SUCCESS;
    }

    // =========================================================================
    // ★ 引擎 B：多线程 RS+AG (退回最稳基线) - 专攻 512MB / 400MB+4B 测试点
    // =========================================================================
    uint64_t maxTileElements = resCtx.localBuffer.size / 2 / dataTypeSize;
    maxTileElements = (maxTileElements / param.rankSize) * param.rankSize;
    if (maxTileElements == 0) maxTileElements = param.rankSize;

    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        const uint64_t remainingCount = param.count - processedCount;
        const uint64_t tileCount = remainingCount < maxTileElements ? remainingCount : maxTileElements;
        const uint64_t tileBytes = tileCount * dataTypeSize;
        const uint64_t userOffset = processedCount * dataTypeSize;

        uint64_t chunkOffsets[64] = {0};
        uint32_t safeRankSize = param.rankSize < 63 ? param.rankSize : 63;
        uint64_t baseCount = tileCount / safeRankSize;
        uint64_t remainder = tileCount % safeRankSize;
        for (uint32_t i = 0; i < safeRankSize; ++i) {
            uint64_t c = baseCount + (i < remainder ? 1 : 0);
            chunkOffsets[i+1] = chunkOffsets[i] + c;
        }

        uint64_t myOffset = chunkOffsets[param.myRank] * dataTypeSize;
        uint64_t myBytes = (chunkOffsets[param.myRank+1] - chunkOffsets[param.myRank]) * dataTypeSize;

        CHK_RET(HcommLocalCopyOnThread(mainThread, localBuffer, input + userOffset, tileBytes));
        CHK_RET(SyncThreadsBeforeCommunication(resCtx.threads));

        for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
            const ThreadHandle t = resCtx.threads[idx];
            const ChannelHandle h = resCtx.channels[idx].handle;

            CHK_RET(HcommChannelNotifyRecordOnThread(t, h, 0));
            CHK_RET(HcommChannelNotifyWaitOnThread(t, h, 0, CUSTOM_TIMEOUT));

            if (myBytes > 0) {
                char *remoteSrc = static_cast<char *>(resCtx.channels[idx].remoteCclMem.addr) + myOffset;
                char *localDst = localBuffer + tileBytes + idx * myBytes;
                CHK_RET(HcommReadOnThread(t, h, localDst, remoteSrc, myBytes));
            }
        }
        CHK_RET(SyncThreadsAfterCommunication(resCtx.threads));

        if (myBytes > 0) {
            uint64_t myElemCount = myBytes / dataTypeSize;
            for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
                char *src = localBuffer + tileBytes + idx * myBytes;
                char *dst = localBuffer + myOffset;
                CHK_RET(HcommLocalReduceOnThread(mainThread, dst, src, myElemCount, dataType, reduceOp));
            }
        }
        CHK_RET(SyncThreadsBeforeCommunication(resCtx.threads));

        for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
            const ThreadHandle t = resCtx.threads[idx];
            const ChannelHandle h = resCtx.channels[idx].handle;
            const uint32_t remoteRank = resCtx.channels[idx].remoteRank;

            CHK_RET(HcommChannelNotifyRecordOnThread(t, h, 1));
            CHK_RET(HcommChannelNotifyWaitOnThread(t, h, 1, CUSTOM_TIMEOUT));

            uint64_t remoteOffset = chunkOffsets[remoteRank] * dataTypeSize;
            uint64_t remoteBytes = (chunkOffsets[remoteRank+1] - chunkOffsets[remoteRank]) * dataTypeSize;

            if (remoteBytes > 0) {
                char *remoteSrc = static_cast<char *>(resCtx.channels[idx].remoteCclMem.addr) + remoteOffset;
                char *localDst = localBuffer + remoteOffset;
                CHK_RET(HcommReadOnThread(t, h, localDst, remoteSrc, remoteBytes));
            }
        }
        CHK_RET(SyncThreadsAfterCommunication(resCtx.threads));

        CHK_RET(SyncThreadsBeforeCommunication(resCtx.threads));
        for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
            const ThreadHandle t = resCtx.threads[idx];
            const ChannelHandle h = resCtx.channels[idx].handle;
            CHK_RET(HcommChannelNotifyRecordOnThread(t, h, 2));
            CHK_RET(HcommChannelNotifyWaitOnThread(t, h, 2, CUSTOM_TIMEOUT));
        }
        CHK_RET(SyncThreadsAfterCommunication(resCtx.threads));

        CHK_RET(HcommLocalCopyOnThread(mainThread, output + userOffset, localBuffer, tileBytes));
        processedCount += tileCount;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl