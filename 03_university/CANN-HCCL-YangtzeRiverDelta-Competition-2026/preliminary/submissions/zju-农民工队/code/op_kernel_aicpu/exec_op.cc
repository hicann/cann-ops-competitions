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
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    // TODO: 算法任务编排
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.rankSize != 16 || param.root >= param.rankSize,
        HCCL_ERROR("ExecOp: invalid rankSize[%u] or root[%u]", param.rankSize, param.root), HCCL_E_PARA);
    auto typeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(),
        HCCL_ERROR("ExecOp: unsupported dataType[%d]", static_cast<int>(param.dataType)), HCCL_E_PARA);

    constexpr uint32_t serverSize = 8;
    constexpr uint64_t smallDataThreshold = 1ULL * 1024 * 1024;
    uint64_t dataTypeSize = typeIt->second;
    CHK_PRT_RET(param.count > ~static_cast<uint64_t>(0) / dataTypeSize,
        HCCL_ERROR("ExecOp: data size overflows uint64_t"), HCCL_E_PARA);
    uint64_t totalBytes = param.count * dataTypeSize;
    uint32_t myServer = param.myRank / serverSize;
    uint32_t localRank = param.myRank % serverSize;
    ThreadHandle mainThread = resCtx.aicpuThread;
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);

    std::unordered_map<uint32_t, ChannelInfo> channelMap;
    for (const auto &channel : resCtx.channels) {
        channelMap.emplace(channel.remoteRank, channel);
    }

    // 小消息：root准备一次数据，其余15个rank直接Read到UserOut。
    if (totalBytes <= smallDataThreshold) {
        CHK_PRT_RET(totalBytes > resCtx.localBuffer.size,
            HCCL_ERROR("ExecOp: local CCL buffer is too small for small-message path"), HCCL_E_INTERNAL);
        if (param.myRank == param.root) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(mainThread, resCtx.localBuffer.addr, output, totalBytes)));
            for (uint32_t rank = 0; rank < param.rankSize; rank++) {
                if (rank != param.root) {
                    auto peer = channelMap.find(rank);
                    CHK_PRT_RET(peer == channelMap.end(),
                        HCCL_ERROR("ExecOp: channel to rank[%u] not found", rank), HCCL_E_INTERNAL);
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                        mainThread, peer->second.handle, NOTIFY_IDX_ACK)));
                }
            }
            for (uint32_t rank = 0; rank < param.rankSize; rank++) {
                if (rank != param.root) {
                    const ChannelInfo &peer = channelMap.find(rank)->second;
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                        mainThread, peer.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                }
            }
        } else {
            auto rootChannel = channelMap.find(param.root);
            CHK_PRT_RET(rootChannel == channelMap.end(),
                HCCL_ERROR("ExecOp: channel to root[%u] not found", param.root), HCCL_E_INTERNAL);
            CHK_PRT_RET(totalBytes > rootChannel->second.remoteCclMem.size,
                HCCL_ERROR("ExecOp: remote root CCL buffer is too small"), HCCL_E_INTERNAL);
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                mainThread, rootChannel->second.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(mainThread, rootChannel->second.handle,
                output, rootChannel->second.remoteCclMem.addr, totalBytes)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                mainThread, rootChannel->second.handle, NOTIFY_IDX_DATA_SIGNAL)));
        }
        return HCCL_SUCCESS;
    }

    // 大消息：188 MiB完整wave、双缓冲、三wave六阶段流水。
    constexpr uint32_t LOGICAL_SHARD_NUM = 16;
    constexpr uint32_t HALF_SHARD_NUM = 8;
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t MAX_WAVE_NUM = 3;
    constexpr uint32_t THREAD_NUM = 6;
    constexpr uint64_t WAVE_BYTES = 188ULL * 1024 * 1024;
    constexpr uint32_t NOTIFY_SCATTER_READY = 0;
    constexpr uint32_t NOTIFY_SCATTER_DONE = 1;
    constexpr uint32_t NOTIFY_DOUBLING_READY = 2;
    constexpr uint32_t NOTIFY_DOUBLING_DONE = 3;
    constexpr uint32_t NOTIFY_WAVE_DONE_BASE = MAX_WAVE_NUM;
    CHK_PRT_RET(resCtx.threads.size() < THREAD_NUM,
        HCCL_ERROR("ExecOp: 6 AICPU threads are required, actual[%zu]", resCtx.threads.size()), HCCL_E_INTERNAL);
    CHK_PRT_RET(BUFFER_NUM * WAVE_BYTES > resCtx.localBuffer.size,
        HCCL_ERROR("ExecOp: local CCL buffer cannot hold two 188 MiB waves"), HCCL_E_INTERNAL);
    for (const auto &channel : resCtx.channels) {
        CHK_PRT_RET(BUFFER_NUM * WAVE_BYTES > channel.remoteCclMem.size,
            HCCL_ERROR("ExecOp: remote CCL buffer of rank[%u] is too small", channel.remoteRank), HCCL_E_INTERNAL);
    }

    uint64_t maxWaveCount = WAVE_BYTES / dataTypeSize;
    uint32_t waveCount = static_cast<uint32_t>((param.count - 1) / maxWaveCount + 1);
    CHK_PRT_RET(waveCount > MAX_WAVE_NUM,
        HCCL_ERROR("ExecOp: waveCount[%u] exceeds supported maximum", waveCount), HCCL_E_PARA);
    uint32_t myLogicalShard = 2 * localRank + myServer;
    uint32_t halfStartShard = localRank / (serverSize / 2) * HALF_SHARD_NUM;
    uint32_t remoteHalfStartShard = halfStartShard ^ HALF_SHARD_NUM;
    uint32_t remoteServerBegin = myServer == 0 ? serverSize : 0;
    uint32_t stagePartnerLocal[4] = {
        localRank,
        localRank ^ 1U,
        localRank ^ 2U,
        localRank ^ 4U
    };
    const ChannelInfo *stageChannels[4];
    for (uint32_t stage = 0; stage < 4; stage++) {
        uint32_t partnerRank = remoteServerBegin + stagePartnerLocal[stage];
        auto partner = channelMap.find(partnerRank);
        CHK_PRT_RET(partner == channelMap.end(),
            HCCL_ERROR("ExecOp: stage[%u] channel to rank[%u] not found", stage, partnerRank), HCCL_E_INTERNAL);
        stageChannels[stage] = &partner->second;
    }

    ThreadHandle foldThread = resCtx.threads[1];
    ThreadHandle double1Thread = resCtx.threads[2];
    ThreadHandle double2Thread = resCtx.threads[3];
    ThreadHandle finalThread = resCtx.threads[4];
    ThreadHandle copyThread = resCtx.threads[5];
    uint8_t *localCcl = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint64_t processedCount = 0;
    for (uint32_t wave = 0; wave < waveCount; wave++) {
        uint32_t parity = wave % BUFFER_NUM;
        if (wave >= BUFFER_NUM) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                mainThread, NOTIFY_WAVE_DONE_BASE + wave - BUFFER_NUM, CUSTOM_TIMEOUT)));
        }

        uint64_t currentCount = std::min(maxWaveCount, param.count - processedCount);
        uint64_t currentBytes = currentCount * dataTypeSize;
        uint64_t outputOffset = processedCount * dataTypeSize;
        uint64_t bufferOffset = static_cast<uint64_t>(parity) * WAVE_BYTES;
        uint8_t *waveBuffer = localCcl + bufferOffset;
        uint64_t baseCount = currentCount / LOGICAL_SHARD_NUM;
        uint64_t remainder = currentCount % LOGICAL_SHARD_NUM;
        uint64_t shardOffsets[LOGICAL_SHARD_NUM + 1];
        shardOffsets[0] = 0;
        for (uint32_t shard = 0; shard < LOGICAL_SHARD_NUM; shard++) {
            uint64_t shardCount = baseCount + (shard < remainder ? 1 : 0);
            shardOffsets[shard + 1] = shardOffsets[shard] + shardCount * dataTypeSize;
        }

        // Stage 0：root准备完整wave，Mesh和Clos上的15个rank同时Scatter Read。
        if (param.myRank == param.root) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                mainThread, waveBuffer, output + outputOffset, currentBytes)));
            for (uint32_t rank = 0; rank < param.rankSize; rank++) {
                if (rank != param.root) {
                    const ChannelInfo &peer = channelMap.find(rank)->second;
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                        mainThread, peer.handle, NOTIFY_SCATTER_READY)));
                }
            }
            for (uint32_t rank = 0; rank < param.rankSize; rank++) {
                if (rank != param.root) {
                    const ChannelInfo &peer = channelMap.find(rank)->second;
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                        mainThread, peer.handle, NOTIFY_SCATTER_DONE, CUSTOM_TIMEOUT)));
                }
            }
        } else {
            auto rootChannel = channelMap.find(param.root);
            CHK_PRT_RET(rootChannel == channelMap.end(),
                HCCL_ERROR("ExecOp: channel to root[%u] not found", param.root), HCCL_E_INTERNAL);
            uint64_t myOffset = shardOffsets[myLogicalShard];
            uint64_t myBytes = shardOffsets[myLogicalShard + 1] - myOffset;
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                mainThread, rootChannel->second.handle, NOTIFY_SCATTER_READY, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(mainThread, rootChannel->second.handle,
                waveBuffer + myOffset,
                static_cast<uint8_t *>(rootChannel->second.remoteCclMem.addr) + bufferOffset + myOffset,
                myBytes)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                mainThread, rootChannel->second.handle, NOTIFY_SCATTER_DONE)));
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(mainThread, foldThread, wave)));

        // Stage 1：1/16 Fold，得到连续1/8。
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(foldThread, wave, CUSTOM_TIMEOUT)));
        uint32_t foldShard = 2 * localRank + (1 - myServer);
        uint64_t foldOffset = shardOffsets[foldShard];
        uint64_t foldBytes = shardOffsets[foldShard + 1] - foldOffset;
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            foldThread, stageChannels[0]->handle, NOTIFY_DOUBLING_READY)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            foldThread, stageChannels[0]->handle, NOTIFY_DOUBLING_READY, CUSTOM_TIMEOUT)));
        if (param.myRank != param.root) {
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(foldThread, stageChannels[0]->handle,
                waveBuffer + foldOffset,
                static_cast<uint8_t *>(stageChannels[0]->remoteCclMem.addr) + bufferOffset + foldOffset,
                foldBytes)));
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            foldThread, stageChannels[0]->handle, NOTIFY_DOUBLING_DONE)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            foldThread, stageChannels[0]->handle, NOTIFY_DOUBLING_DONE, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(foldThread, double1Thread, wave)));

        // Stage 2：1/8 Doubling，得到连续1/4。
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(double1Thread, wave, CUSTOM_TIMEOUT)));
        uint32_t double1Start = stagePartnerLocal[1] * 2;
        uint64_t double1Offset = shardOffsets[double1Start];
        uint64_t double1Bytes = shardOffsets[double1Start + 2] - double1Offset;
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            double1Thread, stageChannels[1]->handle, NOTIFY_DOUBLING_READY)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            double1Thread, stageChannels[1]->handle, NOTIFY_DOUBLING_READY, CUSTOM_TIMEOUT)));
        if (param.myRank != param.root) {
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(double1Thread, stageChannels[1]->handle,
                waveBuffer + double1Offset,
                static_cast<uint8_t *>(stageChannels[1]->remoteCclMem.addr) + bufferOffset + double1Offset,
                double1Bytes)));
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            double1Thread, stageChannels[1]->handle, NOTIFY_DOUBLING_DONE)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            double1Thread, stageChannels[1]->handle, NOTIFY_DOUBLING_DONE, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(double1Thread, double2Thread, wave)));

        // Stage 3：1/4 Doubling，补齐本rank的own half。
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(double2Thread, wave, CUSTOM_TIMEOUT)));
        uint32_t double2Start = (stagePartnerLocal[2] & ~1U) * 2;
        uint64_t double2Offset = shardOffsets[double2Start];
        uint64_t double2Bytes = shardOffsets[double2Start + 4] - double2Offset;
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            double2Thread, stageChannels[2]->handle, NOTIFY_DOUBLING_READY)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            double2Thread, stageChannels[2]->handle, NOTIFY_DOUBLING_READY, CUSTOM_TIMEOUT)));
        if (param.myRank != param.root) {
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(double2Thread, stageChannels[2]->handle,
                waveBuffer + double2Offset,
                static_cast<uint8_t *>(stageChannels[2]->remoteCclMem.addr) + bufferOffset + double2Offset,
                double2Bytes)));
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            double2Thread, stageChannels[2]->handle, NOTIFY_DOUBLING_DONE)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            double2Thread, stageChannels[2]->handle, NOTIFY_DOUBLING_DONE, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(double2Thread, finalThread, wave)));
        if (param.myRank != param.root) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(double2Thread, copyThread, wave)));
        }

        uint64_t ownOffset = shardOffsets[halfStartShard];
        uint64_t ownBytes = shardOffsets[halfStartShard + HALF_SHARD_NUM] - ownOffset;
        uint64_t remoteOffset = shardOffsets[remoteHalfStartShard];
        uint64_t remoteBytes = shardOffsets[remoteHalfStartShard + HALF_SHARD_NUM] - remoteOffset;

        // Stage 4：own half连续LocalCopy到UserOut。
        if (param.myRank != param.root) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(copyThread, wave, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(copyThread,
                output + outputOffset + ownOffset, waveBuffer + ownOffset, ownBytes)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                copyThread, finalThread, NOTIFY_WAVE_DONE_BASE + wave)));
        }

        // Stage 5：remote half直接Read到UserOut，并在两路输出完成后释放Buffer。
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(finalThread, wave, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            finalThread, stageChannels[3]->handle, NOTIFY_DOUBLING_READY)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            finalThread, stageChannels[3]->handle, NOTIFY_DOUBLING_READY, CUSTOM_TIMEOUT)));
        if (param.myRank != param.root) {
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(finalThread, stageChannels[3]->handle,
                output + outputOffset + remoteOffset,
                static_cast<uint8_t *>(stageChannels[3]->remoteCclMem.addr) + bufferOffset + remoteOffset,
                remoteBytes)));
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            finalThread, stageChannels[3]->handle, NOTIFY_DOUBLING_DONE)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            finalThread, stageChannels[3]->handle, NOTIFY_DOUBLING_DONE, CUSTOM_TIMEOUT)));
        if (param.myRank != param.root) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                finalThread, NOTIFY_WAVE_DONE_BASE + wave, CUSTOM_TIMEOUT)));
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            finalThread, mainThread, NOTIFY_WAVE_DONE_BASE + wave)));
        processedCount += currentCount;
    }

    uint32_t firstPendingWave = waveCount > BUFFER_NUM ? waveCount - BUFFER_NUM : 0;
    for (uint32_t wave = firstPendingWave; wave < waveCount; wave++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            mainThread, NOTIFY_WAVE_DONE_BASE + wave, CUSTOM_TIMEOUT)));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
