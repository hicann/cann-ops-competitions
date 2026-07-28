/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {
constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint32_t TOTAL_RANK_NUM = 16;
constexpr uint32_t SERVER_RANK_NUM = 8;

constexpr uint32_t NOTIFY_IDX_SCATTER_DATA = 0;
constexpr uint32_t NOTIFY_IDX_SCATTER_ACK = 1;
constexpr uint32_t NOTIFY_IDX_SEGMENT_DATA = 2;
constexpr uint32_t NOTIFY_IDX_SEGMENT_ACK = 3;

constexpr uint64_t LARGE_MSG_THRESHOLD_BYTES = 64ULL * 1024ULL * 1024ULL;

// 如果整包放不进 HCCL buffer，fallback 到 384MiB。
// 400MiB+4B 会被切成 384MiB + 16MiB+4B，避免 4B 小尾块。
constexpr uint64_t FALLBACK_CHUNK_BYTES = 384ULL * 1024ULL * 1024ULL;

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

uint64_t GetChunkCapacity(const AlgResourceCtx &resCtx)
{
    uint64_t capacity = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        capacity = std::min(capacity, channel.remoteCclMem.size);
    }
    return capacity - capacity % FP32_BYTES;
}

uint64_t ChooseBlockCapacity(uint64_t totalBytes, uint64_t chunkCapacity)
{
    chunkCapacity -= chunkCapacity % FP32_BYTES;
    if (chunkCapacity == 0) {
        return 0;
    }

    if (totalBytes <= chunkCapacity) {
        return totalBytes;
    }

    uint64_t blockCapacity = std::min<uint64_t>(FALLBACK_CHUNK_BYTES, chunkCapacity);
    blockCapacity -= blockCapacity % FP32_BYTES;
    return blockCapacity;
}

uint32_t GetServerId(uint32_t rank)
{
    return rank / SERVER_RANK_NUM;
}

uint32_t GetLocalId(uint32_t rank)
{
    return rank % SERVER_RANK_NUM;
}

uint32_t MakeRank(uint32_t serverId, uint32_t localId)
{
    return serverId * SERVER_RANK_NUM + localId;
}

uint32_t GetOppositeServer(uint32_t serverId)
{
    return 1U - serverId;
}

void BuildSegmentOffsets8(uint64_t blockBytes, uint64_t *segmentOffsets)
{
    const uint64_t blockElems = blockBytes / FP32_BYTES;
    const uint64_t baseElems = blockElems / SERVER_RANK_NUM;
    const uint64_t remainElems = blockElems % SERVER_RANK_NUM;

    uint64_t curElems = 0;
    segmentOffsets[0] = 0;

    for (uint32_t localId = 0; localId < SERVER_RANK_NUM; ++localId) {
        const uint64_t segElems = baseElems + (localId < remainElems ? 1ULL : 0ULL);
        curElems += segElems;
        segmentOffsets[localId + 1] = curElems * FP32_BYTES;
    }
}

HcclResult DirectFanoutBroadcast(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t totalBytes, uint64_t chunkCapacity)
{
    const ThreadHandle thread = resCtx.threads[0];

    uint8_t *inputBuffer = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *outputBuffer = static_cast<uint8_t *>(param.outputPtr);

    const ChannelInfo *rootChannel = nullptr;
    if (param.myRank != param.root) {
        rootChannel = FindChannel(resCtx, param.root);
        if (rootChannel == nullptr || rootChannel->remoteCclMem.addr == nullptr) {
            HCCL_ERROR("Channel to root[%u] is missing", param.root);
            return HCCL_E_PTR;
        }
    }

    for (uint64_t offset = 0; offset < totalBytes; offset += chunkCapacity) {
        const uint64_t chunkBytes = std::min(chunkCapacity, totalBytes - offset);

        if (param.myRank == param.root) {
            void *inputChunk = static_cast<void *>(inputBuffer + offset);

            CHK_RET(HcommLocalCopyOnThread(thread, resCtx.localBuffer.addr, inputChunk, chunkBytes));

            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(HcommChannelNotifyRecordOnThread(
                    thread, channel.handle, NOTIFY_IDX_SCATTER_DATA));
            }

            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    thread, channel.handle, NOTIFY_IDX_SCATTER_ACK, CUSTOM_TIMEOUT));
            }
        } else {
            void *outputChunk = static_cast<void *>(outputBuffer + offset);

            CHK_RET(HcommChannelNotifyWaitOnThread(
                thread, rootChannel->handle, NOTIFY_IDX_SCATTER_DATA, CUSTOM_TIMEOUT));

            CHK_RET(HcommReadOnThread(
                thread, rootChannel->handle, outputChunk, rootChannel->remoteCclMem.addr, chunkBytes));

            CHK_RET(HcommChannelNotifyRecordOnThread(
                thread, rootChannel->handle, NOTIFY_IDX_SCATTER_ACK));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult RootServerScatterOwnSegment(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t blockOffset, uint64_t blockBytes, const uint64_t *segmentOffsets)
{
    const ThreadHandle thread = resCtx.threads[0];

    const uint32_t rootServer = GetServerId(param.root);
    const uint32_t myServer = GetServerId(param.myRank);
    const uint32_t myLocal = GetLocalId(param.myRank);

    uint8_t *inputBuffer = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);

    if (myServer != rootServer) {
        return HCCL_SUCCESS;
    }

    if (param.myRank == param.root) {
        void *inputBlock = static_cast<void *>(inputBuffer + blockOffset);

        CHK_RET(HcommLocalCopyOnThread(
            thread, localBuffer, inputBlock, blockBytes));

        // 只 scatter 给 root server 内其他 7 个 rank。
        // 每个 rank 只从 root 读自己的 1/8 segment。
        for (uint32_t localId = 0; localId < SERVER_RANK_NUM; ++localId) {
            const uint32_t peerRank = MakeRank(rootServer, localId);
            if (peerRank == param.root) {
                continue;
            }

            const ChannelInfo *peerChannel = FindChannel(resCtx, peerRank);
            if (peerChannel == nullptr || peerChannel->remoteCclMem.addr == nullptr) {
                HCCL_ERROR("Scatter peer channel missing, root[%u], peerRank[%u]",
                    param.root, peerRank);
                return HCCL_E_PTR;
            }

            CHK_RET(HcommChannelNotifyRecordOnThread(
                thread, peerChannel->handle, NOTIFY_IDX_SCATTER_DATA));
        }

        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
    if (rootChannel == nullptr || rootChannel->remoteCclMem.addr == nullptr) {
        HCCL_ERROR("Channel to root[%u] is missing", param.root);
        return HCCL_E_PTR;
    }

    const uint64_t segOffset = segmentOffsets[myLocal];
    const uint64_t segBytes = segmentOffsets[myLocal + 1] - segmentOffsets[myLocal];

    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, rootChannel->handle, NOTIFY_IDX_SCATTER_DATA, CUSTOM_TIMEOUT));

    if (segBytes > 0) {
        uint8_t *src = static_cast<uint8_t *>(rootChannel->remoteCclMem.addr) + segOffset;
        uint8_t *dst = localBuffer + segOffset;

        CHK_RET(HcommReadOnThread(
            thread, rootChannel->handle, dst, src, segBytes));
    }

    CHK_RET(HcommChannelNotifyRecordOnThread(
        thread, rootChannel->handle, NOTIFY_IDX_SCATTER_ACK));

    return HCCL_SUCCESS;
}

HcclResult WaitRootServerScatterAckIfRoot(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.myRank != param.root) {
        return HCCL_SUCCESS;
    }

    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t rootServer = GetServerId(param.root);

    for (uint32_t localId = 0; localId < SERVER_RANK_NUM; ++localId) {
        const uint32_t peerRank = MakeRank(rootServer, localId);
        if (peerRank == param.root) {
            continue;
        }

        const ChannelInfo *peerChannel = FindChannel(resCtx, peerRank);
        if (peerChannel == nullptr || peerChannel->remoteCclMem.addr == nullptr) {
            HCCL_ERROR("Scatter ACK peer channel missing, root[%u], peerRank[%u]",
                param.root, peerRank);
            return HCCL_E_PTR;
        }

        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, peerChannel->handle, NOTIFY_IDX_SCATTER_ACK, CUSTOM_TIMEOUT));
    }

    return HCCL_SUCCESS;
}

HcclResult CrossServerReceiveOwnSegmentIfRemote(const OpParam &param, const AlgResourceCtx &resCtx,
    const uint64_t *segmentOffsets)
{
    const ThreadHandle thread = resCtx.threads[0];

    const uint32_t rootServer = GetServerId(param.root);
    const uint32_t myServer = GetServerId(param.myRank);
    const uint32_t myLocal = GetLocalId(param.myRank);

    if (myServer == rootServer) {
        return HCCL_SUCCESS;
    }

    const uint32_t producerRank = MakeRank(rootServer, myLocal);
    const ChannelInfo *producerChannel = FindChannel(resCtx, producerRank);
    if (producerChannel == nullptr || producerChannel->remoteCclMem.addr == nullptr) {
        HCCL_ERROR("Cross producer channel missing, myRank[%u], producerRank[%u]",
            param.myRank, producerRank);
        return HCCL_E_PTR;
    }

    uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);

    const uint64_t segOffset = segmentOffsets[myLocal];
    const uint64_t segBytes = segmentOffsets[myLocal + 1] - segmentOffsets[myLocal];

    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, producerChannel->handle, NOTIFY_IDX_SEGMENT_DATA, CUSTOM_TIMEOUT));

    if (segBytes > 0) {
        uint8_t *src = static_cast<uint8_t *>(producerChannel->remoteCclMem.addr) + segOffset;
        uint8_t *dst = localBuffer + segOffset;

        CHK_RET(HcommReadOnThread(
            thread, producerChannel->handle, dst, src, segBytes));
    }

    CHK_RET(HcommChannelNotifyRecordOnThread(
        thread, producerChannel->handle, NOTIFY_IDX_SEGMENT_ACK));

    return HCCL_SUCCESS;
}

bool ShouldSendLocalSegmentToConsumer(const OpParam &param, uint32_t consumerRank)
{
    // root 本身已经拥有完整用户 buffer，不需要从其他 rank 读 segment。
    return consumerRank != param.root;
}

HcclResult RecordOwnSegmentReady(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const ThreadHandle thread = resCtx.threads[0];

    const uint32_t rootServer = GetServerId(param.root);
    const uint32_t myServer = GetServerId(param.myRank);
    const uint32_t myLocal = GetLocalId(param.myRank);

    // 1. 通知同 server 内其他 rank：我的 1/8 segment 已经准备好。
    for (uint32_t localId = 0; localId < SERVER_RANK_NUM; ++localId) {
        const uint32_t consumerRank = MakeRank(myServer, localId);
        if (consumerRank == param.myRank || !ShouldSendLocalSegmentToConsumer(param, consumerRank)) {
            continue;
        }

        const ChannelInfo *consumerChannel = FindChannel(resCtx, consumerRank);
        if (consumerChannel == nullptr || consumerChannel->remoteCclMem.addr == nullptr) {
            HCCL_ERROR("Local segment consumer channel missing, myRank[%u], consumerRank[%u]",
                param.myRank, consumerRank);
            return HCCL_E_PTR;
        }

        CHK_RET(HcommChannelNotifyRecordOnThread(
            thread, consumerChannel->handle, NOTIFY_IDX_SEGMENT_DATA));
    }

    // 2. root server 中的每个 local owner，再通知 remote server 对应 local rank。
    if (myServer == rootServer) {
        const uint32_t remoteServer = GetOppositeServer(rootServer);
        const uint32_t remotePairRank = MakeRank(remoteServer, myLocal);

        const ChannelInfo *remotePairChannel = FindChannel(resCtx, remotePairRank);
        if (remotePairChannel == nullptr || remotePairChannel->remoteCclMem.addr == nullptr) {
            HCCL_ERROR("Cross remote pair channel missing, myRank[%u], remotePairRank[%u]",
                param.myRank, remotePairRank);
            return HCCL_E_PTR;
        }

        CHK_RET(HcommChannelNotifyRecordOnThread(
            thread, remotePairChannel->handle, NOTIFY_IDX_SEGMENT_DATA));
    }

    return HCCL_SUCCESS;
}

HcclResult ConsumeOtherSegmentsInMyServer(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t blockOffset, const uint64_t *segmentOffsets)
{
    if (param.myRank == param.root) {
        return HCCL_SUCCESS;
    }

    const ThreadHandle thread = resCtx.threads[0];

    const uint32_t myServer = GetServerId(param.myRank);
    const uint32_t myLocal = GetLocalId(param.myRank);

    uint8_t *outputBuffer = static_cast<uint8_t *>(param.outputPtr);

    for (uint32_t ownerLocal = 0; ownerLocal < SERVER_RANK_NUM; ++ownerLocal) {
        if (ownerLocal == myLocal) {
            continue;
        }

        const uint32_t ownerRank = MakeRank(myServer, ownerLocal);
        const ChannelInfo *ownerChannel = FindChannel(resCtx, ownerRank);
        if (ownerChannel == nullptr || ownerChannel->remoteCclMem.addr == nullptr) {
            HCCL_ERROR("Owner channel missing, myRank[%u], ownerRank[%u]",
                param.myRank, ownerRank);
            return HCCL_E_PTR;
        }

        const uint64_t segOffset = segmentOffsets[ownerLocal];
        const uint64_t segBytes = segmentOffsets[ownerLocal + 1] - segmentOffsets[ownerLocal];

        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, ownerChannel->handle, NOTIFY_IDX_SEGMENT_DATA, CUSTOM_TIMEOUT));

        if (segBytes > 0) {
            uint8_t *src = static_cast<uint8_t *>(ownerChannel->remoteCclMem.addr) + segOffset;
            uint8_t *dst = outputBuffer + blockOffset + segOffset;

            CHK_RET(HcommReadOnThread(
                thread, ownerChannel->handle, dst, src, segBytes));
        }

        CHK_RET(HcommChannelNotifyRecordOnThread(
            thread, ownerChannel->handle, NOTIFY_IDX_SEGMENT_ACK));
    }

    return HCCL_SUCCESS;
}

HcclResult CopyOwnSegmentToOutput(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t blockOffset, const uint64_t *segmentOffsets)
{
    if (param.myRank == param.root) {
        return HCCL_SUCCESS;
    }

    const ThreadHandle thread = resCtx.threads[0];

    const uint32_t myLocal = GetLocalId(param.myRank);
    const uint64_t segOffset = segmentOffsets[myLocal];
    const uint64_t segBytes = segmentOffsets[myLocal + 1] - segmentOffsets[myLocal];

    if (segBytes == 0) {
        return HCCL_SUCCESS;
    }

    uint8_t *outputBuffer = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);

    CHK_RET(HcommLocalCopyOnThread(
        thread,
        outputBuffer + blockOffset + segOffset,
        localBuffer + segOffset,
        segBytes));

    return HCCL_SUCCESS;
}

HcclResult WaitOwnSegmentAck(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const ThreadHandle thread = resCtx.threads[0];

    const uint32_t rootServer = GetServerId(param.root);
    const uint32_t myServer = GetServerId(param.myRank);
    const uint32_t myLocal = GetLocalId(param.myRank);

    // 1. 等同 server 内消费者读完我的 1/8 segment。
    for (uint32_t localId = 0; localId < SERVER_RANK_NUM; ++localId) {
        const uint32_t consumerRank = MakeRank(myServer, localId);
        if (consumerRank == param.myRank || !ShouldSendLocalSegmentToConsumer(param, consumerRank)) {
            continue;
        }

        const ChannelInfo *consumerChannel = FindChannel(resCtx, consumerRank);
        if (consumerChannel == nullptr || consumerChannel->remoteCclMem.addr == nullptr) {
            HCCL_ERROR("Consumer ACK channel missing, myRank[%u], consumerRank[%u]",
                param.myRank, consumerRank);
            return HCCL_E_PTR;
        }

        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, consumerChannel->handle, NOTIFY_IDX_SEGMENT_ACK, CUSTOM_TIMEOUT));
    }

    // 2. root server 的 segment owner 还要等 remote pair 跨 server 读完。
    if (myServer == rootServer) {
        const uint32_t remoteServer = GetOppositeServer(rootServer);
        const uint32_t remotePairRank = MakeRank(remoteServer, myLocal);

        const ChannelInfo *remotePairChannel = FindChannel(resCtx, remotePairRank);
        if (remotePairChannel == nullptr || remotePairChannel->remoteCclMem.addr == nullptr) {
            HCCL_ERROR("Cross remote pair ACK channel missing, myRank[%u], remotePairRank[%u]",
                param.myRank, remotePairRank);
            return HCCL_E_PTR;
        }

        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, remotePairChannel->handle, NOTIFY_IDX_SEGMENT_ACK, CUSTOM_TIMEOUT));
    }

    return HCCL_SUCCESS;
}

HcclResult Hierarchical8BroadcastOneBlock(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t blockOffset, uint64_t blockBytes, const uint64_t *segmentOffsets)
{
    const uint32_t rootServer = GetServerId(param.root);
    const uint32_t myServer = GetServerId(param.myRank);

    // Step 1：root server 内 scatter，每个 root-server rank 拿自己的 1/8 segment。
    CHK_RET(RootServerScatterOwnSegment(param, resCtx, blockOffset, blockBytes, segmentOffsets));

    // Step 2：root server 的 8 个 segment owner 先通知本 server peers 和 remote pair。
    if (myServer == rootServer) {
        CHK_RET(RecordOwnSegmentReady(param, resCtx));
    }

    // Step 3：remote server 对应 rank 跨 server 读取自己的 1/8 segment。
    CHK_RET(CrossServerReceiveOwnSegmentIfRemote(param, resCtx, segmentOffsets));

    // Step 4：remote server 拿到 own segment 后，通知 remote server 内其他 peers。
    if (myServer != rootServer) {
        CHK_RET(RecordOwnSegmentReady(param, resCtx));
    }

    // Step 5：两个 server 内分别 allgather。
    // 其他 7/8 segment 直接 read 到用户 output。
    CHK_RET(ConsumeOtherSegmentsInMyServer(param, resCtx, blockOffset, segmentOffsets));

    // Step 6：自己的 1/8 segment 从 localBuffer copy 到用户 output。
    CHK_RET(CopyOwnSegmentToOutput(param, resCtx, blockOffset, segmentOffsets));

    // Step 7：等待消费者读完我的 segment，保证 localBuffer 下一个 block 可安全复用。
    CHK_RET(WaitOwnSegmentAck(param, resCtx));

    // Step 8：root 最后等 root-server scatter ACK。
    CHK_RET(WaitRootServerScatterAckIfRoot(param, resCtx));

    return HCCL_SUCCESS;
}

HcclResult Hierarchical8Broadcast(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t totalBytes, uint64_t chunkCapacity)
{
    const uint64_t blockCapacity = ChooseBlockCapacity(totalBytes, chunkCapacity);
    if (blockCapacity == 0) {
        HCCL_ERROR("Invalid blockCapacity, chunkCapacity[%llu]",
            static_cast<unsigned long long>(chunkCapacity));
        return HCCL_E_PARA;
    }

    uint64_t segmentOffsets[SERVER_RANK_NUM + 1] = {};

    for (uint64_t blockOffset = 0; blockOffset < totalBytes; blockOffset += blockCapacity) {
        const uint64_t blockBytes = std::min(blockCapacity, totalBytes - blockOffset);

        BuildSegmentOffsets8(blockBytes, segmentOffsets);

        CHK_RET(Hierarchical8BroadcastOneBlock(
            param, resCtx, blockOffset, blockBytes, segmentOffsets));
    }

    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing Broadcast hierarchical 8+8 v1 on AICPU_TS");

    if (param.rankSize == 0 || param.myRank >= param.rankSize || param.root >= param.rankSize) {
        HCCL_ERROR("Invalid rank information, myRank[%u], rankSize[%u], root[%u]",
            param.myRank, param.rankSize, param.root);
        return HCCL_E_PARA;
    }

    if (param.dataType != HCCL_DATA_TYPE_FP32) {
        HCCL_ERROR("Unsupported data type[%d]", static_cast<int32_t>(param.dataType));
        return HCCL_E_PARA;
    }

    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    if (param.inputPtr == nullptr || param.outputPtr == nullptr ||
        resCtx.localBuffer.addr == nullptr || resCtx.threads.empty()) {
        HCCL_ERROR("Invalid broadcast buffer or thread resource");
        return HCCL_E_PTR;
    }

    if (resCtx.channels.size() + 1 != param.rankSize) {
        HCCL_ERROR("Channel count mismatch, channelNum[%zu], rankSize[%u]",
            resCtx.channels.size(), param.rankSize);
        return HCCL_E_PARA;
    }

    if (param.count > std::numeric_limits<uint64_t>::max() / FP32_BYTES) {
        HCCL_ERROR("Broadcast byte size overflows uint64_t");
        return HCCL_E_PARA;
    }

    const uint64_t totalBytes = param.count * FP32_BYTES;
    const uint64_t chunkCapacity = GetChunkCapacity(resCtx);
    if (chunkCapacity == 0) {
        HCCL_ERROR("HCCL communication buffer is too small");
        return HCCL_E_PARA;
    }

    // 保护小包性能。
    if (param.rankSize != TOTAL_RANK_NUM || totalBytes <= LARGE_MSG_THRESHOLD_BYTES) {
        return DirectFanoutBroadcast(param, resCtx, totalBytes, chunkCapacity);
    }

    return Hierarchical8Broadcast(param, resCtx, totalBytes, chunkCapacity);
}
} // namespace ops_hccl