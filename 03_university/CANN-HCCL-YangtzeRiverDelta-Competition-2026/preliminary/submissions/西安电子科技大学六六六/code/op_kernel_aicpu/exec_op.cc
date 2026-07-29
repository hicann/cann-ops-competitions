/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_CHUNK_BYTES =
    256ULL * 1024 * 1024;

constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t COMPETITION_RANK_SIZE = 16;

HcclResult GetChannel(
    const AlgResourceCtx &resCtx,
    uint32_t remoteRank,
    const ChannelInfo *&result)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            result = &channel;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR(
        "Channel to rank[%u] was not acquired",
        remoteRank);

    return HCCL_E_INTERNAL;
}

HcclResult WriteAndSignal(
    ThreadHandle thread,
    const ChannelInfo &channel,
    const void *src,
    uint64_t bytes)
{
    CHK_PRT_RET(
        channel.remoteCclMem.size < bytes,
        HCCL_ERROR(
            "Remote buffer is too small, size[%llu], bytes[%llu]",
            static_cast<unsigned long long>(
                channel.remoteCclMem.size),
            static_cast<unsigned long long>(bytes)),
        HCCL_E_INTERNAL);

    CHK_RET(static_cast<HcclResult>(
        HcommWriteOnThread(
            thread,
            channel.handle,
            channel.remoteCclMem.addr,
            src,
            bytes)));

    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(
            thread,
            channel.handle,
            NOTIFY_IDX_DATA_SIGNAL)));

    return HCCL_SUCCESS;
}

HcclResult WaitData(
    ThreadHandle thread,
    const ChannelInfo &channel)
{
    return static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(
            thread,
            channel.handle,
            NOTIFY_IDX_DATA_SIGNAL,
            CUSTOM_TIMEOUT));
}

HcclResult RecordAck(
    ThreadHandle thread,
    const ChannelInfo &channel)
{
    return static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(
            thread,
            channel.handle,
            NOTIFY_IDX_ACK));
}

HcclResult WaitAck(
    ThreadHandle thread,
    const ChannelInfo &channel)
{
    return static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(
            thread,
            channel.handle,
            NOTIFY_IDX_ACK,
            CUSTOM_TIMEOUT));
}

HcclResult ExecTwoServerBroadcast(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t dataBytes,
    uint64_t chunkCapacity)
{
    ThreadHandle thread = resCtx.aicpuThread;
    auto *userBuffer =
        static_cast<uint8_t *>(param.outputPtr);

    void *localBuffer = resCtx.localBuffer.addr;

    const uint32_t sourceServerBase =
        (param.root / RANKS_PER_SERVER) *
        RANKS_PER_SERVER;

    const bool isSourceServer =
        param.myRank >= sourceServerBase &&
        param.myRank <
            sourceServerBase + RANKS_PER_SERVER;

    for (uint64_t offset = 0;
         offset < dataBytes;
         offset += chunkCapacity) {
        const uint64_t bytes =
            std::min(chunkCapacity, dataBytes - offset);

        void *userChunk = userBuffer + offset;

        if (param.myRank == param.root) {
            // root数据先复制到本端HCCL Buffer。
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(
                    thread,
                    localBuffer,
                    userChunk,
                    bytes)));

            // 第一阶段：
            // root通过服务器内Full-Mesh发送给同服务器其他7个rank。
            for (uint32_t rank = sourceServerBase;
                 rank < sourceServerBase + RANKS_PER_SERVER;
                 ++rank) {
                if (rank == param.root) {
                    continue;
                }

                const ChannelInfo *channel = nullptr;

                CHK_RET(GetChannel(
                    resCtx,
                    rank,
                    channel));

                CHK_RET(WriteAndSignal(
                    thread,
                    *channel,
                    localBuffer,
                    bytes));
            }

            // root同时通过自己的Clos链路发送到另一台服务器。
            const ChannelInfo *pairChannel = nullptr;

            CHK_RET(GetChannel(
                resCtx,
                param.myRank ^ RANKS_PER_SERVER,
                pairChannel));

            CHK_RET(WriteAndSignal(
                thread,
                *pairChannel,
                localBuffer,
                bytes));

            CHK_RET(WaitAck(
                thread,
                *pairChannel));

            // 等待同服务器rank完成接收和跨服务器转发。
            for (uint32_t rank = sourceServerBase;
                 rank < sourceServerBase + RANKS_PER_SERVER;
                 ++rank) {
                if (rank == param.root) {
                    continue;
                }

                const ChannelInfo *channel = nullptr;

                CHK_RET(GetChannel(
                    resCtx,
                    rank,
                    channel));

                CHK_RET(WaitAck(
                    thread,
                    *channel));
            }
        } else if (isSourceServer) {
            // 同服务器非root rank：
            // 先从root接收，再通过自己的Clos链路转发。
            const ChannelInfo *rootChannel = nullptr;
            const ChannelInfo *pairChannel = nullptr;

            CHK_RET(GetChannel(
                resCtx,
                param.root,
                rootChannel));

            CHK_RET(GetChannel(
                resCtx,
                param.myRank ^ RANKS_PER_SERVER,
                pairChannel));

            CHK_RET(WaitData(
                thread,
                *rootChannel));

            CHK_RET(WriteAndSignal(
                thread,
                *pairChannel,
                localBuffer,
                bytes));

            // 必须等另一台服务器读取完成，
            // 才能复用本地HCCL Buffer。
            CHK_RET(WaitAck(
                thread,
                *pairChannel));

            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(
                    thread,
                    userChunk,
                    localBuffer,
                    bytes)));

            CHK_RET(RecordAck(
                thread,
                *rootChannel));
        } else {
            // 目标服务器：
            // 从拥有相同本地NPU编号的rank接收。
            const ChannelInfo *pairChannel = nullptr;

            CHK_RET(GetChannel(
                resCtx,
                param.myRank ^ RANKS_PER_SERVER,
                pairChannel));

            CHK_RET(WaitData(
                thread,
                *pairChannel));

            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(
                    thread,
                    userChunk,
                    localBuffer,
                    bytes)));

            CHK_RET(RecordAck(
                thread,
                *pairChannel));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult ExecDirectBroadcast(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t dataBytes,
    uint64_t chunkCapacity)
{
    ThreadHandle thread = resCtx.aicpuThread;
    auto *userBuffer =
        static_cast<uint8_t *>(param.outputPtr);

    void *localBuffer = resCtx.localBuffer.addr;

    for (uint64_t offset = 0;
         offset < dataBytes;
         offset += chunkCapacity) {
        const uint64_t bytes =
            std::min(chunkCapacity, dataBytes - offset);

        void *userChunk = userBuffer + offset;

        if (param.myRank == param.root) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(
                    thread,
                    localBuffer,
                    userChunk,
                    bytes)));

            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(WriteAndSignal(
                    thread,
                    channel,
                    localBuffer,
                    bytes));
            }

            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(WaitAck(
                    thread,
                    channel));
            }
        } else {
            const ChannelInfo *rootChannel = nullptr;

            CHK_RET(GetChannel(
                resCtx,
                param.root,
                rootChannel));

            CHK_RET(WaitData(
                thread,
                *rootChannel));

            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(
                    thread,
                    userChunk,
                    localBuffer,
                    bytes)));

            CHK_RET(RecordAck(
                thread,
                *rootChannel));
        }
    }

    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(
    const OpParam &param,
    const AlgResourceCtx &resCtx)
{
    if (param.count == 0 || param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    const auto typeIter =
        SIZE_TABLE.find(param.dataType);

    CHK_PRT_RET(
        typeIter == SIZE_TABLE.end(),
        HCCL_ERROR(
            "Unsupported data type[%d]",
            static_cast<int>(param.dataType)),
        HCCL_E_PARA);

    const uint64_t dataBytes =
        param.count * typeIter->second;

    const uint64_t chunkCapacity =
        std::min(
            MAX_CHUNK_BYTES,
            resCtx.localBuffer.size);

    CHK_PRT_RET(
        chunkCapacity == 0,
        HCCL_ERROR("Local HCCL buffer is empty"),
        HCCL_E_INTERNAL);

    if (param.rankSize == COMPETITION_RANK_SIZE) {
        return ExecTwoServerBroadcast(
            param,
            resCtx,
            dataBytes,
            chunkCapacity);
    }

    return ExecDirectBroadcast(
        param,
        resCtx,
        dataBytes,
        chunkCapacity);
}
} // namespace ops_hccl