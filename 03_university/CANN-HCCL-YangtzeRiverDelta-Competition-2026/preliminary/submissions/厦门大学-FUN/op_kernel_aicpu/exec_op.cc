/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0 (the "License").
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {

constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t EXPECTED_RANK_SIZE = 16;
constexpr uint32_t THREAD_NUM = 16;

constexpr uint32_t READY_NOTIFY = 0;
constexpr uint32_t DATA_NOTIFY = 1;
constexpr uint32_t READ_NOTIFY = 2;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

constexpr uint64_t SMALL_THRESHOLD = 1ULL * 1024ULL * 1024ULL;
constexpr uint64_t SLOT_ALIGN = 128;
constexpr uint64_t FP32_BYTES = sizeof(float);

struct SendTask {
    const ChannelInfo *channel = nullptr;
    const void *source = nullptr;
    uint64_t remoteOffset = 0;
    uint64_t bytes = 0;
};

enum class TaskKind {
    SEND,
    RECV,
    EXCHANGE
};

struct OpTask {
    TaskKind kind = TaskKind::SEND;
    const ChannelInfo *channel = nullptr;

    const void *sendSource = nullptr;
    uint64_t sendRemoteOffset = 0;
    uint64_t sendBytes = 0;

    void *recvDestination = nullptr;
    uint64_t recvLocalOffset = 0;
    uint64_t recvBytes = 0;
};

HcclResult Primitive(int32_t ret)
{
    return ret == static_cast<int32_t>(HCCL_SUCCESS) ? HCCL_SUCCESS : static_cast<HcclResult>(ret);
}

uint32_t ServerOf(uint32_t rank)
{
    return rank / RANKS_PER_SERVER;
}

uint32_t LocalOf(uint32_t rank)
{
    return rank % RANKS_PER_SERVER;
}

uint32_t LogicalRank(uint32_t rank, uint32_t rootLocal)
{
    return (LocalOf(rank) + RANKS_PER_SERVER - rootLocal) % RANKS_PER_SERVER;
}

uint32_t ActualRank(uint32_t server, uint32_t logical, uint32_t rootLocal)
{
    return server * RANKS_PER_SERVER + (rootLocal + logical) % RANKS_PER_SERVER;
}

uint64_t AlignDown(uint64_t value, uint64_t align)
{
    return value / align * align;
}

uint64_t StripeBeginCount(uint64_t totalCount, uint32_t stripe)
{
    uint64_t base = totalCount / RANKS_PER_SERVER;
    uint64_t rem = totalCount % RANKS_PER_SERVER;
    uint64_t s = static_cast<uint64_t>(stripe);
    return s * base + std::min(s, rem);
}

uint64_t StripeCount(uint64_t totalCount, uint32_t stripe)
{
    uint64_t base = totalCount / RANKS_PER_SERVER;
    uint64_t rem = totalCount % RANKS_PER_SERVER;
    return base + (static_cast<uint64_t>(stripe) < rem ? 1ULL : 0ULL);
}

uint64_t SliceCount(uint64_t totalCount, uint32_t stripe, uint64_t processed, uint64_t maxSliceCount)
{
    uint64_t stripeCount = StripeCount(totalCount, stripe);

    if (processed >= stripeCount) {
        return 0;
    }

    return std::min(maxSliceCount, stripeCount - processed);
}

uint64_t SliceBytes(uint64_t totalCount, uint32_t stripe, uint64_t processed, uint64_t maxSliceCount)
{
    return SliceCount(totalCount, stripe, processed, maxSliceCount) * FP32_BYTES;
}

uint64_t UserOffset(uint64_t totalCount, uint32_t stripe, uint64_t processed)
{
    return (StripeBeginCount(totalCount, stripe) + processed) * FP32_BYTES;
}

uint64_t SlotOffset(uint32_t stripe, uint64_t slotStride)
{
    return static_cast<uint64_t>(stripe) * slotStride;
}

bool BufferRangeValid(uint64_t offset, uint64_t bytes, uint64_t size)
{
    return offset <= size && bytes <= size - offset;
}

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }

    return nullptr;
}

HcclResult CheckChannel(const ChannelInfo *channel)
{
    if (channel == nullptr || channel->handle == 0 ||
        channel->notifyNum < CHANNEL_NOTIFY_NUM ||
        channel->remoteCclMem.addr == nullptr ||
        channel->remoteCclMem.size == 0) {
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult StartWorkers(const std::vector<ThreadHandle> &threads, uint32_t active)
{
    if (active == 0 || active > threads.size()) {
        return HCCL_E_INTERNAL;
    }

    for (uint32_t i = 1; i < active; ++i) {
        CHK_RET(Primitive(HcommThreadNotifyRecordOnThread(threads[0], threads[i], 0)));
    }

    for (uint32_t i = 1; i < active; ++i) {
        CHK_RET(Primitive(HcommThreadNotifyWaitOnThread(threads[i], 0, CUSTOM_TIMEOUT)));
    }

    return HCCL_SUCCESS;
}

HcclResult JoinWorkers(const std::vector<ThreadHandle> &threads, uint32_t active)
{
    if (active == 0 || active > threads.size()) {
        return HCCL_E_INTERNAL;
    }

    for (uint32_t i = 1; i < active; ++i) {
        CHK_RET(Primitive(HcommThreadNotifyWaitOnThread(threads[0], i, CUSTOM_TIMEOUT)));
    }

    for (uint32_t i = 1; i < active; ++i) {
        CHK_RET(Primitive(HcommThreadNotifyRecordOnThread(threads[i], threads[0], i)));
    }

    return HCCL_SUCCESS;
}

HcclResult SendOne(ThreadHandle thread, const SendTask &task)
{
    CHK_RET(CheckChannel(task.channel));

    if (task.bytes == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        !BufferRangeValid(task.remoteOffset, task.bytes, task.channel->remoteCclMem.size),
        HCCL_ERROR("remote HCCL buffer is too small"),
        HCCL_E_INTERNAL);

    void *remote = static_cast<uint8_t *>(task.channel->remoteCclMem.addr) + task.remoteOffset;

    CHK_RET(Primitive(HcommChannelNotifyWaitOnThread(
        thread,
        task.channel->handle,
        READY_NOTIFY,
        CUSTOM_TIMEOUT)));

    CHK_RET(Primitive(HcommWriteOnThread(
        thread,
        task.channel->handle,
        remote,
        task.source,
        task.bytes)));

    CHK_RET(Primitive(HcommChannelNotifyRecordOnThread(
        thread,
        task.channel->handle,
        DATA_NOTIFY)));

    CHK_RET(Primitive(HcommChannelNotifyWaitOnThread(
        thread,
        task.channel->handle,
        READ_NOTIFY,
        CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult ReceiveOne(
    ThreadHandle thread,
    const ChannelInfo &channel,
    void *localBuffer,
    uint64_t localSize,
    uint64_t localOffset,
    void *destination,
    uint64_t bytes,
    bool recordRead)
{
    CHK_RET(CheckChannel(&channel));

    if (bytes == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        localBuffer == nullptr || !BufferRangeValid(localOffset, bytes, localSize),
        HCCL_ERROR("local HCCL buffer is too small"),
        HCCL_E_INTERNAL);

    void *source = static_cast<uint8_t *>(localBuffer) + localOffset;

    CHK_RET(Primitive(HcommChannelNotifyRecordOnThread(
        thread,
        channel.handle,
        READY_NOTIFY)));

    CHK_RET(Primitive(HcommChannelNotifyWaitOnThread(
        thread,
        channel.handle,
        DATA_NOTIFY,
        CUSTOM_TIMEOUT)));

    CHK_RET(Primitive(HcommLocalCopyOnThread(
        thread,
        destination,
        source,
        bytes)));

    if (recordRead) {
        CHK_RET(Primitive(HcommChannelNotifyRecordOnThread(
            thread,
            channel.handle,
            READ_NOTIFY)));
    }

    return HCCL_SUCCESS;
}

HcclResult ExchangeOne(
    ThreadHandle thread,
    const ChannelInfo &channel,
    void *localBuffer,
    uint64_t localSize,
    uint64_t sendRemoteOffset,
    const void *sendSource,
    uint64_t sendBytes,
    uint64_t recvLocalOffset,
    void *recvDestination,
    uint64_t recvBytes)
{
    CHK_RET(CheckChannel(&channel));

    CHK_PRT_RET(
        !BufferRangeValid(sendRemoteOffset, sendBytes, channel.remoteCclMem.size) ||
        !BufferRangeValid(recvLocalOffset, recvBytes, localSize),
        HCCL_ERROR("invalid exchange range"),
        HCCL_E_INTERNAL);

    CHK_RET(Primitive(HcommChannelNotifyRecordOnThread(
        thread,
        channel.handle,
        READY_NOTIFY)));

    CHK_RET(Primitive(HcommChannelNotifyWaitOnThread(
        thread,
        channel.handle,
        READY_NOTIFY,
        CUSTOM_TIMEOUT)));

    if (sendBytes > 0) {
        void *remote = static_cast<uint8_t *>(channel.remoteCclMem.addr) + sendRemoteOffset;

        CHK_RET(Primitive(HcommWriteOnThread(
            thread,
            channel.handle,
            remote,
            sendSource,
            sendBytes)));
    }

    CHK_RET(Primitive(HcommChannelNotifyRecordOnThread(
        thread,
        channel.handle,
        DATA_NOTIFY)));

    CHK_RET(Primitive(HcommChannelNotifyWaitOnThread(
        thread,
        channel.handle,
        DATA_NOTIFY,
        CUSTOM_TIMEOUT)));

    if (recvBytes > 0) {
        void *source = static_cast<uint8_t *>(localBuffer) + recvLocalOffset;

        CHK_RET(Primitive(HcommLocalCopyOnThread(
            thread,
            recvDestination,
            source,
            recvBytes)));
    }

    CHK_RET(Primitive(HcommChannelNotifyRecordOnThread(
        thread,
        channel.handle,
        READ_NOTIFY)));

    CHK_RET(Primitive(HcommChannelNotifyWaitOnThread(
        thread,
        channel.handle,
        READ_NOTIFY,
        CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult ParallelSend(const AlgResourceCtx &resCtx, const SendTask *tasks, uint32_t taskNum)
{
    if (taskNum == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        taskNum > resCtx.threads.size(),
        HCCL_ERROR("not enough AICPU threads"),
        HCCL_E_INTERNAL);

    CHK_RET(StartWorkers(resCtx.threads, taskNum));

    for (uint32_t i = 0; i < taskNum; ++i) {
        CHK_RET(SendOne(resCtx.threads[i], tasks[i]));
    }

    CHK_RET(JoinWorkers(resCtx.threads, taskNum));
    return HCCL_SUCCESS;
}

HcclResult RunTasks(const AlgResourceCtx &resCtx, const OpTask *tasks, uint32_t taskNum)
{
    if (taskNum == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        taskNum > resCtx.threads.size(),
        HCCL_ERROR("not enough AICPU threads for mixed tasks"),
        HCCL_E_INTERNAL);

    CHK_RET(StartWorkers(resCtx.threads, taskNum));

    for (uint32_t i = 0; i < taskNum; ++i) {
        const OpTask &task = tasks[i];
        ThreadHandle thread = resCtx.threads[i];

        if (task.kind == TaskKind::SEND) {
            SendTask sendTask{task.channel, task.sendSource, task.sendRemoteOffset, task.sendBytes};
            CHK_RET(SendOne(thread, sendTask));
        } else if (task.kind == TaskKind::RECV) {
            CHK_RET(ReceiveOne(
                thread,
                *task.channel,
                resCtx.localBuffer.addr,
                resCtx.localBuffer.size,
                task.recvLocalOffset,
                task.recvDestination,
                task.recvBytes,
                true));
        } else {
            CHK_RET(ExchangeOne(
                thread,
                *task.channel,
                resCtx.localBuffer.addr,
                resCtx.localBuffer.size,
                task.sendRemoteOffset,
                task.sendSource,
                task.sendBytes,
                task.recvLocalOffset,
                task.recvDestination,
                task.recvBytes));
        }
    }

    CHK_RET(JoinWorkers(resCtx.threads, taskNum));
    return HCCL_SUCCESS;
}

HcclResult RunSmallTwoBridge(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t offset,
    uint64_t bytes)
{
    if (bytes == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        bytes > resCtx.localBuffer.size,
        HCCL_ERROR("small message exceeds local HCCL buffer"),
        HCCL_E_INTERNAL);

    uint32_t rootServer = ServerOf(param.root);
    uint32_t rootLocal = LocalOf(param.root);
    uint32_t otherServer = 1U - rootServer;
    uint32_t bridge0 = ActualRank(otherServer, 0, rootLocal);
    uint32_t bridge4 = ActualRank(otherServer, 4, rootLocal);

    auto *input = static_cast<uint8_t *>(param.inputPtr);
    auto *output = static_cast<uint8_t *>(param.outputPtr);

    if (param.myRank == param.root) {
        SendTask tasks[RANKS_PER_SERVER + 1];
        uint32_t taskNum = 0;

        for (uint32_t logical = 1; logical < RANKS_PER_SERVER; ++logical) {
            uint32_t peer = ActualRank(rootServer, logical, rootLocal);

            tasks[taskNum++] = SendTask{
                FindChannel(resCtx, peer),
                input + offset,
                0,
                bytes
            };
        }

        tasks[taskNum++] = SendTask{
            FindChannel(resCtx, bridge0),
            input + offset,
            0,
            bytes
        };

        tasks[taskNum++] = SendTask{
            FindChannel(resCtx, bridge4),
            input + offset,
            0,
            bytes
        };

        return ParallelSend(resCtx, tasks, taskNum);
    }

    if (param.myRank == bridge0 || param.myRank == bridge4) {
        const ChannelInfo *parent = FindChannel(resCtx, param.root);
        CHK_RET(CheckChannel(parent));

        CHK_RET(ReceiveOne(
            resCtx.threads[0],
            *parent,
            resCtx.localBuffer.addr,
            resCtx.localBuffer.size,
            0,
            output + offset,
            bytes,
            true));

        SendTask tasks[3];
        uint32_t taskNum = 0;

        uint32_t beginLogical = param.myRank == bridge0 ? 1U : 5U;
        uint32_t endLogical = param.myRank == bridge0 ? 4U : 8U;

        for (uint32_t logical = beginLogical; logical < endLogical; ++logical) {
            uint32_t peer = ActualRank(otherServer, logical, rootLocal);

            tasks[taskNum++] = SendTask{
                FindChannel(resCtx, peer),
                output + offset,
                0,
                bytes
            };
        }

        return ParallelSend(resCtx, tasks, taskNum);
    }

    uint32_t parentRank;

    if (ServerOf(param.myRank) == rootServer) {
        parentRank = param.root;
    } else {
        uint32_t logical = LogicalRank(param.myRank, rootLocal);
        parentRank = logical < 4 ? bridge0 : bridge4;
    }

    const ChannelInfo *parent = FindChannel(resCtx, parentRank);
    CHK_RET(CheckChannel(parent));

    return ReceiveOne(
        resCtx.threads[0],
        *parent,
        resCtx.localBuffer.addr,
        resCtx.localBuffer.size,
        0,
        output + offset,
        bytes,
        true);
}

HcclResult ScatterSliceV8(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t processed,
    uint64_t maxSliceCount,
    uint64_t slotStride)
{
    uint32_t rootServer = ServerOf(param.root);
    uint32_t rootLocal = LocalOf(param.root);
    uint32_t myServer = ServerOf(param.myRank);

    if (myServer != rootServer) {
        return HCCL_SUCCESS;
    }

    auto *input = static_cast<uint8_t *>(param.inputPtr);
    auto *output = static_cast<uint8_t *>(param.outputPtr);

    if (param.myRank == param.root) {
        SendTask tasks[RANKS_PER_SERVER - 1];
        uint32_t taskNum = 0;

        for (uint32_t stripe = 1; stripe < RANKS_PER_SERVER; ++stripe) {
            uint64_t bytes = SliceBytes(param.count, stripe, processed, maxSliceCount);

            if (bytes == 0) {
                continue;
            }

            uint32_t peer = ActualRank(rootServer, stripe, rootLocal);
            const ChannelInfo *channel = FindChannel(resCtx, peer);

            CHK_RET(CheckChannel(channel));

            tasks[taskNum++] = SendTask{
                channel,
                input + UserOffset(param.count, stripe, processed),
                SlotOffset(stripe, slotStride),
                bytes
            };
        }

        return ParallelSend(resCtx, tasks, taskNum);
    }

    uint32_t logical = LogicalRank(param.myRank, rootLocal);
    uint64_t bytes = SliceBytes(param.count, logical, processed, maxSliceCount);

    if (bytes == 0) {
        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
    CHK_RET(CheckChannel(rootChannel));

    return ReceiveOne(
        resCtx.threads[0],
        *rootChannel,
        resCtx.localBuffer.addr,
        resCtx.localBuffer.size,
        SlotOffset(logical, slotStride),
        output + UserOffset(param.count, logical, processed),
        bytes,
        true);
}

HcclResult DirectDisseminateRoot(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t processed,
    uint64_t maxSliceCount,
    uint64_t slotStride)
{
    uint32_t rootServer = ServerOf(param.root);
    uint32_t rootLocal = LocalOf(param.root);
    uint32_t otherServer = 1U - rootServer;

    auto *input = static_cast<uint8_t *>(param.inputPtr);

    uint64_t rootBytes = SliceBytes(param.count, 0, processed, maxSliceCount);

    if (rootBytes == 0) {
        return HCCL_SUCCESS;
    }

    SendTask tasks[EXPECTED_RANK_SIZE - 1];
    uint32_t taskNum = 0;

    for (uint32_t logical = 1; logical < RANKS_PER_SERVER; ++logical) {
        uint32_t peer = ActualRank(rootServer, logical, rootLocal);
        const ChannelInfo *channel = FindChannel(resCtx, peer);

        CHK_RET(CheckChannel(channel));

        tasks[taskNum++] = SendTask{
            channel,
            input + UserOffset(param.count, 0, processed),
            SlotOffset(0, slotStride),
            rootBytes
        };
    }

    for (uint32_t logical = 0; logical < RANKS_PER_SERVER; ++logical) {
        uint32_t peer = ActualRank(otherServer, logical, rootLocal);
        const ChannelInfo *channel = FindChannel(resCtx, peer);

        CHK_RET(CheckChannel(channel));

        tasks[taskNum++] = SendTask{
            channel,
            input + UserOffset(param.count, 0, processed),
            SlotOffset(0, slotStride),
            rootBytes
        };
    }

    return ParallelSend(resCtx, tasks, taskNum);
}

HcclResult DirectDisseminateRootServerNonRoot(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t processed,
    uint64_t maxSliceCount,
    uint64_t slotStride)
{
    uint32_t rootServer = ServerOf(param.root);
    uint32_t rootLocal = LocalOf(param.root);
    uint32_t otherServer = 1U - rootServer;
    uint32_t logical = LogicalRank(param.myRank, rootLocal);

    auto *output = static_cast<uint8_t *>(param.outputPtr);

    uint64_t ownBytes = SliceBytes(param.count, logical, processed, maxSliceCount);
    uint64_t ownOffset = UserOffset(param.count, logical, processed);

    OpTask tasks[EXPECTED_RANK_SIZE - 1];
    uint32_t taskNum = 0;

    uint64_t rootBytes = SliceBytes(param.count, 0, processed, maxSliceCount);

    if (rootBytes > 0) {
        const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
        CHK_RET(CheckChannel(rootChannel));

        tasks[taskNum++] = OpTask{
            TaskKind::RECV,
            rootChannel,
            nullptr,
            0,
            0,
            output + UserOffset(param.count, 0, processed),
            SlotOffset(0, slotStride),
            rootBytes
        };
    }

    for (uint32_t peerLogical = 1; peerLogical < RANKS_PER_SERVER; ++peerLogical) {
        if (peerLogical == logical) {
            continue;
        }

        uint64_t peerBytes = SliceBytes(param.count, peerLogical, processed, maxSliceCount);

        if (ownBytes == 0 && peerBytes == 0) {
            continue;
        }

        uint32_t peer = ActualRank(rootServer, peerLogical, rootLocal);
        const ChannelInfo *channel = FindChannel(resCtx, peer);

        CHK_RET(CheckChannel(channel));

        tasks[taskNum++] = OpTask{
            TaskKind::EXCHANGE,
            channel,
            output + ownOffset,
            SlotOffset(logical, slotStride),
            ownBytes,
            output + UserOffset(param.count, peerLogical, processed),
            SlotOffset(peerLogical, slotStride),
            peerBytes
        };
    }

    if (ownBytes > 0) {
        for (uint32_t peerLogical = 0; peerLogical < RANKS_PER_SERVER; ++peerLogical) {
            uint32_t peer = ActualRank(otherServer, peerLogical, rootLocal);
            const ChannelInfo *channel = FindChannel(resCtx, peer);

            CHK_RET(CheckChannel(channel));

            tasks[taskNum++] = OpTask{
                TaskKind::SEND,
                channel,
                output + ownOffset,
                SlotOffset(logical, slotStride),
                ownBytes,
                nullptr,
                0,
                0
            };
        }
    }

    return RunTasks(resCtx, tasks, taskNum);
}

HcclResult DirectDisseminateOtherServer(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t processed,
    uint64_t maxSliceCount,
    uint64_t slotStride)
{
    uint32_t rootServer = ServerOf(param.root);
    uint32_t rootLocal = LocalOf(param.root);

    auto *output = static_cast<uint8_t *>(param.outputPtr);

    OpTask tasks[RANKS_PER_SERVER];
    uint32_t taskNum = 0;

    for (uint32_t stripe = 0; stripe < RANKS_PER_SERVER; ++stripe) {
        uint64_t bytes = SliceBytes(param.count, stripe, processed, maxSliceCount);

        if (bytes == 0) {
            continue;
        }

        uint32_t sourceRank = ActualRank(rootServer, stripe, rootLocal);
        const ChannelInfo *channel = FindChannel(resCtx, sourceRank);

        CHK_RET(CheckChannel(channel));

        tasks[taskNum++] = OpTask{
            TaskKind::RECV,
            channel,
            nullptr,
            0,
            0,
            output + UserOffset(param.count, stripe, processed),
            SlotOffset(stripe, slotStride),
            bytes
        };
    }

    return RunTasks(resCtx, tasks, taskNum);
}

HcclResult DirectDisseminateSliceV8(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t processed,
    uint64_t maxSliceCount,
    uint64_t slotStride)
{
    uint32_t rootServer = ServerOf(param.root);
    uint32_t myServer = ServerOf(param.myRank);

    if (param.myRank == param.root) {
        return DirectDisseminateRoot(
            param,
            resCtx,
            processed,
            maxSliceCount,
            slotStride);
    }

    if (myServer == rootServer) {
        return DirectDisseminateRootServerNonRoot(
            param,
            resCtx,
            processed,
            maxSliceCount,
            slotStride);
    }

    return DirectDisseminateOtherServer(
        param,
        resCtx,
        processed,
        maxSliceCount,
        slotStride);
}

HcclResult GetCommonSlotStride(const AlgResourceCtx &resCtx, uint64_t &slotStride)
{
    uint64_t commonBufferSize = resCtx.localBuffer.size;

    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size == 0) {
            return HCCL_E_INTERNAL;
        }

        commonBufferSize = std::min(commonBufferSize, channel.remoteCclMem.size);
    }

    slotStride = AlignDown(commonBufferSize / RANKS_PER_SERVER, SLOT_ALIGN);

    CHK_PRT_RET(
        slotStride < FP32_BYTES,
        HCCL_ERROR("HCCL buffer cannot hold eight stripe slots"),
        HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult RunLargeV8(const OpParam &param, const AlgResourceCtx &resCtx)
{
    uint64_t slotStride = 0;
    CHK_RET(GetCommonSlotStride(resCtx, slotStride));

    uint64_t maxSliceCount = slotStride / FP32_BYTES;
    uint64_t maxStripeCount = StripeCount(param.count, 0);

    for (uint64_t processed = 0; processed < maxStripeCount; processed += maxSliceCount) {
        CHK_RET(ScatterSliceV8(
            param,
            resCtx,
            processed,
            maxSliceCount,
            slotStride));

        CHK_RET(DirectDisseminateSliceV8(
            param,
            resCtx,
            processed,
            maxSliceCount,
            slotStride));
    }

    return HCCL_SUCCESS;
}

} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.dataType != HCCL_DATA_TYPE_FP32 ||
        param.inputPtr == nullptr ||
        param.outputPtr == nullptr) {
        return HCCL_E_INTERNAL;
    }

    if (param.rankSize != EXPECTED_RANK_SIZE ||
        param.myRank >= param.rankSize ||
        param.root >= param.rankSize) {
        return HCCL_E_INTERNAL;
    }

    if (param.count > ~0ULL / FP32_BYTES) {
        return HCCL_E_INTERNAL;
    }

    uint64_t totalBytes = param.count * FP32_BYTES;

    if (totalBytes == 0) {
        return HCCL_SUCCESS;
    }

    if (resCtx.threads.size() < THREAD_NUM ||
        resCtx.localBuffer.addr == nullptr ||
        resCtx.localBuffer.size == 0 ||
        resCtx.channels.size() != param.rankSize - 1) {
        return HCCL_E_INTERNAL;
    }

    if (totalBytes <= SMALL_THRESHOLD) {
        return RunSmallTwoBridge(param, resCtx, 0, totalBytes);
    }

    return RunLargeV8(param, resCtx);
}

} // namespace ops_hccl
