/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "custom.h"
#include "log.h"
#include "exec_op.h"

#include <limits>

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_TRANSFER_BYTES = 256ULL * 1024 * 1024;
constexpr uint32_t DATA_ACK = 0;
constexpr uint32_t DATA_READY = SCATTER_LANES * SCATTER_WINDOW_DEPTH;
constexpr uint32_t RELAY_ACK = 2 * SCATTER_LANES * SCATTER_WINDOW_DEPTH;
constexpr uint32_t RELAY_READY = 3 * SCATTER_LANES * SCATTER_WINDOW_DEPTH;

HcclResult BatchWriteAndSignal(ThreadHandle thread, ChannelHandle channel,
    void *dst, const void *src, uint64_t bytes, uint32_t signal)
{
    HcommBatchTransferDesc descs[2]{};
    descs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
    descs[0].transferInfo.write.dst = dst;
    descs[0].transferInfo.write.src = const_cast<void *>(src);
    descs[0].transferInfo.write.len = bytes;
    descs[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    descs[1].transferInfo.notifyRecord.notifyIdx = signal;
    return static_cast<HcclResult>(HcommBatchTransferOnThread(thread, channel, descs, 2));
}

HcclResult FindChannel(const AlgResourceCtx &ctx, uint32_t rank, const ChannelInfo **channel)
{
    for (const auto &item : ctx.channels) {
        if (item.remoteRank == rank) {
            *channel = &item;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("FindChannel: channel to rank[%u] not found", rank);
    return HCCL_E_INTERNAL;
}

uint32_t WorkerIndex(uint32_t rank, uint32_t lane)
{
    return 1 + rank * SCATTER_LANES + lane;
}

uint64_t MinBytes(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

HcclResult StartWorker(const AlgResourceCtx &ctx, uint32_t index, std::vector<uint32_t> &workers)
{
    CHK_PRT_RET(index >= ctx.threads.size(),
        HCCL_ERROR("StartWorker: worker[%u] unavailable", index), HCCL_E_INTERNAL);
    workers.push_back(index);
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(ctx.threads[0], ctx.threads[index], 0)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(ctx.threads[index], 0, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult FinishWorker(const AlgResourceCtx &ctx, uint32_t index)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(ctx.threads[index], ctx.threads[0], index));
}

HcclResult ExecHalvingScatter(const OpParam &param, const AlgResourceCtx &ctx,
    uint64_t sliceBytes, uint64_t inputBytes)
{
    const ThreadHandle thread = ctx.threads[0];
    const uint32_t virtualRank = param.myRank ^ param.root;
    constexpr uint32_t masks[] = {8, 4, 2, 1};

    // Every range is kept at its final physical-rank offset in CCL memory.
    // XOR by root only permutes ranks inside each aligned power-of-two range,
    // so each transferred subgroup remains one contiguous byte interval.
    for (uint32_t mask : masks) {
        const uint32_t position = virtualRank & (2 * mask - 1);
        if (position == 0) {
            const uint32_t child = param.myRank ^ mask;
            const ChannelInfo *channel = nullptr;
            CHK_RET(FindChannel(ctx, child, &channel));

            const uint32_t virtualSendBase = virtualRank + mask;
            const uint32_t physicalSendBase =
                (virtualSendBase ^ param.root) & ~(mask - 1);
            const uint64_t offset = static_cast<uint64_t>(physicalSendBase) * sliceBytes;
            const uint64_t bytes = static_cast<uint64_t>(mask) * sliceBytes;
            CHK_PRT_RET(offset > inputBytes || bytes > inputBytes - offset ||
                    offset > channel->remoteCclMem.size || bytes > channel->remoteCclMem.size - offset,
                HCCL_ERROR("ExecHalvingScatter: range overflow"), HCCL_E_MEMORY);

            const auto *base = param.myRank == param.root ?
                static_cast<const uint8_t *>(param.inputPtr) :
                static_cast<const uint8_t *>(ctx.localBuffer.addr);
            auto *remote = static_cast<uint8_t *>(channel->remoteCclMem.addr) + offset;
            CHK_RET(BatchWriteAndSignal(
                thread, channel->handle, remote, base + offset, bytes, DATA_READY));
        } else if (position == mask) {
            const uint32_t parent = param.myRank ^ mask;
            const ChannelInfo *channel = nullptr;
            CHK_RET(FindChannel(ctx, parent, &channel));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel->handle, DATA_READY, CUSTOM_TIMEOUT)));
        }
    }

    const auto *result = param.myRank == param.root ?
        static_cast<const uint8_t *>(param.inputPtr) + param.myRank * sliceBytes :
        static_cast<const uint8_t *>(ctx.localBuffer.addr) + param.myRank * sliceBytes;
    return static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, param.outputPtr, result, sliceBytes));
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &ctx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.rankSize == 0 || param.root >= param.rankSize || param.myRank >= param.rankSize,
        HCCL_ERROR("ExecOp: invalid root[%u] for rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(ctx.threads.empty(), HCCL_ERROR("ExecOp: no AICPU thread"), HCCL_E_INTERNAL);
    CHK_PRT_RET(param.outputPtr == nullptr, HCCL_ERROR("ExecOp: output pointer is null"), HCCL_E_PARA);
    CHK_PRT_RET(param.myRank == param.root && param.inputPtr == nullptr,
        HCCL_ERROR("ExecOp: root input pointer is null"), HCCL_E_PARA);

    auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(),
        HCCL_ERROR("ExecOp: unsupported data type[%d]", param.dataType), HCCL_E_NOT_SUPPORT);
    const uint64_t typeBytes = sizeIt->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeBytes,
        HCCL_ERROR("ExecOp: receive size overflow"), HCCL_E_PARA);
    const uint64_t sliceBytes = param.count * typeBytes;
    CHK_PRT_RET(param.rankSize > std::numeric_limits<uint64_t>::max() / sliceBytes,
        HCCL_ERROR("ExecOp: input size overflow"), HCCL_E_PARA);

    const ThreadHandle mainThread = ctx.threads[0];
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(mainThread, param.outputPtr, param.inputPtr, sliceBytes));
    }
    CHK_PRT_RET(ctx.localBuffer.addr == nullptr || ctx.localBuffer.size < typeBytes,
        HCCL_ERROR("ExecOp: local CCL buffer unavailable"), HCCL_E_MEMORY);
    CHK_PRT_RET(ctx.channels.size() != param.rankSize - 1,
        HCCL_ERROR("ExecOp: expected[%u] channels, got[%zu]", param.rankSize - 1, ctx.channels.size()),
        HCCL_E_INTERNAL);

    uint64_t bufferBytes = ctx.localBuffer.size;
    for (const auto &channel : ctx.channels) {
        bufferBytes = MinBytes(bufferBytes, channel.remoteCclMem.size);
    }

    // Bound OneShot by total input bytes, not per-rank receive bytes.
    const uint64_t inputBytes = sliceBytes * param.rankSize;
    if (param.rankSize == 2 * SCATTER_SERVER_RANKS &&
        inputBytes == SCATTER_HALVING_INPUT_BYTES && inputBytes <= bufferBytes) {
        return ExecHalvingScatter(param, ctx, sliceBytes, inputBytes);
    }
    if (inputBytes <= SCATTER_ONESHOT_BYTES && inputBytes <= bufferBytes) {
        if (param.myRank == param.root) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                mainThread, ctx.localBuffer.addr, param.inputPtr, inputBytes)));
            for (const auto &channel : ctx.channels) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    mainThread, channel.handle, DATA_READY)));
            }
            const auto *src = static_cast<const uint8_t *>(ctx.localBuffer.addr) + param.root * sliceBytes;
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                mainThread, param.outputPtr, src, sliceBytes)));
            for (const auto &channel : ctx.channels) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    mainThread, channel.handle, DATA_ACK, CUSTOM_TIMEOUT)));
            }
        } else {
            const ChannelInfo *channel = nullptr;
            CHK_RET(FindChannel(ctx, param.root, &channel));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                mainThread, channel->handle, DATA_READY, CUSTOM_TIMEOUT)));
            const auto *src = static_cast<const uint8_t *>(channel->remoteCclMem.addr) +
                param.myRank * sliceBytes;
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
                mainThread, channel->handle, param.outputPtr, src, sliceBytes)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                mainThread, channel->handle, DATA_ACK)));
        }
        return HCCL_SUCCESS;
    }

    struct Route {
        uint32_t destination;
        uint32_t producer;
        uint32_t lane;
        uint64_t begin;
        uint64_t bytes;
    };
    std::vector<Route> routes;
    const bool hybrid = param.rankSize == 2 * SCATTER_SERVER_RANKS &&
        sliceBytes >= SCATTER_HYBRID_MIN_BYTES;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.root) {
            continue;
        }
        const bool split = hybrid && rank / SCATTER_SERVER_RANKS != param.root / SCATTER_SERVER_RANKS &&
            rank % SCATTER_SERVER_RANKS != param.root % SCATTER_SERVER_RANKS;
        uint64_t relayBytes = split ?
            (param.count / SCATTER_ROUTE_DENOMINATOR * SCATTER_RELAY_NUMERATOR) * typeBytes : 0;
        routes.push_back({rank, param.root, 0, 0, sliceBytes - relayBytes});
        if (relayBytes != 0) {
            uint32_t producer = param.root / SCATTER_SERVER_RANKS * SCATTER_SERVER_RANKS +
                rank % SCATTER_SERVER_RANKS;
            routes.push_back({rank, producer, 1, sliceBytes - relayBytes, relayBytes});
        }
    }
    // Compact allocation: one pair of slots per active route (22 routes for
    // hybrid 16-rank Scatter). Memory capacity, not a nominal tile target, wins.
    const uint64_t slotCount = routes.size() * SCATTER_WINDOW_DEPTH;
    const uint64_t capacity = MinBytes(bufferBytes / slotCount, SCATTER_CHUNK_BYTES) /
        typeBytes * typeBytes;
    CHK_PRT_RET(capacity == 0 || ctx.threads.size() < 1ULL + param.rankSize * SCATTER_LANES,
        HCCL_ERROR("ExecOp: insufficient window resources"), HCCL_E_MEMORY);
    std::vector<uint32_t> workers;
    for (size_t routeIndex = 0; routeIndex < routes.size(); ++routeIndex) {
        const Route &route = routes[routeIndex];
        const bool root = param.myRank == param.root;
        const bool receiver = param.myRank == route.destination;
        const bool relay = route.producer != param.root && param.myRank == route.producer;
        if (!root && !receiver && !relay) {
            continue;
        }
        // Balance the final tile instead of leaving a tiny trailing task.
        const uint64_t elements = route.bytes / typeBytes;
        const uint64_t tiles = route.bytes / capacity + (route.bytes % capacity != 0);
        const uint64_t tileBytes = (elements / tiles + (elements % tiles != 0)) * typeBytes;
        const uint32_t index = WorkerIndex(route.destination, route.lane);
        CHK_RET(StartWorker(ctx, index, workers));
        const ThreadHandle worker = ctx.threads[index];
        const ChannelInfo *channel = nullptr;
        const ChannelInfo *downstream = nullptr;
        CHK_RET(FindChannel(ctx, root ? (route.producer == param.root ? route.destination : route.producer) :
            (receiver ? route.producer : param.root), &channel));
        if (relay) {
            CHK_RET(FindChannel(ctx, route.destination, &downstream));
        }
        const bool rootToRelay = root && route.producer != param.root;
        const uint32_t readyBase = rootToRelay || relay ? RELAY_READY : DATA_READY;
        const uint32_t ackBase = rootToRelay || relay ? RELAY_ACK : DATA_ACK;
        for (uint64_t tile = 0; tile < tiles; ++tile) {
            const uint32_t window = tile % SCATTER_WINDOW_DEPTH;
            const uint32_t signal = route.lane * SCATTER_WINDOW_DEPTH + window;
            const uint64_t slot = (routeIndex * SCATTER_WINDOW_DEPTH + window) * capacity;
            const uint64_t offset = tile * tileBytes;
            const uint64_t bytes = MinBytes(route.bytes - offset, tileBytes);
            if (root) {
                // First fill both slots; wait only before reusing the same slot.
                if (tile >= SCATTER_WINDOW_DEPTH) {
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                        worker, channel->handle, ackBase + signal, CUSTOM_TIMEOUT)));
                }
                auto *staging = static_cast<uint8_t *>(ctx.localBuffer.addr) + slot;
                const auto *src = static_cast<const uint8_t *>(param.inputPtr) +
                    route.destination * sliceBytes + route.begin + offset;
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(worker, staging, src, bytes)));
                if (rootToRelay) {
                    auto *dst = static_cast<uint8_t *>(channel->remoteCclMem.addr) + slot;
                    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                        worker, channel->handle, dst, staging, bytes)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(worker, channel->handle)));
                }
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    worker, channel->handle, readyBase + signal)));
            } else if (receiver) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    worker, channel->handle, DATA_READY + signal, CUSTOM_TIMEOUT)));
                auto *dst = static_cast<uint8_t *>(param.outputPtr) + route.begin + offset;
                const auto *src = static_cast<const uint8_t *>(channel->remoteCclMem.addr) + slot;
                CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
                    worker, channel->handle, dst, src, bytes)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    worker, channel->handle, DATA_ACK + signal)));
            } else {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    worker, channel->handle, RELAY_READY + signal, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    worker, downstream->handle, DATA_READY + signal)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    worker, downstream->handle, DATA_ACK + signal, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    worker, channel->handle, RELAY_ACK + signal)));
            }
        }
        if (root) {
            // Drain the last credits even if there is no next tile/invocation.
            const uint64_t first = tiles > SCATTER_WINDOW_DEPTH ? tiles - SCATTER_WINDOW_DEPTH : 0;
            for (uint64_t tile = first; tile < tiles; ++tile) {
                const uint32_t signal = route.lane * SCATTER_WINDOW_DEPTH + tile % SCATTER_WINDOW_DEPTH;
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    worker, channel->handle, ackBase + signal, CUSTOM_TIMEOUT)));
            }
        }
        CHK_RET(FinishWorker(ctx, index));
    }
    if (param.myRank == param.root) {
        for (uint64_t offset = 0; offset < sliceBytes;) {
            const uint64_t bytes = MinBytes(sliceBytes - offset, MAX_TRANSFER_BYTES);
            const auto *src = static_cast<const uint8_t *>(param.inputPtr) + param.root * sliceBytes + offset;
            auto *dst = static_cast<uint8_t *>(param.outputPtr) + offset;
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dst, src, bytes)));
            offset += bytes;
        }
    }
    for (uint32_t index : workers) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(mainThread, index, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
