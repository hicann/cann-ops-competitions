/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_CHUNK_BYTES = 256ULL * 1024 * 1024;
constexpr uint32_t SERVER_RANKS = 8;
constexpr uint32_t SERVER_NUM = 2;
constexpr uint32_t BIRS_PAIR_SIZE = 2;
constexpr uint32_t BIRS_PAIR_NUM = SERVER_RANKS / BIRS_PAIR_SIZE;
constexpr uint32_t BIRS_SCRATCH_SLOTS = 24;
// Phase one can accumulate in its first receive slot, so the cross-server
// receive uses slot 7 and no ninth chunk-sized staging slot is needed.
constexpr uint32_t DIRECT_SCRATCH_SLOTS = 8;
constexpr uint32_t DIRECT_WORKER_NUM = SERVER_RANKS - 1;
constexpr uint64_t DIRECT_THRESHOLD_BYTES = 4ULL * 1024 * 1024;
constexpr uint64_t RHD_SMALL_OUTPUT_BYTES = 32ULL * 1024;
constexpr uint32_t HALVING_SCRATCH_SLOTS = SERVER_RANKS + 1;
constexpr bool USE_HIERARCHICAL_HALVING = false;
// Clos-first packs one full chunk from each of the eight local blocks into
// contiguous CCL slots. It is deliberately used only when all blocks fit in
// one chunk; a split input has a full-output-block stride between chunks.
constexpr uint32_t CLOS_FIRST_SCRATCH_SLOTS = SERVER_RANKS * 2 - 1;
constexpr bool USE_CLOS_FIRST_DIRECT = true;
constexpr bool USE_CHUNKED_CLOS_FIRST_DIRECT = true;

HcclResult FindChannel(const std::unordered_map<uint32_t, ChannelInfo> &channels, uint32_t peer,
    const ChannelInfo **channel)
{
    const auto it = channels.find(peer);
    CHK_PRT_RET(it == channels.end(), HCCL_ERROR("channel to rank[%u] not found", peer), HCCL_E_INTERNAL);
    *channel = &it->second;
    return HCCL_SUCCESS;
}

HcclResult ExecHierarchicalDirect(const OpParam &param, const AlgResourceCtx &resource,
    const std::unordered_map<uint32_t, ChannelInfo> &channels, uint64_t elementSize)
{
    CHK_PRT_RET(resource.threads.size() < DIRECT_WORKER_NUM + 1,
        HCCL_ERROR("hierarchical direct path requires eight AICPU threads"), HCCL_E_INTERNAL);

    const uint64_t outputBytes = param.count * elementSize;
    const uint64_t maxChunkCount = std::min(MAX_CHUNK_BYTES,
        resource.localBuffer.size / DIRECT_SCRATCH_SLOTS) / elementSize;
    CHK_PRT_RET(maxChunkCount == 0, HCCL_ERROR("CCL buffer is too small for direct path"), HCCL_E_INTERNAL);
    const uint64_t chunkNum = 1 + (param.count - 1) / maxChunkCount;
    const uint64_t chunkCount = param.count / chunkNum + (param.count % chunkNum != 0);

    const ThreadHandle mainThread = resource.aicpuThread;
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCcl = static_cast<uint8_t *>(resource.localBuffer.addr);
    const uint32_t serverBegin = (param.myRank / SERVER_RANKS) * SERVER_RANKS;
    const uint32_t myLocal = param.myRank % SERVER_RANKS;
    const uint32_t crossPeer = (param.myRank + SERVER_RANKS) % param.rankSize;
    const ChannelInfo *crossChannel = nullptr;
    CHK_RET(FindChannel(channels, crossPeer, &crossChannel));

    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t count = std::min(chunkCount, param.count - offset);
        const uint64_t bytes = count * elementSize;
        const uint64_t offsetBytes = offset * elementSize;
        auto Scratch = [&](uint32_t slot) -> uint8_t * {
            return localCcl + static_cast<uint64_t>(slot) * bytes;
        };
        auto RemoteScratch = [&](const ChannelInfo *channel, uint32_t slot) -> uint8_t * {
            return static_cast<uint8_t *>(channel->remoteCclMem.addr) + static_cast<uint64_t>(slot) * bytes;
        };
        auto SlotForSource = [](uint32_t sourceLocal, uint32_t destinationLocal) -> uint32_t {
            return sourceLocal < destinationLocal ? sourceLocal : sourceLocal - 1;
        };

        // Each phase reduces eight output blocks within both servers. The
        // seven local peers use different worker streams and different Mesh
        // links, so all direct transfers can progress concurrently.
        for (uint32_t phase = 0; phase < SERVER_NUM; ++phase) {
            for (uint32_t worker = 1; worker <= DIRECT_WORKER_NUM; ++worker) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(mainThread,
                    resource.threads[worker], phase)));
            }
            for (uint32_t worker = 1; worker <= DIRECT_WORKER_NUM; ++worker) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[worker], phase,
                    CUSTOM_TIMEOUT)));
            }
            for (uint32_t destinationLocal = 0; destinationLocal < SERVER_RANKS; ++destinationLocal) {
                if (destinationLocal == myLocal) {
                    continue;
                }
                const uint32_t destination = serverBegin + destinationLocal;
                const ChannelInfo *channel = nullptr;
                CHK_RET(FindChannel(channels, destination, &channel));
                const uint32_t workerIndex = SlotForSource(destinationLocal, myLocal);
                const uint32_t sourceSlot = SlotForSource(myLocal, destinationLocal);
                // Phase zero produces this server's own output block. Phase
                // one produces the matching block on the other server, which
                // is later exchanged with crossPeer.
                const uint32_t targetBlock = phase == 0 ? destination :
                    (destination + SERVER_RANKS) % param.rankSize;
                uint8_t *source = input + static_cast<uint64_t>(targetBlock) * outputBytes + offsetBytes;
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(resource.threads[workerIndex + 1],
                    channel->handle, RemoteScratch(channel, sourceSlot), source, bytes)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.threads[workerIndex + 1],
                    channel->handle, NOTIFY_IDX_DATA_SIGNAL)));
            }

            const uint32_t block = phase == 0 ? param.myRank : crossPeer;
            uint8_t *selfInput = input + static_cast<uint64_t>(block) * outputBytes + offsetBytes;
            uint8_t *accumulator = phase == 0 ? output + offsetBytes : Scratch(0);
            if (phase == 0) {
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, accumulator, selfInput, bytes)));
            }

            // Do not add a worker-completion barrier here. The channel DATA
            // wait already orders this reduction after its producer's write.
            // Reducing and acknowledging each source immediately turns the
            // per-peer full-duplex links into a safe phase-to-phase pipeline.
            bool phaseOneSeeded = false;
            for (uint32_t sourceLocal = 0; sourceLocal < SERVER_RANKS; ++sourceLocal) {
                if (sourceLocal == myLocal) {
                    continue;
                }
                const ChannelInfo *channel = nullptr;
                CHK_RET(FindChannel(channels, serverBegin + sourceLocal, &channel));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, channel->handle,
                    NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                const uint32_t sourceSlot = SlotForSource(sourceLocal, myLocal);
                if (phase == 1 && !phaseOneSeeded) {
                    // Slot 0 is the first consumed receive slot on every
                    // rank. Seed it with this rank's local input rather than
                    // copying that input into a separate accumulator slot.
                    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, Scratch(sourceSlot), selfInput,
                        count, static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
                    phaseOneSeeded = true;
                } else {
                    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, accumulator, Scratch(sourceSlot),
                        count, static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
                }
                if (phase == 0) {
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, channel->handle,
                        NOTIFY_IDX_ACK)));
                }
            }

            // Only phase zero reuses the receive slots.  After phase one, the
            // cross-server handshake orders both ranks before the next chunk,
            // so a second per-peer ACK round is redundant.
            if (phase == 0) {
                for (uint32_t destinationLocal = 0; destinationLocal < SERVER_RANKS; ++destinationLocal) {
                    if (destinationLocal == myLocal) {
                        continue;
                    }
                    const ChannelInfo *channel = nullptr;
                    CHK_RET(FindChannel(channels, serverBegin + destinationLocal, &channel));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, channel->handle,
                        NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                }
            }
        }

        // Slot 7 is first used by this chunk's cross-server write.  A
        // readiness ACK is needed only when a previous chunk occupied it.
        if (offset != 0) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, crossChannel->handle,
                NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        }
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(mainThread, crossChannel->handle,
            RemoteScratch(crossChannel, SERVER_RANKS - 1), Scratch(0), bytes)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, crossChannel->handle,
            NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, crossChannel->handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, output + offsetBytes,
            Scratch(SERVER_RANKS - 1), count, static_cast<HcommDataType>(param.dataType),
            static_cast<HcommReduceOp>(param.reduceType))));
        if (offset + count < param.count) {
            // Tell the peer that its previous cross-server write has been
            // consumed before it reuses this slot in the next chunk.
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, crossChannel->handle,
                NOTIFY_IDX_ACK)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecHierarchicalHalving(const OpParam &param, const AlgResourceCtx &resource,
    const std::unordered_map<uint32_t, ChannelInfo> &channels, uint64_t elementSize)
{
    const uint64_t outputBytes = param.count * elementSize;
    const uint64_t maxChunkCount = std::min(MAX_CHUNK_BYTES,
        resource.localBuffer.size / HALVING_SCRATCH_SLOTS) / elementSize;
    CHK_PRT_RET(maxChunkCount == 0, HCCL_ERROR("CCL buffer is too small for hierarchical halving"), HCCL_E_INTERNAL);
    const uint64_t chunkNum = 1 + (param.count - 1) / maxChunkCount;
    const uint64_t chunkCount = param.count / chunkNum + (param.count % chunkNum != 0);

    const ThreadHandle thread = resource.aicpuThread;
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCcl = static_cast<uint8_t *>(resource.localBuffer.addr);
    const uint32_t serverBegin = (param.myRank / SERVER_RANKS) * SERVER_RANKS;
    const uint32_t myLocal = param.myRank % SERVER_RANKS;
    const uint32_t crossPeer = (param.myRank + SERVER_RANKS) % param.rankSize;
    const ChannelInfo *crossChannel = nullptr;
    CHK_RET(FindChannel(channels, crossPeer, &crossChannel));

    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t count = std::min(chunkCount, param.count - offset);
        const uint64_t bytes = count * elementSize;
        const uint64_t offsetBytes = offset * elementSize;
        auto Scratch = [&](uint32_t slot) -> uint8_t * {
            return localCcl + static_cast<uint64_t>(slot) * bytes;
        };
        auto RemoteScratch = [&](const ChannelInfo *channel, uint32_t slot) -> uint8_t * {
            return static_cast<uint8_t *>(channel->remoteCclMem.addr) + static_cast<uint64_t>(slot) * bytes;
        };

        for (uint32_t phase = 0; phase < SERVER_NUM; ++phase) {
            const uint32_t inputBlockBegin = phase == 0 ? serverBegin :
                (serverBegin + SERVER_RANKS) % param.rankSize;
            // Stage zero retains only one four-block half locally. The other
            // half is consumed directly from the input by the outgoing DMA.
            constexpr uint32_t firstHalfBlocks = SERVER_RANKS / 2;
            const uint32_t firstKeepOffset = ((myLocal >> 2U) & 1U) * firstHalfBlocks;
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, Scratch(firstKeepOffset),
                input + static_cast<uint64_t>(inputBlockBegin + firstKeepOffset) * outputBytes + offsetBytes,
                static_cast<uint64_t>(firstHalfBlocks) * bytes)));

            for (uint32_t stage = 0; stage < 3; ++stage) {
                const uint32_t bit = 2U - stage;
                const uint32_t partner = serverBegin + (myLocal ^ (1U << bit));
                const ChannelInfo *channel = nullptr;
                CHK_RET(FindChannel(channels, partner, &channel));

                const uint32_t halfBlocks = 1U << (2U - stage);
                const uint32_t groupBlocks = halfBlocks * 2U;
                const uint32_t groupStart = (myLocal >> (bit + 1U)) * groupBlocks;
                const uint32_t keepOffset = ((myLocal >> bit) & 1U) * halfBlocks;
                const uint32_t sendOffset = keepOffset == 0 ? halfBlocks : 0;
                const uint64_t transferCount = static_cast<uint64_t>(halfBlocks) * count;
                uint8_t *send = stage == 0 ?
                    input + static_cast<uint64_t>(inputBlockBegin + groupStart + sendOffset) * outputBytes + offsetBytes :
                    Scratch(groupStart + sendOffset);

                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel->handle,
                    NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, channel->handle,
                    NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, channel->handle,
                    RemoteScratch(channel, groupStart + sendOffset), send, transferCount,
                    static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel->handle,
                    NOTIFY_IDX_DATA_SIGNAL)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, channel->handle,
                    NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }

            if (phase == 0) {
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, output + offsetBytes,
                    Scratch(myLocal), bytes)));
            }
        }

        if (offset != 0) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, crossChannel->handle,
                NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        }
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, crossChannel->handle,
            RemoteScratch(crossChannel, SERVER_RANKS), Scratch(myLocal), bytes)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, crossChannel->handle,
            NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, crossChannel->handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, output + offsetBytes,
            Scratch(SERVER_RANKS), count, static_cast<HcommDataType>(param.dataType),
            static_cast<HcommReduceOp>(param.reduceType))));
        if (offset + count < param.count) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, crossChannel->handle,
                NOTIFY_IDX_ACK)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecClosFirstDirect(const OpParam &param, const AlgResourceCtx &resource,
    const std::unordered_map<uint32_t, ChannelInfo> &channels, uint64_t elementSize)
{
    CHK_PRT_RET(resource.threads.size() < DIRECT_WORKER_NUM + 1,
        HCCL_ERROR("Clos-first direct path requires eight AICPU threads"), HCCL_E_INTERNAL);

    const uint64_t outputBytes = param.count * elementSize;
    const uint64_t maxChunkCount = std::min(MAX_CHUNK_BYTES,
        resource.localBuffer.size / CLOS_FIRST_SCRATCH_SLOTS) / elementSize;
    CHK_PRT_RET(maxChunkCount == 0, HCCL_ERROR("CCL buffer is too small for Clos-first direct"), HCCL_E_INTERNAL);
    const uint64_t chunkNum = 1 + (param.count - 1) / maxChunkCount;
    const uint64_t chunkCount = param.count / chunkNum + (param.count % chunkNum != 0);

    const ThreadHandle mainThread = resource.aicpuThread;
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCcl = static_cast<uint8_t *>(resource.localBuffer.addr);
    const uint32_t serverBegin = (param.myRank / SERVER_RANKS) * SERVER_RANKS;
    const uint32_t myLocal = param.myRank % SERVER_RANKS;
    const uint32_t crossPeer = (param.myRank + SERVER_RANKS) % param.rankSize;
    const ChannelInfo *crossChannel = nullptr;
    CHK_RET(FindChannel(channels, crossPeer, &crossChannel));

    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t count = std::min(chunkCount, param.count - offset);
        const uint64_t bytes = count * elementSize;
        const uint64_t offsetBytes = offset * elementSize;
        auto Scratch = [&](uint32_t slot) -> uint8_t * {
            return localCcl + static_cast<uint64_t>(slot) * bytes;
        };
        auto RemoteScratch = [&](const ChannelInfo *channel, uint32_t slot) -> uint8_t * {
            return static_cast<uint8_t *>(channel->remoteCclMem.addr) + static_cast<uint64_t>(slot) * bytes;
        };
        auto SlotForSource = [](uint32_t sourceLocal, uint32_t destinationLocal) -> uint32_t {
            return sourceLocal < destinationLocal ? sourceLocal : sourceLocal - 1;
        };

        // Clos is eight times a single HCCS link. Pair the two matching ranks
        // across servers first, then reduce only eight paired partials locally.
        const uint32_t remoteServerBegin = (serverBegin + SERVER_RANKS) % param.rankSize;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, Scratch(0),
            input + static_cast<uint64_t>(serverBegin) * outputBytes + offsetBytes,
            SERVER_RANKS * bytes)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, crossChannel->handle,
            NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, crossChannel->handle,
            NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(mainThread, crossChannel->handle,
            RemoteScratch(crossChannel, 0),
            input + static_cast<uint64_t>(remoteServerBegin) * outputBytes + offsetBytes,
            SERVER_RANKS * count, static_cast<HcommDataType>(param.dataType),
            static_cast<HcommReduceOp>(param.reduceType))));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, crossChannel->handle,
            NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, crossChannel->handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

        for (uint32_t worker = 1; worker <= DIRECT_WORKER_NUM; ++worker) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(mainThread,
                resource.threads[worker], 0)));
        }
        for (uint32_t worker = 1; worker <= DIRECT_WORKER_NUM; ++worker) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[worker], 0,
                CUSTOM_TIMEOUT)));
        }
        for (uint32_t destinationLocal = 0; destinationLocal < SERVER_RANKS; ++destinationLocal) {
            if (destinationLocal == myLocal) {
                continue;
            }
            const uint32_t destination = serverBegin + destinationLocal;
            const ChannelInfo *channel = nullptr;
            CHK_RET(FindChannel(channels, destination, &channel));
            const uint32_t workerIndex = SlotForSource(destinationLocal, myLocal);
            const uint32_t sourceSlot = SlotForSource(myLocal, destinationLocal);
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(resource.threads[workerIndex + 1], channel->handle,
                RemoteScratch(channel, SERVER_RANKS + sourceSlot), Scratch(destinationLocal), bytes)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.threads[workerIndex + 1],
                channel->handle, NOTIFY_IDX_DATA_SIGNAL)));
        }

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, output + offsetBytes,
            Scratch(myLocal), bytes)));
        for (uint32_t sourceLocal = 0; sourceLocal < SERVER_RANKS; ++sourceLocal) {
            if (sourceLocal == myLocal) {
                continue;
            }
            const ChannelInfo *channel = nullptr;
            CHK_RET(FindChannel(channels, serverBegin + sourceLocal, &channel));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, channel->handle,
                NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            const uint32_t sourceSlot = SlotForSource(sourceLocal, myLocal);
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, output + offsetBytes,
                Scratch(SERVER_RANKS + sourceSlot), count, static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
            if (offset + count < param.count) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, channel->handle,
                    NOTIFY_IDX_ACK)));
            }
        }
        if (offset + count < param.count) {
            for (uint32_t destinationLocal = 0; destinationLocal < SERVER_RANKS; ++destinationLocal) {
                if (destinationLocal == myLocal) {
                    continue;
                }
                const ChannelInfo *channel = nullptr;
                CHK_RET(FindChannel(channels, serverBegin + destinationLocal, &channel));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, channel->handle,
                    NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecChunkedClosFirstDirect(const OpParam &param, const AlgResourceCtx &resource,
    const std::unordered_map<uint32_t, ChannelInfo> &channels, uint64_t elementSize)
{
    CHK_PRT_RET(resource.threads.size() < DIRECT_WORKER_NUM + 1,
        HCCL_ERROR("chunked Clos-first path requires eight AICPU threads"), HCCL_E_INTERNAL);

    const uint64_t outputBytes = param.count * elementSize;
    const uint64_t maxChunkCount = std::min(MAX_CHUNK_BYTES,
        resource.localBuffer.size / CLOS_FIRST_SCRATCH_SLOTS) / elementSize;
    CHK_PRT_RET(maxChunkCount == 0, HCCL_ERROR("CCL buffer is too small for chunked Clos-first"), HCCL_E_INTERNAL);
    const uint64_t chunkNum = 1 + (param.count - 1) / maxChunkCount;
    const uint64_t chunkCount = param.count / chunkNum + (param.count % chunkNum != 0);

    const ThreadHandle mainThread = resource.aicpuThread;
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCcl = static_cast<uint8_t *>(resource.localBuffer.addr);
    const uint32_t serverBegin = (param.myRank / SERVER_RANKS) * SERVER_RANKS;
    const uint32_t myLocal = param.myRank % SERVER_RANKS;
    const uint32_t crossPeer = (param.myRank + SERVER_RANKS) % param.rankSize;
    const ChannelInfo *crossChannel = nullptr;
    CHK_RET(FindChannel(channels, crossPeer, &crossChannel));

    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t count = std::min(chunkCount, param.count - offset);
        const uint64_t bytes = count * elementSize;
        const uint64_t offsetBytes = offset * elementSize;
        auto Scratch = [&](uint32_t slot) -> uint8_t * {
            return localCcl + static_cast<uint64_t>(slot) * bytes;
        };
        auto RemoteScratch = [&](const ChannelInfo *channel, uint32_t slot) -> uint8_t * {
            return static_cast<uint8_t *>(channel->remoteCclMem.addr) + static_cast<uint64_t>(slot) * bytes;
        };
        auto SlotForSource = [](uint32_t sourceLocal, uint32_t destinationLocal) -> uint32_t {
            return sourceLocal < destinationLocal ? sourceLocal : sourceLocal - 1;
        };

        // Input chunks are strided by full output blocks. Pack the eight local
        // blocks separately before pairing the two servers.
        for (uint32_t block = 0; block < SERVER_RANKS; ++block) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, Scratch(block),
                input + static_cast<uint64_t>(serverBegin + block) * outputBytes + offsetBytes, bytes)));
        }

        const uint32_t remoteServerBegin = (serverBegin + SERVER_RANKS) % param.rankSize;
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, crossChannel->handle,
            NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, crossChannel->handle,
            NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        for (uint32_t block = 0; block < SERVER_RANKS; ++block) {
            CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(mainThread, crossChannel->handle,
                RemoteScratch(crossChannel, block),
                input + static_cast<uint64_t>(remoteServerBegin + block) * outputBytes + offsetBytes, count,
                static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, crossChannel->handle,
            NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, crossChannel->handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

        for (uint32_t worker = 1; worker <= DIRECT_WORKER_NUM; ++worker) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(mainThread,
                resource.threads[worker], 0)));
        }
        for (uint32_t worker = 1; worker <= DIRECT_WORKER_NUM; ++worker) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[worker], 0,
                CUSTOM_TIMEOUT)));
        }
        for (uint32_t destinationLocal = 0; destinationLocal < SERVER_RANKS; ++destinationLocal) {
            if (destinationLocal == myLocal) {
                continue;
            }
            const ChannelInfo *channel = nullptr;
            CHK_RET(FindChannel(channels, serverBegin + destinationLocal, &channel));
            const uint32_t workerIndex = SlotForSource(destinationLocal, myLocal);
            const uint32_t sourceSlot = SlotForSource(myLocal, destinationLocal);
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(resource.threads[workerIndex + 1], channel->handle,
                RemoteScratch(channel, SERVER_RANKS + sourceSlot), Scratch(destinationLocal), bytes)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resource.threads[workerIndex + 1],
                channel->handle, NOTIFY_IDX_DATA_SIGNAL)));
        }

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, output + offsetBytes,
            Scratch(myLocal), bytes)));
        for (uint32_t sourceLocal = 0; sourceLocal < SERVER_RANKS; ++sourceLocal) {
            if (sourceLocal == myLocal) {
                continue;
            }
            const ChannelInfo *channel = nullptr;
            CHK_RET(FindChannel(channels, serverBegin + sourceLocal, &channel));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, channel->handle,
                NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            const uint32_t sourceSlot = SlotForSource(sourceLocal, myLocal);
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, output + offsetBytes,
                Scratch(SERVER_RANKS + sourceSlot), count, static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
            if (offset + count < param.count) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, channel->handle,
                    NOTIFY_IDX_ACK)));
            }
        }
        if (offset + count < param.count) {
            for (uint32_t destinationLocal = 0; destinationLocal < SERVER_RANKS; ++destinationLocal) {
                if (destinationLocal == myLocal) {
                    continue;
                }
                const ChannelInfo *channel = nullptr;
                CHK_RET(FindChannel(channels, serverBegin + destinationLocal, &channel));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, channel->handle,
                    NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecSmallRecursiveHalving(const OpParam &param, const AlgResourceCtx &resource,
    const std::unordered_map<uint32_t, ChannelInfo> &channels, uint64_t elementSize)
{
    const uint64_t outputBytes = param.count * elementSize;
    CHK_PRT_RET(resource.localBuffer.size < param.rankSize * outputBytes,
        HCCL_ERROR("CCL buffer is too small for small RHD"), HCCL_E_INTERNAL);

    const ThreadHandle thread = resource.aicpuThread;
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCcl = static_cast<uint8_t *>(resource.localBuffer.addr);
    auto Scratch = [&](uint32_t slot) -> uint8_t * {
        return localCcl + static_cast<uint64_t>(slot) * outputBytes;
    };
    // The first halving round retains only eight blocks.  Preload that half
    // and source the one-shot outgoing half directly from the input buffer.
    constexpr uint32_t INITIAL_HALF_BLOCKS = SERVER_RANKS;
    const uint32_t initialKeepOffset = ((param.myRank >> 3U) & 1U) * INITIAL_HALF_BLOCKS;
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, Scratch(initialKeepOffset),
        input + static_cast<uint64_t>(initialKeepOffset) * outputBytes,
        static_cast<uint64_t>(INITIAL_HALF_BLOCKS) * outputBytes)));

    for (uint32_t stage = 0; stage < 4; ++stage) {
        const uint32_t bit = 3U - stage;
        const uint32_t partner = param.myRank ^ (1U << bit);
        const ChannelInfo *channel = nullptr;
        CHK_RET(FindChannel(channels, partner, &channel));

        const uint32_t halfBlocks = 1U << (3U - stage);
        const uint32_t groupBlocks = halfBlocks * 2U;
        const uint32_t completedPrefix = param.myRank >> (bit + 1U);
        const uint32_t groupStart = completedPrefix * groupBlocks;
        const uint32_t keepOffset = ((param.myRank >> bit) & 1U) * halfBlocks;
        const uint32_t sendOffset = keepOffset == 0 ? halfBlocks : 0;
        const uint64_t transferCount = static_cast<uint64_t>(halfBlocks) * param.count;
        uint8_t *send = stage == 0 ? input + static_cast<uint64_t>(sendOffset) * outputBytes :
            Scratch(groupStart + sendOffset);
        uint8_t *remote = static_cast<uint8_t *>(channel->remoteCclMem.addr) +
            static_cast<uint64_t>(groupStart + sendOffset) * outputBytes;

        // Before overwriting the peer's discarded half, wait until it has
        // completed its preload (stage zero) or consumed the previous stage.
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel->handle,
            NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, channel->handle,
            NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, channel->handle, remote, send,
            transferCount, static_cast<HcommDataType>(param.dataType),
            static_cast<HcommReduceOp>(param.reduceType))));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel->handle,
            NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, channel->handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, output,
        Scratch(param.myRank), outputBytes)));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resource)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    const auto typeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(), HCCL_ERROR("unsupported data type"), HCCL_E_PARA);
    const uint64_t elementSize = typeIt->second;
    const uint64_t outputBytes = param.count * elementSize;
    const ThreadHandle thread = resource.aicpuThread;

    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(thread, param.outputPtr, param.inputPtr, outputBytes));
    }

    std::unordered_map<uint32_t, ChannelInfo> channels;
    for (const auto &channel : resource.channels) {
        channels.emplace(channel.remoteRank, channel);
    }


    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCcl = static_cast<uint8_t *>(resource.localBuffer.addr);

    if (param.rankSize == SERVER_NUM * SERVER_RANKS && outputBytes >= DIRECT_THRESHOLD_BYTES) {
        const uint64_t closFirstMaxBytes = std::min(MAX_CHUNK_BYTES,
            resource.localBuffer.size / CLOS_FIRST_SCRATCH_SLOTS);
        if (USE_CLOS_FIRST_DIRECT && outputBytes <= closFirstMaxBytes) {
            return ExecClosFirstDirect(param, resource, channels, elementSize);
        }
        if (USE_CHUNKED_CLOS_FIRST_DIRECT && outputBytes <= closFirstMaxBytes * 2) {
            return ExecChunkedClosFirstDirect(param, resource, channels, elementSize);
        }
        return USE_HIERARCHICAL_HALVING ?
            ExecHierarchicalHalving(param, resource, channels, elementSize) :
            ExecHierarchicalDirect(param, resource, channels, elementSize);
    }

    if (param.rankSize == SERVER_NUM * SERVER_RANKS && outputBytes == RHD_SMALL_OUTPUT_BYTES) {
        return ExecSmallRecursiveHalving(param, resource, channels, elementSize);
    }

    // For small outputs, reserve one private source slot per rank. All writes
    // are posted before waits, then the destination reduces in source-rank order.
    // For the fixed 2x8 topology, use the threaded BIRS schedule for all
    // non-trivial messages.  The old direct path serializes 15 peer
    // transfers and is particularly costly at 512KB.  Keep the one-element
    // case on the simple path to avoid paying the BIRS setup cost for the
    // smallest functional test.
    if (param.rankSize != SERVER_NUM * SERVER_RANKS || outputBytes <= elementSize ||
        outputBytes > resource.localBuffer.size) {
        const uint64_t slotBytes = resource.localBuffer.size / param.rankSize;
        const uint64_t chunkCount = std::min(MAX_CHUNK_BYTES, slotBytes) / elementSize;
        CHK_PRT_RET(chunkCount == 0, HCCL_ERROR("CCL buffer slot is too small"), HCCL_E_INTERNAL);

        for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
            const uint64_t count = std::min(chunkCount, param.count - offset);
            const uint64_t bytes = count * elementSize;
            const uint64_t offsetBytes = offset * elementSize;
            uint8_t *selfInput = input + static_cast<uint64_t>(param.myRank) * outputBytes + offsetBytes;
            uint8_t *selfSlot = localCcl + static_cast<uint64_t>(param.myRank) * slotBytes;
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, selfSlot, selfInput, bytes)));

            for (uint32_t destination = 0; destination < param.rankSize; ++destination) {
                if (destination == param.myRank) {
                    continue;
                }
                const ChannelInfo *channel = nullptr;
                CHK_RET(FindChannel(channels, destination, &channel));
                uint8_t *source = input + static_cast<uint64_t>(destination) * outputBytes + offsetBytes;
                uint8_t *remoteSlot = static_cast<uint8_t *>(channel->remoteCclMem.addr) +
                    static_cast<uint64_t>(param.myRank) * slotBytes;
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel->handle,
                    remoteSlot, source, bytes)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel->handle,
                    NOTIFY_IDX_DATA_SIGNAL)));
            }
            for (uint32_t sourceRank = 0; sourceRank < param.rankSize; ++sourceRank) {
                if (sourceRank == param.myRank) {
                    continue;
                }
                const ChannelInfo *channel = nullptr;
                CHK_RET(FindChannel(channels, sourceRank, &channel));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, channel->handle,
                    NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }

            uint8_t *outputChunk = output + offsetBytes;
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, outputChunk, localCcl, bytes)));
            for (uint32_t sourceRank = 1; sourceRank < param.rankSize; ++sourceRank) {
                uint8_t *sourceSlot = localCcl + static_cast<uint64_t>(sourceRank) * slotBytes;
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, outputChunk, sourceSlot, count,
                    static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
            }
            if (offset + count < param.count) {
                for (uint32_t sourceRank = 0; sourceRank < param.rankSize; ++sourceRank) {
                    if (sourceRank == param.myRank) {
                        continue;
                    }
                    const ChannelInfo *channel = nullptr;
                    CHK_RET(FindChannel(channels, sourceRank, &channel));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel->handle,
                        NOTIFY_IDX_ACK)));
                }
                for (uint32_t destination = 0; destination < param.rankSize; ++destination) {
                    if (destination == param.myRank) {
                        continue;
                    }
                    const ChannelInfo *channel = nullptr;
                    CHK_RET(FindChannel(channels, destination, &channel));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, channel->handle,
                        NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                }
            }
        }
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(param.rankSize != SERVER_NUM * SERVER_RANKS, HCCL_ERROR("unsupported rank size"),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(resource.threads.size() < 3, HCCL_ERROR("BIRS requires three AICPU threads"), HCCL_E_INTERNAL);

    const ThreadHandle hccsThread = resource.threads[1];
    const ThreadHandle copyThread = resource.threads[2];
    const uint64_t maxChunkCount =
        std::min(MAX_CHUNK_BYTES, resource.localBuffer.size / BIRS_SCRATCH_SLOTS) / elementSize;
    CHK_PRT_RET(maxChunkCount == 0, HCCL_ERROR("CCL buffer is too small for BIRS"), HCCL_E_INTERNAL);
    const uint64_t chunkNum = 1 + (param.count - 1) / maxChunkCount;
    const uint64_t chunkCount = param.count / chunkNum + (param.count % chunkNum != 0);

    const uint32_t serverIndex = param.myRank / SERVER_RANKS;
    const uint32_t myLocal = param.myRank % SERVER_RANKS;
    const uint32_t parity = myLocal % BIRS_PAIR_SIZE;
    const uint32_t pairIndex = myLocal / BIRS_PAIR_SIZE;
    const uint32_t sioPeer = param.myRank ^ 1U;
    const uint32_t crossPeer = (param.myRank + SERVER_RANKS) % param.rankSize;
    const ChannelInfo *sioChannel = nullptr;
    const ChannelInfo *crossChannel = nullptr;
    CHK_RET(FindChannel(channels, sioPeer, &sioChannel));
    CHK_RET(FindChannel(channels, crossPeer, &crossChannel));

    uint32_t hccsRanks[BIRS_PAIR_NUM - 1];
    uint32_t hccsNeighbours[BIRS_PAIR_NUM - 1];
    const uint32_t serverBegin = serverIndex * SERVER_RANKS;
    for (uint32_t index = 0; index < BIRS_PAIR_NUM - 1; ++index) {
        const uint32_t localPeer = (myLocal + BIRS_PAIR_SIZE * (index + 1)) % SERVER_RANKS;
        hccsRanks[index] = serverBegin + localPeer;
        hccsNeighbours[index] = hccsRanks[index] ^ 1U;
    }

    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t count = std::min(chunkCount, param.count - offset);
        const uint64_t bytes = count * elementSize;
        const uint64_t offsetBytes = offset * elementSize;
        auto Scratch = [&](uint32_t slot) -> uint8_t * {
            return localCcl + static_cast<uint64_t>(slot) * bytes;
        };
        auto RemoteScratch = [&](const ChannelInfo *channel, uint32_t slot) -> uint8_t * {
            return static_cast<uint8_t *>(channel->remoteCclMem.addr) + static_cast<uint64_t>(slot) * bytes;
        };
        auto InputChunk = [&](uint32_t targetRank) -> uint8_t * {
            return input + static_cast<uint64_t>(targetRank) * outputBytes + offsetBytes;
        };

        // Slots 0..7 hold the parity-owned pair blocks and slots 8..15 hold
        // HCCS-produced pair blocks. Each SIO round gets a private two-slot
        // input buffer at 16..23, which removes the old double-buffer reuse.
        uint32_t slot = 0;
        for (uint32_t pair = 0; pair < BIRS_PAIR_NUM; ++pair) {
            for (uint32_t server = 0; server < SERVER_NUM; ++server) {
                const uint32_t target = 2 * pair + parity + server * SERVER_RANKS;
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, Scratch(slot++),
                    InputChunk(target), bytes)));
            }
        }
        for (uint32_t server = 0; server < SERVER_NUM; ++server) {
            const uint32_t target = (hccsNeighbours[0] % SERVER_RANKS) + server * SERVER_RANKS;
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, Scratch(16 + server),
                InputChunk(target), bytes)));
        }

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(copyThread, 0, CUSTOM_TIMEOUT)));
        for (uint32_t stage = 1; stage < BIRS_PAIR_NUM; ++stage) {
            const uint32_t nextTarget = (stage < BIRS_PAIR_NUM - 1) ? hccsNeighbours[stage] : sioPeer;
            const uint32_t nextSlot = 16 + stage * SERVER_NUM;
            for (uint32_t server = 0; server < SERVER_NUM; ++server) {
                const uint32_t target = (nextTarget % SERVER_RANKS) + server * SERVER_RANKS;
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(copyThread, Scratch(nextSlot + server),
                    InputChunk(target), bytes)));
            }
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(copyThread, thread, 2)));

        // HCCS round N consumes the SIO reduction produced in round N - 1.
        // The worker keeps this ordering, while the main stream no longer
        // waits for a completed HCCS round before it posts the next SIO round.
        for (uint32_t round = 1; round < BIRS_PAIR_NUM; ++round) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(hccsThread, round - 1,
                CUSTOM_TIMEOUT)));
            const ChannelInfo *sendChannel = nullptr;
            const ChannelInfo *recvChannel = nullptr;
            CHK_RET(FindChannel(channels, hccsRanks[round - 1], &sendChannel));
            CHK_RET(FindChannel(channels, hccsRanks[BIRS_PAIR_NUM - round - 1], &recvChannel));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(hccsThread,
                recvChannel->handle, NOTIFY_IDX_ACK)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(hccsThread,
                sendChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            const uint32_t sourcePair = (hccsRanks[round - 1] % SERVER_RANKS) / BIRS_PAIR_SIZE;
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(hccsThread, sendChannel->handle,
                RemoteScratch(sendChannel, 8 + pairIndex * SERVER_NUM), Scratch(sourcePair * SERVER_NUM),
                SERVER_NUM * bytes)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(hccsThread,
                sendChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(hccsThread,
                recvChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(hccsThread, thread, 1)));

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread, copyThread, 0)));
        for (uint32_t round = 0; round < BIRS_PAIR_NUM; ++round) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread,
                sioChannel->handle, NOTIFY_IDX_ACK)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread,
                sioChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            const uint32_t destinationPair = (round < BIRS_PAIR_NUM - 1) ?
                (hccsRanks[round] % SERVER_RANKS) / BIRS_PAIR_SIZE : pairIndex;
            const uint32_t sourceSlot = 16 + round * SERVER_NUM;
            CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, sioChannel->handle,
                RemoteScratch(sioChannel, destinationPair * SERVER_NUM), Scratch(sourceSlot),
                SERVER_NUM * count, static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread,
                sioChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread,
                sioChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

            if (round < BIRS_PAIR_NUM - 1) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread, hccsThread, round)));
                if (round == 0) {
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, 2, CUSTOM_TIMEOUT)));
                }
            }
        }

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, 1, CUSTOM_TIMEOUT)));
        uint32_t groups[BIRS_PAIR_NUM];
        for (uint32_t index = 0; index < BIRS_PAIR_NUM; ++index) {
            groups[index] = (index == pairIndex) ? index * SERVER_NUM : (BIRS_PAIR_NUM + index) * SERVER_NUM;
        }
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, Scratch(groups[0]), Scratch(groups[1]),
            SERVER_NUM * count, static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, Scratch(groups[2]), Scratch(groups[3]),
            SERVER_NUM * count, static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, Scratch(groups[0]), Scratch(groups[2]),
            SERVER_NUM * count, static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));

        // Stage this server's partial in slot 2 before the Clos exchange.
        // The official BIRS inter-server template performs this copy in
        // PreprocInterServer; omitting it would reduce stale data left in
        // slot 2 by an earlier intra-server round.
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, Scratch(2),
            Scratch(groups[0]) + static_cast<uint64_t>(serverIndex) * bytes, bytes)));

        // One Clos exchange combines the two server-local partials.  The
        // official layout uses slots 2 and 3 for this final exchange.
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread,
            crossChannel->handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread,
            crossChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        const uint32_t remoteServer = (serverIndex + 1) % SERVER_NUM;
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, crossChannel->handle,
            RemoteScratch(crossChannel, 3), Scratch(groups[0]) + static_cast<uint64_t>(remoteServer) * bytes, bytes)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread,
            crossChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread,
            crossChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, Scratch(2), Scratch(3), count,
            static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, output + offsetBytes, Scratch(2), bytes)));

    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
