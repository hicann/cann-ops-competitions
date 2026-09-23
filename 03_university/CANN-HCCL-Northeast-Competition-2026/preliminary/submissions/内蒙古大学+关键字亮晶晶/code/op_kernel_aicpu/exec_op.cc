/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

#include <algorithm>
#include <cstdint>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t COMPETITION_RANK_SIZE = 16;
constexpr uint64_t TINY_LIMIT_BYTES = 4ULL * 1024ULL;
constexpr uint64_t SMALL_LIMIT_BYTES = 1ULL * 1024ULL * 1024ULL;
constexpr uint64_t MAX_PIPELINE_CHUNK = 128ULL * 1024ULL * 1024ULL;
constexpr uint64_t ALIGN_BYTES = 4ULL * 1024ULL;
constexpr uint64_t LARGE_SPLIT_NUM = 4;
constexpr uint64_t LARGE_SPLIT_DEN = 11;
constexpr uint64_t PERF_512M_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t PERF_400M4_BYTES = 400ULL * 1024ULL * 1024ULL + 4ULL;
// V17: size-specialized traffic balance. 512 MiB already has two Clos
// submission workers, so use a moderate relay share; 400 MiB + 4 B keeps
// one Clos worker and shifts more traffic to the seven parallel relays.
constexpr uint64_t SPLIT_512_NUM = 5;
constexpr uint64_t SPLIT_512_DEN = 11;
constexpr uint64_t SPLIT_400_NUM = 1;
constexpr uint64_t SPLIT_400_DEN = 2;
constexpr uint64_t PERF_400M4_CHUNK = 100ULL * 1024ULL * 1024ULL + 4ULL;

inline uint32_t ServerBase(uint32_t rank)
{
    return (rank / RANKS_PER_SERVER) * RANKS_PER_SERVER;
}

inline uint32_t RemoteLeader(uint32_t root)
{
    return root ^ RANKS_PER_SERVER;
}

inline uint32_t DataNotify(uint32_t slot)
{
    return slot * 2;
}

inline uint32_t AckNotify(uint32_t slot)
{
    return slot * 2 + 1;
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (value == 0) {
        return 0;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

void GetSplitRatio(uint64_t blockBytes, uint64_t &num, uint64_t &den)
{
    if (blockBytes == PERF_512M_BYTES) {
        num = SPLIT_512_NUM;
        den = SPLIT_512_DEN;
        return;
    }
    if (blockBytes == PERF_400M4_BYTES) {
        num = SPLIT_400_NUM;
        den = SPLIT_400_DEN;
        return;
    }
    num = LARGE_SPLIT_NUM;
    den = LARGE_SPLIT_DEN;
}

uint64_t SplitA(uint64_t bytes, uint64_t blockBytes)
{
    uint64_t num = LARGE_SPLIT_NUM;
    uint64_t den = LARGE_SPLIT_DEN;
    GetSplitRatio(blockBytes, num, den);
    if (bytes <= ALIGN_BYTES) {
        return (bytes * num) / den;
    }
    const uint64_t raw = (bytes * num + den - 1) / den;
    return std::min(bytes, AlignUp(raw, ALIGN_BYTES));
}

struct PipelinePlan {
    uint64_t chunk = 0;
    uint64_t aCap = 0;
    uint64_t bCap = 0;
    uint64_t relayStride = 0;
    uint64_t leaderStride = 0;
    uint64_t rounds = 0;
};

bool FitsPipeline(uint64_t bufferSize, uint64_t chunk, uint64_t blockBytes)
{
    const uint64_t aCap = SplitA(chunk, blockBytes);
    const uint64_t relayNeed = 2 * (chunk + aCap);
    const uint64_t leafNeed = 2 * chunk;
    return std::max(relayNeed, leafNeed) <= bufferSize;
}

PipelinePlan MakePlan(uint64_t blockBytes, uint64_t bufferSize)
{
    const uint64_t candidates[] = {
        MAX_PIPELINE_CHUNK,
        112ULL * 1024ULL * 1024ULL,
        96ULL * 1024ULL * 1024ULL,
        80ULL * 1024ULL * 1024ULL,
        64ULL * 1024ULL * 1024ULL,
        48ULL * 1024ULL * 1024ULL,
        40ULL * 1024ULL * 1024ULL,
        32ULL * 1024ULL * 1024ULL,
        24ULL * 1024ULL * 1024ULL,
        16ULL * 1024ULL * 1024ULL,
        8ULL * 1024ULL * 1024ULL,
        4ULL * 1024ULL * 1024ULL,
    };

    uint64_t chunk = 0;
    // The 400 MiB + 4 B benchmark is badly imbalanced by the generic
    // 128/128/128/16 MiB schedule. Four ~100 MiB chunks keep the two-stage
    // Mesh/Clos pipeline busy until the end without adding another round.
    if (blockBytes == PERF_400M4_BYTES &&
        FitsPipeline(bufferSize, PERF_400M4_CHUNK, blockBytes)) {
        chunk = PERF_400M4_CHUNK;
    } else if (blockBytes <= MAX_PIPELINE_CHUNK) {
        chunk = blockBytes;
    } else {
        for (uint64_t candidate : candidates) {
            if (FitsPipeline(bufferSize, candidate, blockBytes)) {
                chunk = candidate;
                break;
            }
        }
    }
    if (chunk == 0 || !FitsPipeline(bufferSize, chunk, blockBytes)) {
        return PipelinePlan{};
    }

    PipelinePlan plan{};
    plan.chunk = chunk;
    plan.aCap = SplitA(chunk, blockBytes);
    plan.bCap = chunk - plan.aCap;
    plan.relayStride = chunk + plan.aCap;
    plan.leaderStride = plan.aCap + RANKS_PER_SERVER * plan.bCap;
    plan.rounds = (blockBytes + chunk - 1) / chunk;
    return plan;
}

inline uint8_t *BytePtr(void *base, uint64_t offset)
{
    return static_cast<uint8_t *>(base) + offset;
}

inline const uint8_t *BytePtr(const void *base, uint64_t offset)
{
    return static_cast<const uint8_t *>(base) + offset;
}

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    if (remoteRank >= SCATTER_MAX_RANK_SIZE) {
        return nullptr;
    }
    const uint32_t index = resCtx.channelIndex[remoteRank];
    if (index == 0xFFU || index >= resCtx.channelCount) {
        return nullptr;
    }
    const ChannelInfo &channel = resCtx.channels[index];
    return (channel.remoteRank == remoteRank) ? &channel : nullptr;
}

HcclResult CheckBuffer(const CommBuffer &buffer, uint64_t required, const char *where)
{
    CHK_PRT_RET(buffer.addr == nullptr || buffer.size < required,
        HCCL_ERROR("%s: buffer too small, size[%llu], required[%llu]", where,
            static_cast<unsigned long long>(buffer.size), static_cast<unsigned long long>(required)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult StartWorkers(const AlgResourceCtx &resCtx)
{
    ThreadHandle main = resCtx.threads[0];
    for (uint32_t i = 1; i < resCtx.threadCount; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[i], 0, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(main, resCtx.threads[i], 0)));
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateResource(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(!resCtx.IsValid(), HCCL_ERROR("ExecOp: invalid resource context"), HCCL_E_INTERNAL);
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize || param.root >= param.rankSize,
        HCCL_ERROR("ExecOp: invalid rank tuple my[%u], root[%u], size[%u]", param.myRank, param.root, param.rankSize),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

// ------------------------------ tiny/direct path ------------------------------
HcclResult ExecTinyRoot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes)
{
    ThreadHandle thread = resCtx.aicpuThread;
    const uint64_t totalBytes = blockBytes * param.rankSize;
    CHK_RET(CheckBuffer(resCtx.localBuffer, totalBytes, "ExecTinyRoot"));
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, resCtx.localBuffer.addr, param.inputPtr, totalBytes)));

    for (uint32_t i = 0; i < resCtx.channelCount; ++i) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, resCtx.channels[i].handle, DataNotify(0))));
    }

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, param.outputPtr,
        BytePtr(resCtx.localBuffer.addr, static_cast<uint64_t>(param.root) * blockBytes), blockBytes)));

    for (uint32_t i = 0; i < resCtx.channelCount; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, resCtx.channels[i].handle, AckNotify(0), CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecTinyRecv(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes)
{
    CHK_PRT_RET(resCtx.channelCount != 1,
        HCCL_ERROR("ExecTinyRecv: expected one root channel, got[%u]", resCtx.channelCount), HCCL_E_INTERNAL);
    const ChannelInfo &rootChannel = resCtx.channels[0];
    const uint64_t totalBytes = blockBytes * param.rankSize;
    CHK_RET(CheckBuffer(rootChannel.remoteCclMem, totalBytes, "ExecTinyRecv"));

    ThreadHandle thread = resCtx.aicpuThread;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, rootChannel.handle, DataNotify(0), CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, rootChannel.handle, param.outputPtr,
        BytePtr(rootChannel.remoteCclMem.addr, static_cast<uint64_t>(param.myRank) * blockBytes), blockBytes)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, AckNotify(0))));
    return HCCL_SUCCESS;
}

// ------------------------------ small latency path ------------------------------
HcclResult ScheduleSmallLocalServer(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes)
{
    ThreadHandle worker = resCtx.threads[1];
    ThreadHandle main = resCtx.threads[0];
    const uint32_t localBase = ServerBase(param.root);
    const uint64_t serverBytes = static_cast<uint64_t>(RANKS_PER_SERVER) * blockBytes;
    CHK_RET(CheckBuffer(resCtx.localBuffer, serverBytes, "SmallLocalServer/local"));

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(worker, resCtx.localBuffer.addr,
        BytePtr(param.inputPtr, static_cast<uint64_t>(localBase) * blockBytes), serverBytes)));

    for (uint32_t rank = localBase; rank < localBase + RANKS_PER_SERVER; ++rank) {
        if (rank == param.root) {
            continue;
        }
        const ChannelInfo *channel = FindChannel(resCtx, rank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("SmallLocalServer: missing channel to rank[%u]", rank), HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(worker, channel->handle, DataNotify(0))));
    }

    for (uint32_t rank = localBase; rank < localBase + RANKS_PER_SERVER; ++rank) {
        if (rank == param.root) {
            continue;
        }
        const ChannelInfo *channel = FindChannel(resCtx, rank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("SmallLocalServer: missing ACK channel to rank[%u]", rank), HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            worker, channel->handle, AckNotify(0), CUSTOM_TIMEOUT)));
    }

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(worker, main, 1)));
    return HCCL_SUCCESS;
}

HcclResult ExecSmallRoot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes)
{
    CHK_PRT_RET(resCtx.threadCount != 2,
        HCCL_ERROR("ExecSmallRoot: expected 2 threads, got[%u]", resCtx.threadCount), HCCL_E_INTERNAL);
    const uint64_t serverBytes = static_cast<uint64_t>(RANKS_PER_SERVER) * blockBytes;
    const uint32_t leader = RemoteLeader(param.root);
    const uint32_t remoteBase = ServerBase(leader);
    const ChannelInfo *leaderChannel = FindChannel(resCtx, leader);
    CHK_PRT_RET(leaderChannel == nullptr,
        HCCL_ERROR("ExecSmallRoot: missing remote leader[%u] channel", leader), HCCL_E_INTERNAL);
    CHK_RET(CheckBuffer(leaderChannel->remoteCclMem, serverBytes, "ExecSmallRoot/leader"));
    CHK_RET(StartWorkers(resCtx));

    // Thread 1 stages the local 4 MiB half and releases seven local pulls.
    CHK_RET(ScheduleSmallLocalServer(param, resCtx, blockBytes));

    // Main sends the entire remote 4 MiB half in one Clos transfer. The remote
    // leader then serves seven local pulls and returns one aggregated ACK.
    ThreadHandle main = resCtx.threads[0];
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(main, leaderChannel->handle,
        leaderChannel->remoteCclMem.addr,
        BytePtr(param.inputPtr, static_cast<uint64_t>(remoteBase) * blockBytes), serverBytes)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(main, leaderChannel->handle, DataNotify(0))));

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(main, param.outputPtr,
        BytePtr(param.inputPtr, static_cast<uint64_t>(param.root) * blockBytes), blockBytes)));

    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        main, leaderChannel->handle, AckNotify(0), CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(main, 1, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ExecSmallLocalLeaf(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes)
{
    CHK_PRT_RET(resCtx.channelCount != 1,
        HCCL_ERROR("ExecSmallLocalLeaf: expected one root channel, got[%u]", resCtx.channelCount), HCCL_E_INTERNAL);
    const ChannelInfo &rootChannel = resCtx.channels[0];
    const uint32_t localBase = ServerBase(param.root);
    const uint64_t serverBytes = static_cast<uint64_t>(RANKS_PER_SERVER) * blockBytes;
    CHK_RET(CheckBuffer(rootChannel.remoteCclMem, serverBytes, "ExecSmallLocalLeaf/root"));

    ThreadHandle thread = resCtx.aicpuThread;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, rootChannel.handle, DataNotify(0), CUSTOM_TIMEOUT)));
    const uint64_t offset = static_cast<uint64_t>(param.myRank - localBase) * blockBytes;
    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, rootChannel.handle, param.outputPtr,
        BytePtr(rootChannel.remoteCclMem.addr, offset), blockBytes)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, AckNotify(0))));
    return HCCL_SUCCESS;
}

HcclResult ExecSmallRemoteLeader(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes)
{
    const uint32_t leader = RemoteLeader(param.root);
    CHK_PRT_RET(param.myRank != leader || resCtx.channelCount != RANKS_PER_SERVER,
        HCCL_ERROR("ExecSmallRemoteLeader: invalid leader/channel count[%u]", resCtx.channelCount), HCCL_E_INTERNAL);
    const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
    CHK_PRT_RET(rootChannel == nullptr, HCCL_ERROR("ExecSmallRemoteLeader: missing root channel"), HCCL_E_INTERNAL);

    const uint32_t remoteBase = ServerBase(leader);
    const uint64_t serverBytes = static_cast<uint64_t>(RANKS_PER_SERVER) * blockBytes;
    CHK_RET(CheckBuffer(resCtx.localBuffer, serverBytes, "ExecSmallRemoteLeader/local"));

    ThreadHandle thread = resCtx.aicpuThread;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, rootChannel->handle, DataNotify(0), CUSTOM_TIMEOUT)));

    // Release remote leaves first so their reads overlap the leader's own copy.
    for (uint32_t rank = remoteBase; rank < remoteBase + RANKS_PER_SERVER; ++rank) {
        if (rank == leader) {
            continue;
        }
        const ChannelInfo *leaf = FindChannel(resCtx, rank);
        CHK_PRT_RET(leaf == nullptr,
            HCCL_ERROR("ExecSmallRemoteLeader: missing leaf[%u] channel", rank), HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, leaf->handle, DataNotify(0))));
    }

    const uint64_t ownOffset = static_cast<uint64_t>(leader - remoteBase) * blockBytes;
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        thread, param.outputPtr, BytePtr(resCtx.localBuffer.addr, ownOffset), blockBytes)));

    for (uint32_t rank = remoteBase; rank < remoteBase + RANKS_PER_SERVER; ++rank) {
        if (rank == leader) {
            continue;
        }
        const ChannelInfo *leaf = FindChannel(resCtx, rank);
        CHK_PRT_RET(leaf == nullptr,
            HCCL_ERROR("ExecSmallRemoteLeader: missing leaf ACK channel[%u]", rank), HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, leaf->handle, AckNotify(0), CUSTOM_TIMEOUT)));
    }

    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, rootChannel->handle, AckNotify(0))));
    return HCCL_SUCCESS;
}

HcclResult ExecSmallRemoteLeaf(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes)
{
    CHK_PRT_RET(resCtx.channelCount != 1,
        HCCL_ERROR("ExecSmallRemoteLeaf: expected one leader channel, got[%u]", resCtx.channelCount), HCCL_E_INTERNAL);
    const uint32_t leader = RemoteLeader(param.root);
    const uint32_t remoteBase = ServerBase(leader);
    const ChannelInfo &leaderChannel = resCtx.channels[0];
    const uint64_t serverBytes = static_cast<uint64_t>(RANKS_PER_SERVER) * blockBytes;
    CHK_RET(CheckBuffer(leaderChannel.remoteCclMem, serverBytes, "ExecSmallRemoteLeaf/leader"));

    ThreadHandle thread = resCtx.aicpuThread;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, leaderChannel.handle, DataNotify(0), CUSTOM_TIMEOUT)));
    const uint64_t offset = static_cast<uint64_t>(param.myRank - remoteBase) * blockBytes;
    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, leaderChannel.handle, param.outputPtr,
        BytePtr(leaderChannel.remoteCclMem.addr, offset), blockBytes)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, leaderChannel.handle, AckNotify(0))));
    return HCCL_SUCCESS;
}

HcclResult ExecSmall(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes)
{
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE || blockBytes > SMALL_LIMIT_BYTES,
        HCCL_ERROR("ExecSmall: invalid competition shape size[%u], bytes[%llu]", param.rankSize,
            static_cast<unsigned long long>(blockBytes)), HCCL_E_PARA);
    if (param.myRank == param.root) {
        return ExecSmallRoot(param, resCtx, blockBytes);
    }
    if (ServerBase(param.myRank) == ServerBase(param.root)) {
        return ExecSmallLocalLeaf(param, resCtx, blockBytes);
    }
    if (param.myRank == RemoteLeader(param.root)) {
        return ExecSmallRemoteLeader(param, resCtx, blockBytes);
    }
    return ExecSmallRemoteLeaf(param, resCtx, blockBytes);
}

// ------------------------------ direct fallback ------------------------------
HcclResult ExecDirectFallback(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes)
{
    if (blockBytes <= TINY_LIMIT_BYTES) {
        return (param.myRank == param.root) ? ExecTinyRoot(param, resCtx, blockBytes)
                                           : ExecTinyRecv(param, resCtx, blockBytes);
    }

    // Non-competition fallback: direct, chunk-major fan-out so peer waits do
    // not unnecessarily serialize one full rank after another.
    const uint64_t chunk = std::min<uint64_t>(32ULL * 1024ULL * 1024ULL, resCtx.localBuffer.size);
    CHK_PRT_RET(chunk == 0, HCCL_ERROR("ExecDirectFallback: empty HCCL buffer"), HCCL_E_INTERNAL);
    ThreadHandle thread = resCtx.aicpuThread;

    if (param.myRank == param.root) {
        for (uint64_t offset = 0; offset < blockBytes; offset += chunk) {
            const uint64_t bytes = std::min(chunk, blockBytes - offset);
            for (uint32_t i = 0; i < resCtx.channelCount; ++i) {
                const ChannelInfo &ch = resCtx.channels[i];
                CHK_RET(CheckBuffer(ch.remoteCclMem, bytes, "ExecDirectFallback"));
                const uint8_t *src = BytePtr(param.inputPtr,
                    static_cast<uint64_t>(ch.remoteRank) * blockBytes + offset);
                CHK_RET(static_cast<HcclResult>(
                    HcommWriteOnThread(thread, ch.handle, ch.remoteCclMem.addr, const_cast<uint8_t *>(src), bytes)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, ch.handle, DataNotify(0))));
            }
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
                BytePtr(param.outputPtr, offset),
                BytePtr(param.inputPtr, static_cast<uint64_t>(param.root) * blockBytes + offset), bytes)));
            for (uint32_t i = 0; i < resCtx.channelCount; ++i) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    thread, resCtx.channels[i].handle, AckNotify(0), CUSTOM_TIMEOUT)));
            }
        }
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(resCtx.channelCount != 1,
        HCCL_ERROR("ExecDirectFallback: receiver channel count[%u]", resCtx.channelCount), HCCL_E_INTERNAL);
    const ChannelInfo &ch = resCtx.channels[0];
    for (uint64_t offset = 0; offset < blockBytes; offset += chunk) {
        const uint64_t bytes = std::min(chunk, blockBytes - offset);
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, ch.handle, DataNotify(0), CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, BytePtr(param.outputPtr, offset), resCtx.localBuffer.addr, bytes)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, ch.handle, AckNotify(0))));
    }
    return HCCL_SUCCESS;
}

// ------------------------------ 2D root ------------------------------
HcclResult ScheduleRootMeshWorker(const OpParam &param, const AlgResourceCtx &resCtx, const PipelinePlan &plan,
    uint32_t workerIndex, uint32_t localRank, const ChannelInfo &relayChannel, uint64_t blockBytes)
{
    ThreadHandle worker = resCtx.threads[workerIndex];
    ThreadHandle main = resCtx.threads[0];
    const uint32_t pairRank = localRank ^ RANKS_PER_SERVER;
    CHK_RET(CheckBuffer(relayChannel.remoteCclMem, 2 * plan.relayStride, "RootMeshWorker"));

    bool outstanding[2] = {false, false};
    for (uint64_t round = 0; round < plan.rounds; ++round) {
        const uint32_t slot = static_cast<uint32_t>(round & 1ULL);
        if (outstanding[slot]) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, relayChannel.handle, AckNotify(slot), CUSTOM_TIMEOUT)));
            outstanding[slot] = false;
        }

        const uint64_t offset = round * plan.chunk;
        const uint64_t bytes = std::min(plan.chunk, blockBytes - offset);
        const uint64_t aBytes = SplitA(bytes, blockBytes);
        const uint64_t slotBase = static_cast<uint64_t>(slot) * plan.relayStride;

        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(worker, relayChannel.handle,
            BytePtr(relayChannel.remoteCclMem.addr, slotBase),
            BytePtr(param.inputPtr, static_cast<uint64_t>(localRank) * blockBytes + offset), bytes)));
        if (aBytes != 0) {
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(worker, relayChannel.handle,
                BytePtr(relayChannel.remoteCclMem.addr, slotBase + plan.chunk),
                BytePtr(param.inputPtr, static_cast<uint64_t>(pairRank) * blockBytes + offset), aBytes)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(worker, relayChannel.handle, DataNotify(slot))));
        outstanding[slot] = true;
    }

    for (uint32_t slot = 0; slot < 2; ++slot) {
        if (outstanding[slot]) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, relayChannel.handle, AckNotify(slot), CUSTOM_TIMEOUT)));
        }
    }
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(worker, main, workerIndex)));
    return HCCL_SUCCESS;
}

HcclResult ScheduleRootClosDirectWorker(const OpParam &param, const AlgResourceCtx &resCtx,
    const PipelinePlan &plan, uint64_t blockBytes, uint32_t workerIndex, uint32_t rankBegin, uint32_t rankEnd)
{
    ThreadHandle worker = resCtx.threads[workerIndex];
    ThreadHandle main = resCtx.threads[0];
    const uint32_t leader = RemoteLeader(param.root);

    bool outstanding[2] = {false, false};
    for (uint64_t round = 0; round < plan.rounds; ++round) {
        const uint32_t slot = static_cast<uint32_t>(round & 1ULL);
        if (outstanding[slot]) {
            for (uint32_t rank = rankBegin; rank < rankEnd; ++rank) {
                const ChannelInfo *channel = FindChannel(resCtx, rank);
                CHK_PRT_RET(channel == nullptr,
                    HCCL_ERROR("RootClosDirect: no channel to remote rank[%u]", rank), HCCL_E_INTERNAL);
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    worker, channel->handle, AckNotify(slot), CUSTOM_TIMEOUT)));
            }
            outstanding[slot] = false;
        }

        const uint64_t offset = round * plan.chunk;
        const uint64_t bytes = std::min(plan.chunk, blockBytes - offset);
        const uint64_t aBytes = SplitA(bytes, blockBytes);
        const uint64_t bBytes = bytes - aBytes;
        const uint64_t leafBase = static_cast<uint64_t>(slot) * plan.chunk;

        for (uint32_t rank = rankBegin; rank < rankEnd; ++rank) {
            const ChannelInfo *channel = FindChannel(resCtx, rank);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("RootClosDirect: no channel to remote rank[%u]", rank), HCCL_E_INTERNAL);
            CHK_RET(CheckBuffer(channel->remoteCclMem, 2 * plan.chunk, "RootClosDirect/remote"));

            if (rank == leader && aBytes != 0) {
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(worker, channel->handle,
                    BytePtr(channel->remoteCclMem.addr, leafBase),
                    BytePtr(param.inputPtr, static_cast<uint64_t>(rank) * blockBytes + offset), aBytes)));
            }
            if (bBytes != 0) {
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(worker, channel->handle,
                    BytePtr(channel->remoteCclMem.addr, leafBase + plan.aCap),
                    BytePtr(param.inputPtr,
                        static_cast<uint64_t>(rank) * blockBytes + offset + aBytes), bBytes)));
            }
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(worker, channel->handle, DataNotify(slot))));
        }
        outstanding[slot] = true;
    }

    for (uint32_t slot = 0; slot < 2; ++slot) {
        if (!outstanding[slot]) {
            continue;
        }
        for (uint32_t rank = rankBegin; rank < rankEnd; ++rank) {
            const ChannelInfo *channel = FindChannel(resCtx, rank);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("RootClosDirect: no channel to remote rank[%u]", rank), HCCL_E_INTERNAL);
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, channel->handle, AckNotify(slot), CUSTOM_TIMEOUT)));
        }
    }

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(worker, main, workerIndex)));
    return HCCL_SUCCESS;
}

HcclResult Exec2DRoot(const OpParam &param, const AlgResourceCtx &resCtx, const PipelinePlan &plan,
    uint64_t blockBytes)
{
    const bool dualClos = (blockBytes == PERF_512M_BYTES);
    const uint32_t expectedThreads = dualClos ? 10U : 9U;
    CHK_PRT_RET(resCtx.threadCount != expectedThreads,
        HCCL_ERROR("Exec2DRoot: expected[%u] threads, got[%u]", expectedThreads, resCtx.threadCount),
        HCCL_E_INTERNAL);
    CHK_RET(StartWorkers(resCtx));

    const uint32_t rootBase = ServerBase(param.root);
    uint32_t workerIndex = 1;
    for (uint32_t rank = rootBase; rank < rootBase + RANKS_PER_SERVER; ++rank) {
        if (rank == param.root) {
            continue;
        }
        const ChannelInfo *channel = FindChannel(resCtx, rank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Exec2DRoot: no channel to local rank[%u]", rank), HCCL_E_INTERNAL);
        CHK_RET(ScheduleRootMeshWorker(param, resCtx, plan, workerIndex, rank, *channel, blockBytes));
        ++workerIndex;
    }

    const uint32_t remoteBase = ServerBase(RemoteLeader(param.root));
    if (dualClos) {
        CHK_RET(ScheduleRootClosDirectWorker(
            param, resCtx, plan, blockBytes, 8, remoteBase, remoteBase + 4));
        CHK_RET(ScheduleRootClosDirectWorker(
            param, resCtx, plan, blockBytes, 9, remoteBase + 4, remoteBase + RANKS_PER_SERVER));
    } else {
        CHK_RET(ScheduleRootClosDirectWorker(
            param, resCtx, plan, blockBytes, 8, remoteBase, remoteBase + RANKS_PER_SERVER));
    }

    ThreadHandle main = resCtx.threads[0];
    for (uint64_t offset = 0; offset < blockBytes; offset += plan.chunk) {
        const uint64_t bytes = std::min(plan.chunk, blockBytes - offset);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(main, BytePtr(param.outputPtr, offset),
            BytePtr(param.inputPtr, static_cast<uint64_t>(param.root) * blockBytes + offset), bytes)));
    }

    for (uint32_t i = 1; i < resCtx.threadCount; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(main, i, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

// ------------------------------ local relay ------------------------------
HcclResult Exec2DLocalRelay(const OpParam &param, const AlgResourceCtx &resCtx, const PipelinePlan &plan,
    uint64_t blockBytes)
{
    const uint32_t pairRank = param.myRank ^ RANKS_PER_SERVER;
    const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
    const ChannelInfo *pairChannel = FindChannel(resCtx, pairRank);
    CHK_PRT_RET(rootChannel == nullptr || pairChannel == nullptr,
        HCCL_ERROR("Exec2DLocalRelay: missing root/pair channel"), HCCL_E_INTERNAL);
    CHK_RET(CheckBuffer(resCtx.localBuffer, 2 * plan.relayStride, "Exec2DLocalRelay/local"));
    CHK_RET(CheckBuffer(pairChannel->remoteCclMem, 2 * plan.chunk, "Exec2DLocalRelay/remote"));

    ThreadHandle thread = resCtx.aicpuThread;
    bool outstanding[2] = {false, false};
    for (uint64_t round = 0; round < plan.rounds; ++round) {
        const uint32_t slot = static_cast<uint32_t>(round & 1ULL);
        if (outstanding[slot]) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, pairChannel->handle, AckNotify(slot), CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, rootChannel->handle, AckNotify(slot))));
            outstanding[slot] = false;
        }

        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, rootChannel->handle, DataNotify(slot), CUSTOM_TIMEOUT)));
        const uint64_t offset = round * plan.chunk;
        const uint64_t bytes = std::min(plan.chunk, blockBytes - offset);
        const uint64_t aBytes = SplitA(bytes, blockBytes);
        const uint64_t relayBase = static_cast<uint64_t>(slot) * plan.relayStride;
        const uint64_t leafBase = static_cast<uint64_t>(slot) * plan.chunk;

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            thread, BytePtr(param.outputPtr, offset), BytePtr(resCtx.localBuffer.addr, relayBase), bytes)));
        if (aBytes != 0) {
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, pairChannel->handle,
                BytePtr(pairChannel->remoteCclMem.addr, leafBase),
                BytePtr(resCtx.localBuffer.addr, relayBase + plan.chunk), aBytes)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, pairChannel->handle, DataNotify(slot))));
        outstanding[slot] = true;
    }

    for (uint32_t slot = 0; slot < 2; ++slot) {
        if (outstanding[slot]) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, pairChannel->handle, AckNotify(slot), CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, rootChannel->handle, AckNotify(slot))));
        }
    }
    return HCCL_SUCCESS;
}

// ------------------------------ remote ranks ------------------------------
HcclResult Exec2DRemotePair(const OpParam &param, const AlgResourceCtx &resCtx, const PipelinePlan &plan,
    uint64_t blockBytes)
{
    // This rank is root ^ 8. Both A and B are written directly by root over
    // the same channel, respecting the one-channel-per-peer rule.
    CHK_PRT_RET(resCtx.channelCount != 1,
        HCCL_ERROR("Exec2DRemotePair: expected one root channel, got[%u]", resCtx.channelCount), HCCL_E_INTERNAL);
    const ChannelInfo &rootChannel = resCtx.channels[0];
    CHK_RET(CheckBuffer(resCtx.localBuffer, 2 * plan.chunk, "Exec2DRemotePair/local"));

    ThreadHandle thread = resCtx.aicpuThread;
    for (uint64_t round = 0; round < plan.rounds; ++round) {
        const uint32_t slot = static_cast<uint32_t>(round & 1ULL);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, rootChannel.handle, DataNotify(slot), CUSTOM_TIMEOUT)));

        const uint64_t offset = round * plan.chunk;
        const uint64_t bytes = std::min(plan.chunk, blockBytes - offset);
        const uint64_t aBytes = SplitA(bytes, blockBytes);
        const uint64_t bBytes = bytes - aBytes;
        const uint64_t leafBase = static_cast<uint64_t>(slot) * plan.chunk;

        if (aBytes != 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                thread, BytePtr(param.outputPtr, offset), BytePtr(resCtx.localBuffer.addr, leafBase), aBytes)));
        }
        if (bBytes != 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
                BytePtr(param.outputPtr, offset + aBytes),
                BytePtr(resCtx.localBuffer.addr, leafBase + plan.aCap), bBytes)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, AckNotify(slot))));
    }
    return HCCL_SUCCESS;
}

HcclResult Exec2DRemoteLeaf(const OpParam &param, const AlgResourceCtx &resCtx, const PipelinePlan &plan,
    uint64_t blockBytes)
{
    const uint32_t relayRank = param.myRank ^ RANKS_PER_SERVER;
    const ChannelInfo *relayChannel = FindChannel(resCtx, relayRank);
    const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
    CHK_PRT_RET(relayChannel == nullptr || rootChannel == nullptr,
        HCCL_ERROR("Exec2DRemoteLeaf: missing relay/root channel"), HCCL_E_INTERNAL);
    CHK_RET(CheckBuffer(resCtx.localBuffer, 2 * plan.chunk, "Exec2DRemoteLeaf/local"));

    ThreadHandle thread = resCtx.aicpuThread;
    for (uint64_t round = 0; round < plan.rounds; ++round) {
        const uint32_t slot = static_cast<uint32_t>(round & 1ULL);
        const uint64_t offset = round * plan.chunk;
        const uint64_t bytes = std::min(plan.chunk, blockBytes - offset);
        const uint64_t aBytes = SplitA(bytes, blockBytes);
        const uint64_t bBytes = bytes - aBytes;
        const uint64_t leafBase = static_cast<uint64_t>(slot) * plan.chunk;

        // Drain the root-Clos B part first and ACK it immediately. Root can
        // then reuse its slot without waiting for the independent A path.
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, rootChannel->handle, DataNotify(slot), CUSTOM_TIMEOUT)));
        if (bBytes != 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
                BytePtr(param.outputPtr, offset + aBytes),
                BytePtr(resCtx.localBuffer.addr, leafBase + plan.aCap), bBytes)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, rootChannel->handle, AckNotify(slot))));

        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, relayChannel->handle, DataNotify(slot), CUSTOM_TIMEOUT)));
        if (aBytes != 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                thread, BytePtr(param.outputPtr, offset), BytePtr(resCtx.localBuffer.addr, leafBase), aBytes)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, relayChannel->handle, AckNotify(slot))));
    }
    return HCCL_SUCCESS;
}

HcclResult Exec2D(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes)
{
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE,
        HCCL_ERROR("Exec2D: expected 16 ranks, got[%u]", param.rankSize), HCCL_E_PARA);
    const PipelinePlan plan = MakePlan(blockBytes, resCtx.localBuffer.size);
    CHK_PRT_RET(plan.chunk == 0,
        HCCL_ERROR("Exec2D: HCCL buffer too small for pipeline, size[%llu]",
            static_cast<unsigned long long>(resCtx.localBuffer.size)),
        HCCL_E_INTERNAL);

    switch (resCtx.role) {
        case SCATTER_ROLE_ROOT:
            return Exec2DRoot(param, resCtx, plan, blockBytes);
        case SCATTER_ROLE_LOCAL_RELAY:
            return Exec2DLocalRelay(param, resCtx, plan, blockBytes);
        case SCATTER_ROLE_REMOTE_LEADER:
            return Exec2DRemotePair(param, resCtx, plan, blockBytes);
        case SCATTER_ROLE_REMOTE_LEAF:
            return Exec2DRemoteLeaf(param, resCtx, plan, blockBytes);
        default:
            HCCL_ERROR("Exec2D: invalid role[%u]", resCtx.role);
            return HCCL_E_INTERNAL;
    }
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("ExecOp: unsupported dataType[%d]", param.dataType), HCCL_E_PARA);
    CHK_RET(ValidateResource(param, resCtx));
    CHK_PRT_RET(param.count > UINT64_MAX / sizeof(float),
        HCCL_ERROR("ExecOp: count overflow[%llu]", static_cast<unsigned long long>(param.count)), HCCL_E_PARA);
    const uint64_t blockBytes = param.count * sizeof(float);

    if (resCtx.algo == SCATTER_ALGO_SMALL) {
        return ExecSmall(param, resCtx, blockBytes);
    }
    if (resCtx.algo == SCATTER_ALGO_2D) {
        return Exec2D(param, resCtx, blockBytes);
    }
    return ExecDirectFallback(param, resCtx, blockBytes);
}
} // namespace ops_hccl
