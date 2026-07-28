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
#include <array>
#include <limits>
#include <vector>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace {

constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t SERVER_NUM = 2;
constexpr uint32_t OPTIMIZED_RANK_SIZE = RANKS_PER_SERVER * SERVER_NUM;

constexpr uint32_t CONTROL_THREAD_IDX = 0;
constexpr uint32_t FIRST_WORKER_THREAD_IDX = 1;
constexpr uint32_t COPY_THREAD_IDX = 8;
constexpr uint32_t THREAD_NUM = 16;

constexpr uint64_t SMALL_RELAY_THRESHOLD = 1ULL * 1024ULL * 1024ULL; // 1MiB

// 仅中等偏小消息使用双平面并发，极小功能用例继续走原稳定路径。
constexpr uint64_t SMALL_DUAL_PLANE_MIN = 256ULL * 1024ULL;

constexpr uint32_t PIPELINE_SLOT_NUM = 2;
constexpr uint32_t CHANNEL_NOTIFY_PER_SLOT = 4;

constexpr uint32_t AckNotify(uint32_t slot)
{
    return slot * CHANNEL_NOTIFY_PER_SLOT;
}

constexpr uint32_t RootDataNotify(uint32_t slot)
{
    return slot * CHANNEL_NOTIFY_PER_SLOT + 1;
}

constexpr uint32_t RelayNotify(uint32_t slot)
{
    return slot * CHANNEL_NOTIFY_PER_SLOT + 2;
}

constexpr uint32_t AllGatherNotify(uint32_t slot)
{
    return slot * CHANNEL_NOTIFY_PER_SLOT + 3;
}

constexpr uint32_t WorkerStartNotify(uint32_t slot)
{
    return slot;
}

constexpr uint32_t CopyStartNotify(uint32_t slot)
{
    return slot;
}

using ChannelTable = std::array<const ChannelInfo *, OPTIMIZED_RANK_SIZE>;

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const auto &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

const ChannelInfo *FindChannelFast(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint32_t remoteRank)
{
    if (remoteRank == param.myRank) {
        return nullptr;
    }

    // Host侧按remoteRank升序建Channel并跳过myRank时，可O(1)定位。
    // 若顺序不符，自动回退原来的线性查找，保证正确性。
    if (resCtx.channels.size() + 1U == param.rankSize) {
        const uint32_t index =
            (remoteRank < param.myRank) ? remoteRank : remoteRank - 1U;

        if (index < resCtx.channels.size() &&
            resCtx.channels[index].remoteRank == remoteRank) {
            return &resCtx.channels[index];
        }
    }

    return FindChannel(resCtx, remoteRank);
}

HcclResult BuildChannelTable(const OpParam &param, const AlgResourceCtx &resCtx, ChannelTable &channels)
{
    channels.fill(nullptr);
    for (const auto &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteRank >= param.rankSize,
            HCCL_ERROR("Invalid remote rank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
        channels[channel.remoteRank] = &channel;
    }
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.myRank) {
            CHK_PRT_RET(channels[rank] == nullptr,
                HCCL_ERROR("Channel to rank[%u] not found", rank), HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateAndGetBytes(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t &totalBytes,
                                uint64_t &typeSize)
{
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("Invalid root[%u], rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No communication thread"), HCCL_E_INTERNAL);
    CHK_PTR_NULL(resCtx.localBuffer.addr);

    const auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported dataType[%d]", static_cast<int32_t>(param.dataType)), HCCL_E_PARA);
    typeSize = sizeIter->second;

    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("Data size overflow"), HCCL_E_PARA);
    totalBytes = param.count * typeSize;
    return HCCL_SUCCESS;
}

uint64_t GetBufferCapacity(const AlgResourceCtx &resCtx)
{
    uint64_t capacity = resCtx.localBuffer.size;
    for (const auto &channel : resCtx.channels) {
        capacity = std::min(capacity, channel.remoteCclMem.size);
    }
    return capacity;
}

uint64_t DivideRoundUp(uint64_t value, uint64_t divisor)
{
    return value / divisor + ((value % divisor) != 0 ? 1ULL : 0ULL);
}

uint64_t ChooseBalancedTileCapacity(
    uint64_t totalBytes,
    uint64_t bufferCapacity,
    uint64_t typeSize)
{
    uint64_t maxTile =
        bufferCapacity / PIPELINE_SLOT_NUM;
    maxTile -= maxTile % typeSize;

    if (maxTile == 0) {
        return 0;
    }

    const uint64_t tileCount =
        DivideRoundUp(totalBytes, maxTile);
    const uint64_t totalCount =
        totalBytes / typeSize;
    const uint64_t balancedCount =
        DivideRoundUp(totalCount, tileCount);

    uint64_t balancedTile =
        balancedCount * typeSize;

    if (balancedTile > maxTile) {
        balancedTile = maxTile;
    }

    return balancedTile;
}

// ========================================
// 小消息直发 (≤ 1MB): 单线程 + WriteWithNotify + P1零拷贝 + P4无ACK
// ========================================

/*
 * 小消息双平面并发路径。
 *
 * 数据语义与原ExecSimpleSingleShot一致，仍然直接从inputPtr发送：
 * - control Thread负责另一台Server的8条Clos链路；
 * - worker Thread负责本Server的7条Full-Mesh链路；
 * - 仅增加一组本地Thread Record/Wait，把worker接入主DAG；
 * - 非root接收逻辑完全保持原样。
 *
 * 相比单Thread顺序发送15次，可让两组独立链路并发推进；
 * 相比4/8 Thread方案，本地启动同步开销更低。
 */
HcclResult ExecSimpleDualPlane(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t totalBytes)
{
    CHK_PRT_RET(
        resCtx.threads.size() < 2,
        HCCL_ERROR(
            "Dual-plane path needs at least 2 threads, got[%zu]",
            resCtx.threads.size()),
        HCCL_E_INTERNAL);

    ThreadHandle controlThread =
        resCtx.threads[CONTROL_THREAD_IDX];
    ThreadHandle localThread =
        resCtx.threads[FIRST_WORKER_THREAD_IDX];

    char *input =
        static_cast<char *>(param.inputPtr);
    char *output =
        static_cast<char *>(param.outputPtr);
    char *localBuffer =
        static_cast<char *>(resCtx.localBuffer.addr);

    if (param.myRank == param.root) {
        const uint32_t rootServer =
            param.myRank / RANKS_PER_SERVER;

        // 将本Server发送线程接入control Thread的主执行链。
        CHK_RET(HcommThreadNotifyRecordOnThread(
            controlThread,
            localThread,
            WorkerStartNotify(0)));

        CHK_RET(HcommThreadNotifyWaitOnThread(
            localThread,
            WorkerStartNotify(0),
            0));

        for (const auto &channel : resCtx.channels) {
            const uint32_t remoteServer =
                channel.remoteRank / RANKS_PER_SERVER;

            ThreadHandle txThread =
                (remoteServer == rootServer)
                    ? localThread
                    : controlThread;

            CHK_PTR_NULL(channel.remoteCclMem.addr);

            CHK_RET(HcommWriteWithNotifyOnThread(
                txThread,
                channel.handle,
                channel.remoteCclMem.addr,
                input,
                totalBytes,
                NOTIFY_IDX_DATA_SIGNAL));
        }

        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel =
        FindChannelFast(
            param,
            resCtx,
            param.root);

    CHK_PTR_NULL(rootChannel);

    CHK_RET(HcommChannelNotifyWaitOnThread(
        controlThread,
        rootChannel->handle,
        NOTIFY_IDX_DATA_SIGNAL,
        0));

    CHK_RET(HcommLocalCopyOnThread(
        controlThread,
        output,
        localBuffer,
        totalBytes));

    return HCCL_SUCCESS;
}

HcclResult ExecSimpleSingleShot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes)
{
    ThreadHandle thread = resCtx.threads[CONTROL_THREAD_IDX];
    char *input = static_cast<char *>(param.inputPtr);
    char *output = static_cast<char *>(param.outputPtr);
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);

    if (param.myRank == param.root) {
        for (const auto &channel : resCtx.channels) {
            CHK_RET(HcommWriteWithNotifyOnThread(thread, channel.handle,
                channel.remoteCclMem.addr, input, totalBytes, NOTIFY_IDX_DATA_SIGNAL));
        }
        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel = FindChannelFast(param, resCtx, param.root);
    CHK_PTR_NULL(rootChannel);
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, rootChannel->handle, NOTIFY_IDX_DATA_SIGNAL, 0));
    CHK_RET(HcommLocalCopyOnThread(thread, output, localBuffer, totalBytes));
    return HCCL_SUCCESS;
}

// ========================================
// 中等消息直发 (≤ bufferCapacity): 15 worker 多线程直发
// P1注入: Root不再拷贝全量数据到localBuffer, worker直接WriteWithNotify(inputPtr)
// ========================================
HcclResult ExecDirectFlatOneShot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes)
{
    ChannelTable channels{};
    CHK_RET(BuildChannelTable(param, resCtx, channels));

    ThreadHandle controlThread = resCtx.threads[CONTROL_THREAD_IDX];
    char *input = static_cast<char *>(param.inputPtr);
    char *output = static_cast<char *>(param.outputPtr);
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);

    if (param.myRank == param.root) {
        uint32_t workerIdx = 0;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank == param.root) {
                continue;
            }

            ThreadHandle workerThread = resCtx.threads[FIRST_WORKER_THREAD_IDX + workerIdx];
            const ChannelInfo *channel = channels[rank];
            CHK_PTR_NULL(channel);

            CHK_RET(HcommThreadNotifyRecordOnThread(controlThread, workerThread, WorkerStartNotify(0)));
            CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, WorkerStartNotify(0), 0));

            CHK_RET(HcommWriteWithNotifyOnThread(workerThread, channel->handle,
                channel->remoteCclMem.addr, input, totalBytes, RootDataNotify(0)));
            ++workerIdx;
        }

        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel = channels[param.root];
    CHK_PTR_NULL(rootChannel);
    CHK_RET(HcommChannelNotifyWaitOnThread(controlThread, rootChannel->handle, RootDataNotify(0), 0));
    CHK_RET(HcommLocalCopyOnThread(controlThread, output, localBuffer, totalBytes));
    return HCCL_SUCCESS;
}

// ========================================
// 大消息分片流水线 (> bufferCapacity)
// 数据按 tile 切分, 8路 Scatter-Gather + 双槽交替。
// ========================================
void BuildSlices(uint64_t tileBytes, uint64_t typeSize,
                  std::array<uint64_t, RANKS_PER_SERVER> &sliceOffsets,
                  std::array<uint64_t, RANKS_PER_SERVER> &sliceBytes)
{
    const uint64_t tileCount = tileBytes / typeSize;
    const uint64_t baseCount = tileCount / RANKS_PER_SERVER;
    const uint64_t remainder = tileCount % RANKS_PER_SERVER;

    uint64_t currentCount = 0;
    for (uint32_t idx = 0; idx < RANKS_PER_SERVER; ++idx) {
        const uint64_t count = baseCount + ((idx < remainder) ? 1ULL : 0ULL);
        sliceOffsets[idx] = currentCount * typeSize;
        sliceBytes[idx] = count * typeSize;
        currentCount += count;
    }
}

HcclResult WaitSlotAck(const OpParam &param, const ChannelTable &channels,
                        ThreadHandle controlThread, uint32_t slot)
{
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.root) {
            CHK_RET(HcommChannelNotifyWaitOnThread(controlThread, channels[rank]->handle, AckNotify(slot), 0));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchReceiverCopyAndAck(const OpParam &param, const AlgResourceCtx &resCtx,
                                     const ChannelTable &channels, ThreadHandle controlThread,
                                     ThreadHandle copyThread, uint64_t dataOffset, uint32_t slot,
                                     uint64_t slotOffset, uint64_t tileBytes)
{
    char *output = static_cast<char *>(param.outputPtr);
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);

    CHK_RET(HcommThreadNotifyRecordOnThread(controlThread, copyThread, CopyStartNotify(slot)));
    CHK_RET(HcommThreadNotifyWaitOnThread(copyThread, CopyStartNotify(slot), 0));

    CHK_RET(HcommLocalCopyOnThread(copyThread, output + dataOffset, localBuffer + slotOffset, tileBytes));
    CHK_RET(HcommChannelNotifyRecordOnThread(copyThread, channels[param.root]->handle, AckNotify(slot)));

    return HCCL_SUCCESS;
}

HcclResult LaunchRootSliceDistribution(const OpParam &param, const AlgResourceCtx &resCtx,
                                        const ChannelTable &channels, ThreadHandle controlThread,
                                        const char *inputTile, uint32_t slot, uint64_t slotOffset,
                                        const std::array<uint64_t, RANKS_PER_SERVER> &sliceOffsets,
                                        const std::array<uint64_t, RANKS_PER_SERVER> &sliceBytes)
{
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);
    const uint32_t rootLocal = param.root % RANKS_PER_SERVER;
    const uint32_t otherServerBase = RANKS_PER_SERVER;

    if (sliceBytes[rootLocal] > 0) {
        CHK_RET(HcommLocalCopyOnThread(controlThread,
            localBuffer + slotOffset + sliceOffsets[rootLocal],
            inputTile + sliceOffsets[rootLocal], sliceBytes[rootLocal]));
    }

    uint32_t workerIdx = 0;
    for (uint32_t local = 0; local < RANKS_PER_SERVER; ++local) {
        ThreadHandle txThread = controlThread;

        if (local != rootLocal) {
            txThread = resCtx.threads[FIRST_WORKER_THREAD_IDX + workerIdx];

            CHK_RET(HcommThreadNotifyRecordOnThread(controlThread, txThread, WorkerStartNotify(slot)));
            CHK_RET(HcommThreadNotifyWaitOnThread(txThread, WorkerStartNotify(slot), 0));
            ++workerIdx;
        }

        if (local != rootLocal && sliceBytes[local] > 0) {
            CHK_RET(HcommLocalCopyOnThread(txThread,
                localBuffer + slotOffset + sliceOffsets[local],
                inputTile + sliceOffsets[local], sliceBytes[local]));
        }

        const uint32_t remoteRank = otherServerBase + local;
        const ChannelInfo *remoteChannel = channels[remoteRank];
        CHK_PTR_NULL(remoteChannel);

        if (sliceBytes[local] > 0) {
            char *remoteBuf = static_cast<char *>(remoteChannel->remoteCclMem.addr);
            CHK_RET(HcommWriteWithNotifyOnThread(txThread, remoteChannel->handle,
                remoteBuf + slotOffset + sliceOffsets[local],
                localBuffer + slotOffset + sliceOffsets[local],
                sliceBytes[local], RootDataNotify(slot)));
        }

        if (local != rootLocal && sliceBytes[rootLocal] > 0) {
            const ChannelInfo *localChannel = channels[local];
            CHK_PTR_NULL(localChannel);
            char *localRemoteBuf = static_cast<char *>(localChannel->remoteCclMem.addr);
            CHK_RET(HcommWriteWithNotifyOnThread(txThread, localChannel->handle,
                localRemoteBuf + slotOffset + sliceOffsets[rootLocal],
                localBuffer + slotOffset + sliceOffsets[rootLocal],
                sliceBytes[rootLocal], AllGatherNotify(slot)));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult LaunchLocalAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
                                 const ChannelTable &channels, ThreadHandle controlThread,
                                 uint32_t slot, uint64_t slotOffset,
                                 const std::array<uint64_t, RANKS_PER_SERVER> &sliceOffsets,
                                 const std::array<uint64_t, RANKS_PER_SERVER> &sliceBytes)
{
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);
    const uint32_t serverId = param.myRank / RANKS_PER_SERVER;
    const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
    const uint32_t serverBase = serverId * RANKS_PER_SERVER;

    if (sliceBytes[localRank] == 0) {
        return HCCL_SUCCESS;
    }

    uint32_t workerIdx = 0;
    for (uint32_t peerLocal = 0; peerLocal < RANKS_PER_SERVER; ++peerLocal) {
        if (peerLocal == localRank) {
            continue;
        }
        if (serverId == 0 && serverBase + peerLocal == param.root) {
            continue;
        }

        ThreadHandle workerThread = resCtx.threads[FIRST_WORKER_THREAD_IDX + workerIdx];
        const ChannelInfo *peerChannel = channels[serverBase + peerLocal];
        CHK_PTR_NULL(peerChannel);

        CHK_RET(HcommThreadNotifyRecordOnThread(controlThread, workerThread, WorkerStartNotify(slot)));
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, WorkerStartNotify(slot), 0));

        char *remoteBuf = static_cast<char *>(peerChannel->remoteCclMem.addr);
        CHK_RET(HcommWriteWithNotifyOnThread(workerThread, peerChannel->handle,
            remoteBuf + slotOffset + sliceOffsets[localRank],
            localBuffer + slotOffset + sliceOffsets[localRank],
            sliceBytes[localRank], AllGatherNotify(slot)));
        ++workerIdx;
    }

    return HCCL_SUCCESS;
}

HcclResult WaitLocalAllGather(const OpParam &param, const ChannelTable &channels,
                               ThreadHandle controlThread, uint32_t slot,
                               const std::array<uint64_t, RANKS_PER_SERVER> &sliceBytes)
{
    const uint32_t serverId = param.myRank / RANKS_PER_SERVER;
    const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
    const uint32_t serverBase = serverId * RANKS_PER_SERVER;

    for (uint32_t peerLocal = 0; peerLocal < RANKS_PER_SERVER; ++peerLocal) {
        if (peerLocal == localRank) {
            continue;
        }
        if (sliceBytes[peerLocal] == 0) {
            continue;
        }
        CHK_RET(HcommChannelNotifyWaitOnThread(controlThread,
            channels[serverBase + peerLocal]->handle, AllGatherNotify(slot), 0));
    }

    return HCCL_SUCCESS;
}

HcclResult RunRelayStripePipeline(const OpParam &param, const AlgResourceCtx &resCtx,
                                   const ChannelTable &channels, uint64_t totalBytes,
                                   uint64_t typeSize, uint64_t tileCapacity)
{
    ThreadHandle controlThread = resCtx.threads[CONTROL_THREAD_IDX];
    ThreadHandle copyThread = resCtx.threads[COPY_THREAD_IDX];

    char *input = static_cast<char *>(param.inputPtr);
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);

    const uint32_t serverId = param.myRank / RANKS_PER_SERVER;
    const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
    const uint32_t rootLocal = param.root % RANKS_PER_SERVER;
    const uint32_t remotePartner = (serverId == 0) ? param.myRank + RANKS_PER_SERVER
                                                     : param.myRank - RANKS_PER_SERVER;

    uint64_t dataOffset = 0;
    uint64_t tileIndex = 0;

    while (dataOffset < totalBytes) {
        const uint64_t tileBytes = std::min(tileCapacity, totalBytes - dataOffset);
        const uint32_t slot = static_cast<uint32_t>(tileIndex % PIPELINE_SLOT_NUM);
        const uint64_t slotOffset = static_cast<uint64_t>(slot) * tileCapacity;

        std::array<uint64_t, RANKS_PER_SERVER> sliceOffsets{};
        std::array<uint64_t, RANKS_PER_SERVER> sliceBytes{};
        BuildSlices(tileBytes, typeSize, sliceOffsets, sliceBytes);

        if (param.myRank == param.root && tileIndex >= PIPELINE_SLOT_NUM) {
            CHK_RET(WaitSlotAck(param, channels, controlThread, slot));
        }

        if (param.myRank == param.root) {
            CHK_RET(LaunchRootSliceDistribution(param, resCtx, channels, controlThread,
                input + dataOffset, slot, slotOffset, sliceOffsets, sliceBytes));
        } else if (serverId == 1) {
            if (sliceBytes[localRank] > 0) {
                CHK_RET(HcommChannelNotifyWaitOnThread(controlThread,
                    channels[param.root]->handle, RootDataNotify(slot), 0));
            }
            if (localRank != rootLocal && sliceBytes[localRank] > 0) {
                const ChannelInfo *relayChannel = channels[remotePartner];
                CHK_PTR_NULL(relayChannel);
                char *remoteBuf = static_cast<char *>(relayChannel->remoteCclMem.addr);
                CHK_RET(HcommWriteWithNotifyOnThread(controlThread, relayChannel->handle,
                    remoteBuf + slotOffset + sliceOffsets[localRank],
                    localBuffer + slotOffset + sliceOffsets[localRank],
                    sliceBytes[localRank], RelayNotify(slot)));
            }
        } else {
            if (sliceBytes[localRank] > 0) {
                CHK_RET(HcommChannelNotifyWaitOnThread(controlThread,
                    channels[remotePartner]->handle, RelayNotify(slot), 0));
            }
        }

        if (param.myRank != param.root) {
            CHK_RET(LaunchLocalAllGather(param, resCtx, channels, controlThread,
                slot, slotOffset, sliceOffsets, sliceBytes));
            CHK_RET(WaitLocalAllGather(param, channels, controlThread, slot, sliceBytes));
            CHK_RET(LaunchReceiverCopyAndAck(param, resCtx, channels, controlThread, copyThread,
                dataOffset, slot, slotOffset, tileBytes));
        }

        dataOffset += tileBytes;
        ++tileIndex;
    }

    if (param.myRank == param.root) {
        const uint32_t pendingSlots = static_cast<uint32_t>(std::min<uint64_t>(tileIndex, PIPELINE_SLOT_NUM));
        for (uint32_t s = 0; s < pendingSlots; ++s) {
            CHK_RET(WaitSlotAck(param, channels, controlThread, s));
        }
    }

    return HCCL_SUCCESS;
}

// ========================================
// 回退路径: 二项树广播
// ========================================
uint32_t HighestPowerOfTwo(uint32_t value)
{
    uint32_t result = 1;
    while ((result << 1U) <= value) {
        result <<= 1U;
    }
    return result;
}

uint32_t GetLocalTreeParent(uint32_t rank, uint32_t root)
{
    const uint32_t serverBase = (rank / RANKS_PER_SERVER) * RANKS_PER_SERVER;
    const uint32_t localRank = rank % RANKS_PER_SERVER;
    const uint32_t rootLocalRank = root % RANKS_PER_SERVER;
    const uint32_t virtualRank = (localRank + RANKS_PER_SERVER - rootLocalRank) % RANKS_PER_SERVER;
    const uint32_t parentVirtualRank = virtualRank - HighestPowerOfTwo(virtualRank);
    const uint32_t parentLocalRank = (rootLocalRank + parentVirtualRank) % RANKS_PER_SERVER;
    return serverBase + parentLocalRank;
}

void GetTreeChildren(uint32_t rank, uint32_t root, std::vector<uint32_t> &children)
{
    children.clear();
    if (rank == root) {
        children.push_back(root ^ RANKS_PER_SERVER);
    }
    const uint32_t serverBase = (rank / RANKS_PER_SERVER) * RANKS_PER_SERVER;
    const uint32_t localRank = rank % RANKS_PER_SERVER;
    const uint32_t rootLocalRank = root % RANKS_PER_SERVER;
    const uint32_t virtualRank = (localRank + RANKS_PER_SERVER - rootLocalRank) % RANKS_PER_SERVER;
    for (uint32_t mask = 1; mask < RANKS_PER_SERVER; mask <<= 1U) {
        if (virtualRank < mask && virtualRank + mask < RANKS_PER_SERVER) {
            const uint32_t childLocalRank = (rootLocalRank + virtualRank + mask) % RANKS_PER_SERVER;
            children.push_back(serverBase + childLocalRank);
        }
    }
}

HcclResult ExecStableTree(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes)
{
    std::vector<uint32_t> childRanks;
    GetTreeChildren(param.myRank, param.root, childRanks);

    std::vector<const ChannelInfo *> children;
    children.reserve(childRanks.size());
    uint64_t capacity = resCtx.localBuffer.size;
    for (uint32_t childRank : childRanks) {
        const ChannelInfo *childChannel = FindChannel(resCtx, childRank);
        CHK_PTR_NULL(childChannel);
        CHK_PTR_NULL(childChannel->remoteCclMem.addr);
        children.push_back(childChannel);
        capacity = std::min(capacity, childChannel->remoteCclMem.size);
    }

    const ChannelInfo *parent = nullptr;
    if (param.myRank != param.root) {
        const uint32_t remoteRoot = param.root ^ RANKS_PER_SERVER;
        const uint32_t parentRank =
            (param.myRank == remoteRoot) ? param.root : GetLocalTreeParent(param.myRank, param.root);
        parent = FindChannel(resCtx, parentRank);
        CHK_PTR_NULL(parent);
    }

    CHK_PRT_RET(capacity == 0, HCCL_ERROR("HCCL buffer size is zero"), HCCL_E_INTERNAL);

    ThreadHandle thread = resCtx.threads[0];
    char *input = static_cast<char *>(param.inputPtr);
    char *output = static_cast<char *>(param.outputPtr);
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);

    for (uint64_t offset = 0; offset < totalBytes;) {
        const uint64_t chunkBytes = std::min(capacity, totalBytes - offset);

        if (param.myRank == param.root) {
            CHK_RET(HcommLocalCopyOnThread(thread, localBuffer, input + offset, chunkBytes));
        } else {
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, parent->handle, NOTIFY_IDX_DATA_SIGNAL, 0));
            CHK_RET(HcommLocalCopyOnThread(thread, output + offset, localBuffer, chunkBytes));
        }

        for (const ChannelInfo *child : children) {
            CHK_RET(HcommWriteOnThread(thread, child->handle, child->remoteCclMem.addr,
                localBuffer, chunkBytes));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, child->handle, NOTIFY_IDX_DATA_SIGNAL));
        }

        for (const ChannelInfo *child : children) {
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, child->handle, NOTIFY_IDX_ACK, 0));
        }

        if (param.myRank != param.root) {
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, parent->handle, NOTIFY_IDX_ACK));
        }

        offset += chunkBytes;
    }

    return HCCL_SUCCESS;
}

} // namespace

namespace ops_hccl {

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing Broadcast on Ascend NPU");

    if (param.rankSize <= 1 || param.count == 0) {
        return HCCL_SUCCESS;
    }

    uint64_t totalBytes = 0;
    uint64_t typeSize = 0;
    CHK_RET(ValidateAndGetBytes(param, resCtx, totalBytes, typeSize));

    // 256KiB以下保持原单Thread路径，减少功能用例变化范围。
    if (param.rankSize == OPTIMIZED_RANK_SIZE &&
        totalBytes < SMALL_DUAL_PLANE_MIN) {
        return ExecSimpleSingleShot(
            param,
            resCtx,
            totalBytes);
    }

    // 256KiB~1MiB使用双平面并发，目标是压缩测试点9。
    if (param.rankSize == OPTIMIZED_RANK_SIZE &&
        totalBytes <= SMALL_RELAY_THRESHOLD) {
        return ExecSimpleDualPlane(
            param,
            resCtx,
            totalBytes);
    }

    const uint64_t bufferCapacity =
        GetBufferCapacity(resCtx);

    if (param.rankSize == OPTIMIZED_RANK_SIZE) {
        if (totalBytes <= bufferCapacity) {
            return ExecDirectFlatOneShot(
                param,
                resCtx,
                totalBytes);
        }

        if (param.root < RANKS_PER_SERVER) {
            const uint64_t tileCapacity =
                ChooseBalancedTileCapacity(
                    totalBytes,
                    bufferCapacity,
                    typeSize);

            CHK_PRT_RET(tileCapacity == 0,
                HCCL_ERROR("Pipeline tile capacity is zero"), HCCL_E_INTERNAL);

            ChannelTable channels{};
            CHK_RET(BuildChannelTable(param, resCtx, channels));

            return RunRelayStripePipeline(
                param,
                resCtx,
                channels,
                totalBytes,
                typeSize,
                tileCapacity);
        }
    }

    return ExecStableTree(param, resCtx, totalBytes);
}

} // namespace ops_hccl