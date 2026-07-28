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
#include <cstdint>
#include <limits>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
    constexpr uint32_t DISTRIBUTE_DATA_NOTIFY = NOTIFY_IDX_DATA_SIGNAL;
    constexpr uint32_t GLOBAL_READY_NOTIFY = 2;

    struct ScatterTopology {
        bool inRootServer = false;
        uint32_t ownerIndex = 0;
        uint32_t rootOwners[CUSTOM_RANKS_PER_SERVER] = {};
        uint32_t remoteOwners[CUSTOM_RANKS_PER_SERVER] = {};
        uint32_t localOwners[CUSTOM_RANKS_PER_SERVER] = {};
    };

    uint32_t GetDataTypeSize(HcclDataType dataType)
    {
        switch (dataType) {
            case HCCL_DATA_TYPE_INT8:
            case HCCL_DATA_TYPE_UINT8:
            case HCCL_DATA_TYPE_HIF8:
            case HCCL_DATA_TYPE_FP8E4M3:
            case HCCL_DATA_TYPE_FP8E5M2:
            case HCCL_DATA_TYPE_FP8E8M0:
                return 1;
            case HCCL_DATA_TYPE_INT16:
            case HCCL_DATA_TYPE_UINT16:
            case HCCL_DATA_TYPE_FP16:
            case HCCL_DATA_TYPE_BFP16:
                return 2;
            case HCCL_DATA_TYPE_INT32:
            case HCCL_DATA_TYPE_UINT32:
            case HCCL_DATA_TYPE_FP32:
                return 4;
            case HCCL_DATA_TYPE_INT64:
            case HCCL_DATA_TYPE_UINT64:
            case HCCL_DATA_TYPE_FP64:
                return 8;
            case HCCL_DATA_TYPE_INT128:
                return 16;
            default:
                return 0;
        }
    }

    uint32_t GetServerBase(uint32_t rank)
    {
        return (rank / CUSTOM_RANKS_PER_SERVER) * CUSTOM_RANKS_PER_SERVER;
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

    uint64_t GetSharedBufferCapacity(const AlgResourceCtx &resCtx)
    {
        uint64_t capacity = resCtx.localBuffer.size;
        for (const ChannelInfo &channel : resCtx.channels) {
            capacity = std::min(capacity, channel.remoteCclMem.size);
        }
        return capacity;
    }

    HcclResult ValidateResources(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        CHK_PRT_RET(param.rankSize == 0 || param.root >= param.rankSize || param.myRank >= param.rankSize,
            HCCL_ERROR("[ValidateResources] Invalid rank metadata"), HCCL_E_PARA);
        CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("[ValidateResources] No algorithm thread"), HCCL_E_NOT_FOUND);

        if (param.rankSize == 1) {
            return HCCL_SUCCESS;
        }

        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0,
            HCCL_ERROR("[ValidateResources] Invalid local HCCL buffer"), HCCL_E_MEMORY);
        CHK_PRT_RET(resCtx.channels.size() != static_cast<size_t>(param.rankSize - 1),
            HCCL_ERROR("[ValidateResources] Channel count[%llu] does not match rankSize[%u]",
                static_cast<unsigned long long>(resCtx.channels.size()), param.rankSize),
            HCCL_E_NOT_FOUND);

        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteCclMem.addr == nullptr
                            || channel.remoteCclMem.size == 0 || channel.notifyNum < CUSTOM_CHANNEL_NOTIFY_NUM,
                HCCL_ERROR("[ValidateResources] Invalid channel to rank[%u]", channel.remoteRank), HCCL_E_NOT_FOUND);
        }
        return HCCL_SUCCESS;
    }

    HcclResult RunPullStar(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
    {
        const ThreadHandle mainThread = resCtx.threads[0];
        const uint64_t chunkCapacity = GetSharedBufferCapacity(resCtx);
        CHK_PRT_RET(chunkCapacity == 0, HCCL_ERROR("[RunPullStar] Empty communication buffer"), HCCL_E_MEMORY);

        uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
        uint8_t *output = static_cast<uint8_t *>(param.outputPtr);

        if (param.myRank == param.root) {
            for (uint64_t offset = 0; offset < dataSize;) {
                const uint64_t chunkSize = std::min(chunkCapacity, dataSize - offset);
                CHK_RET(HcommLocalCopyOnThread(mainThread, localBuffer, input + offset, chunkSize));
                for (const ChannelInfo &channel : resCtx.channels) {
                    CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
                }
                for (const ChannelInfo &channel : resCtx.channels) {
                    CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
                }
                offset += chunkSize;
            }
            return HCCL_SUCCESS;
        }

        const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
        CHK_PRT_RET(rootChannel == nullptr, HCCL_ERROR("[RunPullStar] Root channel not found"), HCCL_E_NOT_FOUND);
        for (uint64_t offset = 0; offset < dataSize;) {
            const uint64_t chunkSize = std::min(chunkCapacity, dataSize - offset);
            CHK_RET(HcommChannelNotifyWaitOnThread(
                mainThread, rootChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
            CHK_RET(HcommReadOnThread(
                mainThread, rootChannel->handle, output + offset, rootChannel->remoteCclMem.addr, chunkSize));
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel->handle, NOTIFY_IDX_ACK));
            offset += chunkSize;
        }
        return HCCL_SUCCESS;
    }

    HcclResult RunHierarchicalBroadcast(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
    {
        CHK_PRT_RET(resCtx.threads.size() < 2,
            HCCL_ERROR("[RunHierarchicalBroadcast] Need at least two threads"), HCCL_E_NOT_FOUND);

        const uint32_t rootServerBase = GetServerBase(param.root);
        const uint32_t otherServerBase
            = rootServerBase == 0 ? CUSTOM_RANKS_PER_SERVER : 0;
        const uint32_t counterpartRank = otherServerBase + (param.root % CUSTOM_RANKS_PER_SERVER);
        const uint32_t myServerBase = GetServerBase(param.myRank);
        const uint32_t localLeaderRank = myServerBase == rootServerBase ? param.root : counterpartRank;

        const ChannelInfo *localPeers[CUSTOM_RANKS_PER_SERVER - 1] = {};
        uint32_t localPeerNum = 0;
        for (uint32_t rank = myServerBase; rank < myServerBase + CUSTOM_RANKS_PER_SERVER; ++rank) {
            if (rank == param.myRank) {
                continue;
            }
            const ChannelInfo *peer = FindChannel(resCtx, rank);
            CHK_PRT_RET(peer == nullptr,
                HCCL_ERROR("[RunHierarchicalBroadcast] Local peer[%u] not found", rank), HCCL_E_NOT_FOUND);
            localPeers[localPeerNum++] = peer;
        }

        const ThreadHandle mainThread = resCtx.threads[0];
        const uint64_t chunkCapacity = GetSharedBufferCapacity(resCtx);
        CHK_PRT_RET(chunkCapacity == 0,
            HCCL_ERROR("[RunHierarchicalBroadcast] Empty communication buffer"), HCCL_E_MEMORY);

        uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
        uint8_t *output = static_cast<uint8_t *>(param.outputPtr);

        const ChannelInfo *leaderChannel = nullptr;
        if (param.myRank != localLeaderRank) {
            leaderChannel = FindChannel(resCtx, localLeaderRank);
            CHK_PRT_RET(leaderChannel == nullptr,
                HCCL_ERROR("[RunHierarchicalBroadcast] Leader channel not found"), HCCL_E_NOT_FOUND);
        }

        const ChannelInfo *counterpart = nullptr;
        if (param.myRank == param.root || param.myRank == counterpartRank) {
            const uint32_t remoteRank = param.myRank == param.root ? counterpartRank : param.root;
            counterpart = FindChannel(resCtx, remoteRank);
            CHK_PRT_RET(counterpart == nullptr,
                HCCL_ERROR("[RunHierarchicalBroadcast] Counterpart channel not found"), HCCL_E_NOT_FOUND);
        }

        for (uint64_t offset = 0; offset < dataSize;) {
            const uint64_t chunkSize = std::min(chunkCapacity, dataSize - offset);

            if (param.myRank == param.root) {
                const ThreadHandle crossThread = resCtx.threads[1];
                CHK_RET(HcommLocalCopyOnThread(mainThread, localBuffer, input + offset, chunkSize));
                CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, crossThread, 0));
                CHK_RET(HcommThreadNotifyWaitOnThread(crossThread, 0, CUSTOM_TIMEOUT));
                CHK_RET(HcommWriteWithNotifyOnThread(crossThread, counterpart->handle,
                    counterpart->remoteCclMem.addr, localBuffer, chunkSize, NOTIFY_IDX_DATA_SIGNAL));

                for (uint32_t index = 0; index < localPeerNum; ++index) {
                    CHK_RET(HcommChannelNotifyRecordOnThread(
                        mainThread, localPeers[index]->handle, NOTIFY_IDX_DATA_SIGNAL));
                }
                for (uint32_t index = 0; index < localPeerNum; ++index) {
                    CHK_RET(HcommChannelNotifyWaitOnThread(
                        mainThread, localPeers[index]->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
                }
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    mainThread, counterpart->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            } else if (param.myRank == counterpartRank) {
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    mainThread, counterpart->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
                for (uint32_t index = 0; index < localPeerNum; ++index) {
                    CHK_RET(HcommChannelNotifyRecordOnThread(
                        mainThread, localPeers[index]->handle, NOTIFY_IDX_DATA_SIGNAL));
                }
                CHK_RET(HcommLocalCopyOnThread(mainThread, output + offset, localBuffer, chunkSize));
                for (uint32_t index = 0; index < localPeerNum; ++index) {
                    CHK_RET(HcommChannelNotifyWaitOnThread(
                        mainThread, localPeers[index]->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
                }
                CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, counterpart->handle, NOTIFY_IDX_ACK));
            } else {
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    mainThread, leaderChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
                CHK_RET(HcommReadOnThread(mainThread, leaderChannel->handle, output + offset,
                    leaderChannel->remoteCclMem.addr, chunkSize));
                CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, leaderChannel->handle, NOTIFY_IDX_ACK));
            }

            offset += chunkSize;
        }
        return HCCL_SUCCESS;
    }

    uint64_t GetShardOffset(uint64_t chunkSize, uint32_t shardIndex, uint32_t dataTypeSize)
    {
        const uint64_t elementCount = chunkSize / dataTypeSize;
        const uint64_t baseCount = elementCount / CUSTOM_RANKS_PER_SERVER;
        const uint64_t remainder = elementCount % CUSTOM_RANKS_PER_SERVER;
        const uint64_t elementOffset = shardIndex * baseCount + std::min<uint64_t>(shardIndex, remainder);
        return elementOffset * dataTypeSize;
    }

    uint64_t GetShardSize(uint64_t chunkSize, uint32_t shardIndex, uint32_t dataTypeSize)
    {
        const uint64_t elementCount = chunkSize / dataTypeSize;
        const uint64_t baseCount = elementCount / CUSTOM_RANKS_PER_SERVER;
        const uint64_t remainder = elementCount % CUSTOM_RANKS_PER_SERVER;
        const uint64_t shardCount = baseCount + (shardIndex < remainder ? 1 : 0);
        return shardCount * dataTypeSize;
    }

    uint64_t GetScatterChunkSize(uint64_t remainingSize, uint64_t chunkCapacity, uint32_t dataTypeSize)
    {
        uint64_t chunkSize = std::min(chunkCapacity, remainingSize);
        const uint64_t minChunkSize = CUSTOM_RANKS_PER_SERVER * static_cast<uint64_t>(dataTypeSize);
        if (remainingSize > chunkCapacity && remainingSize - chunkSize < minChunkSize) {
            chunkSize -= minChunkSize - (remainingSize - chunkSize);
        }
        return chunkSize;
    }

    void *GetRemoteBuffer(const ChannelInfo &channel, uint64_t offset)
    {
        return static_cast<uint8_t *>(channel.remoteCclMem.addr) + offset;
    }

    HcclResult BuildScatterTopology(const OpParam &param, ScatterTopology &topology)
    {
        const uint32_t rootServerBase = GetServerBase(param.root);
        const uint32_t otherServerBase
            = rootServerBase == 0 ? CUSTOM_RANKS_PER_SERVER : 0;
        const uint32_t rootLocalRank = param.root % CUSTOM_RANKS_PER_SERVER;
        const uint32_t myServerBase = GetServerBase(param.myRank);
        const uint32_t myLocalRank = param.myRank % CUSTOM_RANKS_PER_SERVER;

        topology.inRootServer = myServerBase == rootServerBase;
        topology.ownerIndex
            = (myLocalRank + CUSTOM_RANKS_PER_SERVER - rootLocalRank) % CUSTOM_RANKS_PER_SERVER;

        for (uint32_t index = 0; index < CUSTOM_RANKS_PER_SERVER; ++index) {
            const uint32_t localRank = (rootLocalRank + index) % CUSTOM_RANKS_PER_SERVER;
            topology.rootOwners[index] = rootServerBase + localRank;
            topology.remoteOwners[index] = otherServerBase + localRank;
            topology.localOwners[index]
                = topology.inRootServer ? topology.rootOwners[index] : topology.remoteOwners[index];
        }
        return HCCL_SUCCESS;
    }

    HcclResult RunScatterAllGather(
        const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize, uint32_t dataTypeSize)
    {
        CHK_PRT_RET(param.rankSize != CUSTOM_BROADCAST_RANK_SIZE,
            HCCL_ERROR("[RunScatterAllGather] Need %u ranks", CUSTOM_BROADCAST_RANK_SIZE), HCCL_E_PARA);
        CHK_PRT_RET(resCtx.threads.size() < CUSTOM_RANKS_PER_SERVER,
            HCCL_ERROR("[RunScatterAllGather] Need %u threads", CUSTOM_RANKS_PER_SERVER), HCCL_E_NOT_FOUND);

        ScatterTopology topology;
        CHK_RET(BuildScatterTopology(param, topology));

        const ThreadHandle mainThread = resCtx.threads[0];
        const uint64_t bufferCapacity = GetSharedBufferCapacity(resCtx);
        const uint64_t chunkCapacity = bufferCapacity - (bufferCapacity % dataTypeSize);
        const uint64_t minChunkSize = CUSTOM_RANKS_PER_SERVER * static_cast<uint64_t>(dataTypeSize);
        CHK_PRT_RET(chunkCapacity < minChunkSize,
            HCCL_ERROR("[RunScatterAllGather] Communication buffer is too small"), HCCL_E_MEMORY);

        uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
        uint8_t *output = static_cast<uint8_t *>(param.outputPtr);

        const ChannelInfo *rootChannel = nullptr;
        if (param.myRank != param.root) {
            rootChannel = FindChannel(resCtx, param.root);
            CHK_PRT_RET(rootChannel == nullptr,
                HCCL_ERROR("[RunScatterAllGather] Root channel not found"), HCCL_E_NOT_FOUND);
        }

        for (uint64_t dataOffset = 0; dataOffset < dataSize;) {
            const uint64_t chunkSize = GetScatterChunkSize(dataSize - dataOffset, chunkCapacity, dataTypeSize);

            if (param.myRank == param.root) {
                // Scatter through the high-bandwidth Clos links. The seven remote owners relay
                // their shards back to the matching owners in the root server.
                for (uint32_t index = 1; index < CUSTOM_RANKS_PER_SERVER; ++index) {
                    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[index], 0));
                }

                const uint64_t shardZeroSize = GetShardSize(chunkSize, 0, dataTypeSize);
                CHK_RET(HcommLocalCopyOnThread(mainThread, localBuffer, input + dataOffset, shardZeroSize));

                for (uint32_t shardIndex = 1; shardIndex < CUSTOM_RANKS_PER_SERVER; ++shardIndex) {
                    const ChannelInfo *remoteOwner = FindChannel(resCtx, topology.remoteOwners[shardIndex]);
                    CHK_PRT_RET(remoteOwner == nullptr,
                        HCCL_ERROR("[RunScatterAllGather] Remote owner[%u] not found", shardIndex),
                        HCCL_E_NOT_FOUND);

                    const uint64_t shardOffset = GetShardOffset(chunkSize, shardIndex, dataTypeSize);
                    const uint64_t shardSize = GetShardSize(chunkSize, shardIndex, dataTypeSize);
                    const ThreadHandle worker = resCtx.threads[shardIndex];
                    CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
                    CHK_RET(HcommWriteWithNotifyOnThread(worker, remoteOwner->handle,
                        GetRemoteBuffer(*remoteOwner, shardOffset), input + dataOffset + shardOffset, shardSize,
                        DISTRIBUTE_DATA_NOTIFY));
                }

                const ChannelInfo *remoteOwnerZero = FindChannel(resCtx, topology.remoteOwners[0]);
                CHK_PRT_RET(remoteOwnerZero == nullptr,
                    HCCL_ERROR("[RunScatterAllGather] Remote owner zero not found"), HCCL_E_NOT_FOUND);
                CHK_RET(HcommWriteWithNotifyOnThread(mainThread, remoteOwnerZero->handle,
                    remoteOwnerZero->remoteCclMem.addr, input + dataOffset, shardZeroSize, DISTRIBUTE_DATA_NOTIFY));

                // Every non-root rank owns one staged shard. Reusing ACK is safe because all
                // distribution ACKs are consumed before any rank can issue its final ACK.
                for (const ChannelInfo &channel : resCtx.channels) {
                    CHK_RET(HcommChannelNotifyWaitOnThread(
                        mainThread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
                }
                for (const ChannelInfo &channel : resCtx.channels) {
                    CHK_RET(HcommChannelNotifyRecordOnThread(
                        mainThread, channel.handle, GLOBAL_READY_NOTIFY));
                }
                for (const ChannelInfo &channel : resCtx.channels) {
                    CHK_RET(HcommChannelNotifyWaitOnThread(
                        mainThread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
                }
            } else {
                const ChannelInfo *distributionSource = rootChannel;
                if (topology.inRootServer) {
                    distributionSource = FindChannel(resCtx, topology.remoteOwners[topology.ownerIndex]);
                    CHK_PRT_RET(distributionSource == nullptr,
                        HCCL_ERROR("[RunScatterAllGather] Relay owner not found"), HCCL_E_NOT_FOUND);
                }
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    mainThread, distributionSource->handle, DISTRIBUTE_DATA_NOTIFY, CUSTOM_TIMEOUT));

                if (!topology.inRootServer && topology.ownerIndex != 0) {
                    const ChannelInfo *rootOwner = FindChannel(resCtx, topology.rootOwners[topology.ownerIndex]);
                    CHK_PRT_RET(rootOwner == nullptr,
                        HCCL_ERROR("[RunScatterAllGather] Root-server owner not found"), HCCL_E_NOT_FOUND);
                    const uint64_t ownOffset = GetShardOffset(chunkSize, topology.ownerIndex, dataTypeSize);
                    const uint64_t ownSize = GetShardSize(chunkSize, topology.ownerIndex, dataTypeSize);
                    CHK_RET(HcommWriteWithNotifyOnThread(mainThread, rootOwner->handle,
                        GetRemoteBuffer(*rootOwner, ownOffset), localBuffer + ownOffset, ownSize,
                        DISTRIBUTE_DATA_NOTIFY));
                }

                CHK_RET(HcommChannelNotifyRecordOnThread(
                    mainThread, rootChannel->handle, NOTIFY_IDX_ACK));
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    mainThread, rootChannel->handle, GLOBAL_READY_NOTIFY, CUSTOM_TIMEOUT));

                // The allgather reads only same-server owners. Each worker uses a different peer,
                // so the one-channel-per-peer contest constraint remains satisfied.
                for (uint32_t index = 1; index < CUSTOM_RANKS_PER_SERVER; ++index) {
                    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[index], 0));
                }

                uint32_t threadIndex = 1;
                for (uint32_t shardIndex = 0; shardIndex < CUSTOM_RANKS_PER_SERVER; ++shardIndex) {
                    if (shardIndex == topology.ownerIndex) {
                        continue;
                    }
                    const ChannelInfo *owner = FindChannel(resCtx, topology.localOwners[shardIndex]);
                    CHK_PRT_RET(owner == nullptr,
                        HCCL_ERROR("[RunScatterAllGather] Local owner[%u] not found", shardIndex),
                        HCCL_E_NOT_FOUND);

                    const uint64_t shardOffset = GetShardOffset(chunkSize, shardIndex, dataTypeSize);
                    const uint64_t shardSize = GetShardSize(chunkSize, shardIndex, dataTypeSize);
                    const ThreadHandle worker = resCtx.threads[threadIndex];
                    CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
                    CHK_RET(HcommReadOnThread(worker, owner->handle,
                        output + dataOffset + shardOffset, GetRemoteBuffer(*owner, shardOffset), shardSize));
                    CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, threadIndex));
                    ++threadIndex;
                }

                const uint64_t ownOffset = GetShardOffset(chunkSize, topology.ownerIndex, dataTypeSize);
                const uint64_t ownSize = GetShardSize(chunkSize, topology.ownerIndex, dataTypeSize);
                CHK_RET(HcommLocalCopyOnThread(
                    mainThread, output + dataOffset + ownOffset, localBuffer + ownOffset, ownSize));

                for (uint32_t index = 1; index < CUSTOM_RANKS_PER_SERVER; ++index) {
                    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, index, CUSTOM_TIMEOUT));
                }
                CHK_RET(HcommChannelNotifyRecordOnThread(
                    mainThread, rootChannel->handle, NOTIFY_IDX_ACK));
            }

            dataOffset += chunkSize;
        }
        return HCCL_SUCCESS;
    }
    uint64_t GetPipelinePartOffset(
        uint64_t chunkSize, uint32_t partIndex, uint32_t dataTypeSize, uint32_t stageNum)
    {
        const uint64_t elementCount = chunkSize / dataTypeSize;
        const uint64_t baseCount = elementCount / stageNum;
        const uint64_t remainder = elementCount % stageNum;
        const uint64_t elementOffset = partIndex * baseCount + std::min<uint64_t>(partIndex, remainder);
        return elementOffset * dataTypeSize;
    }

    uint64_t GetPipelinePartSize(
        uint64_t chunkSize, uint32_t partIndex, uint32_t dataTypeSize, uint32_t stageNum)
    {
        const uint64_t elementCount = chunkSize / dataTypeSize;
        const uint64_t baseCount = elementCount / stageNum;
        const uint64_t remainder = elementCount % stageNum;
        return (baseCount + (partIndex < remainder ? 1 : 0)) * dataTypeSize;
    }

    uint64_t GetPipelineChunkSize(
        uint64_t remainingSize, uint64_t chunkCapacity, uint32_t dataTypeSize, uint32_t stageNum)
    {
        uint64_t chunkSize = std::min(chunkCapacity, remainingSize);
        const uint64_t minChunkSize = stageNum * CUSTOM_RANKS_PER_SERVER
            * static_cast<uint64_t>(dataTypeSize);
        if (remainingSize > chunkCapacity && remainingSize - chunkSize < minChunkSize) {
            chunkSize -= minChunkSize - (remainingSize - chunkSize);
        }
        return chunkSize;
    }

    HcclResult RunPipelinedScatterAllGather(
        const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize, uint32_t dataTypeSize)
    {
        constexpr uint32_t READY_NOTIFY_BASE = 0;
        constexpr uint32_t COORDINATOR_THREAD_BASE = CUSTOM_BROADCAST_RANK_SIZE;
        constexpr uint32_t COORDINATOR_START_NOTIFY = CUSTOM_THREAD_NOTIFY_NUM - 1;
        const uint32_t pipelineStageNum = dataSize < CUSTOM_BROADCAST_DEEP_PIPELINE_THRESHOLD
            ? CUSTOM_BROADCAST_MEDIUM_PIPELINE_STAGE_NUM
            : CUSTOM_BROADCAST_PIPELINE_STAGE_NUM;
        const uint32_t ackNotifyBase = pipelineStageNum;

        CHK_PRT_RET(param.rankSize != CUSTOM_BROADCAST_RANK_SIZE,
            HCCL_ERROR("[RunPipelinedScatterAllGather] Need %u ranks", CUSTOM_BROADCAST_RANK_SIZE), HCCL_E_PARA);
        CHK_PRT_RET(resCtx.threads.size() < CUSTOM_ALGORITHM_THREAD_NUM,
            HCCL_ERROR("[RunPipelinedScatterAllGather] Need %u threads", CUSTOM_ALGORITHM_THREAD_NUM),
            HCCL_E_NOT_FOUND);

        ScatterTopology topology;
        CHK_RET(BuildScatterTopology(param, topology));

        const ThreadHandle mainThread = resCtx.threads[0];
        const uint64_t bufferCapacity = GetSharedBufferCapacity(resCtx);
        const uint64_t chunkCapacity = bufferCapacity - (bufferCapacity % dataTypeSize);
        const uint64_t minChunkSize = pipelineStageNum * CUSTOM_RANKS_PER_SERVER
            * static_cast<uint64_t>(dataTypeSize);
        CHK_PRT_RET(chunkCapacity < minChunkSize,
            HCCL_ERROR("[RunPipelinedScatterAllGather] Communication buffer is too small"), HCCL_E_MEMORY);

        uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);
        uint8_t *output = static_cast<uint8_t *>(param.outputPtr);

        const ChannelInfo *rootChannel = nullptr;
        if (param.myRank != param.root) {
            rootChannel = FindChannel(resCtx, param.root);
            CHK_PRT_RET(rootChannel == nullptr,
                HCCL_ERROR("[RunPipelinedScatterAllGather] Root channel not found"), HCCL_E_NOT_FOUND);
        }

        for (uint64_t dataOffset = 0; dataOffset < dataSize;) {
            const uint64_t chunkSize = GetPipelineChunkSize(
                dataSize - dataOffset, chunkCapacity, dataTypeSize, pipelineStageNum);

            if (param.myRank == param.root) {
                for (uint32_t partIndex = 0; partIndex < pipelineStageNum; ++partIndex) {
                    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread,
                        resCtx.threads[COORDINATOR_THREAD_BASE + partIndex], COORDINATOR_START_NOTIFY));
                }
                for (uint32_t workerIndex = 1; workerIndex < CUSTOM_BROADCAST_RANK_SIZE; ++workerIndex) {
                    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[workerIndex], 0));
                }

                for (uint32_t partIndex = 0; partIndex < pipelineStageNum; ++partIndex) {
                    const uint64_t partOffset = GetPipelinePartOffset(chunkSize, partIndex, dataTypeSize, pipelineStageNum);
                    const uint64_t partSize = GetPipelinePartSize(chunkSize, partIndex, dataTypeSize, pipelineStageNum);
                    const ThreadHandle coordinator = resCtx.threads[COORDINATOR_THREAD_BASE + partIndex];

                    uint32_t workerIndex = 1;
                    for (uint32_t shardIndex = 0; shardIndex < CUSTOM_RANKS_PER_SERVER; ++shardIndex) {
                        const uint64_t shardOffset = GetShardOffset(partSize, shardIndex, dataTypeSize);
                        const uint64_t shardSize = GetShardSize(partSize, shardIndex, dataTypeSize);
                        const uint64_t bufferOffset = partOffset + shardOffset;

                        if (shardIndex != 0) {
                            const ChannelInfo *rootOwner = FindChannel(resCtx, topology.rootOwners[shardIndex]);
                            CHK_PRT_RET(rootOwner == nullptr,
                                HCCL_ERROR("[RunPipelinedScatterAllGather] Root owner[%u] not found", shardIndex),
                                HCCL_E_NOT_FOUND);
                            const ThreadHandle worker = resCtx.threads[workerIndex];
                            if (partIndex == 0) {
                                CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
                            }
                            CHK_RET(HcommWriteOnThread(worker, rootOwner->handle,
                                GetRemoteBuffer(*rootOwner, bufferOffset),
                                input + dataOffset + bufferOffset, shardSize));
                            CHK_RET(HcommThreadNotifyRecordOnThread(worker, coordinator, workerIndex));
                            ++workerIndex;
                        }

                        const ChannelInfo *remoteOwner = FindChannel(resCtx, topology.remoteOwners[shardIndex]);
                        CHK_PRT_RET(remoteOwner == nullptr,
                            HCCL_ERROR("[RunPipelinedScatterAllGather] Remote owner[%u] not found", shardIndex),
                            HCCL_E_NOT_FOUND);
                        const ThreadHandle worker = resCtx.threads[workerIndex];
                        if (partIndex == 0) {
                            CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
                        }
                        CHK_RET(HcommWriteOnThread(worker, remoteOwner->handle,
                            GetRemoteBuffer(*remoteOwner, bufferOffset),
                            input + dataOffset + bufferOffset, shardSize));
                        CHK_RET(HcommThreadNotifyRecordOnThread(worker, coordinator, workerIndex));
                        ++workerIndex;
                    }

                    const uint64_t ownSize = GetShardSize(partSize, 0, dataTypeSize);
                    CHK_RET(HcommLocalCopyOnThread(mainThread, localBuffer + partOffset,
                        input + dataOffset + partOffset, ownSize));
                    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, coordinator, 0));

                    CHK_RET(HcommThreadNotifyWaitOnThread(
                        coordinator, COORDINATOR_START_NOTIFY, CUSTOM_TIMEOUT));
                    for (uint32_t index = 0; index < CUSTOM_BROADCAST_RANK_SIZE; ++index) {
                        CHK_RET(HcommThreadNotifyWaitOnThread(coordinator, index, CUSTOM_TIMEOUT));
                    }
                    for (const ChannelInfo &channel : resCtx.channels) {
                        CHK_RET(HcommChannelNotifyRecordOnThread(
                            coordinator, channel.handle, READY_NOTIFY_BASE + partIndex));
                    }
                }

                for (uint32_t partIndex = 0; partIndex < pipelineStageNum; ++partIndex) {
                    for (const ChannelInfo &channel : resCtx.channels) {
                        CHK_RET(HcommChannelNotifyWaitOnThread(
                            mainThread, channel.handle, ackNotifyBase + partIndex, CUSTOM_TIMEOUT));
                    }
                }
            } else {
                for (uint32_t partIndex = 0; partIndex < pipelineStageNum; ++partIndex) {
                    const uint64_t partOffset = GetPipelinePartOffset(chunkSize, partIndex, dataTypeSize, pipelineStageNum);
                    const uint64_t partSize = GetPipelinePartSize(chunkSize, partIndex, dataTypeSize, pipelineStageNum);
                    const ThreadHandle coordinator = resCtx.threads[COORDINATOR_THREAD_BASE + partIndex];

                    CHK_RET(HcommChannelNotifyWaitOnThread(
                        mainThread, rootChannel->handle, READY_NOTIFY_BASE + partIndex, CUSTOM_TIMEOUT));
                    CHK_RET(HcommThreadNotifyRecordOnThread(
                        mainThread, coordinator, COORDINATOR_START_NOTIFY));
                    for (uint32_t workerIndex = 1; workerIndex < CUSTOM_RANKS_PER_SERVER; ++workerIndex) {
                        CHK_RET(HcommThreadNotifyRecordOnThread(
                            mainThread, resCtx.threads[workerIndex], partIndex));
                    }

                    uint32_t workerIndex = 1;
                    for (uint32_t shardIndex = 0; shardIndex < CUSTOM_RANKS_PER_SERVER; ++shardIndex) {
                        if (shardIndex == topology.ownerIndex) {
                            continue;
                        }
                        const ChannelInfo *owner = FindChannel(resCtx, topology.localOwners[shardIndex]);
                        CHK_PRT_RET(owner == nullptr,
                            HCCL_ERROR("[RunPipelinedScatterAllGather] Local owner[%u] not found", shardIndex),
                            HCCL_E_NOT_FOUND);

                        const uint64_t shardOffset = GetShardOffset(partSize, shardIndex, dataTypeSize);
                        const uint64_t shardSize = GetShardSize(partSize, shardIndex, dataTypeSize);
                        const uint64_t bufferOffset = partOffset + shardOffset;
                        const ThreadHandle worker = resCtx.threads[workerIndex];
                        CHK_RET(HcommThreadNotifyWaitOnThread(worker, partIndex, CUSTOM_TIMEOUT));
                        CHK_RET(HcommReadOnThread(worker, owner->handle,
                            output + dataOffset + bufferOffset,
                            GetRemoteBuffer(*owner, bufferOffset), shardSize));
                        CHK_RET(HcommThreadNotifyRecordOnThread(worker, coordinator, workerIndex));
                        ++workerIndex;
                    }

                    const uint64_t ownOffset = GetShardOffset(partSize, topology.ownerIndex, dataTypeSize);
                    const uint64_t ownSize = GetShardSize(partSize, topology.ownerIndex, dataTypeSize);
                    const uint64_t ownBufferOffset = partOffset + ownOffset;
                    CHK_RET(HcommLocalCopyOnThread(mainThread,
                        output + dataOffset + ownBufferOffset, localBuffer + ownBufferOffset, ownSize));
                    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, coordinator, 0));

                    CHK_RET(HcommThreadNotifyWaitOnThread(
                        coordinator, COORDINATOR_START_NOTIFY, CUSTOM_TIMEOUT));
                    for (uint32_t index = 0; index < CUSTOM_RANKS_PER_SERVER; ++index) {
                        CHK_RET(HcommThreadNotifyWaitOnThread(coordinator, index, CUSTOM_TIMEOUT));
                    }
                    CHK_RET(HcommChannelNotifyRecordOnThread(
                        coordinator, rootChannel->handle, ackNotifyBase + partIndex));
                }
            }

            dataOffset += chunkSize;
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_RET(ValidateResources(param, resCtx));
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);

    const uint32_t dataTypeSize = GetDataTypeSize(param.dataType);
    CHK_PRT_RET(dataTypeSize == 0,
        HCCL_ERROR("[ExecOp] Unsupported data type[%d]", static_cast<int32_t>(param.dataType)), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("[ExecOp] Data size overflow"), HCCL_E_PARA);
    const uint64_t dataSize = param.count * dataTypeSize;

    if (param.rankSize != CUSTOM_BROADCAST_RANK_SIZE || dataSize < CUSTOM_BROADCAST_STAR_THRESHOLD) {
        return RunPullStar(param, resCtx, dataSize);
    }
    if (dataSize < CUSTOM_BROADCAST_SCATTER_THRESHOLD) {
        return RunHierarchicalBroadcast(param, resCtx, dataSize);
    }
    if (dataSize < CUSTOM_BROADCAST_PIPELINE_THRESHOLD) {
        return RunScatterAllGather(param, resCtx, dataSize, dataTypeSize);
    }
    return RunPipelinedScatterAllGather(param, resCtx, dataSize, dataTypeSize);
}
} // namespace ops_hccl