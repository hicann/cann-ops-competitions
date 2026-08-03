/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <cstdint>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

constexpr uint32_t AR_RANKS = 16;
constexpr uint32_t AR_SERVERS = 2;
constexpr uint32_t AR_RANKS_PER_SERVER = 8;
constexpr uint64_t AR_FLOAT_BYTES = sizeof(float);
constexpr uint64_t AR_STRIPE_ALIGNMENT_BYTES = AR_RANKS_PER_SERVER * AR_FLOAT_BYTES;
constexpr uint64_t AR_STRIPED_THRESHOLD_BYTES = 64ULL * 1024;
constexpr uint64_t AR_PARALLEL_THRESHOLD_BYTES = 1ULL * 1024 * 1024;
constexpr uint32_t AR_PARALLEL_THREAD_NUM = AR_RANKS_PER_SERVER;
constexpr uint32_t AR_PARALLEL_PEER_NUM = AR_RANKS_PER_SERVER - 1U;
constexpr uint32_t AR_THREAD_NOTIFY_NUM = AR_PARALLEL_PEER_NUM;

constexpr uint32_t AR_NOTIFY_READY = 0;
constexpr uint32_t AR_NOTIFY_CONSUMED = 1;
constexpr uint32_t AR_NOTIFY_FINAL = 2;
constexpr uint32_t AR_NOTIFY_BROADCAST = 3;
constexpr uint32_t AR_NOTIFY_GATHER = 4;
constexpr uint32_t AR_CHANNEL_NOTIFY_NUM = 4;
constexpr uint32_t AR_PARALLEL_CHANNEL_NOTIFY_NUM = 5;

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem{nullptr, 0};
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread = 0;
    CommBuffer localBuffer{nullptr, 0};
    std::vector<ThreadHandle> threads;
    std::vector<ChannelInfo> channels;

    uint64_t cclBufferBytes = 0;
    uint64_t chunkCapacity = 0;
    uint64_t inputOffset = 0;
    uint64_t reductionOffset = 0;
    uint64_t outputOffset = 0;
    uint32_t serverOfRank[AR_RANKS]{};
    uint32_t laneOfRank[AR_RANKS]{};
    uint32_t rankOfServerLane[AR_SERVERS][AR_RANKS_PER_SERVER]{};
    uint32_t leaderRank[AR_SERVERS]{};
    uint32_t rootLeader = INVALID_VALUE_RANKID;

    std::vector<char> Serialize() const
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << cclBufferBytes;
        binaryStream << chunkCapacity;
        binaryStream << inputOffset;
        binaryStream << reductionOffset;
        binaryStream << outputOffset;
        for (uint32_t rank = 0; rank < AR_RANKS; ++rank) {
            binaryStream << serverOfRank[rank];
            binaryStream << laneOfRank[rank];
        }
        for (uint32_t server = 0; server < AR_SERVERS; ++server) {
            for (uint32_t lane = 0; lane < AR_RANKS_PER_SERVER; ++lane) {
                binaryStream << rankOfServerLane[server][lane];
            }
            binaryStream << leaderRank[server];
        }
        binaryStream << rootLeader;

        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
        binaryStream >> cclBufferBytes;
        binaryStream >> chunkCapacity;
        binaryStream >> inputOffset;
        binaryStream >> reductionOffset;
        binaryStream >> outputOffset;
        for (uint32_t rank = 0; rank < AR_RANKS; ++rank) {
            binaryStream >> serverOfRank[rank];
            binaryStream >> laneOfRank[rank];
        }
        for (uint32_t server = 0; server < AR_SERVERS; ++server) {
            for (uint32_t lane = 0; lane < AR_RANKS_PER_SERVER; ++lane) {
                binaryStream >> rankOfServerLane[server][lane];
            }
            binaryStream >> leaderRank[server];
        }
        binaryStream >> rootLeader;
    }
};

#endif // OPS_HCCL_CUSTOM_H
