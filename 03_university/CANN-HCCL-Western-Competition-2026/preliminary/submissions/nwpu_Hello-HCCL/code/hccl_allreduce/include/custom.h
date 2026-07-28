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

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr = nullptr;
    uint64_t size = 0;
} CommBuffer;

constexpr uint32_t ALG_RESOURCE_CTX_SCHEMA_VERSION = 1;
constexpr uint64_t SMALL_BUTTERFLY_THRESHOLD_BYTES = 512ULL * 1024ULL;
constexpr uint64_t SMALL_RABENSEIFNER_MIN_BYTES = 256ULL * 1024ULL;
constexpr uint32_t GROUPED_RS_REDUCE_GROUP_NUM = 4;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    uint32_t netLayer = INVALID_VALUE_RANKID;
    CommProtocol protocol = COMM_PROTOCOL_RESERVED;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem{};
};

struct AlgResourceCtx {
    uint32_t schemaVersion = ALG_RESOURCE_CTX_SCHEMA_VERSION;
    uint32_t localRankIndex = INVALID_VALUE_RANKID;
    uint32_t pairRank = INVALID_VALUE_RANKID;
    uint32_t serverIndex = INVALID_VALUE_RANKID;
    uint32_t isLowerServer = 0;
    uint32_t localLayer = INVALID_VALUE_RANKID;
    uint32_t globalLayer = INVALID_VALUE_RANKID;
    uint64_t usableBufferSize = 0;
    ThreadHandle aicpuThread = 0;      ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer{};          ///< 本端HCCL通信内存
    std::vector<uint32_t> localRanks;  ///< 本Server内按全局rank升序排列的rank列表
    std::vector<ThreadHandle> threads; ///< AICPU_TS通信引擎上的thread资源
    std::vector<ChannelInfo> channels; ///< AICPU_TS通信引擎上的channel资源

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << schemaVersion;
        binaryStream << localRankIndex;
        binaryStream << pairRank;
        binaryStream << serverIndex;
        binaryStream << isLowerServer;
        binaryStream << localLayer;
        binaryStream << globalLayer;
        binaryStream << usableBufferSize;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << localRanks;
        binaryStream << threads;
        binaryStream << channels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> schemaVersion;
        binaryStream >> localRankIndex;
        binaryStream >> pairRank;
        binaryStream >> serverIndex;
        binaryStream >> isLowerServer;
        binaryStream >> localLayer;
        binaryStream >> globalLayer;
        binaryStream >> usableBufferSize;
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> localRanks;
        binaryStream >> threads;
        binaryStream >> channels;
    }
};

#endif // OPS_HCCL_CUSTOM_H