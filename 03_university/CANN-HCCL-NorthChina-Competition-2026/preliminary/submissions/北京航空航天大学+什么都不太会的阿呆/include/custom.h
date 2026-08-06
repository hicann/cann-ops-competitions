/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ALLGATHER_COMPETITION_CUSTOM_H
#define HCCL_ALLGATHER_COMPETITION_CUSTOM_H

#include <cstdint>
#include <vector>

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_RESERVED;
    uint32_t bandwidthCoeff = 1;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
    CommBuffer remoteInput;
    CommBuffer remoteOutput;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread; ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer;   ///< 本端HCCL通信内存
    CommBuffer registeredInput;
    CommBuffer registeredOutput;
    uint64_t chunkSize = 0;
    uint32_t serverRankSize = 0;
    std::vector<ThreadHandle> threads; ///< AICPU_TS通信引擎上的thread资源
    std::vector<ChannelInfo> intraChannels;
    ChannelInfo interChannel;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << registeredInput;
        binaryStream << registeredOutput;
        binaryStream << chunkSize;
        binaryStream << serverRankSize;
        binaryStream << threads;
        binaryStream << intraChannels;
        binaryStream << interChannel;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> registeredInput;
        binaryStream >> registeredOutput;
        binaryStream >> chunkSize;
        binaryStream >> serverRankSize;
        binaryStream >> threads;
        binaryStream >> intraChannels;
        binaryStream >> interChannel;
    }
};

#endif // HCCL_ALLGATHER_COMPETITION_CUSTOM_H
