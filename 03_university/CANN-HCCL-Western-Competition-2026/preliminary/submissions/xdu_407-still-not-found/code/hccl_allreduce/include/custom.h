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

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

constexpr uint32_t HCCL_CUSTOM_THREAD_NOTIFY_NUM = 6;

typedef struct {
    void *addr = nullptr;
    uint64_t size = 0;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

inline BinaryStream &operator<<(BinaryStream &stream, const CommBuffer &buffer)
{
    stream << buffer.addr;
    stream << buffer.size;
    return stream;
}

inline BinaryStream &operator>>(BinaryStream &stream, CommBuffer &buffer)
{
    stream >> buffer.addr;
    stream >> buffer.size;
    return stream;
}

inline BinaryStream &operator<<(BinaryStream &stream, const ChannelInfo &channel)
{
    stream << channel.remoteRank;
    stream << channel.notifyNum;
    stream << channel.handle;
    stream << channel.remoteCclMem;
    return stream;
}

inline BinaryStream &operator>>(BinaryStream &stream, ChannelInfo &channel)
{
    stream >> channel.remoteRank;
    stream >> channel.notifyNum;
    stream >> channel.handle;
    stream >> channel.remoteCclMem;
    return stream;
}

struct AlgResourceCtx {
    uint32_t rootRank = INVALID_VALUE_RANKID;
    std::vector<uint32_t> allRanks;
    std::vector<uint32_t> localRanks;
    ThreadHandle aicpuThread = 0;      ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< AICPU_TS通信引擎上的thread资源
    std::vector<ChannelInfo> channels; ///< AICPU_TS通信引擎上的channel资源

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << rootRank;
        binaryStream << allRanks;
        binaryStream << localRanks;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
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
        binaryStream >> rootRank;
        binaryStream >> allRanks;
        binaryStream >> localRanks;
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
    }

    const ChannelInfo *FindChannel(uint32_t remoteRank) const
    {
        for (const ChannelInfo &channel : channels) {
            if (channel.remoteRank == remoteRank) {
                return &channel;
            }
        }
        return nullptr;
    }
};

#endif // OPS_HCCL_CUSTOM_H