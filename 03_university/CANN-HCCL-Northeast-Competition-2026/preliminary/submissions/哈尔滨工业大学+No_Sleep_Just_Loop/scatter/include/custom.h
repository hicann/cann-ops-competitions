/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;                 // 主同步 thread（用于 Host/Device 同步）
    CommBuffer localBuffer;                    // 本地 CCL buffer（通信源/中转）
    std::vector<ThreadHandle> meshThreads;     // Server 内 Mesh 从线程（固定 2×8：7 个）
    std::vector<ThreadHandle> closThreads;     // 跨 Server Clos 线程（固定 2×8：1 个）
    std::vector<ChannelInfo> meshChannels;     // 同 Server peer channel（layer0，7 条）
    std::vector<ChannelInfo> closChannels;     // 跨 Server 对端 channel（layer1，1 条）

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread << localBuffer << meshThreads << closThreads << meshChannels << closChannels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread >> localBuffer >> meshThreads >> closThreads >> meshChannels >> closChannels;
    }
};

#endif // OPS_HCCL_CUSTOM_H
