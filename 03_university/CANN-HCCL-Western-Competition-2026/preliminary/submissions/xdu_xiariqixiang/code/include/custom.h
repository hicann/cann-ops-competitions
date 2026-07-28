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

// 通信算法类型枚举
enum class AllReduceAlgType : uint32_t {
    HIERARCHICAL_REDUCE_SCATTER_GATHER = 0, // 层次化 ReduceScatter + AllGather
};

// 链路类型枚举
enum class LinkType : uint32_t {
    INTRA_SERVER_MESH = 0, // Server 内 Full-Mesh 直连
    INTER_SERVER_CLOS = 1, // Server 间 Clos 网络
};

// 单个 rank 的分组信息
struct RankGroupInfo {
    uint32_t serverId;     // Server ID (0 或 1)
    uint32_t localRankId;  // Server 内 Rank ID (0 ~ 7)
    uint32_t globalRankId; // 全局 Rank ID (0 ~ 15)
    uint32_t rankSize;     // 全局总 Rank 数 (16)
    uint32_t numPerServer; // 每个 Server 的 NPU 数 (8)
};

// Host 侧用于临时存储拓扑解析结果的辅助结构
struct LinkInfo {
    uint32_t remoteRank;
    LinkType linkType;
};

// HCCL Buffer 分区管理
// 将 localBuffer 划分为以下区域:
//   [0, chunkBytes)                   : recvScratch  (接收暂存区)
//   [chunkBytes, 2*chunkBytes)        : crossScratch (跨服务器交换区)
//   [2*chunkBytes, ...)               : 预留
struct BufferPartition {
    void *recvScratch;   // ReduceScatter 阶段接收暂存区 (chunkBytes)
    void *crossScratch;  // 跨服务器交换暂存区 (chunkBytes)
    uint64_t chunkBytes; // 每个 chunk 的字节数
    uint64_t chunkCount; // 每个 chunk 的元素数
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;          // AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer;            // 本端HCCL通信内存
    std::vector<ThreadHandle> threads; // AICPU_TS通信引擎上的thread资源
    std::vector<ChannelInfo> channels; // AICPU_TS通信引擎上的channel资源
    RankGroupInfo rankGroupInfo;       // Rank 分组信息
    std::vector<LinkInfo> linkInfos;   // 拓扑链路信息

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << rankGroupInfo;
        binaryStream << linkInfos;
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
        binaryStream >> threads;
        binaryStream >> channels;
        binaryStream >> rankGroupInfo;
        binaryStream >> linkInfos;
    }
};

#endif // OPS_HCCL_CUSTOM_H
