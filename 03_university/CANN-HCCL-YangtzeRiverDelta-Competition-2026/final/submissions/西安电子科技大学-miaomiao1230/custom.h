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
#include <memory>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

constexpr uint64_t BROADCAST_LARGE_MESSAGE_THRESHOLD = 8ULL * 1024 * 1024;
// v33 merge: v27 4x1 chain pipeline segment count per 256 MiB host chunk.
constexpr uint32_t CHAIN4_PIPELINE_SEGMENT_COUNT = 16;

enum class BroadcastKernelMode : uint32_t {
    TREE = 0,
    MESH_SCATTER_ALLGATHER = 1,
    BIPARTITE_SCATTER_ALLGATHER = 2,
    HIERARCHICAL_INTRA_SCATTER = 3,
    HIERARCHICAL_INTER_TRANSFER = 4,
    HIERARCHICAL_INTRA_ALLGATHER = 5,
    CROSS_FIRST_INTER_SCATTER = 6,
    CROSS_FIRST_INTER_RETURN = 7,
    CROSS_FIRST_INTRA_ALLGATHER = 8,
    RELAY_INTER_TRANSFER = 9,
    TWO_SHOT_INTRA_BROADCAST = 10,
    TWO_SHOT_INTER_PAIRED = 11,
    PARALLEL_INTRA_SAG = 12,
    BALANCED_INTER_SCATTER_RECV = 13,
    BALANCED_INTER_FUSED = 14,
    // v33 merge: ported from v27 for the 4x1 large-message path.
    PIPELINED_CHAIN4 = 15,
};

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    BroadcastKernelMode mode = BroadcastKernelMode::TREE;
    uint32_t myRank = 0;
    uint32_t rankSize = 0;
    uint32_t root = 0;

    ChannelHandle parentChannel = 0;
    ChannelHandle childChannels[MAX_RANK_SIZE]{};
    uint32_t childCount = 0;
    uint32_t hasParent = 0;

    ChannelHandle peerChannels[MAX_RANK_SIZE]{};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t peerCount = 0;

    // Appended so the byte offsets of all teammate-proven hierarchy fields
    // remain unchanged.
    uint32_t childNeedsChunkNotify[MAX_RANK_SIZE]{};
    uint32_t pipelineFullChunks = 0;
    uint64_t pipelineChunkBytes = 0;
    uint64_t pipelineTailBytes = 0;
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64];
    // kernel函数
    void *kernelFunc;
    // KernelArg实例指针
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct AlgResourceCtx {
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> ccuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
    }
};

#endif // OPS_HCCL_CUSTOM_H
