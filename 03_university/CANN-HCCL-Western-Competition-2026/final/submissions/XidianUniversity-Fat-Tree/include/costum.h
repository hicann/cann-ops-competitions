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

#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

constexpr uint32_t MAX_ALG_CHANNEL_NUM = MAX_RANK_SIZE - 1;
constexpr uint32_t HIGH_RADIX = 4;
constexpr uint64_t SMALL_DATA_BYTES = 512ULL * 1024ULL;
constexpr uint64_t HIERARCHICAL_BATCH_CHUNK_BYTES = 64ULL * 1024ULL * 1024ULL;
constexpr uint64_t PIPELINE_CHUNK_BYTES = 32ULL * 1024ULL * 1024ULL;
constexpr uint64_t ADAPTIVE_PIPELINE_THRESHOLD_BYTES = 448ULL * 1024ULL * 1024ULL;
constexpr uint32_t PIPELINE_DEPTH = 4;
constexpr uint32_t MAX_PIPELINE_DEPTH = 4;
constexpr uint32_t RHD_LOCAL_PEER_COUNT = 7;

constexpr uint32_t SelectEqualPipelineDepth(uint32_t rankSize, uint64_t dataBytes)
{
    if (dataBytes <= SMALL_DATA_BYTES || (rankSize != 12U && rankSize != 16U)) {
        return 1U;
    }
    if (rankSize == 12U) {
        return 4U;
    }
    return dataBytes > ADAPTIVE_PIPELINE_THRESHOLD_BYTES ? 4U : 3U;
}

enum class AllReduceAlgorithm : uint32_t {
    CLOS_RECURSIVE_DOUBLING = 0,
    CLOS_RING_RSAG = 1,
    CLOS_RECURSIVE_HALVING_DOUBLING = 2,
    CLOS_RADIX4X3_RSAG = 3,
    CLOS_RADIX4_RSAG = 4,
    CLOS_FULL_LANE_FLAT_OWNER_RSAG = 5,
};

enum class TopologyPhase : uint32_t {
    FULL_CLOS = 0,
    RHD_LOCAL_REDUCE = 1,
    RHD_CROSS_REDUCE = 2,
    RHD_LOCAL_GATHER = 3,
    RADIX_LOCAL_REDUCE = 4,
    RADIX_CROSS_REDUCE_GATHER = 5,
    RADIX_LOCAL_GATHER = 6,
    RHD_LOCAL_SHARED = 7,
    RADIX_LOCAL_SHARED = 8,
};

struct AllReduceKernelArg {
    ChannelHandle channels[MAX_ALG_CHANNEL_NUM] {};
    uint32_t remoteRanks[MAX_ALG_CHANNEL_NUM] {};
    uint32_t channelCount = 0;
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    AllReduceAlgorithm algorithm = AllReduceAlgorithm::CLOS_RECURSIVE_DOUBLING;
    TopologyPhase topologyPhase = TopologyPhase::FULL_CLOS;
    uint32_t pipelineDepth = 1;
    uint64_t dataBytes = 0;
    uint64_t segmentOffsets[MAX_RANK_SIZE] {};
    uint64_t segmentBytes[MAX_RANK_SIZE] {};
    uint64_t laneSegmentOffsets[2][MAX_RANK_SIZE] {};
    uint64_t laneSegmentBytes[2][MAX_RANK_SIZE] {};
    uint64_t laneBytes[2] {};
};

struct AlgResourceCtx {
    CommBuffer localBuffer;
    AllReduceAlgorithm algorithm = AllReduceAlgorithm::CLOS_RECURSIVE_DOUBLING;
    uint32_t rankSize = 0;
    uint64_t dataBytes = 0;
    ThreadHandle thread {};
    ThreadHandle slaveThread {};
    ThreadHandle gatherThread {};
    CcuKernelHandle kernel = 0;
    CcuKernelHandle phase0Kernel = 0;
    CcuKernelHandle phase1Kernel = 0;
    CcuKernelHandle phase2Kernel = 0;

    std::vector<char> Serialize() const
    {
        BinaryStream stream;
        stream << localBuffer;
        stream << algorithm;
        stream << rankSize;
        stream << dataBytes;
        stream << thread;
        stream << slaveThread;
        stream << gatherThread;
        stream << kernel;
        stream << phase0Kernel;
        stream << phase1Kernel;
        stream << phase2Kernel;
        std::vector<char> result;
        stream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream stream(data);
        stream >> localBuffer;
        stream >> algorithm;
        stream >> rankSize;
        stream >> dataBytes;
        stream >> thread;
        stream >> slaveThread;
        stream >> gatherThread;
        stream >> kernel;
        stream >> phase0Kernel;
        stream >> phase1Kernel;
        stream >> phase2Kernel;
    }
};

#endif // OPS_HCCL_CUSTOM_H
