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

constexpr uint32_t ALLREDUCE_CTX_MAGIC = 0x41524343;
constexpr uint32_t ALLREDUCE_CTX_VERSION = 8;
constexpr uint32_t MAX_CCU_DIE_GROUPS = 2;
constexpr uint32_t CCU_THREAD_NOTIFY_NUM = 5;
constexpr uint32_t CCU_NOTIFY_START = 0;
constexpr uint32_t CCU_NOTIFY_PUBLISH_DONE = 1;
constexpr uint32_t CCU_NOTIFY_PARTIAL_DONE = 2;
constexpr uint32_t CCU_NOTIFY_RESULT_READY = 3;
constexpr uint32_t CCU_NOTIFY_ALL_DONE = 4;

enum class AllReduceAlgorithm : uint32_t {
    HOSTSYNC_TREE_MESH_RS_AG = 0,
};

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

struct AllReduceKernelArg : public CcuKernelArgBase {
    uint32_t remoteRanks[MAX_RANK_SIZE]{};
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceType = HCCL_REDUCE_RESERVED;
    uint32_t groupIndex = 0;
    uint32_t groupCount = 0;
    uint32_t includesLocalRank = 0;
    uint32_t sourceCount = 0;
    uint32_t scratchBaseSlot = 0;
    uint32_t partialSlotIndices[MAX_CCU_DIE_GROUPS]{};
    uint32_t outputRootUsesLocalInput = 0;
};

struct SequentialScratchReuseKernelArg : public AllReduceKernelArg {
    uint32_t scratchSlotCount = 0;
};

struct OutputRootKernelArg : public AllReduceKernelArg {
};

constexpr uint32_t SMALL4_CTX_MAGIC = 0x53345041;
constexpr uint32_t SMALL4_CTX_VERSION = 1;

struct Small4KernelArg : public CcuKernelArgBase {
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceType = HCCL_REDUCE_RESERVED;
    uint32_t inPlace = 0;
};

enum Small4TaskArgIndex : uint32_t {
    SMALL4_ARG_INPUT_ADDR = 0,
    SMALL4_ARG_OUTPUT_ADDR,
    SMALL4_ARG_INPUT_TOKEN,
    SMALL4_ARG_OUTPUT_TOKEN,
    SMALL4_ARG_SCRATCH_ADDR,
    SMALL4_ARG_SCRATCH_TOKEN,
    SMALL4_ARG_BYTES,
    SMALL4_ARG_COUNT,
};

struct Small4ResourceCtx {
    uint32_t magic = SMALL4_CTX_MAGIC;
    uint32_t version = SMALL4_CTX_VERSION;
    uint32_t rankSize = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceType = HCCL_REDUCE_RESERVED;
    CommBuffer localBuffer;
    std::vector<uint32_t> kernelDieIds;
    std::vector<CcuKernelHandle> ccuKernels;

    std::vector<char> Serialize() const
    {
        BinaryStream binaryStream;
        binaryStream << magic;
        binaryStream << version;
        binaryStream << rankSize;
        binaryStream << dataType;
        binaryStream << reduceType;
        binaryStream << localBuffer;
        binaryStream << kernelDieIds;
        binaryStream << ccuKernels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }
};

enum MeshPhase : uint64_t {
    MESH_PHASE_PUBLISH = 0,
    MESH_PHASE_REDUCE_PARTIAL,
    MESH_PHASE_COMBINE,
    MESH_PHASE_FINISH,
    MESH_PHASE_SINGLE_DIE_ALL,
};

enum MeshTaskArgIndex : uint32_t {
    MESH_ARG_INPUT_ADDR = 0,
    MESH_ARG_OUTPUT_ADDR,
    MESH_ARG_INPUT_TOKEN,
    MESH_ARG_OUTPUT_TOKEN,
    MESH_ARG_SCRATCH_ADDR,
    MESH_ARG_SCRATCH_TOKEN,
    MESH_ARG_MY_SLICE_OFFSET,
    MESH_ARG_MY_SLICE_BYTES,
    MESH_ARG_IN_PLACE,
    MESH_ARG_NEEDS_HANDSHAKE,
    MESH_ARG_PHASE,
    MESH_ARG_COUNT,
};

struct AlgResourceCtx {
    uint32_t magic = ALLREDUCE_CTX_MAGIC;
    uint32_t version = ALLREDUCE_CTX_VERSION;
    uint32_t rankSize = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceType = HCCL_REDUCE_RESERVED;
    CommBuffer localBuffer;
    uint32_t scratchSlotCount = 0;
    uint32_t sequentialScratchReuse = 0;
    ThreadHandle dualDieThread = 0;
    std::vector<uint32_t> channelLayers;
    std::vector<uint32_t> kernelDieIds;
    std::vector<CcuKernelHandle> ccuKernels;

    std::vector<char> Serialize() const
    {
        BinaryStream binaryStream;
        binaryStream << magic;
        binaryStream << version;
        binaryStream << rankSize;
        binaryStream << dataType;
        binaryStream << reduceType;
        binaryStream << localBuffer;
        binaryStream << scratchSlotCount;
        binaryStream << sequentialScratchReuse;
        binaryStream << dualDieThread;
        binaryStream << channelLayers;
        binaryStream << kernelDieIds;
        binaryStream << ccuKernels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> magic;
        binaryStream >> version;
        binaryStream >> rankSize;
        binaryStream >> dataType;
        binaryStream >> reduceType;
        binaryStream >> localBuffer;
        binaryStream >> scratchSlotCount;
        binaryStream >> sequentialScratchReuse;
        binaryStream >> dualDieThread;
        binaryStream >> channelLayers;
        binaryStream >> kernelDieIds;
        binaryStream >> ccuKernels;
    }
};

inline bool GetCcuDataTypeSize(HcclDataType dataType, uint32_t &size)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_BFP16:
            size = 2;
            return true;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_FP32:
            size = 4;
            return true;
        case HCCL_DATA_TYPE_UINT8:
            size = 1;
            return true;
        default:
            size = 0;
            return false;
    }
}

inline bool IsCcuReduceOpSupported(HcclReduceOp reduceType)
{
    return reduceType == HCCL_REDUCE_SUM || reduceType == HCCL_REDUCE_MAX || reduceType == HCCL_REDUCE_MIN;
}

#endif // OPS_HCCL_CUSTOM_H
