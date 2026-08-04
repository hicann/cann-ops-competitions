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

#include <memory>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

enum class WorkloadCell : uint32_t {
    INVALID = 0,
    P16_512K = 0x1601,
    P16_512M = 0x1602,
    P16_400M4B = 0x1603,
    P4_512K = 0x0401,
    P4_512M = 0x0402,
    P4_400M4B = 0x0403,
    P12_512K = 0x1201,
    P12_512M = 0x1202,
    P12_400M4B = 0x1203,
};

enum class ScheduleKind : uint32_t {
    SLOT_GATHER = 1,
    P4_OUTPUT_ACCUMULATE = 3,
    DIRECT_MESH_SMALL = 6,
    CAPACITY_AWARE_PULL = 8,
};

enum class MeshStage : uint64_t {
    PARTIAL = 0,
    MERGE = 1,
};

constexpr uint32_t kReduceScatterSignature = 0x52534343; // "RSCC"

inline WorkloadCell ClassifyWorkload(uint32_t worldSize, uint64_t recvCount)
{
    if (worldSize == 16) {
        if (recvCount == 8192 || recvCount == 64) {
            return WorkloadCell::P16_512K;
        }
        if (recvCount == 8388608) {
            return WorkloadCell::P16_512M;
        }
        if (recvCount == 6553600) {
            return WorkloadCell::P16_400M4B;
        }
    } else if (worldSize == 4) {
        if (recvCount == 32768 || recvCount == 256) {
            return WorkloadCell::P4_512K;
        }
        if (recvCount == 33554432) {
            return WorkloadCell::P4_512M;
        }
        if (recvCount == 26214400) {
            return WorkloadCell::P4_400M4B;
        }
    } else if (worldSize == 12) {
        if (recvCount == 10923 || recvCount == 86) {
            return WorkloadCell::P12_512K;
        }
        if (recvCount == 11184768) {
            return WorkloadCell::P12_512M;
        }
        if (recvCount == 8738048) {
            return WorkloadCell::P12_400M4B;
        }
    }
    return WorkloadCell::INVALID;
}

inline uint32_t WorkloadEpoch(WorkloadCell cell)
{
    return static_cast<uint32_t>(cell);
}

inline const char *WorkloadKey(WorkloadCell cell)
{
    switch (cell) {
        case WorkloadCell::P16_512K:
            return "hccl_yga_rs_p16_512k_dedup2";
        case WorkloadCell::P16_512M:
            return "hccl_yga_rs_p16_512m_v1slot";
        case WorkloadCell::P16_400M4B:
            return "hccl_yga_rs_p16_400m4b_v1slot";
        case WorkloadCell::P4_512K:
            return "hccl_yga_rs_p4_512k_dedup4";
        case WorkloadCell::P4_512M:
            return "hccl_yga_rs_p4_512m_i3";
        case WorkloadCell::P4_400M4B:
            return "hccl_yga_rs_p4_400m4b_i3";
        case WorkloadCell::P12_512K:
            return "hccl_yga_rs_p12_512k_safev2";
        case WorkloadCell::P12_512M:
            return "hccl_yga_rs_p12_512m_v1slot";
        case WorkloadCell::P12_400M4B:
            return "hccl_yga_rs_p12_400m4b_v1slot";
        default:
            return nullptr;
    }
}

struct KernelLaneSpec {
    ChannelHandle lanes[MAX_RANK_SIZE];
    uint32_t laneCount;
};

struct ExchangeKernelSpec : public KernelLaneSpec {
    uint32_t remoteRanks[MAX_RANK_SIZE];
    uint32_t selfRank;
    uint32_t worldSize;
    uint64_t resultBytes;
    uint64_t stripeOffsets[3];
    uint64_t stripeBytes[3];
};

struct MergeKernelSpec : public KernelLaneSpec {
    uint32_t remoteCount;
};

struct TinyMeshKernelSpec : public KernelLaneSpec {
    uint32_t selfRank;
    uint32_t worldSize;
    uint32_t isLeader;
    uint32_t tileParallelism;
    uint64_t resultBytes;
    uint64_t blockBytes;
    HcclDataType scalarType;
    HcclReduceOp combineOp;
};

struct CapacityInboxSpec : public KernelLaneSpec {
    uint32_t remoteRanks[MAX_RANK_SIZE];
    uint32_t fusedOrder[MAX_RANK_SIZE];
    uint32_t inboxSlots[MAX_RANK_SIZE];
    uint32_t selfRank;
    uint32_t worldSize;
    uint32_t slotTotal;
    uint64_t resultBytes;
    uint64_t slotPitch;
    HcclDataType scalarType;
    HcclReduceOp combineOp;
};

struct CapacityFoldSpec : public KernelLaneSpec {
    uint32_t slotForRank[MAX_RANK_SIZE];
    uint32_t selfRank;
    uint32_t worldSize;
    uint32_t slotTotal;
    uint64_t resultBytes;
    uint64_t slotPitch;
    HcclDataType scalarType;
    HcclReduceOp combineOp;
};


struct DispatchContext {
    uint32_t magic = kReduceScatterSignature;
    uint32_t version = 0;
    WorkloadCell cell = WorkloadCell::INVALID;
    ScheduleKind algorithm = ScheduleKind::SLOT_GATHER;
    std::vector<ThreadHandle> workers; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> programs;
    std::vector<uint32_t> groupCuts;
    std::vector<uint32_t> groupRanks;
    uint64_t stagingBase = 0;
    uint64_t stagingBytes = 0;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << magic;
        binaryStream << version;
        binaryStream << cell;
        binaryStream << algorithm;
        binaryStream << workers;
        binaryStream << programs;
        binaryStream << groupCuts;
        binaryStream << groupRanks;
        binaryStream << stagingBase;
        binaryStream << stagingBytes;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> magic;
        binaryStream >> version;
        binaryStream >> cell;
        binaryStream >> algorithm;
        binaryStream >> workers;
        binaryStream >> programs;
        binaryStream >> groupCuts;
        binaryStream >> groupRanks;
        binaryStream >> stagingBase;
        binaryStream >> stagingBytes;
    }
};

#endif // OPS_HCCL_CUSTOM_H
