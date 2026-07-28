/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root path of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include "hccl/hccl_types.h"
#include "hccl/hccl_res_expt.h"

#include "binary_stream.h"
#include "common.h"

// Complete data type size mapping (template's SIZE_TABLE only has FP32)
inline uint32_t GetDataTypeSize(HcclDataType dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:    return 1;
        case HCCL_DATA_TYPE_INT16:   return 2;
        case HCCL_DATA_TYPE_INT32:   return 4;
        case HCCL_DATA_TYPE_FP16:    return 2;
        case HCCL_DATA_TYPE_FP32:    return 4;
        case HCCL_DATA_TYPE_INT64:   return 8;
        case HCCL_DATA_TYPE_UINT64:  return 8;
        case HCCL_DATA_TYPE_UINT8:   return 1;
        case HCCL_DATA_TYPE_UINT16:  return 2;
        case HCCL_DATA_TYPE_UINT32:  return 4;
        case HCCL_DATA_TYPE_FP64:    return 8;
        case HCCL_DATA_TYPE_BFP16:   return 2;
        case HCCL_DATA_TYPE_INT128:  return 16;
        case HCCL_DATA_TYPE_HIF8:    return 2;
        case HCCL_DATA_TYPE_FP8E4M3: return 1;
        case HCCL_DATA_TYPE_FP8E5M2: return 1;
        case HCCL_DATA_TYPE_FP8E8M0: return 1;
        default:                     return 0;
    }
}

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
    ThreadHandle aicpuThread;
    CommBuffer localBuffer;
    std::vector<ThreadHandle> threads;
    std::vector<ChannelInfo> channels;

    // Ring AllReduce topology info
    uint32_t prevRank = INVALID_VALUE_RANKID;
    uint32_t nextRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t ringPos = 0;  // Position in the ring (may differ from myRank for custom ring topologies)

    // Hierarchical topology info (2-server x 8-NPU)
    bool useHierarchical = false;
    uint32_t serverRankSize = 0;
    uint32_t serverPrevRank = INVALID_VALUE_RANKID;
    uint32_t serverNextRank = INVALID_VALUE_RANKID;
    uint32_t serverPrevIdx = 0;
    uint32_t serverNextIdx = 0;
    uint32_t myServerIdx = 0;
    uint32_t crossServerPartnerRank = INVALID_VALUE_RANKID;
    bool hasCrossServerPartner = false;

    // Recursive Halving/Doubling topology info
    // 4 channels: [0]=XOR^1, [1]=XOR^2, [2]=XOR^4, [3]=cross-server
    bool useRecursiveHalving = false;
    uint32_t meshPartnerRanks[3] = {INVALID_VALUE_RANKID, INVALID_VALUE_RANKID, INVALID_VALUE_RANKID};
    uint32_t crossServerChIdx = 0;

    // Serialization
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << prevRank;
        binaryStream << nextRank;
        binaryStream << rankSize;
        binaryStream << myRank;
        binaryStream << ringPos;
        binaryStream << useHierarchical;
        binaryStream << serverRankSize;
        binaryStream << serverPrevRank;
        binaryStream << serverNextRank;
        binaryStream << serverPrevIdx;
        binaryStream << serverNextIdx;
        binaryStream << myServerIdx;
        binaryStream << crossServerPartnerRank;
        binaryStream << hasCrossServerPartner;
        binaryStream << useRecursiveHalving;
        binaryStream << meshPartnerRanks[0];
        binaryStream << meshPartnerRanks[1];
        binaryStream << meshPartnerRanks[2];
        binaryStream << crossServerChIdx;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // Deserialization
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
        binaryStream >> prevRank;
        binaryStream >> nextRank;
        binaryStream >> rankSize;
        binaryStream >> myRank;
        binaryStream >> ringPos;
        binaryStream >> useHierarchical;
        binaryStream >> serverRankSize;
        binaryStream >> serverPrevRank;
        binaryStream >> serverNextRank;
        binaryStream >> serverPrevIdx;
        binaryStream >> serverNextIdx;
        binaryStream >> myServerIdx;
        binaryStream >> crossServerPartnerRank;
        binaryStream >> hasCrossServerPartner;
        binaryStream >> useRecursiveHalving;
        binaryStream >> meshPartnerRanks[0];
        binaryStream >> meshPartnerRanks[1];
        binaryStream >> meshPartnerRanks[2];
        binaryStream >> crossServerChIdx;
    }
};

#endif // OPS_HCCL_CUSTOM_H
