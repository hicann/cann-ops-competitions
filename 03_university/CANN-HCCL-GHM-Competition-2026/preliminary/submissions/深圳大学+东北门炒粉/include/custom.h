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

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

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
    ThreadHandle aicpuThread;
    CommBuffer localBuffer;
    std::vector<ThreadHandle> threads;
    std::vector<ChannelInfo> channels;
    uint32_t hierarchyMode = 0;
    uint32_t localGroupIndex = 0;
    uint32_t localRankIndex = 0;
    uint32_t peerRank = INVALID_VALUE_RANKID;
    std::vector<uint32_t> group0Ranks;
    std::vector<uint32_t> group1Ranks;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << hierarchyMode;
        binaryStream << localGroupIndex;
        binaryStream << localRankIndex;
        binaryStream << peerRank;
        binaryStream << group0Ranks;
        binaryStream << group1Ranks;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
        binaryStream >> hierarchyMode;
        binaryStream >> localGroupIndex;
        binaryStream >> localRankIndex;
        binaryStream >> peerRank;
        binaryStream >> group0Ranks;
        binaryStream >> group1Ranks;
    }
};

namespace custom_reducescatter {

constexpr uint64_t kSliceAlignment = 128;
constexpr uint64_t kMaxTransferBytes = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t kSmallNhrMaxBytes = 32ULL * 1024ULL;
constexpr uint64_t kFlatTreeMinBytes = 1024ULL * 1024ULL;
constexpr uint32_t kChannelNotifyNum = 3;
constexpr uint32_t kHierarchyDisabled = 0;
constexpr uint32_t kHierarchyTwoServer = 1;
constexpr uint32_t kHierarchySmallNhr = 2;

constexpr uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? 0 : value / alignment * alignment;
}

constexpr uint64_t MaxSliceCount(uint64_t cclBufferBytes, uint32_t slotNum, uint32_t dataTypeBytes)
{
    if (slotNum == 0 || dataTypeBytes == 0) {
        return 0;
    }
    const uint64_t perSlotBytes = cclBufferBytes / slotNum;
    const uint64_t transferBytes = perSlotBytes < kMaxTransferBytes ? perSlotBytes : kMaxTransferBytes;
    return AlignDown(transferBytes, kSliceAlignment) / dataTypeBytes;
}

} // namespace custom_reducescatter

#endif // OPS_HCCL_CUSTOM_H
