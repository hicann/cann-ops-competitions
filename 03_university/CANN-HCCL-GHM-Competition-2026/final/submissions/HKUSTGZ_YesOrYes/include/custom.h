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

#include <array>
#include <memory>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
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

constexpr uint32_t HIERARCHY_MAX_TARGETS = 3;
constexpr uint32_t HIERARCHY_MAX_CROSS_PEERS = 2;
constexpr uint32_t HIERARCHY_ALL_CHUNKS = 0xffffffffU;
constexpr uint32_t MATE_CHUNK_COUNT = 2;
constexpr uint32_t MATE_MAX_PEERS = 8;
constexpr uint32_t MATE_LOCAL_KERNEL_MODE = 0xfffffff0U;
constexpr uint32_t MATE_CROSS_KERNEL_MODE = 0xfffffff1U;
constexpr uint32_t HYBRID_MATE_LOCAL_MODE = 0xffffffe0U;
constexpr uint32_t HYBRID_DIRECT_CROSS_MODE = 0xffffffe1U;
constexpr uint32_t HYBRID_DIRECT_LOCAL_MODE = 0xffffffe2U;
constexpr uint32_t HYBRID_MATE_CROSS_MODE = 0xffffffe3U;
constexpr uint32_t HETERO12_H8_MATE_LOCAL_MODE = 0xffffffc0U;
constexpr uint32_t HETERO12_H8_DIRECT_CROSS_MODE = 0xffffffc1U;
constexpr uint32_t HETERO12_H8_COMBINE_MODE = 0xffffffc2U;
constexpr uint32_t HETERO12_H8_PUBLISH_MODE = 0xffffffc3U;
constexpr uint32_t HETERO12_H4_LOCAL_PREFIX_MODE = 0xffffffc4U;
constexpr uint32_t HETERO12_H4_DIRECT_CROSS_MODE = 0xffffffc5U;
constexpr uint32_t HETERO12_H4_LOCAL_SUFFIX_MODE = 0xffffffc6U;
constexpr uint32_t HETERO12_H4_MATE_CROSS_MODE = 0xffffffc7U;
constexpr uint32_t WRITE_REDUCE_4X1_MODE = 0xffffffc8U;
constexpr uint32_t COMPACT_MS_OUTPUT_MODE = 0xffffffd0U;
constexpr uint32_t COMPACT_MS_SCRATCH_MODE = 0xffffffd1U;
constexpr uint32_t DIRECT_TILE_COUNT = 3;

struct HierarchyRankPlan {
    uint32_t localBegin{0};
    uint32_t localSize{0};
    uint32_t remoteBegin{0};
    uint32_t remoteSize{0};
    uint32_t localIndex{0};
    uint32_t targetCount{0};
    std::array<uint32_t, HIERARCHY_MAX_TARGETS> targetRanks{};
    std::array<uint32_t, HIERARCHY_MAX_TARGETS> targetActiveChunk{};
    uint32_t crossPeerCount{0};
    std::array<uint32_t, HIERARCHY_MAX_CROSS_PEERS> crossPeers{};
    std::array<uint32_t, 2> ownerPeers{};
    uint32_t remoteSlot{0};
};

// The final has exactly two dual-node communicator layouts.  This mapping
// balances the asymmetric 8+4 node across the minimum two pipeline chunks.
inline bool BuildHierarchyRankPlan(uint32_t rankSize, uint32_t myRank, HierarchyRankPlan &plan)
{
    if (myRank >= rankSize) {
        return false;
    }
    plan = HierarchyRankPlan{};
    plan.targetActiveChunk.fill(HIERARCHY_ALL_CHUNKS);

    if (rankSize == 16) {
        plan.localBegin = myRank < 8 ? 0 : 8;
        plan.localSize = 8;
        plan.remoteBegin = myRank < 8 ? 8 : 0;
        plan.remoteSize = 8;
    } else if (rankSize == 12) {
        if (myRank < 8) {
            plan.localBegin = 0;
            plan.localSize = 8;
            plan.remoteBegin = 8;
            plan.remoteSize = 4;
        } else {
            plan.localBegin = 8;
            plan.localSize = 4;
            plan.remoteBegin = 0;
            plan.remoteSize = 8;
        }
    } else {
        return false;
    }

    plan.localIndex = myRank - plan.localBegin;
    plan.targetRanks[plan.targetCount++] = myRank;
    if (plan.localSize >= plan.remoteSize) {
        plan.targetRanks[plan.targetCount] = plan.remoteBegin + plan.localIndex % plan.remoteSize;
        if (plan.localSize > plan.remoteSize) {
            plan.targetActiveChunk[plan.targetCount] = plan.localIndex / plan.remoteSize;
        }
        ++plan.targetCount;
    } else {
        for (uint32_t remoteIndex = plan.localIndex; remoteIndex < plan.remoteSize;
             remoteIndex += plan.localSize) {
            if (plan.targetCount >= HIERARCHY_MAX_TARGETS) {
                return false;
            }
            plan.targetRanks[plan.targetCount++] = plan.remoteBegin + remoteIndex;
        }
    }

    plan.ownerPeers[0] = plan.remoteBegin + plan.localIndex % plan.remoteSize;
    plan.ownerPeers[1] = plan.ownerPeers[0];
    if (plan.localSize < plan.remoteSize && plan.localIndex + plan.localSize < plan.remoteSize) {
        plan.ownerPeers[1] = plan.remoteBegin + plan.localIndex + plan.localSize;
    }
    plan.remoteSlot = plan.localSize > plan.remoteSize ? plan.localIndex / plan.remoteSize : 0;

    auto addCrossPeer = [&plan](uint32_t peer) {
        for (uint32_t i = 0; i < plan.crossPeerCount; ++i) {
            if (plan.crossPeers[i] == peer) {
                return true;
            }
        }
        if (plan.crossPeerCount >= HIERARCHY_MAX_CROSS_PEERS) {
            return false;
        }
        plan.crossPeers[plan.crossPeerCount++] = peer;
        return true;
    };
    if (!addCrossPeer(plan.ownerPeers[0]) || !addCrossPeer(plan.ownerPeers[1])) {
        return false;
    }
    for (uint32_t i = 1; i < plan.targetCount; ++i) {
        if (!addCrossPeer(plan.targetRanks[i])) {
            return false;
        }
    }
    if (plan.crossPeerCount == 2 && plan.crossPeers[0] > plan.crossPeers[1]) {
        std::swap(plan.crossPeers[0], plan.crossPeers[1]);
    }
    return true;
}

struct AlgResourceCtx {
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<uint32_t> stripeCounts;
    CcuKernelHandle combineKernel{0};
    uint64_t localBufferToken{0};
    uint64_t cachedInputAddress{0};
    uint64_t cachedInputSize{0};
    uint64_t cachedInputToken{0};
    uint64_t cachedOutputAddress{0};
    uint64_t cachedOutputSize{0};
    uint64_t cachedOutputToken{0};

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << stripeCounts;
        binaryStream << combineKernel;
        binaryStream << localBufferToken;
        binaryStream << cachedInputAddress;
        binaryStream << cachedInputSize;
        binaryStream << cachedInputToken;
        binaryStream << cachedOutputAddress;
        binaryStream << cachedOutputSize;
        binaryStream << cachedOutputToken;
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
        binaryStream >> stripeCounts;
        binaryStream >> combineKernel;
        binaryStream >> localBufferToken;
        binaryStream >> cachedInputAddress;
        binaryStream >> cachedInputSize;
        binaryStream >> cachedInputToken;
        binaryStream >> cachedOutputAddress;
        binaryStream >> cachedOutputSize;
        binaryStream >> cachedOutputToken;
    }
};

#endif // OPS_HCCL_CUSTOM_H
