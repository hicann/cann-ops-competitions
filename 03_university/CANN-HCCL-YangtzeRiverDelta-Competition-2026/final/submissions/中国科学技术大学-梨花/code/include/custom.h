/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_BROADCAST_CCU_PROBLEM_247_CUSTOM_H
#define HCCL_BROADCAST_CCU_PROBLEM_247_CUSTOM_H

#include <cstdint>
#include <memory>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr = nullptr;
    uint64_t size = 0;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE] = {};
    uint32_t channelCount = 0;
};

constexpr uint32_t BROADCAST_ALG_MESH1D_MEM2MEM = 1;
constexpr uint32_t BROADCAST_ALG_DIRECT = 2;
constexpr uint32_t BROADCAST_ALG_NHR1D_MEM2MEM = 3;
constexpr uint32_t BROADCAST_ALG_PARALLEL_HIERARCHICAL = 4;
constexpr uint32_t BROADCAST_ALG_PARALLEL_DIRECT = 5;
constexpr uint32_t BROADCAST_PARALLEL_2D_RANK_SIZE = 16;
constexpr uint32_t BROADCAST_ASYMMETRIC_2D_RANK_SIZE = 12;
constexpr uint32_t BROADCAST_PARALLEL_2D_LANE_COUNT = 8;
constexpr uint32_t BROADCAST_ASYMMETRIC_2D_SMALL_GROUP_SIZE = 4;
constexpr uint32_t BROADCAST_KERNEL_NOTIFY_NUM = 3;
constexpr uint32_t BROADCAST_KERNEL_IDX = 0;
constexpr uint32_t BROADCAST_KERNEL_NUM = 1;
constexpr uint32_t BROADCAST_HIERARCHY_LAYER0_KERNEL_IDX = 0;
constexpr uint32_t BROADCAST_HIERARCHY_LAYER1_KERNEL_IDX = 1;
// A 先走 Layer-0，B 先走 Layer-1；四阶段分别并行执行两个维度的 Scatter/AllGather。
constexpr uint32_t BROADCAST_HIERARCHY_PHASE_SCATTER_0 = 0;
constexpr uint32_t BROADCAST_HIERARCHY_PHASE_SCATTER_1 = 1;
constexpr uint32_t BROADCAST_HIERARCHY_PHASE_ALLGATHER_0 = 2;
constexpr uint32_t BROADCAST_HIERARCHY_PHASE_ALLGATHER_1 = 3;
constexpr uint64_t BROADCAST_DIRECT_THRESHOLD = 1ULL * 1024 * 1024;
constexpr uint64_t BROADCAST_PARALLEL_DIRECT_DATA_SIZE = 512ULL * 1024;
constexpr bool BROADCAST_ENABLE_NHR_LARGE_PACKET = true;
constexpr bool BROADCAST_ENABLE_PARALLEL_HIERARCHICAL = true;
// 8+4 八 lane 版本实测使大包从 2.63ms 回退到 3.11ms，先恢复 rank12 Mesh1D 基线。
constexpr bool BROADCAST_ENABLE_ASYMMETRIC_HIERARCHICAL = false;
constexpr bool BROADCAST_ENABLE_PARALLEL_DIRECT = true;
// 官方 Parallel Broadcast 的 SCATTER 公式在 2*8 上量化为 Mesh-first 3/8。
// 独立开关关闭时，ExecOp 保持原有 50/50（奇数元素 A 向上取整）行为。
constexpr bool BROADCAST_ENABLE_RANK16_WEIGHTED_SPLIT = true;
constexpr uint32_t BROADCAST_RANK16_WEIGHTED_SPLIT_NUMERATOR = 3;
constexpr uint32_t BROADCAST_RANK16_WEIGHTED_SPLIT_DENOMINATOR = 8;
static_assert(BROADCAST_RANK16_WEIGHTED_SPLIT_NUMERATOR > 0 &&
        BROADCAST_RANK16_WEIGHTED_SPLIT_NUMERATOR < BROADCAST_RANK16_WEIGHTED_SPLIT_DENOMINATOR,
    "rank16 weighted split must keep both paths non-empty");
constexpr bool BROADCAST_REUSE_HIERARCHY_REMOTE_ADDR = true;
constexpr bool BROADCAST_SKIP_REDUNDANT_A_RETURN = true;
// rank4 大包的两个 256MB 分片共享 NHR step 与同步，关闭可回退到逐分片执行。
constexpr bool BROADCAST_ENABLE_NHR_PAIRED_CHUNKS = true;
// 128 Event + 1024 CcuBuffer 在评测 CCU 翻译器上注册失败，保留实现供缩小资源后继续试验。
constexpr bool BROADCAST_ENABLE_GROUP_DIRECT_512K = false;

// 按整数比例计算一个数据窗口中先走 Mesh 的元素数。
// roundUp=true 仅用于关闭加权开关时复现旧的 50/50 奇数取整规则。
constexpr uint64_t BroadcastSplitCountByRatio(
    uint64_t totalCount, uint32_t numerator, uint32_t denominator, bool roundUp)
{
    if (denominator == 0 || numerator > denominator) {
        return totalCount / 2 + totalCount % 2;
    }
    const uint64_t quotient = totalCount / denominator;
    const uint64_t remainder = totalCount % denominator;
    const uint64_t remainderProduct = remainder * numerator;
    const uint64_t remainderCount = remainderProduct / denominator;
    const bool hasFraction = remainderProduct % denominator != 0;
    return quotient * numerator + remainderCount + (roundUp && hasFraction ? 1 : 0);
}

static_assert(BroadcastSplitCountByRatio(8, 3, 8, false) == 3,
    "rank16 weighted split must assign 3/8 to Mesh-first data");
static_assert(BroadcastSplitCountByRatio(9, 3, 8, false) == 3,
    "weighted split must use deterministic floor rounding");
static_assert(BroadcastSplitCountByRatio(9, 1, 2, true) == 5,
    "disabled weighted split must preserve legacy odd-count rounding");

inline uint32_t SelectBroadcastAlgorithmMode(uint64_t dataSize, uint32_t rankSize)
{
    if (dataSize <= BROADCAST_DIRECT_THRESHOLD) {
        const bool enableParallelDirect = BROADCAST_ENABLE_PARALLEL_DIRECT &&
            dataSize == BROADCAST_PARALLEL_DIRECT_DATA_SIZE &&
            (rankSize == BROADCAST_PARALLEL_2D_RANK_SIZE ||
                rankSize == BROADCAST_ASYMMETRIC_2D_RANK_SIZE);
        if (enableParallelDirect) {
            return BROADCAST_ALG_PARALLEL_DIRECT;
        }
        return BROADCAST_ALG_DIRECT;
    }
    const bool enableSymmetric2D = BROADCAST_ENABLE_PARALLEL_HIERARCHICAL &&
        rankSize == BROADCAST_PARALLEL_2D_RANK_SIZE;
    const bool enableAsymmetric2D = BROADCAST_ENABLE_PARALLEL_HIERARCHICAL &&
        BROADCAST_ENABLE_ASYMMETRIC_HIERARCHICAL &&
        rankSize == BROADCAST_ASYMMETRIC_2D_RANK_SIZE;
    if (enableSymmetric2D || enableAsymmetric2D) {
        return BROADCAST_ALG_PARALLEL_HIERARCHICAL;
    }
    if (BROADCAST_ENABLE_NHR_LARGE_PACKET && rankSize == 4) {
        return BROADCAST_ALG_NHR1D_MEM2MEM;
    }
    return BROADCAST_ALG_MESH1D_MEM2MEM;
}

struct CcuBroadcastMesh1DMem2MemKernelArg : public CcuKernelArgBase {
    uint64_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t rootId = 0;
};

struct CcuBroadcastParallelDirectKernelArg : public CcuKernelArgBase {
    uint64_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t rootId = 0;
    uint32_t peerRanks[MAX_RANK_SIZE] = {};
};

struct BroadcastNhrStepInfo {
    uint32_t step = 0;
    uint32_t toRank = 0;
    uint32_t fromRank = 0;
    std::vector<uint32_t> txSliceIdxs;
    std::vector<uint32_t> rxSliceIdxs;
};

struct CcuBroadcastNhr1DMem2MemKernelArg : public CcuBroadcastMesh1DMem2MemKernelArg {
    std::vector<BroadcastNhrStepInfo> stepInfoVector;
};

struct CcuBroadcastHierarchicalKernelArg : public CcuKernelArgBase {
    uint64_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t rootId = 0;
    uint32_t localRankCount = 0;
    uint32_t laneCount = 0;
    uint32_t isSourceGroup = 0;
    uint32_t localRanks[MAX_RANK_SIZE] = {};
    uint32_t sourceGatewayRanks[MAX_RANK_SIZE] = {};
    uint32_t destinationGatewayRanks[MAX_RANK_SIZE] = {};
    uint32_t peerRanks[MAX_RANK_SIZE] = {};
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
    uint32_t algorithmMode = BROADCAST_ALG_MESH1D_MEM2MEM;
    uint32_t hierarchyLaneCount = 0;
    uint32_t hierarchyHasLayer1Kernel = 0;
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << algorithmMode;
        binaryStream << hierarchyLaneCount;
        binaryStream << hierarchyHasLayer1Kernel;
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
        binaryStream >> algorithmMode;
        binaryStream >> hierarchyLaneCount;
        binaryStream >> hierarchyHasLayer1Kernel;
        binaryStream >> ccuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
    }
};

#endif // HCCL_BROADCAST_CCU_PROBLEM_247_CUSTOM_H