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
#include <cstddef>
#include <memory>
#include <type_traits>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "common.h"

constexpr uint64_t SMALL_DATA_THRESHOLD = 1 * 1024 * 1024;
constexpr uint32_t FOUR_RANK_PIPELINE_SLICE_NUM = 16;
constexpr uint64_t TEST_SIZE_400_MB_PLUS_4 = 400ULL * 1024ULL * 1024ULL + 4ULL;

// 拓扑、数据量、Channel 组网和 Kernel 调度相互独立，方便后续按场景组合算法。
enum class TopologyClass : uint32_t {
    GENERIC = 0,
    FOUR_RANK,
    TWELVE_RANK,
    SIXTEEN_RANK,
};

enum class MessageSizeClass : uint32_t {
    SMALL = 0,
    LARGE,
};

enum class ChannelPlanType : uint32_t {
    FLAT_ROOT = 0,
    FOUR_RANK_TREE,
    FOUR_RANK_PIPELINE,
    TWELVE_PIPELINE,
    SIXTEEN_FUSED,
};

enum class KernelPlanType : uint32_t {
    LAYER_PARALLEL = 0,
    FOUR_RANK_TREE,
    FOUR_RANK_PIPELINE,
    SMALL_FLAT,
    TWELVE_PIPELINE,
    SIXTEEN_SCATTER_DOUBLING,
};

struct BroadcastPlan {
    TopologyClass topology = TopologyClass::GENERIC;
    MessageSizeClass messageSize = MessageSizeClass::SMALL;
    ChannelPlanType channelPlan = ChannelPlanType::FLAT_ROOT;
    KernelPlanType kernelPlan = KernelPlanType::LAYER_PARALLEL;
    uint64_t chunkSize = MAX_DATA_SIZE;
    uint32_t executionThreadNum = 2; // 包含用户 stream 绑定的 cpuThread
    uint32_t threadNotifyNum = 1;
    uint32_t sixteenDeferCrossWait = 0;
};

inline TopologyClass DetectTopology(uint32_t rankSize)
{
    if (rankSize == 4) {
        return TopologyClass::FOUR_RANK;
    }
    if (rankSize == 12) {
        return TopologyClass::TWELVE_RANK;
    }
    if (rankSize == 16) {
        return TopologyClass::SIXTEEN_RANK;
    }
    return TopologyClass::GENERIC;
}

inline MessageSizeClass ClassifyMessageSize(uint64_t totalSize)
{
    return totalSize <= SMALL_DATA_THRESHOLD ? MessageSizeClass::SMALL : MessageSizeClass::LARGE;
}

inline uint32_t FourRankFirstPeer(uint32_t root)
{
    return (root + 1) % 4;
}

inline uint32_t FourRankSecondPeer(uint32_t root)
{
    return (root + 2) % 4;
}

inline uint32_t FourRankForwardPeer(uint32_t root)
{
    return (root + 3) % 4;
}

inline BroadcastPlan SelectBroadcastPlan(uint32_t rankSize, uint64_t totalSize, uint32_t root)
{
    BroadcastPlan plan;
    plan.topology = DetectTopology(rankSize);
    plan.messageSize = ClassifyMessageSize(totalSize);

    if (plan.messageSize == MessageSizeClass::SMALL) {
        // Small 数据统一使用单向 root 直发，去掉接收端不需要的反向资源交换和后同步。
        plan.channelPlan = ChannelPlanType::FLAT_ROOT;
        plan.kernelPlan = KernelPlanType::SMALL_FLAT;
        if (plan.topology == TopologyClass::FOUR_RANK) {
            plan.executionThreadNum = 1;
            plan.threadNotifyNum = 0;
        }
    } else if (plan.topology == TopologyClass::FOUR_RANK) {
        if (plan.messageSize == MessageSizeClass::LARGE) {
            plan.channelPlan = ChannelPlanType::FOUR_RANK_PIPELINE;
            plan.kernelPlan = KernelPlanType::FOUR_RANK_PIPELINE;
            // 16 分片的接收、转发收敛到单个 persistent kernel，不再需要 Host thread 间逐片同步。
            plan.executionThreadNum = 1;
            plan.threadNotifyNum = 0;
        }
    } else if (plan.topology == TopologyClass::TWELVE_RANK && plan.messageSize == MessageSizeClass::LARGE && root < 8) {
        // 8+4 Large：layer 0 推进双机内流水，layer 1 推进两轮全双工跨机 wave。
        plan.channelPlan = ChannelPlanType::TWELVE_PIPELINE;
        plan.kernelPlan = KernelPlanType::TWELVE_PIPELINE;
        plan.executionThreadNum = 2;
        plan.threadNotifyNum = 3;
    } else if (plan.topology == TopologyClass::SIXTEEN_RANK && plan.messageSize == MessageSizeClass::LARGE) {
        // 16-rank Large: root Scatter 16分片，再并行执行本 Server 7-peer 与跨 Server 8轮匹配。
        plan.channelPlan = ChannelPlanType::SIXTEEN_FUSED;
        plan.kernelPlan = KernelPlanType::SIXTEEN_SCATTER_DOUBLING;
        plan.executionThreadNum = 2;
        plan.threadNotifyNum = 1;
        plan.sixteenDeferCrossWait = static_cast<uint32_t>(totalSize == TEST_SIZE_400_MB_PLUS_4);
    }
    return plan;
}

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
    uint32_t isSender;
};

struct CcuFourRankPipelineKernelArg : public CcuKernelArgBase {
    uint32_t position; // 沿 root -> rank1 -> rank2 -> rank3 链的位置
};

struct CcuSixteenScatterKernelArg : public CcuKernelArgBase {
    uint32_t peerSlices[MAX_RANK_SIZE];
};

struct CcuSixteenFlatKernelArg : public CcuKernelArgBase {
    uint32_t myRank;
    uint32_t serialized;
    uint32_t deferCompletionWait;
};

constexpr uint32_t TWELVE_STAGE_NUM = 8;
struct CcuTwelveKernelArg : public CcuKernelArgBase {
    uint32_t myRank;
    uint32_t root;
    uint32_t peerRanks[MAX_RANK_SIZE];
    uint32_t stageChannelCount[TWELVE_STAGE_NUM];
    uint32_t stageChannelIndices[TWELVE_STAGE_NUM][MAX_RANK_SIZE];
};

// Small 路径只需最多两个 thread/kernel，固定布局避免 vector 反序列化和扫描。
constexpr uint64_t SMALL_RESOURCE_MAGIC = 0x534D414C4C435458ULL;
struct SmallResourceCtx {
    uint64_t magic = SMALL_RESOURCE_MAGIC;
    uint32_t version = 4;
    uint32_t threadCount = 0;
    uint32_t kernelCount = 0;
    std::array<ThreadHandle, 2> threads{};
    std::array<CcuKernelHandle, 2> kernels{};
    std::array<uint32_t, 2> kernelLayers{};
    std::array<uint32_t, 2> layerKernels{2, 2};
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

template <typename T, std::size_t N> struct FixedList {
    std::array<T, N> values{};
    uint32_t count = 0;

    bool empty() const
    {
        return count == 0;
    }

    std::size_t size() const
    {
        return count;
    }

    constexpr std::size_t capacity() const
    {
        return N;
    }

    void resize(std::size_t newSize)
    {
        count = static_cast<uint32_t>(newSize);
    }

    void push_back(const T &value)
    {
        values[count++] = value;
    }

    T *begin()
    {
        return values.data();
    }

    const T *begin() const
    {
        return values.data();
    }

    T *end()
    {
        return values.data() + count;
    }

    const T *end() const
    {
        return values.data() + count;
    }

    T &operator[](std::size_t index)
    {
        return values[index];
    }

    const T &operator[](std::size_t index) const
    {
        return values[index];
    }
};

constexpr uint64_t LARGE_RESOURCE_MAGIC = 0x4C41524745435458ULL;
constexpr uint32_t LARGE_RESOURCE_VERSION = 1;
constexpr std::size_t RESOURCE_THREAD_CAPACITY = 2;
constexpr std::size_t RESOURCE_KERNEL_CAPACITY = MAX_RANK_SIZE;
constexpr std::size_t RESOURCE_PEER_CAPACITY = MAX_RANK_SIZE;

struct AlgResourceCtx {
    uint64_t magic = LARGE_RESOURCE_MAGIC;
    uint32_t version = LARGE_RESOURCE_VERSION;
    FixedList<ThreadHandle, RESOURCE_THREAD_CAPACITY> threads;
    FixedList<CcuKernelHandle, RESOURCE_KERNEL_CAPACITY> ccuKernels;
    FixedList<uint32_t, RESOURCE_KERNEL_CAPACITY> kernelLayers;
    FixedList<uint32_t, RESOURCE_PEER_CAPACITY> kernelPeers;
};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value, "AlgResourceCtx must support raw engine copy");

#endif // OPS_HCCL_CUSTOM_H
