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

#include <cstring>
#include <memory>
#include <type_traits>
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

enum class BroadcastKernelRole : uint32_t {
    PUSH_RECEIVER = 0,
    PUSH_SENDER = 1,
    SCATTER_RECEIVER = 2,
    SCATTER_SENDER = 3,
    PULL_ALLGATHER = 4,
    PUSH_ALLGATHER = 5,
    PULL_ROOT = 6,
    PULL_RECEIVER = 7,
    RECURSIVE_SCATTER_SENDER = 10,
    RECURSIVE_SCATTER_RECEIVER = 11,
    RECURSIVE_EXCHANGE = 12,
    RECURSIVE_DYNAMIC = 19,
    PIPELINE_SPLIT_ROOT = 32,
    PIPELINE_SPLIT_RELAY = 33,
    PIPELINE_SPLIT_TAIL = 34,
    FUSED_SCATTER_ROOT = 35,
    FUSED_SCATTER_RECEIVER = 36,
    FUSED_PUSH_ALLGATHER = 37,
};

enum class BroadcastAlgorithm : uint32_t {
    DIRECT = 0,
    SCATTER_ALLGATHER = 1,
    DIRECT_PULL = 2,
    RANK4_RECURSIVE = 4,
    RANK4_PIPELINE_SPLIT = 10,
    FUSED_SCATTER_ALLGATHER = 11,
};

struct CcuKernelArgBroadcast : public CcuKernelArgBase {
    uint32_t peerRanks[MAX_RANK_SIZE];
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t root;
    BroadcastKernelRole role;
    uint32_t relayRankA;
    uint32_t partnerRank;
    bool pipelineCrossDie;
};

inline HcclResult CcuResultToHccl(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        case CCU_E_INTERNAL:
        default:
            return HCCL_E_INTERNAL;
    }
}

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
    static constexpr uint32_t MAX_KERNEL_GROUPS = MAX_RANK_SIZE;

    ThreadHandle ccuThread{}; ///< CCU通信引擎上的thread资源
    ThreadHandle parallelThread{}; ///< 第二个die使用的独立CCU thread
    CommBuffer localBuffer{}; ///< 本端HCCL通信内存
    BroadcastAlgorithm algorithm{};
    bool rank4DualLane = false;
    bool stageOneParallel = false;
    bool stageTwoParallel = false;
    uint32_t stageOneKernelCount = 0;
    uint32_t stageTwoKernelCount = 0;
    CcuKernelHandle stageOneKernels[MAX_KERNEL_GROUPS]{};
    CcuKernelHandle stageTwoKernels[MAX_KERNEL_GROUPS]{};
    uint32_t stageOnePeerCounts[MAX_KERNEL_GROUPS]{};
    uint32_t stageTwoPeerCounts[MAX_KERNEL_GROUPS]{};
    uint32_t stageOnePeerRanks[MAX_KERNEL_GROUPS][MAX_RANK_SIZE]{};
    uint32_t stageTwoPeerRanks[MAX_KERNEL_GROUPS][MAX_RANK_SIZE]{};
    uint32_t recursiveKernelMask = 0;
    CcuKernelHandle recursiveKernels[4]{};

    // Engine context is created and consumed by the same binary. Keep it as a
    // fixed-layout POD so every invocation needs only one bounded memcpy and no
    // vector allocation or nested BinaryStream deserialization.
    std::vector<char> Serialize() const
    {
        const char *begin = reinterpret_cast<const char *>(this);
        return std::vector<char>(begin, begin + sizeof(*this));
    }

    HcclResult DeSerialize(const void *data, uint64_t size)
    {
        if (data == nullptr || size != sizeof(*this)) {
            return HCCL_E_PARA;
        }
        std::memcpy(this, data, sizeof(*this));
        return HCCL_SUCCESS;
    }
};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value,
    "AlgResourceCtx must remain safe for raw engine-context copies");

#endif // OPS_HCCL_CUSTOM_H
