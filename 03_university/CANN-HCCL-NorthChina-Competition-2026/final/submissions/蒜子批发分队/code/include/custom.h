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
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "common.h"

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE] = {};
    uint32_t channelCount = 0;
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64] = {};
    // kernel函数
    void *kernelFunc = nullptr;
    // KernelArg实例指针
    void *kernelArg = nullptr;

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
    static constexpr uint32_t MAX_KERNEL_COUNT = 2;
    static constexpr uint32_t BASE_TASK_ARG_COUNT = 6;
    static constexpr uint32_t FUSED_BASE_TASK_ARG_COUNT = 7;
    static constexpr uint32_t DUAL_RAIL_BASE_TASK_ARG_COUNT = 8;
    static constexpr uint32_t DUAL_RAIL_FUSED_BASE_TASK_ARG_COUNT = 9;
    static constexpr uint32_t DUAL_RAIL_FULL_TASK_ARG_COUNT = 12;
    static constexpr uint32_t DUAL_RAIL_FUSED_FULL_TASK_ARG_COUNT = 13;
    static constexpr uint32_t DIRECT_LOCAL_COPY_TASK_ARG_COUNT = 8;
    static constexpr uint32_t FUSED_DIRECT_LOCAL_COPY_TASK_ARG_COUNT = 9;
    static constexpr uint32_t RANK12_NHR_TASK_ARG_COUNT = 8;
    static constexpr uint32_t RANK12_MESH_TASK_ARG_COUNT = 9;
    static constexpr uint32_t RANK12_CLOS_TASK_ARG_COUNT = 13;
    static constexpr uint32_t RANK16_MESH_TASK_ARG_COUNT = 13;
    static constexpr uint32_t RANK16_CLOS_TASK_ARG_COUNT = 10;
    static constexpr uint32_t GROUP_BROADCAST_TASK_ARG_COUNT = 9;
    static constexpr uint32_t HYBRID_TASK_ARG_COUNT = 10;
    static constexpr uint32_t FULL_TASK_ARG_COUNT = 11;
    static constexpr uint32_t FUSED_FULL_TASK_ARG_COUNT = 12;
    static constexpr uint32_t MAX_TASK_ARG_COUNT =
        RANK16_MESH_TASK_ARG_COUNT;

    uint32_t layoutVersion = 143;
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t kernelCount = 0;
    ThreadHandle threads[MAX_KERNEL_COUNT] = {};
    CcuKernelHandle ccuKernels[MAX_KERNEL_COUNT] = {};
    uint32_t taskArgCounts[MAX_KERNEL_COUNT] = {};
    uint32_t kernelLinkLoads[MAX_KERNEL_COUNT] = {};
    bool rank12DualRail = false;
    uint32_t rank12PrimaryRailWeight = 0;
    uint32_t rank12AlternateRailWeight = 0;
    bool rank12DirectLocalCopy = false;
    bool rank12DirectReuse = false;
    bool rank12Nhr = false;
    bool rank12SplitRelay = false;
    bool rank12SmallServer = false;
    uint32_t rank12MeshKernelIndex = MAX_KERNEL_COUNT;
    uint32_t rank12ClosKernelIndex = MAX_KERNEL_COUNT;
    uint32_t rank12FanoutSourceRanks[2] = {
        MAX_RANK_SIZE,
        MAX_RANK_SIZE,
    };
    uint32_t rank12FanoutSourceCount = 0;
    bool rank16MixedRelay = false;
    uint32_t rank16MeshKernelIndex = MAX_KERNEL_COUNT;
    uint32_t rank16ClosKernelIndex = MAX_KERNEL_COUNT;
    uint32_t rank16RelayPeerRank = MAX_RANK_SIZE;
    uint64_t cachedStreamAddress = 0;
    uint64_t cachedInputAddress = 0;
    uint64_t cachedInputBytes = 0;
    uint64_t cachedInputToken = 0;
    uint64_t cachedOutputAddress = 0;
    uint64_t cachedOutputBytes = 0;
    uint64_t cachedOutputToken = 0;

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
    "AlgResourceCtx must be safe for engine-context copies");

inline HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_INTERNAL:
            return HCCL_E_INTERNAL;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

#endif // OPS_HCCL_CUSTOM_H
