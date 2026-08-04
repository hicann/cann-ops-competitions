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
#include <type_traits>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

constexpr uint32_t INVALID_KERNEL_INDEX = 0xFFFFFFFFU;
constexpr uint32_t INVALID_PEER_INDEX = 0xFFFFFFFFU;
constexpr uint32_t MAX_CCU_KERNELS = 3;

constexpr uint32_t SMALL_TOPO_DIRECT = 0;
constexpr uint32_t SMALL_TOPO_TWO_BY_EIGHT = 1;
constexpr uint32_t SMALL_TOPO_EIGHT_PLUS_FOUR_LARGE = 2;
constexpr uint32_t SMALL_TOPO_EIGHT_PLUS_FOUR_SMALL = 3;

constexpr uint32_t LARGE_ALGORITHM_NONE = 0;
constexpr uint32_t LARGE_ALGORITHM_V10 = 1;
constexpr uint32_t LARGE_ALGORITHM_V12_P11 = 2;
constexpr uint32_t LARGE_ALGORITHM_V13_P17 = 3;
constexpr uint32_t LARGE_ALGORITHM_V23_TWO_BY_EIGHT = 4;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount = 0;
};

/**
 * A layer kernel only contains channels from one topology layer.  The
 * translator can therefore place the kernel on the CCU that belongs to that
 * layer's IO die.
 */
struct CcuKernelArgReduceLayer : public CcuKernelArgBase {
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE] = {};
    uint32_t peerCount = 0;
    uint32_t includeSelf = 0;
    uint32_t layer = 0;
    uint32_t smallTopology = SMALL_TOPO_DIRECT;
    uint32_t selectedPeerIndices[MAX_RANK_SIZE] = {};
    uint32_t selectedPeerCount = 0;
    uint32_t groupPeerIndices[MAX_RANK_SIZE] = {};
    uint32_t groupPeerCount = 0;
    uint32_t retainedGroupIndex = 0;
    uint32_t groupRankIndex = 0;
};

/**
 * The finalize kernel is local-only. Layer 0 has already produced the local
 * partial directly in the user output; this kernel folds the layer-1 partial
 * into it after both CCUs have completed.
 */
struct CcuKernelArgFinalize : public CcuKernelArgBase {};

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
    CommBuffer localBuffer{};
    ThreadHandle auxThread = 0;
    CcuKernelHandle ccuKernels[MAX_CCU_KERNELS] = {};
    uint32_t kernelCount = 0;
    uint32_t meshKernelIndex = INVALID_KERNEL_INDEX;
    uint32_t closKernelIndex = INVALID_KERNEL_INDEX;
    uint32_t finalizeKernelIndex = INVALID_KERNEL_INDEX;
    uint32_t localRankSize = 0;
    uint32_t localRankIndex = 0;
    uint32_t remoteRankSize = 0;
    uint32_t smallTopology = SMALL_TOPO_DIRECT;
    uint32_t groupedLargeKernel = LARGE_ALGORITHM_NONE;
};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value,
    "The reusable engine context must be copied without heap-backed state");

#endif // OPS_HCCL_CUSTOM_H
