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
#include <memory>
#include <type_traits>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount{0};
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

struct DescriptorCacheEntry {
    uint64_t addr{0};
    uint64_t token{0};
};

struct DescriptorCacheState {
    uint32_t kind{0};
    uint32_t root{INVALID_VALUE_RANKID};
    uint32_t path{0};
};

struct AlgResourceCtx {
    static constexpr uint32_t MAGIC = 0x53434355U; // "SCCU"
    // v28 adds a host-validated Small descriptor cache.  A cache hit removes
    // both address/token SyncXN operations while a changed or non-consecutive
    // descriptor falls back to the complete publication protocol.
    // Keep a fresh context version so cached resources from earlier builds cannot be
    // interpreted as this layout.
    static constexpr uint32_t VERSION = 10;
    static constexpr uint32_t MAX_KERNEL_GROUPS = 2;

    uint32_t magic{MAGIC};
    uint32_t version{VERSION};
    uint32_t myRank{INVALID_VALUE_RANKID};
    uint32_t rankSize{0};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    ThreadHandle ccuThread{}; ///< initial user-stream thread; retained for diagnostics
    CommBuffer localBuffer{};
    ThreadHandle threads[MAX_KERNEL_GROUPS]{};
    uint32_t threadCount{0};
    CcuKernelHandle ccuKernels[MAX_KERNEL_GROUPS]{};
    uint32_t ccuKernelCount{0};
    CcuKernelHandle smallReceiverKernels[MAX_KERNEL_GROUPS]{};
    uint32_t smallReceiverKernelCount{0};
    CcuKernelHandle helperKernels[MAX_KERNEL_GROUPS]{};
    uint32_t helperKernelCount{0};
    CcuKernelHandle widePushKernels[MAX_KERNEL_GROUPS]{};
    uint32_t widePushKernelCount{0};
    uint32_t peerKernelGroups[MAX_RANK_SIZE]{};
    uint32_t localGroupByRank[MAX_RANK_SIZE]{};
    uint32_t instanceSizes[MAX_RANK_SIZE]{};
    uint32_t instanceCount{0};
    uint32_t helperTopologyValid{0};
    uint32_t rank12SmallSpecialized{0};
    uint32_t rank4SmallSpecialized{0};
    uint32_t largeBlock{0};
    DescriptorCacheEntry pushRecvCache[MAX_RANK_SIZE]{};
    DescriptorCacheEntry pullSendCache{};
    DescriptorCacheState lastDescriptorCache{};
};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value,
    "Engine context must remain a fixed byte-copyable POD");

#endif // OPS_HCCL_CUSTOM_H
