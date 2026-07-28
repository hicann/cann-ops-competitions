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

struct BroadcastKernelArg : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t phaseMask;
    uint32_t remoteRanks[MAX_RANK_SIZE];
};

inline HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS: return HCCL_SUCCESS;
        case CCU_E_PARA: return HCCL_E_PARA;
        case CCU_E_PTR: return HCCL_E_PTR;
        case CCU_E_INTERNAL: return HCCL_E_INTERNAL;
        case CCU_E_NOT_SUPPORT: return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND: return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL: return HCCL_E_UNAVAIL;
        default: return HCCL_E_INTERNAL;
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
    static constexpr uint32_t MAX_THREAD_COUNT = 2;
    static constexpr uint32_t MAX_KERNEL_COUNT = 6;

    ThreadHandle ccuThread{};            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer{};            ///< 本端HCCL通信内存
    ThreadHandle threads[MAX_THREAD_COUNT]{};
    CcuKernelHandle ccuKernels[MAX_KERNEL_COUNT]{};
    uint32_t kernelDies[MAX_KERNEL_COUNT]{};
    uint32_t threadCount = 0;
    uint32_t kernelCount = 0;
    uint32_t directKernelCount = 0;

    HcclResult AddThread(ThreadHandle thread)
    {
        if (threadCount >= MAX_THREAD_COUNT) {
            return HCCL_E_PARA;
        }
        threads[threadCount++] = thread;
        return HCCL_SUCCESS;
    }

    HcclResult AddKernel(CcuKernelHandle kernel, uint32_t dieId)
    {
        if (kernelCount >= MAX_KERNEL_COUNT) {
            return HCCL_E_PARA;
        }
        ccuKernels[kernelCount] = kernel;
        kernelDies[kernelCount] = dieId;
        ++kernelCount;
        return HCCL_SUCCESS;
    }

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
        if (threadCount > MAX_THREAD_COUNT || kernelCount > MAX_KERNEL_COUNT ||
            directKernelCount > kernelCount) {
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }
};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value,
    "AlgResourceCtx must remain trivially copyable");

#endif // OPS_HCCL_CUSTOM_H
