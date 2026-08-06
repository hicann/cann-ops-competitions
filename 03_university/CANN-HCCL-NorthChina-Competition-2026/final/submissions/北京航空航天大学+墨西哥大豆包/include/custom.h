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
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

enum class AllGatherTopology : uint32_t {
    TWO_SERVER_EIGHT_NPU = 0,
    FOUR_SERVER_ONE_NPU = 1,
    EIGHT_PLUS_FOUR = 2,
    RESERVED = 3,
};

enum class AllGatherSizeClass : uint32_t {
    SMALL = 0,
    LARGE = 1,
};

struct AllGatherRoute {
    AllGatherTopology topology = AllGatherTopology::RESERVED;
    AllGatherSizeClass sizeClass = AllGatherSizeClass::SMALL;
};

struct AllGatherKernelMeta {
    uint32_t netLayer = 0;
    AllGatherSizeClass sizeClass = AllGatherSizeClass::SMALL;
    uint32_t threadIndex = 0;
};

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t remoteRanks[MAX_RANK_SIZE];
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

struct AlgResourceCtx {
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<AllGatherKernelMeta> ccuKernelMetas;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << ccuKernelMetas;
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
        binaryStream >> ccuKernelMetas;
    }
};

constexpr uint64_t SMALL_512_FAST_CTX_MAGIC = 0x3531324B46415354ULL;
constexpr uint32_t SMALL_512_FAST_KERNEL_NUM = 1;
constexpr uint32_t SMALL_512_FAST_MODE = 1;

struct Small512FastResourceCtx {
    uint64_t magic = SMALL_512_FAST_CTX_MAGIC;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    ThreadHandle thread{};
    CcuKernelHandle kernels[SMALL_512_FAST_KERNEL_NUM]{};
    uint64_t cachedInputAddr = 0;
    uint64_t cachedOutputAddr = 0;
    uint64_t cachedToken = 0;
    uint32_t cacheValid = 0;
    uint32_t reserved = 0;
};

#endif // OPS_HCCL_CUSTOM_H
