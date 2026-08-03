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

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

// ccu kernel register 注册所需信息
struct CcuKernelInfo {
    // kernel 名称
    char kernelFuncName[64];
    // kernel 函数
    void *kernelFunc;
    // KernelArg 实例指针
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
    ThreadHandle ccuThread;                      ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;                      ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads;           ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;     ///< 小消息 one-shot kernels
    std::vector<CcuKernelHandle> stripedKernels; ///< 大消息 striped ReadReduce kernels
    CcuKernelHandle mergeKernel{0};              ///< 双层拓扑的本地 partial merge kernel
    std::vector<uint32_t> kernelChannelCounts;
    // ccuKernels[i]在threads[kernelThreadIndices[i]]上启动。当第0层和第1层
    // 编译到不同的 CCU 时，需要在上下文中保留此映射关系。
    std::vector<uint32_t> kernelThreadIndices;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << stripedKernels;
        binaryStream << mergeKernel;
        binaryStream << kernelChannelCounts;
        binaryStream << kernelThreadIndices;
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
        binaryStream >> stripedKernels;
        binaryStream >> mergeKernel;
        binaryStream >> kernelChannelCounts;
        binaryStream >> kernelThreadIndices;
    }
};

#endif // OPS_HCCL_CUSTOM_H
