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
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

constexpr uint32_t REDUCE_SCATTER_ALGORITHM_FULL_MESH = 0;
constexpr uint32_t REDUCE_SCATTER_ALGORITHM_PARALLEL_2D_OMNIPIPE = 1;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64]{};
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
    ThreadHandle ccuThread = 0;        ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer{};          ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<uint32_t> channelCounts;
    uint32_t algorithm = REDUCE_SCATTER_ALGORITHM_FULL_MESH;
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> columnRanks;
    std::vector<uint32_t> globalRanks;
    uint32_t localRankIndex = INVALID_VALUE_RANKID;
    uint32_t columnRankIndex = INVALID_VALUE_RANKID;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << channelCounts;
        binaryStream << algorithm;
        binaryStream << localRanks;
        binaryStream << columnRanks;
        binaryStream << globalRanks;
        binaryStream << localRankIndex;
        binaryStream << columnRankIndex;
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
        binaryStream >> channelCounts;
        binaryStream >> algorithm;
        binaryStream >> localRanks;
        binaryStream >> columnRanks;
        binaryStream >> globalRanks;
        binaryStream >> localRankIndex;
        binaryStream >> columnRankIndex;
    }
};

#endif // OPS_HCCL_CUSTOM_H
