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

constexpr uint32_t BROADCAST_MAX_NHR_STEPS = 8;
constexpr uint32_t BROADCAST_SMALL_DATA_SIZE = 512 * 1024;

enum class BroadcastAlgorithm : uint32_t {
    FLAT_MIXED = 0,
    GLOBAL_MESH = 1,
    NHR_LAYER1 = 2,
};

enum class BroadcastKernelRole : uint32_t {
    FLAT_LAYER0 = 0,
    FLAT_LAYER1 = 1,
    SCATTER_LAYER0 = 2,
    SCATTER_LAYER1 = 3,
    ALLGATHER_LAYER0 = 4,
    ALLGATHER_LAYER1 = 5,
    NHR_LAYER1 = 6,
};

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount = 0;
};

struct CcuKernelArgPeers : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t rootId = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
};

struct BroadcastNhrStep {
    uint32_t toRank = INVALID_VALUE_RANKID;
    uint32_t fromRank = INVALID_VALUE_RANKID;
    uint32_t txSliceCount = 0;
    uint32_t rxSliceCount = 0;
    uint32_t txSliceIdxs[MAX_RANK_SIZE]{};
};

struct CcuKernelArgNhr : public CcuKernelArgPeers {
    uint32_t scatterStepCount = 0;
    uint32_t totalStepCount = 0;
    BroadcastNhrStep steps[BROADCAST_MAX_NHR_STEPS]{};
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
    CommBuffer localBuffer; ///< 本端HCCL通信内存
    BroadcastAlgorithm algorithm = BroadcastAlgorithm::FLAT_MIXED;
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<uint32_t> kernelRoles;
    std::vector<uint32_t> kernelThreadIdx;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << localBuffer;
        binaryStream << algorithm;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << kernelRoles;
        binaryStream << kernelThreadIdx;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> localBuffer;
        binaryStream >> algorithm;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> kernelRoles;
        binaryStream >> kernelThreadIdx;
    }
};

#endif // OPS_HCCL_CUSTOM_H
