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
    ChannelHandle channels[MAX_RANK_SIZE] = {};
    uint32_t peerRanks[MAX_RANK_SIZE] = {};
    uint32_t channelCount = 0;
};

struct CcuReduceScatterKernelArg : public CcuKernelArgBase {
    uint64_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t hasWorker = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
};

struct CcuReduceScatter2DKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t serverIndex = INVALID_VALUE_RANKID;
    uint32_t localIndex = INVALID_VALUE_RANKID;
    uint32_t topologyRanks[MAX_RANK_SIZE] = {};
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
};

struct CcuReduceScatterAsymKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t serverIndex = INVALID_VALUE_RANKID;
    uint32_t localIndex = INVALID_VALUE_RANKID;
    uint32_t localRankCount = 0;
    uint32_t crossRankCount = 0;
    uint32_t remoteProxyRank = INVALID_VALUE_RANKID;
    uint32_t topologyRanks[MAX_RANK_SIZE] = {};
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
};

enum class ReduceScatterAlgorithm : uint32_t {
    SINGLE_LAYER = 0,
    DUAL_LAYER = 1,
    PARALLEL_2X8 = 2,
    ASYMMETRIC_8X4 = 3,
    LATIN_4X1 = 4,
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
    CommBuffer localBuffer{}; ///< 本端HCCL通信内存
    std::vector<CcuKernelHandle> ccuKernels;
    ThreadHandle workerThread{};
    ReduceScatterAlgorithm algorithm = ReduceScatterAlgorithm::SINGLE_LAYER;
    uint32_t localPeerCount = 0;
    uint32_t crossPeerCount = 0;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << localBuffer;
        binaryStream << ccuKernels;
        binaryStream << workerThread;
        binaryStream << algorithm;
        binaryStream << localPeerCount;
        binaryStream << crossPeerCount;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> localBuffer;
        binaryStream >> ccuKernels;
        binaryStream >> workerThread;
        binaryStream >> algorithm;
        binaryStream >> localPeerCount;
        binaryStream >> crossPeerCount;
    }
};

#endif // OPS_HCCL_CUSTOM_H
