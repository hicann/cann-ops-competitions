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
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
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
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<uint32_t> kernelLayers;
    CcuKernelHandle small4x1PullKernel = 0;
    std::vector<ThreadHandle> slaveThreads;
    uint32_t pairKernelIndex = MAX_RANK_SIZE;
    uint32_t pairedRemoteRank = MAX_RANK_SIZE;
    std::vector<uint32_t> relayKernelIndices;
    std::vector<uint32_t> asymmetricPairedRanks;
    uint32_t asymmetricPairRoundMask = 0;
    uint32_t small2x8DirectPull = 0;
    uint32_t small8Plus4DirectPull = 0;
    uint32_t large8Plus4DirectPull = 0;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuKernels;
        binaryStream << kernelLayers;
        binaryStream << small4x1PullKernel;
        binaryStream << slaveThreads;
        binaryStream << pairKernelIndex;
        binaryStream << pairedRemoteRank;
        binaryStream << relayKernelIndices;
        binaryStream << asymmetricPairedRanks;
        binaryStream << asymmetricPairRoundMask;
        binaryStream << small2x8DirectPull;
        binaryStream << small8Plus4DirectPull;
        binaryStream << large8Plus4DirectPull;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> ccuKernels;
        binaryStream >> kernelLayers;
        binaryStream >> small4x1PullKernel;
        binaryStream >> slaveThreads;
        binaryStream >> pairKernelIndex;
        binaryStream >> pairedRemoteRank;
        binaryStream >> relayKernelIndices;
        binaryStream >> asymmetricPairedRanks;
        binaryStream >> asymmetricPairRoundMask;
        binaryStream >> small2x8DirectPull;
        binaryStream >> small8Plus4DirectPull;
        binaryStream >> large8Plus4DirectPull;
    }
};

// The formal 2x8 small-message profile uses a distinct engine tag and a
// deliberately minimal serialized context. Keeping it separate preserves the
// byte layout of AlgResourceCtx used by the v052 tag, including when the same
// communicator invokes large and small profiles in either order.
struct Small2x8PullResourceCtx {
    std::vector<CcuKernelHandle> kernels;
    std::vector<uint32_t> layers;
    std::vector<ThreadHandle> slaveThreads;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << kernels;
        binaryStream << layers;
        binaryStream << slaveThreads;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> kernels;
        binaryStream >> layers;
        binaryStream >> slaveThreads;
    }
};

struct Small8Plus4PullResourceCtx {
    std::vector<CcuKernelHandle> kernels;
    std::vector<uint32_t> layers;
    std::vector<ThreadHandle> slaveThreads;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << kernels;
        binaryStream << layers;
        binaryStream << slaveThreads;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> kernels;
        binaryStream >> layers;
        binaryStream >> slaveThreads;
    }
};

#endif // OPS_HCCL_CUSTOM_H
