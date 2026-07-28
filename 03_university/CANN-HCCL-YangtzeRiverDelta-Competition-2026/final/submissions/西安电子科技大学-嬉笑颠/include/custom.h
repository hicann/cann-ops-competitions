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

constexpr uint32_t CCU_BROADCAST_MAX_OPS = MAX_RANK_SIZE - 1;

enum class CcuBroadcastPlanKind : uint32_t {
    DIRECT = 0,
    SCATTER_GATHER = 1,
    TREE = 2,
    GLOBAL_DIRECT = 3,
    GLOBAL_SCATTER = 4,
    GLOBAL_ALLGATHER = 5,
    GLOBAL_COMPLETE = 6,
    ROOT_EXCLUDED_SCATTER_GATHER = 7,
};

enum class CcuBroadcastOpKind : uint32_t {
    SEND = 0,
    RECV = 1,
};

struct CcuBroadcastOpDesc {
    uint32_t channelIndex = 0;
    uint32_t kind = static_cast<uint32_t>(CcuBroadcastOpKind::SEND);
    uint32_t remoteRank = 0;
};

struct CcuKernelArgBroadcast : CcuKernelArgBase {
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    uint32_t root = 0;
    uint32_t ckeIndex = 0;
    uint32_t planKind = static_cast<uint32_t>(CcuBroadcastPlanKind::DIRECT);
    uint32_t groupRank = 0;
    uint32_t groupSize = 0;
    uint32_t groupRoot = 0;
    uint32_t opCount = 0;
    CcuBroadcastOpDesc ops[CCU_BROADCAST_MAX_OPS] = {};
    // Channels are partitioned by local CCU die.  Keep the global rank for
    // each channel so a partial-die kernel can still address global shards.
    uint32_t peerRanks[MAX_RANK_SIZE] = {};
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
    std::vector<uint32_t> kernelGroupSizes;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << kernelGroupSizes;
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
        binaryStream >> kernelGroupSizes;
    }
};

#endif // OPS_HCCL_CUSTOM_H
