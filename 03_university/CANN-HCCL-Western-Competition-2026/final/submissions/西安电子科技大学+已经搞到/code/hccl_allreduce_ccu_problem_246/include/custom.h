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

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct CcuKernelArgBase {
    static constexpr uint32_t kMaxRankSize = 32;
    ChannelHandle channels[kMaxRankSize];
    uint32_t channelCount;
};

// V4.8：V4.4 大包图保持不变；三种拓扑的 512 KiB 使用独立小包图。
struct CcuKernelArgAllReduce : CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t peerRanks[kMaxRankSize]{};
    uint32_t topologyKind = 0; // 4: Clos4, 12: Mesh8+Mesh4, 16: Mesh8+Mesh8
    uint32_t stage = 0; // 注册期常量：0 local reduce, 1 cross exchange, 3 local gather
    bool includeLocalInput = false;
    bool initOutput = false;
    bool combineOnly = false;
    bool deferLocalInput = false;
    bool combineScratch = true;
    bool combineInput = false;
    // hierarchical experimental graph: actual peer group on this die
    uint32_t localGroupBase = 0;
    uint32_t localGroupSize = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceType = HCCL_REDUCE_SUM;
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
    CommBuffer localBuffer;
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    // 0: legacy small one-shot
    // 1: large V2.3 topology-selected reduce-scatter/allgather
    // 2: 512 KiB topology-specialized owner-lane multi-kernel
    // 6: topology3 8+8 small one-shot (独立资源图)
    // 7: topology3 8+8 large path (独立资源图)
    // 8: topology1 4-rank large path (独立资源图)
    // 9: topology2 8+4 large path (独立资源图)
    // 10: topology1 512 KiB tiled reduce + fused gather
    // 11: topology2 512 KiB dual-die tiled full reduce
    // 12: topology3 512 KiB dual-die tiled full reduce
    uint32_t algorithmKind{0};
    uint32_t topologyKind{0};
    uint32_t localGroupBase{0};
    uint32_t localGroupSize{0};
    uint32_t reduceCombineMask{0}; // bit0: stage0 需要跨 die combine
    // 固定阶段大包采用两个独立的 HCOMM registration resource group。
    // 该计数既是序列化后的阶段边界，也阻止 ExecOp 将 Gather handle
    // 当作 Reduce/Combine handle 使用。
    uint32_t reduceKernelCount{0};
    uint32_t gatherKernelCount{0};

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << algorithmKind;
        binaryStream << topologyKind;
        binaryStream << localGroupBase;
        binaryStream << localGroupSize;
        binaryStream << reduceCombineMask;
        binaryStream << reduceKernelCount;
        binaryStream << gatherKernelCount;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> algorithmKind;
        binaryStream >> topologyKind;
        binaryStream >> localGroupBase;
        binaryStream >> localGroupSize;
        binaryStream >> reduceCombineMask;
        binaryStream >> reduceKernelCount;
        binaryStream >> gatherKernelCount;
    }
};

#endif // OPS_HCCL_CUSTOM_H
