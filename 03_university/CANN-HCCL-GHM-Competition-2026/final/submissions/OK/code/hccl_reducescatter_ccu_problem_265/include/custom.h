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

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

struct CcuKernelArgReduceScatter : public CcuKernelArgBase {
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t includeSelf = 0;
    uint32_t writeOutput = 0;
    uint32_t scratchStartSlot = 0;
};

struct CcuKernelArgReduceScatterMerge : public CcuKernelArgBase {
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t scratchStartSlot = 0;
};

struct CcuKernelArgReduceScatterHierLocal : public CcuKernelArgBase {
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t serverStart = 0;
    uint32_t serverSize = 0;
    uint32_t assignedOutputCount = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
};

struct CcuKernelArgReduceScatterHierCross : public CcuKernelArgBase {
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t rankSize = 0;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t localServerSize = 0;
    uint32_t remoteServerSize = 0;
    uint32_t remoteReadChannelIndex = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
    CommBuffer localBuffer; ///< 本端HCCL通信内存
    ThreadHandle secondaryThread = 0;
    uint32_t partialKernelCount = 0;
    uint32_t secondaryScratchStartSlot = 0;
    std::vector<CcuKernelHandle> ccuKernels;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << localBuffer;
        binaryStream << secondaryThread;
        binaryStream << partialKernelCount;
        binaryStream << secondaryScratchStartSlot;
        binaryStream << ccuKernels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> localBuffer;
        binaryStream >> secondaryThread;
        binaryStream >> partialKernelCount;
        binaryStream >> secondaryScratchStartSlot;
        binaryStream >> ccuKernels;
    }
};

#endif // OPS_HCCL_CUSTOM_H
