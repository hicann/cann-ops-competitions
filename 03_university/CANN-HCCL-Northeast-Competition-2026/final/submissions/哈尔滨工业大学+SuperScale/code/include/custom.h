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

// Small calls take the acknowledgement round trip off the critical path: root
// does not wait for its readers to finish. Host and kernel decide this the same
// way from the call parameters, so every rank agrees. The reserve below is kept
// out of the relay capacity so large-call plans stay as validated.
constexpr uint64_t CCU_SMALL_BYTES = 1024 * 1024;
constexpr uint64_t CCU_MIN_SCRATCH = 4ULL * 1024 * 1024;
constexpr uint64_t CCU_STAGE_RESERVE = 4ULL * 1024 * 1024;
inline bool CcuStageReserved(uint64_t capacity)
{
    return capacity >= CCU_MIN_SCRATCH + CCU_STAGE_RESERVE;
}
// Two staging slots alternate, so a call never rewrites the slot the previous
// call's readers may still be pulling from.
inline uint64_t CcuStageSlot(uint64_t bytes, uint32_t rankSize)
{
    return bytes * rankSize;
}
inline bool CcuStaged(uint64_t bytes, uint32_t rankSize, uint64_t capacity)
{
    // Every receiver of a small call reads its block straight from root's input
    // and nothing reports back: the critical path is launch + one metadata hop
    // + the read. The platform measured the staging copy at +1us (point 22:
    // 10us staged against the 9us of the plain read), so no copy is made; the
    // generation-separated slots and the entry rendezvous remain, because root
    // still runs ahead of its readers.
    (void)capacity;
    return bytes != 0 && bytes <= CCU_SMALL_BYTES && rankSize >= 2;
}

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
    uint32_t rank = 0, rankSize = 0, root = 0;
    uint64_t bytes = 0;
    uint32_t nearMask = 0;
    uint32_t relayMask = 0;
    uint32_t die = 0, activeDies = 1;
    uint32_t channelDie[MAX_RANK_SIZE] = {};
    bool relayAllowed = true;
    uint64_t scratchCapacity = 0;
    uint32_t phase = 0, relayDies = 0, commitDies = 0;
    uint32_t localMask = 0;  // this rank's own Layer-0 instance members (0 if unknown)
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
    ThreadHandle ccuThread = 0;        ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer{};          ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    uint32_t dataKernelCount = 0;
    std::vector<ChannelHandle> channels;
    uint64_t requiredScratch = 0;
    uint64_t minCclSize = 0;
    std::vector<uint32_t> channelDie;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << dataKernelCount;
        binaryStream << channels;
        binaryStream << requiredScratch;
        binaryStream << minCclSize;
        binaryStream << channelDie;
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
        binaryStream >> dataKernelCount;
        binaryStream >> channels;
        binaryStream >> requiredScratch;
        binaryStream >> minCclSize;
        binaryStream >> channelDie;
    }
};

#endif // OPS_HCCL_CUSTOM_H
