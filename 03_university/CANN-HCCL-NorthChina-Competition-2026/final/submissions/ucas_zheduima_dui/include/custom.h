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

// 传输模式（taskArgs[6]）：扁平直推 / 三路拆分阶段一 / 三路拆分阶段二
constexpr uint64_t TRANSFER_FLAT = 0;
constexpr uint64_t TRANSFER_PHASE_ONE = 1;
constexpr uint64_t TRANSFER_PHASE_TWO = 2;
// 4*1 大消息递归倍增（单 launch 融合两轮）：round1 与 r^1 交换 own S，
// 内部 barrier 后 round2 与 r^2 交换相邻两片（2S）；每轮单 channel 独占全 NIC
constexpr uint64_t TRANSFER_ROUND_FUSED = 3;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

// 每个 Kernel 只持有同一网络层、同一 IO Die 上的 Channel。
// copySelf 用于把本 Rank 的本地拷贝分配给负载较轻的 Kernel。
struct CcuKernelArgAllGather : public CcuKernelArgBase {
    uint32_t copySelf = 0;
    uint32_t isLocalLayer = 0;
    uint32_t hierarchicalEnabled = 0;
    uint32_t myRank = 0;
    uint32_t pairRank = 0;
    uint32_t pairChannelIdx = 0;
    uint32_t pairChannelIdxR2 = 0; // 4*1 递归倍增 round-2 的配对 channel（r^2）
    uint32_t localRankCount = 0;
    uint32_t remoteRankCount = 0;
    uint32_t localRanks[MAX_RANK_SIZE]{};
    uint32_t ownedRemoteCount = 0;
    uint32_t ownedRemoteRanks[MAX_RANK_SIZE]{};
    // owned-remote 在 layer-1 group 内的 channel 下标（与 ownedRemoteRanks 一一对应）
    uint32_t ownedRemoteChannelIdx[MAX_RANK_SIZE]{};
    uint32_t stageTwoTargetCount = 0;
    uint32_t stageTwoTargetChannelIdx[MAX_RANK_SIZE]{};
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
    ThreadHandle ccuThread{};          ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer{};          ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;     ///< 分层 kernel（与 threads 一一对应）
    std::vector<CcuKernelHandle> flatKernels;    ///< 极简扁平 kernel（与 threads 一一对应）

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << flatKernels;
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
        binaryStream >> flatKernels;
    }
};

#endif // OPS_HCCL_CUSTOM_H
