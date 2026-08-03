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

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include <type_traits>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

constexpr uint64_t CASE05_KERNEL_PARAM_MAGIC = 0x4147474154483035ULL;
constexpr uint64_t CASE05_RESOURCE_PLAN_MAGIC = 0x5244345341473035ULL;
constexpr uint32_t CASE05_PLAN_STAGE_COUNT = 4;
constexpr uint64_t BIG_DIRECT_PLAN_MAGIC = 0x4249474449524543ULL;
constexpr uint32_t BIG_DIRECT_THREAD_COUNT = 16;
constexpr uint32_t BIG_DIRECT_PEER_COUNT = 15;

// Case05执行计划以平坦结构直接存放在AICPU Context中。热调用仅传递
// Context地址；magic用于区分同一Kernel入口的紧凑参数与通用OpParam。
struct Case05KernelParam {
    uint64_t magic = CASE05_KERNEL_PARAM_MAGIC;
    void *resCtx = nullptr;
};
static_assert(sizeof(Case05KernelParam) == 16,
    "Unexpected compact Case05 parameter ABI");

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
    CommBuffer remoteInput;
    CommBuffer remoteOutput;
};

struct Case05StageResource {
    ChannelHandle channel = 0;
    void *remoteDst = nullptr;
    void *localSrc = nullptr;
    uint64_t len = 0;
};

// Host在首次建链时将四阶段recursive-doubling编译进资源Context。
// 每阶段的WriteWithNotify/Wait既传播数据，也证明下一阶段输入已就绪。
struct Case05ResourcePlan {
    uint64_t magic = 0;
    ThreadHandle thread = 0;
    void *inputPtr = nullptr;
    void *localDst = nullptr;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    Case05StageResource stages[CASE05_PLAN_STAGE_COUNT] = {};
};
static_assert(sizeof(Case05ResourcePlan) == 168,
    "Unexpected flat Case05 plan ABI");
static_assert(std::is_trivially_copyable<Case05ResourcePlan>::value,
    "Case05 plan must support direct Context copy");

struct BigDirectPeerPlan {
    ThreadHandle thread = 0;
    ChannelHandle channel = 0;
    HcommBatchTransferDesc transfers[2] = {};
};

// Case06/07继续使用已验证的逐对端Read+ACK，只把不随调用变化的
// Thread、Channel、地址和Batch描述符提前固化，减少AICPU编排开销。
struct BigDirectPlan {
    uint64_t magic = 0;
    void *inputPtr = nullptr;
    void *localDst = nullptr;
    uint64_t dataSize = 0;
    uint32_t readyThreadNum = 0;
    uint32_t activeThreadNum = 0;
    uint32_t peerCount = 0;
    uint32_t earlyLocalCopy = 0;
    ThreadHandle threads[BIG_DIRECT_THREAD_COUNT] = {};
    BigDirectPeerPlan peers[BIG_DIRECT_PEER_COUNT] = {};
};
static_assert(std::is_trivially_copyable<BigDirectPlan>::value,
    "Big direct plan must support resource serialization");

struct AlgResourceCtx {
    ThreadHandle aicpuThread;          ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< AICPU_TS通信引擎上的thread资源
    std::vector<ChannelInfo> channels; ///< AICPU_TS通信引擎上的channel资源
    BigDirectPlan bigDirectPlan;        ///< Host预编译的Case06/07直读计划

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << bigDirectPlan;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
        binaryStream >> bigDirectPlan;
    }
};

#endif // OPS_HCCL_CUSTOM_H
