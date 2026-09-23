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

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

// Scatter 算子 CCU Kernel 注册参数。通道按本地 IO Die 分组，每组注册一个 Kernel。
// peerRanks[] 记录每条 channel 对应的对端 rank。
struct CcuKernelArgScatter : CcuKernelArgBase {
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t rootId;
    uint32_t peerRanks[MAX_RANK_SIZE];
    uint32_t doLocalCopy; ///< ROOT：本 Kernel 是否执行自留块本地拷贝（仅首个 Die 分组为 1，避免重复拷贝）
    uint32_t relayRole;     ///< 中继角色：0=非中继（直发/拉模型），1=relay（本地中转），2=target（跨机目标）
    uint32_t relayPeerRank; ///< relay：要转发的目标 rank；target：来源 relay rank
    uint32_t relayIsMesh;   ///< root/relay 的 kernel：1=mesh 组（layer0），0=clos 组（layer1）
    uint32_t doNotify;      ///< relay 的 clos(forward) kernel：1=最后一片发 DONE，0=非末片不发
    uint8_t relayIsRelayed[MAX_RANK_SIZE]; ///< root 的 clos 组：peer 是否被中继（1=尾块直发，0=整块直发）
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
    uint8_t usePull = 0;                   ///< 1=小消息拉模型（对端读），0=推模型（root 写）
    uint8_t useLarge = 0;                  ///< 1=大消息（>256MB）分片下放到 kernel 内（单次 launch 发 2 片）
    uint8_t useRelay = 0;                  ///< 1=中继（2×8/4×3 大消息，root 转发部分跨机块给本地 relay）
    uint32_t relayRole = 0;                ///< 中继角色：0=非中继，1=relay，2=target（exec_op 与 kernel 复用）
    uint32_t relayPeerRank = INVALID_VALUE_RANKID; ///< relay：目标 rank；target：来源 relay rank
    std::vector<ThreadHandle> threads;       ///< CCU通信引擎上的thread资源（[0]=主线程）
    std::vector<CcuKernelHandle> ccuKernels; ///< 每个 IO Die 分组一个 CCU Kernel
    std::vector<std::vector<uint32_t>> peersPerKernel; ///< 每个 Kernel 中每条 channel 对应的对端 rank
    uint64_t relayStagingAddr = 0;         ///< relay 的中转缓冲地址（CCL buffer），0=无中继
    uint64_t relayBytes = 0;               ///< 中继转发量（chunkBytes×4/11 或 ×5/6）
    uint64_t relaySliceBytes = 0;          ///< 中继单次转发上限（=staging 容量；>0 且 <relayBytes 时切片转发）
    CcuKernelHandle relayOwnKernel = 0;    ///< relay 自己块读 kernel（与 gather 同 thread/die/channel，仅 relay 使用）
    CcuKernelHandle relayClosNoDoneKernel = 0; ///< relay 切片非末片的 forward kernel（不发 DONE，仅切片时注册）
    std::vector<uint32_t> relayTargetOf;   ///< 按 rank 索引：relay rank→转发目标 rank（非 relay 为 INVALID_VALUE_RANKID）

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << usePull;
        binaryStream << useLarge;
        binaryStream << useRelay;
        binaryStream << relayRole;
        binaryStream << relayPeerRank;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << peersPerKernel;
        binaryStream << relayTargetOf;
        binaryStream << relayStagingAddr;
        binaryStream << relayBytes;
        binaryStream << relaySliceBytes;
        binaryStream << relayOwnKernel;
        binaryStream << relayClosNoDoneKernel;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> usePull;
        binaryStream >> useLarge;
        binaryStream >> useRelay;
        binaryStream >> relayRole;
        binaryStream >> relayPeerRank;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> peersPerKernel;
        binaryStream >> relayTargetOf;
        binaryStream >> relayStagingAddr;
        binaryStream >> relayBytes;
        binaryStream >> relaySliceBytes;
        binaryStream >> relayOwnKernel;
        binaryStream >> relayClosNoDoneKernel;
    }
};

#endif // OPS_HCCL_CUSTOM_H
