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
#include <unordered_map>
#include <vector>

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

// ===========================================================================
// Channel 通信相关数据结构
// ===========================================================================

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    uint32_t netLayer = 0; ///< 网络层：0=Server内，1=跨Server；注意 netLayer 不等于 IO die
    uint32_t localDie = 0; ///< 本地端点真实 IO die，CCU kernel 必须按该字段分组/注册
    ChannelHandle handle = 0;
};

// ===========================================================================
// CCU Kernel Arg 基类与 ReduceScatter 特化
// ===========================================================================

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

// ReduceScatter CCU Kernel 静态参数（Host 侧构造，随 Kernel 注册固化）
// 继承 CcuKernelArgBase 以兼容 CcuKernelInfo::setKernelArg 的模板类型擦除
struct ReduceScatterKernelArgHost : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    HcclDataType dataType;
    HcclReduceOp reduceOp;
    bool initOutput;     // true=当前 kernel 归约集合包含本 rank 的输入
    bool reduceToOutput; // true=将局部和累加到 output；false=复制到 output
    bool skipOutput;     // true=局部和留在 scratch，由 host 后处理
    uint32_t stepOffset; // RH: 当前 die 的起始全局步号
};

constexpr uint32_t RS_ALGO_MESH = 0;
constexpr uint32_t RS_ALGO_RH = 1;
constexpr uint32_t RS_ALGO_GROUP_DRR = 2;
constexpr uint32_t RS_ALGO_SMALL_4X1 = 3;

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

// ===========================================================================
// AlgResourceCtx —— 序列化到通信引擎上下文的资源包
// ===========================================================================

struct AlgResourceCtx {
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<ChannelInfo> channels;       ///< CCU通信引擎上的channel资源（所有）
    std::vector<ChannelInfo> dieChannels[2]; ///< 按本地端点真实 IO die 分组的 channel
    uint64_t localCclToken = 0;              ///< 本端 CCL Buffer 的 CCU 访问 token
    uint32_t algoType = RS_ALGO_MESH;        ///< 当前资源注册的算法类型

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << channels;
        binaryStream << dieChannels[0];
        binaryStream << dieChannels[1];
        binaryStream << localCclToken;
        binaryStream << algoType;
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
        binaryStream >> channels;
        binaryStream >> dieChannels[0];
        binaryStream >> dieChannels[1];
        binaryStream >> localCclToken;
        binaryStream >> algoType;
    }
};

// ===========================================================================
// 公共扩展函数（Host/Device 共用）
// ===========================================================================

namespace ops_hccl {

// HCCL 数据类型 -> Hcomm 数据类型
inline HcommDataType ToHdt(HcclDataType t)
{
    return static_cast<HcommDataType>(t);
}

// HCCL 归约类型 -> Hcomm 归约类型
inline HcommReduceOp ToHop(HcclReduceOp o)
{
    return static_cast<HcommReduceOp>(o);
}

// 获取数据类型字节数（FP32 快速路径 + SIZE_TABLE 兜底）
inline uint32_t GetElemSize(HcclDataType dt)
{
    if (dt == HCCL_DATA_TYPE_FP32) {
        return sizeof(float);
    }
    const auto it = SIZE_TABLE.find(dt);
    return (it != SIZE_TABLE.end()) ? it->second : 0;
}

inline uint64_t EstimateDieReadCost(const std::vector<ChannelInfo> &channels)
{
    uint64_t innerCount = 0;
    uint64_t closCount = 0;
    for (const auto &ch : channels) {
        if (ch.netLayer == 1) {
            closCount++;
        } else {
            innerCount++;
        }
    }

    // Scale by 4: one server-internal link costs 4 units, while the shared Clos
    // uplink has roughly 4x that bandwidth, so N cross-server peers cost N units.
    const uint64_t innerCost = (innerCount == 0) ? 0U : 4U;
    const uint64_t closCost = closCount;
    return (innerCost > closCost) ? innerCost : closCost;
}

inline uint64_t EstimateDiePipelineCost(const std::vector<ChannelInfo> &channels)
{
    // Remote read bandwidth alone underestimates a die with many peers: after
    // reads finish, that die also performs roughly one local-reduce pass per
    // peer-sized chunk. Keep the weight modest so bandwidth still dominates.
    constexpr uint64_t LOCAL_REDUCE_COST_PER_PEER = 2U;
    return EstimateDieReadCost(channels) + LOCAL_REDUCE_COST_PER_PEER * channels.size();
}

inline uint32_t SelectMeshOutputDie(const AlgResourceCtx &resCtx)
{
    const bool hasDie0 = !resCtx.dieChannels[0].empty();
    const bool hasDie1 = !resCtx.dieChannels[1].empty();
    if (!hasDie0 || !hasDie1) {
        return hasDie0 ? 0U : 1U;
    }

    const uint64_t die0Cost = EstimateDiePipelineCost(resCtx.dieChannels[0]);
    const uint64_t die1Cost = EstimateDiePipelineCost(resCtx.dieChannels[1]);
    if (die0Cost != die1Cost) {
        return (die0Cost < die1Cost) ? 0U : 1U;
    }
    if (resCtx.dieChannels[0].size() != resCtx.dieChannels[1].size()) {
        return (resCtx.dieChannels[0].size() <= resCtx.dieChannels[1].size()) ? 0U : 1U;
    }
    return 0U;
}

inline uint32_t GetGroupDrrMaxGroupSize(uint32_t channelCount)
{
    (void)channelCount;
    return 2U;
}

inline uint32_t GetGroupDrrGroupCount(uint32_t channelCount, uint32_t maxGroupSize)
{
    if (channelCount == 0 || maxGroupSize == 0) {
        return 0;
    }
    return (channelCount + maxGroupSize - 1U) / maxGroupSize;
}

inline uint32_t GetGroupDrrGroupSize(uint32_t channelCount, uint32_t maxGroupSize, uint32_t groupIdx)
{
    if (maxGroupSize == 0) {
        return 0;
    }
    const uint32_t start = groupIdx * maxGroupSize;
    if (start >= channelCount) {
        return 0;
    }
    const uint32_t remain = channelCount - start;
    return (remain < maxGroupSize) ? remain : maxGroupSize;
}

inline uint32_t SelectGroupDrrOutputDie(const AlgResourceCtx &resCtx)
{
    const uint32_t die0Count = static_cast<uint32_t>(resCtx.dieChannels[0].size());
    const uint32_t die1Count = static_cast<uint32_t>(resCtx.dieChannels[1].size());
    if (die0Count != die1Count) {
        return (die0Count > die1Count) ? 0U : 1U;
    }
    return SelectMeshOutputDie(resCtx);
}

// 建立 remoteRank -> ChannelInfo* 的直接索引数组
inline void BuildChannelArray(const std::vector<ChannelInfo> &channels, const ChannelInfo *chArray[MAX_RANK_SIZE])
{
    for (uint32_t i = 0; i < MAX_RANK_SIZE; ++i) {
        chArray[i] = nullptr;
    }
    for (const auto &ch : channels) {
        if (ch.remoteRank < MAX_RANK_SIZE) {
            chArray[ch.remoteRank] = &ch;
        }
    }
}

} // namespace ops_hccl

#endif // OPS_HCCL_CUSTOM_H
