/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

// 北京科技大学南风的队伍：决赛自定义参数与通信计划定义。

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

// 所有 CCU Kernel 都通过这个公共前缀接收通信通道表。
struct CcuKernelArgBase {
    // 通道句柄的顺序必须与各算法计划中的 rank 数组保持一致。
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

// Kernel 模式用于选择远端直写路径或小消息拉取路径。
constexpr uint32_t ALLGATHER_KERNEL_DIRECT_WRITE = 0;
constexpr uint32_t ALLGATHER_KERNEL_SMALL_PULL = 1;

// 跨组、本地组和扁平路径的 CCU Kernel 共用的参数结构。
struct AllGatherKernelArg : public CcuKernelArgBase {
    // 集合通信使用的输入和输出设备缓冲区。
    void *sendBuf;
    void *recvBuf;
    // 总元素数量，以及本次 Kernel 负责处理的数据分片。
    uint64_t sendCount;
    uint64_t sliceOffset;
    uint64_t sliceCount;
    // 用于校验远端访问合法性的 CCU 内存令牌。
    uint64_t inputToken;
    uint64_t outputToken;
    // 运行时拓扑和数据类型元信息。
    uint32_t dataTypeSize;
    uint32_t myRank;
    uint32_t rankSize;
    // 标记本次调用是否同时复制本 Rank 的数据分片。
    uint32_t copyLocalOutput;
    uint32_t kernelMode;
    // 将通道槽位映射到该通道所代表的远端 Rank。
    uint32_t channelIndexToRank[MAX_RANK_SIZE];
};

// 当前评测中的 Rank-12/Rank-16 输入可由一次 CCU 传输完成；G2 保持单轮，
// 避免按子分片执行时产生额外的跨线程数据依赖。
constexpr uint32_t G2_PIPELINE_CHUNK_COUNT = 1;
constexpr uint32_t G2_MAX_FORWARD_TRANSFERS = 15;

// Gateway 算法第二阶段转发 Kernel 的参数结构。
struct G2ForwardKernelArg : public CcuKernelArgBase {
    void *recvBuf;
    uint64_t sendCount;
    uint64_t outputToken;
    uint32_t dataTypeSize;
    uint32_t forwardTransferCount;
    uint32_t forwardChannelIndices[G2_MAX_FORWARD_TRANSFERS];
    uint64_t forwardSliceOffsets[G2_MAX_FORWARD_TRANSFERS];
    uint64_t forwardSliceCounts[G2_MAX_FORWARD_TRANSFERS];
    uint32_t forwardSourceRanks[G2_MAX_FORWARD_TRANSFERS];
};

// 单层路径（小消息或扁平路径）缓存的执行资源。
struct FlatPlan {
    CcuKernelHandle kernel = 0;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> remoteRanks;
    uint32_t dieId = 0;
    uint32_t kernelMode = ALLGATHER_KERNEL_DIRECT_WRITE;
};

// Rank-12/Rank-16 拓扑缓存的执行资源和转发调度计划。
struct GatewayPlan {
    uint32_t valid = 0;
    uint32_t forwardSegmentCount = 0;
    CcuKernelHandle crossKernel = 0;
    CcuKernelHandle localKernel = 0;
    CcuKernelHandle forwardKernel = 0;
    std::vector<ChannelHandle> localChannels;
    std::vector<uint32_t> localRanks;
    std::vector<ChannelHandle> crossChannels;
    std::vector<uint32_t> crossRanks;
    std::vector<uint32_t> forwardChannelIndices;
    std::vector<uint32_t> forwardSourceRanks;
    std::vector<uint32_t> forwardSegmentBegins;
    std::vector<uint32_t> forwardSegmentSpans;
    uint32_t localDieId = 0;
    uint32_t crossDieId = 0;
};

// HcclAllGather 编译并由 ExecOp 后续消费的全部资源。
struct AlgResourceCtx {
    std::vector<ThreadHandle> threads;
    FlatPlan flat;
    GatewayPlan gateway;
    uint32_t myRank = 0;
    uint32_t rankSize = 0;
    uint32_t dataTypeSize = 0;

    // 将句柄和拓扑计划序列化后写入通信域缓存。
    std::vector<char> Serialize()
    {
        // 字段顺序必须保持稳定，DeSerialize 需要按相同顺序读取。
        BinaryStream stream;
        stream << threads;
        stream << flat.kernel << flat.channels << flat.remoteRanks << flat.dieId;
        stream << flat.kernelMode;
        stream << gateway.valid << gateway.forwardSegmentCount;
        stream << gateway.crossKernel << gateway.localKernel;
        stream << gateway.forwardKernel;
        stream << gateway.localChannels << gateway.localRanks;
        stream << gateway.crossChannels << gateway.crossRanks;
        stream << gateway.forwardChannelIndices << gateway.forwardSourceRanks;
        stream << gateway.forwardSegmentBegins << gateway.forwardSegmentSpans;
        stream << gateway.localDieId << gateway.crossDieId;
        stream << myRank << rankSize << dataTypeSize;
        std::vector<char> data;
        stream.Dump(data);
        return data;
    }

    // 在启动集合通信前恢复缓存的资源上下文。
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream stream(data);
        stream >> threads;
        stream >> flat.kernel >> flat.channels >> flat.remoteRanks >> flat.dieId;
        stream >> flat.kernelMode;
        stream >> gateway.valid >> gateway.forwardSegmentCount;
        stream >> gateway.crossKernel >> gateway.localKernel;
        stream >> gateway.forwardKernel;
        stream >> gateway.localChannels >> gateway.localRanks;
        stream >> gateway.crossChannels >> gateway.crossRanks;
        stream >> gateway.forwardChannelIndices >> gateway.forwardSourceRanks;
        stream >> gateway.forwardSegmentBegins >> gateway.forwardSegmentSpans;
        stream >> gateway.localDieId >> gateway.crossDieId;
        stream >> myRank >> rankSize >> dataTypeSize;
    }
};

#endif // OPS_HCCL_CUSTOM_H
