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

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

// 小数据使用较少的通信线程，降低线程启动和汇合的固定开销。
// 口径按“单 rank 接收量”计算（recvCount * dataTypeSize）。
constexpr uint64_t SCATTER_SMALL_DATA_THRESHOLD = 1ULL * 1024 * 1024;
constexpr uint32_t SCATTER_SMALL_COMM_THREAD_NUM = 2;

// 本轮在总输入不超过512KiB时试验Read；就绪及输入生命周期由外层保证。
constexpr uint64_t SCATTER_READ_MAX_TOTAL_SIZE = 512ULL * 1024;

inline bool ScatterUseTinyRead(uint32_t rankSize, uint64_t recvSize)
{
    return rankSize > 1 && recvSize != 0 && recvSize <= SCATTER_READ_MAX_TOTAL_SIZE / rankSize;
}

// ---- 跨 Server 段的 Clos 出口分摊（大数据路径）----
// root 的 Mesh 总出口 = 7 条链路；Clos 出口只有 1 个接入口，带宽为单条 Mesh 的 R 倍。
// 朴素直发时跨机 8 份全挤 root 一个 Clos 口，耗时 8/R，是 Mesh 侧(=1)的 8/R 倍。
// 解法：把每份跨机数据切成 root 直发段(alpha) + 邻居代发段(1-alpha)。
// 代发采用一对一：A 内 local=i 的邻居代发给 B 内 local=i 的卡（i != rootLocal）；
// B 内与 root 同 local 的那张卡没有代发搭档，由 root 全量直发。
//
// 负载（归一化 B_mesh = 1，单 rank 接收量 = S = 1）：
//   root 每条 Mesh = 1(邻居自己那份) + (1-alpha)(代发段)     = 2 - alpha
//   root Clos 口   = (7*alpha + 1) / R
//   邻居 Clos 口   = (1 - alpha) / R                          （很轻）
// 邻居先从 root 读取待代发段，再并行转发和读取自己的整份。
// 假设 Mesh 与 Clos 可并发且带宽恒定，其通信时间近似为：
//   T = max((7*alpha+1)/R, 2-alpha, (1-alpha)*(1+1/R))。
// 当 (1-alpha)/R <= 1，整段转发可被邻居自己那份的 Mesh 传输覆盖。
// 此时配平 root 两类出口得 alpha=(2R-1)/(7+R)，按题面 R 约为 4 取 7/11。
// v12按单rank接收量选择分段比例：512 MiB档使用0.617，400 MiB+4 B档使用0.613。
// Host建链和AICPU数据面必须使用同一选择。
// 模型未计入本地拷贝、通知、下发和共享资源竞争，平台实测才能判断净收益。
// 每个邻居仍只转发搭档的整个后缀，不引入分块流水；用本地线程同步衔接 Read 与转发。
// 分段按元素取整，余量并入后段；验收与实测记录见 VALIDATION.md。
constexpr uint64_t SCATTER_ALPHA_DEN = 1000;
constexpr uint64_t SCATTER_ALPHA_LARGE_NUM = 617;
constexpr uint64_t SCATTER_ALPHA_DEFAULT_NUM = 613;
constexpr uint64_t SCATTER_ALPHA_LARGE_RECV_SIZE = 512ULL * 1024 * 1024;

inline uint64_t ScatterAlphaNum(uint64_t recvSize)
{
    return recvSize >= SCATTER_ALPHA_LARGE_RECV_SIZE
        ? SCATTER_ALPHA_LARGE_NUM : SCATTER_ALPHA_DEFAULT_NUM;
}
// 小于或等于该阈值时不做分摊，直接走朴素直发路径。
constexpr uint64_t SCATTER_RELAY_MIN_RECV_SIZE = 1ULL * 1024 * 1024;

// 评测拓扑：2 Server x 8 NPU
constexpr uint32_t SCATTER_RANKS_PER_SERVER = 8;
constexpr uint32_t SCATTER_RANKS_TOTAL = 16;

// root 直发段字节数 = recvSize * alpha，按 dataTypeSize 对齐；
// 余量归入代发段，保证两段地址连续且完整覆盖。
inline uint64_t ScatterDirectSize(uint64_t recvSize, uint32_t dataTypeSize)
{
    const uint64_t alphaNum = ScatterAlphaNum(recvSize);
    if (alphaNum == 0) {
        return 0;
    }
    if (alphaNum >= SCATTER_ALPHA_DEN) {
        return recvSize;
    }
    uint64_t elemCount = recvSize / dataTypeSize;
    uint64_t unit = elemCount / SCATTER_ALPHA_DEN;
    return unit * alphaNum * dataTypeSize;
}

// 本 rank 在本次 Scatter 中承担的角色。
// RELAY 只在大数据分摊路径下出现。
constexpr uint32_t SCATTER_ROLE_NONE = 0;   // 仅接收自己那份
constexpr uint32_t SCATTER_ROLE_ROOT = 1;   // 数据源
constexpr uint32_t SCATTER_ROLE_RELAY = 2;       // 同 Server 邻居，代发一份给跨机搭档
constexpr uint32_t SCATTER_ROLE_RELAY_RECV = 3;  // 跨机接收方，后段由代发邻居写入
constexpr uint32_t SCATTER_ROLE_FULL_READ = 4;   // 无代发搭档的跨机叶子，从root读取整份

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
    // Scatter 需要直接写到对端用户输出 Buffer，这里缓存建链时交换到的远端地址。
    CommBuffer remoteOutputMem;
    // root注册的完整输入，供小数据接收方、大数据中转及跨机叶子主动读取。
    CommBuffer remoteInputMem;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;          ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< AICPU_TS通信引擎上的thread资源
    std::vector<ChannelInfo> channels; ///< AICPU_TS通信引擎上的channel资源

    uint32_t role = SCATTER_ROLE_NONE;
    // RELAY 角色：代发目标 rank 及其在 channels[] 中的下标
    uint32_t relayPeerRank = INVALID_VALUE_RANKID;
    uint32_t relayChannelIdx = 0;
    // 非 root 视角：与 root 那条 channel 在 channels[] 中的下标
    uint32_t rootChannelIdx = 0;
    // root视角：与root同local id的跨机rank，没有代发搭档，由它从root读取整份
    uint32_t closDirectFullRank = INVALID_VALUE_RANKID;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << role;
        binaryStream << relayPeerRank;
        binaryStream << relayChannelIdx;
        binaryStream << rootChannelIdx;
        binaryStream << closDirectFullRank;
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
        binaryStream >> role;
        binaryStream >> relayPeerRank;
        binaryStream >> relayChannelIdx;
        binaryStream >> rootChannelIdx;
        binaryStream >> closDirectFullRank;
    }
};

#endif // OPS_HCCL_CUSTOM_H
