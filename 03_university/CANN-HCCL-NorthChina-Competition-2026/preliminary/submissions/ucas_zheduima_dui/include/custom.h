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
    void *addr = nullptr;
    uint64_t size = 0;
} CommBuffer;

// HCCL Buffer被切分为PIPELINE_REGION_NUM个等长区域：前PIPELINE_SLOT_NUM个为流水线复用slot，
// 最后一个为首个chunk专用区域（由全体Worker并行填充，消除主Thread串行拷贝的启动瓶颈）。
constexpr uint32_t PIPELINE_SLOT_NUM = 3;
constexpr uint32_t PIPELINE_REGION_NUM = PIPELINE_SLOT_NUM + 1;
constexpr uint64_t PIPELINE_SLICE_ALIGNMENT = 4096;
// 小消息优先减少任务数；超过该阈值且完整输入可放入CCL Buffer时，改用并行全量staging。
constexpr uint64_t SERIAL_SINGLE_SHOT_MAX_SIZE = 512ULL * 1024;

// 通用流水路径复用前8个Channel Notify，每个Notify在整个算子内只有一个Record方和一个Wait方：
// [0]                首chunk READY，由主Thread聚合全部Worker的staged上报后Record
// [1, 3]             流水线slot READY，由独占该slot的拷贝Thread Record
// [4, 6]             流水线slot READ_DONE，由读取该slot的对端Worker Record
// [7]                首chunk READ_DONE，防止下一次算子提前覆盖专用区域
constexpr uint32_t CHANNEL_NOTIFY_CHUNK0_READY = 0;
constexpr uint32_t CHANNEL_NOTIFY_READY_BASE = 1;
constexpr uint32_t CHANNEL_NOTIFY_READ_DONE_BASE = CHANNEL_NOTIFY_READY_BASE + PIPELINE_SLOT_NUM;
constexpr uint32_t CHANNEL_NOTIFY_CHUNK0_READ_DONE = CHANNEL_NOTIFY_READ_DONE_BASE + PIPELINE_SLOT_NUM;

// 大消息mate-owner连续流水：每个rank将输入按2MiB连续块Write到对面同位置rank的CCL。
// mate Channel的[0, 15]用于逐chunk STAGE，[16]用于最终RELEASE；其余14个Channel的
// [0, 15]用于逐chunk READY，[16]用于最终DONE。每个物理Notify每次调用严格一打一收。
constexpr uint64_t MATE_PIPELINE_CHUNK_SIZE = 2ULL * 1024 * 1024;
constexpr uint32_t MATE_PIPELINE_MAX_CHUNK_NUM = 16;
constexpr uint32_t CHANNEL_NOTIFY_MATE_FINAL = MATE_PIPELINE_MAX_CHUNK_NUM;
constexpr uint32_t CHANNEL_NOTIFY_NUM = CHANNEL_NOTIFY_MATE_FINAL + 1;

// copier0逐chunk接收mate STAGE后，用copier1的[1, 16]号本地Notify转发就绪信号；
// copier1的[0]号Notify仍用于START。chunk使用独占Notify，允许前后块安全重叠下发。
constexpr uint32_t THREAD_NOTIFY_MATE_CHUNK_BASE = 1;
constexpr uint32_t THREAD_NOTIFY_MATE_RELAY_NUM = THREAD_NOTIFY_MATE_CHUNK_BASE + MATE_PIPELINE_MAX_CHUNK_NUM;
// copier0的1号Notify独占用于接收copier1的最终完成信号，避免与0号START复用。
constexpr uint32_t THREAD_NOTIFY_MATE_COPIER_DONE = 1;

// 双来源大消息路径复用同一组Channel Notify：本端CCL同时保存own-A和mate-B。
// mate Channel的idx 1用于B写入完成握手，其余Channel的idx 1用于B_READY；二者物理资源不同。
constexpr uint32_t CHANNEL_NOTIFY_DUAL_A_READY = CHANNEL_NOTIFY_CHUNK0_READY;
constexpr uint32_t CHANNEL_NOTIFY_DUAL_B_READY_OR_STAGE = CHANNEL_NOTIFY_READY_BASE;
constexpr uint32_t CHANNEL_NOTIFY_DUAL_RELEASE = CHANNEL_NOTIFY_READY_BASE + 1;
constexpr uint32_t CHANNEL_NOTIFY_DUAL_PEER_DONE = CHANNEL_NOTIFY_CHUNK0_READ_DONE;

// 8路空间条带大消息路径：每个rank把第j条带写入远端组owner-j的CCL。跨服Worker在
// Write后Record STAGE_READY，对端main收齐8个Wait后发布BUNDLE_READY。消费者
// 完成读取后回DATA_DONE，owner收齐全部消费者与本地拷贝后再向8个跨服writer发RELEASE，
// 从而保护CCL在下一次调用中不被提前覆盖。四个Notify均为每Channel每轮一打一收。
constexpr uint32_t CHANNEL_NOTIFY_STRIPE_STAGE_READY = CHANNEL_NOTIFY_CHUNK0_READY;
constexpr uint32_t CHANNEL_NOTIFY_STRIPE_BUNDLE_READY = CHANNEL_NOTIFY_READY_BASE;
constexpr uint32_t CHANNEL_NOTIFY_STRIPE_OWNER_RELEASE = CHANNEL_NOTIFY_READY_BASE + 1;
constexpr uint32_t CHANNEL_NOTIFY_STRIPE_DATA_DONE = CHANNEL_NOTIFY_CHUNK0_READ_DONE;

// 注册输出直推路径复用同一组Channel Notify：[0]为对端Write落盘READY（对端Worker Record、
// 本端main Wait），[2]为RELEASE（本端main收齐全部READY后Record、对端Worker Wait），
// 防止下一次调用提前覆盖本端recvBuf。两个Notify均为每Channel每轮一打一收。
constexpr uint32_t CHANNEL_NOTIFY_PUSH_READY = CHANNEL_NOTIFY_CHUNK0_READY;
constexpr uint32_t CHANNEL_NOTIFY_PUSH_RELEASE = CHANNEL_NOTIFY_READY_BASE + 1;

// 拷贝Thread数量，与流水线slot一一对应，避免同一Notify出现多个Record方
constexpr uint32_t COPIER_THREAD_NUM = PIPELINE_SLOT_NUM;

// 竞赛拓扑专用快速路径：2个连续编号的8-rank layer-0组，layer-1覆盖全部16个rank。
// 拓扑关系由Host查询并验证后写入Context；验证失败时Device自动回退通用直连路径。
constexpr uint32_t OPTIMIZED_RANK_NUM = 16;
constexpr uint32_t OPTIMIZED_GROUP_SIZE = 8;
constexpr uint32_t RECURSIVE_DOUBLING_ROUND_NUM = 4;

// Worker/拷贝Thread的0号Notify：主Thread下发的启动信号
constexpr uint32_t THREAD_NOTIFY_START = 0;
// 主Thread的Notify布局（workerNum = rankSize - 1）：
// [0]                            Host/AICPU同步（aicpu_kernel.cc使用）
// [1, workerNum]                 各Worker的首chunk切片staged上报
// [workerNum + 1, 2 * workerNum] 各Worker完成上报
// [2 * workerNum + 1, 2 * workerNum + COPIER_THREAD_NUM] 各拷贝Thread完成上报
constexpr uint32_t MAIN_NOTIFY_STAGED_BASE = 1;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    // Proxy路径中，该Channel对端CCL Buffer里承载的数据源rank。
    uint32_t representedRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread = 0;      ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< AICPU_TS通信引擎上的thread资源
    std::vector<ChannelInfo> channels; ///< AICPU_TS通信引擎上的channel资源

    // 经Host严格验证的2x8拓扑计划。topologyPlanValid为0时其余字段不可使用。
    uint32_t topologyPlanValid = 0;
    uint32_t localGroupBaseRank = INVALID_VALUE_RANKID;
    uint32_t remoteGroupBaseRank = INVALID_VALUE_RANKID;
    uint32_t localGroupIndex = INVALID_VALUE_RANKID;
    uint32_t proxyRank = INVALID_VALUE_RANKID;
    std::vector<uint32_t> recursiveDoublingChannelIndices;

    // 注册输出直推路径：registeredOutput记录首次建资源时注册的本端recvBuf；
    // remoteOutputs与channels一一对应。数组为空表示注册/交换不可用，Device回退原有路径。
    CommBuffer registeredOutput;
    std::vector<CommBuffer> remoteOutputs;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << topologyPlanValid;
        binaryStream << localGroupBaseRank;
        binaryStream << remoteGroupBaseRank;
        binaryStream << localGroupIndex;
        binaryStream << proxyRank;
        binaryStream << recursiveDoublingChannelIndices;
        binaryStream << registeredOutput;
        binaryStream << remoteOutputs;
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
        binaryStream >> topologyPlanValid;
        binaryStream >> localGroupBaseRank;
        binaryStream >> remoteGroupBaseRank;
        binaryStream >> localGroupIndex;
        binaryStream >> proxyRank;
        binaryStream >> recursiveDoublingChannelIndices;
        binaryStream >> registeredOutput;
        binaryStream >> remoteOutputs;
    }
};

#endif // OPS_HCCL_CUSTOM_H
