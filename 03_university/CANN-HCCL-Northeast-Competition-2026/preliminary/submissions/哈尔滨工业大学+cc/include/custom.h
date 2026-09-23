/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include <cstdint>
#include <vector>

#include "binary_stream.h"
#include "common.h"

// 通道通知槽位（common.h 定义 ACK=0 / DATA_SIGNAL=1；RELAY 为跨机中继方案新增槽位）
constexpr uint32_t NOTIFY_IDX_RELAY = 2;

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t stripeIdx = 0;       ///< 同一对端的条带通道索引（规则锁 1 通道/对端，恒为 0）
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    void *remoteOutput = nullptr; ///< 对端 recvBuf 地址（HcclCommMemReg 交换，root 单边直写目标）
    void *remoteInput = nullptr;  ///< 对端 sendBuf 地址（HcclCommMemReg 交换，peer 单边直读数据源）
    CommBuffer remoteCclMem;      ///< 对端 cclBuffer（中继载荷落点，root 写入 / peer 读取）
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;             ///< AICPU_TS通信引擎上的thread资源（主线程，兼 Host/Device 同步）
    std::vector<ThreadHandle> threads;    ///< [0]=主线程；[1..jobThreadNum]=写任务专属线程；[jobThreadNum+1..+copyThreadNum]=拷贝线程
    std::vector<ChannelInfo> channels;    ///< 每对端 1 条通道（赛题 2.2 合规约束）
    std::vector<uint8_t> peerIsLocal;     ///< 按 rank 索引：1=同 server（layer-0 直连），0=跨机
    uint64_t smallMsgThresholdBytes = 0;  ///< 大小消息路径切换阈值（每 rank 字节数，host 侧 env 可调）
    uint32_t jobThreadNum = 0;            ///< 写任务线程数。同步写原语会阻塞所在线程直至传输完成，
                                          ///< 写并发度由线程数决定，故每个写任务独占一线程
    uint32_t copyThreadNum = 0;           ///< 自留块本地拷贝线程数（AICPU 单线程拷贝与网络写同量级，
                                          ///< 须并行拷贝避免成为临界路径）
    uint8_t relayEnabled = 0;             ///< 跨机中继分流开关（env HCCL_SCATTER_RELAY；三侧公式一致）
    uint8_t nbiEnabled = 0;               ///< 小消息 NBI 直发开关（env HCCL_SCATTER_NBI；主线程无句柄融合写+对端ACK）
    CommBuffer localBuffer;               ///< 本端 cclBuffer（中继节点转发载荷的读取源）

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << peerIsLocal;
        binaryStream << smallMsgThresholdBytes;
        binaryStream << jobThreadNum;
        binaryStream << copyThreadNum;
        binaryStream << relayEnabled;
        binaryStream << nbiEnabled;
        binaryStream << localBuffer;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> threads;
        binaryStream >> channels;
        binaryStream >> peerIsLocal;
        binaryStream >> smallMsgThresholdBytes;
        binaryStream >> jobThreadNum;
        binaryStream >> copyThreadNum;
        binaryStream >> relayEnabled;
        binaryStream >> nbiEnabled;
        binaryStream >> localBuffer;
    }
};

#endif // OPS_HCCL_CUSTOM_H
