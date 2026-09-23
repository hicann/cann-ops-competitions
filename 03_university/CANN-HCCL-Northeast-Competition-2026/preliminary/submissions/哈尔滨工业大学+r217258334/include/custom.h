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

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem = {nullptr, 0};
    CommBuffer remoteUserOutput = {nullptr, 0}; ///< 对端通过HcclCommMemReg暴露的recvBuf
    uint32_t hasRemoteUserOutput = 0;
    CommBuffer remoteUserInput = {nullptr, 0};  ///< root在小包路径暴露的完整sendBuf
    uint32_t hasRemoteUserInput = 0;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;          ///< AICPU_TS通信引擎上的主thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< threads[0]为主thread，threads[1]为并发辅助thread
    std::vector<ChannelInfo> channels; ///< 以remote rank为下标；本rank位置无有效channel

    // 由RankGraph第一层推导出的本Server/另一Server rank集合，均按rank id升序。
    // 2x8拓扑下用两组相同下标构造一一对应的Clos relay。
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> remoteRanks;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << localRanks;
        binaryStream << remoteRanks;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
        binaryStream >> localRanks;
        binaryStream >> remoteRanks;
    }
};

#endif // OPS_HCCL_CUSTOM_H
