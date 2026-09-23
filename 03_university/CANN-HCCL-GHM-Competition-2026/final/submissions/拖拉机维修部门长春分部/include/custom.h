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

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

struct CcuKernelArgScatter : public CcuKernelArgBase {
    uint32_t myRank;
    uint32_t root;
    uint32_t rankSize;
    uint32_t peerRanks[MAX_RANK_SIZE];
    uint32_t relayTargets[MAX_RANK_SIZE];
    uint32_t relayHelpers[MAX_RANK_SIZE];
    uint32_t relayHelper;
    uint32_t relayTarget;
    bool handleOwnCopy = false;
    // Relay paths can release the helper after its relay prefix reaches scratch.
    // The helper's own output completion is joined separately afterwards.
    bool relayPrefixFirst = false;
    // Reserved for a future per-tile path. It is disabled until every
    // cross-die helper has an explicit tile-level signal channel.
    bool relayTilePipeline = false;
    // Small-message direct path: receivers pull their slices after root
    // publishes its input address and token.
    bool receiverPull = false;
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
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<uint32_t> kernelRoles; ///< 1 marks the helper's second-hop kernel.
    uint32_t relayHelper = MAX_RANK_SIZE;
    uint32_t relayTarget = MAX_RANK_SIZE;
    uint32_t relayNumerator = 0;
    uint32_t relayDenominator = 1;
    uint32_t overlapOwnCopy = 0;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << kernelRoles;
        binaryStream << relayHelper;
        binaryStream << relayTarget;
        binaryStream << relayNumerator;
        binaryStream << relayDenominator;
        binaryStream << overlapOwnCopy;
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
        binaryStream >> kernelRoles;
        binaryStream >> relayHelper;
        binaryStream >> relayTarget;
        binaryStream >> relayNumerator;
        binaryStream >> relayDenominator;
        binaryStream >> overlapOwnCopy;
    }
};

#endif // OPS_HCCL_CUSTOM_H
