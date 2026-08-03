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

#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

constexpr uint32_t THREAD_NOTIFY_NUM = 2;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t peerRanks[MAX_RANK_SIZE];
    uint32_t channelCount = 0;
};

struct AlgResourceCtx {
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> directKernels;
    CcuKernelHandle relayKernel = 0;
    uint32_t relayEnabled = 0;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << threads;
        binaryStream << directKernels;
        binaryStream << relayKernel;
        binaryStream << relayEnabled;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> threads;
        binaryStream >> directKernels;
        binaryStream >> relayKernel;
        binaryStream >> relayEnabled;
    }
};

#endif // OPS_HCCL_CUSTOM_H
