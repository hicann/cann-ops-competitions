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

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

constexpr uint32_t BROADCAST_MAX_PEERS = MAX_RANK_SIZE - 1U;
constexpr uint32_t BROADCAST_PHASE_COUNT = 3;

enum class BroadcastKernelMode : uint32_t {
    NOOP = 0,
    SCATTER_SEND = 1,
    SCATTER_RECV = 2,
    ALLGATHER = 3,
};

// Registered once for a (communicator, root) pair.  Non-root ranks have one
// channel, while the root owns one channel per destination rank.
struct BroadcastKernelArg {
    ChannelHandle channels[BROADCAST_MAX_PEERS]{};
    uint32_t channelCount = 0;
    uint32_t root = 0;
    uint32_t myRank = 0;
};

struct HierarchicalBroadcastKernelArg {
    ChannelHandle channels[BROADCAST_MAX_PEERS]{};
    uint64_t transferOffsets[BROADCAST_MAX_PEERS]{};
    uint64_t transferLengths[BROADCAST_MAX_PEERS]{};
    uint32_t channelCount = 0;
    BroadcastKernelMode mode = BroadcastKernelMode::NOOP;
    uint32_t groupSize = 0;
    uint32_t myIndex = 0;
    uint64_t dataBytes = 0;
};

struct AlgResourceCtx {
    ThreadHandle thread = 0;
    ThreadHandle auxiliaryThread = 0;
    CcuKernelHandle kernels[BROADCAST_PHASE_COUNT]{};
    uint32_t kernelCount = 0;
    bool hierarchical = false;

    std::vector<char> Serialize() const
    {
        BinaryStream stream;
        stream << thread << auxiliaryThread;
        for (CcuKernelHandle kernel : kernels) {
            stream << kernel;
        }
        stream << kernelCount << hierarchical;
        std::vector<char> result;
        stream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream stream(data);
        stream >> thread >> auxiliaryThread;
        for (CcuKernelHandle &kernel : kernels) {
            stream >> kernel;
        }
        stream >> kernelCount >> hierarchical;
    }
};

#endif // OPS_HCCL_CUSTOM_H