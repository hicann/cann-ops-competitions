/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_EXEC_OP_H
#define OPS_HCCL_CCU_EXEC_OP_H

#include <hccl/hcomm_primitives.h>
#include "common.h"
#include "custom.h"

namespace ops_hccl {

enum class TopologyType : uint32_t {
    INVALID = 0,
    SINGLE_RANK = 1,
    MESH_4X1 = 2,
    DUAL_SERVER_2X8 = 3,
    ASYMMETRIC_8P4 = 4,
    GENERIC = 5,
};

enum class PayloadType : uint32_t {
    LATENCY = 0,
    BANDWIDTH_ALIGNED = 1,
    BANDWIDTH_TAIL = 2,
};

struct SlicePair {
    uint64_t first = 0;
    uint64_t second = 0;

    uint64_t Total() const
    {
        return first + second;
    }
};

struct ExecPlan {
    TopologyType topology = TopologyType::INVALID;
    PayloadType payload = PayloadType::LATENCY;
    uint64_t rankBytes = 0;
    uint64_t outputBytes = 0;
    bool inlineOwnCopy = false;
};

// 4x1双平面统一流量计划。seedBytes经64B向下对齐，所有奇尾均落入
// suffixBytes；二者都不超过CCU单次通信256MB上限。
struct Balanced4x1Flow {
    uint64_t batchBytes = 0;
    uint64_t seedBytes = 0;
    uint64_t suffixBytes = 0;

    uint64_t Total() const
    {
        return seedBytes + suffixBytes;
    }
};

constexpr uint64_t INLINE_COPY_THRESHOLD_BYTES = 1ULL * 1024ULL * 1024ULL;
constexpr uint64_t DMA_ALIGNMENT_BYTES = 64ULL;
constexpr uint32_t MAX_CHANNELS_PER_DIE = 8;
constexpr uint32_t MAX_CCU_TASK_ARGS = 13;

ExecPlan SelectExecPlan(const OpParam &param, uint32_t topologyType);
SlicePair BuildSlicePair(const ExecPlan &plan, uint64_t remainingBytes);
Balanced4x1Flow BuildBalanced4x1Flow(uint64_t remainingBytes);

// 执行算法任务编排
HcclResult ExecOp(const OpParam &param);
HcclResult Exec4x1Exact512WarmOp(const OpParam &param);
} // namespace ops_hccl
#endif // OPS_HCCL_CCU_EXEC_OP_H
