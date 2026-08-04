/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

#include "custom.h"

namespace ops_hccl {

constexpr uint64_t CCU_REDUCE_SCATTER_MODE_REDUCE_PEERS = 0;
constexpr uint64_t CCU_REDUCE_SCATTER_MODE_MERGE_PARTIALS = 1;
constexpr uint64_t CCU_REDUCE_SCATTER_MODE_REDUCE_PEERS_SCRATCH = 2;
constexpr uint64_t CCU_REDUCE_SCATTER_MODE_MS_PIPELINE = 3;
constexpr uint64_t CCU_REDUCE_SCATTER_MODE_REDUCE_REMOTE = 4;
constexpr uint64_t CCU_REDUCE_SCATTER_MODE_IN_PLACE_SCRATCH = 5;
constexpr uint64_t CCU_REDUCE_SCATTER_MODE_OMNIPIPE_MESH = 6;

constexpr uint32_t CCU_REDUCE_SCATTER_KERNEL_GENERIC = 0;
constexpr uint32_t CCU_REDUCE_SCATTER_KERNEL_OMNIPIPE_MESH = 1;
constexpr uint32_t CCU_REDUCE_SCATTER_KERNEL_NHR = 2;

struct CcuKernelArgReduceScatter : public CcuKernelArgBase {
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
    bool destinationInitialized = false;
    uint32_t kernelKind = CCU_REDUCE_SCATTER_KERNEL_GENERIC;
    uint32_t axisRankSize = 0;
    uint32_t axisRankIndex = 0;
    uint32_t peerSubRanks[MAX_RANK_SIZE]{};
};

CcuResult CcuReduceScatterReadReduceKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterOmnipipeMeshKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterOmnipipeNhrKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
