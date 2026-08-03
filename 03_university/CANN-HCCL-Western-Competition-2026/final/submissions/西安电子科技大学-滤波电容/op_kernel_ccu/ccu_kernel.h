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

// Phase values passed as task argument slot 7.
// Tree phases use [0, chunkSize). Ordered reduce-scatter/all-gather phases
// use [transferOffset, transferOffset + transferSize).
enum class PairPhase : uint64_t {
    REDUCE_RECEIVE      = 0, // parent: read child output to scratch, LocalReduce into own output
    REDUCE_SEND         = 1, // child:  passthrough (parent drives reduction)
    REDUCE_RECEIVE_INIT = 2, // parent: init output from input first, then reduce
    REDUCE_SEND_INIT    = 3, // child:  init output from input first (out-of-place, leaf)
    BROADCAST_SEND      = 4, // parent: write own output to child output
    BROADCAST_RECEIVE   = 5, // child:  receive broadcast (passthrough; parent drives)
    // WriteReduce variants: child actively pushes and reduces into parent output.
    REDUCE_WR_SEND      = 6, // child:  WriteReduce accumulated output into remote output
    REDUCE_WR_SEND_INIT = 7, // leaf:   WriteReduce input directly into remote output
    REDUCE_WR_RECV      = 8, // parent: wait for child WriteReduce, no local op
    REDUCE_WR_RECV_INIT = 9, // parent: init output from input first, wait for WriteReduce
    RHD_REDUCE_SEND_FIRST = 10, // first level: WriteReduce input range into peer output
    RHD_REDUCE_RECV_FIRST = 11, // first level: initialize the locally retained range
    RHD_REDUCE_SEND       = 12, // later level: WriteReduce output range into peer output
    RHD_REDUCE_RECV       = 13, // later level: wait for peer WriteReduce
    RHD_GATHER_SEND       = 14, // write the locally owned range into peer output
    RHD_GATHER_RECV       = 15, // wait for peer write
    RHD_REDUCE_BIDIR_FIRST = 16, // both peers initialize keep ranges and reduce input ranges
    RHD_REDUCE_BIDIR       = 17, // both peers reduce output ranges into each other
    RHD_GATHER_BIDIR       = 18, // both peers write owned ranges into each other
    RHD_REDUCE_BIDIR_SCRATCH = 19, // both peers reduce stable scratch copies into each other
};

struct CcuKernelArgAllReduce : CcuKernelArgBase {
    uint32_t rankId;
    uint32_t peerRank;
    HcclDataType dataType;
    HcclReduceOp reduceType;
};

struct CcuKernelArgShardedMesh : CcuKernelArgBase {
    uint32_t rankId;
    uint32_t rankSize;
    HcclDataType dataType;
    HcclReduceOp reduceType;
    uint32_t peerRanks[MAX_RANK_SIZE];
    uint32_t scratchLanes[MAX_RANK_SIZE];
    uint32_t secondaryPartialLane;
    bool primary;
};

// CCU kernel entry point.
CcuResult CcuKernel(CcuKernelArg arg);
CcuResult CcuMeshBootstrapKernel(CcuKernelArg arg);
CcuResult CcuShardedMeshKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
