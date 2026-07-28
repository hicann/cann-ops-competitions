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

#include <cstdint>

#include <ccu/ccu_types.h>

#include "custom.h"

namespace ops_hccl {

enum class DirectKernelPhase : uint32_t {
    INPUT_RECORD = 0,
    ONESHOT_READ_RECORD = 1,
    ONESHOT_DONE_WAIT = 2,
    ONESHOT_LOCAL_REDUCE = 3,
    DIRECT_REDUCE_SCATTER = 4,
    DIRECT_ALL_GATHER_RECORD = 5,
    DIRECT_ALL_GATHER_WAIT = 6,
    PARALLEL_REDUCE_SCATTER = 7,
    PARALLEL_REDUCE_MERGE = 8,
    DIRECT_BUFFER_REDUCE_SCATTER = 9,
    DIRECT_BUFFER_MERGE = 10,
    DIRECT_BUFFER_ALL_GATHER = 11,
    FUSED_ONESHOT_COMMUNICATION = 12,
    FUSED_ONESHOT_ALL_REDUCE = 13,
    RANK16_CLOS_PAIR_TREE = 14,
};

struct CcuKernelArgGroup : CcuKernelArgBase {
    uint32_t myRank = 0;
    uint32_t rankSize = 0;
    uint32_t groupIndex = 0;
    uint32_t groupCount = 0;
    uint32_t remoteRanks[MAX_RANK_SIZE]{};
};

CcuResult CcuGroupedPairKernel(CcuKernelArg arg);
CcuResult CcuParallelRsKernel(CcuKernelArg arg);
CcuResult CcuDirectBufferKernel(CcuKernelArg arg);
CcuResult CcuRank12ClosTreeKernel(CcuKernelArg arg);
CcuResult CcuFourRankNhrKernel(CcuKernelArg arg);

// F081 non-production registration-capacity probes.  Each wrapper preserves
// the F066 task graph byte-for-byte while giving every scored route an
// independent registration entry and handle.
CcuResult CcuRank4SmallProbeKernel(CcuKernelArg arg);
CcuResult CcuRank4Large512ProbeKernel(CcuKernelArg arg);
CcuResult CcuRank4Large400MiB4BProbeKernel(CcuKernelArg arg);
CcuResult CcuRank12SmallProbeKernel(CcuKernelArg arg);
CcuResult CcuRank12Large512ProbeKernel(CcuKernelArg arg);
CcuResult CcuRank12Large400MiB4BProbeKernel(CcuKernelArg arg);
CcuResult CcuRank16SmallProbeKernel(CcuKernelArg arg);
CcuResult CcuRank16Large512ProbeKernel(CcuKernelArg arg);
CcuResult CcuRank16Large400MiB4BProbeKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
