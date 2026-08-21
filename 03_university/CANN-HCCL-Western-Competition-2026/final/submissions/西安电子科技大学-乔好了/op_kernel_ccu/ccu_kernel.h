/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef HCCL_COMPETITION_ALLREDUCE_246_CCU_KERNEL_H_
#define HCCL_COMPETITION_ALLREDUCE_246_CCU_KERNEL_H_

#include <ccu/ccu_event.hpp>
#include <ccu/ccu_primitives.hpp>
#include <ccu/ccu_types.h>
#include <ccu/ccu_variable.hpp>
#include <hccl/hcomm_primitives.h>

#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

constexpr uint32_t CCU_MAX_RANK_SIZE = CcuKernelArgBase::CCU_MAX_RANK_SIZE;
static_assert(CCU_MAX_RANK_SIZE == MAX_RANK_SIZE, "CCU rank capacity mismatch");

enum CcuTaskArgIndex : uint32_t {
    CCU_ARG_INPUT = 0,
    CCU_ARG_OUTPUT,
    CCU_ARG_TOKEN,
    CCU_ARG_SCRATCH_SAVE,
    CCU_ARG_SCRATCH_PARTIAL,
    CCU_ARG_TILE_OFFSET,
    CCU_ARG_SEGMENT_SIZE,
    CCU_ARG_LAST_SEGMENT_SIZE,
    CCU_ARG_SCRATCH_STRIDE,
    CCU_ARG_OWNER_MAIN_SIZE,
    CCU_ARG_OWNER_TAIL_SIZE,
    CCU_ARG_INPUT_OUTPUT_EQUAL,
    CCU_TASK_ARG_NUM
};

static_assert(CCU_TASK_ARG_NUM == 12, "CCU task argument count mismatch");

struct CcuKernelArgAllReduce : CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t dieId;
    uint32_t channelRanks[CCU_MAX_RANK_SIZE];
    uint32_t reduceWithLocal;
    uint32_t ownsLocalSource;
    uint32_t useP16DualChain;
    uint32_t kernelCount;
    HcclDataType dataType;
    HcclReduceOp reduceOp;
};

CcuResult CcuKernelPartial(CcuKernelArg arg);
CcuResult CcuKernelFinalize(CcuKernelArg arg);
CcuResult CcuKernelPartialCombinedExchange(CcuKernelArg arg);
CcuResult CcuKernelFinalizeReuseExchange(CcuKernelArg arg);
CcuResult CcuKernelP4Fused(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_COMPETITION_ALLREDUCE_246_CCU_KERNEL_H_
