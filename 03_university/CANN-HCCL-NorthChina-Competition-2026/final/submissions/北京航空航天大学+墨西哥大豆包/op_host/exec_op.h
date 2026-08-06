/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMPETITION_2026_ALLGATHER_EXEC_OP_H
#define HCCL_COMPETITION_2026_ALLGATHER_EXEC_OP_H

#include <hccl/hcomm_primitives.h>

#include "common.h"
#include "custom.h"

namespace ops_hccl {

HcclResult SelectAllGatherRoute(uint64_t dataSize, uint32_t rankSize, AllGatherRoute &route);

HcclResult ExecSmall512Fast(
    void *inputPtr, void *outputPtr, void *resCtx, uint64_t ctxSize);

HcclResult ExecOp(const OpParam &param);

} // namespace ops_hccl

#endif // HCCL_COMPETITION_2026_ALLGATHER_EXEC_OP_H
