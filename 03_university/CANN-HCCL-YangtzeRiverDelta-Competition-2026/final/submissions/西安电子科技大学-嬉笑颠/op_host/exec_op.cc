/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.count == 0 || param.rankSize == 1, HCCL_INFO("Empty or single-rank broadcast, skip CCU launch"),
        HCCL_SUCCESS);
    CHK_PRT_RET(
        param.dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("Only FP32 broadcast is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("Broadcast byte size overflows"), HCCL_E_PARA);

    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);
    CHK_PRT_RET(resCtx.threads.empty() || resCtx.ccuKernels.empty()
                    || resCtx.ccuKernels.size() != resCtx.kernelGroupSizes.size(),
        HCCL_ERROR("Broadcast CCU resources are incomplete"), HCCL_E_INTERNAL);

    const uint64_t totalBytes = param.count * sizeof(float);
    uint64_t outputToken = 0;
    CcuResult ret = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), totalBytes, &outputToken);
    CHK_PRT_RET(ret != CCU_SUCCESS, HCCL_ERROR("Failed to get broadcast output token[%d]", ret), ConvertCcuToHccl(ret));

    uint64_t offset = 0;
    while (offset < totalBytes) {
        const uint64_t chunkBytes0 = std::min<uint64_t>(MAX_DATA_SIZE, totalBytes - offset);
        const uint64_t chunkBytes1 = std::min<uint64_t>(MAX_DATA_SIZE, totalBytes - offset - chunkBytes0);
        CHK_PRT_RET(chunkBytes0 == 0 || chunkBytes0 % sizeof(float) != 0 || chunkBytes1 % sizeof(float) != 0,
            HCCL_ERROR("Invalid broadcast chunk sizes[%llu, %llu]",
                static_cast<unsigned long long>(chunkBytes0), static_cast<unsigned long long>(chunkBytes1)),
            HCCL_E_PARA);
        for (uint32_t kernelIndex = 0; kernelIndex < resCtx.ccuKernels.size(); ++kernelIndex) {
            const uint64_t groupSize = resCtx.kernelGroupSizes[kernelIndex];
            CHK_PRT_RET(groupSize < 2 || groupSize > MAX_RANK_SIZE,
                HCCL_ERROR("Invalid broadcast group size[%llu]", static_cast<unsigned long long>(groupSize)),
                HCCL_E_INTERNAL);
            const uint64_t elementCount0 = chunkBytes0 / sizeof(float);
            const uint64_t normalSliceBytes0 = (elementCount0 / groupSize) * sizeof(float);
            const uint64_t lastSliceBytes0 = chunkBytes0 - normalSliceBytes0 * (groupSize - 1);
            const uint64_t elementCount1 = chunkBytes1 / sizeof(float);
            const uint64_t normalSliceBytes1 = (elementCount1 / groupSize) * sizeof(float);
            const uint64_t lastSliceBytes1 = chunkBytes1 - normalSliceBytes1 * (groupSize - 1);
            const uint64_t taskArgs[] = {
                reinterpret_cast<uint64_t>(param.outputPtr),
                outputToken,
                chunkBytes0,
                offset,
                normalSliceBytes0,
                lastSliceBytes0,
                chunkBytes1,
                normalSliceBytes1,
                lastSliceBytes1,
            };
            ret = HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[kernelIndex], taskArgs,
                static_cast<uint32_t>(sizeof(taskArgs) / sizeof(taskArgs[0])));
            CHK_PRT_RET(
                ret != CCU_SUCCESS, HCCL_ERROR("Broadcast CCU kernel launch failed[%d]", ret), ConvertCcuToHccl(ret));
        }
        offset += chunkBytes0 + chunkBytes1;
    }
    HCCL_INFO("Hierarchical CCU broadcast launched, bytes[%llu]", static_cast<unsigned long long>(totalBytes));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
