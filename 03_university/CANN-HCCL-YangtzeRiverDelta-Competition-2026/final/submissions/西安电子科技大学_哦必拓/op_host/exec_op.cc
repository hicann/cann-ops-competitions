/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_res.h>
#include <hcomm/ccu/ccu_launch.h>
#include <hcomm/hcomm_primitives.h>

#include <algorithm>
#include <array>
#include <vector>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
    HcclResult GetToken(uint64_t address, uint64_t bytes, uint64_t &token)
    {
        const CcuResult result = HcommCcuGetMemToken(address, bytes, &token);
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("HcommCcuGetMemToken failed: %d", result);
            return ConvertCcuToHccl(result);
        }
        return HCCL_SUCCESS;
    }

    HcclResult Launch(ThreadHandle thread, CcuKernelHandle kernel, const uint64_t *args, uint32_t argCount)
    {
        const CcuResult result = HcommCcuKernelLaunch(thread, kernel, args, argCount);
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("broadcast kernel launch failed: %d", result);
            return ConvertCcuToHccl(result);
        }
        return HCCL_SUCCESS;
    }

    HcclResult PreSyncThreads(ThreadHandle mainThread, ThreadHandle auxiliaryThread)
    {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(mainThread, auxiliaryThread, 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(auxiliaryThread, 0)));
        return HCCL_SUCCESS;
    }

    HcclResult PostSyncThreads(ThreadHandle mainThread, ThreadHandle auxiliaryThread)
    {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(mainThread, 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(auxiliaryThread, mainThread, 0)));
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *context = static_cast<char *>(param.resCtx);
    std::vector<char> serialized(context, context + param.ctxSize);
    AlgResourceCtx resources;
    resources.DeSerialize(serialized);

    const uint64_t dataBytes = param.count * sizeof(float);
    uint64_t token = 0;
    CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.outputPtr), dataBytes, token));

    if (resources.hierarchical) {
        const std::array<uint64_t, 2> args = {reinterpret_cast<uint64_t>(param.outputPtr), token};
        for (uint32_t phase = 0; phase < resources.kernelCount; ++phase) {
            CHK_RET(Launch(resources.thread, resources.kernels[phase], args.data(), args.size()));
        }
        return HCCL_SUCCESS;
    }

    // CCU transfer descriptors accept at most 256 MiB.  Launches share the
    // stream-backed thread, so chunks complete in buffer order.
    uint64_t offset = 0;
    while (offset < dataBytes) {
        const uint64_t length = std::min<uint64_t>(MAX_DATA_SIZE, dataBytes - offset);
        const std::array<uint64_t, 4> args = {reinterpret_cast<uint64_t>(param.outputPtr), token, offset, length};
        if (resources.auxiliaryThread != 0 && resources.kernelCount == 2) {
            CHK_RET(PreSyncThreads(resources.thread, resources.auxiliaryThread));
            CHK_RET(Launch(resources.thread, resources.kernels[0], args.data(), args.size()));
            CHK_RET(Launch(resources.auxiliaryThread, resources.kernels[1], args.data(), args.size()));
            CHK_RET(PostSyncThreads(resources.thread, resources.auxiliaryThread));
        } else {
            for (uint32_t kernel = 0; kernel < resources.kernelCount; ++kernel) {
                CHK_RET(Launch(resources.thread, resources.kernels[kernel], args.data(), args.size()));
            }
        }
        offset += length;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl