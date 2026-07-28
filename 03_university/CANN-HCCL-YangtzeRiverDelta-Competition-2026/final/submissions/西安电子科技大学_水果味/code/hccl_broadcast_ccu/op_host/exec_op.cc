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
#include <vector>

#include "ccu_launch.h"
#include "ccu_res.h"

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

    constexpr uint64_t SLICE_ALIGN = 128;

    bool RoleInList(uint32_t role, const std::vector<BroadcastKernelRole> &roles)
    {
        for (BroadcastKernelRole candidate : roles) {
            if (role == static_cast<uint32_t>(candidate)) {
                return true;
            }
        }
        return false;
    }

    HcclResult PreSyncSubThread(const std::vector<ThreadHandle> &threads)
    {
        if (threads.size() < 2) {
            return HCCL_SUCCESS;
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[1], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[1], 0, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    HcclResult PostSyncSubThread(const std::vector<ThreadHandle> &threads)
    {
        if (threads.size() < 2) {
            return HCCL_SUCCESS;
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], 0, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[1], threads[0], 0)));
        return HCCL_SUCCESS;
    }

    HcclResult LaunchPhase(
        const AlgResourceCtx &resCtx, const std::vector<BroadcastKernelRole> &roles, std::vector<uint64_t> &taskArgs)
    {
        int32_t mainKernel = -1;
        int32_t subKernel = -1;
        for (uint32_t i = 0; i < resCtx.ccuKernels.size(); ++i) {
            if (!RoleInList(resCtx.kernelRoles[i], roles)) {
                continue;
            }
            if (resCtx.kernelThreadIdx[i] == 0) {
                mainKernel = static_cast<int32_t>(i);
            } else if (resCtx.kernelThreadIdx[i] == 1) {
                subKernel = static_cast<int32_t>(i);
            } else {
                HCCL_ERROR("[Broadcast] invalid kernel thread index %u", resCtx.kernelThreadIdx[i]);
                return HCCL_E_INTERNAL;
            }
        }
        CHK_PRT_RET(
            mainKernel < 0 && subKernel < 0, HCCL_ERROR("[Broadcast] no kernel found for phase"), HCCL_E_INTERNAL);
        if (subKernel >= 0) {
            CHK_RET(PreSyncSubThread(resCtx.threads));
        }
        if (mainKernel >= 0) {
            const uint32_t idx = static_cast<uint32_t>(mainKernel);
            const CcuResult ret = HcommCcuKernelLaunch(
                resCtx.threads[resCtx.kernelThreadIdx[idx]], resCtx.ccuKernels[idx], taskArgs.data(), taskArgs.size());
            CHK_PRT_RET(
                ret != CCU_SUCCESS, HCCL_ERROR("[Broadcast] main kernel launch failed %d", ret), ConvertCcuToHccl(ret));
        }
        if (subKernel >= 0) {
            const uint32_t idx = static_cast<uint32_t>(subKernel);
            const CcuResult ret = HcommCcuKernelLaunch(
                resCtx.threads[resCtx.kernelThreadIdx[idx]], resCtx.ccuKernels[idx], taskArgs.data(), taskArgs.size());
            CHK_PRT_RET(
                ret != CCU_SUCCESS, HCCL_ERROR("[Broadcast] sub kernel launch failed %d", ret), ConvertCcuToHccl(ret));
            CHK_RET(PostSyncSubThread(resCtx.threads));
        }
        return HCCL_SUCCESS;
    }

    void CalcSlices(uint64_t tileSize, uint32_t rankSize, uint64_t &normalSliceSize, uint64_t &lastSliceSize)
    {
        normalSliceSize = tileSize / rankSize;
        normalSliceSize = normalSliceSize / SLICE_ALIGN * SLICE_ALIGN;
        if (normalSliceSize == 0) {
            normalSliceSize = tileSize / rankSize;
            normalSliceSize = normalSliceSize / sizeof(float) * sizeof(float);
        }
        lastSliceSize = tileSize - normalSliceSize * (rankSize - 1);
    }

    HcclResult LaunchTile(
        const OpParam &param, const AlgResourceCtx &resCtx, uint64_t tileAddr, uint64_t token, uint64_t tileSize)
    {
        uint64_t normalSliceSize = 0;
        uint64_t lastSliceSize = 0;
        CalcSlices(tileSize, param.rankSize, normalSliceSize, lastSliceSize);
        std::vector<uint64_t> taskArgs = {tileAddr, token, tileSize, normalSliceSize, lastSliceSize};

        switch (resCtx.algorithm) {
            case BroadcastAlgorithm::FLAT_MIXED:
                return LaunchPhase(
                    resCtx, {BroadcastKernelRole::FLAT_LAYER0, BroadcastKernelRole::FLAT_LAYER1}, taskArgs);
            case BroadcastAlgorithm::NHR_LAYER1:
                return LaunchPhase(resCtx, {BroadcastKernelRole::NHR_LAYER1}, taskArgs);
            case BroadcastAlgorithm::GLOBAL_MESH:
                CHK_RET(LaunchPhase(
                    resCtx, {BroadcastKernelRole::SCATTER_LAYER0, BroadcastKernelRole::SCATTER_LAYER1}, taskArgs));
                CHK_RET(LaunchPhase(
                    resCtx, {BroadcastKernelRole::ALLGATHER_LAYER0, BroadcastKernelRole::ALLGATHER_LAYER1}, taskArgs));
                return HCCL_SUCCESS;
            default:
                HCCL_ERROR("[Broadcast] invalid algorithm %u", static_cast<uint32_t>(resCtx.algorithm));
                return HCCL_E_INTERNAL;
        }
    }

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(ctx);
    std::vector<char> serialized(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(serialized);
    CHK_PRT_RET(resCtx.ccuKernels.size() != resCtx.kernelRoles.size()
                    || resCtx.ccuKernels.size() != resCtx.kernelThreadIdx.size() || resCtx.threads.empty(),
        HCCL_ERROR("[Broadcast] invalid resource context"), HCCL_E_INTERNAL);
    // Engine contexts are shared by a communicator, while the user is allowed to
    // submit later calls on another stream. Always launch thread-0 work on the
    // thread obtained from this invocation's stream.
    resCtx.threads[0] = param.cpuThread;
    for (uint32_t threadIdx : resCtx.kernelThreadIdx) {
        CHK_PRT_RET(threadIdx >= resCtx.threads.size(),
            HCCL_ERROR("[Broadcast] kernel thread index %u exceeds thread count %zu", threadIdx, resCtx.threads.size()),
            HCCL_E_INTERNAL);
    }

    const uint64_t dataSize = param.count * sizeof(float);
    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t token = 0;
    const CcuResult tokenRet = HcommCcuGetMemToken(baseAddr, dataSize, &token);
    CHK_PRT_RET(tokenRet != CCU_SUCCESS, HCCL_ERROR("[Broadcast] get memory token failed %d", tokenRet),
        ConvertCcuToHccl(tokenRet));

    uint64_t processed = 0;
    while (processed < dataSize) {
        const uint64_t tileSize = std::min<uint64_t>(MAX_DATA_SIZE, dataSize - processed);
        CHK_RET(LaunchTile(param, resCtx, baseAddr + processed, token, tileSize));
        processed += tileSize;
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
