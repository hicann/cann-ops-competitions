/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
HcclResult SyncThreads(ThreadHandle producer, ThreadHandle consumer, uint32_t notifyIndex)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(producer, consumer, notifyIndex)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(consumer, notifyIndex, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

ThreadHandle KernelThread(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t kernelIndex)
{
    return kernelIndex == 0 ? param.cpuThread : resCtx.extraThreads[kernelIndex - 1];
}

bool ContainsPeer(const BroadcastKernelMeta &meta, uint32_t peerRank)
{
    return std::find(meta.peerRanks, meta.peerRanks + meta.peerCount, peerRank)
        != meta.peerRanks + meta.peerCount;
}

bool KernelIsActive(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t kernelIndex)
{
    if (resCtx.mode == static_cast<uint32_t>(BroadcastMode::SCATTER_ALLGATHER) || param.myRank == param.root) {
        return true;
    }
    return ContainsPeer(resCtx.kernelMeta[kernelIndex], param.root);
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0,
        HCCL_ERROR("[BroadcastV3] CCU context is unavailable"), HCCL_E_INTERNAL);

    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);
    CHK_PRT_RET(resCtx.ccuKernels.empty() || resCtx.ccuKernels.size() != resCtx.kernelMeta.size()
            || resCtx.ccuKernels.size() > 2 || resCtx.windows.empty()
            || resCtx.windows.size() > BROADCAST_MAX_WINDOWS
            || resCtx.extraThreads.size() + 1 != resCtx.ccuKernels.size(),
        HCCL_ERROR("[BroadcastV3] invalid resource shape: kernels[%llu], windows[%llu]",
            static_cast<unsigned long long>(resCtx.ccuKernels.size()),
            static_cast<unsigned long long>(resCtx.windows.size())), HCCL_E_INTERNAL);

    std::vector<uint64_t> taskArgs(2 + 2 * resCtx.windows.size(), 0);
    taskArgs[0] = param.root;
    taskArgs[1] = static_cast<uint32_t>(BroadcastPhase::ONE_SHOT);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    for (uint32_t window = 0; window < resCtx.windows.size(); ++window) {
        const uint64_t address = outputBase + resCtx.windows[window].offset;
        uint64_t token = 0;
        CHK_RET_CCU(HcommCcuGetMemToken(address, resCtx.windows[window].bytes, &token));
        taskArgs[2 + 2 * window] = address;
        taskArgs[3 + 2 * window] = token;
    }

    std::vector<uint32_t> activeKernels;
    for (uint32_t kernelIndex = 0; kernelIndex < resCtx.ccuKernels.size(); ++kernelIndex) {
        if (KernelIsActive(param, resCtx, kernelIndex)) {
            activeKernels.push_back(kernelIndex);
        }
    }
    CHK_PRT_RET(activeKernels.empty(),
        HCCL_ERROR("[BroadcastV3] no kernel owns root[%u]", param.root), HCCL_E_INTERNAL);

    auto launchKernels = [&](BroadcastPhase phase) -> HcclResult {
        taskArgs[1] = static_cast<uint32_t>(phase);
        for (uint32_t kernelIndex : activeKernels) {
            CHK_RET_CCU(HcommCcuKernelLaunch(KernelThread(param, resCtx, kernelIndex),
                resCtx.ccuKernels[kernelIndex], taskArgs.data(), taskArgs.size()));
        }
        return HCCL_SUCCESS;
    };

    if (resCtx.mode == static_cast<uint32_t>(BroadcastMode::ONE_SHOT)) {
        for (uint32_t kernelIndex : activeKernels) {
            if (kernelIndex != 0) {
                CHK_RET(SyncThreads(param.cpuThread, KernelThread(param, resCtx, kernelIndex), 0));
            }
        }
        CHK_RET(launchKernels(BroadcastPhase::ONE_SHOT));
        for (uint32_t kernelIndex : activeKernels) {
            if (kernelIndex != 0) {
                CHK_RET(SyncThreads(KernelThread(param, resCtx, kernelIndex), param.cpuThread, 1));
            }
        }
        return HCCL_SUCCESS;
    }

    if (resCtx.ccuKernels.size() == 1) {
        CHK_RET(launchKernels(BroadcastPhase::SCATTER));
        CHK_RET(launchKernels(BroadcastPhase::ALLGATHER));
        return HCCL_SUCCESS;
    }

    const ThreadHandle secondaryThread = resCtx.extraThreads[0];
    CHK_RET(SyncThreads(param.cpuThread, secondaryThread, 0));
    CHK_RET(launchKernels(BroadcastPhase::SCATTER));
    CHK_RET(SyncThreads(secondaryThread, param.cpuThread, 1));

    // The main thread is now ordered after both DIE Scatter kernels.  Fan this
    // completion back to the secondary DIE before either AllGather kernel can
    // read its owner slice.  Keeping this dependency in the host task graph
    // avoids the cross-DIE named Event deadlock seen by CheckerV3 at 16 ranks.
    CHK_RET(SyncThreads(param.cpuThread, secondaryThread, 0));
    CHK_RET(launchKernels(BroadcastPhase::ALLGATHER));
    CHK_RET(SyncThreads(secondaryThread, param.cpuThread, 1));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
