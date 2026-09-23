/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hcomm/hcomm_primitives.h>

#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint32_t WORKER_START_NOTIFY = 0;
constexpr uint32_t RELAY_READY_NOTIFY = 1;
constexpr uint32_t WORKER_FINISH_NOTIFY = 0;

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS: return HCCL_SUCCESS;
        case CCU_E_PARA: return HCCL_E_PARA;
        case CCU_E_PTR: return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT: return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND: return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL: return HCCL_E_UNAVAIL;
        default: return HCCL_E_INTERNAL;
    }
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0,
        HCCL_ERROR("[Scatter] missing CCU resource context"), HCCL_E_INTERNAL);
    const char *raw = static_cast<const char *>(param.resCtx);
    std::vector<char> serialized(raw, raw + param.ctxSize);
    AlgResourceCtx ctx;
    ctx.DeSerialize(serialized);
    CHK_PRT_RET(ctx.threads.empty() || ctx.threads.size() != ctx.ccuKernels.size() ||
        ctx.kernelRoles.size() != ctx.ccuKernels.size(),
        HCCL_ERROR("[Scatter] CCU kernel/thread count mismatch"), HCCL_E_INTERNAL);
    ctx.threads[0] = param.cpuThread;

    const uint64_t sliceBytes = param.count * sizeof(float);
    const uint64_t totalBytes = sliceBytes * param.rankSize;
    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchAddr = 0;
    uint64_t scratchToken = 0;
    const uint64_t maxChunkBytes = std::min<uint64_t>(sliceBytes, MAX_DATA_SIZE);
    const uint64_t maxRelayBytes = ((maxChunkBytes * ctx.relayNumerator /
        ctx.relayDenominator) / 4) * 4;
    if (param.myRank == param.root) {
        CcuResult ret = HcommCcuGetMemToken(inputAddr, totalBytes, &inputToken);
        if (ret != CCU_SUCCESS) {
            HCCL_ERROR("[Scatter] input memory token failed: %d", ret);
            return ConvertCcuResult(ret);
        }
    }
    if (param.myRank != param.root || ctx.overlapOwnCopy != 0) {
        CcuResult ret = HcommCcuGetMemToken(outputAddr, sliceBytes, &outputToken);
        if (ret != CCU_SUCCESS) {
            HCCL_ERROR("[Scatter] output memory token failed: %d", ret);
            return ConvertCcuResult(ret);
        }
    }
    if (ctx.relayTarget < param.rankSize) {
        CHK_PRT_RET(ctx.localBuffer.addr == nullptr || ctx.localBuffer.size < maxRelayBytes,
            HCCL_ERROR("[Scatter] insufficient relay scratch buffer"), HCCL_E_UNAVAIL);
        scratchAddr = reinterpret_cast<uint64_t>(ctx.localBuffer.addr);
        CcuResult ret = HcommCcuGetMemToken(scratchAddr, maxRelayBytes, &scratchToken);
        if (ret != CCU_SUCCESS) {
            return ConvertCcuResult(ret);
        }
    }

    // The root supplies its source buffer; receivers publish their output
    // addresses and tokens to the root before each push.
    if (ctx.threads.size() == 2) {
        // The worker must start with a local wait in the CCU checker graph.
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            ctx.threads[0], ctx.threads[1], WORKER_START_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            ctx.threads[1], WORKER_START_NOTIFY, CUSTOM_TIMEOUT)));
    }
    for (uint64_t offset = 0; offset < sliceBytes;) {
        const uint64_t chunkBytes = std::min<uint64_t>(MAX_DATA_SIZE, sliceBytes - offset);
        const uint64_t relayBytes = ((chunkBytes * ctx.relayNumerator /
            ctx.relayDenominator) / 4) * 4;
        uint64_t args[] = {
            inputAddr + offset,
            outputAddr + offset,
            inputToken,
            outputToken,
            chunkBytes,
            sliceBytes,
            scratchAddr,
            scratchToken,
            relayBytes,
            chunkBytes - relayBytes,
        };
        // Launch root-facing kernels first.  On a two-die helper, the
        // target-facing kernel is released by the existing cross-thread
        // handshake after the prefix-ready point.  This avoids dereferencing
        // a missing root channel while still allowing prefix forwarding.
        for (size_t i = 0; i < ctx.ccuKernels.size(); ++i) {
            if (ctx.kernelRoles[i] != 0) {
                continue;
            }
            CcuResult ret = HcommCcuKernelLaunch(ctx.threads[i], ctx.ccuKernels[i], args,
                ctx.relayNumerator == 0 ? 6 : 10);
            if (ret != CCU_SUCCESS) {
                HCCL_ERROR("[Scatter] CCU launch %zu failed: %d", i, ret);
                return ConvertCcuResult(ret);
            }
        }
        for (size_t i = 0; i < ctx.ccuKernels.size(); ++i) {
            if (ctx.kernelRoles[i] == 0) {
                continue;
            }
            if (ctx.threads.size() == 2) {
                const size_t source = i == 0 ? 1 : 0;
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    ctx.threads[source], ctx.threads[i], RELAY_READY_NOTIFY)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    ctx.threads[i], RELAY_READY_NOTIFY, CUSTOM_TIMEOUT)));
            }
            CcuResult ret = HcommCcuKernelLaunch(ctx.threads[i], ctx.ccuKernels[i], args,
                ctx.relayNumerator == 0 ? 6 : 10);
            if (ret != CCU_SUCCESS) {
                return ConvertCcuResult(ret);
            }
        }
        offset += chunkBytes;
    }
    if (ctx.threads.size() == 2) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            ctx.threads[1], ctx.threads[0], WORKER_FINISH_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            ctx.threads[0], WORKER_FINISH_NOTIFY, CUSTOM_TIMEOUT)));
    }
    if (param.myRank == param.root && ctx.overlapOwnCopy == 0) {
        // Keep the full root input untouched until every remote write completes.
        for (uint64_t offset = 0; offset < sliceBytes;) {
            const uint64_t chunkBytes = std::min<uint64_t>(MAX_DATA_SIZE, sliceBytes - offset);
            const auto *src = static_cast<const uint8_t *>(param.inputPtr) +
                static_cast<uint64_t>(param.root) * sliceBytes + offset;
            auto *dst = static_cast<uint8_t *>(param.outputPtr) + offset;
            if (src != dst) {
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                    ctx.threads[0], dst, src, chunkBytes)));
            }
            offset += chunkBytes;
        }
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
