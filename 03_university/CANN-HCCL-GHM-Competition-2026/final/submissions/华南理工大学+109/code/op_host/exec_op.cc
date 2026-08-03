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
#include <numeric>

#include <ccu/ccu_res.h>
#include <hcomm/ccu/ccu_launch.h>

#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    constexpr uint64_t kStripedThresholdBytes = 1ULL << 20;
    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    const auto typeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type %d", param.dataType), HCCL_E_NOT_SUPPORT);
    const uint64_t unitSize = typeIt->second;
    CHK_PRT_RET(param.count > UINT64_MAX / unitSize / param.rankSize, HCCL_ERROR("ReduceScatter byte count overflows"),
        HCCL_E_PARA);

    const uint64_t recvBytes = param.count * unitSize;
    const uint64_t inputBytes = recvBytes * param.rankSize;
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("CCU thread resource is missing"), HCCL_E_INTERNAL);
    if (param.rankSize == 1) {
        CHK_RET(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, recvBytes));
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < param.rankSize * unitSize,
        HCCL_ERROR("HCCL buffer is too small"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernels.size() != resCtx.kernelThreadIndices.size(),
        HCCL_ERROR("CCU kernel/thread resource mapping is invalid"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernels.size() != resCtx.kernelChannelCounts.size(),
        HCCL_ERROR("CCU kernel/channel resource mapping is invalid"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.stripedKernels.size() != resCtx.ccuKernels.size(),
        HCCL_ERROR("CCU striped kernel resource mapping is invalid"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernels.size() == 2 && resCtx.mergeKernel == 0,
        HCCL_ERROR("CCU partial merge kernel is missing"), HCCL_E_INTERNAL);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t cclToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, &outputToken));
    CHK_RET_CCU(
        HcommCcuGetMemToken(reinterpret_cast<uint64_t>(resCtx.localBuffer.addr), resCtx.localBuffer.size, &cclToken));

    const bool striped = inputBytes > kStripedThresholdBytes;
    const bool parallelLayerKernels = resCtx.ccuKernels.size() == 2;
    const uint32_t scratchSlotCount = striped ? 1U
                                              : std::accumulate(resCtx.kernelChannelCounts.begin(),
                                                    resCtx.kernelChannelCounts.end(), static_cast<uint32_t>(0));
    CHK_PRT_RET(scratchSlotCount == 0, HCCL_ERROR("CCU kernel has no channels"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernels.size() > resCtx.threads.size(), HCCL_ERROR("Insufficient CCU thread resources"),
        HCCL_E_INTERNAL);

    uint64_t chunkCapacityBytes = std::min<uint64_t>(MAX_DATA_SIZE, resCtx.localBuffer.size / scratchSlotCount);
    chunkCapacityBytes -= chunkCapacityBytes % unitSize;
    CHK_PRT_RET(chunkCapacityBytes == 0, HCCL_ERROR("HCCL buffer cannot hold one element per rank"), HCCL_E_INTERNAL);
    const uint64_t chunkCapacityElements = chunkCapacityBytes / unitSize;
    const uint64_t chunkCount
        = param.count == 0 ? 1
                           : param.count / chunkCapacityElements + (param.count % chunkCapacityElements != 0 ? 1 : 0);
    const uint64_t balancedChunkElements = param.count / chunkCount + (param.count % chunkCount != 0 ? 1 : 0);
    const uint64_t maxChunkBytes = balancedChunkElements * unitSize;

    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t cclBase = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    for (uint64_t offset = 0; offset < recvBytes; offset += maxChunkBytes) {
        const uint64_t chunkBytes = std::min(maxChunkBytes, recvBytes - offset);
        const uint64_t inputOffset = static_cast<uint64_t>(param.myRank) * recvBytes + offset;
        if (parallelLayerKernels) {
            const ThreadHandle mainThread = resCtx.threads[resCtx.kernelThreadIndices[0]];
            const ThreadHandle slaveThread = resCtx.threads[resCtx.kernelThreadIndices[1]];
            int32_t syncRet = HcommThreadNotifyRecordOnThread(mainThread, slaveThread, 0);
            CHK_PRT_RET(syncRet != HCCL_SUCCESS, HCCL_ERROR("CCU pre-sync record failed: %d", syncRet),
                static_cast<HcclResult>(syncRet));
            syncRet = HcommThreadNotifyWaitOnThread(slaveThread, 0, CUSTOM_TIMEOUT);
            CHK_PRT_RET(syncRet != HCCL_SUCCESS, HCCL_ERROR("CCU pre-sync wait failed: %d", syncRet),
                static_cast<HcclResult>(syncRet));
        }
        uint64_t scratchSlotOffset = 0;
        for (uint32_t i = 0; i < resCtx.ccuKernels.size(); ++i) {
            const uint32_t threadIndex = parallelLayerKernels ? resCtx.kernelThreadIndices[i] : 0;
            CHK_PRT_RET(threadIndex >= resCtx.threads.size(), HCCL_ERROR("Invalid CCU thread index %u", threadIndex),
                HCCL_E_INTERNAL);
            CcuResult ret;
            if (striped) {
                const uint32_t stripeCount = resCtx.kernelChannelCounts[i];
                const uint64_t chunkElements = chunkBytes / unitSize;
                const uint64_t stripeElements = chunkElements / stripeCount;
                const uint64_t stripeBytes = stripeElements * unitSize;
                const uint64_t lastStripeBytes = chunkBytes - stripeBytes * (stripeCount - 1);
                const uint64_t taskArgs[] = {inputBase, inputToken, outputBase + offset, outputToken, cclBase, cclToken,
                    inputOffset, chunkBytes, stripeBytes, lastStripeBytes};
                ret = HcommCcuKernelLaunch(resCtx.threads[threadIndex], resCtx.stripedKernels[i], taskArgs,
                    sizeof(taskArgs) / sizeof(taskArgs[0]));
            } else {
                const uint64_t kernelScratchBase = cclBase + scratchSlotOffset * chunkBytes;
                const uint64_t taskArgs[] = {inputBase, inputToken, outputBase + offset, outputToken, kernelScratchBase,
                    cclToken, inputOffset, chunkBytes, chunkBytes};
                ret = HcommCcuKernelLaunch(resCtx.threads[threadIndex], resCtx.ccuKernels[i], taskArgs,
                    sizeof(taskArgs) / sizeof(taskArgs[0]));
                scratchSlotOffset += resCtx.kernelChannelCounts[i];
            }
            CHK_PRT_RET(ret != CCU_SUCCESS, HCCL_ERROR("CCU kernel launch failed: %d", ret), ConvertCcuToHccl(ret));
        }

        if (parallelLayerKernels) {
            const ThreadHandle mainThread = resCtx.threads[resCtx.kernelThreadIndices[0]];
            const ThreadHandle slaveThread = resCtx.threads[resCtx.kernelThreadIndices[1]];
            int32_t syncRet = HcommThreadNotifyWaitOnThread(mainThread, 0, CUSTOM_TIMEOUT);
            CHK_PRT_RET(syncRet != HCCL_SUCCESS, HCCL_ERROR("CCU post-sync wait failed: %d", syncRet),
                static_cast<HcclResult>(syncRet));
            syncRet = HcommThreadNotifyRecordOnThread(slaveThread, mainThread, 0);
            CHK_PRT_RET(syncRet != HCCL_SUCCESS, HCCL_ERROR("CCU post-sync record failed: %d", syncRet),
                static_cast<HcclResult>(syncRet));
        }
        if (parallelLayerKernels) {
            const ThreadHandle mainThread = resCtx.threads[0];
            const uint64_t partialBase
                = striped ? cclBase : cclBase + static_cast<uint64_t>(resCtx.kernelChannelCounts[0]) * chunkBytes;
            const uint64_t mergeArgs[] = {outputBase + offset, outputToken, partialBase, cclToken, chunkBytes};
            const CcuResult mergeRet = HcommCcuKernelLaunch(
                mainThread, resCtx.mergeKernel, mergeArgs, sizeof(mergeArgs) / sizeof(mergeArgs[0]));
            CHK_PRT_RET(mergeRet != CCU_SUCCESS, HCCL_ERROR("CCU partial merge failed: %d", mergeRet),
                ConvertCcuToHccl(mergeRet));
        }
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
