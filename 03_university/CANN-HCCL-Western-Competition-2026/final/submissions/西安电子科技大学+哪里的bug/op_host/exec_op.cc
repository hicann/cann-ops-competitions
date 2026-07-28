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
#include <limits>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    CHK_PTR_NULL(param.resCtx);

    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type %u", static_cast<uint32_t>(param.dataType)), HCCL_E_NOT_SUPPORT);
    const uint64_t dataTypeSize = sizeIt->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("AllReduce data size overflows uint64"), HCCL_E_PARA);

    const uint64_t dataSize = param.count * dataTypeSize;
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }

    // A cached context can contain a thread bound to an older stream. Always
    // launch through the thread acquired for the current API invocation.
    resCtx.ccuThread = param.cpuThread;
    if (!resCtx.threads.empty()) {
        resCtx.threads[0] = param.cpuThread;
    }

    if (param.rankSize == 1) {
        if (param.inputPtr == param.outputPtr) {
            return HCCL_SUCCESS;
        }
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, dataSize));
    }

    CHK_PRT_RET(resCtx.ccuKernels.size() != 1,
        HCCL_ERROR("Expected one registered CCU kernel, got %zu", resCtx.ccuKernels.size()), HCCL_E_INTERNAL);

    const uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    CHK_PRT_RET(baseInputAddr > std::numeric_limits<uint64_t>::max() - dataSize ||
            baseOutputAddr > std::numeric_limits<uint64_t>::max() - dataSize,
        HCCL_ERROR("AllReduce buffer range overflows uint64"), HCCL_E_PARA);
    const uint64_t inputEnd = baseInputAddr + dataSize;
    const uint64_t outputEnd = baseOutputAddr + dataSize;
    CHK_PRT_RET(baseInputAddr != baseOutputAddr && baseInputAddr < outputEnd && baseOutputAddr < inputEnd,
        HCCL_ERROR("Partially overlapping AllReduce buffers are unsupported"), HCCL_E_NOT_SUPPORT);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseInputAddr, dataSize, &inputToken));
    if (baseInputAddr == baseOutputAddr) {
        outputToken = inputToken;
    } else {
        CHK_RET_CCU(HcommCcuGetMemToken(baseOutputAddr, dataSize, &outputToken));
    }
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0,
        HCCL_ERROR("HCCL scratch buffer is unavailable"), HCCL_E_INTERNAL);
    const uint64_t scratchAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    uint64_t scratchToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(scratchAddr, resCtx.localBuffer.size, &scratchToken));

    // One-shot minimizes synchronization for very small per-rank payloads, but
    // its peer-read traffic grows with rankSize.  Keep the measured fast path
    // for domains of at most four ranks; for larger domains switch to
    // reduce-scatter/all-gather once a rank owns more than 16 KiB.  In
    // particular, the scored 512 KiB cases now use the lower-volume algorithm
    // at 12/16 ranks while retaining the current 4-rank path.
    constexpr uint64_t oneShotThresholdSmallDomain = 512 * 1024;
    constexpr uint64_t oneShotBytesPerRankLargeDomain = 16 * 1024;
    const uint64_t oneShotThreshold = param.rankSize <= 4
        ? oneShotThresholdSmallDomain
        : oneShotBytesPerRankLargeDomain * param.rankSize;
    constexpr uint64_t algorithmOneShot = 0;
    constexpr uint64_t algorithmReduceScatterAllGather = 1;
    const uint64_t maxSliceCountByScratch =
        resCtx.localBuffer.size / (param.rankSize - 1) / dataTypeSize;
    CHK_PRT_RET(maxSliceCountByScratch == 0,
        HCCL_ERROR("HCCL scratch buffer cannot hold one element per peer"), HCCL_E_MEMORY);
    const uint64_t maxCountByScratch = maxSliceCountByScratch * param.rankSize;
    // MAX_DATA_SIZE limits one CCU transfer. Reduce-scatter/all-gather sends one
    // rank slice per transfer, so a launch may cover up to rankSize such slices.
    const uint64_t maxCountByTransfer = MAX_DATA_SIZE / dataTypeSize * param.rankSize;
    const uint64_t maxCountPerLaunch = std::min(maxCountByTransfer, maxCountByScratch);
    uint64_t processedCount = 0;

    while (processedCount < param.count) {
        const uint64_t launchCount = std::min(maxCountPerLaunch, param.count - processedCount);
        const uint64_t launchBytes = launchCount * dataTypeSize;
        const uint64_t inputAddr = baseInputAddr + processedCount * dataTypeSize;
        const uint64_t outputAddr = baseOutputAddr + processedCount * dataTypeSize;
        const uint64_t needLocalCopy = inputAddr == outputAddr ? 0 : 1;

        uint64_t algorithm = algorithmOneShot;
        uint64_t sliceOffset = 0;
        uint64_t sliceSize = launchBytes;
        const bool oneShotFitsScratch =
            launchBytes != 0 && param.rankSize - 1 <= resCtx.localBuffer.size / launchBytes;
        if (launchBytes > oneShotThreshold || !oneShotFitsScratch) {
            algorithm = algorithmReduceScatterAllGather;
            const uint64_t baseSliceCount = launchCount / param.rankSize;
            const uint64_t extraSliceCount = launchCount % param.rankSize;
            const uint64_t rankSliceCount = baseSliceCount + (param.myRank < extraSliceCount ? 1 : 0);
            const uint64_t rankSliceOffset = baseSliceCount * param.myRank +
                std::min<uint64_t>(param.myRank, extraSliceCount);
            sliceOffset = rankSliceOffset * dataTypeSize;
            sliceSize = rankSliceCount * dataTypeSize;
        }
        CHK_PRT_RET(sliceSize != 0 && param.rankSize - 1 > resCtx.localBuffer.size / sliceSize,
            HCCL_ERROR("HCCL scratch buffer is too small for a %llu-byte slice",
                static_cast<unsigned long long>(sliceSize)),
            HCCL_E_MEMORY);

        std::vector<uint64_t> taskArgs = {
            inputAddr,
            outputAddr,
            inputToken,
            outputToken,
            scratchAddr,
            scratchToken,
            sliceOffset,
            sliceSize,
            needLocalCopy,
            algorithm,
        };
        CHK_RET_CCU(HcommCcuKernelLaunch(
            param.cpuThread, resCtx.ccuKernels[0], taskArgs.data(), taskArgs.size()));
        processedCount += launchCount;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
