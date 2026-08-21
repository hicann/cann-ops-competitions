/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hcomm/hcomm_primitives.h>

#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
    constexpr uint64_t MAX_CHUNK_BYTES = 256ULL * 1024ULL * 1024ULL;
    constexpr uint64_t PARALLEL_COMBINE_MIN_BYTES = 256ULL * 1024ULL;
    constexpr uint64_t READ_PIPELINE_MIN_BYTES = 512ULL * 1024ULL;
    constexpr uint64_t READ_PIPELINE_ALIGNMENT = 4ULL * 1024ULL;
    constexpr uint64_t BUFFERED_REDUCE_MIN_BYTES = 256ULL * 1024ULL;
    constexpr uint64_t BUFFERED_REDUCE_SLICE_BYTES = 4ULL * 1024ULL;
    constexpr uint64_t BUFFERED_REDUCE_LOOP_COUNT = 16ULL;

    struct BufferedReduceArgs {
        uint64_t enabled = 0;
        uint64_t addressOffset = 0;
        uint64_t loopIterations = 0;
        uint64_t parallelConfig = 0;
        uint64_t residualBytes = 0;
    };

    constexpr uint64_t BitMask(uint16_t bitCount)
    {
        return (uint64_t{1} << bitCount) - 1;
    }

    uint64_t PackParallelConfig(uint64_t repeatCount, uint64_t repeatLoopIndex, uint64_t totalLoopCount)
    {
        return ((repeatCount & BitMask(7)) << 55) | ((repeatLoopIndex & BitMask(7)) << 48)
            | ((totalLoopCount & BitMask(7)) << 41);
    }

    BufferedReduceArgs CalcBufferedReduceArgs(uint64_t bytes)
    {
        BufferedReduceArgs args;
        if (bytes < BUFFERED_REDUCE_MIN_BYTES) {
            return args;
        }

        constexpr uint64_t loopBytes = BUFFERED_REDUCE_SLICE_BYTES * BUFFERED_REDUCE_LOOP_COUNT;
        const uint64_t serialIterations = bytes / loopBytes;
        const uint64_t parallelSlices = (bytes % loopBytes) / BUFFERED_REDUCE_SLICE_BYTES;
        const uint64_t residualBytes = bytes % BUFFERED_REDUCE_SLICE_BYTES;

        args.enabled = 1;
        args.addressOffset = serialIterations * loopBytes;
        args.loopIterations = serialIterations;
        if (parallelSlices == 0 && residualBytes == 0) {
            return args;
        }
        if (parallelSlices != 0 && residualBytes == 0) {
            args.parallelConfig = PackParallelConfig(parallelSlices - 1, 0, 1);
            args.residualBytes = BUFFERED_REDUCE_SLICE_BYTES;
        } else if (parallelSlices == 0) {
            args.parallelConfig = PackParallelConfig(0, 0, 1);
            args.residualBytes = residualBytes;
        } else {
            args.parallelConfig = PackParallelConfig(parallelSlices - 1, 1, 2);
            args.residualBytes = residualBytes;
        }
        return args;
    }

    HcclResult ConvertCcuResult(CcuResult result)
    {
        switch (result) {
            case CCU_SUCCESS:
                return HCCL_SUCCESS;
            case CCU_E_PARA:
                return HCCL_E_PARA;
            case CCU_E_PTR:
                return HCCL_E_PTR;
            case CCU_E_NOT_SUPPORT:
                return HCCL_E_NOT_SUPPORT;
            case CCU_E_NOT_FOUND:
                return HCCL_E_NOT_FOUND;
            case CCU_E_UNAVAIL:
                return HCCL_E_UNAVAIL;
            default:
                return HCCL_E_INTERNAL;
        }
    }

    HcclResult PreSyncThreads(ThreadHandle mainThread, ThreadHandle subThread)
    {
        HcclResult result
            = static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(mainThread, subThread, 0));
        if (result != HCCL_SUCCESS) {
            return result;
        }
        return static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(subThread, 0));
    }

    HcclResult PostSyncThreads(ThreadHandle mainThread, ThreadHandle subThread)
    {
        HcclResult result
            = static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(mainThread, 0));
        if (result != HCCL_SUCCESS) {
            return result;
        }
        return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(subThread, mainThread, 0));
    }

    HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel, uint64_t inputAddress,
        uint64_t destinationAddress, uint64_t inputToken, uint64_t destinationToken, uint64_t scratchAddress,
        uint64_t scratchToken, uint64_t sourceBaseOffset, uint64_t chunkBytes, uint64_t pipelineFirstBytes,
        uint64_t pipelineSecondBytes, const BufferedReduceArgs &firstReduce,
        const BufferedReduceArgs &secondReduce)
    {
        const uint64_t taskArgs[] = {
            inputAddress, destinationAddress, inputToken, destinationToken, scratchAddress, scratchToken,
            sourceBaseOffset, chunkBytes, pipelineFirstBytes, pipelineSecondBytes,
            firstReduce.enabled, firstReduce.addressOffset, firstReduce.loopIterations,
            firstReduce.parallelConfig, firstReduce.residualBytes,
            secondReduce.enabled, secondReduce.addressOffset, secondReduce.loopIterations,
            secondReduce.parallelConfig, secondReduce.residualBytes};
        CcuResult launchResult = HcommCcuKernelLaunch(thread, kernel, taskArgs, sizeof(taskArgs) / sizeof(taskArgs[0]));
        if (launchResult != CCU_SUCCESS) {
            return ConvertCcuResult(launchResult);
        }
        return HCCL_SUCCESS;
    }

    HcclResult LaunchCombineKernel(ThreadHandle thread, CcuKernelHandle kernel, uint64_t destinationAddress,
        uint64_t sourceAddress, uint64_t destinationToken, uint64_t sourceToken, uint64_t chunkBytes)
    {
        const uint64_t taskArgs[] = {
            destinationAddress, sourceAddress, destinationToken, sourceToken, chunkBytes};
        CcuResult launchResult = HcommCcuKernelLaunch(thread, kernel, taskArgs, sizeof(taskArgs) / sizeof(taskArgs[0]));
        if (launchResult != CCU_SUCCESS) {
            return ConvertCcuResult(launchResult);
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE) {
        return HCCL_E_PARA;
    }

    constexpr uint64_t dataTypeBytes = sizeof(float);
    if (param.count > std::numeric_limits<uint64_t>::max() / dataTypeBytes) {
        return HCCL_E_PARA;
    }
    const uint64_t recvBytes = param.count * dataTypeBytes;
    if (recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t inputBytes = recvBytes * param.rankSize;

    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, recvBytes));
    }
    if (resCtx.threads.empty() || resCtx.ccuKernels.empty()) {
        return HCCL_E_INTERNAL;
    }
    const bool useTwoDies = resCtx.threads.size() == 2 && resCtx.ccuKernels.size() == 4;
    if (!useTwoDies && (resCtx.threads.size() != 1 || resCtx.ccuKernels.size() != 1)) {
        return HCCL_E_INTERNAL;
    }

    const uint64_t baseInputAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t baseOutputAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CcuResult tokenResult = HcommCcuGetMemToken(baseInputAddress, inputBytes, &inputToken);
    if (tokenResult != CCU_SUCCESS) {
        return ConvertCcuResult(tokenResult);
    }
    tokenResult = HcommCcuGetMemToken(baseOutputAddress, recvBytes, &outputToken);
    if (tokenResult != CCU_SUCCESS) {
        return ConvertCcuResult(tokenResult);
    }

    if (resCtx.localBuffer.addr == nullptr
        || resCtx.localBuffer.size < static_cast<uint64_t>(param.rankSize) * dataTypeBytes) {
        return HCCL_E_MEMORY;
    }

    const uint64_t scratchSlotCount = param.rankSize;
    uint64_t maxChunkBytes = std::min(MAX_CHUNK_BYTES, resCtx.localBuffer.size / scratchSlotCount);
    maxChunkBytes -= maxChunkBytes % dataTypeBytes;
    if (maxChunkBytes == 0) {
        return HCCL_E_MEMORY;
    }
    const uint64_t scratchBytes = maxChunkBytes * scratchSlotCount;
    const uint64_t scratchAddress = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    uint64_t scratchToken = 0;
    tokenResult = HcommCcuGetMemToken(scratchAddress, scratchBytes, &scratchToken);
    if (tokenResult != CCU_SUCCESS) {
        return ConvertCcuResult(tokenResult);
    }

    uint64_t processedBytes = 0;
    while (processedBytes < recvBytes) {
        const uint64_t chunkBytes = std::min(maxChunkBytes, recvBytes - processedBytes);
        const uint64_t outputAddress = baseOutputAddress + processedBytes;
        const uint64_t sourceBaseOffset = static_cast<uint64_t>(param.myRank) * recvBytes + processedBytes;
        uint64_t pipelineFirstBytes = 0;
        uint64_t pipelineSecondBytes = 0;
        if (chunkBytes >= READ_PIPELINE_MIN_BYTES) {
            pipelineFirstBytes = (chunkBytes / 2 / READ_PIPELINE_ALIGNMENT) * READ_PIPELINE_ALIGNMENT;
            if (pipelineFirstBytes == 0 || pipelineFirstBytes >= chunkBytes) {
                pipelineFirstBytes = 0;
            } else {
                pipelineSecondBytes = chunkBytes - pipelineFirstBytes;
            }
        }
        const uint64_t firstReduceBytes = pipelineFirstBytes == 0 ? chunkBytes : pipelineFirstBytes;
        const BufferedReduceArgs firstReduce = CalcBufferedReduceArgs(firstReduceBytes);
        const BufferedReduceArgs secondReduce = CalcBufferedReduceArgs(pipelineSecondBytes);

        if (!useTwoDies) {
            CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[0], baseInputAddress, outputAddress, inputToken,
                outputToken, scratchAddress, scratchToken, sourceBaseOffset, chunkBytes, pipelineFirstBytes,
                pipelineSecondBytes, firstReduce, secondReduce));
        } else {
            const uint64_t partialAddress = scratchAddress;
            CHK_RET(PreSyncThreads(resCtx.threads[0], resCtx.threads[1]));
            CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[0], baseInputAddress, outputAddress, inputToken,
                outputToken, scratchAddress, scratchToken, sourceBaseOffset, chunkBytes, pipelineFirstBytes,
                pipelineSecondBytes, firstReduce, secondReduce));
            CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[1], baseInputAddress, partialAddress, inputToken,
                scratchToken, scratchAddress, scratchToken, sourceBaseOffset, chunkBytes, pipelineFirstBytes,
                pipelineSecondBytes, firstReduce, secondReduce));
            CHK_RET(PostSyncThreads(resCtx.threads[0], resCtx.threads[1]));

            const uint64_t firstHalfBytes = pipelineFirstBytes != 0
                ? pipelineFirstBytes
                : (chunkBytes / dataTypeBytes / 2) * dataTypeBytes;
            const uint64_t secondHalfBytes = chunkBytes - firstHalfBytes;
            if (chunkBytes < PARALLEL_COMBINE_MIN_BYTES || firstHalfBytes == 0) {
                CHK_RET(LaunchCombineKernel(resCtx.threads[0], resCtx.ccuKernels[2], outputAddress, partialAddress,
                    outputToken, scratchToken, chunkBytes));
            } else {
                const uint64_t secondPartialAddress = pipelineFirstBytes == 0
                    ? partialAddress + firstHalfBytes
                    : scratchAddress + static_cast<uint64_t>(param.rankSize) * pipelineFirstBytes;
                CHK_RET(PreSyncThreads(resCtx.threads[0], resCtx.threads[1]));
                CHK_RET(LaunchCombineKernel(resCtx.threads[0], resCtx.ccuKernels[2], outputAddress, partialAddress,
                    outputToken, scratchToken, firstHalfBytes));
                CHK_RET(LaunchCombineKernel(resCtx.threads[1], resCtx.ccuKernels[3],
                    outputAddress + firstHalfBytes, secondPartialAddress,
                    outputToken, scratchToken, secondHalfBytes));
                CHK_RET(PostSyncThreads(resCtx.threads[0], resCtx.threads[1]));
            }
        }
        processedBytes += chunkBytes;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
