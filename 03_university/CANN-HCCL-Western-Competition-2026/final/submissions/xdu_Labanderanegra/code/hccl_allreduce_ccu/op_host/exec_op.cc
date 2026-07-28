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

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {

namespace {

constexpr uint32_t DIE_NUM = 2;
constexpr uint32_t LEVEL_NUM = 2;
constexpr uint32_t LOCAL_MESH_LEVEL = 0;
constexpr uint32_t COLUMN_TREE_LEVEL = 1;
constexpr uint32_t REDUCE_PHASE = 0;
constexpr uint32_t BROADCAST_PHASE = 1;
constexpr uint32_t LOCAL_EXCHANGE_INPUT = 0;
constexpr uint32_t LOCAL_REDUCE = 1;
constexpr uint32_t LOCAL_EXCHANGE_OUTPUT = 2;
constexpr uint32_t LOCAL_GATHER = 3;
constexpr uint32_t LOCAL_BARRIER = 4;
constexpr uint32_t SYNC_NOTIFY_INDEX = 0;
constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
constexpr uint64_t CHUNK_SIZE = MAX_DATA_SIZE;

struct ChunkSlice {
    uint64_t offset = 0;
    uint64_t size = 0;
};

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
        case CCU_E_INTERNAL:
        default:
            return HCCL_E_INTERNAL;
    }
}

HcclResult MainToSlave(const std::vector<ThreadHandle> &threads)
{
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(threads[0], threads[1], SYNC_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(threads[1], SYNC_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult SlaveToMain(const std::vector<ThreadHandle> &threads)
{
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(threads[0], SYNC_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(threads[1], threads[0], SYNC_NOTIFY_INDEX)));
    return HCCL_SUCCESS;
}

ChunkSlice CalcSlice(uint64_t bytes, uint32_t groupSize, uint32_t index)
{
    const uint64_t elementCount = bytes / DATA_TYPE_SIZE;
    const uint64_t baseCount = elementCount / groupSize;
    const uint64_t extraCount = elementCount % groupSize;
    const uint64_t sliceCount = baseCount + (index < extraCount ? 1 : 0);
    const uint64_t sliceOffset = index * baseCount + std::min<uint64_t>(index, extraCount);
    return ChunkSlice{sliceOffset * DATA_TYPE_SIZE, sliceCount * DATA_TYPE_SIZE};
}

HcclResult LaunchKernel(const AlgResourceCtx &resCtx, uint32_t level, uint32_t dieId,
    const std::vector<uint64_t> &taskArgs)
{
    const uint32_t kernelIdx = level * DIE_NUM + dieId;
    if (resCtx.kernelValid[kernelIdx] == 0) {
        return HCCL_SUCCESS;
    }

    CcuResult ccuRet = HcommCcuKernelLaunch(resCtx.threads[dieId], resCtx.ccuKernels[kernelIdx],
        taskArgs.data(), taskArgs.size());
    return ConvertCcuResult(ccuRet);
}

bool HasKernel(const AlgResourceCtx &resCtx, uint32_t level)
{
    return resCtx.kernelValid[level * DIE_NUM] != 0 ||
        resCtx.kernelValid[level * DIE_NUM + 1] != 0;
}

HcclResult RunSerialStage(const AlgResourceCtx &resCtx, uint32_t level,
    const std::vector<uint64_t> &taskArgs)
{
    if (!HasKernel(resCtx, level)) {
        return HCCL_SUCCESS;
    }

    CHK_RET(LaunchKernel(resCtx, level, 0, taskArgs));
    CHK_RET(MainToSlave(resCtx.threads));
    CHK_RET(LaunchKernel(resCtx, level, 1, taskArgs));
    CHK_RET(SlaveToMain(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult RunParallelStage(const AlgResourceCtx &resCtx, uint32_t level,
    const std::vector<uint64_t> &taskArgs)
{
    if (!HasKernel(resCtx, level)) {
        return HCCL_SUCCESS;
    }

    CHK_RET(LaunchKernel(resCtx, level, 0, taskArgs));
    CHK_RET(LaunchKernel(resCtx, level, 1, taskArgs));
    CHK_RET(SlaveToMain(resCtx.threads));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (resCtx.threads.empty() || param.count > std::numeric_limits<uint64_t>::max() / DATA_TYPE_SIZE) {
        return HCCL_E_PARA;
    }

    const uint64_t totalBytes = param.count * DATA_TYPE_SIZE;
    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);

    if (param.rankSize == 1) {
        if (resCtx.threads.size() != 1) {
            return HCCL_E_INTERNAL;
        }
        if (param.inputPtr == param.outputPtr) {
            return HCCL_SUCCESS;
        }
        for (uint64_t offset = 0; offset < totalBytes; offset += CHUNK_SIZE) {
            const uint64_t bytes = std::min(CHUNK_SIZE, totalBytes - offset);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0],
                reinterpret_cast<void *>(outputBase + offset), reinterpret_cast<void *>(inputBase + offset), bytes)));
        }
        return HCCL_SUCCESS;
    }

    const uint32_t groupSize = resCtx.localGroupSize;
    const uint32_t expectedKernelNum = LEVEL_NUM * DIE_NUM;
    if (groupSize == 0 || groupSize > param.rankSize || resCtx.localIndex >= groupSize ||
        resCtx.threads.size() != DIE_NUM || resCtx.ccuKernels.size() != expectedKernelNum ||
        resCtx.kernelValid.size() != expectedKernelNum) {
        HCCL_ERROR("Invalid GCD RSAG resource layout, groupSize[%u] localIndex[%u] threads[%zu] "
                   "kernels[%zu] valid[%zu]",
            groupSize, resCtx.localIndex,
            resCtx.threads.size(), resCtx.ccuKernels.size(), resCtx.kernelValid.size());
        return HCCL_E_INTERNAL;
    }

    uint64_t chunkBytes = CHUNK_SIZE;
    if (groupSize > 1) {
        if (resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < DATA_TYPE_SIZE) {
            HCCL_ERROR("The local Mesh stage requires a non-empty HCCL scratch buffer");
            return HCCL_E_INTERNAL;
        }
        const uint64_t scratchElements = resCtx.localBuffer.size / DATA_TYPE_SIZE;
        const uint64_t scratchSlots = groupSize - 1;
        const uint64_t maxSliceElements = scratchElements / scratchSlots;
        const uint64_t chunkElements = std::min<uint64_t>(
            CHUNK_SIZE / DATA_TYPE_SIZE,
            maxSliceElements > std::numeric_limits<uint64_t>::max() / groupSize
                ? std::numeric_limits<uint64_t>::max()
                : maxSliceElements * groupSize);
        chunkBytes = chunkElements * DATA_TYPE_SIZE;
        if (chunkBytes == 0) {
            return HCCL_E_INTERNAL;
        }
    }

    for (uint64_t offset = 0; offset < totalBytes; offset += chunkBytes) {
        const uint64_t bytes = std::min(chunkBytes, totalBytes - offset);
        const uint64_t inputAddr = inputBase + offset;
        const uint64_t outputAddr = outputBase + offset;

        uint64_t inputToken = 0;
        uint64_t outputToken = 0;
        CcuResult ccuRet = HcommCcuGetMemToken(inputAddr, bytes, &inputToken);
        if (ccuRet != CCU_SUCCESS) {
            return ConvertCcuResult(ccuRet);
        }
        ccuRet = HcommCcuGetMemToken(outputAddr, bytes, &outputToken);
        if (ccuRet != CCU_SUCCESS) {
            return ConvertCcuResult(ccuRet);
        }

        const ChunkSlice localSlice = CalcSlice(bytes, groupSize, resCtx.localIndex);
        // ReduceScatter 只读取本 rank 拥有的 output 分片；其余区域会由 AllGather 覆盖。
        if (inputAddr != outputAddr && localSlice.size != 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0],
                reinterpret_cast<void *>(outputAddr + localSlice.offset),
                reinterpret_cast<void *>(inputAddr + localSlice.offset), localSlice.size)));
        }

        // 从流的第一条任务固定为 local WAIT；各层严格按 die0 -> die1 排列。
        CHK_RET(MainToSlave(resCtx.threads));

        uint64_t scratchAddr = 0;
        uint64_t scratchToken = 0;
        std::vector<uint64_t> localArgs;
        if (groupSize > 1) {
            scratchAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
            const uint64_t scratchSlots = groupSize - 1;
            if (localSlice.size > std::numeric_limits<uint64_t>::max() / scratchSlots) {
                return HCCL_E_INTERNAL;
            }
            const uint64_t scratchTokenBytes
                = std::max<uint64_t>(localSlice.size * scratchSlots, DATA_TYPE_SIZE);
            if (scratchTokenBytes > resCtx.localBuffer.size) {
                return HCCL_E_INTERNAL;
            }
            ccuRet = HcommCcuGetMemToken(scratchAddr, scratchTokenBytes, &scratchToken);
            if (ccuRet != CCU_SUCCESS) {
                return ConvertCcuResult(ccuRet);
            }
            localArgs = {inputAddr, outputAddr, inputToken, outputToken,
                scratchAddr, scratchToken, localSlice.size, localSlice.offset, LOCAL_EXCHANGE_INPUT};
            CHK_RET(RunParallelStage(resCtx, LOCAL_MESH_LEVEL, localArgs));

            if (localSlice.size != 0) {
                std::vector<uint64_t> localReduceArgs = localArgs;
                localReduceArgs.back() = LOCAL_REDUCE;
                CHK_RET(RunSerialStage(resCtx, LOCAL_MESH_LEVEL, localReduceArgs));
            }
        }

        if (localSlice.size != 0) {
            const std::vector<uint64_t> columnArgs
                = {inputAddr, outputAddr, inputToken, outputToken,
                   localSlice.size, localSlice.offset, REDUCE_PHASE};
            CHK_RET(RunSerialStage(resCtx, COLUMN_TREE_LEVEL, columnArgs));

            std::vector<uint64_t> columnBroadcastArgs = columnArgs;
            columnBroadcastArgs.back() = BROADCAST_PHASE;
            CHK_RET(RunSerialStage(resCtx, COLUMN_TREE_LEVEL, columnBroadcastArgs));
        }

        if (groupSize > 1) {
            localArgs.back() = LOCAL_EXCHANGE_OUTPUT;
            CHK_RET(RunParallelStage(resCtx, LOCAL_MESH_LEVEL, localArgs));

            if (localSlice.size != 0) {
                std::vector<uint64_t> localGatherArgs = localArgs;
                localGatherArgs.back() = LOCAL_GATHER;
                CHK_RET(RunParallelStage(resCtx, LOCAL_MESH_LEVEL, localGatherArgs));
            }

            localArgs.back() = LOCAL_BARRIER;
            CHK_RET(RunParallelStage(resCtx, LOCAL_MESH_LEVEL, localArgs));
        }
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
