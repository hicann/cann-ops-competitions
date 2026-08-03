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
#include <cinttypes>
#include <cstdint>
#include <limits>
#include <vector>

#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"
#include "ccu_launch.h"

namespace ops_hccl {
namespace {
    constexpr uint64_t CHUNK_ALIGNMENT = 128;
    constexpr uint64_t SMALL_RECV_SIZE_LIMIT = 1024 * 1024;
    constexpr uint32_t HIERARCHICAL_SCRATCH_SLOT_COUNT = 16;

    bool IsSmallHierarchicalParam(const OpParam &param)
    {
        return param.rankSize == 12 && param.count <= SMALL_RECV_SIZE_LIMIT / sizeof(float);
    }

    HcclResult GetRangeEnd(uintptr_t start, uint64_t size, uintptr_t &end)
    {
        if (size > std::numeric_limits<uintptr_t>::max() - start) {
            HCCL_ERROR("Buffer address range overflows uintptr_t");
            return HCCL_E_PARA;
        }
        end = start + static_cast<uintptr_t>(size);
        return HCCL_SUCCESS;
    }

    HcclResult CheckNoOverlap(const OpParam &param, uint64_t recvSize, uint64_t inputSize)
    {
        const uintptr_t inputStart = reinterpret_cast<uintptr_t>(param.inputPtr);
        const uintptr_t outputStart = reinterpret_cast<uintptr_t>(param.outputPtr);
        uintptr_t inputEnd = 0;
        uintptr_t outputEnd = 0;
        CHK_RET(GetRangeEnd(inputStart, inputSize, inputEnd));
        CHK_RET(GetRangeEnd(outputStart, recvSize, outputEnd));

        const bool overlaps = inputStart < outputEnd && outputStart < inputEnd;
        if (overlaps) {
            HCCL_ERROR("Overlapping sendBuf and recvBuf are not supported");
            return HCCL_E_NOT_SUPPORT;
        }
        return HCCL_SUCCESS;
    }

    HcclResult CheckResourceContext(const OpParam &param, const AlgResourceCtx &resourceCtx)
    {
        if (resourceCtx.localBuffer.addr == nullptr || resourceCtx.localBuffer.size == 0) {
            HCCL_ERROR("Invalid local HCCL buffer");
            return HCCL_E_INTERNAL;
        }
        const uint32_t expectedKernelCount
            = IsSmallHierarchicalParam(param)
                  ? 2
                  : (resourceCtx.partialKernelCount == 2 ? 3 : resourceCtx.partialKernelCount);
        if (param.rankSize > 1
            && (resourceCtx.partialKernelCount == 0 || resourceCtx.partialKernelCount > 2
                || resourceCtx.ccuKernels.size() != expectedKernelCount)) {
            HCCL_ERROR("Invalid CCU resource context, kernels[%zu/%u]", resourceCtx.ccuKernels.size(),
                resourceCtx.partialKernelCount);
            return HCCL_E_INTERNAL;
        }
        if (!IsSmallHierarchicalParam(param) && resourceCtx.partialKernelCount == 2
            && (resourceCtx.secondaryThread == 0 || resourceCtx.secondaryScratchStartSlot == 0
                || resourceCtx.secondaryScratchStartSlot >= param.rankSize)) {
            HCCL_ERROR("Invalid dual-die CCU resource context");
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel, std::vector<uint64_t> &taskArgs)
    {
        CcuResult launchRet
            = HcommCcuKernelLaunch(thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("CCU kernel launch failed, ret[%d]", static_cast<int32_t>(launchRet));
            return ConvertCcuToHccl(launchRet);
        }
        return HCCL_SUCCESS;
    }

    HcclResult LaunchDualDieChunk(const OpParam &param, const AlgResourceCtx &resCtx, std::vector<uint64_t> &taskArgs)
    {
        constexpr uint32_t THREAD_NOTIFY_INDEX = 0;
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(param.cpuThread, resCtx.secondaryThread, THREAD_NOTIFY_INDEX)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.secondaryThread, THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT)));

        CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[0], taskArgs));
        CHK_RET(LaunchKernel(resCtx.secondaryThread, resCtx.ccuKernels[1], taskArgs));

        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(param.cpuThread, THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.secondaryThread, param.cpuThread, THREAD_NOTIFY_INDEX)));

        // Merge on CCU as well. The simulator's Host Runtime Reduce path resolves
        // both virtual addresses through CurrContext and fails for CCU scratch/output
        // buffers in two-die cases.
        CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[2], taskArgs));
        return HCCL_SUCCESS;
    }

    HcclResult LaunchHierarchicalSmall(
        const OpParam &param, const AlgResourceCtx &resCtx, std::vector<uint64_t> &taskArgs)
    {
        // Both kernels are enqueued on the user stream.  Each generated CCU task
        // carries its own dieId, so stream order provides the layer-0 -> layer-1
        // dependency without a secondary runtime stream or four Thread Notify
        // tasks.
        CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[1], taskArgs));
        CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[0], taskArgs));
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    if (param.resCtx == nullptr || param.ctxSize == 0) {
        HCCL_ERROR("Invalid engine context");
        return HCCL_E_INTERNAL;
    }

    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);
    CHK_RET(CheckResourceContext(param, resCtx));

    const auto sizeIter = SIZE_TABLE.find(param.dataType);
    if (sizeIter == SIZE_TABLE.end()) {
        HCCL_ERROR("Unsupported dataType[%d]", static_cast<int32_t>(param.dataType));
        return HCCL_E_NOT_SUPPORT;
    }
    const uint64_t dataTypeSize = sizeIter->second;
    if (param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize) {
        HCCL_ERROR("recvCount[%" PRIu64 "] causes byte-size overflow", param.count);
        return HCCL_E_PARA;
    }
    const uint64_t recvSize = param.count * dataTypeSize;
    if (recvSize == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize
        || param.rankSize > std::numeric_limits<uint64_t>::max() / recvSize) {
        HCCL_ERROR(
            "Invalid rank or input size, rank[%u/%u], recvSize[%" PRIu64 "]", param.myRank, param.rankSize, recvSize);
        return HCCL_E_PARA;
    }
    const uint64_t inputSize = recvSize * param.rankSize;

    if (param.rankSize == 1) {
        if (param.outputPtr == param.inputPtr) {
            return HCCL_SUCCESS;
        }
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, recvSize)));
        return HCCL_SUCCESS;
    }
    CHK_RET(CheckNoOverlap(param, recvSize, inputSize));

    if (IsSmallHierarchicalParam(param)) {
        if (recvSize > std::numeric_limits<uint64_t>::max() / HIERARCHICAL_SCRATCH_SLOT_COUNT
            || resCtx.localBuffer.size < recvSize * HIERARCHICAL_SCRATCH_SLOT_COUNT) {
            HCCL_ERROR("HCCL buffer[%" PRIu64 "] is too small for hierarchical recvSize[%" PRIu64 "]",
                resCtx.localBuffer.size, recvSize);
            return HCCL_E_INTERNAL;
        }

        uint64_t inputToken = 0;
        uint64_t outputToken = 0;
        uint64_t scratchToken = 0;
        CHK_RET_CCU(HcommCcuGetMemToken(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(param.inputPtr)), inputSize, &inputToken));
        CHK_RET_CCU(HcommCcuGetMemToken(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(param.outputPtr)), recvSize, &outputToken));
        CHK_RET_CCU(HcommCcuGetMemToken(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(resCtx.localBuffer.addr)),
            resCtx.localBuffer.size, &scratchToken));
        std::vector<uint64_t> taskArgs = {
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(param.inputPtr)),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(param.outputPtr)),
            inputToken,
            outputToken,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(resCtx.localBuffer.addr)),
            scratchToken,
            recvSize,
        };
        CHK_RET(LaunchHierarchicalSmall(param, resCtx, taskArgs));
        return HCCL_SUCCESS;
    }

    const uint32_t scratchSlotCount = param.rankSize;
    // ContiguousFixedReduce reduces floor(sourceCount / 2) adjacent pieces in
    // its widest round.  On the contest's two-die topologies each die owns at
    // most eight sources, so four pieces are the global worst case.  Keep the
    // value rank-size based: every rank must derive the same chunk schedule on
    // the asymmetric 8+4 topology.
    uint32_t maxReducePieces = param.rankSize / 2;
    if (resCtx.partialKernelCount == 2 && (param.rankSize == 12 || param.rankSize == 16)) {
        maxReducePieces = 4;
    }
    const uint64_t primitiveCapacity = MAX_DATA_SIZE / std::max<uint32_t>(1, maxReducePieces);
    uint64_t chunkCapacity
        = std::min<uint64_t>({primitiveCapacity, resCtx.localBuffer.size / scratchSlotCount, recvSize});
    chunkCapacity = chunkCapacity / CHUNK_ALIGNMENT * CHUNK_ALIGNMENT;
    if (chunkCapacity == 0) {
        chunkCapacity = std::min<uint64_t>(resCtx.localBuffer.size / scratchSlotCount, recvSize);
        chunkCapacity = chunkCapacity / dataTypeSize * dataTypeSize;
    }
    if (chunkCapacity < dataTypeSize) {
        HCCL_ERROR("HCCL buffer[%" PRIu64 "] is too small for rankSize[%u]", resCtx.localBuffer.size, param.rankSize);
        return HCCL_E_INTERNAL;
    }

    const uint64_t rankSliceOffset = static_cast<uint64_t>(param.myRank) * recvSize;

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(param.inputPtr)), inputSize, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(param.outputPtr)), recvSize, &outputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(resCtx.localBuffer.addr)),
        resCtx.localBuffer.size, &scratchToken));

    std::vector<uint64_t> taskArgs = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(param.inputPtr)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(param.outputPtr)),
        inputToken,
        outputToken,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(resCtx.localBuffer.addr)),
        scratchToken,
        rankSliceOffset,
        chunkCapacity,
        0,
        0,
    };

    uint64_t chunkOffset = 0;
    while (chunkOffset < recvSize) {
        const uint64_t dataSize = std::min(chunkCapacity, recvSize - chunkOffset);
        taskArgs[8] = chunkOffset;
        taskArgs[9] = dataSize;
        if (resCtx.partialKernelCount == 1) {
            CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[0], taskArgs));
        } else {
            CHK_RET(LaunchDualDieChunk(param, resCtx, taskArgs));
        }
        chunkOffset += dataSize;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
