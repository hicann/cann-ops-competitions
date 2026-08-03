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
#include <array>
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

constexpr uint64_t RELAY_MIN_BYTES_PER_RANK = 1024ULL * 1024ULL;

constexpr uint64_t SetBits(uint16_t end)
{
    return (uint64_t(1) << (end + 1)) - uint64_t(1);
}

uint64_t GetMaxLoopIterNum()
{
    constexpr uint16_t LOOP_NUM_END_BIT = 12;
    return SetBits(LOOP_NUM_END_BIT);
}

uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t REPEAT_NUM_END_BIT = 6;
    constexpr uint16_t REPEAT_NUM_SHIFT = 55;
    constexpr uint16_t REPEAT_LOOP_END_BIT = 6;
    constexpr uint16_t REPEAT_LOOP_SHIFT = 48;
    constexpr uint16_t TOTAL_LOOP_END_BIT = 6;
    constexpr uint16_t TOTAL_LOOP_SHIFT = 41;
    return ((repeatNum & SetBits(REPEAT_NUM_END_BIT)) << REPEAT_NUM_SHIFT) |
           ((repeatLoopIndex & SetBits(REPEAT_LOOP_END_BIT)) << REPEAT_LOOP_SHIFT) |
           ((totalLoopNum & SetBits(TOTAL_LOOP_END_BIT)) << TOTAL_LOOP_SHIFT);
}

std::array<uint64_t, 4> CalculateCopySize(uint64_t size)
{
    const uint64_t memSlice = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;
    const uint64_t loopSize = CCU_MS_LOCAL_COPY_LOOP_COUNT * memSlice;
    const uint64_t maxSize = loopSize * (GetMaxLoopIterNum() + 1);

    uint64_t fullLoops = size / loopSize;
    uint64_t fullSlices = (size - fullLoops * loopSize) / memSlice;
    uint64_t tailSize = size - fullLoops * loopSize - fullSlices * memSlice;
    if (size == maxSize) {
        fullLoops = GetMaxLoopIterNum();
        fullSlices = CCU_MS_LOCAL_COPY_LOOP_COUNT - 1;
        tailSize = memSlice;
    }

    const uint64_t addressOffset = memSlice * CCU_MS_LOCAL_COPY_LOOP_COUNT * fullLoops;
    uint64_t parallelParam = 0;
    uint64_t residual = 0;
    if (fullSlices == 0 && tailSize == 0) {
        parallelParam = 0;
    } else if (fullSlices != 0 && tailSize == 0) {
        parallelParam = GetParallelParam(fullSlices - 1, 0, 1);
        residual = memSlice;
    } else if (fullSlices == 0) {
        parallelParam = GetParallelParam(0, 0, 1);
        residual = tailSize;
    } else {
        parallelParam = GetParallelParam(fullSlices - 1, 1, 2);
        residual = tailSize;
    }
    return {addressOffset, fullLoops, parallelParam, residual};
}

HcclResult GetDataSizes(const OpParam &param, uint64_t &dataTypeSize, uint64_t &dataSize, uint64_t &outputSize)
{
    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("unsupported data type: %d", param.dataType),
        HCCL_E_NOT_SUPPORT);
    dataTypeSize = sizeIt->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("input size overflows uint64"), HCCL_E_PARA);
    dataSize = param.count * dataTypeSize;
    CHK_PRT_RET(param.rankSize != 0 && dataSize > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("output size overflows uint64"), HCCL_E_PARA);
    outputSize = dataSize * param.rankSize;
    return HCCL_SUCCESS;
}

HcclResult SignalSlaves(const AlgResourceCtx &resCtx)
{
    constexpr uint32_t syncId = 0;
    for (std::size_t index = 1; index < resCtx.threads.size(); ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[index], syncId)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[index], syncId, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult JoinSlaves(const AlgResourceCtx &resCtx)
{
    constexpr uint32_t syncId = 0;
    for (std::size_t index = 1; index < resCtx.threads.size(); ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[0], syncId, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[index], resCtx.threads[0], syncId)));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchDirectSlice(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken, uint64_t dataSize, uint64_t sliceSize)
{
    const uint64_t rankOutputOffset = dataSize * param.myRank;
    const auto goSize = CalculateCopySize(sliceSize);
    const std::array<uint64_t, 11> taskArgs = {
        inputAddr,
        outputAddr,
        inputToken,
        outputToken,
        0,
        rankOutputOffset,
        sliceSize,
        goSize[0],
        goSize[1],
        goSize[2],
        goSize[3],
    };
    CHK_RET(SignalSlaves(resCtx));
    for (std::size_t index = 0; index < resCtx.directKernels.size(); ++index) {
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[index], resCtx.directKernels[index], taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size())));
    }
    CHK_RET(JoinSlaves(resCtx));
    return HCCL_SUCCESS;
}

HcclResult ExecuteDirect(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataTypeSize,
    uint64_t dataSize, uint64_t inputToken, uint64_t outputToken)
{
    const uint64_t maxCountPerLaunch = MAX_DATA_SIZE / dataTypeSize;
    uint64_t processedCount = 0;
    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    while (processedCount < param.count) {
        const uint64_t sliceCount = std::min(maxCountPerLaunch, param.count - processedCount);
        const uint64_t processedBytes = processedCount * dataTypeSize;
        CHK_RET(LaunchDirectSlice(param, resCtx, inputBase + processedBytes, outputBase + processedBytes,
            inputToken, outputToken, dataSize, sliceCount * dataTypeSize));
        processedCount += sliceCount;
    }
    return HCCL_SUCCESS;
}

HcclResult ExecuteRelay(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataTypeSize,
    uint64_t dataSize, uint64_t inputToken, uint64_t outputToken)
{
    std::array<uint64_t, RELAY_CHUNK_COUNT> chunkOffsets{};
    std::array<uint64_t, RELAY_CHUNK_COUNT> chunkSizes{};
    const uint64_t baseCount = param.count / RELAY_CHUNK_COUNT;
    const uint64_t remainder = param.count % RELAY_CHUNK_COUNT;
    uint64_t processedCount = 0;
    for (uint32_t chunk = 0; chunk < RELAY_CHUNK_COUNT; ++chunk) {
        const uint64_t chunkCount = baseCount + static_cast<uint64_t>(chunk < remainder);
        chunkOffsets[chunk] = processedCount * dataTypeSize;
        chunkSizes[chunk] = chunkCount * dataTypeSize;
        CHK_PRT_RET(chunkSizes[chunk] > MAX_DATA_SIZE,
            HCCL_ERROR("relay chunk %u exceeds the 256 MiB transfer limit", chunk), HCCL_E_PARA);
        processedCount += chunkCount;
    }

    std::vector<uint64_t> taskArgs;
    taskArgs.reserve(4 + MAX_RANK_SIZE + RELAY_CHUNK_COUNT * 6);
    taskArgs.push_back(reinterpret_cast<uint64_t>(param.inputPtr));
    taskArgs.push_back(reinterpret_cast<uint64_t>(param.outputPtr));
    taskArgs.push_back(inputToken);
    taskArgs.push_back(outputToken);
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        taskArgs.push_back(dataSize * rank);
    }
    for (uint64_t offset : chunkOffsets) {
        taskArgs.push_back(offset);
    }
    for (uint64_t size : chunkSizes) {
        taskArgs.push_back(size);
    }
    for (uint64_t size : chunkSizes) {
        const auto goSize = CalculateCopySize(size);
        taskArgs.insert(taskArgs.end(), goSize.begin(), goSize.end());
    }

    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.relayKernel, taskArgs.data(),
        static_cast<uint32_t>(taskArgs.size())));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *serialized = static_cast<char *>(param.resCtx);
    std::vector<char> sequence(serialized, serialized + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(sequence);
    if (!resCtx.threads.empty()) {
        resCtx.threads[0] = param.cpuThread;
    }

    uint64_t dataTypeSize = 0;
    uint64_t dataSize = 0;
    uint64_t outputSize = 0;
    CHK_RET(GetDataSizes(param, dataTypeSize, dataSize, outputSize));
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("CCU thread was not registered"), HCCL_E_INTERNAL);
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize));
    }

    CHK_PRT_RET(resCtx.directKernels.empty() || resCtx.directKernels.size() != resCtx.threads.size(),
        HCCL_ERROR("CCU thread/kernel groups are incomplete"), HCCL_E_INTERNAL);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), dataSize, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), outputSize, &outputToken));

    const bool useRelay = resCtx.relayEnabled != 0 && resCtx.relayKernel != 0 &&
        dataSize > RELAY_MIN_BYTES_PER_RANK;
    if (useRelay) {
        return ExecuteRelay(param, resCtx, dataTypeSize, dataSize, inputToken, outputToken);
    }
    return ExecuteDirect(param, resCtx, dataTypeSize, dataSize, inputToken, outputToken);
}

} // namespace ops_hccl
