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
#include <cstdint>

#include <ccu/ccu_res.h>
#include <ccu_launch.h>

#include "log.h"
#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {

namespace {
constexpr uint64_t DEFAULT_PIPELINE_STEPS = 4;
constexpr uint64_t FINE_PIPELINE_STEPS = 8;
constexpr uint64_t DEEP_PIPELINE_STEPS = 12;
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);

    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    CHK_PRT_RET(
        resCtx.threads.empty() || resCtx.directKernels.empty(),
        HCCL_ERROR("[ExecOp] CCU resource is incomplete"),
        HCCL_E_INTERNAL);

    const auto sizeIt = SIZE_TABLE.find(param.dataType);

    CHK_PRT_RET(
        sizeIt == SIZE_TABLE.end(),
        HCCL_ERROR(
            "[ExecOp] unsupported data type: %d",
            static_cast<int32_t>(param.dataType)),
        HCCL_E_NOT_SUPPORT);

    const uint64_t dataTypeSize = sizeIt->second;

    CHK_PRT_RET(
        param.count > UINT64_MAX / dataTypeSize,
        HCCL_ERROR("[ExecOp] data size overflow"),
        HCCL_E_PARA);

    const uint64_t dataSize =
        param.count * dataTypeSize;

    uint64_t token = 0;
    const uint64_t baseAddress =
        reinterpret_cast<uint64_t>(param.outputPtr);

    HcommCcuGetMemToken(
        baseAddress,
        dataSize,
        &token);

    constexpr uint64_t smallDataThreshold = 1024 * 1024;
    constexpr uint64_t dmaAlignment = 16 * 1024;
    const uint64_t ownerCount = param.rankSize - 1;
    uint64_t blockSize = dataSize;
    if (ownerCount > 1) {
        const uint64_t unalignedBlockSize = dataSize / ownerCount + (dataSize % ownerCount != 0);
        blockSize = unalignedBlockSize;
        if (unalignedBlockSize <= UINT64_MAX - (dmaAlignment - 1)) {
            const uint64_t alignedBlockSize =
                ((unalignedBlockSize + dmaAlignment - 1) / dmaAlignment) * dmaAlignment;
            if (alignedBlockSize <= (dataSize - 1) / (ownerCount - 1)) {
                blockSize = alignedBlockSize;
            }
        }
    }
    const bool canUsePartition = ownerCount > 1 && blockSize <= MAX_DATA_SIZE &&
        blockSize <= (dataSize - 1) / (ownerCount - 1);

    if (resCtx.meshEnabled && dataSize > smallDataThreshold && canUsePartition) {
        /*
         * 单层全互联拓扑：一个Kernel内流水执行Scatter和AllGather。
         * root每完成一个分块就唤醒对应owner，使扩散与剩余Scatter重叠。
        */
        uint64_t pipelineSteps = DEFAULT_PIPELINE_STEPS;
        if (param.rankSize == MAX_RANK_SIZE) {
            pipelineSteps = DEEP_PIPELINE_STEPS;
        } else if (param.rankSize == 4 || param.rankSize == 12) {
            pipelineSteps = FINE_PIPELINE_STEPS;
        }
        const uint64_t lastBlockSize = dataSize - blockSize * (ownerCount - 1);
        const uint64_t blockSegmentSize = blockSize / pipelineSteps;
        const uint64_t blockLastSegmentSize =
            blockSize - blockSegmentSize * (pipelineSteps - 1);
        const uint64_t tailSegmentSize = lastBlockSize / pipelineSteps;
        const uint64_t tailLastSegmentSize =
            lastBlockSize - tailSegmentSize * (pipelineSteps - 1);
        const std::array<uint64_t, 9> taskArgs{
            baseAddress,
            token,
            0,
            blockSize,
            lastBlockSize,
            blockSegmentSize,
            blockLastSegmentSize,
            tailSegmentSize,
            tailLastSegmentSize};

        for (uint32_t netLayer = 0; netLayer < resCtx.meshKernels.size(); ++netLayer) {
            const CcuKernelHandle kernel = resCtx.meshKernels[netLayer];
            if (kernel == CcuKernelHandle{}) {
                continue;
            }

            const uint32_t threadIndex = std::min<uint32_t>(netLayer, resCtx.threads.size() - 1);
            CHK_RET_CCU(HcommCcuKernelLaunch(
                resCtx.threads[threadIndex], kernel, taskArgs.data(), taskArgs.size()));
        }
        return HCCL_SUCCESS;
    }

    if (resCtx.stagedEnabled && dataSize > smallDataThreshold && canUsePartition) {
        /*
         * 跨layer拓扑：两个IO Die分别由独立stream驱动。
         * Scatter在本地HBM写入就绪标记，Gather确认本rank分块已到达后再扩散，
         * 因此两层能够安全重叠，而不会读取尚未到达的数据。
         */
        CHK_PRT_RET(
            resCtx.threads.size() < 2 || resCtx.primaryStream == nullptr ||
                resCtx.auxiliaryStream == nullptr || resCtx.startEvent == nullptr ||
                resCtx.finishEvent == nullptr || resCtx.localBuffer.addr == nullptr ||
                resCtx.localBuffer.size < sizeof(uint64_t),
            HCCL_ERROR("[ExecOp] parallel layer resource is incomplete"),
            HCCL_E_INTERNAL);

        const uint64_t lastBlockSize = dataSize - blockSize * (ownerCount - 1);
        void *readyMarker = static_cast<char *>(resCtx.localBuffer.addr) +
            resCtx.localBuffer.size - sizeof(uint64_t);
        const uint64_t readyAddress = reinterpret_cast<uint64_t>(readyMarker);
        constexpr uint64_t readyValue = 1;
        const std::array<uint64_t, 7> taskArgs{
            baseAddress, token, 0, blockSize, lastBlockSize, readyAddress, readyValue};

        ACLCHECK(aclrtMemsetAsync(
            readyMarker, sizeof(uint64_t), 0, sizeof(uint64_t), resCtx.primaryStream));
        ACLCHECK(aclrtRecordEvent(resCtx.startEvent, resCtx.primaryStream));
        ACLCHECK(aclrtStreamWaitEvent(resCtx.auxiliaryStream, resCtx.startEvent));

        for (uint32_t netLayer = 0; netLayer < resCtx.scatterKernels.size(); ++netLayer) {
            const CcuKernelHandle kernel = resCtx.scatterKernels[netLayer];
            if (kernel != CcuKernelHandle{}) {
                CHK_RET_CCU(HcommCcuKernelLaunch(
                    resCtx.threads[netLayer], kernel, taskArgs.data(), taskArgs.size()));
            }
        }
        for (uint32_t netLayer = 0; netLayer < resCtx.gatherKernels.size(); ++netLayer) {
            const CcuKernelHandle kernel = resCtx.gatherKernels[netLayer];
            if (kernel != CcuKernelHandle{}) {
                CHK_RET_CCU(HcommCcuKernelLaunch(
                    resCtx.threads[netLayer], kernel, taskArgs.data(), taskArgs.size()));
            }
        }

        ACLCHECK(aclrtRecordEvent(resCtx.finishEvent, resCtx.auxiliaryStream));
        ACLCHECK(aclrtStreamWaitEvent(resCtx.primaryStream, resCtx.finishEvent));
        return HCCL_SUCCESS;
    }

    /*
     * UBC CTP单次最多传输256MB。
     * 直接对用户buffer切片，不需要把完整数据放入400MB HCCL Buffer。
     */
    const bool parallelDirect = resCtx.stagedEnabled && resCtx.threads.size() >= 2 &&
        resCtx.primaryStream != nullptr && resCtx.auxiliaryStream != nullptr &&
        resCtx.startEvent != nullptr && resCtx.finishEvent != nullptr;
    if (parallelDirect) {
        ACLCHECK(aclrtRecordEvent(resCtx.startEvent, resCtx.primaryStream));
        ACLCHECK(aclrtStreamWaitEvent(resCtx.auxiliaryStream, resCtx.startEvent));
    }

    for (uint64_t offset = 0; offset < dataSize;) {
        const uint64_t sliceSize =
            std::min<uint64_t>(
                MAX_DATA_SIZE,
                dataSize - offset);

        const std::array<uint64_t, 4> taskArgs{baseAddress, token, offset, sliceSize};

        /*
         * 每个切片每层最多下发一个Kernel任务。
         * 不同层的网络设备不会出现在同一个Kernel中。
         */
        for (uint32_t netLayer = 0; netLayer < resCtx.directKernels.size(); ++netLayer) {
            const CcuKernelHandle kernel = resCtx.directKernels[netLayer];
            if (kernel == CcuKernelHandle{}) {
                continue;
            }

            const uint32_t threadIndex = parallelDirect
                ? std::min<uint32_t>(netLayer, resCtx.threads.size() - 1)
                : 0;
            CHK_RET_CCU(HcommCcuKernelLaunch(
                resCtx.threads[threadIndex], kernel, taskArgs.data(), taskArgs.size()));
        }

        offset += sliceSize;
    }

    if (parallelDirect) {
        ACLCHECK(aclrtRecordEvent(resCtx.finishEvent, resCtx.auxiliaryStream));
        ACLCHECK(aclrtStreamWaitEvent(resCtx.primaryStream, resCtx.finishEvent));
    }

    return HCCL_SUCCESS;
}

} // namespace ops_hccl