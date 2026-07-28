/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint32_t TOPOLOGY_PHASE_TASK_ARG_INDEX = 6;
constexpr uint64_t TOPOLOGY_LOCAL_REDUCE_SELECTOR = 0;
constexpr uint64_t TOPOLOGY_LOCAL_GATHER_SELECTOR = 1;

HcclResult GetDataBytes(const OpParam &param, uint64_t &dataBytes)
{
    auto iter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(iter == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", param.dataType), HCCL_E_PARA);
    CHK_PRT_RET(param.count > UINT64_MAX / iter->second,
        HCCL_ERROR("Data bytes overflow, count[%llu]", static_cast<unsigned long long>(param.count)), HCCL_E_PARA);
    dataBytes = param.count * iter->second;
    return HCCL_SUCCESS;
}

HcclResult GetMemToken(const void *address, uint64_t size, uint64_t &token)
{
    CcuResult ccuRet = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(address), size, &token);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("Get CCU memory token failed, ret[%d]", ccuRet),
        ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult RecordThreadNotify(ThreadHandle source, ThreadHandle destination, uint32_t notifyIndex)
{
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(source, destination, notifyIndex)));
    return HCCL_SUCCESS;
}

HcclResult WaitThreadNotify(ThreadHandle destination, uint32_t notifyIndex)
{
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(destination, notifyIndex, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult SyncThreads(ThreadHandle source, ThreadHandle destination)
{
    CHK_RET(RecordThreadNotify(source, destination, 0));
    CHK_RET(WaitThreadNotify(destination, 0));
    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel,
    const std::vector<uint64_t> &taskArgs, const char *phaseName)
{
    const CcuResult ccuRet = HcommCcuKernelLaunch(thread, kernel,
        taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("Launch CCU AllReduce %s kernel failed, ret[%d]", phaseName, ccuRet),
        ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult GetRadix4ScratchBytes(const OpParam &param, uint64_t &scratchBytes)
{
    CHK_PRT_RET(param.rankSize != HIGH_RADIX,
        HCCL_ERROR("Radix-4 RSAG requires 4 ranks"), HCCL_E_PARA);

    const uint64_t baseElements = param.count / HIGH_RADIX;
    const uint64_t remainder = param.count % HIGH_RADIX;
    const uint64_t maxSegmentElements = baseElements + (remainder == 0 ? 0ULL : 1ULL);
    CHK_PRT_RET(maxSegmentElements > UINT64_MAX / sizeof(float),
        HCCL_ERROR("Radix-4 RSAG scratch size overflow"), HCCL_E_PARA);

    const uint64_t maxSegmentBytes = maxSegmentElements * sizeof(float);
    const uint64_t batchBytes = maxSegmentBytes > HIERARCHICAL_BATCH_CHUNK_BYTES ?
        HIERARCHICAL_BATCH_CHUNK_BYTES : maxSegmentBytes;
    scratchBytes = batchBytes * (HIGH_RADIX - 1U);
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resource)
{
    uint64_t dataBytes = 0;
    CHK_RET(GetDataBytes(param, dataBytes));
    if (dataBytes == 0) {
        return HCCL_SUCCESS;
    }

    if (param.rankSize == 1) {
        if (param.inputPtr != param.outputPtr) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(resource.thread, param.outputPtr, param.inputPtr, dataBytes)));
        }
        return HCCL_SUCCESS;
    }

    const bool topologyAware =
        (resource.algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_HALVING_DOUBLING && param.rankSize == 16U) ||
        (resource.algorithm == AllReduceAlgorithm::CLOS_RADIX4X3_RSAG && param.rankSize == 12U) ||
        (resource.algorithm == AllReduceAlgorithm::CLOS_FULL_LANE_FLAT_OWNER_RSAG &&
            (param.rankSize == 12U || param.rankSize == 16U));
    const bool separateRhdPhases = resource.algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_HALVING_DOUBLING &&
        param.rankSize == 16U;
    const uint32_t pipelineDepth = SelectEqualPipelineDepth(param.rankSize, dataBytes);
    const bool pipelineFlatOwner =
        resource.algorithm == AllReduceAlgorithm::CLOS_FULL_LANE_FLAT_OWNER_RSAG &&
        (param.rankSize == 12U || param.rankSize == 16U) && pipelineDepth > 1U;
    const bool invalidKernel = topologyAware ?
        resource.slaveThread == 0 || (pipelineFlatOwner && resource.gatherThread == 0) ||
            resource.phase0Kernel == 0 || resource.phase1Kernel == 0 || resource.phase2Kernel == 0 :
        resource.kernel == 0;
    CHK_PRT_RET(invalidKernel || resource.rankSize != param.rankSize || resource.dataBytes != dataBytes,
        HCCL_ERROR("Invalid CCU AllReduce resource"), HCCL_E_INTERNAL);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET(GetMemToken(param.inputPtr, dataBytes, inputToken));
    CHK_RET(GetMemToken(param.outputPtr, dataBytes, outputToken));

    uint64_t scratchAddr = 0;
    uint64_t scratchToken = 0;
    if (resource.algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_DOUBLING) {
        CHK_PRT_RET(dataBytes > UINT64_MAX / HIGH_RADIX,
            HCCL_ERROR("High-radix RD scratch size overflow"), HCCL_E_PARA);
        const uint64_t scratchBytes = dataBytes * HIGH_RADIX;
        CHK_PRT_RET(resource.localBuffer.addr == nullptr || resource.localBuffer.size < scratchBytes,
            HCCL_ERROR("HCCL scratch buffer is too small for Recursive Doubling"), HCCL_E_INTERNAL);
        scratchAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
        CHK_RET(GetMemToken(resource.localBuffer.addr, scratchBytes, scratchToken));
    } else if (resource.algorithm == AllReduceAlgorithm::CLOS_RADIX4_RSAG) {
        uint64_t scratchBytes = 0;
        CHK_RET(GetRadix4ScratchBytes(param, scratchBytes));
        CHK_PRT_RET(resource.localBuffer.addr == nullptr || resource.localBuffer.size < scratchBytes,
            HCCL_ERROR("HCCL scratch buffer is too small for radix-4 RSAG"), HCCL_E_INTERNAL);
        scratchAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
        CHK_RET(GetMemToken(resource.localBuffer.addr, scratchBytes, scratchToken));
    }

    const std::vector<uint64_t> taskArgs = {
        reinterpret_cast<uint64_t>(param.inputPtr),
        reinterpret_cast<uint64_t>(param.outputPtr),
        inputToken,
        outputToken,
        scratchAddr,
        scratchToken,
    };
    if (topologyAware) {
        if (separateRhdPhases) {
            CHK_RET(LaunchKernel(resource.thread, resource.phase0Kernel, taskArgs, "local-loop-reduce"));
            CHK_RET(SyncThreads(resource.thread, resource.slaveThread));
            CHK_RET(LaunchKernel(resource.slaveThread, resource.phase1Kernel, taskArgs, "cross"));
            CHK_RET(SyncThreads(resource.slaveThread, resource.thread));
            CHK_RET(LaunchKernel(resource.thread, resource.phase2Kernel, taskArgs, "local-gather"));
            return HCCL_SUCCESS;
        }

        if (pipelineFlatOwner) {
            for (uint32_t slice = 0; slice < pipelineDepth; ++slice) {
                std::vector<uint64_t> localTaskArgs = taskArgs;
                localTaskArgs.push_back(TOPOLOGY_LOCAL_REDUCE_SELECTOR);
                localTaskArgs.push_back(slice);
                CHK_RET(LaunchKernel(resource.thread, resource.phase0Kernel,
                    localTaskArgs, "local-reduce-slice"));
                CHK_RET(RecordThreadNotify(resource.thread, resource.slaveThread, slice));
                CHK_RET(WaitThreadNotify(resource.slaveThread, slice));

                std::vector<uint64_t> crossTaskArgs = taskArgs;
                crossTaskArgs.push_back(slice);
                CHK_RET(LaunchKernel(resource.slaveThread, resource.phase1Kernel,
                    crossTaskArgs, "cross-slice"));
                CHK_RET(RecordThreadNotify(resource.slaveThread, resource.gatherThread, slice));
                CHK_RET(WaitThreadNotify(resource.gatherThread, slice));

                localTaskArgs[TOPOLOGY_PHASE_TASK_ARG_INDEX] = TOPOLOGY_LOCAL_GATHER_SELECTOR;
                CHK_RET(LaunchKernel(resource.gatherThread, resource.phase2Kernel,
                    localTaskArgs, "local-gather-slice"));
            }
            CHK_RET(SyncThreads(resource.gatherThread, resource.thread));
            return HCCL_SUCCESS;
        }

        std::vector<uint64_t> localTaskArgs = taskArgs;
        localTaskArgs.push_back(TOPOLOGY_LOCAL_REDUCE_SELECTOR);
        CHK_RET(LaunchKernel(resource.thread, resource.phase0Kernel, localTaskArgs, "local-reduce"));
        CHK_RET(SyncThreads(resource.thread, resource.slaveThread));
        CHK_RET(LaunchKernel(resource.slaveThread, resource.phase1Kernel, taskArgs, "cross"));
        CHK_RET(SyncThreads(resource.slaveThread, resource.thread));
        localTaskArgs[TOPOLOGY_PHASE_TASK_ARG_INDEX] = TOPOLOGY_LOCAL_GATHER_SELECTOR;
        CHK_RET(LaunchKernel(resource.thread, resource.phase2Kernel, localTaskArgs, "local-gather"));
        return HCCL_SUCCESS;
    }

    CHK_RET(LaunchKernel(resource.thread, resource.kernel, taskArgs, "full"));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
