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
#include <vector>

#include <ccu/ccu_res.h>
#include <ccu_launch.h>

#include "ccu_kernel.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

    constexpr uint64_t MAX_TRANSFER_BYTES = 256ULL * 1024ULL * 1024ULL;
    constexpr uint64_t RELAY_MIN_BYTES = 8ULL * 1024ULL * 1024ULL;
    constexpr uint32_t RELAY_PIPELINE_PARTS = 4;
    constexpr uint32_t PIPELINE_NOTIFY_BASE = 1;
    // 3/5 split balances die0 (1×) local-distribution load against die1 (4×)
    // per-channel throughput.  Raising the direct share overloads die1 channels
    // before die0 is sufficiently relieved.
    constexpr uint64_t RELAY_DIRECT_NUMERATOR = 3;
    constexpr uint64_t RELAY_DIRECT_DENOMINATOR = 5;
    constexpr uint64_t RELAY_ALIGNMENT = 128;
    constexpr uint64_t DIRECT_PHASE = 0;
    constexpr uint64_t RELAY_PHASE = 1;
    constexpr uint32_t RANKS_PER_SERVER = 8;

    constexpr uint64_t SetLowBits(uint16_t end)
    {
        return (uint64_t(1) << (end + 1)) - uint64_t(1);
    }

    uint64_t MaxLoopIteration()
    {
        constexpr uint16_t LOOP_NUMBER_BITS = 12;
        return SetLowBits(LOOP_NUMBER_BITS);
    }

    uint64_t PackParallelParameter(uint64_t repeatNumber, uint64_t repeatLoopIndex, uint64_t totalLoops)
    {
        constexpr uint16_t FIELD_BITS = 7;
        constexpr uint16_t REPEAT_SHIFT = 55;
        constexpr uint16_t REPEAT_LOOP_SHIFT = 48;
        constexpr uint16_t TOTAL_LOOP_SHIFT = 41;
        return ((repeatNumber & SetLowBits(FIELD_BITS)) << REPEAT_SHIFT)
               | ((repeatLoopIndex & SetLowBits(FIELD_BITS)) << REPEAT_LOOP_SHIFT)
               | ((totalLoops & SetLowBits(FIELD_BITS)) << TOTAL_LOOP_SHIFT);
    }

    std::array<uint64_t, 4> CalculateGroupCopySize(uint64_t bytes)
    {
        constexpr uint64_t MEMORY_SLICE_BYTES = CCU_LOCAL_COPY_SLICES_PER_LOOP * CCU_MEMORY_SLICE_BYTES;
        constexpr uint64_t LOOP_BYTES = CCU_LOCAL_COPY_LOOP_COUNT * MEMORY_SLICE_BYTES;

        uint64_t completeLoops = bytes / LOOP_BYTES;
        uint64_t completeSlices = (bytes % LOOP_BYTES) / MEMORY_SLICE_BYTES;
        uint64_t residualBytes = bytes % MEMORY_SLICE_BYTES;
        const uint64_t maxBytes = LOOP_BYTES * (MaxLoopIteration() + 1);

        if (bytes == maxBytes) {
            completeLoops = MaxLoopIteration();
            completeSlices = CCU_LOCAL_COPY_LOOP_COUNT - 1;
            residualBytes = MEMORY_SLICE_BYTES;
        }

        const uint64_t addressOffset = MEMORY_SLICE_BYTES * CCU_LOCAL_COPY_LOOP_COUNT * completeLoops;
        uint64_t parallelParameter = 0;
        uint64_t tailBytes = 0;
        if (completeSlices != 0 && residualBytes == 0) {
            parallelParameter = PackParallelParameter(completeSlices - 1, 0, 1);
            tailBytes = MEMORY_SLICE_BYTES;
        } else if (completeSlices == 0 && residualBytes != 0) {
            parallelParameter = PackParallelParameter(0, 0, 1);
            tailBytes = residualBytes;
        } else if (completeSlices != 0) {
            parallelParameter = PackParallelParameter(completeSlices - 1, 1, 2);
            tailBytes = residualBytes;
        }

        return {addressOffset, completeLoops, parallelParameter, tailBytes};
    }

    HcclResult PreSynchronizeThreads(const std::vector<ThreadHandle> &threads)
    {
        for (size_t i = 1; i < threads.size(); ++i) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[i], 0)));
        }
        for (size_t i = 1; i < threads.size(); ++i) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[i], 0, CUSTOM_TIMEOUT)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult PostSynchronizeThreads(const std::vector<ThreadHandle> &threads)
    {
        for (size_t i = 1; i < threads.size(); ++i) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], 0, CUSTOM_TIMEOUT)));
        }
        for (size_t i = 1; i < threads.size(); ++i) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[i], threads[0], 0)));
        }
        return HCCL_SUCCESS;
    }

    struct TransferSlice {
        uint64_t offset = 0;
        uint64_t bytes = 0;
    };

    std::vector<TransferSlice> BuildDirectSlices(uint64_t totalBytes)
    {
        std::vector<TransferSlice> slices;
        for (uint64_t offset = 0; offset < totalBytes;) {
            const uint64_t bytes = std::min(MAX_TRANSFER_BYTES, totalBytes - offset);
            slices.push_back(TransferSlice{offset, bytes});
            offset += bytes;
        }
        return slices;
    }

    std::vector<TransferSlice> BuildRelaySlices(uint64_t totalBytes)
    {
        std::vector<TransferSlice> slices;
        slices.reserve(RELAY_PIPELINE_PARTS);
        uint64_t offset = 0;
        for (uint32_t part = 0; part < RELAY_PIPELINE_PARTS; ++part) {
            const uint32_t remainingParts = RELAY_PIPELINE_PARTS - part;
            const uint64_t remainingBytes = totalBytes - offset;
            uint64_t bytes = remainingBytes;
            if (remainingParts > 1) {
                bytes = (remainingBytes / remainingParts / RELAY_ALIGNMENT) * RELAY_ALIGNMENT;
            }
            slices.push_back(TransferSlice{offset, bytes});
            offset += bytes;
        }
        return slices;
    }

    uint32_t RelayPeerRank(const OpParam &param)
    {
        constexpr uint32_t RANKS_PER_SERVER = 8;
        if (param.myRank < RANKS_PER_SERVER && param.myRank < param.rankSize - RANKS_PER_SERVER) {
            return param.myRank + RANKS_PER_SERVER;
        }
        if (param.myRank >= RANKS_PER_SERVER) {
            return param.myRank - RANKS_PER_SERVER;
        }
        return INVALID_VALUE_RANKID;
    }

    std::array<uint64_t, 16> BuildTaskArguments(const OpParam &param, uint64_t inputAddress,
        uint64_t outputAddress, uint64_t token, uint64_t totalInputBytes, const TransferSlice &slice,
        bool relayEnabled, bool firstSlice, bool lastSlice, uint64_t phase)
    {
        const std::array<uint64_t, 4> groupCopySize = CalculateGroupCopySize(slice.bytes);
        const uint64_t outputOffset
            = static_cast<uint64_t>(param.myRank) * totalInputBytes + slice.offset;
        const uint64_t directBytes = relayEnabled
            ? (slice.bytes * RELAY_DIRECT_NUMERATOR / RELAY_DIRECT_DENOMINATOR / RELAY_ALIGNMENT) * RELAY_ALIGNMENT
            : slice.bytes;
        const uint64_t relayBytes = slice.bytes - directBytes;
        const uint32_t relayPeerRank = RelayPeerRank(param);
        const uint64_t relayOutputOffset = relayPeerRank == INVALID_VALUE_RANKID
            ? 0
            : static_cast<uint64_t>(relayPeerRank) * totalInputBytes + slice.offset;
        return {inputAddress, outputAddress, token, slice.offset, outputOffset, slice.bytes, groupCopySize[0],
            groupCopySize[1], groupCopySize[2], groupCopySize[3], lastSlice ? 1ULL : 0ULL,
            firstSlice ? 1ULL : 0ULL, relayOutputOffset, directBytes, relayBytes, phase};
    }

    HcclResult LaunchDirect(const OpParam &param, const AlgResourceCtx &resource,
        const std::vector<ThreadHandle> &threads, uint64_t inputAddress, uint64_t outputAddress, uint64_t token,
        uint64_t totalInputBytes)
    {
        const std::vector<TransferSlice> slices = BuildDirectSlices(totalInputBytes);
        CHK_PRT_RET(slices.size() > 1 && resource.continuationKernels.size() != resource.ccuKernels.size(),
            HCCL_ERROR("Missing CCU continuation kernels for multi-slice direct AllGather"), HCCL_E_INTERNAL);
        // 按分片分组：同一切片内 die0/die1 并行完成后再进入下一片
        for (size_t sliceIndex = 0; sliceIndex < slices.size(); ++sliceIndex) {
            if (threads.size() > 1) {
                CHK_RET(PreSynchronizeThreads(threads));
            }
            for (size_t kernelIndex = 0; kernelIndex < resource.ccuKernels.size(); ++kernelIndex) {
                const bool lastSlice = (sliceIndex + 1 == slices.size());
                const std::array<uint64_t, 16> taskArguments = BuildTaskArguments(param, inputAddress,
                    outputAddress, token, totalInputBytes, slices[sliceIndex], false, sliceIndex == 0,
                    lastSlice, DIRECT_PHASE);
                const CcuKernelHandle kernel = (slices.size() > 1 && sliceIndex > 0)
                    ? resource.continuationKernels[kernelIndex]
                    : resource.ccuKernels[kernelIndex];
                const CcuResult result = HcommCcuKernelLaunch(
                    threads[kernelIndex], kernel, taskArguments.data(), 11);
                CHK_PRT_RET(result != CCU_SUCCESS,
                    HCCL_ERROR("CCU AllGather kernel[%zu] launch failed[%d]", kernelIndex, result),
                    ConvertCcuToHccl(result));
            }
            if (threads.size() > 1) {
                CHK_RET(PostSynchronizeThreads(threads));
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult LaunchRelayPipeline(const OpParam &param, const AlgResourceCtx &resource,
        const std::vector<ThreadHandle> &threads, uint64_t inputAddress, uint64_t outputAddress, uint64_t token,
        uint64_t totalInputBytes)
    {
        const uint32_t localKernelIndex = resource.relayKernelIndex;
        const uint32_t crossKernelIndex = 1U - localKernelIndex;
        const std::vector<TransferSlice> slices = BuildRelaySlices(totalInputBytes);
        CHK_RET(PreSynchronizeThreads(threads));

        // Fill the local link pipeline first.  Relay tails are queued after all
        // local direct chunks so they overlap the independent cross-die queue.
        for (size_t sliceIndex = 0; sliceIndex < slices.size(); ++sliceIndex) {
            const bool lastSlice = (sliceIndex + 1 == slices.size());
            const std::array<uint64_t, 16> taskArguments = BuildTaskArguments(param, inputAddress,
                outputAddress, token, totalInputBytes, slices[sliceIndex], true, sliceIndex == 0,
                lastSlice, DIRECT_PHASE);
            const CcuResult result = HcommCcuKernelLaunch(threads[localKernelIndex],
                resource.relayKernels[localKernelIndex], taskArguments.data(), taskArguments.size());
            CHK_PRT_RET(result != CCU_SUCCESS,
                HCCL_ERROR("CCU AllGather local pipeline[%zu] launch failed[%d]", sliceIndex, result),
                ConvertCcuToHccl(result));
        }

        for (size_t sliceIndex = 0; sliceIndex < slices.size(); ++sliceIndex) {
            const bool lastSlice = (sliceIndex + 1 == slices.size());
            const std::array<uint64_t, 16> taskArguments = BuildTaskArguments(param, inputAddress,
                outputAddress, token, totalInputBytes, slices[sliceIndex], true, sliceIndex == 0,
                lastSlice, DIRECT_PHASE);
            const CcuResult result = HcommCcuKernelLaunch(threads[crossKernelIndex],
                resource.relayKernels[crossKernelIndex], taskArguments.data(), taskArguments.size());
            CHK_PRT_RET(result != CCU_SUCCESS,
                HCCL_ERROR("CCU AllGather cross pipeline[%zu] launch failed[%d]", sliceIndex, result),
                ConvertCcuToHccl(result));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[crossKernelIndex],
                threads[localKernelIndex], PIPELINE_NOTIFY_BASE + static_cast<uint32_t>(sliceIndex))));
        }

        for (size_t sliceIndex = 0; sliceIndex < slices.size(); ++sliceIndex) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[localKernelIndex],
                PIPELINE_NOTIFY_BASE + static_cast<uint32_t>(sliceIndex), CUSTOM_TIMEOUT)));
            const bool lastSlice = (sliceIndex + 1 == slices.size());
            const std::array<uint64_t, 16> taskArguments = BuildTaskArguments(param, inputAddress,
                outputAddress, token, totalInputBytes, slices[sliceIndex], true, false,
                lastSlice, RELAY_PHASE);
            const CcuResult result = HcommCcuKernelLaunch(threads[localKernelIndex],
                resource.relayKernels[localKernelIndex], taskArguments.data(), taskArguments.size());
            CHK_PRT_RET(result != CCU_SUCCESS,
                HCCL_ERROR("CCU AllGather relay pipeline[%zu] launch failed[%d]", sliceIndex, result),
                ConvertCcuToHccl(result));
        }
        CHK_RET(PostSynchronizeThreads(threads));
        return HCCL_SUCCESS;
    }

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0, HCCL_ERROR("Invalid AllGather resource context"),
        HCCL_E_INTERNAL);

    const char *context = static_cast<const char *>(param.resCtx);
    std::vector<char> serialized(context, context + param.ctxSize);
    AlgResourceCtx resource;
    resource.DeSerialize(serialized);

    const auto typeSize = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeSize == SIZE_TABLE.end() || typeSize->second == 0,
        HCCL_ERROR("Unsupported AllGather data type[%d]", static_cast<int>(param.dataType)), HCCL_E_NOT_SUPPORT);
    const uint64_t inputBytes = param.count * typeSize->second;

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, inputBytes));
    }
    CHK_PRT_RET(resource.ccuKernels.empty() || resource.ccuKernels.size() > 2
            || resource.threads.size() != resource.ccuKernels.size()
            || (!resource.continuationKernels.empty()
                && resource.continuationKernels.size() != resource.ccuKernels.size())
            || (!resource.relayKernels.empty() && resource.relayKernels.size() != resource.ccuKernels.size()),
        HCCL_ERROR("Invalid CCU kernel/thread resource count[%zu/%zu]", resource.ccuKernels.size(),
            resource.threads.size()),
        HCCL_E_INTERNAL);

    resource.threads[0] = param.cpuThread;

    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t memoryToken = 0;
    HcommCcuGetMemToken(inputBase, inputBytes, &memoryToken);

    const bool relayEnabled = (param.rankSize == 16 || param.rankSize == 12) && resource.ccuKernels.size() == 2
        && resource.relayKernels.size() == 2
        && resource.relayKernelIndex < 2 && inputBytes > RELAY_MIN_BYTES
        && inputBytes <= MAX_TRANSFER_BYTES * RELAY_PIPELINE_PARTS;
    if (relayEnabled) {
        return LaunchRelayPipeline(
            param, resource, resource.threads, inputBase, outputBase, memoryToken, inputBytes);
    }
    return LaunchDirect(param, resource, resource.threads, inputBase, outputBase, memoryToken, inputBytes);
}
} // namespace ops_hccl
