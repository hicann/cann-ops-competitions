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
#include <cstring>
#include <limits>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t CCL_ALIGNMENT = 4096;
constexpr uint64_t MAX_CCU_TRANSFER_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t SMALL_INPUT_BYTES = 512ULL * 1024ULL;
constexpr uint64_t FANIN_MIN_INPUT_BYTES = 256ULL * 1024ULL;
constexpr uint32_t PIPELINE_NOTIFY = 0;
constexpr uint64_t TWO_BY_EIGHT_MESH_WINDOW = 4;

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value / alignment * alignment;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (value == 0) {
        return 0;
    }
    return (value - 1U) / alignment * alignment + alignment;
}

HcclResult GetToken(uint64_t address, uint64_t size, uint64_t &token)
{
    CHK_RET_CCU(HcommCcuGetMemToken(address, size, &token));
    return HCCL_SUCCESS;
}

HcclResult LaunchLayer(ThreadHandle thread, CcuKernelHandle kernel,
    uint64_t inputAddress, uint64_t inputToken, uint64_t outputAddress,
    uint64_t outputToken, uint64_t sliceOffset, uint64_t sliceSize,
    uint32_t sourceCount, bool initializeOutput, bool sequential)
{
    CHK_PRT_RET(sourceCount == 0 || sourceCount > MAX_LAYER_SOURCES,
        HCCL_ERROR("Invalid layer source count[%u]", sourceCount), HCCL_E_PARA);
    const uint64_t elementCount = sliceSize / sizeof(float);
    const uint64_t segmentBytes = sequential ? 0 :
        (elementCount / sourceCount) * sizeof(float);
    const uint64_t lastSegmentBytes =
        sliceSize - segmentBytes * (sourceCount - 1U);
    const std::array<uint64_t, CCU_TASK_ARG_COUNT> taskArgs = {
        CCU_MODE_GENERIC,
        inputAddress,
        inputToken,
        outputAddress,
        outputToken,
        0,
        0,
        sliceOffset,
        sliceSize,
        segmentBytes,
        lastSegmentBytes,
        initializeOutput ? 1U : 0U,
    };
    CHK_RET_CCU(HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
    return HCCL_SUCCESS;
}

HcclResult LaunchSmall(ThreadHandle thread, CcuKernelHandle kernel,
    uint64_t mode, uint64_t inputAddress, uint64_t inputToken,
    uint64_t outputAddress, uint64_t outputToken,
    uint64_t scratchAddress, uint64_t scratchToken,
    uint64_t sliceOffset, uint64_t sliceSize,
    uint64_t stageBytes, uint64_t finalOffset)
{
    const std::array<uint64_t, CCU_TASK_ARG_COUNT> taskArgs = {
        mode,
        inputAddress,
        inputToken,
        outputAddress,
        outputToken,
        scratchAddress,
        scratchToken,
        sliceOffset,
        sliceSize,
        stageBytes,
        finalOffset,
        0,
    };
    CHK_RET_CCU(HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
    return HCCL_SUCCESS;
}

HcclResult LaunchSmallFanin(ThreadHandle thread, CcuKernelHandle kernel,
    uint64_t inputAddress, uint64_t inputToken, uint64_t outputAddress,
    uint64_t outputToken, uint64_t scratchAddress, uint64_t scratchToken,
    uint64_t sliceOffset, uint64_t sliceSize, bool copyResult)
{
    const std::array<uint64_t, CCU_TASK_ARG_COUNT> taskArgs = {
        CCU_MODE_SMALL_FANIN,
        inputAddress,
        inputToken,
        outputAddress,
        outputToken,
        scratchAddress,
        scratchToken,
        sliceOffset,
        sliceSize,
        0,
        0,
        copyResult ? 1U : 0U,
    };
    CHK_RET_CCU(HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
    return HCCL_SUCCESS;
}

HcclResult LaunchLargeGrouped(ThreadHandle thread, CcuKernelHandle kernel,
    uint64_t mode, uint64_t inputAddress, uint64_t inputToken,
    uint64_t outputAddress,
    uint64_t outputToken, uint64_t scratchAddress, uint64_t scratchToken,
    uint64_t sliceOffset, uint64_t sliceSize, uint64_t partialStride,
    uint64_t firstStripeBytes, uint64_t secondStripeBytes)
{
    const std::array<uint64_t, CCU_TASK_ARG_COUNT> taskArgs = {
        mode,
        inputAddress,
        inputToken,
        outputAddress,
        outputToken,
        scratchAddress,
        scratchToken,
        sliceOffset,
        sliceSize,
        partialStride,
        firstStripeBytes,
        secondStripeBytes,
    };
    CHK_RET_CCU(HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
    return HCCL_SUCCESS;
}

HcclResult LaunchV23(ThreadHandle thread, CcuKernelHandle kernel,
    uint64_t mode, uint64_t inputAddress, uint64_t inputToken,
    uint64_t outputAddress, uint64_t outputToken,
    uint64_t scratchAddress, uint64_t scratchToken,
    uint64_t sliceOffset, uint64_t sliceSize, uint64_t partialStride,
    uint64_t firstHalfBytes, uint64_t secondHalfBytes,
    uint64_t regularStripeBytes, uint64_t lastStripeBytes,
    uint64_t remoteGroupOffset)
{
    const std::array<uint64_t, CCU_V23_TASK_ARG_COUNT> taskArgs = {
        mode,
        inputAddress,
        inputToken,
        outputAddress,
        outputToken,
        scratchAddress,
        scratchToken,
        sliceOffset,
        sliceSize,
        partialStride,
        firstHalfBytes,
        secondHalfBytes,
        regularStripeBytes,
        lastStripeBytes,
        remoteGroupOffset,
    };
    CHK_RET_CCU(HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
    return HCCL_SUCCESS;
}

HcclResult LaunchFinalize(ThreadHandle thread, CcuKernelHandle kernel,
    uint64_t remotePartial, uint64_t cclToken, uint64_t outputAddress,
    uint64_t outputToken, uint64_t dataSize)
{
    const std::array<uint64_t, 5> taskArgs = {
        remotePartial,
        cclToken,
        outputAddress,
        outputToken,
        dataSize,
    };
    CHK_RET_CCU(HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(ctx);
    CHK_PRT_RET(param.ctxSize != sizeof(AlgResourceCtx),
        HCCL_ERROR("Unexpected reusable context size[%llu]",
            static_cast<unsigned long long>(param.ctxSize)),
        HCCL_E_INTERNAL);
    AlgResourceCtx resCtx;
    (void)std::memcpy(&resCtx, ctx, sizeof(resCtx));

    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 ||
            param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("Unsupported ReduceScatter type or operation"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("recvCount overflow"), HCCL_E_PARA);
    const uint64_t outputBytes = param.count * sizeof(float);
    if (outputBytes == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.rankSize == 0 ||
            param.rankSize > std::numeric_limits<uint64_t>::max() / outputBytes,
        HCCL_ERROR("Input byte size overflow"), HCCL_E_PARA);
    const uint64_t inputBytes = outputBytes * param.rankSize;

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            param.cpuThread, param.outputPtr, param.inputPtr, outputBytes)));
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(resCtx.closKernelIndex >= resCtx.kernelCount,
        HCCL_ERROR("Missing layer-1 CCU kernel"), HCCL_E_INTERNAL);

    const uint64_t inputAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET(GetToken(inputAddress, inputBytes, inputToken));
    CHK_RET(GetToken(outputAddress, outputBytes, outputToken));

    const bool twoLayer = resCtx.meshKernelIndex < resCtx.kernelCount;
    const bool eightPlusFour =
        resCtx.smallTopology == SMALL_TOPO_EIGHT_PLUS_FOUR_LARGE ||
        resCtx.smallTopology == SMALL_TOPO_EIGHT_PLUS_FOUR_SMALL;
    // The 12-rank 512 KiB case is rounded up to one complete FP32 element
    // per rank: 524304 bytes.  Extend the small threshold only for this
    // topology.  Keeping 2x8 on the exact V7 threshold is intentional: the
    // global padded threshold used by V8 moved point 10 from 19 us to 37 us.
    const uint64_t smallPaddingBytes = eightPlusFour ?
        static_cast<uint64_t>(param.rankSize) * sizeof(float) : 0;
    const bool smallMessage =
        inputBytes <= SMALL_INPUT_BYTES + smallPaddingBytes;
    bool faninEligible = smallMessage &&
        inputBytes >= FANIN_MIN_INPUT_BYTES &&
        resCtx.localBuffer.size >= inputBytes;
    uint64_t cclBase = 0;
    uint64_t cclToken = 0;
    if (faninEligible) {
        cclBase = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
        CHK_RET(GetToken(cclBase, resCtx.localBuffer.size, cclToken));
    }

    if (!twoLayer) {
        if (faninEligible) {
            CHK_RET(LaunchSmallFanin(param.cpuThread,
                resCtx.ccuKernels[resCtx.closKernelIndex],
                inputAddress, inputToken, outputAddress, outputToken,
                cclBase, cclToken,
                param.myRank * outputBytes, outputBytes, true));
            return HCCL_SUCCESS;
        }
        const bool sequential = inputBytes <= SMALL_INPUT_BYTES;
        uint64_t processed = 0;
        while (processed < outputBytes) {
            const uint64_t chunkBytes =
                std::min(MAX_CCU_TRANSFER_BYTES, outputBytes - processed);
            CHK_PRT_RET(param.myRank >
                    (std::numeric_limits<uint64_t>::max() - processed) / outputBytes,
                HCCL_ERROR("Input slice offset overflow"), HCCL_E_PARA);
            const uint64_t inputSliceOffset =
                param.myRank * outputBytes + processed;
            CHK_RET(LaunchLayer(param.cpuThread,
                resCtx.ccuKernels[resCtx.closKernelIndex],
                inputAddress, inputToken, outputAddress + processed, outputToken,
                inputSliceOffset, chunkBytes, param.rankSize, true, sequential));
            processed += chunkBytes;
        }
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(resCtx.finalizeKernelIndex >= resCtx.kernelCount,
        HCCL_ERROR("Incomplete two-layer CCU resources"), HCCL_E_INTERNAL);

    CHK_PRT_RET(param.myRank >
            std::numeric_limits<uint64_t>::max() / outputBytes,
        HCCL_ERROR("Input slice offset overflow"), HCCL_E_PARA);
    const uint64_t rankSliceBase = param.myRank * outputBytes;

    if (faninEligible) {
        CHK_PRT_RET(resCtx.localRankSize == 0 ||
                resCtx.remoteRankSize == 0 ||
                resCtx.localRankSize >
                    std::numeric_limits<uint64_t>::max() / outputBytes,
            HCCL_ERROR("Invalid fan-in layer size"), HCCL_E_PARA);
        const uint64_t localBytes =
            static_cast<uint64_t>(resCtx.localRankSize) * outputBytes;
        CHK_PRT_RET(localBytes > resCtx.localBuffer.size ||
                cclBase > std::numeric_limits<uint64_t>::max() - localBytes,
            HCCL_ERROR("CCL buffer overflow for fan-in layers"), HCCL_E_MEMORY);
        const uint64_t remoteScratch = cclBase + localBytes;

        // CheckerV3 requires every acquired slave stream to start with a
        // local Wait.  Keep this V7 start edge even though the previous
        // invocation is already closed by the reverse completion edge.
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            param.cpuThread, resCtx.auxThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.auxThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchSmallFanin(param.cpuThread,
            resCtx.ccuKernels[resCtx.meshKernelIndex],
            inputAddress, inputToken, outputAddress, outputToken,
            cclBase, cclToken, rankSliceBase, outputBytes, true));
        CHK_RET(LaunchSmallFanin(resCtx.auxThread,
            resCtx.ccuKernels[resCtx.closKernelIndex],
            inputAddress, inputToken, remoteScratch, cclToken,
            remoteScratch, cclToken, rankSliceBase, outputBytes, false));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.auxThread, param.cpuThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            param.cpuThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchFinalize(param.cpuThread,
            resCtx.ccuKernels[resCtx.finalizeKernelIndex],
            remoteScratch, cclToken, outputAddress, outputToken, outputBytes));
        return HCCL_SUCCESS;
    }

    if (smallMessage && resCtx.smallTopology != SMALL_TOPO_DIRECT) {
        const uint64_t fallbackCclBase =
            reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
        uint64_t fallbackCclToken = 0;
        CHK_RET(GetToken(fallbackCclBase, resCtx.localBuffer.size,
            fallbackCclToken));

        if (resCtx.smallTopology == SMALL_TOPO_TWO_BY_EIGHT) {
            const uint64_t retainedBytes = inputBytes / 2U;
            CHK_PRT_RET(retainedBytes > resCtx.localBuffer.size,
                HCCL_ERROR("CCL buffer cannot hold the 2x8 retained half"),
                HCCL_E_MEMORY);
            const uint64_t retainedOffset =
                static_cast<uint64_t>(param.myRank / 8U) * retainedBytes;
            const uint64_t finalOffset =
                static_cast<uint64_t>(param.myRank % 8U) * outputBytes;
            const uint64_t meshStripeBytes =
                outputBytes >= TWO_BY_EIGHT_MESH_WINDOW * sizeof(float) ?
                AlignDown(outputBytes / TWO_BY_EIGHT_MESH_WINDOW,
                    sizeof(float)) : 0;
            const uint64_t meshLastStripeBytes = meshStripeBytes == 0 ? 0 :
                outputBytes - meshStripeBytes * (TWO_BY_EIGHT_MESH_WINDOW - 1U);

            // The two kernels are already serialized. Launching both on the
            // operation thread preserves that order without four cross-thread
            // notify tasks; the kernel resources still place each layer on its
            // own IO die.
            CHK_RET(LaunchSmall(param.cpuThread,
                resCtx.ccuKernels[resCtx.closKernelIndex],
                CCU_MODE_TWO_BY_EIGHT_CLOS,
                inputAddress, inputToken, outputAddress, outputToken,
                fallbackCclBase, fallbackCclToken, retainedOffset,
                retainedBytes, 0, 0));
            CHK_RET(LaunchSmall(param.cpuThread,
                resCtx.ccuKernels[resCtx.meshKernelIndex],
                CCU_MODE_TWO_BY_EIGHT_MESH,
                inputAddress, inputToken, outputAddress, outputToken,
                fallbackCclBase, fallbackCclToken, finalOffset, outputBytes,
                meshStripeBytes, meshLastStripeBytes));
            return HCCL_SUCCESS;
        }

        const uint64_t groupBytes = outputBytes * 4U;
        CHK_PRT_RET(groupBytes > resCtx.localBuffer.size,
            HCCL_ERROR("CCL buffer cannot hold the 8+4 retained group"),
            HCCL_E_MEMORY);
        const uint64_t retainedOffset =
            static_cast<uint64_t>(param.myRank / 4U) * groupBytes;

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            param.cpuThread, resCtx.auxThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.auxThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchSmall(resCtx.auxThread,
            resCtx.ccuKernels[resCtx.closKernelIndex], CCU_MODE_MIXED_CLOS,
            inputAddress, inputToken, outputAddress, outputToken,
            fallbackCclBase, fallbackCclToken, retainedOffset, groupBytes,
            0, 0));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.auxThread, param.cpuThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            param.cpuThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchSmall(param.cpuThread,
            resCtx.ccuKernels[resCtx.meshKernelIndex], CCU_MODE_MIXED_MESH,
            inputAddress, inputToken, outputAddress, outputToken,
            fallbackCclBase, fallbackCclToken, retainedOffset, outputBytes,
            groupBytes,
            static_cast<uint64_t>(param.myRank % 4U) * outputBytes));
        return HCCL_SUCCESS;
    }

    if (smallMessage) {
        // Keep one registered kernel per network layer. The local layer
        // initializes output, then the remote layer accumulates into it.
        CHK_RET(LaunchLayer(param.cpuThread,
            resCtx.ccuKernels[resCtx.meshKernelIndex], inputAddress, inputToken,
            outputAddress, outputToken, rankSliceBase, outputBytes,
            resCtx.localRankSize, true, true));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            param.cpuThread, resCtx.auxThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.auxThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchLayer(resCtx.auxThread,
            resCtx.ccuKernels[resCtx.closKernelIndex], inputAddress, inputToken,
            outputAddress, outputToken, rankSliceBase, outputBytes,
            param.rankSize - resCtx.localRankSize, false, true));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.auxThread, param.cpuThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            param.cpuThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    const uint64_t largeCclBase =
        reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    uint64_t largeCclToken = 0;
    CHK_RET(GetToken(largeCclBase, resCtx.localBuffer.size, largeCclToken));

    // V10 grouped partial fan-in. Keep the two network layers on their own
    // CCUs and launch them concurrently, but collapse 8/4 serialized network
    // waves into two striped waves inside each kernel. Each source pair writes
    // a private partial; a fixed local tree merges those partials. Unsupported
    // shapes and insufficient CCL capacity fall through to the V9 path.
    const bool groupedSourceCounts =
        (resCtx.localRankSize == 4U || resCtx.localRankSize == 8U) &&
        (resCtx.remoteRankSize == 4U || resCtx.remoteRankSize == 8U);
    if (resCtx.groupedLargeKernel == LARGE_ALGORITHM_V23_TWO_BY_EIGHT) {
        CHK_PRT_RET(resCtx.localRankSize != 8U || resCtx.remoteRankSize != 8U ||
                outputBytes > MAX_CCU_TRANSFER_BYTES,
            HCCL_ERROR("Invalid V23 2x8 context"), HCCL_E_INTERNAL);

        const uint64_t partialStride = AlignUp(outputBytes, CCL_ALIGNMENT);
        CHK_PRT_RET(partialStride == 0 ||
                partialStride > std::numeric_limits<uint64_t>::max() / 5U,
            HCCL_ERROR("V23 CCL layout overflow"), HCCL_E_PARA);
        const uint64_t requiredCclBytes = partialStride * 5U;
        CHK_PRT_RET(requiredCclBytes > resCtx.localBuffer.size,
            HCCL_ERROR("V23 CCL layout exceeds capacity"), HCCL_E_MEMORY);

        const uint64_t firstHalfBytes =
            AlignDown(outputBytes / 2U, sizeof(float));
        const uint64_t secondHalfBytes = outputBytes - firstHalfBytes;
        const uint64_t regularStripeBytes =
            AlignDown(outputBytes / 7U, sizeof(float));
        const uint64_t lastStripeBytes =
            outputBytes - regularStripeBytes * 6U;
        CHK_PRT_RET(firstHalfBytes == 0 || secondHalfBytes == 0 ||
                regularStripeBytes == 0 || lastStripeBytes == 0,
            HCCL_ERROR("V23 stripe layout is empty"), HCCL_E_PARA);

        const uint64_t meshScratch = largeCclBase;
        const uint64_t helperScratch = meshScratch + partialStride * 3U;
        const uint64_t directScratch = helperScratch + partialStride;
        const uint64_t remoteGroupOffset = param.myRank < 8U ?
            outputBytes * 8U : 0U;

        // Start the layer-1 thread once, then keep layer-0 and layer-1 active
        // concurrently for the complete first stage. The final helper receive
        // reuses the already registered layer-1 kernel handle.
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            param.cpuThread, resCtx.auxThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.auxThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));

        CHK_RET(LaunchV23(param.cpuThread,
            resCtx.ccuKernels[resCtx.meshKernelIndex],
            CCU_MODE_V23_MESH_HELPER,
            inputAddress, inputToken, outputAddress, outputToken,
            meshScratch, largeCclToken, rankSliceBase, outputBytes,
            partialStride, firstHalfBytes, secondHalfBytes,
            regularStripeBytes, lastStripeBytes, remoteGroupOffset));
        CHK_RET(LaunchV23(resCtx.auxThread,
            resCtx.ccuKernels[resCtx.closKernelIndex],
            CCU_MODE_V23_CLOS_DIRECT,
            inputAddress, inputToken, directScratch, largeCclToken,
            directScratch, largeCclToken, rankSliceBase, outputBytes,
            0, 0, 0, regularStripeBytes, lastStripeBytes, 0));

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.auxThread, param.cpuThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            param.cpuThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));

        CHK_RET(LaunchV23(param.cpuThread,
            resCtx.ccuKernels[resCtx.closKernelIndex],
            CCU_MODE_V23_CLOS_FINAL,
            helperScratch, largeCclToken, outputAddress, outputToken,
            directScratch, largeCclToken, 0, outputBytes,
            0, 0, 0, regularStripeBytes, lastStripeBytes, 0));
        return HCCL_SUCCESS;
    }

    if (resCtx.groupedLargeKernel == LARGE_ALGORITHM_V13_P17) {
        CHK_PRT_RET(!groupedSourceCounts || outputBytes > MAX_CCU_TRANSFER_BYTES,
            HCCL_ERROR("Invalid V13 point-17 context"), HCCL_E_INTERNAL);
        const uint64_t firstColourBytes =
            AlignDown(outputBytes / 2U, sizeof(float));
        const uint64_t secondColourBytes = outputBytes - firstColourBytes;
        CHK_PRT_RET(firstColourBytes < 4U * sizeof(float) ||
                secondColourBytes < 4U * sizeof(float),
            HCCL_ERROR("V13 point-17 colour is too small"), HCCL_E_PARA);
        const uint64_t partialStride = AlignUp(
            std::max(firstColourBytes, secondColourBytes), CCL_ALIGNMENT);
        const uint64_t meshGroupCount = resCtx.localRankSize / 4U;
        const uint64_t closGroupCount = resCtx.remoteRankSize / 4U;
        CHK_PRT_RET(partialStride == 0 ||
                meshGroupCount > std::numeric_limits<uint64_t>::max() / partialStride ||
                closGroupCount > std::numeric_limits<uint64_t>::max() / partialStride,
            HCCL_ERROR("V13 point-17 partial layout overflow"), HCCL_E_PARA);
        const uint64_t meshScratchBytes =
            (meshGroupCount - 1U) * partialStride;
        const uint64_t closScratchBytes =
            (closGroupCount - 1U) * partialStride;
        CHK_PRT_RET(meshScratchBytes > resCtx.localBuffer.size ||
                closScratchBytes > resCtx.localBuffer.size - meshScratchBytes,
            HCCL_ERROR("V13 point-17 CCL layout overflow"), HCCL_E_MEMORY);

        const uint64_t meshScratch = largeCclBase;
        const uint64_t closScratch = largeCclBase + meshScratchBytes;
        const uint64_t firstRegularStripeBytes =
            AlignDown(firstColourBytes / 4U, sizeof(float));
        const uint64_t firstLastStripeBytes = firstColourBytes -
            firstRegularStripeBytes * 3U;
        const uint64_t secondRegularStripeBytes =
            AlignDown(secondColourBytes / 4U, sizeof(float));
        const uint64_t secondLastStripeBytes = secondColourBytes -
            secondRegularStripeBytes * 3U;

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            param.cpuThread, resCtx.auxThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.auxThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));

        CHK_RET(LaunchLargeGrouped(param.cpuThread,
            resCtx.ccuKernels[resCtx.meshKernelIndex],
            CCU_MODE_LARGE_GROUPED4_INIT,
            inputAddress, inputToken, outputAddress, outputToken,
            meshScratch, largeCclToken, rankSliceBase, firstColourBytes,
            partialStride, firstRegularStripeBytes, firstLastStripeBytes));
        CHK_RET(LaunchLargeGrouped(resCtx.auxThread,
            resCtx.ccuKernels[resCtx.closKernelIndex],
            CCU_MODE_LARGE_GROUPED4_INIT,
            inputAddress, inputToken, outputAddress + firstColourBytes,
            outputToken, closScratch, largeCclToken,
            rankSliceBase + firstColourBytes, secondColourBytes,
            partialStride, secondRegularStripeBytes, secondLastStripeBytes));

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            param.cpuThread, resCtx.auxThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.auxThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.auxThread, param.cpuThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            param.cpuThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));

        CHK_RET(LaunchLargeGrouped(param.cpuThread,
            resCtx.ccuKernels[resCtx.meshKernelIndex],
            CCU_MODE_LARGE_GROUPED4_ACCUM,
            inputAddress, inputToken, outputAddress + firstColourBytes,
            outputToken, meshScratch, largeCclToken,
            rankSliceBase + firstColourBytes, secondColourBytes,
            partialStride, secondRegularStripeBytes, secondLastStripeBytes));
        CHK_RET(LaunchLargeGrouped(resCtx.auxThread,
            resCtx.ccuKernels[resCtx.closKernelIndex],
            CCU_MODE_LARGE_GROUPED4_ACCUM,
            inputAddress, inputToken, outputAddress, outputToken,
            closScratch, largeCclToken, rankSliceBase, firstColourBytes,
            partialStride, firstRegularStripeBytes, firstLastStripeBytes));

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.auxThread, param.cpuThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            param.cpuThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    if (resCtx.groupedLargeKernel == LARGE_ALGORITHM_V10 ||
        resCtx.groupedLargeKernel == LARGE_ALGORITHM_V12_P11) {
        CHK_PRT_RET(!groupedSourceCounts || outputBytes > MAX_CCU_TRANSFER_BYTES,
            HCCL_ERROR("Invalid grouped large-message context"), HCCL_E_INTERNAL);
        const uint64_t partialStride = AlignUp(outputBytes, CCL_ALIGNMENT);
        const uint64_t meshGroupCount = resCtx.localRankSize / 2U;
        const uint64_t closGroupCount = resCtx.remoteRankSize / 2U;
        CHK_PRT_RET(partialStride == 0 ||
                meshGroupCount > std::numeric_limits<uint64_t>::max() / partialStride ||
                closGroupCount > std::numeric_limits<uint64_t>::max() / partialStride,
            HCCL_ERROR("Grouped partial layout overflow"), HCCL_E_PARA);
        const uint64_t meshScratchBytes =
            (meshGroupCount - 1U) * partialStride;
        const uint64_t closBytes = closGroupCount * partialStride;
        if (meshScratchBytes <= resCtx.localBuffer.size &&
            closBytes <= resCtx.localBuffer.size - meshScratchBytes) {
            const uint64_t meshScratch = largeCclBase;
            const uint64_t remotePartial = largeCclBase + meshScratchBytes;
            const uint64_t closScratch = remotePartial + partialStride;
            const uint64_t firstStripeBytes =
                AlignDown(outputBytes / 2U, sizeof(float));
            const uint64_t secondStripeBytes = outputBytes - firstStripeBytes;

            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                param.cpuThread, resCtx.auxThread, PIPELINE_NOTIFY)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.auxThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));
            CHK_RET(LaunchLargeGrouped(param.cpuThread,
                resCtx.ccuKernels[resCtx.meshKernelIndex],
                CCU_MODE_LARGE_GROUPED,
                inputAddress, inputToken, outputAddress, outputToken,
                meshScratch, largeCclToken, rankSliceBase, outputBytes,
                partialStride, firstStripeBytes, secondStripeBytes));
            CHK_RET(LaunchLargeGrouped(resCtx.auxThread,
                resCtx.ccuKernels[resCtx.closKernelIndex],
                CCU_MODE_LARGE_GROUPED,
                inputAddress, inputToken, remotePartial, largeCclToken,
                closScratch, largeCclToken, rankSliceBase, outputBytes,
                partialStride, firstStripeBytes, secondStripeBytes));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.auxThread, param.cpuThread, PIPELINE_NOTIFY)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                param.cpuThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));
            CHK_RET(LaunchFinalize(param.cpuThread,
                resCtx.ccuKernels[resCtx.finalizeKernelIndex],
                remotePartial, largeCclToken, outputAddress, outputToken,
                outputBytes));
            return HCCL_SUCCESS;
        }
    }

    const uint64_t chunkCapacity =
        std::min(AlignDown(resCtx.localBuffer.size, CCL_ALIGNMENT),
            MAX_CCU_TRANSFER_BYTES);
    CHK_PRT_RET(chunkCapacity == 0,
        HCCL_ERROR("CCL buffer is too small for the layer-1 partial"),
        HCCL_E_INTERNAL);
    const uint64_t remotePartial = largeCclBase;

    // This is the Version 2 large-message resource graph: one kernel per
    // network layer plus one local finalize kernel. It is the online-proven
    // fallback while the hierarchical flow is redesigned to reuse kernels.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        param.cpuThread, resCtx.auxThread, PIPELINE_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.auxThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));

    uint64_t processed = 0;
    while (processed < outputBytes) {
        const uint64_t chunkBytes =
            std::min(chunkCapacity, outputBytes - processed);
        CHK_PRT_RET(param.myRank >
                (std::numeric_limits<uint64_t>::max() - processed) / outputBytes,
            HCCL_ERROR("Input slice offset overflow"), HCCL_E_PARA);
        const uint64_t inputSliceOffset =
            param.myRank * outputBytes + processed;

        CHK_RET(LaunchLayer(param.cpuThread,
            resCtx.ccuKernels[resCtx.meshKernelIndex], inputAddress, inputToken,
            outputAddress + processed, outputToken, inputSliceOffset, chunkBytes,
            resCtx.localRankSize, true, false));
        CHK_RET(LaunchLayer(resCtx.auxThread,
            resCtx.ccuKernels[resCtx.closKernelIndex], inputAddress, inputToken,
            remotePartial, largeCclToken, inputSliceOffset, chunkBytes,
            param.rankSize - resCtx.localRankSize, true, false));

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.auxThread, param.cpuThread, PIPELINE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            param.cpuThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));

        CHK_RET(LaunchFinalize(param.cpuThread,
            resCtx.ccuKernels[resCtx.finalizeKernelIndex],
            remotePartial, largeCclToken, outputAddress + processed, outputToken,
            chunkBytes));
        processed += chunkBytes;

        if (processed < outputBytes) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                param.cpuThread, resCtx.auxThread, PIPELINE_NOTIFY)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.auxThread, PIPELINE_NOTIFY, CUSTOM_TIMEOUT)));
        }
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
