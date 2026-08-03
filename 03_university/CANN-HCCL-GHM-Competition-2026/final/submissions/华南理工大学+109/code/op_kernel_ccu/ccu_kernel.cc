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

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

namespace {
    constexpr uint32_t kInputAddressVar = 0;
    constexpr uint32_t kInputTokenVar = 1;
    constexpr uint32_t kChannelNotify = 0;
    constexpr uint16_t kAddressAndTokenMask = (1U << kInputAddressVar) | (1U << kInputTokenVar);
    constexpr uint16_t kCompletionMask = 1U << 2;
} // namespace

#define KERNEL_CHK_RET(call) \
    do { \
        const CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

CcuResult CcuReduceScatterReadKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatter *>(arg);
    if (kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> peerInputs(kernelArg->channelCount);
    std::vector<ccu::Variable> peerTokens(kernelArg->channelCount);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        peerInputs[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kInputAddressVar);
        peerTokens[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kInputTokenVar);
    }

    ccu::Variable inputAddress;
    ccu::Variable inputToken;
    ccu::Variable outputAddress;
    ccu::Variable outputToken;
    ccu::Variable scratchAddress;
    ccu::Variable scratchToken;
    ccu::Variable inputOffset;
    ccu::Variable transferSize;
    ccu::Variable scratchStride;
    uint32_t argId = 0;
    KERNEL_CHK_RET(ccu::LoadArg(inputAddress, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(inputToken, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(outputAddress, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(outputToken, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(scratchAddress, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(scratchToken, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(inputOffset, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(transferSize, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(scratchStride, argId++));

    // 每个通道仅发布一次地址和 token，然后等待对端恰好一次匹配的发布，再发起读操作
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        KERNEL_CHK_RET(ccu::WriteVariableWithNotify(
            kernelArg->channels[i], inputAddress, kInputAddressVar, kChannelNotify, 1U << kInputAddressVar));
        KERNEL_CHK_RET(ccu::WriteVariableWithNotify(
            kernelArg->channels[i], inputToken, kInputTokenVar, kChannelNotify, 1U << kInputTokenVar));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        KERNEL_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kChannelNotify, kAddressAndTokenMask));
    }

    ccu::LocalAddr output;
    output.addr = outputAddress;
    output.token = outputToken;
    std::vector<ccu::LocalAddr> scratches(kernelArg->channelCount);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        scratches[i].addr = scratchAddress;
        for (uint32_t slot = 0; slot < i; ++slot) {
            scratches[i].addr += scratchStride;
        }
        scratches[i].token = scratchToken;
    }
    ccu::Event completion;
    uint16_t readCompletionMask = 0;
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        ccu::RemoteAddr remote;
        remote.addr = peerInputs[i];
        remote.addr += inputOffset;
        remote.token = peerTokens[i];
        const uint16_t peerMask = static_cast<uint16_t>(1U << i);
        readCompletionMask = static_cast<uint16_t>(readCompletionMask | peerMask);
        KERNEL_CHK_RET(ccu::Read(kernelArg->channels[i], scratches[i], remote, transferSize, completion, peerMask));
    }
    KERNEL_CHK_RET(ccu::EventWait(completion, readCompletionMask));

    // 在固定连续的归约树中对并发读取到的对端数据块进行归约
    uint32_t remaining = kernelArg->channelCount;
    while (remaining > 1) {
        const uint32_t reducePairs = remaining / 2;
        const uint32_t sourceSlot = remaining - reducePairs;
        ccu::Variable reduceLength;
        reduceLength = transferSize;
        for (uint32_t pair = 1; pair < reducePairs; ++pair) {
            reduceLength += transferSize;
        }
        KERNEL_CHK_RET(ccu::LocalReduce(
            scratches[0], scratches[sourceSlot], reduceLength, kernelArg->dataType, kernelArg->reduceOp, completion));
        KERNEL_CHK_RET(ccu::EventWait(completion));
        remaining -= reducePairs;
    }

    if (kernelArg->initializeOutput) {
        ccu::LocalAddr localInput;
        localInput.addr = inputAddress;
        localInput.addr += inputOffset;
        localInput.token = inputToken;
        KERNEL_CHK_RET(ccu::LocalCopy(output, localInput, transferSize, completion));
        KERNEL_CHK_RET(ccu::EventWait(completion));
        KERNEL_CHK_RET(
            ccu::LocalReduce(output, scratches[0], transferSize, kernelArg->dataType, kernelArg->reduceOp, completion));
        KERNEL_CHK_RET(ccu::EventWait(completion));
    }

    // 非初始化层将其确定性的部分结果留在 slot 0 中，
    // 等待两个 CCU 都完成后由主线程进行合并

    // 完成握手可以防止后续分块在对端消费完当前分块的记录之前，
    // 在同一通道 notify 上发起新的记录
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        KERNEL_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], kChannelNotify, kCompletionMask));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        KERNEL_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kChannelNotify, kCompletionMask));
    }

    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterStripedKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatter *>(arg);
    if (kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> peerInputs(kernelArg->channelCount);
    std::vector<ccu::Variable> peerTokens(kernelArg->channelCount);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        peerInputs[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kInputAddressVar);
        peerTokens[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kInputTokenVar);
    }

    ccu::Variable inputAddress;
    ccu::Variable inputToken;
    ccu::Variable outputAddress;
    ccu::Variable outputToken;
    ccu::Variable partialAddress;
    ccu::Variable partialToken;
    ccu::Variable inputOffset;
    ccu::Variable transferSize;
    ccu::Variable stripeSize;
    ccu::Variable lastStripeSize;
    uint32_t argId = 0;
    KERNEL_CHK_RET(ccu::LoadArg(inputAddress, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(inputToken, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(outputAddress, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(outputToken, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(partialAddress, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(partialToken, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(inputOffset, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(transferSize, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(stripeSize, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(lastStripeSize, argId++));

    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        KERNEL_CHK_RET(ccu::WriteVariableWithNotify(
            kernelArg->channels[i], inputAddress, kInputAddressVar, kChannelNotify, 1U << kInputAddressVar));
        KERNEL_CHK_RET(ccu::WriteVariableWithNotify(
            kernelArg->channels[i], inputToken, kInputTokenVar, kChannelNotify, 1U << kInputTokenVar));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        KERNEL_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kChannelNotify, kAddressAndTokenMask));
    }

    ccu::Event completion;
    ccu::LocalAddr accumulator;
    accumulator.addr = kernelArg->initializeOutput ? outputAddress : partialAddress;
    accumulator.token = kernelArg->initializeOutput ? outputToken : partialToken;

    const uint32_t stripeCount = kernelArg->channelCount;
    std::vector<ccu::Variable> stripeOffsets(stripeCount);
    std::vector<ccu::Variable> stripeLengths(stripeCount);
    std::vector<ccu::LocalAddr> localStripes(stripeCount);
    std::vector<ccu::RemoteAddr> remoteStripes(stripeCount);
    ccu::Variable stripeOffset;
    stripeOffset = 0;
    for (uint32_t stripe = 0; stripe < stripeCount; ++stripe) {
        stripeOffsets[stripe] = stripeOffset;
        stripeLengths[stripe] = stripe + 1 == stripeCount ? lastStripeSize : stripeSize;
        stripeOffset += stripeSize;
    }

    if (kernelArg->initializeOutput) {
        ccu::LocalAddr localInput;
        localInput.addr = inputAddress;
        localInput.addr += inputOffset;
        localInput.token = inputToken;
        KERNEL_CHK_RET(ccu::LocalCopy(accumulator, localInput, transferSize, completion));
        KERNEL_CHK_RET(ccu::EventWait(completion));
    } else {
        uint16_t initializeMask = 0;
        for (uint32_t i = 0; i < stripeCount; ++i) {
            localStripes[i].addr = accumulator.addr;
            localStripes[i].addr += stripeOffsets[i];
            localStripes[i].token = accumulator.token;
            remoteStripes[i].addr = peerInputs[i];
            remoteStripes[i].addr += inputOffset;
            remoteStripes[i].addr += stripeOffsets[i];
            remoteStripes[i].token = peerTokens[i];
            const uint16_t mask = static_cast<uint16_t>(1U << i);
            initializeMask = static_cast<uint16_t>(initializeMask | mask);
            KERNEL_CHK_RET(ccu::Read(
                kernelArg->channels[i], localStripes[i], remoteStripes[i], stripeLengths[i], completion, mask));
        }
        KERNEL_CHK_RET(ccu::EventWait(completion, initializeMask));
    }

    // 每一步都针对互不重叠的条带。仅远端的 partial 从对端 i 初始化条带 i，
    // 因此从第 1 步开始，且不会重复计算该对端。负责初始化输出的层从本地输入开始，
    // 在步骤 [0, N) 内依次访问所有远端对端
    const uint32_t firstStep = kernelArg->initializeOutput ? 0U : 1U;
    for (uint32_t step = firstStep; step < stripeCount; ++step) {
        uint16_t completionMask = 0;
        for (uint32_t i = 0; i < stripeCount; ++i) {
            const uint32_t stripeIndex = (i + step) % stripeCount;
            localStripes[i].addr = accumulator.addr;
            localStripes[i].addr += stripeOffsets[stripeIndex];
            localStripes[i].token = accumulator.token;
            remoteStripes[i].addr = peerInputs[i];
            remoteStripes[i].addr += inputOffset;
            remoteStripes[i].addr += stripeOffsets[stripeIndex];
            remoteStripes[i].token = peerTokens[i];
            const uint16_t mask = static_cast<uint16_t>(1U << i);
            completionMask = static_cast<uint16_t>(completionMask | mask);
            KERNEL_CHK_RET(ccu::ReadReduce(kernelArg->channels[i], localStripes[i], remoteStripes[i],
                stripeLengths[stripeIndex], kernelArg->dataType, kernelArg->reduceOp, completion, mask));
        }
        KERNEL_CHK_RET(ccu::EventWait(completion, completionMask));
    }

    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        KERNEL_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], kChannelNotify, kCompletionMask));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        KERNEL_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kChannelNotify, kCompletionMask));
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterMergeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterMerge *>(arg);
    if (kernelArg->channelCount != 1) {
        return CCU_E_PARA;
    }

    // 将地址翻译锚定到与第一个网络层 kernel 相同的 IO Die 上。
    // 合并操作本身仅访问本地内存
    const auto deploymentAnchor = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[0], kInputAddressVar);
    (void)deploymentAnchor;

    ccu::Variable outputAddress;
    ccu::Variable outputToken;
    ccu::Variable partialAddress;
    ccu::Variable partialToken;
    ccu::Variable transferSize;
    uint32_t argId = 0;
    KERNEL_CHK_RET(ccu::LoadArg(outputAddress, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(outputToken, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(partialAddress, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(partialToken, argId++));
    KERNEL_CHK_RET(ccu::LoadArg(transferSize, argId++));

    ccu::LocalAddr output;
    output.addr = outputAddress;
    output.token = outputToken;
    ccu::LocalAddr partial;
    partial.addr = partialAddress;
    partial.token = partialToken;
    ccu::Event completion;
    KERNEL_CHK_RET(
        ccu::LocalReduce(output, partial, transferSize, kernelArg->dataType, kernelArg->reduceOp, completion));
    KERNEL_CHK_RET(ccu::EventWait(completion));
    return CCU_SUCCESS;
}

} // namespace ops_hccl

#undef KERNEL_CHK_RET
