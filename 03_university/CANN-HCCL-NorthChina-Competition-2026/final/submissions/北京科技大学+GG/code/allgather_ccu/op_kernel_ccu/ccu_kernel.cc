/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

namespace ccu = ::AscendC::ccu;

namespace {

constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t CKE_INDEX = 0;
constexpr uint32_t POST_SYNC_ID = 3;

#define CCU_RETURN_IF_ERROR(call) \
    do { \
        CcuResult ccuResult = (call); \
        if (ccuResult != CCU_SUCCESS) { \
            return ccuResult; \
        } \
    } while (0)

uint16_t EventMask(uint32_t bitCount)
{
    if (bitCount >= 16) {
        return static_cast<uint16_t>(0xFFFFU);
    }
    return static_cast<uint16_t>((1U << bitCount) - 1U);
}

} // namespace

namespace ops_hccl {

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuAllGatherKernelArg *>(arg);
    const bool distributePhase =
        kernelArg != nullptr &&
        kernelArg->phase == static_cast<uint32_t>(AllGatherKernelPhase::DISTRIBUTE);
    const bool relayPhase =
        kernelArg != nullptr &&
        kernelArg->phase == static_cast<uint32_t>(AllGatherKernelPhase::RELAY);
    const bool directAlgorithm =
        kernelArg != nullptr &&
        kernelArg->algorithm == static_cast<uint32_t>(AllGatherAlgorithm::DIRECT);
    const bool mixedAlgorithm =
        kernelArg != nullptr &&
        kernelArg->algorithm == static_cast<uint32_t>(AllGatherAlgorithm::TWO_SERVER_EIGHT);
    if (kernelArg == nullptr || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE ||
        kernelArg->rankId >= kernelArg->rankSize || kernelArg->channelCount >= kernelArg->rankSize ||
        kernelArg->handleLocalCopy > 1 || (!directAlgorithm && !mixedAlgorithm) ||
        (!distributePhase && !relayPhase) ||
        (relayPhase && (!mixedAlgorithm || kernelArg->handleLocalCopy != 0))) {
        return CCU_E_PARA;
    }
    if (distributePhase) {
        for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
            if (kernelArg->transferModes[index] >
                static_cast<uint32_t>(AllGatherTransferMode::DIRECT_PREFIX)) {
                return CCU_E_PARA;
            }
        }
    }

    std::vector<ccu::Variable> peerOutput(kernelArg->channelCount);
    std::vector<ccu::Variable> peerOutputToken(kernelArg->channelCount);
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        peerOutput[index] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[index], OUTPUT_XN_ID);
        peerOutputToken[index] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[index], TOKEN_XN_ID);
    }

    ccu::Variable inputAddress;
    ccu::Variable outputAddress;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable selfOffset;
    ccu::Variable chunkSize;
    ccu::Variable localCopyFlag;
    ccu::Variable directSize;
    ccu::Variable relaySourceAddress;
    ccu::Variable relaySourceOffset;
    ccu::Variable relaySize;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputAddress, TASK_INPUT_ADDR));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputAddress, TASK_OUTPUT_ADDR));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, TASK_INPUT_TOKEN));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, TASK_OUTPUT_TOKEN));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(selfOffset, TASK_SELF_OFFSET));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunkSize, TASK_CHUNK_SIZE));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(localCopyFlag, TASK_LOCAL_COPY));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(directSize, TASK_DIRECT_SIZE));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(relaySourceAddress, TASK_RELAY_SOURCE_ADDR));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(relaySourceOffset, TASK_RELAY_SOURCE_OFFSET));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(relaySize, TASK_RELAY_SIZE));

    const uint32_t addressAndTokenBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[index], outputAddress,
            OUTPUT_XN_ID, CKE_INDEX, static_cast<uint16_t>(1U << OUTPUT_XN_ID)));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[index], outputToken,
            TOKEN_XN_ID, CKE_INDEX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[index], CKE_INDEX,
            static_cast<uint16_t>(addressAndTokenBits)));
    }

    if (distributePhase) {
        ccu::LocalAddr source;
        source.addr = inputAddress;
        source.token = inputToken;

        ccu::LocalAddr localDestination;
        localDestination.addr = outputAddress;
        localDestination.addr += selfOffset;
        localDestination.token = outputToken;

        ccu::Event completionEvent;
        for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
            ccu::RemoteAddr remoteDestination;
            remoteDestination.addr = peerOutput[index];
            remoteDestination.addr += selfOffset;
            remoteDestination.token = peerOutputToken[index];
            const uint16_t eventBit = static_cast<uint16_t>(1U << index);
            if (kernelArg->transferModes[index] ==
                static_cast<uint32_t>(AllGatherTransferMode::DIRECT_PREFIX)) {
                CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[index], remoteDestination,
                    source, directSize, completionEvent, eventBit));
            } else {
                CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[index], remoteDestination,
                    source, chunkSize, completionEvent, eventBit));
            }
        }

        uint32_t completionBitCount = kernelArg->channelCount;
        if (kernelArg->handleLocalCopy != 0) {
            const uint16_t localCopyBit = static_cast<uint16_t>(1U << completionBitCount);
            CCU_IF(localCopyFlag != 0) {
                CCU_RETURN_IF_ERROR(ccu::LocalCopy(
                    localDestination, source, chunkSize, completionEvent, localCopyBit));
            }
            CCU_ELSE {
                CCU_RETURN_IF_ERROR(ccu::EventRecord(completionEvent, localCopyBit));
            }
            ++completionBitCount;
        }

        if (completionBitCount != 0) {
            CCU_RETURN_IF_ERROR(
                ccu::EventWait(completionEvent, EventMask(completionBitCount)));
        }
    } else {
        ccu::LocalAddr relaySource;
        relaySource.addr = relaySourceAddress;
        relaySource.token = outputToken;

        ccu::Event completionEvent;
        for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
            ccu::RemoteAddr remoteDestination;
            remoteDestination.addr = peerOutput[index];
            remoteDestination.addr += relaySourceOffset;
            remoteDestination.token = peerOutputToken[index];
            const uint16_t eventBit = static_cast<uint16_t>(1U << index);
            CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[index], remoteDestination,
                relaySource, relaySize, completionEvent, eventBit));
        }
        if (kernelArg->channelCount != 0) {
            CCU_RETURN_IF_ERROR(
                ccu::EventWait(completionEvent, EventMask(kernelArg->channelCount)));
        }
    }

    const uint16_t postSyncBit = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyRecord(kernelArg->channels[index], CKE_INDEX, postSyncBit));
    }
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(kernelArg->channels[index], CKE_INDEX, postSyncBit));
    }

    return CCU_SUCCESS;
}

} // namespace ops_hccl
