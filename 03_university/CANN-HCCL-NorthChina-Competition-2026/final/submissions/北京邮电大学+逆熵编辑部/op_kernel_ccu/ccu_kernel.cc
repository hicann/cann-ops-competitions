/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hcomm/hcomm_primitives.h>

#include <ccu/ccu_control_flow_macro.h>

#include <vector>

#include "ccu_kernel.h"

namespace ops_hccl {

namespace ccu = ::AscendC::ccu;

#define CCU_RETURN_IF_ERROR(call) \
    do { \
        const CcuResult ccuResult = static_cast<CcuResult>(call); \
        if (ccuResult != CCU_SUCCESS) { \
            return ccuResult; \
        } \
    } while (0)

namespace {

constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t OUTPUT_TOKEN_XN_ID = 2;
constexpr uint32_t OUTPUT_NOTIFY_INDEX = 0;
constexpr uint32_t OUTPUT_TOKEN_NOTIFY_INDEX = 1;
constexpr uint32_t POST_SYNC_NOTIFY_INDEX = 2;
constexpr uint32_t COMPLETION_ACK_NOTIFY_INDEX = 3;
constexpr uint32_t TASK_ARGUMENT_COUNT = 7;
constexpr uint32_t SMALL_PULL_TASK_ARGUMENT_COUNT = 5;
constexpr uint32_t LARGE_PULL_TASK_ARGUMENT_COUNT = 6;
constexpr uint64_t PUBLISH_STAGE = 0;
constexpr uint64_t WRITE_RECORD_STAGE = 1;
constexpr uint64_t FINAL_WAIT_STAGE = 2;
constexpr uint64_t PAIR_PUBLISH_STAGE = 3;
constexpr uint64_t PAIR_WRITE_RECORD_STAGE = 4;
constexpr uint64_t PAIR_FINAL_WAIT_STAGE = 5;
constexpr uint64_t ACK_FINAL_WAIT_STAGE = 6;
constexpr uint64_t FUSED_WRITE_WAIT_STAGE = 7;
constexpr uint64_t MONOLITHIC_STAGE = 8;
constexpr uint64_t RELAY_PUBLISH_STAGE = 9;
constexpr uint64_t RELAY_WRITE_RECORD_STAGE = 10;
constexpr uint64_t RELAY_FINAL_WAIT_STAGE = 11;
constexpr uint64_t RELAY_ACK_FINAL_WAIT_STAGE = 12;
constexpr uint64_t ASYMMETRIC_PAIR_ROUND0_STAGE = 13;
constexpr uint64_t ASYMMETRIC_PAIR_ROUND1_STAGE = 14;
constexpr uint64_t ASYMMETRIC_NONPAIR_STAGE = 15;
constexpr uint64_t ASYMMETRIC_LOCAL_WRITE_STAGE = 16;
constexpr uint64_t ASYMMETRIC_RELAY_WRITE_STAGE = 17;
constexpr uint64_t ASYMMETRIC_RELAY_WRITE_RECORD_STAGE = 18;
constexpr uint64_t ASYMMETRIC_RELAY_READ_ROUND0_STAGE = 19;
constexpr uint64_t ASYMMETRIC_RELAY_READ_ROUND1_RECORD_STAGE = 20;
constexpr uint64_t ASYMMETRIC_FUSED_PULL_RELAY_STAGE = 21;
constexpr uint64_t ASYMMETRIC_FUSED_THIRD_WRITE_PULL_RELAY_STAGE = 22;
constexpr uint64_t ASYMMETRIC_SEGMENTED_DIRECT_WRITE_RECORD_STAGE = 23;
constexpr uint64_t ASYMMETRIC_FUSED_QUARTER_WRITE_PULL_RELAY_STAGE = 24;

struct TaskVariables {
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable remoteOutputOffset;
    ccu::Variable sliceSize;
    ccu::Variable stage;
};

struct SmallPullTaskVariables {
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable sliceSize;
};

struct LargePullTaskVariables {
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankStride;
    ccu::Variable transferSize;
};

CcuResult ValidateKernelArg(CcuKernelArg arg, uint32_t expectedLayer, CcuKernelArgDirect *&kernelArg)
{
    kernelArg = static_cast<CcuKernelArgDirect *>(arg);
    if (kernelArg == nullptr || kernelArg->layerId != expectedLayer ||
        kernelArg->localRank >= MAX_RANK_SIZE ||
        kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE ||
        kernelArg->pairChannelCount > kernelArg->channelCount ||
        kernelArg->relayChannelCount > kernelArg->channelCount ||
        (kernelArg->asymmetricPairRound0Channel != MAX_RANK_SIZE &&
         kernelArg->asymmetricPairRound0Channel >= kernelArg->channelCount) ||
        (kernelArg->asymmetricPairRound1Channel != MAX_RANK_SIZE &&
         kernelArg->asymmetricPairRound1Channel >= kernelArg->channelCount) ||
        (kernelArg->pairChannelCount != 0U &&
         (expectedLayer != 1U || kernelArg->relayChannelCount != 0U)) ||
        (kernelArg->relayChannelCount != 0U &&
         (expectedLayer != 0U || kernelArg->relaySourceCount == 0U ||
          kernelArg->relaySourceCount > 2U))) {
        return CCU_E_PARA;
    }
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->relayChannelCount; ++channelIndex) {
        if (kernelArg->relaySourceRanks0[channelIndex] >= MAX_RANK_SIZE ||
            (kernelArg->relaySourceCount == 2U &&
             kernelArg->relaySourceRanks1[channelIndex] >= MAX_RANK_SIZE)) {
            return CCU_E_PARA;
        }
    }
    return CCU_SUCCESS;
}

inline CcuResult RunSendSingleChannel(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars, uint32_t channelIndex)
{
    if (channelIndex >= kernelArg->channelCount) {
        return CCU_E_PARA;
    }
    const ChannelHandle channel = kernelArg->channels[channelIndex];
    ccu::Variable remoteOutput =
        ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_XN_ID);
    ccu::Variable remoteOutputToken =
        ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_XN_ID);
    CCU_RETURN_IF_ERROR(ccu::NotifyWait(channel, OUTPUT_NOTIFY_INDEX, 1U));
    CCU_RETURN_IF_ERROR(ccu::NotifyWait(channel, OUTPUT_TOKEN_NOTIFY_INDEX, 1U));

    ccu::LocalAddr source;
    source.addr = vars.input;
    source.token = vars.inputToken;
    ccu::RemoteAddr destination;
    destination.addr = remoteOutput;
    destination.addr += vars.remoteOutputOffset;
    destination.token = remoteOutputToken;
    ccu::Event completion;
    CCU_RETURN_IF_ERROR(ccu::Write(
        channel, destination, source, vars.sliceSize, completion, 1U));
    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, 1U));
    CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
        channel, POST_SYNC_NOTIFY_INDEX, 1U));
    return CCU_SUCCESS;
}

inline CcuResult RunSymmetricBidirectionalSingleChannel(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars, uint32_t channelIndex)
{
    if (channelIndex >= kernelArg->channelCount) {
        return CCU_E_PARA;
    }
    const ChannelHandle channel = kernelArg->channels[channelIndex];
    CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
        channel, vars.output, OUTPUT_XN_ID, OUTPUT_NOTIFY_INDEX, 1U));
    CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
        channel, vars.outputToken, OUTPUT_TOKEN_XN_ID,
        OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
    CCU_RETURN_IF_ERROR(RunSendSingleChannel(
        kernelArg, vars, channelIndex));
    CCU_RETURN_IF_ERROR(ccu::NotifyWait(
        channel, POST_SYNC_NOTIFY_INDEX, 1U));
    CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
        channel, COMPLETION_ACK_NOTIFY_INDEX, 1U));
    CCU_RETURN_IF_ERROR(ccu::NotifyWait(
        channel, COMPLETION_ACK_NOTIFY_INDEX, 1U));
    return CCU_SUCCESS;
}

inline CcuResult RunMonolithicChannelSuffix(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars)
{
    const uint32_t firstChannel = kernelArg->pairChannelCount;
    const uint32_t selectedCount = kernelArg->channelCount - firstChannel;
    if (firstChannel == 0U || selectedCount == 0U) {
        return CCU_E_PARA;
    }
    std::vector<ccu::Variable> remoteOutput(selectedCount);
    std::vector<ccu::Variable> remoteOutputToken(selectedCount);
    for (uint32_t selectedIndex = 0; selectedIndex < selectedCount; ++selectedIndex) {
        const uint32_t channelIndex = firstChannel + selectedIndex;
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.output, OUTPUT_XN_ID, OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.outputToken, OUTPUT_TOKEN_XN_ID, OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
        remoteOutput[selectedIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_XN_ID);
        remoteOutputToken[selectedIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_XN_ID);
    }
    for (uint32_t selectedIndex = 0; selectedIndex < selectedCount; ++selectedIndex) {
        const ChannelHandle channel =
            kernelArg->channels[firstChannel + selectedIndex];
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            channel, OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            channel, OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
    }

    ccu::LocalAddr source;
    source.addr = vars.input;
    source.token = vars.inputToken;
    ccu::Event completion;
    for (uint32_t selectedIndex = 0; selectedIndex < selectedCount; ++selectedIndex) {
        const ChannelHandle channel =
            kernelArg->channels[firstChannel + selectedIndex];
        ccu::RemoteAddr destination;
        destination.addr = remoteOutput[selectedIndex];
        destination.addr += vars.remoteOutputOffset;
        destination.token = remoteOutputToken[selectedIndex];
        const uint16_t completionMask =
            static_cast<uint16_t>(1U << selectedIndex);
        CCU_RETURN_IF_ERROR(ccu::Write(
            channel, destination, source, vars.sliceSize,
            completion, completionMask));
    }
    const uint16_t allCompletionMask =
        static_cast<uint16_t>((1U << selectedCount) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, allCompletionMask));
    for (uint32_t selectedIndex = 0; selectedIndex < selectedCount; ++selectedIndex) {
        const ChannelHandle channel =
            kernelArg->channels[firstChannel + selectedIndex];
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            channel, POST_SYNC_NOTIFY_INDEX, 1U));
    }
    for (uint32_t selectedIndex = 0; selectedIndex < selectedCount; ++selectedIndex) {
        const ChannelHandle channel =
            kernelArg->channels[firstChannel + selectedIndex];
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            channel, POST_SYNC_NOTIFY_INDEX, 1U));
    }
    return CCU_SUCCESS;
}

inline CcuResult RunAsymmetricLocalWrite(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars)
{
    std::vector<ccu::Variable> remoteOutput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteOutputToken(kernelArg->channelCount);
    for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        remoteOutput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_XN_ID);
        remoteOutputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_XN_ID);
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            channel, OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            channel, OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
    }

    ccu::LocalAddr source;
    source.addr = vars.input;
    source.token = vars.inputToken;
    ccu::Event completion;
    for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
        ccu::RemoteAddr destination;
        destination.addr = remoteOutput[channelIndex];
        destination.addr += vars.remoteOutputOffset;
        destination.token = remoteOutputToken[channelIndex];
        const uint16_t completionMask =
            static_cast<uint16_t>(1U << channelIndex);
        CCU_RETURN_IF_ERROR(ccu::Write(
            kernelArg->channels[channelIndex], destination, source,
            vars.sliceSize, completion, completionMask));
    }
    const uint16_t allCompletionMask =
        static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, allCompletionMask));

    // Relay-prefix Channels remain open until their final relay Write.
    for (uint32_t channelIndex = kernelArg->relayChannelCount;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
    }
    return CCU_SUCCESS;
}

inline CcuResult RunAsymmetricRelayWrite(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars, bool recordCompletion)
{
    if (kernelArg->relayChannelCount == 0U) {
        return CCU_E_PARA;
    }
    ccu::LocalAddr source;
    source.addr = vars.input;
    source.token = vars.inputToken;
    ccu::Event completion;
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->relayChannelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        ccu::RemoteAddr destination;
        destination.addr =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_XN_ID);
        destination.addr += vars.remoteOutputOffset;
        destination.token =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_XN_ID);
        const uint16_t completionMask =
            static_cast<uint16_t>(1U << channelIndex);
        CCU_RETURN_IF_ERROR(ccu::Write(
            channel, destination, source, vars.sliceSize,
            completion, completionMask));
    }
    const uint16_t allCompletionMask =
        static_cast<uint16_t>((1U << kernelArg->relayChannelCount) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, allCompletionMask));
    if (recordCompletion) {
        for (uint32_t channelIndex = 0;
            channelIndex < kernelArg->relayChannelCount; ++channelIndex) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
                kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
        }
    }
    return CCU_SUCCESS;
}

inline CcuResult RunAsymmetricRelayRead(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars,
    uint32_t sourceIndex, bool recordCompletion)
{
    if (kernelArg->relayChannelCount == 0U ||
        kernelArg->relaySourceCount == 0U ||
        kernelArg->relaySourceCount > 2U ||
        sourceIndex >= kernelArg->relaySourceCount) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> remoteOutput(kernelArg->relayChannelCount);
    std::vector<ccu::Variable> remoteOutputToken(kernelArg->relayChannelCount);
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->relayChannelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        remoteOutput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_XN_ID);
        remoteOutputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_XN_ID);
    }

    ccu::Event completion;
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->relayChannelCount; ++channelIndex) {
        const uint32_t sourceRank = sourceIndex == 0U ?
            kernelArg->relaySourceRanks0[channelIndex] :
            kernelArg->relaySourceRanks1[channelIndex];
        ccu::Variable sourceOffset;
        sourceOffset = 0U;
        for (uint32_t rankIndex = 0; rankIndex < sourceRank; ++rankIndex) {
            sourceOffset += vars.remoteOutputOffset;
        }

        ccu::RemoteAddr source;
        source.addr = remoteOutput[channelIndex];
        source.addr += sourceOffset;
        source.addr += vars.output;
        source.token = remoteOutputToken[channelIndex];

        ccu::LocalAddr destination;
        destination.addr = vars.input;
        destination.addr += sourceOffset;
        destination.token = vars.inputToken;

        const uint16_t completionMask =
            static_cast<uint16_t>(1U << channelIndex);
        CCU_RETURN_IF_ERROR(ccu::Read(
            kernelArg->channels[channelIndex], destination, source,
            vars.sliceSize, completion, completionMask));
    }

    const uint16_t allCompletionMask =
        static_cast<uint16_t>((1U << kernelArg->relayChannelCount) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, allCompletionMask));
    if (recordCompletion) {
        for (uint32_t channelIndex = 0;
            channelIndex < kernelArg->relayChannelCount; ++channelIndex) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
                kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
        }
    }
    return CCU_SUCCESS;
}

inline CcuResult RunAsymmetricFusedPullRelay(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars)
{
    if (kernelArg->relayChannelCount == 0U ||
        kernelArg->relaySourceCount == 0U ||
        kernelArg->relaySourceCount > 2U) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> remoteOutput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteOutputToken(kernelArg->channelCount);
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        remoteOutput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_XN_ID);
        remoteOutputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_XN_ID);
    }
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
    }

    ccu::Variable localOwnerOffset;
    localOwnerOffset = 0U;
    for (uint32_t rankIndex = 0; rankIndex < kernelArg->localRank; ++rankIndex) {
        localOwnerOffset += vars.remoteOutputOffset;
    }

    ccu::LocalAddr localSuffix;
    localSuffix.addr = vars.input;
    localSuffix.token = vars.inputToken;
    ccu::Event writeCompletion;
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        ccu::RemoteAddr destination;
        destination.addr = remoteOutput[channelIndex];
        destination.addr += localOwnerOffset;
        destination.token = remoteOutputToken[channelIndex];
        const uint16_t eventBit =
            static_cast<uint16_t>(1U << channelIndex);
        CCU_RETURN_IF_ERROR(ccu::Write(
            kernelArg->channels[channelIndex], destination, localSuffix,
            vars.sliceSize, writeCompletion, eventBit));
    }
    const uint16_t allWriteBits =
        static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(writeCompletion, allWriteBits));

    // The 8-rank side pulls one paired suffix per relay Channel; the 4-rank
    // side pulls two. A separate Event per round keeps one operation per
    // Channel in flight while retaining the same XN resources.
    for (uint32_t sourceIndex = 0;
        sourceIndex < kernelArg->relaySourceCount; ++sourceIndex) {
        ccu::Event readCompletion;
        for (uint32_t channelIndex = 0;
            channelIndex < kernelArg->relayChannelCount; ++channelIndex) {
            const uint32_t sourceRank = sourceIndex == 0U ?
                kernelArg->relaySourceRanks0[channelIndex] :
                kernelArg->relaySourceRanks1[channelIndex];
            ccu::Variable sourceOffset;
            sourceOffset = 0U;
            for (uint32_t rankIndex = 0; rankIndex < sourceRank; ++rankIndex) {
                sourceOffset += vars.remoteOutputOffset;
            }

            ccu::RemoteAddr source;
            source.addr = remoteOutput[channelIndex];
            source.addr += sourceOffset;
            source.token = remoteOutputToken[channelIndex];
            ccu::LocalAddr destination;
            destination.addr = vars.output;
            destination.addr += sourceOffset;
            destination.token = vars.outputToken;
            const uint16_t eventBit =
                static_cast<uint16_t>(1U << channelIndex);
            CCU_RETURN_IF_ERROR(ccu::Read(
                kernelArg->channels[channelIndex], destination, source,
                vars.sliceSize, readCompletion, eventBit));
        }
        const uint16_t allReadBits =
            static_cast<uint16_t>((1U << kernelArg->relayChannelCount) - 1U);
        CCU_RETURN_IF_ERROR(ccu::EventWait(readCompletion, allReadBits));
    }

    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
    }
    return CCU_SUCCESS;
}

inline CcuResult RunAsymmetricFusedThirdWritePullRelay(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars)
{
    if (kernelArg->relayChannelCount == 0U ||
        kernelArg->relaySourceCount == 0U ||
        kernelArg->relaySourceCount > 2U) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> remoteOutput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteOutputToken(kernelArg->channelCount);
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        remoteOutput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_XN_ID);
        remoteOutputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_XN_ID);
    }
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
    }

    ccu::Variable directSize;
    directSize = vars.remoteOutputOffset;
    directSize += vars.remoteOutputOffset;
    ccu::Variable fullSize;
    fullSize = directSize;
    fullSize += vars.sliceSize;

    ccu::Variable localOwnerOffset;
    localOwnerOffset = 0U;
    for (uint32_t rankIndex = 0; rankIndex < kernelArg->localRank; ++rankIndex) {
        localOwnerOffset += fullSize;
    }

    ccu::Variable segmentOffset;
    segmentOffset = 0U;
    for (uint32_t segmentIndex = 0; segmentIndex < 2U; ++segmentIndex) {
        ccu::LocalAddr localSegment;
        localSegment.addr = vars.input;
        localSegment.addr += segmentOffset;
        localSegment.token = vars.inputToken;
        ccu::Event writeCompletion;
        for (uint32_t channelIndex = 0;
            channelIndex < kernelArg->channelCount; ++channelIndex) {
            ccu::RemoteAddr destination;
            destination.addr = remoteOutput[channelIndex];
            destination.addr += localOwnerOffset;
            destination.addr += segmentOffset;
            destination.token = remoteOutputToken[channelIndex];
            const uint16_t eventBit =
                static_cast<uint16_t>(1U << channelIndex);
            CCU_RETURN_IF_ERROR(ccu::Write(
                kernelArg->channels[channelIndex], destination, localSegment,
                vars.remoteOutputOffset, writeCompletion, eventBit));
        }
        const uint16_t allWriteBits =
            static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U);
        CCU_RETURN_IF_ERROR(ccu::EventWait(writeCompletion, allWriteBits));
        segmentOffset += vars.remoteOutputOffset;
    }

    ccu::LocalAddr localSuffix;
    localSuffix.addr = vars.input;
    localSuffix.addr += directSize;
    localSuffix.token = vars.inputToken;
    ccu::Event suffixWriteCompletion;
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        ccu::RemoteAddr destination;
        destination.addr = remoteOutput[channelIndex];
        destination.addr += localOwnerOffset;
        destination.addr += directSize;
        destination.token = remoteOutputToken[channelIndex];
        const uint16_t eventBit =
            static_cast<uint16_t>(1U << channelIndex);
        CCU_RETURN_IF_ERROR(ccu::Write(
            kernelArg->channels[channelIndex], destination, localSuffix,
            vars.sliceSize, suffixWriteCompletion, eventBit));
    }
    const uint16_t allSuffixWriteBits =
        static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(
        suffixWriteCompletion, allSuffixWriteBits));

    for (uint32_t sourceIndex = 0;
        sourceIndex < kernelArg->relaySourceCount; ++sourceIndex) {
        ccu::Event readCompletion;
        for (uint32_t channelIndex = 0;
            channelIndex < kernelArg->relayChannelCount; ++channelIndex) {
            const uint32_t sourceRank = sourceIndex == 0U ?
                kernelArg->relaySourceRanks0[channelIndex] :
                kernelArg->relaySourceRanks1[channelIndex];
            ccu::Variable sourceOffset;
            sourceOffset = 0U;
            for (uint32_t rankIndex = 0; rankIndex < sourceRank; ++rankIndex) {
                sourceOffset += fullSize;
            }

            ccu::RemoteAddr source;
            source.addr = remoteOutput[channelIndex];
            source.addr += sourceOffset;
            source.addr += directSize;
            source.token = remoteOutputToken[channelIndex];
            ccu::LocalAddr destination;
            destination.addr = vars.output;
            destination.addr += sourceOffset;
            destination.addr += directSize;
            destination.token = vars.outputToken;
            const uint16_t eventBit =
                static_cast<uint16_t>(1U << channelIndex);
            CCU_RETURN_IF_ERROR(ccu::Read(
                kernelArg->channels[channelIndex], destination, source,
                vars.sliceSize, readCompletion, eventBit));
        }
        const uint16_t allReadBits =
            static_cast<uint16_t>((1U << kernelArg->relayChannelCount) - 1U);
        CCU_RETURN_IF_ERROR(ccu::EventWait(readCompletion, allReadBits));
    }

    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
    }
    return CCU_SUCCESS;
}

inline CcuResult RunAsymmetricFusedQuarterWritePullRelay(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars)
{
    if (kernelArg->relayChannelCount == 0U ||
        kernelArg->relaySourceCount == 0U ||
        kernelArg->relaySourceCount > 2U) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> remoteOutput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteOutputToken(kernelArg->channelCount);
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        remoteOutput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_XN_ID);
        remoteOutputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_XN_ID);
    }
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
    }

    ccu::Variable directSize;
    directSize = vars.remoteOutputOffset;
    directSize += vars.remoteOutputOffset;
    directSize += vars.remoteOutputOffset;
    ccu::Variable fullSize;
    fullSize = directSize;
    fullSize += vars.sliceSize;

    ccu::Variable localOwnerOffset;
    localOwnerOffset = 0U;
    for (uint32_t rankIndex = 0; rankIndex < kernelArg->localRank; ++rankIndex) {
        localOwnerOffset += fullSize;
    }

    ccu::Variable segmentOffset;
    segmentOffset = 0U;
    for (uint32_t segmentIndex = 0; segmentIndex < 3U; ++segmentIndex) {
        ccu::LocalAddr localQuarter;
        localQuarter.addr = vars.input;
        localQuarter.addr += segmentOffset;
        localQuarter.token = vars.inputToken;
        ccu::Event writeCompletion;
        for (uint32_t channelIndex = 0;
            channelIndex < kernelArg->channelCount; ++channelIndex) {
            ccu::RemoteAddr destination;
            destination.addr = remoteOutput[channelIndex];
            destination.addr += localOwnerOffset;
            destination.addr += segmentOffset;
            destination.token = remoteOutputToken[channelIndex];
            const uint16_t eventBit =
                static_cast<uint16_t>(1U << channelIndex);
            CCU_RETURN_IF_ERROR(ccu::Write(
                kernelArg->channels[channelIndex], destination, localQuarter,
                vars.remoteOutputOffset, writeCompletion, eventBit));
        }
        const uint16_t allWriteBits =
            static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U);
        CCU_RETURN_IF_ERROR(ccu::EventWait(writeCompletion, allWriteBits));
        segmentOffset += vars.remoteOutputOffset;
    }

    ccu::LocalAddr localSuffix;
    localSuffix.addr = vars.input;
    localSuffix.addr += directSize;
    localSuffix.token = vars.inputToken;
    ccu::Event suffixWriteCompletion;
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        ccu::RemoteAddr destination;
        destination.addr = remoteOutput[channelIndex];
        destination.addr += localOwnerOffset;
        destination.addr += directSize;
        destination.token = remoteOutputToken[channelIndex];
        const uint16_t eventBit =
            static_cast<uint16_t>(1U << channelIndex);
        CCU_RETURN_IF_ERROR(ccu::Write(
            kernelArg->channels[channelIndex], destination, localSuffix,
            vars.sliceSize, suffixWriteCompletion, eventBit));
    }
    const uint16_t allSuffixWriteBits =
        static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(
        suffixWriteCompletion, allSuffixWriteBits));

    for (uint32_t sourceIndex = 0;
        sourceIndex < kernelArg->relaySourceCount; ++sourceIndex) {
        ccu::Event readCompletion;
        for (uint32_t channelIndex = 0;
            channelIndex < kernelArg->relayChannelCount; ++channelIndex) {
            const uint32_t sourceRank = sourceIndex == 0U ?
                kernelArg->relaySourceRanks0[channelIndex] :
                kernelArg->relaySourceRanks1[channelIndex];
            ccu::Variable sourceOffset;
            sourceOffset = 0U;
            for (uint32_t rankIndex = 0; rankIndex < sourceRank; ++rankIndex) {
                sourceOffset += fullSize;
            }

            ccu::RemoteAddr source;
            source.addr = remoteOutput[channelIndex];
            source.addr += sourceOffset;
            source.addr += directSize;
            source.token = remoteOutputToken[channelIndex];
            ccu::LocalAddr destination;
            destination.addr = vars.output;
            destination.addr += sourceOffset;
            destination.addr += directSize;
            destination.token = vars.outputToken;
            const uint16_t eventBit =
                static_cast<uint16_t>(1U << channelIndex);
            CCU_RETURN_IF_ERROR(ccu::Read(
                kernelArg->channels[channelIndex], destination, source,
                vars.sliceSize, readCompletion, eventBit));
        }
        const uint16_t allReadBits =
            static_cast<uint16_t>((1U << kernelArg->relayChannelCount) - 1U);
        CCU_RETURN_IF_ERROR(ccu::EventWait(readCompletion, allReadBits));
    }

    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
    }
    return CCU_SUCCESS;
}

inline CcuResult RunAsymmetricSegmentedDirectWriteRecord(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars)
{
    if (kernelArg->channelCount == 0U) {
        return CCU_E_PARA;
    }

    std::vector<ccu::Variable> remoteOutput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteOutputToken(kernelArg->channelCount);
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        remoteOutput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_XN_ID);
        remoteOutputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_XN_ID);
    }
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
    }

    ccu::Variable directSize;
    directSize = vars.remoteOutputOffset;
    directSize += vars.remoteOutputOffset;
    ccu::Variable fullSize;
    fullSize = directSize;
    fullSize += vars.sliceSize;
    ccu::Variable localOwnerOffset;
    localOwnerOffset = 0U;
    for (uint32_t rankIndex = 0; rankIndex < kernelArg->localRank; ++rankIndex) {
        localOwnerOffset += fullSize;
    }

    ccu::Variable segmentOffset;
    segmentOffset = 0U;
    for (uint32_t segmentIndex = 0; segmentIndex < 2U; ++segmentIndex) {
        ccu::LocalAddr source;
        source.addr = vars.input;
        source.addr += segmentOffset;
        source.token = vars.inputToken;
        ccu::Event completion;
        for (uint32_t channelIndex = 0;
            channelIndex < kernelArg->channelCount; ++channelIndex) {
            ccu::RemoteAddr destination;
            destination.addr = remoteOutput[channelIndex];
            destination.addr += localOwnerOffset;
            destination.addr += segmentOffset;
            destination.token = remoteOutputToken[channelIndex];
            const uint16_t eventBit =
                static_cast<uint16_t>(1U << channelIndex);
            CCU_RETURN_IF_ERROR(ccu::Write(
                kernelArg->channels[channelIndex], destination, source,
                vars.remoteOutputOffset, completion, eventBit));
        }
        const uint16_t allCompletionBits =
            static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U);
        CCU_RETURN_IF_ERROR(
            ccu::EventWait(completion, allCompletionBits));
        segmentOffset += vars.remoteOutputOffset;
    }

    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
    }
    return CCU_SUCCESS;
}

CcuResult LoadAllTaskArgs(TaskVariables &vars)
{
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.output, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.remoteOutputOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.sliceSize, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.stage, argId++));
    if (argId != TASK_ARGUMENT_COUNT) {
        return CCU_E_INTERNAL;
    }
    return CCU_SUCCESS;
}

CcuResult LoadSmallPullTaskArgs(SmallPullTaskVariables &vars)
{
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.output, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.sliceSize, argId++));
    if (argId != SMALL_PULL_TASK_ARGUMENT_COUNT) {
        return CCU_E_INTERNAL;
    }
    return CCU_SUCCESS;
}

CcuResult LoadLargePullTaskArgs(LargePullTaskVariables &vars)
{
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.output, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.rankStride, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(vars.transferSize, argId++));
    return argId == LARGE_PULL_TASK_ARGUMENT_COUNT ?
        CCU_SUCCESS : CCU_E_INTERNAL;
}

inline CcuResult PublishChannelPrefix(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars, uint32_t selectedCount)
{
    if (selectedCount == 0 || selectedCount > kernelArg->channelCount) {
        return CCU_E_PARA;
    }
    for (uint32_t channelIndex = 0; channelIndex < selectedCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIndex],
            vars.output, OUTPUT_XN_ID, OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIndex],
            vars.outputToken, OUTPUT_TOKEN_XN_ID, OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
    }
    return CCU_SUCCESS;
}

inline CcuResult WriteRecordChannelPrefix(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars, uint32_t selectedCount)
{
    if (selectedCount == 0 || selectedCount > kernelArg->channelCount) {
        return CCU_E_PARA;
    }
    std::vector<ccu::Variable> remoteOutput(selectedCount);
    std::vector<ccu::Variable> remoteOutputToken(selectedCount);
    for (uint32_t channelIndex = 0; channelIndex < selectedCount; ++channelIndex) {
        remoteOutput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                kernelArg->channels[channelIndex], OUTPUT_XN_ID);
        remoteOutputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                kernelArg->channels[channelIndex], OUTPUT_TOKEN_XN_ID);
    }
    for (uint32_t channelIndex = 0; channelIndex < selectedCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
    }

    ccu::LocalAddr source;
    source.addr = vars.input;
    source.token = vars.inputToken;
    ccu::Event completion;
    for (uint32_t channelIndex = 0; channelIndex < selectedCount; ++channelIndex) {
        ccu::RemoteAddr destination;
        destination.addr = remoteOutput[channelIndex];
        destination.addr += vars.remoteOutputOffset;
        destination.token = remoteOutputToken[channelIndex];
        const uint16_t completionMask = static_cast<uint16_t>(1U << channelIndex);
        CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[channelIndex], destination, source,
            vars.sliceSize, completion, completionMask));
    }
    const uint16_t allCompletionMask =
        static_cast<uint16_t>((1U << selectedCount) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, allCompletionMask));
    for (uint32_t channelIndex = 0; channelIndex < selectedCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
    }
    return CCU_SUCCESS;
}

inline CcuResult FinalWaitChannelPrefix(CcuKernelArgDirect *kernelArg,
    uint32_t selectedCount, bool acknowledgeCompletion)
{
    if (selectedCount == 0 || selectedCount > kernelArg->channelCount) {
        return CCU_E_PARA;
    }
    for (uint32_t channelIndex = 0; channelIndex < selectedCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
    }
    if (!acknowledgeCompletion) {
        return CCU_SUCCESS;
    }
    for (uint32_t channelIndex = 0; channelIndex < selectedCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            kernelArg->channels[channelIndex], COMPLETION_ACK_NOTIFY_INDEX, 1U));
    }
    for (uint32_t channelIndex = 0; channelIndex < selectedCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], COMPLETION_ACK_NOTIFY_INDEX, 1U));
    }
    return CCU_SUCCESS;
}

inline CcuResult RunWriteRecordWaitAll(
    CcuKernelArgDirect *kernelArg, TaskVariables &vars)
{
    std::vector<ccu::Variable> remoteOutput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteOutputToken(kernelArg->channelCount);
    for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
        remoteOutput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                kernelArg->channels[channelIndex], OUTPUT_XN_ID);
        remoteOutputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                kernelArg->channels[channelIndex], OUTPUT_TOKEN_XN_ID);
    }

    for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
    }

    ccu::LocalAddr source;
    source.addr = vars.input;
    source.token = vars.inputToken;

    ccu::Event completion;
    for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
        ccu::RemoteAddr destination;
        destination.addr = remoteOutput[channelIndex];
        destination.addr += vars.remoteOutputOffset;
        destination.token = remoteOutputToken[channelIndex];
        const uint16_t completionMask = static_cast<uint16_t>(1U << channelIndex);
        CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[channelIndex], destination, source,
            vars.sliceSize, completion, completionMask));
    }

    const uint16_t allCompletionMask =
        static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, allCompletionMask));

    for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
    }
    for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
    }
    return CCU_SUCCESS;
}

CcuResult StagedPushKernel(CcuKernelArg arg, uint32_t expectedLayer)
{
    CcuKernelArgDirect *kernelArg = nullptr;
    CCU_RETURN_IF_ERROR(ValidateKernelArg(arg, expectedLayer, kernelArg));

    TaskVariables vars;
    CCU_RETURN_IF_ERROR(LoadAllTaskArgs(vars));

    CCU_IF(vars.stage ==
        ASYMMETRIC_SEGMENTED_DIRECT_WRITE_RECORD_STAGE) {
        CCU_RETURN_IF_ERROR(
            RunAsymmetricSegmentedDirectWriteRecord(kernelArg, vars));
    } CCU_ELSE {
    CCU_IF(vars.stage == MONOLITHIC_STAGE) {
        for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIndex],
                vars.output, OUTPUT_XN_ID, OUTPUT_NOTIFY_INDEX, 1U));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIndex],
                vars.outputToken, OUTPUT_TOKEN_XN_ID, OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
        }
        CCU_RETURN_IF_ERROR(RunWriteRecordWaitAll(kernelArg, vars));
    } CCU_ELSE {
    CCU_IF(vars.stage == FUSED_WRITE_WAIT_STAGE) {
        CCU_RETURN_IF_ERROR(RunWriteRecordWaitAll(kernelArg, vars));
    } CCU_ELSE {
    CCU_IF(vars.stage == PUBLISH_STAGE) {
        for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIndex],
                vars.output, OUTPUT_XN_ID, OUTPUT_NOTIFY_INDEX, 1U));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIndex],
                vars.outputToken, OUTPUT_TOKEN_XN_ID, OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
        }
    } CCU_ELSE {
        CCU_IF(vars.stage == WRITE_RECORD_STAGE) {
            std::vector<ccu::Variable> remoteOutput(kernelArg->channelCount);
            std::vector<ccu::Variable> remoteOutputToken(kernelArg->channelCount);
            for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
                remoteOutput[channelIndex] =
                    ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIndex], OUTPUT_XN_ID);
                remoteOutputToken[channelIndex] =
                    ccu::GetResByChannel<ccu::Variable>(
                        kernelArg->channels[channelIndex], OUTPUT_TOKEN_XN_ID);
            }

            for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                    kernelArg->channels[channelIndex], OUTPUT_NOTIFY_INDEX, 1U));
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                    kernelArg->channels[channelIndex], OUTPUT_TOKEN_NOTIFY_INDEX, 1U));
            }

            ccu::LocalAddr source;
            source.addr = vars.input;
            source.token = vars.inputToken;

            ccu::Event completion;
            for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
                ccu::RemoteAddr destination;
                destination.addr = remoteOutput[channelIndex];
                destination.addr += vars.remoteOutputOffset;
                destination.token = remoteOutputToken[channelIndex];
                const uint16_t completionMask = static_cast<uint16_t>(1U << channelIndex);
                CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[channelIndex], destination, source,
                    vars.sliceSize, completion, completionMask));
            }

            const uint16_t allCompletionMask =
                static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U);
            CCU_RETURN_IF_ERROR(ccu::EventWait(completion, allCompletionMask));

            for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
                CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
                    kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
            }
        } CCU_ELSE {
            CCU_IF(vars.stage == FINAL_WAIT_STAGE) {
                for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
                    CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                        kernelArg->channels[channelIndex], POST_SYNC_NOTIFY_INDEX, 1U));
                }
            } CCU_ELSE {
                // These counts are host-side registration constants. Only one
                // subset family is translated for a given IO-Die kernel.
                const bool hasAsymmetricPairRound0 =
                    kernelArg->asymmetricPairRound0Channel != MAX_RANK_SIZE;
                const bool hasAsymmetricPairRound1 =
                    kernelArg->asymmetricPairRound1Channel != MAX_RANK_SIZE;
                if (kernelArg->asymmetricLayer0) {
                    CCU_IF(vars.stage ==
                        ASYMMETRIC_FUSED_THIRD_WRITE_PULL_RELAY_STAGE) {
                        CCU_RETURN_IF_ERROR(
                            RunAsymmetricFusedThirdWritePullRelay(
                                kernelArg, vars));
                    } CCU_ELSE {
                        CCU_IF(vars.stage ==
                            ASYMMETRIC_FUSED_QUARTER_WRITE_PULL_RELAY_STAGE) {
                            CCU_RETURN_IF_ERROR(
                                RunAsymmetricFusedQuarterWritePullRelay(
                                    kernelArg, vars));
                        } CCU_ELSE {
                            CCU_IF(vars.stage ==
                                ASYMMETRIC_FUSED_PULL_RELAY_STAGE) {
                                CCU_RETURN_IF_ERROR(
                                    RunAsymmetricFusedPullRelay(
                                        kernelArg, vars));
                            } CCU_ELSE {
                                // The only other asymmetric layer-0 stage is
                                // ACK_FINAL_WAIT_STAGE.
                                CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                    kernelArg, kernelArg->channelCount, true));
                            }
                        }
                    }
                } else if (hasAsymmetricPairRound0 && hasAsymmetricPairRound1) {
                    CCU_IF(vars.stage == PAIR_PUBLISH_STAGE) {
                        CCU_RETURN_IF_ERROR(PublishChannelPrefix(
                            kernelArg, vars, kernelArg->pairChannelCount));
                    } CCU_ELSE {
                        CCU_IF(vars.stage == PAIR_WRITE_RECORD_STAGE) {
                            CCU_RETURN_IF_ERROR(WriteRecordChannelPrefix(
                                kernelArg, vars, kernelArg->pairChannelCount));
                        } CCU_ELSE {
                            CCU_IF(vars.stage == PAIR_FINAL_WAIT_STAGE) {
                                CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                    kernelArg, kernelArg->pairChannelCount, true));
                            } CCU_ELSE {
                                CCU_IF(vars.stage == ASYMMETRIC_PAIR_ROUND0_STAGE) {
                                    CCU_RETURN_IF_ERROR(RunSymmetricBidirectionalSingleChannel(
                                        kernelArg, vars,
                                        kernelArg->asymmetricPairRound0Channel));
                                } CCU_ELSE {
                                    CCU_IF(vars.stage == ASYMMETRIC_PAIR_ROUND1_STAGE) {
                                        CCU_RETURN_IF_ERROR(RunSymmetricBidirectionalSingleChannel(
                                            kernelArg, vars,
                                            kernelArg->asymmetricPairRound1Channel));
                                    } CCU_ELSE {
                                        CCU_IF(vars.stage == ASYMMETRIC_NONPAIR_STAGE) {
                                            CCU_RETURN_IF_ERROR(
                                                RunMonolithicChannelSuffix(kernelArg, vars));
                                        } CCU_ELSE {
                                            CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                                kernelArg, kernelArg->channelCount, true));
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if (hasAsymmetricPairRound0) {
                    CCU_IF(vars.stage == PAIR_PUBLISH_STAGE) {
                        CCU_RETURN_IF_ERROR(PublishChannelPrefix(
                            kernelArg, vars, kernelArg->pairChannelCount));
                    } CCU_ELSE {
                        CCU_IF(vars.stage == PAIR_WRITE_RECORD_STAGE) {
                            CCU_RETURN_IF_ERROR(WriteRecordChannelPrefix(
                                kernelArg, vars, kernelArg->pairChannelCount));
                        } CCU_ELSE {
                            CCU_IF(vars.stage == PAIR_FINAL_WAIT_STAGE) {
                                CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                    kernelArg, kernelArg->pairChannelCount, true));
                            } CCU_ELSE {
                                CCU_IF(vars.stage == ASYMMETRIC_PAIR_ROUND0_STAGE) {
                                    CCU_RETURN_IF_ERROR(RunSymmetricBidirectionalSingleChannel(
                                        kernelArg, vars,
                                        kernelArg->asymmetricPairRound0Channel));
                                } CCU_ELSE {
                                    CCU_IF(vars.stage == ASYMMETRIC_NONPAIR_STAGE) {
                                        CCU_RETURN_IF_ERROR(
                                            RunMonolithicChannelSuffix(kernelArg, vars));
                                    } CCU_ELSE {
                                        CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                            kernelArg, kernelArg->channelCount, true));
                                    }
                                }
                            }
                        }
                    }
                } else if (hasAsymmetricPairRound1) {
                    CCU_IF(vars.stage == PAIR_PUBLISH_STAGE) {
                        CCU_RETURN_IF_ERROR(PublishChannelPrefix(
                            kernelArg, vars, kernelArg->pairChannelCount));
                    } CCU_ELSE {
                        CCU_IF(vars.stage == PAIR_WRITE_RECORD_STAGE) {
                            CCU_RETURN_IF_ERROR(WriteRecordChannelPrefix(
                                kernelArg, vars, kernelArg->pairChannelCount));
                        } CCU_ELSE {
                            CCU_IF(vars.stage == PAIR_FINAL_WAIT_STAGE) {
                                CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                    kernelArg, kernelArg->pairChannelCount, true));
                            } CCU_ELSE {
                                CCU_IF(vars.stage == ASYMMETRIC_PAIR_ROUND1_STAGE) {
                                    CCU_RETURN_IF_ERROR(RunSymmetricBidirectionalSingleChannel(
                                        kernelArg, vars,
                                        kernelArg->asymmetricPairRound1Channel));
                                } CCU_ELSE {
                                    CCU_IF(vars.stage == ASYMMETRIC_NONPAIR_STAGE) {
                                        CCU_RETURN_IF_ERROR(
                                            RunMonolithicChannelSuffix(kernelArg, vars));
                                    } CCU_ELSE {
                                        CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                            kernelArg, kernelArg->channelCount, true));
                                    }
                                }
                            }
                        }
                    }
                } else if (kernelArg->pairChannelCount != 0U) {
                    CCU_IF(vars.stage == PAIR_PUBLISH_STAGE) {
                        CCU_RETURN_IF_ERROR(PublishChannelPrefix(
                            kernelArg, vars, kernelArg->pairChannelCount));
                    } CCU_ELSE {
                        CCU_IF(vars.stage == PAIR_WRITE_RECORD_STAGE) {
                            CCU_RETURN_IF_ERROR(WriteRecordChannelPrefix(
                                kernelArg, vars, kernelArg->pairChannelCount));
                        } CCU_ELSE {
                            CCU_IF(vars.stage == PAIR_FINAL_WAIT_STAGE) {
                                CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                    kernelArg, kernelArg->pairChannelCount, true));
                            } CCU_ELSE {
                                CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                    kernelArg, kernelArg->channelCount, true));
                            }
                        }
                    }
                } else if (kernelArg->relayChannelCount != 0U) {
                    CCU_IF(vars.stage == RELAY_PUBLISH_STAGE) {
                        CCU_RETURN_IF_ERROR(PublishChannelPrefix(
                            kernelArg, vars, kernelArg->relayChannelCount));
                    } CCU_ELSE {
                        CCU_IF(vars.stage == RELAY_WRITE_RECORD_STAGE) {
                            CCU_RETURN_IF_ERROR(WriteRecordChannelPrefix(
                                kernelArg, vars, kernelArg->relayChannelCount));
                        } CCU_ELSE {
                            CCU_IF(vars.stage == RELAY_FINAL_WAIT_STAGE) {
                                CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                    kernelArg, kernelArg->relayChannelCount, false));
                            } CCU_ELSE {
                                CCU_IF(vars.stage == RELAY_ACK_FINAL_WAIT_STAGE) {
                                    CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                        kernelArg, kernelArg->relayChannelCount, true));
                                } CCU_ELSE {
                                    CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                                        kernelArg, kernelArg->channelCount, true));
                                }
                            }
                        }
                    }
                } else {
                    CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                        kernelArg, kernelArg->channelCount, true));
                }
            }
        }
    }
    }
    }
    }

    return CCU_SUCCESS;
}

CcuResult Small4x1DirectPullKernel(CcuKernelArg arg)
{
    CcuKernelArgDirect *kernelArg = nullptr;
    CCU_RETURN_IF_ERROR(ValidateKernelArg(arg, 1U, kernelArg));
    if (kernelArg->localRank >= 4U ||
        kernelArg->channelCount != 3U ||
        kernelArg->pairChannelCount != 0U ||
        kernelArg->relayChannelCount != 0U) {
        return CCU_E_PARA;
    }

    SmallPullTaskVariables vars;
    CCU_RETURN_IF_ERROR(LoadSmallPullTaskArgs(vars));

    // This kernel exchanges source-buffer metadata, not output-buffer
    // metadata. The existing resource slots are reused only as numeric Channel
    // resources; the dedicated handle gives this protocol its own translated
    // instruction stream.
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.input, OUTPUT_XN_ID,
            OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.inputToken, OUTPUT_TOKEN_XN_ID,
            OUTPUT_NOTIFY_INDEX, 2U));
    }

    ccu::LocalAddr localSource;
    localSource.addr = vars.input;
    localSource.token = vars.inputToken;
    ccu::LocalAddr localDestination;
    localDestination.addr = vars.output;
    for (uint32_t rankIndex = 0;
        rankIndex < kernelArg->localRank; ++rankIndex) {
        localDestination.addr += vars.sliceSize;
    }
    localDestination.token = vars.outputToken;

    // Start the local DMA before waiting for peer metadata, so it overlaps the
    // source-identity rendezvous and the subsequent network Reads.
    ccu::Event completion;
    uint16_t completionMask = 1U;
    CCU_RETURN_IF_ERROR(ccu::LocalCopy(
        localDestination, localSource, vars.sliceSize,
        completion, 1U));

    std::vector<ccu::Variable> remoteInput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->channelCount);
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        remoteInput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                channel, OUTPUT_XN_ID);
        remoteInputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                channel, OUTPUT_TOKEN_XN_ID);
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            channel, OUTPUT_NOTIFY_INDEX, 3U));
    }

    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        ccu::RemoteAddr remoteSource;
        remoteSource.addr = remoteInput[channelIndex];
        remoteSource.token = remoteInputToken[channelIndex];

        ccu::LocalAddr peerDestination;
        peerDestination.addr = vars.output;
        for (uint32_t rankIndex = 0;
            rankIndex < kernelArg->peerRanks[channelIndex]; ++rankIndex) {
            peerDestination.addr += vars.sliceSize;
        }
        peerDestination.token = vars.outputToken;

        const uint16_t readCompletionBit =
            static_cast<uint16_t>(1U << (channelIndex + 1U));
        completionMask =
            static_cast<uint16_t>(completionMask | readCompletionBit);
        CCU_RETURN_IF_ERROR(ccu::Read(
            kernelArg->channels[channelIndex],
            peerDestination, remoteSource, vars.sliceSize,
            completion, readCompletionBit));
    }

    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, completionMask));
    return CCU_SUCCESS;
}

CcuResult Small2x8Layer0DirectPullKernel(CcuKernelArg arg)
{
    CcuKernelArgDirect *kernelArg = nullptr;
    CCU_RETURN_IF_ERROR(ValidateKernelArg(arg, 0U, kernelArg));
    if (kernelArg->localRank >= 16U ||
        kernelArg->channelCount > 15U ||
        kernelArg->pairChannelCount != 0U ||
        kernelArg->relayChannelCount != 0U) {
        return CCU_E_PARA;
    }

    SmallPullTaskVariables vars;
    CCU_RETURN_IF_ERROR(LoadSmallPullTaskArgs(vars));

    // This handle is translated separately from the staged layer-0 kernel.
    // It exchanges immutable source metadata once, pulls every peer owned by
    // this (layer, Die) resource group, and waits for local Read completion.
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.input, OUTPUT_XN_ID,
            OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.inputToken, OUTPUT_TOKEN_XN_ID,
            OUTPUT_NOTIFY_INDEX, 2U));
    }

    // Start LocalCopy on the Mesh CCU before consuming peer metadata.
    ccu::LocalAddr localSource;
    localSource.addr = vars.input;
    localSource.token = vars.inputToken;
    ccu::LocalAddr localDestination;
    localDestination.addr = vars.output;
    for (uint32_t rankIndex = 0;
        rankIndex < kernelArg->localRank; ++rankIndex) {
        localDestination.addr += vars.sliceSize;
    }
    localDestination.token = vars.outputToken;

    ccu::Event completion;
    uint16_t completionMask = 1U;
    CCU_RETURN_IF_ERROR(ccu::LocalCopy(
        localDestination, localSource, vars.sliceSize,
        completion, 1U));

    std::vector<ccu::Variable> remoteInput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->channelCount);
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        remoteInput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                channel, OUTPUT_XN_ID);
        remoteInputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                channel, OUTPUT_TOKEN_XN_ID);
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            channel, OUTPUT_NOTIFY_INDEX, 3U));
    }

    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        ccu::RemoteAddr remoteSource;
        remoteSource.addr = remoteInput[channelIndex];
        remoteSource.token = remoteInputToken[channelIndex];

        ccu::LocalAddr peerDestination;
        peerDestination.addr = vars.output;
        for (uint32_t rankIndex = 0;
            rankIndex < kernelArg->peerRanks[channelIndex]; ++rankIndex) {
            peerDestination.addr += vars.sliceSize;
        }
        peerDestination.token = vars.outputToken;

        const uint16_t readCompletionBit =
            static_cast<uint16_t>(1U << (channelIndex + 1U));
        completionMask =
            static_cast<uint16_t>(completionMask | readCompletionBit);
        CCU_RETURN_IF_ERROR(ccu::Read(
            kernelArg->channels[channelIndex],
            peerDestination, remoteSource, vars.sliceSize,
            completion, readCompletionBit));
    }

    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, completionMask));
    return CCU_SUCCESS;
}

CcuResult Small2x8Layer1DirectPullKernel(CcuKernelArg arg)
{
    CcuKernelArgDirect *kernelArg = nullptr;
    CCU_RETURN_IF_ERROR(ValidateKernelArg(arg, 1U, kernelArg));
    if (kernelArg->localRank >= 16U ||
        kernelArg->channelCount > 14U ||
        kernelArg->pairChannelCount != 0U ||
        kernelArg->relayChannelCount != 0U) {
        return CCU_E_PARA;
    }

    SmallPullTaskVariables vars;
    CCU_RETURN_IF_ERROR(LoadSmallPullTaskArgs(vars));

    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.input, OUTPUT_XN_ID,
            OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.inputToken, OUTPUT_TOKEN_XN_ID,
            OUTPUT_NOTIFY_INDEX, 2U));
    }

    std::vector<ccu::Variable> remoteInput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->channelCount);
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        remoteInput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                channel, OUTPUT_XN_ID);
        remoteInputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                channel, OUTPUT_TOKEN_XN_ID);
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            channel, OUTPUT_NOTIFY_INDEX, 3U));
    }

    ccu::Event completion;
    uint16_t completionMask = 0U;

    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        ccu::RemoteAddr remoteSource;
        remoteSource.addr = remoteInput[channelIndex];
        remoteSource.token = remoteInputToken[channelIndex];

        ccu::LocalAddr peerDestination;
        peerDestination.addr = vars.output;
        for (uint32_t rankIndex = 0;
            rankIndex < kernelArg->peerRanks[channelIndex]; ++rankIndex) {
            peerDestination.addr += vars.sliceSize;
        }
        peerDestination.token = vars.outputToken;

        const uint16_t readCompletionBit =
            static_cast<uint16_t>(1U << channelIndex);
        completionMask =
            static_cast<uint16_t>(completionMask | readCompletionBit);
        CCU_RETURN_IF_ERROR(ccu::Read(
            kernelArg->channels[channelIndex],
            peerDestination, remoteSource, vars.sliceSize,
            completion, readCompletionBit));
    }

    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, completionMask));
    return CCU_SUCCESS;
}

CcuResult Large8Plus4Layer0DirectPullKernel(CcuKernelArg arg)
{
    CcuKernelArgDirect *kernelArg = nullptr;
    CCU_RETURN_IF_ERROR(ValidateKernelArg(arg, 0U, kernelArg));
    if (kernelArg->localRank >= 12U ||
        kernelArg->channelCount > 11U ||
        kernelArg->pairChannelCount != 0U ||
        kernelArg->relayChannelCount != 0U) {
        return CCU_E_PARA;
    }

    LargePullTaskVariables vars;
    CCU_RETURN_IF_ERROR(LoadLargePullTaskArgs(vars));
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.input, OUTPUT_XN_ID,
            OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.inputToken, OUTPUT_TOKEN_XN_ID,
            OUTPUT_NOTIFY_INDEX, 2U));
    }

    // The formal small 8+4 path reuses this handle with transferSize equal to
    // one rank slice. Keep its LocalCopy on Mesh and overlap metadata waits.
    ccu::LocalAddr localSource;
    localSource.addr = vars.input;
    localSource.token = vars.inputToken;
    ccu::LocalAddr localDestination;
    localDestination.addr = vars.output;
    for (uint32_t rankIndex = 0;
        rankIndex < kernelArg->localRank; ++rankIndex) {
        localDestination.addr += vars.rankStride;
    }
    localDestination.token = vars.outputToken;

    ccu::Event completion;
    uint16_t completionMask = 1U;
    CCU_RETURN_IF_ERROR(ccu::LocalCopy(
        localDestination, localSource, vars.transferSize,
        completion, 1U));

    std::vector<ccu::Variable> remoteInput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->channelCount);
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        remoteInput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_XN_ID);
        remoteInputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                channel, OUTPUT_TOKEN_XN_ID);
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            channel, OUTPUT_NOTIFY_INDEX, 3U));
    }

    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        ccu::RemoteAddr remoteSource;
        remoteSource.addr = remoteInput[channelIndex];
        remoteSource.token = remoteInputToken[channelIndex];
        ccu::LocalAddr peerDestination;
        peerDestination.addr = vars.output;
        for (uint32_t rankIndex = 0;
            rankIndex < kernelArg->peerRanks[channelIndex]; ++rankIndex) {
            peerDestination.addr += vars.rankStride;
        }
        peerDestination.token = vars.outputToken;
        const uint16_t readCompletionBit =
            static_cast<uint16_t>(1U << (channelIndex + 1U));
        completionMask = static_cast<uint16_t>(
            completionMask | readCompletionBit);
        CCU_RETURN_IF_ERROR(ccu::Read(
            kernelArg->channels[channelIndex], peerDestination,
            remoteSource, vars.transferSize, completion,
            readCompletionBit));
    }
    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, completionMask));
    return CCU_SUCCESS;
}

CcuResult Large8Plus4Layer1DirectPullKernel(CcuKernelArg arg)
{
    CcuKernelArgDirect *kernelArg = nullptr;
    CCU_RETURN_IF_ERROR(ValidateKernelArg(arg, 1U, kernelArg));
    if (kernelArg->localRank >= 12U ||
        kernelArg->channelCount > 11U ||
        kernelArg->pairChannelCount != 0U ||
        kernelArg->relayChannelCount != 0U) {
        return CCU_E_PARA;
    }

    LargePullTaskVariables vars;
    CCU_RETURN_IF_ERROR(LoadLargePullTaskArgs(vars));
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.input, OUTPUT_XN_ID,
            OUTPUT_NOTIFY_INDEX, 1U));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, vars.inputToken, OUTPUT_TOKEN_XN_ID,
            OUTPUT_NOTIFY_INDEX, 2U));
    }

    std::vector<ccu::Variable> remoteInput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->channelCount);
    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        const ChannelHandle channel = kernelArg->channels[channelIndex];
        remoteInput[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_XN_ID);
        remoteInputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                channel, OUTPUT_TOKEN_XN_ID);
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            channel, OUTPUT_NOTIFY_INDEX, 3U));
    }

    ccu::Event completion;
    uint16_t completionMask = 0U;

    for (uint32_t channelIndex = 0;
        channelIndex < kernelArg->channelCount; ++channelIndex) {
        ccu::RemoteAddr remoteSource;
        remoteSource.addr = remoteInput[channelIndex];
        remoteSource.token = remoteInputToken[channelIndex];
        ccu::LocalAddr peerDestination;
        peerDestination.addr = vars.output;
        for (uint32_t rankIndex = 0;
            rankIndex < kernelArg->peerRanks[channelIndex]; ++rankIndex) {
            peerDestination.addr += vars.rankStride;
        }
        peerDestination.token = vars.outputToken;
        const uint16_t readCompletionBit =
            static_cast<uint16_t>(1U << channelIndex);
        completionMask = static_cast<uint16_t>(
            completionMask | readCompletionBit);
        CCU_RETURN_IF_ERROR(ccu::Read(
            kernelArg->channels[channelIndex], peerDestination,
            remoteSource, vars.transferSize, completion,
            readCompletionBit));
    }
    CCU_RETURN_IF_ERROR(ccu::EventWait(completion, completionMask));
    return CCU_SUCCESS;
}

#if 0
CcuResult PairPushKernel(CcuKernelArg arg)
{
    CcuKernelArgDirect *kernelArg = nullptr;
    CCU_RETURN_IF_ERROR(ValidateKernelArg(arg, 1U, kernelArg));
    TaskVariables vars;
    CCU_RETURN_IF_ERROR(LoadAllTaskArgs(vars));

    CCU_IF(vars.stage == MONOLITHIC_STAGE) {
        CCU_RETURN_IF_ERROR(PublishChannelPrefix(
            kernelArg, vars, kernelArg->channelCount));
        CCU_RETURN_IF_ERROR(RunWriteRecordWaitAll(kernelArg, vars));
    } CCU_ELSE {
    CCU_IF(vars.stage == FUSED_WRITE_WAIT_STAGE) {
        CCU_RETURN_IF_ERROR(RunWriteRecordWaitAll(kernelArg, vars));
    } CCU_ELSE {
    CCU_IF(vars.stage == PUBLISH_STAGE) {
        CCU_RETURN_IF_ERROR(PublishChannelPrefix(
            kernelArg, vars, kernelArg->channelCount));
    } CCU_ELSE {
    CCU_IF(vars.stage == WRITE_RECORD_STAGE) {
        CCU_RETURN_IF_ERROR(WriteRecordChannelPrefix(
            kernelArg, vars, kernelArg->channelCount));
    } CCU_ELSE {
    CCU_IF(vars.stage == FINAL_WAIT_STAGE) {
        CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
            kernelArg, kernelArg->channelCount, false));
    } CCU_ELSE {
    CCU_IF(vars.stage == ACK_FINAL_WAIT_STAGE) {
        CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
            kernelArg, kernelArg->channelCount, true));
    } CCU_ELSE {
    CCU_IF(vars.stage == PAIR_PUBLISH_STAGE) {
        CCU_RETURN_IF_ERROR(PublishChannelPrefix(
            kernelArg, vars, kernelArg->channelCount));
    } CCU_ELSE {
        CCU_IF(vars.stage == PAIR_WRITE_RECORD_STAGE) {
            CCU_RETURN_IF_ERROR(WriteRecordChannelPrefix(
                kernelArg, vars, kernelArg->channelCount));
        } CCU_ELSE {
            CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                kernelArg, kernelArg->channelCount, true));
        }
    }
    }
    }
    }
    }
    }
    }
    return CCU_SUCCESS;
}

CcuResult RelayPushKernel(CcuKernelArg arg)
{
    CcuKernelArgDirect *kernelArg = nullptr;
    CCU_RETURN_IF_ERROR(ValidateKernelArg(arg, 0U, kernelArg));
    TaskVariables vars;
    CCU_RETURN_IF_ERROR(LoadAllTaskArgs(vars));

    CCU_IF(vars.stage == MONOLITHIC_STAGE) {
        CCU_RETURN_IF_ERROR(PublishChannelPrefix(
            kernelArg, vars, kernelArg->channelCount));
        CCU_RETURN_IF_ERROR(RunWriteRecordWaitAll(kernelArg, vars));
    } CCU_ELSE {
    CCU_IF(vars.stage == FUSED_WRITE_WAIT_STAGE) {
        CCU_RETURN_IF_ERROR(RunWriteRecordWaitAll(kernelArg, vars));
    } CCU_ELSE {
    CCU_IF(vars.stage == PUBLISH_STAGE) {
        CCU_RETURN_IF_ERROR(PublishChannelPrefix(
            kernelArg, vars, kernelArg->channelCount));
    } CCU_ELSE {
    CCU_IF(vars.stage == WRITE_RECORD_STAGE) {
        CCU_RETURN_IF_ERROR(WriteRecordChannelPrefix(
            kernelArg, vars, kernelArg->channelCount));
    } CCU_ELSE {
    CCU_IF(vars.stage == FINAL_WAIT_STAGE) {
        CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
            kernelArg, kernelArg->channelCount, false));
    } CCU_ELSE {
    CCU_IF(vars.stage == ACK_FINAL_WAIT_STAGE) {
        CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
            kernelArg, kernelArg->channelCount, true));
    } CCU_ELSE {
    CCU_IF(vars.stage == RELAY_PUBLISH_STAGE) {
        CCU_RETURN_IF_ERROR(PublishChannelPrefix(
            kernelArg, vars, kernelArg->channelCount));
    } CCU_ELSE {
        CCU_IF(vars.stage == RELAY_WRITE_RECORD_STAGE) {
            CCU_RETURN_IF_ERROR(WriteRecordChannelPrefix(
                kernelArg, vars, kernelArg->channelCount));
        } CCU_ELSE {
            CCU_IF(vars.stage == RELAY_FINAL_WAIT_STAGE) {
                CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                    kernelArg, kernelArg->channelCount, false));
            } CCU_ELSE {
                CCU_RETURN_IF_ERROR(FinalWaitChannelPrefix(
                    kernelArg, kernelArg->channelCount, true));
            }
        }
    }
    }
    }
    }
    }
    }
    }
    return CCU_SUCCESS;
}
#endif

} // namespace

CcuResult CcuStagedLayer0(CcuKernelArg arg)
{
    return StagedPushKernel(arg, 0);
}

CcuResult CcuStagedLayer1(CcuKernelArg arg)
{
    return StagedPushKernel(arg, 1);
}

CcuResult CcuSmall4x1DirectPull(CcuKernelArg arg)
{
    return Small4x1DirectPullKernel(arg);
}

CcuResult CcuSmall2x8Layer0DirectPull(CcuKernelArg arg)
{
    return Small2x8Layer0DirectPullKernel(arg);
}

CcuResult CcuSmall2x8Layer1DirectPull(CcuKernelArg arg)
{
    return Small2x8Layer1DirectPullKernel(arg);
}

CcuResult CcuLarge8Plus4Layer0DirectPull(CcuKernelArg arg)
{
    return Large8Plus4Layer0DirectPullKernel(arg);
}

CcuResult CcuLarge8Plus4Layer1DirectPull(CcuKernelArg arg)
{
    return Large8Plus4Layer1DirectPullKernel(arg);
}

#if 0
CcuResult CcuPairLayer1(CcuKernelArg arg)
{
    return PairPushKernel(arg);
}

CcuResult CcuRelayLayer0(CcuKernelArg arg)
{
    return RelayPushKernel(arg);
}
#endif

#undef CCU_RETURN_IF_ERROR

} // namespace ops_hccl
