/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel.h"
#include "log.h"

#define CCU_KERNEL_CHK_RET(call) \
    do { \
        const CcuResult ccuKernelResult = static_cast<CcuResult>(call); \
        if (ccuKernelResult != CCU_SUCCESS) { \
            return ccuKernelResult; \
        } \
    } while (0)

namespace ops_hccl {
namespace {

    constexpr uint32_t REMOTE_OUTPUT_VARIABLE_ID = 1;
    constexpr uint32_t REMOTE_TOKEN_VARIABLE_ID = 2;
    constexpr uint32_t POST_SYNC_NOTIFY_ID = 3;
    constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
    constexpr uint64_t DIRECT_PHASE = 0;

    constexpr uint64_t SetLowBits(uint16_t end)
    {
        return (uint64_t(1) << (end + 1)) - uint64_t(1);
    }

    constexpr uint64_t PackLoopParameter(uint64_t contextId, uint64_t addressOffset, uint64_t loopIterations)
    {
        constexpr uint16_t CONTEXT_BITS = 8;
        constexpr uint16_t CONTEXT_SHIFT = 45;
        constexpr uint16_t ADDRESS_BITS = 32;
        constexpr uint16_t ADDRESS_SHIFT = 13;
        constexpr uint16_t ITERATION_BITS = 13;
        return ((contextId & SetLowBits(CONTEXT_BITS)) << CONTEXT_SHIFT)
               | ((addressOffset & SetLowBits(ADDRESS_BITS)) << ADDRESS_SHIFT)
               | (loopIterations & SetLowBits(ITERATION_BITS));
    }

    constexpr uint64_t PackParallelParameter(uint64_t repeatNumber, uint64_t repeatLoopIndex, uint64_t totalLoops)
    {
        constexpr uint16_t FIELD_BITS = 7;
        constexpr uint16_t REPEAT_SHIFT = 55;
        constexpr uint16_t REPEAT_LOOP_SHIFT = 48;
        constexpr uint16_t TOTAL_LOOP_SHIFT = 41;
        return ((repeatNumber & SetLowBits(FIELD_BITS)) << REPEAT_SHIFT)
               | ((repeatLoopIndex & SetLowBits(FIELD_BITS)) << REPEAT_LOOP_SHIFT)
               | ((totalLoops & SetLowBits(FIELD_BITS)) << TOTAL_LOOP_SHIFT);
    }

    constexpr uint64_t PackOffsetParameter(
        uint64_t globalAddressOffset, uint64_t memorySliceOffset, uint64_t eventOffset)
    {
        constexpr uint16_t GLOBAL_ADDRESS_BITS = 32;
        constexpr uint16_t GLOBAL_ADDRESS_SHIFT = 21;
        constexpr uint16_t MEMORY_SLICE_BITS = 11;
        constexpr uint16_t MEMORY_SLICE_SHIFT = 10;
        constexpr uint16_t EVENT_BITS = 10;
        return ((globalAddressOffset & SetLowBits(GLOBAL_ADDRESS_BITS)) << GLOBAL_ADDRESS_SHIFT)
               | ((memorySliceOffset & SetLowBits(MEMORY_SLICE_BITS)) << MEMORY_SLICE_SHIFT)
               | (eventOffset & SetLowBits(EVENT_BITS));
    }

    void InitializeGroupCopy(
        AllGatherKernelContext &context, ccu::LocalAddr *sources, ccu::LocalAddr *destinations, ccu::Variable *lengths)
    {
        if (!context.copyResourceAllocated) {
            context.copyConfig.memoryInterleave = CCU_MEMORY_INTERLEAVE;
            context.copyConfig.loopCount = CCU_LOCAL_COPY_LOOP_COUNT;
            context.copyConfig.memorySliceBytes = CCU_LOCAL_COPY_SLICES_PER_LOOP * CCU_MEMORY_SLICE_BYTES;
            context.copyResource.eventCount = context.copyConfig.loopCount;
            context.copyResource.completedEvents = ccu::Array<ccu::Event>(context.copyResource.eventCount);
            context.copyResource.bufferCount = context.copyConfig.loopCount * context.copyConfig.memoryInterleave;
            context.copyResource.buffers = ccu::Array<ccu::CcuBuffer>(context.copyResource.bufferCount);
            context.copyResourceAllocated = true;
        }

        constexpr const char *LOOP_NAME = "local_copy";
        if (context.loopEntities.count(LOOP_NAME) != 0) {
            return;
        }

        context.loopEntities.emplace(LOOP_NAME, LoopEntity());
        LoopEntity &entity = context.loopEntities[LOOP_NAME];
        for (uint32_t index = 0; index < 2; ++index) {
            const uint32_t bufferBase = index * context.copyConfig.memoryInterleave;
            const ccu::Event event = context.copyResource.completedEvents[index];
            entity.body[index].reset(new ccu::Func([&context, index, bufferBase, event, sources, destinations,
                                                       lengths]() {
                ccu::LocalCopy(context.copyResource.buffers[bufferBase], sources[index], lengths[index], event, 1);
                ccu::EventWait(event, 1);
                ccu::LocalCopy(destinations[index], context.copyResource.buffers[bufferBase], lengths[index], event, 1);
                ccu::EventWait(event, 1);
            }));
            entity.loops[index].reset(new ccu::Loop(entity.loopParameter[index], *entity.body[index]));
        }
    }

    CcuResult GroupCopy(
        AllGatherKernelContext &context, ccu::LocalAddr destination, ccu::LocalAddr source, GroupCopySize size)
    {
        ccu::LocalAddr loopSources[2];
        ccu::LocalAddr loopDestinations[2];
        ccu::Variable loopLengths[2];
        InitializeGroupCopy(context, loopSources, loopDestinations, loopLengths);
        LoopEntity &entity = context.loopEntities["local_copy"];

        CCU_IF(size.addressOffset != 0)
        {
            ccu::Variable loopParameter;
            loopParameter = PackLoopParameter(0, context.copyConfig.memorySliceBytes * context.copyConfig.loopCount, 0);
            loopParameter += size.loopParameter;

            ccu::Variable memorySliceBytes;
            memorySliceBytes = context.copyConfig.memorySliceBytes;
            loopSources[0].addr = source.addr;
            loopSources[0].token = source.token;
            loopDestinations[0].addr = destination.addr;
            loopDestinations[0].token = destination.token;
            loopLengths[0] = memorySliceBytes;
            entity.loopParameter[0] = loopParameter;

            ccu::Variable parallelParameter;
            parallelParameter = PackParallelParameter(context.copyConfig.loopCount - 1, 0, 1);
            ccu::Variable offsetParameter;
            offsetParameter
                = PackOffsetParameter(context.copyConfig.memorySliceBytes, context.copyConfig.memoryInterleave, 1);
            std::vector<ccu::Loop> loops{*entity.loops[0]};
            ccu::LoopGroup group(parallelParameter, offsetParameter, context.copyConfig.loopCount, loops);
        }

        CCU_IF(size.parallelParameter != 0)
        {
            source.addr += size.addressOffset;
            destination.addr += size.addressOffset;
            loopSources[0].addr = source.addr;
            loopSources[0].token = source.token;
            loopDestinations[0].addr = destination.addr;
            loopDestinations[0].token = destination.token;
            loopLengths[0] = size.residualBytes;

            source.addr += size.residualBytes;
            destination.addr += size.residualBytes;
            ccu::Variable memorySliceBytes;
            memorySliceBytes = context.copyConfig.memorySliceBytes;
            loopSources[1].addr = source.addr;
            loopSources[1].token = source.token;
            loopDestinations[1].addr = destination.addr;
            loopDestinations[1].token = destination.token;
            loopLengths[1] = memorySliceBytes;

            entity.loopParameter[0] = PackLoopParameter(0, 0, 1);
            entity.loopParameter[1] = PackLoopParameter(0, 0, 1);
            ccu::Variable offsetParameter;
            offsetParameter
                = PackOffsetParameter(context.copyConfig.memorySliceBytes, context.copyConfig.memoryInterleave, 1);
            std::vector<ccu::Loop> loops{*entity.loops[0], *entity.loops[1]};
            ccu::LoopGroup group(size.parallelParameter, offsetParameter, context.copyConfig.loopCount, loops);
        }
        return CCU_SUCCESS;
    }

    CcuResult LoadArguments(AllGatherKernelContext &context)
    {
        uint32_t argumentId = 0;
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.input, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.outputs[context.arg->rankId], argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.tokens[context.arg->rankId], argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.inputOffset, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.outputOffset, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.sliceBytes, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.groupCopySize.addressOffset, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.groupCopySize.loopParameter, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.groupCopySize.parallelParameter, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.groupCopySize.residualBytes, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.lastSlice, argumentId++));
        return CCU_SUCCESS;
    }

    CcuResult LoadRelayArguments(AllGatherKernelContext &context)
    {
        CCU_KERNEL_CHK_RET(LoadArguments(context));
        uint32_t argumentId = 11;
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.firstSlice, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.relayOutputOffset, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.directBytes, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.relayBytes, argumentId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(context.phase, argumentId++));
        return CCU_SUCCESS;
    }

    CcuResult TransferOriginal(AllGatherKernelContext &context)
    {
        ccu::LocalAddr source;
        source.addr = context.input;
        source.addr += context.inputOffset;
        source.token = context.tokens[context.arg->rankId];

        std::vector<ccu::RemoteAddr> remoteDestinations(context.arg->channelCount);
        for (uint32_t channelIndex = 0; channelIndex < context.arg->channelCount; ++channelIndex) {
            const uint32_t remoteRank = context.arg->remoteRanks[channelIndex];
            remoteDestinations[channelIndex].addr = context.outputs[remoteRank];
            remoteDestinations[channelIndex].addr += context.outputOffset;
            remoteDestinations[channelIndex].token = context.tokens[remoteRank];
        }

        CCU_IF(context.sliceBytes != 0)
        {
            for (uint32_t channelIndex = 0; channelIndex < context.arg->channelCount; ++channelIndex) {
                const uint32_t remoteRank = context.arg->remoteRanks[channelIndex];
                const uint16_t completionMask = static_cast<uint16_t>(1U << remoteRank);
                CCU_KERNEL_CHK_RET(ccu::Write(context.arg->channels[channelIndex],
                    remoteDestinations[channelIndex], source, context.sliceBytes, context.completionEvent,
                    completionMask));
            }
        }

        if (context.arg->handleLocalCopy) {
            ccu::LocalAddr localDestination;
            localDestination.addr = context.outputs[context.arg->rankId];
            localDestination.addr += context.outputOffset;
            localDestination.token = context.tokens[context.arg->rankId];
            CCU_KERNEL_CHK_RET(GroupCopy(context, localDestination, source, context.groupCopySize));
        }

        uint16_t remoteMask = 0;
        for (uint32_t channelIndex = 0; channelIndex < context.arg->channelCount; ++channelIndex) {
            remoteMask |= static_cast<uint16_t>(1U << context.arg->remoteRanks[channelIndex]);
        }
        CCU_KERNEL_CHK_RET(ccu::EventWait(context.completionEvent, remoteMask));
        return CCU_SUCCESS;
    }

    CcuResult SynchronizeAddresses(AllGatherKernelContext &context)
    {
        for (uint32_t i = 0; i < context.arg->channelCount; ++i) {
            CCU_KERNEL_CHK_RET(
                ccu::WriteVariableWithNotify(context.arg->channels[i], context.outputs[context.arg->rankId],
                    REMOTE_OUTPUT_VARIABLE_ID, CHANNEL_NOTIFY_INDEX, 1U << REMOTE_OUTPUT_VARIABLE_ID));
            CCU_KERNEL_CHK_RET(
                ccu::WriteVariableWithNotify(context.arg->channels[i], context.tokens[context.arg->rankId],
                    REMOTE_TOKEN_VARIABLE_ID, CHANNEL_NOTIFY_INDEX, 1U << REMOTE_TOKEN_VARIABLE_ID));
        }

        constexpr uint32_t ADDRESS_READY_MASK = (1U << REMOTE_OUTPUT_VARIABLE_ID) | (1U << REMOTE_TOKEN_VARIABLE_ID);
        for (uint32_t i = 0; i < context.arg->channelCount; ++i) {
            CCU_KERNEL_CHK_RET(ccu::NotifyWait(context.arg->channels[i], CHANNEL_NOTIFY_INDEX, ADDRESS_READY_MASK));
        }
        return CCU_SUCCESS;
    }

    CcuResult TransferDirect(AllGatherKernelContext &context)
    {
        ccu::LocalAddr source;
        source.addr = context.input;
        source.addr += context.inputOffset;
        source.token = context.tokens[context.arg->rankId];

        std::vector<ccu::RemoteAddr> remoteDestinations(context.arg->channelCount);
        for (uint32_t channelIndex = 0; channelIndex < context.arg->channelCount; ++channelIndex) {
            const uint32_t remoteRank = context.arg->remoteRanks[channelIndex];
            remoteDestinations[channelIndex].addr = context.outputs[remoteRank];
            remoteDestinations[channelIndex].addr += context.outputOffset;
            remoteDestinations[channelIndex].token = context.tokens[remoteRank];
        }

        ccu::Variable remoteBytes;
        remoteBytes = context.sliceBytes;
        if (context.arg->relaySource) {
            remoteBytes = context.directBytes;
        }
        if (context.arg->fullLocalTransfer) {
            remoteBytes = context.sliceBytes;
        }
        CCU_IF(remoteBytes != 0)
        {
            for (uint32_t channelIndex = 0; channelIndex < context.arg->channelCount; ++channelIndex) {
                const uint32_t remoteRank = context.arg->remoteRanks[channelIndex];
                const uint16_t completionMask = static_cast<uint16_t>(1U << remoteRank);
                CCU_KERNEL_CHK_RET(ccu::Write(context.arg->channels[channelIndex],
                    remoteDestinations[channelIndex], source, remoteBytes, context.completionEvent,
                    completionMask));
            }
        }

        uint16_t completionMask = 0;
        if (context.arg->relaySource && context.arg->relayPeerChannelIndex != INVALID_VALUE_RANKID) {
            const uint32_t relayChannel = context.arg->relayPeerChannelIndex;
            ccu::LocalAddr relaySource;
            relaySource.addr = source.addr;
            relaySource.addr += context.directBytes;
            relaySource.token = source.token;
            ccu::RemoteAddr relayDestination;
            relayDestination.addr = remoteDestinations[relayChannel].addr;
            relayDestination.addr += context.directBytes;
            relayDestination.token = remoteDestinations[relayChannel].token;
            const uint16_t relayMask = static_cast<uint16_t>(1U << context.arg->rankId);
            CCU_IF(context.relayBytes != 0)
            {
                CCU_KERNEL_CHK_RET(ccu::Write(context.arg->channels[relayChannel], relayDestination, relaySource,
                    context.relayBytes, context.completionEvent, relayMask));
            }
            CCU_ELSE
            {
                CCU_KERNEL_CHK_RET(ccu::EventRecord(context.completionEvent, relayMask));
            }
            completionMask |= relayMask;
        }

        if (context.arg->handleLocalCopy) {
            ccu::LocalAddr localDestination;
            localDestination.addr = context.outputs[context.arg->rankId];
            localDestination.addr += context.outputOffset;
            localDestination.token = context.tokens[context.arg->rankId];
            CCU_KERNEL_CHK_RET(GroupCopy(context, localDestination, source, context.groupCopySize));
        }

        for (uint32_t channelIndex = 0; channelIndex < context.arg->channelCount; ++channelIndex) {
            completionMask |= static_cast<uint16_t>(1U << context.arg->remoteRanks[channelIndex]);
        }
        CCU_KERNEL_CHK_RET(ccu::EventWait(context.completionEvent, completionMask));
        return CCU_SUCCESS;
    }

    CcuResult TransferRelay(AllGatherKernelContext &context)
    {
        if (!context.arg->forwardRelayData) {
            return CCU_SUCCESS;
        }

        ccu::LocalAddr source;
        source.addr = context.outputs[context.arg->rankId];
        source.addr += context.relayOutputOffset;
        source.addr += context.directBytes;
        source.token = context.tokens[context.arg->rankId];

        uint16_t completionMask = 0;
        CCU_IF(context.relayBytes != 0)
        {
            for (uint32_t channelIndex = 0; channelIndex < context.arg->channelCount; ++channelIndex) {
                const uint32_t remoteRank = context.arg->remoteRanks[channelIndex];
                ccu::RemoteAddr destination;
                destination.addr = context.outputs[remoteRank];
                destination.addr += context.relayOutputOffset;
                destination.addr += context.directBytes;
                destination.token = context.tokens[remoteRank];
                const uint16_t remoteMask = static_cast<uint16_t>(1U << remoteRank);
                CCU_KERNEL_CHK_RET(ccu::Write(context.arg->channels[channelIndex], destination, source,
                    context.relayBytes, context.completionEvent, remoteMask));
                completionMask |= remoteMask;
            }
        }
        CCU_KERNEL_CHK_RET(ccu::EventWait(context.completionEvent, completionMask));
        return CCU_SUCCESS;
    }

    CcuResult PostSynchronize(const AllGatherKernelContext &context)
    {
        constexpr uint32_t POST_SYNC_MASK = 1U << POST_SYNC_NOTIFY_ID;
        for (uint32_t i = 0; i < context.arg->channelCount; ++i) {
            CCU_KERNEL_CHK_RET(ccu::NotifyRecord(context.arg->channels[i], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
        }
        for (uint32_t i = 0; i < context.arg->channelCount; ++i) {
            CCU_KERNEL_CHK_RET(ccu::NotifyWait(context.arg->channels[i], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
        }
        return CCU_SUCCESS;
    }

CcuResult DirectKernel(CcuKernelArg arg, bool synchronizeAddresses)
{
    auto *kernelArg = static_cast<AllGatherKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize <= 1 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->channelCount == 0
        || kernelArg->channelCount >= kernelArg->rankSize) {
        HCCL_ERROR("Invalid CCU AllGather kernel resources");
        return CCU_E_INTERNAL;
    }

    AllGatherKernelContext context;
    context.arg = kernelArg;
    context.outputs.resize(kernelArg->rankSize);
    context.tokens.resize(kernelArg->rankSize);

    for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
        const uint32_t remoteRank = kernelArg->remoteRanks[channelIndex];
        if (remoteRank >= kernelArg->rankSize || remoteRank == kernelArg->rankId) {
            HCCL_ERROR("Invalid remote rank in CCU AllGather kernel resources");
            return CCU_E_INTERNAL;
        }
        context.outputs[remoteRank]
            = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIndex], REMOTE_OUTPUT_VARIABLE_ID);
        context.tokens[remoteRank]
            = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIndex], REMOTE_TOKEN_VARIABLE_ID);
    }

    CCU_KERNEL_CHK_RET(LoadArguments(context));
    if (synchronizeAddresses) {
        CCU_KERNEL_CHK_RET(SynchronizeAddresses(context));
    }
    CCU_KERNEL_CHK_RET(TransferOriginal(context));
    CCU_IF(context.lastSlice != 0)
    {
        CCU_KERNEL_CHK_RET(PostSynchronize(context));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    return DirectKernel(arg, true);
}

CcuResult CcuContinuationKernel(CcuKernelArg arg)
{
    return DirectKernel(arg, false);
}

CcuResult CcuRelayKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<AllGatherKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize <= 1 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->channelCount == 0
        || kernelArg->channelCount >= kernelArg->rankSize) {
        HCCL_ERROR("Invalid CCU AllGather relay kernel resources");
        return CCU_E_INTERNAL;
    }

    AllGatherKernelContext context;
    context.arg = kernelArg;
    context.outputs.resize(kernelArg->rankSize);
    context.tokens.resize(kernelArg->rankSize);

    for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
        const uint32_t remoteRank = kernelArg->remoteRanks[channelIndex];
        if (remoteRank >= kernelArg->rankSize || remoteRank == kernelArg->rankId) {
            HCCL_ERROR("Invalid remote rank in CCU AllGather relay kernel resources");
            return CCU_E_INTERNAL;
        }
        context.outputs[remoteRank]
            = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIndex], REMOTE_OUTPUT_VARIABLE_ID);
        context.tokens[remoteRank]
            = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIndex], REMOTE_TOKEN_VARIABLE_ID);
    }

    CCU_KERNEL_CHK_RET(LoadRelayArguments(context));
    CCU_IF(context.firstSlice != 0)
    {
        CCU_KERNEL_CHK_RET(SynchronizeAddresses(context));
    }
    CCU_IF(context.phase == DIRECT_PHASE)
    {
        CCU_KERNEL_CHK_RET(TransferDirect(context));
    }
    CCU_ELSE
    {
        CCU_KERNEL_CHK_RET(TransferRelay(context));
    }
    CCU_KERNEL_CHK_RET(PostSynchronize(context));
    return CCU_SUCCESS;
}

} // namespace ops_hccl
