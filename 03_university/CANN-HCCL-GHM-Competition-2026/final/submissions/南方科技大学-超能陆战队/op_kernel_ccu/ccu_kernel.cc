/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <memory>
#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call) \
    do { \
        CcuResult ccuResult = (call); \
        if (ccuResult != CCU_SUCCESS) { \
            return ccuResult; \
        } \
    } while (0)
#endif

namespace ops_hccl {
namespace {
    constexpr uint16_t INPUT_ADDRESS_RESOURCE_ID = 0;
    constexpr uint16_t INPUT_TOKEN_RESOURCE_ID = 1;
    constexpr uint16_t POST_SYNC_ID = 2;
    constexpr uint16_t CHANNEL_EVENT_INDEX = 0;
    constexpr uint16_t TRANSFER_EVENT_MASK = 1;
    constexpr uint32_t BUFFERED_REDUCE_MAX_SOURCES = 8;
    constexpr uint32_t BUFFERED_REDUCE_LOOP_COUNT = 16;
    constexpr uint32_t BUFFERED_REDUCE_INTERLEAVE = 8;
    constexpr uint64_t BUFFERED_REDUCE_SLICE_BYTES = 4ULL * 1024ULL;

    struct BufferedReduceGoSize {
        ccu::Variable enabled;
        ccu::Variable addressOffset;
        ccu::Variable loopIterations;
        ccu::Variable parallelConfig;
        ccu::Variable residualBytes;
    };

    struct BufferedReduceResources {
        bool initialized = false;
        bool loopsCreated = false;
        ccu::Array<ccu::Event> completedEvents{0};
        ccu::Array<ccu::CcuBuffer> buffers{0};
        std::array<std::vector<ccu::LocalAddr>, 2> loopSources;
        ccu::LocalAddr loopDestination[2];
        ccu::Variable loopLength[2];
        ccu::Variable loopParameters[2];
        std::array<std::unique_ptr<ccu::Func>, 2> bodies;
        std::array<std::unique_ptr<ccu::Loop>, 2> loops;
    };

    struct ReduceScatterContext {
        const CcuReduceScatterKernelArg *arg;
        std::vector<ccu::Variable> remoteInputAddresses;
        std::vector<ccu::Variable> remoteInputTokens;
        ccu::Variable localInputAddress;
        ccu::Variable localInputToken;
        ccu::Variable destinationAddress;
        ccu::Variable destinationToken;
        ccu::Variable scratchAddress;
        ccu::Variable scratchToken;
        ccu::Variable sourceBaseOffset;
        ccu::Variable chunkBytes;
        ccu::Variable pipelineFirstBytes;
        ccu::Variable pipelineSecondBytes;
        BufferedReduceGoSize firstReduce;
        BufferedReduceGoSize secondReduce;
        BufferedReduceResources bufferedReduce;
        ccu::Event event;
        ccu::Event pipelineEvent;
    };

    constexpr uint64_t BitMask(uint16_t bitCount)
    {
        return (uint64_t{1} << bitCount) - 1;
    }

    uint64_t PackLoopConfig(uint64_t addressOffset, uint64_t loopIterations)
    {
        return ((addressOffset & BitMask(32)) << 13) | (loopIterations & BitMask(13));
    }

    uint64_t PackParallelConfig(uint64_t repeatCount, uint64_t repeatLoopIndex, uint64_t totalLoopCount)
    {
        return ((repeatCount & BitMask(7)) << 55) | ((repeatLoopIndex & BitMask(7)) << 48)
            | ((totalLoopCount & BitMask(7)) << 41);
    }

    uint64_t PackOffsetConfig(uint64_t addressOffset, uint64_t bufferOffset, uint64_t eventOffset)
    {
        return ((addressOffset & BitMask(32)) << 21) | ((bufferOffset & BitMask(11)) << 10)
            | (eventOffset & BitMask(10));
    }

    CcuResult InitResources(ReduceScatterContext &context)
    {
        const auto *arg = context.arg;
        context.remoteInputAddresses.resize(arg->channelCount);
        context.remoteInputTokens.resize(arg->channelCount);

        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            context.remoteInputAddresses[channel]
                = ccu::GetResByChannel<ccu::Variable>(arg->channels[channel], INPUT_ADDRESS_RESOURCE_ID);
            context.remoteInputTokens[channel]
                = ccu::GetResByChannel<ccu::Variable>(arg->channels[channel], INPUT_TOKEN_RESOURCE_ID);
        }
        const uint32_t sourceCount = arg->channelCount + (arg->includeLocal ? 1U : 0U);
        if (sourceCount <= BUFFERED_REDUCE_MAX_SOURCES) {
            context.bufferedReduce.completedEvents = ccu::Array<ccu::Event>(BUFFERED_REDUCE_LOOP_COUNT);
            context.bufferedReduce.buffers = ccu::Array<ccu::CcuBuffer>(
                BUFFERED_REDUCE_LOOP_COUNT * BUFFERED_REDUCE_INTERLEAVE);
            context.bufferedReduce.initialized = true;
        }
        return CCU_SUCCESS;
    }

    CcuResult LoadBufferedReduceArgs(BufferedReduceGoSize &goSize, uint32_t &argumentIndex)
    {
        CCU_CHK_RET(ccu::LoadArg(goSize.enabled, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(goSize.addressOffset, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(goSize.loopIterations, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(goSize.parallelConfig, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(goSize.residualBytes, argumentIndex++));
        return CCU_SUCCESS;
    }

    CcuResult LoadArguments(ReduceScatterContext &context)
    {
        uint32_t argumentIndex = 0;
        CCU_CHK_RET(ccu::LoadArg(context.localInputAddress, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(context.destinationAddress, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(context.localInputToken, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(context.destinationToken, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(context.scratchAddress, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(context.scratchToken, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(context.sourceBaseOffset, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(context.chunkBytes, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(context.pipelineFirstBytes, argumentIndex++));
        CCU_CHK_RET(ccu::LoadArg(context.pipelineSecondBytes, argumentIndex++));
        CCU_CHK_RET(LoadBufferedReduceArgs(context.firstReduce, argumentIndex));
        CCU_CHK_RET(LoadBufferedReduceArgs(context.secondReduce, argumentIndex));
        return CCU_SUCCESS;
    }

    CcuResult PreSync(ReduceScatterContext &context)
    {
        const auto *arg = context.arg;
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channel], context.localInputAddress,
                INPUT_ADDRESS_RESOURCE_ID, CHANNEL_EVENT_INDEX, 1U << INPUT_ADDRESS_RESOURCE_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channel], context.localInputToken,
                INPUT_TOKEN_RESOURCE_ID, CHANNEL_EVENT_INDEX, 1U << INPUT_TOKEN_RESOURCE_ID));
        }

        constexpr uint32_t preSyncMask = (1U << INPUT_ADDRESS_RESOURCE_ID) | (1U << INPUT_TOKEN_RESOURCE_ID);
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[channel], CHANNEL_EVENT_INDEX, preSyncMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult PostSync(ReduceScatterContext &context)
    {
        const auto *arg = context.arg;
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channel], CHANNEL_EVENT_INDEX, 1U << POST_SYNC_ID));
        }
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[channel], CHANNEL_EVENT_INDEX, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult IssueSegmentReads(ReduceScatterContext &context, ccu::Variable scratchSegmentOffset,
        ccu::Variable sourceSegmentOffset, ccu::Variable segmentBytes, ccu::Event segmentEvent)
    {
        const auto *arg = context.arg;
        const uint32_t sourceCount = arg->channelCount + (arg->includeLocal ? 1U : 0U);
        ccu::LocalAddr scratchBase;
        scratchBase.addr = context.scratchAddress;
        scratchBase.addr += scratchSegmentOffset;
        for (uint32_t slot = 0; slot < arg->scratchSlotBase; ++slot) {
            scratchBase.addr += segmentBytes;
        }
        scratchBase.token = context.scratchToken;

        uint32_t sourceIndex = 0;
        uint32_t readEventMask = 0;
        for (uint32_t sourceRank = 0; sourceRank < arg->rankSize; ++sourceRank) {
            ccu::LocalAddr scratchSlot;
            scratchSlot.addr = scratchBase.addr;
            for (uint32_t slot = 0; slot < sourceIndex; ++slot) {
                scratchSlot.addr += segmentBytes;
            }
            scratchSlot.token = context.scratchToken;

            if (sourceRank == arg->rankId) {
                if (arg->includeLocal) {
                    ccu::LocalAddr source;
                    source.addr = context.localInputAddress;
                    source.addr += context.sourceBaseOffset;
                    source.addr += sourceSegmentOffset;
                    source.token = context.localInputToken;

                    const uint32_t eventMask = 1U << sourceIndex;
                    CCU_CHK_RET(ccu::LocalCopy(
                        scratchSlot, source, segmentBytes, segmentEvent, eventMask));
                    readEventMask |= eventMask;
                    ++sourceIndex;
                }
                continue;
            }

            for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
                if (arg->remoteRanks[channel] != sourceRank) {
                    continue;
                }

                ccu::RemoteAddr remoteSource;
                remoteSource.addr = context.remoteInputAddresses[channel];
                remoteSource.addr += context.sourceBaseOffset;
                remoteSource.addr += sourceSegmentOffset;
                remoteSource.token = context.remoteInputTokens[channel];

                const uint32_t eventMask = 1U << sourceIndex;
                CCU_CHK_RET(ccu::Read(arg->channels[channel], scratchSlot, remoteSource, segmentBytes,
                    segmentEvent, eventMask));
                readEventMask |= eventMask;
                ++sourceIndex;
                break;
            }
        }

        if (sourceIndex != sourceCount || readEventMask == 0) {
            return CCU_E_PARA;
        }
        return CCU_SUCCESS;
    }

    CcuResult CreateBufferedReduceLoops(ReduceScatterContext &context, uint32_t sourceCount)
    {
        auto &resources = context.bufferedReduce;
        if (!resources.initialized || sourceCount == 0 || sourceCount > BUFFERED_REDUCE_MAX_SOURCES) {
            return CCU_E_PARA;
        }
        if (resources.loopsCreated) {
            return CCU_SUCCESS;
        }

        for (uint32_t index = 0; index < 2; ++index) {
            resources.loopSources[index].resize(sourceCount);
            const uint32_t bufferBase = index * BUFFERED_REDUCE_INTERLEAVE;
            ccu::Event loopEvent = resources.completedEvents[index];
            resources.bodies[index].reset(new ccu::Func(
                [&resources, index, bufferBase, loopEvent, sourceCount]() {
                    for (uint32_t source = 0; source < sourceCount; ++source) {
                        ccu::LocalCopy(resources.buffers[bufferBase + source],
                            resources.loopSources[index][source], resources.loopLength[index], loopEvent,
                            1U << source);
                    }
                    ccu::EventWait(loopEvent, (1U << sourceCount) - 1U);

                    if (sourceCount > 1) {
                        std::vector<ccu::CcuBuffer> reduceBuffers;
                        reduceBuffers.reserve(sourceCount);
                        for (uint32_t source = 0; source < sourceCount; ++source) {
                            reduceBuffers.push_back(resources.buffers[bufferBase + source]);
                        }
                        ccu::LocalReduce(reduceBuffers.data(), sourceCount, HCCL_DATA_TYPE_FP32,
                            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, resources.loopLength[index], loopEvent,
                            TRANSFER_EVENT_MASK);
                        ccu::EventWait(loopEvent, TRANSFER_EVENT_MASK);
                    }

                    ccu::LocalCopy(resources.loopDestination[index], resources.buffers[bufferBase],
                        resources.loopLength[index], loopEvent, TRANSFER_EVENT_MASK);
                    ccu::EventWait(loopEvent, TRANSFER_EVENT_MASK);
                }));
            resources.loops[index].reset(
                new ccu::Loop(resources.loopParameters[index], *resources.bodies[index]));
        }
        resources.loopsCreated = true;
        return CCU_SUCCESS;
    }

    CcuResult RunBufferedReduce(ReduceScatterContext &context, ccu::LocalAddr destination,
        std::vector<ccu::LocalAddr> sources, BufferedReduceGoSize &goSize)
    {
        auto &resources = context.bufferedReduce;
        ccu::Variable loopConfig;
        ccu::Variable sliceBytes;
        ccu::Variable parallelConfig;
        ccu::Variable offsetConfig;

        CCU_IF(goSize.loopIterations != 0)
        {
            loopConfig = PackLoopConfig(
                BUFFERED_REDUCE_SLICE_BYTES * BUFFERED_REDUCE_LOOP_COUNT, 0);
            loopConfig += goSize.loopIterations;
            sliceBytes = BUFFERED_REDUCE_SLICE_BYTES;
            for (uint32_t source = 0; source < sources.size(); ++source) {
                resources.loopSources[0][source].addr = sources[source].addr;
                resources.loopSources[0][source].token = sources[source].token;
            }
            resources.loopDestination[0].addr = destination.addr;
            resources.loopDestination[0].token = destination.token;
            resources.loopLength[0] = sliceBytes;
            parallelConfig = PackParallelConfig(BUFFERED_REDUCE_LOOP_COUNT - 1, 0, 1);
            offsetConfig = PackOffsetConfig(
                BUFFERED_REDUCE_SLICE_BYTES, BUFFERED_REDUCE_INTERLEAVE, 1);
            resources.loopParameters[0] = loopConfig;
            std::vector<ccu::Loop> loops{*resources.loops[0]};
            ccu::LoopGroup group(parallelConfig, offsetConfig, BUFFERED_REDUCE_LOOP_COUNT, loops);
        }

        CCU_IF(goSize.parallelConfig != 0)
        {
            for (uint32_t source = 0; source < sources.size(); ++source) {
                sources[source].addr += goSize.addressOffset;
            }
            destination.addr += goSize.addressOffset;

            for (uint32_t source = 0; source < sources.size(); ++source) {
                resources.loopSources[0][source].addr = sources[source].addr;
                resources.loopSources[0][source].token = sources[source].token;
            }
            resources.loopDestination[0].addr = destination.addr;
            resources.loopDestination[0].token = destination.token;
            resources.loopLength[0] = goSize.residualBytes;

            for (uint32_t source = 0; source < sources.size(); ++source) {
                sources[source].addr += goSize.residualBytes;
            }
            destination.addr += goSize.residualBytes;
            sliceBytes = BUFFERED_REDUCE_SLICE_BYTES;
            for (uint32_t source = 0; source < sources.size(); ++source) {
                resources.loopSources[1][source].addr = sources[source].addr;
                resources.loopSources[1][source].token = sources[source].token;
            }
            resources.loopDestination[1].addr = destination.addr;
            resources.loopDestination[1].token = destination.token;
            resources.loopLength[1] = sliceBytes;

            resources.loopParameters[0] = PackLoopConfig(0, 1);
            resources.loopParameters[1] = PackLoopConfig(0, 1);
            offsetConfig = PackOffsetConfig(
                BUFFERED_REDUCE_SLICE_BYTES, BUFFERED_REDUCE_INTERLEAVE, 1);
            std::vector<ccu::Loop> loops{*resources.loops[0], *resources.loops[1]};
            ccu::LoopGroup group(
                goSize.parallelConfig, offsetConfig, BUFFERED_REDUCE_LOOP_COUNT, loops);
        }
        return CCU_SUCCESS;
    }

    CcuResult BufferedReduceSegment(ReduceScatterContext &context, ccu::Variable scratchSegmentOffset,
        ccu::Variable sourceSegmentOffset, ccu::Variable segmentBytes, ccu::Event segmentEvent,
        BufferedReduceGoSize &goSize)
    {
        const auto *arg = context.arg;
        const uint32_t sourceCount = arg->channelCount + (arg->includeLocal ? 1U : 0U);
        const uint32_t readEventMask = (1U << sourceCount) - 1U;
        CCU_CHK_RET(ccu::EventWait(segmentEvent, readEventMask));
        CCU_CHK_RET(CreateBufferedReduceLoops(context, sourceCount));

        ccu::LocalAddr scratchBase;
        scratchBase.addr = context.scratchAddress;
        scratchBase.addr += scratchSegmentOffset;
        for (uint32_t slot = 0; slot < arg->scratchSlotBase; ++slot) {
            scratchBase.addr += segmentBytes;
        }
        scratchBase.token = context.scratchToken;

        std::vector<ccu::LocalAddr> sources;
        sources.reserve(sourceCount);
        for (uint32_t source = 0; source < sourceCount; ++source) {
            ccu::LocalAddr sourceAddress;
            sourceAddress.addr = scratchBase.addr;
            for (uint32_t slot = 0; slot < source; ++slot) {
                sourceAddress.addr += segmentBytes;
            }
            sourceAddress.token = context.scratchToken;
            sources.push_back(sourceAddress);
        }

        ccu::LocalAddr destination;
        if (arg->includeLocal) {
            destination.addr = context.destinationAddress;
            destination.addr += sourceSegmentOffset;
            destination.token = context.destinationToken;
        } else {
            destination.addr = scratchBase.addr;
            destination.token = scratchBase.token;
        }
        return RunBufferedReduce(context, destination, sources, goSize);
    }

    CcuResult TreeReduceSegment(ReduceScatterContext &context, ccu::Variable scratchSegmentOffset,
        ccu::Variable sourceSegmentOffset, ccu::Variable segmentBytes, ccu::Event segmentEvent)
    {
        const auto *arg = context.arg;
        const uint32_t sourceCount = arg->channelCount + (arg->includeLocal ? 1U : 0U);
        const uint32_t readEventMask = (1U << sourceCount) - 1U;
        CCU_CHK_RET(ccu::EventWait(segmentEvent, readEventMask));

        ccu::LocalAddr scratchBase;
        scratchBase.addr = context.scratchAddress;
        scratchBase.addr += scratchSegmentOffset;
        for (uint32_t slot = 0; slot < arg->scratchSlotBase; ++slot) {
            scratchBase.addr += segmentBytes;
        }
        scratchBase.token = context.scratchToken;

        uint32_t remainingSources = sourceCount;
        while (remainingSources > 1) {
            const uint32_t reduceSources = remainingSources / 2;
            const uint32_t sourceSlot = remainingSources - reduceSources;

            ccu::LocalAddr reduceSource;
            reduceSource.addr = scratchBase.addr;
            for (uint32_t slot = 0; slot < sourceSlot; ++slot) {
                reduceSource.addr += segmentBytes;
            }
            reduceSource.token = context.scratchToken;

            ccu::Variable reduceBytes;
            reduceBytes = segmentBytes;
            for (uint32_t source = 1; source < reduceSources; ++source) {
                reduceBytes += segmentBytes;
            }

            CCU_CHK_RET(ccu::LocalReduce(scratchBase, reduceSource, reduceBytes, arg->dataType, arg->reduceOp,
                segmentEvent, TRANSFER_EVENT_MASK));
            CCU_CHK_RET(ccu::EventWait(segmentEvent, TRANSFER_EVENT_MASK));
            remainingSources -= reduceSources;
        }

        if (arg->includeLocal) {
            ccu::LocalAddr destination;
            destination.addr = context.destinationAddress;
            destination.addr += sourceSegmentOffset;
            destination.token = context.destinationToken;
            CCU_CHK_RET(
                ccu::LocalCopy(destination, scratchBase, segmentBytes, segmentEvent, TRANSFER_EVENT_MASK));
            CCU_CHK_RET(ccu::EventWait(segmentEvent, TRANSFER_EVENT_MASK));
        }
        return CCU_SUCCESS;
    }

    CcuResult ReduceSegment(ReduceScatterContext &context, ccu::Variable scratchSegmentOffset,
        ccu::Variable sourceSegmentOffset, ccu::Variable segmentBytes, ccu::Event segmentEvent,
        BufferedReduceGoSize &goSize)
    {
        const uint32_t sourceCount
            = context.arg->channelCount + (context.arg->includeLocal ? 1U : 0U);
        if (sourceCount > BUFFERED_REDUCE_MAX_SOURCES) {
            return TreeReduceSegment(
                context, scratchSegmentOffset, sourceSegmentOffset, segmentBytes, segmentEvent);
        }

        CCU_IF(goSize.enabled == 0)
        {
            CCU_CHK_RET(TreeReduceSegment(
                context, scratchSegmentOffset, sourceSegmentOffset, segmentBytes, segmentEvent));
        }
        CCU_IF(goSize.enabled != 0)
        {
            CCU_CHK_RET(BufferedReduceSegment(context, scratchSegmentOffset, sourceSegmentOffset,
                segmentBytes, segmentEvent, goSize));
        }
        return CCU_SUCCESS;
    }

    CcuResult ExecuteReduceScatter(ReduceScatterContext &context)
    {
        ccu::Variable zero;
        zero = 0;

        CCU_IF(context.pipelineFirstBytes == 0)
        {
            CCU_CHK_RET(IssueSegmentReads(context, zero, zero, context.chunkBytes, context.event));
            CCU_CHK_RET(ReduceSegment(
                context, zero, zero, context.chunkBytes, context.event, context.firstReduce));
        }

        CCU_IF(context.pipelineFirstBytes != 0)
        {
            ccu::Variable secondScratchOffset;
            secondScratchOffset = context.pipelineFirstBytes;
            for (uint32_t rank = 1; rank < context.arg->rankSize; ++rank) {
                secondScratchOffset += context.pipelineFirstBytes;
            }

            // Match the official mesh mem2mem schedule: issue the next read
            // batch before waiting for and reducing the current batch.
            CCU_CHK_RET(IssueSegmentReads(
                context, zero, zero, context.pipelineFirstBytes, context.event));
            CCU_CHK_RET(IssueSegmentReads(context, secondScratchOffset, context.pipelineFirstBytes,
                context.pipelineSecondBytes, context.pipelineEvent));
            CCU_CHK_RET(ReduceSegment(
                context, zero, zero, context.pipelineFirstBytes, context.event, context.firstReduce));
            CCU_CHK_RET(ReduceSegment(context, secondScratchOffset, context.pipelineFirstBytes,
                context.pipelineSecondBytes, context.pipelineEvent, context.secondReduce));
        }
        return CCU_SUCCESS;
    }
} // namespace

CcuResult CcuReduceScatterKernel(CcuKernelArg kernelArgument)
{
    auto *kernelArg = static_cast<CcuReduceScatterKernelArg *>(kernelArgument);
    if (kernelArg == nullptr || kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->channelCount >= kernelArg->rankSize
        || (!kernelArg->includeLocal && kernelArg->channelCount == 0)
        || kernelArg->scratchSlotBase >= kernelArg->rankSize
        || kernelArg->scratchSlotBase + kernelArg->channelCount + (kernelArg->includeLocal ? 1U : 0U)
            > kernelArg->rankSize) {
        return CCU_E_PARA;
    }

    ReduceScatterContext context;
    context.arg = kernelArg;

    CCU_CHK_RET(InitResources(context));
    CCU_CHK_RET(LoadArguments(context));
    CCU_CHK_RET(PreSync(context));
    CCU_CHK_RET(ExecuteReduceScatter(context));
    CCU_CHK_RET(PostSync(context));

    return CCU_SUCCESS;
}

CcuResult CcuCombineKernel(CcuKernelArg kernelArgument)
{
    auto *kernelArg = static_cast<CcuCombineKernelArg *>(kernelArgument);
    if (kernelArg == nullptr || kernelArg->channelCount != 1) {
        return CCU_E_PARA;
    }

    // Associate this local-only kernel with the same die as the primary
    // reduction kernel. The resource itself is not used for data transfer.
    (void)ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[0], INPUT_ADDRESS_RESOURCE_ID);

    ccu::Variable destinationAddress;
    ccu::Variable sourceAddress;
    ccu::Variable destinationToken;
    ccu::Variable sourceToken;
    ccu::Variable chunkBytes;
    uint32_t argumentIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(destinationAddress, argumentIndex++));
    CCU_CHK_RET(ccu::LoadArg(sourceAddress, argumentIndex++));
    CCU_CHK_RET(ccu::LoadArg(destinationToken, argumentIndex++));
    CCU_CHK_RET(ccu::LoadArg(sourceToken, argumentIndex++));
    CCU_CHK_RET(ccu::LoadArg(chunkBytes, argumentIndex++));

    ccu::LocalAddr destination;
    destination.addr = destinationAddress;
    destination.token = destinationToken;
    ccu::LocalAddr source;
    source.addr = sourceAddress;
    source.token = sourceToken;
    ccu::Event event;

    CCU_CHK_RET(ccu::LocalReduce(destination, source, chunkBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, event,
        TRANSFER_EVENT_MASK));
    CCU_CHK_RET(ccu::EventWait(event, TRANSFER_EVENT_MASK));
    return CCU_SUCCESS;
}

} // namespace ops_hccl
