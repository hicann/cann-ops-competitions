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
#include <memory>
#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "log.h"
#include "ccu_kernel.h"

namespace ops_hccl {
namespace {
    namespace ccu = ::AscendC::ccu;

    constexpr uint32_t INPUT_VAR_INDEX = 0;
    constexpr uint32_t TOKEN_VAR_INDEX = 1;
    constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
    constexpr uint16_t INPUT_READY_MASK = 1U << 0;
    constexpr uint16_t TOKEN_READY_MASK = 1U << 1;
    constexpr uint16_t POST_SYNC_MASK = 1U << 2;
    constexpr uint16_t NHR_READY_MASK = 1U << 3;
    constexpr uint16_t NHR_DONE_MASK = 1U << 4;
    constexpr uint32_t MAX_MS_PIPELINE_INPUTS = 8;
    constexpr uint32_t MS_PIPELINE_PARALLEL_TILES = 16;
    constexpr uint32_t MS_PIPELINE_BUFFER_INTERLEAVE = 8;
    constexpr uint32_t MS_PIPELINE_LOOP_COUNT = 2;
    constexpr uint32_t MS_PIPELINE_GRID_LOOP = 0;
    constexpr uint32_t MS_PIPELINE_TAIL_LOOP = 0;
    constexpr uint32_t MS_PIPELINE_FULL_TILE_LOOP = 1;
    constexpr uint64_t MS_PIPELINE_TILE_BYTES = 4096;
    static_assert(MAX_RANK_SIZE <= 16, "CCU event masks are 16-bit");
    static_assert(
        MAX_MS_PIPELINE_INPUTS <= MS_PIPELINE_BUFFER_INTERLEAVE, "each MS pipeline input needs a distinct CCU buffer");

    struct MsPipelineGroupOpSizeVars {
        ccu::Variable addrOffset;
        ccu::Variable loopParam;
        ccu::Variable parallelParam;
        ccu::Variable residual;
    };

    struct ReduceScatterContext {
        const CcuKernelArgReduceScatter *arg = nullptr;
        ccu::Variable localInput;
        ccu::Variable localOutput;
        ccu::Variable localInputToken;
        ccu::Variable localOutputToken;
        ccu::Variable remoteSliceOffset;
        ccu::Variable sliceSize;
        ccu::Variable executionMode;
        ccu::Variable scratchBase;
        ccu::Variable scratchToken;
        ccu::Variable scratchStride;
        MsPipelineGroupOpSizeVars msPipelineGroupSize;
        std::vector<ccu::Variable> nhrSliceOffsets;
        std::vector<ccu::Variable> nhrSliceSizes;
        std::vector<ccu::Variable> peerInputs;
        std::vector<ccu::Variable> peerTokens;
        std::vector<ccu::RemoteAddr> peerSources;
        std::vector<ccu::LocalAddr> scratchSlots;
        ccu::LocalAddr peerDestination;
        ccu::Event transferDone;
        ccu::Array<ccu::Event> pipelineEvents{0};
        ccu::Array<ccu::CcuBuffer> pipelineBuffers{0};
        std::unique_ptr<ccu::Func> pipelineBodies[MS_PIPELINE_LOOP_COUNT];
        std::unique_ptr<ccu::Loop> pipelineLoops[MS_PIPELINE_LOOP_COUNT];
        ccu::Variable pipelineLoopParams[MS_PIPELINE_LOOP_COUNT];
        ccu::LocalAddr pipelineDestinations[MS_PIPELINE_LOOP_COUNT];
        ccu::LocalAddr pipelineLocalSources[MS_PIPELINE_LOOP_COUNT];
        ccu::RemoteAddr pipelineRemoteSources[MS_PIPELINE_LOOP_COUNT][MAX_MS_PIPELINE_INPUTS];
        ccu::Variable pipelineLengths[MS_PIPELINE_LOOP_COUNT];
        std::unique_ptr<ccu::Func> scratchReduceBodies[MS_PIPELINE_LOOP_COUNT];
        std::unique_ptr<ccu::Loop> scratchReduceLoops[MS_PIPELINE_LOOP_COUNT];
        ccu::Variable scratchReduceLoopParams[MS_PIPELINE_LOOP_COUNT];
        ccu::LocalAddr scratchReduceDestinations[MS_PIPELINE_LOOP_COUNT];
        std::vector<ccu::LocalAddr> scratchReduceInputs[MS_PIPELINE_LOOP_COUNT];
        ccu::Variable scratchReduceLengths[MS_PIPELINE_LOOP_COUNT];
        bool pipelineResourcesAllocated = false;
        bool pipelineLoopsCreated = false;
        bool scratchReduceLoopsCreated = false;
    };

    constexpr uint64_t SetLowBits(uint16_t end)
    {
        return (uint64_t{1} << (end + 1)) - uint64_t{1};
    }

    constexpr uint64_t GetLoopParam(uint64_t loopContextId, uint64_t addressOffset, uint64_t loopIterNum)
    {
        constexpr uint16_t CONTEXT_ID_BIT_END = 8;
        constexpr uint16_t CONTEXT_ID_SHIFT = 45;
        constexpr uint16_t ADDRESS_OFFSET_BIT_END = 32;
        constexpr uint16_t ADDRESS_OFFSET_SHIFT = 13;
        constexpr uint16_t LOOP_NUM_BIT_END = 13;
        return ((loopContextId & SetLowBits(CONTEXT_ID_BIT_END)) << CONTEXT_ID_SHIFT)
               | ((addressOffset & SetLowBits(ADDRESS_OFFSET_BIT_END)) << ADDRESS_OFFSET_SHIFT)
               | (loopIterNum & SetLowBits(LOOP_NUM_BIT_END));
    }

    constexpr uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
    {
        constexpr uint16_t REPEAT_BIT_END = 7;
        constexpr uint16_t REPEAT_NUM_SHIFT = 55;
        constexpr uint16_t REPEAT_LOOP_BIT_END = 7;
        constexpr uint16_t REPEAT_LOOP_SHIFT = 48;
        constexpr uint16_t TOTAL_LOOP_BIT_END = 7;
        constexpr uint16_t TOTAL_LOOP_SHIFT = 41;
        return ((repeatNum & SetLowBits(REPEAT_BIT_END)) << REPEAT_NUM_SHIFT)
               | ((repeatLoopIndex & SetLowBits(REPEAT_LOOP_BIT_END)) << REPEAT_LOOP_SHIFT)
               | ((totalLoopNum & SetLowBits(TOTAL_LOOP_BIT_END)) << TOTAL_LOOP_SHIFT);
    }

    constexpr uint64_t GetOffsetParam(uint64_t addressOffset, uint64_t bufferOffset, uint64_t eventOffset)
    {
        constexpr uint16_t ADDRESS_OFFSET_BIT_END = 32;
        constexpr uint16_t ADDRESS_OFFSET_SHIFT = 21;
        constexpr uint16_t BUFFER_OFFSET_BIT_END = 11;
        constexpr uint16_t BUFFER_OFFSET_SHIFT = 10;
        constexpr uint16_t EVENT_OFFSET_BIT_END = 10;
        return ((addressOffset & SetLowBits(ADDRESS_OFFSET_BIT_END)) << ADDRESS_OFFSET_SHIFT)
               | ((bufferOffset & SetLowBits(BUFFER_OFFSET_BIT_END)) << BUFFER_OFFSET_SHIFT)
               | (eventOffset & SetLowBits(EVENT_OFFSET_BIT_END));
    }

    inline CcuResult InitRemoteResources(ReduceScatterContext &ctx)
    {
        ctx.peerInputs.resize(ctx.arg->channelCount);
        ctx.peerTokens.resize(ctx.arg->channelCount);
        ctx.peerSources.resize(ctx.arg->channelCount);
        ctx.scratchSlots.resize(ctx.arg->channelCount);
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            ctx.peerInputs[index] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[index], INPUT_VAR_INDEX);
            ctx.peerTokens[index] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[index], TOKEN_VAR_INDEX);
        }
        return CCU_SUCCESS;
    }

    inline CcuResult LoadTaskArgs(ReduceScatterContext &ctx)
    {
        uint32_t argIndex = 0;
        CcuResult result = ccu::LoadArg(ctx.localInput, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.localOutput, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.localInputToken, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.localOutputToken, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.remoteSliceOffset, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.sliceSize, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.executionMode, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.scratchBase, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.scratchToken, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.scratchStride, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.msPipelineGroupSize.addrOffset, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.msPipelineGroupSize.loopParam, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.msPipelineGroupSize.parallelParam, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        return ccu::LoadArg(ctx.msPipelineGroupSize.residual, argIndex);
    }

    inline CcuResult LoadNhrTaskArgs(ReduceScatterContext &ctx)
    {
        if (ctx.arg->axisRankSize <= 1 || ctx.arg->axisRankSize > MAX_RANK_SIZE
            || ctx.arg->axisRankIndex >= ctx.arg->axisRankSize
            || ctx.arg->channelCount + 1 != ctx.arg->axisRankSize) {
            return CCU_E_PARA;
        }

        uint32_t argIndex = 0;
        CcuResult result = ccu::LoadArg(ctx.localInput, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.localOutput, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.localInputToken, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ccu::LoadArg(ctx.executionMode, argIndex++);
        if (result != CCU_SUCCESS) {
            return result;
        }

        ctx.nhrSliceOffsets.resize(ctx.arg->axisRankSize);
        ctx.nhrSliceSizes.resize(ctx.arg->axisRankSize);
        for (uint32_t rankIndex = 0; rankIndex < ctx.arg->axisRankSize; ++rankIndex) {
            result = ccu::LoadArg(ctx.nhrSliceOffsets[rankIndex], argIndex++);
            if (result != CCU_SUCCESS) {
                return result;
            }
        }
        for (uint32_t rankIndex = 0; rankIndex < ctx.arg->axisRankSize; ++rankIndex) {
            result = ccu::LoadArg(ctx.nhrSliceSizes[rankIndex], argIndex++);
            if (result != CCU_SUCCESS) {
                return result;
            }
        }
        return CCU_SUCCESS;
    }

    inline void PreparePeerSources(ReduceScatterContext &ctx)
    {
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            ctx.peerSources[index].addr = ctx.peerInputs[index];
            ctx.peerSources[index].addr += ctx.remoteSliceOffset;
            ctx.peerSources[index].token = ctx.peerTokens[index];
        }
    }

    inline void PreparePeerDestination(ReduceScatterContext &ctx)
    {
        ctx.peerDestination.addr = ctx.localOutput;
        ctx.peerDestination.token = ctx.localOutputToken;
    }

    inline CcuResult InitializeLocalContribution(ReduceScatterContext &ctx)
    {
        ccu::LocalAddr source;
        source.addr = ctx.localInput;
        source.addr += ctx.remoteSliceOffset;
        source.token = ctx.localInputToken;

        CcuResult result = ccu::LocalCopy(ctx.peerDestination, source, ctx.sliceSize, ctx.transferDone, 1);
        if (result != CCU_SUCCESS) {
            return result;
        }
        return ccu::EventWait(ctx.transferDone, 1);
    }

    inline void PrepareScratchSlots(ReduceScatterContext &ctx)
    {
        ccu::Variable scratchOffset;
        scratchOffset = 0;
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            ctx.scratchSlots[index].addr = ctx.scratchBase;
            ctx.scratchSlots[index].addr += scratchOffset;
            ctx.scratchSlots[index].token = ctx.scratchToken;
            scratchOffset += ctx.scratchStride;
        }
    }

    inline CcuResult PreSync(ReduceScatterContext &ctx)
    {
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            CcuResult result = ccu::WriteVariableWithNotify(
                ctx.arg->channels[index], ctx.localInput, INPUT_VAR_INDEX, CHANNEL_NOTIFY_INDEX, INPUT_READY_MASK);
            if (result != CCU_SUCCESS) {
                return result;
            }
            result = ccu::WriteVariableWithNotify(
                ctx.arg->channels[index], ctx.localInputToken, TOKEN_VAR_INDEX, CHANNEL_NOTIFY_INDEX, TOKEN_READY_MASK);
            if (result != CCU_SUCCESS) {
                return result;
            }
        }

        constexpr uint16_t readyMask = INPUT_READY_MASK | TOKEN_READY_MASK;
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            CcuResult result = ccu::NotifyWait(ctx.arg->channels[index], CHANNEL_NOTIFY_INDEX, readyMask);
            if (result != CCU_SUCCESS) {
                return result;
            }
        }
        return CCU_SUCCESS;
    }

    inline CcuResult ReadPeerContributionDirect(ReduceScatterContext &ctx, uint32_t channelIndex, bool reduce)
    {
        CcuResult result;
        if (reduce) {
            result
                = ccu::ReadReduce(ctx.arg->channels[channelIndex], ctx.peerDestination, ctx.peerSources[channelIndex],
                    ctx.sliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.transferDone, 1);
        } else {
            result = ccu::Read(ctx.arg->channels[channelIndex], ctx.peerDestination, ctx.peerSources[channelIndex],
                ctx.sliceSize, ctx.transferDone, 1);
        }
        if (result != CCU_SUCCESS) {
            return result;
        }
        return ccu::EventWait(ctx.transferDone, 1);
    }

    inline CcuResult ReducePeerContributionsDirect(ReduceScatterContext &ctx)
    {
        uint32_t firstReduceIndex = 0;
        if (!ctx.arg->destinationInitialized) {
            CcuResult result = ReadPeerContributionDirect(ctx, 0, false);
            if (result != CCU_SUCCESS) {
                return result;
            }
            firstReduceIndex = 1;
        }

        for (uint32_t index = firstReduceIndex; index < ctx.arg->channelCount; ++index) {
            CcuResult result = ReadPeerContributionDirect(ctx, index, true);
            if (result != CCU_SUCCESS) {
                return result;
            }
        }
        return CCU_SUCCESS;
    }

    inline CcuResult ReadPeerContributionsToScratch(ReduceScatterContext &ctx)
    {
        uint16_t waitMask = 0;
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            const uint16_t eventMask = static_cast<uint16_t>(1U << index);
            CcuResult result = ccu::Read(ctx.arg->channels[index], ctx.scratchSlots[index], ctx.peerSources[index],
                ctx.sliceSize, ctx.transferDone, eventMask);
            if (result != CCU_SUCCESS) {
                return result;
            }
            waitMask |= eventMask;
        }
        return ccu::EventWait(ctx.transferDone, waitMask);
    }

    inline CcuResult ReduceScratchContributions(ReduceScatterContext &ctx)
    {
        uint32_t firstScratchIndex = 0;
        if (!ctx.arg->destinationInitialized) {
            firstScratchIndex = 1;
        }
        for (uint32_t index = firstScratchIndex; index < ctx.arg->channelCount; ++index) {
            CcuResult result = ccu::LocalReduce(ctx.peerDestination, ctx.scratchSlots[index], ctx.sliceSize,
                ctx.arg->dataType, ctx.arg->reduceOp, ctx.transferDone, 1);
            if (result != CCU_SUCCESS) {
                return result;
            }
            result = ccu::EventWait(ctx.transferDone, 1);
            if (result != CCU_SUCCESS) {
                return result;
            }
        }
        return CCU_SUCCESS;
    }

    inline CcuResult ReducePeerContributionsWithScratch(ReduceScatterContext &ctx)
    {
        CcuResult result = ReadPeerContributionsToScratch(ctx);
        if (result != CCU_SUCCESS) {
            return result;
        }
        return ReduceScratchContributions(ctx);
    }

    inline CcuResult AllocateMsPipelineResources(ReduceScatterContext &ctx)
    {
        if (ctx.pipelineResourcesAllocated) {
            return CCU_SUCCESS;
        }
        ctx.pipelineEvents = ccu::Array<ccu::Event>(MS_PIPELINE_PARALLEL_TILES);
        ctx.pipelineBuffers = ccu::Array<ccu::CcuBuffer>(MS_PIPELINE_PARALLEL_TILES * MS_PIPELINE_BUFFER_INTERLEAVE);
        ctx.pipelineResourcesAllocated = true;
        return CCU_SUCCESS;
    }

    inline CcuResult CreateScratchReduceLoops(ReduceScatterContext &ctx)
    {
        CcuResult result = AllocateMsPipelineResources(ctx);
        if (result != CCU_SUCCESS) {
            return result;
        }
        if (ctx.scratchReduceLoopsCreated) {
            return CCU_SUCCESS;
        }

        const uint32_t inputCount = ctx.arg->channelCount + 1;
        if (inputCount <= 1 || inputCount > MAX_MS_PIPELINE_INPUTS) {
            return CCU_E_PARA;
        }
        for (uint32_t loopIndex = 0; loopIndex < MS_PIPELINE_LOOP_COUNT; ++loopIndex) {
            const uint32_t bufferBase = loopIndex * MS_PIPELINE_BUFFER_INTERLEAVE;
            const ccu::Event pipelineEvent = ctx.pipelineEvents[loopIndex];
            ctx.scratchReduceInputs[loopIndex].resize(inputCount);
            ctx.scratchReduceBodies[loopIndex].reset(
                new ccu::Func([&ctx, loopIndex, inputCount, bufferBase, pipelineEvent]() {
                    uint16_t waitMask = 0;
                    for (uint32_t inputIndex = 0; inputIndex < inputCount; ++inputIndex) {
                        const uint16_t copyMask = static_cast<uint16_t>(1U << inputIndex);
                        ccu::LocalCopy(ctx.pipelineBuffers[bufferBase + inputIndex],
                            ctx.scratchReduceInputs[loopIndex][inputIndex], ctx.scratchReduceLengths[loopIndex],
                            pipelineEvent, copyMask);
                        waitMask |= copyMask;
                    }
                    ccu::EventWait(pipelineEvent, waitMask);
                    ccu::LocalReduce(&ctx.pipelineBuffers[bufferBase], inputCount, ctx.arg->dataType,
                        ctx.arg->dataType, ctx.arg->reduceOp, ctx.scratchReduceLengths[loopIndex], pipelineEvent, 1);
                    ccu::EventWait(pipelineEvent, 1);
                    ccu::LocalCopy(ctx.scratchReduceDestinations[loopIndex], ctx.pipelineBuffers[bufferBase],
                        ctx.scratchReduceLengths[loopIndex], pipelineEvent, 1);
                    ccu::EventWait(pipelineEvent, 1);
                }));
            ctx.scratchReduceLoops[loopIndex].reset(
                new ccu::Loop(ctx.scratchReduceLoopParams[loopIndex], *ctx.scratchReduceBodies[loopIndex]));
        }
        ctx.scratchReduceLoopsCreated = true;
        return CCU_SUCCESS;
    }

    inline void ConfigureScratchReduceLoop(
        ReduceScatterContext &ctx, uint32_t loopIndex, ccu::Variable offset, ccu::Variable length)
    {
        ctx.scratchReduceDestinations[loopIndex].addr = ctx.peerDestination.addr;
        ctx.scratchReduceDestinations[loopIndex].addr += offset;
        ctx.scratchReduceDestinations[loopIndex].token = ctx.peerDestination.token;

        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ctx.scratchReduceInputs[loopIndex][channelIndex].addr = ctx.scratchSlots[channelIndex].addr;
            ctx.scratchReduceInputs[loopIndex][channelIndex].addr += offset;
            ctx.scratchReduceInputs[loopIndex][channelIndex].token = ctx.scratchSlots[channelIndex].token;
        }
        const uint32_t localInputIndex = ctx.arg->channelCount;
        ctx.scratchReduceInputs[loopIndex][localInputIndex].addr = ctx.localInput;
        ctx.scratchReduceInputs[loopIndex][localInputIndex].addr += ctx.remoteSliceOffset;
        ctx.scratchReduceInputs[loopIndex][localInputIndex].addr += offset;
        ctx.scratchReduceInputs[loopIndex][localInputIndex].token = ctx.localInputToken;
        ctx.scratchReduceLengths[loopIndex] = length;
    }

    inline CcuResult ReduceScratchInputsWithCcuBuffers(ReduceScatterContext &ctx)
    {
        CcuResult result = CreateScratchReduceLoops(ctx);
        if (result != CCU_SUCCESS) {
            return result;
        }

        ccu::Variable tileLength;
        tileLength = MS_PIPELINE_TILE_BYTES;
        CCU_IF(ctx.msPipelineGroupSize.loopParam != 0)
        {
            ccu::Variable zeroOffset;
            zeroOffset = 0;
            ConfigureScratchReduceLoop(ctx, MS_PIPELINE_GRID_LOOP, zeroOffset, tileLength);
            ctx.scratchReduceLoopParams[MS_PIPELINE_GRID_LOOP] = ctx.msPipelineGroupSize.loopParam;

            ccu::Variable parallelConfig;
            parallelConfig = GetParallelParam(MS_PIPELINE_PARALLEL_TILES - 1, 0, 1);
            ccu::Variable offsetConfig;
            offsetConfig = GetOffsetParam(MS_PIPELINE_TILE_BYTES, MS_PIPELINE_BUFFER_INTERLEAVE, 1);
            std::vector<ccu::Loop> loops{*ctx.scratchReduceLoops[MS_PIPELINE_GRID_LOOP]};
            ccu::LoopGroup group(parallelConfig, offsetConfig, MS_PIPELINE_PARALLEL_TILES, loops);
        }

        CCU_IF(ctx.msPipelineGroupSize.parallelParam != 0)
        {
            ConfigureScratchReduceLoop(
                ctx, MS_PIPELINE_TAIL_LOOP, ctx.msPipelineGroupSize.addrOffset, ctx.msPipelineGroupSize.residual);
            ccu::Variable fullTileOffset;
            fullTileOffset = ctx.msPipelineGroupSize.addrOffset;
            fullTileOffset += ctx.msPipelineGroupSize.residual;
            ConfigureScratchReduceLoop(ctx, MS_PIPELINE_FULL_TILE_LOOP, fullTileOffset, tileLength);

            ctx.scratchReduceLoopParams[MS_PIPELINE_TAIL_LOOP] = GetLoopParam(0, 0, 1);
            ctx.scratchReduceLoopParams[MS_PIPELINE_FULL_TILE_LOOP] = GetLoopParam(0, 0, 1);
            ccu::Variable offsetConfig;
            offsetConfig = GetOffsetParam(MS_PIPELINE_TILE_BYTES, MS_PIPELINE_BUFFER_INTERLEAVE, 1);
            std::vector<ccu::Loop> loops{
                *ctx.scratchReduceLoops[MS_PIPELINE_TAIL_LOOP],
                *ctx.scratchReduceLoops[MS_PIPELINE_FULL_TILE_LOOP],
            };
            ccu::LoopGroup group(
                ctx.msPipelineGroupSize.parallelParam, offsetConfig, MS_PIPELINE_PARALLEL_TILES, loops);
        }
        return CCU_SUCCESS;
    }

    inline CcuResult ReduceOmnipipeMeshWithScratch(ReduceScatterContext &ctx)
    {
        CcuResult result = ReadPeerContributionsToScratch(ctx);
        if (result != CCU_SUCCESS) {
            return result;
        }
        return ReduceScratchInputsWithCcuBuffers(ctx);
    }

    inline CcuResult CreateMsPipelineLoops(ReduceScatterContext &ctx)
    {
        CcuResult result = AllocateMsPipelineResources(ctx);
        if (result != CCU_SUCCESS) {
            return result;
        }
        if (ctx.pipelineLoopsCreated) {
            return CCU_SUCCESS;
        }

        const bool pipelineIncludesLocal
            = ctx.arg->destinationInitialized && ctx.arg->channelCount < MAX_MS_PIPELINE_INPUTS;
        const uint32_t inputCount = ctx.arg->channelCount + (pipelineIncludesLocal ? 1U : 0U);
        if (inputCount == 0 || inputCount > MAX_MS_PIPELINE_INPUTS) {
            return CCU_E_PARA;
        }

        for (uint32_t loopIndex = 0; loopIndex < MS_PIPELINE_LOOP_COUNT; ++loopIndex) {
            const uint32_t bufferBase = loopIndex * MS_PIPELINE_BUFFER_INTERLEAVE;
            const ccu::Event pipelineEvent = ctx.pipelineEvents[loopIndex];
            ctx.pipelineBodies[loopIndex].reset(
                new ccu::Func([&ctx, loopIndex, pipelineIncludesLocal, inputCount, bufferBase, pipelineEvent]() {
                    uint32_t bufferIndex = 0;
                    uint16_t waitMask = 0;
                    if (pipelineIncludesLocal) {
                        const uint16_t localMask = static_cast<uint16_t>(1U << bufferIndex);
                        ccu::LocalCopy(ctx.pipelineBuffers[bufferBase + bufferIndex],
                            ctx.pipelineLocalSources[loopIndex], ctx.pipelineLengths[loopIndex], pipelineEvent,
                            localMask);
                        waitMask |= localMask;
                        ++bufferIndex;
                    }
                    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
                        const uint16_t readMask = static_cast<uint16_t>(1U << bufferIndex);
                        ccu::Read(ctx.arg->channels[channelIndex], ctx.pipelineBuffers[bufferBase + bufferIndex],
                            ctx.pipelineRemoteSources[loopIndex][channelIndex], ctx.pipelineLengths[loopIndex],
                            pipelineEvent, readMask);
                        waitMask |= readMask;
                        ++bufferIndex;
                    }
                    ccu::EventWait(pipelineEvent, waitMask);

                    if (inputCount > 1) {
                        ccu::LocalReduce(&ctx.pipelineBuffers[bufferBase], inputCount, ctx.arg->dataType,
                            ctx.arg->dataType, ctx.arg->reduceOp, ctx.pipelineLengths[loopIndex], pipelineEvent, 1);
                        ccu::EventWait(pipelineEvent, 1);
                    }

                    ccu::LocalCopy(ctx.pipelineDestinations[loopIndex], ctx.pipelineBuffers[bufferBase],
                        ctx.pipelineLengths[loopIndex], pipelineEvent, 1);
                    ccu::EventWait(pipelineEvent, 1);
                }));
            ctx.pipelineLoops[loopIndex].reset(
                new ccu::Loop(ctx.pipelineLoopParams[loopIndex], *ctx.pipelineBodies[loopIndex]));
        }
        ctx.pipelineLoopsCreated = true;
        return CCU_SUCCESS;
    }

    inline void ConfigureMsPipelineLoop(
        ReduceScatterContext &ctx, uint32_t loopIndex, ccu::Variable offset, ccu::Variable length)
    {
        ctx.pipelineDestinations[loopIndex].addr = ctx.peerDestination.addr;
        ctx.pipelineDestinations[loopIndex].addr += offset;
        ctx.pipelineDestinations[loopIndex].token = ctx.peerDestination.token;

        ctx.pipelineLocalSources[loopIndex].addr = ctx.localInput;
        ctx.pipelineLocalSources[loopIndex].addr += ctx.remoteSliceOffset;
        ctx.pipelineLocalSources[loopIndex].addr += offset;
        ctx.pipelineLocalSources[loopIndex].token = ctx.localInputToken;

        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ctx.pipelineRemoteSources[loopIndex][channelIndex].addr = ctx.peerSources[channelIndex].addr;
            ctx.pipelineRemoteSources[loopIndex][channelIndex].addr += offset;
            ctx.pipelineRemoteSources[loopIndex][channelIndex].token = ctx.peerSources[channelIndex].token;
        }
        ctx.pipelineLengths[loopIndex] = length;
    }

    inline CcuResult ReducePeerContributionsWithMsPipeline(ReduceScatterContext &ctx)
    {
        CcuResult result = CreateMsPipelineLoops(ctx);
        if (result != CCU_SUCCESS) {
            return result;
        }

        ccu::Variable tileLength;
        tileLength = MS_PIPELINE_TILE_BYTES;

        CCU_IF(ctx.msPipelineGroupSize.loopParam != 0)
        {
            ccu::Variable zeroOffset;
            zeroOffset = 0;
            ConfigureMsPipelineLoop(ctx, MS_PIPELINE_GRID_LOOP, zeroOffset, tileLength);
            ctx.pipelineLoopParams[MS_PIPELINE_GRID_LOOP] = ctx.msPipelineGroupSize.loopParam;

            ccu::Variable parallelConfig;
            parallelConfig = GetParallelParam(MS_PIPELINE_PARALLEL_TILES - 1, 0, 1);
            ccu::Variable offsetConfig;
            offsetConfig = GetOffsetParam(MS_PIPELINE_TILE_BYTES, MS_PIPELINE_BUFFER_INTERLEAVE, 1);
            std::vector<ccu::Loop> loops{*ctx.pipelineLoops[MS_PIPELINE_GRID_LOOP]};
            ccu::LoopGroup group(parallelConfig, offsetConfig, MS_PIPELINE_PARALLEL_TILES, loops);
        }

        CCU_IF(ctx.msPipelineGroupSize.parallelParam != 0)
        {
            ConfigureMsPipelineLoop(
                ctx, MS_PIPELINE_TAIL_LOOP, ctx.msPipelineGroupSize.addrOffset, ctx.msPipelineGroupSize.residual);
            ccu::Variable fullTileOffset;
            fullTileOffset = ctx.msPipelineGroupSize.addrOffset;
            fullTileOffset += ctx.msPipelineGroupSize.residual;
            ConfigureMsPipelineLoop(ctx, MS_PIPELINE_FULL_TILE_LOOP, fullTileOffset, tileLength);

            ctx.pipelineLoopParams[MS_PIPELINE_TAIL_LOOP] = GetLoopParam(0, 0, 1);
            ctx.pipelineLoopParams[MS_PIPELINE_FULL_TILE_LOOP] = GetLoopParam(0, 0, 1);
            ccu::Variable offsetConfig;
            offsetConfig = GetOffsetParam(MS_PIPELINE_TILE_BYTES, MS_PIPELINE_BUFFER_INTERLEAVE, 1);
            std::vector<ccu::Loop> loops{
                *ctx.pipelineLoops[MS_PIPELINE_TAIL_LOOP],
                *ctx.pipelineLoops[MS_PIPELINE_FULL_TILE_LOOP],
            };
            ccu::LoopGroup group(
                ctx.msPipelineGroupSize.parallelParam, offsetConfig, MS_PIPELINE_PARALLEL_TILES, loops);
        }
        return CCU_SUCCESS;
    }

    inline CcuResult MergeLocalContributionAfterMsPipeline(ReduceScatterContext &ctx)
    {
        ccu::LocalAddr source;
        source.addr = ctx.localInput;
        source.addr += ctx.remoteSliceOffset;
        source.token = ctx.localInputToken;

        CcuResult result = ccu::LocalReduce(
            ctx.peerDestination, source, ctx.sliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.transferDone, 1);
        if (result != CCU_SUCCESS) {
            return result;
        }
        return ccu::EventWait(ctx.transferDone, 1);
    }

    inline CcuResult PostSync(ReduceScatterContext &ctx)
    {
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            CcuResult result = ccu::NotifyRecord(ctx.arg->channels[index], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK);
            if (result != CCU_SUCCESS) {
                return result;
            }
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            CcuResult result = ccu::NotifyWait(ctx.arg->channels[index], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK);
            if (result != CCU_SUCCESS) {
                return result;
            }
        }
        return CCU_SUCCESS;
    }

    uint32_t FindChannelForSubRank(const ReduceScatterContext &ctx, uint32_t subRank)
    {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (ctx.arg->peerSubRanks[channelIndex] == subRank) {
                return channelIndex;
            }
        }
        return ctx.arg->channelCount;
    }

    inline CcuResult ReduceScatterNhr(ReduceScatterContext &ctx)
    {
        uint32_t step = 0;
        for (uint32_t remaining = ctx.arg->axisRankSize - 1; remaining != 0; remaining >>= 1, ++step) {
            const uint32_t deltaRank = 1U << step;
            const uint32_t sendTo
                = (ctx.arg->axisRankIndex + ctx.arg->axisRankSize - deltaRank) % ctx.arg->axisRankSize;
            const uint32_t recvFrom = (ctx.arg->axisRankIndex + deltaRank) % ctx.arg->axisRankSize;
            const uint32_t sendChannelIndex = FindChannelForSubRank(ctx, sendTo);
            const uint32_t recvChannelIndex = FindChannelForSubRank(ctx, recvFrom);
            if (sendChannelIndex >= ctx.arg->channelCount || recvChannelIndex >= ctx.arg->channelCount) {
                return CCU_E_PARA;
            }
            const ChannelHandle sendChannel = ctx.arg->channels[sendChannelIndex];
            const ChannelHandle recvChannel = ctx.arg->channels[recvChannelIndex];

            if (step != 0) {
                CcuResult result = ccu::NotifyRecord(recvChannel, CHANNEL_NOTIFY_INDEX, NHR_READY_MASK);
                if (result != CCU_SUCCESS) {
                    return result;
                }
                result = ccu::NotifyWait(sendChannel, CHANNEL_NOTIFY_INDEX, NHR_READY_MASK);
                if (result != CCU_SUCCESS) {
                    return result;
                }
            }

            const uint32_t deltaSlice = 1U << (step + 1);
            const uint32_t sliceCount = (ctx.arg->axisRankSize - 1 + deltaRank) / deltaSlice;
            uint32_t recvSliceIndex = ctx.arg->axisRankIndex;
            for (uint32_t slice = 0; slice < sliceCount; ++slice) {
                ccu::LocalAddr destination;
                destination.addr = ctx.localInput;
                destination.addr += ctx.nhrSliceOffsets[recvSliceIndex];
                destination.token = ctx.localInputToken;

                ccu::RemoteAddr source;
                source.addr = ctx.peerInputs[recvChannelIndex];
                source.addr += ctx.nhrSliceOffsets[recvSliceIndex];
                source.token = ctx.peerTokens[recvChannelIndex];

                CCU_IF(ctx.nhrSliceSizes[recvSliceIndex] != 0)
                {
                    CcuResult result = ccu::ReadReduce(recvChannel, destination, source,
                        ctx.nhrSliceSizes[recvSliceIndex], ctx.arg->dataType, ctx.arg->reduceOp, ctx.transferDone, 1);
                    if (result != CCU_SUCCESS) {
                        return result;
                    }
                }
                CCU_IF(ctx.nhrSliceSizes[recvSliceIndex] == 0)
                {
                    CcuResult result = ccu::EventRecord(ctx.transferDone, 1);
                    if (result != CCU_SUCCESS) {
                        return result;
                    }
                }
                CcuResult result = ccu::EventWait(ctx.transferDone, 1);
                if (result != CCU_SUCCESS) {
                    return result;
                }
                recvSliceIndex
                    = (recvSliceIndex + ctx.arg->axisRankSize - deltaSlice) % ctx.arg->axisRankSize;
            }

            CcuResult result = ccu::NotifyRecord(sendChannel, CHANNEL_NOTIFY_INDEX, NHR_DONE_MASK);
            if (result != CCU_SUCCESS) {
                return result;
            }
            result = ccu::NotifyWait(recvChannel, CHANNEL_NOTIFY_INDEX, NHR_DONE_MASK);
            if (result != CCU_SUCCESS) {
                return result;
            }
        }
        return CCU_SUCCESS;
    }

    inline CcuResult MergePartialResults(ReduceScatterContext &ctx)
    {
        ccu::LocalAddr destination;
        destination.addr = ctx.localOutput;
        destination.token = ctx.localOutputToken;

        ccu::LocalAddr source;
        source.addr = ctx.localInput;
        source.token = ctx.localInputToken;

        CcuResult result = ccu::LocalReduce(
            destination, source, ctx.sliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.transferDone, 1);
        if (result != CCU_SUCCESS) {
            return result;
        }
        return ccu::EventWait(ctx.transferDone, 1);
    }

} // namespace

CcuResult CcuReduceScatterReadReduceKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatter *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        HCCL_ERROR("[CcuReduceScatterReadReduceKernel] invalid kernel argument");
        return CCU_E_PARA;
    }

    ReduceScatterContext ctx;
    ctx.arg = kernelArg;

    CcuResult result = InitRemoteResources(ctx);
    if (result != CCU_SUCCESS) {
        return result;
    }
    if (ctx.arg->kernelKind == CCU_REDUCE_SCATTER_KERNEL_NHR) {
        result = LoadNhrTaskArgs(ctx);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = PreSync(ctx);
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = ReduceScatterNhr(ctx);
        if (result != CCU_SUCCESS) {
            return result;
        }
        return PostSync(ctx);
    }
    result = LoadTaskArgs(ctx);
    if (result != CCU_SUCCESS) {
        return result;
    }
    CCU_IF(ctx.executionMode != CCU_REDUCE_SCATTER_MODE_MERGE_PARTIALS)
    {
        result = PreSync(ctx);
        if (result != CCU_SUCCESS) {
            return result;
        }
        PreparePeerSources(ctx);
        PreparePeerDestination(ctx);

        CCU_IF(ctx.sliceSize != 0)
        {
            if (ctx.arg->destinationInitialized) {
                CCU_IF(ctx.executionMode == CCU_REDUCE_SCATTER_MODE_REDUCE_PEERS)
                {
                    result = InitializeLocalContribution(ctx);
                    if (result != CCU_SUCCESS) {
                        return result;
                    }
                }
                CCU_IF(ctx.executionMode == CCU_REDUCE_SCATTER_MODE_REDUCE_PEERS_SCRATCH)
                {
                    result = InitializeLocalContribution(ctx);
                    if (result != CCU_SUCCESS) {
                        return result;
                    }
                }
            }
            CCU_IF(ctx.executionMode == CCU_REDUCE_SCATTER_MODE_REDUCE_PEERS)
            {
                result = ReducePeerContributionsDirect(ctx);
            }
            CCU_IF(ctx.executionMode == CCU_REDUCE_SCATTER_MODE_REDUCE_REMOTE)
            {
                result = ReducePeerContributionsDirect(ctx);
            }
            CCU_IF(ctx.executionMode == CCU_REDUCE_SCATTER_MODE_REDUCE_PEERS_SCRATCH)
            {
                PrepareScratchSlots(ctx);
                result = ReducePeerContributionsWithScratch(ctx);
            }
            CCU_IF(ctx.executionMode == CCU_REDUCE_SCATTER_MODE_IN_PLACE_SCRATCH)
            {
                PrepareScratchSlots(ctx);
                result = ReducePeerContributionsWithScratch(ctx);
            }
            CCU_IF(ctx.executionMode == CCU_REDUCE_SCATTER_MODE_OMNIPIPE_MESH)
            {
                PrepareScratchSlots(ctx);
                result = ReduceOmnipipeMeshWithScratch(ctx);
            }
            if (ctx.arg->channelCount <= MAX_MS_PIPELINE_INPUTS) {
                CCU_IF(ctx.executionMode == CCU_REDUCE_SCATTER_MODE_MS_PIPELINE)
                {
                    result = ReducePeerContributionsWithMsPipeline(ctx);
                    if (result == CCU_SUCCESS && ctx.arg->destinationInitialized
                        && ctx.arg->channelCount == MAX_MS_PIPELINE_INPUTS) {
                        result = MergeLocalContributionAfterMsPipeline(ctx);
                    }
                }
            }
        }
        if (result != CCU_SUCCESS) {
            return result;
        }
        result = PostSync(ctx);
    }
    if (result != CCU_SUCCESS) {
        return result;
    }

    CCU_IF(ctx.executionMode == CCU_REDUCE_SCATTER_MODE_MERGE_PARTIALS)
    {
        result = MergePartialResults(ctx);
    }
    return result;
}

CcuResult CcuReduceScatterOmnipipeMeshKernel(CcuKernelArg arg)
{
    return CcuReduceScatterReadReduceKernel(arg);
}

CcuResult CcuReduceScatterOmnipipeNhrKernel(CcuKernelArg arg)
{
    return CcuReduceScatterReadReduceKernel(arg);
}

} // namespace ops_hccl
