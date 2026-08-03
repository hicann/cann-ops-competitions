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
#include <limits>
#include <memory>
#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

namespace ops_hccl {

namespace {

#define CCU_RETURN_IF_ERROR(call) \
    do { \
        CcuResult ccuResult = (call); \
        if (ccuResult != CCU_SUCCESS) { \
            return ccuResult; \
        } \
    } while (0)

    constexpr uint32_t INPUT_XN_ID = 0;
    // Bit 1 is intentionally separate from the stage-1/stage-2 POST bit.
    // CKE notifications are set/clear masks, not counters, so re-recording a
    // bit before every peer has consumed its previous value can lose an epoch.
    constexpr uint32_t PINNED_COMBINE_SYNC_ID = 1;
    constexpr uint32_t WRITE_REDUCE_ACK_SYNC_ID
        = PINNED_COMBINE_SYNC_ID;
    constexpr uint32_t TOKEN_XN_ID = 2;
    constexpr uint32_t POST_SYNC_ID = 3;
    constexpr uint32_t CKE_IDX_0 = 0;

    constexpr uint32_t MAX_GROUP_SOURCE_COUNT = 8;
    constexpr uint32_t MAX_ASYNC_STRIPES = 4;
    constexpr uint32_t MAX_COMBINE_STRIPES = 8;
    constexpr uint32_t WIDE_LANE_MAX_SOURCES = 8;
    constexpr uint32_t WIDE_LANE_TILE_COUNT = 3;
    constexpr uint32_t WIDE_LANE_METADATA_COUNT = 5;
    constexpr uint32_t WIDE_LANE_WORKSPACE_BANKS = 2;
    constexpr uint32_t REDUCE_LANE_COUNT = 4;
    constexpr uint32_t WORKSPACE_SLOT_COUNT = REDUCE_LANE_COUNT - 1;
    constexpr uint32_t PARTIAL_BASE_ARG_COUNT = 8;
    constexpr uint32_t STRIPE_OFFSET_ARG_BASE = PARTIAL_BASE_ARG_COUNT;
    constexpr uint32_t STRIPE_LENGTH_ARG_BASE = STRIPE_OFFSET_ARG_BASE + MAX_ASYNC_STRIPES;

    constexpr uint32_t MS_INTERLEAVE = 8;
    // Match the CCU_SCHED resource pack: 16 block loop engines and
    // 16 * 8 = 128 memory-slice buffers on each IO die.
    constexpr uint32_t MS_PARALLEL_LOOPS = 16;
    constexpr uint64_t MS_SLICE_BYTES = 4096;
    constexpr uint64_t MS_LOOP_BYTES = MS_SLICE_BYTES * MS_PARALLEL_LOOPS;
    constexpr uint64_t RANK4_PACKED_MS_BYTES = 2U * MS_LOOP_BYTES;
    constexpr uint32_t RANK4_PACKED_MS_SOURCES = 4U;

    constexpr uint64_t EncodeLoopParam(uint64_t gsaOffset, uint64_t loopCount)
    {
        return (gsaOffset << 13U) | loopCount;
    }

    constexpr uint64_t EncodeParallelParam(
        uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
    {
        return (repeatNum << 55U) | (repeatLoopIndex << 48U) | (totalLoopNum << 41U);
    }

    constexpr uint64_t EncodeOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t eventOffset)
    {
        return (gsaOffset << 21U) | (msOffset << 10U) | eventOffset;
    }

    struct PartialReduceContext {
        const CcuKernelArgPartialReduce *arg{nullptr};

        ccu::Variable myInput;
        ccu::Variable myOutput;
        ccu::Variable myScratch;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable scratchToken;
        ccu::Variable compactDestination;
        ccu::Variable compactDestinationToken;
        std::vector<ccu::Variable> peerInput;
        std::vector<ccu::Variable> peerToken;
        ccu::Variable sliceOffset;
        ccu::Variable sliceBytes;
        std::array<ccu::Variable, WIDE_LANE_METADATA_COUNT> stripeOffset;
        std::array<ccu::Variable, MAX_ASYNC_STRIPES> stripeLength;
        ccu::Variable msMainBytes;
        ccu::Variable msMainLoopCount;
        ccu::Variable msTailParallel;
        ccu::Variable msTailBytes;
    };

    static CcuResult InitResources(PartialReduceContext &ctx)
    {
        const auto *arg = ctx.arg;
        if (arg->channelCount == 0 || arg->channelCount > MAX_GROUP_SOURCE_COUNT
            || arg->stripeCount > MAX_ASYNC_STRIPES
            || arg->stripeCount > arg->channelCount
            || (arg->enableMsPrefix && arg->stripeCount != 0)
            || (arg->enableWideLane && arg->stripeCount != 0)
            || (arg->enableMsPrefix && arg->enableWideLane)
            || (arg->enableRank4PackedMs
                && (!arg->enableMsPrefix || !arg->compactArgs || !arg->includeSelf
                    || arg->channelCount != RANK4_PACKED_MS_SOURCES - 1U
                    || arg->staticMsGoSize[0] != RANK4_PACKED_MS_BYTES
                    || arg->staticMsGoSize[1] != 2U
                    || arg->staticMsGoSize[2] != 0U
                    || arg->staticMsGoSize[3] != 0U))) {
            return CcuResult::CCU_E_PARA;
        }
        const uint32_t sourceCount = arg->channelCount + static_cast<uint32_t>(arg->includeSelf);
        if (sourceCount == 0 || sourceCount > MAX_GROUP_SOURCE_COUNT) {
            return CcuResult::CCU_E_PARA;
        }

        ctx.peerInput.resize(arg->channelCount);
        ctx.peerToken.resize(arg->channelCount);
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            ctx.peerInput[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], INPUT_XN_ID);
            ctx.peerToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], TOKEN_XN_ID);
        }
        return CCU_SUCCESS;
    }

    static CcuResult LoadArgs(PartialReduceContext &ctx)
    {
        uint32_t argId = 0;
        if (ctx.arg->compactArgs) {
            CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.myInput, argId++));
            CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.compactDestination, argId++));
            CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
            CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.compactDestinationToken, argId++));
            ctx.sliceOffset = ctx.arg->staticSliceOffset;
            ctx.msMainBytes = ctx.arg->staticMsGoSize[0];
            ctx.msMainLoopCount = ctx.arg->staticMsGoSize[1];
            ctx.msTailParallel = ctx.arg->staticMsGoSize[2];
            ctx.msTailBytes = ctx.arg->staticMsGoSize[3];
            return CCU_SUCCESS;
        }
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.myInput, argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.myOutput, argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.myScratch, argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sliceOffset, argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sliceBytes, argId++));
        if (ctx.arg->enableWideLane) {
            for (uint32_t i = 0; i < WIDE_LANE_METADATA_COUNT; ++i) {
                CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.stripeOffset[i], PARTIAL_BASE_ARG_COUNT + i));
            }
            return CCU_SUCCESS;
        }
        if (ctx.arg->enableMsPrefix) {
            CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.msMainBytes, PARTIAL_BASE_ARG_COUNT));
            CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.msMainLoopCount, PARTIAL_BASE_ARG_COUNT + 1));
            CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.msTailParallel, PARTIAL_BASE_ARG_COUNT + 2));
            CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.msTailBytes, PARTIAL_BASE_ARG_COUNT + 3));
            ctx.stripeOffset[MAX_ASYNC_STRIPES - 1] = ctx.sliceBytes;
            ctx.stripeLength[MAX_ASYNC_STRIPES - 1] = 0;
            return CCU_SUCCESS;
        }
        for (uint32_t i = 0; i < MAX_ASYNC_STRIPES; ++i) {
            CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.stripeOffset[i], STRIPE_OFFSET_ARG_BASE + i));
            CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.stripeLength[i], STRIPE_LENGTH_ARG_BASE + i));
        }
        return CCU_SUCCESS;
    }

    static CcuResult PreSync(PartialReduceContext &ctx)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[i], ctx.myInput, INPUT_XN_ID, CKE_IDX_0, 1U << INPUT_XN_ID));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[i], ctx.inputToken, TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
        }

        constexpr uint32_t WAIT_BITS = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, WAIT_BITS));
        }
        return CCU_SUCCESS;
    }

    static CcuResult RecordPostSync(PartialReduceContext &ctx)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    static CcuResult WaitPostSync(PartialReduceContext &ctx)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    static CcuResult PostSync(PartialReduceContext &ctx)
    {
        CCU_RETURN_IF_ERROR(RecordPostSync(ctx));
        CCU_RETURN_IF_ERROR(WaitPostSync(ctx));
        return CCU_SUCCESS;
    }

    static CcuResult ReduceRank4PackedMs(PartialReduceContext &ctx)
    {
        // The exact rank-4 512-KiB input has 32 output slices. Each of the
        // 16 lanes retains two four-source slices in eight disjoint MS buffers.
        // The first synchronous LoopGroup completes every input transfer before
        // POST is recorded; the second consumes the same persistent MS handles
        // while POST propagation overlaps local reduction and output copies.
        ccu::Array<ccu::Event> completed(MS_PARALLEL_LOOPS);
        ccu::Array<ccu::CcuBuffer> buffers(MS_PARALLEL_LOOPS * MS_INTERLEAVE);
        ccu::Variable halfOffset;
        halfOffset = MS_LOOP_BYTES;

        ccu::LocalAddr localSource0;
        localSource0.addr = ctx.myInput;
        localSource0.addr += ctx.sliceOffset;
        localSource0.token = ctx.inputToken;
        ccu::LocalAddr localSource1;
        localSource1.addr = ctx.myInput;
        localSource1.addr += ctx.sliceOffset;
        localSource1.addr += halfOffset;
        localSource1.token = ctx.inputToken;

        std::array<ccu::RemoteAddr, RANK4_PACKED_MS_SOURCES - 1U> remoteSource0;
        std::array<ccu::RemoteAddr, RANK4_PACKED_MS_SOURCES - 1U> remoteSource1;
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            remoteSource0[channel].addr = ctx.peerInput[channel];
            remoteSource0[channel].addr += ctx.sliceOffset;
            remoteSource0[channel].token = ctx.peerToken[channel];
            remoteSource1[channel].addr = ctx.peerInput[channel];
            remoteSource1[channel].addr += ctx.sliceOffset;
            remoteSource1[channel].addr += halfOffset;
            remoteSource1[channel].token = ctx.peerToken[channel];
        }

        ccu::LocalAddr destination0;
        destination0.addr = ctx.compactDestination;
        destination0.token = ctx.compactDestinationToken;
        ccu::LocalAddr destination1;
        destination1.addr = ctx.compactDestination;
        destination1.addr += halfOffset;
        destination1.token = ctx.compactDestinationToken;

        ccu::Variable length;
        length = MS_SLICE_BYTES;
        ccu::Event laneEvent = completed[0];
        ccu::Variable loadLoopParam;
        loadLoopParam = EncodeLoopParam(0, 1);
        ccu::Func loadBody([&ctx, &buffers, &remoteSource0, &remoteSource1,
                              &localSource0, &localSource1, &length, laneEvent]() {
            for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
                ccu::Read(ctx.arg->channels[channel], buffers[channel], remoteSource0[channel],
                    length, laneEvent, static_cast<uint16_t>(1U << channel));
                ccu::Read(ctx.arg->channels[channel], buffers[4U + channel], remoteSource1[channel],
                    length, laneEvent, static_cast<uint16_t>(1U << (4U + channel)));
            }
            ccu::LocalCopy(
                buffers[3], localSource0, length, laneEvent, static_cast<uint16_t>(1U << 3U));
            ccu::LocalCopy(
                buffers[7], localSource1, length, laneEvent, static_cast<uint16_t>(1U << 7U));
            ccu::EventWait(laneEvent, 0xFFU);
        });
        ccu::Loop loadLoop(loadLoopParam, loadBody);

        ccu::Variable parallelParam;
        parallelParam = EncodeParallelParam(MS_PARALLEL_LOOPS - 1U, 0, 1);
        ccu::Variable offsetParam;
        offsetParam = EncodeOffsetParam(MS_SLICE_BYTES, MS_INTERLEAVE, 1);
        {
            std::vector<ccu::Loop> loops{loadLoop};
            ccu::LoopGroup loadGroup(parallelParam, offsetParam, MS_PARALLEL_LOOPS, loops);
        }

        CCU_RETURN_IF_ERROR(RecordPostSync(ctx));

        ccu::Variable reduceLoopParam;
        reduceLoopParam = EncodeLoopParam(0, 1);
        ccu::Func reduceBody([&ctx, &buffers, &destination0, &destination1, &length, laneEvent]() {
            ccu::LocalReduce(&buffers[0], RANK4_PACKED_MS_SOURCES,
                ctx.arg->dataType, ctx.arg->outputDataType, ctx.arg->reduceOp,
                length, laneEvent, 1U);
            ccu::LocalReduce(&buffers[4], RANK4_PACKED_MS_SOURCES,
                ctx.arg->dataType, ctx.arg->outputDataType, ctx.arg->reduceOp,
                length, laneEvent, 2U);
            ccu::EventWait(laneEvent, 3U);
            ccu::LocalCopy(destination0, buffers[0], length, laneEvent, 1U);
            ccu::LocalCopy(destination1, buffers[4], length, laneEvent, 2U);
            ccu::EventWait(laneEvent, 3U);
        });
        ccu::Loop reduceLoop(reduceLoopParam, reduceBody);
        ccu::Variable reduceParallelParam;
        reduceParallelParam = EncodeParallelParam(MS_PARALLEL_LOOPS - 1U, 0, 1);
        ccu::Variable reduceOffsetParam;
        reduceOffsetParam = EncodeOffsetParam(MS_SLICE_BYTES, MS_INTERLEAVE, 1);
        {
            std::vector<ccu::Loop> loops{reduceLoop};
            ccu::LoopGroup reduceGroup(
                reduceParallelParam, reduceOffsetParam, MS_PARALLEL_LOOPS, loops);
        }

        CCU_RETURN_IF_ERROR(WaitPostSync(ctx));
        return CCU_SUCCESS;
    }

    static CcuResult ReduceLayerLanes(
        PartialReduceContext &ctx, ccu::Variable dataOffset, ccu::Variable dataBytes)
    {
        std::vector<ccu::RemoteAddr> remoteSrc(ctx.arg->channelCount);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            remoteSrc[i].addr = ctx.peerInput[i];
            remoteSrc[i].addr += ctx.sliceOffset;
            remoteSrc[i].addr += dataOffset;
            remoteSrc[i].token = ctx.peerToken[i];
        }

        ccu::LocalAddr localSrc;
        localSrc.addr = ctx.myInput;
        localSrc.addr += ctx.sliceOffset;
        localSrc.addr += dataOffset;
        localSrc.token = ctx.inputToken;

        ccu::LocalAddr dst;
        dst.addr = ctx.arg->includeSelf ? ctx.myOutput : ctx.myScratch;
        dst.addr += dataOffset;
        dst.token = ctx.arg->includeSelf ? ctx.outputToken : ctx.scratchToken;

        std::array<ccu::LocalAddr, WORKSPACE_SLOT_COUNT> workspace;
        for (uint32_t i = 0; i < WORKSPACE_SLOT_COUNT; ++i) {
            workspace[i].addr = ctx.stripeOffset[i];
            workspace[i].token = ctx.stripeLength[i];
        }

        const uint32_t sourceCount = ctx.arg->channelCount + static_cast<uint32_t>(ctx.arg->includeSelf);
        const uint32_t laneCount = std::min<uint32_t>(REDUCE_LANE_COUNT, sourceCount);
        std::array<ccu::LocalAddr, REDUCE_LANE_COUNT> lanes;
        lanes[0] = dst;
        for (uint32_t lane = 1; lane < laneCount; ++lane) {
            lanes[lane] = workspace[lane - 1];
        }

        ccu::Event completed;
        uint32_t nextChannel = 0;
        if (ctx.arg->includeSelf) {
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(lanes[0], localSrc, dataBytes, completed, 1U));
        }
        const uint32_t firstRemoteLane = static_cast<uint32_t>(ctx.arg->includeSelf);
        for (uint32_t lane = firstRemoteLane; lane < laneCount; ++lane) {
            const uint16_t eventBit = static_cast<uint16_t>(1U << lane);
            CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[nextChannel], lanes[lane], remoteSrc[nextChannel],
                dataBytes, completed, eventBit));
            ++nextChannel;
        }

        uint16_t secondReadMask = 0;
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            const uint16_t eventBit = static_cast<uint16_t>(1U << lane);
            CCU_RETURN_IF_ERROR(ccu::EventWait(completed, eventBit));
            if (nextChannel < ctx.arg->channelCount) {
                CCU_RETURN_IF_ERROR(ccu::ReadReduce(ctx.arg->channels[nextChannel], lanes[lane],
                    remoteSrc[nextChannel], dataBytes, ctx.arg->dataType, ctx.arg->reduceOp, completed,
                    eventBit));
                secondReadMask |= eventBit;
                ++nextChannel;
            }
        }

        uint16_t pairReduceMask = 0;
        uint32_t pairIndex = 0;
        for (uint32_t lane = 0; lane + 1 < laneCount; lane += 2, ++pairIndex) {
            const uint16_t lanesReadyMask
                = secondReadMask & static_cast<uint16_t>((1U << lane) | (1U << (lane + 1)));
            if (lanesReadyMask != 0) {
                CCU_RETURN_IF_ERROR(ccu::EventWait(completed, lanesReadyMask));
            }
            const uint16_t pairBit = static_cast<uint16_t>(1U << (8U + pairIndex));
            CCU_RETURN_IF_ERROR(ccu::LocalReduce(lanes[lane], lanes[lane + 1], dataBytes, ctx.arg->dataType,
                ctx.arg->reduceOp, completed, pairBit));
            pairReduceMask |= pairBit;
        }

        if ((laneCount & 1U) != 0) {
            const uint16_t lastLaneBit = static_cast<uint16_t>(1U << (laneCount - 1));
            if ((secondReadMask & lastLaneBit) != 0) {
                CCU_RETURN_IF_ERROR(ccu::EventWait(completed, lastLaneBit));
            }
        }
        if (pairReduceMask != 0) {
            CCU_RETURN_IF_ERROR(ccu::EventWait(completed, pairReduceMask));
        }
        for (uint32_t lane = 2; lane < laneCount; lane += 2) {
            CCU_RETURN_IF_ERROR(ccu::LocalReduce(
                lanes[0], lanes[lane], dataBytes, ctx.arg->dataType, ctx.arg->reduceOp, completed, 1U));
            CCU_RETURN_IF_ERROR(ccu::EventWait(completed, 1U));
        }
        return CCU_SUCCESS;
    }

    static void ComposeMsReduceBody(PartialReduceContext &ctx, ccu::Array<ccu::CcuBuffer> &buffers,
        uint32_t bufferBase, ccu::Event completed, std::vector<ccu::RemoteAddr> &remoteSrc,
        ccu::LocalAddr &localSrc, ccu::LocalAddr &dst, ccu::Variable &length)
    {
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            ccu::Read(ctx.arg->channels[channel], buffers[bufferBase + channel], remoteSrc[channel], length,
                completed, static_cast<uint16_t>(1U << channel));
        }
        if (ctx.arg->includeSelf) {
            ccu::LocalCopy(buffers[bufferBase + ctx.arg->channelCount], localSrc, length, completed,
                static_cast<uint16_t>(1U << ctx.arg->channelCount));
        }

        const uint32_t sourceCount
            = ctx.arg->channelCount + static_cast<uint32_t>(ctx.arg->includeSelf);
        ccu::EventWait(completed, static_cast<uint16_t>((1U << sourceCount) - 1U));
        if (sourceCount > 1) {
            ccu::LocalReduce(&buffers[bufferBase], sourceCount, ctx.arg->dataType, ctx.arg->outputDataType,
                ctx.arg->reduceOp, length, completed, 1U);
            ccu::EventWait(completed, 1U);
        }
        ccu::LocalCopy(dst, buffers[bufferBase], length, completed, 1U);
        ccu::EventWait(completed, 1U);
    }

    static CcuResult ReduceLayerMsPrefix(PartialReduceContext &ctx)
    {
        ccu::Array<ccu::Event> completed(MS_PARALLEL_LOOPS);
        ccu::Array<ccu::CcuBuffer> buffers(MS_PARALLEL_LOOPS * MS_INTERLEAVE);

        ccu::LocalAddr mainLocalSrc;
        mainLocalSrc.addr = ctx.myInput;
        mainLocalSrc.addr += ctx.sliceOffset;
        mainLocalSrc.token = ctx.inputToken;
        ccu::LocalAddr mainDst;
        if (ctx.arg->compactArgs) {
            mainDst.addr = ctx.compactDestination;
            mainDst.token = ctx.compactDestinationToken;
        } else {
            mainDst.addr = ctx.arg->includeSelf ? ctx.myOutput : ctx.myScratch;
            mainDst.token = ctx.arg->includeSelf ? ctx.outputToken : ctx.scratchToken;
        }
        std::vector<ccu::RemoteAddr> mainRemoteSrc(ctx.arg->channelCount);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            mainRemoteSrc[channel].addr = ctx.peerInput[channel];
            mainRemoteSrc[channel].addr += ctx.sliceOffset;
            mainRemoteSrc[channel].token = ctx.peerToken[channel];
        }

        ccu::Variable mainLength;
        mainLength = MS_SLICE_BYTES;
        ccu::Variable mainLoopParam;
        if (ctx.arg->compactArgs) {
            mainLoopParam = EncodeLoopParam(MS_LOOP_BYTES, ctx.arg->staticMsGoSize[1]);
        } else {
            mainLoopParam = EncodeLoopParam(MS_LOOP_BYTES, 0);
            mainLoopParam += ctx.msMainLoopCount;
        }
        ccu::Event mainEvent = completed[0];
        ccu::Func mainBody([&ctx, &buffers, &mainRemoteSrc, &mainLocalSrc, &mainDst, &mainLength, mainEvent]() {
            ComposeMsReduceBody(ctx, buffers, 0, mainEvent, mainRemoteSrc, mainLocalSrc, mainDst, mainLength);
        });
        ccu::Loop mainLoop(mainLoopParam, mainBody);

        if (ctx.arg->compactArgs) {
            if (ctx.arg->staticMsGoSize[0] != 0) {
                ccu::Variable parallelParam;
                parallelParam = EncodeParallelParam(MS_PARALLEL_LOOPS - 1U, 0, 1);
                ccu::Variable offsetParam;
                offsetParam = EncodeOffsetParam(MS_SLICE_BYTES, MS_INTERLEAVE, 1);
                std::vector<ccu::Loop> loops{mainLoop};
                ccu::LoopGroup group(parallelParam, offsetParam, MS_PARALLEL_LOOPS, loops);
            }
        } else {
            CCU_IF(ctx.msMainBytes != 0)
            {
                ccu::Variable parallelParam;
                parallelParam = EncodeParallelParam(MS_PARALLEL_LOOPS - 1U, 0, 1);
                ccu::Variable offsetParam;
                offsetParam = EncodeOffsetParam(MS_SLICE_BYTES, MS_INTERLEAVE, 1);
                std::vector<ccu::Loop> loops{mainLoop};
                ccu::LoopGroup group(parallelParam, offsetParam, MS_PARALLEL_LOOPS, loops);
            }
        }

        ccu::LocalAddr tailLocalSrc0;
        tailLocalSrc0.addr = ctx.myInput;
        tailLocalSrc0.addr += ctx.sliceOffset;
        tailLocalSrc0.addr += ctx.msMainBytes;
        tailLocalSrc0.token = ctx.inputToken;
        ccu::LocalAddr tailDst0;
        if (ctx.arg->compactArgs) {
            tailDst0.addr = ctx.compactDestination;
            tailDst0.token = ctx.compactDestinationToken;
        } else {
            tailDst0.addr = ctx.arg->includeSelf ? ctx.myOutput : ctx.myScratch;
            tailDst0.token = ctx.arg->includeSelf ? ctx.outputToken : ctx.scratchToken;
        }
        tailDst0.addr += ctx.msMainBytes;
        std::vector<ccu::RemoteAddr> tailRemoteSrc0(ctx.arg->channelCount);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            tailRemoteSrc0[channel].addr = ctx.peerInput[channel];
            tailRemoteSrc0[channel].addr += ctx.sliceOffset;
            tailRemoteSrc0[channel].addr += ctx.msMainBytes;
            tailRemoteSrc0[channel].token = ctx.peerToken[channel];
        }

        ccu::LocalAddr tailLocalSrc1;
        tailLocalSrc1.addr = ctx.myInput;
        tailLocalSrc1.addr += ctx.sliceOffset;
        tailLocalSrc1.addr += ctx.msMainBytes;
        tailLocalSrc1.addr += ctx.msTailBytes;
        tailLocalSrc1.token = ctx.inputToken;
        ccu::LocalAddr tailDst1;
        if (ctx.arg->compactArgs) {
            tailDst1.addr = ctx.compactDestination;
            tailDst1.token = ctx.compactDestinationToken;
        } else {
            tailDst1.addr = ctx.arg->includeSelf ? ctx.myOutput : ctx.myScratch;
            tailDst1.token = ctx.arg->includeSelf ? ctx.outputToken : ctx.scratchToken;
        }
        tailDst1.addr += ctx.msMainBytes;
        tailDst1.addr += ctx.msTailBytes;
        std::vector<ccu::RemoteAddr> tailRemoteSrc1(ctx.arg->channelCount);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            tailRemoteSrc1[channel].addr = ctx.peerInput[channel];
            tailRemoteSrc1[channel].addr += ctx.sliceOffset;
            tailRemoteSrc1[channel].addr += ctx.msMainBytes;
            tailRemoteSrc1[channel].addr += ctx.msTailBytes;
            tailRemoteSrc1[channel].token = ctx.peerToken[channel];
        }

        ccu::Variable tailLength0;
        tailLength0 = ctx.msTailBytes;
        ccu::Variable tailLength1;
        tailLength1 = MS_SLICE_BYTES;
        ccu::Variable tailLoopParam0;
        tailLoopParam0 = EncodeLoopParam(0, 1);
        ccu::Variable tailLoopParam1;
        tailLoopParam1 = EncodeLoopParam(0, 1);
        ccu::Event tailEvent0 = completed[0];
        ccu::Event tailEvent1 = completed[1];
        ccu::Func tailBody0(
            [&ctx, &buffers, &tailRemoteSrc0, &tailLocalSrc0, &tailDst0, &tailLength0, tailEvent0]() {
                ComposeMsReduceBody(
                    ctx, buffers, 0, tailEvent0, tailRemoteSrc0, tailLocalSrc0, tailDst0, tailLength0);
            });
        ccu::Func tailBody1(
            [&ctx, &buffers, &tailRemoteSrc1, &tailLocalSrc1, &tailDst1, &tailLength1, tailEvent1]() {
                ComposeMsReduceBody(
                    ctx, buffers, MS_INTERLEAVE, tailEvent1, tailRemoteSrc1, tailLocalSrc1, tailDst1, tailLength1);
            });
        ccu::Loop tailLoop0(tailLoopParam0, tailBody0);
        ccu::Loop tailLoop1(tailLoopParam1, tailBody1);

        if (ctx.arg->compactArgs) {
            if (ctx.arg->staticMsGoSize[2] != 0) {
                ccu::Variable tailParallel;
                tailParallel = ctx.arg->staticMsGoSize[2];
                ccu::Variable offsetParam;
                offsetParam = EncodeOffsetParam(MS_SLICE_BYTES, MS_INTERLEAVE, 1);
                std::vector<ccu::Loop> loops{tailLoop0, tailLoop1};
                ccu::LoopGroup group(tailParallel, offsetParam, MS_PARALLEL_LOOPS, loops);
            }
        } else {
            CCU_IF(ctx.msTailParallel != 0)
            {
                ccu::Variable offsetParam;
                offsetParam = EncodeOffsetParam(MS_SLICE_BYTES, MS_INTERLEAVE, 1);
                std::vector<ccu::Loop> loops{tailLoop0, tailLoop1};
                ccu::LoopGroup group(ctx.msTailParallel, offsetParam, MS_PARALLEL_LOOPS, loops);
            }
        }
        return CCU_SUCCESS;
    }

    static CcuResult ReduceLayerStriped(PartialReduceContext &ctx)
    {
        ccu::LocalAddr localSrc;
        localSrc.addr = ctx.myInput;
        localSrc.addr += ctx.sliceOffset;
        localSrc.token = ctx.inputToken;

        ccu::LocalAddr dst;
        dst.addr = ctx.arg->includeSelf ? ctx.myOutput : ctx.myScratch;
        dst.token = ctx.arg->includeSelf ? ctx.outputToken : ctx.scratchToken;

        const uint32_t stripeCount = ctx.arg->stripeCount;
        std::array<ccu::LocalAddr, MAX_ASYNC_STRIPES> dstStripe;
        std::array<ccu::LocalAddr, MAX_ASYNC_STRIPES> localStripe;
        for (uint32_t stripe = 0; stripe < stripeCount; ++stripe) {
            dstStripe[stripe] = dst;
            dstStripe[stripe].addr += ctx.stripeOffset[stripe];
            localStripe[stripe] = localSrc;
            localStripe[stripe].addr += ctx.stripeOffset[stripe];
        }
        // Address wrappers refer to CCU address registers; copying a wrapper
        // and then applying += can mutate the shared register.  Materialize
        // every (peer, stripe) address exactly once so cyclic waves can never
        // accumulate offsets across uses of the same peer.
        std::array<std::array<ccu::RemoteAddr, MAX_ASYNC_STRIPES>, MAX_GROUP_SOURCE_COUNT> remoteStripe;
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            for (uint32_t stripe = 0; stripe < stripeCount; ++stripe) {
                remoteStripe[channel][stripe].addr = ctx.peerInput[channel];
                remoteStripe[channel][stripe].addr += ctx.sliceOffset;
                remoteStripe[channel][stripe].addr += ctx.stripeOffset[stripe];
                remoteStripe[channel][stripe].token = ctx.peerToken[channel];
            }
        }

        // A stripe is owned by one operation in every wave.  Consequently all
        // operations inside a wave target disjoint HBM ranges and distinct
        // peers, while every stripe observes a stable cyclic source order.
        ccu::Event completed;
        uint16_t activeMask = 0;
        if (ctx.arg->includeSelf) {
            for (uint32_t stripe = 0; stripe < stripeCount; ++stripe) {
                const uint16_t eventBit = static_cast<uint16_t>(1U << stripe);
                CCU_RETURN_IF_ERROR(ccu::LocalCopy(
                    dstStripe[stripe], localStripe[stripe], ctx.stripeLength[stripe], completed, eventBit));
                activeMask |= eventBit;
            }
        } else {
            for (uint32_t stripe = 0; stripe < stripeCount; ++stripe) {
                const uint16_t eventBit = static_cast<uint16_t>(1U << stripe);
                CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[stripe], dstStripe[stripe],
                    remoteStripe[stripe][stripe], ctx.stripeLength[stripe], completed, eventBit));
                activeMask |= eventBit;
            }
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(completed, activeMask));

        const uint32_t waveCount
            = ctx.arg->channelCount - static_cast<uint32_t>(!ctx.arg->includeSelf);
        for (uint32_t wave = 0; wave < waveCount; ++wave) {
            activeMask = 0;
            for (uint32_t stripe = 0; stripe < stripeCount; ++stripe) {
                const uint32_t channelIndex = ctx.arg->includeSelf
                    ? (wave + stripe) % ctx.arg->channelCount
                    : (wave + stripe + 1U) % ctx.arg->channelCount;
                const uint16_t eventBit = static_cast<uint16_t>(1U << stripe);
                CCU_RETURN_IF_ERROR(ccu::ReadReduce(ctx.arg->channels[channelIndex], dstStripe[stripe],
                    remoteStripe[channelIndex][stripe], ctx.stripeLength[stripe], ctx.arg->dataType,
                    ctx.arg->reduceOp, completed, eventBit));
                activeMask |= eventBit;
            }
            CCU_RETURN_IF_ERROR(ccu::EventWait(completed, activeMask));
        }
        return CCU_SUCCESS;
    }

    static CcuResult ReduceLayerWide(PartialReduceContext &ctx);

    static CcuResult ReduceLayer(PartialReduceContext &ctx)
    {
        if (ctx.arg->enableWideLane) {
            return ReduceLayerWide(ctx);
        }
        if (ctx.arg->compactArgs) {
            return ReduceLayerMsPrefix(ctx);
        }
        if (ctx.arg->enableMsPrefix) {
            CCU_RETURN_IF_ERROR(ReduceLayerMsPrefix(ctx));
            CCU_IF(ctx.stripeLength[MAX_ASYNC_STRIPES - 1] != 0)
            {
                CCU_RETURN_IF_ERROR(ReduceLayerLanes(ctx, ctx.stripeOffset[MAX_ASYNC_STRIPES - 1],
                    ctx.stripeLength[MAX_ASYNC_STRIPES - 1]));
            }
            return CCU_SUCCESS;
        }
        if (ctx.arg->stripeCount == 0) {
            ccu::Variable zero;
            zero = 0;
            return ReduceLayerLanes(ctx, zero, ctx.sliceBytes);
        }
        return ReduceLayerStriped(ctx);
    }

    static CcuResult StartWideTile(PartialReduceContext &ctx, ccu::Variable &tileOffset,
        ccu::Variable &tileLength, uint32_t bank,
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> &lanes, ccu::Event completed)
    {
        const uint32_t sourceCount
            = ctx.arg->channelCount + static_cast<uint32_t>(ctx.arg->includeSelf);
        ccu::Variable bankBase;
        bankBase = ctx.stripeOffset[0];
        const uint32_t workspaceLanes = sourceCount - 1U;
        for (uint32_t slot = 0; slot < bank * workspaceLanes; ++slot) {
            bankBase += ctx.stripeOffset[1];
        }

        ccu::LocalAddr dst;
        dst.addr = ctx.arg->includeSelf ? ctx.myOutput : ctx.myScratch;
        dst.addr += tileOffset;
        dst.token = ctx.arg->includeSelf ? ctx.outputToken : ctx.scratchToken;
        lanes[0] = dst;
        for (uint32_t lane = 1; lane < sourceCount; ++lane) {
            lanes[lane].addr = bankBase;
            for (uint32_t slot = 1; slot < lane; ++slot) {
                lanes[lane].addr += ctx.stripeOffset[1];
            }
            lanes[lane].token = ctx.scratchToken;
        }

        std::vector<ccu::RemoteAddr> remoteSrc(ctx.arg->channelCount);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            remoteSrc[channel].addr = ctx.peerInput[channel];
            remoteSrc[channel].addr += ctx.sliceOffset;
            remoteSrc[channel].addr += tileOffset;
            remoteSrc[channel].token = ctx.peerToken[channel];
        }
        ccu::LocalAddr localSrc;
        localSrc.addr = ctx.myInput;
        localSrc.addr += ctx.sliceOffset;
        localSrc.addr += tileOffset;
        localSrc.token = ctx.inputToken;

        uint32_t nextChannel = 0;
        if (ctx.arg->includeSelf) {
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(lanes[0], localSrc, tileLength, completed, 1U));
        } else {
            CCU_RETURN_IF_ERROR(
                ccu::Read(ctx.arg->channels[0], lanes[0], remoteSrc[0], tileLength, completed, 1U));
            nextChannel = 1;
        }
        for (uint32_t lane = 1; lane < sourceCount; ++lane, ++nextChannel) {
            CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[nextChannel], lanes[lane],
                remoteSrc[nextChannel], tileLength, completed, static_cast<uint16_t>(1U << lane)));
        }
        return CCU_SUCCESS;
    }

    static CcuResult WaitWideTileReads(PartialReduceContext &ctx, ccu::Event completed)
    {
        const uint32_t sourceCount
            = ctx.arg->channelCount + static_cast<uint32_t>(ctx.arg->includeSelf);
        const uint16_t mask = static_cast<uint16_t>((1U << sourceCount) - 1U);
        CCU_RETURN_IF_ERROR(ccu::EventWait(completed, mask));
        return CCU_SUCCESS;
    }

    static CcuResult ReduceWideTile(PartialReduceContext &ctx,
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> &lanes,
        ccu::Variable &tileLength, ccu::Event completed)
    {
        const uint32_t sourceCount
            = ctx.arg->channelCount + static_cast<uint32_t>(ctx.arg->includeSelf);
        std::array<uint32_t, WIDE_LANE_MAX_SOURCES> active{};
        std::array<uint32_t, WIDE_LANE_MAX_SOURCES> next{};
        for (uint32_t i = 0; i < sourceCount; ++i) {
            active[i] = i;
        }
        uint32_t activeCount = sourceCount;
        while (activeCount > 1U) {
            uint32_t nextCount = 0;
            uint32_t pairIndex = 0;
            uint16_t pairMask = 0;
            uint32_t index = 0;
            for (; index + 1U < activeCount; index += 2U, ++pairIndex) {
                const uint16_t bit = static_cast<uint16_t>(1U << (8U + pairIndex));
                CCU_RETURN_IF_ERROR(ccu::LocalReduce(lanes[active[index]], lanes[active[index + 1U]],
                    tileLength, ctx.arg->dataType, ctx.arg->reduceOp, completed, bit));
                pairMask |= bit;
                next[nextCount++] = active[index];
            }
            if (index < activeCount) {
                next[nextCount++] = active[index];
            }
            CCU_RETURN_IF_ERROR(ccu::EventWait(completed, pairMask));
            for (uint32_t i = 0; i < nextCount; ++i) {
                active[i] = next[i];
            }
            activeCount = nextCount;
        }
        return CCU_SUCCESS;
    }

    static CcuResult ReduceLayerWide(PartialReduceContext &ctx)
    {
        const uint32_t sourceCount
            = ctx.arg->channelCount + static_cast<uint32_t>(ctx.arg->includeSelf);
        if (sourceCount < 2U || sourceCount > WIDE_LANE_MAX_SOURCES
            || WIDE_LANE_WORKSPACE_BANKS != 2U) {
            return CcuResult::CCU_E_PARA;
        }

        ccu::Variable tileOffset0;
        tileOffset0 = 0;
        ccu::Variable tileOffset1;
        tileOffset1 = ctx.stripeOffset[2];
        ccu::Variable tileOffset2;
        tileOffset2 = ctx.stripeOffset[2];
        tileOffset2 += ctx.stripeOffset[3];

        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> lanes0;
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> lanes1;
        ccu::Event completed;

        CCU_RETURN_IF_ERROR(
            StartWideTile(ctx, tileOffset0, ctx.stripeOffset[2], 0, lanes0, completed));
        CCU_RETURN_IF_ERROR(WaitWideTileReads(ctx, completed));
        CCU_RETURN_IF_ERROR(
            StartWideTile(ctx, tileOffset1, ctx.stripeOffset[3], 1, lanes1, completed));
        CCU_RETURN_IF_ERROR(ReduceWideTile(ctx, lanes0, ctx.stripeOffset[2], completed));

        CCU_RETURN_IF_ERROR(WaitWideTileReads(ctx, completed));
        CCU_RETURN_IF_ERROR(
            StartWideTile(ctx, tileOffset2, ctx.stripeOffset[4], 0, lanes0, completed));
        CCU_RETURN_IF_ERROR(ReduceWideTile(ctx, lanes1, ctx.stripeOffset[3], completed));

        CCU_RETURN_IF_ERROR(WaitWideTileReads(ctx, completed));
        CCU_RETURN_IF_ERROR(ReduceWideTile(ctx, lanes0, ctx.stripeOffset[4], completed));
        return CCU_SUCCESS;
    }

    struct MateLocalContext {
        const CcuKernelArgMateLocal *arg{nullptr};
        ccu::Variable input;
        ccu::Variable output;
        ccu::Variable scratch;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable scratchToken;
        std::vector<ccu::Variable> peerInput;
        std::vector<ccu::Variable> peerToken;
        std::array<ccu::Variable, MATE_MAX_PEERS> inputOffset;
        std::array<ccu::Variable, MATE_MAX_PEERS> destinationOffset;
        std::array<ccu::Variable, MATE_MAX_PEERS> length;
        std::array<ccu::LocalAddr, MATE_MAX_PEERS> localSource;
        std::array<ccu::LocalAddr, MATE_MAX_PEERS> destination;
        std::array<ccu::RemoteAddr, MATE_MAX_PEERS> remoteSource;
        ccu::Event completed;
    };

    static CcuResult InitMateLocal(MateLocalContext &ctx)
    {
        if (ctx.arg->channelCount == 0 || ctx.arg->channelCount >= MATE_MAX_PEERS
            || ctx.arg->ownPieceCount != ctx.arg->channelCount + 1U
            || ctx.arg->pieceCount <= ctx.arg->ownPieceCount
            || ctx.arg->pieceCount > MATE_MAX_PIECES) {
            return CcuResult::CCU_E_PARA;
        }
        for (uint32_t piece = 0; piece < ctx.arg->pieceCount; ++piece) {
            if (ctx.arg->pieceLengths[piece] == 0) {
                return CcuResult::CCU_E_PARA;
            }
        }
        ctx.peerInput.resize(ctx.arg->channelCount);
        ctx.peerToken.resize(ctx.arg->channelCount);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            ctx.peerInput[channel]
                = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], INPUT_XN_ID);
            ctx.peerToken[channel]
                = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], TOKEN_XN_ID);
        }
        return CCU_SUCCESS;
    }

    static CcuResult InitMateLocalExpanded(MateLocalContext &ctx)
    {
        if (ctx.arg->channelCount == 0 || ctx.arg->channelCount >= MATE_MAX_PEERS
            || ctx.arg->ownPieceCount < ctx.arg->channelCount + 1U
            || ctx.arg->pieceCount <= ctx.arg->ownPieceCount
            || ctx.arg->pieceCount > MATE_MAX_PIECES) {
            return CcuResult::CCU_E_PARA;
        }
        for (uint32_t piece = 0; piece < ctx.arg->pieceCount; ++piece) {
            if (ctx.arg->pieceLengths[piece] == 0) {
                return CcuResult::CCU_E_PARA;
            }
        }
        ctx.peerInput.resize(ctx.arg->channelCount);
        ctx.peerToken.resize(ctx.arg->channelCount);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            ctx.peerInput[channel]
                = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], INPUT_XN_ID);
            ctx.peerToken[channel]
                = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], TOKEN_XN_ID);
        }
        return CCU_SUCCESS;
    }

    static CcuResult LoadMateLocalArgs(MateLocalContext &ctx)
    {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, 0));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, 1));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratch, 2));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, 3));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, 4));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, 5));
        return CCU_SUCCESS;
    }

    static CcuResult MateLocalPreSync(MateLocalContext &ctx)
    {
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channel], ctx.input, INPUT_XN_ID, CKE_IDX_0, 1U << INPUT_XN_ID));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channel], ctx.inputToken, TOKEN_XN_ID, CKE_IDX_0,
                1U << TOKEN_XN_ID));
        }
        constexpr uint32_t WAIT_BITS = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0, WAIT_BITS));
        }
        return CCU_SUCCESS;
    }

    static CcuResult MateLocalPostSync(MateLocalContext &ctx)
    {
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(
                ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(
                ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    static CcuResult RunMateLocalCyclic(MateLocalContext &ctx)
    {
        const uint32_t laneCapacity = ctx.arg->channelCount;
        const uint32_t pieceCount = ctx.arg->pieceCount;
        if (laneCapacity > 0 && pieceCount >= laneCapacity) {
            const uint16_t remoteMask
                = static_cast<uint16_t>((1U << laneCapacity) - 1U);
            for (uint32_t piece = 0; piece < laneCapacity; ++piece) {
                const uint32_t slot = piece;
                ctx.inputOffset[slot] = ctx.arg->inputOffsets[piece];
                ctx.destinationOffset[slot] = ctx.arg->outputOffsets[piece];
                ctx.length[slot] = ctx.arg->pieceLengths[piece];
                ctx.localSource[slot].addr = ctx.input;
                ctx.localSource[slot].addr += ctx.inputOffset[slot];
                ctx.localSource[slot].token = ctx.inputToken;
                if (piece < ctx.arg->ownPieceCount) {
                    ctx.destination[slot].addr = ctx.output;
                    ctx.destination[slot].token = ctx.outputToken;
                } else {
                    ctx.destination[slot].addr = ctx.scratch;
                    ctx.destination[slot].token = ctx.scratchToken;
                }
                ctx.destination[slot].addr += ctx.destinationOffset[slot];
                CCU_RETURN_IF_ERROR(ccu::LocalCopy(ctx.destination[slot], ctx.localSource[slot],
                    ctx.length[slot], ctx.completed, static_cast<uint16_t>(1U << slot)));
            }
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.completed, remoteMask));

            for (uint32_t wave = 0; wave < pieceCount; ++wave) {
                uint16_t waveMask = remoteMask;
                for (uint32_t channel = 0; channel < laneCapacity; ++channel) {
                    const uint32_t slot = channel;
                    const uint32_t piece = (wave + channel) % pieceCount;
                    ctx.inputOffset[slot] = ctx.arg->inputOffsets[piece];
                    ctx.destinationOffset[slot] = ctx.arg->outputOffsets[piece];
                    ctx.length[slot] = ctx.arg->pieceLengths[piece];
                    if (piece < ctx.arg->ownPieceCount) {
                        ctx.destination[slot].addr = ctx.output;
                        ctx.destination[slot].token = ctx.outputToken;
                    } else {
                        ctx.destination[slot].addr = ctx.scratch;
                        ctx.destination[slot].token = ctx.scratchToken;
                    }
                    ctx.destination[slot].addr += ctx.destinationOffset[slot];
                    ctx.remoteSource[slot].addr = ctx.peerInput[channel];
                    ctx.remoteSource[slot].addr += ctx.inputOffset[slot];
                    ctx.remoteSource[slot].token = ctx.peerToken[channel];
                    CCU_RETURN_IF_ERROR(ccu::ReadReduce(ctx.arg->channels[channel],
                        ctx.destination[slot], ctx.remoteSource[slot], ctx.length[slot],
                        ctx.arg->dataType, ctx.arg->reduceOp, ctx.completed,
                        static_cast<uint16_t>(1U << slot)));
                }

                const uint32_t copyPiece = wave + laneCapacity;
                if (copyPiece < pieceCount) {
                    const uint32_t copySlot = laneCapacity;
                    ctx.inputOffset[copySlot] = ctx.arg->inputOffsets[copyPiece];
                    ctx.destinationOffset[copySlot] = ctx.arg->outputOffsets[copyPiece];
                    ctx.length[copySlot] = ctx.arg->pieceLengths[copyPiece];
                    ctx.localSource[copySlot].addr = ctx.input;
                    ctx.localSource[copySlot].addr += ctx.inputOffset[copySlot];
                    ctx.localSource[copySlot].token = ctx.inputToken;
                    if (copyPiece < ctx.arg->ownPieceCount) {
                        ctx.destination[copySlot].addr = ctx.output;
                        ctx.destination[copySlot].token = ctx.outputToken;
                    } else {
                        ctx.destination[copySlot].addr = ctx.scratch;
                        ctx.destination[copySlot].token = ctx.scratchToken;
                    }
                    ctx.destination[copySlot].addr += ctx.destinationOffset[copySlot];
                    CCU_RETURN_IF_ERROR(ccu::LocalCopy(ctx.destination[copySlot],
                        ctx.localSource[copySlot], ctx.length[copySlot], ctx.completed,
                        static_cast<uint16_t>(1U << copySlot)));
                    waveMask = static_cast<uint16_t>(waveMask | (1U << copySlot));
                }
                CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.completed, waveMask));
            }
            return CCU_SUCCESS;
        }

        for (uint32_t batchStart = 0; batchStart < ctx.arg->pieceCount;
            batchStart += laneCapacity) {
            const uint32_t batchCount
                = std::min(laneCapacity, ctx.arg->pieceCount - batchStart);
            const uint16_t activeMask
                = static_cast<uint16_t>((1U << batchCount) - 1U);
            for (uint32_t lane = 0; lane < batchCount; ++lane) {
                const uint32_t piece = batchStart + lane;
                ctx.inputOffset[lane] = ctx.arg->inputOffsets[piece];
                ctx.destinationOffset[lane] = ctx.arg->outputOffsets[piece];
                ctx.length[lane] = ctx.arg->pieceLengths[piece];
                ctx.localSource[lane].addr = ctx.input;
                ctx.localSource[lane].addr += ctx.inputOffset[lane];
                ctx.localSource[lane].token = ctx.inputToken;
                if (piece < ctx.arg->ownPieceCount) {
                    ctx.destination[lane].addr = ctx.output;
                    ctx.destination[lane].token = ctx.outputToken;
                } else {
                    ctx.destination[lane].addr = ctx.scratch;
                    ctx.destination[lane].token = ctx.scratchToken;
                }
                ctx.destination[lane].addr += ctx.destinationOffset[lane];
                CCU_RETURN_IF_ERROR(ccu::LocalCopy(ctx.destination[lane], ctx.localSource[lane],
                    ctx.length[lane], ctx.completed, static_cast<uint16_t>(1U << lane)));
            }
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.completed, activeMask));

            for (uint32_t wave = 0; wave < laneCapacity; ++wave) {
                for (uint32_t lane = 0; lane < batchCount; ++lane) {
                    const uint32_t channel = (lane + wave) % laneCapacity;
                    ctx.remoteSource[lane].addr = ctx.peerInput[channel];
                    ctx.remoteSource[lane].addr += ctx.inputOffset[lane];
                    ctx.remoteSource[lane].token = ctx.peerToken[channel];
                    CCU_RETURN_IF_ERROR(ccu::ReadReduce(ctx.arg->channels[channel],
                        ctx.destination[lane], ctx.remoteSource[lane], ctx.length[lane],
                        ctx.arg->dataType, ctx.arg->reduceOp, ctx.completed,
                        static_cast<uint16_t>(1U << lane)));
                }
                CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.completed, activeMask));
            }
        }
        return CCU_SUCCESS;
    }

    struct MateCrossContext {
        const CcuKernelArgMateCross *arg{nullptr};
        ccu::Variable input;
        ccu::Variable output;
        ccu::Variable scratch;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable scratchToken;
        std::vector<ccu::Variable> peerScratch;
        std::vector<ccu::Variable> peerToken;
        std::vector<ccu::RemoteAddr> remoteSource;
        std::vector<ccu::LocalAddr> destination;
        ccu::Variable remotePartialOffset;
        std::vector<ccu::Variable> outputOffset;
        std::vector<ccu::Variable> length;
        ccu::Event completed;
    };

    static CcuResult InitMateCross(MateCrossContext &ctx)
    {
        if (ctx.arg->channelCount == 0 || ctx.arg->channelCount > MATE_MAX_PEERS) {
            return CcuResult::CCU_E_PARA;
        }
        ctx.peerScratch.resize(ctx.arg->channelCount);
        ctx.peerToken.resize(ctx.arg->channelCount);
        ctx.remoteSource.resize(ctx.arg->channelCount);
        ctx.destination.resize(ctx.arg->channelCount);
        ctx.outputOffset.resize(ctx.arg->channelCount);
        ctx.length.resize(ctx.arg->channelCount);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            if (ctx.arg->shardLengths[channel] == 0) {
                return CcuResult::CCU_E_PARA;
            }
            ctx.peerScratch[channel]
                = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], INPUT_XN_ID);
            ctx.peerToken[channel]
                = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], TOKEN_XN_ID);
        }
        return CCU_SUCCESS;
    }

    static CcuResult LoadMateCrossArgs(MateCrossContext &ctx)
    {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, 0));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, 1));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratch, 2));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, 3));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, 4));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, 5));
        return CCU_SUCCESS;
    }

    static CcuResult MateCrossPreSync(MateCrossContext &ctx)
    {
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channel], ctx.scratch, INPUT_XN_ID, CKE_IDX_0,
                1U << INPUT_XN_ID));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channel], ctx.scratchToken, TOKEN_XN_ID, CKE_IDX_0,
                1U << TOKEN_XN_ID));
        }
        constexpr uint32_t WAIT_BITS = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0, WAIT_BITS));
        }
        return CCU_SUCCESS;
    }

    static CcuResult ReduceMateCross(MateCrossContext &ctx)
    {
        ctx.remotePartialOffset = ctx.arg->remotePartialOffset;
        uint16_t activeMask = 0;
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            ctx.remoteSource[channel].addr = ctx.peerScratch[channel];
            ctx.remoteSource[channel].addr += ctx.remotePartialOffset;
            ctx.remoteSource[channel].token = ctx.peerToken[channel];
            ctx.outputOffset[channel] = ctx.arg->outputOffsets[channel];
            ctx.destination[channel].addr = ctx.output;
            ctx.destination[channel].addr += ctx.outputOffset[channel];
            ctx.destination[channel].token = ctx.outputToken;
            ctx.length[channel] = ctx.arg->shardLengths[channel];
            const uint16_t bit = static_cast<uint16_t>(1U << channel);
            CCU_RETURN_IF_ERROR(ccu::ReadReduce(ctx.arg->channels[channel], ctx.destination[channel],
                ctx.remoteSource[channel], ctx.length[channel], ctx.arg->dataType,
                ctx.arg->reduceOp, ctx.completed, bit));
            activeMask |= bit;
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.completed, activeMask));
        return CCU_SUCCESS;
    }

    static CcuResult MateCrossPostSync(MateCrossContext &ctx)
    {
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(
                ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(
                ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    struct DirectContext {
        const CcuKernelArgDirect *arg{nullptr};
        ccu::Variable input;
        ccu::Variable output;
        ccu::Variable scratch;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable scratchToken;
        std::vector<ccu::Variable> peerInput;
        std::vector<ccu::Variable> peerToken;
        std::vector<ccu::RemoteAddr> remoteSource;
        ccu::Variable inputOffset;
        ccu::Variable outputOffset;
        ccu::Variable length;
        ccu::Variable workspaceOffset;
        ccu::Variable workspaceStride;
        std::array<ccu::Variable, DIRECT_TILE_COUNT> tileLength;
    };

    static CcuResult InitDirect(DirectContext &ctx, bool accumulateLocal)
    {
        const uint32_t sourceCount
            = ctx.arg->channelCount + static_cast<uint32_t>(accumulateLocal);
        if (ctx.arg->channelCount == 0 || ctx.arg->channelCount > MATE_MAX_PEERS
            || sourceCount < 2U || sourceCount > WIDE_LANE_MAX_SOURCES
            || ctx.arg->length == 0 || ctx.arg->workspaceStride == 0
            || DIRECT_TILE_COUNT != 3U || WIDE_LANE_WORKSPACE_BANKS != 2U) {
            return CcuResult::CCU_E_PARA;
        }
        uint64_t totalLength = 0;
        for (uint32_t tile = 0; tile < DIRECT_TILE_COUNT; ++tile) {
            if (ctx.arg->tileLengths[tile] == 0
                || totalLength > std::numeric_limits<uint64_t>::max()
                    - ctx.arg->tileLengths[tile]) {
                return CcuResult::CCU_E_PARA;
            }
            totalLength += ctx.arg->tileLengths[tile];
        }
        if (totalLength != ctx.arg->length) {
            return CcuResult::CCU_E_PARA;
        }
        ctx.peerInput.resize(ctx.arg->channelCount);
        ctx.peerToken.resize(ctx.arg->channelCount);
        ctx.remoteSource.resize(ctx.arg->channelCount);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            ctx.peerInput[channel]
                = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], INPUT_XN_ID);
            ctx.peerToken[channel]
                = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], TOKEN_XN_ID);
        }
        return CCU_SUCCESS;
    }

    static CcuResult LoadDirectArgs(DirectContext &ctx)
    {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, 0));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, 1));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratch, 2));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, 3));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, 4));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, 5));
        ctx.inputOffset = ctx.arg->inputOffset;
        ctx.outputOffset = ctx.arg->outputOffset;
        ctx.length = ctx.arg->length;
        ctx.workspaceOffset = ctx.arg->workspaceOffset;
        ctx.workspaceStride = ctx.arg->workspaceStride;
        for (uint32_t tile = 0; tile < DIRECT_TILE_COUNT; ++tile) {
            ctx.tileLength[tile] = ctx.arg->tileLengths[tile];
        }
        return CCU_SUCCESS;
    }

    static CcuResult DirectPreSync(DirectContext &ctx)
    {
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channel], ctx.input, INPUT_XN_ID, CKE_IDX_0, 1U << INPUT_XN_ID));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channel], ctx.inputToken, TOKEN_XN_ID, CKE_IDX_0,
                1U << TOKEN_XN_ID));
        }
        constexpr uint32_t WAIT_BITS = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0, WAIT_BITS));
        }
        return CCU_SUCCESS;
    }

    static CcuResult StartDirectWideTile(DirectContext &ctx, ccu::Variable &tileOffset,
        ccu::Variable &tileLength, uint32_t bank, bool accumulateLocal,
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> &lanes,
        ccu::Event completed)
    {
        const uint32_t sourceCount
            = ctx.arg->channelCount + static_cast<uint32_t>(accumulateLocal);
        const uint32_t workspaceLanes = sourceCount - 1U;
        lanes[0].addr = ctx.output;
        lanes[0].addr += ctx.outputOffset;
        lanes[0].addr += tileOffset;
        lanes[0].token = ctx.outputToken;
        for (uint32_t lane = 1; lane < sourceCount; ++lane) {
            lanes[lane].addr = ctx.scratch;
            lanes[lane].addr += ctx.workspaceOffset;
            const uint32_t slot = bank * workspaceLanes + lane - 1U;
            for (uint32_t step = 0; step < slot; ++step) {
                lanes[lane].addr += ctx.workspaceStride;
            }
            lanes[lane].token = ctx.scratchToken;
        }

        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            ctx.remoteSource[channel].addr = ctx.peerInput[channel];
            ctx.remoteSource[channel].addr += ctx.inputOffset;
            ctx.remoteSource[channel].addr += tileOffset;
            ctx.remoteSource[channel].token = ctx.peerToken[channel];
        }
        if (accumulateLocal) {
            ccu::LocalAddr localSource;
            localSource.addr = ctx.input;
            localSource.addr += ctx.inputOffset;
            localSource.addr += tileOffset;
            localSource.token = ctx.inputToken;
            CCU_RETURN_IF_ERROR(ccu::LocalReduce(lanes[0], localSource, tileLength,
                ctx.arg->dataType, ctx.arg->reduceOp, completed, 1U));
            for (uint32_t lane = 1; lane < sourceCount; ++lane) {
                const uint32_t channel = lane - 1U;
                CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[channel], lanes[lane],
                    ctx.remoteSource[channel], tileLength, completed,
                    static_cast<uint16_t>(1U << lane)));
            }
        } else {
            CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[0], lanes[0],
                ctx.remoteSource[0], tileLength, completed, 1U));
            for (uint32_t lane = 1; lane < sourceCount; ++lane) {
                CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[lane], lanes[lane],
                    ctx.remoteSource[lane], tileLength, completed,
                    static_cast<uint16_t>(1U << lane)));
            }
        }
        return CCU_SUCCESS;
    }

    static CcuResult WaitDirectWideReads(
        DirectContext &ctx, bool accumulateLocal, ccu::Event completed)
    {
        const uint32_t sourceCount
            = ctx.arg->channelCount + static_cast<uint32_t>(accumulateLocal);
        CCU_RETURN_IF_ERROR(ccu::EventWait(
            completed, static_cast<uint16_t>((1U << sourceCount) - 1U)));
        return CCU_SUCCESS;
    }

    static CcuResult ReduceDirectWideTile(DirectContext &ctx,
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> &lanes,
        ccu::Variable &tileLength, uint32_t sourceCount, ccu::Event completed)
    {
        std::array<uint32_t, WIDE_LANE_MAX_SOURCES> active{};
        std::array<uint32_t, WIDE_LANE_MAX_SOURCES> next{};
        for (uint32_t source = 0; source < sourceCount; ++source) {
            active[source] = source;
        }
        uint32_t activeCount = sourceCount;
        while (activeCount > 1U) {
            uint32_t nextCount = 0;
            uint32_t pairIndex = 0;
            uint16_t pairMask = 0;
            uint32_t index = 0;
            for (; index + 1U < activeCount; index += 2U, ++pairIndex) {
                const uint16_t bit = static_cast<uint16_t>(1U << (8U + pairIndex));
                CCU_RETURN_IF_ERROR(ccu::LocalReduce(lanes[active[index]],
                    lanes[active[index + 1U]], tileLength, ctx.arg->dataType,
                    ctx.arg->reduceOp, completed, bit));
                pairMask |= bit;
                next[nextCount++] = active[index];
            }
            if (index < activeCount) {
                next[nextCount++] = active[index];
            }
            CCU_RETURN_IF_ERROR(ccu::EventWait(completed, pairMask));
            for (uint32_t nextIndex = 0; nextIndex < nextCount; ++nextIndex) {
                active[nextIndex] = next[nextIndex];
            }
            activeCount = nextCount;
        }
        return CCU_SUCCESS;
    }

    static CcuResult RunDirectWide(DirectContext &ctx, bool accumulateLocal)
    {
        const uint32_t sourceCount
            = ctx.arg->channelCount + static_cast<uint32_t>(accumulateLocal);
        ccu::Variable tileOffset0;
        tileOffset0 = 0;
        ccu::Variable tileOffset1;
        tileOffset1 = ctx.tileLength[0];
        ccu::Variable tileOffset2;
        tileOffset2 = ctx.tileLength[0];
        tileOffset2 += ctx.tileLength[1];
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> lanes0;
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> lanes1;
        ccu::Event completed;

        CCU_RETURN_IF_ERROR(StartDirectWideTile(ctx, tileOffset0,
            ctx.tileLength[0], 0, accumulateLocal, lanes0, completed));
        CCU_RETURN_IF_ERROR(WaitDirectWideReads(ctx, accumulateLocal, completed));
        CCU_RETURN_IF_ERROR(StartDirectWideTile(ctx, tileOffset1,
            ctx.tileLength[1], 1, accumulateLocal, lanes1, completed));
        CCU_RETURN_IF_ERROR(ReduceDirectWideTile(
            ctx, lanes0, ctx.tileLength[0], sourceCount, completed));

        CCU_RETURN_IF_ERROR(WaitDirectWideReads(ctx, accumulateLocal, completed));
        CCU_RETURN_IF_ERROR(StartDirectWideTile(ctx, tileOffset2,
            ctx.tileLength[2], 0, accumulateLocal, lanes0, completed));
        CCU_RETURN_IF_ERROR(ReduceDirectWideTile(
            ctx, lanes1, ctx.tileLength[1], sourceCount, completed));

        CCU_RETURN_IF_ERROR(WaitDirectWideReads(ctx, accumulateLocal, completed));
        CCU_RETURN_IF_ERROR(ReduceDirectWideTile(
            ctx, lanes0, ctx.tileLength[2], sourceCount, completed));
        return CCU_SUCCESS;
    }

    static CcuResult StartDirectWideScratchTile(DirectContext &ctx,
        ccu::Variable &tileOffset, ccu::Variable &tileLength, uint32_t bank,
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> &lanes,
        ccu::Event completed)
    {
        const uint32_t sourceCount = ctx.arg->channelCount;
        const uint32_t workspaceLanes = sourceCount - 1U;
        lanes[0].addr = ctx.scratch;
        lanes[0].addr += ctx.outputOffset;
        lanes[0].addr += tileOffset;
        lanes[0].token = ctx.scratchToken;
        for (uint32_t lane = 1; lane < sourceCount; ++lane) {
            lanes[lane].addr = ctx.scratch;
            lanes[lane].addr += ctx.workspaceOffset;
            const uint32_t slot = bank * workspaceLanes + lane - 1U;
            for (uint32_t step = 0; step < slot; ++step) {
                lanes[lane].addr += ctx.workspaceStride;
            }
            lanes[lane].token = ctx.scratchToken;
        }

        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            ctx.remoteSource[channel].addr = ctx.peerInput[channel];
            ctx.remoteSource[channel].addr += ctx.inputOffset;
            ctx.remoteSource[channel].addr += tileOffset;
            ctx.remoteSource[channel].token = ctx.peerToken[channel];
        }
        CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[0], lanes[0],
            ctx.remoteSource[0], tileLength, completed, 1U));
        for (uint32_t lane = 1; lane < sourceCount; ++lane) {
            CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[lane], lanes[lane],
                ctx.remoteSource[lane], tileLength, completed,
                static_cast<uint16_t>(1U << lane)));
        }
        return CCU_SUCCESS;
    }

    static CcuResult StartDirectWideLocalInitTile(DirectContext &ctx,
        ccu::Variable &tileOffset, ccu::Variable &tileLength, uint32_t bank,
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> &lanes,
        ccu::Event completed)
    {
        const uint32_t sourceCount = ctx.arg->channelCount + 1U;
        const uint32_t workspaceLanes = sourceCount - 1U;
        lanes[0].addr = ctx.output;
        lanes[0].addr += ctx.outputOffset;
        lanes[0].addr += tileOffset;
        lanes[0].token = ctx.outputToken;
        for (uint32_t lane = 1; lane < sourceCount; ++lane) {
            lanes[lane].addr = ctx.scratch;
            lanes[lane].addr += ctx.workspaceOffset;
            const uint32_t slot = bank * workspaceLanes + lane - 1U;
            for (uint32_t step = 0; step < slot; ++step) {
                lanes[lane].addr += ctx.workspaceStride;
            }
            lanes[lane].token = ctx.scratchToken;
        }

        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            ctx.remoteSource[channel].addr = ctx.peerInput[channel];
            ctx.remoteSource[channel].addr += ctx.inputOffset;
            ctx.remoteSource[channel].addr += tileOffset;
            ctx.remoteSource[channel].token = ctx.peerToken[channel];
        }
        ccu::LocalAddr localSource;
        localSource.addr = ctx.input;
        localSource.addr += ctx.inputOffset;
        localSource.addr += tileOffset;
        localSource.token = ctx.inputToken;
        CCU_RETURN_IF_ERROR(
            ccu::LocalCopy(lanes[0], localSource, tileLength, completed, 1U));
        for (uint32_t lane = 1; lane < sourceCount; ++lane) {
            const uint32_t channel = lane - 1U;
            CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[channel], lanes[lane],
                ctx.remoteSource[channel], tileLength, completed,
                static_cast<uint16_t>(1U << lane)));
        }
        return CCU_SUCCESS;
    }

    static CcuResult RunDirectWideScratch(DirectContext &ctx)
    {
        const uint32_t sourceCount = ctx.arg->channelCount;
        ccu::Variable tileOffset0;
        tileOffset0 = 0;
        ccu::Variable tileOffset1;
        tileOffset1 = ctx.tileLength[0];
        ccu::Variable tileOffset2;
        tileOffset2 = ctx.tileLength[0];
        tileOffset2 += ctx.tileLength[1];
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> lanes0;
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> lanes1;
        ccu::Event completed;

        CCU_RETURN_IF_ERROR(StartDirectWideScratchTile(
            ctx, tileOffset0, ctx.tileLength[0], 0, lanes0, completed));
        CCU_RETURN_IF_ERROR(WaitDirectWideReads(ctx, false, completed));
        CCU_RETURN_IF_ERROR(StartDirectWideScratchTile(
            ctx, tileOffset1, ctx.tileLength[1], 1, lanes1, completed));
        CCU_RETURN_IF_ERROR(ReduceDirectWideTile(
            ctx, lanes0, ctx.tileLength[0], sourceCount, completed));

        CCU_RETURN_IF_ERROR(WaitDirectWideReads(ctx, false, completed));
        CCU_RETURN_IF_ERROR(StartDirectWideScratchTile(
            ctx, tileOffset2, ctx.tileLength[2], 0, lanes0, completed));
        CCU_RETURN_IF_ERROR(ReduceDirectWideTile(
            ctx, lanes1, ctx.tileLength[1], sourceCount, completed));

        CCU_RETURN_IF_ERROR(WaitDirectWideReads(ctx, false, completed));
        CCU_RETURN_IF_ERROR(ReduceDirectWideTile(
            ctx, lanes0, ctx.tileLength[2], sourceCount, completed));
        return CCU_SUCCESS;
    }

    static CcuResult RunDirectWideLocalInit(DirectContext &ctx)
    {
        const uint32_t sourceCount = ctx.arg->channelCount + 1U;
        ccu::Variable tileOffset0;
        tileOffset0 = 0;
        ccu::Variable tileOffset1;
        tileOffset1 = ctx.tileLength[0];
        ccu::Variable tileOffset2;
        tileOffset2 = ctx.tileLength[0];
        tileOffset2 += ctx.tileLength[1];
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> lanes0;
        std::array<ccu::LocalAddr, WIDE_LANE_MAX_SOURCES> lanes1;
        ccu::Event completed;

        CCU_RETURN_IF_ERROR(StartDirectWideLocalInitTile(
            ctx, tileOffset0, ctx.tileLength[0], 0, lanes0, completed));
        CCU_RETURN_IF_ERROR(WaitDirectWideReads(ctx, true, completed));
        CCU_RETURN_IF_ERROR(StartDirectWideLocalInitTile(
            ctx, tileOffset1, ctx.tileLength[1], 1, lanes1, completed));
        CCU_RETURN_IF_ERROR(ReduceDirectWideTile(
            ctx, lanes0, ctx.tileLength[0], sourceCount, completed));

        CCU_RETURN_IF_ERROR(WaitDirectWideReads(ctx, true, completed));
        CCU_RETURN_IF_ERROR(StartDirectWideLocalInitTile(
            ctx, tileOffset2, ctx.tileLength[2], 0, lanes0, completed));
        CCU_RETURN_IF_ERROR(ReduceDirectWideTile(
            ctx, lanes1, ctx.tileLength[1], sourceCount, completed));

        CCU_RETURN_IF_ERROR(WaitDirectWideReads(ctx, true, completed));
        CCU_RETURN_IF_ERROR(ReduceDirectWideTile(
            ctx, lanes0, ctx.tileLength[2], sourceCount, completed));
        return CCU_SUCCESS;
    }

    static CcuResult DirectPostSync(DirectContext &ctx)
    {
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(
                ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(
                ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    struct NodeLocalContext {
        const CcuKernelArgNodeLocal *arg{nullptr};
        ccu::Variable input;
        ccu::Variable output;
        ccu::Variable scratch;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable scratchToken;
        ccu::Variable chunkOffset;
        ccu::Variable alignedStride;
        std::array<ccu::Variable, HIERARCHY_MAX_TARGETS> targetOffset;
        std::array<ccu::Variable, HIERARCHY_MAX_TARGETS> targetLength;
        std::vector<ccu::Variable> peerInput;
        std::vector<ccu::Variable> peerToken;
    };

    static CcuResult InitNodeLocal(NodeLocalContext &ctx)
    {
        if (ctx.arg->targetCount == 0 || ctx.arg->targetCount > HIERARCHY_MAX_TARGETS
            || ctx.arg->channelCount == 0
            || ctx.arg->channelCount + 1U > MAX_GROUP_SOURCE_COUNT) {
            return CcuResult::CCU_E_PARA;
        }
        ctx.peerInput.resize(ctx.arg->channelCount);
        ctx.peerToken.resize(ctx.arg->channelCount);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            ctx.peerInput[i] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], INPUT_XN_ID);
            ctx.peerToken[i] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
        }
        return CCU_SUCCESS;
    }

    static CcuResult LoadNodeLocalArgs(NodeLocalContext &ctx)
    {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, 0));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, 1));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratch, 2));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, 3));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, 4));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, 5));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkOffset, 6));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.alignedStride, 7));
        // ExecuteHierarchy uses a fixed-width argument ABI so every registered
        // node-local kernel must emit the same 14 LoadArg instructions.  Slots
        // above targetCount are zero-filled by the host and are never consumed
        // by the reduction loop.
        for (uint32_t target = 0; target < HIERARCHY_MAX_TARGETS; ++target) {
            CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.targetOffset[target], 8 + target));
            CCU_RETURN_IF_ERROR(
                ccu::LoadArg(ctx.targetLength[target], 8 + HIERARCHY_MAX_TARGETS + target));
        }
        return CCU_SUCCESS;
    }

    static CcuResult NodeLocalPreSync(NodeLocalContext &ctx)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[i], ctx.input, INPUT_XN_ID, CKE_IDX_0, 1U << INPUT_XN_ID));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[i], ctx.inputToken, TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
        }
        constexpr uint32_t WAIT_BITS = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, WAIT_BITS));
        }
        return CCU_SUCCESS;
    }

    static CcuResult ReduceNodeTarget(NodeLocalContext &ctx, uint32_t target)
    {
        std::vector<ccu::RemoteAddr> remoteSrc(ctx.arg->channelCount);
        for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
            remoteSrc[channel].addr = ctx.peerInput[channel];
            remoteSrc[channel].addr += ctx.targetOffset[target];
            remoteSrc[channel].token = ctx.peerToken[channel];
        }
        ccu::LocalAddr localSrc;
        localSrc.addr = ctx.input;
        localSrc.addr += ctx.targetOffset[target];
        localSrc.token = ctx.inputToken;

        ccu::LocalAddr dst;
        if (target == 0) {
            dst.addr = ctx.output;
            dst.addr += ctx.chunkOffset;
            dst.token = ctx.outputToken;
        } else {
            dst.addr = ctx.scratch;
            for (uint32_t slot = 1; slot < target; ++slot) {
                dst.addr += ctx.alignedStride;
            }
            dst.addr += ctx.chunkOffset;
            dst.token = ctx.scratchToken;
        }

        std::array<ccu::LocalAddr, WORKSPACE_SLOT_COUNT> workspace;
        for (uint32_t slot = 0; slot < WORKSPACE_SLOT_COUNT; ++slot) {
            workspace[slot].addr = ctx.scratch;
            for (uint32_t offset = 0; offset < ctx.arg->targetCount - 1U + slot; ++offset) {
                workspace[slot].addr += ctx.alignedStride;
            }
            workspace[slot].token = ctx.scratchToken;
        }

        constexpr uint32_t LANE_COUNT = WORKSPACE_SLOT_COUNT + 1U;
        const uint32_t sourceCount = ctx.arg->channelCount + 1U;
        const uint32_t laneCount = std::min<uint32_t>(LANE_COUNT, sourceCount);
        std::array<ccu::LocalAddr, LANE_COUNT> lanes;
        lanes[0] = dst;
        for (uint32_t lane = 1; lane < laneCount; ++lane) {
            lanes[lane] = workspace[lane - 1U];
        }

        ccu::Event completed;
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(lanes[0], localSrc, ctx.targetLength[target], completed, 1U));
        uint32_t nextChannel = 0;
        for (uint32_t lane = 1; lane < laneCount; ++lane) {
            const uint16_t eventBit = static_cast<uint16_t>(1U << lane);
            CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[nextChannel], lanes[lane], remoteSrc[nextChannel],
                ctx.targetLength[target], completed, eventBit));
            ++nextChannel;
        }

        uint16_t secondReadMask = 0;
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            const uint16_t eventBit = static_cast<uint16_t>(1U << lane);
            CCU_RETURN_IF_ERROR(ccu::EventWait(completed, eventBit));
            if (nextChannel < ctx.arg->channelCount) {
                CCU_RETURN_IF_ERROR(ccu::ReadReduce(ctx.arg->channels[nextChannel], lanes[lane],
                    remoteSrc[nextChannel], ctx.targetLength[target], ctx.arg->dataType, ctx.arg->reduceOp,
                    completed, eventBit));
                secondReadMask |= eventBit;
                ++nextChannel;
            }
        }

        uint16_t pairMask = 0;
        uint32_t pairIndex = 0;
        for (uint32_t lane = 0; lane + 1U < laneCount; lane += 2U, ++pairIndex) {
            const uint16_t readyMask
                = secondReadMask & static_cast<uint16_t>((1U << lane) | (1U << (lane + 1U)));
            if (readyMask != 0) {
                CCU_RETURN_IF_ERROR(ccu::EventWait(completed, readyMask));
            }
            const uint16_t pairBit = static_cast<uint16_t>(1U << (8U + pairIndex));
            CCU_RETURN_IF_ERROR(ccu::LocalReduce(lanes[lane], lanes[lane + 1U], ctx.targetLength[target],
                ctx.arg->dataType, ctx.arg->reduceOp, completed, pairBit));
            pairMask |= pairBit;
        }
        if ((laneCount & 1U) != 0) {
            const uint16_t lastBit = static_cast<uint16_t>(1U << (laneCount - 1U));
            if ((secondReadMask & lastBit) != 0) {
                CCU_RETURN_IF_ERROR(ccu::EventWait(completed, lastBit));
            }
        }
        if (pairMask != 0) {
            CCU_RETURN_IF_ERROR(ccu::EventWait(completed, pairMask));
        }
        for (uint32_t lane = 2; lane < laneCount; lane += 2U) {
            CCU_RETURN_IF_ERROR(ccu::LocalReduce(lanes[0], lanes[lane], ctx.targetLength[target],
                ctx.arg->dataType, ctx.arg->reduceOp, completed, 1U));
            CCU_RETURN_IF_ERROR(ccu::EventWait(completed, 1U));
        }
        return CCU_SUCCESS;
    }

    static CcuResult NodeLocalPostSync(NodeLocalContext &ctx)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    struct CrossFinalizeContext {
        const CcuKernelArgCrossFinalize *arg{nullptr};
        ccu::Variable output;
        ccu::Variable scratch;
        ccu::Variable outputToken;
        ccu::Variable scratchToken;
        ccu::Variable chunkOffset;
        ccu::Variable chunkLength;
        ccu::Variable alignedStride;
        ccu::Variable chunkIndex;
        std::vector<ccu::Variable> peerScratch;
        std::vector<ccu::Variable> peerToken;
    };

    static CcuResult InitCrossFinalize(CrossFinalizeContext &ctx)
    {
        if (ctx.arg->channelCount == 0 || ctx.arg->channelCount > HIERARCHY_MAX_CROSS_PEERS
            || ctx.arg->ownerChannelIndex0 >= ctx.arg->channelCount
            || ctx.arg->ownerChannelIndex1 >= ctx.arg->channelCount
            || ctx.arg->remoteSlot > 1U) {
            return CcuResult::CCU_E_PARA;
        }
        ctx.peerScratch.resize(ctx.arg->channelCount);
        ctx.peerToken.resize(ctx.arg->channelCount);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            ctx.peerScratch[i] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], INPUT_XN_ID);
            ctx.peerToken[i] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
        }
        return CCU_SUCCESS;
    }

    static CcuResult LoadCrossFinalizeArgs(CrossFinalizeContext &ctx)
    {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, 0));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratch, 1));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, 2));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, 3));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkOffset, 4));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkLength, 5));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.alignedStride, 6));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkIndex, 7));
        return CCU_SUCCESS;
    }

    static CcuResult CrossFinalizePreSync(CrossFinalizeContext &ctx)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[i], ctx.scratch, INPUT_XN_ID, CKE_IDX_0, 1U << INPUT_XN_ID));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[i], ctx.scratchToken, TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
        }
        constexpr uint32_t WAIT_BITS = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, WAIT_BITS));
        }
        return CCU_SUCCESS;
    }

    static CcuResult ReadRemotePartial(CrossFinalizeContext &ctx, uint32_t channelIndex)
    {
        ccu::RemoteAddr src;
        src.addr = ctx.peerScratch[channelIndex];
        for (uint32_t slot = 0; slot < ctx.arg->remoteSlot; ++slot) {
            src.addr += ctx.alignedStride;
        }
        src.addr += ctx.chunkOffset;
        src.token = ctx.peerToken[channelIndex];
        ccu::LocalAddr dst;
        dst.addr = ctx.output;
        dst.addr += ctx.chunkOffset;
        dst.token = ctx.outputToken;
        ccu::Event completed;
        CCU_RETURN_IF_ERROR(ccu::ReadReduce(ctx.arg->channels[channelIndex], dst, src, ctx.chunkLength,
            ctx.arg->dataType, ctx.arg->reduceOp, completed, 1U));
        CCU_RETURN_IF_ERROR(ccu::EventWait(completed, 1U));
        return CCU_SUCCESS;
    }

    static CcuResult CrossFinalizePostSync(CrossFinalizeContext &ctx)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

} // namespace

CcuResult CcuReduceScatterPartialKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgPartialReduce *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }

    PartialReduceContext ctx;
    ctx.arg = kernelArg;

    CCU_RETURN_IF_ERROR(InitResources(ctx));
    CCU_RETURN_IF_ERROR(LoadArgs(ctx));
    if (kernelArg->compactArgs && kernelArg->includeSelf && kernelArg->channelCount == 3U
        && kernelArg->staticMsGoSize[0] == 0 && kernelArg->staticMsGoSize[1] == 0
        && kernelArg->staticMsGoSize[2] == EncodeParallelParam(0, 0, 1)
        && kernelArg->staticMsGoSize[3] == MS_SLICE_BYTES) {
        constexpr uint32_t WAIT_BITS
            = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
        constexpr uint16_t SOURCE_MASK = 0xFU;
        constexpr uint32_t SOURCE_COUNT = 4U;
        ccu::Variable length;
        length = kernelArg->staticMsGoSize[3];
        ccu::Event completed;
        ccu::Array<ccu::CcuBuffer> buffers(SOURCE_COUNT);
        ccu::LocalAddr localSource;
        ccu::RemoteAddr remoteSources[SOURCE_COUNT - 1U];
        ccu::LocalAddr destination;

        localSource.addr = ctx.myInput;
        localSource.addr += ctx.sliceOffset;
        localSource.token = ctx.inputToken;
        destination.addr = ctx.compactDestination;
        destination.token = ctx.compactDestinationToken;
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(buffers[3], localSource, length,
            completed, static_cast<uint16_t>(1U << 3U)));

        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i],
                ctx.myInput, INPUT_XN_ID, CKE_IDX_0, 1U << INPUT_XN_ID));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i],
                ctx.inputToken, TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
        }

        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(
                ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, WAIT_BITS));
            remoteSources[i].addr = ctx.peerInput[i];
            remoteSources[i].addr += ctx.sliceOffset;
            remoteSources[i].token = ctx.peerToken[i];
            CCU_RETURN_IF_ERROR(ccu::Read(kernelArg->channels[i], buffers[i],
                remoteSources[i], length, completed, static_cast<uint16_t>(1U << i)));
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(completed, SOURCE_MASK));

        CCU_RETURN_IF_ERROR(ccu::LocalReduce(&buffers[0], SOURCE_COUNT,
            kernelArg->dataType, kernelArg->outputDataType, kernelArg->reduceOp,
            length, completed, 1U));
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0,
                1U << POST_SYNC_ID));
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(completed, 1U));

        CCU_RETURN_IF_ERROR(
            ccu::LocalCopy(destination, buffers[0], length, completed, 1U));
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0,
                1U << POST_SYNC_ID));
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(completed, 1U));
        return CCU_SUCCESS;
    }
    if (kernelArg->enableRank4PackedMs) {
        CCU_RETURN_IF_ERROR(PreSync(ctx));
        CCU_RETURN_IF_ERROR(ReduceRank4PackedMs(ctx));
        return CCU_SUCCESS;
    }
    CCU_RETURN_IF_ERROR(PreSync(ctx));
    CCU_RETURN_IF_ERROR(ReduceLayer(ctx));
    CCU_RETURN_IF_ERROR(PostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterWriteReduce4x1Kernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgWriteReduce4x1 *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    if (kernelArg->channelCount != WRITE_REDUCE_4X1_STRIPES
        || kernelArg->myRank >= 4U || kernelArg->outputBytes == 0) {
        return CcuResult::CCU_E_PARA;
    }
    uint64_t coveredBytes = 0;
    std::array<bool, 4> seenTarget{};
    for (uint32_t stripe = 0; stripe < WRITE_REDUCE_4X1_STRIPES;
        ++stripe) {
        if (kernelArg->stripeOffsets[stripe] != coveredBytes
            || kernelArg->stripeLengths[stripe] == 0
            || kernelArg->stripeLengths[stripe] > kernelArg->outputBytes
            || coveredBytes
                > kernelArg->outputBytes - kernelArg->stripeLengths[stripe]) {
            return CcuResult::CCU_E_PARA;
        }
        coveredBytes += kernelArg->stripeLengths[stripe];
    }
    if (coveredBytes != kernelArg->outputBytes) {
        return CcuResult::CCU_E_PARA;
    }
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        const uint32_t target = kernelArg->targetRanks[channel];
        if (target >= 4U || target == kernelArg->myRank
            || seenTarget[target]
            || kernelArg->sourceIndicesAtTarget[channel]
                >= WRITE_REDUCE_4X1_STRIPES) {
            return CcuResult::CCU_E_PARA;
        }
        seenTarget[target] = true;
    }

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(input, 0));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(output, 1));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, 2));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, 3));

    std::array<ccu::Variable, WRITE_REDUCE_4X1_STRIPES>
        peerOutput;
    std::array<ccu::Variable, WRITE_REDUCE_4X1_STRIPES>
        peerOutputToken;
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        peerOutput[channel] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[channel], INPUT_XN_ID);
        peerOutputToken[channel]
            = ccu::GetResByChannel<ccu::Variable>(
                kernelArg->channels[channel], TOKEN_XN_ID);
    }

    ccu::Variable outputLength;
    outputLength = kernelArg->outputBytes;
    ccu::Variable selfSourceOffset;
    selfSourceOffset
        = static_cast<uint64_t>(kernelArg->myRank)
        * kernelArg->outputBytes;
    ccu::LocalAddr selfSource;
    selfSource.addr = input;
    selfSource.addr += selfSourceOffset;
    selfSource.token = inputToken;
    ccu::LocalAddr localOutput;
    localOutput.addr = output;
    localOutput.token = outputToken;
    ccu::Event completed;
    constexpr uint16_t SELF_COPY_BIT = 1U << 8U;
    CCU_RETURN_IF_ERROR(ccu::LocalCopy(localOutput, selfSource,
        outputLength, completed, SELF_COPY_BIT));

    // The recvBuf address publication overlaps the self copy.  Its token is
    // published only after initialization completes, so observing both XN
    // bits proves that endpoint WriteReduce may safely update peer recvBuf.
    // After this copy the local kernel never writes recvBuf again: within a
    // Latin wave the three inbound writers own disjoint stripes, and the
    // global wave barrier serializes reuse of each stripe across senders.
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            kernelArg->channels[channel], output, INPUT_XN_ID,
            CKE_IDX_0, 1U << INPUT_XN_ID));
    }
    CCU_RETURN_IF_ERROR(ccu::EventWait(completed, SELF_COPY_BIT));
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            kernelArg->channels[channel], outputToken, TOKEN_XN_ID,
            CKE_IDX_0, 1U << TOKEN_XN_ID));
    }
    constexpr uint32_t READY_BITS
        = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channel], CKE_IDX_0, READY_BITS));
    }

    std::array<ccu::RemoteAddr, WRITE_REDUCE_4X1_STRIPES>
        remoteDestination;
    std::array<ccu::LocalAddr, WRITE_REDUCE_4X1_STRIPES> localSource;
    std::array<ccu::Variable, WRITE_REDUCE_4X1_STRIPES>
        targetBaseOffset;
    std::array<ccu::Variable, WRITE_REDUCE_4X1_STRIPES> stripeOffset;
    std::array<ccu::Variable, WRITE_REDUCE_4X1_STRIPES> stripeLength;
    std::array<ccu::Variable, WRITE_REDUCE_4X1_STRIPES> transferLength;
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        targetBaseOffset[channel]
            = static_cast<uint64_t>(kernelArg->targetRanks[channel])
            * kernelArg->outputBytes;
    }
    for (uint32_t stripe = 0; stripe < WRITE_REDUCE_4X1_STRIPES;
        ++stripe) {
        stripeOffset[stripe] = kernelArg->stripeOffsets[stripe];
        stripeLength[stripe] = kernelArg->stripeLengths[stripe];
    }
    constexpr uint16_t ALL_CHANNELS_MASK
        = (1U << WRITE_REDUCE_4X1_STRIPES) - 1U;
    for (uint32_t round = 0; round < WRITE_REDUCE_4X1_STRIPES;
        ++round) {
        for (uint32_t channel = 0; channel < kernelArg->channelCount;
            ++channel) {
            const uint32_t stripe
                = (kernelArg->sourceIndicesAtTarget[channel]
                    + WRITE_REDUCE_4X1_STRIPES - round)
                % WRITE_REDUCE_4X1_STRIPES;
            remoteDestination[channel].addr = peerOutput[channel];
            remoteDestination[channel].addr += stripeOffset[stripe];
            remoteDestination[channel].token
                = peerOutputToken[channel];
            localSource[channel].addr = input;
            localSource[channel].addr += targetBaseOffset[channel];
            localSource[channel].addr += stripeOffset[stripe];
            localSource[channel].token = inputToken;
            transferLength[channel] = stripeLength[stripe];
            const uint16_t eventBit
                = static_cast<uint16_t>(1U << channel);
            CCU_RETURN_IF_ERROR(ccu::WriteReduce(
                kernelArg->channels[channel], remoteDestination[channel],
                localSource[channel], transferLength[channel],
                kernelArg->dataType, kernelArg->reduceOp, completed,
                eventBit));
        }
        CCU_RETURN_IF_ERROR(
            ccu::EventWait(completed, ALL_CHANNELS_MASK));

        // A local EventWait alone does not align different senders: a fast
        // rank could otherwise enter wave r+1 while another rank is still
        // updating the same target in wave r.  Because 4x1 is a complete
        // four-rank graph, record-all then wait-all is a global wave barrier.
        //
        // DATA uses 3/1/3 rather than consuming READY bits 0/2. Before a rank
        // records wave-2 bit 3, it has waited for every peer's wave-1 bit 1.
        // A peer can record bit 1 only after consuming this rank's wave-0 bit
        // 3, so the repeated bit is safe under arbitrary rank skew.
        const uint32_t barrierBit = round == 1U
            ? (1U << WRITE_REDUCE_ACK_SYNC_ID)
            : (1U << POST_SYNC_ID);
        for (uint32_t channel = 0; channel < kernelArg->channelCount;
            ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
                kernelArg->channels[channel], CKE_IDX_0, barrierBit));
        }
        for (uint32_t channel = 0; channel < kernelArg->channelCount;
            ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                kernelArg->channels[channel], CKE_IDX_0, barrierBit));
        }
    }

    // No fourth ACK is required. READY bits 0/2 are disjoint from the final
    // wave-2 bit 3. Receiving the next mission/op READY proves that the peer
    // returned from its prior mission and consumed this rank's final bit 3
    // before the following wave 0 may reuse it.
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterWriteReduceDualKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgWriteReduceDual *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    const bool validRank16Shape = kernelArg->rankSize == 16U
        && ((kernelArg->channelCount == 7U
                && kernelArg->stripeCount == 7U)
            || (kernelArg->channelCount == 8U
                && kernelArg->stripeCount == 8U));
    const bool validRank12Shape = kernelArg->rankSize == 12U
        && ((kernelArg->channelCount == 3U
                && kernelArg->stripeCount == 3U)
            || (kernelArg->channelCount == 7U
                && kernelArg->stripeCount == 7U)
            || (kernelArg->channelCount == 4U
                && kernelArg->stripeCount == 8U)
            || (kernelArg->channelCount == 8U
                && kernelArg->stripeCount == 8U));
    if ((!validRank16Shape && !validRank12Shape)
        || kernelArg->myRank >= kernelArg->rankSize
        || kernelArg->outputBytes == 0
        || kernelArg->phaseLength == 0
        || kernelArg->phaseOffset > kernelArg->outputBytes
        || kernelArg->phaseLength
            > kernelArg->outputBytes - kernelArg->phaseOffset) {
        return CcuResult::CCU_E_PARA;
    }

    uint64_t coveredBytes = 0;
    for (uint32_t stripe = 0; stripe < kernelArg->stripeCount; ++stripe) {
        if (kernelArg->stripeOffsets[stripe] != coveredBytes
            || kernelArg->stripeLengths[stripe] == 0
            || kernelArg->stripeLengths[stripe] > kernelArg->phaseLength
            || coveredBytes
                > kernelArg->phaseLength - kernelArg->stripeLengths[stripe]) {
            return CcuResult::CCU_E_PARA;
        }
        coveredBytes += kernelArg->stripeLengths[stripe];
    }
    if (coveredBytes != kernelArg->phaseLength) {
        return CcuResult::CCU_E_PARA;
    }

    std::array<bool, 16> seenTarget{};
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        const uint32_t target = kernelArg->targetRanks[channel];
        if (target >= kernelArg->rankSize || target == kernelArg->myRank
            || seenTarget[target]
            || kernelArg->sourceIndicesAtTarget[channel]
                >= kernelArg->stripeCount) {
            return CcuResult::CCU_E_PARA;
        }
        seenTarget[target] = true;
    }

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(input, 0));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(output, 1));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, 2));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, 3));

    std::array<ccu::Variable, WRITE_REDUCE_DUAL_MAX_CHANNELS> peerOutput;
    std::array<ccu::Variable, WRITE_REDUCE_DUAL_MAX_CHANNELS>
        peerOutputToken;
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        peerOutput[channel] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[channel], INPUT_XN_ID);
        peerOutputToken[channel]
            = ccu::GetResByChannel<ccu::Variable>(
                kernelArg->channels[channel], TOKEN_XN_ID);
    }

    ccu::Variable phaseOffset;
    phaseOffset = kernelArg->phaseOffset;
    ccu::Variable phaseLength;
    phaseLength = kernelArg->phaseLength;
    ccu::Event completed;
    constexpr uint16_t SELF_COPY_BIT = 1U << 8U;
    if (kernelArg->initializeRange) {
        ccu::Variable selfSourceOffset;
        selfSourceOffset
            = static_cast<uint64_t>(kernelArg->myRank)
            * kernelArg->outputBytes;
        selfSourceOffset += phaseOffset;
        ccu::LocalAddr selfSource;
        selfSource.addr = input;
        selfSource.addr += selfSourceOffset;
        selfSource.token = inputToken;
        ccu::LocalAddr localAccumulator;
        localAccumulator.addr = output;
        localAccumulator.addr += phaseOffset;
        localAccumulator.token = outputToken;
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(localAccumulator, selfSource,
            phaseLength, completed, SELF_COPY_BIT));
    }

    // Stage-1 address publication overlaps the self copy.  Stage 2 is ordered
    // after both stage-1 missions by the host-thread barrier, so it can publish
    // the already initialized range immediately.
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            kernelArg->channels[channel], output, INPUT_XN_ID,
            CKE_IDX_0, 1U << INPUT_XN_ID));
    }
    if (kernelArg->initializeRange) {
        CCU_RETURN_IF_ERROR(ccu::EventWait(completed, SELF_COPY_BIT));
    }
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            kernelArg->channels[channel], outputToken, TOKEN_XN_ID,
            CKE_IDX_0, 1U << TOKEN_XN_ID));
    }
    constexpr uint32_t READY_BITS
        = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channel], CKE_IDX_0, READY_BITS));
    }

    std::array<ccu::RemoteAddr, WRITE_REDUCE_DUAL_MAX_CHANNELS>
        remoteDestination;
    std::array<ccu::LocalAddr, WRITE_REDUCE_DUAL_MAX_CHANNELS> localSource;
    std::array<ccu::Variable, WRITE_REDUCE_DUAL_MAX_CHANNELS>
        targetBaseOffset;
    std::array<ccu::Variable, WRITE_REDUCE_DUAL_MAX_CHANNELS> stripeOffset;
    std::array<ccu::Variable, WRITE_REDUCE_DUAL_MAX_CHANNELS> stripeLength;
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        targetBaseOffset[channel]
            = static_cast<uint64_t>(kernelArg->targetRanks[channel])
            * kernelArg->outputBytes;
        targetBaseOffset[channel] += phaseOffset;
    }
    for (uint32_t stripe = 0; stripe < kernelArg->stripeCount; ++stripe) {
        stripeOffset[stripe] = kernelArg->stripeOffsets[stripe];
        stripeLength[stripe] = kernelArg->stripeLengths[stripe];
    }

    const uint16_t allChannelsMask = static_cast<uint16_t>(
        (1U << kernelArg->channelCount) - 1U);
    // K=8 is exactly the bipartite cross-node shape for every supported
    // rank-12/rank-16 layout.  Same-side writers have no direct channel, so a
    // target must first collect DATA from all incoming writers and then send
    // ACK to all of them.  K=3/K=7 are complete local cliques: their existing
    // one-phase alternating barrier already lets every writer wait directly
    // for every other writer.
    const bool requiresAggregateAck
        = kernelArg->stripeCount == WRITE_REDUCE_DUAL_MAX_CHANNELS;
    constexpr uint32_t WAVE_DATA_BIT = 1U << POST_SYNC_ID;
    constexpr uint32_t WAVE_ACK_BIT
        = 1U << WRITE_REDUCE_ACK_SYNC_ID;
    for (uint32_t round = 0; round < kernelArg->stripeCount; ++round) {
        for (uint32_t channel = 0; channel < kernelArg->channelCount;
            ++channel) {
            const uint32_t stripe
                = (kernelArg->sourceIndicesAtTarget[channel]
                    + kernelArg->stripeCount - round)
                % kernelArg->stripeCount;
            remoteDestination[channel].addr = peerOutput[channel];
            remoteDestination[channel].addr += phaseOffset;
            remoteDestination[channel].addr += stripeOffset[stripe];
            remoteDestination[channel].token = peerOutputToken[channel];
            localSource[channel].addr = input;
            localSource[channel].addr += targetBaseOffset[channel];
            localSource[channel].addr += stripeOffset[stripe];
            localSource[channel].token = inputToken;
            const uint16_t eventBit
                = static_cast<uint16_t>(1U << channel);
            CCU_RETURN_IF_ERROR(ccu::WriteReduce(
                kernelArg->channels[channel], remoteDestination[channel],
                localSource[channel], stripeLength[stripe],
                kernelArg->dataType, kernelArg->reduceOp, completed,
                eventBit));
        }
        CCU_RETURN_IF_ERROR(
            ccu::EventWait(completed, allChannelsMask));

        // A local clique needs one alternating DATA epoch.  A cross phase
        // needs DATA(bit 3) followed by ACK(bit 1): target T records ACK only
        // after consuming DATA from every incoming writer, so writer S cannot
        // enter the next wave until every previous writer to each target has
        // completed.  Waiting ACK also proves that T consumed S's DATA before
        // bit 3 is reused; waiting the next DATA proves that the peer consumed
        // the prior ACK before bit 1 is reused.  READY bits 0/2 remain
        // disjoint and close the final ACK epoch across mission/op boundaries.
        const uint32_t waveBit = requiresAggregateAck
            ? WAVE_DATA_BIT
            : ((round & 1U) == 0U ? WAVE_DATA_BIT : WAVE_ACK_BIT);
        for (uint32_t channel = 0; channel < kernelArg->channelCount;
            ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
                kernelArg->channels[channel], CKE_IDX_0, waveBit));
        }
        for (uint32_t channel = 0; channel < kernelArg->channelCount;
            ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                kernelArg->channels[channel], CKE_IDX_0, waveBit));
        }
        if (requiresAggregateAck) {
            for (uint32_t channel = 0;
                channel < kernelArg->channelCount; ++channel) {
                CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
                    kernelArg->channels[channel], CKE_IDX_0,
                    WAVE_ACK_BIT));
            }
            for (uint32_t channel = 0;
                channel < kernelArg->channelCount; ++channel) {
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                    kernelArg->channels[channel], CKE_IDX_0,
                    WAVE_ACK_BIT));
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterOwnerPullDualKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgWriteReduceDual *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }

    // The host registers this kernel only for the two official rank-16 large
    // receive counts. Rank-12, rank-4, all small points, and hidden shapes
    // remain on the byte-identical measured V12w paths.
    const bool validRank16Shape = kernelArg->rankSize == 16U
        && ((kernelArg->channelCount == 7U
                && kernelArg->stripeCount == 7U)
            || (kernelArg->channelCount == 8U
                && kernelArg->stripeCount == 8U));
    if (!validRank16Shape || kernelArg->myRank >= kernelArg->rankSize
        || kernelArg->outputBytes == 0U
        || kernelArg->phaseLength == 0U
        || kernelArg->phaseOffset > kernelArg->outputBytes
        || kernelArg->phaseLength
            > kernelArg->outputBytes - kernelArg->phaseOffset) {
        return CcuResult::CCU_E_PARA;
    }

    // Local P7 uses all seven partitions; cross P1 is encoded as the
    // one-element non-zero prefix of the eight synchronized color slots.
    uint32_t partitionCount = kernelArg->stripeCount;
    if (kernelArg->stripeCount == WRITE_REDUCE_DUAL_MAX_CHANNELS) {
        partitionCount = 0U;
        while (partitionCount < WRITE_REDUCE_DUAL_MAX_CHANNELS
            && kernelArg->stripeLengths[partitionCount] != 0U) {
            ++partitionCount;
        }
        if (partitionCount != 1U && partitionCount != 2U
            && partitionCount != 4U
            && partitionCount != WRITE_REDUCE_DUAL_MAX_CHANNELS) {
            return CcuResult::CCU_E_PARA;
        }
        for (uint32_t stripe = partitionCount;
            stripe < WRITE_REDUCE_DUAL_MAX_CHANNELS; ++stripe) {
            if (kernelArg->stripeOffsets[stripe] != 0U
                || kernelArg->stripeLengths[stripe] != 0U) {
                return CcuResult::CCU_E_PARA;
            }
        }
    }

    uint64_t coveredBytes = 0U;
    for (uint32_t stripe = 0; stripe < partitionCount; ++stripe) {
        if (kernelArg->stripeOffsets[stripe] != coveredBytes
            || kernelArg->stripeLengths[stripe] == 0U
            || kernelArg->stripeLengths[stripe] > kernelArg->phaseLength
            || coveredBytes
                > kernelArg->phaseLength - kernelArg->stripeLengths[stripe]) {
            return CcuResult::CCU_E_PARA;
        }
        coveredBytes += kernelArg->stripeLengths[stripe];
    }
    if (coveredBytes != kernelArg->phaseLength) {
        return CcuResult::CCU_E_PARA;
    }

    std::array<bool, 16> seenPeer{};
    std::array<bool, WRITE_REDUCE_DUAL_MAX_CHANNELS> seenResidue{};
    for (uint32_t channel = 0; channel < kernelArg->channelCount;
        ++channel) {
        const uint32_t peer = kernelArg->targetRanks[channel];
        const uint32_t residue
            = kernelArg->sourceIndicesAtTarget[channel];
        if (peer >= kernelArg->rankSize || peer == kernelArg->myRank
            || seenPeer[peer] || residue >= kernelArg->stripeCount
            || seenResidue[residue]) {
            return CcuResult::CCU_E_PARA;
        }
        seenPeer[peer] = true;
        seenResidue[residue] = true;
    }

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(input, 0));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(output, 1));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, 2));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, 3));

    std::array<ccu::Variable, WRITE_REDUCE_DUAL_MAX_CHANNELS>
        peerInput;
    std::array<ccu::Variable, WRITE_REDUCE_DUAL_MAX_CHANNELS>
        peerInputToken;
    for (uint32_t channel = 0; channel < kernelArg->channelCount;
        ++channel) {
        peerInput[channel] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[channel], INPUT_XN_ID);
        peerInputToken[channel] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[channel], TOKEN_XN_ID);
    }

    ccu::Variable phaseOffset;
    phaseOffset = kernelArg->phaseOffset;
    ccu::Variable phaseLength;
    phaseLength = kernelArg->phaseLength;
    ccu::Event completed;
    constexpr uint16_t SELF_COPY_BIT = 1U << 8U;
    if (kernelArg->initializeRange) {
        ccu::Variable selfSourceOffset;
        selfSourceOffset
            = static_cast<uint64_t>(kernelArg->myRank)
            * kernelArg->outputBytes;
        selfSourceOffset += phaseOffset;
        ccu::LocalAddr selfSource;
        selfSource.addr = input;
        selfSource.addr += selfSourceOffset;
        selfSource.token = inputToken;
        ccu::LocalAddr localAccumulator;
        localAccumulator.addr = output;
        localAccumulator.addr += phaseOffset;
        localAccumulator.token = outputToken;
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(localAccumulator, selfSource,
            phaseLength, completed, SELF_COPY_BIT));

        // Publish immutable sendBuf metadata once for this channel set. Both
        // XNs overlap the independent self copy; EventWait still orders every
        // remote accumulation after initialization.
        for (uint32_t channel = 0; channel < kernelArg->channelCount;
            ++channel) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                kernelArg->channels[channel], input, INPUT_XN_ID,
                CKE_IDX_0, 1U << INPUT_XN_ID));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                kernelArg->channels[channel], inputToken, TOKEN_XN_ID,
                CKE_IDX_0, 1U << TOKEN_XN_ID));
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(completed, SELF_COPY_BIT));

        constexpr uint32_t READY_BITS
            = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t channel = 0; channel < kernelArg->channelCount;
            ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                kernelArg->channels[channel], CKE_IDX_0, READY_BITS));
        }
    }

    ccu::Variable sourceBaseOffset;
    sourceBaseOffset
        = static_cast<uint64_t>(kernelArg->myRank)
        * kernelArg->outputBytes;
    sourceBaseOffset += phaseOffset;
    std::array<ccu::RemoteAddr, WRITE_REDUCE_DUAL_MAX_CHANNELS>
        remoteSource;
    std::array<ccu::LocalAddr, WRITE_REDUCE_DUAL_MAX_CHANNELS>
        localDestination;
    std::array<ccu::Variable, WRITE_REDUCE_DUAL_MAX_CHANNELS>
        stripeOffset;
    std::array<ccu::Variable, WRITE_REDUCE_DUAL_MAX_CHANNELS>
        stripeLength;
    for (uint32_t stripe = 0; stripe < partitionCount; ++stripe) {
        stripeOffset[stripe] = kernelArg->stripeOffsets[stripe];
        stripeLength[stripe] = kernelArg->stripeLengths[stripe];
    }
    for (uint32_t channel = 0; channel < kernelArg->channelCount;
        ++channel) {
        const uint32_t residue
            = kernelArg->sourceIndicesAtTarget[channel];
        const uint32_t initialStripe
            = residue < partitionCount ? residue : 0U;
        remoteSource[channel].addr = peerInput[channel];
        remoteSource[channel].addr += sourceBaseOffset;
        remoteSource[channel].addr += stripeOffset[initialStripe];
        remoteSource[channel].token = peerInputToken[channel];
        localDestination[channel].addr = output;
        localDestination[channel].addr += phaseOffset;
        localDestination[channel].addr += stripeOffset[initialStripe];
        localDestination[channel].token = outputToken;
    }

    // Each round maps active channels to disjoint owner partitions.
    // EventWait completes the round before any destination address advances,
    // eliminating the distributed target WAW of sender-side WriteReduce.
    for (uint32_t round = 0; round < kernelArg->stripeCount; ++round) {
        uint16_t activeChannelsMask = 0U;
        for (uint32_t channel = 0; channel < kernelArg->channelCount;
            ++channel) {
            const uint32_t stripe
                = (kernelArg->sourceIndicesAtTarget[channel] + round)
                % kernelArg->stripeCount;
            if (stripe >= partitionCount) {
                continue;
            }
            const uint16_t eventBit
                = static_cast<uint16_t>(1U << channel);
            activeChannelsMask = static_cast<uint16_t>(
                activeChannelsMask | eventBit);
            CCU_RETURN_IF_ERROR(ccu::ReadReduce(
                kernelArg->channels[channel], localDestination[channel],
                remoteSource[channel], stripeLength[stripe],
                kernelArg->dataType, kernelArg->reduceOp, completed,
                eventBit));
        }
        if (activeChannelsMask != 0U) {
            CCU_RETURN_IF_ERROR(
                ccu::EventWait(completed, activeChannelsMask));
        }

        if (round + 1U < kernelArg->stripeCount) {
            for (uint32_t channel = 0;
                channel < kernelArg->channelCount; ++channel) {
                const uint32_t stripe
                    = (kernelArg->sourceIndicesAtTarget[channel] + round)
                    % kernelArg->stripeCount;
                if (stripe >= partitionCount) {
                    continue;
                }
                if (stripe + 1U == partitionCount) {
                    remoteSource[channel].addr = peerInput[channel];
                    remoteSource[channel].addr += sourceBaseOffset;
                    localDestination[channel].addr = output;
                    localDestination[channel].addr += phaseOffset;
                } else {
                    remoteSource[channel].addr += stripeLength[stripe];
                    localDestination[channel].addr
                        += stripeLength[stripe];
                }
            }
        }
    }

    if (!kernelArg->initializeRange) {
        // Stage 2 is the sole source-lifetime close. A target records only
        // after all remote reads complete, then waits every source. Return
        // therefore proves all peers stopped reading this sendBuf.
        constexpr uint32_t POST_BIT = 1U << POST_SYNC_ID;
        for (uint32_t channel = 0; channel < kernelArg->channelCount;
            ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
                kernelArg->channels[channel], CKE_IDX_0, POST_BIT));
        }
        for (uint32_t channel = 0; channel < kernelArg->channelCount;
            ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                kernelArg->channels[channel], CKE_IDX_0, POST_BIT));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterMateLocalKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgMateLocal *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    MateLocalContext ctx;
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitMateLocal(ctx));
    CCU_RETURN_IF_ERROR(LoadMateLocalArgs(ctx));
    CCU_RETURN_IF_ERROR(MateLocalPreSync(ctx));
    CCU_RETURN_IF_ERROR(RunMateLocalCyclic(ctx));
    CCU_RETURN_IF_ERROR(MateLocalPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterMateLocalExpandedKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgMateLocal *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    MateLocalContext ctx;
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitMateLocalExpanded(ctx));
    CCU_RETURN_IF_ERROR(LoadMateLocalArgs(ctx));
    CCU_RETURN_IF_ERROR(MateLocalPreSync(ctx));
    CCU_RETURN_IF_ERROR(RunMateLocalCyclic(ctx));
    CCU_RETURN_IF_ERROR(MateLocalPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterMateCrossKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgMateCross *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    MateCrossContext ctx;
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitMateCross(ctx));
    CCU_RETURN_IF_ERROR(LoadMateCrossArgs(ctx));
    CCU_RETURN_IF_ERROR(MateCrossPreSync(ctx));
    CCU_RETURN_IF_ERROR(ReduceMateCross(ctx));
    CCU_RETURN_IF_ERROR(MateCrossPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterDirectCrossKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDirect *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    DirectContext ctx;
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitDirect(ctx, false));
    CCU_RETURN_IF_ERROR(LoadDirectArgs(ctx));
    CCU_RETURN_IF_ERROR(DirectPreSync(ctx));
    CCU_RETURN_IF_ERROR(RunDirectWide(ctx, false));
    CCU_RETURN_IF_ERROR(DirectPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterDirectCrossScratchKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDirect *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    DirectContext ctx;
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitDirect(ctx, false));
    CCU_RETURN_IF_ERROR(LoadDirectArgs(ctx));
    CCU_RETURN_IF_ERROR(DirectPreSync(ctx));
    CCU_RETURN_IF_ERROR(RunDirectWideScratch(ctx));
    CCU_RETURN_IF_ERROR(DirectPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterDirectLocalKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDirect *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    DirectContext ctx;
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitDirect(ctx, true));
    CCU_RETURN_IF_ERROR(LoadDirectArgs(ctx));
    CCU_RETURN_IF_ERROR(DirectPreSync(ctx));
    CCU_RETURN_IF_ERROR(RunDirectWide(ctx, true));
    CCU_RETURN_IF_ERROR(DirectPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterDirectLocalInitKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDirect *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    DirectContext ctx;
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitDirect(ctx, true));
    CCU_RETURN_IF_ERROR(LoadDirectArgs(ctx));
    CCU_RETURN_IF_ERROR(DirectPreSync(ctx));
    CCU_RETURN_IF_ERROR(RunDirectWideLocalInit(ctx));
    CCU_RETURN_IF_ERROR(DirectPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterPublisherKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgPublisher *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 || kernelArg->channelCount > MATE_MAX_PEERS) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable scratch;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchToken;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(input, 0));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(output, 1));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(scratch, 2));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, 3));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, 4));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(scratchToken, 5));

    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            kernelArg->channels[channel], scratch, INPUT_XN_ID, CKE_IDX_0,
            1U << INPUT_XN_ID));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            kernelArg->channels[channel], scratchToken, TOKEN_XN_ID, CKE_IDX_0,
            1U << TOKEN_XN_ID));
    }
    constexpr uint32_t WAIT_BITS = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(kernelArg->channels[channel], CKE_IDX_0, WAIT_BITS));
    }
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            kernelArg->channels[channel], CKE_IDX_0, 1U << POST_SYNC_ID));
    }
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channel], CKE_IDX_0, 1U << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterPinnedCombineKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgCombine *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0
        || kernelArg->channelCount > MATE_MAX_PEERS
        || !kernelArg->compactArgs || kernelArg->stripeCount != 1
        || kernelArg->staticStripeLength == 0) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::Variable output;
    ccu::Variable scratch;
    ccu::Variable outputToken;
    ccu::Variable scratchToken;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(output, 0));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(scratch, 1));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, 2));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(scratchToken, 3));

    ccu::Variable length;
    length = kernelArg->staticStripeLength;
    ccu::LocalAddr destination;
    destination.addr = output;
    destination.token = outputToken;
    ccu::LocalAddr source;
    source.addr = scratch;
    source.token = scratchToken;
    ccu::Event completed;
    CCU_RETURN_IF_ERROR(ccu::LocalReduce(destination, source, length,
        kernelArg->dataType, kernelArg->reduceOp, completed, 1U));
    CCU_RETURN_IF_ERROR(ccu::EventWait(completed, 1U));

    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            kernelArg->channels[channel], CKE_IDX_0,
            1U << PINNED_COMBINE_SYNC_ID));
    }
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            kernelArg->channels[channel], CKE_IDX_0,
            1U << PINNED_COMBINE_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterNodeLocalKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgNodeLocal *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    NodeLocalContext ctx;
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitNodeLocal(ctx));
    CCU_RETURN_IF_ERROR(LoadNodeLocalArgs(ctx));
    CCU_RETURN_IF_ERROR(NodeLocalPreSync(ctx));
    for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
        CCU_IF(ctx.targetLength[target] != 0)
        {
            CCU_RETURN_IF_ERROR(ReduceNodeTarget(ctx, target));
        }
    }
    CCU_RETURN_IF_ERROR(NodeLocalPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterCrossFinalizeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgCrossFinalize *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    CrossFinalizeContext ctx;
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitCrossFinalize(ctx));
    CCU_RETURN_IF_ERROR(LoadCrossFinalizeArgs(ctx));
    CCU_RETURN_IF_ERROR(CrossFinalizePreSync(ctx));
    if (kernelArg->ownerChannelIndex0 == kernelArg->ownerChannelIndex1) {
        CCU_RETURN_IF_ERROR(ReadRemotePartial(ctx, kernelArg->ownerChannelIndex0));
    } else {
        CCU_IF(ctx.chunkIndex == 0)
        {
            CCU_RETURN_IF_ERROR(ReadRemotePartial(ctx, kernelArg->ownerChannelIndex0));
        }
        CCU_IF(ctx.chunkIndex != 0)
        {
            CCU_RETURN_IF_ERROR(ReadRemotePartial(ctx, kernelArg->ownerChannelIndex1));
        }
    }
    CCU_RETURN_IF_ERROR(CrossFinalizePostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterCombineKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgCombine *>(arg);
    if (kernelArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }

    ccu::Variable output;
    ccu::Variable scratch;
    ccu::Variable outputToken;
    ccu::Variable scratchToken;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(output, 0));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(scratch, 1));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, 2));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(scratchToken, 3));
    if (kernelArg->stripeCount == 0 || kernelArg->stripeCount > MAX_COMBINE_STRIPES) {
        return CcuResult::CCU_E_PARA;
    }

    if (kernelArg->compactArgs) {
        if (kernelArg->stripeCount != 1 || kernelArg->staticStripeLength == 0) {
            return CcuResult::CCU_E_PARA;
        }
        ccu::Variable length;
        length = kernelArg->staticStripeLength;
        ccu::LocalAddr dst;
        dst.addr = output;
        dst.token = outputToken;
        ccu::LocalAddr src;
        src.addr = scratch;
        src.token = scratchToken;
        ccu::Event completed;
        CCU_RETURN_IF_ERROR(ccu::LocalReduce(
            dst, src, length, kernelArg->dataType, kernelArg->reduceOp, completed, 1U));
        CCU_RETURN_IF_ERROR(ccu::EventWait(completed, 1U));
        return CCU_SUCCESS;
    }

    std::array<ccu::Variable, MAX_COMBINE_STRIPES> stripeOffset;
    std::array<ccu::Variable, MAX_COMBINE_STRIPES> stripeLength;
    for (uint32_t stripe = 0; stripe < kernelArg->stripeCount; ++stripe) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(stripeOffset[stripe], 4 + stripe));
        CCU_RETURN_IF_ERROR(
            ccu::LoadArg(stripeLength[stripe], 4 + kernelArg->stripeCount + stripe));
    }

    ccu::Event completed;
    uint16_t activeMask = 0;
    for (uint32_t stripe = 0; stripe < kernelArg->stripeCount; ++stripe) {
        ccu::LocalAddr dst;
        dst.addr = output;
        dst.addr += stripeOffset[stripe];
        dst.token = outputToken;
        ccu::LocalAddr src;
        src.addr = scratch;
        src.addr += stripeOffset[stripe];
        src.token = scratchToken;
        const uint16_t eventBit = static_cast<uint16_t>(1U << stripe);
        CCU_RETURN_IF_ERROR(ccu::LocalReduce(
            dst, src, stripeLength[stripe], kernelArg->dataType, kernelArg->reduceOp, completed, eventBit));
        activeMask |= eventBit;
    }
    CCU_RETURN_IF_ERROR(ccu::EventWait(completed, activeMask));
    return CCU_SUCCESS;
}

} // namespace ops_hccl
