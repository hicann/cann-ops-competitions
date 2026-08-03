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

#define ALLGATHER_CCU_CHK_RET(call) \
    do { \
        CcuResult ccuRet = static_cast<CcuResult>(call); \
        if (ccuRet != CCU_SUCCESS) { \
            return ccuRet; \
        } \
    } while (0)

namespace ops_hccl {
namespace {
    constexpr uint32_t OUTPUT_RESOURCE_ID = 1;
    constexpr uint32_t TOKEN_RESOURCE_ID = 2;
    constexpr uint32_t CHANNEL_EVENT_ID = 0;
    constexpr uint32_t POST_SYNC_BIT = 3;

    uint64_t GetParallelParam(uint64_t repeatCount, uint64_t repeatLoopIndex, uint64_t totalLoopCount)
    {
        constexpr uint64_t FIELD_MASK = (uint64_t{1} << 7) - 1;
        return ((repeatCount & FIELD_MASK) << 55) | ((repeatLoopIndex & FIELD_MASK) << 48)
               | ((totalLoopCount & FIELD_MASK) << 41);
    }

    uint64_t GetOffsetParam(uint64_t globalOffset, uint64_t memorySliceOffset, uint64_t eventOffset)
    {
        constexpr uint64_t GLOBAL_MASK = (uint64_t{1} << 32) - 1;
        constexpr uint64_t MEMORY_SLICE_MASK = (uint64_t{1} << 11) - 1;
        constexpr uint64_t EVENT_MASK = (uint64_t{1} << 10) - 1;
        return ((globalOffset & GLOBAL_MASK) << 21) | ((memorySliceOffset & MEMORY_SLICE_MASK) << 10)
               | (eventOffset & EVENT_MASK);
    }

    uint64_t GetLoopParam(uint64_t globalOffset, uint64_t iterations)
    {
        constexpr uint64_t GLOBAL_MASK = (uint64_t{1} << 32) - 1;
        constexpr uint64_t ITERATION_MASK = (uint64_t{1} << 13) - 1;
        return ((globalOffset & GLOBAL_MASK) << 13) | (iterations & ITERATION_MASK);
    }

    void InitializeCopyLoops(
        AllGatherContext &ctx, ccu::LocalAddr *sources, ccu::LocalAddr *destinations, ccu::Variable *lengths)
    {
        if (!ctx.copyResourcesAllocated) {
            ctx.copyEvents = ccu::Array<ccu::Event>(LOCAL_COPY_LOOP_COUNT);
            ctx.copyBuffers = ccu::Array<ccu::CcuBuffer>(LOCAL_COPY_LOOP_COUNT * CCU_MEMORY_SLICE_INTERLEAVE);
            ctx.copyResourcesAllocated = true;
        }

        const std::string key = "allgather_local_copy";
        if (ctx.loopEntities.count(key) != 0) {
            return;
        }

        ctx.loopEntities.emplace(key, LoopEntity());
        LoopEntity &entity = ctx.loopEntities[key];
        for (uint32_t index = 0; index < 2; ++index) {
            const uint32_t bufferBase = index * CCU_MEMORY_SLICE_INTERLEAVE;
            const ccu::Event copyEvent = ctx.copyEvents[index];
            entity.bodies[index].reset(
                new ccu::Func([&ctx, index, bufferBase, copyEvent, sources, destinations, lengths]() {
                    ccu::LocalCopy(ctx.copyBuffers[bufferBase], sources[index], lengths[index], copyEvent, 1);
                    ccu::EventWait(copyEvent, 1);
                    ccu::LocalCopy(destinations[index], ctx.copyBuffers[bufferBase], lengths[index], copyEvent, 1);
                    ccu::EventWait(copyEvent, 1);
                }));
            entity.loops[index].reset(new ccu::Loop(entity.parameters[index], *entity.bodies[index]));
        }
    }

    CcuResult GroupCopy(AllGatherContext &ctx, ccu::LocalAddr destination, ccu::LocalAddr source)
    {
        ccu::LocalAddr sources[2];
        ccu::LocalAddr destinations[2];
        ccu::Variable lengths[2];
        InitializeCopyLoops(ctx, sources, destinations, lengths);
        LoopEntity &entity = ctx.loopEntities["allgather_local_copy"];

        CCU_IF(ctx.groupCopySize.addressOffset != 0)
        {
            ccu::Variable loopParameter;
            loopParameter = GetLoopParam(LOCAL_COPY_MEMORY_SLICE * LOCAL_COPY_LOOP_COUNT, 0);
            loopParameter += ctx.groupCopySize.loopParam;
            ccu::Variable fixedLength;
            fixedLength = LOCAL_COPY_MEMORY_SLICE;

            sources[0].addr = source.addr;
            sources[0].token = source.token;
            destinations[0].addr = destination.addr;
            destinations[0].token = destination.token;
            lengths[0] = fixedLength;
            entity.parameters[0] = loopParameter;

            ccu::Variable parallelParameter;
            parallelParameter = GetParallelParam(LOCAL_COPY_LOOP_COUNT - 1, 0, 1);
            ccu::Variable offsetParameter;
            offsetParameter = GetOffsetParam(LOCAL_COPY_MEMORY_SLICE, CCU_MEMORY_SLICE_INTERLEAVE, 1);
            std::vector<ccu::Loop> loops{*entity.loops[0]};
            ccu::LoopGroup group(parallelParameter, offsetParameter, LOCAL_COPY_LOOP_COUNT, loops);
        }

        CCU_IF(ctx.groupCopySize.parallelParam != 0)
        {
            source.addr += ctx.groupCopySize.addressOffset;
            destination.addr += ctx.groupCopySize.addressOffset;

            sources[0].addr = source.addr;
            sources[0].token = source.token;
            destinations[0].addr = destination.addr;
            destinations[0].token = destination.token;
            lengths[0] = ctx.groupCopySize.residual;

            source.addr += ctx.groupCopySize.residual;
            destination.addr += ctx.groupCopySize.residual;
            ccu::Variable fixedLength;
            fixedLength = LOCAL_COPY_MEMORY_SLICE;
            sources[1].addr = source.addr;
            sources[1].token = source.token;
            destinations[1].addr = destination.addr;
            destinations[1].token = destination.token;
            lengths[1] = fixedLength;

            ccu::Variable singleIteration;
            singleIteration = GetLoopParam(0, 1);
            entity.parameters[0] = singleIteration;
            entity.parameters[1] = singleIteration;
            ccu::Variable offsetParameter;
            offsetParameter = GetOffsetParam(LOCAL_COPY_MEMORY_SLICE, CCU_MEMORY_SLICE_INTERLEAVE, 1);
            std::vector<ccu::Loop> loops{*entity.loops[0], *entity.loops[1]};
            ccu::LoopGroup group(ctx.groupCopySize.parallelParam, offsetParameter, LOCAL_COPY_LOOP_COUNT, loops);
        }
        return CCU_SUCCESS;
    }

    // 优化版 Transfer：
    //   1. 所有 remote Write 并发发出（充分利用多 channel 并发带宽，不等待完成）
    //   2. Write 发出后立即执行 local copy（隐藏网络传输延迟，overlap 计算与通信）
    //   3. 最后统一 EventWait（等待 remote write + local copy 全部完成）
    CcuResult Transfer(AllGatherContext &ctx)
    {
        ccu::LocalAddr source;
        source.addr = ctx.input;
        source.addr += ctx.inputOffset;
        source.token = ctx.tokens[ctx.arg->rankId];

        ccu::LocalAddr localDestination;
        std::vector<ccu::RemoteAddr> remoteDestinations;
        remoteDestinations.resize(ctx.arg->rankSize);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const uint32_t rank = ctx.arg->peerRanks[channelIndex];
            remoteDestinations[rank].addr = ctx.outputs[rank];
            remoteDestinations[rank].addr += ctx.outputOffset;
            remoteDestinations[rank].token = ctx.tokens[rank];
        }
        if (ctx.arg->copyLocal) {
            localDestination.addr = ctx.outputs[ctx.arg->rankId];
            localDestination.addr += ctx.outputOffset;
            localDestination.token = ctx.tokens[ctx.arg->rankId];
        }

        // 阶段1：并发发出所有 remote Write（DMA 开始传输，不阻塞 CCU）
        uint16_t waitMask = 0;
        CCU_IF(ctx.sliceSize != 0)
        {
            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
                const uint32_t rank = ctx.arg->peerRanks[channelIndex];
                const uint16_t rankMask = static_cast<uint16_t>(uint32_t{1} << rank);
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], remoteDestinations[rank], source,
                    ctx.sliceSize, ctx.transferEvent, rankMask));
                waitMask |= rankMask;
            }
        }

        // 阶段2：Write 在网络传输期间执行 local copy，overlap 网络延迟
        if (ctx.arg->copyLocal) {
            ALLGATHER_CCU_CHK_RET(GroupCopy(ctx, localDestination, source));
            const uint16_t localMask = static_cast<uint16_t>(uint32_t{1} << ctx.arg->rankId);
            ALLGATHER_CCU_CHK_RET(ccu::EventRecord(ctx.transferEvent, localMask));
            waitMask |= localMask;
        }

        // 阶段3：等待所有传输完成（remote write + local copy）
        ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.transferEvent, waitMask));
        return CCU_SUCCESS;
    }
} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<AllGatherKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0
        || kernelArg->channelCount != kernelArg->peerRanks.size()) {
        return CCU_E_PARA;
    }

    AllGatherContext ctx;
    ctx.arg = kernelArg;
    ctx.outputs.resize(ctx.arg->rankSize);
    ctx.tokens.resize(ctx.arg->rankSize);

    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        const uint32_t rank = ctx.arg->peerRanks[channelIndex];
        if (rank >= ctx.arg->rankSize || rank == ctx.arg->rankId) {
            return CCU_E_PARA;
        }
        ctx.outputs[rank] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], OUTPUT_RESOURCE_ID);
        ctx.tokens[rank] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], TOKEN_RESOURCE_ID);
    }

    uint32_t argIndex = 0;
    ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.input, argIndex++));
    ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.outputs[ctx.arg->rankId], argIndex++));
    ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.tokens[ctx.arg->rankId], argIndex++));
    ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.inputOffset, argIndex++));
    ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argIndex++));
    ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argIndex++));
    ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.addressOffset, argIndex++));
    ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.loopParam, argIndex++));
    ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.parallelParam, argIndex++));
    ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.residual, argIndex++));

    // 广播本 rank 的 output 地址和 token 给所有对端，对端收到后才能向本端 Write
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[index], ctx.outputs[ctx.arg->rankId],
            OUTPUT_RESOURCE_ID, CHANNEL_EVENT_ID, uint32_t{1} << OUTPUT_RESOURCE_ID));
        ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[index], ctx.tokens[ctx.arg->rankId],
            TOKEN_RESOURCE_ID, CHANNEL_EVENT_ID, uint32_t{1} << TOKEN_RESOURCE_ID));
    }
    // 等待所有对端的地址就绪通知
    const uint32_t addressReadyMask = (uint32_t{1} << OUTPUT_RESOURCE_ID) | (uint32_t{1} << TOKEN_RESOURCE_ID);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ALLGATHER_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[index], CHANNEL_EVENT_ID, addressReadyMask));
    }

    // 执行数据传输（remote write 与 local copy 流水线化）
    ALLGATHER_CCU_CHK_RET(Transfer(ctx));

    // Post-sync：先并发发出所有 NotifyRecord，再统一等待所有 NotifyWait，减少串行等待
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ALLGATHER_CCU_CHK_RET(
            ccu::NotifyRecord(ctx.arg->channels[index], CHANNEL_EVENT_ID, uint32_t{1} << POST_SYNC_BIT));
    }
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ALLGATHER_CCU_CHK_RET(
            ccu::NotifyWait(ctx.arg->channels[index], CHANNEL_EVENT_ID, uint32_t{1} << POST_SYNC_BIT));
    }
    return CCU_SUCCESS;
}
} // namespace ops_hccl
namespace ops_hccl {
namespace large {
static_assert(ALLGATHER_SOURCE_VERSION == 0x090000, "Mixed AllGather source version");
namespace {
    constexpr uint32_t OUTPUT_RESOURCE_ID = 1;
    constexpr uint32_t TOKEN_RESOURCE_ID = 2;
    constexpr uint32_t CHANNEL_EVENT_ID = 0;
    constexpr uint32_t POST_SYNC_BIT = 3;
    constexpr uint32_t BALANCED_MODE_DIRECT = 0;
    constexpr uint32_t BALANCED_MODE_PIPELINE = 3;
    constexpr uint32_t TEST12_STAGE_MODE_BASE = 16;
    constexpr uint32_t TEST12_DATA_STAGE_COUNT = 7;
    constexpr uint32_t NETWORK_CLASS_MESH = 0;
    constexpr uint32_t NETWORK_CLASS_CLOS = 1;
    constexpr uint32_t PIPELINE_NOTIFY_BIT_BASE = 4;
    constexpr uint32_t RANK16_ACTIVE_STRIPE_COUNT = 4;

    bool UsesBalancedTaskArgs(uint32_t rankSize)
    {
        return rankSize == BALANCED_SERVER_RANKS * 2
               || rankSize == BALANCED_SERVER_RANKS + BALANCED_SMALL_SERVER_RANKS;
    }

    uint64_t GetParallelParam(uint64_t repeatCount, uint64_t repeatLoopIndex, uint64_t totalLoopCount)
    {
        constexpr uint64_t FIELD_MASK = (uint64_t{1} << 7) - 1;
        return ((repeatCount & FIELD_MASK) << 55) | ((repeatLoopIndex & FIELD_MASK) << 48)
               | ((totalLoopCount & FIELD_MASK) << 41);
    }

    uint64_t GetOffsetParam(uint64_t globalOffset, uint64_t memorySliceOffset, uint64_t eventOffset)
    {
        constexpr uint64_t GLOBAL_MASK = (uint64_t{1} << 32) - 1;
        constexpr uint64_t MEMORY_SLICE_MASK = (uint64_t{1} << 11) - 1;
        constexpr uint64_t EVENT_MASK = (uint64_t{1} << 10) - 1;
        return ((globalOffset & GLOBAL_MASK) << 21) | ((memorySliceOffset & MEMORY_SLICE_MASK) << 10)
               | (eventOffset & EVENT_MASK);
    }

    uint64_t GetLoopParam(uint64_t globalOffset, uint64_t iterations)
    {
        constexpr uint64_t GLOBAL_MASK = (uint64_t{1} << 32) - 1;
        constexpr uint64_t ITERATION_MASK = (uint64_t{1} << 13) - 1;
        return ((globalOffset & GLOBAL_MASK) << 13) | (iterations & ITERATION_MASK);
    }

    void InitializeCopyLoops(AllGatherContext &ctx, const std::string &key, ccu::LocalAddr *sources,
        ccu::LocalAddr *destinations, ccu::Variable *lengths)
    {
        if (!ctx.copyResourcesAllocated) {
            ctx.copyEvents = ccu::Array<ccu::Event>(LOCAL_COPY_LOOP_COUNT);
            ctx.copyBuffers = ccu::Array<ccu::CcuBuffer>(LOCAL_COPY_LOOP_COUNT * CCU_MEMORY_SLICE_INTERLEAVE);
            ctx.copyResourcesAllocated = true;
        }

        if (ctx.loopEntities.count(key) != 0) {
            return;
        }

        ctx.loopEntities.emplace(key, LoopEntity());
        LoopEntity &entity = ctx.loopEntities[key];
        for (uint32_t index = 0; index < 2; ++index) {
            const uint32_t bufferBase = index * CCU_MEMORY_SLICE_INTERLEAVE;
            const ccu::Event copyEvent = ctx.copyEvents[index];
            entity.bodies[index].reset(
                new ccu::Func([&ctx, index, bufferBase, copyEvent, sources, destinations, lengths]() {
                    ccu::LocalCopy(ctx.copyBuffers[bufferBase], sources[index], lengths[index], copyEvent, 1);
                    ccu::EventWait(copyEvent, 1);
                    ccu::LocalCopy(destinations[index], ctx.copyBuffers[bufferBase], lengths[index], copyEvent, 1);
                    ccu::EventWait(copyEvent, 1);
                }));
            entity.loops[index].reset(new ccu::Loop(entity.parameters[index], *entity.bodies[index]));
        }
    }

    CcuResult GroupCopy(
        AllGatherContext &ctx, const std::string &key, ccu::LocalAddr destination, ccu::LocalAddr source)
    {
        ccu::LocalAddr sources[2];
        ccu::LocalAddr destinations[2];
        ccu::Variable lengths[2];
        InitializeCopyLoops(ctx, key, sources, destinations, lengths);
        LoopEntity &entity = ctx.loopEntities[key];

        CCU_IF(ctx.groupCopySize.addressOffset != 0)
        {
            ccu::Variable loopParameter;
            loopParameter = GetLoopParam(LOCAL_COPY_MEMORY_SLICE * LOCAL_COPY_LOOP_COUNT, 0);
            loopParameter += ctx.groupCopySize.loopParam;
            ccu::Variable fixedLength;
            fixedLength = LOCAL_COPY_MEMORY_SLICE;

            sources[0].addr = source.addr;
            sources[0].token = source.token;
            destinations[0].addr = destination.addr;
            destinations[0].token = destination.token;
            lengths[0] = fixedLength;
            entity.parameters[0] = loopParameter;

            ccu::Variable parallelParameter;
            parallelParameter = GetParallelParam(LOCAL_COPY_LOOP_COUNT - 1, 0, 1);
            ccu::Variable offsetParameter;
            offsetParameter = GetOffsetParam(LOCAL_COPY_MEMORY_SLICE, CCU_MEMORY_SLICE_INTERLEAVE, 1);
            std::vector<ccu::Loop> loops{*entity.loops[0]};
            ccu::LoopGroup group(parallelParameter, offsetParameter, LOCAL_COPY_LOOP_COUNT, loops);
        }

        CCU_IF(ctx.groupCopySize.parallelParam != 0)
        {
            source.addr += ctx.groupCopySize.addressOffset;
            destination.addr += ctx.groupCopySize.addressOffset;

            sources[0].addr = source.addr;
            sources[0].token = source.token;
            destinations[0].addr = destination.addr;
            destinations[0].token = destination.token;
            lengths[0] = ctx.groupCopySize.residual;

            source.addr += ctx.groupCopySize.residual;
            destination.addr += ctx.groupCopySize.residual;
            ccu::Variable fixedLength;
            fixedLength = LOCAL_COPY_MEMORY_SLICE;
            sources[1].addr = source.addr;
            sources[1].token = source.token;
            destinations[1].addr = destination.addr;
            destinations[1].token = destination.token;
            lengths[1] = fixedLength;

            ccu::Variable singleIteration;
            singleIteration = GetLoopParam(0, 1);
            entity.parameters[0] = singleIteration;
            entity.parameters[1] = singleIteration;
            ccu::Variable offsetParameter;
            offsetParameter = GetOffsetParam(LOCAL_COPY_MEMORY_SLICE, CCU_MEMORY_SLICE_INTERLEAVE, 1);
            std::vector<ccu::Loop> loops{*entity.loops[0], *entity.loops[1]};
            ccu::LoopGroup group(ctx.groupCopySize.parallelParam, offsetParameter, LOCAL_COPY_LOOP_COUNT, loops);
        }
        return CCU_SUCCESS;
    }

    int32_t FindChannelIndex(const AllGatherKernelArg &arg, uint32_t peerRank)
    {
        for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
            if (arg.peerRanks[channelIndex] == peerRank) {
                return static_cast<int32_t>(channelIndex);
            }
        }
        return -1;
    }

    CcuResult TransferDirect(AllGatherContext &ctx, const std::string &copyKey)
    {
        ccu::Event &transferEvent =
            UsesBalancedTaskArgs(ctx.arg->rankSize) ? ctx.balancedEvents[0] : ctx.transferEvents[0];
        ccu::LocalAddr source;
        source.addr = ctx.input;
        source.addr += ctx.inputOffset;
        source.token = ctx.tokens[ctx.arg->rankId];

        ccu::LocalAddr localDestination;
        std::vector<ccu::RemoteAddr> remoteDestinations;
        remoteDestinations.resize(ctx.arg->rankSize);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const uint32_t rank = ctx.arg->peerRanks[channelIndex];
            remoteDestinations[rank].addr = ctx.outputs[rank];
            remoteDestinations[rank].addr += ctx.outputOffset;
            remoteDestinations[rank].token = ctx.tokens[rank];
        }
        if (ctx.arg->copyLocal) {
            localDestination.addr = ctx.outputs[ctx.arg->rankId];
            localDestination.addr += ctx.outputOffset;
            localDestination.token = ctx.tokens[ctx.arg->rankId];
        }

        uint16_t waitMask = 0;
        CCU_IF(ctx.sliceSize != 0)
        {
            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
                const uint32_t rank = ctx.arg->peerRanks[channelIndex];
                const uint16_t rankMask = static_cast<uint16_t>(uint32_t{1} << rank);
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], remoteDestinations[rank], source,
                    ctx.sliceSize, transferEvent, rankMask));
                waitMask |= rankMask;
            }
        }

        if (ctx.arg->copyLocal) {
            ALLGATHER_CCU_CHK_RET(GroupCopy(ctx, copyKey, localDestination, source));
            const uint16_t localMask = static_cast<uint16_t>(uint32_t{1} << ctx.arg->rankId);
            ALLGATHER_CCU_CHK_RET(ccu::EventRecord(transferEvent, localMask));
            waitMask |= localMask;
        }
        ALLGATHER_CCU_CHK_RET(ccu::EventWait(transferEvent, waitMask));
        return CCU_SUCCESS;
    }

#if 0
    CcuResult TransferBalancedPhaseA(AllGatherContext &ctx)
    {
        const uint32_t localRank = ctx.arg->rankId % BALANCED_SERVER_RANKS;
        const uint32_t localServerBase = (ctx.arg->rankId / BALANCED_SERVER_RANKS) * BALANCED_SERVER_RANKS;
        const uint32_t remoteServerBase =
            (ctx.arg->rankId < BALANCED_SERVER_RANKS) ? BALANCED_SERVER_RANKS : 0;
        uint16_t waitMasks[BALANCED_STRIPE_COUNT] = {};

        for (uint32_t stripe = 0; stripe < RANK16_ACTIVE_STRIPE_COUNT; ++stripe) {
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.addr += ctx.stripeOffsets[stripe];
            source.token = ctx.tokens[ctx.arg->rankId];

            for (uint32_t peerLocalRank = 0; peerLocalRank < BALANCED_SERVER_RANKS; ++peerLocalRank) {
                if (peerLocalRank == localRank) {
                    continue;
                }
                const uint32_t peerRank = localServerBase + peerLocalRank;
                const int32_t channelIndex = FindChannelIndex(*ctx.arg, peerRank);
                if (channelIndex < 0) {
                    continue;
                }
                ccu::RemoteAddr destination;
                destination.addr = ctx.outputs[peerRank];
                destination.addr += ctx.outputOffset;
                destination.addr += ctx.stripeOffsets[stripe];
                destination.token = ctx.tokens[peerRank];
                const uint16_t rankMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                    ctx.stripeSizes[stripe], ctx.balancedEvents[stripe], rankMask));
                waitMasks[stripe] |= rankMask;
            }

            const uint32_t directRemoteCount = RANK16_DIRECT_REMOTE_COUNTS[stripe];
            for (uint32_t offset = 0; offset < directRemoteCount; ++offset) {
                const uint32_t peerLocalRank = (localRank + offset) % BALANCED_SERVER_RANKS;
                const uint32_t peerRank = remoteServerBase + peerLocalRank;
                const int32_t channelIndex = FindChannelIndex(*ctx.arg, peerRank);
                if (channelIndex < 0) {
                    continue;
                }
                ccu::RemoteAddr destination;
                destination.addr = ctx.outputs[peerRank];
                destination.addr += ctx.outputOffset;
                destination.addr += ctx.stripeOffsets[stripe];
                destination.token = ctx.tokens[peerRank];
                const uint16_t rankMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                    ctx.stripeSizes[stripe], ctx.balancedEvents[stripe], rankMask));
                waitMasks[stripe] |= rankMask;
            }
        }

        if (ctx.arg->copyLocal) {
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.token = ctx.tokens[ctx.arg->rankId];
            ccu::LocalAddr destination;
            destination.addr = ctx.outputs[ctx.arg->rankId];
            destination.addr += ctx.outputOffset;
            destination.token = ctx.tokens[ctx.arg->rankId];
            ALLGATHER_CCU_CHK_RET(GroupCopy(ctx, "allgather_rank16_phase_a_copy", destination, source));
        }

        for (uint32_t stripe = 0; stripe < RANK16_ACTIVE_STRIPE_COUNT; ++stripe) {
            if (waitMasks[stripe] != 0) {
                ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.balancedEvents[stripe], waitMasks[stripe]));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult TransferBalancedPhaseB(AllGatherContext &ctx)
    {
        const uint32_t localRank = ctx.arg->rankId % BALANCED_SERVER_RANKS;
        const uint32_t localServerBase = (ctx.arg->rankId / BALANCED_SERVER_RANKS) * BALANCED_SERVER_RANKS;
        const uint32_t remoteServerBase =
            (ctx.arg->rankId < BALANCED_SERVER_RANKS) ? BALANCED_SERVER_RANKS : 0;
        uint16_t waitMasks[BALANCED_STRIPE_COUNT] = {};

        for (uint32_t stripe = 0; stripe < RANK16_ACTIVE_STRIPE_COUNT; ++stripe) {
            const uint32_t sourceLocalRank =
                (localRank + BALANCED_SERVER_RANKS - RANK16_RELAY_OFFSETS[stripe]) % BALANCED_SERVER_RANKS;
            const uint32_t sourceRank = remoteServerBase + sourceLocalRank;
            const uint32_t directRemoteCount = RANK16_DIRECT_REMOTE_COUNTS[stripe];
            const uint32_t relayRemoteCount = BALANCED_SERVER_RANKS - directRemoteCount;

            ccu::Variable sourceSlotOffset;
            sourceSlotOffset = 0;
            for (uint32_t rank = 0; rank < sourceRank; ++rank) {
                sourceSlotOffset += ctx.rankDataSize;
            }
            ccu::LocalAddr source;
            source.addr = ctx.outputs[ctx.arg->rankId];
            source.addr += sourceSlotOffset;
            source.addr += ctx.stripeOffsets[stripe];
            source.token = ctx.tokens[ctx.arg->rankId];

            for (uint32_t offset = 0; offset < relayRemoteCount; ++offset) {
                const uint32_t peerLocalRank =
                    (sourceLocalRank + directRemoteCount + offset) % BALANCED_SERVER_RANKS;
                if (peerLocalRank == localRank) {
                    continue;
                }
                const uint32_t peerRank = localServerBase + peerLocalRank;
                const int32_t channelIndex = FindChannelIndex(*ctx.arg, peerRank);
                if (channelIndex < 0) {
                    continue;
                }
                ccu::RemoteAddr destination;
                destination.addr = ctx.outputs[peerRank];
                destination.addr += sourceSlotOffset;
                destination.addr += ctx.stripeOffsets[stripe];
                destination.token = ctx.tokens[peerRank];
                const uint16_t rankMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                    ctx.stripeSizes[stripe], ctx.balancedEvents[stripe], rankMask));
                waitMasks[stripe] |= rankMask;
            }
        }

        for (uint32_t stripe = 0; stripe < RANK16_ACTIVE_STRIPE_COUNT; ++stripe) {
            if (waitMasks[stripe] != 0) {
                ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.balancedEvents[stripe], waitMasks[stripe]));
            }
        }
        return CCU_SUCCESS;
    }

#endif
    CcuResult LaunchPipelineDirect(
        AllGatherContext &ctx, uint32_t stripe, uint16_t &directWaitMask)
    {
        ccu::LocalAddr source;
        source.addr = ctx.input;
        source.addr += ctx.stripeOffsets[stripe];
        source.token = ctx.tokens[ctx.arg->rankId];

        directWaitMask = 0;
        for (uint32_t peerRank : ctx.arg->pipelineDirectRanks) {
            const int32_t channelIndex = FindChannelIndex(*ctx.arg, peerRank);
            if (channelIndex < 0) {
                return CCU_E_PARA;
            }
            ccu::RemoteAddr destination;
            destination.addr = ctx.outputs[peerRank];
            destination.addr += ctx.outputOffset;
            destination.addr += ctx.stripeOffsets[stripe];
            destination.token = ctx.tokens[peerRank];
            const uint16_t rankMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
            ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                ctx.stripeSizes[stripe], ctx.balancedEvents[stripe], rankMask));
            directWaitMask |= rankMask;
        }
        return CCU_SUCCESS;
    }

    CcuResult CompletePipelineDirect(
        AllGatherContext &ctx, uint32_t stripe, uint16_t directWaitMask)
    {
        if (directWaitMask != 0) {
            ALLGATHER_CCU_CHK_RET(
                ccu::EventWait(ctx.balancedEvents[stripe], directWaitMask));
        }

        // The relay may read this source slot only after all source-side
        // writes in this die group have completed.
        const uint32_t stripeMask =
            uint32_t{1} << (PIPELINE_NOTIFY_BIT_BASE + stripe);
        for (uint32_t relayRank : ctx.arg->pipelineRemoteRelayRanks) {
            const int32_t channelIndex = FindChannelIndex(*ctx.arg, relayRank);
            if (channelIndex < 0) {
                return CCU_E_PARA;
            }
            ALLGATHER_CCU_CHK_RET(
                ccu::NotifyRecord(ctx.arg->channels[channelIndex], CHANNEL_EVENT_ID, stripeMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunPipelineRelays(AllGatherContext &ctx, uint32_t stripe)
    {
        const uint32_t stripeMask =
            uint32_t{1} << (PIPELINE_NOTIFY_BIT_BASE + stripe);
        for (const PipelineRelayTask &relayTask : ctx.arg->pipelineRelayTasks) {
            const int32_t sourceChannelIndex =
                FindChannelIndex(*ctx.arg, relayTask.sourceRank);
            if (sourceChannelIndex < 0) {
                return CCU_E_PARA;
            }
            ALLGATHER_CCU_CHK_RET(
                ccu::NotifyWait(ctx.arg->channels[sourceChannelIndex], CHANNEL_EVENT_ID, stripeMask));

            ccu::Variable sourceSlotOffset;
            sourceSlotOffset = 0;
            for (uint32_t rank = 0; rank < relayTask.sourceRank; ++rank) {
                sourceSlotOffset += ctx.rankDataSize;
            }
            ccu::LocalAddr source;
            source.addr = ctx.outputs[ctx.arg->rankId];
            source.addr += sourceSlotOffset;
            source.addr += ctx.stripeOffsets[stripe];
            source.token = ctx.tokens[ctx.arg->rankId];
            uint16_t relayWaitMask = 0;
            for (uint32_t destinationRank : relayTask.destinationRanks) {
                const int32_t destinationChannelIndex =
                    FindChannelIndex(*ctx.arg, destinationRank);
                if (destinationChannelIndex < 0) {
                    return CCU_E_PARA;
                }
                ccu::RemoteAddr destination;
                destination.addr = ctx.outputs[destinationRank];
                destination.addr += sourceSlotOffset;
                destination.addr += ctx.stripeOffsets[stripe];
                destination.token = ctx.tokens[destinationRank];
                const uint16_t rankMask =
                    static_cast<uint16_t>(uint32_t{1} << destinationRank);
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[destinationChannelIndex],
                    destination, source, ctx.stripeSizes[stripe],
                    ctx.balancedEvents[stripe], rankMask));
                relayWaitMask |= rankMask;
            }
            if (relayWaitMask != 0) {
                ALLGATHER_CCU_CHK_RET(
                    ccu::EventWait(ctx.balancedEvents[stripe], relayWaitMask));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult TransferPipeline(AllGatherContext &ctx)
    {
        uint16_t directWaitMasks[RANK16_ACTIVE_STRIPE_COUNT] = {};
        ALLGATHER_CCU_CHK_RET(
            LaunchPipelineDirect(ctx, 0, directWaitMasks[0]));

        // Hide the mandatory full-rank local copy behind the first batch of
        // in-flight Direct/Clos writes.  V9.1 performed this copy after every
        // relay stripe had completed, adding an extra data-size-proportional
        // tail to tests 11, 12 and 17.
        if (ctx.arg->copyLocal) {
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.token = ctx.tokens[ctx.arg->rankId];
            ccu::LocalAddr destination;
            destination.addr = ctx.outputs[ctx.arg->rankId];
            destination.addr += ctx.outputOffset;
            destination.token = ctx.tokens[ctx.arg->rankId];
            ALLGATHER_CCU_CHK_RET(
                GroupCopy(ctx, "allgather_rank16_pipeline_copy", destination, source));
        }

        for (uint32_t stripe = 0; stripe < RANK16_ACTIVE_STRIPE_COUNT; ++stripe) {
            ALLGATHER_CCU_CHK_RET(
                CompletePipelineDirect(ctx, stripe, directWaitMasks[stripe]));

            // Start Clos transfer of stripe N+1 before doing the same-die Mesh
            // fanout of stripe N. Distinct events make the overlap explicit
            // and keep Checker-visible write/read dependencies intact.
            if (stripe + 1 < RANK16_ACTIVE_STRIPE_COUNT) {
                ALLGATHER_CCU_CHK_RET(
                    LaunchPipelineDirect(ctx, stripe + 1, directWaitMasks[stripe + 1]));
            }
            ALLGATHER_CCU_CHK_RET(RunPipelineRelays(ctx, stripe));
        }
        return CCU_SUCCESS;
    }

#if 0
    CcuResult TransferBalancedPhaseA12(AllGatherContext &ctx)
    {
        if (ctx.arg->rankId < BALANCED_SERVER_RANKS) {
            ctx.inputOffset = 0;
            ctx.sliceSize = ctx.rankDataSize;
            return TransferDirect(ctx, "allgather_rank12_phase_a_copy");
        }

        const uint32_t sourceLocalRank = ctx.arg->rankId - BALANCED_SERVER_RANKS;
        const uint32_t directBase = sourceLocalRank * 2;
        uint16_t waitMasks[BALANCED_STRIPE_COUNT] = {};
        for (uint32_t stripe = 0; stripe < BALANCED_STRIPE_COUNT; ++stripe) {
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.addr += ctx.stripeOffsets[stripe];
            source.token = ctx.tokens[ctx.arg->rankId];

            for (uint32_t peerLocalRank = 0; peerLocalRank < BALANCED_SMALL_SERVER_RANKS; ++peerLocalRank) {
                const uint32_t peerRank = BALANCED_SERVER_RANKS + peerLocalRank;
                if (peerRank == ctx.arg->rankId) {
                    continue;
                }
                const int32_t channelIndex = FindChannelIndex(*ctx.arg, peerRank);
                if (channelIndex < 0) {
                    continue;
                }
                ccu::RemoteAddr destination;
                destination.addr = ctx.outputs[peerRank];
                destination.addr += ctx.outputOffset;
                destination.addr += ctx.stripeOffsets[stripe];
                destination.token = ctx.tokens[peerRank];
                const uint16_t rankMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                    ctx.stripeSizes[stripe], ctx.balancedEvents[stripe], rankMask));
                waitMasks[stripe] |= rankMask;
            }

            const uint32_t directRemoteCount = BALANCED_DIRECT_REMOTE_COUNTS[stripe];
            for (uint32_t offset = 0; offset < directRemoteCount; ++offset) {
                const uint32_t peerRank = (directBase + offset) % BALANCED_SERVER_RANKS;
                const int32_t channelIndex = FindChannelIndex(*ctx.arg, peerRank);
                if (channelIndex < 0) {
                    continue;
                }
                ccu::RemoteAddr destination;
                destination.addr = ctx.outputs[peerRank];
                destination.addr += ctx.outputOffset;
                destination.addr += ctx.stripeOffsets[stripe];
                destination.token = ctx.tokens[peerRank];
                const uint16_t rankMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                    ctx.stripeSizes[stripe], ctx.balancedEvents[stripe], rankMask));
                waitMasks[stripe] |= rankMask;
            }
        }

        if (ctx.arg->copyLocal) {
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.token = ctx.tokens[ctx.arg->rankId];
            ccu::LocalAddr destination;
            destination.addr = ctx.outputs[ctx.arg->rankId];
            destination.addr += ctx.outputOffset;
            destination.token = ctx.tokens[ctx.arg->rankId];
            ALLGATHER_CCU_CHK_RET(GroupCopy(ctx, "allgather_rank12_phase_a_copy", destination, source));
        }
        for (uint32_t stripe = 0; stripe < BALANCED_STRIPE_COUNT; ++stripe) {
            if (waitMasks[stripe] != 0) {
                ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.balancedEvents[stripe], waitMasks[stripe]));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult TransferBalancedPhaseB12(AllGatherContext &ctx)
    {
        if (ctx.arg->rankId >= BALANCED_SERVER_RANKS) {
            return CCU_SUCCESS;
        }

        uint16_t waitMasks[BALANCED_STRIPE_COUNT] = {};
        for (uint32_t sourceLocalRank = 0; sourceLocalRank < BALANCED_SMALL_SERVER_RANKS; ++sourceLocalRank) {
            const uint32_t sourceRank = BALANCED_SERVER_RANKS + sourceLocalRank;
            const uint32_t directBase = sourceLocalRank * 2;
            ccu::Variable sourceSlotOffset;
            sourceSlotOffset = 0;
            for (uint32_t rank = 0; rank < sourceRank; ++rank) {
                sourceSlotOffset += ctx.rankDataSize;
            }

            for (uint32_t stripe = 0; stripe < BALANCED_STRIPE_COUNT; ++stripe) {
                const uint32_t relayRank =
                    (directBase + BALANCED_RELAY_OFFSETS[stripe]) % BALANCED_SERVER_RANKS;
                if (ctx.arg->rankId != relayRank) {
                    continue;
                }
                ccu::LocalAddr source;
                source.addr = ctx.outputs[ctx.arg->rankId];
                source.addr += sourceSlotOffset;
                source.addr += ctx.stripeOffsets[stripe];
                source.token = ctx.tokens[ctx.arg->rankId];

                const uint32_t directRemoteCount = BALANCED_DIRECT_REMOTE_COUNTS[stripe];
                const uint32_t relayRemoteCount = BALANCED_SERVER_RANKS - directRemoteCount;
                for (uint32_t offset = 0; offset < relayRemoteCount; ++offset) {
                    const uint32_t peerRank =
                        (directBase + directRemoteCount + offset) % BALANCED_SERVER_RANKS;
                    const int32_t channelIndex = FindChannelIndex(*ctx.arg, peerRank);
                    if (channelIndex < 0) {
                        continue;
                    }
                    ccu::RemoteAddr destination;
                    destination.addr = ctx.outputs[peerRank];
                    destination.addr += sourceSlotOffset;
                    destination.addr += ctx.stripeOffsets[stripe];
                    destination.token = ctx.tokens[peerRank];
                    const uint16_t rankMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
                    ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                        ctx.stripeSizes[stripe], ctx.balancedEvents[stripe], rankMask));
                    waitMasks[stripe] |= rankMask;
                }
            }
        }
        for (uint32_t stripe = 0; stripe < BALANCED_STRIPE_COUNT; ++stripe) {
            if (waitMasks[stripe] != 0) {
                ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.balancedEvents[stripe], waitMasks[stripe]));
            }
        }
        return CCU_SUCCESS;
    }

#endif
    template <uint32_t Stripe>
    bool IsTest12FanoutOffset(uint32_t offset)
    {
        static_assert(Stripe < TEST12_DATA_STAGE_COUNT,
            "Invalid test12 stripe");
        constexpr uint32_t first = Stripe % 7U + 1U;
        constexpr uint32_t second = (Stripe + 1U) % 7U + 1U;
        constexpr uint32_t third = (Stripe + 2U) % 7U + 1U;
        return offset == first || offset == second || offset == third;
    }

    template <uint32_t Stage>
    CcuResult TransferTest12Stage(AllGatherContext &ctx)
    {
        static_assert(Stage <= TEST12_DATA_STAGE_COUNT,
            "Invalid test12 stage");
        if (ctx.arg->rankSize != BALANCED_SERVER_RANKS * 2
            || ctx.arg->networkClass > NETWORK_CLASS_CLOS) {
            return CCU_E_PARA;
        }

        const uint32_t localRank =
            ctx.arg->rankId % BALANCED_SERVER_RANKS;
        const uint32_t localServerBase =
            (ctx.arg->rankId / BALANCED_SERVER_RANKS)
            * BALANCED_SERVER_RANKS;
        const uint32_t remoteServerBase = localServerBase == 0
            ? BALANCED_SERVER_RANKS
            : 0;
        uint16_t currentWaitMask = 0;
        uint16_t fanoutWaitMask = 0;

        if constexpr (Stage < TEST12_DATA_STAGE_COUNT) {
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.addr += ctx.stripeOffsets[0];
            source.token = ctx.tokens[ctx.arg->rankId];

            if (ctx.arg->networkClass == NETWORK_CLASS_MESH) {
                for (uint32_t channelIndex = 0;
                     channelIndex < ctx.arg->channelCount;
                     ++channelIndex) {
                    const uint32_t peerRank =
                        ctx.arg->peerRanks[channelIndex];
                    ccu::RemoteAddr destination;
                    destination.addr = ctx.outputs[peerRank];
                    destination.addr += ctx.outputOffset;
                    destination.addr += ctx.stripeOffsets[0];
                    destination.token = ctx.tokens[peerRank];
                    const uint16_t rankMask = static_cast<uint16_t>(
                        uint32_t{1} << peerRank);
                    ALLGATHER_CCU_CHK_RET(ccu::Write(
                        ctx.arg->channels[channelIndex], destination,
                        source, ctx.stripeSizes[0],
                        ctx.balancedEvents[0], rankMask));
                    currentWaitMask |= rankMask;
                }
            } else {
                const uint32_t relayLocalRank =
                    (localRank + Stage) % BALANCED_SERVER_RANKS;
                for (uint32_t channelIndex = 0;
                     channelIndex < ctx.arg->channelCount;
                     ++channelIndex) {
                    const uint32_t peerRank =
                        ctx.arg->peerRanks[channelIndex];
                    const uint32_t peerLocalRank =
                        peerRank - remoteServerBase;
                    const uint32_t relayOffset =
                        (peerLocalRank + BALANCED_SERVER_RANKS
                            - relayLocalRank)
                        % BALANCED_SERVER_RANKS;
                    if (IsTest12FanoutOffset<Stage>(relayOffset)) {
                        continue;
                    }
                    ccu::RemoteAddr destination;
                    destination.addr = ctx.outputs[peerRank];
                    destination.addr += ctx.outputOffset;
                    destination.addr += ctx.stripeOffsets[0];
                    destination.token = ctx.tokens[peerRank];
                    const uint16_t rankMask = static_cast<uint16_t>(
                        uint32_t{1} << peerRank);
                    ALLGATHER_CCU_CHK_RET(ccu::Write(
                        ctx.arg->channels[channelIndex], destination,
                        source, ctx.stripeSizes[0],
                        ctx.balancedEvents[0], rankMask));
                    currentWaitMask |= rankMask;
                }
            }
        }

        if constexpr (Stage > 0) {
            if (ctx.arg->networkClass == NETWORK_CLASS_MESH) {
                constexpr uint32_t previous = Stage - 1;
                const uint32_t sourceLocalRank =
                    (localRank + BALANCED_SERVER_RANKS
                        - previous % BALANCED_SERVER_RANKS)
                    % BALANCED_SERVER_RANKS;
                const uint32_t sourceRank =
                    remoteServerBase + sourceLocalRank;
                ccu::Variable sourceSlotOffset;
                sourceSlotOffset = 0;
                for (uint32_t rank = 0; rank < sourceRank; ++rank) {
                    sourceSlotOffset += ctx.rankDataSize;
                }
                ccu::LocalAddr relaySource;
                relaySource.addr = ctx.outputs[ctx.arg->rankId];
                relaySource.addr += sourceSlotOffset;
                relaySource.addr += ctx.stripeOffsets[1];
                relaySource.token = ctx.tokens[ctx.arg->rankId];

                for (uint32_t channelIndex = 0;
                     channelIndex < ctx.arg->channelCount;
                     ++channelIndex) {
                    const uint32_t peerRank =
                        ctx.arg->peerRanks[channelIndex];
                    const uint32_t peerLocalRank =
                        peerRank - localServerBase;
                    const uint32_t relayOffset =
                        (peerLocalRank + BALANCED_SERVER_RANKS
                            - localRank)
                        % BALANCED_SERVER_RANKS;
                    if (!IsTest12FanoutOffset<previous>(relayOffset)) {
                        continue;
                    }
                    ccu::RemoteAddr destination;
                    destination.addr = ctx.outputs[peerRank];
                    destination.addr += sourceSlotOffset;
                    destination.addr += ctx.stripeOffsets[1];
                    destination.token = ctx.tokens[peerRank];
                    const uint16_t rankMask = static_cast<uint16_t>(
                        uint32_t{1} << peerRank);
                    ALLGATHER_CCU_CHK_RET(ccu::Write(
                        ctx.arg->channels[channelIndex], destination,
                        relaySource, ctx.stripeSizes[1],
                        ctx.balancedEvents[1], rankMask));
                    fanoutWaitMask |= rankMask;
                }
            }
        }

        if constexpr (Stage == 0) {
            if (ctx.arg->copyLocal) {
                ccu::LocalAddr source;
                source.addr = ctx.input;
                source.token = ctx.tokens[ctx.arg->rankId];
                ccu::LocalAddr destination;
                destination.addr = ctx.outputs[ctx.arg->rankId];
                destination.addr += ctx.outputOffset;
                destination.token = ctx.tokens[ctx.arg->rankId];
                ALLGATHER_CCU_CHK_RET(GroupCopy(ctx,
                    "allgather_test12_pipeline_copy", destination,
                    source));
            }
        }

        if (currentWaitMask != 0) {
            ALLGATHER_CCU_CHK_RET(ccu::EventWait(
                ctx.balancedEvents[0], currentWaitMask));
        }
        if (fanoutWaitMask != 0) {
            ALLGATHER_CCU_CHK_RET(ccu::EventWait(
                ctx.balancedEvents[1], fanoutWaitMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult PreSync(AllGatherContext &ctx)
    {
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[index],
                ctx.outputs[ctx.arg->rankId], OUTPUT_RESOURCE_ID, CHANNEL_EVENT_ID,
                uint32_t{1} << OUTPUT_RESOURCE_ID));
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[index],
                ctx.tokens[ctx.arg->rankId], TOKEN_RESOURCE_ID, CHANNEL_EVENT_ID,
                uint32_t{1} << TOKEN_RESOURCE_ID));
        }
        const uint32_t addressReadyMask =
            (uint32_t{1} << OUTPUT_RESOURCE_ID) | (uint32_t{1} << TOKEN_RESOURCE_ID);
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            ALLGATHER_CCU_CHK_RET(
                ccu::NotifyWait(ctx.arg->channels[index], CHANNEL_EVENT_ID, addressReadyMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult PostSync(AllGatherContext &ctx)
    {
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            ALLGATHER_CCU_CHK_RET(
                ccu::NotifyRecord(ctx.arg->channels[index], CHANNEL_EVENT_ID, uint32_t{1} << POST_SYNC_BIT));
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            ALLGATHER_CCU_CHK_RET(
                ccu::NotifyWait(ctx.arg->channels[index], CHANNEL_EVENT_ID, uint32_t{1} << POST_SYNC_BIT));
        }
        return CCU_SUCCESS;
    }
} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<AllGatherKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0
        || kernelArg->channelCount != kernelArg->peerRanks.size()) {
        return CCU_E_PARA;
    }

    AllGatherContext ctx;
    ctx.arg = kernelArg;
    ctx.outputs.resize(ctx.arg->rankSize);
    ctx.tokens.resize(ctx.arg->rankSize);

    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        const uint32_t rank = ctx.arg->peerRanks[channelIndex];
        if (rank >= ctx.arg->rankSize || rank == ctx.arg->rankId) {
            return CCU_E_PARA;
        }
        ctx.outputs[rank] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], OUTPUT_RESOURCE_ID);
        ctx.tokens[rank] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], TOKEN_RESOURCE_ID);
    }

    uint32_t argIndex = 0;
    if (UsesBalancedTaskArgs(ctx.arg->rankSize)) {
        ctx.balancedEvents = ccu::Array<ccu::Event>(BALANCED_STRIPE_COUNT);
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.balancedMode, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.input, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.outputs[ctx.arg->rankId], argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.tokens[ctx.arg->rankId], argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.rankDataSize, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argIndex++));
        for (uint32_t stripe = 0; stripe < BALANCED_STRIPE_COUNT; ++stripe) {
            ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.stripeOffsets[stripe], argIndex++));
        }
        for (uint32_t stripe = 0; stripe < BALANCED_STRIPE_COUNT; ++stripe) {
            ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.stripeSizes[stripe], argIndex++));
        }
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.addressOffset, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.loopParam, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.parallelParam, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.residual, argIndex++));
        ALLGATHER_CCU_CHK_RET(PreSync(ctx));
        if (ctx.arg->test12Special) {
            CCU_IF(ctx.balancedMode == TEST12_STAGE_MODE_BASE + 0)
            {
                ALLGATHER_CCU_CHK_RET(TransferTest12Stage<0>(ctx));
            }
            CCU_IF(ctx.balancedMode == TEST12_STAGE_MODE_BASE + 1)
            {
                ALLGATHER_CCU_CHK_RET(TransferTest12Stage<1>(ctx));
            }
            CCU_IF(ctx.balancedMode == TEST12_STAGE_MODE_BASE + 2)
            {
                ALLGATHER_CCU_CHK_RET(TransferTest12Stage<2>(ctx));
            }
            CCU_IF(ctx.balancedMode == TEST12_STAGE_MODE_BASE + 3)
            {
                ALLGATHER_CCU_CHK_RET(TransferTest12Stage<3>(ctx));
            }
            CCU_IF(ctx.balancedMode == TEST12_STAGE_MODE_BASE + 4)
            {
                ALLGATHER_CCU_CHK_RET(TransferTest12Stage<4>(ctx));
            }
            CCU_IF(ctx.balancedMode == TEST12_STAGE_MODE_BASE + 5)
            {
                ALLGATHER_CCU_CHK_RET(TransferTest12Stage<5>(ctx));
            }
            CCU_IF(ctx.balancedMode == TEST12_STAGE_MODE_BASE + 6)
            {
                ALLGATHER_CCU_CHK_RET(TransferTest12Stage<6>(ctx));
            }
            CCU_IF(ctx.balancedMode == TEST12_STAGE_MODE_BASE + 7)
            {
                ALLGATHER_CCU_CHK_RET(TransferTest12Stage<7>(ctx));
            }
        } else {
            CCU_IF(ctx.balancedMode == BALANCED_MODE_DIRECT)
            {
                ctx.inputOffset = 0;
                ctx.sliceSize = ctx.rankDataSize;
                ALLGATHER_CCU_CHK_RET(
                    TransferDirect(ctx, "allgather_direct_copy"));
            }
            CCU_IF(ctx.balancedMode == BALANCED_MODE_PIPELINE)
            {
                ALLGATHER_CCU_CHK_RET(TransferPipeline(ctx));
            }
        }
        ALLGATHER_CCU_CHK_RET(PostSync(ctx));
    } else {
        ctx.transferEvents = ccu::Array<ccu::Event>(1);
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.input, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.outputs[ctx.arg->rankId], argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.tokens[ctx.arg->rankId], argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.inputOffset, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.addressOffset, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.loopParam, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.parallelParam, argIndex++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.groupCopySize.residual, argIndex++));
        ALLGATHER_CCU_CHK_RET(PreSync(ctx));
        ALLGATHER_CCU_CHK_RET(TransferDirect(ctx, "allgather_direct_copy"));
        ALLGATHER_CCU_CHK_RET(PostSync(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuKernelRank16(CcuKernelArg arg)
{
    return CcuKernel(arg);
}

CcuResult CcuKernelRank12(CcuKernelArg arg)
{
    return CcuKernel(arg);
}
} // namespace large
} // namespace ops_hccl
