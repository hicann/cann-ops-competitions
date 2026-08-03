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

#include "ccu_kernel.h"
#include "log.h"

namespace ops_hccl {

#define CCU_CHK_RET(call) \
    do { \
        const CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

namespace {

constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t PRE_SYNC_CKE = 0;
constexpr uint32_t RELAY_READY_CKE = 1;
constexpr uint32_t FINAL_SYNC_CKE = 2;
constexpr uint16_t FINAL_SYNC_BIT = 1;

struct DirectContext {
    const CcuKernelArgDirect *arg = nullptr;
    ccu::Variable input;
    std::vector<ccu::Variable> output;
    ccu::Variable inputToken;
    std::vector<ccu::Variable> outputToken;
    ccu::Variable inputOffset;
    ccu::Variable outputOffset;
    ccu::Variable sliceSize;
    GroupOpSizeVars copySize;
    ccu::Event networkEvent;
    CopyContext copy;
};

struct RelayContext {
    const CcuKernelArgRelay2x8 *arg = nullptr;
    ccu::Variable input;
    std::vector<ccu::Variable> output;
    ccu::Variable inputToken;
    std::vector<ccu::Variable> outputToken;
    std::array<ccu::Variable, MAX_RANK_SIZE> rankOffset;
    std::array<ccu::Variable, RELAY_CHUNK_COUNT> chunkOffset;
    std::array<ccu::Variable, RELAY_CHUNK_COUNT> chunkSize;
    std::array<GroupOpSizeVars, RELAY_CHUNK_COUNT> copySize;
    std::array<ccu::LocalAddr, MAX_MESH_OPS_PER_SLOT + MAX_CLOS_OPS_PER_SLOT> transferSource;
    std::array<ccu::RemoteAddr, MAX_MESH_OPS_PER_SLOT + MAX_CLOS_OPS_PER_SLOT> transferDestination;
    ccu::LocalAddr localCopySource;
    ccu::LocalAddr localCopyDestination;
    ccu::Event networkEvent;
    CopyContext copy;
};

constexpr uint64_t SetBits(uint16_t end)
{
    return (uint64_t(1) << (end + 1)) - uint64_t(1);
}

uint64_t GetLoopParam(uint64_t loopContextId, uint64_t addressOffset, uint64_t loopIterations)
{
    constexpr uint16_t CONTEXT_END_BIT = 7;
    constexpr uint16_t CONTEXT_SHIFT = 45;
    constexpr uint16_t ADDRESS_END_BIT = 31;
    constexpr uint16_t ADDRESS_SHIFT = 13;
    constexpr uint16_t LOOP_END_BIT = 12;
    return ((loopContextId & SetBits(CONTEXT_END_BIT)) << CONTEXT_SHIFT) |
           ((addressOffset & SetBits(ADDRESS_END_BIT)) << ADDRESS_SHIFT) |
           (loopIterations & SetBits(LOOP_END_BIT));
}

uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t REPEAT_END_BIT = 6;
    constexpr uint16_t REPEAT_SHIFT = 55;
    constexpr uint16_t REPEAT_LOOP_END_BIT = 6;
    constexpr uint16_t REPEAT_LOOP_SHIFT = 48;
    constexpr uint16_t TOTAL_LOOP_END_BIT = 6;
    constexpr uint16_t TOTAL_LOOP_SHIFT = 41;
    return ((repeatNum & SetBits(REPEAT_END_BIT)) << REPEAT_SHIFT) |
           ((repeatLoopIndex & SetBits(REPEAT_LOOP_END_BIT)) << REPEAT_LOOP_SHIFT) |
           ((totalLoopNum & SetBits(TOTAL_LOOP_END_BIT)) << TOTAL_LOOP_SHIFT);
}

uint64_t GetOffsetParam(uint64_t addressOffset, uint64_t msOffset, uint64_t ckeOffset)
{
    constexpr uint16_t ADDRESS_END_BIT = 31;
    constexpr uint16_t ADDRESS_SHIFT = 21;
    constexpr uint16_t MS_END_BIT = 10;
    constexpr uint16_t MS_SHIFT = 10;
    constexpr uint16_t CKE_END_BIT = 9;
    return ((addressOffset & SetBits(ADDRESS_END_BIT)) << ADDRESS_SHIFT) |
           ((msOffset & SetBits(MS_END_BIT)) << MS_SHIFT) | (ckeOffset & SetBits(CKE_END_BIT));
}

void InitializeCopyResources(CopyContext &ctx)
{
    if (!ctx.resourceAllocated) {
        ctx.config.msInterleave = CCU_MS_INTERLEAVE;
        ctx.config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
        ctx.config.memSlice = CCU_LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
        ctx.resources.eventCount = ctx.config.loopCount;
        ctx.resources.completedEvent = ccu::Array<ccu::Event>(ctx.resources.eventCount);
        ctx.resources.bufCount = ctx.config.loopCount * ctx.config.msInterleave;
        ctx.resources.ccuBuf = ccu::Array<ccu::CcuBuffer>(ctx.resources.bufCount);
        ctx.resourceAllocated = true;
    }

    const std::string loopType = "localcopy";
    if (ctx.IsLoopEntityRegistered(loopType)) {
        return;
    }
    ctx.CreateLoopEntity(loopType);
    CcuLoopEntity &entity = ctx.loopMap[loopType];
    for (uint32_t index = 0; index < 2; ++index) {
        const uint32_t bufferBase = index * ctx.config.msInterleave;
        const ccu::Event loopEvent = ctx.resources.completedEvent[index];
        entity.body[index].reset(new ccu::Func([&ctx, index, bufferBase, loopEvent]() {
            ccu::LocalCopy(
                ctx.resources.ccuBuf[bufferBase], ctx.loopSource[index], ctx.loopLength[index], loopEvent, 1);
            ccu::EventWait(loopEvent, 1);
            ccu::LocalCopy(ctx.loopDestination[index], ctx.resources.ccuBuf[bufferBase], ctx.loopLength[index],
                loopEvent, 1);
            ccu::EventWait(loopEvent, 1);
        }));
        entity.loops[index].reset(new ccu::Loop(entity.loopParam[index], *entity.body[index]));
    }
}

CcuResult GroupCopy(CopyContext &ctx, ccu::LocalAddr destination, ccu::LocalAddr source, GroupOpSizeVars size)
{
    InitializeCopyResources(ctx);
    CcuLoopEntity &loops = ctx.loopMap["localcopy"];

    CCU_IF(size.addrOffset != 0)
    {
        ctx.scratchLoopParam = GetLoopParam(0, ctx.config.memSlice * ctx.config.loopCount, 0);
        ctx.scratchLoopParam += size.loopParam;
        ctx.scratchMemSlice = ctx.config.memSlice;
        ctx.loopSource[0].addr = source.addr;
        ctx.loopSource[0].token = source.token;
        ctx.loopDestination[0].addr = destination.addr;
        ctx.loopDestination[0].token = destination.token;
        ctx.loopLength[0] = ctx.scratchMemSlice;
        loops.loopParam[0] = ctx.scratchLoopParam;
        ctx.scratchParallelConfig = GetParallelParam(ctx.config.loopCount - 1, 0, 1);
        ctx.scratchOffsetConfig = GetOffsetParam(ctx.config.memSlice, ctx.config.msInterleave, 1);
        std::vector<ccu::Loop> groupLoops{*loops.loops[0]};
        ccu::LoopGroup group(
            ctx.scratchParallelConfig, ctx.scratchOffsetConfig, ctx.config.loopCount, groupLoops);
    }

    CCU_IF(size.parallelParam != 0)
    {
        source.addr += size.addrOffset;
        destination.addr += size.addrOffset;
        ctx.loopSource[0].addr = source.addr;
        ctx.loopSource[0].token = source.token;
        ctx.loopDestination[0].addr = destination.addr;
        ctx.loopDestination[0].token = destination.token;
        ctx.loopLength[0] = size.residual;
        source.addr += size.residual;
        destination.addr += size.residual;

        ctx.scratchMemSlice = ctx.config.memSlice;
        ctx.loopSource[1].addr = source.addr;
        ctx.loopSource[1].token = source.token;
        ctx.loopDestination[1].addr = destination.addr;
        ctx.loopDestination[1].token = destination.token;
        ctx.loopLength[1] = ctx.scratchMemSlice;
        ctx.scratchLoopConfig[0] = GetLoopParam(0, 0, 1);
        ctx.scratchLoopConfig[1] = GetLoopParam(0, 0, 1);
        loops.loopParam[0] = ctx.scratchLoopConfig[0];
        loops.loopParam[1] = ctx.scratchLoopConfig[1];
        ctx.scratchOffsetConfig = GetOffsetParam(ctx.config.memSlice, ctx.config.msInterleave, 1);
        std::vector<ccu::Loop> groupLoops{*loops.loops[0], *loops.loops[1]};
        ccu::LoopGroup group(size.parallelParam, ctx.scratchOffsetConfig, ctx.config.loopCount, groupLoops);
    }
    return CCU_SUCCESS;
}

uint32_t ChannelIndex(uint32_t rankId, uint32_t peerRank)
{
    return peerRank < rankId ? peerRank : peerRank - 1;
}

template <typename KernelArg>
CcuResult ExchangeOutputResources(const KernelArg &arg, std::vector<ccu::Variable> &output,
    std::vector<ccu::Variable> &outputToken)
{
    for (uint32_t channel = 0; channel < arg.channelCount; ++channel) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg.channels[channel], output[arg.rankId], OUTPUT_XN_ID, PRE_SYNC_CKE, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg.channels[channel], outputToken[arg.rankId], TOKEN_XN_ID, PRE_SYNC_CKE, 1U << TOKEN_XN_ID));
    }
    const uint32_t resourceMask = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channel = 0; channel < arg.channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyWait(arg.channels[channel], PRE_SYNC_CKE, resourceMask));
    }
    return CCU_SUCCESS;
}

template <typename KernelArg> CcuResult FinalSynchronize(const KernelArg &arg)
{
    for (uint32_t channel = 0; channel < arg.channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyRecord(arg.channels[channel], FINAL_SYNC_CKE, FINAL_SYNC_BIT));
    }
    for (uint32_t channel = 0; channel < arg.channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyWait(arg.channels[channel], FINAL_SYNC_CKE, FINAL_SYNC_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult IssueRelayTransfer(
    RelayContext &ctx, const TransferDesc &operation, uint32_t operationIndex, uint16_t eventBit)
{
    ccu::LocalAddr &source = ctx.transferSource[operationIndex];
    ccu::RemoteAddr &destination = ctx.transferDestination[operationIndex];
    if (operation.kind == TransferKind::RELAY) {
        source.addr = ctx.output[ctx.arg->rankId];
        source.addr += ctx.rankOffset[operation.ownerRank];
        source.addr += ctx.chunkOffset[operation.chunkId];
        source.token = ctx.outputToken[ctx.arg->rankId];
    } else {
        source.addr = ctx.input;
        source.addr += ctx.chunkOffset[operation.chunkId];
        source.token = ctx.inputToken;
    }
    destination.addr = ctx.output[operation.peerRank];
    destination.addr += ctx.rankOffset[operation.ownerRank];
    destination.addr += ctx.chunkOffset[operation.chunkId];
    destination.token = ctx.outputToken[operation.peerRank];
    const uint32_t channel = ChannelIndex(ctx.arg->rankId, operation.peerRank);
    return ccu::Write(ctx.arg->channels[channel], destination, source, ctx.chunkSize[operation.chunkId],
        ctx.networkEvent, eventBit);
}

CcuResult CopyLocalRelayChunks(RelayContext &ctx, uint8_t chunkMask)
{
    for (uint32_t chunk = 0; chunk < RELAY_CHUNK_COUNT; ++chunk) {
        if ((chunkMask & (1U << chunk)) == 0) {
            continue;
        }
        ctx.localCopySource.addr = ctx.input;
        ctx.localCopySource.addr += ctx.chunkOffset[chunk];
        ctx.localCopySource.token = ctx.inputToken;
        ctx.localCopyDestination.addr = ctx.output[ctx.arg->rankId];
        ctx.localCopyDestination.addr += ctx.rankOffset[ctx.arg->rankId];
        ctx.localCopyDestination.addr += ctx.chunkOffset[chunk];
        ctx.localCopyDestination.token = ctx.outputToken[ctx.arg->rankId];
        CCU_CHK_RET(GroupCopy(
            ctx.copy, ctx.localCopyDestination, ctx.localCopySource, ctx.copySize[chunk]));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult DirectAllGatherKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDirect *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize < 2 || kernelArg->rankSize > MAX_RANK_SIZE ||
        kernelArg->rankId >= kernelArg->rankSize || kernelArg->channelCount == 0 ||
        kernelArg->channelCount >= kernelArg->rankSize) {
        return CCU_E_INTERNAL;
    }

    DirectContext ctx;
    ctx.arg = kernelArg;
    ctx.output.resize(ctx.arg->rankSize);
    ctx.outputToken.resize(ctx.arg->rankSize);
    for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
        const uint32_t peer = ctx.arg->peerRanks[channel];
        if (peer >= ctx.arg->rankSize || peer == ctx.arg->rankId) {
            return CCU_E_INTERNAL;
        }
        ctx.output[peer] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], OUTPUT_XN_ID);
        ctx.outputToken[peer] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], TOKEN_XN_ID);
    }

    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.copySize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.copySize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.copySize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.copySize.residual, argId++));
    CCU_CHK_RET(ExchangeOutputResources(*ctx.arg, ctx.output, ctx.outputToken));

    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += ctx.inputOffset;
    source.token = ctx.inputToken;
    uint16_t completionMask = 0;
    for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
        const uint32_t peer = ctx.arg->peerRanks[channel];
        ccu::RemoteAddr destination;
        destination.addr = ctx.output[peer];
        destination.addr += ctx.outputOffset;
        destination.token = ctx.outputToken[peer];
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channel], destination, source, ctx.sliceSize,
            ctx.networkEvent, eventBit));
        completionMask |= eventBit;
    }

    if (ctx.arg->ownsLocalCopy != 0) {
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.output[ctx.arg->rankId];
        localDestination.addr += ctx.outputOffset;
        localDestination.token = ctx.outputToken[ctx.arg->rankId];
        const uint16_t localBit = static_cast<uint16_t>(1U << ctx.arg->rankId);
        if (ctx.arg->useSmallMessageFastPath != 0) {
            CCU_CHK_RET(ccu::LocalCopy(localDestination, source, ctx.sliceSize, ctx.networkEvent, localBit));
        } else {
            CCU_CHK_RET(GroupCopy(ctx.copy, localDestination, source, ctx.copySize));
            CCU_CHK_RET(ccu::EventRecord(ctx.networkEvent, localBit));
        }
        completionMask |= localBit;
    }
    CCU_CHK_RET(ccu::EventWait(ctx.networkEvent, completionMask));
    if (ctx.arg->useSmallMessageFastPath != 0) {
        return CCU_SUCCESS;
    }
    return FinalSynchronize(*ctx.arg);
}

CcuResult RelayAllGather2x8Kernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgRelay2x8 *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize != MAX_RANK_SIZE ||
        kernelArg->channelCount != MAX_RANK_SIZE - 1) {
        return CCU_E_INTERNAL;
    }

    RelayContext ctx;
    ctx.arg = kernelArg;
    ctx.output.resize(ctx.arg->rankSize);
    ctx.outputToken.resize(ctx.arg->rankSize);
    uint32_t channel = 0;
    for (uint32_t peer = 0; peer < ctx.arg->rankSize; ++peer) {
        if (peer == ctx.arg->rankId) {
            continue;
        }
        ctx.output[peer] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], OUTPUT_XN_ID);
        ctx.outputToken[peer] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channel], TOKEN_XN_ID);
        ++channel;
    }

    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken[ctx.arg->rankId], argId++));
    for (uint32_t rank = 0; rank < ctx.arg->rankSize; ++rank) {
        CCU_CHK_RET(ccu::LoadArg(ctx.rankOffset[rank], argId++));
    }
    for (uint32_t chunk = 0; chunk < RELAY_CHUNK_COUNT; ++chunk) {
        CCU_CHK_RET(ccu::LoadArg(ctx.chunkOffset[chunk], argId++));
    }
    for (uint32_t chunk = 0; chunk < RELAY_CHUNK_COUNT; ++chunk) {
        CCU_CHK_RET(ccu::LoadArg(ctx.chunkSize[chunk], argId++));
    }
    for (uint32_t chunk = 0; chunk < RELAY_CHUNK_COUNT; ++chunk) {
        CCU_CHK_RET(ccu::LoadArg(ctx.copySize[chunk].addrOffset, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.copySize[chunk].loopParam, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.copySize[chunk].parallelParam, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.copySize[chunk].residual, argId++));
    }
    CCU_CHK_RET(ExchangeOutputResources(*ctx.arg, ctx.output, ctx.outputToken));

    for (uint32_t slotIndex = 0; slotIndex < RELAY_SLOT_COUNT; ++slotIndex) {
        const RelaySlotSchedule &slot = ctx.arg->slots[slotIndex];
        uint32_t operationIndex = 0;
        uint16_t completionMask = 0;
        uint16_t relayReadyMask = 0;

        for (uint32_t index = 0; index < slot.closCount; ++index) {
            const uint16_t eventBit = static_cast<uint16_t>(1U << operationIndex++);
            CCU_CHK_RET(IssueRelayTransfer(ctx, slot.closOps[index], operationIndex - 1, eventBit));
            completionMask |= eventBit;
            if (slot.closOps[index].signalReady != 0) {
                relayReadyMask |= eventBit;
            }
        }
        for (uint32_t index = 0; index < slot.meshCount; ++index) {
            const TransferDesc &operation = slot.meshOps[index];
            if (operation.kind == TransferKind::RELAY) {
                continue;
            }
            const uint16_t eventBit = static_cast<uint16_t>(1U << operationIndex++);
            CCU_CHK_RET(IssueRelayTransfer(ctx, operation, operationIndex - 1, eventBit));
            completionMask |= eventBit;
        }

        CCU_CHK_RET(CopyLocalRelayChunks(ctx, slot.localCopyMask));

        if (relayReadyMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.networkEvent, relayReadyMask));
            completionMask = static_cast<uint16_t>(completionMask & ~relayReadyMask);
            for (uint32_t index = 0; index < slot.closCount; ++index) {
                const TransferDesc &operation = slot.closOps[index];
                if (operation.signalReady == 0) {
                    continue;
                }
                const uint32_t targetChannel = ChannelIndex(ctx.arg->rankId, operation.peerRank);
                CCU_CHK_RET(ccu::NotifyRecord(
                    ctx.arg->channels[targetChannel], RELAY_READY_CKE, 1U << operation.chunkId));
            }
        }

        for (uint32_t index = 0; index < slot.meshCount; ++index) {
            const TransferDesc &operation = slot.meshOps[index];
            if (operation.kind != TransferKind::RELAY) {
                continue;
            }
            if (operation.waitReady != 0) {
                const uint32_t sourceChannel = ChannelIndex(ctx.arg->rankId, operation.ownerRank);
                CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[sourceChannel], RELAY_READY_CKE,
                    1U << operation.chunkId));
            }
            const uint16_t eventBit = static_cast<uint16_t>(1U << operationIndex++);
            CCU_CHK_RET(IssueRelayTransfer(ctx, operation, operationIndex - 1, eventBit));
            completionMask |= eventBit;
        }

        if (completionMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.networkEvent, completionMask));
        }
    }
    return FinalSynchronize(*ctx.arg);
}

#undef CCU_CHK_RET

} // namespace ops_hccl
