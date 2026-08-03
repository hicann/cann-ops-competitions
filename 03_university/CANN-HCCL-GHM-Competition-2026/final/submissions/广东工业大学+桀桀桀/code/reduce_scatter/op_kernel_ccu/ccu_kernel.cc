/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 */

#include <array>
#include <memory>
#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;
namespace {

constexpr uint32_t CKE_INDEX = 0;
constexpr uint32_t INPUT_ADDR_XN = 0;
constexpr uint32_t INPUT_TOKEN_XN = 1;
constexpr uint32_t POST_SYNC_XN = 2;
constexpr uint32_t NHR_STEP_READY_XN = 3;
constexpr uint32_t NHR_STEP_DONE_XN = 4;
// Reads only consume immutable peer input, while every scratch slot has one
// local writer.  The host-side cross-CCU wait provides the only completion
// dependency required before the final local reduction.
constexpr bool ENABLE_GATHER_POST_SYNC = false;
constexpr bool ENABLE_PARALLEL_GATHER_READS = true;
constexpr uint32_t GROUP_BUFFER_STRIDE = 8;

#define CCU_RETURN_IF_ERROR(call) \
    do { \
        const CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

struct GatherContext {
    const ReduceScatterCcuKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable scratch;
    ccu::Variable inputToken;
    ccu::Variable scratchToken;
    ccu::Variable targetBlockOffset;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    ccu::Variable scratchStride;
    ccu::Variable hierarchicalTargetBlockBytes;
    ccu::Variable groupFullOffset;
    ccu::Variable groupFullLoopCount;
    ccu::Variable groupParallelParam;
    ccu::Variable groupResidualBytes;
    std::vector<ccu::Variable> peerInputs;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event event;
};

static CcuResult LoadGatherArgs(GatherContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratch, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.targetBlockOffset, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkOffset, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkBytes, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchStride, argIndex++));
    return CCU_SUCCESS;
}

static CcuResult LoadGroupReduceArgs(GatherContext &ctx)
{
    CCU_RETURN_IF_ERROR(LoadGatherArgs(ctx));
    uint32_t argIndex = 8;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.groupFullOffset, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.groupFullLoopCount, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.groupParallelParam, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.groupResidualBytes, argIndex++));
    return CCU_SUCCESS;
}

static CcuResult GatherFromPeers(GatherContext &ctx)
{
    const auto *arg = ctx.arg;
    ctx.peerInputs.resize(arg->commChannelCount);
    ctx.peerTokens.resize(arg->commChannelCount);
    uint16_t allReadMask = 0;

    // This slot is disjoint from every peer write below.  Staging it in the
    // first gather kernel overlaps the copy with the other network-layer
    // kernel and removes the standalone copy from the final reduction.
    if (arg->stageSelfInput) {
        ccu::LocalAddr selfInput;
        selfInput.addr = ctx.input;
        selfInput.addr += ctx.targetBlockOffset;
        selfInput.addr += ctx.chunkOffset;
        selfInput.token = ctx.inputToken;

        ccu::LocalAddr selfScratch;
        selfScratch.addr = ctx.scratch;
        ccu::Variable selfSlotOffset;
        selfSlotOffset = 0;
        for (uint32_t slot = 0; slot < arg->rankId; ++slot) {
            selfSlotOffset += ctx.scratchStride;
        }
        selfScratch.addr += selfSlotOffset;
        selfScratch.token = ctx.scratchToken;
        ccu::LocalCopy(selfScratch, selfInput, ctx.chunkBytes, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        ctx.peerInputs[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_ADDR_XN);
        ctx.peerTokens[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_TOKEN_XN);
    }

    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.input, INPUT_ADDR_XN, CKE_INDEX,
            1U << INPUT_ADDR_XN);
        ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.inputToken, INPUT_TOKEN_XN, CKE_INDEX,
            1U << INPUT_TOKEN_XN);
    }
    const uint32_t preSyncBits = (1U << INPUT_ADDR_XN) | (1U << INPUT_TOKEN_XN);
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        ccu::NotifyWait(arg->commChannels[channelIndex], CKE_INDEX, preSyncBits);
    }

    // Every source rank owns one CCL slot.  Both network-layer kernels use
    // disjoint peer sets, so their writes are safe to execute concurrently.
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        ccu::RemoteAddr remoteInput;
        remoteInput.addr = ctx.peerInputs[channelIndex];
        remoteInput.addr += ctx.targetBlockOffset;
        remoteInput.addr += ctx.chunkOffset;
        remoteInput.token = ctx.peerTokens[channelIndex];

        ccu::LocalAddr localScratch;
        localScratch.addr = ctx.scratch;
        ccu::Variable slotOffset;
        slotOffset = 0;
        for (uint32_t slot = 0; slot < arg->peerRanks[channelIndex]; ++slot) {
            slotOffset += ctx.scratchStride;
        }
        localScratch.addr += slotOffset;
        localScratch.token = ctx.scratchToken;

        const uint16_t readMask = ENABLE_PARALLEL_GATHER_READS ? static_cast<uint16_t>(1U << channelIndex) : 1;
        ccu::Read(arg->commChannels[channelIndex], localScratch, remoteInput, ctx.chunkBytes, ctx.event, readMask);
        if (ENABLE_PARALLEL_GATHER_READS) {
            allReadMask |= readMask;
        } else {
            ccu::EventWait(ctx.event, 1);
        }
    }
    if (ENABLE_PARALLEL_GATHER_READS && allReadMask != 0) {
        ccu::EventWait(ctx.event, allReadMask);
    }

    if (ENABLE_GATHER_POST_SYNC) {
        for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
            ccu::NotifyRecord(arg->commChannels[channelIndex], CKE_INDEX, 1U << POST_SYNC_XN);
        }
        for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
            ccu::NotifyWait(arg->commChannels[channelIndex], CKE_INDEX, 1U << POST_SYNC_XN);
        }
    }
    return CCU_SUCCESS;
}

static CcuResult GroupReduceFromPeers(GatherContext &ctx)
{
    const auto *arg = ctx.arg;
    const uint32_t inputCount = arg->commChannelCount + (arg->includeSelf ? 1U : 0U);
    if (inputCount < 2 || inputCount > GROUP_BUFFER_STRIDE) {
        return CCU_E_PARA;
    }
    ctx.peerInputs.resize(arg->commChannelCount);
    ctx.peerTokens.resize(arg->commChannelCount);
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        ctx.peerInputs[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_ADDR_XN);
        ctx.peerTokens[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_TOKEN_XN);
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.input, INPUT_ADDR_XN,
            CKE_INDEX, 1U << INPUT_ADDR_XN));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.inputToken,
            INPUT_TOKEN_XN, CKE_INDEX, 1U << INPUT_TOKEN_XN));
    }
    const uint32_t readyBits = (1U << INPUT_ADDR_XN) | (1U << INPUT_TOKEN_XN);
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(arg->commChannels[channelIndex], CKE_INDEX, readyBits));
    }

    ccu::Array<ccu::CcuBuffer> buffers(GROUP_BUFFER_STRIDE);
    uint16_t readyMask = 0;
    for (uint32_t peer = 0; peer < arg->commChannelCount; ++peer) {
        ccu::RemoteAddr remote;
        remote.addr = ctx.peerInputs[peer];
        remote.addr += ctx.targetBlockOffset;
        remote.addr += ctx.chunkOffset;
        remote.token = ctx.peerTokens[peer];
        const uint16_t eventMask = static_cast<uint16_t>(1U << peer);
        ccu::Read(arg->commChannels[peer], buffers[peer], remote, ctx.chunkBytes, ctx.event, eventMask);
        readyMask |= eventMask;
    }
    if (arg->includeSelf) {
        ccu::LocalAddr self;
        self.addr = ctx.input;
        self.addr += ctx.targetBlockOffset;
        self.addr += ctx.chunkOffset;
        self.token = ctx.inputToken;
        const uint16_t eventMask = static_cast<uint16_t>(1U << arg->commChannelCount);
        ccu::LocalCopy(buffers[arg->commChannelCount], self, ctx.chunkBytes, ctx.event, eventMask);
        readyMask |= eventMask;
    }
    ccu::EventWait(ctx.event, readyMask);
    ccu::LocalReduce(&buffers[0], inputCount, arg->dataType, arg->dataType, arg->reduceOp, ctx.chunkBytes, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
    ccu::LocalAddr output;
    output.addr = ctx.scratch;
    ccu::Variable partialOffset;
    partialOffset = 0;
    for (uint32_t slot = 0; slot < arg->partialResultSlot; ++slot) {
        partialOffset += ctx.scratchStride;
    }
    output.addr += partialOffset;
    output.addr += ctx.chunkOffset;
    output.token = ctx.scratchToken;
    ccu::LocalCopy(output, buffers[0], ctx.chunkBytes, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
    return CCU_SUCCESS;
}

struct FinalReduceContext {
    const ReduceScatterCcuKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable scratch;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchToken;
    ccu::Variable selfInputOffset;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    ccu::Variable scratchStride;
    ccu::Variable usePackedTree;
    ccu::Event event;
};

static CcuResult LoadFinalReduceArgs(FinalReduceContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratch, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.selfInputOffset, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkOffset, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkBytes, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchStride, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.usePackedTree, argIndex++));
    return CCU_SUCCESS;
}

static ccu::LocalAddr ScratchSlot(FinalReduceContext &ctx, uint32_t sourceRank)
{
    ccu::LocalAddr source;
    source.addr = ctx.scratch;
    ccu::Variable slotOffset;
    slotOffset = 0;
    for (uint32_t slot = 0; slot < sourceRank; ++slot) {
        slotOffset += ctx.scratchStride;
    }
    source.addr += slotOffset;
    source.token = ctx.scratchToken;
    return source;
}

static CcuResult ReduceInFixedTree(FinalReduceContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.chunkOffset;
    output.token = ctx.outputToken;

    std::vector<ccu::LocalAddr> active;
    active.reserve(arg->rankSize);
    for (uint32_t sourceRank = 0; sourceRank < arg->rankSize; ++sourceRank) {
        active.push_back(ScratchSlot(ctx, sourceRank));
    }

    if (!arg->selfInputStaged) {
        ccu::LocalAddr selfInput;
        selfInput.addr = ctx.input;
        selfInput.addr += ctx.selfInputOffset;
        selfInput.addr += ctx.chunkOffset;
        selfInput.token = ctx.inputToken;
        ccu::LocalCopy(active[arg->rankId], selfInput, ctx.chunkBytes, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }
    while (active.size() > 1) {
        std::vector<ccu::LocalAddr> next;
        next.reserve((active.size() + 1) / 2);
        uint16_t completeMask = 0;
        for (uint32_t pair = 0; pair + 1 < active.size(); pair += 2) {
            const uint16_t eventMask = static_cast<uint16_t>(1U << (pair / 2));
            ccu::LocalReduce(active[pair], active[pair + 1], ctx.chunkBytes, arg->dataType, arg->reduceOp,
                ctx.event, eventMask);
            completeMask |= eventMask;
            next.push_back(active[pair]);
        }
        if ((active.size() & 1U) != 0) {
            next.push_back(active.back());
        }
        ccu::EventWait(ctx.event, completeMask);
        active.swap(next);
    }

    ccu::LocalCopy(output, active.front(), ctx.chunkBytes, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
    return CCU_SUCCESS;
}

// When the chunk exactly fills every rank's CCL slot, corresponding source
// ranges are adjacent in memory.  One LocalReduce can then cover a whole tree
// level without crossing a slot boundary.  Tail chunks keep the general path
// because their source ranges are separated by the full slot stride.
static CcuResult ReduceInPackedFixedTree(FinalReduceContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.chunkOffset;
    output.token = ctx.outputToken;

    ccu::LocalAddr selfInput;
    selfInput.addr = ctx.input;
    selfInput.addr += ctx.selfInputOffset;
    selfInput.addr += ctx.chunkOffset;
    selfInput.token = ctx.inputToken;

    ccu::LocalAddr scratch = ScratchSlot(ctx, arg->rankId);
    // The rank-12 Mesh gather stages this exact source slot before the final
    // kernel is launched.  Reusing it removes one HBM copy while retaining
    // the same deterministic packed reduction tree.
    if (!arg->selfInputStaged) {
        ccu::LocalCopy(scratch, selfInput, ctx.chunkBytes, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }

    uint32_t activeCount = arg->rankSize;
    while (activeCount > 1) {
        const uint32_t reduceCount = activeCount / 2;
        ccu::LocalAddr dst = ScratchSlot(ctx, 0);
        ccu::LocalAddr src = ScratchSlot(ctx, activeCount - reduceCount);

        ccu::Variable bytes;
        bytes = ctx.chunkBytes;
        for (uint32_t index = 1; index < reduceCount; ++index) {
            bytes += ctx.chunkBytes;
        }
        ccu::LocalReduce(dst, src, bytes, arg->dataType, arg->reduceOp, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
        activeCount -= reduceCount;
    }

    scratch = ScratchSlot(ctx, 0);
    ccu::LocalCopy(output, scratch, ctx.chunkBytes, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
    return CCU_SUCCESS;
}

static CcuResult ReduceInSourceRankOrder(FinalReduceContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.chunkOffset;
    output.token = ctx.outputToken;

    ccu::LocalAddr selfInput;
    selfInput.addr = ctx.input;
    selfInput.addr += ctx.selfInputOffset;
    selfInput.addr += ctx.chunkOffset;
    selfInput.token = ctx.inputToken;

    ccu::LocalCopy(output, selfInput, ctx.chunkBytes, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);

    // The order is fixed by source rank, independent of the relative timing
    // of the two CCU kernels or any network transfer.
    for (uint32_t sourceRank = 0; sourceRank < arg->rankSize; ++sourceRank) {
        if (sourceRank == arg->rankId) {
            continue;
        }
        ccu::LocalAddr source = ScratchSlot(ctx, sourceRank);
        ccu::LocalReduce(output, source, ctx.chunkBytes, arg->dataType, arg->reduceOp, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }
    return CCU_SUCCESS;
}

struct PartialReduceContext {
    const ReduceScatterCcuKernelArg *arg = nullptr;
    ccu::Variable scratch;
    ccu::Variable output;
    ccu::Variable scratchToken;
    ccu::Variable outputToken;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    ccu::Variable scratchStride;
    ccu::Event event;
};

static ccu::LocalAddr PartialScratchSlot(PartialReduceContext &ctx, uint32_t sourceRank)
{
    ccu::LocalAddr source;
    source.addr = ctx.scratch;
    ccu::Variable offset;
    offset = 0;
    for (uint32_t slot = 0; slot < sourceRank; ++slot) {
        offset += ctx.scratchStride;
    }
    source.addr += offset;
    source.token = ctx.scratchToken;
    return source;
}

static CcuResult LoadLocalServerReduceArgs(PartialReduceContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratch, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkBytes, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchStride, argIndex++));
    return CCU_SUCCESS;
}

static CcuResult ReduceLocalServer(PartialReduceContext &ctx)
{
    const auto *arg = ctx.arg;
    std::vector<uint32_t> activeRanks;
    std::vector<ccu::LocalAddr> active;
    activeRanks.reserve(arg->localRankCount);
    active.reserve(arg->localRankCount);
    for (uint32_t index = 0; index < arg->localRankCount; ++index) {
        activeRanks.push_back(arg->localRanks[index]);
        active.push_back(PartialScratchSlot(ctx, arg->localRanks[index]));
    }
    while (active.size() > 1) {
        std::vector<uint32_t> nextRanks;
        std::vector<ccu::LocalAddr> next;
        uint16_t completeMask = 0;
        for (uint32_t pair = 0; pair + 1 < active.size(); pair += 2) {
            const uint16_t eventMask = static_cast<uint16_t>(1U << (pair / 2));
            ccu::LocalReduce(active[pair], active[pair + 1], ctx.chunkBytes, arg->dataType, arg->reduceOp,
                ctx.event, eventMask);
            completeMask |= eventMask;
            nextRanks.push_back(activeRanks[pair]);
            next.push_back(active[pair]);
        }
        if ((active.size() & 1U) != 0) {
            nextRanks.push_back(activeRanks.back());
            next.push_back(active.back());
        }
        ccu::EventWait(ctx.event, completeMask);
        activeRanks.swap(nextRanks);
        active.swap(next);
    }
    if (arg->partialResultSlot >= arg->rankSize) {
        return CCU_E_PARA;
    }
    if (activeRanks.front() != arg->partialResultSlot) {
        ccu::LocalCopy(PartialScratchSlot(ctx, arg->partialResultSlot), active.front(), ctx.chunkBytes, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }
    return CCU_SUCCESS;
}

static CcuResult LoadClosPairGatherArgs(PartialReduceContext &ctx)
{
    return LoadLocalServerReduceArgs(ctx);
}

static CcuResult GatherClosPair(PartialReduceContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::Variable peerScratch = ccu::GetResByChannel<ccu::Variable>(arg->commChannels[0], INPUT_ADDR_XN);
    ccu::Variable peerToken = ccu::GetResByChannel<ccu::Variable>(arg->commChannels[0], INPUT_TOKEN_XN);
    ccu::WriteVariableWithNotify(arg->commChannels[0], ctx.scratch, INPUT_ADDR_XN, CKE_INDEX,
        1U << INPUT_ADDR_XN);
    ccu::WriteVariableWithNotify(arg->commChannels[0], ctx.scratchToken, INPUT_TOKEN_XN, CKE_INDEX,
        1U << INPUT_TOKEN_XN);
    const uint32_t readyBits = (1U << INPUT_ADDR_XN) | (1U << INPUT_TOKEN_XN);
    ccu::NotifyWait(arg->commChannels[0], CKE_INDEX, readyBits);
    ccu::RemoteAddr remote;
    remote.addr = peerScratch;
    ccu::Variable peerOffset;
    peerOffset = 0;
    for (uint32_t slot = 0; slot < arg->closPeerRank; ++slot) {
        peerOffset += ctx.scratchStride;
    }
    remote.addr += peerOffset;
    remote.token = peerToken;
    ccu::Read(arg->commChannels[0], PartialScratchSlot(ctx, arg->closPeerRank), remote, ctx.chunkBytes,
        ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
    return CCU_SUCCESS;
}

static CcuResult LoadFinalPairReduceArgs(PartialReduceContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratch, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkOffset, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkBytes, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchStride, argIndex++));
    return CCU_SUCCESS;
}

static CcuResult ReduceFinalPair(PartialReduceContext &ctx)
{
    const auto *arg = ctx.arg;
    const uint32_t firstServerRank = arg->rankId < arg->closPeerRank ? arg->rankId : arg->closPeerRank;
    const uint32_t secondServerRank = arg->rankId < arg->closPeerRank ? arg->closPeerRank : arg->rankId;
    ccu::LocalAddr first = PartialScratchSlot(ctx, firstServerRank);
    ccu::LocalAddr second = PartialScratchSlot(ctx, secondServerRank);
    if (firstServerRank != secondServerRank) {
        ccu::LocalReduce(first, second, ctx.chunkBytes, arg->dataType, arg->reduceOp, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }

    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.chunkOffset;
    output.token = ctx.outputToken;
    ccu::LocalCopy(output, first, ctx.chunkBytes, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
    return CCU_SUCCESS;
}

// The 2x8 hierarchy uses a compact, per-chunk CCL layout:
// [0..7] local-server source blocks, [8] local partial, [9] remote partial.
// A partial slot is never overwritten after the Clos peer can observe it.
constexpr uint32_t HIERARCHICAL_LOCAL_SOURCE_COUNT = 8;
constexpr uint32_t HIERARCHICAL_LOCAL_PARTIAL_SLOT = HIERARCHICAL_LOCAL_SOURCE_COUNT;
constexpr uint32_t HIERARCHICAL_EXPORTED_PARTIAL_SLOT = HIERARCHICAL_LOCAL_PARTIAL_SLOT + 1;
constexpr uint32_t HIERARCHICAL_REMOTE_PARTIAL_SLOT = HIERARCHICAL_EXPORTED_PARTIAL_SLOT + 1;

static ccu::LocalAddr HierarchicalSlot(PartialReduceContext &ctx, uint32_t slot)
{
    ccu::LocalAddr address;
    address.addr = ctx.scratch;
    ccu::Variable offset;
    offset = 0;
    for (uint32_t index = 0; index < slot; ++index) {
        offset += ctx.scratchStride;
    }
    address.addr += offset;
    address.token = ctx.scratchToken;
    return address;
}

static CcuResult LoadHierarchicalMeshArgs(GatherContext &ctx)
{
    CCU_RETURN_IF_ERROR(LoadGatherArgs(ctx));
    return ccu::LoadArg(ctx.hierarchicalTargetBlockBytes, 8);
}

static uint32_t LocalSourceIndex(const ReduceScatterCcuKernelArg *arg, uint32_t rank)
{
    for (uint32_t index = 0; index < arg->localRankCount; ++index) {
        if (arg->localRanks[index] == rank) {
            return index;
        }
    }
    return INVALID_VALUE_RANKID;
}

static CcuResult ComputeHierarchicalPartial(GatherContext &ctx, PartialReduceContext &partial,
    ccu::Variable targetBlockOffset, uint32_t partialSlot)
{
    const auto *arg = ctx.arg;
    const uint32_t selfIndex = LocalSourceIndex(arg, arg->rankId);
    if (selfIndex == INVALID_VALUE_RANKID) {
        return CCU_E_PARA;
    }
    ccu::LocalAddr selfInput;
    selfInput.addr = ctx.input;
    selfInput.addr += targetBlockOffset;
    selfInput.addr += ctx.chunkOffset;
    selfInput.token = ctx.inputToken;
    ccu::LocalCopy(HierarchicalSlot(partial, selfIndex), selfInput, ctx.chunkBytes, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);

    uint16_t readMask = 0;
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        const uint32_t sourceIndex = LocalSourceIndex(arg, arg->peerRanks[channelIndex]);
        if (sourceIndex == INVALID_VALUE_RANKID) {
            return CCU_E_PARA;
        }
        ccu::RemoteAddr remoteInput;
        remoteInput.addr = ctx.peerInputs[channelIndex];
        remoteInput.addr += targetBlockOffset;
        remoteInput.addr += ctx.chunkOffset;
        remoteInput.token = ctx.peerTokens[channelIndex];
        const uint16_t eventMask = static_cast<uint16_t>(1U << channelIndex);
        ccu::Read(arg->commChannels[channelIndex], HierarchicalSlot(partial, sourceIndex), remoteInput,
            ctx.chunkBytes, ctx.event, eventMask);
        readMask |= eventMask;
    }
    ccu::EventWait(ctx.event, readMask);

    std::vector<ccu::LocalAddr> active;
    active.reserve(HIERARCHICAL_LOCAL_SOURCE_COUNT);
    for (uint32_t index = 0; index < HIERARCHICAL_LOCAL_SOURCE_COUNT; ++index) {
        active.push_back(HierarchicalSlot(partial, index));
    }
    while (active.size() > 1) {
        std::vector<ccu::LocalAddr> next;
        uint16_t completeMask = 0;
        for (uint32_t pair = 0; pair + 1 < active.size(); pair += 2) {
            const uint16_t eventMask = static_cast<uint16_t>(1U << (pair / 2));
            ccu::LocalReduce(active[pair], active[pair + 1], ctx.chunkBytes, arg->dataType, arg->reduceOp,
                ctx.event, eventMask);
            completeMask |= eventMask;
            next.push_back(active[pair]);
        }
        ccu::EventWait(ctx.event, completeMask);
        active.swap(next);
    }
    ccu::LocalCopy(HierarchicalSlot(partial, partialSlot), active.front(), ctx.chunkBytes,
        ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
    return CCU_SUCCESS;
}

static CcuResult RunHierarchicalMeshPartial(GatherContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->localRankCount != HIERARCHICAL_LOCAL_SOURCE_COUNT ||
        arg->commChannelCount + 1 != HIERARCHICAL_LOCAL_SOURCE_COUNT) {
        return CCU_E_PARA;
    }
    ctx.peerInputs.resize(arg->commChannelCount);
    ctx.peerTokens.resize(arg->commChannelCount);
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        ctx.peerInputs[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_ADDR_XN);
        ctx.peerTokens[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_TOKEN_XN);
        ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.input, INPUT_ADDR_XN, CKE_INDEX,
            1U << INPUT_ADDR_XN);
        ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.inputToken, INPUT_TOKEN_XN, CKE_INDEX,
            1U << INPUT_TOKEN_XN);
    }
    const uint32_t readyBits = (1U << INPUT_ADDR_XN) | (1U << INPUT_TOKEN_XN);
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        ccu::NotifyWait(arg->commChannels[channelIndex], CKE_INDEX, readyBits);
    }
    PartialReduceContext partial;
    partial.arg = arg;
    partial.scratch = ctx.scratch;
    partial.scratchToken = ctx.scratchToken;
    partial.chunkBytes = ctx.chunkBytes;
    partial.scratchStride = ctx.scratchStride;
    CCU_RETURN_IF_ERROR(ComputeHierarchicalPartial(ctx, partial, ctx.targetBlockOffset,
        HIERARCHICAL_LOCAL_PARTIAL_SLOT));
    ccu::Variable partnerBlockOffset;
    partnerBlockOffset = 0;
    for (uint32_t rank = 0; rank < arg->closPeerRank; ++rank) {
        partnerBlockOffset += ctx.hierarchicalTargetBlockBytes;
    }
    return ComputeHierarchicalPartial(ctx, partial, partnerBlockOffset, HIERARCHICAL_EXPORTED_PARTIAL_SLOT);
}

static CcuResult RunHierarchicalClosPair(PartialReduceContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->commChannelCount != 1) {
        return CCU_E_PARA;
    }
    ccu::Variable peerScratch = ccu::GetResByChannel<ccu::Variable>(arg->commChannels[0], INPUT_ADDR_XN);
    ccu::Variable peerToken = ccu::GetResByChannel<ccu::Variable>(arg->commChannels[0], INPUT_TOKEN_XN);
    ccu::WriteVariableWithNotify(arg->commChannels[0], ctx.scratch, INPUT_ADDR_XN, CKE_INDEX,
        1U << INPUT_ADDR_XN);
    ccu::WriteVariableWithNotify(arg->commChannels[0], ctx.scratchToken, INPUT_TOKEN_XN, CKE_INDEX,
        1U << INPUT_TOKEN_XN);
    const uint32_t readyBits = (1U << INPUT_ADDR_XN) | (1U << INPUT_TOKEN_XN);
    ccu::NotifyWait(arg->commChannels[0], CKE_INDEX, readyBits);

    ccu::RemoteAddr remotePartial;
    remotePartial.addr = peerScratch;
    for (uint32_t index = 0; index < HIERARCHICAL_EXPORTED_PARTIAL_SLOT; ++index) {
        remotePartial.addr += ctx.scratchStride;
    }
    remotePartial.token = peerToken;
    ccu::Read(arg->commChannels[0], HierarchicalSlot(ctx, HIERARCHICAL_REMOTE_PARTIAL_SLOT), remotePartial,
        ctx.chunkBytes, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
    return CCU_SUCCESS;
}

static CcuResult RunHierarchicalFinal(PartialReduceContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.chunkOffset;
    output.token = ctx.outputToken;
    ccu::LocalCopy(output, HierarchicalSlot(ctx, HIERARCHICAL_LOCAL_PARTIAL_SLOT), ctx.chunkBytes, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
    ccu::LocalReduce(output, HierarchicalSlot(ctx, HIERARCHICAL_REMOTE_PARTIAL_SLOT), ctx.chunkBytes,
        arg->dataType, arg->reduceOp, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
    return CCU_SUCCESS;
}

struct NhrContext {
    const ReduceScatterCcuKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable outputBlockBytes;
    std::vector<ccu::Variable> peerInputs;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event event;
};

static CcuResult LoadNhrArgs(NhrContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputBlockBytes, argIndex++));
    return CCU_SUCCESS;
}

static uint32_t FindNhrChannel(const ReduceScatterCcuKernelArg *arg, uint32_t peerRank)
{
    for (uint32_t index = 0; index < arg->commChannelCount; ++index) {
        if (arg->peerRanks[index] == peerRank) {
            return index;
        }
    }
    return INVALID_VALUE_RANKID;
}

static CcuResult RunNhr(NhrContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->nhrStepCount == 0 || arg->nhrStepCount > 4) {
        return CCU_E_PARA;
    }
    ctx.peerInputs.resize(arg->commChannelCount);
    ctx.peerTokens.resize(arg->commChannelCount);
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        ctx.peerInputs[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_ADDR_XN);
        ctx.peerTokens[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_TOKEN_XN);
        ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.input, INPUT_ADDR_XN, CKE_INDEX,
            1U << INPUT_ADDR_XN);
        ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.inputToken, INPUT_TOKEN_XN, CKE_INDEX,
            1U << INPUT_TOKEN_XN);
    }
    const uint32_t readyBits = (1U << INPUT_ADDR_XN) | (1U << INPUT_TOKEN_XN);
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        ccu::NotifyWait(arg->commChannels[channelIndex], CKE_INDEX, readyBits);
    }

    for (uint32_t stepIndex = 0; stepIndex < arg->nhrStepCount; ++stepIndex) {
        const uint32_t distance = arg->nhrDistances[stepIndex];
        if (distance == 0 || distance >= arg->rankSize || (distance & (distance - 1U)) != 0) {
            return CCU_E_PARA;
        }
        const uint32_t peerRank = arg->rankId ^ distance;
        const uint32_t channelIndex = FindNhrChannel(arg, peerRank);
        if (channelIndex == INVALID_VALUE_RANKID) {
            return CCU_E_PARA;
        }
        const uint32_t groupStart = arg->rankId & ~(2U * distance - 1U);
        const uint32_t sendBlockStart = groupStart + ((arg->rankId & distance) == 0 ? distance : 0);

        // Each peer sees exactly one record and one wait on both step bits.
        // That makes the remote destination stable before WriteReduce and
        // confirms the peer's write before the next halving round reads it.
        ccu::NotifyRecord(arg->commChannels[channelIndex], CKE_INDEX, 1U << NHR_STEP_READY_XN);
        ccu::NotifyWait(arg->commChannels[channelIndex], CKE_INDEX, 1U << NHR_STEP_READY_XN);

        ccu::LocalAddr localSource;
        localSource.addr = ctx.input;
        ccu::Variable sendOffset;
        sendOffset = 0;
        for (uint32_t block = 0; block < sendBlockStart; ++block) {
            sendOffset += ctx.outputBlockBytes;
        }
        localSource.addr += sendOffset;
        localSource.token = ctx.inputToken;

        ccu::RemoteAddr remoteDestination;
        remoteDestination.addr = ctx.peerInputs[channelIndex];
        remoteDestination.addr += sendOffset;
        remoteDestination.token = ctx.peerTokens[channelIndex];

        ccu::Variable transferBytes;
        transferBytes = ctx.outputBlockBytes;
        for (uint32_t block = 1; block < distance; ++block) {
            transferBytes += ctx.outputBlockBytes;
        }
        ccu::WriteReduce(arg->commChannels[channelIndex], remoteDestination, localSource, transferBytes,
            arg->dataType, arg->reduceOp, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);

        ccu::NotifyRecord(arg->commChannels[channelIndex], CKE_INDEX, 1U << NHR_STEP_DONE_XN);
        ccu::NotifyWait(arg->commChannels[channelIndex], CKE_INDEX, 1U << NHR_STEP_DONE_XN);
    }

    if (arg->nhrWriteOutput) {
        ccu::LocalAddr finalInput;
        finalInput.addr = ctx.input;
        ccu::Variable finalOffset;
        finalOffset = 0;
        for (uint32_t block = 0; block < arg->rankId; ++block) {
            finalOffset += ctx.outputBlockBytes;
        }
        finalInput.addr += finalOffset;
        finalInput.token = ctx.inputToken;

        ccu::LocalAddr finalOutput;
        finalOutput.addr = ctx.output;
        finalOutput.token = ctx.outputToken;
        ccu::LocalCopy(finalOutput, finalInput, ctx.outputBlockBytes, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }
    return CCU_SUCCESS;
}

struct StripedClosDirectContext {
    const ReduceScatterCcuKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable targetBlockOffset;
    ccu::Variable stripeOffsets[3];
    ccu::Variable stripeBytes[3];
    std::vector<ccu::Variable> peerInputs;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event event;
};

static CcuResult LoadStripedClosDirectArgs(StripedClosDirectContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.targetBlockOffset, argIndex++));
    for (uint32_t stripe = 0; stripe < 3; ++stripe) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.stripeOffsets[stripe], argIndex++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.stripeBytes[stripe], argIndex++));
    }
    return CCU_SUCCESS;
}

static CcuResult RunStripedClosDirect(StripedClosDirectContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankSize != 4 || arg->commChannelCount != 3) {
        return CCU_E_PARA;
    }

    ctx.peerInputs.resize(arg->commChannelCount);
    ctx.peerTokens.resize(arg->commChannelCount);
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        ctx.peerInputs[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_ADDR_XN);
        ctx.peerTokens[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_TOKEN_XN);
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.input,
            INPUT_ADDR_XN, CKE_INDEX, 1U << INPUT_ADDR_XN));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.inputToken,
            INPUT_TOKEN_XN, CKE_INDEX, 1U << INPUT_TOKEN_XN));
    }
    const uint32_t readyBits = (1U << INPUT_ADDR_XN) | (1U << INPUT_TOKEN_XN);
    for (uint32_t channelIndex = 0; channelIndex < arg->commChannelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(arg->commChannels[channelIndex], CKE_INDEX, readyBits));
    }

    if (arg->stageSelfInput) {
        // Queue all three independent self copies first.  The first network
        // phase only waits for its own stripe, allowing the local copy engine
        // to overlap the remaining copies with Clos traffic.
        for (uint32_t stripe = 0; stripe < 3; ++stripe) {
            ccu::LocalAddr selfInput;
            selfInput.addr = ctx.input;
            selfInput.addr += ctx.targetBlockOffset;
            selfInput.addr += ctx.stripeOffsets[stripe];
            selfInput.token = ctx.inputToken;
            ccu::LocalAddr selfOutput;
            selfOutput.addr = ctx.output;
            selfOutput.addr += ctx.stripeOffsets[stripe];
            selfOutput.token = ctx.outputToken;
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(selfOutput, selfInput, ctx.stripeBytes[stripe], ctx.event,
                static_cast<uint16_t>(1U << stripe)));
        }
    } else {
        ccu::LocalAddr selfInput;
        selfInput.addr = ctx.input;
        selfInput.addr += ctx.targetBlockOffset;
        selfInput.token = ctx.inputToken;
        ccu::LocalAddr selfOutput;
        selfOutput.addr = ctx.output;
        selfOutput.token = ctx.outputToken;
        ccu::Variable outputBytes;
        outputBytes = ctx.stripeBytes[0];
        outputBytes += ctx.stripeBytes[1];
        outputBytes += ctx.stripeBytes[2];
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(selfOutput, selfInput, outputBytes, ctx.event, 1));
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, 1));
    }

    // Each phase sends a different Clos channel to each disjoint stripe.
    // Phase barriers fix the fp32 operand order for every output element.
    for (uint32_t phase = 0; phase < 3; ++phase) {
        uint16_t phaseMask = 0;
        for (uint32_t stripe = 0; stripe < 3; ++stripe) {
            if (phase == 0 && arg->stageSelfInput) {
                CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, static_cast<uint16_t>(1U << stripe)));
            }
            const uint32_t channelIndex = (phase + stripe) % 3;
            ccu::LocalAddr localOutput;
            localOutput.addr = ctx.output;
            localOutput.addr += ctx.stripeOffsets[stripe];
            localOutput.token = ctx.outputToken;
            ccu::RemoteAddr remoteInput;
            remoteInput.addr = ctx.peerInputs[channelIndex];
            remoteInput.addr += ctx.targetBlockOffset;
            remoteInput.addr += ctx.stripeOffsets[stripe];
            remoteInput.token = ctx.peerTokens[channelIndex];
            const uint16_t eventMask = static_cast<uint16_t>(1U << stripe);
            CCU_RETURN_IF_ERROR(ccu::ReadReduce(arg->commChannels[channelIndex], localOutput, remoteInput,
                ctx.stripeBytes[stripe], arg->dataType, arg->reduceOp, ctx.event, eventMask));
            phaseMask |= eventMask;
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, phaseMask));
    }
    return CCU_SUCCESS;
}

struct StripedLayerDirectContext {
    const ReduceScatterCcuKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable targetBlockOffset;
    ccu::Variable regularStripeBytes;
    ccu::Variable lastStripeBytes;
    std::vector<ccu::Variable> peerInputs;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event event;
};

static CcuResult LoadStripedLayerDirectArgs(StripedLayerDirectContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.targetBlockOffset, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.regularStripeBytes, argIndex++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.lastStripeBytes, argIndex++));
    return CCU_SUCCESS;
}

static CcuResult RunStripedLayerDirect(StripedLayerDirectContext &ctx)
{
    const auto *arg = ctx.arg;
    const uint32_t stripeCount = arg->commChannelCount;
    if (stripeCount == 0 || stripeCount > 15) {
        return CCU_E_PARA;
    }

    ctx.peerInputs.resize(stripeCount);
    ctx.peerTokens.resize(stripeCount);
    for (uint32_t channelIndex = 0; channelIndex < stripeCount; ++channelIndex) {
        ctx.peerInputs[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_ADDR_XN);
        ctx.peerTokens[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(arg->commChannels[channelIndex], INPUT_TOKEN_XN);
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.input,
            INPUT_ADDR_XN, CKE_INDEX, 1U << INPUT_ADDR_XN));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(arg->commChannels[channelIndex], ctx.inputToken,
            INPUT_TOKEN_XN, CKE_INDEX, 1U << INPUT_TOKEN_XN));
    }
    const uint32_t readyBits = (1U << INPUT_ADDR_XN) | (1U << INPUT_TOKEN_XN);
    for (uint32_t channelIndex = 0; channelIndex < stripeCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(arg->commChannels[channelIndex], CKE_INDEX, readyBits));
    }

    if (arg->directInitializeOutput) {
        ccu::LocalAddr selfInput;
        selfInput.addr = ctx.input;
        selfInput.addr += ctx.targetBlockOffset;
        selfInput.token = ctx.inputToken;
        ccu::LocalAddr selfOutput;
        selfOutput.addr = ctx.output;
        selfOutput.token = ctx.outputToken;
        ccu::Variable outputBytes;
        outputBytes = ctx.lastStripeBytes;
        for (uint32_t stripe = 1; stripe < stripeCount; ++stripe) {
            outputBytes += ctx.regularStripeBytes;
        }
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(selfOutput, selfInput, outputBytes, ctx.event, 1));
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, 1));
    }

    // A cyclic channel assignment makes every physical link active in each
    // phase.  A stripe is updated at most once per phase, which fixes its
    // fp32 accumulation order without cross-Die concurrent writes.
    for (uint32_t phase = 0; phase < stripeCount; ++phase) {
        uint16_t phaseMask = 0;
        for (uint32_t stripe = 0; stripe < stripeCount; ++stripe) {
            const uint32_t channelIndex = (phase + stripe) % stripeCount;
            ccu::Variable stripeOffset;
            stripeOffset = 0;
            for (uint32_t prefix = 0; prefix < stripe; ++prefix) {
                stripeOffset += ctx.regularStripeBytes;
            }
            ccu::Variable stripeBytes;
            stripeBytes = stripe + 1 == stripeCount ? ctx.lastStripeBytes : ctx.regularStripeBytes;
            ccu::LocalAddr localOutput;
            localOutput.addr = ctx.output;
            localOutput.addr += stripeOffset;
            localOutput.token = ctx.outputToken;
            ccu::RemoteAddr remoteInput;
            remoteInput.addr = ctx.peerInputs[channelIndex];
            remoteInput.addr += ctx.targetBlockOffset;
            remoteInput.addr += stripeOffset;
            remoteInput.token = ctx.peerTokens[channelIndex];
            const uint16_t eventMask = static_cast<uint16_t>(1U << stripe);
            CCU_RETURN_IF_ERROR(ccu::ReadReduce(arg->commChannels[channelIndex], localOutput, remoteInput,
                stripeBytes, arg->dataType, arg->reduceOp, ctx.event, eventMask));
            phaseMask |= eventMask;
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, phaseMask));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuGatherKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<ReduceScatterCcuKernelArg *>(kernelArg);
    GatherContext ctx;
    ctx.arg = arg;
    if (arg->kind == KernelKind::GROUP_REDUCE) {
        CCU_RETURN_IF_ERROR(LoadGroupReduceArgs(ctx));
        return GroupReduceFromPeers(ctx);
    }
    CCU_RETURN_IF_ERROR(LoadGatherArgs(ctx));
    CCU_RETURN_IF_ERROR(GatherFromPeers(ctx));
    if (arg->kind == KernelKind::LOCAL_SERVER_REDUCE) {
        PartialReduceContext partial;
        partial.arg = arg;
        partial.scratch = ctx.scratch;
        partial.scratchToken = ctx.scratchToken;
        partial.chunkBytes = ctx.chunkBytes;
        partial.scratchStride = ctx.scratchStride;
        return ReduceLocalServer(partial);
    }
    return CCU_SUCCESS;
}

CcuResult CcuStripedClosDirectKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<ReduceScatterCcuKernelArg *>(kernelArg);
    StripedClosDirectContext ctx;
    ctx.arg = arg;
    CCU_RETURN_IF_ERROR(LoadStripedClosDirectArgs(ctx));
    return RunStripedClosDirect(ctx);
}

CcuResult CcuStripedLayerDirectKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<ReduceScatterCcuKernelArg *>(kernelArg);
    StripedLayerDirectContext ctx;
    ctx.arg = arg;
    CCU_RETURN_IF_ERROR(LoadStripedLayerDirectArgs(ctx));
    return RunStripedLayerDirect(ctx);
}

CcuResult CcuFinalReduceKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<ReduceScatterCcuKernelArg *>(kernelArg);
    FinalReduceContext ctx;
    ctx.arg = arg;
    CCU_RETURN_IF_ERROR(LoadFinalReduceArgs(ctx));
    // slotOffsets[1] is the host-provided CCL slot stride.  CCU_IF keeps the
    // two data-dependent paths in one registered kernel, which is required by
    // the competition CCU graph registration model.
    if (arg->rankSize == 16 || arg->rankSize == 12) {
        CCU_IF(ctx.usePackedTree == 1) {
            CCU_RETURN_IF_ERROR(ReduceInPackedFixedTree(ctx));
        }
        CCU_IF(ctx.usePackedTree != 1) {
            if (arg->rankSize == 16) {
                CCU_RETURN_IF_ERROR(ReduceInFixedTree(ctx));
            } else if (arg->useSmallTree) {
                CCU_RETURN_IF_ERROR(ReduceInFixedTree(ctx));
            } else {
                CCU_RETURN_IF_ERROR(ReduceInSourceRankOrder(ctx));
            }
        }
    } else {
        // This value is fixed when the CCU kernel is registered.  Trees
        // remove launch overhead for the 512KB performance case; larger
        // messages use source-rank order to avoid local HBM contention.
        if (arg->useSmallTree) {
            CCU_RETURN_IF_ERROR(ReduceInFixedTree(ctx));
        } else {
            CCU_RETURN_IF_ERROR(ReduceInSourceRankOrder(ctx));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuLocalServerReduceKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<ReduceScatterCcuKernelArg *>(kernelArg);
    PartialReduceContext ctx;
    ctx.arg = arg;
    CCU_RETURN_IF_ERROR(LoadLocalServerReduceArgs(ctx));
    return ReduceLocalServer(ctx);
}

CcuResult CcuClosPairGatherKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<ReduceScatterCcuKernelArg *>(kernelArg);
    PartialReduceContext ctx;
    ctx.arg = arg;
    CCU_RETURN_IF_ERROR(LoadClosPairGatherArgs(ctx));
    return GatherClosPair(ctx);
}

CcuResult CcuFinalPairReduceKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<ReduceScatterCcuKernelArg *>(kernelArg);
    PartialReduceContext ctx;
    ctx.arg = arg;
    CCU_RETURN_IF_ERROR(LoadFinalPairReduceArgs(ctx));
    return ReduceFinalPair(ctx);
}

CcuResult CcuHierarchicalMeshPartialKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<ReduceScatterCcuKernelArg *>(kernelArg);
    GatherContext ctx;
    ctx.arg = arg;
    CCU_RETURN_IF_ERROR(LoadHierarchicalMeshArgs(ctx));
    return RunHierarchicalMeshPartial(ctx);
}

CcuResult CcuHierarchicalClosPairKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<ReduceScatterCcuKernelArg *>(kernelArg);
    PartialReduceContext ctx;
    ctx.arg = arg;
    CCU_RETURN_IF_ERROR(LoadClosPairGatherArgs(ctx));
    return RunHierarchicalClosPair(ctx);
}

CcuResult CcuHierarchicalFinalKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<ReduceScatterCcuKernelArg *>(kernelArg);
    PartialReduceContext ctx;
    ctx.arg = arg;
    CCU_RETURN_IF_ERROR(LoadFinalPairReduceArgs(ctx));
    return RunHierarchicalFinal(ctx);
}

CcuResult CcuNhrKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<ReduceScatterCcuKernelArg *>(kernelArg);
    NhrContext ctx;
    ctx.arg = arg;
    CCU_RETURN_IF_ERROR(LoadNhrArgs(ctx));
    return RunNhr(ctx);
}

} // namespace ops_hccl

#undef CCU_RETURN_IF_ERROR
