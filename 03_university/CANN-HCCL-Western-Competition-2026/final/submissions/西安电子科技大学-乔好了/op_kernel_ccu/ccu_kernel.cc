#include "ccu_kernel.h"

#include <vector>

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
constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t PARTIAL_DONE_BIT = 4;
constexpr uint32_t OWNER_DONE_BIT = 5;
constexpr uint32_t CKE_INDEX = 0;
constexpr uint32_t P4_RANK_SIZE = 4;
constexpr uint32_t P16_RANK_SIZE = 16;
constexpr uint32_t P16_MAX_GROUP_SOURCES = 9;
constexpr uint32_t P16_DUAL_CHAIN_SOURCES = 8;
constexpr uint32_t P16_DUAL_CHAIN_WIDTH = 4;
struct KernelContext {
    const CcuKernelArgAllReduce *arg;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable token;
    ccu::Variable scratchSave;
    ccu::Variable scratchPartial;
    ccu::Variable tileOffset;
    ccu::Variable segmentSize;
    ccu::Variable lastSegmentSize;
    ccu::Variable scratchStride;
    ccu::Variable ownerMainSize;
    ccu::Variable ownerTailSize;
    ccu::Variable inputOutputEqual;
    std::vector<ccu::Variable> peerInput;
    std::vector<ccu::Variable> peerOutput;
    std::vector<ccu::Variable> peerToken;
    ccu::Event event;
};

CcuResult InitResource(KernelContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankSize == 0 || arg->rankSize > MAX_RANK_SIZE ||
        arg->channelCount == 0 || arg->channelCount >= arg->rankSize ||
        arg->kernelCount == 0 || arg->kernelCount > 2 || arg->dieId >= 2 ||
        arg->reduceWithLocal > 1 || arg->ownsLocalSource > 1 ||
        arg->useP16DualChain > 2) {
        return CcuResult::CCU_E_PARA;
    }
    ctx.peerInput.resize(arg->channelCount);
    ctx.peerOutput.resize(arg->channelCount);
    ctx.peerToken.resize(arg->channelCount);
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        if (arg->channelRanks[channel] >= arg->rankSize ||
            arg->channelRanks[channel] == arg->rankId ||
            (channel != 0 && arg->channelRanks[channel - 1] >= arg->channelRanks[channel])) {
            return CcuResult::CCU_E_PARA;
        }
        ctx.peerInput[channel] =
            ccu::GetResByChannel<ccu::Variable>(arg->channels[channel], INPUT_XN_ID);
        ctx.peerOutput[channel] =
            ccu::GetResByChannel<ccu::Variable>(arg->channels[channel], OUTPUT_XN_ID);
        ctx.peerToken[channel] =
            ccu::GetResByChannel<ccu::Variable>(arg->channels[channel], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(KernelContext &ctx)
{
    CCU_CHK_RET(ccu::LoadArg(ctx.input, CCU_ARG_INPUT));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, CCU_ARG_OUTPUT));
    CCU_CHK_RET(ccu::LoadArg(ctx.token, CCU_ARG_TOKEN));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchSave, CCU_ARG_SCRATCH_SAVE));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchPartial, CCU_ARG_SCRATCH_PARTIAL));
    CCU_CHK_RET(ccu::LoadArg(ctx.tileOffset, CCU_ARG_TILE_OFFSET));
    CCU_CHK_RET(ccu::LoadArg(ctx.segmentSize, CCU_ARG_SEGMENT_SIZE));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSegmentSize, CCU_ARG_LAST_SEGMENT_SIZE));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchStride, CCU_ARG_SCRATCH_STRIDE));
    CCU_CHK_RET(ccu::LoadArg(ctx.ownerMainSize, CCU_ARG_OWNER_MAIN_SIZE));
    CCU_CHK_RET(ccu::LoadArg(ctx.ownerTailSize, CCU_ARG_OWNER_TAIL_SIZE));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOutputEqual, CCU_ARG_INPUT_OUTPUT_EQUAL));
    return CCU_SUCCESS;
}

CcuResult ExchangePartialAddress(KernelContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint16_t mask = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channel], ctx.input, INPUT_XN_ID, CKE_INDEX, 1U << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channel], ctx.token, TOKEN_XN_ID, CKE_INDEX, 1U << TOKEN_XN_ID));
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channel], CKE_INDEX, mask));
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeFinalizeAddress(KernelContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint16_t mask = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channel], ctx.output, OUTPUT_XN_ID, CKE_INDEX, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channel], ctx.token, TOKEN_XN_ID, CKE_INDEX, 1U << TOKEN_XN_ID));
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channel], CKE_INDEX, mask));
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeAllAddresses(KernelContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint16_t mask =
        (1U << INPUT_XN_ID) | (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channel], ctx.input, INPUT_XN_ID, CKE_INDEX, 1U << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channel], ctx.output, OUTPUT_XN_ID, CKE_INDEX, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channel], ctx.token, TOKEN_XN_ID, CKE_INDEX, 1U << TOKEN_XN_ID));
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channel], CKE_INDEX, mask));
    }
    return CCU_SUCCESS;
}

CcuResult GroupBarrier(KernelContext &ctx, uint32_t bit)
{
    const auto *arg = ctx.arg;
    const uint16_t mask = 1U << bit;
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channel], CKE_INDEX, mask));
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channel], CKE_INDEX, mask));
    }
    return CCU_SUCCESS;
}

ccu::LocalAddr LocalAddress(const ccu::Variable &base, const ccu::Variable &token,
    const ccu::Variable &offset)
{
    ccu::LocalAddr address;
    address.addr = base;
    address.addr += offset;
    address.token = token;
    return address;
}

ccu::RemoteAddr RemoteAddress(const ccu::Variable &base, const ccu::Variable &token,
    const ccu::Variable &offset)
{
    ccu::RemoteAddr address;
    address.addr = base;
    address.addr += offset;
    address.token = token;
    return address;
}

void SegmentOffset(KernelContext &ctx, uint32_t rank, ccu::Variable &offset)
{
    offset = ctx.tileOffset;
    for (uint32_t index = 0; index < rank; ++index) {
        offset += ctx.segmentSize;
    }
}

ccu::Variable &SegmentSize(KernelContext &ctx, uint32_t rank)
{
    return rank + 1 == ctx.arg->rankSize ? ctx.lastSegmentSize : ctx.segmentSize;
}

uint32_t FindChannel(const CcuKernelArgAllReduce *arg, uint32_t rank)
{
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        if (arg->channelRanks[channel] == rank) {
            return channel;
        }
    }
    return CCU_MAX_RANK_SIZE;
}

CcuResult ReduceP4ScratchFanIn(KernelContext &ctx,
    const uint32_t sources[P4_RANK_SIZE], const ccu::Variable &ownerOffset,
    ccu::Variable &ownerSize)
{
    constexpr uint32_t sourceCount = P4_RANK_SIZE;
    uint16_t readMask = 0;
    ccu::LocalAddr slots[sourceCount];

    CCU_IF(ownerSize != 0)
    {
        for (uint32_t index = 0; index < sourceCount; ++index) {
            ccu::Variable slotOffset;
            slotOffset = 0;
            for (uint32_t prior = 0; prior < index; ++prior) {
                slotOffset += ctx.lastSegmentSize;
            }
            slots[index] = LocalAddress(ctx.scratchSave, ctx.token, slotOffset);

            const uint16_t bit = static_cast<uint16_t>(1U << index);
            const uint32_t source = sources[index];
            if (source == ctx.arg->rankId) {
                ccu::LocalAddr localInput =
                    LocalAddress(ctx.input, ctx.token, ownerOffset);
                CCU_CHK_RET(ccu::LocalCopy(
                    slots[index], localInput, ownerSize, ctx.event, bit));
            } else {
                const uint32_t channel = FindChannel(ctx.arg, source);
                if (channel >= ctx.arg->channelCount) {
                    return CcuResult::CCU_E_PARA;
                }
                ccu::RemoteAddr remote = RemoteAddress(
                    ctx.peerInput[channel], ctx.peerToken[channel], ownerOffset);
                CCU_CHK_RET(ccu::Read(ctx.arg->channels[channel],
                    slots[index], remote, ownerSize, ctx.event, bit));
            }
            readMask = static_cast<uint16_t>(readMask | bit);
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, readMask));

        // Preserve B0's rank-relative FP32 order: local, then rank+1..rank+3.
        for (uint32_t index = 1; index < sourceCount; ++index) {
            CCU_CHK_RET(ccu::LocalReduce(slots[0], slots[index], ownerSize,
                ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReduceP16ScratchFanIn(KernelContext &ctx, const uint32_t *sources,
    uint32_t sourceCount, const ccu::Variable &ownerOffset,
    ccu::Variable &ownerSize)
{
    if (sourceCount < 2 || sourceCount > P16_MAX_GROUP_SOURCES) {
        return CcuResult::CCU_E_PARA;
    }
    uint16_t readMask = 0;
    ccu::LocalAddr slots[P16_MAX_GROUP_SOURCES];

    CCU_IF(ownerSize != 0)
    {
        ccu::Variable zero;
        zero = 0;
        ccu::LocalAddr accumulator = ctx.arg->reduceWithLocal ?
            LocalAddress(ctx.scratchSave, ctx.token, zero) :
            LocalAddress(ctx.scratchPartial, ctx.token, zero);
        for (uint32_t index = 0; index < sourceCount; ++index) {
            if (index == 0) {
                // Seed the fixed partial directly with the first source. This
                // preserves the left-fold order while removing a full-shard
                // LocalCopy and its completion wait after the fan-in reads.
                slots[index] = accumulator;
            } else {
                ccu::Variable slotOffset;
                slotOffset = 0;
                // Slots 0/1 are the fixed primary/secondary partials. Source
                // slots are globally keyed by rank so the two dies cannot
                // overlap.
                for (uint32_t slot = 0; slot < sources[index] + 2; ++slot) {
                    slotOffset += ctx.scratchStride;
                }
                slots[index] = LocalAddress(ctx.scratchSave, ctx.token, slotOffset);
            }

            const uint16_t bit = static_cast<uint16_t>(1U << index);
            const uint32_t source = sources[index];
            if (source == ctx.arg->rankId) {
                ccu::LocalAddr localInput =
                    LocalAddress(ctx.input, ctx.token, ownerOffset);
                CCU_CHK_RET(ccu::LocalCopy(
                    slots[index], localInput, ownerSize, ctx.event, bit));
            } else {
                const uint32_t channel = FindChannel(ctx.arg, source);
                if (channel >= ctx.arg->channelCount) {
                    return CcuResult::CCU_E_PARA;
                }
                ccu::RemoteAddr remote = RemoteAddress(
                    ctx.peerInput[channel], ctx.peerToken[channel], ownerOffset);
                CCU_CHK_RET(ccu::Read(ctx.arg->channels[channel],
                    slots[index], remote, ownerSize, ctx.event, bit));
            }
            readMask = static_cast<uint16_t>(readMask | bit);
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, readMask));

        for (uint32_t index = 1; index < sourceCount; ++index) {
            CCU_CHK_RET(ccu::LocalReduce(accumulator, slots[index], ownerSize,
                ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReduceP16DualHbmChains(KernelContext &ctx, const uint32_t *sources,
    uint32_t sourceCount, const ccu::Variable &ownerOffset,
    ccu::Variable &ownerSize)
{
    if (sourceCount != P16_DUAL_CHAIN_SOURCES) {
        return CcuResult::CCU_E_PARA;
    }
    uint16_t readMask = 0;
    ccu::LocalAddr slots[P16_DUAL_CHAIN_SOURCES];

    CCU_IF(ownerSize != 0)
    {
        ccu::Variable zero;
        zero = 0;
        const bool preseedFinalize = ctx.arg->useP16DualChain == 2 &&
            ctx.arg->reduceWithLocal != 0;
        ccu::LocalAddr accumulator0 = preseedFinalize ?
            LocalAddress(ctx.output, ctx.token, ownerOffset) :
            (ctx.arg->reduceWithLocal ?
                LocalAddress(ctx.scratchSave, ctx.token, zero) :
                LocalAddress(ctx.scratchPartial, ctx.token, zero));

        // source[0] is seeded directly into accumulator0, so its globally
        // rank-keyed source slot is dead and can hold the second accumulator.
        ccu::Variable accumulator1Offset;
        accumulator1Offset = 0;
        for (uint32_t slot = 0; slot < sources[0] + 2; ++slot) {
            accumulator1Offset += ctx.scratchStride;
        }
        ccu::LocalAddr accumulator1 =
            LocalAddress(ctx.scratchSave, ctx.token, accumulator1Offset);

        // Rotate only issue order. Source-indexed slots and the fixed 4+4
        // arithmetic tree below remain unchanged, so FP32 evaluation order is
        // identical to F1 while different owners start on different sources.
        const uint32_t issueStart = ctx.arg->rankId % sourceCount;
        for (uint32_t issue = 0; issue < sourceCount; ++issue) {
            const uint32_t index = (issueStart + issue) % sourceCount;
            if (index == 0) {
                slots[index] = accumulator0;
            } else if (index == P16_DUAL_CHAIN_WIDTH) {
                slots[index] = accumulator1;
            } else {
                ccu::Variable slotOffset;
                slotOffset = 0;
                for (uint32_t slot = 0; slot < sources[index] + 2; ++slot) {
                    slotOffset += ctx.scratchStride;
                }
                slots[index] = LocalAddress(ctx.scratchSave, ctx.token, slotOffset);
            }

            const uint16_t bit = static_cast<uint16_t>(1U << index);
            const uint32_t source = sources[index];
            if (source == ctx.arg->rankId) {
                ccu::LocalAddr localInput =
                    LocalAddress(ctx.input, ctx.token, ownerOffset);
                CCU_CHK_RET(ccu::LocalCopy(
                    slots[index], localInput, ownerSize, ctx.event, bit));
            } else {
                const uint32_t channel = FindChannel(ctx.arg, source);
                if (channel >= ctx.arg->channelCount) {
                    return CcuResult::CCU_E_PARA;
                }
                ccu::RemoteAddr remote = RemoteAddress(
                    ctx.peerInput[channel], ctx.peerToken[channel], ownerOffset);
                CCU_CHK_RET(ccu::Read(ctx.arg->channels[channel],
                    slots[index], remote, ownerSize, ctx.event, bit));
            }
            readMask = static_cast<uint16_t>(readMask | bit);
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, readMask));

        constexpr uint16_t chain0Bit = 1U;
        constexpr uint16_t chain1Bit = 2U;
        constexpr uint16_t chainMask = chain0Bit | chain1Bit;
        for (uint32_t step = 1; step < P16_DUAL_CHAIN_WIDTH; ++step) {
            CCU_CHK_RET(ccu::LocalReduce(accumulator0, slots[step], ownerSize,
                ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, chain0Bit));
            CCU_CHK_RET(ccu::LocalReduce(accumulator1,
                slots[P16_DUAL_CHAIN_WIDTH + step], ownerSize,
                ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, chain1Bit));
            CCU_CHK_RET(ccu::EventWait(ctx.event, chainMask));
        }
        CCU_CHK_RET(ccu::LocalReduce(accumulator0, accumulator1, ownerSize,
            ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, chain0Bit));
        CCU_CHK_RET(ccu::EventWait(ctx.event, chain0Bit));
        if (preseedFinalize) {
            ccu::LocalAddr sharedPrimary =
                LocalAddress(ctx.scratchSave, ctx.token, zero);
            CCU_CHK_RET(ccu::LocalCopy(sharedPrimary, accumulator0,
                ownerSize, ctx.event, chain0Bit));
            CCU_CHK_RET(ccu::EventWait(ctx.event, chain0Bit));
        }
    }
    return CCU_SUCCESS;
}

#if 0
uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
{
    return ((loopCtxId & 0xffULL) << 45) |
        ((gsaOffset & 0xffffffffULL) << 13) |
        (loopIterNum & 0x1fffULL);
}

uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    return ((repeatNum & 0x7fULL) << 55) |
        ((repeatLoopIndex & 0x7fULL) << 48) |
        ((totalLoopNum & 0x7fULL) << 41);
}

uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
{
    return ((gsaOffset & 0xffffffffULL) << 21) |
        ((msOffset & 0x7ffULL) << 10) |
        (ckeOffset & 0x3ffULL);
}

CcuResult RunP4Chunk(KernelContext &ctx, const uint32_t sources[P4_SOURCE_COUNT],
    const ccu::Variable &sourceOffset, const ccu::Variable &destinationOffset,
    const ccu::Variable &length, uint32_t lane)
{
    const uint32_t bufferBase = lane * P4_MS_INTERLEAVE;
    const ccu::Event laneEvent = ctx.p4Events[lane];
    for (uint32_t sourceIndex = 0; sourceIndex < P4_SOURCE_COUNT; ++sourceIndex) {
        const uint16_t mask = 1U << sourceIndex;
        const uint32_t source = sources[sourceIndex];
        if (source == ctx.arg->rankId) {
            ccu::LocalAddr local = LocalAddress(ctx.input, ctx.inputToken, sourceOffset);
            CCU_CHK_RET(ccu::LocalCopy(
                ctx.p4Buffers[bufferBase + sourceIndex], local, length, laneEvent, mask));
        } else {
            const uint32_t channel = FindChannel(ctx.arg, source);
            if (channel >= ctx.arg->channelCount) {
                return CcuResult::CCU_E_PARA;
            }
            ccu::RemoteAddr remote = RemoteAddress(
                ctx.peerInput[channel], ctx.peerInputToken[channel], sourceOffset);
            CCU_CHK_RET(ccu::Read(ctx.arg->channels[channel],
                ctx.p4Buffers[bufferBase + sourceIndex], remote, length, laneEvent, mask));
        }
    }
    CCU_CHK_RET(ccu::EventWait(laneEvent, P4_READ_MASK));
    CCU_CHK_RET(ccu::LocalReduce(&ctx.p4Buffers[bufferBase], P4_SOURCE_COUNT,
        ctx.arg->dataType, ctx.arg->dataType, ctx.arg->reduceOp,
        length, laneEvent, P4_REDUCE_MASK));
    CCU_CHK_RET(ccu::EventWait(laneEvent, P4_REDUCE_MASK));
    ccu::LocalAddr destination =
        LocalAddress(ctx.scratchSave, ctx.scratchToken, destinationOffset);
    CCU_CHK_RET(ccu::LocalCopy(destination, ctx.p4Buffers[bufferBase],
        length, laneEvent, P4_WRITE_MASK));
    CCU_CHK_RET(ccu::EventWait(laneEvent, P4_WRITE_MASK));
    return CCU_SUCCESS;
}

CcuResult ReduceP4Owner(KernelContext &ctx, const uint32_t sources[P4_SOURCE_COUNT],
    const ccu::Variable &ownerOffset)
{
    ctx.p4Events = ccu::Array<ccu::Event>(P4_LOOP_LANES);
    ctx.p4Buffers =
        ccu::Array<ccu::CcuBuffer>(P4_LOOP_LANES * P4_MS_INTERLEAVE);

    const ccu::Event loopEvent = ctx.p4Events[0];
    ctx.p4Loop.body.reset(new ccu::Func([&ctx, sources, loopEvent]() {
        for (uint32_t sourceIndex = 0; sourceIndex < P4_SOURCE_COUNT; ++sourceIndex) {
            const uint16_t mask = 1U << sourceIndex;
            const uint32_t source = sources[sourceIndex];
            if (source == ctx.arg->rankId) {
                ccu::LocalCopy(ctx.p4Buffers[sourceIndex],
                    ctx.p4Vars.localSrc[sourceIndex], ctx.p4Vars.len, loopEvent, mask);
            } else {
                const uint32_t channel = FindChannel(ctx.arg, source);
                ccu::Read(ctx.arg->channels[channel], ctx.p4Buffers[sourceIndex],
                    ctx.p4Vars.remoteSrc[sourceIndex], ctx.p4Vars.len, loopEvent, mask);
            }
        }
        ccu::EventWait(loopEvent, P4_READ_MASK);
        ccu::LocalReduce(&ctx.p4Buffers[0], P4_SOURCE_COUNT,
            ctx.arg->dataType, ctx.arg->dataType, ctx.arg->reduceOp,
            ctx.p4Vars.len, loopEvent, P4_REDUCE_MASK);
        ccu::EventWait(loopEvent, P4_REDUCE_MASK);
        ccu::LocalCopy(ctx.p4Vars.dst, ctx.p4Buffers[0],
            ctx.p4Vars.len, loopEvent, P4_WRITE_MASK);
        ccu::EventWait(loopEvent, P4_WRITE_MASK);
    }));
    ctx.p4Loop.loop.reset(new ccu::Loop(ctx.p4Loop.loopParam, *ctx.p4Loop.body));

    CCU_IF(ctx.p4LoopCount != 0)
    {
        for (uint32_t sourceIndex = 0; sourceIndex < P4_SOURCE_COUNT; ++sourceIndex) {
            const uint32_t source = sources[sourceIndex];
            if (source == ctx.arg->rankId) {
                ctx.p4Vars.localSrc[sourceIndex] =
                    LocalAddress(ctx.input, ctx.inputToken, ownerOffset);
            } else {
                const uint32_t channel = FindChannel(ctx.arg, source);
                if (channel >= ctx.arg->channelCount) {
                    return CcuResult::CCU_E_PARA;
                }
                ctx.p4Vars.remoteSrc[sourceIndex] = RemoteAddress(
                    ctx.peerInput[channel], ctx.peerInputToken[channel], ownerOffset);
            }
        }
        ccu::Variable zero;
        zero = 0;
        ctx.p4Vars.dst = LocalAddress(ctx.scratchSave, ctx.scratchToken, zero);
        ctx.p4Vars.len = P4_MICRO_TILE_BYTES;
        ctx.p4Loop.loopParam = GetLoopParam(
            0, P4_MICRO_TILE_BYTES * P4_LOOP_LANES, 0);
        ctx.p4Loop.loopParam += ctx.p4LoopCount;
        ccu::Variable parallel;
        parallel = GetParallelParam(P4_LOOP_LANES - 1, 0, 1);
        ccu::Variable offset;
        offset = GetOffsetParam(P4_MICRO_TILE_BYTES, P4_MS_INTERLEAVE, 1);
        std::vector<ccu::Loop> loops{*ctx.p4Loop.loop};
        ccu::LoopGroup group(parallel, offset, P4_LOOP_LANES, loops);
    }

    for (uint32_t lane = 0; lane < P4_LOOP_LANES; ++lane) {
        CCU_IF(ctx.p4EpilogueLen[lane] != 0)
        {
            ccu::Variable laneOffset;
            laneOffset = lane * P4_MICRO_TILE_BYTES;
            ccu::Variable sourceOffset;
            sourceOffset = ownerOffset + ctx.p4MainBytes;
            sourceOffset += laneOffset;
            ccu::Variable destinationOffset;
            destinationOffset = ctx.p4MainBytes;
            destinationOffset += laneOffset;
            CCU_CHK_RET(RunP4Chunk(
                ctx, sources, sourceOffset, destinationOffset,
                ctx.p4EpilogueLen[lane], lane));
        }
    }

    CCU_IF(ctx.p4TailBytes != 0)
    {
        ccu::Variable sourceOffset;
        sourceOffset = ownerOffset + ctx.p4TailOffset;
        CCU_CHK_RET(RunP4Chunk(
            ctx, sources, sourceOffset, ctx.p4TailOffset, ctx.p4TailBytes, 0));
    }
    return CCU_SUCCESS;
}

#endif

CcuResult ReduceRange(KernelContext &ctx, const uint32_t *sources, uint32_t sourceCount,
    const ccu::Variable &ownerOffset, const ccu::Variable &rangeOffset,
    ccu::Variable &rangeSize)
{
    const auto *arg = ctx.arg;
    ccu::Variable sourceOffset;
    sourceOffset = ownerOffset + rangeOffset;
    ccu::LocalAddr accumulator = arg->reduceWithLocal ?
        LocalAddress(ctx.scratchSave, ctx.token, rangeOffset) :
        LocalAddress(ctx.scratchPartial, ctx.token, rangeOffset);

    CCU_IF(rangeSize != 0)
    {
        for (uint32_t index = 0; index < sourceCount; ++index) {
            const uint32_t source = sources[index];
            if (source == arg->rankId) {
                ccu::LocalAddr localInput = LocalAddress(ctx.input, ctx.token, sourceOffset);
                if (index == 0) {
                    CCU_CHK_RET(ccu::LocalCopy(accumulator, localInput, rangeSize, ctx.event, 1));
                } else {
                    CCU_CHK_RET(ccu::LocalReduce(accumulator, localInput, rangeSize,
                        arg->dataType, arg->reduceOp, ctx.event, 1));
                }
            } else {
                const uint32_t actualChannel = FindChannel(arg, source);
                if (actualChannel >= arg->channelCount || arg->channelRanks[actualChannel] != source) {
                    return CcuResult::CCU_E_PARA;
                }
                ccu::RemoteAddr remote = RemoteAddress(
                    ctx.peerInput[actualChannel], ctx.peerToken[actualChannel], sourceOffset);
                if (index == 0) {
                    CCU_CHK_RET(ccu::Read(arg->channels[actualChannel], accumulator, remote,
                        rangeSize, ctx.event, 1));
                } else {
                    CCU_CHK_RET(ccu::ReadReduce(arg->channels[actualChannel], accumulator, remote,
                        rangeSize, arg->dataType, arg->reduceOp, ctx.event, 1));
                }
            }
            CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReducePartial(KernelContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t sources[CCU_MAX_RANK_SIZE];
    uint32_t sourceCount = 0;
    if (arg->rankSize == P4_RANK_SIZE && arg->kernelCount == 1 && arg->reduceWithLocal) {
        // P4 uses one balanced peer per round.  The old globally sorted order
        // made ranks 1/2/3 issue their first large Read against rank 0 at the
        // same time, creating a real-device hot spot that could time out even
        // though the VM graph was valid.
        sources[sourceCount++] = arg->rankId;
        for (uint32_t step = 1; step < arg->rankSize; ++step) {
            sources[sourceCount++] = (arg->rankId + step) % arg->rankSize;
        }
    } else {
        bool localInserted = false;
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            if (arg->ownsLocalSource && !localInserted &&
                arg->rankId < arg->channelRanks[channel]) {
                sources[sourceCount++] = arg->rankId;
                localInserted = true;
            }
            sources[sourceCount++] = arg->channelRanks[channel];
        }
        if (arg->ownsLocalSource && !localInserted) {
            sources[sourceCount++] = arg->rankId;
        }
    }

    ccu::Variable ownerOffset;
    SegmentOffset(ctx, arg->rankId, ownerOffset);
    ccu::Variable &ownerSize = SegmentSize(ctx, arg->rankId);
    ccu::Variable zero;
    zero = 0;
    if (arg->rankSize == P16_RANK_SIZE && arg->kernelCount == 2) {
        if (arg->useP16DualChain != 0) {
            CCU_CHK_RET(ReduceP16DualHbmChains(
                ctx, sources, sourceCount, ownerOffset, ownerSize));
        } else {
            CCU_CHK_RET(ReduceP16ScratchFanIn(
                ctx, sources, sourceCount, ownerOffset, ownerSize));
        }
    } else if (arg->rankSize == P4_RANK_SIZE && arg->kernelCount == 1) {
        CCU_CHK_RET(ReduceP4ScratchFanIn(ctx, sources, ownerOffset, ownerSize));
    } else {
        CCU_CHK_RET(ReduceRange(ctx, sources, sourceCount, ownerOffset, zero, ownerSize));
    }
    return CCU_SUCCESS;
}

CcuResult MergePartials(KernelContext &ctx)
{
    if (ctx.arg->kernelCount == 2 && !ctx.arg->reduceWithLocal) {
        return CCU_SUCCESS;
    }
    ccu::Variable zero;
    zero = 0;
    ccu::Variable ownerOffset;
    SegmentOffset(ctx, ctx.arg->rankId, ownerOffset);
    ccu::Variable &ownerSize = SegmentSize(ctx, ctx.arg->rankId);

    CCU_IF(ownerSize != 0)
    {
        ccu::LocalAddr destination = LocalAddress(ctx.output, ctx.token, ownerOffset);
        if ((ctx.arg->rankSize == P4_RANK_SIZE && ctx.arg->kernelCount == 1) ||
            (ctx.arg->rankSize == P16_RANK_SIZE && ctx.arg->kernelCount == 2)) {
            ccu::LocalAddr primary = LocalAddress(ctx.scratchSave, ctx.token, zero);
            CCU_CHK_RET(ccu::LocalCopy(destination, primary, ownerSize, ctx.event, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
        } else {
            CCU_IF(ctx.inputOutputEqual != 0)
            {
                ccu::LocalAddr primary = LocalAddress(ctx.scratchSave, ctx.token, zero);
                CCU_CHK_RET(ccu::LocalCopy(destination, primary, ownerSize, ctx.event, 1));
                CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
            }
        }

        if (ctx.arg->kernelCount == 2) {
            ccu::LocalAddr secondary = LocalAddress(ctx.scratchPartial, ctx.token, zero);
            CCU_CHK_RET(ccu::LocalReduce(destination, secondary, ownerSize,
                ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
        }
    }
    return CCU_SUCCESS;
}

CcuResult MergeP16Partials(KernelContext &ctx)
{
    ccu::Variable zero;
    zero = 0;
    ccu::Variable ownerOffset;
    SegmentOffset(ctx, ctx.arg->rankId, ownerOffset);
    ccu::Variable &ownerSize = SegmentSize(ctx, ctx.arg->rankId);
    CCU_IF(ownerSize != 0)
    {
        ccu::Variable mergedOffset;
        mergedOffset = ctx.scratchStride;
        mergedOffset += ctx.scratchStride;
        ccu::LocalAddr primary = LocalAddress(ctx.scratchSave, ctx.token, zero);
        ccu::LocalAddr secondary = LocalAddress(ctx.scratchPartial, ctx.token, zero);
        const bool preseedFinalize = ctx.arg->useP16DualChain == 2;
        ccu::LocalAddr destination = ctx.arg->reduceWithLocal ?
            LocalAddress(ctx.output, ctx.token, ownerOffset) :
            LocalAddress(ctx.scratchSave, ctx.token,
                preseedFinalize ? zero : mergedOffset);
        // Both finalizers use the same fixed FP32 left-fold order.  Slots 0/1
        // remain immutable; secondary reuses dead source slot 2 as its output.
        if (!preseedFinalize) {
            CCU_CHK_RET(ccu::LocalCopy(destination, primary,
                ownerSize, ctx.event, 1U));
            CCU_CHK_RET(ccu::EventWait(ctx.event, 1U));
        }
        CCU_CHK_RET(ccu::LocalReduce(destination, secondary, ownerSize,
            ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1U));
        CCU_CHK_RET(ccu::EventWait(ctx.event, 1U));
    }
    return CCU_SUCCESS;
}

CcuResult GatherSegments(KernelContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::Variable ownerOffset;
    SegmentOffset(ctx, arg->rankId, ownerOffset);
    ccu::Variable &ownerSize = SegmentSize(ctx, arg->rankId);
    CCU_IF(ownerSize != 0)
    {
        ccu::LocalAddr source = LocalAddress(ctx.output, ctx.token, ownerOffset);
        if (arg->rankSize == 4 && arg->kernelCount == 1) {
            // Preserve the balanced round-robin issue order while allowing the
            // three independent channels to progress concurrently.
            uint16_t completionMask = 0;
            for (uint32_t step = 1; step < arg->rankSize; ++step) {
                const uint32_t peer = (arg->rankId + step) % arg->rankSize;
                const uint32_t channel = FindChannel(arg, peer);
                if (channel >= arg->channelCount) {
                    return CcuResult::CCU_E_PARA;
                }
                ccu::RemoteAddr destination = RemoteAddress(
                    ctx.peerOutput[channel], ctx.peerToken[channel], ownerOffset);
                const uint16_t stepMask = static_cast<uint16_t>(1U << (step - 1));
                CCU_CHK_RET(ccu::Write(
                    arg->channels[channel], destination, source, ownerSize, ctx.event, stepMask));
                completionMask = static_cast<uint16_t>(completionMask | stepMask);
            }
            CCU_CHK_RET(ccu::EventWait(ctx.event, completionMask));
        } else {
            uint16_t completionMask = 0;
            for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
                const uint16_t channelMask = static_cast<uint16_t>(1U << channel);
                ccu::RemoteAddr destination = RemoteAddress(
                    ctx.peerOutput[channel], ctx.peerToken[channel], ownerOffset);
                CCU_CHK_RET(ccu::Write(
                    arg->channels[channel], destination, source, ownerSize, ctx.event, channelMask));
                completionMask = static_cast<uint16_t>(completionMask | channelMask);
            }
            CCU_CHK_RET(ccu::EventWait(ctx.event, completionMask));
        }
    }
    return CCU_SUCCESS;
}

CcuResult GatherP16Segments(KernelContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::Variable zero;
    zero = 0;
    ccu::Variable ownerOffset;
    SegmentOffset(ctx, arg->rankId, ownerOffset);
    ccu::Variable &ownerSize = SegmentSize(ctx, arg->rankId);
    CCU_IF(ownerSize != 0)
    {
        ccu::Variable mergedOffset;
        mergedOffset = ctx.scratchStride;
        mergedOffset += ctx.scratchStride;
        ccu::LocalAddr source = arg->reduceWithLocal ?
            LocalAddress(ctx.output, ctx.token, ownerOffset) :
            LocalAddress(ctx.scratchSave, ctx.token,
                arg->useP16DualChain == 2 ? zero : mergedOffset);
        uint16_t completionMask = 0;
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            const uint16_t channelMask = static_cast<uint16_t>(1U << channel);
            ccu::RemoteAddr destination = RemoteAddress(
                ctx.peerOutput[channel], ctx.peerToken[channel], ownerOffset);
            CCU_CHK_RET(ccu::Write(
                arg->channels[channel], destination, source, ownerSize, ctx.event, channelMask));
            completionMask = static_cast<uint16_t>(completionMask | channelMask);
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, completionMask));
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuKernelPartial(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    KernelContext ctx{};
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(ExchangePartialAddress(ctx));
    CCU_CHK_RET(ReducePartial(ctx));
    CCU_CHK_RET(GroupBarrier(ctx, PARTIAL_DONE_BIT));
    return CCU_SUCCESS;
}

CcuResult CcuKernelFinalize(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    KernelContext ctx{};
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(ExchangeFinalizeAddress(ctx));
    if (kernelArg->rankSize == P16_RANK_SIZE && kernelArg->kernelCount == 2) {
        CCU_CHK_RET(MergeP16Partials(ctx));
        CCU_CHK_RET(GroupBarrier(ctx, OWNER_DONE_BIT));
        CCU_CHK_RET(GatherP16Segments(ctx));
        return CCU_SUCCESS;
    }
    CCU_CHK_RET(MergePartials(ctx));
    CCU_CHK_RET(GroupBarrier(ctx, OWNER_DONE_BIT));
    CCU_CHK_RET(GatherSegments(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuKernelPartialCombinedExchange(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if ((kernelArg->rankSize != 12 && kernelArg->rankSize != P16_RANK_SIZE) ||
        kernelArg->kernelCount != 2) {
        return CcuResult::CCU_E_PARA;
    }
    KernelContext ctx{};
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(ExchangeAllAddresses(ctx));
    CCU_CHK_RET(ReducePartial(ctx));
    CCU_CHK_RET(GroupBarrier(ctx, PARTIAL_DONE_BIT));
    return CCU_SUCCESS;
}

CcuResult CcuKernelFinalizeReuseExchange(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if ((kernelArg->rankSize != 12 && kernelArg->rankSize != P16_RANK_SIZE) ||
        kernelArg->kernelCount != 2) {
        return CcuResult::CCU_E_PARA;
    }
    KernelContext ctx{};
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    if (kernelArg->rankSize == P16_RANK_SIZE) {
        CCU_CHK_RET(MergeP16Partials(ctx));
        CCU_CHK_RET(GroupBarrier(ctx, OWNER_DONE_BIT));
        CCU_CHK_RET(GatherP16Segments(ctx));
        return CCU_SUCCESS;
    }
    CCU_CHK_RET(MergePartials(ctx));
    CCU_CHK_RET(GroupBarrier(ctx, OWNER_DONE_BIT));
    CCU_CHK_RET(GatherSegments(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuKernelP4Fused(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if (kernelArg->rankSize != 4 || kernelArg->kernelCount != 1) {
        return CcuResult::CCU_E_PARA;
    }
    KernelContext ctx{};
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(ExchangePartialAddress(ctx));
    CCU_CHK_RET(ReducePartial(ctx));
    CCU_CHK_RET(GroupBarrier(ctx, PARTIAL_DONE_BIT));
    CCU_CHK_RET(ExchangeFinalizeAddress(ctx));
    CCU_CHK_RET(MergePartials(ctx));
    CCU_CHK_RET(GroupBarrier(ctx, OWNER_DONE_BIT));
    CCU_CHK_RET(GatherSegments(ctx));
    return CCU_SUCCESS;
}

} // namespace ops_hccl
