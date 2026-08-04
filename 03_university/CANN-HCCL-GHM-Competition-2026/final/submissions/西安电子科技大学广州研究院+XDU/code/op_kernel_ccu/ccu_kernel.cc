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
#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"

#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call) \
    do { \
        const CcuResult ccuResult = (call); \
        if (ccuResult != CCU_SUCCESS) { \
            return ccuResult; \
        } \
    } while (0)
#endif

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

namespace {
constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t TOKEN_XN_ID = 1;
constexpr uint32_t PRE_NOTIFY_IDX = 0;
constexpr uint32_t POST_NOTIFY_IDX = 1;
constexpr uint16_t INPUT_READY_MASK = 1U << 0;
constexpr uint16_t TOKEN_READY_MASK = 1U << 1;
constexpr uint16_t POST_READY_MASK = 1U << 0;
constexpr uint32_t TWO_BY_EIGHT_MESH_PEERS = 7;
constexpr uint32_t TWO_BY_EIGHT_MESH_WINDOW = 4;
constexpr uint16_t TWO_BY_EIGHT_MESH_WINDOW_MASK =
    (1U << TWO_BY_EIGHT_MESH_WINDOW) - 1U;
constexpr uint32_t V23_LOCAL_SIZE = 8;
constexpr uint32_t V23_STRIPE_COUNT = 7;
constexpr uint32_t V23_DIRECT_ROUNDS = 5;
constexpr uint16_t V23_STRIPE_MASK = (1U << V23_STRIPE_COUNT) - 1U;

// Relative source ids for the five perfect matchings of the 7x8 direct
// source graph. Adding the output owner's local id modulo eight gives the
// remote source id. Every row uses seven different Channels and every column
// covers exactly the five sources not assigned to that stripe's helper.
constexpr uint32_t V23_DIRECT_SOURCE[V23_DIRECT_ROUNDS][V23_STRIPE_COUNT] = {
    {3, 7, 0, 6, 5, 2, 4},
    {4, 6, 3, 5, 0, 7, 1},
    {5, 0, 4, 1, 7, 6, 2},
    {6, 2, 7, 4, 3, 1, 0},
    {7, 5, 1, 2, 6, 0, 3},
};

struct SourcePlan {
    bool isSelf;
    uint32_t peerIndex;
};

struct ReduceLayerContext {
    const CcuKernelArgReduceLayer *arg = nullptr;

    ccu::Variable mode;
    ccu::Variable input;
    ccu::Variable inputToken;
    ccu::Variable output;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable sliceOffset;
    ccu::Variable sliceSize;
    ccu::Variable segmentBytes;
    ccu::Variable lastSegmentBytes;
    ccu::Variable initializeOutput;

    std::vector<ccu::Variable> peerInput;
    std::vector<ccu::Variable> peerToken;
    ccu::Event event;
};

struct V23ReduceLayerContext : public ReduceLayerContext {
    ccu::Variable regularStripeBytes;
    ccu::Variable finalStripeBytes;
    ccu::Variable remoteGroupOffset;
};

static std::vector<SourcePlan> BuildSourcePlan(const CcuKernelArgReduceLayer &arg)
{
    std::vector<std::pair<uint32_t, SourcePlan>> ordered;
    ordered.reserve(arg.peerCount + arg.includeSelf);
    if (arg.includeSelf != 0U) {
        ordered.push_back({arg.rankId, SourcePlan{true, 0}});
    }
    for (uint32_t peerIndex = 0; peerIndex < arg.peerCount; ++peerIndex) {
        ordered.push_back({arg.peerRanks[peerIndex], SourcePlan{false, peerIndex}});
    }
    std::sort(ordered.begin(), ordered.end(),
        [](const std::pair<uint32_t, SourcePlan> &lhs,
            const std::pair<uint32_t, SourcePlan> &rhs) {
            return lhs.first < rhs.first;
        });

    std::vector<SourcePlan> result;
    result.reserve(ordered.size());
    for (const auto &entry : ordered) {
        result.push_back(entry.second);
    }
    // Rotate the deterministic rank order.  In the sequential small-message
    // path this makes every step a permutation instead of making all ranks
    // read the same source at once.
    if (!result.empty()) {
        const uint32_t rotation =
            arg.rankId % static_cast<uint32_t>(result.size());
        std::rotate(result.begin(), result.begin() + rotation, result.end());
    }
    return result;
}

static ccu::LocalAddr MakeLocal(ccu::Variable address, ccu::Variable token)
{
    ccu::LocalAddr result;
    result.addr = address;
    result.token = token;
    return result;
}

static ccu::RemoteAddr MakeRemote(ccu::Variable address, ccu::Variable token)
{
    ccu::RemoteAddr result;
    result.addr = address;
    result.token = token;
    return result;
}

static CcuResult InitializeSegment(ReduceLayerContext &ctx,
    const SourcePlan &source, ccu::LocalAddr destination,
    ccu::Variable offset, ccu::Variable length, uint16_t mask)
{
    if (source.isSelf) {
        ccu::LocalAddr localSource = MakeLocal(ctx.input, ctx.inputToken);
        localSource.addr += ctx.sliceOffset;
        localSource.addr += offset;
        CCU_CHK_RET(ccu::LocalCopy(
            destination, localSource, length, ctx.event, mask));
    } else {
        ccu::RemoteAddr remoteSource =
            MakeRemote(ctx.peerInput[source.peerIndex], ctx.peerToken[source.peerIndex]);
        remoteSource.addr += ctx.sliceOffset;
        remoteSource.addr += offset;
        CCU_CHK_RET(ccu::Read(ctx.arg->channels[source.peerIndex],
            destination, remoteSource, length, ctx.event, mask));
    }
    return CCU_SUCCESS;
}

static CcuResult ReduceSegment(ReduceLayerContext &ctx,
    const SourcePlan &source, ccu::LocalAddr destination,
    ccu::Variable offset, ccu::Variable length, uint16_t mask)
{
    if (source.isSelf) {
        ccu::LocalAddr localSource = MakeLocal(ctx.input, ctx.inputToken);
        localSource.addr += ctx.sliceOffset;
        localSource.addr += offset;
        CCU_CHK_RET(ccu::LocalReduce(destination, localSource, length,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, mask));
    } else {
        ccu::RemoteAddr remoteSource =
            MakeRemote(ctx.peerInput[source.peerIndex], ctx.peerToken[source.peerIndex]);
        remoteSource.addr += ctx.sliceOffset;
        remoteSource.addr += offset;
        CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[source.peerIndex],
            destination, remoteSource, length, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event, mask));
    }
    return CCU_SUCCESS;
}

/**
 * Deterministic small-message fallback. It is selected only when the output
 * contains fewer FP32 elements than sources, so a non-empty stripe per source
 * cannot be formed.
 */
static CcuResult ReduceSequential(ReduceLayerContext &ctx,
    const std::vector<SourcePlan> &sources)
{
    ccu::LocalAddr destination = MakeLocal(ctx.output, ctx.outputToken);
    ccu::Variable zeroOffset;
    zeroOffset = 0;

    CCU_IF(ctx.initializeOutput != 0)
    {
        CCU_CHK_RET(InitializeSegment(
            ctx, sources[0], destination, zeroOffset, ctx.sliceSize, 1));
        ccu::EventWait(ctx.event, 1);
        for (uint32_t sourceIndex = 1; sourceIndex < sources.size(); ++sourceIndex) {
            CCU_CHK_RET(ReduceSegment(
                ctx, sources[sourceIndex], destination, zeroOffset, ctx.sliceSize, 1));
            ccu::EventWait(ctx.event, 1);
        }
    }
    CCU_IF(ctx.initializeOutput == 0)
    {
        for (uint32_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
            CCU_CHK_RET(ReduceSegment(
                ctx, sources[sourceIndex], destination, zeroOffset, ctx.sliceSize, 1));
            ccu::EventWait(ctx.event, 1);
        }
    }
    return CCU_SUCCESS;
}

/**
 * MESH stripe schedule:
 *   round 0 initializes every destination stripe from one distinct source;
 *   subsequent rounds rotate sources over stripes and reduce in place.
 *
 * Every round has exactly one writer per stripe. Rounds are separated by an
 * event wait, making the FP32 accumulation order fixed, while all sources in a
 * round use their links concurrently.
 */
static CcuResult ReduceStriped(ReduceLayerContext &ctx,
    const std::vector<SourcePlan> &sources)
{
    const uint32_t sourceCount = static_cast<uint32_t>(sources.size());
    const uint16_t allSourceMask =
        static_cast<uint16_t>((1U << sourceCount) - 1U);

    std::vector<ccu::Variable> segmentOffsets(sourceCount);
    std::vector<ccu::Variable> segmentLengths(sourceCount);
    ccu::Variable nextOffset;
    nextOffset = 0;
    for (uint32_t segment = 0; segment < sourceCount; ++segment) {
        segmentOffsets[segment] = nextOffset;
        segmentLengths[segment] =
            (segment + 1U == sourceCount) ? ctx.lastSegmentBytes : ctx.segmentBytes;
        nextOffset += ctx.segmentBytes;
    }

    for (uint32_t round = 0; round < sourceCount; ++round) {
        for (uint32_t sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
            const uint32_t destinationIndex =
                (sourceIndex + round) % sourceCount;
            ccu::LocalAddr destination = MakeLocal(ctx.output, ctx.outputToken);
            destination.addr += segmentOffsets[destinationIndex];
            const uint16_t mask = static_cast<uint16_t>(1U << sourceIndex);

            if (round == 0) {
                CCU_IF(ctx.initializeOutput != 0)
                {
                    CCU_CHK_RET(InitializeSegment(ctx, sources[sourceIndex],
                        destination, segmentOffsets[destinationIndex],
                        segmentLengths[destinationIndex], mask));
                }
                CCU_IF(ctx.initializeOutput == 0)
                {
                    CCU_CHK_RET(ReduceSegment(ctx, sources[sourceIndex],
                        destination, segmentOffsets[destinationIndex],
                        segmentLengths[destinationIndex], mask));
                }
            } else {
                CCU_CHK_RET(ReduceSegment(ctx, sources[sourceIndex],
                    destination, segmentOffsets[destinationIndex],
                    segmentLengths[destinationIndex], mask));
            }
        }
        ccu::EventWait(ctx.event, allSourceMask);
    }
    return CCU_SUCCESS;
}

static CcuResult ExchangeAddressSet(ReduceLayerContext &ctx,
    const uint32_t *peerIndices, uint32_t peerCount,
    ccu::Variable address, ccu::Variable token)
{
    for (uint32_t index = 0; index < peerCount; ++index) {
        const uint32_t peerIndex = peerIndices[index];
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
            address, INPUT_XN_ID, PRE_NOTIFY_IDX, INPUT_READY_MASK));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
            token, TOKEN_XN_ID, PRE_NOTIFY_IDX, TOKEN_READY_MASK));
    }
    for (uint32_t index = 0; index < peerCount; ++index) {
        const uint32_t peerIndex = peerIndices[index];
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[peerIndex], PRE_NOTIFY_IDX,
            INPUT_READY_MASK | TOKEN_READY_MASK));
    }
    return CCU_SUCCESS;
}

static CcuResult BarrierAddressSet(ReduceLayerContext &ctx,
    const uint32_t *peerIndices, uint32_t peerCount)
{
    for (uint32_t index = 0; index < peerCount; ++index) {
        const uint32_t peerIndex = peerIndices[index];
        CCU_CHK_RET(ccu::NotifyRecord(
            ctx.arg->channels[peerIndex], POST_NOTIFY_IDX, POST_READY_MASK));
    }
    for (uint32_t index = 0; index < peerCount; ++index) {
        const uint32_t peerIndex = peerIndices[index];
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[peerIndex], POST_NOTIFY_IDX, POST_READY_MASK));
    }
    return CCU_SUCCESS;
}

/**
 * One-round deterministic fan-in for the 512 KiB paths.
 *
 * Network operations never write the same destination concurrently.  Every
 * source first lands in a private CCL slot, so all links in this layer can be
 * active in one transfer round.  A fixed binary tree then reduces the slots
 * in place.  The tree is deterministic and its intermediate destinations are
 * disjoint within each round.
 *
 * The host passes a separate scratch range to each layer.  Thus a Mesh
 * kernel and a Clos kernel may run on their different IO-die CCUs at the
 * same time without a memory conflict.  For the layer-0 kernel,
 * initializeOutput is non-zero and the root is copied to the user output.
 * For layer-1, the root remains at scratch so the host finalize kernel can
 * add it to the layer-0 result.
 */
static CcuResult ReduceSmallFanin(ReduceLayerContext &ctx,
    const std::vector<SourcePlan> &sources)
{
    const uint32_t sourceCount = static_cast<uint32_t>(sources.size());
    if (sourceCount == 0U || sourceCount > MAX_LAYER_SOURCES) {
        return CCU_E_PARA;
    }
    // The kernel that includes the local rank initializes the deterministic
    // tree. Rooting slot zero directly in output preserves the exact source
    // and FP32 tree order while removing the final full-length LocalCopy.
    const bool directRootOutput = ctx.arg->includeSelf != 0U;

    std::vector<ccu::Variable> slotOffsets(sourceCount);
    ccu::Variable nextOffset;
    nextOffset = 0;
    for (uint32_t sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
        slotOffsets[sourceIndex] = nextOffset;
        nextOffset += ctx.sliceSize;
    }

    // Publish the input address/token and wait for all peers before issuing
    // the one-round fan-in.  The generic kernel uses channel indices
    // directly; peerRanks contains global rank ids and cannot be passed to
    // ExchangeAddressSet.
    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
            ctx.input, INPUT_XN_ID, PRE_NOTIFY_IDX, INPUT_READY_MASK));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
            ctx.inputToken, TOKEN_XN_ID, PRE_NOTIFY_IDX, TOKEN_READY_MASK));
    }
    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[peerIndex],
            PRE_NOTIFY_IDX, INPUT_READY_MASK | TOKEN_READY_MASK));
    }

    for (uint32_t sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
        ccu::LocalAddr destination;
        if (directRootOutput && sourceIndex == 0U) {
            destination = MakeLocal(ctx.output, ctx.outputToken);
        } else {
            destination = MakeLocal(ctx.scratch, ctx.scratchToken);
            destination.addr += slotOffsets[sourceIndex];
        }
        const uint16_t mask = static_cast<uint16_t>(1U << sourceIndex);
        const SourcePlan &source = sources[sourceIndex];
        if (source.isSelf) {
            ccu::LocalAddr localSource = MakeLocal(ctx.input, ctx.inputToken);
            localSource.addr += ctx.sliceOffset;
            CCU_CHK_RET(ccu::LocalCopy(
                destination, localSource, ctx.sliceSize, ctx.event, mask));
        } else {
            ccu::RemoteAddr remoteSource =
                MakeRemote(ctx.peerInput[source.peerIndex],
                    ctx.peerToken[source.peerIndex]);
            remoteSource.addr += ctx.sliceOffset;
            CCU_CHK_RET(ccu::Read(ctx.arg->channels[source.peerIndex],
                destination, remoteSource, ctx.sliceSize, ctx.event, mask));
        }
    }
    const uint16_t sourceMask =
        static_cast<uint16_t>((1U << sourceCount) - 1U);
    ccu::EventWait(ctx.event, sourceMask);

    // The data reads are complete.  Notify peers now and overlap their
    // post-notify wait with our local deterministic reduction tree.
    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyRecord(
            ctx.arg->channels[peerIndex], POST_NOTIFY_IDX, POST_READY_MASK));
    }

    for (uint32_t stride = 1U; stride < sourceCount; stride <<= 1U) {
        uint16_t roundMask = 0;
        uint32_t pairIndex = 0;
        for (uint32_t destinationIndex = 0;
            destinationIndex + stride < sourceCount;
            destinationIndex += stride << 1U) {
            ccu::LocalAddr destination;
            if (directRootOutput && destinationIndex == 0U) {
                destination = MakeLocal(ctx.output, ctx.outputToken);
            } else {
                destination = MakeLocal(ctx.scratch, ctx.scratchToken);
                destination.addr += slotOffsets[destinationIndex];
            }
            ccu::LocalAddr source =
                MakeLocal(ctx.scratch, ctx.scratchToken);
            source.addr += slotOffsets[destinationIndex + stride];
            const uint16_t mask = static_cast<uint16_t>(1U << pairIndex++);
            roundMask = static_cast<uint16_t>(roundMask | mask);
            CCU_CHK_RET(ccu::LocalReduce(destination, source, ctx.sliceSize,
                HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, mask));
        }
        if (roundMask != 0) {
            ccu::EventWait(ctx.event, roundMask);
        }
    }

    if (!directRootOutput) {
        CCU_IF(ctx.initializeOutput != 0)
        {
            ccu::LocalAddr destination = MakeLocal(ctx.output, ctx.outputToken);
            ccu::LocalAddr source = MakeLocal(ctx.scratch, ctx.scratchToken);
            CCU_CHK_RET(ccu::LocalCopy(
                destination, source, ctx.sliceSize, ctx.event, 1));
            ccu::EventWait(ctx.event, 1);
        }
    }

    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[peerIndex], POST_NOTIFY_IDX, POST_READY_MASK));
    }
    return CCU_SUCCESS;
}

/**
 * Two-round grouped partial fan-in for large two-layer messages.
 *
 * Sources are divided into deterministic pairs. In each of two rounds, the
 * two sources in every pair target different halves of that pair's private
 * partial. The second round swaps the halves, so both sources contribute to
 * the complete partial while all groups remain disjoint. A fixed binary tree
 * then merges the group partials locally.
 */
static CcuResult ReduceLargeGroupedV10(ReduceLayerContext &ctx,
    const std::vector<SourcePlan> &sources, bool earlyEightSourceSubtrees)
{
    const uint32_t sourceCount = static_cast<uint32_t>(sources.size());
    if (sourceCount != 4U && sourceCount != 8U) {
        return CCU_E_PARA;
    }
    const uint32_t groupCount = sourceCount / 2U;
    const uint16_t sourceMask =
        static_cast<uint16_t>((1U << sourceCount) - 1U);

    std::vector<ccu::Variable> partialOffsets(groupCount);
    ccu::Variable nextPartialOffset;
    nextPartialOffset = 0;
    for (uint32_t group = 1U; group < groupCount; ++group) {
        partialOffsets[group] = nextPartialOffset;
        nextPartialOffset += ctx.segmentBytes;
    }

    std::vector<ccu::Variable> stripeOffsets(2U);
    std::vector<ccu::Variable> stripeLengths(2U);
    stripeOffsets[0] = 0;
    stripeOffsets[1] = ctx.lastSegmentBytes;
    stripeLengths[0] = ctx.lastSegmentBytes;
    stripeLengths[1] = ctx.initializeOutput;

    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
            ctx.input, INPUT_XN_ID, PRE_NOTIFY_IDX, INPUT_READY_MASK));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
            ctx.inputToken, TOKEN_XN_ID, PRE_NOTIFY_IDX, TOKEN_READY_MASK));
    }
    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[peerIndex],
            PRE_NOTIFY_IDX, INPUT_READY_MASK | TOKEN_READY_MASK));
    }

    for (uint32_t round = 0; round < 2U; ++round) {
        for (uint32_t group = 0; group < groupCount; ++group) {
            for (uint32_t member = 0; member < 2U; ++member) {
                const uint32_t sourceIndex = group * 2U + member;
                const uint32_t stripe = (member + round) % 2U;
                // Construct a fresh address object for every instruction.
                // ccu::Variable copies may share translator state, so copying
                // a pre-offset LocalAddr and mutating it can accidentally make
                // two operations target the same stripe.
                ccu::LocalAddr destination;
                if (group == 0U) {
                    destination = MakeLocal(ctx.output, ctx.outputToken);
                } else {
                    destination = MakeLocal(ctx.scratch, ctx.scratchToken);
                    destination.addr += partialOffsets[group];
                }
                destination.addr += stripeOffsets[stripe];
                const uint16_t mask =
                    static_cast<uint16_t>(1U << sourceIndex);
                if (round == 0U) {
                    CCU_CHK_RET(InitializeSegment(ctx, sources[sourceIndex],
                        destination, stripeOffsets[stripe],
                        stripeLengths[stripe], mask));
                } else {
                    CCU_CHK_RET(ReduceSegment(ctx, sources[sourceIndex],
                        destination, stripeOffsets[stripe],
                        stripeLengths[stripe], mask));
                }
            }
        }
        if (!earlyEightSourceSubtrees || round == 0U || groupCount != 4U) {
            ccu::EventWait(ctx.event, sourceMask);
        }
    }

    if (earlyEightSourceSubtrees && groupCount == 4U) {
        constexpr uint16_t FIRST_SUBTREE_MASK = 0x0FU;
        constexpr uint16_t SECOND_SUBTREE_MASK = 0xF0U;
        constexpr uint16_t FIRST_MERGE_MASK = 1U << 0;
        constexpr uint16_t SECOND_MERGE_MASK = 1U << 4;

        ccu::EventWait(ctx.event, FIRST_SUBTREE_MASK);
        for (uint32_t sourceIndex = 0; sourceIndex < 4U; ++sourceIndex) {
            if (!sources[sourceIndex].isSelf) {
                CCU_CHK_RET(ccu::NotifyRecord(
                    ctx.arg->channels[sources[sourceIndex].peerIndex],
                    POST_NOTIFY_IDX, POST_READY_MASK));
            }
        }
        ccu::LocalAddr firstDestination =
            MakeLocal(ctx.output, ctx.outputToken);
        ccu::LocalAddr firstSource = MakeLocal(ctx.scratch, ctx.scratchToken);
        firstSource.addr += partialOffsets[1U];
        CCU_CHK_RET(ccu::LocalReduce(firstDestination, firstSource,
            ctx.sliceSize, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
            ctx.event, FIRST_MERGE_MASK));

        ccu::EventWait(ctx.event, SECOND_SUBTREE_MASK);
        for (uint32_t sourceIndex = 4U; sourceIndex < 8U; ++sourceIndex) {
            if (!sources[sourceIndex].isSelf) {
                CCU_CHK_RET(ccu::NotifyRecord(
                    ctx.arg->channels[sources[sourceIndex].peerIndex],
                    POST_NOTIFY_IDX, POST_READY_MASK));
            }
        }
        ccu::LocalAddr secondDestination =
            MakeLocal(ctx.scratch, ctx.scratchToken);
        secondDestination.addr += partialOffsets[2U];
        ccu::LocalAddr secondSource =
            MakeLocal(ctx.scratch, ctx.scratchToken);
        secondSource.addr += partialOffsets[3U];
        CCU_CHK_RET(ccu::LocalReduce(secondDestination, secondSource,
            ctx.sliceSize, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
            ctx.event, SECOND_MERGE_MASK));

        ccu::EventWait(ctx.event, FIRST_MERGE_MASK | SECOND_MERGE_MASK);
        ccu::LocalAddr finalDestination =
            MakeLocal(ctx.output, ctx.outputToken);
        ccu::LocalAddr finalSource =
            MakeLocal(ctx.scratch, ctx.scratchToken);
        finalSource.addr += partialOffsets[2U];
        CCU_CHK_RET(ccu::LocalReduce(finalDestination, finalSource,
            ctx.sliceSize, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
            ctx.event, FIRST_MERGE_MASK));
        ccu::EventWait(ctx.event, FIRST_MERGE_MASK);
    } else {
        // Exact V10 schedule for P12/P18 and every non-selected shape.
        for (uint32_t peerIndex = 0;
            peerIndex < ctx.arg->peerCount; ++peerIndex) {
            CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[peerIndex],
                POST_NOTIFY_IDX, POST_READY_MASK));
        }
        for (uint32_t stride = 1U; stride < groupCount; stride <<= 1U) {
            uint16_t roundMask = 0;
            uint32_t pairIndex = 0;
            for (uint32_t destinationGroup = 0;
                destinationGroup + stride < groupCount;
                destinationGroup += stride << 1U) {
                ccu::LocalAddr destination;
                if (destinationGroup == 0U) {
                    destination = MakeLocal(ctx.output, ctx.outputToken);
                } else {
                    destination = MakeLocal(ctx.scratch, ctx.scratchToken);
                    destination.addr += partialOffsets[destinationGroup];
                }
                ccu::LocalAddr source = MakeLocal(ctx.scratch, ctx.scratchToken);
                source.addr += partialOffsets[destinationGroup + stride];
                const uint16_t mask = static_cast<uint16_t>(1U << pairIndex++);
                roundMask = static_cast<uint16_t>(roundMask | mask);
                CCU_CHK_RET(ccu::LocalReduce(destination, source, ctx.sliceSize,
                    HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, mask));
            }
            if (roundMask != 0) {
                ccu::EventWait(ctx.event, roundMask);
            }
        }
    }

    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[peerIndex], POST_NOTIFY_IDX, POST_READY_MASK));
    }
    return CCU_SUCCESS;
}

/**
 * V13 point-17 path. Four sources form one deterministic partial. INIT owns
 * one output colour and ACCUM adds the opposite layer after the host barrier.
 * This entry is registered only for the 8+4, 512 MiB performance case.
 */
static CcuResult ReduceLargeGroupedV13(ReduceLayerContext &ctx,
    const std::vector<SourcePlan> &sources)
{
    const uint32_t sourceCount = static_cast<uint32_t>(sources.size());
    if (sourceCount != 4U && sourceCount != 8U) {
        return CCU_E_PARA;
    }
    const uint32_t groupCount = sourceCount / 4U;
    const uint16_t sourceMask =
        static_cast<uint16_t>((1U << sourceCount) - 1U);

    std::vector<ccu::Variable> partialOffsets(groupCount);
    ccu::Variable nextPartialOffset;
    nextPartialOffset = 0;
    for (uint32_t group = 1U; group < groupCount; ++group) {
        partialOffsets[group] = nextPartialOffset;
        nextPartialOffset += ctx.segmentBytes;
    }

    std::vector<ccu::Variable> stripeOffsets(4U);
    std::vector<ccu::Variable> stripeLengths(4U);
    ccu::Variable nextStripeOffset;
    nextStripeOffset = 0;
    for (uint32_t stripe = 0; stripe < 4U; ++stripe) {
        stripeOffsets[stripe] = nextStripeOffset;
        if (stripe == 3U) {
            stripeLengths[stripe] = ctx.initializeOutput;
        } else {
            stripeLengths[stripe] = ctx.lastSegmentBytes;
        }
        nextStripeOffset += ctx.lastSegmentBytes;
    }

    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
            ctx.input, INPUT_XN_ID, PRE_NOTIFY_IDX, INPUT_READY_MASK));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
            ctx.inputToken, TOKEN_XN_ID, PRE_NOTIFY_IDX, TOKEN_READY_MASK));
    }
    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[peerIndex],
            PRE_NOTIFY_IDX, INPUT_READY_MASK | TOKEN_READY_MASK));
    }

    for (uint32_t round = 0; round < 4U; ++round) {
        for (uint32_t group = 0; group < groupCount; ++group) {
            for (uint32_t member = 0; member < 4U; ++member) {
                const uint32_t sourceIndex = group * 4U + member;
                const uint32_t stripe = (member + round) % 4U;
                ccu::LocalAddr destination;
                if (group == 0U) {
                    destination = MakeLocal(ctx.output, ctx.outputToken);
                } else {
                    destination = MakeLocal(ctx.scratch, ctx.scratchToken);
                    destination.addr += partialOffsets[group];
                }
                destination.addr += stripeOffsets[stripe];
                const uint16_t mask =
                    static_cast<uint16_t>(1U << sourceIndex);
                if (round == 0U && group == 0U) {
                    CCU_IF(ctx.mode == CCU_MODE_LARGE_GROUPED4_INIT)
                    {
                        CCU_CHK_RET(InitializeSegment(ctx, sources[sourceIndex],
                            destination, stripeOffsets[stripe],
                            stripeLengths[stripe], mask));
                    }
                    CCU_IF(ctx.mode == CCU_MODE_LARGE_GROUPED4_ACCUM)
                    {
                        CCU_CHK_RET(ReduceSegment(ctx, sources[sourceIndex],
                            destination, stripeOffsets[stripe],
                            stripeLengths[stripe], mask));
                    }
                } else if (round == 0U) {
                    CCU_CHK_RET(InitializeSegment(ctx, sources[sourceIndex],
                        destination, stripeOffsets[stripe],
                        stripeLengths[stripe], mask));
                } else {
                    CCU_CHK_RET(ReduceSegment(ctx, sources[sourceIndex],
                        destination, stripeOffsets[stripe],
                        stripeLengths[stripe], mask));
                }
            }
        }
        ccu::EventWait(ctx.event, sourceMask);
    }

    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyRecord(
            ctx.arg->channels[peerIndex], POST_NOTIFY_IDX, POST_READY_MASK));
    }

    for (uint32_t stride = 1U; stride < groupCount; stride <<= 1U) {
        uint16_t roundMask = 0;
        uint32_t pairIndex = 0;
        for (uint32_t destinationGroup = 0;
            destinationGroup + stride < groupCount;
            destinationGroup += stride << 1U) {
            ccu::LocalAddr destination;
            if (destinationGroup == 0U) {
                destination = MakeLocal(ctx.output, ctx.outputToken);
            } else {
                destination = MakeLocal(ctx.scratch, ctx.scratchToken);
                destination.addr += partialOffsets[destinationGroup];
            }
            ccu::LocalAddr source = MakeLocal(ctx.scratch, ctx.scratchToken);
            source.addr += partialOffsets[destinationGroup + stride];
            const uint16_t mask = static_cast<uint16_t>(1U << pairIndex++);
            roundMask = static_cast<uint16_t>(roundMask | mask);
            CCU_CHK_RET(ccu::LocalReduce(destination, source, ctx.sliceSize,
                HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, mask));
        }
        if (roundMask != 0) {
            ccu::EventWait(ctx.event, roundMask);
        }
    }

    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[peerIndex], POST_NOTIFY_IDX, POST_READY_MASK));
    }
    return CCU_SUCCESS;
}

static bool FindV23PeerIndex(const CcuKernelArgReduceLayer &arg,
    uint32_t localRank, uint32_t &peerIndex)
{
    for (uint32_t index = 0; index < arg.peerCount; ++index) {
        if (arg.peerRanks[index] % V23_LOCAL_SIZE == localRank) {
            peerIndex = index;
            return true;
        }
    }
    return false;
}

static void BuildV23StripeLayout(V23ReduceLayerContext &ctx,
    std::vector<ccu::Variable> &offsets,
    std::vector<ccu::Variable> &lengths)
{
    offsets.resize(V23_STRIPE_COUNT);
    lengths.resize(V23_STRIPE_COUNT);
    ccu::Variable nextOffset;
    nextOffset = 0;
    for (uint32_t stripe = 0; stripe < V23_STRIPE_COUNT; ++stripe) {
        offsets[stripe] = nextOffset;
        if (stripe + 1U == V23_STRIPE_COUNT) {
            lengths[stripe] = ctx.finalStripeBytes;
        } else {
            lengths[stripe] = ctx.regularStripeBytes;
        }
        nextOffset += ctx.regularStripeBytes;
    }
}

static CcuResult ExchangeV23Address(ReduceLayerContext &ctx,
    ccu::Variable address, ccu::Variable token)
{
    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
            address, INPUT_XN_ID, PRE_NOTIFY_IDX, INPUT_READY_MASK));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
            token, TOKEN_XN_ID, PRE_NOTIFY_IDX, TOKEN_READY_MASK));
    }
    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[peerIndex],
            PRE_NOTIFY_IDX, INPUT_READY_MASK | TOKEN_READY_MASK));
    }
    return CCU_SUCCESS;
}

static CcuResult FinishV23Channels(ReduceLayerContext &ctx)
{
    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[peerIndex],
            POST_NOTIFY_IDX, POST_READY_MASK));
    }
    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[peerIndex],
            POST_NOTIFY_IDX, POST_READY_MASK));
    }
    return CCU_SUCCESS;
}

/**
 * V23 layer-0 stage for 2x8.
 *
 * The first 19 data operations preserve V10's two-half local reduction for
 * this rank's own output. The remaining 21 operations produce seven helper
 * stripes for owners on the other server. Every helper contains exactly
 * three sources: this rank and two distinct local peers. Across the seven
 * helpers each directed Mesh Channel is used exactly twice, so the added 2O
 * traffic is balanced over all seven links.
 */
static CcuResult ReduceV23MeshHelper(V23ReduceLayerContext &ctx,
    const std::vector<SourcePlan> &sources)
{
    if (ctx.arg->layer != 0U || ctx.arg->includeSelf == 0U ||
        sources.size() != V23_LOCAL_SIZE || ctx.arg->peerCount != 7U) {
        return CCU_E_PARA;
    }
    CCU_CHK_RET(ExchangeV23Address(ctx, ctx.input, ctx.inputToken));

    constexpr uint32_t PAIR_COUNT = V23_LOCAL_SIZE / 2U;
    constexpr uint16_t SOURCE_MASK = (1U << V23_LOCAL_SIZE) - 1U;
    std::vector<ccu::Variable> partialOffsets(PAIR_COUNT);
    ccu::Variable nextPartialOffset;
    nextPartialOffset = 0;
    for (uint32_t group = 1U; group < PAIR_COUNT; ++group) {
        partialOffsets[group] = nextPartialOffset;
        nextPartialOffset += ctx.segmentBytes;
    }

    std::vector<ccu::Variable> halfOffsets(2U);
    std::vector<ccu::Variable> halfLengths(2U);
    halfOffsets[0] = 0;
    halfOffsets[1] = ctx.lastSegmentBytes;
    halfLengths[0] = ctx.lastSegmentBytes;
    halfLengths[1] = ctx.initializeOutput;

    for (uint32_t round = 0; round < 2U; ++round) {
        for (uint32_t group = 0; group < PAIR_COUNT; ++group) {
            for (uint32_t member = 0; member < 2U; ++member) {
                const uint32_t sourceIndex = group * 2U + member;
                const uint32_t half = (member + round) % 2U;
                ccu::LocalAddr destination;
                if (group == 0U) {
                    destination = MakeLocal(ctx.output, ctx.outputToken);
                } else {
                    destination = MakeLocal(ctx.scratch, ctx.scratchToken);
                    destination.addr += partialOffsets[group];
                }
                destination.addr += halfOffsets[half];
                const uint16_t mask =
                    static_cast<uint16_t>(1U << sourceIndex);
                if (round == 0U) {
                    CCU_CHK_RET(InitializeSegment(ctx, sources[sourceIndex],
                        destination, halfOffsets[half], halfLengths[half], mask));
                } else {
                    CCU_CHK_RET(ReduceSegment(ctx, sources[sourceIndex],
                        destination, halfOffsets[half], halfLengths[half], mask));
                }
            }
        }
        ccu::EventWait(ctx.event, SOURCE_MASK);
    }

    ccu::LocalAddr firstRoot = MakeLocal(ctx.output, ctx.outputToken);
    ccu::LocalAddr firstPeer = MakeLocal(ctx.scratch, ctx.scratchToken);
    ccu::LocalAddr secondRoot = MakeLocal(ctx.scratch, ctx.scratchToken);
    secondRoot.addr += partialOffsets[2U];
    ccu::LocalAddr secondPeer = MakeLocal(ctx.scratch, ctx.scratchToken);
    secondPeer.addr += partialOffsets[3U];
    constexpr uint16_t FIRST_TREE_MASK = 1U << 8;
    constexpr uint16_t SECOND_TREE_MASK = 1U << 9;
    CCU_CHK_RET(ccu::LocalReduce(firstRoot, firstPeer, ctx.sliceSize,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, FIRST_TREE_MASK));
    CCU_CHK_RET(ccu::LocalReduce(secondRoot, secondPeer, ctx.sliceSize,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, SECOND_TREE_MASK));

    std::vector<ccu::Variable> stripeOffsets;
    std::vector<ccu::Variable> stripeLengths;
    BuildV23StripeLayout(ctx, stripeOffsets, stripeLengths);

    const uint32_t rootLocal = ctx.arg->rankId % V23_LOCAL_SIZE;
    std::vector<ccu::Variable> ownerOffsets(V23_STRIPE_COUNT);
    for (uint32_t stripe = 0; stripe < V23_STRIPE_COUNT; ++stripe) {
        const uint32_t ownerLocal =
            (rootLocal + V23_LOCAL_SIZE - stripe) % V23_LOCAL_SIZE;
        ownerOffsets[stripe] = ctx.remoteGroupOffset;
        for (uint32_t index = 0; index < ownerLocal; ++index) {
            ownerOffsets[stripe] += ctx.sliceSize;
        }

        ccu::LocalAddr destination = MakeLocal(ctx.scratch, ctx.scratchToken);
        destination.addr += ctx.segmentBytes;
        destination.addr += ctx.segmentBytes;
        destination.addr += ctx.segmentBytes;
        destination.addr += stripeOffsets[stripe];
        ccu::LocalAddr source = MakeLocal(ctx.input, ctx.inputToken);
        source.addr += ownerOffsets[stripe];
        source.addr += stripeOffsets[stripe];
        const uint16_t mask = static_cast<uint16_t>(1U << stripe);
        CCU_CHK_RET(ccu::LocalCopy(destination, source, stripeLengths[stripe],
            ctx.event, mask));
    }
    ccu::EventWait(ctx.event, V23_STRIPE_MASK);

    for (uint32_t round = 0; round < 2U; ++round) {
        for (uint32_t stripe = 0; stripe < V23_STRIPE_COUNT; ++stripe) {
            const uint32_t firstPartner =
                (rootLocal + stripe + 1U) % V23_LOCAL_SIZE;
            const uint32_t secondPartner = (rootLocal +
                ((stripe + 1U) % V23_STRIPE_COUNT) + 1U) % V23_LOCAL_SIZE;
            const uint32_t partnerLocal =
                round == 0U ? firstPartner : secondPartner;
            uint32_t peerIndex = 0;
            if (!FindV23PeerIndex(*ctx.arg, partnerLocal, peerIndex)) {
                return CCU_E_PARA;
            }

            ccu::LocalAddr destination = MakeLocal(ctx.scratch, ctx.scratchToken);
            destination.addr += ctx.segmentBytes;
            destination.addr += ctx.segmentBytes;
            destination.addr += ctx.segmentBytes;
            destination.addr += stripeOffsets[stripe];
            ccu::RemoteAddr source =
                MakeRemote(ctx.peerInput[peerIndex], ctx.peerToken[peerIndex]);
            source.addr += ownerOffsets[stripe];
            source.addr += stripeOffsets[stripe];
            const uint16_t mask = static_cast<uint16_t>(1U << stripe);
            CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[peerIndex],
                destination, source, stripeLengths[stripe], HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, ctx.event, mask));
        }
        if (round == 0U) {
            // The first pair-tree level and helper transfers use disjoint
            // memory. Wait only when the dependent final tree edge is ready,
            // then let that local reduction overlap the second Mesh round.
            ccu::EventWait(ctx.event,
                FIRST_TREE_MASK | SECOND_TREE_MASK);
            ccu::LocalAddr finalRoot = MakeLocal(ctx.output, ctx.outputToken);
            ccu::LocalAddr finalPeer = MakeLocal(ctx.scratch, ctx.scratchToken);
            finalPeer.addr += partialOffsets[2U];
            CCU_CHK_RET(ccu::LocalReduce(finalRoot, finalPeer, ctx.sliceSize,
                HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event,
                FIRST_TREE_MASK));
        }
        ccu::EventWait(ctx.event, V23_STRIPE_MASK);
    }
    ccu::EventWait(ctx.event, FIRST_TREE_MASK);

    CCU_CHK_RET(FinishV23Channels(ctx));
    return CCU_SUCCESS;
}

/** Five channel-disjoint Clos rounds produce the direct five-source partial. */
static CcuResult ReduceV23ClosDirect(V23ReduceLayerContext &ctx)
{
    if (ctx.arg->layer != 1U || ctx.arg->includeSelf != 0U ||
        ctx.arg->peerCount != V23_LOCAL_SIZE) {
        return CCU_E_PARA;
    }
    CCU_CHK_RET(ExchangeV23Address(ctx, ctx.input, ctx.inputToken));

    std::vector<ccu::Variable> stripeOffsets;
    std::vector<ccu::Variable> stripeLengths;
    BuildV23StripeLayout(ctx, stripeOffsets, stripeLengths);
    const uint32_t ownerLocal = ctx.arg->rankId % V23_LOCAL_SIZE;

    for (uint32_t round = 0; round < V23_DIRECT_ROUNDS; ++round) {
        for (uint32_t stripe = 0; stripe < V23_STRIPE_COUNT; ++stripe) {
            const uint32_t sourceLocal =
                (ownerLocal + V23_DIRECT_SOURCE[round][stripe]) % V23_LOCAL_SIZE;
            uint32_t peerIndex = 0;
            if (!FindV23PeerIndex(*ctx.arg, sourceLocal, peerIndex)) {
                return CCU_E_PARA;
            }
            ccu::LocalAddr destination = MakeLocal(ctx.output, ctx.outputToken);
            destination.addr += stripeOffsets[stripe];
            ccu::RemoteAddr source =
                MakeRemote(ctx.peerInput[peerIndex], ctx.peerToken[peerIndex]);
            source.addr += ctx.sliceOffset;
            source.addr += stripeOffsets[stripe];
            const uint16_t mask = static_cast<uint16_t>(1U << stripe);
            if (round == 0U) {
                CCU_CHK_RET(ccu::Read(ctx.arg->channels[peerIndex], destination,
                    source, stripeLengths[stripe], ctx.event, mask));
            } else {
                CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[peerIndex],
                    destination, source, stripeLengths[stripe],
                    HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, mask));
            }
        }
        ccu::EventWait(ctx.event, V23_STRIPE_MASK);
    }

    CCU_CHK_RET(FinishV23Channels(ctx));
    return CCU_SUCCESS;
}

/** Receive one helper per stripe, then fold the complete remote partial once. */
static CcuResult ReduceV23ClosFinal(V23ReduceLayerContext &ctx)
{
    if (ctx.arg->layer != 1U || ctx.arg->includeSelf != 0U ||
        ctx.arg->peerCount != V23_LOCAL_SIZE) {
        return CCU_E_PARA;
    }
    CCU_CHK_RET(ExchangeV23Address(ctx, ctx.input, ctx.inputToken));

    std::vector<ccu::Variable> stripeOffsets;
    std::vector<ccu::Variable> stripeLengths;
    BuildV23StripeLayout(ctx, stripeOffsets, stripeLengths);
    const uint32_t ownerLocal = ctx.arg->rankId % V23_LOCAL_SIZE;

    for (uint32_t stripe = 0; stripe < V23_STRIPE_COUNT; ++stripe) {
        const uint32_t helperLocal =
            (ownerLocal + stripe) % V23_LOCAL_SIZE;
        uint32_t peerIndex = 0;
        if (!FindV23PeerIndex(*ctx.arg, helperLocal, peerIndex)) {
            return CCU_E_PARA;
        }
        ccu::LocalAddr destination = MakeLocal(ctx.scratch, ctx.scratchToken);
        destination.addr += stripeOffsets[stripe];
        ccu::RemoteAddr source =
            MakeRemote(ctx.peerInput[peerIndex], ctx.peerToken[peerIndex]);
        source.addr += stripeOffsets[stripe];
        const uint16_t mask = static_cast<uint16_t>(1U << stripe);
        CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[peerIndex], destination,
            source, stripeLengths[stripe], HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event, mask));
    }
    ccu::EventWait(ctx.event, V23_STRIPE_MASK);

    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[peerIndex],
            POST_NOTIFY_IDX, POST_READY_MASK));
    }

    ccu::LocalAddr destination = MakeLocal(ctx.output, ctx.outputToken);
    ccu::LocalAddr source = MakeLocal(ctx.scratch, ctx.scratchToken);
    CCU_CHK_RET(ccu::LocalReduce(destination, source, ctx.sliceSize,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 1U));
    ccu::EventWait(ctx.event, 1U);

    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[peerIndex],
            POST_NOTIFY_IDX, POST_READY_MASK));
    }
    return CCU_SUCCESS;
}

static ccu::RemoteAddr MakePeerRemote(ReduceLayerContext &ctx,
    uint32_t peerIndex, ccu::Variable offset)
{
    ccu::RemoteAddr remote =
        MakeRemote(ctx.peerInput[peerIndex], ctx.peerToken[peerIndex]);
    remote.addr += offset;
    return remote;
}

static CcuResult ReduceTwoByEightClos(ReduceLayerContext &ctx)
{
    if (ctx.arg->selectedPeerCount != 1U) {
        return CCU_E_PARA;
    }
    CCU_CHK_RET(ExchangeAddressSet(ctx, ctx.arg->selectedPeerIndices,
        ctx.arg->selectedPeerCount, ctx.input, ctx.inputToken));

    ccu::LocalAddr destination = MakeLocal(ctx.scratch, ctx.scratchToken);
    ccu::LocalAddr localSource = MakeLocal(ctx.input, ctx.inputToken);
    localSource.addr += ctx.sliceOffset;
    CCU_CHK_RET(ccu::LocalCopy(
        destination, localSource, ctx.sliceSize, ctx.event, 1));
    ccu::EventWait(ctx.event, 1);

    const uint32_t peerIndex = ctx.arg->selectedPeerIndices[0];
    ccu::RemoteAddr remoteSource =
        MakePeerRemote(ctx, peerIndex, ctx.sliceOffset);
    CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[peerIndex], destination,
        remoteSource, ctx.sliceSize, HCCL_DATA_TYPE_FP32,
        HCCL_REDUCE_SUM, ctx.event, 1));
    ccu::EventWait(ctx.event, 1);

    CCU_CHK_RET(BarrierAddressSet(ctx, ctx.arg->selectedPeerIndices,
        ctx.arg->selectedPeerCount));
    return CCU_SUCCESS;
}

static CcuResult ReduceTwoByEightMesh(ReduceLayerContext &ctx)
{
    if (ctx.arg->groupPeerCount != TWO_BY_EIGHT_MESH_PEERS) {
        return CCU_E_PARA;
    }
    CCU_CHK_RET(ExchangeAddressSet(ctx, ctx.arg->groupPeerIndices,
        ctx.arg->groupPeerCount, ctx.scratch, ctx.scratchToken));

    ccu::LocalAddr destination = MakeLocal(ctx.output, ctx.outputToken);
    ccu::LocalAddr localSource = MakeLocal(ctx.scratch, ctx.scratchToken);
    localSource.addr += ctx.sliceOffset;
    CCU_CHK_RET(ccu::LocalCopy(
        destination, localSource, ctx.sliceSize, ctx.event, 1));
    ccu::EventWait(ctx.event, 1);

    CCU_IF(ctx.segmentBytes == 0)
    {
        for (uint32_t index = 0; index < ctx.arg->groupPeerCount; ++index) {
            const uint32_t peerIndex = ctx.arg->groupPeerIndices[index];
            ccu::RemoteAddr remoteSource =
                MakePeerRemote(ctx, peerIndex, ctx.sliceOffset);
            CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[peerIndex], destination,
                remoteSource, ctx.sliceSize, HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, ctx.event, 1));
            ccu::EventWait(ctx.event, 1);
        }
    }

    CCU_IF(ctx.segmentBytes != 0)
    {
        // Four disjoint destination stripes are active in every round. The
        // peer rotation is fixed, covers all seven peers for every stripe,
        // and gives each peer at most one operation per round.
        for (uint32_t round = 0; round < TWO_BY_EIGHT_MESH_PEERS; ++round) {
            ccu::Variable stripeOffset;
            stripeOffset = 0;
            for (uint32_t stripe = 0; stripe < TWO_BY_EIGHT_MESH_WINDOW;
                ++stripe) {
                const uint32_t peerOrder =
                    (round + stripe) % TWO_BY_EIGHT_MESH_PEERS;
                const uint32_t peerIndex =
                    ctx.arg->groupPeerIndices[peerOrder];
                const uint16_t mask = static_cast<uint16_t>(1U << stripe);

                ccu::Variable stripeLength;
                if (stripe + 1U == TWO_BY_EIGHT_MESH_WINDOW) {
                    stripeLength = ctx.lastSegmentBytes;
                } else {
                    stripeLength = ctx.segmentBytes;
                }

                ccu::LocalAddr stripeDestination =
                    MakeLocal(ctx.output, ctx.outputToken);
                stripeDestination.addr += stripeOffset;
                ccu::RemoteAddr remoteSource =
                    MakePeerRemote(ctx, peerIndex, ctx.sliceOffset);
                remoteSource.addr += stripeOffset;
                CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[peerIndex],
                    stripeDestination, remoteSource, stripeLength,
                    HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, mask));
                stripeOffset += ctx.segmentBytes;
            }
            ccu::EventWait(ctx.event, TWO_BY_EIGHT_MESH_WINDOW_MASK);
        }
    }
    CCU_CHK_RET(BarrierAddressSet(ctx, ctx.arg->groupPeerIndices,
        ctx.arg->groupPeerCount));
    return CCU_SUCCESS;
}

static CcuResult ReduceMixedClos(ReduceLayerContext &ctx)
{
    if (ctx.arg->selectedPeerCount == 0U ||
        ctx.arg->selectedPeerCount > 2U) {
        return CCU_E_PARA;
    }
    CCU_CHK_RET(ExchangeAddressSet(ctx, ctx.arg->selectedPeerIndices,
        ctx.arg->selectedPeerCount, ctx.input, ctx.inputToken));

    ccu::LocalAddr destination = MakeLocal(ctx.scratch, ctx.scratchToken);
    for (uint32_t index = 0; index < ctx.arg->selectedPeerCount; ++index) {
        const uint32_t peerIndex = ctx.arg->selectedPeerIndices[index];
        ccu::RemoteAddr remoteSource =
            MakePeerRemote(ctx, peerIndex, ctx.sliceOffset);
        if (index == 0U) {
            CCU_CHK_RET(ccu::Read(ctx.arg->channels[peerIndex], destination,
                remoteSource, ctx.sliceSize, ctx.event, 1));
        } else {
            CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[peerIndex], destination,
                remoteSource, ctx.sliceSize, HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, ctx.event, 1));
        }
        ccu::EventWait(ctx.event, 1);
    }

    CCU_CHK_RET(BarrierAddressSet(ctx, ctx.arg->selectedPeerIndices,
        ctx.arg->selectedPeerCount));
    return CCU_SUCCESS;
}

static CcuResult ReduceMixedMesh(ReduceLayerContext &ctx)
{
    if (ctx.arg->selectedPeerCount > 1U ||
        ctx.arg->groupPeerCount != 3U) {
        return CCU_E_PARA;
    }
    ccu::LocalAddr scratch = MakeLocal(ctx.scratch, ctx.scratchToken);
    ccu::LocalAddr localInput = MakeLocal(ctx.input, ctx.inputToken);
    localInput.addr += ctx.sliceOffset;
    CCU_CHK_RET(ccu::LocalReduce(scratch, localInput, ctx.segmentBytes,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 1));
    ccu::EventWait(ctx.event, 1);

    if (ctx.arg->selectedPeerCount == 1U) {
        CCU_CHK_RET(ExchangeAddressSet(ctx, ctx.arg->selectedPeerIndices,
            ctx.arg->selectedPeerCount, ctx.input, ctx.inputToken));
        const uint32_t peerIndex = ctx.arg->selectedPeerIndices[0];
        ccu::RemoteAddr pairInput =
            MakePeerRemote(ctx, peerIndex, ctx.sliceOffset);
        CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[peerIndex], scratch,
            pairInput, ctx.segmentBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event, 1));
        ccu::EventWait(ctx.event, 1);
        CCU_CHK_RET(BarrierAddressSet(ctx, ctx.arg->selectedPeerIndices,
            ctx.arg->selectedPeerCount));
    }

    CCU_CHK_RET(ExchangeAddressSet(ctx, ctx.arg->groupPeerIndices,
        ctx.arg->groupPeerCount, ctx.scratch, ctx.scratchToken));

    ccu::LocalAddr destination = MakeLocal(ctx.output, ctx.outputToken);
    ccu::LocalAddr localPartial = MakeLocal(ctx.scratch, ctx.scratchToken);
    localPartial.addr += ctx.lastSegmentBytes;
    CCU_CHK_RET(ccu::LocalCopy(
        destination, localPartial, ctx.sliceSize, ctx.event, 1));
    ccu::EventWait(ctx.event, 1);

    for (uint32_t index = 0; index < ctx.arg->groupPeerCount; ++index) {
        const uint32_t peerIndex = ctx.arg->groupPeerIndices[index];
        ccu::RemoteAddr remotePartial =
            MakePeerRemote(ctx, peerIndex, ctx.lastSegmentBytes);
        CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[peerIndex], destination,
            remotePartial, ctx.sliceSize, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event, 1));
        ccu::EventWait(ctx.event, 1);
    }
    CCU_CHK_RET(BarrierAddressSet(ctx, ctx.arg->groupPeerIndices,
        ctx.arg->groupPeerCount));
    return CCU_SUCCESS;
}

static CcuResult ReduceLayerKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceLayer *>(arg);
    if (kernelArg == nullptr || kernelArg->peerCount != kernelArg->channelCount ||
        kernelArg->peerCount > MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    const std::vector<SourcePlan> sources = BuildSourcePlan(*kernelArg);
    if (sources.empty() || sources.size() > MAX_LAYER_SOURCES) {
        return CCU_E_PARA;
    }

    ReduceLayerContext ctx;
    ctx.arg = kernelArg;
    ctx.peerInput.resize(ctx.arg->peerCount);
    ctx.peerToken.resize(ctx.arg->peerCount);

    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        ctx.peerInput[peerIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[peerIndex], INPUT_XN_ID);
        ctx.peerToken[peerIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[peerIndex], TOKEN_XN_ID);
    }

    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.mode, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.segmentBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSegmentBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.initializeOutput, argId++));

    CCU_IF(ctx.mode == CCU_MODE_GENERIC)
    {
        for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
                ctx.input, INPUT_XN_ID, PRE_NOTIFY_IDX, INPUT_READY_MASK));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[peerIndex],
                ctx.inputToken, TOKEN_XN_ID, PRE_NOTIFY_IDX, TOKEN_READY_MASK));
        }
        for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
            CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[peerIndex], PRE_NOTIFY_IDX,
                INPUT_READY_MASK | TOKEN_READY_MASK));
        }

        CCU_IF(ctx.sliceSize != 0)
        {
            CCU_IF(ctx.segmentBytes == 0)
            {
                CCU_CHK_RET(ReduceSequential(ctx, sources));
            }
            CCU_IF(ctx.segmentBytes != 0)
            {
                CCU_CHK_RET(ReduceStriped(ctx, sources));
            }
        }

        for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
            CCU_CHK_RET(ccu::NotifyRecord(
                ctx.arg->channels[peerIndex], POST_NOTIFY_IDX, POST_READY_MASK));
        }
        for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
            CCU_CHK_RET(ccu::NotifyWait(
                ctx.arg->channels[peerIndex], POST_NOTIFY_IDX, POST_READY_MASK));
        }
    }

    CCU_IF(ctx.mode == CCU_MODE_SMALL_FANIN)
    {
        CCU_CHK_RET(ReduceSmallFanin(ctx, sources));
    }

    if (kernelArg->smallTopology == SMALL_TOPO_TWO_BY_EIGHT) {
        if (kernelArg->layer == 1U) {
            CCU_IF(ctx.mode == CCU_MODE_TWO_BY_EIGHT_CLOS)
            {
                CCU_CHK_RET(ReduceTwoByEightClos(ctx));
            }
        } else {
            CCU_IF(ctx.mode == CCU_MODE_TWO_BY_EIGHT_MESH)
            {
                CCU_CHK_RET(ReduceTwoByEightMesh(ctx));
            }
        }
    }

    if (kernelArg->smallTopology == SMALL_TOPO_EIGHT_PLUS_FOUR_LARGE ||
        kernelArg->smallTopology == SMALL_TOPO_EIGHT_PLUS_FOUR_SMALL) {
        if (kernelArg->layer == 1U) {
            CCU_IF(ctx.mode == CCU_MODE_MIXED_CLOS)
            {
                CCU_CHK_RET(ReduceMixedClos(ctx));
            }
        } else {
            CCU_IF(ctx.mode == CCU_MODE_MIXED_MESH)
            {
                CCU_CHK_RET(ReduceMixedMesh(ctx));
            }
        }
    }
    return CCU_SUCCESS;
}

static CcuResult ReduceGroupedLayerKernel(CcuKernelArg arg,
    uint32_t largeAlgorithm)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceLayer *>(arg);
    if (kernelArg == nullptr || kernelArg->peerCount != kernelArg->channelCount ||
        kernelArg->peerCount > MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    const std::vector<SourcePlan> sources = BuildSourcePlan(*kernelArg);
    if (sources.size() != 4U && sources.size() != 8U) {
        return CCU_E_PARA;
    }

    ReduceLayerContext ctx;
    ctx.arg = kernelArg;
    ctx.peerInput.resize(ctx.arg->peerCount);
    ctx.peerToken.resize(ctx.arg->peerCount);
    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        ctx.peerInput[peerIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[peerIndex], INPUT_XN_ID);
        ctx.peerToken[peerIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[peerIndex], TOKEN_XN_ID);
    }

    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.mode, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.segmentBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSegmentBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.initializeOutput, argId++));

    if (largeAlgorithm == LARGE_ALGORITHM_V12_P11) {
        CCU_CHK_RET(ReduceLargeGroupedV10(ctx, sources, true));
    } else if (largeAlgorithm == LARGE_ALGORITHM_V13_P17) {
        CCU_CHK_RET(ReduceLargeGroupedV13(ctx, sources));
    } else {
        CCU_CHK_RET(ReduceLargeGroupedV10(ctx, sources, false));
    }
    return CCU_SUCCESS;
}

static CcuResult ReduceGroupedLayerV23Kernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceLayer *>(arg);
    if (kernelArg == nullptr || kernelArg->peerCount != kernelArg->channelCount ||
        kernelArg->peerCount > MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    const std::vector<SourcePlan> sources = BuildSourcePlan(*kernelArg);
    if (sources.size() != V23_LOCAL_SIZE) {
        return CCU_E_PARA;
    }

    V23ReduceLayerContext ctx;
    ctx.arg = kernelArg;
    ctx.peerInput.resize(ctx.arg->peerCount);
    ctx.peerToken.resize(ctx.arg->peerCount);
    for (uint32_t peerIndex = 0; peerIndex < ctx.arg->peerCount; ++peerIndex) {
        ctx.peerInput[peerIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[peerIndex], INPUT_XN_ID);
        ctx.peerToken[peerIndex] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[peerIndex], TOKEN_XN_ID);
    }

    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.mode, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.segmentBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSegmentBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.initializeOutput, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.regularStripeBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.finalStripeBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.remoteGroupOffset, argId++));

    if (kernelArg->layer == 0U) {
        CCU_IF(ctx.mode == CCU_MODE_V23_MESH_HELPER)
        {
            CCU_CHK_RET(ReduceV23MeshHelper(ctx, sources));
        }
    } else if (kernelArg->layer == 1U) {
        CCU_IF(ctx.mode == CCU_MODE_V23_CLOS_DIRECT)
        {
            CCU_CHK_RET(ReduceV23ClosDirect(ctx));
        }
        CCU_IF(ctx.mode == CCU_MODE_V23_CLOS_FINAL)
        {
            CCU_CHK_RET(ReduceV23ClosFinal(ctx));
        }
    } else {
        return CCU_E_PARA;
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceLayer0Kernel(CcuKernelArg arg)
{
    return ReduceLayerKernel(arg);
}

CcuResult CcuReduceLayer1Kernel(CcuKernelArg arg)
{
    return ReduceLayerKernel(arg);
}

CcuResult CcuReduceLayer0GroupedKernel(CcuKernelArg arg)
{
    return ReduceGroupedLayerKernel(arg, LARGE_ALGORITHM_V10);
}

CcuResult CcuReduceLayer1GroupedKernel(CcuKernelArg arg)
{
    return ReduceGroupedLayerKernel(arg, LARGE_ALGORITHM_V10);
}

CcuResult CcuReduceLayer0GroupedEarlyKernel(CcuKernelArg arg)
{
    return ReduceGroupedLayerKernel(arg, LARGE_ALGORITHM_V12_P11);
}

CcuResult CcuReduceLayer1GroupedEarlyKernel(CcuKernelArg arg)
{
    return ReduceGroupedLayerKernel(arg, LARGE_ALGORITHM_V12_P11);
}

CcuResult CcuReduceLayer0GroupedV13Kernel(CcuKernelArg arg)
{
    return ReduceGroupedLayerKernel(arg, LARGE_ALGORITHM_V13_P17);
}

CcuResult CcuReduceLayer1GroupedV13Kernel(CcuKernelArg arg)
{
    return ReduceGroupedLayerKernel(arg, LARGE_ALGORITHM_V13_P17);
}

CcuResult CcuReduceLayer0GroupedV23Kernel(CcuKernelArg arg)
{
    return ReduceGroupedLayerV23Kernel(arg);
}

CcuResult CcuReduceLayer1GroupedV23Kernel(CcuKernelArg arg)
{
    return ReduceGroupedLayerV23Kernel(arg);
}

CcuResult CcuReduceFinalizeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgFinalize *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PARA;
    }

    ccu::Variable remotePartial;
    ccu::Variable cclToken;
    ccu::Variable output;
    ccu::Variable outputToken;
    ccu::Variable dataSize;
    ccu::Event event;

    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(remotePartial, argId++));
    CCU_CHK_RET(ccu::LoadArg(cclToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(output, argId++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(dataSize, argId++));

    ccu::LocalAddr source = MakeLocal(remotePartial, cclToken);
    ccu::LocalAddr destination = MakeLocal(output, outputToken);

    CCU_IF(dataSize != 0)
    {
        CCU_CHK_RET(ccu::LocalReduce(destination, source, dataSize,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, event, 1));
        ccu::EventWait(event, 1);
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
