/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REDUCE_SCATTER_LAYER_DIE_PAIRWISE_H
#define CCU_REDUCE_SCATTER_LAYER_DIE_PAIRWISE_H

#include <cstdint>

#include "ccu_primitives.hpp"
#include "ccu_types.h"
#include "custom.h"

namespace ccu = ::AscendC::ccu;

#define CCU_CHK_RET(call)            \
    do {                             \
        CcuResult ret = (call);      \
        if (ret != CCU_SUCCESS) {    \
            return ret;              \
        }                            \
    } while (0)

namespace {
constexpr uint32_t RS_MAX_GROUP_PEERS = 8;
constexpr uint32_t RS_PIPELINE_CHUNKS = 4;
constexpr uint32_t RS_PIPELINE_BANKS = 2;
constexpr uint32_t RS_PIPELINE_SELF_EVENT_BIT = 8;
constexpr int RS_INPUT_XN_ID = 0;
constexpr int RS_TOKEN_XN_ID = 1;
constexpr int RS_POST_SYNC_ID = 2;
constexpr int RS_CKE_INDEX = 0;

struct ReduceScatterContext {
    const ReduceScatterKernelArg *arg;

    ccu::Variable remoteInputBase[RS_MAX_GROUP_PEERS];
    ccu::Variable remoteToken[RS_MAX_GROUP_PEERS];
    ccu::Variable localInputBase;
    ccu::Variable localOutputBase;
    ccu::Variable localToken;
    ccu::Variable localScratchBase[3];
    ccu::Variable inputOffset;
    ccu::Variable outputOffset;
    ccu::Variable activeBytes;
    ccu::Variable operationMode;
    ccu::Variable pipelineSlotStride;
    ccu::Variable pipelineSetStride;
    ccu::Variable pipelineChunkSize[RS_PIPELINE_CHUNKS];

    ccu::RemoteAddr remoteInput[RS_MAX_GROUP_PEERS];
    ccu::LocalAddr localInput;
    ccu::LocalAddr finalOutput;
    ccu::LocalAddr scratch[3];
    ccu::LocalAddr pipelinePeerSlot[RS_PIPELINE_BANKS][RS_MAX_GROUP_PEERS];
    ccu::Event event;
    ccu::Event pipelineReadEvent[RS_PIPELINE_BANKS];
    ccu::Event pipelineReduceEvent;
};

CcuResult InitResource(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->channelCount == 0 ||
        arg->channelCount > RS_MAX_GROUP_PEERS) {
        return CcuResult::CCU_E_PARA;
    }

    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ctx.remoteInputBase[channel] =
            ccu::GetResByChannel<ccu::Variable>(
                arg->channels[channel], RS_INPUT_XN_ID);
        ctx.remoteToken[channel] =
            ccu::GetResByChannel<ccu::Variable>(
                arg->channels[channel], RS_TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(ReduceScatterContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.localInputBase, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localOutputBase, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localScratchBase[0], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localScratchBase[1], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localScratchBase[2], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.activeBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.operationMode, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pipelineSlotStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pipelineSetStride, argId++));
    ctx.pipelineChunkSize[0] = ctx.activeBytes;
    for (uint32_t index = 1; index < RS_PIPELINE_CHUNKS; ++index) {
        CCU_CHK_RET(ccu::LoadArg(
            ctx.pipelineChunkSize[index], argId++));
    }
    return CCU_SUCCESS;
}

void PreSync(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint32_t inputMask = 1U << RS_INPUT_XN_ID;
    constexpr uint32_t tokenMask = 1U << RS_TOKEN_XN_ID;
    constexpr uint32_t readyMask = inputMask | tokenMask;

    // Publish every endpoint before waiting on any endpoint.  This keeps the
    // address exchange independent of peer iteration order.
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ccu::WriteVariableWithNotify(
            arg->channels[channel], ctx.localInputBase,
            RS_INPUT_XN_ID, RS_CKE_INDEX, inputMask);
        ccu::WriteVariableWithNotify(
            arg->channels[channel], ctx.localToken,
            RS_TOKEN_XN_ID, RS_CKE_INDEX, tokenMask);
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ccu::NotifyWait(
            arg->channels[channel], RS_CKE_INDEX, readyMask);
    }
}

void PostSync(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint32_t postMask = 1U << RS_POST_SYNC_ID;

    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ccu::NotifyRecord(
            arg->channels[channel], RS_CKE_INDEX, postMask);
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ccu::NotifyWait(
            arg->channels[channel], RS_CKE_INDEX, postMask);
    }
}

void PostSyncRecord(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint32_t postMask = 1U << RS_POST_SYNC_ID;
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        // This record is enqueued after every Read on the same channel.
        // Remote completion can therefore overlap the remaining local tree.
        ccu::NotifyRecord(
            arg->channels[channel], RS_CKE_INDEX, postMask);
    }
}

void PostSyncWait(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint32_t postMask = 1U << RS_POST_SYNC_ID;
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ccu::NotifyWait(
            arg->channels[channel], RS_CKE_INDEX, postMask);
    }
}

CcuResult ReduceOneChunk(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.localInput.addr = ctx.localInputBase;
    ctx.localInput.addr += ctx.inputOffset;
    ctx.localInput.token = ctx.localToken;

    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.addr += ctx.outputOffset;
    ctx.finalOutput.token = ctx.localToken;

    ctx.scratch[0].addr = ctx.localScratchBase[0];
    ctx.scratch[0].token = ctx.localToken;

    CCU_IF(ctx.activeBytes != 0) {
        if (arg->includeSelf) {
            ccu::LocalCopy(
                ctx.finalOutput, ctx.localInput, ctx.activeBytes,
                ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
        }

        // Channels are registered in ascending peer-rank order.  Reusing one
        // scratch tile after each wait gives a fixed FP32 reduction order and
        // keeps every kernel's static XN/GSA request small enough that two
        // kernels can coexist on the same IO Die.
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            ctx.remoteInput[channel].addr =
                ctx.remoteInputBase[channel];
            ctx.remoteInput[channel].addr += ctx.inputOffset;
            ctx.remoteInput[channel].token = ctx.remoteToken[channel];

            ccu::Read(
                arg->channels[channel], ctx.scratch[0],
                ctx.remoteInput[channel], ctx.activeBytes,
                ctx.event, 1);
            ccu::EventWait(ctx.event, 1);

            if (arg->includeSelf || channel != 0) {
                ccu::LocalReduce(
                    ctx.finalOutput, ctx.scratch[0], ctx.activeBytes,
                    arg->dataType, arg->reduceOp, ctx.event, 1);
            } else {
                ccu::LocalCopy(
                    ctx.finalOutput, ctx.scratch[0], ctx.activeBytes,
                    ctx.event, 1);
            }
            ccu::EventWait(ctx.event, 1);
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReduceOneChunkDepth2(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.localInput.addr = ctx.localInputBase;
    ctx.localInput.addr += ctx.inputOffset;
    ctx.localInput.token = ctx.localToken;

    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.addr += ctx.outputOffset;
    ctx.finalOutput.token = ctx.localToken;

    for (uint32_t slot = 0; slot < 2; ++slot) {
        ctx.scratch[slot].addr = ctx.localScratchBase[slot];
        ctx.scratch[slot].token = ctx.localToken;
    }

    CCU_IF(ctx.activeBytes != 0) {
        uint32_t batchBegin = 0;
        if (arg->includeSelf) {
            ccu::LocalCopy(
                ctx.finalOutput, ctx.localInput, ctx.activeBytes,
                ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
        } else if (arg->channelCount != 0) {
            // The group accumulator is already a private, disjoint local
            // tile.  Seed it directly with the first remote contribution,
            // then use the two existing scratch tiles for the next peers.
            // All reads complete before reduction, matching the official
            // multi-peer Read -> mask Wait task graph.
            uint32_t initialCount = arg->channelCount;
            if (initialCount > 3) {
                initialCount = 3;
            }
            uint32_t initialReadyMask = 0;
            for (uint32_t initial = 0; initial < initialCount; ++initial) {
                ctx.remoteInput[initial].addr =
                    ctx.remoteInputBase[initial];
                ctx.remoteInput[initial].addr += ctx.inputOffset;
                ctx.remoteInput[initial].token = ctx.remoteToken[initial];
                const uint32_t signal = 1U << initial;
                if (initial == 0) {
                    ccu::Read(
                        arg->channels[initial], ctx.finalOutput,
                        ctx.remoteInput[initial], ctx.activeBytes,
                        ctx.event, signal);
                } else {
                    ccu::Read(
                        arg->channels[initial],
                        ctx.scratch[initial - 1],
                        ctx.remoteInput[initial], ctx.activeBytes,
                        ctx.event, signal);
                }
                initialReadyMask |= signal;
            }
            ccu::EventWait(ctx.event, initialReadyMask);

            for (uint32_t initial = 1; initial < initialCount; ++initial) {
                ccu::LocalReduce(
                    ctx.finalOutput, ctx.scratch[initial - 1],
                    ctx.activeBytes, arg->dataType, arg->reduceOp,
                    ctx.event, 1);
                ccu::EventWait(ctx.event, 1);
            }
            batchBegin = initialCount;
        }

        // Both reads stay inside this kernel's fixed (layer, scope, die)
        // group.  Distinct scratch tiles and event bits allow at most two
        // peers to make progress concurrently; accumulation remains in the
        // original ascending peer-rank order.
        for (uint32_t batch = batchBegin;
            batch < arg->channelCount; batch += 2) {
            uint32_t batchCount = arg->channelCount - batch;
            if (batchCount > 2) {
                batchCount = 2;
            }
            uint32_t readyMask = 0;
            for (uint32_t slot = 0; slot < batchCount; ++slot) {
                const uint32_t channel = batch + slot;
                const uint32_t signal = 1U << slot;
                ctx.remoteInput[channel].addr =
                    ctx.remoteInputBase[channel];
                ctx.remoteInput[channel].addr += ctx.inputOffset;
                ctx.remoteInput[channel].token = ctx.remoteToken[channel];

                ccu::Read(
                    arg->channels[channel], ctx.scratch[slot],
                    ctx.remoteInput[channel], ctx.activeBytes,
                    ctx.event, signal);
                readyMask |= signal;
            }
            ccu::EventWait(ctx.event, readyMask);

            for (uint32_t slot = 0; slot < batchCount; ++slot) {
                ccu::LocalReduce(
                    ctx.finalOutput, ctx.scratch[slot], ctx.activeBytes,
                    arg->dataType, arg->reduceOp, ctx.event, 1);
                ccu::EventWait(ctx.event, 1);
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReduceOneChunkReadReduce(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.localInput.addr = ctx.localInputBase;
    ctx.localInput.addr += ctx.inputOffset;
    ctx.localInput.token = ctx.localToken;

    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.addr += ctx.outputOffset;
    ctx.finalOutput.token = ctx.localToken;

    CCU_IF(ctx.activeBytes != 0) {
        // The single-kernel 4x1 group always owns the local contribution.
        // Seed the accumulator once, then consume remote peers in the same
        // ascending channel order as the passing LocalReduce path.  Waiting
        // after every fused primitive keeps FP32 accumulation deterministic.
        ccu::LocalCopy(
            ctx.finalOutput, ctx.localInput, ctx.activeBytes,
            ctx.event, 1);
        ccu::EventWait(ctx.event, 1);

        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            ctx.remoteInput[channel].addr =
                ctx.remoteInputBase[channel];
            ctx.remoteInput[channel].addr += ctx.inputOffset;
            ctx.remoteInput[channel].token = ctx.remoteToken[channel];

            ccu::ReadReduce(
                arg->channels[channel], ctx.finalOutput,
                ctx.remoteInput[channel], ctx.activeBytes,
                arg->dataType, arg->reduceOp, ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReduceOneChunkMultiKernelReadReduce(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.localInput.addr = ctx.localInputBase;
    ctx.localInput.addr += ctx.inputOffset;
    ctx.localInput.token = ctx.localToken;

    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.addr += ctx.outputOffset;
    ctx.finalOutput.token = ctx.localToken;

    CCU_IF(ctx.activeBytes != 0) {
        uint32_t channelBegin = 0;
        if (arg->includeSelf) {
            ccu::LocalCopy(
                ctx.finalOutput, ctx.localInput, ctx.activeBytes,
                ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
        } else if (arg->channelCount != 0) {
            // F1R23R1 already proved that the first remote contribution can
            // seed this group's disjoint accumulator directly.
            ctx.remoteInput[0].addr = ctx.remoteInputBase[0];
            ctx.remoteInput[0].addr += ctx.inputOffset;
            ctx.remoteInput[0].token = ctx.remoteToken[0];
            ccu::Read(
                arg->channels[0], ctx.finalOutput, ctx.remoteInput[0],
                ctx.activeBytes, ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
            channelBegin = 1;
        }

        // The accumulator is now initialized by the same first contribution
        // as the passing path.  Fuse every later ascending peer contribution
        // with the network read, waiting after each primitive so the FP32
        // accumulation order remains unchanged.
        for (uint32_t channel = channelBegin;
            channel < arg->channelCount; ++channel) {
            ctx.remoteInput[channel].addr =
                ctx.remoteInputBase[channel];
            ctx.remoteInput[channel].addr += ctx.inputOffset;
            ctx.remoteInput[channel].token = ctx.remoteToken[channel];

            ccu::ReadReduce(
                arg->channels[channel], ctx.finalOutput,
                ctx.remoteInput[channel], ctx.activeBytes,
                arg->dataType, arg->reduceOp, ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReduceOneChunkSeedOverlapReadReduce(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.localInput.addr = ctx.localInputBase;
    ctx.localInput.addr += ctx.inputOffset;
    ctx.localInput.token = ctx.localToken;

    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.addr += ctx.outputOffset;
    ctx.finalOutput.token = ctx.localToken;

    ctx.scratch[0].addr = ctx.localScratchBase[0];
    ctx.scratch[0].token = ctx.localToken;

    CCU_IF(ctx.activeBytes != 0) {
        uint32_t channelBegin = 0;
        uint32_t readyMask = 1;
        bool reduceScratch = false;

        if (arg->includeSelf) {
            // The seed and first ascending peer write disjoint destinations.
            // Waiting on their mask before LocalReduce preserves the exact
            // local, peer0, peer1... FP32 accumulation order.
            ccu::LocalCopy(
                ctx.finalOutput, ctx.localInput, ctx.activeBytes,
                ctx.event, 1);
            if (arg->channelCount != 0) {
                ctx.remoteInput[0].addr = ctx.remoteInputBase[0];
                ctx.remoteInput[0].addr += ctx.inputOffset;
                ctx.remoteInput[0].token = ctx.remoteToken[0];
                ccu::Read(
                    arg->channels[0], ctx.scratch[0],
                    ctx.remoteInput[0], ctx.activeBytes,
                    ctx.event, 2);
                readyMask |= 2;
                channelBegin = 1;
                reduceScratch = true;
            }
        } else {
            // A non-self group retains its direct peer0 seed.  Peer1 may be
            // fetched concurrently into the private scratch tile because the
            // two writes are disjoint.
            ctx.remoteInput[0].addr = ctx.remoteInputBase[0];
            ctx.remoteInput[0].addr += ctx.inputOffset;
            ctx.remoteInput[0].token = ctx.remoteToken[0];
            ccu::Read(
                arg->channels[0], ctx.finalOutput,
                ctx.remoteInput[0], ctx.activeBytes,
                ctx.event, 1);
            channelBegin = 1;
            if (arg->channelCount > 1) {
                ctx.remoteInput[1].addr = ctx.remoteInputBase[1];
                ctx.remoteInput[1].addr += ctx.inputOffset;
                ctx.remoteInput[1].token = ctx.remoteToken[1];
                ccu::Read(
                    arg->channels[1], ctx.scratch[0],
                    ctx.remoteInput[1], ctx.activeBytes,
                    ctx.event, 2);
                readyMask |= 2;
                channelBegin = 2;
                reduceScratch = true;
            }
        }
        ccu::EventWait(ctx.event, readyMask);

        if (reduceScratch) {
            ccu::LocalReduce(
                ctx.finalOutput, ctx.scratch[0], ctx.activeBytes,
                arg->dataType, arg->reduceOp, ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
        }

        // Every later contribution remains fused and strictly ascending.
        for (uint32_t channel = channelBegin;
            channel < arg->channelCount; ++channel) {
            ctx.remoteInput[channel].addr =
                ctx.remoteInputBase[channel];
            ctx.remoteInput[channel].addr += ctx.inputOffset;
            ctx.remoteInput[channel].token = ctx.remoteToken[channel];

            ccu::ReadReduce(
                arg->channels[channel], ctx.finalOutput,
                ctx.remoteInput[channel], ctx.activeBytes,
                arg->dataType, arg->reduceOp, ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReduceOneChunkFrontier3ReadReduce(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.localInput.addr = ctx.localInputBase;
    ctx.localInput.addr += ctx.inputOffset;
    ctx.localInput.token = ctx.localToken;

    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.addr += ctx.outputOffset;
    ctx.finalOutput.token = ctx.localToken;

    for (uint32_t slot = 0; slot < 2; ++slot) {
        ctx.scratch[slot].addr = ctx.localScratchBase[slot];
        ctx.scratch[slot].token = ctx.localToken;
    }

    CCU_IF(ctx.activeBytes != 0) {
        uint32_t channelBegin = 0;
        uint32_t readyMask = 0;
        uint32_t scratchCount = 0;

        if (arg->includeSelf) {
            // The local seed and the first two ascending peers write three
            // disjoint destinations.  Ordered reductions after the mask wait
            // restore local, peer0, peer1... exactly.
            ccu::LocalCopy(
                ctx.finalOutput, ctx.localInput, ctx.activeBytes,
                ctx.event, 1);
            readyMask |= 1;

            scratchCount = arg->channelCount;
            if (scratchCount > 2) {
                scratchCount = 2;
            }
            for (uint32_t slot = 0; slot < scratchCount; ++slot) {
                ctx.remoteInput[slot].addr = ctx.remoteInputBase[slot];
                ctx.remoteInput[slot].addr += ctx.inputOffset;
                ctx.remoteInput[slot].token = ctx.remoteToken[slot];
                const uint32_t signal = 1U << (slot + 1);
                ccu::Read(
                    arg->channels[slot], ctx.scratch[slot],
                    ctx.remoteInput[slot], ctx.activeBytes,
                    ctx.event, signal);
                readyMask |= signal;
            }
            channelBegin = scratchCount;
        } else if (arg->channelCount != 0) {
            // The direct peer0 seed remains first.  At most peer1 and peer2
            // fill the two private scratch tiles concurrently.
            ctx.remoteInput[0].addr = ctx.remoteInputBase[0];
            ctx.remoteInput[0].addr += ctx.inputOffset;
            ctx.remoteInput[0].token = ctx.remoteToken[0];
            ccu::Read(
                arg->channels[0], ctx.finalOutput,
                ctx.remoteInput[0], ctx.activeBytes,
                ctx.event, 1);
            readyMask |= 1;
            channelBegin = 1;

            scratchCount = arg->channelCount - channelBegin;
            if (scratchCount > 2) {
                scratchCount = 2;
            }
            for (uint32_t slot = 0; slot < scratchCount; ++slot) {
                const uint32_t channel = channelBegin + slot;
                ctx.remoteInput[channel].addr =
                    ctx.remoteInputBase[channel];
                ctx.remoteInput[channel].addr += ctx.inputOffset;
                ctx.remoteInput[channel].token =
                    ctx.remoteToken[channel];
                const uint32_t signal = 1U << (slot + 1);
                ccu::Read(
                    arg->channels[channel], ctx.scratch[slot],
                    ctx.remoteInput[channel], ctx.activeBytes,
                    ctx.event, signal);
                readyMask |= signal;
            }
            channelBegin += scratchCount;
        }

        if (readyMask != 0) {
            ccu::EventWait(ctx.event, readyMask);
        }
        for (uint32_t slot = 0; slot < scratchCount; ++slot) {
            ccu::LocalReduce(
                ctx.finalOutput, ctx.scratch[slot], ctx.activeBytes,
                arg->dataType, arg->reduceOp, ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
        }

        // Any remaining peers stay fused and strictly ascending.
        for (uint32_t channel = channelBegin;
            channel < arg->channelCount; ++channel) {
            ctx.remoteInput[channel].addr =
                ctx.remoteInputBase[channel];
            ctx.remoteInput[channel].addr += ctx.inputOffset;
            ctx.remoteInput[channel].token = ctx.remoteToken[channel];

            ccu::ReadReduce(
                arg->channels[channel], ctx.finalOutput,
                ctx.remoteInput[channel], ctx.activeBytes,
                arg->dataType, arg->reduceOp, ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReduceOneChunkPhaseSharedFrontier4(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.localInput.addr = ctx.localInputBase;
    ctx.localInput.addr += ctx.inputOffset;
    ctx.localInput.token = ctx.localToken;

    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.addr += ctx.outputOffset;
    ctx.finalOutput.token = ctx.localToken;

    for (uint32_t slot = 0; slot < 3; ++slot) {
        ctx.scratch[slot].addr = ctx.localScratchBase[slot];
        ctx.scratch[slot].token = ctx.localToken;
    }

    CCU_IF(ctx.activeBytes != 0) {
        uint32_t channelBegin = 0;
        uint32_t readyMask = 0;
        uint32_t scratchCount = 0;

        if (arg->includeSelf) {
            // Local seed and up to three ascending peers own four mutually
            // disjoint destinations.  A single mask wait closes every async
            // receive before strictly ordered FP32 LocalReduce.
            ccu::LocalCopy(
                ctx.finalOutput, ctx.localInput, ctx.activeBytes,
                ctx.event, 1);
            readyMask |= 1;

            scratchCount = arg->channelCount;
            if (scratchCount > 3) {
                scratchCount = 3;
            }
            for (uint32_t slot = 0; slot < scratchCount; ++slot) {
                ctx.remoteInput[slot].addr = ctx.remoteInputBase[slot];
                ctx.remoteInput[slot].addr += ctx.inputOffset;
                ctx.remoteInput[slot].token = ctx.remoteToken[slot];
                const uint32_t signal = 1U << (slot + 1);
                ccu::Read(
                    arg->channels[slot], ctx.scratch[slot],
                    ctx.remoteInput[slot], ctx.activeBytes,
                    ctx.event, signal);
                readyMask |= signal;
            }
            channelBegin = scratchCount;
        } else if (arg->channelCount != 0) {
            // Peer0 is the canonical direct seed.  The next three ascending
            // peers use the three thread-owned scratch tiles.
            ctx.remoteInput[0].addr = ctx.remoteInputBase[0];
            ctx.remoteInput[0].addr += ctx.inputOffset;
            ctx.remoteInput[0].token = ctx.remoteToken[0];
            ccu::Read(
                arg->channels[0], ctx.finalOutput,
                ctx.remoteInput[0], ctx.activeBytes,
                ctx.event, 1);
            readyMask |= 1;
            channelBegin = 1;

            scratchCount = arg->channelCount - channelBegin;
            if (scratchCount > 3) {
                scratchCount = 3;
            }
            for (uint32_t slot = 0; slot < scratchCount; ++slot) {
                const uint32_t channel = channelBegin + slot;
                ctx.remoteInput[channel].addr =
                    ctx.remoteInputBase[channel];
                ctx.remoteInput[channel].addr += ctx.inputOffset;
                ctx.remoteInput[channel].token =
                    ctx.remoteToken[channel];
                const uint32_t signal = 1U << (slot + 1);
                ccu::Read(
                    arg->channels[channel], ctx.scratch[slot],
                    ctx.remoteInput[channel], ctx.activeBytes,
                    ctx.event, signal);
                readyMask |= signal;
            }
            channelBegin += scratchCount;
        }

        if (readyMask != 0) {
            ccu::EventWait(ctx.event, readyMask);
        }
        for (uint32_t slot = 0; slot < scratchCount; ++slot) {
            ccu::LocalReduce(
                ctx.finalOutput, ctx.scratch[slot], ctx.activeBytes,
                arg->dataType, arg->reduceOp, ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
        }

        // Contributions outside the frontier stay fused and strictly
        // ascending, so network bytes and the canonical FP32 tree are stable.
        for (uint32_t channel = channelBegin;
            channel < arg->channelCount; ++channel) {
            ctx.remoteInput[channel].addr =
                ctx.remoteInputBase[channel];
            ctx.remoteInput[channel].addr += ctx.inputOffset;
            ctx.remoteInput[channel].token = ctx.remoteToken[channel];

            ccu::ReadReduce(
                arg->channels[channel], ctx.finalOutput,
                ctx.remoteInput[channel], ctx.activeBytes,
                arg->dataType, arg->reduceOp, ctx.event, 1);
            ccu::EventWait(ctx.event, 1);
        }
    }
    return CCU_SUCCESS;
}

void BindPipelineOutput(ReduceScatterContext &ctx, uint32_t chunk)
{
    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.addr += ctx.outputOffset;
    for (uint32_t index = 0; index < chunk; ++index) {
        // Scratch banks use the maximum slot stride, but data coordinates
        // advance by the actual non-uniform chunk prefix.
        ctx.finalOutput.addr += ctx.pipelineChunkSize[index];
    }
    ctx.finalOutput.token = ctx.localToken;
}

// F1R43 v3 keeps the accepted teammate network graph and applies one
// independently proved Pareto improvement: on a non-primary kernel, peer0
// receives directly into that kernel's private partial root.  This removes
// one full-chunk scratch-to-partial LocalCopy without changing the remote
// Read set, fixed contribution tree, merge order, or network bytes.
void BindTeammateOutput(ReduceScatterContext &ctx, uint32_t chunk)
{
    ctx.finalOutput.addr = ctx.localOutputBase;
    for (uint32_t index = 0; index < chunk; ++index) {
        ctx.finalOutput.addr += ctx.pipelineSlotStride;
    }
    ctx.finalOutput.token = ctx.localToken;
}

void IssueTeammateChunk(ReduceScatterContext &ctx, uint32_t chunk)
{
    const auto *arg = ctx.arg;
    ccu::Event &readEvent =
        ctx.pipelineReadEvent[chunk % RS_PIPELINE_BANKS];

    ccu::Variable inputOffset;
    inputOffset = ctx.inputOffset;
    for (uint32_t index = 0; index < chunk; ++index) {
        inputOffset += ctx.pipelineChunkSize[index];
    }

    if (arg->includeSelf) {
        ctx.localInput.addr = ctx.localInputBase;
        ctx.localInput.addr += inputOffset;
        ctx.localInput.token = ctx.localToken;
        BindTeammateOutput(ctx, chunk);
        ccu::LocalCopy(
            ctx.finalOutput, ctx.localInput,
            ctx.pipelineChunkSize[chunk],
            readEvent, 1U << RS_PIPELINE_SELF_EVENT_BIT);
    }

    ccu::Variable slotOffset;
    slotOffset = 0;
    for (uint32_t bank = 0;
        bank < chunk % RS_PIPELINE_BANKS; ++bank) {
        slotOffset += ctx.pipelineSetStride;
    }
    for (uint32_t channel = 0;
        channel < arg->channelCount; ++channel) {
        if (!arg->includeSelf && channel == 0) {
            BindTeammateOutput(ctx, chunk);
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel].addr =
                ctx.finalOutput.addr;
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel].token =
                ctx.finalOutput.token;
        } else {
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel].addr =
                ctx.localScratchBase[0];
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel].addr +=
                slotOffset;
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel].token =
                ctx.localToken;
            slotOffset += ctx.pipelineSlotStride;
        }

        ctx.remoteInput[channel].addr =
            ctx.remoteInputBase[channel];
        ctx.remoteInput[channel].addr += inputOffset;
        ctx.remoteInput[channel].token =
            ctx.remoteToken[channel];
        ccu::Read(
            arg->channels[channel],
                ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel],
            ctx.remoteInput[channel],
            ctx.pipelineChunkSize[chunk],
            readEvent, 1U << channel);
    }
}

void ReduceTeammateChunkTree(
    ReduceScatterContext &ctx, uint32_t chunk)
{
    const auto *arg = ctx.arg;
    const uint32_t peerCount = arg->channelCount;
    ccu::Event &readEvent =
        ctx.pipelineReadEvent[chunk % RS_PIPELINE_BANKS];

    uint32_t waitMask = (1U << peerCount) - 1;
    if (arg->includeSelf) {
        waitMask |= 1U << RS_PIPELINE_SELF_EVENT_BIT;
    }
    ccu::EventWait(readEvent, waitMask);

    uint32_t live = peerCount;
    while (live > 1) {
        uint32_t half = 1;
        while (half * 2 < live) {
            half *= 2;
        }
        uint32_t stepMask = 0;
        for (uint32_t index = 0;
            index + half < live; ++index) {
            ccu::LocalReduce(
                ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][index],
                ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][index + half],
                ctx.pipelineChunkSize[chunk],
                arg->dataType, arg->reduceOp,
                ctx.pipelineReduceEvent, 1U << index);
            stepMask |= 1U << index;
        }
        ccu::EventWait(ctx.pipelineReduceEvent, stepMask);
        live = half;
    }

    BindTeammateOutput(ctx, chunk);
    if (arg->includeSelf) {
        ccu::LocalReduce(
            ctx.finalOutput,
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][0],
            ctx.pipelineChunkSize[chunk],
            arg->dataType, arg->reduceOp,
            ctx.pipelineReduceEvent, 1U);
        ccu::EventWait(ctx.pipelineReduceEvent, 1U);
    }
}

void ReduceTeammateTreePipeline(ReduceScatterContext &ctx)
{
    IssueTeammateChunk(ctx, 0);
    CCU_IF(ctx.pipelineChunkSize[1] != 0) {
        IssueTeammateChunk(ctx, 1);
    }
    ReduceTeammateChunkTree(ctx, 0);
    CCU_IF(ctx.pipelineChunkSize[2] != 0) {
        IssueTeammateChunk(ctx, 2);
    }
    CCU_IF(ctx.pipelineChunkSize[1] != 0) {
        ReduceTeammateChunkTree(ctx, 1);
    }
    CCU_IF(ctx.pipelineChunkSize[3] != 0) {
        IssueTeammateChunk(ctx, 3);
    }
    CCU_IF(ctx.pipelineChunkSize[2] != 0) {
        ReduceTeammateChunkTree(ctx, 2);
    }
    CCU_IF(ctx.pipelineChunkSize[3] != 0) {
        ReduceTeammateChunkTree(ctx, 3);
    }
}

void IssuePipelineChunk(ReduceScatterContext &ctx, uint32_t chunk)
{
    const auto *arg = ctx.arg;
    ccu::Event &readEvent =
        ctx.pipelineReadEvent[chunk % RS_PIPELINE_BANKS];

    ccu::Variable inputOffset;
    inputOffset = ctx.inputOffset;
    for (uint32_t index = 0; index < chunk; ++index) {
        inputOffset += ctx.pipelineChunkSize[index];
    }

    if (arg->includeSelf) {
        ctx.localInput.addr = ctx.localInputBase;
        ctx.localInput.addr += inputOffset;
        ctx.localInput.token = ctx.localToken;
        BindPipelineOutput(ctx, chunk);
        ccu::LocalCopy(
            ctx.finalOutput, ctx.localInput,
            ctx.pipelineChunkSize[chunk], readEvent,
            1U << RS_PIPELINE_SELF_EVENT_BIT);
    }

    ccu::Variable slotOffset;
    slotOffset = 0;
    for (uint32_t bank = 0;
        bank < chunk % RS_PIPELINE_BANKS; ++bank) {
        slotOffset += ctx.pipelineSetStride;
    }
    for (uint32_t channel = 0;
        channel < arg->channelCount; ++channel) {
        if (!arg->includeSelf && channel == 0) {
            // A non-primary kernel owns a private partial.  Make peer0's
            // receive that tree's canonical root, so the fixed tree reduces
            // in place and no terminal scratch-to-partial LocalCopy is needed.
            // This does not alias any peer: every later channel still owns a
            // distinct parity scratch slot, and every chunk owns a disjoint
            // partial interval.
            BindPipelineOutput(ctx, chunk);
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel].addr =
                ctx.finalOutput.addr;
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel].token =
                ctx.finalOutput.token;
        } else {
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel].addr =
                ctx.localScratchBase[0];
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel].addr +=
                slotOffset;
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel].token =
                ctx.localToken;
            slotOffset += ctx.pipelineSlotStride;
        }

        ctx.remoteInput[channel].addr =
            ctx.remoteInputBase[channel];
        ctx.remoteInput[channel].addr += inputOffset;
        ctx.remoteInput[channel].token =
            ctx.remoteToken[channel];
        ccu::Read(
            arg->channels[channel],
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][channel],
            ctx.remoteInput[channel],
            ctx.pipelineChunkSize[chunk],
            readEvent, 1U << channel);
    }
}

void ReducePipelineChunkTree(
    ReduceScatterContext &ctx, uint32_t chunk)
{
    const auto *arg = ctx.arg;
    const uint32_t peerCount = arg->channelCount;
    ccu::Event &readEvent =
        ctx.pipelineReadEvent[chunk % RS_PIPELINE_BANKS];

    uint32_t waitMask = (1U << peerCount) - 1;
    if (arg->includeSelf) {
        waitMask |= 1U << RS_PIPELINE_SELF_EVENT_BIT;
    }
    ccu::EventWait(readEvent, waitMask);

    // The tree is fixed by ascending channel ordinal.  Every peer appears
    // exactly once and every level waits before the next level.  This changes
    // FP32 parentheses relative to the linear path but is deterministic; the
    // complete VM Checker/Semantic gates remain mandatory for tolerance.
    uint32_t live = peerCount;
    while (live > 1) {
        uint32_t half = 1;
        while (half * 2 < live) {
            half *= 2;
        }
        uint32_t stepMask = 0;
        for (uint32_t index = 0;
            index + half < live; ++index) {
            ccu::LocalReduce(
                ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][index],
                ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][index + half],
                ctx.pipelineChunkSize[chunk],
                arg->dataType, arg->reduceOp,
                ctx.pipelineReduceEvent, 1U << index);
            stepMask |= 1U << index;
        }
        ccu::EventWait(ctx.pipelineReduceEvent, stepMask);
        live = half;
    }

    BindPipelineOutput(ctx, chunk);
    if (arg->includeSelf) {
        ccu::LocalReduce(
            ctx.finalOutput,
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][0],
            ctx.pipelineChunkSize[chunk],
            arg->dataType, arg->reduceOp,
            ctx.pipelineReduceEvent, 1);
        ccu::EventWait(ctx.pipelineReduceEvent, 1);
    }
}

CcuResult ReducePipelineChunkReadyTree(
    ReduceScatterContext &ctx, uint32_t chunk)
{
    const auto *arg = ctx.arg;
    const uint32_t peerCount = arg->channelCount;
    ccu::Event &readEvent =
        ctx.pipelineReadEvent[chunk % RS_PIPELINE_BANKS];

    if (peerCount == 1) {
        ccu::EventWait(readEvent, 1U);
    } else {
        uint32_t half = 1;
        while (half * 2 < peerCount) {
            half *= 2;
        }
        const uint32_t paired = peerCount - half;
        uint32_t firstLevelMask = 0;
        for (uint32_t index = 0; index < paired; ++index) {
            const uint32_t pairMask =
                (1U << index) | (1U << (index + half));
            ccu::EventWait(readEvent, pairMask);
            ccu::LocalReduce(
                ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][index],
                ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][index + half],
                ctx.pipelineChunkSize[chunk],
                arg->dataType, arg->reduceOp,
                ctx.pipelineReduceEvent, 1U << index);
            firstLevelMask |= 1U << index;
        }
        // A non-power-of-two tree carries the untouched middle destinations
        // into the next level.  Consume their Read completion bits exactly
        // once while the first pair reductions are in flight.
        for (uint32_t index = paired; index < half; ++index) {
            ccu::EventWait(readEvent, 1U << index);
        }
        ccu::EventWait(ctx.pipelineReduceEvent, firstLevelMask);

        uint32_t live = half;
        while (live > 1) {
            uint32_t nextHalf = 1;
            while (nextHalf * 2 < live) {
                nextHalf *= 2;
            }
            uint32_t stepMask = 0;
            for (uint32_t index = 0;
                index + nextHalf < live; ++index) {
                ccu::LocalReduce(
                    ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][index],
                    ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][index + nextHalf],
                    ctx.pipelineChunkSize[chunk],
                    arg->dataType, arg->reduceOp,
                    ctx.pipelineReduceEvent, 1U << index);
                stepMask |= 1U << index;
            }
            ccu::EventWait(ctx.pipelineReduceEvent, stepMask);
            live = nextHalf;
        }
    }

    BindPipelineOutput(ctx, chunk);
    if (arg->includeSelf) {
        // Self LocalCopy is independent of the remote tree.  Waiting here,
        // rather than in the initial full mask, overlaps it with every remote
        // Read and first-level pair reduction.
        ccu::EventWait(
            readEvent, 1U << RS_PIPELINE_SELF_EVENT_BIT);
        ccu::LocalReduce(
            ctx.finalOutput,
            ctx.pipelinePeerSlot[chunk % RS_PIPELINE_BANKS][0],
            ctx.pipelineChunkSize[chunk],
            arg->dataType, arg->reduceOp,
            ctx.pipelineReduceEvent, 1U);
        ccu::EventWait(ctx.pipelineReduceEvent, 1U);
    }
    return CCU_SUCCESS;
}

CcuResult ReduceSingleChunkReadyTree(ReduceScatterContext &ctx)
{
    IssuePipelineChunk(ctx, 0);
    return ReducePipelineChunkReadyTree(ctx, 0);
}

CcuResult ReduceFusedReadyTreePipeline(ReduceScatterContext &ctx)
{
    IssuePipelineChunk(ctx, 0);
    CCU_IF(ctx.pipelineChunkSize[1] != 0) {
        IssuePipelineChunk(ctx, 1);
    }
    CCU_IF(ctx.pipelineChunkSize[2] != 0) {
        CCU_CHK_RET(ReducePipelineChunkReadyTree(ctx, 0));
        IssuePipelineChunk(ctx, 2);
    }
    CCU_IF(ctx.pipelineChunkSize[3] != 0) {
        CCU_CHK_RET(ReducePipelineChunkReadyTree(ctx, 1));
        IssuePipelineChunk(ctx, 3);
    }
    // Every network Read is now queued.  Per-channel ordering makes this
    // record a precise read-completion fence while the remaining local trees
    // execute independently.
    PostSyncRecord(ctx);
    CCU_IF(ctx.pipelineChunkSize[2] == 0) {
        CCU_CHK_RET(ReducePipelineChunkReadyTree(ctx, 0));
    }
    CCU_IF(ctx.pipelineChunkSize[1] != 0) {
        CCU_IF(ctx.pipelineChunkSize[3] == 0) {
            CCU_CHK_RET(ReducePipelineChunkReadyTree(ctx, 1));
        }
    }
    CCU_IF(ctx.pipelineChunkSize[2] != 0) {
        CCU_CHK_RET(ReducePipelineChunkReadyTree(ctx, 2));
    }
    CCU_IF(ctx.pipelineChunkSize[3] != 0) {
        CCU_CHK_RET(ReducePipelineChunkReadyTree(ctx, 3));
    }
    PostSyncWait(ctx);
    return CCU_SUCCESS;
}

CcuResult ReduceTreePipeline(ReduceScatterContext &ctx)
{
    IssuePipelineChunk(ctx, 0);
    CCU_IF(ctx.pipelineChunkSize[1] != 0) {
        IssuePipelineChunk(ctx, 1);
    }
    ReducePipelineChunkTree(ctx, 0);
    CCU_IF(ctx.pipelineChunkSize[2] != 0) {
        IssuePipelineChunk(ctx, 2);
    }
    CCU_IF(ctx.pipelineChunkSize[1] != 0) {
        ReducePipelineChunkTree(ctx, 1);
    }
    CCU_IF(ctx.pipelineChunkSize[3] != 0) {
        IssuePipelineChunk(ctx, 3);
    }
    CCU_IF(ctx.pipelineChunkSize[2] != 0) {
        ReducePipelineChunkTree(ctx, 2);
    }
    CCU_IF(ctx.pipelineChunkSize[3] != 0) {
        ReducePipelineChunkTree(ctx, 3);
    }
    return CCU_SUCCESS;
}

CcuResult MergePipelinePartials(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    // A private partial is contiguous over the active super-chunk.  Merge it
    // with one LocalReduce instead of one command and wait per internal chunk.
    // Different partials are still launched by Host in deterministic kernel
    // order, so the cross-group FP32 contribution order is unchanged.
    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.addr += ctx.outputOffset;
    ctx.finalOutput.token = ctx.localToken;
    ctx.scratch[0].addr = ctx.localScratchBase[0];
    ctx.scratch[0].token = ctx.localToken;
    ccu::LocalReduce(
        ctx.finalOutput, ctx.scratch[0],
        ctx.activeBytes,
        arg->dataType, arg->reduceOp,
        ctx.pipelineReduceEvent, 1U);
    ccu::EventWait(ctx.pipelineReduceEvent, 1U);
    return CCU_SUCCESS;
}

CcuResult MergePipelinePartialBatch(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.addr += ctx.outputOffset;
    ctx.finalOutput.token = ctx.localToken;

    ctx.scratch[0].addr = ctx.localScratchBase[0];
    ctx.scratch[0].token = ctx.localToken;
    for (uint32_t ordinal = 0;
        ordinal < RS_PIPELINE_CHUNKS; ++ordinal) {
        CCU_IF(ctx.pipelineChunkSize[ordinal] != 0) {
            ccu::LocalReduce(
                ctx.finalOutput, ctx.scratch[0],
                ctx.pipelineChunkSize[ordinal],
                arg->dataType, arg->reduceOp,
                ctx.pipelineReduceEvent, 1U);
            ccu::EventWait(ctx.pipelineReduceEvent, 1U);
            ctx.scratch[0].addr += ctx.pipelineSlotStride;
        }
    }
    return CCU_SUCCESS;
}

CcuResult MergePrivatePartial(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.addr += ctx.outputOffset;
    ctx.finalOutput.token = ctx.localToken;

    ctx.scratch[0].addr = ctx.localScratchBase[0];
    ctx.scratch[0].token = ctx.localToken;

    CCU_IF(ctx.activeBytes != 0) {
        ccu::LocalReduce(
            ctx.finalOutput, ctx.scratch[0], ctx.activeBytes,
            arg->dataType, arg->reduceOp, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterLayerDiePairwiseKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    ReduceScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.operationMode == 15) {
        CCU_CHK_RET(MergePipelinePartialBatch(ctx));
    } CCU_ELSE {
    CCU_IF(ctx.operationMode == 9) {
        CCU_CHK_RET(MergePipelinePartials(ctx));
    } CCU_ELSE {
        CCU_IF(ctx.operationMode == 1) {
            CCU_CHK_RET(MergePrivatePartial(ctx));
        } CCU_ELSE {
            PreSync(ctx);
            CCU_IF(ctx.operationMode == 16) {
                ReduceTeammateTreePipeline(ctx);
            } CCU_ELSE {
            CCU_IF(ctx.operationMode == 17) {
                CCU_CHK_RET(ReduceFusedReadyTreePipeline(ctx));
            } CCU_ELSE {
            CCU_IF(ctx.operationMode == 13) {
                CCU_CHK_RET(ReduceSingleChunkReadyTree(ctx));
            } CCU_ELSE {
            CCU_IF(ctx.operationMode == 12) {
                CCU_CHK_RET(ReduceTreePipeline(ctx));
                } CCU_ELSE {
                CCU_IF(ctx.operationMode == 7) {
                    CCU_CHK_RET(
                        ReduceOneChunkPhaseSharedFrontier4(ctx));
                } CCU_ELSE {
                    CCU_IF(ctx.operationMode == 6) {
                        CCU_CHK_RET(
                            ReduceOneChunkFrontier3ReadReduce(ctx));
                    } CCU_ELSE {
                        CCU_IF(ctx.operationMode == 5) {
                            CCU_CHK_RET(
                                ReduceOneChunkSeedOverlapReadReduce(ctx));
                        } CCU_ELSE {
                            CCU_IF(ctx.operationMode == 4) {
                                CCU_CHK_RET(
                                    ReduceOneChunkMultiKernelReadReduce(
                                        ctx));
                            } CCU_ELSE {
                                CCU_IF(ctx.operationMode == 3) {
                                    CCU_CHK_RET(
                                        ReduceOneChunkReadReduce(ctx));
                                } CCU_ELSE {
                                    CCU_IF(ctx.operationMode == 2) {
                                        CCU_CHK_RET(
                                            ReduceOneChunkDepth2(ctx));
                                    } CCU_ELSE {
                                        CCU_CHK_RET(
                                            ReduceOneChunk(ctx));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            }
            }
            }
            CCU_IF(ctx.operationMode != 17) {
                PostSync(ctx);
            }
        }
    }
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterSmallReadyKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    ReduceScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.operationMode == 15) {
        CCU_CHK_RET(MergePipelinePartialBatch(ctx));
    } CCU_ELSE {
        CCU_IF(ctx.operationMode == 9) {
            CCU_CHK_RET(MergePipelinePartials(ctx));
        } CCU_ELSE {
            CCU_IF(ctx.operationMode == 1) {
                CCU_CHK_RET(MergePrivatePartial(ctx));
            } CCU_ELSE {
                PreSync(ctx);
                CCU_CHK_RET(ReduceSingleChunkReadyTree(ctx));
                PostSync(ctx);
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterFusedReadyKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    ReduceScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.operationMode == 15) {
        CCU_CHK_RET(MergePipelinePartialBatch(ctx));
    } CCU_ELSE {
        CCU_IF(ctx.operationMode == 9) {
            CCU_CHK_RET(MergePipelinePartials(ctx));
        } CCU_ELSE {
            CCU_IF(ctx.operationMode == 1) {
                CCU_CHK_RET(MergePrivatePartial(ctx));
            } CCU_ELSE {
                PreSync(ctx);
                CCU_CHK_RET(ReduceFusedReadyTreePipeline(ctx));
                PostSync(ctx);
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterFlatTreeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    ReduceScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.operationMode == 15) {
        CCU_CHK_RET(MergePipelinePartialBatch(ctx));
    } CCU_ELSE {
        CCU_IF(ctx.operationMode == 9) {
            CCU_CHK_RET(MergePipelinePartials(ctx));
        } CCU_ELSE {
            CCU_IF(ctx.operationMode == 1) {
                CCU_CHK_RET(MergePrivatePartial(ctx));
            } CCU_ELSE {
                PreSync(ctx);
                CCU_CHK_RET(ReduceTreePipeline(ctx));
                PostSync(ctx);
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterTeammateTreeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    ReduceScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.operationMode == 15) {
        CCU_CHK_RET(MergePipelinePartialBatch(ctx));
    } CCU_ELSE {
        CCU_IF(ctx.operationMode == 9) {
            CCU_CHK_RET(MergePipelinePartials(ctx));
        } CCU_ELSE {
            CCU_IF(ctx.operationMode == 1) {
                CCU_CHK_RET(MergePrivatePartial(ctx));
            } CCU_ELSE {
                PreSync(ctx);
                ReduceTeammateTreePipeline(ctx);
                PostSync(ctx);
            }
        }
    }
    return CCU_SUCCESS;
}

// Rank-4 SMALL is a fixed two-stage recursive-halving graph. The xor-2
// partner first forms two-rank partials for the retained target half. The
// xor-1 partner then supplies the complementary two-rank partial for this
// rank's output. Both generations close before their source is reused.
CcuResult CcuReduceScatterRank4SmallRhKernel(CcuKernelArg arg)
{
    constexpr int INPUT_XN_ID = 0;
    constexpr int TOKEN_XN_ID = 1;
    constexpr int DONE_NOTIFY_ID = 2;
    constexpr int CKE_INDEX = 0;
    constexpr uint32_t INPUT_MASK = 1U << INPUT_XN_ID;
    constexpr uint32_t TOKEN_MASK = 1U << TOKEN_XN_ID;
    constexpr uint32_t READY_MASK = INPUT_MASK | TOKEN_MASK;
    constexpr uint32_t DONE_MASK = 1U << DONE_NOTIFY_ID;

    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    if (kernelArg->channelCount != 3 ||
        kernelArg->rank4SmallMyRank >= 4 ||
        kernelArg->rank4SmallXorHighChannelSlot >=
            kernelArg->channelCount ||
        kernelArg->rank4SmallXorLowChannelSlot >=
            kernelArg->channelCount ||
        kernelArg->rank4SmallXorHighChannelSlot ==
            kernelArg->rank4SmallXorLowChannelSlot) {
        return CcuResult::CCU_E_PARA;
    }

    const ChannelHandle highChannel =
        kernelArg->channels[
            kernelArg->rank4SmallXorHighChannelSlot];
    const ChannelHandle lowChannel =
        kernelArg->channels[
            kernelArg->rank4SmallXorLowChannelSlot];

    ccu::Variable remoteHighBase;
    ccu::Variable remoteHighToken;
    ccu::Variable remoteLowBase;
    ccu::Variable remoteLowToken;
    remoteHighBase =
        ccu::GetResByChannel<ccu::Variable>(
            highChannel, INPUT_XN_ID);
    remoteHighToken =
        ccu::GetResByChannel<ccu::Variable>(
            highChannel, TOKEN_XN_ID);
    remoteLowBase =
        ccu::GetResByChannel<ccu::Variable>(
            lowChannel, INPUT_XN_ID);
    remoteLowToken =
        ccu::GetResByChannel<ccu::Variable>(
            lowChannel, TOKEN_XN_ID);

    ccu::Variable localInputBase;
    ccu::Variable localOutputBase;
    ccu::Variable localInputToken;
    ccu::Variable localOutputToken;
    ccu::Variable localScratchBase;
    ccu::Variable localScratchToken;
    ccu::Variable sliceBytes;
    CCU_CHK_RET(ccu::LoadArg(localInputBase, 0));
    CCU_CHK_RET(ccu::LoadArg(localOutputBase, 1));
    CCU_CHK_RET(ccu::LoadArg(localInputToken, 2));
    CCU_CHK_RET(ccu::LoadArg(localOutputToken, 3));
    CCU_CHK_RET(ccu::LoadArg(localScratchBase, 4));
    CCU_CHK_RET(ccu::LoadArg(localScratchToken, 5));
    CCU_CHK_RET(ccu::LoadArg(sliceBytes, 6));

    const uint32_t halfFirstSlice =
        kernelArg->rank4SmallMyRank & 2U;
    const uint32_t ownOrdinal =
        kernelArg->rank4SmallMyRank & 1U;

    ccu::Variable halfBytes;
    halfBytes = sliceBytes;
    halfBytes += sliceBytes;

    ccu::LocalAddr localInputHalf;
    localInputHalf.addr = localInputBase;
    for (uint32_t index = 0;
        index < halfFirstSlice; ++index) {
        localInputHalf.addr += sliceBytes;
    }
    localInputHalf.token = localInputToken;

    ccu::LocalAddr localHalf;
    localHalf.addr = localScratchBase;
    localHalf.token = localScratchToken;

    ccu::LocalAddr remoteHalf;
    remoteHalf.addr = localScratchBase;
    remoteHalf.addr += halfBytes;
    remoteHalf.token = localScratchToken;

    ccu::Event event;

    ccu::WriteVariableWithNotify(
        highChannel, localInputBase,
        INPUT_XN_ID, CKE_INDEX, INPUT_MASK);
    ccu::WriteVariableWithNotify(
        highChannel, localInputToken,
        TOKEN_XN_ID, CKE_INDEX, TOKEN_MASK);
    ccu::NotifyWait(
        highChannel, CKE_INDEX, READY_MASK);

    // XN-backed remote values may only be materialized after the matching
    // ready notification.  R138 proved that constructing this address above
    // NotifyWait emits an early LoadGSAXn and fails CheckerV3 ErrorCode 108.
    ccu::RemoteAddr remoteInputHalf;
    remoteInputHalf.addr = remoteHighBase;
    for (uint32_t index = 0;
        index < halfFirstSlice; ++index) {
        remoteInputHalf.addr += sliceBytes;
    }
    remoteInputHalf.token = remoteHighToken;

    ccu::LocalCopy(
        localHalf, localInputHalf, halfBytes,
        event, 1U);
    ccu::Read(
        highChannel, remoteHalf, remoteInputHalf,
        halfBytes, event, 2U);
    ccu::EventWait(event, 3U);
    ccu::LocalReduce(
        localHalf, remoteHalf, halfBytes,
        kernelArg->dataType, kernelArg->reduceOp,
        event, 1U);
    ccu::EventWait(event, 1U);

    ccu::NotifyRecord(
        highChannel, CKE_INDEX, DONE_MASK);
    ccu::NotifyWait(
        highChannel, CKE_INDEX, DONE_MASK);

    ccu::WriteVariableWithNotify(
        lowChannel, localScratchBase,
        INPUT_XN_ID, CKE_INDEX, INPUT_MASK);
    ccu::WriteVariableWithNotify(
        lowChannel, localScratchToken,
        TOKEN_XN_ID, CKE_INDEX, TOKEN_MASK);
    ccu::NotifyWait(
        lowChannel, CKE_INDEX, READY_MASK);

    ccu::LocalAddr localTarget;
    localTarget.addr = localScratchBase;
    for (uint32_t index = 0;
        index < ownOrdinal; ++index) {
        localTarget.addr += sliceBytes;
    }
    localTarget.token = localScratchToken;

    ccu::RemoteAddr remoteTarget;
    remoteTarget.addr = remoteLowBase;
    for (uint32_t index = 0;
        index < ownOrdinal; ++index) {
        remoteTarget.addr += sliceBytes;
    }
    remoteTarget.token = remoteLowToken;

    ccu::LocalAddr finalOutput;
    finalOutput.addr = localOutputBase;
    finalOutput.token = localOutputToken;

    ccu::LocalCopy(
        finalOutput, localTarget, sliceBytes,
        event, 1U);
    ccu::Read(
        lowChannel, remoteHalf, remoteTarget,
        sliceBytes, event, 2U);
    ccu::EventWait(event, 3U);
    ccu::LocalReduce(
        finalOutput, remoteHalf, sliceBytes,
        kernelArg->dataType, kernelArg->reduceOp,
        event, 1U);
    ccu::EventWait(event, 1U);

    ccu::NotifyRecord(
        lowChannel, CKE_INDEX, DONE_MASK);
    ccu::NotifyWait(
        lowChannel, CKE_INDEX, DONE_MASK);
    return CCU_SUCCESS;
}

// P10 keeps the measured F1R57 mode-13/15 graph and Host schedule.  This
// dedicated entry removes modes that the normalized rank-16 SMALL cell can
// never issue, without changing any command, address or contribution edge.
CcuResult CcuReduceScatterRank16SmallThinKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    ReduceScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.operationMode == 15) {
        CCU_CHK_RET(MergePipelinePartialBatch(ctx));
    } CCU_ELSE {
        PreSync(ctx);
        CCU_CHK_RET(ReduceSingleChunkReadyTree(ctx));
        PostSync(ctx);
    }
    return CCU_SUCCESS;
}

namespace teammate_exact_kernel {
constexpr uint32_t RS_MAX_GROUP_PEERS = 8;
constexpr uint32_t RS_MAX_CHUNKS = 4;
constexpr uint32_t RS_BANKS = 2;
constexpr int RS_INPUT_XN_ID = 0;
constexpr int RS_TOKEN_XN_ID = 1;
constexpr int RS_POST_SYNC_ID = 2;
constexpr int RS_CKE_INDEX = 0;
constexpr uint32_t RS_SELF_EVENT_BIT = 8; // peer 读占用 bit0..7

// taskArgs 布局（与 LoadArg 顺序一致）：
// 0 inputBase, 1 outputBase, 2 token, 3 scratchBase,
// 4 rankSliceOffset, 5 slotStride, 6 setStride,
// 7..10 size0..size3, 11 operationMode

struct ReduceScatterContext {
    const ReduceScatterKernelArg *arg;

    ccu::Variable remoteInputBase[RS_MAX_GROUP_PEERS];
    ccu::Variable remoteToken[RS_MAX_GROUP_PEERS];
    ccu::Variable localInputBase;
    ccu::Variable localOutputBase;
    ccu::Variable localToken;
    ccu::Variable localScratchBase;
    ccu::Variable rankSliceOffset;
    ccu::Variable slotStride;
    ccu::Variable setStride;
    ccu::Variable chunkSize[RS_MAX_CHUNKS];
    ccu::Variable operationMode;

    ccu::RemoteAddr remoteInput[RS_MAX_GROUP_PEERS];
    ccu::LocalAddr peerSlot[RS_BANKS][RS_MAX_GROUP_PEERS];
    ccu::LocalAddr localInput;
    ccu::LocalAddr finalOutput;
    ccu::LocalAddr scratch;
    ccu::Event readEvent[RS_BANKS];
    ccu::Event reduceEvent;
};

CcuResult InitResource(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->channelCount == 0 ||
        arg->channelCount > RS_MAX_GROUP_PEERS) {
        return CcuResult::CCU_E_PARA;
    }

    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ctx.remoteInputBase[channel] =
            ccu::GetResByChannel<ccu::Variable>(
                arg->channels[channel], RS_INPUT_XN_ID);
        ctx.remoteToken[channel] =
            ccu::GetResByChannel<ccu::Variable>(
                arg->channels[channel], RS_TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(ReduceScatterContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.localInputBase, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localOutputBase, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localScratchBase, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rankSliceOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.slotStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.setStride, argId++));
    for (uint32_t index = 0; index < RS_MAX_CHUNKS; ++index) {
        CCU_CHK_RET(ccu::LoadArg(ctx.chunkSize[index], argId++));
    }
    CCU_CHK_RET(ccu::LoadArg(ctx.operationMode, argId++));
    return CCU_SUCCESS;
}

void PreSync(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint32_t inputMask = 1U << RS_INPUT_XN_ID;
    constexpr uint32_t tokenMask = 1U << RS_TOKEN_XN_ID;
    constexpr uint32_t readyMask = inputMask | tokenMask;

    // Publish every endpoint before waiting on any endpoint.  This keeps the
    // address exchange independent of peer iteration order.
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ccu::WriteVariableWithNotify(
            arg->channels[channel], ctx.localInputBase,
            RS_INPUT_XN_ID, RS_CKE_INDEX, inputMask);
        ccu::WriteVariableWithNotify(
            arg->channels[channel], ctx.localToken,
            RS_TOKEN_XN_ID, RS_CKE_INDEX, tokenMask);
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ccu::NotifyWait(
            arg->channels[channel], RS_CKE_INDEX, readyMask);
    }
}

void PostSync(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint32_t postMask = 1U << RS_POST_SYNC_ID;

    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ccu::NotifyRecord(
            arg->channels[channel], RS_CKE_INDEX, postMask);
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ccu::NotifyWait(
            arg->channels[channel], RS_CKE_INDEX, postMask);
    }
}

void PostSyncRecord(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint32_t postMask = 1U << RS_POST_SYNC_ID;
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ccu::NotifyRecord(
            arg->channels[channel], RS_CKE_INDEX, postMask);
    }
}

void PostSyncWait(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint32_t postMask = 1U << RS_POST_SYNC_ID;
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        ccu::NotifyWait(
            arg->channels[channel], RS_CKE_INDEX, postMask);
    }
}

// 绑定 chunk k 的输出地址；数据坐标按真实非均匀 chunk 前缀推进。
void BindChunkOutput(ReduceScatterContext &ctx, uint32_t k)
{
    ctx.finalOutput.addr = ctx.localOutputBase;
    for (uint32_t index = 0; index < k; ++index) {
        ctx.finalOutput.addr += ctx.chunkSize[index];
    }
    ctx.finalOutput.token = ctx.localToken;
}

// 下发 chunk k 的 self 拷贝与全部 peer 读（只传输，不占归约链）
void IssueChunk(ReduceScatterContext &ctx, uint32_t k)
{
    const auto *arg = ctx.arg;
    ccu::Event &readEvent = ctx.readEvent[k % RS_BANKS];

    // chunk k 的输入偏移：rankSliceOffset + 真实 chunk 前缀。
    ccu::Variable inputOff;
    inputOff = ctx.rankSliceOffset;
    for (uint32_t index = 0; index < k; ++index) {
        inputOff += ctx.chunkSize[index];
    }

    if (arg->includeSelf) {
        ctx.localInput.addr = ctx.localInputBase;
        ctx.localInput.addr += inputOff;
        ctx.localInput.token = ctx.localToken;
        BindChunkOutput(ctx, k);
        ccu::LocalCopy(
            ctx.finalOutput, ctx.localInput, ctx.chunkSize[k],
            readEvent, 1U << RS_SELF_EVENT_BIT);
    }

    ccu::Variable slotOff;
    slotOff = 0;
    for (uint32_t bank = 0; bank < k % RS_BANKS; ++bank) {
        slotOff += ctx.setStride;
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        if (!arg->includeSelf && channel == 0) {
            // Materialize the non-primary root directly in its private
            // partial.  This saves one slot in every bank and one terminal
            // LocalCopy without changing contribution order.
            BindChunkOutput(ctx, k);
            ctx.peerSlot[k % RS_BANKS][channel].addr =
                ctx.finalOutput.addr;
            ctx.peerSlot[k % RS_BANKS][channel].token =
                ctx.finalOutput.token;
        } else {
            ctx.peerSlot[k % RS_BANKS][channel].addr =
                ctx.localScratchBase;
            ctx.peerSlot[k % RS_BANKS][channel].addr += slotOff;
            ctx.peerSlot[k % RS_BANKS][channel].token = ctx.localToken;
            slotOff += ctx.slotStride;
        }

        ctx.remoteInput[channel].addr =
            ctx.remoteInputBase[channel];
        ctx.remoteInput[channel].addr += inputOff;
        ctx.remoteInput[channel].token = ctx.remoteToken[channel];

        ccu::Read(
            arg->channels[channel], ctx.peerSlot[k % RS_BANKS][channel],
            ctx.remoteInput[channel], ctx.chunkSize[k],
            readEvent, 1U << channel);
    }
}

// chunk k 的归约阶段：等齐传输，树形归约（形状固定 -> 确定性），根写入输出
void ReduceChunkTree(ReduceScatterContext &ctx, uint32_t k)
{
    const auto *arg = ctx.arg;
    const uint32_t peerNum = arg->channelCount;
    ccu::Event &readEvent = ctx.readEvent[k % RS_BANKS];

    // 等齐本 chunk 的全部读（以及 self 拷贝）
    uint32_t waitMask = (1U << peerNum) - 1;
    if (arg->includeSelf) {
        waitMask |= (1U << RS_SELF_EVENT_BIT);
    }
    ccu::EventWait(readEvent, waitMask);

    // 树形归约：每步把 slot[i+half] 并入 slot[i]，步内并行、步间串行
    uint32_t live = peerNum;
    while (live > 1) {
        uint32_t half = 1;
        while (half * 2 < live) {
            half *= 2;
        }
        uint32_t stepMask = 0;
        for (uint32_t i = 0; i + half < live; ++i) {
            ccu::LocalReduce(
                ctx.peerSlot[k % RS_BANKS][i],
                ctx.peerSlot[k % RS_BANKS][i + half],
                ctx.chunkSize[k],
                arg->dataType, arg->reduceOp, ctx.reduceEvent, 1U << i);
            stepMask |= (1U << i);
        }
        ccu::EventWait(ctx.reduceEvent, stepMask);
        live = half;
    }

    // 根并入输出：primary 组 self 已在输出中；其余组为覆盖写
    BindChunkOutput(ctx, k);
    if (arg->includeSelf) {
        ccu::LocalReduce(
            ctx.finalOutput, ctx.peerSlot[k % RS_BANKS][0],
            ctx.chunkSize[k],
            arg->dataType, arg->reduceOp, ctx.reduceEvent, 1);
        ccu::EventWait(ctx.reduceEvent, 1);
    }
}

void ReduceChunkReadyTree(ReduceScatterContext &ctx, uint32_t k)
{
    const auto *arg = ctx.arg;
    const uint32_t peerNum = arg->channelCount;
    ccu::Event &readEvent = ctx.readEvent[k % RS_BANKS];

    if (peerNum == 1) {
        ccu::EventWait(readEvent, 1U);
    } else {
        uint32_t half = 1;
        while (half * 2 < peerNum) {
            half *= 2;
        }
        const uint32_t paired = peerNum - half;
        uint32_t firstLevelMask = 0;
        for (uint32_t i = 0; i < paired; ++i) {
            const uint32_t pairMask =
                (1U << i) | (1U << (i + half));
            ccu::EventWait(readEvent, pairMask);
            ccu::LocalReduce(
                ctx.peerSlot[k % RS_BANKS][i],
                ctx.peerSlot[k % RS_BANKS][i + half],
                ctx.chunkSize[k],
                arg->dataType, arg->reduceOp,
                ctx.reduceEvent, 1U << i);
            firstLevelMask |= 1U << i;
        }
        for (uint32_t i = paired; i < half; ++i) {
            ccu::EventWait(readEvent, 1U << i);
        }
        ccu::EventWait(ctx.reduceEvent, firstLevelMask);

        uint32_t live = half;
        while (live > 1) {
            uint32_t nextHalf = 1;
            while (nextHalf * 2 < live) {
                nextHalf *= 2;
            }
            uint32_t stepMask = 0;
            for (uint32_t i = 0; i + nextHalf < live; ++i) {
                ccu::LocalReduce(
                    ctx.peerSlot[k % RS_BANKS][i],
                    ctx.peerSlot[k % RS_BANKS][i + nextHalf],
                    ctx.chunkSize[k],
                    arg->dataType, arg->reduceOp,
                    ctx.reduceEvent, 1U << i);
                stepMask |= 1U << i;
            }
            ccu::EventWait(ctx.reduceEvent, stepMask);
            live = nextHalf;
        }
    }

    BindChunkOutput(ctx, k);
    if (arg->includeSelf) {
        ccu::EventWait(readEvent, 1U << RS_SELF_EVENT_BIT);
        ccu::LocalReduce(
            ctx.finalOutput, ctx.peerSlot[k % RS_BANKS][0],
            ctx.chunkSize[k],
            arg->dataType, arg->reduceOp,
            ctx.reduceEvent, 1U);
        ccu::EventWait(ctx.reduceEvent, 1U);
    }
}

CcuResult RunNetworkMode(ReduceScatterContext &ctx)
{
    PreSync(ctx);

    // Two-bank early-fence wavefront.  A bank is recycled only after its
    // previous chunk has completed its deterministic local tree.  Once every
    // Read is queued, the completion fence overlaps the remaining trees.
    IssueChunk(ctx, 0);
    CCU_IF(ctx.chunkSize[1] != 0) {
        IssueChunk(ctx, 1);
    }
    CCU_IF(ctx.chunkSize[2] != 0) {
        ReduceChunkTree(ctx, 0);
        IssueChunk(ctx, 2);
    }
    CCU_IF(ctx.chunkSize[3] != 0) {
        ReduceChunkTree(ctx, 1);
        IssueChunk(ctx, 3);
    }
    PostSyncRecord(ctx);
    CCU_IF(ctx.chunkSize[2] == 0) {
        ReduceChunkTree(ctx, 0);
    }
    CCU_IF(ctx.chunkSize[1] != 0) {
        CCU_IF(ctx.chunkSize[3] == 0) {
            ReduceChunkTree(ctx, 1);
        }
    }
    CCU_IF(ctx.chunkSize[2] != 0) {
        ReduceChunkTree(ctx, 2);
    }
    CCU_IF(ctx.chunkSize[3] != 0) {
        ReduceChunkTree(ctx, 3);
    }

    PostSyncWait(ctx);
    return CCU_SUCCESS;
}

// P16 is statically a single chunk.  Keep the teammate all-peer/fixed-tree
// sequence but remove the three impossible chunk-condition branches.
CcuResult RunSingleChunkNetworkMode(ReduceScatterContext &ctx)
{
    PreSync(ctx);
    IssueChunk(ctx, 0);
    PostSyncRecord(ctx);
    ReduceChunkTree(ctx, 0);
    PostSyncWait(ctx);
    return CCU_SUCCESS;
}

CcuResult RunReadyNetworkMode(ReduceScatterContext &ctx)
{
    PreSync(ctx);

    IssueChunk(ctx, 0);
    CCU_IF(ctx.chunkSize[1] != 0) {
        IssueChunk(ctx, 1);
    }
    CCU_IF(ctx.chunkSize[2] != 0) {
        ReduceChunkReadyTree(ctx, 0);
        IssueChunk(ctx, 2);
    }
    CCU_IF(ctx.chunkSize[3] != 0) {
        ReduceChunkReadyTree(ctx, 1);
        IssueChunk(ctx, 3);
    }
    PostSyncRecord(ctx);
    CCU_IF(ctx.chunkSize[2] == 0) {
        ReduceChunkReadyTree(ctx, 0);
    }
    CCU_IF(ctx.chunkSize[1] != 0) {
        CCU_IF(ctx.chunkSize[3] == 0) {
            ReduceChunkReadyTree(ctx, 1);
        }
    }
    CCU_IF(ctx.chunkSize[2] != 0) {
        ReduceChunkReadyTree(ctx, 2);
    }
    CCU_IF(ctx.chunkSize[3] != 0) {
        ReduceChunkReadyTree(ctx, 3);
    }

    PostSyncWait(ctx);
    return CCU_SUCCESS;
}

CcuResult MergePrivatePartial(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.finalOutput.addr = ctx.localOutputBase;
    ctx.finalOutput.token = ctx.localToken;

    ctx.scratch.addr = ctx.localScratchBase;
    ctx.scratch.token = ctx.localToken;

    CCU_IF(ctx.chunkSize[0] != 0) {
        ccu::LocalReduce(
            ctx.finalOutput, ctx.scratch, ctx.chunkSize[0],
            arg->dataType, arg->reduceOp, ctx.reduceEvent, 1);
        ccu::EventWait(ctx.reduceEvent, 1);
    }
    return CCU_SUCCESS;
}
CcuResult CcuReduceScatterLayerDiePairwiseKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    ReduceScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.operationMode == 0) {
        CCU_CHK_RET(RunNetworkMode(ctx));
    } CCU_ELSE {
        CCU_CHK_RET(MergePrivatePartial(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterSmallKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    ReduceScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.operationMode == 0) {
        CCU_CHK_RET(RunSingleChunkNetworkMode(ctx));
    } CCU_ELSE {
        CCU_CHK_RET(MergePrivatePartial(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterReadyPipelineKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    ReduceScatterContext ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.operationMode == 0) {
        CCU_CHK_RET(RunReadyNetworkMode(ctx));
    } CCU_ELSE {
        CCU_CHK_RET(MergePrivatePartial(ctx));
    }
    return CCU_SUCCESS;
}

} // namespace teammate_exact_kernel

#endif // CCU_REDUCE_SCATTER_LAYER_DIE_PAIRWISE_H
