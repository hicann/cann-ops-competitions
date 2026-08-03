/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CCU kernel implementation for tree and ordered reduce-scatter/all-gather AllReduce.
//
// Phases:
//   REDUCE_RECEIVE_INIT / REDUCE_WR_RECV_INIT :
//       Parent node: copy input->output first (out-of-place init), then receive child data.
//   REDUCE_SEND_INIT :
//       Child node (leaf, out-of-place): copy input->output first, then send.
//   REDUCE_WR_SEND_INIT :
//       Child node (leaf, out-of-place): WriteReduce input directly into parent output.
//   REDUCE_RECEIVE / REDUCE_WR_RECV :
//       Parent node: receive child data (output already initialized from prior reduce step).
//   REDUCE_SEND / REDUCE_WR_SEND :
//       Child node: send (output already in place from prior init or reduce step).
//   BROADCAST_SEND :
//       Parent node: write own output to child output buffer.
//   BROADCAST_RECEIVE :
//       Child node: passthrough; parent drives the write.
//
// Tree WriteReduce optimization (phases 6-9):
//   Child calls WriteReduce to atomically add into parent output.
//   This eliminates parent's Read+LocalReduce pair and saves one RTT on the critical path.
//   Correctness: tree layers are serialized via PairSync, so FP32 accumulation order is
//   deterministic across runs with identical inputs.
//
// Ordered RHD optimization (phases 10-15):
//   Each XOR peer pair transfers one direction at a time. PairSync separates the
//   low-to-high and high-to-low directions, avoiding simultaneous bidirectional
//   WriteReduce on a single channel. All ranges are aligned by the host.
//
// Bidirectional RHD optimization (phases 16-18):
//   Used only by the four-rank Clos topology, where both peers can transfer
//   concurrently and previous hardware runs established this path as stable.

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call) \
    do { \
        const CcuResult _r = (call); \
        if (_r != CCU_SUCCESS) { \
            return _r; \
        } \
    } while (0)
#endif

namespace {

// Slot indices in the per-channel resource table.
constexpr uint32_t OUTPUT_ADDR_SLOT  = 0;
constexpr uint32_t OUTPUT_TOKEN_SLOT = 1;
constexpr uint32_t POST_SYNC_SLOT    = 2;
constexpr uint32_t NOTIFY_INDEX      = 0;

// Bitmasks for the notify word.
constexpr uint16_t ADDR_MASK  = static_cast<uint16_t>(1u << OUTPUT_ADDR_SLOT);
constexpr uint16_t TOKEN_MASK = static_cast<uint16_t>(1u << OUTPUT_TOKEN_SLOT);
constexpr uint16_t SYNC_MASK  = static_cast<uint16_t>(1u << POST_SYNC_SLOT);
constexpr uint16_t EXCH_MASK  = static_cast<uint16_t>(ADDR_MASK | TOKEN_MASK);

struct KernelContext {
    const CcuKernelArgAllReduce *arg;
    // Task arguments (loaded from CCU arg slots).
    ccu::Variable localInput;
    ccu::Variable localOutput;
    ccu::Variable localInputToken;
    ccu::Variable localOutputToken;
    ccu::Variable remoteOutput;      // exchanged remote address
    ccu::Variable remoteOutputToken; // exchanged remote token
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable chunkSize;
    ccu::Variable phase;
    ccu::Variable transferOffset;
    ccu::Variable transferSize;
    ccu::Variable copyOffset;
    ccu::Variable copySize;
    ccu::Event event;
};

// Load the 12 task arguments from CCU arg slots 0-11.
CcuResult LoadArgs(KernelContext &ctx)
{
    uint32_t slot = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.localInput,       slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localOutput,      slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localInputToken,  slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localOutputToken, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch,          slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken,     slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.chunkSize,        slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.phase,            slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.transferOffset,   slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.transferSize,     slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.copyOffset,       slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.copySize,         slot++));
    return CCU_SUCCESS;
}

// Populate the per-channel resource variables from the channel resource table.
CcuResult InitResources(KernelContext &ctx)
{
    const ChannelHandle ch = ctx.arg->channels[0];
    ctx.remoteOutput      = ccu::GetResByChannel<ccu::Variable>(ch, OUTPUT_ADDR_SLOT);
    ctx.remoteOutputToken = ccu::GetResByChannel<ccu::Variable>(ch, OUTPUT_TOKEN_SLOT);
    return CCU_SUCCESS;
}

// Bidirectional address exchange: each side publishes its output address/token
// and waits until it has received the peer's address/token.
CcuResult ExchangeAddresses(KernelContext &ctx)
{
    const ChannelHandle ch = ctx.arg->channels[0];
    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        ch, ctx.localOutput,      OUTPUT_ADDR_SLOT,  NOTIFY_INDEX, ADDR_MASK));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        ch, ctx.localOutputToken, OUTPUT_TOKEN_SLOT, NOTIFY_INDEX, TOKEN_MASK));
    CCU_CHK_RET(ccu::NotifyWait(ch, NOTIFY_INDEX, EXCH_MASK));
    return CCU_SUCCESS;
}

// Copy input buffer to output buffer (out-of-place initialization).
CcuResult InitializeOutput(KernelContext &ctx)
{
    ccu::LocalAddr src;
    src.addr  = ctx.localInput;
    src.token = ctx.localInputToken;

    ccu::LocalAddr dst;
    dst.addr  = ctx.localOutput;
    dst.token = ctx.localOutputToken;

    CCU_CHK_RET(ccu::LocalCopy(dst, src, ctx.chunkSize, ctx.event, 1));
    CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

// Initialize only the range retained by this rank at the first reduce-scatter level.
CcuResult InitializeRange(KernelContext &ctx)
{
    ccu::LocalAddr src;
    src.addr = ctx.localInput;
    src.addr += ctx.copyOffset;
    src.token = ctx.localInputToken;

    ccu::LocalAddr dst;
    dst.addr = ctx.localOutput;
    dst.addr += ctx.copyOffset;
    dst.token = ctx.localOutputToken;

    CCU_IF(ctx.copySize != 0)
    {
        CCU_CHK_RET(ccu::LocalCopy(dst, src, ctx.copySize, ctx.event, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    }
    return CCU_SUCCESS;
}

// Preserve the local partial in scratch before a same-range bidirectional
// WriteReduce. This avoids reading output while the peer updates it.
CcuResult CopyTransferRangeToScratch(KernelContext &ctx)
{
    ccu::LocalAddr src;
    src.addr = ctx.localOutput;
    src.addr += ctx.transferOffset;
    src.token = ctx.localOutputToken;

    ccu::LocalAddr dst;
    dst.addr = ctx.scratch;
    dst.token = ctx.scratchToken;

    CCU_IF(ctx.transferSize != 0)
    {
        CCU_CHK_RET(ccu::LocalCopy(
            dst, src, ctx.transferSize, ctx.event, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    }
    return CCU_SUCCESS;
}

// Send one aligned reduce-scatter range. Only one side of the pair executes
// this operation in each launch.
CcuResult ReduceScatterWrite(KernelContext &ctx, bool useInput)
{
    ccu::LocalAddr src;
    if (useInput) {
        src.addr = ctx.localInput;
        src.token = ctx.localInputToken;
    } else {
        src.addr = ctx.localOutput;
        src.token = ctx.localOutputToken;
    }
    src.addr += ctx.transferOffset;

    ccu::RemoteAddr dst;
    dst.addr = ctx.remoteOutput;
    dst.addr += ctx.transferOffset;
    dst.token = ctx.remoteOutputToken;

    CCU_IF(ctx.transferSize != 0)
    {
        CCU_CHK_RET(ccu::WriteReduce(ctx.arg->channels[0], dst, src,
            ctx.transferSize, ctx.arg->dataType, ctx.arg->reduceType, ctx.event, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    }
    return CCU_SUCCESS;
}

// Reduce a stable scratch copy into the peer's matching output range.
CcuResult ReduceScratchToPeer(KernelContext &ctx)
{
    ccu::LocalAddr src;
    src.addr = ctx.scratch;
    src.token = ctx.scratchToken;

    ccu::RemoteAddr dst;
    dst.addr = ctx.remoteOutput;
    dst.addr += ctx.transferOffset;
    dst.token = ctx.remoteOutputToken;

    CCU_IF(ctx.transferSize != 0)
    {
        CCU_CHK_RET(ccu::WriteReduce(ctx.arg->channels[0], dst, src,
            ctx.transferSize, ctx.arg->dataType, ctx.arg->reduceType,
            ctx.event, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    }
    return CCU_SUCCESS;
}

// Send one aligned all-gather range. Only one side of the pair executes this
// operation in each launch.
CcuResult AllGatherWrite(KernelContext &ctx)
{
    ccu::LocalAddr src;
    src.addr = ctx.localOutput;
    src.addr += ctx.transferOffset;
    src.token = ctx.localOutputToken;

    ccu::RemoteAddr dst;
    dst.addr = ctx.remoteOutput;
    dst.addr += ctx.transferOffset;
    dst.token = ctx.remoteOutputToken;

    CCU_IF(ctx.transferSize != 0)
    {
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[0], dst, src,
            ctx.transferSize, ctx.event, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    }
    return CCU_SUCCESS;
}

// Parent-side: read child's output into scratch, then reduce into own output.
// Used in the classic pull-based tree reduce (phases REDUCE_RECEIVE / REDUCE_RECEIVE_INIT).
CcuResult ReducePeer(KernelContext &ctx)
{
    ccu::LocalAddr scratchAddr;
    scratchAddr.addr  = ctx.scratch;
    scratchAddr.token = ctx.scratchToken;

    ccu::RemoteAddr remoteAddr;
    remoteAddr.addr  = ctx.remoteOutput;
    remoteAddr.token = ctx.remoteOutputToken;

    // Pull child data into local scratch.
    CCU_CHK_RET(ccu::Read(ctx.arg->channels[0], scratchAddr, remoteAddr,
        ctx.chunkSize, ctx.event, 1));
    CCU_CHK_RET(ccu::EventWait(ctx.event, 1));

    // Reduce scratch into local output (output += scratch).
    ccu::LocalAddr outputAddr;
    outputAddr.addr  = ctx.localOutput;
    outputAddr.token = ctx.localOutputToken;

    CCU_CHK_RET(ccu::LocalReduce(outputAddr, scratchAddr, ctx.chunkSize,
        ctx.arg->dataType, ctx.arg->reduceType, ctx.event, 1));
    CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

// Child-side WriteReduce for an internal node whose output already contains an
// accumulated subtree result.
CcuResult WriteReduceOutputToPeer(KernelContext &ctx)
{
    ccu::LocalAddr srcAddr;
    srcAddr.addr  = ctx.localOutput;
    srcAddr.token = ctx.localOutputToken;

    ccu::RemoteAddr dstAddr;
    dstAddr.addr  = ctx.remoteOutput;
    dstAddr.token = ctx.remoteOutputToken;

    CCU_CHK_RET(ccu::WriteReduce(ctx.arg->channels[0], dstAddr, srcAddr,
        ctx.chunkSize, ctx.arg->dataType, ctx.arg->reduceType, ctx.event, 1));
    CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

// Child-side WriteReduce for an out-of-place leaf. The leaf never needs its
// local output before broadcast, so using input directly avoids a full copy.
CcuResult WriteReduceInputToPeer(KernelContext &ctx)
{
    ccu::LocalAddr srcAddr;
    srcAddr.addr  = ctx.localInput;
    srcAddr.token = ctx.localInputToken;

    ccu::RemoteAddr dstAddr;
    dstAddr.addr  = ctx.remoteOutput;
    dstAddr.token = ctx.remoteOutputToken;

    CCU_CHK_RET(ccu::WriteReduce(ctx.arg->channels[0], dstAddr, srcAddr,
        ctx.chunkSize, ctx.arg->dataType, ctx.arg->reduceType, ctx.event, 1));
    CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

// Parent-side broadcast: write own output to child's output buffer.
CcuResult BroadcastPeer(KernelContext &ctx)
{
    ccu::LocalAddr srcAddr;
    srcAddr.addr  = ctx.localOutput;
    srcAddr.token = ctx.localOutputToken;

    ccu::RemoteAddr dstAddr;
    dstAddr.addr  = ctx.remoteOutput;
    dstAddr.token = ctx.remoteOutputToken;

    CCU_CHK_RET(ccu::Write(ctx.arg->channels[0], dstAddr, srcAddr,
        ctx.chunkSize, ctx.event, 1));
    CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

// Bidirectional end-of-phase synchronization.
CcuResult PairSync(KernelContext &ctx)
{
    const ChannelHandle ch = ctx.arg->channels[0];
    CCU_CHK_RET(ccu::NotifyRecord(ch, NOTIFY_INDEX, SYNC_MASK));
    CCU_CHK_RET(ccu::NotifyWait(ch,  NOTIFY_INDEX, SYNC_MASK));
    return CCU_SUCCESS;
}

} // anonymous namespace

// Entry point invoked for every CCU kernel launch.
// Task argument layout (12 slots):
//   [0] localInput       - input buffer device address
//   [1] localOutput      - output buffer device address
//   [2] localInputToken  - input buffer CCU token
//   [3] localOutputToken - output buffer CCU token
//   [4] scratch          - HCCL scratch buffer address
//   [5] scratchToken     - HCCL scratch buffer token
//   [6] chunkSize        - bytes in this chunk
//   [7] phase            - PairPhase enum value (uint64_t)
//   [8] transferOffset   - byte offset of the ordered transfer range
//   [9] transferSize     - bytes in the ordered transfer range
//  [10] copyOffset       - byte offset of the first-level retained range
//  [11] copySize         - bytes in the first-level retained range
CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount != 1 ||
        kernelArg->rankId == kernelArg->peerRank) {
        return CCU_E_PARA;
    }

    KernelContext ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    // ------------------------------------------------------------------
    // Phase: optional output initialization (out-of-place first call)
    // CCU_IF does not support || between CondExpr; use separate blocks.
    // ------------------------------------------------------------------
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::REDUCE_RECEIVE_INIT))
    {
        CCU_CHK_RET(InitializeOutput(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::REDUCE_WR_RECV_INIT))
    {
        CCU_CHK_RET(InitializeOutput(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::REDUCE_SEND_INIT))
    {
        CCU_CHK_RET(InitializeOutput(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::RHD_REDUCE_RECV_FIRST))
    {
        CCU_CHK_RET(InitializeRange(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::RHD_REDUCE_BIDIR_FIRST))
    {
        CCU_CHK_RET(InitializeRange(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::RHD_REDUCE_BIDIR_SCRATCH))
    {
        CCU_CHK_RET(CopyTransferRangeToScratch(ctx));
    }
    // ------------------------------------------------------------------
    // Address exchange (both sides publish/receive output address+token)
    // ------------------------------------------------------------------
    CCU_CHK_RET(ExchangeAddresses(ctx));

    // ------------------------------------------------------------------
    // Phase: data movement
    // ------------------------------------------------------------------

    // Classic pull-based reduce (parent reads from child, reduces locally).
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::REDUCE_RECEIVE))
    {
        CCU_CHK_RET(ReducePeer(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::REDUCE_RECEIVE_INIT))
    {
        CCU_CHK_RET(ReducePeer(ctx));
    }

    // Push-based reduce: child writes-and-reduces into parent output.
    // Parent side (REDUCE_WR_RECV / REDUCE_WR_RECV_INIT) has no data operation;
    // it only participates in address exchange and PairSync.
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::REDUCE_WR_SEND))
    {
        CCU_CHK_RET(WriteReduceOutputToPeer(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::REDUCE_WR_SEND_INIT))
    {
        CCU_CHK_RET(WriteReduceInputToPeer(ctx));
    }

    // Ordered reduce-scatter. Receiver phases only wait at PairSync.
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::RHD_REDUCE_SEND_FIRST))
    {
        CCU_CHK_RET(ReduceScatterWrite(ctx, true));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::RHD_REDUCE_SEND))
    {
        CCU_CHK_RET(ReduceScatterWrite(ctx, false));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::RHD_REDUCE_BIDIR_FIRST))
    {
        CCU_CHK_RET(ReduceScatterWrite(ctx, true));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::RHD_REDUCE_BIDIR))
    {
        CCU_CHK_RET(ReduceScatterWrite(ctx, false));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::RHD_REDUCE_BIDIR_SCRATCH))
    {
        CCU_CHK_RET(ReduceScratchToPeer(ctx));
    }

    // Ordered all-gather. Receiver phases only wait at PairSync.
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::RHD_GATHER_SEND))
    {
        CCU_CHK_RET(AllGatherWrite(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::RHD_GATHER_BIDIR))
    {
        CCU_CHK_RET(AllGatherWrite(ctx));
    }

    // Broadcast: parent writes to child.
    CCU_IF(ctx.phase == static_cast<uint64_t>(PairPhase::BROADCAST_SEND))
    {
        CCU_CHK_RET(BroadcastPeer(ctx));
    }
    // Receive and pass-through phases have no data op and fall through to sync.

    // ------------------------------------------------------------------
    // End-of-phase bilateral sync (peer pair must both progress together)
    // ------------------------------------------------------------------
    CCU_CHK_RET(PairSync(ctx));
    return CCU_SUCCESS;
}

namespace sharded_mesh {

constexpr uint32_t INPUT_ADDR_SLOT = 0;
constexpr uint32_t INPUT_TOKEN_SLOT = 1;
constexpr uint32_t OUTPUT_ADDR_SLOT = 2;
constexpr uint32_t OUTPUT_TOKEN_SLOT = 3;
constexpr uint32_t MESH_NOTIFY_INDEX = 0;
constexpr uint16_t INPUT_ADDR_MASK = static_cast<uint16_t>(1u << INPUT_ADDR_SLOT);
constexpr uint16_t INPUT_TOKEN_MASK = static_cast<uint16_t>(1u << INPUT_TOKEN_SLOT);
constexpr uint16_t OUTPUT_ADDR_MASK = static_cast<uint16_t>(1u << OUTPUT_ADDR_SLOT);
constexpr uint16_t OUTPUT_TOKEN_MASK = static_cast<uint16_t>(1u << OUTPUT_TOKEN_SLOT);
constexpr uint16_t ADDRESS_MASK =
    static_cast<uint16_t>(INPUT_ADDR_MASK | INPUT_TOKEN_MASK |
        OUTPUT_ADDR_MASK | OUTPUT_TOKEN_MASK);

struct MeshContext {
    const CcuKernelArgShardedMesh *arg;
    ccu::Variable localInputBase;
    ccu::Variable localOutputBase;
    ccu::Variable localInputToken;
    ccu::Variable localOutputToken;
    ccu::Variable scratchBase;
    ccu::Variable scratchToken;
    ccu::Variable scratchStride;
    ccu::Variable operationMode;
    ccu::Variable operationSize;
    ccu::Variable peerOffset;
    ccu::LocalAddr localInput;
    ccu::LocalAddr localOutput;
    ccu::LocalAddr secondaryPartial;
    std::vector<ccu::RemoteAddr> peerInputs;
    std::vector<ccu::RemoteAddr> peerOutputs;
    std::vector<ccu::LocalAddr> peerScratch;
    ccu::Event event;
};

CcuResult LoadMeshArgs(MeshContext &ctx)
{
    uint32_t slot = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.localInputBase, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localOutputBase, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localInputToken, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localOutputToken, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchBase, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchStride, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.operationMode, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.operationSize, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.peerOffset, slot++));
    return CCU_SUCCESS;
}

CcuResult InitMeshResources(MeshContext &ctx)
{
    const uint32_t channelCount = ctx.arg->channelCount;
    ctx.peerInputs.resize(channelCount);
    ctx.peerOutputs.resize(channelCount);
    ctx.peerScratch.resize(channelCount);

    for (uint32_t i = 0; i < channelCount; ++i) {
        const ChannelHandle channel = ctx.arg->channels[i];
        ctx.peerInputs[i].addr = ccu::GetResByChannel<ccu::Variable>(channel, INPUT_ADDR_SLOT);
        ctx.peerInputs[i].token = ccu::GetResByChannel<ccu::Variable>(channel, INPUT_TOKEN_SLOT);
        ctx.peerOutputs[i].addr =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_ADDR_SLOT);
        ctx.peerOutputs[i].token =
            ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_SLOT);
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeMeshAddresses(MeshContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const ChannelHandle channel = ctx.arg->channels[i];
        CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.localInputBase,
            INPUT_ADDR_SLOT, MESH_NOTIFY_INDEX, INPUT_ADDR_MASK));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.localInputToken,
            INPUT_TOKEN_SLOT, MESH_NOTIFY_INDEX, INPUT_TOKEN_MASK));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.localOutputBase,
            OUTPUT_ADDR_SLOT, MESH_NOTIFY_INDEX, OUTPUT_ADDR_MASK));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.localOutputToken,
            OUTPUT_TOKEN_SLOT, MESH_NOTIFY_INDEX, OUTPUT_TOKEN_MASK));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], MESH_NOTIFY_INDEX, ADDRESS_MASK));
    }
    return CCU_SUCCESS;
}

void ReadPeerChunks(MeshContext &ctx, ccu::Variable len)
{
    uint16_t waitMask = 0;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint16_t bit = static_cast<uint16_t>(1u << i);
        ccu::Read(ctx.arg->channels[i], ctx.peerScratch[i], ctx.peerInputs[i], len,
            ctx.event, bit);
        waitMask = static_cast<uint16_t>(waitMask | bit);
    }
    ccu::EventWait(ctx.event, waitMask);
}

void ReadPeerOutputChunks(MeshContext &ctx, ccu::Variable len)
{
    uint16_t waitMask = 0;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint16_t bit = static_cast<uint16_t>(1u << i);
        ctx.peerOutputs[i].addr += ctx.peerOffset;
        ccu::Read(ctx.arg->channels[i], ctx.peerScratch[i],
            ctx.peerOutputs[i], len, ctx.event, bit);
        waitMask = static_cast<uint16_t>(waitMask | bit);
    }
    ccu::EventWait(ctx.event, waitMask);
}

void ReducePeerChunks(MeshContext &ctx, ccu::Variable len)
{
    std::vector<ccu::LocalAddr> inputs;
    if (ctx.arg->primary) {
        ccu::LocalCopy(ctx.localOutput, ctx.localInput, len, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
        inputs.push_back(ctx.localOutput);
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        inputs.push_back(ctx.peerScratch[i]);
    }

    while (inputs.size() > 1) {
        std::vector<ccu::LocalAddr> next;
        uint16_t waitMask = 0;
        uint32_t eventIndex = 0;
        uint32_t i = 0;
        for (; i + 1 < inputs.size(); i += 2) {
            const uint16_t bit = static_cast<uint16_t>(1u << eventIndex++);
            ccu::LocalReduce(inputs[i], inputs[i + 1], len,
                ctx.arg->dataType, ctx.arg->reduceType, ctx.event, bit);
            waitMask = static_cast<uint16_t>(waitMask | bit);
            next.push_back(inputs[i]);
        }
        if (i < inputs.size()) {
            next.push_back(inputs[i]);
        }
        ccu::EventWait(ctx.event, waitMask);
        inputs.swap(next);
    }
}

void ReducePeerOutputChunks(MeshContext &ctx, ccu::Variable len)
{
    std::vector<ccu::LocalAddr> inputs;
    if (ctx.arg->primary) {
        inputs.push_back(ctx.localOutput);
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        inputs.push_back(ctx.peerScratch[i]);
    }

    while (inputs.size() > 1) {
        std::vector<ccu::LocalAddr> next;
        uint16_t waitMask = 0;
        uint32_t eventIndex = 0;
        uint32_t i = 0;
        for (; i + 1 < inputs.size(); i += 2) {
            const uint16_t bit = static_cast<uint16_t>(1u << eventIndex++);
            ccu::LocalReduce(inputs[i], inputs[i + 1], len,
                ctx.arg->dataType, ctx.arg->reduceType, ctx.event, bit);
            waitMask = static_cast<uint16_t>(waitMask | bit);
            next.push_back(inputs[i]);
        }
        if (i < inputs.size()) {
            next.push_back(inputs[i]);
        }
        ccu::EventWait(ctx.event, waitMask);
        inputs.swap(next);
    }
}

void ProcessPartialChunk(MeshContext &ctx, ccu::Variable len)
{
    ReadPeerChunks(ctx, len);
    ReducePeerChunks(ctx, len);
}

void ProcessInPlacePartialChunk(MeshContext &ctx, ccu::Variable len)
{
    ReadPeerOutputChunks(ctx, len);
    ReducePeerOutputChunks(ctx, len);
}

void WriteOwnedChunk(MeshContext &ctx, ccu::Variable len)
{
    uint16_t waitMask = 0;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint16_t bit = static_cast<uint16_t>(1u << i);
        ctx.peerOutputs[i].addr += ctx.peerOffset;
        ccu::Write(ctx.arg->channels[i], ctx.peerOutputs[i], ctx.localOutput,
            len, ctx.event, bit);
        waitMask = static_cast<uint16_t>(waitMask | bit);
    }
    ccu::EventWait(ctx.event, waitMask);
}

void MergePartialChunk(MeshContext &ctx, ccu::Variable split,
    ccu::Variable primarySize, ccu::Variable secondarySize)
{
    ccu::LocalAddr output = ctx.localOutput;
    ccu::LocalAddr partial = ctx.secondaryPartial;
    ccu::Variable mergeSize;
    if (ctx.arg->primary) {
        mergeSize = primarySize;
    } else {
        output.addr += split;
        partial.addr += split;
        mergeSize = secondarySize;
    }
    CCU_IF(mergeSize != 0)
    {
        ccu::LocalReduce(output, partial, mergeSize,
            ctx.arg->dataType, ctx.arg->reduceType, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }
}

CcuResult PrepareMeshAddresses(MeshContext &ctx)
{
    ctx.localInput.addr = ctx.localInputBase;
    ctx.localInput.token = ctx.localInputToken;
    ctx.localOutput.addr = ctx.localOutputBase;
    ctx.localOutput.token = ctx.localOutputToken;

    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        ctx.peerScratch[i].addr = ctx.scratchBase;
        const uint32_t scratchLane = ctx.arg->scratchLanes[i];
        for (uint32_t lane = 0; lane < scratchLane; ++lane) {
            ctx.peerScratch[i].addr += ctx.scratchStride;
        }
        ctx.peerScratch[i].token = ctx.scratchToken;
    }

    ctx.secondaryPartial.addr = ctx.scratchBase;
    for (uint32_t lane = 0; lane < ctx.arg->secondaryPartialLane; ++lane) {
        ctx.secondaryPartial.addr += ctx.scratchStride;
    }
    ctx.secondaryPartial.token = ctx.scratchToken;
    return CCU_SUCCESS;
}

} // namespace sharded_mesh

CcuResult CcuMeshBootstrapKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgShardedMesh *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 ||
        kernelArg->channelCount >= 16 || kernelArg->rankSize > MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    sharded_mesh::MeshContext ctx;
    ctx.arg = kernelArg;
    uint32_t slot = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.localInputBase, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localInputToken, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localOutputBase, slot++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localOutputToken, slot++));
    CCU_CHK_RET(sharded_mesh::ExchangeMeshAddresses(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuShardedMeshKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgShardedMesh *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 ||
        kernelArg->channelCount >= 16 || kernelArg->rankSize > MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    sharded_mesh::MeshContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(sharded_mesh::LoadMeshArgs(ctx));
    CCU_CHK_RET(sharded_mesh::InitMeshResources(ctx));
    CCU_CHK_RET(sharded_mesh::PrepareMeshAddresses(ctx));

    CCU_IF(ctx.operationMode == 0)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            ctx.peerInputs[i].addr += ctx.peerOffset;
        }
        sharded_mesh::ProcessPartialChunk(ctx, ctx.operationSize);
    }

    CCU_IF(ctx.operationMode == 1)
    {
        sharded_mesh::MergePartialChunk(ctx, ctx.operationSize,
            ctx.operationSize, ctx.operationSize);
    }

    CCU_IF(ctx.operationMode == 2)
    {
        sharded_mesh::WriteOwnedChunk(ctx, ctx.operationSize);
    }

    CCU_IF(ctx.operationMode == 3)
    {
        sharded_mesh::ProcessInPlacePartialChunk(
            ctx, ctx.operationSize);
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
