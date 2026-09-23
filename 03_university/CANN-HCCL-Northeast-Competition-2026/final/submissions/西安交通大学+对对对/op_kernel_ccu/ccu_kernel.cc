/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel.h"

#include <limits>
#include <vector>

namespace ops_hccl {
namespace {
namespace ccu = AscendC::ccu;

constexpr uint32_t OUTPUT_VAR = 0;
constexpr uint32_t TOKEN_VAR = 1;
constexpr uint32_t READY_NOTIFY = 0;
constexpr uint32_t DONE_NOTIFY = 1;
constexpr uint32_t ACK_NOTIFY = 2;
constexpr uint16_t OUTPUT_READY = 1;
constexpr uint16_t TOKEN_READY = 2;
constexpr uint16_t BOTH_READY = OUTPUT_READY | TOKEN_READY;
constexpr uint16_t COMPLETE = 1;

// This file returns CcuResult, whereas CHK_RET_CCU is for Host HcclResult APIs.
#define SCATTER_CCU_TRY(call) \
    do { \
        const CcuResult scatterRet = (call); \
        if (scatterRet != CCU_SUCCESS) { \
            return scatterRet; \
        } \
    } while (0)

struct TaskVariables {
    ccu::Variable input;
    ccu::Variable inputToken;
    ccu::Variable output;
    ccu::Variable outputToken;
    ccu::Variable bytes;
    ccu::Variable loopCount;
    ccu::Variable lastBytes;
    ccu::Variable copyEnabled;
    ccu::Variable rootSource;
};

CcuResult LoadTaskVariables(TaskVariables &vars)
{
    // All roles load the same contiguous nine uint64_t arguments. Keep these
    // as the first emitted instructions, within the 13-argument SQE capacity.
    SCATTER_CCU_TRY(ccu::LoadArg(vars.input, SCATTER_SRC_ADDR));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.inputToken, SCATTER_SRC_TOKEN));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.output, SCATTER_DST_ADDR));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.outputToken, SCATTER_DST_TOKEN));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.bytes, SCATTER_BYTES));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.loopCount, SCATTER_LOOP_COUNT));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.lastBytes, SCATTER_LAST_BYTES));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.copyEnabled, SCATTER_COPY_ENABLED));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.rootSource, SCATTER_ROOT_SRC_ADDR));
    return CCU_SUCCESS;
}

CcuResult ValidateKernelArg(const ScatterKernelArg &arg)
{
    if (arg.shortPush > 1 || (arg.shortPush != 0 &&
        (arg.dataPath != ScatterDataPath::SINGLE_WAVE || arg.role == ScatterKernelRole::LOCAL_COPY ||
            arg.pullRead != 0 || arg.pullEarlyCopy != 0 || arg.pullMsBytes != 0 || arg.pullMsTail != 0 ||
            arg.pullAddressMode != ScatterPullAddressMode::SLICE || arg.pullRoot != 0 || arg.pullPeerOrder != 0))) {
        return CCU_E_PARA;
    }
    if (arg.pullMsTail > 1 ||
        (arg.pullMsTail != 0 && arg.pullMsBytes <= SCATTER_PULL_MS_TAIL_BYTES)) {
        return CCU_E_PARA;
    }
    if (arg.pullMsBytes != 0 && (arg.pullMsBytes > SCATTER_PULL_MS_MAX_BYTES || arg.pullMsBytes % 4U != 0 ||
        arg.pullRead == 0 || arg.role != ScatterKernelRole::RECEIVE ||
        arg.dataPath != ScatterDataPath::SINGLE_WAVE || arg.peerCount != 1)) {
        return CCU_E_PARA;
    }
    if (arg.pullAddressMode != ScatterPullAddressMode::SLICE &&
        arg.pullAddressMode != ScatterPullAddressMode::THREE_SOURCES &&
        arg.pullAddressMode != ScatterPullAddressMode::BASE_OFFSET) {
        return CCU_E_PARA;
    }
    const uint32_t pullRanks = arg.pullAddressMode == ScatterPullAddressMode::THREE_SOURCES ? 4U : 12U;
    if (arg.pullAddressMode != ScatterPullAddressMode::SLICE &&
        (arg.pullRead == 0 || arg.dataPath != ScatterDataPath::SINGLE_WAVE ||
            arg.role == ScatterKernelRole::LOCAL_COPY || arg.pullRoot >= pullRanks ||
            arg.peerCount >= pullRanks)) {
        return CCU_E_PARA;
    }
    if ((arg.dataPath != ScatterDataPath::SINGLE_WAVE && arg.dataPath != ScatterDataPath::CHUNKED) ||
        arg.copyOwner > 1 || arg.mergeEvents > 1 || arg.readyWrite > 1 || arg.earlyCopy > 1 || arg.pullRead > 1 ||
        arg.pullEarlyCopy > 1 || arg.deferCopyWait > 1 || arg.pullPeerOrder > 1 ||
        (arg.deferCopyWait != 0 && (arg.role != ScatterKernelRole::SEND ||
            arg.dataPath != ScatterDataPath::SINGLE_WAVE || arg.copyOwner == 0 ||
            arg.earlyCopy != 0 || arg.pullRead != 0)) ||
        (arg.pullPeerOrder != 0 && (arg.role != ScatterKernelRole::SEND || arg.pullRead == 0 ||
            arg.pullAddressMode != ScatterPullAddressMode::BASE_OFFSET)) ||
        (arg.pullEarlyCopy != 0 && (arg.pullRead == 0 || arg.copyOwner == 0 ||
            arg.role != ScatterKernelRole::SEND)) ||
        (arg.pullRead != 0 && (arg.dataPath != ScatterDataPath::SINGLE_WAVE ||
            arg.role == ScatterKernelRole::LOCAL_COPY || arg.mergeEvents != 0 || arg.readyWrite != 0 ||
            arg.earlyCopy != 0)) ||
        (arg.earlyCopy != 0 && (arg.copyOwner == 0 || arg.dataPath != ScatterDataPath::SINGLE_WAVE)) ||
        ((arg.copyOwner != 0 || arg.mergeEvents != 0 || arg.readyWrite != 0) &&
            arg.role != ScatterKernelRole::SEND) ||
        ((arg.mergeEvents != 0 || arg.readyWrite != 0) && arg.dataPath != ScatterDataPath::SINGLE_WAVE) ||
        (arg.readyWrite != 0 && arg.peerCount < 2)) {
        return CCU_E_PARA;
    }
    if (arg.role == ScatterKernelRole::LOCAL_COPY) {
        return arg.peerCount == 0 ? CCU_SUCCESS : CCU_E_PARA;
    }
    if (arg.role != ScatterKernelRole::SEND && arg.role != ScatterKernelRole::RECEIVE) {
        return CCU_E_PARA;
    }
    if (arg.peerCount == 0 || arg.peerCount >= MAX_RANK_SIZE ||
        (arg.role == ScatterKernelRole::RECEIVE && arg.peerCount != 1)) {
        return CCU_E_PARA;
    }
    for (uint32_t i = 0; i < arg.peerCount; ++i) {
        if (arg.channels[i] == 0 || arg.peers[i] >= MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }
        if (arg.pullAddressMode != ScatterPullAddressMode::SLICE &&
            (arg.peers[i] >= pullRanks ||
                (arg.role == ScatterKernelRole::SEND && arg.peers[i] == arg.pullRoot) ||
                (arg.role == ScatterKernelRole::RECEIVE && arg.peers[i] != arg.pullRoot))) {
            return CCU_E_PARA;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (arg.channels[i] == arg.channels[j] || arg.peers[i] == arg.peers[j]) {
                return CCU_E_PARA;
            }
        }
    }
    return CCU_SUCCESS;
}

void InitSources(const ScatterKernelArg &arg, const TaskVariables &vars,
    std::vector<ccu::LocalAddr> &sources)
{
    uint32_t lastRank = 0;
    for (uint32_t i = 0; i < arg.peerCount; ++i) {
        if (arg.peers[i] > lastRank) {
            lastRank = arg.peers[i];
        }
    }

    // The peer order is chosen by topology and need not be numerical. Walking
    // ranks once computes peer * recvBytes using at most 15 CCU additions,
    // without expanding the launch arguments to one pointer per peer.
    ccu::Variable cursor;
    cursor = vars.input;
    for (uint32_t rank = 0; rank <= lastRank; ++rank) {
        for (uint32_t i = 0; i < arg.peerCount; ++i) {
            if (arg.peers[i] == rank) {
                sources[i].addr = cursor;
                sources[i].token = vars.inputToken;
            }
        }
        if (rank != lastRank) {
            cursor += vars.bytes;
        }
    }
}

struct FusedCopyState {
    ccu::LocalAddr source;
    ccu::LocalAddr destination;
    ccu::Event event;
};

CcuResult SendWave(const ScatterKernelArg &arg, const std::vector<ccu::LocalAddr> &sources,
    const std::vector<ccu::RemoteAddr> &destinations, const std::vector<ccu::Event> &events,
    const std::vector<FusedCopyState> &copies, ccu::Variable copyEnabled,
    ccu::Variable bytes, const char *copyLabel)
{
    // Do not put an EventWait in this loop: all independent transfers must be
    // issued before waiting for any completion.
    if (arg.readyWrite == 0) {
        for (uint32_t i = 0; i < arg.peerCount; ++i) {
            const uint32_t eventIndex = arg.mergeEvents != 0 ? 0 : i;
            const uint16_t bit = arg.mergeEvents != 0 ? static_cast<uint16_t>(1U << i) : COMPLETE;
            SCATTER_CCU_TRY(ccu::Write(arg.channels[i], destinations[i], sources[i], bytes, events[eventIndex], bit));
        }
    }
    if (arg.copyOwner != 0 && arg.earlyCopy == 0) {
        // Only one registered SEND group owns this instruction branch. Host
        // enables it for this invocation only when output is disjoint from the
        // entire input; other groups may still be reading any input slice.
        // Launch Copy before waiting for Writes, so memory and network work can
        // overlap. A disabled branch neither launches nor waits on its event.
        SCATTER_CCU_TRY(CcuIfBegin(copyEnabled.handle, 0, CCU_CONDITION_NE, copyLabel));
        SCATTER_CCU_TRY(ccu::LocalCopy(copies[0].destination, copies[0].source,
            bytes, copies[0].event, COMPLETE));
        if (arg.deferCopyWait == 0) {
            SCATTER_CCU_TRY(ccu::EventWait(copies[0].event, COMPLETE));
        }
        SCATTER_CCU_TRY(CcuIfEnd(copyLabel));
    }
    if (arg.mergeEvents != 0) {
        const uint16_t mask = static_cast<uint16_t>((1U << arg.peerCount) - 1U);
        SCATTER_CCU_TRY(ccu::EventWait(events[0], mask));
    } else {
        for (uint32_t i = 0; i < arg.peerCount; ++i) {
            // EventWait clears its completion bit before reuse.
            SCATTER_CCU_TRY(ccu::EventWait(events[i], COMPLETE));
        }
    }
    return CCU_SUCCESS;
}

CcuResult SendPeers(const ScatterKernelArg &arg, TaskVariables &vars)
{
    std::vector<ccu::LocalAddr> sources(arg.peerCount);
    std::vector<ccu::RemoteAddr> destinations(arg.peerCount);
    std::vector<ccu::Event> events(arg.mergeEvents != 0 ? 1 : arg.peerCount);
    // A non-owner group allocates no fused-copy address or event resources.
    std::vector<FusedCopyState> copies(arg.copyOwner);
    InitSources(arg, vars, sources);
    if (arg.copyOwner != 0) {
        copies[0].source.addr = vars.rootSource;
        copies[0].source.token = vars.inputToken;
        copies[0].destination.addr = vars.output;
        copies[0].destination.token = vars.outputToken;
    }

    if (arg.earlyCopy != 0) {
        // No remote metadata is needed for self-copy. Full-input disjointness
        // is checked by Host on every call; an in-place/alias call skips both
        // this launch and its matching wait at the end of this kernel.
        SCATTER_CCU_TRY(CcuIfBegin(vars.copyEnabled.handle, 0, CCU_CONDITION_NE, "scatter_early_copy_start"));
        SCATTER_CCU_TRY(ccu::LocalCopy(copies[0].destination, copies[0].source,
            vars.bytes, copies[0].event, COMPLETE));
        SCATTER_CCU_TRY(CcuIfEnd("scatter_early_copy_start"));
    }

    // Receive the output base address and token once for the whole invocation.
    // Copy them into independent address/token registers before any data write.
    for (uint32_t i = 0; i < arg.peerCount; ++i) {
        SCATTER_CCU_TRY(ccu::NotifyWait(arg.channels[i], READY_NOTIFY, BOTH_READY));
        const ccu::Variable peerOutput = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], OUTPUT_VAR);
        const ccu::Variable peerToken = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], TOKEN_VAR);
        destinations[i].addr = peerOutput;
        destinations[i].token = peerToken;
        if (arg.readyWrite != 0) {
            const uint32_t eventIndex = arg.mergeEvents != 0 ? 0 : i;
            const uint16_t bit = arg.mergeEvents != 0 ? static_cast<uint16_t>(1U << i) : COMPLETE;
            SCATTER_CCU_TRY(ccu::Write(arg.channels[i], destinations[i], sources[i],
                vars.bytes, events[eventIndex], bit));
        }
    }

    if (arg.dataPath == ScatterDataPath::SINGLE_WAVE) {
        // This is a registration-time choice. No While, chunk constants or
        // address-advancement instructions are generated for B <= 256MiB.
        SCATTER_CCU_TRY(SendWave(arg, sources, destinations, events, copies,
            vars.copyEnabled, vars.bytes, "scatter_fused_copy_single"));
    } else {
        ccu::Variable chunkBytes;
        ccu::Variable minusOne;
        chunkBytes = MAX_DATA_SIZE;
        minusOne = std::numeric_limits<uint64_t>::max();

        // Host supplies (B - 1) / MAX_DATA_SIZE full waves and a strictly
        // positive final wave. Both network and optional local-copy addresses
        // advance by exactly one completed full wave before reuse.
        constexpr const char *LOOP_LABEL = "scatter_send_full_waves";
        SCATTER_CCU_TRY(CcuWhileBegin(vars.loopCount.handle, 0, CCU_CONDITION_NE, LOOP_LABEL));
        SCATTER_CCU_TRY(SendWave(arg, sources, destinations, events, copies,
            vars.copyEnabled, chunkBytes, "scatter_fused_copy_full"));
        for (uint32_t i = 0; i < arg.peerCount; ++i) {
            sources[i].addr += chunkBytes;
            destinations[i].addr += chunkBytes;
        }
        if (arg.copyOwner != 0) {
            copies[0].source.addr += chunkBytes;
            copies[0].destination.addr += chunkBytes;
        }
        vars.loopCount += minusOne;
        SCATTER_CCU_TRY(CcuWhileEnd(LOOP_LABEL));
        SCATTER_CCU_TRY(SendWave(arg, sources, destinations, events, copies,
            vars.copyEnabled, vars.lastBytes, "scatter_fused_copy_tail"));
    }

    // Complete every receiver before waiting for ACK from any one receiver.
    // Ordinary Push drains ACK here. Short Push instead gates every next
    // same-direction DONE on fresh receiver READY, emitted only after the
    // previous DONE was consumed. All Writes have completed before either
    // protocol can release root input; local Copy and worker joins still drain.
    for (uint32_t i = 0; i < arg.peerCount; ++i) {
        SCATTER_CCU_TRY(ccu::NotifyRecord(arg.channels[i], DONE_NOTIFY, COMPLETE));
    }
    if (arg.shortPush == 0) {
        for (uint32_t i = 0; i < arg.peerCount; ++i) {
            SCATTER_CCU_TRY(ccu::NotifyWait(arg.channels[i], ACK_NOTIFY, COMPLETE));
        }
    }
    if (arg.earlyCopy != 0) {
        // Remote Writes have completed and terminal notices were sent. They
        // can progress while the independent local copy finishes, but root
        // cannot return or reuse its input/output until this event is consumed.
        SCATTER_CCU_TRY(CcuIfBegin(vars.copyEnabled.handle, 0, CCU_CONDITION_NE, "scatter_early_copy_finish"));
        SCATTER_CCU_TRY(ccu::EventWait(copies[0].event, COMPLETE));
        SCATTER_CCU_TRY(CcuIfEnd("scatter_early_copy_finish"));
    }
    if (arg.deferCopyWait != 0) {
        // Copy began after all Writes from this group were issued. Its wait
        // must not delay remote DONE/ACK, but must finish before kernel return.
        // Use exactly the same per-call condition as its launch in SendWave.
        SCATTER_CCU_TRY(CcuIfBegin(vars.copyEnabled.handle, 0, CCU_CONDITION_NE, "scatter_deferred_copy_finish"));
        SCATTER_CCU_TRY(ccu::EventWait(copies[0].event, COMPLETE));
        SCATTER_CCU_TRY(CcuIfEnd("scatter_deferred_copy_finish"));
    }
    return CCU_SUCCESS;
}

CcuResult ReceivePeers(const ScatterKernelArg &arg, const TaskVariables &vars)
{
    const ChannelHandle channel = arg.channels[0];
    SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(channel, vars.output, OUTPUT_VAR,
        READY_NOTIFY, OUTPUT_READY));
    SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(channel, vars.outputToken, TOKEN_VAR,
        READY_NOTIFY, TOKEN_READY));
    SCATTER_CCU_TRY(ccu::NotifyWait(channel, DONE_NOTIFY, COMPLETE));
    // Next local invocation is ordered after this consumption (including the
    // Host worker join). Never apply this omission to Pull or Adaptive.
    if (arg.shortPush == 0) {
        SCATTER_CCU_TRY(ccu::NotifyRecord(channel, ACK_NOTIFY, COMPLETE));
    }
    return CCU_SUCCESS;
}

// Pull uses the same two per-channel metadata slots in the opposite direction:
// root publishes this peer's input slice and its token; only the receiver owns
// the output address. READY means source metadata is available, ACK means the
// Read event has completed. Root must drain ACK from every peer before input
// reuse, including before a separate aliasing root Copy in ExecOp.
CcuResult PublishPullSources(const ScatterKernelArg &arg, const TaskVariables &vars)
{
    // V020 C: only the admitted 4x1 copy owner allocates these resources.
    // Do not wait here: address publication and remote Reads must overlap Copy.
    std::vector<FusedCopyState> earlyCopies(arg.pullEarlyCopy);
    if (arg.pullEarlyCopy != 0) {
        earlyCopies[0].source.addr = vars.rootSource;
        earlyCopies[0].source.token = vars.inputToken;
        earlyCopies[0].destination.addr = vars.output;
        earlyCopies[0].destination.token = vars.outputToken;
        SCATTER_CCU_TRY(CcuIfBegin(vars.copyEnabled.handle, 0, CCU_CONDITION_NE, "scatter_pull_early_copy"));
        SCATTER_CCU_TRY(ccu::LocalCopy(earlyCopies[0].destination, earlyCopies[0].source,
            vars.bytes, earlyCopies[0].event, COMPLETE));
        SCATTER_CCU_TRY(CcuIfEnd("scatter_pull_early_copy"));
    }
    if (arg.pullPeerOrder != 0) {
        // BASE_OFFSET peers share this dynamic input base. Host has already
        // sorted remote peers first; emit each address/token pair in that
        // order, with no new address arithmetic or early ACK wait.
        for (uint32_t i = 0; i < arg.peerCount; ++i) {
            SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(arg.channels[i], vars.input, OUTPUT_VAR,
                READY_NOTIFY, OUTPUT_READY));
            SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(arg.channels[i], vars.inputToken, TOKEN_VAR,
                READY_NOTIFY, TOKEN_READY));
        }
    } else if (arg.pullAddressMode != ScatterPullAddressMode::SLICE) {
        // Registration-time loops preserve the original numerical peer order.
        // The references below emit no Variable allocation or cursor arithmetic.
        const uint32_t rankCount = arg.pullAddressMode == ScatterPullAddressMode::THREE_SOURCES ? 4U : 12U;
        for (uint32_t rank = 0; rank < rankCount; ++rank) {
            for (uint32_t i = 0; i < arg.peerCount; ++i) {
                if (arg.peers[i] != rank) { continue; }
                const uint32_t ordinal = rank < arg.pullRoot ? rank : rank - 1;
                const ccu::Variable &source = arg.pullAddressMode == ScatterPullAddressMode::BASE_OFFSET ||
                    ordinal == 0 ? vars.input : (ordinal == 1 ? vars.loopCount : vars.lastBytes);
                SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(arg.channels[i], source, OUTPUT_VAR,
                    READY_NOTIFY, OUTPUT_READY));
                SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(arg.channels[i], vars.inputToken, TOKEN_VAR,
                    READY_NOTIFY, TOKEN_READY));
            }
        }
    } else {
        uint32_t lastRank = 0;
        for (uint32_t i = 0; i < arg.peerCount; ++i) {
            if (arg.peers[i] > lastRank) { lastRank = arg.peers[i]; }
        }
        ccu::Variable cursor;
        cursor = vars.input;
        for (uint32_t rank = 0; rank <= lastRank; ++rank) {
            for (uint32_t i = 0; i < arg.peerCount; ++i) {
                if (arg.peers[i] != rank) { continue; }
                SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(arg.channels[i], cursor, OUTPUT_VAR,
                    READY_NOTIFY, OUTPUT_READY));
                SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(arg.channels[i], vars.inputToken, TOKEN_VAR,
                    READY_NOTIFY, TOKEN_READY));
            }
            if (rank != lastRank) { cursor += vars.bytes; }
        }
    }
    std::vector<FusedCopyState> copies(arg.pullEarlyCopy == 0 ? arg.copyOwner : 0);
    if (arg.copyOwner != 0 && arg.pullEarlyCopy == 0) {
        copies[0].source.addr = vars.rootSource;
        copies[0].source.token = vars.inputToken;
        copies[0].destination.addr = vars.output;
        copies[0].destination.token = vars.outputToken;
        SCATTER_CCU_TRY(CcuIfBegin(vars.copyEnabled.handle, 0, CCU_CONDITION_NE, "scatter_pull_copy_start"));
        SCATTER_CCU_TRY(ccu::LocalCopy(copies[0].destination, copies[0].source,
            vars.bytes, copies[0].event, COMPLETE));
        SCATTER_CCU_TRY(CcuIfEnd("scatter_pull_copy_start"));
    }
    for (uint32_t i = 0; i < arg.peerCount; ++i) {
        SCATTER_CCU_TRY(ccu::NotifyWait(arg.channels[i], ACK_NOTIFY, COMPLETE));
    }
    if (arg.copyOwner != 0) {
        SCATTER_CCU_TRY(CcuIfBegin(vars.copyEnabled.handle, 0, CCU_CONDITION_NE, "scatter_pull_copy_finish"));
        SCATTER_CCU_TRY(ccu::EventWait(arg.pullEarlyCopy != 0 ? earlyCopies[0].event : copies[0].event, COMPLETE));
        SCATTER_CCU_TRY(CcuIfEnd("scatter_pull_copy_finish"));
    }
    return CCU_SUCCESS;
}

CcuResult ReceivePullMs(const ScatterKernelArg &arg, const ccu::RemoteAddr &source,
    const ccu::LocalAddr &destination)
{
    constexpr uint32_t EVENT_BITS = 16;
    const ChannelHandle channel = arg.channels[0];
    const uint32_t pageCount = static_cast<uint32_t>(
        (arg.pullMsBytes + SCATTER_PULL_MS_SLICE_BYTES - 1) / SCATTER_PULL_MS_SLICE_BYTES);
    const uint32_t eventCount = (pageCount + EVENT_BITS - 1) / EVENT_BITS;
    ccu::Array<ccu::CcuBuffer> buffers(pageCount);
    ccu::Array<ccu::Event> events(eventCount);
    ccu::Variable pageBytes;
    ccu::Variable lastBytes;
    pageBytes = SCATTER_PULL_MS_SLICE_BYTES;
    lastBytes = (arg.pullMsBytes - 1) % SCATTER_PULL_MS_SLICE_BYTES + 1;
    ccu::RemoteAddr readCursor;
    readCursor.addr = source.addr;
    readCursor.token = source.token;
    ccu::LocalAddr outputCursor;
    outputCursor.addr = destination.addr;
    outputCursor.token = destination.token;

    // Emit independent transfers on the enclosing queue. A terminal
    // LoopGroup has no same-queue successor for the platform Checker, so
    // neither stage uses loops/clones. Issued transfers capture the current
    // address; reusing a cursor avoids allocating one address per fragment.
    for (uint32_t i = 0; i < pageCount; ++i) {
        const auto &bytes = i + 1 == pageCount ? lastBytes : pageBytes;
        SCATTER_CCU_TRY(ccu::Read(channel, buffers[i], readCursor, bytes,
            events[i / EVENT_BITS], static_cast<uint16_t>(1U << (i % EVENT_BITS))));
        if (i + 1 != pageCount) { readCursor.addr += pageBytes; }
    }
    for (uint32_t g = 0; g < eventCount; ++g) {
        const uint32_t remaining = pageCount - g * EVENT_BITS;
        const uint32_t bits = remaining < EVENT_BITS ? remaining : EVENT_BITS;
        SCATTER_CCU_TRY(ccu::EventWait(events[g], static_cast<uint16_t>((1U << bits) - 1U)));
    }
    // ACK releases the input only after every fragment has reached private
    // MS. No root memory/channel metadata is read after this point.
    SCATTER_CCU_TRY(ccu::NotifyRecord(channel, ACK_NOTIFY, COMPLETE));
    for (uint32_t i = 0; i < pageCount; ++i) {
        const auto &bytes = i + 1 == pageCount ? lastBytes : pageBytes;
        SCATTER_CCU_TRY(ccu::LocalCopy(outputCursor, buffers[i], bytes,
            events[i / EVENT_BITS], static_cast<uint16_t>(1U << (i % EVENT_BITS))));
        if (i + 1 != pageCount) { outputCursor.addr += pageBytes; }
    }
    for (uint32_t g = 0; g < eventCount; ++g) {
        const uint32_t remaining = pageCount - g * EVENT_BITS;
        const uint32_t bits = remaining < EVENT_BITS ? remaining : EVENT_BITS;
        SCATTER_CCU_TRY(ccu::EventWait(events[g], static_cast<uint16_t>((1U << bits) - 1U)));
    }
    return CCU_SUCCESS;
}

CcuResult ReceivePullMsTail(const ScatterKernelArg &arg, const ccu::RemoteAddr &source,
    const ccu::LocalAddr &destination)
{
    constexpr uint16_t TAIL_MASK = (1U << SCATTER_PULL_MS_TAIL_BUFFERS) - 1U;
    constexpr uint16_t PREFIX_MASK = 1U << SCATTER_PULL_MS_TAIL_BUFFERS;
    const ChannelHandle channel = arg.channels[0];
    ccu::Array<ccu::CcuBuffer> buffers(SCATTER_PULL_MS_TAIL_BUFFERS);
    ccu::Event event;
    ccu::Variable prefixBytes;
    ccu::Variable pageBytes;
    prefixBytes = arg.pullMsBytes - SCATTER_PULL_MS_TAIL_BYTES;
    pageBytes = SCATTER_PULL_MS_SLICE_BYTES;
    ccu::RemoteAddr readCursor;
    readCursor.addr = source.addr;
    readCursor.addr += prefixBytes;
    readCursor.token = source.token;
    ccu::LocalAddr outputCursor;
    outputCursor.addr = destination.addr;
    outputCursor.addr += prefixBytes;
    outputCursor.token = destination.token;

    // Prefix and all four tail reads are issued before waiting. They access
    // disjoint byte ranges. Each tail slice owns a distinct private buffer.
    SCATTER_CCU_TRY(ccu::Read(channel, destination, source, prefixBytes, event, PREFIX_MASK));
    for (uint32_t i = 0; i < SCATTER_PULL_MS_TAIL_BUFFERS; ++i) {
        SCATTER_CCU_TRY(ccu::Read(channel, buffers[i], readCursor, pageBytes,
            event, static_cast<uint16_t>(1U << i)));
        if (i + 1 != SCATTER_PULL_MS_TAIL_BUFFERS) { readCursor.addr += pageBytes; }
    }
    // ACK includes prefix completion, not just MS arrival. After this wait
    // all input is in output/private MS, and all current tokens are captured.
    SCATTER_CCU_TRY(ccu::EventWait(event, TAIL_MASK | PREFIX_MASK));
    SCATTER_CCU_TRY(ccu::NotifyRecord(channel, ACK_NOTIFY, COMPLETE));
    for (uint32_t i = 0; i < SCATTER_PULL_MS_TAIL_BUFFERS; ++i) {
        SCATTER_CCU_TRY(ccu::LocalCopy(outputCursor, buffers[i], pageBytes,
            event, static_cast<uint16_t>(1U << i)));
        if (i + 1 != SCATTER_PULL_MS_TAIL_BUFFERS) { outputCursor.addr += pageBytes; }
    }
    // Reuse only the consumed read bits; local return requires full output.
    return ccu::EventWait(event, TAIL_MASK);
}

CcuResult ReceivePull(const ScatterKernelArg &arg, const TaskVariables &vars)
{
    const ChannelHandle channel = arg.channels[0];
    SCATTER_CCU_TRY(ccu::NotifyWait(channel, READY_NOTIFY, BOTH_READY));
    const ccu::Variable peerInput = ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_VAR);
    const ccu::Variable peerToken = ccu::GetResByChannel<ccu::Variable>(channel, TOKEN_VAR);
    ccu::RemoteAddr source;
    source.addr = peerInput;
    if (arg.pullAddressMode == ScatterPullAddressMode::BASE_OFFSET) {
        source.addr += vars.loopCount;
    }
    source.token = peerToken;
    ccu::LocalAddr destination;
    destination.addr = vars.output;
    destination.token = vars.outputToken;
    if (arg.pullMsBytes != 0) {
        return arg.pullMsTail != 0 ? ReceivePullMsTail(arg, source, destination) :
            ReceivePullMs(arg, source, destination);
    }
    ccu::Event event;
    SCATTER_CCU_TRY(ccu::Read(channel, destination, source, vars.bytes, event, COMPLETE));
    SCATTER_CCU_TRY(ccu::EventWait(event, COMPLETE));
    return ccu::NotifyRecord(channel, ACK_NOTIFY, COMPLETE);
}

CcuResult CopyLocal(const ScatterKernelArg &arg, TaskVariables &vars)
{
    ccu::LocalAddr source;
    ccu::LocalAddr destination;
    source.addr = vars.input;
    source.token = vars.inputToken;
    destination.addr = vars.output;
    destination.token = vars.outputToken;
    ccu::Event event;
    if (arg.dataPath == ScatterDataPath::SINGLE_WAVE) {
        SCATTER_CCU_TRY(ccu::LocalCopy(destination, source, vars.bytes, event, COMPLETE));
        SCATTER_CCU_TRY(ccu::EventWait(event, COMPLETE));
        return CCU_SUCCESS;
    }
    ccu::Variable chunkBytes;
    ccu::Variable minusOne;
    chunkBytes = MAX_DATA_SIZE;
    minusOne = std::numeric_limits<uint64_t>::max();

    // Host invokes this separate kernel only after all network groups join, so
    // root's output may overlap a slice that another rank needed to receive.
    constexpr const char *LOOP_LABEL = "scatter_copy_full_waves";
    SCATTER_CCU_TRY(CcuWhileBegin(vars.loopCount.handle, 0, CCU_CONDITION_NE, LOOP_LABEL));
    SCATTER_CCU_TRY(ccu::LocalCopy(destination, source, chunkBytes, event, COMPLETE));
    SCATTER_CCU_TRY(ccu::EventWait(event, COMPLETE));
    source.addr += chunkBytes;
    destination.addr += chunkBytes;
    vars.loopCount += minusOne;
    SCATTER_CCU_TRY(CcuWhileEnd(LOOP_LABEL));
    SCATTER_CCU_TRY(ccu::LocalCopy(destination, source, vars.lastBytes, event, COMPLETE));
    SCATTER_CCU_TRY(ccu::EventWait(event, COMPLETE));
    return CCU_SUCCESS;
}

constexpr uint32_t SCRATCH_VAR = 2;
constexpr uint32_t SCRATCH_TOKEN_VAR = 3;
constexpr uint16_t SCRATCH_READY = 4;
constexpr uint16_t SCRATCH_TOKEN_READY = 8;
constexpr uint16_t SCRATCH_BOTH_READY = SCRATCH_READY | SCRATCH_TOKEN_READY;
constexpr uint16_t ALL_BUFFERS_READY = BOTH_READY | SCRATCH_BOTH_READY;

bool ValidReadyMask(uint16_t mask)
{
    return mask == BOTH_READY || mask == SCRATCH_BOTH_READY || mask == ALL_BUFFERS_READY;
}

bool ValidTransfer(const ScatterTransfer &transfer)
{
    const bool validSource = transfer.source == ScatterBuffer::INPUT ||
        transfer.source == ScatterBuffer::OUTPUT || transfer.source == ScatterBuffer::SCRATCH;
    const bool validDestination = transfer.destination == ScatterBuffer::OUTPUT ||
        transfer.destination == ScatterBuffer::SCRATCH;
    return validSource && validDestination && transfer.reserved == 0 && transfer.bytes != 0 &&
        transfer.bytes <= MAX_DATA_SIZE && transfer.bytes % sizeof(float) == 0 &&
        transfer.sourceOffset % sizeof(float) == 0 && transfer.destinationOffset % sizeof(float) == 0 &&
        transfer.sourceOffset <= std::numeric_limits<uint64_t>::max() - transfer.bytes &&
        transfer.destinationOffset <= std::numeric_limits<uint64_t>::max() - transfer.bytes;
}

CcuResult ValidateAdaptiveArg(const AdaptiveKernelArg &arg)
{
    if (arg.transferCount > SCATTER_MAX_TRANSFERS || arg.deferAck > 1 || arg.fuseAckRequest > 1 ||
        arg.earlyTail > 2 || arg.copyOwner > 1 || arg.interleaveTails > 1 ||
        arg.readyWrite > 1 || arg.deferHelperOutput > 1 ||
        (arg.deferHelperOutput != 0 && (arg.role != AdaptiveRole::SEND_DATA ||
            arg.earlyTail == 0 || arg.tailChannel != 0)) ||
        (arg.readyWrite != 0 && (arg.role != AdaptiveRole::SEND_DATA ||
            arg.peerCount < 2 || arg.tailChannel != 0)) ||
        (arg.interleaveTails != 0 && arg.earlyTail != 2) ||
        ((arg.earlyTail != 0 || arg.copyOwner != 0 || arg.tailChannel != 0) &&
            arg.role != AdaptiveRole::SEND_DATA) ||
        (arg.deferAck != 0 && arg.role != AdaptiveRole::RECEIVE) ||
        (arg.fuseAckRequest != 0 && arg.role != AdaptiveRole::SEND_DATA)) {
        return CCU_E_PARA;
    }
    if ((arg.copyOwner != 0 && (arg.copyBytes == 0 || arg.copyBytes > MAX_DATA_SIZE ||
            (arg.copySourceOffset | arg.copyBytes) % 4 != 0 ||
            arg.copySourceOffset > std::numeric_limits<uint64_t>::max() - arg.copyBytes)) ||
        (arg.copyOwner == 0 && (arg.copyBytes != 0 || arg.copySourceOffset != 0)) ||
        (arg.role == AdaptiveRole::RECEIVE_TAIL &&
            (arg.tailWaitMask == 0 || (arg.tailWaitMask & COMPLETE) != 0))) {
        return CCU_E_PARA;
    }
    if (arg.role == AdaptiveRole::LOCAL_COPY) {
        if (arg.peerCount != 0 || arg.transferCount != 1 || arg.transfers[0].channelIndex != 0 ||
            arg.transfers[0].source != ScatterBuffer::SCRATCH ||
            arg.transfers[0].destination != ScatterBuffer::OUTPUT || !ValidTransfer(arg.transfers[0])) {
            return CCU_E_PARA;
        }
        return CCU_SUCCESS;
    }
    if (arg.role != AdaptiveRole::SEND_DATA && arg.role != AdaptiveRole::RECEIVE &&
        arg.role != AdaptiveRole::WAIT_ACK && arg.role != AdaptiveRole::ACK_ONLY &&
        arg.role != AdaptiveRole::REQUEST_ACK && arg.role != AdaptiveRole::PUBLISH_READY &&
        arg.role != AdaptiveRole::RECEIVE_TAIL && arg.role != AdaptiveRole::FINISH_RECEIVE) {
        return CCU_E_PARA;
    }
    if (arg.peerCount == 0 || arg.peerCount >= MAX_RANK_SIZE ||
        ((arg.role == AdaptiveRole::RECEIVE || arg.role == AdaptiveRole::ACK_ONLY ||
          arg.role == AdaptiveRole::PUBLISH_READY || arg.role == AdaptiveRole::RECEIVE_TAIL ||
          arg.role == AdaptiveRole::FINISH_RECEIVE) && arg.peerCount != 1) ||
        (arg.role == AdaptiveRole::SEND_DATA ? arg.transferCount == 0 : arg.transferCount != 0)) {
        return CCU_E_PARA;
    }
    for (uint32_t i = 0; i < arg.peerCount; ++i) {
        if (arg.channels[i] == 0 || arg.peers[i] >= MAX_RANK_SIZE ||
            ((arg.role == AdaptiveRole::SEND_DATA || arg.role == AdaptiveRole::RECEIVE ||
              arg.role == AdaptiveRole::PUBLISH_READY || arg.role == AdaptiveRole::RECEIVE_TAIL) &&
            !ValidReadyMask(arg.readyMasks[i]))) {
            return CCU_E_PARA;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (arg.channels[i] == arg.channels[j] || arg.peers[i] == arg.peers[j]) {
                return CCU_E_PARA;
            }
        }
    }
    uint16_t expectedMasks[MAX_RANK_SIZE] = {};
    uint16_t consumedTailMask = 0;
    uint32_t tailCounts[MAX_RANK_SIZE] = {};
    for (uint32_t i = 0; i < arg.transferCount; ++i) {
        const ScatterTransfer &transfer = arg.transfers[i];
        if (!ValidTransfer(transfer) || transfer.channelIndex >= arg.peerCount) {
            return CCU_E_PARA;
        }
        if (arg.copyOwner != 0 && transfer.source != ScatterBuffer::INPUT) { return CCU_E_PARA; }
        if (arg.tailChannel != 0) {
            const uint16_t mask = arg.tailWaitMasks[i];
            if (arg.earlyTail != 0 || arg.copyOwner != 0 || transfer.source != ScatterBuffer::SCRATCH ||
                mask == 0 || (mask & COMPLETE) != 0 || (mask & (mask - 1)) != 0 ||
                (consumedTailMask & mask) != 0) { return CCU_E_PARA; }
            consumedTailMask |= mask;
        } else if (arg.tailWaitMasks[i] != 0) { return CCU_E_PARA; }
        if (arg.earlyTail == 2 && transfer.destination == ScatterBuffer::SCRATCH &&
            ++tailCounts[transfer.channelIndex] > 15) { return CCU_E_PARA; }
        expectedMasks[transfer.channelIndex] |= transfer.destination == ScatterBuffer::OUTPUT ?
            BOTH_READY : SCRATCH_BOTH_READY;
        for (uint32_t j = 0; j < i; ++j) {
            const ScatterTransfer &previous = arg.transfers[j];
            if (transfer.channelIndex == previous.channelIndex && transfer.destination == previous.destination &&
                transfer.destinationOffset < previous.destinationOffset + previous.bytes &&
                previous.destinationOffset < transfer.destinationOffset + transfer.bytes) {
                return CCU_E_PARA;
            }
        }
    }
    if (arg.role == AdaptiveRole::SEND_DATA) {
        bool hasDelayedPeer = false;
        for (uint32_t i = 0; i < arg.peerCount; ++i) {
            if (arg.tailChannel != 0 && arg.tailChannel == arg.channels[i]) { return CCU_E_PARA; }
            if (expectedMasks[i] != arg.readyMasks[i]) {
                return CCU_E_PARA;
            }
            hasDelayedPeer = hasDelayedPeer || expectedMasks[i] == (BOTH_READY | SCRATCH_BOTH_READY);
        }
        if (arg.deferHelperOutput != 0 && !hasDelayedPeer) { return CCU_E_PARA; }
    }
    if (arg.role == AdaptiveRole::RECEIVE_TAIL && (arg.readyMasks[0] & SCRATCH_BOTH_READY) == 0) {
        return CCU_E_PARA;
    }
    if (arg.earlyTail != 0) {
        bool hasScratch = false;
        uint32_t helperCount = 0;
        for (uint32_t peer = 0; peer < arg.peerCount; ++peer) {
            helperCount += tailCounts[peer] != 0 ? 1U : 0U;
        }
        if (arg.interleaveTails != 0 && helperCount < 2) { return CCU_E_PARA; }
        for (uint32_t i = 0; i < arg.transferCount; ++i) {
            if (arg.transfers[i].source != ScatterBuffer::INPUT) { return CCU_E_PARA; }
            hasScratch = hasScratch || arg.transfers[i].destination == ScatterBuffer::SCRATCH;
        }
        if (!hasScratch) { return CCU_E_PARA; }
    }
    return CCU_SUCCESS;
}

struct AdaptiveVariables {
    ccu::Variable input;
    ccu::Variable inputToken;
    ccu::Variable output;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable waveOffset;
    ccu::Variable copyEnabled;
};

CcuResult LoadAdaptiveVariables(AdaptiveVariables &vars)
{
    // Preserve data slots 0..6 and consume all 13 slots of one launch SQE.
    SCATTER_CCU_TRY(ccu::LoadArg(vars.input, ADAPTIVE_INPUT));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.inputToken, ADAPTIVE_INPUT_TOKEN));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.output, ADAPTIVE_OUTPUT));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.outputToken, ADAPTIVE_OUTPUT_TOKEN));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.scratch, ADAPTIVE_SCRATCH));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.scratchToken, ADAPTIVE_SCRATCH_TOKEN));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.waveOffset, ADAPTIVE_WAVE_OFFSET));
    SCATTER_CCU_TRY(ccu::LoadArg(vars.copyEnabled, ADAPTIVE_COPY_ENABLED));
    // These loads use an independent variable and never affect data addresses,
    // offsets or notification state. Each argument ID emits its own LoadArg.
    ccu::Variable padding;
    for (uint32_t index = ADAPTIVE_PAD_BEGIN; index < ADAPTIVE_ARG_COUNT; ++index) {
        SCATTER_CCU_TRY(ccu::LoadArg(padding, index));
    }
    return CCU_SUCCESS;
}

CcuResult AddFixedOffset(ccu::Address &address, uint64_t offset)
{
    if (offset != 0) {
        ccu::Variable delta;
        SCATTER_CCU_TRY(CcuVariableAssignImm(delta.handle, offset));
        SCATTER_CCU_TRY(CcuAddressAddAssignVar(address.handle, delta.handle));
    }
    return CCU_SUCCESS;
}

CcuResult InitAdaptiveLocal(ccu::LocalAddr &address, ScatterBuffer buffer, uint64_t offset,
    const AdaptiveVariables &vars)
{
    const ccu::Variable &base = buffer == ScatterBuffer::INPUT ? vars.input :
        (buffer == ScatterBuffer::OUTPUT ? vars.output : vars.scratch);
    const ccu::Variable &token = buffer == ScatterBuffer::INPUT ? vars.inputToken :
        (buffer == ScatterBuffer::OUTPUT ? vars.outputToken : vars.scratchToken);
    // Explicit assignments retain the independently allocated address/token
    // handles; assigning a temporary Variable would instead alias its handle.
    SCATTER_CCU_TRY(CcuAddressAssignVar(address.addr.handle, base.handle));
    SCATTER_CCU_TRY(CcuVariableAssignVar(address.token.handle, token.handle));
    if (buffer != ScatterBuffer::SCRATCH) {
        SCATTER_CCU_TRY(CcuAddressAddAssignVar(address.addr.handle, vars.waveOffset.handle));
    }
    SCATTER_CCU_TRY(AddFixedOffset(address.addr, offset));
    return CCU_SUCCESS;
}

CcuResult InitAdaptiveRemote(ccu::RemoteAddr &address, ChannelHandle channel,
    const ScatterTransfer &transfer, const AdaptiveVariables &vars)
{
    const uint32_t baseIndex = transfer.destination == ScatterBuffer::OUTPUT ? OUTPUT_VAR : SCRATCH_VAR;
    const uint32_t tokenIndex = transfer.destination == ScatterBuffer::OUTPUT ? TOKEN_VAR : SCRATCH_TOKEN_VAR;
    const ccu::Variable base = ccu::GetResByChannel<ccu::Variable>(channel, baseIndex);
    const ccu::Variable token = ccu::GetResByChannel<ccu::Variable>(channel, tokenIndex);
    SCATTER_CCU_TRY(CcuAddressAssignVar(address.addr.handle, base.handle));
    SCATTER_CCU_TRY(CcuVariableAssignVar(address.token.handle, token.handle));
    if (transfer.destination == ScatterBuffer::OUTPUT) {
        SCATTER_CCU_TRY(CcuAddressAddAssignVar(address.addr.handle, vars.waveOffset.handle));
    }
    SCATTER_CCU_TRY(AddFixedOffset(address.addr, transfer.destinationOffset));
    return CCU_SUCCESS;
}

struct AdaptiveWriteState {
    ccu::LocalAddr source;
    ccu::RemoteAddr destination;
    ccu::Variable bytes;
    ccu::Event event;
};

CcuResult AdaptiveSend(const AdaptiveKernelArg &arg, const AdaptiveVariables &vars)
{
    // One event and private address/length state per transfer. There are at
    // most 32 events, and the Host registers only same-die channels together.
    std::vector<AdaptiveWriteState> writes(arg.transferCount);
    // These arrays and the lambda exist only while constructing instructions.
    // No CCU counter, new event, local notification or runtime branch is added.
    uint32_t tailsLeft[MAX_RANK_SIZE] = {};
    bool delayed[SCATTER_MAX_TRANSFERS] = {};
    if (arg.deferHelperOutput != 0) {
        for (uint32_t i = 0; i < arg.transferCount; ++i) {
            const auto &transfer = arg.transfers[i];
            if (transfer.destination == ScatterBuffer::SCRATCH) { ++tailsLeft[transfer.channelIndex]; }
        }
        for (uint32_t i = 0; i < arg.transferCount; ++i) {
            const auto &transfer = arg.transfers[i];
            delayed[i] = transfer.destination == ScatterBuffer::OUTPUT && tailsLeft[transfer.channelIndex] != 0;
        }
    }
    const auto issueDelayedOutput = [&](uint32_t peer) -> CcuResult {
        for (uint32_t i = 0; i < arg.transferCount; ++i) {
            if (!delayed[i] || arg.transfers[i].channelIndex != peer) { continue; }
            SCATTER_CCU_TRY(ccu::Write(arg.channels[peer], writes[i].destination,
                writes[i].source, writes[i].bytes, writes[i].event, COMPLETE));
        }
        return CCU_SUCCESS;
    };
    std::vector<FusedCopyState> copies(arg.copyOwner);
    std::vector<ccu::Variable> copyBytes(arg.copyOwner);
    if (arg.copyOwner != 0) {
        SCATTER_CCU_TRY(InitAdaptiveLocal(copies[0].source, ScatterBuffer::INPUT, arg.copySourceOffset, vars));
        SCATTER_CCU_TRY(InitAdaptiveLocal(copies[0].destination, ScatterBuffer::OUTPUT, 0, vars));
        SCATTER_CCU_TRY(CcuVariableAssignImm(copyBytes[0].handle, arg.copyBytes));
    }
    if (arg.tailChannel != 0) {
        // Host guarantees this upstream channel and every downstream channel
        // share one known die. Explicitly bind the upstream wait dependency.
        CcuVariableHandle bindingHandle = 0;
        SCATTER_CCU_TRY(CcuVariableCreateByChannel(arg.tailChannel, OUTPUT_VAR, &bindingHandle));
    }
    for (uint32_t i = 0; i < arg.peerCount; ++i) {
        SCATTER_CCU_TRY(ccu::NotifyRecord(arg.channels[i], ADAPTIVE_REQUEST_NOTIFY, ADAPTIVE_DATA_REQUEST));
    }
    if (arg.readyWrite != 0) {
        // Every DATA_REQUEST is already sent. Start one peer after its READY;
        // Issue every peer's first-phase data before waiting for completion.
        // V020 may postpone helper OUTPUT, but never a tail or direct prefix.
        for (uint32_t peer = 0; peer < arg.peerCount; ++peer) {
            SCATTER_CCU_TRY(ccu::NotifyWait(arg.channels[peer], READY_NOTIFY, arg.readyMasks[peer]));
            for (uint32_t i = 0; i < arg.transferCount; ++i) {
                const auto &transfer = arg.transfers[i];
                if (transfer.channelIndex != peer) { continue; }
                SCATTER_CCU_TRY(InitAdaptiveLocal(writes[i].source, transfer.source, transfer.sourceOffset, vars));
                SCATTER_CCU_TRY(InitAdaptiveRemote(writes[i].destination,
                    arg.channels[peer], transfer, vars));
                SCATTER_CCU_TRY(CcuVariableAssignImm(writes[i].bytes.handle, transfer.bytes));
                if (!delayed[i]) {
                    SCATTER_CCU_TRY(ccu::Write(arg.channels[peer], writes[i].destination,
                        writes[i].source, writes[i].bytes, writes[i].event, COMPLETE));
                }
            }
        }
    } else {
        for (uint32_t i = 0; i < arg.peerCount; ++i) {
            SCATTER_CCU_TRY(ccu::NotifyWait(arg.channels[i], READY_NOTIFY, arg.readyMasks[i]));
        }
        for (uint32_t i = 0; i < arg.transferCount; ++i) {
            const ScatterTransfer &transfer = arg.transfers[i];
            SCATTER_CCU_TRY(InitAdaptiveLocal(writes[i].source, transfer.source, transfer.sourceOffset, vars));
            SCATTER_CCU_TRY(InitAdaptiveRemote(writes[i].destination, arg.channels[transfer.channelIndex], transfer, vars));
            SCATTER_CCU_TRY(CcuVariableAssignImm(writes[i].bytes.handle, transfer.bytes));
        }
        // All non-delayed Writes precede event waits. The optional helper
        // OUTPUT is issued after that peer's existing tail-completion signal.
        for (uint32_t i = 0; i < arg.transferCount; ++i) {
            if (delayed[i]) { continue; }
            if (arg.tailChannel != 0) {
                SCATTER_CCU_TRY(ccu::NotifyWait(arg.tailChannel, DONE_NOTIFY, arg.tailWaitMasks[i]));
            }
            SCATTER_CCU_TRY(ccu::Write(arg.channels[arg.transfers[i].channelIndex],
                writes[i].destination, writes[i].source, writes[i].bytes, writes[i].event, COMPLETE));
        }
    }
    if (arg.copyOwner != 0) {
        SCATTER_CCU_TRY(CcuIfBegin(vars.copyEnabled.handle, 0, CCU_CONDITION_NE, "relay_copy_start"));
        SCATTER_CCU_TRY(ccu::LocalCopy(copies[0].destination, copies[0].source,
            copyBytes[0], copies[0].event, COMPLETE));
        SCATTER_CCU_TRY(CcuIfEnd("relay_copy_start"));
    }
    if (arg.interleaveTails != 0) {
        // Registration-time loops only: reorder the existing wait/notify pairs,
        // without allocating CCU variables/events. V020 inserts a delayed
        // helper OUTPUT immediately after that peer's final tail signal.
        // Each peer's original scratch-transfer ordinal still defines its bit.
        // All scratch Writes are in flight, so waiting for one cannot prevent
        // another helper's tail from starting. These are ordered waits.
        uint32_t next[MAX_RANK_SIZE] = {};
        for (uint32_t ordinal = 1; ordinal <= 15; ++ordinal) {
            bool emitted = false;
            for (uint32_t peer = 0; peer < arg.peerCount; ++peer) {
                while (next[peer] < arg.transferCount &&
                    (arg.transfers[next[peer]].channelIndex != peer ||
                     arg.transfers[next[peer]].destination != ScatterBuffer::SCRATCH)) {
                    ++next[peer];
                }
                if (next[peer] == arg.transferCount) { continue; }
                SCATTER_CCU_TRY(ccu::EventWait(writes[next[peer]].event, COMPLETE));
                SCATTER_CCU_TRY(ccu::NotifyRecord(arg.channels[peer], DONE_NOTIFY,
                    static_cast<uint16_t>(1U << ordinal)));
                // Release this helper's own output immediately after its last
                // tail, without waiting for another helper's remaining tails.
                if (arg.deferHelperOutput != 0 && --tailsLeft[peer] == 0) {
                    SCATTER_CCU_TRY(issueDelayedOutput(peer));
                }
                ++next[peer];
                emitted = true;
            }
            if (!emitted) { break; }
        }
    } else if (arg.earlyTail != 0) {
        // Release each helper's complete scratch payload before waiting for
        // its own output. All scratch Writes were already submitted. Separate
        // DONE bits prevent whole-peer completion from being consumed early.
        for (uint32_t peer = 0; peer < arg.peerCount; ++peer) {
            bool hasTail = false;
            uint32_t tailIndex = 0;
            for (uint32_t i = 0; i < arg.transferCount; ++i) {
                if (arg.transfers[i].channelIndex == peer &&
                    arg.transfers[i].destination == ScatterBuffer::SCRATCH) {
                    SCATTER_CCU_TRY(ccu::EventWait(writes[i].event, COMPLETE));
                    hasTail = true;
                    if (arg.earlyTail == 2) {
                        const uint16_t mask = static_cast<uint16_t>(1U << (++tailIndex));
                        SCATTER_CCU_TRY(ccu::NotifyRecord(arg.channels[peer], DONE_NOTIFY, mask));
                    }
                }
            }
            if (hasTail && arg.earlyTail == 1) {
                SCATTER_CCU_TRY(ccu::NotifyRecord(arg.channels[peer], DONE_NOTIFY, ADAPTIVE_TAIL_COMPLETE));
            }
            if (hasTail && arg.deferHelperOutput != 0) {
                SCATTER_CCU_TRY(issueDelayedOutput(peer));
            }
        }
    }
    for (uint32_t peer = 0; peer < arg.peerCount; ++peer) {
        for (uint32_t i = 0; i < arg.transferCount; ++i) {
            if (arg.transfers[i].channelIndex == peer &&
                !(arg.earlyTail != 0 && arg.transfers[i].destination == ScatterBuffer::SCRATCH)) {
                SCATTER_CCU_TRY(ccu::EventWait(writes[i].event, COMPLETE));
            }
        }
        SCATTER_CCU_TRY(ccu::NotifyRecord(arg.channels[peer], DONE_NOTIFY, COMPLETE));
        if (arg.fuseAckRequest != 0) {
            SCATTER_CCU_TRY(ccu::NotifyRecord(arg.channels[peer], ADAPTIVE_REQUEST_NOTIFY, ADAPTIVE_ACK_REQUEST));
        }
    }
    if (arg.copyOwner != 0) {
        // All TAIL/DONE/ACK_REQUEST notifications are already published; a
        // slow local Copy cannot hold back helper forwarding readiness.
        SCATTER_CCU_TRY(CcuIfBegin(vars.copyEnabled.handle, 0, CCU_CONDITION_NE, "relay_copy_finish"));
        SCATTER_CCU_TRY(ccu::EventWait(copies[0].event, COMPLETE));
        SCATTER_CCU_TRY(CcuIfEnd("relay_copy_finish"));
    }
    // ACK waiting stays a separate Host batch after ALL data groups launch.
    // A fused ACK_REQUEST above records a bit without waiting for its ACK.
    // Waiting here can deadlock when a helper forwards to a leaf whose root
    // prefix is assigned to a later, sequentially launched die group.
    return CCU_SUCCESS;
}

CcuResult AdaptiveReceive(const AdaptiveKernelArg &arg, const AdaptiveVariables &vars)
{
    const ChannelHandle channel = arg.channels[0];
    const uint16_t mask = arg.readyMasks[0];
    SCATTER_CCU_TRY(ccu::NotifyWait(channel, ADAPTIVE_REQUEST_NOTIFY, ADAPTIVE_DATA_REQUEST));
    if ((mask & BOTH_READY) != 0) {
        SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(channel, vars.output,
            OUTPUT_VAR, READY_NOTIFY, OUTPUT_READY));
        SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(channel, vars.outputToken,
            TOKEN_VAR, READY_NOTIFY, TOKEN_READY));
    }
    if ((mask & SCRATCH_BOTH_READY) != 0) {
        SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(channel, vars.scratch,
            SCRATCH_VAR, READY_NOTIFY, SCRATCH_READY));
        SCATTER_CCU_TRY(ccu::WriteVariableWithNotify(channel, vars.scratchToken,
            SCRATCH_TOKEN_VAR, READY_NOTIFY, SCRATCH_TOKEN_READY));
    }
    if (arg.role == AdaptiveRole::PUBLISH_READY) { return CCU_SUCCESS; }
    if (arg.role == AdaptiveRole::RECEIVE_TAIL) {
        return ccu::NotifyWait(channel, DONE_NOTIFY, arg.tailWaitMask);
    }
    SCATTER_CCU_TRY(ccu::NotifyWait(channel, DONE_NOTIFY, COMPLETE));
    if (arg.deferAck == 0) {
        SCATTER_CCU_TRY(ccu::NotifyWait(channel, ADAPTIVE_REQUEST_NOTIFY, ADAPTIVE_ACK_REQUEST));
        SCATTER_CCU_TRY(ccu::NotifyRecord(channel, ACK_NOTIFY, COMPLETE));
    }
    // A forwarding receiver delays ACK until its downstream ACK batch and
    // optional local copy both complete, protecting reuse of this scratch wave.
    return CCU_SUCCESS;
}

CcuResult AdaptiveCopy(const ScatterTransfer &transfer, const AdaptiveVariables &vars)
{
    ccu::LocalAddr source;
    ccu::LocalAddr destination;
    ccu::Variable bytes;
    ccu::Event event;
    SCATTER_CCU_TRY(InitAdaptiveLocal(source, transfer.source, transfer.sourceOffset, vars));
    SCATTER_CCU_TRY(InitAdaptiveLocal(destination, transfer.destination, transfer.destinationOffset, vars));
    SCATTER_CCU_TRY(CcuVariableAssignImm(bytes.handle, transfer.bytes));
    SCATTER_CCU_TRY(ccu::LocalCopy(destination, source, bytes, event, COMPLETE));
    SCATTER_CCU_TRY(ccu::EventWait(event, COMPLETE));
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }
    const auto &kernelArg = *static_cast<const ScatterKernelArg *>(arg);
    SCATTER_CCU_TRY(ValidateKernelArg(kernelArg));

    TaskVariables vars;
    SCATTER_CCU_TRY(LoadTaskVariables(vars));
    if (kernelArg.pullRead != 0) {
        return kernelArg.role == ScatterKernelRole::SEND ? PublishPullSources(kernelArg, vars)
            : ReceivePull(kernelArg, vars);
    }
    if (kernelArg.role == ScatterKernelRole::SEND) {
        return SendPeers(kernelArg, vars);
    }
    if (kernelArg.role == ScatterKernelRole::RECEIVE) {
        return ReceivePeers(kernelArg, vars);
    }
    return CopyLocal(kernelArg, vars);
}

CcuResult AdaptiveCcuKernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }
    const auto &kernelArg = *static_cast<const AdaptiveKernelArg *>(arg);
    SCATTER_CCU_TRY(ValidateAdaptiveArg(kernelArg));
    AdaptiveVariables vars;
    SCATTER_CCU_TRY(LoadAdaptiveVariables(vars));
    if (kernelArg.role == AdaptiveRole::SEND_DATA) {
        return AdaptiveSend(kernelArg, vars);
    }
    if (kernelArg.role == AdaptiveRole::RECEIVE || kernelArg.role == AdaptiveRole::PUBLISH_READY ||
        kernelArg.role == AdaptiveRole::RECEIVE_TAIL) {
        return AdaptiveReceive(kernelArg, vars);
    }
    if (kernelArg.role == AdaptiveRole::FINISH_RECEIVE) {
        // This wait-only kernel must explicitly bind to the upstream die,
        // exactly as WAIT_ACK does; default-die placement is unsafe.
        CcuVariableHandle bindingHandle = 0;
        SCATTER_CCU_TRY(CcuVariableCreateByChannel(kernelArg.channels[0], OUTPUT_VAR, &bindingHandle));
        SCATTER_CCU_TRY(ccu::NotifyWait(kernelArg.channels[0], DONE_NOTIFY, COMPLETE));
        SCATTER_CCU_TRY(ccu::NotifyWait(kernelArg.channels[0], ADAPTIVE_REQUEST_NOTIFY, ADAPTIVE_ACK_REQUEST));
        return ccu::NotifyRecord(kernelArg.channels[0], ACK_NOTIFY, COMPLETE);
    }
    if (kernelArg.role == AdaptiveRole::REQUEST_ACK) {
        for (uint32_t i = 0; i < kernelArg.peerCount; ++i) {
            SCATTER_CCU_TRY(ccu::NotifyRecord(kernelArg.channels[i], ADAPTIVE_REQUEST_NOTIFY, ADAPTIVE_ACK_REQUEST));
        }
        return CCU_SUCCESS;
    }
    if (kernelArg.role == AdaptiveRole::WAIT_ACK) {
        for (uint32_t i = 0; i < kernelArg.peerCount; ++i) {
            // Bind this wait-only kernel to the channel's CCU die. Creating
            // the handle records a compilation dependency on the existing
            // channel XN; it neither reads/writes the XN nor adds communication.
            CcuVariableHandle bindingHandle = 0;
            SCATTER_CCU_TRY(CcuVariableCreateByChannel(kernelArg.channels[i], OUTPUT_VAR, &bindingHandle));
            SCATTER_CCU_TRY(ccu::NotifyWait(kernelArg.channels[i], ACK_NOTIFY, COMPLETE));
        }
        return CCU_SUCCESS;
    }
    if (kernelArg.role == AdaptiveRole::ACK_ONLY) {
        SCATTER_CCU_TRY(ccu::NotifyWait(kernelArg.channels[0], ADAPTIVE_REQUEST_NOTIFY, ADAPTIVE_ACK_REQUEST));
        return ccu::NotifyRecord(kernelArg.channels[0], ACK_NOTIFY, COMPLETE);
    }
    return AdaptiveCopy(kernelArg.transfers[0], vars);
}

#undef SCATTER_CCU_TRY
} // namespace ops_hccl
