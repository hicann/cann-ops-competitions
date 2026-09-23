/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include "custom.h"

namespace ops_hccl {

enum class ScatterKernelRole : uint32_t {
    SEND = 0,
    RECEIVE = 1,
    LOCAL_COPY = 2
};

// MS capture is bounded by the registered input size. The CCU_SCHED pack has at most 128
// 4-KiB slices per die; physical admission is still decided by registration.
constexpr uint64_t SCATTER_PULL_MS_SLICE_BYTES = 4096;
constexpr uint32_t SCATTER_PULL_MS_MAX_BUFFERS = 128;
constexpr uint64_t SCATTER_PULL_MS_MAX_BYTES =
    SCATTER_PULL_MS_SLICE_BYTES * SCATTER_PULL_MS_MAX_BUFFERS;

constexpr uint32_t SCATTER_PULL_MS_TAIL_BUFFERS = 4;
constexpr uint64_t SCATTER_PULL_MS_TAIL_BYTES =
    SCATTER_PULL_MS_SLICE_BYTES * SCATTER_PULL_MS_TAIL_BUFFERS;

// Registration fixes the peer list. All channels of a SEND kernel must belong
// to the same local CCU die; Host launches separate groups with separate resources.
// dataPath specializes the instruction graph at registration. Exactly one SEND
// group can own the optional root-copy branch; other roles must set copyOwner=0.
struct ScatterKernelArg {
    ScatterKernelRole role = ScatterKernelRole::LOCAL_COPY;
    uint32_t peerCount = 0;
    ScatterDataPath dataPath = ScatterDataPath::SINGLE_WAVE;
    uint32_t copyOwner = 0;
    uint32_t mergeEvents = 0;
    uint32_t readyWrite = 0;
    // Single-wave copyOwner only; Host still gates each call by full-input alias safety.
    uint32_t earlyCopy = 0;
    // SINGLE_WAVE SEND publishes source metadata; RECEIVE reads its own slice.
    // ACK still confirms completed data access before root can reuse input.
    uint32_t pullRead = 0;
    // SINGLE_WAVE Push only. READY and completed Writes precede DONE; the
    // receiver consumes DONE before publishing any next-invocation READY.
    // This does not apply to Pull, whose ACK protects root input lifetime.
    uint32_t shortPush = 0;
    // Pull SEND/copyOwner only. Dynamic full-input alias guard still applies.
    uint32_t pullEarlyCopy = 0;
    // Address representation is fixed at registration; pointers remain dynamic.
    ScatterPullAddressMode pullAddressMode = ScatterPullAddressMode::SLICE;
    uint32_t pullRoot = 0; // Used only by non-SLICE Pull modes.
    // SINGLE_WAVE Push SEND/copyOwner only. Keep Copy launch after Writes;
    // defer only its completion wait until this group's DONE/ACK is drained.
    uint32_t deferCopyWait = 0;
    // BASE_OFFSET Pull SEND only: publish in Host's topology-aware peer order.
    uint32_t pullPeerOrder = 0;
    // RECEIVE Pull only. Zero keeps direct GM Read. Nonzero is the exact B
    // represented by this registered plan, not a runtime pointer or token.
    uint64_t pullMsBytes = 0;
    // RECEIVE only: B > 16 KiB. Prefix goes to output; only the tail uses MS.
    // The complete B remains in pullMsBytes and in the registration/cache key.
    uint32_t pullMsTail = 0;
    ChannelHandle channels[MAX_RANK_SIZE] = {};
    uint32_t peers[MAX_RANK_SIZE] = {};
};

// Host passes B > 0, loopCount = (B - 1) / MAX_DATA_SIZE and
// lastBytes = (B - 1) % MAX_DATA_SIZE + 1. SEND uses the full input base;
// LOCAL_COPY uses the already-offset root slice as its source. SINGLE_WAVE
// requires 0 < B <= MAX_DATA_SIZE and emits no While instructions. CHUNKED
// uses the full-wave count and strictly positive lastBytes supplied by Host.
// copyEnabled is decided for each call, after checking output against the
// complete root input range; rootSource is the already-offset root slice.
// V021 specializations keep all nine leading LoadArg instructions:
// THREE_SOURCES SEND uses slots 0/5/6 for non-root ranks in global rank order.
// BASE_OFFSET RECEIVE uses slot 5 for myRank * B; root still uses input base.
// These modes are SINGLE_WAVE only. LOCAL_COPY keeps its original source slot.
enum ScatterTaskArg : uint32_t {
    SCATTER_SRC_ADDR = 0,
    SCATTER_SRC_TOKEN,
    SCATTER_DST_ADDR,
    SCATTER_DST_TOKEN,
    SCATTER_BYTES,
    SCATTER_LOOP_COUNT,
    SCATTER_LAST_BYTES,
    SCATTER_COPY_ENABLED,
    SCATTER_ROOT_SRC_ADDR,
    SCATTER_TASK_ARG_COUNT
};

CcuResult CcuKernel(CcuKernelArg arg);

constexpr uint32_t SCATTER_MAX_TRANSFERS = 32;
enum class AdaptiveRole : uint32_t {
    SEND_DATA = 0, RECEIVE = 1, WAIT_ACK = 2, ACK_ONLY = 3, LOCAL_COPY = 4, REQUEST_ACK = 5,
    PUBLISH_READY = 6, RECEIVE_TAIL = 7, FINISH_RECEIVE = 8
};
constexpr AdaptiveRole SCATTER_SEND_PHASES[] = {
    AdaptiveRole::SEND_DATA, AdaptiveRole::REQUEST_ACK, AdaptiveRole::WAIT_ACK
};
// Consume each request bit separately; the other phase's bit can arrive early.
constexpr uint32_t ADAPTIVE_REQUEST_NOTIFY = 3;
constexpr uint16_t ADAPTIVE_DATA_REQUEST = 1;
constexpr uint16_t ADAPTIVE_ACK_REQUEST = 2;
// DONE notify bit 1 remains whole-peer completion. Bit 2 releases only the
// helper's scratch tails; FINISH_RECEIVE still drains whole-peer DONE and ACK.
constexpr uint16_t ADAPTIVE_TAIL_COMPLETE = 2;
enum class ScatterBuffer : uint32_t { INPUT = 0, OUTPUT = 1, SCRATCH = 2 };
struct ScatterTransfer {
    uint32_t channelIndex = 0;
    ScatterBuffer source = ScatterBuffer::INPUT;
    ScatterBuffer destination = ScatterBuffer::OUTPUT;
    uint32_t reserved = 0;
    uint64_t sourceOffset = 0;
    uint64_t destinationOffset = 0;
    uint64_t bytes = 0;
};
struct AdaptiveKernelArg {
    AdaptiveRole role = AdaptiveRole::SEND_DATA;
    uint32_t peerCount = 0;
    uint32_t transferCount = 0;
    uint32_t deferAck = 0;
    // Registration-time SEND_DATA option; Host owns root/scene admission.
    // It only moves ACK_REQUEST after each peer's DONE, never ACK waiting.
    uint32_t fuseAckRequest = 0;
    uint32_t earlyTail = 0;
    uint32_t readyWrite = 0; // B only: 2x8 multi-peer SEND_DATA.
    // Root single-wave SEND only: issue helper OUTPUT after its SCRATCH tails.
    // Reuses existing per-transfer events and earlyTail notifications.
    uint32_t deferHelperOutput = 0;
    // Round-robin tail completion across multiple helpers within this send group.
    uint32_t interleaveTails = 0;
    uint32_t copyOwner = 0;
    uint64_t copySourceOffset = 0;
    uint64_t copyBytes = 0;
    // earlyTail=2 emits one DONE bit per scratch transfer, starting at bit 1.
    // A helper either consumes the union in RECEIVE_TAIL or consumes each bit
    // just before its corresponding Write. Host admits the latter on one die.
    uint16_t tailWaitMask = ADAPTIVE_TAIL_COMPLETE;
    ChannelHandle tailChannel = 0;
    uint16_t tailWaitMasks[SCATTER_MAX_TRANSFERS] = {};
    ChannelHandle channels[MAX_RANK_SIZE] = {};
    uint32_t peers[MAX_RANK_SIZE] = {};
    uint16_t readyMasks[MAX_RANK_SIZE] = {};
    ScatterTransfer transfers[SCATTER_MAX_TRANSFERS] = {};
};
// Runtime addresses are never part of a registered plan. INPUT/OUTPUT addresses
// include WAVE_OFFSET; scratch offsets always address the single reusable wave.
enum AdaptiveTaskArg : uint32_t {
    ADAPTIVE_INPUT = 0, ADAPTIVE_INPUT_TOKEN, ADAPTIVE_OUTPUT, ADAPTIVE_OUTPUT_TOKEN,
    ADAPTIVE_SCRATCH, ADAPTIVE_SCRATCH_TOKEN, ADAPTIVE_WAVE_OFFSET,
    ADAPTIVE_COPY_ENABLED = 7, ADAPTIVE_PAD_BEGIN = 8, ADAPTIVE_ARG_COUNT = 13
};
// The fixed SDK carries 13 arguments per SQE. Loading the complete argument
// block keeps a standalone short kernel distinct from a continuation SQE.
constexpr uint32_t SCATTER_SQE_ARG_CAPACITY = 13;
static_assert(ADAPTIVE_WAVE_OFFSET == 6 && ADAPTIVE_COPY_ENABLED == 7 && ADAPTIVE_PAD_BEGIN == 8,
    "Adaptive data argument slots must remain unchanged");
static_assert(ADAPTIVE_ARG_COUNT == SCATTER_SQE_ARG_CAPACITY,
    "Adaptive kernels must consume one complete SQE argument block");
static_assert(SCATTER_TASK_ARG_COUNT <= SCATTER_SQE_ARG_CAPACITY,
    "Direct kernels must fit within one SQE argument block");
CcuResult AdaptiveCcuKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
