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
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>

#include <hccl/hcomm_primitives.h>
#include "ccu_launch.h"
#include "ccu_res.h"
#include "ccu_kernel.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t RELAY_ALIGNMENT = 512ULL * SCATTER_RELAY_STAGES;
constexpr uint32_t START_NOTIFY = 0U;
// Relay helper forward lanes use START_NOTIFY=0 for Begin().  A single
// READY notify cannot be reused for stage0 and stage1 because both RECORDs
// can be enqueued on the producer stream before the first WAIT is consumed
// on the forward stream.  Keep two READY slots (1/2) and reuse them only
// after the matching release handshake closes the previous slot lifetime.
constexpr uint32_t READY_NOTIFY_BASE = 1U;
constexpr uint32_t RELAY_READY_SLOT_COUNT = 2U;

struct TransferArgs {
    uint64_t input = 0;
    uint64_t output = 0;
    uint64_t scratch = 0;
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    uint64_t stride = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t direct = 0;
    uint64_t relay = 0;
    uint64_t rootCopyEnabled = 0;
};

bool RelayGeometry(uint64_t bytes, uint64_t &direct, uint64_t &relay)
{
    // bytes <= 256 MiB in this implementation, but avoid multiplication overflow.
    if (bytes > std::numeric_limits<uint64_t>::max() / SCATTER_RELAY_NUM) {
        return false;
    }
    relay = ((bytes * SCATTER_RELAY_NUM / SCATTER_RELAY_DEN) / RELAY_ALIGNMENT) * RELAY_ALIGNMENT;
    direct = bytes - relay;
    return relay != 0 && relay < bytes;
}

// A1 functional audit for the proven 4x3 Full-4Pair plan.  P1 showed that
// top-level useRelay/capability gates alone are not enough to explain why the
// measured time stayed at the old Direct baseline.  This check makes every
// non-tiny 4x3 window prove that the actual 5/12 split is valid before any
// kernel is launched.  Tiny residuals remain eligible for Direct so odd byte
// counts stay semantically safe.
HcclResult Audit4x3RelayWindow(const ScatterExecutionPlan &plan, const TransferArgs &x, bool relayWindow)
{
    constexpr uint64_t MIN_RELAY_WINDOW =
        (RELAY_ALIGNMENT * SCATTER_RELAY_DEN + SCATTER_RELAY_NUM - 1ULL) / SCATTER_RELAY_NUM;
    const bool majorWindow = x.bytes >= MIN_RELAY_WINDOW;
    if (!majorWindow) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(!relayWindow || x.relay == 0U || x.direct == 0U ||
            x.relay >= x.bytes || x.direct + x.relay != x.bytes,
        HCCL_ERROR("A1 4x3 major window fell out of Relay geometry offset[%llu] bytes[%llu] direct[%llu] relay[%llu]",
            static_cast<unsigned long long>(x.offset), static_cast<unsigned long long>(x.bytes),
            static_cast<unsigned long long>(x.direct), static_cast<unsigned long long>(x.relay)),
        HCCL_E_INTERNAL);

    CHK_PRT_RET((x.relay % RELAY_ALIGNMENT) != 0U || x.relay > plan.scratchBytes,
        HCCL_ERROR("A1 4x3 Relay alignment/scratch mismatch offset[%llu] relay[%llu] align[%llu] scratch[%llu]",
            static_cast<unsigned long long>(x.offset), static_cast<unsigned long long>(x.relay),
            static_cast<unsigned long long>(RELAY_ALIGNMENT),
            static_cast<unsigned long long>(plan.scratchBytes)),
        HCCL_E_INTERNAL);

    const uint64_t idealRelay = x.bytes * SCATTER_RELAY_NUM / SCATTER_RELAY_DEN;
    CHK_PRT_RET(x.relay > idealRelay || idealRelay - x.relay >= RELAY_ALIGNMENT,
        HCCL_ERROR("A1 4x3 Relay ratio mismatch offset[%llu] bytes[%llu] direct[%llu] relay[%llu] ideal[%llu]",
            static_cast<unsigned long long>(x.offset), static_cast<unsigned long long>(x.bytes),
            static_cast<unsigned long long>(x.direct), static_cast<unsigned long long>(x.relay),
            static_cast<unsigned long long>(idealRelay)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult MemToken(uint64_t addr, uint64_t bytes, uint64_t &token)
{
    CHK_PRT_RET(addr == 0 || bytes == 0 || addr > std::numeric_limits<uint64_t>::max() - bytes,
        HCCL_ERROR("invalid memory token range addr[%llu] size[%llu]",
            static_cast<unsigned long long>(addr), static_cast<unsigned long long>(bytes)), HCCL_E_PARA);
    CHK_RET_CCU(HcommCcuGetMemToken(addr, bytes, &token));
    return HCCL_SUCCESS;
}

HcclResult Begin(const std::vector<ThreadHandle> &threads)
{
    for (size_t i = 1; i < threads.size(); ++i) {
        CHK_RET(HcommThreadNotifyRecordOnThread(threads[0], threads[i], START_NOTIFY));
        CHK_RET(HcommThreadNotifyWaitOnThread(threads[i], START_NOTIFY, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult JoinAtBase(const std::vector<ThreadHandle> &threads, uint32_t joinBase)
{
    for (size_t i = 1; i < threads.size(); ++i) {
        const uint32_t idx = joinBase + static_cast<uint32_t>(i - 1U);
        CHK_RET(HcommThreadNotifyRecordOnThread(threads[i], threads[0], idx));
        CHK_RET(HcommThreadNotifyWaitOnThread(threads[0], idx, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult Join(const std::vector<ThreadHandle> &threads)
{
    return JoinAtBase(threads, SCATTER_JOIN_BASE);
}

HcclResult RelayJoin(const std::vector<ThreadHandle> &threads)
{
    return JoinAtBase(threads, SCATTER_RELAY_JOIN_BASE);
}

HcclResult LegacyBegin(const std::vector<ThreadHandle> &threads)
{
    for (size_t i = 1; i < threads.size(); ++i) {
        CHK_RET(HcommThreadNotifyRecordOnThread(threads[0], threads[i], 0U));
        CHK_RET(HcommThreadNotifyWaitOnThread(threads[i], 0U, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult LegacyJoin(const std::vector<ThreadHandle> &threads)
{
    for (size_t i = 1; i < threads.size(); ++i) {
        const uint32_t idx = static_cast<uint32_t>(i - 1U);
        CHK_RET(HcommThreadNotifyRecordOnThread(threads[i], threads[0], idx));
        CHK_RET(HcommThreadNotifyWaitOnThread(threads[0], idx, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

bool LegacyAssignLanes(const std::vector<uint32_t> &channelCounts, size_t laneCount,
    std::vector<size_t> &kernelLanes)
{
    const size_t kernelCount = channelCounts.size();
    if (kernelCount == 0 || laneCount == 0 || laneCount > kernelCount) { return false; }
    std::vector<size_t> order(kernelCount);
    for (size_t i = 0; i < kernelCount; ++i) { order[i] = i; }
    std::sort(order.begin(), order.end(), [&channelCounts](size_t lhs, size_t rhs) {
        return channelCounts[lhs] > channelCounts[rhs];
    });
    std::vector<uint64_t> laneLoad(laneCount, 0U);
    kernelLanes.assign(kernelCount, 0U);
    for (const size_t kernelIdx : order) {
        size_t lane = 0U;
        for (size_t i = 1U; i < laneCount; ++i) {
            if (laneLoad[i] < laneLoad[lane]) { lane = i; }
        }
        kernelLanes[kernelIdx] = lane;
        laneLoad[lane] += channelCounts[kernelIdx];
    }
    return true;
}

HcclResult LegacyDirectLaunch(ThreadHandle thread, CcuKernelHandle kernel, const TransferArgs &x)
{
    const uint64_t args[] = {x.input, x.output, x.inputToken, x.outputToken,
        x.stride, x.offset, x.bytes};
    CHK_RET_CCU(HcommCcuKernelLaunch(thread, kernel, args, 7U));
    return HCCL_SUCCESS;
}

HcclResult RunLegacy16Direct(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<ThreadHandle> &threads, const ScatterExecutionPlan &plan, TransferArgs x)
{
    if (param.myRank == param.root) {
        const uint64_t localInput = x.input + static_cast<uint64_t>(param.root) * plan.sliceBytes;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(param.cpuThread, param.outputPtr,
            reinterpret_cast<void *>(localInput), plan.sliceBytes)));
    }
    std::vector<size_t> lanes;
    const bool parallel = param.myRank == param.root && threads.size() >= 2U &&
        resCtx.directKernelChannelCounts.size() == resCtx.directKernels.size() &&
        LegacyAssignLanes(resCtx.directKernelChannelCounts, threads.size(), lanes);
    if (parallel) { CHK_RET(LegacyBegin(threads)); }
    while (x.offset < x.stride) {
        x.bytes = std::min(SCATTER_WINDOW_BYTES, x.stride - x.offset);
        for (size_t i = 0; i < resCtx.directKernels.size(); ++i) {
            const ThreadHandle thread = parallel ? threads[lanes[i]] : param.cpuThread;
            CHK_RET(LegacyDirectLaunch(thread, resCtx.directKernels[i], x));
        }
        x.offset += x.bytes;
    }
    if (parallel) { CHK_RET(LegacyJoin(threads)); }
    return HCCL_SUCCESS;
}

HcclResult Direct(ThreadHandle thread, CcuKernelHandle kernel, const TransferArgs &x,
    bool batchMode, uint64_t secondBytes)
{
    CHK_PRT_RET(x.bytes == 0 || x.bytes > SCATTER_WINDOW_BYTES || secondBytes > SCATTER_WINDOW_BYTES ||
            x.offset > x.stride || x.bytes > x.stride - x.offset ||
            secondBytes > x.stride - x.offset - x.bytes || (!batchMode && secondBytes != 0),
        HCCL_ERROR("invalid direct window geometry"), HCCL_E_INTERNAL);
    const uint64_t args[] = {x.input, x.output, x.inputToken, x.outputToken,
        x.stride, x.offset, x.bytes, x.rootCopyEnabled, secondBytes};
    CHK_RET_CCU(HcommCcuKernelLaunch(thread, kernel, args, batchMode ? 9U : 8U));
    return HCCL_SUCCESS;
}

HcclResult Relay(ThreadHandle thread, CcuKernelHandle kernel, const TransferArgs &x,
    uint64_t relayOffset, uint64_t chunkBytes, uint64_t scratchOffset)
{
    CHK_PRT_RET(x.bytes > SCATTER_WINDOW_BYTES || x.offset > x.stride || x.bytes > x.stride - x.offset ||
            x.direct > x.bytes || x.relay != x.bytes - x.direct ||
            relayOffset > x.relay || chunkBytes > x.relay - relayOffset ||
            scratchOffset > x.relay || chunkBytes > x.relay - scratchOffset,
        HCCL_ERROR("invalid relay window geometry"), HCCL_E_INTERNAL);
    // Fifteen LoadArg declarations; the final scalar gates in-place self-copy.
    const uint64_t args[] = {x.input, x.output, x.scratch, x.inputToken, x.outputToken, x.scratchToken,
        x.stride, x.direct, x.relay, relayOffset, chunkBytes, scratchOffset, x.offset, x.bytes, x.rootCopyEnabled};
    CHK_RET_CCU(HcommCcuKernelLaunch(thread, kernel, args, sizeof(args) / sizeof(args[0])));
    return HCCL_SUCCESS;
}


HcclResult RelayLoadOrFinal(ThreadHandle thread, CcuKernelHandle kernel, const TransferArgs &x,
    uint64_t mode, uint64_t relayOffset, uint64_t chunkBytes, uint64_t scratchOffset)
{
    CHK_PRT_RET(mode > 2U || x.bytes > SCATTER_WINDOW_BYTES || x.offset > x.stride ||
            x.bytes > x.stride - x.offset || x.direct > x.bytes || x.relay != x.bytes - x.direct ||
            relayOffset > x.relay || chunkBytes > x.relay - relayOffset ||
            scratchOffset > x.relay || chunkBytes > x.relay - scratchOffset,
        HCCL_ERROR("invalid fused relay-link geometry mode[%llu]",
            static_cast<unsigned long long>(mode)), HCCL_E_INTERNAL);
    // CcuScatterRelayLoadKernel repurposes runtime arg #8 as a small mode:
    // 0/1 choose the helper's two relay targets; 2 delivers the helper's own
    // final slice.  All other runtime argument positions retain the Relay ABI.
    const uint64_t args[] = {x.input, x.output, x.scratch, x.inputToken, x.outputToken, x.scratchToken,
        x.stride, x.direct, mode, relayOffset, chunkBytes, scratchOffset, x.offset, x.bytes, x.rootCopyEnabled};
    CHK_RET_CCU(HcommCcuKernelLaunch(thread, kernel, args, sizeof(args) / sizeof(args[0])));
    return HCCL_SUCCESS;
}

std::vector<size_t> AssignLanes(const std::vector<uint32_t> &weights, size_t laneCount)
{
    std::vector<size_t> order(weights.size());
    for (size_t i = 0; i < order.size(); ++i) { order[i] = i; }
    std::stable_sort(order.begin(), order.end(), [&weights](size_t a, size_t b) { return weights[a] > weights[b]; });
    std::vector<uint64_t> loads(laneCount, 0);
    std::vector<size_t> lanes(weights.size(), 0);
    for (size_t i : order) {
        const size_t lane = static_cast<size_t>(std::min_element(loads.begin(), loads.end()) - loads.begin());
        lanes[i] = lane;
        loads[lane] += weights[i];
    }
    return lanes;
}

HcclResult RunDirect(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<ThreadHandle> &threads, const ScatterExecutionPlan &plan, TransferArgs x,
    uint64_t secondBytes)
{
    const bool parallel = param.myRank == param.root && threads.size() > 1U;
    std::vector<size_t> lanes(resCtx.directKernels.size(), 0);
    if (parallel) {
        lanes = AssignLanes(resCtx.directKernelChannelCounts, threads.size());
    }
    for (size_t i = 0; i < resCtx.directKernels.size(); ++i) {
        const ThreadHandle thread = parallel ? threads[lanes[i]] : param.cpuThread;
        CHK_RET(Direct(thread, resCtx.directKernels[i], x, plan.batchDirect, secondBytes));
    }
    return HCCL_SUCCESS;
}

struct RootWork {
    bool helper = false;
    size_t index = 0;
    uint64_t estimatedTimeUnits = 0;
    size_t lane = 0;
};

// A2: each 4x3 helper owns two relay pairs.  Four stages are mapped onto two
// ping-pong slots per pair so Load(stage+1) can overlap Forward(stage).
// With fanout=2 and stages=4, required helper scratch is still exactly relay:
//   2 pairs * 2 slots * (relay / 4) == relay.
uint64_t RelayScratchOffset(size_t pair, uint32_t stage, uint64_t chunk)
{
    const uint64_t slot = static_cast<uint64_t>(stage & 1U);
    return (static_cast<uint64_t>(pair) * 2ULL + slot) * chunk;
}

uint32_t RelayReleaseNotify(size_t pair, uint32_t stage)
{
    return static_cast<uint32_t>(pair) * SCATTER_RELAY_SLOT_COUNT +
        (stage & (SCATTER_RELAY_SLOT_COUNT - 1U));
}

HcclResult AuditStreamingRelayGeometry(size_t pairCount, uint64_t relay, uint64_t chunk)
{
    static_assert(SCATTER_RELAY_STAGES == 4U, "A2 requires four relay stages");
    static_assert(SCATTER_RELAY_FANOUT == 2U, "A2 4x3 scratch proof assumes fanout two");
    CHK_PRT_RET(pairCount == 0U || pairCount > SCATTER_RELAY_FANOUT ||
            chunk == 0U || chunk * SCATTER_RELAY_STAGES != relay,
        HCCL_ERROR("A2 invalid streaming geometry pairs[%zu] relay[%llu] chunk[%llu] stages[%u]",
            pairCount, static_cast<unsigned long long>(relay), static_cast<unsigned long long>(chunk),
            SCATTER_RELAY_STAGES), HCCL_E_INTERNAL);
    const uint64_t required = static_cast<uint64_t>(pairCount) * 2ULL * chunk;
    CHK_PRT_RET(required > relay,
        HCCL_ERROR("A2 ping-pong scratch exceeds relay pairs[%zu] required[%llu] relay[%llu]",
            pairCount, static_cast<unsigned long long>(required),
            static_cast<unsigned long long>(relay)), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RunRelayRoot(const AlgResourceCtx &resCtx, const std::vector<ThreadHandle> &threads,
    const TransferArgs &x)
{
    const size_t helperCount = resCtx.relayHelperFinalKernels.size();
    const size_t directCount = resCtx.relayDirectKernels.size();
    CHK_PRT_RET(helperCount == 0U || directCount == 0U || threads.empty() ||
            resCtx.relayLoadKernels.empty() || resCtx.relayLoadKernels.size() % helperCount != 0U ||
            resCtx.relayDirectKernelWeightUnits.size() != directCount,
        HCCL_ERROR("invalid P4 Relay root geometry helpers[%zu] direct[%zu] loads[%zu] threads[%zu] weights[%zu]",
            helperCount, directCount, resCtx.relayLoadKernels.size(), threads.size(),
            resCtx.relayDirectKernelWeightUnits.size()), HCCL_E_INTERNAL);
    const size_t fanout = resCtx.relayLoadKernels.size() / helperCount;

    // P4 execution-resource scheduler.  RegisterPeerGroups already creates one
    // RelayDirect kernel per (layerId, dieId) PathKey.  When enough TS threads
    // were acquired, give every such path group and every helper its own lane.
    // This removes the P3 four-lane multiplexing that could serialize otherwise
    // independent root resources.  If an exotic topology exposes more work
    // groups than the eight-lane safety cap, retain the proven service-time
    // list scheduler as a deterministic fallback.
    std::vector<size_t> helperLanes(helperCount, 0U);
    std::vector<size_t> directLanes(directCount, 0U);
    const size_t workCount = helperCount + directCount;
    if (threads.size() >= workCount) {
        for (size_t i = 0; i < helperCount; ++i) { helperLanes[i] = i; }
        for (size_t i = 0; i < directCount; ++i) { directLanes[i] = helperCount + i; }
    } else {
        std::vector<RootWork> work;
        for (size_t i = 0; i < directCount; ++i) {
            work.push_back(RootWork{false, i, resCtx.relayDirectKernelWeightUnits[i], 0});
        }
        // Compare estimated service TIME, not raw bytes. Local link B, root Clos 4B.
        // Per-helper bytes = (12 + fanout*5)/12 S. All weights use S/(12*4B).
        const uint64_t helperWeight = 4ULL *
            (SCATTER_RELAY_DEN + static_cast<uint64_t>(fanout) * SCATTER_RELAY_NUM);
        for (size_t i = 0; i < helperCount; ++i) {
            work.push_back(RootWork{true, i, helperWeight, 0});
        }
        std::stable_sort(work.begin(), work.end(), [](const RootWork &a, const RootWork &b) {
            if (a.estimatedTimeUnits != b.estimatedTimeUnits) {
                return a.estimatedTimeUnits > b.estimatedTimeUnits;
            }
            if (a.helper != b.helper) { return a.helper; }
            return a.index < b.index;
        });
        std::vector<uint64_t> loads(threads.size(), 0U);
        for (auto &w : work) {
            w.lane = static_cast<size_t>(std::min_element(loads.begin(), loads.end()) - loads.begin());
            loads[w.lane] += w.estimatedTimeUnits;
            if (w.helper) {
                helperLanes[w.index] = w.lane;
            } else {
                directLanes[w.index] = w.lane;
            }
        }
    }

    const uint64_t chunk = x.relay / SCATTER_RELAY_STAGES;
    CHK_RET(AuditStreamingRelayGeometry(fanout, x.relay, chunk));

    const auto launchHelperStage = [&](size_t helper, uint32_t stage) -> HcclResult {
        const ThreadHandle lane = threads[helperLanes[helper]];
        for (size_t pair = 0; pair < fanout; ++pair) {
            const size_t idx = helper * fanout + pair;
            const uint64_t scratchOffset = RelayScratchOffset(pair, stage, chunk);
            CHK_RET(RelayLoadOrFinal(lane, resCtx.relayLoadKernels[idx], x, pair,
                static_cast<uint64_t>(stage) * chunk, chunk, scratchOffset));
        }
        return HCCL_SUCCESS;
    };

    // Seed one chunk on BOTH helper links first.  As soon as stage0 is in flight,
    // enqueue every root->remote PathKey on its assigned lane.  Then continue the
    // helper pipeline stage-major so helper0 cannot queue all four stages ahead
    // of helper1/direct traffic.  This is the concrete P4 test of whether the
    // theoretical 4B + B + B capacity can be exposed by TS/CCU concurrency.
    for (size_t helper = 0; helper < helperCount; ++helper) {
        CHK_RET(launchHelperStage(helper, 0U));
    }
    for (size_t i = 0; i < directCount; ++i) {
        CHK_RET(Relay(threads[directLanes[i]], resCtx.relayDirectKernels[i], x, 0, 0, 0));
    }
    for (uint32_t stage = 1U; stage < SCATTER_RELAY_STAGES; ++stage) {
        for (size_t helper = 0; helper < helperCount; ++helper) {
            CHK_RET(launchHelperStage(helper, stage));
        }
    }
    for (size_t helper = 0; helper < helperCount; ++helper) {
        CHK_RET(RelayLoadOrFinal(threads[helperLanes[helper]],
            resCtx.relayHelperFinalKernels[helper], x, 2U, 0U, 0U, 0U));
    }
    return HCCL_SUCCESS;
}

HcclResult RunRelayPeer(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<ThreadHandle> &threads, const TransferArgs &x)
{
    const uint64_t chunk = x.relay / SCATTER_RELAY_STAGES;
    if (resCtx.localRole == SCATTER_ROLE_HELPER) {
        const size_t pairCount = resCtx.relayLoadKernels.size();
        CHK_PRT_RET(pairCount == 0U || resCtx.relayForwardKernels.size() != pairCount ||
                threads.size() != pairCount + 1U,
            HCCL_ERROR("invalid helper Relay schedule loads[%zu] forwards[%zu] threads[%zu]",
                pairCount, resCtx.relayForwardKernels.size(), threads.size()), HCCL_E_INTERNAL);
        CHK_RET(AuditStreamingRelayGeometry(pairCount, x.relay, chunk));

        // A2 true two-slot pipeline.  stage0/1 fill different slots without
        // waiting for the preceding Forward.  Before stage2/3 reuses slot0/1,
        // consume the completion token produced by Forward(stage-2).
        // The pair's forward TS lane serializes its own writes, while the main
        // helper lane is free to receive the next relay chunk concurrently.
        for (uint32_t stage = 0; stage < SCATTER_RELAY_STAGES; ++stage) {
            for (size_t pair = 0; pair < pairCount; ++pair) {
                const ThreadHandle forward = threads[1U + pair];
                if (stage >= SCATTER_RELAY_SLOT_COUNT) {
                    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread,
                        RelayReleaseNotify(pair, stage), CUSTOM_TIMEOUT));
                }
                const uint64_t scratchOffset = RelayScratchOffset(pair, stage, chunk);
                CHK_RET(RelayLoadOrFinal(param.cpuThread, resCtx.relayLoadKernels[pair], x, pair,
                    static_cast<uint64_t>(stage) * chunk, chunk, scratchOffset));
                const uint32_t readyNotify = READY_NOTIFY_BASE +
                    (stage & (RELAY_READY_SLOT_COUNT - 1U));
                CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, forward, readyNotify));
                CHK_RET(HcommThreadNotifyWaitOnThread(forward, readyNotify, CUSTOM_TIMEOUT));
                CHK_RET(Relay(forward, resCtx.relayForwardKernels[pair], x,
                    static_cast<uint64_t>(stage) * chunk, chunk, scratchOffset));
                // Only stages that will have their slot reused need an explicit
                // release token.  The final two stages are fenced by Join().
                if (stage + SCATTER_RELAY_SLOT_COUNT < SCATTER_RELAY_STAGES) {
                    CHK_RET(HcommThreadNotifyRecordOnThread(forward, param.cpuThread,
                        RelayReleaseNotify(pair, stage)));
                }
            }
        }
        CHK_RET(RelayLoadOrFinal(param.cpuThread, resCtx.relayHelperFinalKernels[0],
            x, 2U, 0U, 0U, 0U));
        return HCCL_SUCCESS;
    }

    // A relayed target receives its disjoint prefix and tail on separate lanes.
    const ThreadHandle directThread = resCtx.localRole == SCATTER_ROLE_TARGET ? threads[1] : param.cpuThread;
    for (auto handle : resCtx.relayDirectKernels) {
        CHK_RET(Relay(directThread, handle, x, 0, 0, 0));
    }
    for (uint32_t stage = 0; stage < SCATTER_RELAY_STAGES; ++stage) {
        for (auto handle : resCtx.relayForwardKernels) {
            CHK_RET(Relay(param.cpuThread, handle, x, stage * chunk, chunk, 0));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunRelayResidual(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<ThreadHandle> &threads, const TransferArgs &x)
{
    CHK_PRT_RET(x.bytes == 0U || x.relay != 0U || x.direct != x.bytes,
        HCCL_ERROR("V30.6 invalid relay residual bytes[%llu] direct[%llu] relay[%llu]",
            static_cast<unsigned long long>(x.bytes), static_cast<unsigned long long>(x.direct),
            static_cast<unsigned long long>(x.relay)), HCCL_E_INTERNAL);

#if HCCL_SCATTER_DIAGNOSTICS
    std::fprintf(stderr,
        "[V306_RESIDUAL] rank=%u root=%u role=%u bytes=%llu rdirect=%zu hfinal=%zu fwd=%zu threads=%zu\n",
        param.myRank, param.root, resCtx.localRole, static_cast<unsigned long long>(x.bytes),
        resCtx.relayDirectKernels.size(), resCtx.relayHelperFinalKernels.size(),
        resCtx.relayForwardKernels.size(), threads.size());
#endif

    if (param.myRank == param.root) {
        // For a sub-alignment tail there is no relay payload.  The already
        // registered RelayDirect kernels send the whole residual to ordinary
        // and relay-target peers because directBytes == sliceSize; the helper
        // final kernels cover the two local helpers.  This replaces the
        // duplicate Direct kernel family in a large 4x3 resource context.
        for (auto handle : resCtx.relayDirectKernels) {
            CHK_RET(Relay(param.cpuThread, handle, x, 0U, 0U, 0U));
        }
        for (auto handle : resCtx.relayHelperFinalKernels) {
            CHK_RET(RelayLoadOrFinal(param.cpuThread, handle, x, 2U, 0U, 0U, 0U));
        }
        return HCCL_SUCCESS;
    }

    if (resCtx.localRole == SCATTER_ROLE_HELPER) {
        CHK_PRT_RET(resCtx.relayHelperFinalKernels.size() != 1U,
            HCCL_ERROR("V30.6 helper residual requires one helper-final kernel"), HCCL_E_INTERNAL);
        return RelayLoadOrFinal(param.cpuThread, resCtx.relayHelperFinalKernels[0],
            x, 2U, 0U, 0U, 0U);
    }

    CHK_PRT_RET(resCtx.relayDirectKernels.size() != 1U,
        HCCL_ERROR("V30.6 peer residual requires one relay-direct kernel role[%u] kernels[%zu]",
            resCtx.localRole, resCtx.relayDirectKernels.size()), HCCL_E_INTERNAL);
    const ThreadHandle lane = (resCtx.localRole == SCATTER_ROLE_TARGET && threads.size() > 1U) ?
        threads[1] : param.cpuThread;
    return Relay(lane, resCtx.relayDirectKernels[0], x, 0U, 0U, 0U);
}

HcclResult HubCopy(ThreadHandle thread, CcuKernelHandle kernel, const TransferArgs &x,
    uint64_t sourceBase, uint64_t sourceToken, uint64_t receiveBase, uint64_t receiveToken,
    uint64_t sourceStride, uint64_t sourceOffset, uint64_t destinationOffset, uint64_t bytes,
    bool ingress = false, uint64_t pitch = 0)
{
    CHK_PRT_RET(bytes == 0 || bytes > SCATTER_WINDOW_BYTES || sourceBase == 0 || receiveBase == 0 ||
            (ingress && (pitch < bytes || x.bytes > SCATTER_HUB_WINDOW_BYTES)),
        HCCL_ERROR("invalid hub transfer arguments"), HCCL_E_INTERNAL);
    const uint64_t args[] = {sourceBase, sourceToken, receiveBase, receiveToken,
        sourceStride, sourceOffset, destinationOffset, bytes, x.rootCopyEnabled, pitch};
    CHK_RET_CCU(HcommCcuKernelLaunch(thread, kernel, args, ingress ? 10U : 9U));
    return HCCL_SUCCESS;
}

HcclResult HubJoin(const std::vector<ThreadHandle> &threads)
{
    for (size_t i = 1; i < threads.size(); ++i) {
        const uint32_t index = SCATTER_HUB_JOIN_BASE + static_cast<uint32_t>(i - 1U);
        CHK_RET(HcommThreadNotifyRecordOnThread(threads[i], threads[0], index));
        CHK_RET(HcommThreadNotifyWaitOnThread(threads[0], index, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult RunHubWindow(const OpParam &param, const AlgResourceCtx &res,
    const std::vector<ThreadHandle> &threads, const TransferArgs &x, const HubWindowGeometry &g)
{
    const auto rootCopy = [&](ThreadHandle lane, CcuKernelHandle kernel,
            bool scratchReceive, uint64_t srcOffset, uint64_t dstOffset, uint64_t length,
            bool ingress = false, uint64_t pitch = 0) -> HcclResult {
        return HubCopy(lane, kernel, x, x.input, x.inputToken,
            scratchReceive ? x.scratch : x.output, scratchReceive ? x.scratchToken : x.outputToken,
            x.stride, srcOffset, dstOffset, length, ingress, pitch);
    };
    if (param.myRank == param.root) {
        // Every local path group runs its own Load -> own-data sequence. All
        // groups overlap with root->B0 on the main lane. No lane reuses a peer.
        for (size_t i = 0; i < res.hubLocalLoad.size(); ++i) {
            CHK_RET(rootCopy(threads[i + 1U], res.hubLocalLoad[i], false,
                x.offset + g.prefix, 0, g.relay));
            CHK_RET(rootCopy(threads[i + 1U], res.hubLocalOwn[i], false,
                x.offset, x.offset, g.bytes));
        }
        // D8: root-local output copy is a normal local TS memcpy, not fused
        // into a Hub CCU kernel.  Queue it on a helper lane so it overlaps the
        // root->B0 traffic and never activates the HubRootCopy CCU_IF path.
        if (x.rootCopyEnabled != 0U) {
            const uint64_t localSrc = x.input + static_cast<uint64_t>(param.root) * x.stride + x.offset;
            const uint64_t localDst = x.output + x.offset;
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(threads[1],
                reinterpret_cast<void *>(localDst), reinterpret_cast<void *>(localSrc), g.bytes)));
        }
        for (uint32_t r = 0; r < g.rounds; ++r) {
            const uint64_t prefixOffset = r * g.firstChunk;
            const uint64_t chunk = r + 1U == g.rounds ? g.prefix - prefixOffset : g.firstChunk;
            const uint64_t slot = static_cast<uint64_t>(r % 2U) * SCATTER_HUB_PEERS * g.slotPitch;
            // Z9: one registered ingress kernel writes all seven remote-target
            // segments into the seven B0 scratch slots in one launch.
            CHK_RET(rootCopy(threads[0], res.hubIngress[0], false,
                x.offset + prefixOffset, slot, chunk, true, g.slotPitch));
        }
        // B0 own-data LAST: this communication overlaps the last prefix forwards.
        CHK_RET(rootCopy(threads[0], res.hubOwn[0], false, x.offset, x.offset, g.bytes));
        return HCCL_SUCCESS;
    }
    if (res.localRole == SCATTER_ROLE_HELPER) {
        CHK_RET(rootCopy(threads[0], res.hubLocalLoad[0], true, x.offset + g.prefix, 0, g.relay));
        CHK_RET(HcommThreadNotifyRecordOnThread(threads[0], threads[1], READY_NOTIFY_BASE));
        CHK_RET(HcommThreadNotifyWaitOnThread(threads[1], READY_NOTIFY_BASE, CUSTOM_TIMEOUT));
        CHK_RET(HubCopy(threads[1], res.hubTail[0], x, x.scratch, x.scratchToken,
            x.output, x.outputToken, 0, 0, x.offset + g.prefix, g.relay));
        // Own-data receive can overlap Forward; scratch is not reused until Join.
        CHK_RET(rootCopy(threads[0], res.hubLocalOwn[0], false, x.offset, x.offset, g.bytes));
        return HCCL_SUCCESS;
    }
    if (res.localRole == SCATTER_ROLE_HUB) {
        for (uint32_t r = 0; r < g.rounds; ++r) {
            const uint32_t parity = r % 2U;
            const uint64_t prefixOffset = r * g.firstChunk;
            const uint64_t chunk = r + 1U == g.rounds ? g.prefix - prefixOffset : g.firstChunk;
            const uint64_t slot = static_cast<uint64_t>(parity) * SCATTER_HUB_PEERS * g.slotPitch;
            if (r >= 2U) {
                for (size_t i = 0; i < res.hubPrefix.size(); ++i) {
                    // One free acknowledgement per physical forward group and slot.
                    CHK_RET(HcommThreadNotifyWaitOnThread(threads[0],
                        static_cast<uint32_t>(i) * 2U + parity, CUSTOM_TIMEOUT));
                }
            }
            // Z9: one matching receive-side ingress launch synchronizes the
            // whole seven-target packed transfer for this round.
            CHK_RET(rootCopy(threads[0], res.hubIngress[0], true,
                x.offset + prefixOffset, slot, chunk, true, g.slotPitch));
            for (size_t i = 0; i < res.hubPrefix.size(); ++i) {
                const ThreadHandle lane = threads[1U + i];
                CHK_RET(HcommThreadNotifyRecordOnThread(threads[0], lane, 1U + parity));
                CHK_RET(HcommThreadNotifyWaitOnThread(lane, 1U + parity, CUSTOM_TIMEOUT));
                CHK_RET(HubCopy(lane, res.hubPrefix[i], x, x.scratch, x.scratchToken,
                    x.output, x.outputToken, g.slotPitch, slot, x.offset + prefixOffset, chunk));
                if (r + 2U < g.rounds) {
                    CHK_RET(HcommThreadNotifyRecordOnThread(lane, threads[0],
                        static_cast<uint32_t>(i) * 2U + parity));
                }
            }
        }
        CHK_RET(rootCopy(threads[0], res.hubOwn[0], false, x.offset, x.offset, g.bytes));
        return HCCL_SUCCESS;
    }
    // Bi: two independent peers, disjoint prefix/tail output intervals.
    CHK_RET(rootCopy(threads[1], res.hubTail[0], false, 0, x.offset + g.prefix, g.relay));
    for (uint32_t r = 0; r < g.rounds; ++r) {
        const uint64_t prefixOffset = r * g.firstChunk;
        const uint64_t chunk = r + 1U == g.rounds ? g.prefix - prefixOffset : g.firstChunk;
        CHK_RET(rootCopy(threads[0], res.hubPrefix[0], false, 0, x.offset + prefixOffset, chunk));
    }
    return HCCL_SUCCESS;
}

} // namespace

bool BuildHubWindowGeometry(uint64_t bytes, HubWindowGeometry &g)
{
    g = HubWindowGeometry{};
    if (bytes < 8192ULL || bytes > SCATTER_HUB_WINDOW_BYTES || bytes % 4ULL != 0ULL) { return false; }
    g.bytes = bytes;
    g.relay = ((bytes * 4ULL / 11ULL) / 512ULL) * 512ULL;
    g.prefix = bytes - g.relay;
    if (g.relay == 0ULL || g.prefix == 0ULL) { return false; }

    // D6 correctness-first Real Hub geometry.
    // D5 jumped directly to the tuned two-round schedule.  That makes the B0
    // ping-pong scratch very large (about 181 MiB for a 32 MiB slice), so a
    // perfectly valid Hub plan can fail before the first data transfer when the
    // communicator HCCL buffer is smaller than that tuned scratch demand.
    //
    // Keep the real 4/11 + 7/11 Hub data plane, but execute at least four
    // ingress/forward rounds.  Four rounds reduce B0 scratch by roughly 2.5x
    // while preserving exactly the same bytes, peers, channels and Hub roles.
    // Larger windows may use more than four rounds only to stay inside the
    // fixed global scratch budget.  There is no Direct fallback for a Hub
    // window: if this geometry cannot be built, the target call fails loudly.
    const uint64_t slotCapacity =
        (SCATTER_HUB_SCRATCH_BUDGET / (2ULL * SCATTER_HUB_PEERS) / 512ULL) * 512ULL;
    if (slotCapacity == 0ULL) { return false; }

    const uint64_t requiredByCapacity = 1ULL + (g.prefix - 1ULL) / slotCapacity;
    const uint32_t firstRoundCount = static_cast<uint32_t>(
        std::max<uint64_t>(SCATTER_HUB_ROUNDS, requiredByCapacity));

    for (uint32_t rounds = firstRoundCount; rounds <= 64U; ++rounds) {
        const uint64_t mean = (g.prefix + rounds - 1ULL) / rounds;
        const uint64_t regular = ((mean + 511ULL) / 512ULL) * 512ULL;
        if (regular == 0ULL || regular > slotCapacity) { continue; }
        if (static_cast<uint64_t>(rounds - 1U) >
            std::numeric_limits<uint64_t>::max() / regular) { continue; }
        const uint64_t beforeLast = static_cast<uint64_t>(rounds - 1U) * regular;
        if (beforeLast >= g.prefix) { continue; }
        const uint64_t last = g.prefix - beforeLast;
        if (last == 0ULL || last > regular || last > slotCapacity) { continue; }

        g.rounds = rounds;
        g.firstChunk = regular;
        g.slotPitch = regular;
        g.scratchBytes = 2ULL * SCATTER_HUB_PEERS * g.slotPitch;
        return g.scratchBytes <= SCATTER_HUB_SCRATCH_BUDGET;
    }
    return false;
}

namespace {
HcclResult BuildHubExecutionPlan(const OpParam &param, const AlgResourceCtx &res, ScatterExecutionPlan &p)
{
    CHK_PRT_RET(param.rankSize != 16U || res.hubRank >= 16U || res.hubRank == param.root ||
            res.hubHelpers.size() != 7U || res.hubTargets.size() != 7U,
        HCCL_ERROR("malformed global hub plan"), HCCL_E_INTERNAL);
    std::vector<uint32_t> ranks{param.root, res.hubRank};
    ranks.insert(ranks.end(), res.hubHelpers.begin(), res.hubHelpers.end());
    ranks.insert(ranks.end(), res.hubTargets.begin(), res.hubTargets.end());
    std::sort(ranks.begin(), ranks.end());
    for (uint32_t i = 0; i < 16U; ++i) {
        CHK_PRT_RET(ranks[i] != i, HCCL_ERROR("hub roles are not a rank partition"), HCCL_E_INTERNAL);
    }
    const uint32_t role = param.myRank == param.root ? SCATTER_ROLE_ROOT :
        param.myRank == res.hubRank ? SCATTER_ROLE_HUB :
        std::find(res.hubHelpers.begin(), res.hubHelpers.end(), param.myRank) != res.hubHelpers.end() ?
            SCATTER_ROLE_HELPER : SCATTER_ROLE_TARGET;
    CHK_PRT_RET(res.localRole != role, HCCL_ERROR("local hub role mismatch"), HCCL_E_INTERNAL);
    p.useHub = true;
    p.windowBytes = std::min(p.sliceBytes, SCATTER_HUB_WINDOW_BYTES);
    p.windowCount = 1ULL + (p.sliceBytes - 1ULL) / p.windowBytes;
    HubWindowGeometry g;
    CHK_PRT_RET(!BuildHubWindowGeometry(p.windowBytes, g), HCCL_ERROR("invalid hub window"), HCCL_E_INTERNAL);
    if (role == SCATTER_ROLE_ROOT) {
        CHK_PRT_RET(res.hubLocalLoad.empty() || res.hubLocalLoad.size() > 7U ||
                res.hubLocalOwn.size() != res.hubLocalLoad.size() ||
                res.hubIngress.size() != 1U || res.hubOwn.size() != 1U,
            HCCL_ERROR("incomplete Z9 root hub resources"), HCCL_E_INTERNAL);
        p.threadCount = static_cast<uint32_t>(1U + res.hubLocalLoad.size());
    } else if (role == SCATTER_ROLE_HELPER) {
        CHK_PRT_RET(res.hubLocalLoad.size() != 1U || res.hubLocalOwn.size() != 1U || res.hubTail.size() != 1U,
            HCCL_ERROR("incomplete helper hub resources"), HCCL_E_INTERNAL);
        p.threadCount = 2U; p.scratchBytes = g.relay;
    } else if (role == SCATTER_ROLE_HUB) {
        CHK_PRT_RET(res.hubIngress.size() != 1U || res.hubOwn.size() != 1U ||
                res.hubPrefix.empty() || res.hubPrefix.size() > 7U,
            HCCL_ERROR("incomplete B0 hub resources"), HCCL_E_INTERNAL);
        p.threadCount = static_cast<uint32_t>(1U + res.hubPrefix.size());
        // Reserve the exact maximum two-slot geometry across this call's
        // windows.  The optimized two-round schedule intentionally uses a
        // larger first slot than the old four-round schedule.
        uint64_t maxScratch = 0ULL;
        uint64_t done = 0ULL;
        while (done < p.sliceBytes) {
            const uint64_t bytes = std::min(p.windowBytes, p.sliceBytes - done);
            HubWindowGeometry windowGeometry;
            CHK_PRT_RET(!BuildHubWindowGeometry(bytes, windowGeometry),
                HCCL_ERROR("invalid residual hub window"), HCCL_E_INTERNAL);
            maxScratch = std::max(maxScratch, windowGeometry.scratchBytes);
            done += bytes;
        }
        p.scratchBytes = maxScratch;
    } else {
        CHK_PRT_RET(res.hubPrefix.size() != 1U || res.hubTail.size() != 1U,
            HCCL_ERROR("incomplete Bi hub resources"), HCCL_E_INTERNAL);
        p.threadCount = 2U;
    }
    CHK_PRT_RET(p.threadCount > SCATTER_HUB_MAX_THREADS ||
            (p.scratchBytes != 0 && (res.localBuffer.addr == nullptr || res.localBuffer.size < p.scratchBytes)),
        HCCL_ERROR("hub buffer too small: rank[%u] need[%llu] have[%llu]", param.myRank,
            static_cast<unsigned long long>(p.scratchBytes), static_cast<unsigned long long>(res.localBuffer.size)),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}
} // namespace

HcclResult BuildExecutionPlan(const OpParam &param, const AlgResourceCtx &resCtx, ScatterExecutionPlan &plan)
{
    plan = ScatterExecutionPlan{};
    const auto type = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.root >= param.rankSize ||
            param.myRank >= param.rankSize || type == SIZE_TABLE.end() || type->second == 0 ||
            param.count > std::numeric_limits<uint64_t>::max() / type->second,
        HCCL_ERROR("invalid Scatter plan parameters"), HCCL_E_PARA);
    plan.sliceBytes = param.count * type->second;
    CHK_PRT_RET(plan.sliceBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("total input byte count overflow"), HCCL_E_PARA);
    plan.totalBytes = plan.sliceBytes * param.rankSize;
    if (plan.sliceBytes == 0 || param.rankSize == 1U) { return HCCL_SUCCESS; }
    CHK_PRT_RET(resCtx.schemaVersion != 25U ||
            resCtx.directKernelChannelCounts.size() != resCtx.directKernels.size() ||
            resCtx.directKernelDieIds.size() != resCtx.directKernels.size() ||
            (resCtx.hubEnabled == 0U && resCtx.relayCapable == 0U && resCtx.directKernels.empty()),
        HCCL_ERROR("invalid V30.6 resource context"), HCCL_E_INTERNAL);
    plan.windowBytes = std::min(plan.sliceBytes, SCATTER_WINDOW_BYTES);
    plan.windowCount = 1ULL + (plan.sliceBytes - 1ULL) / plan.windowBytes;
    // A successfully registered Hub context is the globally agreed 2x8
    // capacity-balanced protocol.  Legacy16 remains the safe small-message and
    // topology-rejection fallback only.
    plan.legacy16 = HCCL_SCATTER_LEGACY_2X8_SAFE != 0 && param.rankSize == 16U &&
        resCtx.hubEnabled == 0U;
    plan.batchDirect = !plan.legacy16 && resCtx.directBatchMode != 0U;
    if (plan.legacy16) {
        CHK_PRT_RET(resCtx.directBatchMode != 0U || resCtx.hubEnabled != 0U || resCtx.relayCapable != 0U,
            HCCL_ERROR("legacy 2x8 context contains optimized-path resources"), HCCL_E_INTERNAL);
        if (param.myRank == param.root && plan.totalBytes <= SCATTER_SMALL_FAST_THRESHOLD &&
            resCtx.directKernels.size() > 1U) {
            // Keep the exact V23 Legacy16 Channel/kernel protocol.  Only the
            // already-proven Begin/Join scheduler uses two independent path
            // groups for the 512 KiB class.
            plan.threadCount = static_cast<uint32_t>(std::min<size_t>(
                SCATTER_SMALL_FAST_THREADS, resCtx.directKernels.size()));
        } else if (param.myRank == param.root && plan.totalBytes >= SCATTER_PARALLEL_THRESHOLD) {
            plan.threadCount = static_cast<uint32_t>(std::min<size_t>(SCATTER_MAX_THREADS, resCtx.directKernels.size()));
        }
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(plan.batchDirect != (plan.sliceBytes > SCATTER_WINDOW_BYTES),
        HCCL_ERROR("stale Direct kernel ABI/cache class"), HCCL_E_INTERNAL);

    if (resCtx.hubEnabled != 0U && param.rankSize == 16U &&
        plan.totalBytes >= SCATTER_PARALLEL_THRESHOLD) {
        CHK_PRT_RET(!resCtx.directKernels.empty(),
            HCCL_ERROR("Z8 Hub context unexpectedly contains Direct kernels"), HCCL_E_INTERNAL);
        return BuildHubExecutionPlan(param, resCtx, plan);
    }
    uint64_t direct = 0;
    uint64_t relay = 0;
    const bool geometryOk = RelayGeometry(plan.windowBytes, direct, relay);
    plan.useRelay = param.rankSize == 12U && resCtx.relayCapable != 0U &&
        plan.totalBytes >= SCATTER_PARALLEL_THRESHOLD && geometryOk;
    if (!plan.useRelay) {
        plan.smallFastDirect = param.rankSize == 12U && resCtx.directSmallFast != 0U &&
            plan.totalBytes <= SCATTER_SMALL_FAST_THRESHOLD;
        if (param.myRank == param.root && plan.smallFastDirect && resCtx.directKernels.size() > 1U) {
            plan.threadCount = static_cast<uint32_t>(std::min<size_t>(
                SCATTER_SMALL_FAST_THREADS, resCtx.directKernels.size()));
        } else if (param.myRank == param.root && plan.totalBytes >= SCATTER_PARALLEL_THRESHOLD &&
            (param.rankSize == 4U || param.rankSize == 12U || param.rankSize == 16U)) {
            plan.threadCount = static_cast<uint32_t>(std::min<size_t>(SCATTER_MAX_THREADS, resCtx.directKernels.size()));
        }
        return HCCL_SUCCESS;
    }

    // Never silently switch just one rank back to Direct if Relay resources are
    // incomplete. Such a local fallback creates unmatched cross-rank protocols.
    CHK_PRT_RET(resCtx.relaySplitCapable == 0U,
        HCCL_ERROR("incomplete globally selected Relay plan, rank[%u] hash[%llu]", param.myRank,
            static_cast<unsigned long long>(resCtx.relayPlanHash)), HCCL_E_INTERNAL);
    plan.scratchBytes = relay;
    if (param.myRank == param.root) {
        CHK_PRT_RET(resCtx.relayDirectKernels.empty() || resCtx.relayHelperFinalKernels.empty() ||
                resCtx.relayLoadKernels.empty() ||
                resCtx.relayDirectKernelWeightUnits.size() != resCtx.relayDirectKernels.size(),
            HCCL_ERROR("incomplete root Relay resources"), HCCL_E_INTERNAL);
        // P4 root-only expansion: each path-grouped RelayDirect kernel plus each
        // helper may own one TS lane.  Ordinary Direct paths remain capped by
        // SCATTER_MAX_THREADS, so 4x1 / 8+4 regression anchors are untouched.
        plan.threadCount = static_cast<uint32_t>(std::min<size_t>(SCATTER_RELAY_ROOT_MAX_THREADS,
            resCtx.relayDirectKernels.size() + resCtx.relayHelperFinalKernels.size()));
    } else if (resCtx.localRole == SCATTER_ROLE_HELPER) {
        CHK_PRT_RET(resCtx.relayLoadKernels.empty() ||
                resCtx.relayForwardKernels.size() != resCtx.relayLoadKernels.size() ||
                resCtx.relayHelperFinalKernels.size() != 1U || resCtx.localBuffer.addr == nullptr ||
                resCtx.localBuffer.size < relay,
            HCCL_ERROR("helper resources/scratch too small: rank[%u] need[%llu] have[%llu]", param.myRank,
                static_cast<unsigned long long>(relay), static_cast<unsigned long long>(resCtx.localBuffer.size)),
            HCCL_E_PARA);
        plan.threadCount = static_cast<uint32_t>(1U + resCtx.relayForwardKernels.size());
    } else if (resCtx.localRole == SCATTER_ROLE_TARGET) {
        CHK_PRT_RET(resCtx.relayDirectKernels.size() != 1U || resCtx.relayForwardKernels.size() != 1U,
            HCCL_ERROR("incomplete target Relay resources"), HCCL_E_INTERNAL);
        plan.threadCount = 2U;
    } else {
        CHK_PRT_RET(resCtx.relayDirectKernels.size() != 1U || !resCtx.relayForwardKernels.empty(),
            HCCL_ERROR("incomplete ordinary-rank Relay resources"), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<ThreadHandle> &parallelThreads, const ScatterExecutionPlan &plan)
{
    if (plan.sliceBytes == 0) { return HCCL_SUCCESS; }
    if (param.rankSize == 1U) {
        if (param.inputPtr == param.outputPtr) { return HCCL_SUCCESS; }
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(param.cpuThread,
            param.outputPtr, param.inputPtr, plan.sliceBytes)));
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET((plan.threadCount > 1U && (parallelThreads.size() != plan.threadCount ||
                    parallelThreads[0] != param.cpuThread)) ||
            (plan.threadCount == 1U && !parallelThreads.empty()),
        HCCL_ERROR("thread schedule differs from selected execution plan"), HCCL_E_INTERNAL);
    TransferArgs x;
    x.stride = plan.sliceBytes;
    x.output = reinterpret_cast<uint64_t>(param.outputPtr);
    CHK_RET(MemToken(x.output, plan.sliceBytes, x.outputToken));
    x.input = x.output;
    x.inputToken = x.outputToken;
    if (param.myRank == param.root) {
        x.input = reinterpret_cast<uint64_t>(param.inputPtr);
        CHK_RET(MemToken(x.input, plan.totalBytes, x.inputToken));
    }
    if (plan.legacy16) {
        return RunLegacy16Direct(param, resCtx, parallelThreads, plan, x);
    }

    if ((plan.useRelay && resCtx.localRole == SCATTER_ROLE_HELPER) ||
        (plan.useHub && plan.scratchBytes != 0)) {
        x.scratch = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
        CHK_RET(MemToken(x.scratch, plan.scratchBytes, x.scratchToken));
        CHK_PRT_RET(plan.useHub && x.scratch < x.output + plan.sliceBytes &&
                x.output < x.scratch + plan.scratchBytes,
            HCCL_ERROR("hub scratch aliases user output"), HCCL_E_PARA);
    }
#if HCCL_SCATTER_DIAGNOSTICS
    std::fprintf(stderr,
        "[SCATTER_V23_EXEC] rank=%u root=%u n=%u count=%llu slice=%llu total=%llu "
        "role=%u mode=%s windows=%llu window=%llu scratch=%llu threads=%u directK=%zu "
        "loadK=%zu fwdK=%zu batch=%u smallFast=%u planHash=%llu hub=%u hlocal=%zu hingress=%zu hprefix=%zu htail=%zu\n",
        param.myRank, param.root, param.rankSize, static_cast<unsigned long long>(param.count),
        static_cast<unsigned long long>(plan.sliceBytes), static_cast<unsigned long long>(plan.totalBytes),
        resCtx.localRole, plan.useHub ? "HUB_2X8_Z9_COMPACT" : plan.useRelay ? "WINDOW_RELAY_5_12" : "DIRECT",
        static_cast<unsigned long long>(plan.windowCount), static_cast<unsigned long long>(plan.windowBytes),
        static_cast<unsigned long long>(plan.scratchBytes), plan.threadCount, resCtx.directKernels.size(),
        resCtx.relayLoadKernels.size(), resCtx.relayForwardKernels.size(), plan.batchDirect ? 1U : 0U,
        plan.smallFastDirect ? 1U : 0U, static_cast<unsigned long long>(resCtx.relayPlanHash), resCtx.hubRank,
        resCtx.hubLocalLoad.size(), resCtx.hubIngress.size(), resCtx.hubPrefix.size(), resCtx.hubTail.size());
#endif
    // Root self-copy is fused into exactly one active kernel PER WINDOW:
    // Direct group 0, RelayDirect group 0, or HubOwn. Network traffic no longer
    // waits for a separate whole-slice TS memcpy before it can begin.
    if (param.myRank == param.root) {
        const uint64_t local = x.input + static_cast<uint64_t>(param.root) * plan.sliceBytes;
        x.rootCopyEnabled = x.output != local ? 1U : 0U;
        const bool overlaps = x.output < x.input + plan.totalBytes && x.input < x.output + plan.sliceBytes;
        CHK_PRT_RET(overlaps && x.output != local,
            HCCL_ERROR("root output overlaps input other than its exact own slice"), HCCL_E_PARA);
    }

    if (plan.useHub) {
        while (x.offset < x.stride) {
            x.bytes = std::min(plan.windowBytes, x.stride - x.offset);
            HubWindowGeometry geometry;
            const bool hubWindow = BuildHubWindowGeometry(x.bytes, geometry);
            if (hubWindow && resCtx.localRole == SCATTER_ROLE_HUB) {
                CHK_PRT_RET(geometry.scratchBytes > plan.scratchBytes,
                    HCCL_ERROR("hub window exceeds reserved scratch"), HCCL_E_INTERNAL);
            }
            CHK_PRT_RET(!hubWindow,
                HCCL_ERROR("D8 target Hub window has no valid 4-round geometry: rank[%u] offset[%llu] bytes[%llu]",
                    param.myRank, static_cast<unsigned long long>(x.offset),
                    static_cast<unsigned long long>(x.bytes)), HCCL_E_INTERNAL);
            CHK_RET(Begin(parallelThreads));
            CHK_RET(RunHubWindow(param, resCtx, parallelThreads, x, geometry));
            CHK_RET(HubJoin(parallelThreads));
            x.offset += x.bytes;
        }
        return HCCL_SUCCESS;
    }

    if (!plan.useRelay) {
        // All multi-TS Direct schedules must enter the worker start protocol before launches.
        // Channel PreSync synchronizes peer endpoints, but does not replace TS-thread Begin/Join ordering.
        CHK_RET(Begin(parallelThreads));
        while (x.offset < x.stride) {
            x.bytes = std::min(SCATTER_WINDOW_BYTES, x.stride - x.offset);
            const uint64_t rest = x.stride - x.offset - x.bytes;
            const uint64_t next = plan.batchDirect ? std::min(SCATTER_WINDOW_BYTES, rest) : 0;
            CHK_RET(RunDirect(param, resCtx, parallelThreads, plan, x, next));
            x.offset += x.bytes + next;
        }
        CHK_RET(Join(parallelThreads));
        return HCCL_SUCCESS;
    }

    while (x.offset < x.stride) {
        x.bytes = std::min(plan.windowBytes, x.stride - x.offset);
        const bool relayWindow = RelayGeometry(x.bytes, x.direct, x.relay);
        // A1: useRelay is selected only for the validated 4x3 large-message
        // path in this P1 baseline.  Every non-tiny window must therefore
        // prove the exact aligned 5/12 Relay split before execution.  If this
        // gate trips on 11/12, P1's unchanged 2.06/1.61 ms was caused by a
        // window-level fallback/geometry defect rather than by pipeline cost.
        CHK_RET(Audit4x3RelayWindow(plan, x, relayWindow));
        CHK_RET(Begin(parallelThreads));
        if (!relayWindow) {
            // Only a truly tiny residual that cannot form one aligned Relay
            // block is allowed to use Direct.  Major-window fallback is
            // rejected by Audit4x3RelayWindow above.
            CHK_RET(RunRelayResidual(param, resCtx, parallelThreads, x));
        } else if (param.myRank == param.root) {
            CHK_RET(RunRelayRoot(resCtx, parallelThreads, x));
        } else {
            CHK_RET(RunRelayPeer(param, resCtx, parallelThreads, x));
        }
        // Explicit window fence: no scratch overwrite, reused ready/start notify,
        // or channel XN address overwrite may race the previous window's Forward.
        CHK_RET(RelayJoin(parallelThreads));
        x.offset += x.bytes;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
