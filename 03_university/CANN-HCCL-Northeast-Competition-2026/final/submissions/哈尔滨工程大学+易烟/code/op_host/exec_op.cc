/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_res.h>
#if __has_include(<ccu/ccu_launch.h>)
#include <ccu/ccu_launch.h>
#endif
#include <algorithm>
#include <array>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
bool Overlaps(uint64_t a, uint64_t aBytes, uint64_t b, uint64_t bBytes)
{
    return a < b ? b - a < aBytes : a - b < bBytes;
}

bool CanOverlapRootCopy(const OpParam &param, uint64_t bytes)
{
    // A destination inside another rank's source slice is unsafe as well.
    // ScatterImpl already validated both ranges and the N * bytes product.
    return param.myRank == param.root &&
        !Overlaps(reinterpret_cast<uintptr_t>(param.inputPtr), bytes * param.rankSize,
            reinterpret_cast<uintptr_t>(param.outputPtr), bytes);
}

bool OverlapLargeCopy(const OpParam &param, uint64_t bytes)
{
    const bool enabled = param.rankSize == 4 ? V13_LARGE_COPY_FOUR_BY_ONE : V13_LARGE_COPY_OTHERS;
    return enabled && ClassifyMessage(param.count) == MessageClass::LARGE && CanOverlapRootCopy(param, bytes);
}

uint64_t FrameCapacity(uint64_t count)
{
    // Balance the bounded frames so a 256 MiB + 4 B slice does not create a
    // separate four-byte protocol round. All ranks use the same count only.
    const uint64_t maxElements = MAX_DATA_SIZE / ELEMENT_BYTES;
    const uint64_t frames = (count - 1) / maxElements + 1;
    return (count / frames + (count % frames != 0 ? 1 : 0)) * ELEMENT_BYTES;
}

SupplyPlan BuildSupplyPlan(const OpParam &param, const AlgResourceCtx &ctx,
    uint64_t bytes, uint64_t frameOffset, uint64_t frameBytes)
{
    SupplyPlan plan;
    const uint32_t mask = ctx.topology.localMemberMask;
    if (param.myRank == param.root || (mask & RankBit(param.root)) == 0 ||
        ctx.localBuffer.size <= PULL_PAYLOAD_OFFSET || ctx.peerLinks[param.root].hop != 1) {
        return plan;
    }
    plan.layout = RootPullLayout(ctx.topology);
    if (plan.layout == PullLayout::DIRECT) {
        return plan;
    }
    plan.rootMask = mask;
    uint32_t helperOrdinal = 0;
    uint32_t remoteCount = 0;
    uint32_t remoteRanks[MAX_RANK_SIZE]{};
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if ((mask & RankBit(rank)) == 0) {
            remoteRanks[remoteCount++] = rank;
        } else if (rank != param.root && rank < param.myRank) {
            ++helperOrdinal;
        }
    }
    const uint64_t capacity = (ctx.localBuffer.size - PULL_PAYLOAD_OFFSET) / ELEMENT_BYTES * ELEMENT_BYTES;
    const uint64_t elements = frameBytes / ELEMENT_BYTES;
    if (plan.layout == PullLayout::TWO_BY_EIGHT_PREFIXES) {
        // Seven root-local helpers each supply one distinct remote target.
        // The eighth remote target reads its complete frame from root.
        if (remoteCount != 8 || helperOrdinal >= 7) {
            return SupplyPlan{};
        }
        const uint32_t target = remoteRanks[helperOrdinal];
        const uint64_t nominal = ((elements / 11) * 4 + (elements % 11) * 4 / 11) * ELEMENT_BYTES;
        const uint64_t length = std::min(nominal, capacity);
        auto &piece = plan.pieces[target];
        piece.bytes = length;
        piece.negativeEnd = uint64_t{0} - length;
        piece.bufferOffset = PULL_PAYLOAD_OFFSET;
        piece.inputOffset = target * bytes + frameOffset;
        return plan;
    }
    if (plan.layout == PullLayout::FOUR_BY_THREE_PREFIXES) {
        // Two root-local helpers each supply one distinct remote target.
        // The other seven remote targets read complete frames from root.
        if (remoteCount != 9 || helperOrdinal >= 2) {
            return SupplyPlan{};
        }
        const uint32_t target = remoteRanks[helperOrdinal];
        const uint64_t nominal = ((elements / 6) * 5 + (elements % 6) * 5 / 6) * ELEMENT_BYTES;
        const uint64_t length = std::min(nominal, capacity);
        auto &piece = plan.pieces[target];
        piece.bytes = length;
        piece.negativeEnd = uint64_t{0} - length;
        piece.bufferOffset = PULL_PAYLOAD_OFFSET;
        piece.inputOffset = target * bytes + frameOffset;
        return plan;
    }

    if (plan.layout == PullLayout::EIGHT_PLUS_FOUR_PREFIXES) {
        // Root on the four-rank side: three helpers, three distinct targets.
        if (remoteCount != 8 || helperOrdinal >= 3) {
            return SupplyPlan{};
        }
        const uint32_t target = remoteRanks[helperOrdinal];
        const uint64_t nominal = ((elements / 7) * 4 + (elements % 7) * 4 / 7) * ELEMENT_BYTES;
        const uint64_t length = std::min(nominal, capacity);
        auto &piece = plan.pieces[target];
        piece.bytes = length;
        piece.negativeEnd = uint64_t{0} - length;
        piece.bufferOffset = PULL_PAYLOAD_OFFSET;
        piece.inputOffset = target * bytes + frameOffset;
        return plan;
    }

    // Retain the V12 layout for the archived pre-prefix comparison stages.
    constexpr uint64_t denominator = 7;
    if (plan.layout != PullLayout::EIGHT_PLUS_FOUR_BANDS || remoteCount != 8 || helperOrdinal >= 3) {
        return SupplyPlan{};
    }
    const uint64_t nominal = ((elements / denominator) * 4 +
        (elements % denominator) * 4 / denominator) * ELEMENT_BYTES;
    const uint64_t start = helperOrdinal * nominal;
    const uint64_t end = start + std::min(nominal, capacity);
    for (uint32_t j = 0; j < remoteCount; ++j) {
        const uint64_t targetStart = j * frameBytes;
        const uint64_t first = std::max(start, targetStart);
        const uint64_t last = std::min(end, targetStart + frameBytes);
        if (last <= first) {
            continue;
        }
        const uint32_t target = remoteRanks[j];
        auto &piece = plan.pieces[target];
        piece.offset = first - targetStart;
        piece.bytes = last - first;
        piece.negativeEnd = uint64_t{0} - (piece.offset + piece.bytes);
        piece.bufferOffset = PULL_PAYLOAD_OFFSET + first - start;
        // A band crossing a virtual target boundary becomes two real Reads.
        piece.inputOffset = target * bytes + frameOffset + piece.offset;
    }
    return plan;
}

HcclResult LaunchGroups(const OpParam &param, const AlgResourceCtx &ctx,
    const CcuKernelHandle *kernels, const uint64_t *args, uint32_t argCount)
{
    if (ctx.groupCount == 2) {
        CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, ctx.worker, DIRECT_THREAD_START));
        CHK_RET(HcommThreadNotifyWaitOnThread(ctx.worker, DIRECT_THREAD_START, CUSTOM_TIMEOUT));
        CHK_RET_CCU(HcommCcuKernelLaunch(ctx.worker, kernels[1], args, argCount));
        CHK_RET(HcommThreadNotifyRecordOnThread(ctx.worker, param.cpuThread, DIRECT_THREAD_DONE));
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, kernels[0], args, argCount));
        CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, DIRECT_THREAD_DONE, CUSTOM_TIMEOUT));
    } else if (ctx.groupCount == 1) {
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, kernels[0], args, argCount));
    }
    return HCCL_SUCCESS;
}

struct HelperOverlapPlan {
    bool enabled = false;
    uint32_t target = INVALID_VALUE_RANKID;
    uint32_t rootGroup = INVALID_VALUE_RANKID;
    uint32_t targetGroup = INVALID_VALUE_RANKID;
};

HcclResult BuildHelperOverlapPlan(const OpParam &param, const AlgResourceCtx &ctx, const SupplyPlan &supply,
    uint64_t bytes, uint64_t frameOffset, uint64_t frameBytes, HelperOverlapPlan &plan)
{
    const uint32_t mask = ctx.topology.localMemberMask;
    if (param.rankSize != 16 || param.myRank == param.root ||
        ClassifyMessage(param.count) != MessageClass::LARGE || ctx.topology.kind != TopologyKind::TWO_BY_EIGHT ||
        RankMaskSize(mask) != 8 || (mask & ~AllRankMask(param.rankSize)) != 0 ||
        (mask & RankBit(param.myRank)) == 0 || (mask & RankBit(param.root)) == 0 ||
        supply.layout != PullLayout::TWO_BY_EIGHT_PREFIXES || supply.rootMask != mask) {
        return HCCL_SUCCESS;
    }
    uint32_t nonzeroPieces = 0;
    uint32_t target = INVALID_VALUE_RANKID;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (supply.pieces[peer].bytes != 0) {
            ++nonzeroPieces;
            target = peer;
        }
    }
    if (nonzeroPieces != 1 || ctx.groupCount != DIRECT_MAX_DIE_GROUPS) {
        return HCCL_SUCCESS;
    }
    const auto &piece = supply.pieces[target];
    CHK_PRT_RET(target == param.myRank || target == param.root || (mask & RankBit(target)) != 0 ||
        piece.offset != 0 || piece.negativeEnd != uint64_t{0} - piece.bytes ||
        piece.bufferOffset != PULL_PAYLOAD_OFFSET || piece.bytes > frameBytes ||
        ctx.localBuffer.size <= PULL_PAYLOAD_OFFSET ||
        piece.bytes > ctx.localBuffer.size - PULL_PAYLOAD_OFFSET ||
        piece.inputOffset != target * bytes + frameOffset,
        HCCL_ERROR("Invalid V20 helper prefix for READY overlap"), HCCL_E_INTERNAL);
    const auto &rootLink = ctx.peerLinks[param.root];
    const auto &targetLink = ctx.peerLinks[target];
    CHK_PRT_RET(rootLink.valid != 1 || targetLink.valid != 1 || rootLink.channel == 0 ||
        targetLink.channel == 0 || rootLink.hop != 1,
        HCCL_ERROR("Invalid V20 helper channels for READY overlap"), HCCL_E_INTERNAL);
    for (uint32_t g = 0; g < ctx.groupCount; ++g) {
        const auto &group = ctx.groups[g];
        for (uint32_t i = 0; i < group.kernelArg.peerCount; ++i) {
            const uint32_t peer = group.kernelArg.peers[i];
            if (peer != param.root && peer != target) {
                continue;
            }
            const auto &link = ctx.peerLinks[peer];
            CHK_PRT_RET(group.dieId != link.dieId || group.kernelArg.channels[i] != link.channel,
                HCCL_ERROR("V20 helper group/channel die mismatch"), HCCL_E_INTERNAL);
            uint32_t &index = peer == param.root ? plan.rootGroup : plan.targetGroup;
            CHK_PRT_RET(index != INVALID_VALUE_RANKID,
                HCCL_ERROR("Duplicate V20 helper channel group"), HCCL_E_INTERNAL);
            index = g;
        }
    }
    CHK_PRT_RET(plan.rootGroup == INVALID_VALUE_RANKID || plan.targetGroup == INVALID_VALUE_RANKID,
        HCCL_ERROR("Missing V20 helper channel group"), HCCL_E_INTERNAL);
    if (plan.rootGroup == plan.targetGroup) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(rootLink.dieId == targetLink.dieId || ctx.workerReady != 1 || ctx.worker == 0 ||
        ctx.worker == param.cpuThread || ctx.pullGroups[param.root][plan.rootGroup] == 0 ||
        ctx.pullGroups[param.root][plan.targetGroup] == 0 || ctx.offerReadies[target] == 0,
        HCCL_ERROR("Invalid V20 cross-die helper resources"), HCCL_E_INTERNAL);
    plan.target = target;
    plan.enabled = true;
    return HCCL_SUCCESS;
}

HcclResult LaunchHelperGroupsWithReady(const OpParam &param, const AlgResourceCtx &ctx,
    const HelperOverlapPlan &plan, const uint64_t *args)
{
    // Prefetch and all outbound META precede START on main. READY references
    // only the target die, so it cannot reuse the concurrent root die's XNs.
    // After any enqueue failure, propagate the error without a second launch.
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, ctx.worker, DIRECT_THREAD_START));
    CHK_RET(HcommThreadNotifyWaitOnThread(ctx.worker, DIRECT_THREAD_START, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(ctx.worker, ctx.pullGroups[param.root][plan.rootGroup], args, GT_ARG_COUNT));
    CHK_RET(HcommThreadNotifyRecordOnThread(ctx.worker, param.cpuThread, DIRECT_THREAD_DONE));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.offerReadies[plan.target], nullptr, 0));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.pullGroups[param.root][plan.targetGroup],
        args, GT_ARG_COUNT));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, DIRECT_THREAD_DONE, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult BuildHelperPreparationArgs(const OpParam &param, const AlgResourceCtx &ctx,
    const HelperPreparationPlan &preparation, const SupplyPlan &supply, uint64_t bytes,
    uint64_t frameOffset, uint64_t frameBytes, uint64_t bufferToken,
    std::array<uint64_t, HP_ARG_COUNT> &args)
{
    CHK_PRT_RET(ctx.groupCount != DIRECT_MAX_DIE_GROUPS || preparation.target >= param.rankSize ||
        preparation.rootGroup >= ctx.groupCount || preparation.targetGroup >= ctx.groupCount ||
        preparation.rootGroup == preparation.targetGroup,
        HCCL_ERROR("Invalid V20 helper preparation groups"), HCCL_E_INTERNAL);
    CHK_PRT_RET(ctx.workerReady != 1 || ctx.worker == 0 || ctx.worker == param.cpuThread ||
        ctx.offerGroups[param.root][preparation.rootGroup] == 0 ||
        ctx.offerGroups[param.root][preparation.targetGroup] == 0,
        HCCL_ERROR("Invalid V20 helper preparation resources"), HCCL_E_INTERNAL);
    CHK_PRT_RET(supply.layout != PullLayout::TWO_BY_EIGHT_PREFIXES ||
        supply.rootMask != ctx.topology.localMemberMask || frameOffset > bytes ||
        frameBytes == 0 || frameBytes > MAX_DATA_SIZE || frameBytes > bytes - frameOffset ||
        frameOffset % ELEMENT_BYTES != 0 || frameBytes % ELEMENT_BYTES != 0,
        HCCL_ERROR("Invalid V20 helper preparation layout or frame"), HCCL_E_INTERNAL);
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        CHK_PRT_RET(peer != preparation.target && supply.pieces[peer].bytes != 0,
            HCCL_ERROR("V20 helper preparation requires its unique target"), HCCL_E_INTERNAL);
    }
    const auto &piece = supply.pieces[preparation.target];
    CHK_PRT_RET(piece.offset != 0 || piece.negativeEnd != uint64_t{0} - piece.bytes ||
        piece.bufferOffset != PULL_PAYLOAD_OFFSET || piece.bytes > frameBytes || piece.bytes > MAX_DATA_SIZE ||
        piece.bytes % ELEMENT_BYTES != 0 || ctx.localBuffer.size <= PULL_PAYLOAD_OFFSET ||
        piece.bytes > ctx.localBuffer.size - PULL_PAYLOAD_OFFSET ||
        piece.inputOffset != preparation.target * bytes + frameOffset,
        HCCL_ERROR("Invalid V20 helper preparation prefix"), HCCL_E_INTERNAL);
    args = {
        reinterpret_cast<uintptr_t>(ctx.localBuffer.addr) + piece.bufferOffset, bufferToken,
        uint64_t{0} - static_cast<uint64_t>(supply.layout), uint64_t{0} - supply.rootMask,
        preparation.target, piece.bytes, piece.negativeEnd, piece.inputOffset};
    return HCCL_SUCCESS;
}

HcclResult LaunchHelperPreparation(const OpParam &param, const AlgResourceCtx &ctx,
    const HelperPreparationPlan &preparation, const uint64_t *args)
{
    // The slot contract is HP8 even for R=0. Only the worker's root-containing
    // group consumes root META; both groups publish offers before this join.
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, ctx.worker, DIRECT_THREAD_START));
    CHK_RET(HcommThreadNotifyWaitOnThread(ctx.worker, DIRECT_THREAD_START, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(ctx.worker, ctx.offerGroups[param.root][preparation.rootGroup],
        args, HP_ARG_COUNT));
    CHK_RET(HcommThreadNotifyRecordOnThread(ctx.worker, param.cpuThread, DIRECT_THREAD_DONE));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.offerGroups[param.root][preparation.targetGroup],
        args, HP_ARG_COUNT));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, DIRECT_THREAD_DONE, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

struct ContinuousHelperFrame {
    bool enabled = false;
    HelperOverlapPlan overlap;
    std::array<uint64_t, HP_ARG_COUNT> preparationArgs{};
    std::array<uint64_t, GT_ARG_COUNT> pullArgs{};
};

HcclResult BuildContinuousHelperFrame(const OpParam &param, const AlgResourceCtx &ctx,
    const HelperPreparationPlan &preparation, const HelperOverlapPlan &overlap, const SupplyPlan &supply,
    uint64_t bytes, uint64_t frameOffset, uint64_t frameBytes, uint64_t outputToken, uint64_t bufferToken,
    ContinuousHelperFrame &frame)
{
    // This complete preflight is pure Host work. No partial continuous frame
    // may fall back to the old launch sequence after notifications are queued.
    CHK_RET(BuildHelperPreparationArgs(param, ctx, preparation, supply, bytes,
        frameOffset, frameBytes, bufferToken, frame.preparationArgs));
    frame.pullArgs = {reinterpret_cast<uintptr_t>(param.outputPtr) + frameOffset,
        outputToken, param.myRank * bytes + frameOffset, frameBytes};
    if (!overlap.enabled) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(preparation.target != overlap.target || preparation.rootGroup != overlap.rootGroup ||
        preparation.targetGroup != overlap.targetGroup || frame.preparationArgs[HP_BYTES] == 0 ||
        ctx.groups[preparation.rootGroup].dieId == ctx.groups[preparation.targetGroup].dieId ||
        ctx.groups[preparation.rootGroup].dieId != ctx.peerLinks[param.root].dieId ||
        ctx.groups[preparation.targetGroup].dieId != ctx.peerLinks[preparation.target].dieId,
        HCCL_ERROR("Invalid V20 combined helper preparation/Pull plan"), HCCL_E_INTERNAL);
    frame.overlap = overlap;
    frame.enabled = true;
    return HCCL_SUCCESS;
}

HcclResult LaunchContinuousHelperFrame(const OpParam &param, const AlgResourceCtx &ctx,
    const ContinuousHelperFrame &frame)
{
    const auto &plan = frame.overlap;
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, ctx.worker, DIRECT_THREAD_START));
    CHK_RET(HcommThreadNotifyWaitOnThread(ctx.worker, DIRECT_THREAD_START, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(ctx.worker, ctx.offerGroups[param.root][plan.rootGroup],
        frame.preparationArgs.data(), HP_ARG_COUNT));
    CHK_RET(HcommThreadNotifyRecordOnThread(ctx.worker, param.cpuThread, DIRECT_THREAD_DONE));
    // The own-frame Read follows prefetch on the same actual die immediately.
    // Its final DONE waits for START2_ACK, proving main consumed PREP_DONE and
    // preventing two records from merging in the single main notification slot.
    CHK_RET_CCU(HcommCcuKernelLaunch(ctx.worker, ctx.pullGroups[param.root][plan.rootGroup],
        frame.pullArgs.data(), GT_ARG_COUNT));
    CHK_RET(HcommThreadNotifyWaitOnThread(ctx.worker, DIRECT_THREAD_START, CUSTOM_TIMEOUT));
    CHK_RET(HcommThreadNotifyRecordOnThread(ctx.worker, param.cpuThread, DIRECT_THREAD_DONE));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.offerGroups[param.root][plan.targetGroup],
        frame.preparationArgs.data(), HP_ARG_COUNT));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, DIRECT_THREAD_DONE, CUSTOM_TIMEOUT));
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, ctx.worker, DIRECT_THREAD_START));
    // Authorization rejection still owes READY for the published nonzero offer.
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.offerReadies[plan.target], nullptr, 0));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.pullGroups[param.root][plan.targetGroup],
        frame.pullArgs.data(), GT_ARG_COUNT));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, DIRECT_THREAD_DONE, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult LaunchEightSideRootPhase(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes,
    uint64_t frameOffset, uint64_t frameBytes, uint64_t inputToken, uint64_t outputToken, bool earlyCopy)
{
    CHK_PRT_RET(ctx.groupCount != DIRECT_MAX_DIE_GROUPS || ctx.workerReady != 1 || ctx.worker == 0 ||
        ctx.worker == param.cpuThread || ctx.groups[0].publish == 0 || ctx.groups[1].publish == 0,
        HCCL_ERROR("Invalid V20 fused root phase resources"), HCCL_E_INTERNAL);
    uint64_t args[DIRECT_MAX_DIE_GROUPS][RPH_ARG_COUNT]{};
    for (uint32_t g = 0; g < ctx.groupCount; ++g) {
        args[g][RPH_INPUT] = reinterpret_cast<uintptr_t>(param.inputPtr);
        args[g][RPH_TOKEN] = inputToken;
        args[g][RPH_LAYOUT] = static_cast<uint64_t>(RootPullLayout(ctx.topology));
        args[g][RPH_MASK] = ctx.topology.localMemberMask;
        args[g][RPH_COPY_SOURCE] = reinterpret_cast<uintptr_t>(param.inputPtr) + param.root * bytes + frameOffset;
        args[g][RPH_OUTPUT] = reinterpret_cast<uintptr_t>(param.outputPtr) + frameOffset;
        args[g][RPH_OUTPUT_TOKEN] = outputToken;
        args[g][RPH_COPY_BYTES] = frameBytes;
        args[g][RPH_DO_COPY] = earlyCopy && g == 0 ? 1 : 0;
    }
    // Copy ownership follows group 0, not physical die 0. Both replacement
    // slots use RPH9, including alias/no-copy calls. RELEASE stays after join.
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, ctx.worker, DIRECT_THREAD_START));
    CHK_RET(HcommThreadNotifyWaitOnThread(ctx.worker, DIRECT_THREAD_START, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(ctx.worker, ctx.groups[1].publish, args[1], RPH_ARG_COUNT));
    CHK_RET(HcommThreadNotifyRecordOnThread(ctx.worker, param.cpuThread, DIRECT_THREAD_DONE));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.groups[0].publish, args[0], RPH_ARG_COUNT));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, DIRECT_THREAD_DONE, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult CopyRootOutput(const OpParam &param, const AlgResourceCtx &ctx,
    uint64_t bytes, uint64_t inputToken, uint64_t outputToken, uint64_t bufferToken)
{
    const uint64_t source = reinterpret_cast<uintptr_t>(param.inputPtr) + param.root * bytes;
    const uint64_t output = reinterpret_cast<uintptr_t>(param.outputPtr);
    if (source == output) {
        return HCCL_SUCCESS;
    }
    const bool overlap = Overlaps(source, bytes, output, bytes);
    const uint64_t chunk = overlap ? std::min<uint64_t>(MAX_DATA_SIZE, ctx.localBuffer.size) : MAX_DATA_SIZE;
    std::array<uint64_t, COPY_ARG_COUNT> args{};
    args[COPY_SOURCE_TOKEN] = inputToken;
    args[COPY_OUTPUT_TOKEN] = outputToken;
    args[COPY_BUFFER] = reinterpret_cast<uintptr_t>(ctx.localBuffer.addr);
    args[COPY_BUFFER_TOKEN] = bufferToken;
    args[COPY_STAGING] = overlap ? 1 : 0;
    for (uint64_t copied = 0; copied < bytes;) {
        const uint64_t length = std::min(chunk, bytes - copied);
        const uint64_t offset = overlap && output > source ? bytes - copied - length : copied;
        args[COPY_SOURCE] = source + offset;
        args[COPY_OUTPUT] = output + offset;
        args[COPY_BYTES] = length;
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.copyKernel, args.data(), COPY_ARG_COUNT));
        copied += length;
    }
    return HCCL_SUCCESS;
}

void OrderDirectGroups(const OpParam &param, const AlgResourceCtx &ctx, uint32_t *order)
{
    order[0] = 0;
    order[1] = 1;
    if (param.myRank != param.root || ctx.groupCount != 2 || ctx.topology.kind == TopologyKind::UNKNOWN) {
        return;
    }
    uint32_t remotePeers[DIRECT_MAX_DIE_GROUPS]{};
    for (uint32_t g = 0; g < ctx.groupCount; ++g) {
        const auto &group = ctx.groups[g].kernelArg;
        for (uint32_t i = 0; i < group.peerCount; ++i) {
            remotePeers[g] += (ctx.topology.localMemberMask & RankBit(group.peers[i])) == 0 ? 1U : 0U;
        }
    }
    // The worker is enqueued first. Preserve V12's Clos-first group ordering.
    if (remotePeers[0] > remotePeers[1]) {
        std::swap(order[0], order[1]);
    }
}

HcclResult LaunchSpecializedRoot(const OpParam &param, const AlgResourceCtx &ctx,
    const uint32_t *order, const uint64_t *args, bool copy)
{
    // Ownership follows the actual group index, including after reordering.
    // Every variant used here was registered in the same concurrent round.
    const uint32_t copyGroup = order[0];
    const auto launch = [&](ThreadHandle thread, uint32_t g) -> HcclResult {
        const bool copiesHere = copy && g == copyGroup;
        const CcuKernelHandle kernel = copiesHere ? ctx.groups[g].directRootCopy : ctx.groups[g].directRoot;
        const uint32_t count = copiesHere ? static_cast<uint32_t>(DRC_ARG_COUNT) :
            static_cast<uint32_t>(DR_ARG_COUNT);
        CHK_RET_CCU(HcommCcuKernelLaunch(thread, kernel, args, count));
        return HCCL_SUCCESS;
    };
    if (ctx.groupCount == 2) {
        CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, ctx.worker, DIRECT_THREAD_START));
        CHK_RET(HcommThreadNotifyWaitOnThread(ctx.worker, DIRECT_THREAD_START, CUSTOM_TIMEOUT));
        CHK_RET(launch(ctx.worker, order[1]));
        CHK_RET(HcommThreadNotifyRecordOnThread(ctx.worker, param.cpuThread, DIRECT_THREAD_DONE));
        CHK_RET(launch(param.cpuThread, order[0]));
        CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, DIRECT_THREAD_DONE, CUSTOM_TIMEOUT));
    } else {
        CHK_RET(launch(param.cpuThread, order[0]));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecDirectPull(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes,
    uint64_t inputToken, uint64_t outputToken, uint64_t bufferToken)
{
    // Small: one Read per target. 4x1: the same direct topology, with bounded
    // reads for large slices. There is no useful intra-server helper in 4x1.
    const uint64_t frameCapacity = FrameCapacity(param.count);
    const uint64_t input = reinterpret_cast<uintptr_t>(param.inputPtr);
    const uint64_t output = reinterpret_cast<uintptr_t>(param.outputPtr);
    const uint64_t ownSource = param.myRank == param.root ? input + param.root * bytes : 0;
    const bool overlap = param.myRank == param.root && ownSource != output &&
        Overlaps(ownSource, bytes, output, bytes);
    const bool fused = param.myRank == param.root && ctx.groupCount == 1 &&
        bytes <= MAX_DATA_SIZE && (!overlap || bytes <= ctx.localBuffer.size);
    const bool specialized = V13_SMALL_ROOT_SPECIALIZATION && UseSmallDirect(param) && param.myRank == param.root &&
        ctx.groupCount != 0 && (ownSource == output || CanOverlapRootCopy(param, bytes));
    uint32_t order[DIRECT_MAX_DIE_GROUPS];
    OrderDirectGroups(param, ctx, order);
    const CcuKernelHandle rootKernels[DIRECT_MAX_DIE_GROUPS] = {
        ctx.groups[order[0]].smallRoot, ctx.groups[order[1]].smallRoot};
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t length = std::min(frameCapacity, bytes - offset);
        if (param.myRank != param.root) {
            const uint64_t args[SP_ARG_COUNT] = {output + offset, outputToken, param.myRank * bytes + offset, length};
            CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.smallPeers[param.root], args, SP_ARG_COUNT));
        } else if (specialized) {
            // Both variants share input/token at positions 0 and 1; the no-copy
            // kernel receives exactly two arguments, not the six-field array.
            const uint64_t args[DRC_ARG_COUNT] = {
                input, inputToken, ownSource + offset, output + offset, outputToken, length};
            CHK_RET(LaunchSpecializedRoot(param, ctx, order, args, ownSource != output));
        } else {
            std::array<uint64_t, SR_ARG_COUNT> args{};
            args[SR_INPUT] = input;
            args[SR_INPUT_TOKEN] = inputToken;
            args[SR_COPY_SOURCE] = ownSource;
            args[SR_OUTPUT] = output;
            args[SR_OUTPUT_TOKEN] = outputToken;
            args[SR_BUFFER] = reinterpret_cast<uintptr_t>(ctx.localBuffer.addr);
            args[SR_BUFFER_TOKEN] = bufferToken;
            args[SR_COPY_BYTES] = bytes;
            args[SR_STAGING] = overlap ? 1 : 0;
            const bool copiesHere = fused && ownSource != output && offset + length == bytes;
            args[SR_DO_COPY] = copiesHere ? (OverlapLargeCopy(param, bytes) ? 2 : 1) : 0;
            CHK_RET(LaunchGroups(param, ctx, rootKernels, args.data(), SR_ARG_COUNT));
        }
        offset += length;
    }
    if (param.myRank == param.root && !specialized && !fused) {
        CHK_RET(CopyRootOutput(param, ctx, bytes, inputToken, outputToken, bufferToken));
    }
    return HCCL_SUCCESS;
}

HcclResult PrefetchSupply(const OpParam &param, const AlgResourceCtx &ctx,
    const SupplyPlan &supply, uint64_t bufferToken)
{
    const uint64_t buffer = reinterpret_cast<uintptr_t>(ctx.localBuffer.addr);
    const uint64_t negativeLayout = uint64_t{0} - static_cast<uint64_t>(supply.layout);
    const uint64_t negativeMask = uint64_t{0} - supply.rootMask;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        const auto &piece = supply.pieces[peer];
        if (piece.bytes == 0) {
            continue;
        }
        const uint64_t args[PF_ARG_COUNT] = {buffer, bufferToken, negativeLayout, negativeMask,
            piece.inputOffset, piece.bufferOffset, piece.bytes};
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.prefetches[param.root], args, PF_ARG_COUNT));
    }
    return HCCL_SUCCESS;
}

HcclResult PublishSupply(const OpParam &param, const AlgResourceCtx &ctx,
    const SupplyPlan &supply, uint64_t bufferToken, uint64_t &offeredBytes)
{
    HelperPreparationPlan preparation;
    CHK_PRT_RET(BuildHelperPreparationPlan(ctx, param.root, preparation),
        HCCL_ERROR("V20 HP8 preparation slots cannot publish with BG7"), HCCL_E_INTERNAL);
    const uint64_t buffer = reinterpret_cast<uintptr_t>(ctx.localBuffer.addr);
    const uint64_t negativeLayout = uint64_t{0} - static_cast<uint64_t>(supply.layout);
    const uint64_t negativeMask = uint64_t{0} - supply.rootMask;
    bool batched = V13_BATCH_OFFERS && supply.layout != PullLayout::EIGHT_PLUS_FOUR_BANDS;
    uint32_t target = INVALID_VALUE_RANKID;
    offeredBytes = 0;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank || peer == param.root) {
            continue;
        }
        const auto &piece = supply.pieces[peer];
        offeredBytes += piece.bytes;
        if (piece.bytes != 0) {
            if (target != INVALID_VALUE_RANKID || piece.offset != 0 ||
                piece.bufferOffset != PULL_PAYLOAD_OFFSET || piece.negativeEnd != uint64_t{0} - piece.bytes) {
                batched = false;
            }
            target = peer;
        }
    }
    if (batched) {
        const uint64_t length = target == INVALID_VALUE_RANKID ? 0 : supply.pieces[target].bytes;
        const uint64_t args[BG_ARG_COUNT] = {buffer + PULL_PAYLOAD_OFFSET, bufferToken,
            negativeLayout, negativeMask, target, length, uint64_t{0} - length};
        for (uint32_t g = 0; g < ctx.groupCount; ++g) {
            const auto &group = ctx.groups[g].kernelArg;
            if (group.peerCount == 1 && group.peers[0] == param.root) {
                continue;
            }
            // Groups are serial, and every original nonroot pair still sends
            // seven fields plus 0x7f before any PullGroup may wait for META.
            CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.offerGroups[param.root][g], args, BG_ARG_COUNT));
        }
        return HCCL_SUCCESS;
    }
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank || peer == param.root) {
            continue;
        }
        const auto &piece = supply.pieces[peer];
        const uint64_t args[OF_ARG_COUNT] = {piece.offset, piece.bytes, piece.negativeEnd,
            buffer + piece.bufferOffset, bufferToken, negativeLayout, negativeMask};
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.offerSends[peer], args, OF_ARG_COUNT));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecTopologyPull(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes,
    uint64_t inputToken, uint64_t outputToken, uint64_t bufferToken)
{
    const uint64_t frameCapacity = FrameCapacity(param.count);
    const PullLayout rootLayout = RootPullLayout(ctx.topology);
    const bool earlyCopy = OverlapLargeCopy(param, bytes);
    const bool fusedRoot8 = RootOnEightSide(ctx.topology, param.rankSize, param.root);
    const bool fusedRootPhase = UseEightSideRootPhase(ctx.topology, ctx.rankSize, ctx.myRank, ctx.groupCount);
    HelperPreparationPlan preparation;
    const bool fusedPreparation = BuildHelperPreparationPlan(ctx, param.root, preparation);
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t length = std::min(frameCapacity, bytes - offset);
        if (param.myRank == param.root) {
            if (fusedRootPhase) {
                CHK_RET(LaunchEightSideRootPhase(param, ctx, bytes, offset, length,
                    inputToken, outputToken, earlyCopy));
            } else {
                const uint64_t args[RP_ARG_COUNT] = {reinterpret_cast<uintptr_t>(param.inputPtr), inputToken,
                    static_cast<uint64_t>(rootLayout), ctx.topology.localMemberMask};
                // Original RP4 groups publish to BOTH dies before waiting.
                for (uint32_t g = 0; g < ctx.groupCount; ++g) {
                    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.groups[g].publish, args, RP_ARG_COUNT));
                }
                if (earlyCopy) {
                    // Copy only this bounded frame, overlapping peer Reads.
                    const uint64_t copyArgs[COPY_ARG_COUNT] = {
                        reinterpret_cast<uintptr_t>(param.inputPtr) + param.root * bytes + offset, inputToken,
                        reinterpret_cast<uintptr_t>(param.outputPtr) + offset, outputToken,
                        reinterpret_cast<uintptr_t>(ctx.localBuffer.addr), bufferToken, length, 0};
                    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.copyKernel, copyArgs, COPY_ARG_COUNT));
                }
                for (uint32_t g = 0; g < ctx.groupCount; ++g) {
                    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.groups[g].rootWait, nullptr, 0));
                }
            }
            // Release only after BOTH dies have collected every ROOT_DONE.
            // This also protects the final frame when the next call changes root.
            for (uint32_t g = 0; g < ctx.groupCount; ++g) {
                CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.groups[g].rootRelease, nullptr, 0));
            }
        } else {
            PullLayout supplyLayout = PullLayout::DIRECT;
            uint64_t offeredBytes = 0;
            const uint64_t task[GT_ARG_COUNT] = {reinterpret_cast<uintptr_t>(param.outputPtr) + offset, outputToken,
                param.myRank * bytes + offset, length};
            bool groupsLaunched = false;
            bool continuousHelper = false;
            // Under the same predicate used at registration, BuildSupplyPlan
            // always returns DIRECT, rootMask=0 and zero pieces. The fused
            // kernels consume root META and publish zero BYTES with full META.
            if (!fusedRoot8) {
                const SupplyPlan supply = BuildSupplyPlan(param, ctx, bytes, offset, length);
                supplyLayout = supply.layout;
                const bool early = UsesEarlyOffer(supply.layout);
                HelperOverlapPlan overlapPlan;
                if (early) {
                    CHK_RET(BuildHelperOverlapPlan(param, ctx, supply, bytes, offset, length, overlapPlan));
                }
                if (fusedPreparation) {
                    // HP8 owns capture, complete offer publication and prefetch.
                    // Its static slot selection remains valid when R is zero.
                    ContinuousHelperFrame frame;
                    CHK_RET(BuildContinuousHelperFrame(param, ctx, preparation, overlapPlan, supply,
                        bytes, offset, length, outputToken, bufferToken, frame));
                    offeredBytes = frame.preparationArgs[HP_BYTES];
                    if (frame.enabled) {
                        CHK_RET(LaunchContinuousHelperFrame(param, ctx, frame));
                        groupsLaunched = true;
                        continuousHelper = true;
                    } else {
                        CHK_RET(LaunchHelperPreparation(param, ctx, preparation, frame.preparationArgs.data()));
                    }
                } else {
                    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.captures[param.root], nullptr, 0));
                    // Archived bands retain post-prefetch META without READY.
                    if (!early) {
                        CHK_RET(PrefetchSupply(param, ctx, supply, bufferToken));
                    }
                    CHK_RET(PublishSupply(param, ctx, supply, bufferToken, offeredBytes));
                }
                if (early && !groupsLaunched) {
                    if (!fusedPreparation) {
                        CHK_RET(PrefetchSupply(param, ctx, supply, bufferToken));
                    }
                    if (overlapPlan.enabled) {
                        CHK_RET(LaunchHelperGroupsWithReady(param, ctx, overlapPlan, task));
                        groupsLaunched = true;
                    } else {
                        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
                            if (peer == param.myRank || peer == param.root || supply.pieces[peer].bytes == 0) {
                                continue;
                            }
                            // Even rejected prefetch authorization owes READY
                            // for the original nonzero descriptor.
                            CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.offerReadies[peer], nullptr, 0));
                        }
                    }
                }
            }
            // Each group captures its own META and reads only local-die XNs.
            // Join both groups before ROOT_DONE; the ACK kernel then waits for
            // root RELEASE before this rank can reuse scratch or channel XNs.
            if (!groupsLaunched) {
                CHK_RET(LaunchGroups(param, ctx, ctx.pullGroups[param.root], task, GT_ARG_COUNT));
            }
            CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.rootAcks[param.root], nullptr, 0));
            HCCL_DEBUG("[Scatter v20 supply] rank=%u root=%u kind=%u layout=%llu "
                "frame=%llu bytes=%llu offered=%llu fused=%u readyOverlap=%u prepared=%u continuous=%u",
                param.myRank, param.root, static_cast<uint32_t>(ctx.topology.kind),
                static_cast<unsigned long long>(supplyLayout), static_cast<unsigned long long>(offset),
                static_cast<unsigned long long>(length), static_cast<unsigned long long>(offeredBytes),
                fusedRoot8 ? 1U : 0U, groupsLaunched ? 1U : 0U, fusedPreparation ? 1U : 0U,
                continuousHelper ? 1U : 0U);
        }
        offset += length;
    }
    if (param.myRank == param.root && !earlyCopy) {
        // All destinations and helpers have stopped reading root input.
        CHK_RET(CopyRootOutput(param, ctx, bytes, inputToken, outputToken, bufferToken));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    AlgResourceCtx ctx;
    CHK_PRT_RET(!ctx.DeSerialize(param.resCtx, param.ctxSize) || ctx.myRank != param.myRank ||
        ctx.rankSize != param.rankSize, HCCL_ERROR("Invalid V20 Pull context"), HCCL_E_INTERNAL);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    const uint64_t bytes = param.count * ELEMENT_BYTES;
    const bool root = param.myRank == param.root;
    const bool direct = UseSmallDirect(param) || param.rankSize == 4 || param.rankSize == 1;
    const bool parallel = ctx.groupCount == 2 && (root || !direct);
    CHK_PRT_RET(parallel && (ctx.workerReady != 1 || ctx.worker == 0 || ctx.worker == param.cpuThread),
        HCCL_ERROR("V20 requires a separate worker for both actual dies"), HCCL_E_INTERNAL);
    const uint64_t scratch = reinterpret_cast<uintptr_t>(ctx.localBuffer.addr);
    const uint64_t input = reinterpret_cast<uintptr_t>(param.inputPtr);
    const uint64_t output = reinterpret_cast<uintptr_t>(param.outputPtr);
    const uint64_t ownSource = root ? input + param.root * bytes : 0;
    const bool copyOverlap = root && ownSource != output && Overlaps(ownSource, bytes, output, bytes);
    // HCCL-VM uses encoded device VAs; their low bits do not describe the
    // allocation's physical alignment. Keep the allocator's original base.
    // No HBM control page is reserved: nonroot scratch holds helper payload only.
    const bool useBuffer = (!direct && !root) || copyOverlap;
    CHK_PRT_RET(useBuffer && (Overlaps(scratch, ctx.localBuffer.size, output, bytes) ||
        (root && Overlaps(scratch, ctx.localBuffer.size, input, bytes * param.rankSize))),
        HCCL_ERROR("V20 scratch must not alias accessed user buffers"), HCCL_E_PARA);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t bufferToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(output, bytes, &outputToken));
    if (root) {
        CHK_RET_CCU(HcommCcuGetMemToken(input, bytes * param.rankSize, &inputToken));
    }
    if (useBuffer) {
        CHK_RET_CCU(HcommCcuGetMemToken(scratch, ctx.localBuffer.size, &bufferToken));
    }
    HCCL_DEBUG("[Scatter v20 plan] rank=%u root=%u N=%u bytes=%llu "
        "topology=%u layout=%llu direct=%u groups=%u",
        param.myRank, param.root, param.rankSize, static_cast<unsigned long long>(bytes),
        static_cast<uint32_t>(ctx.topology.kind), static_cast<unsigned long long>(RootPullLayout(ctx.topology)),
        direct ? 1U : 0U, ctx.groupCount);
    if (direct) {
        return ExecDirectPull(param, ctx, bytes, inputToken, outputToken, bufferToken);
    }
    return ExecTopologyPull(param, ctx, bytes, inputToken, outputToken, bufferToken);
}
} // namespace ops_hccl
