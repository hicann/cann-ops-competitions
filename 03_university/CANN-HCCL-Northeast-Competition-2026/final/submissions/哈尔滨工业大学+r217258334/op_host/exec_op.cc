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
#include <cstdint>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

constexpr uint64_t LARGE_MESSAGE_THRESHOLD = 1ULL * 1024 * 1024;
constexpr uint32_t NETWORK_SYNC_NOTIFY_IDX = 0;
constexpr uint32_t COPY_SYNC_NOTIFY_IDX = 0;
constexpr uint32_t MAX_DIE_KERNEL_NUM = 4;
constexpr uint64_t RELAY_PHASE_ROOT_FANOUT = 1;
constexpr uint64_t RELAY_PHASE_FORWARD = 2;
constexpr uint64_t RELAY_PHASE_FINAL_SYNC = 3;

HcclResult PreSyncNetworkThreads(const AlgResourceCtx &resCtx, uint32_t kernelNum)
{
    for (uint32_t i = 1; i < kernelNum; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[0], resCtx.threads[i], NETWORK_SYNC_NOTIFY_IDX)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.threads[i], NETWORK_SYNC_NOTIFY_IDX, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult PostSyncNetworkThreads(const AlgResourceCtx &resCtx, uint32_t kernelNum)
{
    for (uint32_t i = 1; i < kernelNum; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[i], resCtx.threads[0], i - 1)));
    }
    for (uint32_t i = 1; i < kernelNum; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.threads[0], i - 1, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

uint32_t AllDirectKernelMask(uint32_t kernelNum)
{
    return kernelNum >= 32 ? 0xFFFFFFFFU : ((1U << kernelNum) - 1U);
}

uint32_t DirectKernelMaskForPeer(const AlgResourceCtx &resCtx, uint32_t peerRank)
{
    if (peerRank >= 32 || resCtx.directKernelPeerMasks.size() != resCtx.ccuKernels.size()) {
        return 0;
    }

    const uint32_t peerBit = 1U << peerRank;
    uint32_t activeMask = 0;
    for (uint32_t i = 0; i < resCtx.directKernelPeerMasks.size(); ++i) {
        if ((resCtx.directKernelPeerMasks[i] & peerBit) != 0) {
            activeMask |= (1U << i);
        }
    }
    return activeMask;
}

HcclResult PreSyncDirectThreads(
    const AlgResourceCtx &resCtx, uint32_t kernelNum, uint32_t activeKernelMask)
{
    for (uint32_t i = 1; i < kernelNum; ++i) {
        if ((activeKernelMask & (1U << i)) == 0) {
            continue;
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[0], resCtx.threads[i], NETWORK_SYNC_NOTIFY_IDX)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.threads[i], NETWORK_SYNC_NOTIFY_IDX, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult PostSyncDirectThreads(
    const AlgResourceCtx &resCtx, uint32_t kernelNum, uint32_t activeKernelMask)
{
    for (uint32_t i = 1; i < kernelNum; ++i) {
        if ((activeKernelMask & (1U << i)) == 0) {
            continue;
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[i], resCtx.threads[0], i - 1)));
    }
    for (uint32_t i = 1; i < kernelNum; ++i) {
        if ((activeKernelMask & (1U << i)) == 0) {
            continue;
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.threads[0], i - 1, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchDieKernels(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputAddr,
    uint64_t inputToken, uint64_t outputAddr, uint64_t outputToken, uint64_t recvBytes,
    uint64_t chunkOffset, uint64_t chunkSize)
{
    const uint32_t kernelNum = static_cast<uint32_t>(resCtx.ccuKernels.size());
    CHK_PRT_RET(kernelNum == 0 || kernelNum > MAX_DIE_KERNEL_NUM || resCtx.threads.size() < kernelNum,
        HCCL_ERROR("[%s] invalid kernel/thread resources, kernels[%u], threads[%zu]",
            __func__, kernelNum, resCtx.threads.size()),
        HCCL_E_INTERNAL);

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        inputToken,
        outputAddr,
        outputToken,
        recvBytes,
        chunkOffset,
        chunkSize,
        static_cast<uint64_t>(param.root),
    };

    CHK_PRT_RET(resCtx.directKernelPeerMasks.size() != resCtx.ccuKernels.size(),
        HCCL_ERROR("[%s] invalid direct kernel peer metadata, masks[%zu], kernels[%zu]",
            __func__, resCtx.directKernelPeerMasks.size(), resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);

    const uint32_t activeKernelMask = (param.myRank == param.root)
        ? AllDirectKernelMask(kernelNum)
        : DirectKernelMaskForPeer(resCtx, param.root);
    CHK_PRT_RET(activeKernelMask == 0 ||
        (activeKernelMask & ~AllDirectKernelMask(kernelNum)) != 0,
        HCCL_ERROR("[%s] invalid direct active kernel mask[0x%x], myRank[%u], root[%u]",
            __func__, activeKernelMask, param.myRank, param.root),
        HCCL_E_INTERNAL);

    CHK_RET(PreSyncDirectThreads(resCtx, kernelNum, activeKernelMask));
    for (uint32_t i = 0; i < kernelNum; ++i) {
        if ((activeKernelMask & (1U << i)) == 0) {
            continue;
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(
            resCtx.threads[i], resCtx.ccuKernels[i], taskArgs.data(), taskArgs.size()));
    }
    CHK_RET(PostSyncDirectThreads(resCtx, kernelNum, activeKernelMask));
    return HCCL_SUCCESS;
}

HcclResult LaunchRelayDieKernels(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputAddr,
    uint64_t inputToken, uint64_t outputAddr, uint64_t outputToken, uint64_t relayBufferAddr,
    uint64_t relayBufferToken, uint64_t recvBytes, uint64_t chunkOffset, uint64_t chunkSize,
    uint64_t directBytes, uint64_t relayBytes, uint32_t relayTargetRank, uint32_t relaySourceRank, uint64_t phase)
{
    const uint32_t kernelNum = static_cast<uint32_t>(resCtx.ccuKernels.size());
    CHK_PRT_RET(kernelNum == 0 || kernelNum > MAX_DIE_KERNEL_NUM || resCtx.threads.size() < kernelNum,
        HCCL_ERROR("[%s] invalid kernel/thread resources, kernels[%u], threads[%zu]",
            __func__, kernelNum, resCtx.threads.size()),
        HCCL_E_INTERNAL);

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        inputToken,
        outputAddr,
        outputToken,
        relayBufferAddr,
        relayBufferToken,
        recvBytes,
        chunkOffset,
        chunkSize,
        directBytes,
        relayBytes,
        static_cast<uint64_t>(param.root),
        static_cast<uint64_t>(relayTargetRank),
        static_cast<uint64_t>(relaySourceRank),
        phase,
    };

    CHK_RET(PreSyncNetworkThreads(resCtx, kernelNum));
    for (uint32_t i = 0; i < kernelNum; ++i) {
        CHK_RET_CCU(HcommCcuKernelLaunch(
            resCtx.threads[i], resCtx.ccuKernels[i], taskArgs.data(), taskArgs.size()));
    }
    CHK_RET(PostSyncNetworkThreads(resCtx, kernelNum));
    return HCCL_SUCCESS;
}

HcclResult GetRelayChunkLimit(const AlgResourceCtx &resCtx, uint64_t dataTypeSize, uint64_t &chunkLimit)
{
    CHK_PRT_RET(resCtx.relayNumerator == 0 || resCtx.relayDenominator <= resCtx.relayNumerator ||
        resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0 || dataTypeSize == 0,
        HCCL_ERROR("[%s] invalid relay metadata or HCCL buffer", __func__), HCCL_E_INTERNAL);

    chunkLimit = MAX_DATA_SIZE;
    const uint64_t relayBytesAtMax =
        static_cast<uint64_t>(MAX_DATA_SIZE) * resCtx.relayNumerator / resCtx.relayDenominator;
    if (resCtx.localBuffer.size < relayBytesAtMax) {
        // This multiplication is bounded because localBuffer.size is already
        // smaller than relayBytesAtMax (<= MAX_DATA_SIZE).
        chunkLimit = resCtx.localBuffer.size * resCtx.relayDenominator / resCtx.relayNumerator;
        chunkLimit -= chunkLimit % dataTypeSize;
    }
    const uint64_t minRelayChunk =
        static_cast<uint64_t>(resCtx.relayDenominator) * dataTypeSize;
    CHK_PRT_RET(chunkLimit < minRelayChunk,
        HCCL_ERROR("[%s] relay chunk limit[%lu] is smaller than minimum[%lu]",
            __func__, chunkLimit, minRelayChunk),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult StartOverlappedLocalCopy(const AlgResourceCtx &resCtx, uint32_t kernelNum,
    uint64_t localDstAddr, uint64_t localSrcAddr, uint64_t chunkSize)
{
    CHK_PRT_RET(resCtx.threads.size() <= kernelNum,
        HCCL_ERROR("[%s] copy thread is missing, kernels[%u], threads[%zu]",
            __func__, kernelNum, resCtx.threads.size()),
        HCCL_E_INTERNAL);

    const ThreadHandle copyThread = resCtx.threads[kernelNum];
    const ThreadHandle mainThread = resCtx.threads[0];

    // Ensure the copy thread observes all work ordered before this collective on
    // the user/main stream. The completion record is queued after the local copy
    // so the main thread can join it after launching the network kernels.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        mainThread, copyThread, COPY_SYNC_NOTIFY_IDX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        copyThread, COPY_SYNC_NOTIFY_IDX, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(copyThread,
        reinterpret_cast<void *>(localDstAddr), reinterpret_cast<void *>(localSrcAddr), chunkSize)));
    const uint32_t copyDoneMainNotifyIdx = kernelNum - 1;
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        copyThread, mainThread, copyDoneMainNotifyIdx)));
    return HCCL_SUCCESS;
}

HcclResult WaitOverlappedLocalCopy(const AlgResourceCtx &resCtx, uint32_t kernelNum)
{
    const uint32_t copyDoneMainNotifyIdx = kernelNum - 1;
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[0], copyDoneMainNotifyIdx, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("[%s] unsupported dataType[%d]", __func__, param.dataType),
        HCCL_E_NOT_SUPPORT);

    const uint64_t dataTypeSize = sizeIt->second;
    const uint64_t recvBytes = param.count * dataTypeSize;
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("[%s] root[%u] is out of rankSize[%u]", __func__, param.root, param.rankSize), HCCL_E_PARA);

    const uint32_t kernelNum = static_cast<uint32_t>(resCtx.ccuKernels.size());
    CHK_PRT_RET(kernelNum == 0 || kernelNum > MAX_DIE_KERNEL_NUM,
        HCCL_ERROR("[%s] unsupported ccu kernel number[%u]", __func__, kernelNum), HCCL_E_INTERNAL);

    const bool largeCopyThreadExpected = (recvBytes > LARGE_MESSAGE_THRESHOLD);
    const size_t expectedThreadNum =
        static_cast<size_t>(kernelNum + (largeCopyThreadExpected ? 1U : 0U));
    CHK_PRT_RET(resCtx.threads.size() != expectedThreadNum,
        HCCL_ERROR("[%s] thread number[%zu] does not match expected[%zu]",
            __func__, resCtx.threads.size(), expectedThreadNum),
        HCCL_E_INTERNAL);

    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);

    uint64_t inputToken = 0;
    if (param.myRank == param.root) {
        const uint64_t totalInputBytes = recvBytes * static_cast<uint64_t>(param.rankSize);
        CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, totalInputBytes, &inputToken));
    }

    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(outputAddr, recvBytes, &outputToken));

    const bool useTopologyRelay = resCtx.relayNumerator != 0;
    uint64_t relayBufferAddr = 0;
    uint64_t relayBufferToken = 0;
    uint64_t chunkLimit = MAX_DATA_SIZE;
    uint32_t relayTargetRank = INVALID_VALUE_RANKID;
    uint32_t relaySourceRank = INVALID_VALUE_RANKID;
    if (useTopologyRelay) {
        CHK_PRT_RET(resCtx.relayDenominator <= resCtx.relayNumerator ||
            resCtx.relayServerIds.size() != param.rankSize,
            HCCL_ERROR("[%s] invalid relay context, ratio[%u/%u], serverIds[%zu]",
                __func__, resCtx.relayNumerator, resCtx.relayDenominator, resCtx.relayServerIds.size()),
            HCCL_E_INTERNAL);

        std::vector<uint32_t> relayTargetByRank;
        std::vector<uint32_t> relaySourceByRank;
        const uint32_t pairNum = BuildScatterRelayPairs(
            resCtx.relayServerIds, param.root, relayTargetByRank, relaySourceByRank);
        uint32_t expectedPairNum = 0;
        const uint32_t rootServerId = resCtx.relayServerIds[param.root];
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.root && resCtx.relayServerIds[rank] == rootServerId) {
                ++expectedPairNum;
            }
        }
        CHK_PRT_RET(pairNum != expectedPairNum || relayTargetByRank.size() != param.rankSize ||
            relaySourceByRank.size() != param.rankSize,
            HCCL_ERROR("[%s] invalid runtime relay pairing, pairNum[%u]", __func__, pairNum),
            HCCL_E_INTERNAL);
        relayTargetRank = relayTargetByRank[param.myRank];
        relaySourceRank = relaySourceByRank[param.myRank];

        relayBufferAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
        CHK_RET_CCU(HcommCcuGetMemToken(relayBufferAddr, resCtx.localBuffer.size, &relayBufferToken));
        CHK_RET(GetRelayChunkLimit(resCtx, dataTypeSize, chunkLimit));
    }

    // Freeze the 512KB path. For every topology, only messages >1MB enable
    // the dedicated copy thread; root overlaps its local self-copy with the
    // die-parallel network kernels while non-root ranks simply leave it idle.
    const bool useLargeCopyOverlap =
        (recvBytes > LARGE_MESSAGE_THRESHOLD && param.myRank == param.root);

    uint64_t processedBytes = 0;
    while (processedBytes < recvBytes) {
        const uint64_t remainingBytes = recvBytes - processedBytes;
        uint64_t chunkSize = std::min<uint64_t>(chunkLimit, remainingBytes);
        if (useTopologyRelay && remainingBytes > chunkLimit) {
            // Keep a tiny final remainder from producing a zero-length direct
            // or relay sub-transfer. recvBytes/chunkLimit are datatype-aligned,
            // so this adjustment preserves alignment as well.
            const uint64_t minRelayChunk =
                static_cast<uint64_t>(resCtx.relayDenominator) * dataTypeSize;
            const uint64_t tailBytes = remainingBytes - chunkSize;
            if (tailBytes < minRelayChunk) {
                chunkSize -= minRelayChunk - tailBytes;
            }
        }

        bool copyStarted = false;
        if (param.myRank == param.root) {
            const uint64_t localSrcAddr =
                inputAddr + static_cast<uint64_t>(param.root) * recvBytes + processedBytes;
            const uint64_t localDstAddr = outputAddr + processedBytes;

            if (useLargeCopyOverlap) {
                CHK_RET(StartOverlappedLocalCopy(
                    resCtx, kernelNum, localDstAddr, localSrcAddr, chunkSize));
                copyStarted = true;
            } else {
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0],
                    reinterpret_cast<void *>(localDstAddr), reinterpret_cast<void *>(localSrcAddr), chunkSize)));
            }
        }

        if (useTopologyRelay) {
            uint64_t relayBytes = chunkSize * resCtx.relayNumerator / resCtx.relayDenominator;
            relayBytes -= relayBytes % dataTypeSize;
            const uint64_t directBytes = chunkSize - relayBytes;

            // Phase 1 starts root Clos direct traffic and local Mesh staging.
            // Each local relay is released as soon as its staged suffix lands;
            // the per-rank die barrier then lets phase 2 forward that suffix on
            // either IO die while root is still finishing phase 1. Phase 3 is a
            // final root<->all handshake after the relay-forward die barrier, so
            // the relay HCCL buffer cannot be overwritten by the next chunk.
            // Root self-copy remains overlapped across the whole sequence.
            CHK_RET(LaunchRelayDieKernels(param, resCtx, inputAddr, inputToken, outputAddr, outputToken,
                relayBufferAddr, relayBufferToken, recvBytes, processedBytes, chunkSize, directBytes, relayBytes,
                relayTargetRank, relaySourceRank, RELAY_PHASE_ROOT_FANOUT));
            CHK_RET(LaunchRelayDieKernels(param, resCtx, inputAddr, inputToken, outputAddr, outputToken,
                relayBufferAddr, relayBufferToken, recvBytes, processedBytes, chunkSize, directBytes, relayBytes,
                relayTargetRank, relaySourceRank, RELAY_PHASE_FORWARD));
            CHK_RET(LaunchRelayDieKernels(param, resCtx, inputAddr, inputToken, outputAddr, outputToken,
                relayBufferAddr, relayBufferToken, recvBytes, processedBytes, chunkSize, directBytes, relayBytes,
                relayTargetRank, relaySourceRank, RELAY_PHASE_FINAL_SYNC));
        } else {
            // V1.5/V1.6-B die-parallel direct path remains byte-for-byte in
            // behavior for 4x1, 8+4 and all small-message contexts.
            CHK_RET(LaunchDieKernels(param, resCtx, inputAddr, inputToken, outputAddr, outputToken,
                recvBytes, processedBytes, chunkSize));
        }

        if (copyStarted) {
            CHK_RET(WaitOverlappedLocalCopy(resCtx, kernelNum));
        }

        processedBytes += chunkSize;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
