/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root path of the software repository for the full text of the License.
 */

#include <vector>
#include <algorithm>
#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {

static constexpr uint64_t MIN_SLICE_ALIGN = 128;

// Compute floor(log2(n))
static uint32_t Log2(uint32_t n)
{
    uint32_t r = 0;
    while ((1u << r) < n) r++;
    return r;
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AllReduce on AICPU, rank=%u, size=%u, count=%lu, recHalving=%d, hier=%d, cclBuf=%lu",
              resCtx.myRank, resCtx.rankSize, (unsigned long)param.count,
              resCtx.useRecursiveHalving, resCtx.useHierarchical,
              (unsigned long)resCtx.localBuffer.size);

    uint32_t n = resCtx.rankSize;
    uint32_t myRank = resCtx.myRank;

    uint32_t elemSize = GetDataTypeSize(param.dataType);
    if (elemSize == 0) {
        HCCL_ERROR("Unsupported data type: %u", static_cast<uint32_t>(param.dataType));
        return HCCL_E_PARA;
    }
    uint64_t totalBytes = param.count * elemSize;

    if (n <= 1) {
        if (totalBytes > 0) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, totalBytes)));
        }
        return HCCL_SUCCESS;
    }

    ThreadHandle thread = resCtx.threads[0];
    void *localCclBuf = resCtx.localBuffer.addr;
    uint64_t cclBufferSize = resCtx.localBuffer.size;

    HcommDataType hcommDataType = static_cast<HcommDataType>(param.dataType);
    HcommReduceOp hcommReduceOp = static_cast<HcommReduceOp>(param.reduceType);

    uint64_t slotSize = cclBufferSize;
    slotSize = slotSize / MIN_SLICE_ALIGN * MIN_SLICE_ALIGN;
    if (slotSize < elemSize) slotSize = elemSize;

    HCCL_INFO("slotSize=%lu, cclBufferSize=%lu", (unsigned long)slotSize, (unsigned long)cclBufferSize);

    if (resCtx.useRecursiveHalving) {
        // Channel_mapping: [0]=XOR^1, [1]=XOR^2, [2]=XOR^4, [3]=cross-server
        ChannelHandle meshCh[3];
        void *meshRemoteBuf[3];
        uint32_t nServer = resCtx.serverRankSize;
        uint32_t myLocalRank = myRank % nServer;
        uint32_t log2n = Log2(nServer);

        for (uint32_t i = 0; i < 3 && i < log2n; i++) {
            meshCh[i] = resCtx.channels[i].handle;
            meshRemoteBuf[i] = resCtx.channels[i].remoteCclMem.addr;
        }
        ChannelHandle crossCh = resCtx.channels[resCtx.crossServerChIdx].handle;
        void *crossRemoteBuf = resCtx.channels[resCtx.crossServerChIdx].remoteCclMem.addr;

        // Butterfly threshold: for small data (<= 1MB), butterfly is faster
        constexpr uint64_t BUTTERFLY_THRESHOLD = 1 * 1024 * 1024;
        bool useButterfly = (totalBytes <= BUTTERFLY_THRESHOLD);

        if (useButterfly) {
            // ===== Butterfly AllReduce (for small data) =====
            // Per-channel CCL buffer: divide into 4 regions, one per channel/step
            uint64_t perChSlotSize = cclBufferSize / 4;
            perChSlotSize = perChSlotSize / MIN_SLICE_ALIGN * MIN_SLICE_ALIGN;
            if (perChSlotSize < elemSize) perChSlotSize = elemSize;

            uint64_t maxLoopElems = perChSlotSize / elemSize;
            uint64_t loopCount = (param.count + maxLoopElems - 1) / maxLoopElems;
            if (loopCount == 0) loopCount = 1;

            HCCL_INFO("Butterfly: totalBytes=%lu, perChSlotSize=%lu, maxLoopElems=%lu, loopCount=%lu",
                      totalBytes, (unsigned long)perChSlotSize, maxLoopElems, loopCount);

            for (uint64_t loop = 0; loop < loopCount; loop++) {
                uint64_t loopStart = loop * maxLoopElems;
                uint64_t loopElems = std::min(maxLoopElems, param.count - loopStart);
                uint64_t loopBytes = loopElems * elemSize;

                // Copy input to output for this loop
                if (loopBytes > 0) {
                    void *src = static_cast<char *>(param.inputPtr) + loopStart * elemSize;
                    void *dst = static_cast<char *>(param.outputPtr) + loopStart * elemSize;
                    CHK_RET(static_cast<HcclResult>(
                        HcommLocalCopyOnThread(thread, dst, src, loopBytes)));
                }

                // 4 butterfly steps, each uses a distinct per-channel buffer region
                for (uint32_t step = 0; step <= log2n; step++) {
                    ChannelHandle ch;
                    void *remoteBufBase;

                    if (step < log2n) {
                        ch = meshCh[step];
                        remoteBufBase = meshRemoteBuf[step];
                    } else {
                        ch = crossCh;
                        remoteBufBase = crossRemoteBuf;
                    }

                    void *stepLocalBuf = static_cast<char *>(localCclBuf) + step * perChSlotSize;
                    void *stepRemoteBuf = static_cast<char *>(remoteBufBase) + step * perChSlotSize;

                    // 1. Copy current output to per-channel CCL buffer
                    if (loopBytes > 0) {
                        void *src = static_cast<char *>(param.outputPtr) + loopStart * elemSize;
                        CHK_RET(static_cast<HcclResult>(
                            HcommLocalCopyOnThread(thread, stepLocalBuf, src, loopBytes)));
                    }

                    // 2-notify pattern: Record -> Wait -> ReadReduce (2 notify per step).
                    // Safe: each step uses a distinct buffer region, no contention.
                    uint32_t stepMask = (step < log2n) ? (1u << step) : nServer;
                    bool iAmLower = ((myRank & stepMask) == 0);

                    if (iAmLower) {
                        // Lower: Record ACK (data ready) -> Wait DATA_SIGNAL -> ReadReduce
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, ch, NOTIFY_IDX_ACK)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, ch, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                        if (loopElems > 0) {
                            void *dst = static_cast<char *>(param.outputPtr) + loopStart * elemSize;
                            CHK_RET(static_cast<HcclResult>(
                                HcommReadReduceOnThread(thread, ch, dst, stepRemoteBuf,
                                    loopElems, hcommDataType, hcommReduceOp)));
                        }
                    } else {
                        // Higher: Record DATA_SIGNAL (data ready) -> Wait ACK -> ReadReduce
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, ch, NOTIFY_IDX_DATA_SIGNAL)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, ch, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                        if (loopElems > 0) {
                            void *dst = static_cast<char *>(param.outputPtr) + loopStart * elemSize;
                            CHK_RET(static_cast<HcclResult>(
                                HcommReadReduceOnThread(thread, ch, dst, stepRemoteBuf,
                                    loopElems, hcommDataType, hcommReduceOp)));
                        }
                    }
                }
            }

            HCCL_INFO("Butterfly AllReduce completed successfully");
        } else {
            // ===== Recursive Halving/Doubling AllReduce (for large data) =====
            uint64_t maxSendElems = slotSize / elemSize;
            uint64_t maxLoopElems = 2 * maxSendElems;
            uint64_t loopCount = (param.count + maxLoopElems - 1) / maxLoopElems;
            if (loopCount == 0) loopCount = 1;

            HCCL_INFO("RecHalving: nServer=%u, log2n=%u, localRank=%u, totalBytes=%lu, "
                      "slotSize=%lu, maxLoopElems=%lu, loopCount=%lu",
                      nServer, log2n, myLocalRank, totalBytes, slotSize, maxLoopElems, loopCount);

            for (uint64_t loop = 0; loop < loopCount; loop++) {
                uint64_t loopStart = loop * maxLoopElems;
                uint64_t loopElems = std::min(maxLoopElems, param.count - loopStart);

                uint64_t chunkStart[9];
                chunkStart[0] = 0;
                uint64_t base = loopElems / nServer;
                uint32_t rem = static_cast<uint32_t>(loopElems % nServer);
                for (uint32_t i = 0; i < nServer; i++) {
                    chunkStart[i + 1] = chunkStart[i] + base + (i < rem ? 1 : 0);
                }

                // Initial copy deferred to Phase 1 step 0 (optimized: only copy keep+send chunks from input)

                // ===== Phase 1: Recursive Halving ReduceScatter (log2n steps) =====
                uint32_t activeChunkStart = 0;
                uint32_t activeChunkCount = nServer;

                for (uint32_t k = 0; k < log2n; k++) {
                    uint32_t mask = 1u << k;
                    ChannelHandle ch = meshCh[k];
                    void *remoteBuf = meshRemoteBuf[k];

                    uint32_t halfChunkCount = activeChunkCount / 2;
                    uint32_t sendChunkStart, sendChunkCount, keepChunkStart, keepChunkCount;

                    if ((myLocalRank & mask) == 0) {
                        sendChunkStart = activeChunkStart + halfChunkCount;
                        sendChunkCount = halfChunkCount;
                        keepChunkStart = activeChunkStart;
                        keepChunkCount = halfChunkCount;
                    } else {
                        sendChunkStart = activeChunkStart;
                        sendChunkCount = halfChunkCount;
                        keepChunkStart = activeChunkStart + halfChunkCount;
                        keepChunkCount = halfChunkCount;
                    }

                    uint64_t sendStart = chunkStart[sendChunkStart];
                    uint64_t sendLen = chunkStart[sendChunkStart + sendChunkCount] - sendStart;
                    uint64_t keepStart = chunkStart[keepChunkStart];
                    uint64_t keepLen = chunkStart[keepChunkStart + keepChunkCount] - keepStart;

                    if (k == 0) {
                        // Step 0: copy keep chunk from input to output, send chunk from input to CCL buf
                        if (keepLen > 0) {
                            void *src = static_cast<char *>(param.inputPtr) + (loopStart + keepStart) * elemSize;
                            void *dst = static_cast<char *>(param.outputPtr) + (loopStart + keepStart) * elemSize;
                            CHK_RET(static_cast<HcclResult>(
                                HcommLocalCopyOnThread(thread, dst, src, keepLen * elemSize)));
                        }
                        if (sendLen > 0) {
                            void *src = static_cast<char *>(param.inputPtr) + (loopStart + sendStart) * elemSize;
                            CHK_RET(static_cast<HcclResult>(
                                HcommLocalCopyOnThread(thread, localCclBuf, src, sendLen * elemSize)));
                        }
                    } else {
                        if (sendLen > 0) {
                            void *src = static_cast<char *>(param.outputPtr) + (loopStart + sendStart) * elemSize;
                            CHK_RET(static_cast<HcclResult>(
                                HcommLocalCopyOnThread(thread, localCclBuf, src, sendLen * elemSize)));
                        }
                    }

                    // 4-notify pattern: Record -> Wait -> ReadReduce -> Record -> Wait
                    bool iAmLower = ((myLocalRank & mask) == 0);
                    if (iAmLower) {
                        // Lower: Record ACK (data ready) -> Wait DATA_SIGNAL -> ReadReduce -> Record ACK (done) -> Wait DATA_SIGNAL
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, ch, NOTIFY_IDX_ACK)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, ch, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                        if (keepLen > 0) {
                            void *dst = static_cast<char *>(param.outputPtr) + (loopStart + keepStart) * elemSize;
                            CHK_RET(static_cast<HcclResult>(
                                HcommReadReduceOnThread(thread, ch, dst, remoteBuf,
                                    keepLen, hcommDataType, hcommReduceOp)));
                        }
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, ch, NOTIFY_IDX_ACK)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, ch, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                    } else {
                        // Higher: Record DATA_SIGNAL (data ready) -> Wait ACK -> ReadReduce -> Record DATA_SIGNAL (done) -> Wait ACK
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, ch, NOTIFY_IDX_DATA_SIGNAL)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, ch, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                        if (keepLen > 0) {
                            void *dst = static_cast<char *>(param.outputPtr) + (loopStart + keepStart) * elemSize;
                            CHK_RET(static_cast<HcclResult>(
                                HcommReadReduceOnThread(thread, ch, dst, remoteBuf,
                                    keepLen, hcommDataType, hcommReduceOp)));
                        }
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, ch, NOTIFY_IDX_DATA_SIGNAL)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, ch, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                    }

                    activeChunkStart = keepChunkStart;
                    activeChunkCount = keepChunkCount;
                }

                // ===== Phase 2: Cross-server exchange (1 step, 4-notify pattern) =====
                {
                    uint64_t crossStart = chunkStart[activeChunkStart];
                    uint64_t crossLen = chunkStart[activeChunkStart + 1] - crossStart;

                    if (crossLen > 0) {
                        void *src = static_cast<char *>(param.outputPtr) + (loopStart + crossStart) * elemSize;
                        CHK_RET(static_cast<HcclResult>(
                            HcommLocalCopyOnThread(thread, localCclBuf, src, crossLen * elemSize)));
                    }

                    // Parallel ReadReduce pattern: both sides ReadReduce simultaneously.
                    bool iAmLower = ((myRank & nServer) == 0);
                    if (iAmLower) {
                        // Lower: Record ACK (data ready) -> Wait DATA_SIGNAL -> ReadReduce -> Record ACK (done) -> Wait DATA_SIGNAL
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, crossCh, NOTIFY_IDX_ACK)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, crossCh, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                        if (crossLen > 0) {
                            void *dst = static_cast<char *>(param.outputPtr) + (loopStart + crossStart) * elemSize;
                            CHK_RET(static_cast<HcclResult>(
                                HcommReadReduceOnThread(thread, crossCh, dst, crossRemoteBuf,
                                    crossLen, hcommDataType, hcommReduceOp)));
                        }
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, crossCh, NOTIFY_IDX_ACK)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, crossCh, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                    } else {
                        // Higher: Record DATA_SIGNAL (data ready) -> Wait ACK -> ReadReduce -> Record DATA_SIGNAL (done) -> Wait ACK
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, crossCh, NOTIFY_IDX_DATA_SIGNAL)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, crossCh, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                        if (crossLen > 0) {
                            void *dst = static_cast<char *>(param.outputPtr) + (loopStart + crossStart) * elemSize;
                            CHK_RET(static_cast<HcclResult>(
                                HcommReadReduceOnThread(thread, crossCh, dst, crossRemoteBuf,
                                    crossLen, hcommDataType, hcommReduceOp)));
                        }
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, crossCh, NOTIFY_IDX_DATA_SIGNAL)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, crossCh, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                    }
                }

                // ===== Phase 3: Recursive Doubling AllGather (log2n steps) =====
                for (uint32_t k = 0; k < log2n; k++) {
                    uint32_t mask = 1u << (log2n - 1 - k);
                    uint32_t chIdx = log2n - 1 - k;
                    ChannelHandle ch = meshCh[chIdx];
                    void *remoteBuf = meshRemoteBuf[chIdx];

                    uint64_t sendStart = chunkStart[activeChunkStart];
                    uint64_t sendLen = chunkStart[activeChunkStart + activeChunkCount] - sendStart;

                    uint32_t recvChunkStart, recvChunkCount;
                    if ((myLocalRank & mask) == 0) {
                        recvChunkStart = activeChunkStart + activeChunkCount;
                        recvChunkCount = activeChunkCount;
                    } else {
                        recvChunkStart = activeChunkStart - activeChunkCount;
                        recvChunkCount = activeChunkCount;
                    }

                    uint64_t recvStart = chunkStart[recvChunkStart];
                    uint64_t recvLen = chunkStart[recvChunkStart + recvChunkCount] - recvStart;

                    if (sendLen > 0) {
                        void *src = static_cast<char *>(param.outputPtr) + (loopStart + sendStart) * elemSize;
                        CHK_RET(static_cast<HcclResult>(
                            HcommLocalCopyOnThread(thread, localCclBuf, src, sendLen * elemSize)));
                    }

                    // 4-notify pattern with skip for last step of final loop
                    bool skipDoneWait = (loop == loopCount - 1) && (k == log2n - 1);
                    bool iAmLower = ((myLocalRank & mask) == 0);
                    if (iAmLower) {
                        // Lower: Record ACK (data ready) -> Wait DATA_SIGNAL -> Read -> [Record ACK -> Wait DATA_SIGNAL]
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, ch, NOTIFY_IDX_ACK)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, ch, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                        if (recvLen > 0) {
                            void *dst = static_cast<char *>(param.outputPtr) + (loopStart + recvStart) * elemSize;
                            CHK_RET(static_cast<HcclResult>(
                                HcommReadOnThread(thread, ch, dst, remoteBuf, recvLen * elemSize)));
                        }
                        if (!skipDoneWait) {
                            CHK_RET(static_cast<HcclResult>(
                                HcommChannelNotifyRecordOnThread(thread, ch, NOTIFY_IDX_ACK)));
                            CHK_RET(static_cast<HcclResult>(
                                HcommChannelNotifyWaitOnThread(thread, ch, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                        }
                    } else {
                        // Higher: Record DATA_SIGNAL (data ready) -> Wait ACK -> Read -> [Record DATA_SIGNAL -> Wait ACK]
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyRecordOnThread(thread, ch, NOTIFY_IDX_DATA_SIGNAL)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommChannelNotifyWaitOnThread(thread, ch, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                        if (recvLen > 0) {
                            void *dst = static_cast<char *>(param.outputPtr) + (loopStart + recvStart) * elemSize;
                            CHK_RET(static_cast<HcclResult>(
                                HcommReadOnThread(thread, ch, dst, remoteBuf, recvLen * elemSize)));
                        }
                        if (!skipDoneWait) {
                            CHK_RET(static_cast<HcclResult>(
                                HcommChannelNotifyRecordOnThread(thread, ch, NOTIFY_IDX_DATA_SIGNAL)));
                            CHK_RET(static_cast<HcclResult>(
                                HcommChannelNotifyWaitOnThread(thread, ch, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                        }
                    }

                    if ((myLocalRank & mask) == 0) {
                        activeChunkCount = 2 * activeChunkCount;
                    } else {
                        activeChunkStart = recvChunkStart;
                        activeChunkCount = 2 * activeChunkCount;
                    }
                }
            }

            HCCL_INFO("Recursive Halving AllReduce completed successfully");
        }
    } else if (resCtx.useHierarchical) {
        // ===== Fallback: Hierarchical Ring AllReduce (v23 proven algorithm) =====
        uint32_t nServer = resCtx.serverRankSize;
        uint32_t myLocalRank = myRank % nServer;

        ChannelHandle prevCh = resCtx.channels[resCtx.serverPrevIdx].handle;
        ChannelHandle nextCh = resCtx.channels[resCtx.serverNextIdx].handle;
        ChannelHandle partnerCh = resCtx.channels[resCtx.crossServerChIdx].handle;
        void *prevRemoteBuf = resCtx.channels[resCtx.serverPrevIdx].remoteCclMem.addr;
        void *partnerRemoteBuf = resCtx.channels[resCtx.crossServerChIdx].remoteCclMem.addr;

        uint64_t maxChunkElems = slotSize / elemSize;
        uint64_t maxLoopElems = maxChunkElems * nServer;
        uint64_t loopCount = (param.count + maxLoopElems - 1) / maxLoopElems;
        if (loopCount == 0) loopCount = 1;

        HCCL_INFO("Hierarchical ring: nServer=%u, localRank=%u, slotSize=%lu, loopCount=%lu",
                  nServer, myLocalRank, slotSize, loopCount);

        for (uint64_t loop = 0; loop < loopCount; loop++) {
            uint64_t loopStart = loop * maxLoopElems;
            uint64_t loopElems = std::min(maxLoopElems, param.count - loopStart);

            uint64_t chunkStart[9];
            chunkStart[0] = 0;
            uint64_t base = loopElems / nServer;
            uint32_t rem = static_cast<uint32_t>(loopElems % nServer);
            for (uint32_t i = 0; i < nServer; i++) {
                chunkStart[i + 1] = chunkStart[i] + base + (i < rem ? 1 : 0);
            }

            uint64_t loopBytes = loopElems * elemSize;
            if (loopBytes > 0) {
                void *src = static_cast<char *>(param.inputPtr) + loopStart * elemSize;
                void *dst = static_cast<char *>(param.outputPtr) + loopStart * elemSize;
                CHK_RET(static_cast<HcclResult>(
                    HcommLocalCopyOnThread(thread, dst, src, loopBytes)));
            }

            // Phase 1: ReduceScatter (nServer-1 steps)
            for (uint32_t step = 0; step < nServer - 1; step++) {
                uint32_t sendIdx = (myLocalRank - step + nServer) % nServer;
                uint32_t recvIdx = (myLocalRank - step - 1 + nServer) % nServer;
                uint64_t sendLen = chunkStart[sendIdx + 1] - chunkStart[sendIdx];
                uint64_t recvLen = chunkStart[recvIdx + 1] - chunkStart[recvIdx];

                if (sendLen > 0) {
                    void *src = static_cast<char *>(param.outputPtr) +
                        (loopStart + chunkStart[sendIdx]) * elemSize;
                    CHK_RET(static_cast<HcclResult>(
                        HcommLocalCopyOnThread(thread, localCclBuf, src, sendLen * elemSize)));
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, nextCh, NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, prevCh, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                if (recvLen > 0) {
                    void *dst = static_cast<char *>(param.outputPtr) +
                        (loopStart + chunkStart[recvIdx]) * elemSize;
                    CHK_RET(static_cast<HcclResult>(
                        HcommReadReduceOnThread(thread, prevCh, dst, prevRemoteBuf,
                            recvLen, hcommDataType, hcommReduceOp)));
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, prevCh, NOTIFY_IDX_DATA_SIGNAL)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, nextCh, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }

            // Phase 2: Cross-server (1 step)
            {
                uint32_t crossChunkIdx = (myLocalRank + 1) % nServer;
                uint64_t crossLen = chunkStart[crossChunkIdx + 1] - chunkStart[crossChunkIdx];

                if (crossLen > 0) {
                    void *src = static_cast<char *>(param.outputPtr) +
                        (loopStart + chunkStart[crossChunkIdx]) * elemSize;
                    CHK_RET(static_cast<HcclResult>(
                        HcommLocalCopyOnThread(thread, localCclBuf, src, crossLen * elemSize)));
                }
                // Asymmetric notify pattern: each notify index has exactly one writer.
                bool iAmLower = ((myRank & nServer) == 0);
                if (iAmLower) {
                    // Lower rank: Record ACK (data ready) -> Wait DATA_SIGNAL -> ReadReduce -> Record ACK (done reading)
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyRecordOnThread(thread, partnerCh, NOTIFY_IDX_ACK)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(thread, partnerCh, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                    if (crossLen > 0) {
                        void *dst = static_cast<char *>(param.outputPtr) +
                            (loopStart + chunkStart[crossChunkIdx]) * elemSize;
                        CHK_RET(static_cast<HcclResult>(
                            HcommReadReduceOnThread(thread, partnerCh, dst, partnerRemoteBuf,
                                crossLen, hcommDataType, hcommReduceOp)));
                    }
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyRecordOnThread(thread, partnerCh, NOTIFY_IDX_ACK)));
                } else {
                    // Higher rank: Wait ACK (data ready) -> ReadReduce -> Record DATA_SIGNAL (done reading) -> Wait ACK (done reading)
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(thread, partnerCh, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                    if (crossLen > 0) {
                        void *dst = static_cast<char *>(param.outputPtr) +
                            (loopStart + chunkStart[crossChunkIdx]) * elemSize;
                        CHK_RET(static_cast<HcclResult>(
                            HcommReadReduceOnThread(thread, partnerCh, dst, partnerRemoteBuf,
                                crossLen, hcommDataType, hcommReduceOp)));
                    }
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyRecordOnThread(thread, partnerCh, NOTIFY_IDX_DATA_SIGNAL)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(thread, partnerCh, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                }
            }

            // Phase 3: AllGather (nServer-1 steps)
            for (uint32_t step = 0; step < nServer - 1; step++) {
                uint32_t sendIdx = (myLocalRank + 1 - step + nServer) % nServer;
                uint32_t recvIdx = (myLocalRank - step + nServer) % nServer;
                uint64_t sendLen = chunkStart[sendIdx + 1] - chunkStart[sendIdx];
                uint64_t recvLen = chunkStart[recvIdx + 1] - chunkStart[recvIdx];

                if (sendLen > 0) {
                    void *src = static_cast<char *>(param.outputPtr) +
                        (loopStart + chunkStart[sendIdx]) * elemSize;
                    CHK_RET(static_cast<HcclResult>(
                        HcommLocalCopyOnThread(thread, localCclBuf, src, sendLen * elemSize)));
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, nextCh, NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, prevCh, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                if (recvLen > 0) {
                    void *dst = static_cast<char *>(param.outputPtr) +
                        (loopStart + chunkStart[recvIdx]) * elemSize;
                    CHK_RET(static_cast<HcclResult>(
                        HcommReadOnThread(thread, prevCh, dst, prevRemoteBuf, recvLen * elemSize)));
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, prevCh, NOTIFY_IDX_DATA_SIGNAL)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, nextCh, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }
        }
        HCCL_INFO("Hierarchical ring AllReduce completed successfully");
    } else {
        // ===== Fallback: Simple Ring AllReduce =====
        uint32_t ringPos = resCtx.ringPos;
        ChannelHandle prevCh = resCtx.channels[0].handle;
        ChannelHandle nextCh = resCtx.channels[1].handle;
        void *prevRemoteBuf = resCtx.channels[0].remoteCclMem.addr;

        uint64_t maxChunkElems = slotSize / elemSize;
        uint64_t maxLoopElems = maxChunkElems * n;
        uint64_t loopCount = (param.count + maxLoopElems - 1) / maxLoopElems;
        if (loopCount == 0) loopCount = 1;

        HCCL_INFO("Simple ring: n=%u, rank=%u, ringPos=%u, slotSize=%lu, loopCount=%lu",
                  n, myRank, ringPos, slotSize, loopCount);

        for (uint64_t loop = 0; loop < loopCount; loop++) {
            uint64_t loopStart = loop * maxLoopElems;
            uint64_t loopElems = std::min(maxLoopElems, param.count - loopStart);

            uint64_t chunkStart[32];
            chunkStart[0] = 0;
            uint64_t base = loopElems / n;
            uint32_t rem = static_cast<uint32_t>(loopElems % n);
            for (uint32_t i = 0; i < n && i < 31; i++) {
                chunkStart[i + 1] = chunkStart[i] + base + (i < rem ? 1 : 0);
            }

            uint64_t loopBytes = loopElems * elemSize;
            if (loopBytes > 0) {
                void *src = static_cast<char *>(param.inputPtr) + loopStart * elemSize;
                void *dst = static_cast<char *>(param.outputPtr) + loopStart * elemSize;
                CHK_RET(static_cast<HcclResult>(
                    HcommLocalCopyOnThread(thread, dst, src, loopBytes)));
            }

            // Phase 1: ReduceScatter (n-1 steps)
            for (uint32_t step = 0; step < n - 1; step++) {
                uint32_t sendIdx = (ringPos - step + n) % n;
                uint32_t recvIdx = (ringPos - step - 1 + n) % n;
                uint64_t sendLen = chunkStart[sendIdx + 1] - chunkStart[sendIdx];
                uint64_t recvLen = chunkStart[recvIdx + 1] - chunkStart[recvIdx];

                if (sendLen > 0) {
                    void *src = static_cast<char *>(param.outputPtr) +
                        (loopStart + chunkStart[sendIdx]) * elemSize;
                    CHK_RET(static_cast<HcclResult>(
                        HcommLocalCopyOnThread(thread, localCclBuf, src, sendLen * elemSize)));
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, nextCh, NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, prevCh, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                if (recvLen > 0) {
                    void *dst = static_cast<char *>(param.outputPtr) +
                        (loopStart + chunkStart[recvIdx]) * elemSize;
                    CHK_RET(static_cast<HcclResult>(
                        HcommReadReduceOnThread(thread, prevCh, dst, prevRemoteBuf,
                            recvLen, hcommDataType, hcommReduceOp)));
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, prevCh, NOTIFY_IDX_DATA_SIGNAL)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, nextCh, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }

            // Phase 2: AllGather (n-1 steps)
            for (uint32_t step = 0; step < n - 1; step++) {
                uint32_t sendIdx = (ringPos + 1 - step + n) % n;
                uint32_t recvIdx = (ringPos - step + n) % n;
                uint64_t sendLen = chunkStart[sendIdx + 1] - chunkStart[sendIdx];
                uint64_t recvLen = chunkStart[recvIdx + 1] - chunkStart[recvIdx];

                if (sendLen > 0) {
                    void *src = static_cast<char *>(param.outputPtr) +
                        (loopStart + chunkStart[sendIdx]) * elemSize;
                    CHK_RET(static_cast<HcclResult>(
                        HcommLocalCopyOnThread(thread, localCclBuf, src, sendLen * elemSize)));
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, nextCh, NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, prevCh, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                if (recvLen > 0) {
                    void *dst = static_cast<char *>(param.outputPtr) +
                        (loopStart + chunkStart[recvIdx]) * elemSize;
                    CHK_RET(static_cast<HcclResult>(
                        HcommReadOnThread(thread, prevCh, dst, prevRemoteBuf, recvLen * elemSize)));
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, prevCh, NOTIFY_IDX_DATA_SIGNAL)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, nextCh, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }
        }
        HCCL_INFO("Simple ring AllReduce completed successfully");
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
