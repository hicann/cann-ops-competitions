/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
// 在 channels 中按 remoteRank 查找 channel
const ChannelInfo *FindChannel(const std::vector<ChannelInfo> &channels, uint32_t remoteRank)
{
    for (const auto &ch : channels) {
        if (ch.remoteRank == remoteRank) {
            return &ch;
        }
    }
    return nullptr;
}

// 统一检查 Hcomm 返回值
HcclResult CheckRet(int32_t ret, const char *act, uint32_t myRank, uint32_t peer)
{
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[%s] rank[%u] peer[%u] ret[%d]", act, myRank, peer, ret),
                static_cast<HcclResult>(ret));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const uint32_t myRank = param.myRank;
    const uint32_t root = param.root;
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("[%s] rank[%u] has no thread", __func__, myRank), HCCL_E_INTERNAL);
    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t threadNum = static_cast<uint32_t>(resCtx.threads.size());
    const uint32_t channelNum = static_cast<uint32_t>(resCtx.channels.size());
    void *userBuf = param.inputPtr;
    void *localCcl = resCtx.localBuffer.addr;
    const uint64_t cclSize = resCtx.localBuffer.size;

    auto it = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(it == SIZE_TABLE.end(), HCCL_ERROR("[%s] bad dtype[%u]", __func__, param.dataType), HCCL_E_PARA);
    const uint64_t dataLen = param.count * it->second;
    CHK_PRT_RET(param.rankSize == 0 || root >= param.rankSize,
                HCCL_ERROR("[%s] rank[%u] invalid rankSize[%u] or root[%u]", __func__, myRank, param.rankSize, root),
                HCCL_E_PARA);
    if (param.rankSize == 1 || dataLen == 0) {
        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootCh = FindChannel(resCtx.channels, root);
    constexpr uint64_t SMALL_THRESHOLD = 1 * 1024 * 1024; // 1MB: 以下直接全发, 以上 Mesh TwoShot
    constexpr uint64_t UB_MAX_DATA_SIZE = 256 * 1024 * 1024;
    uint64_t compactSlotCapacity = cclSize / 2;
    if (compactSlotCapacity > UB_MAX_DATA_SIZE) {
        compactSlotCapacity = UB_MAX_DATA_SIZE;
    }
    compactSlotCapacity = (compactSlotCapacity / it->second) * it->second;
    CHK_PRT_RET(compactSlotCapacity == 0,
                HCCL_ERROR("[%s] rank[%u] cclSize[%lu] cannot hold compact slots", __func__, myRank, cclSize),
                HCCL_E_INTERNAL);
    const uint64_t maxUint64 = ~static_cast<uint64_t>(0);
    CHK_PRT_RET(compactSlotCapacity > maxUint64 / param.rankSize,
                HCCL_ERROR("[%s] rank[%u] compact chunk size overflow", __func__, myRank), HCCL_E_INTERNAL);
    const uint64_t maxReadChunkSize = compactSlotCapacity * param.rankSize;
    bool usedReadAllGather = false;

    uint64_t offset = 0;
    while (offset < dataLen) {
        uint64_t chunk = dataLen - offset;
        if (chunk > maxReadChunkSize) {
            chunk = maxReadChunkSize;
        }
        const bool smallChunk = (chunk <= SMALL_THRESHOLD);
        const void *srcBase = static_cast<const uint8_t *>(userBuf) + offset;
        void *dstBase = static_cast<uint8_t *>(userBuf) + offset;

        if (smallChunk) {
            if (myRank == root) {
                // 小数据：root只暂存一次，nonroot从root CCL直接Read到用户Buffer。
                CHK_PRT_RET(chunk > cclSize,
                            HCCL_ERROR("[%s] root[%u] local buffer[%lu] < small chunk[%lu]", __func__, root,
                                       cclSize, chunk),
                            HCCL_E_INTERNAL);
                int32_t r = HcommLocalCopyOnThread(thread, localCcl, srcBase, chunk);
                CHK_RET(CheckRet(r, "Stage small root", myRank, root));
                for (const auto &ch : resCtx.channels) {
                    r = HcommChannelNotifyRecordOnThread(thread, ch.handle, NOTIFY_IDX_DATA_SIGNAL);
                    CHK_RET(CheckRet(r, "NotifyDATA", myRank, ch.remoteRank));
                }
                for (const auto &ch : resCtx.channels) {
                    r = HcommChannelNotifyWaitOnThread(thread, ch.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT);
                    CHK_RET(CheckRet(r, "WaitACK", myRank, ch.remoteRank));
                }
            } else {
                // 小数据非root：等待root staging完成，Read直达用户Buffer，再返回ACK。
                CHK_PRT_RET(rootCh == nullptr, HCCL_ERROR("[%s] rank[%u] no rootCh", __func__, myRank),
                            HCCL_E_INTERNAL);
                CHK_PRT_RET(chunk > rootCh->remoteCclMem.size,
                            HCCL_ERROR("[%s] rank[%u] root[%u] remote buffer[%lu] < small chunk[%lu]", __func__,
                                       myRank, root, rootCh->remoteCclMem.size, chunk),
                            HCCL_E_INTERNAL);
                int32_t r = HcommChannelNotifyWaitOnThread(thread, rootCh->handle, NOTIFY_IDX_DATA_SIGNAL,
                                                           CUSTOM_TIMEOUT);
                CHK_RET(CheckRet(r, "WaitDATA root", myRank, root));
                r = HcommReadOnThread(thread, rootCh->handle, dstBase, rootCh->remoteCclMem.addr, chunk);
                CHK_RET(CheckRet(r, "Read small root", myRank, root));
                r = HcommChannelNotifyRecordOnThread(thread, rootCh->handle, NOTIFY_IDX_ACK);
                CHK_RET(CheckRet(r, "NotifyACK root", myRank, root));
            }
        } else {
            usedReadAllGather = true;
            // 大数据：root只Scatter各peer own slice；所有非root再从15个对端CCL Read直达用户Buffer。
            CHK_PRT_RET(threadNum != channelNum,
                        HCCL_ERROR("[%s] rank[%u] threads[%u] != channels[%u]", __func__, myRank, threadNum,
                                   channelNum),
                        HCCL_E_INTERNAL);
            CHK_PRT_RET(chunk % it->second != 0,
                        HCCL_ERROR("[%s] rank[%u] chunk[%lu] is not dtype aligned", __func__, myRank, chunk),
                        HCCL_E_INTERNAL);

            uint32_t rootChannelIdx = channelNum;
            if (myRank != root) {
                for (uint32_t i = 0; i < channelNum; ++i) {
                    if (resCtx.channels[i].remoteRank == root) {
                        rootChannelIdx = i;
                        break;
                    }
                }
            }

            const uint64_t chunkCount = chunk / it->second;
            const uint64_t sliceCount = (chunkCount + param.rankSize - 1) / param.rankSize;
            const uint64_t sliceStride = sliceCount * it->second;
            CHK_PRT_RET(sliceStride > compactSlotCapacity,
                        HCCL_ERROR("[%s] rank[%u] sliceStride[%lu] > slotCapacity[%lu]", __func__, myRank,
                                   sliceStride, compactSlotCapacity),
                        HCCL_E_INTERNAL);
            auto SliceSize = [chunk, sliceStride](uint32_t rank) -> uint64_t {
                const uint64_t userOffset = static_cast<uint64_t>(rank) * sliceStride;
                if (userOffset >= chunk) {
                    return 0;
                }
                const uint64_t remain = chunk - userOffset;
                return remain < sliceStride ? remain : sliceStride;
            };

            constexpr uint64_t OWN_SLOT_OFFSET = 0;
            const uint64_t rootUserOffset = static_cast<uint64_t>(root) * sliceStride;
            const uint64_t rootSliceSize = SliceSize(root);
            const uint64_t myUserOffset = static_cast<uint64_t>(myRank) * sliceStride;
            const uint64_t mySliceSize = SliceSize(myRank);

            if (myRank == root && rootSliceSize > 0) {
                // staging排在main->slave启动信号之前；main stream FIFO保证任何Scatter DATA前root源已就绪。
                int32_t r = HcommLocalCopyOnThread(
                    thread, static_cast<uint8_t *>(localCcl) + OWN_SLOT_OFFSET,
                    static_cast<const uint8_t *>(srcBase) + rootUserOffset, rootSliceSize);
                CHK_RET(CheckRet(r, "Stage root slice", myRank, root));
            }

            if (myRank != root) {
                // DATA保证本rank own slice已到达；root staging早于root的全部worker启动，因此也已经就绪。
                CHK_PRT_RET(rootCh == nullptr, HCCL_ERROR("[%s] rank[%u] no rootCh", __func__, myRank),
                            HCCL_E_INTERNAL);
                CHK_PRT_RET(rootChannelIdx >= threadNum,
                            HCCL_ERROR("[%s] rank[%u] invalid root channel index[%u]", __func__, myRank,
                                       rootChannelIdx),
                            HCCL_E_INTERNAL);
                int32_t r = HcommChannelNotifyWaitOnThread(thread, rootCh->handle, NOTIFY_IDX_DATA_SIGNAL,
                                                           CUSTOM_TIMEOUT);
                CHK_RET(CheckRet(r, "Scatter wait root", myRank, root));
            }

            // main 启动所有 slave；slave notify 0 接收启动，main notify 1..N 等待完成。
            for (uint32_t i = 1; i < channelNum; i++) {
                int32_t r = HcommThreadNotifyRecordOnThread(thread, resCtx.threads[i], 0);
                CHK_RET(CheckRet(r, "Start slave", myRank, resCtx.channels[i].remoteRank));
            }
            for (uint32_t i = 1; i < channelNum; i++) {
                int32_t r = HcommThreadNotifyWaitOnThread(resCtx.threads[i], 0, CUSTOM_TIMEOUT);
                CHK_RET(CheckRet(r, "Wait start", myRank, resCtx.channels[i].remoteRank));
            }

            if (myRank == root) {
                // 标准Scatter：只发送每个非root自己的slice；root slice保留在本端CCL供远端Read。
                for (uint32_t i = 0; i < channelNum; i++) {
                    const ChannelInfo &ch = resCtx.channels[i];
                    const ThreadHandle worker = resCtx.threads[i];
                    const uint64_t peerUserOffset = static_cast<uint64_t>(ch.remoteRank) * sliceStride;
                    const uint64_t peerSliceSize = SliceSize(ch.remoteRank);
                    CHK_PRT_RET(peerSliceSize > ch.remoteCclMem.size,
                                HCCL_ERROR("[%s] root[%u] peer[%u] remote buffer[%lu] < own slice[%lu]",
                                           __func__, root, ch.remoteRank, ch.remoteCclMem.size, peerSliceSize),
                                HCCL_E_INTERNAL);
                    if (peerSliceSize > 0) {
                        void *dst = static_cast<uint8_t *>(ch.remoteCclMem.addr) + OWN_SLOT_OFFSET;
                        const void *src = static_cast<const uint8_t *>(srcBase) + peerUserOffset;
                        int32_t r = HcommWriteOnThread(worker, ch.handle, dst, src, peerSliceSize);
                        CHK_RET(CheckRet(r, "Scatter write peer slice", myRank, ch.remoteRank));
                    }
                    int32_t r = HcommChannelNotifyRecordOnThread(worker, ch.handle, NOTIFY_IDX_DATA_SIGNAL);
                    CHK_RET(CheckRet(r, "Scatter DATA", myRank, ch.remoteRank));
                    r = HcommChannelNotifyWaitOnThread(worker, ch.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT);
                    CHK_RET(CheckRet(r, "TwoShot ACK", myRank, ch.remoteRank));
                }
            } else {
                // root channel无需READY握手：DATA到达时root CCL源已由main FIFO保证就绪，直接Read到用户Buffer。
                const ThreadHandle copyWorker = resCtx.threads[rootChannelIdx];
                CHK_PRT_RET(rootSliceSize > rootCh->remoteCclMem.size,
                            HCCL_ERROR("[%s] rank[%u] root[%u] remote buffer[%lu] < root slice[%lu]", __func__,
                                       myRank, root, rootCh->remoteCclMem.size, rootSliceSize),
                            HCCL_E_INTERNAL);
                if (rootSliceSize > 0) {
                    int32_t r = HcommReadOnThread(
                        copyWorker, rootCh->handle, static_cast<uint8_t *>(dstBase) + rootUserOffset,
                        static_cast<const uint8_t *>(rootCh->remoteCclMem.addr) + OWN_SLOT_OFFSET, rootSliceSize);
                    CHK_RET(CheckRet(r, "Read root slice", myRank, root));
                }
                if (mySliceSize > 0) {
                    int32_t r = HcommLocalCopyOnThread(
                        copyWorker, static_cast<uint8_t *>(dstBase) + myUserOffset,
                        static_cast<const uint8_t *>(localCcl) + OWN_SLOT_OFFSET, mySliceSize);
                    CHK_RET(CheckRet(r, "Copy own slice", myRank, myRank));
                }

                // 其余14个非root之间仍先发布own slice READY，再从对端CCL直读到用户Buffer。
                for (uint32_t i = 0; i < channelNum; i++) {
                    const ChannelInfo &ch = resCtx.channels[i];
                    if (ch.remoteRank == root) {
                        continue;
                    }
                    const ThreadHandle worker = resCtx.threads[i];
                    const uint64_t peerUserOffset = static_cast<uint64_t>(ch.remoteRank) * sliceStride;
                    const uint64_t peerSliceSize = SliceSize(ch.remoteRank);
                    CHK_PRT_RET(peerSliceSize > ch.remoteCclMem.size,
                                HCCL_ERROR("[%s] rank[%u] peer[%u] remote buffer[%lu] < own slice[%lu]", __func__,
                                           myRank, ch.remoteRank, ch.remoteCclMem.size, peerSliceSize),
                                HCCL_E_INTERNAL);
                    int32_t r = HcommChannelNotifyRecordOnThread(worker, ch.handle, NOTIFY_IDX_ACK);
                    CHK_RET(CheckRet(r, "ReadAllGather READY", myRank, ch.remoteRank));
                    r = HcommChannelNotifyWaitOnThread(worker, ch.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT);
                    CHK_RET(CheckRet(r, "ReadAllGather wait READY", myRank, ch.remoteRank));
                    if (peerSliceSize > 0) {
                        r = HcommReadOnThread(
                            worker, ch.handle, static_cast<uint8_t *>(dstBase) + peerUserOffset,
                            static_cast<const uint8_t *>(ch.remoteCclMem.addr) + OWN_SLOT_OFFSET, peerSliceSize);
                        CHK_RET(CheckRet(r, "ReadAllGather read", myRank, ch.remoteRank));
                    }
                }
            }

            // 汇合所有 worker，确保当前分块的 Read 及 root/own LocalCopy 已完成。
            for (uint32_t i = 1; i < channelNum; i++) {
                int32_t r = HcommThreadNotifyWaitOnThread(thread, i, CUSTOM_TIMEOUT);
                CHK_RET(CheckRet(r, "Wait slave", myRank, resCtx.channels[i].remoteRank));
            }
            for (uint32_t i = 1; i < channelNum; i++) {
                int32_t r = HcommThreadNotifyRecordOnThread(resCtx.threads[i], thread, i);
                CHK_RET(CheckRet(r, "Finish slave", myRank, resCtx.channels[i].remoteRank));
            }

            if (myRank != root) {
                int32_t r = HcommChannelNotifyRecordOnThread(thread, rootCh->handle, NOTIFY_IDX_ACK);
                CHK_RET(CheckRet(r, "TwoShot ACK root", myRank, root));
            }
        }

        offset += chunk;
    }

    HCCL_INFO("ExecOp v0.4.10 rank[%u] root[%u] usedReadAllGather=%d done", myRank, root, usedReadAllGather);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
