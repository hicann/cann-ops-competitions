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
#include <limits>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_READY = 0;
constexpr uint32_t CHANNEL_NOTIFY_CONSUMED = 1;
constexpr uint32_t THREAD_NOTIFY_START = 0;
constexpr uint32_t THREAD_NOTIFY_TREE_LEFT = 1;
constexpr uint32_t THREAD_NOTIFY_TREE_RIGHT = 2;
// AICPU主Thread的notify 0在进入ExecOp前已经消费完Host启动信号，可安全复用为本地拷贝完成信号。
constexpr uint32_t THREAD_NOTIFY_LOCAL_COPY_DONE = 0;
constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t SMALL_MESSAGE_THRESHOLD = 512UL * 1024UL;
constexpr uint64_t P6_MESSAGE_BYTES = 512ULL * 1024ULL * 1024ULL;

void *ByteOffset(void *ptr, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(ptr) + offset);
}


HcclResult QueuePeerExchange(ThreadHandle thread, const ChannelInfo &channel, void *dst,
    void *remoteSrc, uint64_t chunkBytes, bool confirmConsumed)
{
    // 对称握手：双方均先声明本端中转数据已就绪，再读取远端。
    // 非最后一块继续执行CONSUMED确认，保证对应Bank再次复用前所有对端读取均已完成；
    // 最后一块之后不会再覆盖HCCL Buffer，因此可对称省略这组确认。
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, CHANNEL_NOTIFY_READY));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, CHANNEL_NOTIFY_READY, CUSTOM_TIMEOUT));
    CHK_RET(HcommReadOnThread(thread, channel.handle, dst, remoteSrc, chunkBytes));
    if (confirmConsumed) {
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, CHANNEL_NOTIFY_CONSUMED));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, CHANNEL_NOTIFY_CONSUMED, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult QueuePeerExchange(ThreadHandle thread, const ChannelInfo &channel, void *dst, uint64_t chunkBytes)
{
    return QueuePeerExchange(thread, channel, dst, channel.remoteCclMem.addr, chunkBytes, true);
}


const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

HcclResult QueueChannelWrite(ThreadHandle thread, const ChannelInfo &channel, void *remoteAddress,
    const void *localAddress, uint64_t size, bool useFusedNotify, uint32_t notifyIndex)
{
    if (useFusedNotify) {
        // Ascend 950能力探针：将Write与远端Notify融合，随后只保留一次本端Wait。
        CHK_RET(HcommWriteWithNotifyOnThread(
            thread, channel.handle, remoteAddress, localAddress, size, notifyIndex));
    } else {
        CHK_RET(HcommWriteOnThread(thread, channel.handle, remoteAddress, localAddress, size));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyIndex));
    }
    CHK_RET(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, notifyIndex, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

// 512KiB及以下采用4轮Recursive Doubling：每轮只与rank^1、rank^2、rank^4、rank^8通信，
// 将当前已聚合的连续rank块直接写入对端HCCL Buffer。16 rank时通信启动次数由15次降至4次。
HcclResult ExecuteRecursiveDoubling(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    ThreadHandle mainThread = resCtx.threads[0];
    void *cclBuffer = resCtx.localBuffer.addr;
    const uint64_t outputSize = static_cast<uint64_t>(param.rankSize) * dataSize;

    CHK_RET(HcommLocalCopyOnThread(mainThread,
        ByteOffset(cclBuffer, static_cast<uint64_t>(param.myRank) * dataSize), param.inputPtr, dataSize));

    for (uint32_t groupSize = 1; groupSize < param.rankSize; groupSize <<= 1U) {
        const uint32_t partnerRank = param.myRank ^ groupSize;
        const ChannelInfo *channel = FindChannel(resCtx, partnerRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Recursive Doubling channel to rank[%u] was not acquired", partnerRank), HCCL_E_INTERNAL);

        const uint32_t sendGroupBase = (param.myRank / groupSize) * groupSize;
        const uint64_t sendOffset = static_cast<uint64_t>(sendGroupBase) * dataSize;
        const uint64_t sendSize = static_cast<uint64_t>(groupSize) * dataSize;
        // 4B和512KiB的四轮均使用融合Write+Notify；机内三轮使用Notify 0，跨Server轮使用Notify 1，避免连续阶段复用同一索引。
        const bool useFusedNotify = (dataSize <= SMALL_MESSAGE_THRESHOLD);
        const uint32_t notifyIndex = (groupSize < 8U)
            ? CHANNEL_NOTIFY_READY
            : CHANNEL_NOTIFY_CONSUMED;
        CHK_RET(QueueChannelWrite(mainThread, *channel,
            ByteOffset(channel->remoteCclMem.addr, sendOffset),
            ByteOffset(cclBuffer, sendOffset), sendSize, useFusedNotify, notifyIndex));
    }

    CHK_RET(HcommLocalCopyOnThread(mainThread, param.outputPtr, cclBuffer, outputSize));
    return HCCL_SUCCESS;
}


// 测试点6专用：512MiB规则大消息的“主控/通信解耦”双Bank流水。
// threads[0]只负责Bank准备、启动和完成等待；threads[1..15]各自固定负责一条Channel。
// 这样15条Peer链路完全对称，主Thread可在Worker通信当前Bank时准备下一Bank。
// 16-rank大消息专用：三路解耦双Bank Direct-Read。
// Main只负责控制和下一Bank准备；15个Worker各固定一条Channel；
// 独立OwnCopy Thread一次性完成input -> output[myRank]。
// 目标是同时消除“Main承担Peer通信”和“OwnCopy占用Bank准备关键路径”。
HcclResult ExecuteLargeTripleDecoupled(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes)
{
    const uint32_t peerCount = static_cast<uint32_t>(resCtx.channels.size());
    CHK_PRT_RET(param.rankSize != 16 || peerCount != 15,
        HCCL_ERROR("Large triple-decoupled path requires 16 ranks and 15 peers, rankSize[%u], peerCount[%u]",
            param.rankSize, peerCount), HCCL_E_PARA);
    CHK_PRT_RET(resCtx.threads.size() != peerCount + 2U,
        HCCL_ERROR("Large triple-decoupled thread count mismatch, threads[%lu], expected[%u]",
            resCtx.threads.size(), peerCount + 2U), HCCL_E_INTERNAL);

    const ThreadHandle mainThread = resCtx.threads[0];
    const ThreadHandle ownCopyThread = resCtx.threads[peerCount + 1U];
    void *ownOutput = ByteOffset(param.outputPtr, static_cast<uint64_t>(param.myRank) * totalBytes);
    const bool needLocalCopy = (param.inputPtr != ownOutput);

    const uint64_t bankCapacity = resCtx.chunkCapacity / 2U;
    CHK_PRT_RET(bankCapacity == 0,
        HCCL_ERROR("Large triple-decoupled bank capacity is zero"), HCCL_E_MEMORY);

    uint64_t currentOffset = 0;
    uint32_t currentBank = 0;
    uint64_t currentBytes = std::min(bankCapacity, totalBytes);

    // 首Bank必须先完成，随后网络即可启动。
    CHK_RET(HcommLocalCopyOnThread(
        mainThread, resCtx.localBuffer.addr, param.inputPtr, currentBytes));

    // OwnCopy只启动一次，并在全部网络轮次期间持续执行。
    if (needLocalCopy) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, ownCopyThread, THREAD_NOTIFY_START));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            ownCopyThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(
            ownCopyThread, ownOutput, param.inputPtr, totalBytes));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            ownCopyThread, mainThread, THREAD_NOTIFY_LOCAL_COPY_DONE));
    }

    while (currentOffset < totalBytes) {
        // 15个通信Worker构成两棵启动/完成树，Main不承担任何Peer通信。
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[1], THREAD_NOTIFY_START));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[2], THREAD_NOTIFY_START));

        const uint64_t currentBankOffset = static_cast<uint64_t>(currentBank) * bankCapacity;

        for (uint32_t workerIdx = 0; workerIdx < peerCount; ++workerIdx) {
            const uint32_t threadIdx = workerIdx + 1U;
            ThreadHandle peerThread = resCtx.threads[threadIdx];
            CHK_RET(HcommThreadNotifyWaitOnThread(
                peerThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));

            const uint32_t leftChild = workerIdx * 2U + 2U;
            const uint32_t rightChild = leftChild + 1U;
            if (leftChild < peerCount) {
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    peerThread, resCtx.threads[leftChild + 1U], THREAD_NOTIFY_START));
            }
            if (rightChild < peerCount) {
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    peerThread, resCtx.threads[rightChild + 1U], THREAD_NOTIFY_START));
            }

            const ChannelInfo &channel = resCtx.channels[workerIdx];
            void *dst = ByteOffset(param.outputPtr,
                static_cast<uint64_t>(channel.remoteRank) * totalBytes + currentOffset);
            void *remoteSrc = ByteOffset(channel.remoteCclMem.addr, currentBankOffset);
            const bool isLastChunk = (currentOffset + currentBytes >= totalBytes);
            CHK_RET(QueuePeerExchange(
                peerThread, channel, dst, remoteSrc, currentBytes, !isLastChunk));

            if (leftChild < peerCount) {
                CHK_RET(HcommThreadNotifyWaitOnThread(
                    peerThread, THREAD_NOTIFY_TREE_LEFT, CUSTOM_TIMEOUT));
            }
            if (rightChild < peerCount) {
                CHK_RET(HcommThreadNotifyWaitOnThread(
                    peerThread, THREAD_NOTIFY_TREE_RIGHT, CUSTOM_TIMEOUT));
            }

            if (workerIdx < 2U) {
                const uint32_t rootNotify = (workerIdx == 0U)
                    ? THREAD_NOTIFY_TREE_LEFT
                    : THREAD_NOTIFY_TREE_RIGHT;
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    peerThread, mainThread, rootNotify));
            } else {
                const uint32_t parentIdx = (workerIdx - 2U) / 2U;
                const uint32_t parentNotify = (workerIdx == parentIdx * 2U + 2U)
                    ? THREAD_NOTIFY_TREE_LEFT
                    : THREAD_NOTIFY_TREE_RIGHT;
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    peerThread, resCtx.threads[parentIdx + 1U], parentNotify));
            }
        }

        // 当前Bank通信期间，Main只准备下一Bank；OwnCopy由独立Thread并行推进。
        const uint64_t nextOffset = currentOffset + currentBytes;
        uint64_t nextBytes = 0;
        const uint32_t nextBank = currentBank ^ 1U;
        const bool hasNext = (nextOffset < totalBytes);
        if (hasNext) {
            nextBytes = std::min(bankCapacity, totalBytes - nextOffset);
            void *nextLocalBank = ByteOffset(
                resCtx.localBuffer.addr, static_cast<uint64_t>(nextBank) * bankCapacity);
            CHK_RET(HcommLocalCopyOnThread(
                mainThread, nextLocalBank, ByteOffset(param.inputPtr, nextOffset), nextBytes));
        }

        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, THREAD_NOTIFY_TREE_LEFT, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, THREAD_NOTIFY_TREE_RIGHT, CUSTOM_TIMEOUT));

        currentOffset = nextOffset;
        currentBytes = nextBytes;
        currentBank = nextBank;
    }

    if (needLocalCopy) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, THREAD_NOTIFY_LOCAL_COPY_DONE, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateResources(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Only FP32 is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank[%u/%u]", param.myRank, param.rankSize), HCCL_E_PARA);

    const uint32_t peerCount = (param.rankSize > 0) ? (param.rankSize - 1) : 0;
    CHK_PRT_RET(resCtx.channels.size() != peerCount,
        HCCL_ERROR("Channel count mismatch, channels[%lu], peers[%u]", resCtx.channels.size(), peerCount),
        HCCL_E_INTERNAL);
    const uint32_t expectedThreadNum = (peerCount == 0)
        ? 1
        : (peerCount + ((param.rankSize == 16) ? 2U : 1U));
    CHK_PRT_RET(resCtx.threads.size() != expectedThreadNum,
        HCCL_ERROR("Thread count mismatch, threads[%lu], expected[%u]", resCtx.threads.size(), expectedThreadNum),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.chunkCapacity == 0,
        HCCL_ERROR("Chunk capacity is zero"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_RET(ValidateResources(param, resCtx));
    CHK_PRT_RET(param.count > (std::numeric_limits<uint64_t>::max() / FP32_BYTES),
        HCCL_ERROR("Input byte size overflow"), HCCL_E_PARA);

    const uint64_t totalBytes = param.count * FP32_BYTES;
    if (totalBytes == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.rankSize > (std::numeric_limits<uint64_t>::max() / totalBytes),
        HCCL_ERROR("Output byte size overflow"), HCCL_E_PARA);

    ThreadHandle mainThread = resCtx.threads[0];

    // 小消息采用第9版已在正式评测中验证有效的4轮Recursive Doubling；
    // 大消息则完全保留V2的15路Direct-Read和完整HCCL Buffer分块路径。
    const uint64_t outputBytes = static_cast<uint64_t>(param.rankSize) * totalBytes;
    if (param.rankSize == 16 && totalBytes <= SMALL_MESSAGE_THRESHOLD
        && outputBytes <= resCtx.chunkCapacity) {
        return ExecuteRecursiveDoubling(param, resCtx, totalBytes);
    }

    // 16-rank大消息统一进入三路解耦双Bank路径，覆盖P6与P7；
    // 小消息Recursive Doubling保持不变。
    if (param.rankSize == 16 && totalBytes > SMALL_MESSAGE_THRESHOLD
        && resCtx.chunkCapacity >= 2U) {
        return ExecuteLargeTripleDecoupled(param, resCtx, totalBytes);
    }

    void *ownOutput = ByteOffset(param.outputPtr, static_cast<uint64_t>(param.myRank) * totalBytes);
    const bool needLocalCopy = (param.inputPtr != ownOutput);
    const uint32_t peerCount = static_cast<uint32_t>(resCtx.channels.size());

    if (peerCount == 0) {
        if (needLocalCopy) {
            CHK_RET(HcommLocalCopyOnThread(mainThread, ownOutput, param.inputPtr, totalBytes));
        }
        return HCCL_SUCCESS;
    }

    // 最后一条Thread专门完成本rank输入到输出对应rank槽位的拷贝。
    // 它在首分块进入中转区后启动，从而避免与首分块中转拷贝争抢本地带宽，并与网络传输重叠。
    const ThreadHandle localCopyThread = resCtx.threads[peerCount];
    bool localCopyQueued = false;

    uint64_t offset = 0;
    while (offset < totalBytes) {
        const uint64_t chunkBytes = std::min(resCtx.chunkCapacity, totalBytes - offset);
        CHK_RET(HcommLocalCopyOnThread(
            mainThread, resCtx.localBuffer.addr, ByteOffset(param.inputPtr, offset), chunkBytes));

        if (needLocalCopy && !localCopyQueued) {
            CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, localCopyThread, THREAD_NOTIFY_START));
            CHK_RET(HcommThreadNotifyWaitOnThread(localCopyThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));
            CHK_RET(HcommLocalCopyOnThread(localCopyThread, ownOutput, param.inputPtr, totalBytes));
            CHK_RET(HcommThreadNotifyRecordOnThread(
                localCopyThread, mainThread, THREAD_NOTIFY_LOCAL_COPY_DONE));
            localCopyQueued = true;
        }

        // 启动信号按与完成汇聚相同的堆式二叉树向下广播。
        // 主Thread只通知左右两个根Worker；每个Worker收到START后再放行自己的两个子节点。
        // 16 rank时主Thread的启动Record由14次降为2次，同时确保每个Worker每分块仅消费一次START。
        if (peerCount > 1) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                mainThread, resCtx.threads[1], THREAD_NOTIFY_START));
        }
        if (peerCount > 2) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                mainThread, resCtx.threads[2], THREAD_NOTIFY_START));
        }

        // Worker收到启动信号后先继续向下广播，再执行自己的固定Channel；
        // 通信完成后仍按同一棵树向上汇聚，主Thread只等待左右两棵子树。
        for (uint32_t peerIdx = 1; peerIdx < peerCount; ++peerIdx) {
            ThreadHandle peerThread = resCtx.threads[peerIdx];
            CHK_RET(HcommThreadNotifyWaitOnThread(peerThread, THREAD_NOTIFY_START, CUSTOM_TIMEOUT));

            const uint32_t leftChild = peerIdx * 2U + 1U;
            const uint32_t rightChild = leftChild + 1U;
            if (leftChild < peerCount) {
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    peerThread, resCtx.threads[leftChild], THREAD_NOTIFY_START));
            }
            if (rightChild < peerCount) {
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    peerThread, resCtx.threads[rightChild], THREAD_NOTIFY_START));
            }

            const ChannelInfo &channel = resCtx.channels[peerIdx];
            void *dst = ByteOffset(param.outputPtr,
                static_cast<uint64_t>(channel.remoteRank) * totalBytes + offset);
            CHK_RET(QueuePeerExchange(peerThread, channel, dst, chunkBytes));

            if (leftChild < peerCount) {
                CHK_RET(HcommThreadNotifyWaitOnThread(
                    peerThread, THREAD_NOTIFY_TREE_LEFT, CUSTOM_TIMEOUT));
            }
            if (rightChild < peerCount) {
                CHK_RET(HcommThreadNotifyWaitOnThread(
                    peerThread, THREAD_NOTIFY_TREE_RIGHT, CUSTOM_TIMEOUT));
            }

            const uint32_t parentIdx = (peerIdx - 1U) / 2U;
            const uint32_t parentNotify = (peerIdx == parentIdx * 2U + 1U)
                ? THREAD_NOTIFY_TREE_LEFT
                : THREAD_NOTIFY_TREE_RIGHT;
            CHK_RET(HcommThreadNotifyRecordOnThread(
                peerThread, resCtx.threads[parentIdx], parentNotify));
        }

        // 主Thread同时承担一个Peer，避免额外占用通信Thread。
        const ChannelInfo &mainChannel = resCtx.channels[0];
        void *mainDst = ByteOffset(param.outputPtr,
            static_cast<uint64_t>(mainChannel.remoteRank) * totalBytes + offset);
        CHK_RET(QueuePeerExchange(mainThread, mainChannel, mainDst, chunkBytes));

        // 根节点只需等待左右两棵完成子树，不再顺序等待14个Worker。
        if (peerCount > 1) {
            CHK_RET(HcommThreadNotifyWaitOnThread(
                mainThread, THREAD_NOTIFY_TREE_LEFT, CUSTOM_TIMEOUT));
        }
        if (peerCount > 2) {
            CHK_RET(HcommThreadNotifyWaitOnThread(
                mainThread, THREAD_NOTIFY_TREE_RIGHT, CUSTOM_TIMEOUT));
        }
        offset += chunkBytes;
    }

    if (needLocalCopy) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, THREAD_NOTIFY_LOCAL_COPY_DONE, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
