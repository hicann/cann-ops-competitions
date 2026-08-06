/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

// 北京科技大学南风的队伍：决赛 CCU Kernel 通信实现。

#include "ccu_kernel.h"
#include "../include/custom.h"

using namespace AscendC::ccu;

namespace ops_hccl {
namespace {
// CCU 资源编号和通知位编号，必须与 Host 侧注册协议保持一致。
constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t NOTIFY_SYNC = 0;
constexpr uint32_t POST_SYNC_ID = 3;
constexpr uint16_t LOCAL_COPY_MASK = 1U << 15;
}

CcuResult CcuKernel(CcuKernelArg arg)
{
    // 读取 Kernel 参数并计算本次分片在输入、输出缓冲区中的字节范围。
    LoadArg(Variable {}, 0);
    AllGatherKernelArg *kernelArg = static_cast<AllGatherKernelArg *>(arg);
    const uint32_t channelCount = kernelArg->channelCount;
    const uint64_t sliceBytes =
        kernelArg->sliceCount * kernelArg->dataTypeSize;
    const uint64_t sourceOffset =
        kernelArg->sliceOffset * kernelArg->dataTypeSize;
    const uint64_t destinationOffset =
        (kernelArg->myRank * kernelArg->sendCount +
            kernelArg->sliceOffset) * kernelArg->dataTypeSize;
    const bool smallPull =
        kernelArg->kernelMode == ALLGATHER_KERNEL_SMALL_PULL;

    // 小消息拉取路径从本地输入读取；直写路径直接使用全局输出作为本地源。
    Variable localOutput;
    Variable localOutputToken;
    if (smallPull) {
        localOutput = reinterpret_cast<uint64_t>(kernelArg->sendBuf);
        localOutputToken = kernelArg->inputToken;
    } else {
        localOutput = reinterpret_cast<uint64_t>(kernelArg->recvBuf);
        localOutputToken = kernelArg->outputToken;
    }
    // 交换每条通道对应的远端输出地址和内存令牌。
    Variable remoteOutput[MAX_RANK_SIZE];
    Variable remoteToken[MAX_RANK_SIZE];
    for (uint32_t channel = 0; channel < channelCount; ++channel) {
        remoteOutput[channel] = GetResByChannel<Variable>(
            kernelArg->channels[channel], OUTPUT_XN_ID);
        remoteToken[channel] = GetResByChannel<Variable>(
            kernelArg->channels[channel], TOKEN_XN_ID);
        WriteVariableWithNotify(kernelArg->channels[channel], localOutput,
            OUTPUT_XN_ID, NOTIFY_SYNC, 1U << OUTPUT_XN_ID);
        WriteVariableWithNotify(kernelArg->channels[channel], localOutputToken,
            TOKEN_XN_ID, NOTIFY_SYNC, 1U << TOKEN_XN_ID);
    }
    // 等待所有通道完成地址和令牌交换后再开始数据传输。
    const uint16_t exchangeMask =
        (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channel = 0; channel < channelCount; ++channel) {
        NotifyWait(kernelArg->channels[channel], NOTIFY_SYNC, exchangeMask);
    }

    if (smallPull) {
        // 小消息路径：从每个远端输入拉取数据，同时复制本 Rank 的本地分片。
        Variable sourceOffsetVar;
        sourceOffsetVar = sourceOffset;
        Variable bytes;
        bytes = sliceBytes;
        Event event;
        uint16_t waitMask = 0;
        for (uint32_t channel = 0; channel < channelCount; ++channel) {
            // 远端 Rank 的目标位置由通道映射表计算得到。
            const uint64_t remoteDestinationOffset =
                (static_cast<uint64_t>(kernelArg->channelIndexToRank[channel]) *
                    kernelArg->sendCount + kernelArg->sliceOffset) *
                kernelArg->dataTypeSize;
            LocalAddr destination;
            destination.addr = reinterpret_cast<uint64_t>(kernelArg->recvBuf) +
                remoteDestinationOffset;
            destination.token = kernelArg->outputToken;
            RemoteAddr source;
            source.addr = remoteOutput[channel];
            source.addr += sourceOffsetVar;
            source.token = remoteToken[channel];
            const uint16_t mask = 1U << channel;
            Read(kernelArg->channels[channel], destination, source,
                bytes, event, mask);
            waitMask |= mask;
        }
        // 远端拉取与本地复制共用一个事件，统一等待后再返回。
        LocalAddr localDestination;
        localDestination.addr = reinterpret_cast<uint64_t>(kernelArg->recvBuf) +
            destinationOffset;
        localDestination.token = kernelArg->outputToken;
        LocalAddr localSource;
        localSource.addr = reinterpret_cast<uint64_t>(kernelArg->sendBuf) +
            sourceOffset;
        localSource.token = kernelArg->inputToken;
        LocalCopy(localDestination, localSource, bytes, event, LOCAL_COPY_MASK);
        waitMask |= LOCAL_COPY_MASK;
        EventWait(event, waitMask);
        return CCU_SUCCESS;
    }

    // 直写路径：以本地输入为源，将当前分片写入每个远端输出。
    LocalAddr source;
    source.addr = reinterpret_cast<uint64_t>(kernelArg->sendBuf) +
        sourceOffset;
    source.token = kernelArg->inputToken;
    Variable bytes;
    bytes = sliceBytes;
    Variable destinationOffsetVar;
    destinationOffsetVar = destinationOffset;
    Event event;
    uint16_t waitMask = 0;
    for (uint32_t channel = 0; channel < channelCount; ++channel) {
        // 每条通道写入相同的源分片，但目标偏移按本地 Rank 计算。
        RemoteAddr destination;
        destination.addr = remoteOutput[channel];
        destination.addr += destinationOffsetVar;
        destination.token = remoteToken[channel];
        const uint16_t mask = 1U << channel;
        Write(kernelArg->channels[channel], destination, source,
            bytes, event, mask);
        waitMask |= mask;
    }
    if (kernelArg->copyLocalOutput != 0) {
        // 本地 Rank 不经过远端通道，单独复制到输出缓冲区对应位置。
        LocalAddr localDestination;
        localDestination.addr =
            reinterpret_cast<uint64_t>(kernelArg->recvBuf) +
            destinationOffset;
        localDestination.token = kernelArg->outputToken;
        LocalCopy(localDestination, source, bytes, event, LOCAL_COPY_MASK);
        waitMask |= LOCAL_COPY_MASK;
    }
    EventWait(event, waitMask);
    // 通知远端写入完成，并等待远端确认，保证下一轮不会覆盖未完成的数据。
    for (uint32_t channel = 0; channel < channelCount; ++channel) {
        NotifyRecord(kernelArg->channels[channel],
            NOTIFY_SYNC, 1U << POST_SYNC_ID);
    }
    for (uint32_t channel = 0; channel < channelCount; ++channel) {
        NotifyWait(kernelArg->channels[channel],
            NOTIFY_SYNC, 1U << POST_SYNC_ID);
    }
    return CCU_SUCCESS;
}

CcuResult G2ForwardKernel(CcuKernelArg arg)
{
    // Gateway 转发阶段复用本地输出作为源，将指定远端 Rank 的分片转发出去。
    LoadArg(Variable {}, 0);
    G2ForwardKernelArg *kernelArg =
        static_cast<G2ForwardKernelArg *>(arg);
    Variable localOutput;
    localOutput = reinterpret_cast<uint64_t>(kernelArg->recvBuf);
    Variable localOutputToken;
    localOutputToken = kernelArg->outputToken;
    // 先交换本地输出地址和令牌，建立远端写入目标。
    Variable remoteOutput[MAX_RANK_SIZE];
    Variable remoteToken[MAX_RANK_SIZE];
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        remoteOutput[channel] = GetResByChannel<Variable>(
            kernelArg->channels[channel], OUTPUT_XN_ID);
        remoteToken[channel] = GetResByChannel<Variable>(
            kernelArg->channels[channel], TOKEN_XN_ID);
        WriteVariableWithNotify(kernelArg->channels[channel], localOutput,
            OUTPUT_XN_ID, NOTIFY_SYNC, 1U << OUTPUT_XN_ID);
        WriteVariableWithNotify(kernelArg->channels[channel], localOutputToken,
            TOKEN_XN_ID, NOTIFY_SYNC, 1U << TOKEN_XN_ID);
    }
    // 地址交换完成后才能安全发起转发写入。
    const uint16_t exchangeMask =
        (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        NotifyWait(kernelArg->channels[channel], NOTIFY_SYNC, exchangeMask);
    }

    // 每条转发记录描述目标通道、源 Rank 和该 Rank 的数据分片。
    Event forwardEvent;
    uint16_t forwardWaitMask = 0;
    for (uint32_t transfer = 0;
        transfer < kernelArg->forwardTransferCount; ++transfer) {
        const uint32_t channelIndex =
            kernelArg->forwardChannelIndices[transfer];
        Variable forwardChunkOffset;
        forwardChunkOffset =
            kernelArg->forwardSliceOffsets[transfer] *
            kernelArg->dataTypeSize;
        Variable forwardBytes;
        forwardBytes =
            kernelArg->forwardSliceCounts[transfer] *
            kernelArg->dataTypeSize;
        Variable offset;
        offset = static_cast<uint64_t>(
            kernelArg->forwardSourceRanks[transfer]) *
            kernelArg->sendCount * kernelArg->dataTypeSize;
        offset += forwardChunkOffset;
        // 根据源 Rank 和分片偏移计算本地输出源地址。
        LocalAddr source;
        source.addr = localOutput;
        source.addr += offset;
        source.token = localOutputToken;
        RemoteAddr destination;
        destination.addr = remoteOutput[channelIndex];
        destination.addr += offset;
        destination.token = remoteToken[channelIndex];
        const uint16_t mask = 1U << transfer;
        Write(kernelArg->channels[channelIndex], destination, source,
            forwardBytes, forwardEvent, mask);
        forwardWaitMask |= mask;
    }
    // 等待所有转发写入完成，再与各本地通道进行收尾同步。
    EventWait(forwardEvent, forwardWaitMask);
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        NotifyRecord(kernelArg->channels[channel],
            NOTIFY_SYNC, 1U << POST_SYNC_ID);
    }
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        NotifyWait(kernelArg->channels[channel],
            NOTIFY_SYNC, 1U << POST_SYNC_ID);
    }
    return CCU_SUCCESS;
}
}
