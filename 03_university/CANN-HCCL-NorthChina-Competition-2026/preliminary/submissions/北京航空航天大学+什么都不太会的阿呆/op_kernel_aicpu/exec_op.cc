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

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace {
constexpr uint64_t RECURSIVE_DOUBLING_THRESHOLD = 8 * 1024 * 1024;
constexpr uint64_t DIRECT_TRANSFER_LIMIT = 256 * 1024 * 1024;
constexpr uint64_t DATA_ALIGN = 4096;
constexpr uint32_t MAX_SERVER_RANK_SIZE = 8;
constexpr uint32_t PARALLEL_RATIO_DENOM = 8;
constexpr uint32_t TOPOLOGY_INTER_BW_FACTOR = 8;
constexpr uint32_t THREAD_START_NOTIFY_IDX = 0;
constexpr uint32_t MESH_CLOS_READY_NOTIFY_IDX = 1;
constexpr uint32_t INTER_MESH_READY_NOTIFY_BASE = 1;
constexpr uint32_t MAIN_DONE_NOTIFY_BASE = 1;
constexpr uint32_t FINAL_ACK_NOTIFY_IDX = 2;
constexpr uint32_t FINAL_DATA_NOTIFY_IDX = 3;

const ChannelInfo &GetIntraChannel(const AlgResourceCtx &resCtx, uint32_t localRankIndex, uint32_t remoteLocalRankIndex)
{
    uint32_t channelIndex = remoteLocalRankIndex < localRankIndex ? remoteLocalRankIndex : remoteLocalRankIndex - 1;
    return resCtx.intraChannels[channelIndex];
}

HcclResult RunRecursiveDoubling(const AlgResourceCtx &resCtx, ThreadHandle thread, const char *input, char *localBuffer,
    uint64_t blockSize, uint32_t localRankIndex)
{
    char *localSlot = localBuffer + static_cast<uint64_t>(localRankIndex) * blockSize;
    CHK_RET(HcommLocalCopyOnThread(thread, localSlot, input, blockSize));

    for (uint32_t distance = 1; distance < resCtx.serverRankSize; distance <<= 1) {
        uint32_t partnerLocalRankIndex = localRankIndex ^ distance;
        uint32_t groupBaseRankIndex = localRankIndex & ~(distance - 1);
        uint64_t transferSize = static_cast<uint64_t>(distance) * blockSize;
        const ChannelInfo &channel = GetIntraChannel(resCtx, localRankIndex, partnerLocalRankIndex);
        char *remoteDst
            = static_cast<char *>(channel.remoteCclMem.addr) + static_cast<uint64_t>(groupBaseRankIndex) * blockSize;
        const char *localSrc = localBuffer + static_cast<uint64_t>(groupBaseRankIndex) * blockSize;
        CHK_RET(HcommWriteWithNotifyOnThread(
            thread, channel.handle, remoteDst, localSrc, transferSize, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult RunSmallMessage(const AlgResourceCtx &resCtx, ThreadHandle thread, const char *input, char *output,
    char *localBuffer, uint64_t dataSize, uint32_t localRankIndex, uint32_t serverBaseRank, uint32_t peerServerBaseRank)
{
    CHK_RET(RunRecursiveDoubling(resCtx, thread, input, localBuffer, dataSize, localRankIndex));

    uint64_t aggregateSize = dataSize * resCtx.serverRankSize;
    char *remoteStage = static_cast<char *>(resCtx.interChannel.remoteCclMem.addr) + aggregateSize;
    CHK_RET(HcommWriteWithNotifyOnThread(
        thread, resCtx.interChannel.handle, remoteStage, localBuffer, aggregateSize, NOTIFY_IDX_DATA_SIGNAL));

    char *localOutput = output + static_cast<uint64_t>(serverBaseRank) * dataSize;
    CHK_RET(HcommLocalCopyOnThread(thread, localOutput, localBuffer, aggregateSize));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, resCtx.interChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

    char *remoteOutput = output + static_cast<uint64_t>(peerServerBaseRank) * dataSize;
    CHK_RET(HcommLocalCopyOnThread(thread, remoteOutput, localBuffer + aggregateSize, aggregateSize));
    return HCCL_SUCCESS;
}

HcclResult ExchangeReadIntermediate(ThreadHandle thread, const ChannelInfo &channel, void *const *dsts,
    const void *const *srcs, const uint64_t *sizes, uint32_t count)
{
    for (uint32_t index = 0; index < count; ++index) {
        CHK_RET(HcommReadOnThread(thread, channel.handle, dsts[index], srcs[index], sizes[index]));
    }
    return HCCL_SUCCESS;
}

HcclResult ExchangeReadFinal(ThreadHandle thread, const ChannelInfo &channel, void *const *dsts,
    const void *const *srcs, const uint64_t *sizes, uint32_t count)
{
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, FINAL_ACK_NOTIFY_IDX));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, FINAL_ACK_NOTIFY_IDX, CUSTOM_TIMEOUT));
    for (uint32_t index = 0; index < count; ++index) {
        CHK_RET(HcommReadOnThread(thread, channel.handle, dsts[index], srcs[index], sizes[index]));
    }
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult ExchangeWriteFinal(ThreadHandle thread, const ChannelInfo &channel, void *const *dsts,
    const void *const *srcs, const uint64_t *sizes, uint32_t count)
{
    if (count == 0) {
        return HCCL_E_PARA;
    }
    for (uint32_t index = 0; index < count; ++index) {
        if (index + 1 == count) {
            CHK_RET(HcommWriteWithNotifyOnThread(
                thread, channel.handle, dsts[index], srcs[index], sizes[index], NOTIFY_IDX_DATA_SIGNAL));
        } else {
            CHK_RET(HcommWriteOnThread(thread, channel.handle, dsts[index], srcs[index], sizes[index]));
        }
    }
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

uint32_t GetMeshRatioUnits(const AlgResourceCtx &resCtx)
{
    uint64_t intraBandwidth = std::max<uint32_t>(resCtx.intraChannels[0].bandwidthCoeff, 1);
    for (uint32_t channelIndex = 1; channelIndex < resCtx.intraChannels.size(); ++channelIndex) {
        intraBandwidth = std::min<uint64_t>(
            intraBandwidth, std::max<uint32_t>(resCtx.intraChannels[channelIndex].bandwidthCoeff, 1));
    }
    uint64_t interBandwidth = std::max<uint32_t>(resCtx.interChannel.bandwidthCoeff, 1);
    if (resCtx.interChannel.protocol != CommProtocol::COMM_PROTOCOL_PCIE) {
        interBandwidth = std::max(interBandwidth, intraBandwidth * TOPOLOGY_INTER_BW_FACTOR);
    }

    uint32_t bestUnits = 1;
    uint64_t bestCost = UINT64_MAX;
    for (uint32_t meshUnits = 1; meshUnits < PARALLEL_RATIO_DENOM; ++meshUnits) {
        uint64_t closUnits = PARALLEL_RATIO_DENOM - meshUnits;
        uint64_t firstPhaseCost = std::max(
            static_cast<uint64_t>(meshUnits) * interBandwidth, closUnits * intraBandwidth);
        uint64_t secondPhaseCost = std::max(static_cast<uint64_t>(resCtx.serverRankSize) * meshUnits * intraBandwidth,
            2 * closUnits * interBandwidth);
        uint64_t totalCost = firstPhaseCost + secondPhaseCost;
        if (totalCost < bestCost) {
            bestCost = totalCost;
            bestUnits = meshUnits;
        }
    }
    return bestUnits;
}

HcclResult StartWorkers(const AlgResourceCtx &resCtx)
{
    ThreadHandle mainThread = resCtx.threads[0];
    for (uint32_t workerIndex = 1; workerIndex < resCtx.threads.size(); ++workerIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[workerIndex], THREAD_START_NOTIFY_IDX));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitWorkers(const AlgResourceCtx &resCtx)
{
    ThreadHandle mainThread = resCtx.threads[0];
    for (uint32_t channelIndex = 0; channelIndex < resCtx.intraChannels.size(); ++channelIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, MAIN_DONE_NOTIFY_BASE + channelIndex, CUSTOM_TIMEOUT));
    }
    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, resCtx.serverRankSize, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult WaitMeshWorkerStart(const AlgResourceCtx &resCtx, uint32_t channelIndex)
{
    if (channelIndex == 0) {
        return HCCL_SUCCESS;
    }
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[channelIndex + 1], THREAD_START_NOTIFY_IDX, CUSTOM_TIMEOUT));
}

HcclResult SignalMeshReady(const AlgResourceCtx &resCtx, uint32_t channelIndex)
{
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[channelIndex + 1],
        resCtx.threads[resCtx.serverRankSize], INTER_MESH_READY_NOTIFY_BASE + channelIndex));
}

HcclResult WaitMeshReady(const AlgResourceCtx &resCtx)
{
    ThreadHandle thread = resCtx.threads[resCtx.serverRankSize];
    for (uint32_t channelIndex = 0; channelIndex < resCtx.intraChannels.size(); ++channelIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            thread, INTER_MESH_READY_NOTIFY_BASE + channelIndex, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitMeshReady(const AlgResourceCtx &resCtx, uint32_t channelIndex)
{
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[resCtx.serverRankSize],
        INTER_MESH_READY_NOTIFY_BASE + channelIndex, CUSTOM_TIMEOUT));
}

HcclResult SignalClosReady(const AlgResourceCtx &resCtx)
{
    ThreadHandle thread = resCtx.threads[resCtx.serverRankSize];
    for (uint32_t channelIndex = 0; channelIndex < resCtx.intraChannels.size(); ++channelIndex) {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            thread, resCtx.threads[channelIndex + 1], MESH_CLOS_READY_NOTIFY_IDX));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitClosReady(const AlgResourceCtx &resCtx, uint32_t channelIndex)
{
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[channelIndex + 1], MESH_CLOS_READY_NOTIFY_IDX, CUSTOM_TIMEOUT));
}

HcclResult FinishMeshWorker(const AlgResourceCtx &resCtx, uint32_t channelIndex)
{
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[channelIndex + 1],
        resCtx.threads[0], MAIN_DONE_NOTIFY_BASE + channelIndex));
}

HcclResult FinishInterWorker(const AlgResourceCtx &resCtx)
{
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[resCtx.serverRankSize], resCtx.threads[0], resCtx.serverRankSize));
}

HcclResult RunMeshPipeline(const AlgResourceCtx &resCtx, const char *input, char *output, uint64_t dataSize,
    uint64_t dataOffset, uint64_t meshSize, uint64_t closSize, uint32_t localRankIndex, uint32_t serverBaseRank)
{
    uint32_t rankSize = resCtx.serverRankSize;
    uint32_t peerServerBaseRank = serverBaseRank == 0 ? rankSize : 0;
    uint32_t localGlobalRank = serverBaseRank + localRankIndex;
    uint32_t peerGlobalRank = peerServerBaseRank + localRankIndex;
    ThreadHandle leader = resCtx.threads[1];
    CHK_RET(HcommThreadNotifyWaitOnThread(leader, THREAD_START_NOTIFY_IDX, CUSTOM_TIMEOUT));

    uint64_t closDataOffset = dataOffset + meshSize;
    for (uint32_t channelIndex = 0; channelIndex < resCtx.intraChannels.size(); ++channelIndex) {
        const ChannelInfo &channel = resCtx.intraChannels[channelIndex];
        ThreadHandle worker = resCtx.threads[channelIndex + 1];
        CHK_RET(WaitMeshWorkerStart(resCtx, channelIndex));
        void *meshDsts[1] = {output + static_cast<uint64_t>(channel.remoteRank) * dataSize + dataOffset};
        const void *meshSrcs[1] = {static_cast<const char *>(channel.remoteInput.addr) + dataOffset};
        uint64_t meshSizes[1] = {meshSize};
        CHK_RET(ExchangeReadIntermediate(worker, channel, meshDsts, meshSrcs, meshSizes, 1));
        CHK_RET(SignalMeshReady(resCtx, channelIndex));
        CHK_RET(WaitClosReady(resCtx, channelIndex));

        void *dsts[2] = {};
        const void *srcs[2] = {};
        uint64_t sizes[2] = {closSize, closSize};
        dsts[0] = static_cast<char *>(channel.remoteOutput.addr)
            + static_cast<uint64_t>(localGlobalRank) * dataSize + closDataOffset;
        srcs[0] = input + closDataOffset;
        dsts[1] = static_cast<char *>(channel.remoteOutput.addr)
            + static_cast<uint64_t>(peerGlobalRank) * dataSize + closDataOffset;
        srcs[1] = output + static_cast<uint64_t>(peerGlobalRank) * dataSize + closDataOffset;
        CHK_RET(ExchangeWriteFinal(worker, channel, dsts, srcs, sizes, 2));
        CHK_RET(FinishMeshWorker(resCtx, channelIndex));
    }
    return HCCL_SUCCESS;
}

HcclResult RunInterPipeline(const AlgResourceCtx &resCtx, const char *input, char *output, uint64_t dataSize,
    uint64_t dataOffset, uint64_t meshSize, uint64_t closSize, uint32_t localRankIndex, uint32_t serverBaseRank)
{
    uint32_t rankSize = resCtx.serverRankSize;
    uint32_t peerServerBaseRank = serverBaseRank == 0 ? rankSize : 0;
    ThreadHandle thread = resCtx.threads[rankSize];
    CHK_RET(HcommThreadNotifyWaitOnThread(thread, THREAD_START_NOTIFY_IDX, CUSTOM_TIMEOUT));

    uint64_t closDataOffset = dataOffset + meshSize;
    uint32_t localGlobalRank = serverBaseRank + localRankIndex;
    uint32_t peerGlobalRank = peerServerBaseRank + localRankIndex;
    uint64_t closSizes[1] = {closSize};
    if (resCtx.interChannel.protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
        void *readDsts[1] = {output + static_cast<uint64_t>(peerGlobalRank) * dataSize + closDataOffset};
        const void *readSrcs[1]
            = {static_cast<const char *>(resCtx.interChannel.remoteInput.addr) + closDataOffset};
        CHK_RET(ExchangeReadIntermediate(thread, resCtx.interChannel, readDsts, readSrcs, closSizes, 1));
    } else {
        void *writeDst = static_cast<char *>(resCtx.interChannel.remoteOutput.addr)
            + static_cast<uint64_t>(localGlobalRank) * dataSize + closDataOffset;
        CHK_RET(HcommWriteWithNotifyOnThread(thread, resCtx.interChannel.handle, writeDst,
            input + closDataOffset, closSize, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, resCtx.interChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    }

    CHK_RET(SignalClosReady(resCtx));
    if (resCtx.interChannel.protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
        CHK_RET(WaitMeshReady(resCtx));
        uint64_t sizes[MAX_SERVER_RANK_SIZE] = {};
        void *readDsts[MAX_SERVER_RANK_SIZE] = {};
        const void *readSrcs[MAX_SERVER_RANK_SIZE] = {};
        for (uint32_t rankIndex = 0; rankIndex < rankSize; ++rankIndex) {
            uint32_t globalRank = peerServerBaseRank + rankIndex;
            readDsts[rankIndex] = output + static_cast<uint64_t>(globalRank) * dataSize + dataOffset;
            readSrcs[rankIndex] = rankIndex == localRankIndex
                ? static_cast<const char *>(resCtx.interChannel.remoteInput.addr) + dataOffset
                : static_cast<const char *>(resCtx.interChannel.remoteOutput.addr)
                    + static_cast<uint64_t>(globalRank) * dataSize + dataOffset;
            sizes[rankIndex] = meshSize;
        }
        CHK_RET(ExchangeReadFinal(thread, resCtx.interChannel, readDsts, readSrcs, sizes, rankSize));
    } else {
        void *writeDst = static_cast<char *>(resCtx.interChannel.remoteOutput.addr)
            + static_cast<uint64_t>(localGlobalRank) * dataSize + dataOffset;
        CHK_RET(HcommWriteOnThread(thread, resCtx.interChannel.handle, writeDst, input + dataOffset, meshSize));
        for (uint32_t channelIndex = 0; channelIndex < resCtx.intraChannels.size(); ++channelIndex) {
            CHK_RET(WaitMeshReady(resCtx, channelIndex));
            uint32_t globalRank = resCtx.intraChannels[channelIndex].remoteRank;
            writeDst = static_cast<char *>(resCtx.interChannel.remoteOutput.addr)
                + static_cast<uint64_t>(globalRank) * dataSize + dataOffset;
            const void *writeSrc = output + static_cast<uint64_t>(globalRank) * dataSize + dataOffset;
            if (channelIndex + 1 == resCtx.intraChannels.size()) {
                CHK_RET(HcommWriteWithNotifyOnThread(thread, resCtx.interChannel.handle, writeDst, writeSrc,
                    meshSize, FINAL_DATA_NOTIFY_IDX));
            } else {
                CHK_RET(HcommWriteOnThread(thread, resCtx.interChannel.handle, writeDst, writeSrc, meshSize));
            }
        }
        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, resCtx.interChannel.handle, FINAL_DATA_NOTIFY_IDX, CUSTOM_TIMEOUT));
    }
    CHK_RET(FinishInterWorker(resCtx));
    return HCCL_SUCCESS;
}

HcclResult RunLargeParallel(const AlgResourceCtx &resCtx, const char *input, char *output,
    uint64_t dataSize, uint32_t localRankIndex, uint32_t serverBaseRank)
{
    uint32_t meshRatioUnits = GetMeshRatioUnits(resCtx);
    uint64_t loopTimes = (dataSize + DIRECT_TRANSFER_LIMIT - 1) / DIRECT_TRANSFER_LIMIT;
    uint64_t processedSize = 0;
    for (uint64_t loopIndex = 0; loopIndex < loopTimes; ++loopIndex) {
        uint64_t currentSize = std::min(dataSize - processedSize, DIRECT_TRANSFER_LIMIT);
        if (currentSize == 0) {
            return HCCL_E_MEMORY;
        }

        uint64_t meshSize = currentSize * meshRatioUnits / PARALLEL_RATIO_DENOM;
        meshSize = meshSize / sizeof(float) * sizeof(float);
        if (loopIndex + 1 < loopTimes && meshSize >= DATA_ALIGN) {
            meshSize = meshSize / DATA_ALIGN * DATA_ALIGN;
        }
        uint64_t closSize = currentSize - meshSize;
        if (meshSize == 0 || closSize == 0) {
            return HCCL_E_MEMORY;
        }

        CHK_RET(StartWorkers(resCtx));
        char *localOutput = output
            + static_cast<uint64_t>(serverBaseRank + localRankIndex) * dataSize + processedSize;
        CHK_RET(HcommLocalCopyOnThread(resCtx.threads[0], localOutput, input + processedSize, currentSize));
        CHK_RET(RunMeshPipeline(resCtx, input, output, dataSize, processedSize, meshSize, closSize,
            localRankIndex, serverBaseRank));
        CHK_RET(RunInterPipeline(resCtx, input, output, dataSize, processedSize, meshSize, closSize,
            localRankIndex, serverBaseRank));
        CHK_RET(WaitWorkers(resCtx));
        processedSize += currentSize;
    }
    return HCCL_SUCCESS;
}
}

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(resCtx.localBuffer.addr);

    if (param.dataType != HCCL_DATA_TYPE_FP32) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (resCtx.serverRankSize < 2 || param.rankSize != resCtx.serverRankSize * 2 || param.myRank >= param.rankSize
        || resCtx.serverRankSize > MAX_SERVER_RANK_SIZE || resCtx.threads.size() != resCtx.serverRankSize + 1
        || resCtx.intraChannels.size() + 1 != resCtx.serverRankSize || resCtx.chunkSize == 0) {
        return HCCL_E_PARA;
    }

    if (param.count > UINT64_MAX / sizeof(float)) {
        return HCCL_E_PARA;
    }
    uint64_t dataSize = param.count * sizeof(float);
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }
    auto *input = static_cast<const char *>(param.inputPtr);
    auto *output = static_cast<char *>(param.outputPtr);
    auto *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);
    uint32_t localRankIndex = param.myRank % resCtx.serverRankSize;
    uint32_t serverBaseRank = param.myRank - localRankIndex;
    uint32_t peerServerBaseRank = serverBaseRank == 0 ? resCtx.serverRankSize : 0;
    uint32_t pairedRank = peerServerBaseRank + localRankIndex;
    uint64_t aggregateCapacity = resCtx.chunkSize * resCtx.serverRankSize;
    if (aggregateCapacity > resCtx.localBuffer.size || resCtx.interChannel.remoteRank != pairedRank) {
        return HCCL_E_MEMORY;
    }

    if (dataSize <= RECURSIVE_DOUBLING_THRESHOLD && dataSize <= resCtx.chunkSize
        && dataSize * resCtx.serverRankSize * 2 <= aggregateCapacity) {
        return RunSmallMessage(resCtx, resCtx.threads[0], input, output, localBuffer, dataSize, localRankIndex,
            serverBaseRank, peerServerBaseRank);
    }
    if (dataSize > UINT64_MAX / param.rankSize) {
        return HCCL_E_PARA;
    }
    uint64_t outputSize = dataSize * param.rankSize;
    if (resCtx.registeredInput.addr != param.inputPtr || resCtx.registeredInput.size < dataSize
        || resCtx.registeredOutput.addr != param.outputPtr || resCtx.registeredOutput.size < outputSize
        || resCtx.interChannel.remoteInput.addr == nullptr || resCtx.interChannel.remoteInput.size < dataSize
        || resCtx.interChannel.remoteOutput.addr == nullptr || resCtx.interChannel.remoteOutput.size < outputSize) {
        return HCCL_E_MEMORY;
    }
    for (const ChannelInfo &channel : resCtx.intraChannels) {
        if (channel.remoteInput.addr == nullptr || channel.remoteInput.size < dataSize
            || channel.remoteOutput.addr == nullptr || channel.remoteOutput.size < outputSize) {
            return HCCL_E_MEMORY;
        }
    }
    return RunLargeParallel(resCtx, input, output, dataSize, localRankIndex, serverBaseRank);
}
}
