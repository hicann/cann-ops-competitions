/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom.h"
#include "log.h"
#include "exec_op.h"
#include <limits>

namespace ops_hccl {
namespace {
HcclResult CopyRootOutput(const OpParam &param, ThreadHandle thread, uint64_t bytes)
{
    const auto *source = static_cast<const uint8_t *>(param.inputPtr) + static_cast<uint64_t>(param.root) * bytes;
    auto *destination = static_cast<uint8_t *>(param.outputPtr);
    if (source == destination) {
        return HCCL_SUCCESS;
    }
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t length = std::min(SCATTER_MAX_COPY_BYTES, bytes - offset);
        CHK_RET(HcommLocalCopyOnThread(thread, destination + offset, source + offset, length));
        offset += length;
    }
    return HCCL_SUCCESS;
}

bool UseSmallScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t bytes)
{
    // Every rank sees the same minimum, including heterogeneous CCL capacities.
    uint64_t capacity = std::min(SCATTER_SMALL_INPUT_BYTES, resCtx.localBuffer.size);
    for (const auto &channel : resCtx.channels) {
        capacity = std::min(capacity, channel.remoteCclMem.size);
    }
    return bytes <= capacity / param.rankSize;
}

const ChannelInfo *ChannelTo(const AlgResourceCtx &resCtx, uint32_t peer)
{
    const auto found = std::find_if(resCtx.channels.begin(), resCtx.channels.end(), [peer](const ChannelInfo &channel) {
        return channel.remoteRank == peer;
    });
    return found == resCtx.channels.end() ? nullptr : &*found;
}

std::vector<std::vector<uint32_t>> SmallScatterGroups(const OpParam &param, const AlgResourceCtx &resCtx)
{
    std::vector<std::vector<uint32_t>> groups;
    const auto append = [&](const std::vector<uint32_t> &members) {
        for (size_t begin = 0; begin < members.size(); begin += 4) {
            const size_t end = std::min(members.size(), begin + 4);
            groups.emplace_back(members.begin() + begin, members.begin() + end);
        }
    };
    if (param.rankSize == 16 && resCtx.serverRanks.size() == 8) {
        const bool rootIsLocal = std::binary_search(resCtx.serverRanks.begin(), resCtx.serverRanks.end(), param.root);
        std::vector<uint32_t> rootServer;
        std::vector<uint32_t> otherServer;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank == param.root) {
                continue;
            }
            const bool local = std::binary_search(resCtx.serverRanks.begin(), resCtx.serverRanks.end(), rank);
            (local == rootIsLocal ? rootServer : otherServer).push_back(rank);
        }
        // Start the two cross-server notifications first. Each representative
        // fans out only inside its own server; group sizes are 4, 4, 4, 3.
        append(otherServer);
        append(rootServer);
    } else {
        std::vector<uint32_t> members;
        for (uint32_t virtualRank = 1; virtualRank < param.rankSize; ++virtualRank) {
            members.push_back(static_cast<uint32_t>((static_cast<uint64_t>(param.root) + virtualRank) % param.rankSize));
        }
        append(members);
    }
    return groups;
}

HcclResult SmallScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t bytes)
{
    const ThreadHandle thread = resCtx.aicpuThread;
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    CHK_PRT_RET(resCtx.channels.size() != param.rankSize - 1,
        HCCL_ERROR("Incomplete Scatter channels"), HCCL_E_INTERNAL);
    const auto groups = SmallScatterGroups(param, resCtx);
    const ChannelInfo *parent = nullptr;
    std::vector<const ChannelInfo *> children;
    if (param.myRank == param.root) {
        for (const auto &group : groups) {
            children.push_back(ChannelTo(resCtx, group.front()));
        }
        CHK_RET(HcommLocalCopyOnThread(thread, resCtx.localBuffer.addr, param.inputPtr, bytes * param.rankSize));
    } else {
        for (const auto &group : groups) {
            if (std::find(group.begin(), group.end(), param.myRank) == group.end()) {
                continue;
            }
            const bool representative = param.myRank == group.front();
            parent = ChannelTo(resCtx, representative ? param.root : group.front());
            if (representative) {
                for (size_t i = 1; i < group.size(); ++i) {
                    children.push_back(ChannelTo(resCtx, group[i]));
                }
            }
            break;
        }
        CHK_PTR_NULL(parent);
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, parent->handle, SCATTER_NOTIFY_SMALL_DATA, CUSTOM_TIMEOUT));
    }
    // Forward readiness before fetching this rank's slice, so subtree readers
    // can progress concurrently. All data still comes directly from root CCL.
    for (const auto *child : children) {
        CHK_PTR_NULL(child);
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, child->handle, SCATTER_NOTIFY_SMALL_DATA));
    }
    if (param.myRank != param.root) {
        const auto *rootChannel = ChannelTo(resCtx, param.root);
        CHK_PTR_NULL(rootChannel);
        CHK_PTR_NULL(rootChannel->remoteCclMem.addr);
        const auto *source = static_cast<const uint8_t *>(rootChannel->remoteCclMem.addr)
            + static_cast<uint64_t>(param.myRank) * bytes;
        CHK_RET(HcommReadOnThread(thread, rootChannel->handle, resCtx.localBuffer.addr, source, bytes));
        CHK_RET(HcommLocalCopyOnThread(thread, param.outputPtr, resCtx.localBuffer.addr, bytes));
    }
    for (const auto *child : children) {
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, child->handle, SCATTER_NOTIFY_SMALL_CONSUMED, CUSTOM_TIMEOUT));
    }
    if (param.myRank == param.root) {
        // Aggregated ACKs cover every reader before any reuse of the root CCL,
        // including a caller choosing that CCL as its own output buffer.
        return CopyRootOutput(param, thread, bytes);
    }
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, parent->handle,
        SCATTER_NOTIFY_SMALL_CONSUMED));
}

bool RelayGroups(const OpParam &param, const AlgResourceCtx &resCtx,
    std::vector<uint32_t> &rootServer, std::vector<uint32_t> &otherServer)
{
    if (param.rankSize != 16 || resCtx.serverRanks.size() != 8 || param.count < 3) {
        return false;
    }
    std::vector<uint32_t> complement;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!std::binary_search(resCtx.serverRanks.begin(), resCtx.serverRanks.end(), rank)) {
            complement.push_back(rank);
        }
    }
    if (complement.size() != 8) {
        return false;
    }
    const bool rootIsLocal = std::binary_search(resCtx.serverRanks.begin(), resCtx.serverRanks.end(), param.root);
    rootServer = rootIsLocal ? resCtx.serverRanks : complement;
    otherServer = rootIsLocal ? complement : resCtx.serverRanks;
    return true;
}

uint64_t ChunkCount(uint64_t bytes, uint64_t chunkBytes)
{
    return bytes == 0 ? 0 : 1 + (bytes - 1) / chunkBytes;
}

HcclResult StartWorker(ThreadHandle mainThread, ThreadHandle worker)
{
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, 0));
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
}

HcclResult BeginReceive(ThreadHandle thread, const ChannelInfo &channel, uint64_t chunks)
{
    for (uint32_t lane = 0; lane < std::min<uint64_t>(2, chunks); ++lane) {
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_NOTIFY_CREDIT + lane));
    }
    return HCCL_SUCCESS;
}

HcclResult ReleaseReceiveSlot(ThreadHandle thread, const ChannelInfo &channel, uint64_t &sequence, uint64_t chunks)
{
    if (sequence + 2 < chunks) {
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle,
            SCATTER_NOTIFY_CREDIT + static_cast<uint32_t>(sequence % 2)));
    }
    ++sequence;
    return HCCL_SUCCESS;
}

HcclResult WriteSlot(ThreadHandle thread, const ChannelInfo &channel, uint8_t *remoteBase,
    const void *source, uint64_t length, uint64_t chunkBytes, uint64_t sequence)
{
    const uint32_t lane = static_cast<uint32_t>(sequence % 2);
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_NOTIFY_CREDIT + lane, CUSTOM_TIMEOUT));
    return static_cast<HcclResult>(HcommWriteWithNotifyOnThread(thread, channel.handle,
        remoteBase + lane * chunkBytes, source, length, SCATTER_NOTIFY_DATA + lane));
}

HcclResult SendInputPiece(ThreadHandle thread, const ChannelInfo &channel, uint8_t *staging,
    const uint8_t *source, uint64_t bytes, uint64_t chunkBytes, uint64_t &sequence)
{
    auto *remoteBase = static_cast<uint8_t *>(channel.remoteCclMem.addr);
    for (uint64_t offset = 0; offset < bytes; ++sequence) {
        const uint64_t length = std::min(chunkBytes, bytes - offset);
        CHK_RET(HcommLocalCopyOnThread(thread, staging, source + offset, length));
        CHK_RET(WriteSlot(thread, channel, remoteBase, staging, length, chunkBytes, sequence));
        offset += length;
    }
    return HCCL_SUCCESS;
}

HcclResult ReceiveOutputPiece(ThreadHandle thread, const ChannelInfo &channel, const uint8_t *staging,
    uint8_t *output, uint64_t bytes, uint64_t chunkBytes, uint64_t &sequence, uint64_t chunks)
{
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t length = std::min(chunkBytes, bytes - offset);
        const uint32_t lane = static_cast<uint32_t>(sequence % 2);
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_NOTIFY_DATA + lane, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(thread, output + offset, staging + lane * chunkBytes, length));
        CHK_RET(ReleaseReceiveSlot(thread, channel, sequence, chunks));
        offset += length;
    }
    return HCCL_SUCCESS;
}

HcclResult ReceivePart(ThreadHandle thread, const ChannelInfo &channel, const uint8_t *staging,
    uint8_t *output, uint64_t bytes, uint64_t chunkBytes)
{
    const uint64_t chunks = ChunkCount(bytes, chunkBytes);
    uint64_t sequence = 0;
    CHK_RET(BeginReceive(thread, channel, chunks));
    CHK_RET(ReceiveOutputPiece(thread, channel, staging, output, bytes, chunkBytes, sequence, chunks));
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_NOTIFY_FINISHED));
}

HcclResult WeightedScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t bytes,
    const std::vector<uint32_t> &rootServer, const std::vector<uint32_t> &otherServer)
{
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    CHK_PRT_RET(resCtx.channels.size() != param.rankSize - 1 || resCtx.threads.size() < param.rankSize,
        HCCL_ERROR("Incomplete relay resources"), HCCL_E_INTERNAL);
    uint64_t capacity = resCtx.localBuffer.size;
    for (const auto &channel : resCtx.channels) {
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        capacity = std::min(capacity, channel.remoteCclMem.size);
    }
    const uint64_t chunkBytes = ScatterSlotBytes(capacity, param.rankSize);
    CHK_PRT_RET(chunkBytes == 0, HCCL_ERROR("Relay buffer is too small"), HCCL_E_INTERNAL);
    // Seven helpers carry 4/11 of their paired remote output. This balances
    // each root mesh link against its cross-server port's 4x link capacity.
    const uint64_t relayBytes = (param.count * 4 / 11) * sizeof(float);
    const ThreadHandle mainThread = resCtx.aicpuThread;
    auto *buffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    auto *output = static_cast<uint8_t *>(param.outputPtr);

    if (param.myRank == param.root) {
        const auto *input = static_cast<const uint8_t *>(param.inputPtr);
        for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
            const auto &channel = resCtx.channels[i];
            const ThreadHandle worker = resCtx.threads[i + 1];
            auto *staging = buffer + i * chunkBytes;
            uint64_t sequence = 0;
            CHK_RET(StartWorker(mainThread, worker));
            const auto local = std::find(rootServer.begin(), rootServer.end(), channel.remoteRank);
            if (local != rootServer.end()) {
                const uint32_t pairedRank = otherServer[local - rootServer.begin()];
                CHK_RET(SendInputPiece(worker, channel, staging,
                    input + static_cast<uint64_t>(pairedRank) * bytes, relayBytes, chunkBytes, sequence));
                // Preserve the sequence at the segment boundary, even for a
                // partial relay tail followed by a new full-sized own block.
                CHK_RET(SendInputPiece(worker, channel, staging,
                    input + static_cast<uint64_t>(channel.remoteRank) * bytes, bytes, chunkBytes, sequence));
            } else {
                const auto remote = std::find(otherServer.begin(), otherServer.end(), channel.remoteRank);
                CHK_PRT_RET(remote == otherServer.end(), HCCL_ERROR("Invalid remote server rank"), HCCL_E_INTERNAL);
                const uint32_t pairedRank = rootServer[remote - otherServer.begin()];
                const uint64_t begin = pairedRank == param.root ? 0 : relayBytes;
                CHK_RET(SendInputPiece(worker, channel, staging,
                    input + static_cast<uint64_t>(channel.remoteRank) * bytes + begin,
                    bytes - begin, chunkBytes, sequence));
            }
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle, SCATTER_NOTIFY_FINISHED, CUSTOM_TIMEOUT));
            CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, i + 1));
        }
        for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, i + 1, CUSTOM_TIMEOUT));
        }
        return CopyRootOutput(param, mainThread, bytes);
    }

    const auto *rootChannel = ChannelTo(resCtx, param.root);
    CHK_PTR_NULL(rootChannel);
    const auto local = std::find(rootServer.begin(), rootServer.end(), param.myRank);
    if (local != rootServer.end()) {
        const uint32_t pairedRank = otherServer[local - rootServer.begin()];
        const auto *forward = ChannelTo(resCtx, pairedRank);
        CHK_PTR_NULL(forward);
        // Each segment is chunked separately; adding lengths before ceil would
        // lose a partial block and leave an unmatched credit at the boundary.
        const uint64_t chunks = ChunkCount(relayBytes, chunkBytes) + ChunkCount(bytes, chunkBytes);
        uint64_t sequence = 0;
        CHK_RET(BeginReceive(mainThread, *rootChannel, chunks));
        auto *remoteRelayBase = static_cast<uint8_t *>(forward->remoteCclMem.addr) + 2 * chunkBytes;
        for (uint64_t offset = 0; offset < relayBytes;) {
            const uint64_t length = std::min(chunkBytes, relayBytes - offset);
            const uint32_t lane = static_cast<uint32_t>(sequence % 2);
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, rootChannel->handle,
                SCATTER_NOTIFY_DATA + lane, CUSTOM_TIMEOUT));
            CHK_RET(WriteSlot(mainThread, *forward, remoteRelayBase,
                buffer + lane * chunkBytes, length, chunkBytes, sequence));
            CHK_RET(ReleaseReceiveSlot(mainThread, *rootChannel, sequence, chunks));
            offset += length;
        }
        CHK_RET(ReceiveOutputPiece(mainThread, *rootChannel, buffer, output, bytes, chunkBytes, sequence, chunks));
        // Root may finish only after the relay destination consumed its prefix.
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, forward->handle, SCATTER_NOTIFY_FINISHED, CUSTOM_TIMEOUT));
        return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, rootChannel->handle,
            SCATTER_NOTIFY_FINISHED));
    }

    const auto remote = std::find(otherServer.begin(), otherServer.end(), param.myRank);
    CHK_PRT_RET(remote == otherServer.end(), HCCL_ERROR("Invalid relay destination"), HCCL_E_INTERNAL);
    const uint32_t pairedRank = rootServer[remote - otherServer.begin()];
    if (pairedRank == param.root) {
        return ReceivePart(mainThread, *rootChannel, buffer, output, bytes, chunkBytes);
    }
    const auto *relayChannel = ChannelTo(resCtx, pairedRank);
    CHK_PTR_NULL(relayChannel);
    const ThreadHandle directWorker = resCtx.threads[1];
    const ThreadHandle relayWorker = resCtx.threads[2];
    CHK_RET(StartWorker(mainThread, directWorker));
    CHK_RET(StartWorker(mainThread, relayWorker));
    CHK_RET(ReceivePart(directWorker, *rootChannel, buffer, output + relayBytes, bytes - relayBytes, chunkBytes));
    CHK_RET(HcommThreadNotifyRecordOnThread(directWorker, mainThread, 1));
    // Incoming sources always use distinct CCL slots, independent of rank order.
    CHK_RET(ReceivePart(relayWorker, *relayChannel, buffer + 2 * chunkBytes, output, relayBytes, chunkBytes));
    CHK_RET(HcommThreadNotifyRecordOnThread(relayWorker, mainThread, 2));
    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, 1, CUSTOM_TIMEOUT));
    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, 2, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult SendScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t bytes)
{
    const uint64_t slotBytes = ScatterSlotBytes(resCtx.localBuffer.size, param.rankSize);
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    CHK_PRT_RET(slotBytes == 0 || resCtx.channels.size() != param.rankSize - 1
                    || resCtx.threads.size() < param.rankSize,
        HCCL_ERROR("Incomplete Scatter sender resources"), HCCL_E_INTERNAL);
    const ThreadHandle mainThread = resCtx.aicpuThread;
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        const auto &channel = resCtx.channels[i];
        const uint32_t lanes = ScatterReceiveLanes(channel.remoteCclMem.size);
        const uint64_t chunkBytes = std::min(slotBytes, channel.remoteCclMem.size / lanes) / sizeof(float) * sizeof(float);
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        CHK_PRT_RET(chunkBytes == 0 || channel.remoteRank >= param.rankSize || channel.remoteRank == param.root,
            HCCL_ERROR("Invalid Scatter destination"), HCCL_E_INTERNAL);
        const ThreadHandle worker = resCtx.threads[i + 1];
        auto *staging = static_cast<uint8_t *>(resCtx.localBuffer.addr) + i * slotBytes;
        const auto *source = static_cast<const uint8_t *>(param.inputPtr)
            + static_cast<uint64_t>(channel.remoteRank) * bytes;

        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        uint64_t chunk = 0;
        for (uint64_t offset = 0; offset < bytes; ++chunk) {
            const uint64_t length = std::min(chunkBytes, bytes - offset);
            CHK_RET(HcommLocalCopyOnThread(worker, staging, source + offset, length));
            const uint32_t lane = static_cast<uint32_t>(chunk % lanes);
            auto *destination = static_cast<uint8_t *>(channel.remoteCclMem.addr) + lane * chunkBytes;
            CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle,
                SCATTER_NOTIFY_CREDIT + lane, CUSTOM_TIMEOUT));
            CHK_RET(HcommWriteWithNotifyOnThread(worker, channel.handle, destination,
                staging, length, SCATTER_NOTIFY_DATA + lane));
            offset += length;
        }
        // Do not return or reuse this channel until the final output copy completed.
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle, SCATTER_NOTIFY_FINISHED, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, i + 1));
    }
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, i + 1, CUSTOM_TIMEOUT));
    }
    // Defer this copy until every sender consumed input, including in-place output.
    return CopyRootOutput(param, mainThread, bytes);
}

HcclResult ReceiveScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t bytes)
{
    const auto found = std::find_if(resCtx.channels.begin(), resCtx.channels.end(), [&param](const ChannelInfo &channel) {
        return channel.remoteRank == param.root;
    });
    CHK_PRT_RET(found == resCtx.channels.end(), HCCL_ERROR("Missing channel to Scatter root"), HCCL_E_INTERNAL);
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    const auto &channel = *found;
    const uint64_t slotBytes = ScatterSlotBytes(channel.remoteCclMem.size, param.rankSize);
    const uint32_t lanes = ScatterReceiveLanes(resCtx.localBuffer.size);
    const uint64_t chunkBytes = std::min(slotBytes, resCtx.localBuffer.size / lanes) / sizeof(float) * sizeof(float);
    CHK_PRT_RET(chunkBytes == 0, HCCL_ERROR("Scatter receive buffer is too small"), HCCL_E_INTERNAL);
    const ThreadHandle thread = resCtx.aicpuThread;
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    const uint64_t chunks = 1 + (bytes - 1) / chunkBytes;
    for (uint32_t lane = 0; lane < std::min<uint64_t>(lanes, chunks); ++lane) {
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_NOTIFY_CREDIT + lane));
    }
    uint64_t chunk = 0;
    for (uint64_t offset = 0; offset < bytes; ++chunk) {
        const uint64_t length = std::min(chunkBytes, bytes - offset);
        const uint32_t lane = static_cast<uint32_t>(chunk % lanes);
        const auto *staging = static_cast<const uint8_t *>(resCtx.localBuffer.addr) + lane * chunkBytes;
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_NOTIFY_DATA + lane, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(thread, output + offset, staging, length));
        // Return a slot only when this invocation will reuse it; no stale credit
        // may survive into a later invocation with a different size or root.
        if (chunk + lanes < chunks) {
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_NOTIFY_CREDIT + lane));
        }
        offset += length;
    }
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_NOTIFY_FINISHED));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.rankSize == 0 || param.root >= param.rankSize || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid Scatter rank or root"), HCCL_E_PARA);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("Scatter supports float32"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize,
        HCCL_ERROR("Scatter size overflows"), HCCL_E_PARA);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(param.outputPtr);
    const uint64_t bytes = param.count * sizeof(float);
    if (param.myRank == param.root) {
        CHK_PTR_NULL(param.inputPtr);
        if (param.rankSize == 1) {
            return CopyRootOutput(param, resCtx.aicpuThread, bytes);
        }
    }
    if (UseSmallScatter(param, resCtx, bytes)) {
        return SmallScatter(param, resCtx, bytes);
    }
    std::vector<uint32_t> rootServer;
    std::vector<uint32_t> otherServer;
    if (RelayGroups(param, resCtx, rootServer, otherServer)) {
        return WeightedScatter(param, resCtx, bytes, rootServer, otherServer);
    }
    if (param.myRank == param.root) {
        return SendScatter(param, resCtx, bytes);
    }
    return ReceiveScatter(param, resCtx, bytes);
}
} // namespace ops_hccl
