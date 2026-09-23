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

#include <algorithm>
#include <limits>

namespace ops_hccl {
namespace {

HcclResult SendWrite(const ThreadHandle &thread, const ChannelInfo &channel, const void *src, uint64_t size,
    void *remoteDst = nullptr)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    void *dst = remoteDst == nullptr ? channel.remoteCclMem.addr : remoteDst;
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle, dst, src, size)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult RecvWrite(const ThreadHandle &thread, const ChannelInfo &channel)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult WaitReadDone(const ThreadHandle &thread, const ChannelInfo &channel)
{
    return static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
}

HcclResult RecvRead(const ThreadHandle &thread, const ChannelInfo &channel, void *localCclDst,
    void *outputDst, const void *remoteSrc, uint64_t size)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    // AICPU_TS builds the read destination's RMA token from the memory
    // registered on this channel. These independent channels register the
    // HCCL buffer, while the per-invocation user output is not a channel
    // memory handle. Read into registered CCL memory, then place the result in
    // the user buffer with a local copy.
    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, channel.handle, localCclDst, remoteSrc, size)));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, outputDst, localCclDst, size)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult CopyLocal(const ThreadHandle &thread, void *dst, const void *src, uint64_t size)
{
    return static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, size));
}

uint8_t *ByteOffset(void *base, uint64_t offset)
{
    return static_cast<uint8_t *>(base) + offset;
}

uint32_t GetMeshWorkerSlot(uint32_t targetRank, uint32_t selfLocalIdx)
{
    const uint32_t targetLocalIdx = targetRank % SCATTER_RANKS_PER_SERVER;
    return targetLocalIdx < selfLocalIdx ? targetLocalIdx : targetLocalIdx - 1;
}

uint64_t GetHelperBytes(uint64_t chunkBytes)
{
    // chunkBytes is FP32-aligned. Compute in elements so the split cannot cut an
    // element, and divide before multiplying to keep the arithmetic overflow-free.
    const uint64_t elements = chunkBytes / sizeof(float);
    const uint64_t helperElements =
        (elements / SCATTER_HELPER_RATIO_DENOMINATOR) * SCATTER_HELPER_RATIO_NUMERATOR +
        ((elements % SCATTER_HELPER_RATIO_DENOMINATOR) * SCATTER_HELPER_RATIO_NUMERATOR) /
            SCATTER_HELPER_RATIO_DENOMINATOR;
    return helperElements * sizeof(float);
}

HcclResult GetChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank, const ChannelInfo *&channel)
{
    CHK_PRT_RET(remoteRank >= SCATTER_EXPECTED_RANK_SIZE ||
                    resCtx.channels[remoteRank].remoteRank != remoteRank ||
                    resCtx.channels[remoteRank].remoteCclMem.addr == nullptr ||
                    resCtx.channels[remoteRank].remoteCclMem.size < SCATTER_REQUIRED_CCL_BYTES,
        HCCL_ERROR("Channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
    channel = &resCtx.channels[remoteRank];
    return HCCL_SUCCESS;
}

HcclResult StartWorker(const ThreadHandle &controlThread, const ThreadHandle &workerThread)
{
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(controlThread, workerThread, 0)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(workerThread, 0, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RecordWorkerDone(
    const ThreadHandle &workerThread, const ThreadHandle &controlThread, uint32_t workerSlot)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(workerThread, controlThread, workerSlot));
}

HcclResult WaitWorkerDone(const ThreadHandle &controlThread, uint32_t workerSlot)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(controlThread, workerSlot, CUSTOM_TIMEOUT));
}

HcclResult RunSmallScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes,
    uint32_t rootServerBase, uint32_t remoteServerBase, uint32_t rootLocalIdx)
{
    const ThreadHandle &controlThread = resCtx.threads[0];

    if (param.myRank == param.root) {
        // The eight remote blocks are one contiguous half of the root input.
        // Stage that half with one SDMA task. Starting workers afterwards
        // establishes staging -> remote-ready ordering without eight small
        // LocalCopy tasks on the worker streams.
        const uint64_t remoteHalfOffset = static_cast<uint64_t>(remoteServerBase) * blockBytes;
        const uint64_t remoteHalfBytes = SCATTER_RANKS_PER_SERVER * blockBytes;
        CHK_RET(CopyLocal(controlThread, ByteOffset(resCtx.localBuffer.addr, remoteHalfOffset),
            ByteOffset(param.inputPtr, remoteHalfOffset), remoteHalfBytes));

        for (uint32_t workerSlot = 0; workerSlot < SCATTER_SMALL_MESH_WORKER_COUNT; ++workerSlot) {
            CHK_RET(StartWorker(controlThread, resCtx.threads[workerSlot + 1]));
        }

        // Remote CLOS targets pull directly from the root CCL buffer while the
        // same worker submits its local Mesh write. This overlaps both fabrics
        // without adding threads or concentrating 15 data operations at root.
        for (uint32_t localIdx = 0; localIdx < SCATTER_RANKS_PER_SERVER; ++localIdx) {
            if (localIdx == rootLocalIdx) {
                continue;
            }
            const uint32_t localTarget = rootServerBase + localIdx;
            const uint32_t remoteTarget = remoteServerBase + localIdx;
            const uint32_t workerSlot =
                GetMeshWorkerSlot(localTarget, rootLocalIdx) % SCATTER_SMALL_MESH_WORKER_COUNT;
            const ThreadHandle &workerThread = resCtx.threads[workerSlot + 1];
            const ChannelInfo *localChannel = nullptr;
            const ChannelInfo *remoteChannel = nullptr;
            CHK_RET(GetChannel(resCtx, localTarget, localChannel));
            CHK_RET(GetChannel(resCtx, remoteTarget, remoteChannel));
            const uint64_t localOffset = static_cast<uint64_t>(localTarget) * blockBytes;
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                workerThread, remoteChannel->handle, NOTIFY_IDX_ACK)));
            CHK_RET(SendWrite(workerThread, *localChannel, ByteOffset(param.inputPtr, localOffset), blockBytes));
        }

        // Control owns the one CLOS target not paired with a Mesh worker, plus
        // the root result. Keep worker release ahead of this control work so
        // the Mesh critical path is not delayed by another control-stream task.
        const uint32_t directTarget = remoteServerBase + rootLocalIdx;
        const ChannelInfo *directChannel = nullptr;
        CHK_RET(GetChannel(resCtx, directTarget, directChannel));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            controlThread, directChannel->handle, NOTIFY_IDX_ACK)));
        const uint64_t selfOffset = static_cast<uint64_t>(param.root) * blockBytes;
        CHK_RET(CopyLocal(controlThread, param.outputPtr, ByteOffset(param.inputPtr, selfOffset), blockBytes));
        CHK_RET(WaitReadDone(controlThread, *directChannel));

        // A worker may own two Mesh/CLOS pairs. Queue all first hops before
        // waiting for any pull completion so the second pair is not delayed by
        // the first receiver's done signal.
        for (uint32_t localIdx = 0; localIdx < SCATTER_RANKS_PER_SERVER; ++localIdx) {
            if (localIdx == rootLocalIdx) {
                continue;
            }
            const uint32_t localTarget = rootServerBase + localIdx;
            const uint32_t remoteTarget = remoteServerBase + localIdx;
            const uint32_t workerSlot =
                GetMeshWorkerSlot(localTarget, rootLocalIdx) % SCATTER_SMALL_MESH_WORKER_COUNT;
            const ChannelInfo *remoteChannel = nullptr;
            CHK_RET(GetChannel(resCtx, remoteTarget, remoteChannel));
            CHK_RET(WaitReadDone(resCtx.threads[workerSlot + 1], *remoteChannel));
        }
        for (uint32_t workerSlot = 0; workerSlot < SCATTER_SMALL_MESH_WORKER_COUNT; ++workerSlot) {
            CHK_RET(RecordWorkerDone(resCtx.threads[workerSlot + 1], controlThread, workerSlot));
        }
        for (uint32_t workerSlot = 0; workerSlot < SCATTER_SMALL_MESH_WORKER_COUNT; ++workerSlot) {
            CHK_RET(WaitWorkerDone(controlThread, workerSlot));
        }
        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel = nullptr;
    CHK_RET(GetChannel(resCtx, param.root, rootChannel));
    if (param.myRank >= rootServerBase &&
        param.myRank < rootServerBase + SCATTER_RANKS_PER_SERVER) {
        CHK_RET(RecvWrite(controlThread, *rootChannel));
        CHK_RET(CopyLocal(controlThread, param.outputPtr, resCtx.localBuffer.addr, blockBytes));
    } else {
        const uint64_t remoteOffset = static_cast<uint64_t>(param.myRank) * blockBytes;
        CHK_RET(RecvRead(controlThread, *rootChannel, resCtx.localBuffer.addr, param.outputPtr,
            ByteOffset(rootChannel->remoteCclMem.addr, remoteOffset), blockBytes));
    }
    return HCCL_SUCCESS;
}

HcclResult RunSmallGroupedRelayScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes,
    uint32_t rootServerBase, uint32_t remoteServerBase, uint32_t rootLocalIdx)
{
    const ThreadHandle &controlThread = resCtx.threads[0];

    if (param.myRank == param.root) {
        for (uint32_t workerSlot = 0; workerSlot < SCATTER_SMALL_MESH_WORKER_COUNT; ++workerSlot) {
            CHK_RET(StartWorker(controlThread, resCtx.threads[workerSlot + 1]));
        }

        // Four independent workers move two contiguous blocks each across
        // CLOS. This halves transaction count without concentrating the whole
        // remote half at one relay, and removes the root staging copy.
        for (uint32_t workerSlot = 0; workerSlot < SCATTER_SMALL_MESH_WORKER_COUNT; ++workerSlot) {
            const uint32_t firstLocalIdx = workerSlot * 2;
            const uint32_t relayRank = remoteServerBase + firstLocalIdx;
            const ChannelInfo *relayChannel = nullptr;
            CHK_RET(GetChannel(resCtx, relayRank, relayChannel));
            CHK_RET(SendWrite(resCtx.threads[workerSlot + 1], *relayChannel,
                ByteOffset(param.inputPtr, static_cast<uint64_t>(relayRank) * blockBytes), 2 * blockBytes,
                relayChannel->remoteCclMem.addr));
        }

        // The same workers scatter the root server after their CLOS bulk. The
        // four CLOS queues still start in parallel rather than serializing on
        // the control thread.
        for (uint32_t localIdx = 0; localIdx < SCATTER_RANKS_PER_SERVER; ++localIdx) {
            if (localIdx == rootLocalIdx) {
                continue;
            }
            const uint32_t targetRank = rootServerBase + localIdx;
            const uint32_t workerSlot =
                GetMeshWorkerSlot(targetRank, rootLocalIdx) % SCATTER_SMALL_MESH_WORKER_COUNT;
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(resCtx, targetRank, channel));
            CHK_RET(SendWrite(resCtx.threads[workerSlot + 1], *channel,
                ByteOffset(param.inputPtr, static_cast<uint64_t>(targetRank) * blockBytes), blockBytes));
        }

        CHK_RET(CopyLocal(controlThread, param.outputPtr,
            ByteOffset(param.inputPtr, static_cast<uint64_t>(param.root) * blockBytes), blockBytes));

        for (uint32_t workerSlot = 0; workerSlot < SCATTER_SMALL_MESH_WORKER_COUNT; ++workerSlot) {
            CHK_RET(RecordWorkerDone(resCtx.threads[workerSlot + 1], controlThread, workerSlot));
        }
        for (uint32_t workerSlot = 0; workerSlot < SCATTER_SMALL_MESH_WORKER_COUNT; ++workerSlot) {
            CHK_RET(WaitWorkerDone(controlThread, workerSlot));
        }
        return HCCL_SUCCESS;
    }

    const bool onRootServer = param.myRank >= rootServerBase &&
        param.myRank < rootServerBase + SCATTER_RANKS_PER_SERVER;
    if (onRootServer) {
        const ChannelInfo *rootChannel = nullptr;
        CHK_RET(GetChannel(resCtx, param.root, rootChannel));
        CHK_RET(RecvWrite(controlThread, *rootChannel));
        CHK_RET(CopyLocal(controlThread, param.outputPtr, resCtx.localBuffer.addr, blockBytes));
        return HCCL_SUCCESS;
    }

    const uint32_t localIdx = param.myRank % SCATTER_RANKS_PER_SERVER;
    const uint32_t relayRank = remoteServerBase + (localIdx & ~1U);
    if (param.myRank != relayRank) {
        const ChannelInfo *relayChannel = nullptr;
        CHK_RET(GetChannel(resCtx, relayRank, relayChannel));
        CHK_RET(RecvWrite(controlThread, *relayChannel));
        CHK_RET(CopyLocal(controlThread, param.outputPtr, resCtx.localBuffer.addr, blockBytes));
        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel = nullptr;
    const ChannelInfo *peerChannel = nullptr;
    CHK_RET(GetChannel(resCtx, param.root, rootChannel));
    CHK_RET(GetChannel(resCtx, relayRank + 1, peerChannel));
    CHK_RET(RecvWrite(controlThread, *rootChannel));
    CHK_RET(SendWrite(
        controlThread, *peerChannel, ByteOffset(resCtx.localBuffer.addr, blockBytes), blockBytes));
    CHK_RET(CopyLocal(controlThread, param.outputPtr, resCtx.localBuffer.addr, blockBytes));
    return HCCL_SUCCESS;
}

HcclResult RunLargeRoot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t blockBytes,
    uint32_t rootServerBase, uint32_t remoteServerBase, uint32_t rootLocalIdx)
{
    const ThreadHandle &controlThread = resCtx.threads[0];

    // Start all Mesh workers before queueing the root CLOS stream. Each worker
    // owns exactly one root->helper channel for the complete invocation.
    for (uint32_t localIdx = 0; localIdx < SCATTER_RANKS_PER_SERVER; ++localIdx) {
        if (localIdx == rootLocalIdx) {
            continue;
        }
        const uint32_t targetRank = rootServerBase + localIdx;
        const uint32_t workerSlot = GetMeshWorkerSlot(targetRank, rootLocalIdx);
        CHK_RET(StartWorker(controlThread, resCtx.threads[workerSlot + 1]));
    }

    for (uint32_t localIdx = 0; localIdx < SCATTER_RANKS_PER_SERVER; ++localIdx) {
        if (localIdx == rootLocalIdx) {
            continue;
        }
        const uint32_t helperRank = rootServerBase + localIdx;
        const uint32_t remoteTarget = remoteServerBase + localIdx;
        const uint32_t workerSlot = GetMeshWorkerSlot(helperRank, rootLocalIdx);
        const ThreadHandle &workerThread = resCtx.threads[workerSlot + 1];
        const ChannelInfo *helperChannel = nullptr;
        CHK_RET(GetChannel(resCtx, helperRank, helperChannel));

        for (uint64_t chunkOffset = 0; chunkOffset < blockBytes;) {
            const uint64_t chunkBytes = std::min(SCATTER_PIPELINE_CHUNK_BYTES, blockBytes - chunkOffset);
            const uint64_t helperBytes = GetHelperBytes(chunkBytes);
            // Prefix first: the helper can start its CLOS forward while this
            // same Mesh link transfers the helper's local result.
            if (helperBytes != 0) {
                const uint64_t helperInputOffset =
                    static_cast<uint64_t>(remoteTarget) * blockBytes + chunkOffset;
                void *remoteHelperSlot =
                    ByteOffset(helperChannel->remoteCclMem.addr, SCATTER_PIPELINE_CHUNK_BYTES);
                CHK_RET(SendWrite(workerThread, *helperChannel,
                    ByteOffset(param.inputPtr, helperInputOffset), helperBytes, remoteHelperSlot));
            }
            const uint64_t localInputOffset = static_cast<uint64_t>(helperRank) * blockBytes + chunkOffset;
            CHK_RET(SendWrite(workerThread, *helperChannel, ByteOffset(param.inputPtr, localInputOffset),
                chunkBytes, helperChannel->remoteCclMem.addr));
            chunkOffset += chunkBytes;
        }
        CHK_RET(RecordWorkerDone(workerThread, controlThread, workerSlot));
    }

    for (uint64_t chunkOffset = 0; chunkOffset < blockBytes;) {
        const uint64_t chunkBytes = std::min(SCATTER_PIPELINE_CHUNK_BYTES, blockBytes - chunkOffset);
        const uint64_t helperBytes = GetHelperBytes(chunkBytes);
        const uint64_t mainBytes = chunkBytes - helperBytes;

        const uint64_t selfInputOffset = static_cast<uint64_t>(param.root) * blockBytes + chunkOffset;
        CHK_RET(CopyLocal(controlThread, ByteOffset(param.outputPtr, chunkOffset),
            ByteOffset(param.inputPtr, selfInputOffset), chunkBytes));

        // The same-local-id target has no helper and receives its full chunk.
        // Send it first because its local copy is larger than a split suffix.
        const uint32_t directTarget = remoteServerBase + rootLocalIdx;
        const ChannelInfo *directChannel = nullptr;
        CHK_RET(GetChannel(resCtx, directTarget, directChannel));
        const uint64_t directInputOffset =
            static_cast<uint64_t>(directTarget) * blockBytes + chunkOffset;
        CHK_RET(SendWrite(controlThread, *directChannel,
            ByteOffset(param.inputPtr, directInputOffset), chunkBytes, directChannel->remoteCclMem.addr));

        for (uint32_t localIdx = 0; localIdx < SCATTER_RANKS_PER_SERVER; ++localIdx) {
            if (localIdx == rootLocalIdx) {
                continue;
            }
            const uint32_t targetRank = remoteServerBase + localIdx;
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(resCtx, targetRank, channel));
            const uint64_t inputOffset =
                static_cast<uint64_t>(targetRank) * blockBytes + chunkOffset + helperBytes;
            void *remoteSlot =
                ByteOffset(channel->remoteCclMem.addr, SCATTER_PIPELINE_CHUNK_BYTES);
            CHK_RET(SendWrite(controlThread, *channel,
                ByteOffset(param.inputPtr, inputOffset), mainBytes, remoteSlot));
        }
        chunkOffset += chunkBytes;
    }

    for (uint32_t workerSlot = 0; workerSlot < SCATTER_MESH_WORKER_COUNT; ++workerSlot) {
        CHK_RET(WaitWorkerDone(controlThread, workerSlot));
    }
    return HCCL_SUCCESS;
}

HcclResult RunLargeSourceHelper(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t blockBytes, uint32_t remoteServerBase)
{
    const ThreadHandle &controlThread = resCtx.threads[0];
    const ThreadHandle &workerThread = resCtx.threads[1];
    const uint32_t localIdx = param.myRank % SCATTER_RANKS_PER_SERVER;
    const uint32_t remoteTarget = remoteServerBase + localIdx;
    const ChannelInfo *rootChannel = nullptr;
    const ChannelInfo *interChannel = nullptr;
    CHK_RET(GetChannel(resCtx, param.root, rootChannel));
    CHK_RET(GetChannel(resCtx, remoteTarget, interChannel));

    bool forwardPending = false;
    for (uint64_t chunkOffset = 0; chunkOffset < blockBytes;) {
        const uint64_t chunkBytes = std::min(SCATTER_PIPELINE_CHUNK_BYTES, blockBytes - chunkOffset);
        const uint64_t helperBytes = GetHelperBytes(chunkBytes);

        if (helperBytes != 0) {
            // Do not acknowledge an overwrite of the forwarding slot until the
            // previous CLOS write has consumed it.
            if (forwardPending) {
                CHK_RET(WaitWorkerDone(controlThread, 0));
            }
            CHK_RET(RecvWrite(controlThread, *rootChannel));
            CHK_RET(StartWorker(controlThread, workerThread));
            CHK_RET(SendWrite(workerThread, *interChannel,
                ByteOffset(resCtx.localBuffer.addr, SCATTER_PIPELINE_CHUNK_BYTES), helperBytes,
                interChannel->remoteCclMem.addr));
            CHK_RET(RecordWorkerDone(workerThread, controlThread, 0));
            forwardPending = true;
        }
        CHK_RET(RecvWrite(controlThread, *rootChannel));
        CHK_RET(CopyLocal(controlThread, ByteOffset(param.outputPtr, chunkOffset),
            resCtx.localBuffer.addr, chunkBytes));
        chunkOffset += chunkBytes;
    }
    if (forwardPending) {
        CHK_RET(WaitWorkerDone(controlThread, 0));
    }
    return HCCL_SUCCESS;
}

HcclResult RunLargeRemoteTarget(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t blockBytes, uint32_t rootServerBase, uint32_t rootLocalIdx)
{
    const ThreadHandle &controlThread = resCtx.threads[0];
    const uint32_t localIdx = param.myRank % SCATTER_RANKS_PER_SERVER;
    const ChannelInfo *rootChannel = nullptr;
    CHK_RET(GetChannel(resCtx, param.root, rootChannel));

    // The same-local-id target is fed entirely by the root and needs no helper
    // stream or inter-thread synchronization.
    if (localIdx == rootLocalIdx) {
        for (uint64_t chunkOffset = 0; chunkOffset < blockBytes;) {
            const uint64_t chunkBytes = std::min(SCATTER_PIPELINE_CHUNK_BYTES, blockBytes - chunkOffset);
            CHK_RET(RecvWrite(controlThread, *rootChannel));
            CHK_RET(CopyLocal(controlThread, ByteOffset(param.outputPtr, chunkOffset),
                resCtx.localBuffer.addr, chunkBytes));
            chunkOffset += chunkBytes;
        }
        return HCCL_SUCCESS;
    }

    const uint32_t sourceHelper = rootServerBase + localIdx;
    const ChannelInfo *helperChannel = nullptr;
    CHK_RET(GetChannel(resCtx, sourceHelper, helperChannel));
    const ThreadHandle &rootRecvThread = resCtx.threads[1];
    CHK_RET(StartWorker(controlThread, rootRecvThread));
    for (uint64_t chunkOffset = 0; chunkOffset < blockBytes;) {
        const uint64_t chunkBytes = std::min(SCATTER_PIPELINE_CHUNK_BYTES, blockBytes - chunkOffset);
        const uint64_t helperBytes = GetHelperBytes(chunkBytes);
        const uint64_t mainBytes = chunkBytes - helperBytes;

        // Prefix and suffix use disjoint CCL slots and disjoint output ranges,
        // so their receives and copies can progress concurrently. Each stream
        // copies its previous wave before acknowledging reuse of its own slot.
        CHK_RET(RecvWrite(rootRecvThread, *rootChannel));
        CHK_RET(CopyLocal(rootRecvThread, ByteOffset(param.outputPtr, chunkOffset + helperBytes),
            ByteOffset(resCtx.localBuffer.addr, SCATTER_PIPELINE_CHUNK_BYTES), mainBytes));
        if (helperBytes != 0) {
            CHK_RET(RecvWrite(controlThread, *helperChannel));
            CHK_RET(CopyLocal(controlThread, ByteOffset(param.outputPtr, chunkOffset),
                resCtx.localBuffer.addr, helperBytes));
        }
        chunkOffset += chunkBytes;
    }
    CHK_RET(RecordWorkerDone(rootRecvThread, controlThread, 0));
    CHK_RET(WaitWorkerDone(controlThread, 0));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.magic != SCATTER_RESOURCE_MAGIC || resCtx.version != SCATTER_RESOURCE_VERSION,
        HCCL_ERROR("Invalid scatter resource ABI, magic[0x%x] version[%u]", resCtx.magic, resCtx.version),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(param.rankSize != SCATTER_EXPECTED_RANK_SIZE || param.myRank >= param.rankSize ||
                    param.root >= param.rankSize,
        HCCL_ERROR("Invalid topology parameters, rankSize[%u] myRank[%u] root[%u]", param.rankSize, param.myRank,
            param.root),
        HCCL_E_PARA);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Scatter direct hybrid only supports FP32, dataType[%d]", static_cast<int>(param.dataType)),
        HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("Scatter count overflows byte size, count[%llu]", static_cast<unsigned long long>(param.count)),
        HCCL_E_PARA);

    const uint64_t blockBytes = param.count * sizeof(float);
    CHK_PRT_RET(blockBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("Scatter input byte size overflows, blockBytes[%llu] rankSize[%u]",
            static_cast<unsigned long long>(blockBytes), param.rankSize),
        HCCL_E_PARA);
    const uint64_t totalBytes = blockBytes * param.rankSize;
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < SCATTER_REQUIRED_CCL_BYTES,
        HCCL_ERROR("Invalid local HCCL buffer, addr[%p] size[%llu]", resCtx.localBuffer.addr,
            static_cast<unsigned long long>(resCtx.localBuffer.size)),
        HCCL_E_INTERNAL);
    const uint32_t rootServer = param.root / SCATTER_RANKS_PER_SERVER;
    const uint32_t rootLocalIdx = param.root % SCATTER_RANKS_PER_SERVER;
    const uint32_t rootServerBase = rootServer * SCATTER_RANKS_PER_SERVER;
    const uint32_t remoteServerBase =
        ((rootServer + 1) % SCATTER_SERVER_NUM) * SCATTER_RANKS_PER_SERVER;

    if (totalBytes <= SCATTER_SMALL_TOTAL_BYTES) {
        HCCL_INFO("Scatter direct small path, rank[%u] root[%u] totalBytes[%llu]", param.myRank, param.root,
            static_cast<unsigned long long>(totalBytes));
        if (totalBytes == SCATTER_BULK_RELAY_TOTAL_BYTES) {
            return RunSmallGroupedRelayScatter(
                param, resCtx, blockBytes, rootServerBase, remoteServerBase, rootLocalIdx);
        }
        return RunSmallScatter(param, resCtx, blockBytes, rootServerBase, remoteServerBase, rootLocalIdx);
    }

    HCCL_INFO("Scatter direct hybrid path, rank[%u] root[%u] totalBytes[%llu]", param.myRank, param.root,
        static_cast<unsigned long long>(totalBytes));
    if (param.myRank == param.root) {
        return RunLargeRoot(param, resCtx, blockBytes, rootServerBase, remoteServerBase, rootLocalIdx);
    }
    if (param.myRank >= rootServerBase &&
        param.myRank < rootServerBase + SCATTER_RANKS_PER_SERVER) {
        return RunLargeSourceHelper(param, resCtx, blockBytes, remoteServerBase);
    }
    return RunLargeRemoteTarget(param, resCtx, blockBytes, rootServerBase, rootLocalIdx);
}
} // namespace ops_hccl
