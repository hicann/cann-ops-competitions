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

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace {

constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint64_t MAX_DATA_SIZE = 256ULL * 1024 * 1024;
constexpr uint32_t WORKER_START_NOTIFY = 0;

HcclResult StartWorker(
    ThreadHandle controlThread,
    ThreadHandle workerThread)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(
            controlThread,
            workerThread,
            WORKER_START_NOTIFY));
}

HcclResult WaitWorkerStart(ThreadHandle workerThread)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(
            workerThread,
            WORKER_START_NOTIFY,
            CUSTOM_TIMEOUT));
}

HcclResult FinishWorker(
    ThreadHandle workerThread,
    ThreadHandle controlThread,
    uint32_t workerIndex)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(
            workerThread,
            controlThread,
            workerIndex));
}

HcclResult WaitWorkerFinish(
    ThreadHandle controlThread,
    uint32_t workerIndex)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(
            controlThread,
            workerIndex,
            CUSTOM_TIMEOUT));
}

HcclResult PeerReady(
    ThreadHandle thread,
    const ChannelInfo &channel)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(
            thread,
            channel.handle,
            NOTIFY_IDX_DATA_SIGNAL)));

    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(
            thread,
            channel.handle,
            NOTIFY_IDX_DATA_SIGNAL,
            CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult SendBuffer(
    const OpParam &param,
    const ChannelInfo &channel,
    ThreadHandle thread,
    uint64_t totalBytes)
{
    CHK_PRT_RET(
        channel.remoteCclMem.addr == nullptr ||
            channel.remoteCclMem.size == 0,
        HCCL_ERROR(
            "SendBuffer: invalid remote CCL buffer for rank[%u]",
            channel.remoteRank),
        HCCL_E_INTERNAL);

    CHK_RET(PeerReady(thread, channel));

    const uint64_t maxBytesPerLoop =
        std::min(MAX_DATA_SIZE, channel.remoteCclMem.size);

    CHK_PRT_RET(
        maxBytesPerLoop == 0,
        HCCL_ERROR(
            "SendBuffer: empty remote CCL buffer for rank[%u]",
            channel.remoteRank),
        HCCL_E_INTERNAL);

    uint8_t *source = static_cast<uint8_t *>(param.inputPtr);

    for (uint64_t offset = 0; offset < totalBytes;) {
        const uint64_t subBytes =
            std::min(maxBytesPerLoop, totalBytes - offset);

        CHK_RET(static_cast<HcclResult>(
            HcommWriteOnThread(
                thread,
                channel.handle,
                channel.remoteCclMem.addr,
                source + offset,
                subBytes)));

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(
                thread,
                channel.handle,
                NOTIFY_IDX_DATA_SIGNAL)));

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(
                thread,
                channel.handle,
                NOTIFY_IDX_ACK,
                CUSTOM_TIMEOUT)));

        offset += subBytes;
    }

    return HCCL_SUCCESS;
}

HcclResult ReceiveBuffer(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    const ChannelInfo &channel,
    ThreadHandle thread,
    uint64_t totalBytes)
{
    CHK_PRT_RET(
        resCtx.localBuffer.addr == nullptr ||
            resCtx.localBuffer.size == 0,
        HCCL_ERROR("ReceiveBuffer: invalid local CCL buffer"),
        HCCL_E_INTERNAL);

    CHK_RET(PeerReady(thread, channel));

    const uint64_t maxBytesPerLoop =
        std::min(MAX_DATA_SIZE, resCtx.localBuffer.size);

    CHK_PRT_RET(
        maxBytesPerLoop == 0,
        HCCL_ERROR("ReceiveBuffer: empty local CCL buffer"),
        HCCL_E_INTERNAL);

    uint8_t *destination = static_cast<uint8_t *>(param.outputPtr);

    for (uint64_t offset = 0; offset < totalBytes;) {
        const uint64_t subBytes =
            std::min(maxBytesPerLoop, totalBytes - offset);

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(
                thread,
                channel.handle,
                NOTIFY_IDX_DATA_SIGNAL,
                CUSTOM_TIMEOUT)));

        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(
                thread,
                destination + offset,
                resCtx.localBuffer.addr,
                subBytes)));

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(
                thread,
                channel.handle,
                NOTIFY_IDX_ACK)));

        offset += subBytes;
    }

    return HCCL_SUCCESS;
}

HcclResult RunSendWorkers(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t totalBytes)
{
    const uint32_t workerNum =
        static_cast<uint32_t>(resCtx.channels.size());
    ThreadHandle controlThread = resCtx.aicpuThread;

    for (uint32_t i = 0; i < workerNum; ++i) {
        CHK_RET(StartWorker(
            controlThread,
            resCtx.threads[i + 1]));
    }

    for (uint32_t i = 0; i < workerNum; ++i) {
        ThreadHandle workerThread = resCtx.threads[i + 1];

        CHK_RET(WaitWorkerStart(workerThread));
        CHK_RET(SendBuffer(
            param,
            resCtx.channels[i],
            workerThread,
            totalBytes));
        CHK_RET(FinishWorker(
            workerThread,
            controlThread,
            i));
    }

    for (uint32_t i = 0; i < workerNum; ++i) {
        CHK_RET(WaitWorkerFinish(controlThread, i));
    }

    return HCCL_SUCCESS;
}

HcclResult RunReceiveWorker(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t totalBytes)
{
    ThreadHandle controlThread = resCtx.aicpuThread;
    ThreadHandle workerThread = resCtx.threads[1];

    CHK_RET(StartWorker(controlThread, workerThread));
    CHK_RET(WaitWorkerStart(workerThread));

    CHK_RET(ReceiveBuffer(
        param,
        resCtx,
        resCtx.channels[0],
        workerThread,
        totalBytes));

    CHK_RET(FinishWorker(
        workerThread,
        controlThread,
        0));

    CHK_RET(WaitWorkerFinish(controlThread, 0));
    return HCCL_SUCCESS;
}

HcclResult RunBridgeWorkers(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t totalBytes)
{
    const uint32_t workerNum =
        static_cast<uint32_t>(resCtx.channels.size());
    ThreadHandle controlThread = resCtx.aicpuThread;
    ThreadHandle receiveThread = resCtx.threads[1];

    // bridge 的 channel[0] 是 root。
    CHK_RET(StartWorker(controlThread, receiveThread));
    CHK_RET(WaitWorkerStart(receiveThread));

    CHK_RET(ReceiveBuffer(
        param,
        resCtx,
        resCtx.channels[0],
        receiveThread,
        totalBytes));

    // 完整接收 root 数据后，再启动第二个 Server 上的发送工作 Thread。
    for (uint32_t i = 1; i < workerNum; ++i) {
        CHK_RET(StartWorker(
            receiveThread,
            resCtx.threads[i + 1]));
    }

    CHK_RET(FinishWorker(
        receiveThread,
        controlThread,
        0));

    for (uint32_t i = 1; i < workerNum; ++i) {
        ThreadHandle workerThread = resCtx.threads[i + 1];

        CHK_RET(WaitWorkerStart(workerThread));
        CHK_RET(SendBuffer(
            param,
            resCtx.channels[i],
            workerThread,
            totalBytes));
        CHK_RET(FinishWorker(
            workerThread,
            controlThread,
            i));
    }

    for (uint32_t i = 0; i < workerNum; ++i) {
        CHK_RET(WaitWorkerFinish(controlThread, i));
    }

    return HCCL_SUCCESS;
}

} // namespace

namespace ops_hccl {

HcclResult ExecOp(
    const OpParam &param,
    const AlgResourceCtx &resCtx)
{
    if (param.count == 0 || param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    const auto typeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(
        typeIt == SIZE_TABLE.end(),
        HCCL_ERROR(
            "Unsupported Broadcast dataType[%d]",
            static_cast<int>(param.dataType)),
        HCCL_E_INTERNAL);

    const uint64_t expectedThreadNum =
        static_cast<uint64_t>(resCtx.channels.size()) + 1;

    CHK_PRT_RET(
        resCtx.channels.empty() ||
            resCtx.threads.size() != expectedThreadNum,
        HCCL_ERROR(
            "Invalid resources: channels[%zu], threads[%zu]",
            resCtx.channels.size(),
            resCtx.threads.size()),
        HCCL_E_INTERNAL);

    const uint64_t totalBytes =
        param.count * typeIt->second;

    HCCL_INFO(
        "Broadcast ExecOp: rank[%u], root[%u], rankSize[%u], "
        "channels[%zu], threads[%zu], bytes[%llu]",
        param.myRank,
        param.root,
        param.rankSize,
        resCtx.channels.size(),
        resCtx.threads.size(),
        static_cast<unsigned long long>(totalBytes));

    if (param.rankSize <= RANKS_PER_SERVER) {
        if (param.myRank == param.root) {
            return RunSendWorkers(
                param,
                resCtx,
                totalBytes);
        }

        return RunReceiveWorker(
            param,
            resCtx,
            totalBytes);
    }

    CHK_PRT_RET(
        param.rankSize != 2 * RANKS_PER_SERVER,
        HCCL_ERROR(
            "Unsupported rankSize[%u]",
            param.rankSize),
        HCCL_E_INTERNAL);

    const uint32_t rootServer =
        param.root / RANKS_PER_SERVER;
    const uint32_t myServer =
        param.myRank / RANKS_PER_SERVER;
    const uint32_t bridgeRank =
        rootServer == 0
            ? param.root + RANKS_PER_SERVER
            : param.root - RANKS_PER_SERVER;

    if (param.myRank == param.root) {
        return RunSendWorkers(
            param,
            resCtx,
            totalBytes);
    }

    if (param.myRank == bridgeRank) {
        return RunBridgeWorkers(
            param,
            resCtx,
            totalBytes);
    }

    if (myServer == rootServer) {
        return RunReceiveWorker(
            param,
            resCtx,
            totalBytes);
    }

    return RunReceiveWorker(
        param,
        resCtx,
        totalBytes);
}

} // namespace ops_hccl
