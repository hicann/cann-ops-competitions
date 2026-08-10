/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <limits>
#include <vector>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace {
constexpr uint32_t DIRECT_NOTIFY_IDX = 0;

void *AddOffset(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
}

HcclResult GetDataTypeSize(HcclDataType dataType, uint32_t &dataTypeSize)
{
    auto iter = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(iter == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type[%d]", static_cast<int32_t>(dataType)), HCCL_E_NOT_SUPPORT);
    dataTypeSize = iter->second;
    return HCCL_SUCCESS;
}

uint32_t GetCommunicationThreadNum(uint32_t rankSize, uint64_t dataSize)
{
    uint32_t channelNum = rankSize > 1 ? rankSize - 1 : 0;
    if (dataSize <= DIRECT_SMALL_DATA_THRESHOLD && channelNum > DIRECT_SMALL_COMM_THREAD_NUM) {
        return DIRECT_SMALL_COMM_THREAD_NUM;
    }
    return channelNum;
}

HcclResult ThreadSyncBefore(const std::vector<ThreadHandle> &threads, bool prioritizeLastThread)
{
    // 独立本地拷贝线程位于末尾，优先唤醒它以尽早与网络传输重叠。
    uint32_t lastThreadIdx = static_cast<uint32_t>(threads.size() - 1);
    if (prioritizeLastThread) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[0], threads[lastThreadIdx], 0)));
    }

    // 主线程唤醒其余从线程，再由所有从线程等待各自的Notify 0。
    for (uint32_t idx = 1; idx < threads.size(); idx++) {
        if (prioritizeLastThread && idx == lastThreadIdx) {
            continue;
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[idx], 0)));
    }
    for (uint32_t idx = 1; idx < threads.size(); idx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[idx], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadSyncAfter(const std::vector<ThreadHandle> &threads)
{
    // 主线程的Notify 0由Host/Device同步占用，因此通信线程完成信号使用Notify 1..N-1。
    for (uint32_t idx = 1; idx < threads.size(); idx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], idx, CUSTOM_TIMEOUT)));
    }
    for (uint32_t idx = 1; idx < threads.size(); idx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[idx], threads[0], idx)));
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateResources(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize, uint64_t outputSize)
{
    uint32_t channelNum = param.rankSize - 1;
    uint32_t communicationThreadNum = GetCommunicationThreadNum(param.rankSize, dataSize);
    bool useDedicatedLocalCopyThread = dataSize > DIRECT_LOCAL_COPY_THREAD_THRESHOLD;
    uint32_t expectedThreadNum = communicationThreadNum + static_cast<uint32_t>(useDedicatedLocalCopyThread);
    CHK_PRT_RET(resCtx.threads.size() != expectedThreadNum || resCtx.channels.size() != channelNum,
        HCCL_ERROR("Resource number mismatch: rankSize[%u], threadNum[%zu/%u], channelNum[%zu/%u]", param.rankSize,
            resCtx.threads.size(), expectedThreadNum, resCtx.channels.size(), channelNum),
        HCCL_E_INTERNAL);

    std::vector<bool> remoteRankSeen(param.rankSize, false);
    remoteRankSeen[param.myRank] = true;
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteRank >= param.rankSize || remoteRankSeen[channel.remoteRank],
            HCCL_ERROR("Invalid or duplicate remote rank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.notifyNum <= DIRECT_NOTIFY_IDX,
            HCCL_ERROR("Insufficient channel notify number[%u] for rank[%u]", channel.notifyNum, channel.remoteRank),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.remoteInputMem.addr == nullptr || channel.remoteInputMem.size < dataSize,
            HCCL_ERROR("Invalid remote input buffer for rank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.remoteOutputMem.addr == nullptr || channel.remoteOutputMem.size < outputSize,
            HCCL_ERROR("Invalid remote output buffer for rank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
        remoteRankSeen[channel.remoteRank] = true;
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateExecutionParameters(
    const OpParam &param, uint32_t &dataTypeSize,
    uint64_t &dataSize, uint64_t &outputSize)
{
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PRT_RET(
        param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR(
            "Invalid rank information: myRank[%u], rankSize[%u]",
            param.myRank, param.rankSize),
        HCCL_E_PARA);
    CHK_RET(GetDataTypeSize(param.dataType, dataTypeSize));
    CHK_PRT_RET(
        param.count >
            std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR(
            "Input data size overflows: count[%llu], dataTypeSize[%u]",
            static_cast<unsigned long long>(param.count), dataTypeSize),
        HCCL_E_PARA);
    dataSize = param.count * dataTypeSize;
    CHK_PRT_RET(
        dataSize >
            std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR(
            "Output data size overflows: dataSize[%llu], rankSize[%u]",
            static_cast<unsigned long long>(dataSize), param.rankSize),
        HCCL_E_PARA);
    outputSize = dataSize * param.rankSize;
    return HCCL_SUCCESS;
}

HcclResult SubmitDirectReads(
    const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t dataSize, uint32_t communicationThreadNum)
{
    for (uint32_t channelIndex = 0;
         channelIndex < resCtx.channels.size(); channelIndex++) {
        const ChannelInfo &channel = resCtx.channels[channelIndex];
        ThreadHandle thread = resCtx.threads[
            (channelIndex + 1) % communicationThreadNum];
        void *localOutput = AddOffset(
            param.outputPtr,
            static_cast<uint64_t>(channel.remoteRank) * dataSize);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
            thread, channel.handle, localOutput,
            channel.remoteInputMem.addr, dataSize)));
    }
    return HCCL_SUCCESS;
}

HcclResult SubmitWrites(
    const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t dataSize, uint64_t outputOffset,
    uint32_t communicationThreadNum)
{
    for (uint32_t channelIndex = 0;
         channelIndex < resCtx.channels.size(); channelIndex++) {
        const ChannelInfo &channel = resCtx.channels[channelIndex];
        ThreadHandle thread =
            resCtx.threads[channelIndex % communicationThreadNum];
        void *remoteOutput =
            AddOffset(channel.remoteOutputMem.addr, outputOffset);
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
            thread, channel.handle, remoteOutput,
            param.inputPtr, dataSize, DIRECT_NOTIFY_IDX)));
    }
    return HCCL_SUCCESS;
}

HcclResult CopyLocalSlice(
    const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t dataSize, uint64_t outputOffset,
    bool useDedicatedLocalCopyThread)
{
    void *localOutput = AddOffset(param.outputPtr, outputOffset);
    if (localOutput == param.inputPtr) {
        return HCCL_SUCCESS;
    }
    ThreadHandle localCopyThread = useDedicatedLocalCopyThread
        ? resCtx.threads.back() : resCtx.threads[0];
    return static_cast<HcclResult>(HcommLocalCopyOnThread(
        localCopyThread, localOutput, param.inputPtr, dataSize));
}

HcclResult WaitForWriteNotifications(
    const AlgResourceCtx &resCtx, uint32_t communicationThreadNum)
{
    for (uint32_t channelIndex = 0;
         channelIndex < resCtx.channels.size(); channelIndex++) {
        const ChannelInfo &channel = resCtx.channels[channelIndex];
        ThreadHandle thread =
            resCtx.threads[channelIndex % communicationThreadNum];
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, DIRECT_NOTIFY_IDX, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

struct ExecutionPlan {
    uint32_t communicationThreadNum;
    bool useDirectRead;
    bool useDedicatedLocalCopyThread;
    uint64_t outputOffset;
};

ExecutionPlan BuildExecutionPlan(const OpParam &param, uint64_t dataSize)
{
    ExecutionPlan plan{};
    plan.communicationThreadNum =
        GetCommunicationThreadNum(param.rankSize, dataSize);
    plan.useDirectRead = param.rankSize > 1;
    plan.useDedicatedLocalCopyThread =
        param.rankSize > 1 &&
        dataSize > DIRECT_LOCAL_COPY_THREAD_THRESHOLD;
    plan.outputOffset =
        static_cast<uint64_t>(param.myRank) * dataSize;
    HCCL_INFO(
        "Direct-buffer AllGather v16: rank[%u/%u], dataSize[%llu], "
        "communicationThreadNum[%u], directRead[%d], "
        "dedicatedLocalCopyThread[%d]",
        param.myRank, param.rankSize,
        static_cast<unsigned long long>(dataSize),
        plan.communicationThreadNum, plan.useDirectRead,
        plan.useDedicatedLocalCopyThread);
    return plan;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");
    uint32_t dataTypeSize = 0;
    uint64_t dataSize = 0;
    uint64_t outputSize = 0;
    CHK_RET(ValidateExecutionParameters(
        param, dataTypeSize, dataSize, outputSize));
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        resCtx.threads.empty(),
        HCCL_ERROR("No AICPU communication thread was acquired"),
        HCCL_E_INTERNAL);

    const ExecutionPlan plan = BuildExecutionPlan(param, dataSize);
    if (param.rankSize > 1) {
        CHK_RET(ValidateResources(param, resCtx, dataSize, outputSize));
        CHK_RET(ThreadSyncBefore(
            resCtx.threads, plan.useDedicatedLocalCopyThread));
    }
    if (plan.useDirectRead) {
        CHK_RET(SubmitDirectReads(
            param, resCtx, dataSize, plan.communicationThreadNum));
    } else {
        CHK_RET(SubmitWrites(
            param, resCtx, dataSize, plan.outputOffset,
            plan.communicationThreadNum));
    }
    CHK_RET(CopyLocalSlice(
        param, resCtx, dataSize, plan.outputOffset,
        plan.useDedicatedLocalCopyThread));
    if (!plan.useDirectRead) {
        CHK_RET(WaitForWriteNotifications(
            resCtx, plan.communicationThreadNum));
    }
    if (param.rankSize > 1) {
        CHK_RET(ThreadSyncAfter(resCtx.threads));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
