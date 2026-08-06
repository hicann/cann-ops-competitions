/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
// 特殊优化路径使用的硬件拓扑和缓冲区约束。
constexpr uint64_t BUFFER_ALIGN = 128;
constexpr uint32_t SERVER_NUM = 2;
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t EXPECTED_RANK_SIZE = SERVER_NUM * RANKS_PER_SERVER;
constexpr uint64_t SMALL_HIER_THRESHOLD = 1ULL * 1024 * 1024;
constexpr uint64_t MIB = 1024ULL * 1024;
constexpr uint64_t PIPELINE_DATA_SIZE = 512ULL * MIB;
constexpr uint64_t PIPELINE_FIRST_SIZE = 288ULL * MIB;
constexpr uint64_t PIPELINE_MIDDLE_SIZE = 112ULL * MIB;
constexpr uint64_t PIPELINE_LAST_SIZE = 112ULL * MIB;
constexpr uint64_t EXPECTED_CCL_BUFFER_SIZE = 400ULL * MIB;

// 按字节偏移通信缓冲区指针。
void *AddOffset(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
}

HcclResult SyncThreadsBefore(
    const std::vector<ThreadHandle> &threads, uint32_t activeThreadNum)
{
    CHK_PRT_RET(activeThreadNum == 0 || activeThreadNum > threads.size(),
        HCCL_ERROR("Invalid active thread count %u for %zu threads",
            activeThreadNum, threads.size()),
        HCCL_E_PARA);
    // 并行阶段开始前，由线程 0 唤醒所有通信线程。
    for (uint32_t idx = 1; idx < activeThreadNum; ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[0], threads[idx], 0)));
    }
    for (uint32_t idx = 1; idx < activeThreadNum; ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[idx], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult SyncThreadsAfter(
    const std::vector<ThreadHandle> &threads, uint32_t activeThreadNum)
{
    CHK_PRT_RET(activeThreadNum == 0 || activeThreadNum > threads.size(),
        HCCL_ERROR("Invalid active thread count %u for %zu threads",
            activeThreadNum, threads.size()),
        HCCL_E_PARA);
    // 并行阶段结束后，由线程 0 等待所有通信线程完成。
    for (uint32_t idx = 1; idx < activeThreadNum; ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[idx], threads[0], idx - 1)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[0], idx - 1, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ReadPeer(ThreadHandle thread, const ChannelInfo &channel,
    void *localDst, const void *remoteSrc, uint64_t size)
{
    // 双方将输入写入本地 HCCL 缓冲区后进行同步，确保远端读取的数据已就绪。
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommReadOnThread(thread, channel.handle, localDst, remoteSrc, size)));
    return HCCL_SUCCESS;
}

HcclResult SmallHierarchicalReadAllGather(const OpParam &param,
    const AlgResourceCtx &resCtx, uint64_t processedCount,
    uint64_t sliceCount, uint64_t dataTypeSize)
{
    const uint64_t dataSize = param.count * dataTypeSize;
    const uint64_t sliceSize = sliceCount * dataTypeSize;
    const uint64_t chunkOffset = processedCount * dataTypeSize;
    const void *input = AddOffset(param.inputPtr, chunkOffset);
    void *localSlot = AddOffset(
        resCtx.localBuffer.addr, sliceSize * param.myRank);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[0], localSlot, input, sliceSize)));

    // 使用两个工作线程进行同步，减少小数据场景下的同步开销。
    constexpr uint32_t meshWorkerNum = 2;
    CHK_RET(SyncThreadsBefore(resCtx.threads, meshWorkerNum));

    // 阶段一：向所有通道发送就绪通知，避免串行等待阻塞后续通道。
    for (uint32_t idx = 0; idx < RANKS_PER_SERVER - 1; ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(
                resCtx.threads[idx % meshWorkerNum],
                resCtx.channels[idx].handle, NOTIFY_IDX_ACK)));
    }

    // 阶段二：等待对端就绪，并将各 Rank 数据读入对应的本地 CCL 分区。
    for (uint32_t idx = 0; idx < RANKS_PER_SERVER - 1; ++idx) {
        const ChannelInfo &channel = resCtx.channels[idx];
        const uint32_t peerRank = channel.remoteRank;
        void *localPeerSlot = AddOffset(resCtx.localBuffer.addr,
            static_cast<uint64_t>(peerRank) * sliceSize);
        const void *remotePeerSlot = AddOffset(channel.remoteCclMem.addr,
            static_cast<uint64_t>(peerRank) * sliceSize);
        ThreadHandle thread = resCtx.threads[idx % meshWorkerNum];
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(
                thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommReadOnThread(thread, channel.handle,
                localPeerSlot, remotePeerSlot, sliceSize)));
    }
    CHK_RET(SyncThreadsAfter(resCtx.threads, meshWorkerNum));

    // 两个通信线程汇合后，唤醒本地拷贝线程。
    constexpr uint32_t copyStartNotifyIdx = 1;
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[0], resCtx.threads[1], copyStartNotifyIdx)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[1], copyStartNotifyIdx, CUSTOM_TIMEOUT)));

    // 通道 7 读取远端服务器的 8 Rank 数据，同时线程 1 拷贝本地数据。
    constexpr uint32_t crossIdx = RANKS_PER_SERVER - 1;
    const uint32_t localServerBase =
        param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
    const uint32_t remoteServerBase =
        (localServerBase + RANKS_PER_SERVER) % EXPECTED_RANK_SIZE;
    const uint64_t blockSize = RANKS_PER_SERVER * sliceSize;

    const ChannelInfo &crossChannel = resCtx.channels[crossIdx];
    void *remoteHalfOutput = AddOffset(param.outputPtr,
        static_cast<uint64_t>(remoteServerBase) * dataSize + chunkOffset);
    const void *remoteBlock = AddOffset(crossChannel.remoteCclMem.addr,
        static_cast<uint64_t>(remoteServerBase) * sliceSize);
    CHK_RET(ReadPeer(resCtx.threads[0], crossChannel,
        remoteHalfOutput, remoteBlock, blockSize));

    void *localHalfOutput = AddOffset(param.outputPtr,
        static_cast<uint64_t>(localServerBase) * dataSize + chunkOffset);
    const void *localBlock = AddOffset(resCtx.localBuffer.addr,
        static_cast<uint64_t>(localServerBase) * sliceSize);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[1], localHalfOutput, localBlock, blockSize)));

    constexpr uint32_t copyDoneNotifyIdx = 1;
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[1], resCtx.threads[0], copyDoneNotifyIdx)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[0], copyDoneNotifyIdx, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ReadDirectAllGather(const OpParam &param,
    const AlgResourceCtx &resCtx, uint64_t processedCount,
    uint64_t sliceCount, uint64_t dataTypeSize)
{
    const uint64_t dataSize = param.count * dataTypeSize;
    const uint64_t sliceSize = sliceCount * dataTypeSize;
    const uint64_t chunkOffset = processedCount * dataTypeSize;
    const void *input = AddOffset(param.inputPtr, chunkOffset);
    const uint32_t copyThreadIdx = resCtx.channels.size();

    // 将专用拷贝线程加入任务流，使本地输出拷贝可与 CCL 缓冲区拷贝并行。
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[0], resCtx.threads[copyThreadIdx], 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[copyThreadIdx], 0, CUSTOM_TIMEOUT)));

    // 所有对端读取同一分片，因此无需按 Rank 划分本地 CCL 缓冲区。
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[0], resCtx.localBuffer.addr, input, sliceSize)));

    void *localOutput = AddOffset(
        param.outputPtr, param.myRank * dataSize + chunkOffset);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[copyThreadIdx], localOutput, input, sliceSize)));

    const uint32_t activeThreadNum = resCtx.channels.size();
    CHK_RET(SyncThreadsBefore(resCtx.threads, activeThreadNum));
    for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
        const ChannelInfo &channel = resCtx.channels[idx];
        void *peerOutput = AddOffset(param.outputPtr,
            channel.remoteRank * dataSize + chunkOffset);
        CHK_RET(ReadPeer(resCtx.threads[idx], channel, peerOutput,
            channel.remoteCclMem.addr, sliceSize));
    }
    CHK_RET(SyncThreadsAfter(resCtx.threads, activeThreadNum));

    // 所有通信线程结束后，等待专用本地拷贝线程完成。
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[copyThreadIdx], resCtx.threads[0], activeThreadNum - 1)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[0], activeThreadNum - 1, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult BatchReadTwoTailSlices(ThreadHandle thread,
    const ChannelInfo &channel, void *outputBase)
{
    // 将两个尾部分片批量读取到最终输出位置。
    HcommBatchTransferDesc descs[2] = {};
    descs[0].transType = HCOMM_TRANSFER_TYPE_READ;
    descs[0].transferInfo.read.len = PIPELINE_MIDDLE_SIZE;
    descs[0].transferInfo.read.src = AddOffset(
        channel.remoteCclMem.addr, PIPELINE_FIRST_SIZE);
    descs[0].transferInfo.read.dst = AddOffset(
        outputBase, PIPELINE_FIRST_SIZE);

    descs[1].transType = HCOMM_TRANSFER_TYPE_READ;
    descs[1].transferInfo.read.len = PIPELINE_LAST_SIZE;
    descs[1].transferInfo.read.src = channel.remoteCclMem.addr;
    descs[1].transferInfo.read.dst = AddOffset(outputBase,
        PIPELINE_FIRST_SIZE + PIPELINE_MIDDLE_SIZE);
    CHK_RET(static_cast<HcclResult>(HcommBatchTransferOnThread(
        thread, channel.handle, descs, 2)));
    return HCCL_SUCCESS;
}

HcclResult ReadDirectTwoRoundBatchTail512(const OpParam &param,
    const AlgResourceCtx &resCtx, uint64_t dataTypeSize)
{
    const uint64_t dataSize = param.count * dataTypeSize;
    const uint32_t activeThreadNum = resCtx.channels.size();
    const uint32_t copyThreadIdx = activeThreadNum;

    CHK_PRT_RET(dataSize != PIPELINE_DATA_SIZE ||
        resCtx.localBuffer.size != EXPECTED_CCL_BUFFER_SIZE ||
        PIPELINE_FIRST_SIZE + PIPELINE_MIDDLE_SIZE + PIPELINE_LAST_SIZE != dataSize ||
        PIPELINE_FIRST_SIZE + PIPELINE_MIDDLE_SIZE != resCtx.localBuffer.size,
        HCCL_ERROR("Invalid 512MiB two-round batch-tail parameters"), HCCL_E_PARA);

    // 拷贝线程先加入任务流，再等待 15 个对端通道的数据就绪信号。
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[0], resCtx.threads[copyThreadIdx], 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[copyThreadIdx], 0, CUSTOM_TIMEOUT)));
    for (uint32_t idx = 0; idx < activeThreadNum; ++idx) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            resCtx.threads[copyThreadIdx], resCtx.channels[idx].handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }

    // 第一次发布将 A（288MiB）和 B（112MiB）写入 400MiB CCL 缓冲区，
    // 通信线程先读取 A。
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[0], resCtx.localBuffer.addr,
        param.inputPtr, EXPECTED_CCL_BUFFER_SIZE)));
    CHK_RET(SyncThreadsBefore(resCtx.threads, activeThreadNum));
    for (uint32_t idx = 0; idx < activeThreadNum; ++idx) {
        const ChannelInfo &channel = resCtx.channels[idx];
        void *peerOutput = AddOffset(param.outputPtr,
            channel.remoteRank * dataSize);
        CHK_RET(ReadPeer(resCtx.threads[idx], channel, peerOutput,
            channel.remoteCclMem.addr, PIPELINE_FIRST_SIZE));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            resCtx.threads[idx], channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            resCtx.threads[idx], channel.handle, NOTIFY_IDX_DATA_SIGNAL,
            CUSTOM_TIMEOUT)));
    }

    // 所有对端读完 A 后，用 C 覆盖 CCL 的前 112MiB，并通知对端继续读取。
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[copyThreadIdx], resCtx.localBuffer.addr,
        AddOffset(param.inputPtr,
            PIPELINE_FIRST_SIZE + PIPELINE_MIDDLE_SIZE),
        PIPELINE_LAST_SIZE)));
    for (uint32_t idx = 0; idx < activeThreadNum; ++idx) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            resCtx.threads[copyThreadIdx], resCtx.channels[idx].handle,
            NOTIFY_IDX_DATA_SIGNAL)));
    }

    // 第二轮通过一次批量传输读取不连续的 B、C 两个尾部分片。
    for (uint32_t idx = 0; idx < activeThreadNum; ++idx) {
        const ChannelInfo &channel = resCtx.channels[idx];
        void *peerOutput = AddOffset(param.outputPtr,
            channel.remoteRank * dataSize);
        CHK_RET(BatchReadTwoTailSlices(
            resCtx.threads[idx], channel, peerOutput));
    }

    // 本地完整输出拷贝与第二轮网络传输并行执行，最后统一等待完成。
    void *localOutput = AddOffset(param.outputPtr, param.myRank * dataSize);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        resCtx.threads[copyThreadIdx], localOutput, param.inputPtr, dataSize)));
    CHK_RET(SyncThreadsAfter(resCtx.threads, activeThreadNum));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[copyThreadIdx], resCtx.threads[0], activeThreadNum - 1)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[0], activeThreadNum - 1, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing read-direct AllGather on Ascend NPU");

    // 下发设备任务前，先校验数据类型和通信资源。
    const auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type %d", static_cast<int32_t>(param.dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(resCtx.threads.empty(),
        HCCL_ERROR("No AICPU thread was allocated"), HCCL_E_INTERNAL);

    const uint64_t dataTypeSize = sizeIter->second;
    const uint64_t dataSize = param.count * dataTypeSize;
    // 单 Rank 场景直接将输入拷贝到输出。
    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize)));
        return HCCL_SUCCESS;
    }

    const uint32_t expectedChannelNum = param.rankSize - 1;
    CHK_PRT_RET(resCtx.channels.size() != expectedChannelNum,
        HCCL_ERROR("Channel count %zu does not match expected count %u",
            resCtx.channels.size(), expectedChannelNum),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.threads.size() != resCtx.channels.size() + 1,
        HCCL_ERROR("Thread count %zu does not match channel count plus copy thread %zu",
            resCtx.threads.size(), resCtx.channels.size() + 1),
        HCCL_E_INTERNAL);

    const uint64_t bytesPerSlice =
        (resCtx.localBuffer.size / BUFFER_ALIGN) * BUFFER_ALIGN;
    CHK_PRT_RET(bytesPerSlice < dataTypeSize,
        HCCL_ERROR("HCCL buffer %llu is too small",
            static_cast<unsigned long long>(resCtx.localBuffer.size)),
        HCCL_E_MEMORY);
    const uint64_t maxCountPerLoop = bytesPerSlice / dataTypeSize;

    // 16 Rank、512MiB 固定场景使用两轮流水优化路径。
    if (param.rankSize == EXPECTED_RANK_SIZE &&
        dataSize == PIPELINE_DATA_SIZE &&
        resCtx.localBuffer.size == EXPECTED_CCL_BUFFER_SIZE) {
        return ReadDirectTwoRoundBatchTail512(param, resCtx, dataTypeSize);
    }

    // 按本地 HCCL 缓冲区容量对数据进行分片处理。
    for (uint64_t processedCount = 0; processedCount < param.count;) {
        const uint64_t sliceCount =
            std::min(maxCountPerLoop, param.count - processedCount);
        // 16 Rank 小数据使用分层聚合，其余场景直接从各对端读取。
        if (param.rankSize == EXPECTED_RANK_SIZE &&
            dataSize <= SMALL_HIER_THRESHOLD) {
            CHK_RET(SmallHierarchicalReadAllGather(
                param, resCtx, processedCount, sliceCount, dataTypeSize));
        } else {
            CHK_RET(ReadDirectAllGather(
                param, resCtx, processedCount, sliceCount, dataTypeSize));
        }
        processedCount += sliceCount;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
