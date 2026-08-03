/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "custom.h"
#include "log.h"
#include "exec_op.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ops_hccl {
namespace {

// A single transport task is limited to 256 MiB.  In the 16-rank case the
// 400 MiB CCL buffer is the tighter bound (26.67 MiB for each peer slot).
constexpr uint64_t MAX_TRANSFER_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t DIRECT_READ_THRESHOLD = 1ULL * 1024ULL * 1024ULL;
constexpr uint64_t MIN_SLICE_ALIGNMENT = 128;
// The extra alignment unit prevents 400 MiB + 4 B from becoming a fourth,
// latency-only 4-byte transport round after 128 MiB + 256 MiB + 16 MiB.
constexpr uint64_t PIPELINE_FIRST_BUFFER_BYTES = 128ULL * 1024ULL * 1024ULL + MIN_SLICE_ALIGNMENT;
constexpr uint32_t CHANNEL_READY = 0;
constexpr uint32_t CHANNEL_DATA = 1;

// 递归加倍算法常量
constexpr uint64_t SLICE_ALIGN_BYTES = 128;
constexpr uint64_t SMALL_MESSAGE_THRESHOLD = 1ULL * 1024 * 1024;     // ≤1MB 走递归加倍
constexpr uint64_t RECURSIVE_MAX_CHUNK_BYTES = 4ULL * 1024 * 1024;   // 递归加倍每轮最大分片

inline uint8_t *AddOffset(void *ptr, uint64_t offset)
{
    return static_cast<uint8_t *>(ptr) + offset;
}

inline const uint8_t *AddOffset(const void *ptr, uint64_t offset)
{
    return static_cast<const uint8_t *>(ptr) + offset;
}

// Rank slot in a layout that omits ownerRank's local contribution.
inline uint32_t PeerSlotIndex(uint32_t rank, uint32_t ownerRank)
{
    return rank < ownerRank ? rank : rank - 1;
}

HcclResult ValidateAndGetSize(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t &dataBytes)
{
    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end() || sizeIt->second == 0,
        HCCL_ERROR("Unsupported AllGather data type[%d]", static_cast<int>(param.dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank information rank[%u] size[%u]", param.myRank, param.rankSize), HCCL_E_INTERNAL);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeIt->second,
        HCCL_ERROR("AllGather input size overflow"), HCCL_E_PARA);

    dataBytes = param.count * sizeIt->second;
    CHK_PRT_RET(dataBytes != 0 && param.rankSize > std::numeric_limits<uint64_t>::max() / dataBytes,
        HCCL_ERROR("AllGather output size overflow"), HCCL_E_PARA);

    if (param.rankSize == 1 || param.count == 0) {
        return HCCL_SUCCESS;
    }

    const uint32_t peerCount = param.rankSize - 1;
    CHK_PRT_RET(resCtx.threads.size() < param.rankSize || resCtx.channels.size() != peerCount,
        HCCL_ERROR("Invalid AllGather resources threads[%zu] channels[%zu]", resCtx.threads.size(),
            resCtx.channels.size()), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr ||
            resCtx.localBuffer.size < static_cast<uint64_t>(peerCount) * sizeIt->second,
        HCCL_ERROR("Local CCL buffer is too small: size[%llu]",
            static_cast<unsigned long long>(resCtx.localBuffer.size)), HCCL_E_INTERNAL);

    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank ||
                channel.remoteCclMem.addr == nullptr,
            HCCL_ERROR("Invalid channel for remote rank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult ExecDirectRead(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataBytes)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    const uint32_t peerCount = param.rankSize - 1;
    uint64_t commonBufferBytes = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        commonBufferBytes = std::min(commonBufferBytes, channel.remoteCclMem.size);
    }
    uint64_t firstBufferBytes = std::min(PIPELINE_FIRST_BUFFER_BYTES, commonBufferBytes / 2);
    firstBufferBytes = (firstBufferBytes / MIN_SLICE_ALIGNMENT) * MIN_SLICE_ALIGNMENT;
    uint64_t secondBufferBytes = std::min(MAX_TRANSFER_BYTES, commonBufferBytes - firstBufferBytes);
    secondBufferBytes = (secondBufferBytes / MIN_SLICE_ALIGNMENT) * MIN_SLICE_ALIGNMENT;
    CHK_PRT_RET(firstBufferBytes == 0 || secondBufferBytes == 0,
        HCCL_ERROR("CCL buffer cannot hold two aligned read slices"), HCCL_E_INTERNAL);

    const uint64_t bufferOffsets[2] = {0, firstBufferBytes};
    const uint64_t bufferCapacities[2] = {firstBufferBytes, secondBufferBytes};

    auto StageSlice = [&](uint32_t bufferIndex, uint64_t offset, uint64_t bytes) -> HcclResult {
        void *stagingBuffer = AddOffset(resCtx.localBuffer.addr, bufferOffsets[bufferIndex]);
        const void *input = AddOffset(param.inputPtr, offset);
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(mainThread, stagingBuffer, input, bytes));
    };

    uint64_t processedBytes = 0;
    uint32_t bufferIndex = 0;
    uint64_t sliceBytes = std::min(bufferCapacities[bufferIndex], dataBytes);
    CHK_RET(StageSlice(bufferIndex, processedBytes, sliceBytes));

    while (processedBytes < dataBytes) {
        // The start notify makes the local staging copy visible to every
        // worker.  Each worker owns exactly one peer channel.
        for (uint32_t i = 0; i < peerCount; ++i) {
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[i + 1], 0)));
        }

        for (uint32_t i = 0; i < peerCount; ++i) {
            const ChannelInfo &channel = resCtx.channels[i];
            const ThreadHandle thread = resCtx.threads[i + 1];
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(thread, 0, CUSTOM_TIMEOUT)));

            // READY proves the peer has staged this slice.  DATA is recorded
            // after Read and therefore proves that our CCL slot may be reused.
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, CHANNEL_READY)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, CHANNEL_READY, CUSTOM_TIMEOUT)));

            void *peerOutput = AddOffset(param.outputPtr,
                static_cast<uint64_t>(channel.remoteRank) * dataBytes + processedBytes);
            const void *remoteStagingBuffer = AddOffset(
                channel.remoteCclMem.addr, bufferOffsets[bufferIndex]);
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
                thread, channel.handle, peerOutput, remoteStagingBuffer, sliceBytes)));

            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, CHANNEL_DATA)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, CHANNEL_DATA, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(thread, mainThread, i)));
        }

        void *selfOutput = AddOffset(param.outputPtr,
            static_cast<uint64_t>(param.myRank) * dataBytes + processedBytes);
        const void *input = AddOffset(param.inputPtr, processedBytes);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(mainThread, selfOutput, input, sliceBytes)));

        const uint64_t nextOffset = processedBytes + sliceBytes;
        const uint32_t nextBufferIndex = bufferIndex ^ 1U;
        uint64_t nextSliceBytes = 0;
        if (nextOffset < dataBytes) {
            nextSliceBytes = std::min(bufferCapacities[nextBufferIndex], dataBytes - nextOffset);
            CHK_RET(StageSlice(nextBufferIndex, nextOffset, nextSliceBytes));
        }

        for (uint32_t i = 0; i < peerCount; ++i) {
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(mainThread, i, CUSTOM_TIMEOUT)));
        }
        processedBytes = nextOffset;
        bufferIndex = nextBufferIndex;
        sliceBytes = nextSliceBytes;
    }
    return HCCL_SUCCESS;
}

HcclResult ExecDirectWrite(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataBytes)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    const uint64_t typeSize = SIZE_TABLE.at(param.dataType);
    const uint32_t peerCount = param.rankSize - 1;

    uint64_t commonBufferBytes = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        commonBufferBytes = std::min(commonBufferBytes, channel.remoteCclMem.size);
    }
    uint64_t maxSliceBytes = std::min(MAX_TRANSFER_BYTES, commonBufferBytes / peerCount);
    maxSliceBytes = (maxSliceBytes / MIN_SLICE_ALIGNMENT) * MIN_SLICE_ALIGNMENT;
    maxSliceBytes = (maxSliceBytes / typeSize) * typeSize;
    CHK_PRT_RET(maxSliceBytes == 0, HCCL_ERROR("CCL buffer cannot hold one aligned write slice"), HCCL_E_INTERNAL);

    uint64_t processedBytes = 0;
    for (uint32_t i = 0; i < peerCount; ++i) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[i + 1], 0)));
    }
    while (processedBytes < dataBytes) {
        const uint64_t sliceBytes = std::min(maxSliceBytes, dataBytes - processedBytes);
        const void *input = AddOffset(param.inputPtr, processedBytes);

        for (uint32_t i = 0; i < peerCount; ++i) {
            const ChannelInfo &channel = resCtx.channels[i];
            const ThreadHandle thread = resCtx.threads[i + 1];

            if (processedBytes == 0) {
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(thread, 0, CUSTOM_TIMEOUT)));
            }

            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, CHANNEL_READY)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, CHANNEL_READY, CUSTOM_TIMEOUT)));

            const uint32_t remoteSlotIndex = PeerSlotIndex(param.myRank, channel.remoteRank);
            void *remoteSlot = AddOffset(channel.remoteCclMem.addr,
                static_cast<uint64_t>(remoteSlotIndex) * sliceBytes);
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                thread, channel.handle, remoteSlot, input, sliceBytes)));

            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, CHANNEL_DATA)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, CHANNEL_DATA, CUSTOM_TIMEOUT)));

            const uint32_t localSlotIndex = PeerSlotIndex(channel.remoteRank, param.myRank);
            const void *localPeerSlot = AddOffset(resCtx.localBuffer.addr,
                static_cast<uint64_t>(localSlotIndex) * sliceBytes);
            void *peerOutput = AddOffset(param.outputPtr,
                static_cast<uint64_t>(channel.remoteRank) * dataBytes + processedBytes);
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, peerOutput, localPeerSlot, sliceBytes)));
        }

        void *selfOutput = AddOffset(param.outputPtr,
            static_cast<uint64_t>(param.myRank) * dataBytes + processedBytes);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(mainThread, selfOutput, input, sliceBytes)));
        processedBytes += sliceBytes;
    }

    for (uint32_t i = 0; i < peerCount; ++i) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[i + 1], mainThread, i)));
    }
    for (uint32_t i = 0; i < peerCount; ++i) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(mainThread, i, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

// ---- 递归加倍 AllGather：小包 O(log₂N) 轮次，延迟极低 ----
bool IsPowerOfTwo(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

const ChannelInfo *GetChannelByRank(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    if (remoteRank >= resCtx.rankToChannelIdx.size()) {
        return nullptr;
    }
    uint32_t channelIdx = resCtx.rankToChannelIdx[remoteRank];
    if (channelIdx == INVALID_VALUE_RANKID || channelIdx >= resCtx.channels.size()) {
        return nullptr;
    }
    return &resCtx.channels[channelIdx];
}

uint64_t AlignDown(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return value;
    }
    return value / align * align;
}

uint64_t SelectChunkBytes(uint64_t preferredBytes, uint64_t capacityBytes)
{
    return AlignDown(std::min(preferredBytes, capacityBytes), SLICE_ALIGN_BYTES);
}

// 写交换：ACK 同步 → RDMA Write → DATA_SIGNAL 同步
HcclResult WriteExchange(ThreadHandle thread, const ChannelInfo &channel, const void *sendSrc, void *remoteDst,
    uint64_t bytes)
{
    if (bytes == 0) {
        return HCCL_SUCCESS;
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle, remoteDst, sendSrc, bytes)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

// 递归加倍 AllGather：每轮与距离为 delta 的伙伴交换已聚合数据
HcclResult RecursiveDoublingAllGather(ThreadHandle thread, const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t totalDataSize)
{
    uint32_t maxPeerGroup = std::max<uint32_t>(1, param.rankSize / 2);
    uint64_t capacityBytes = resCtx.localBuffer.size / maxPeerGroup;
    uint64_t maxChunkBytes = SelectChunkBytes(RECURSIVE_MAX_CHUNK_BYTES, capacityBytes);
    CHK_PRT_RET(maxChunkBytes == 0, HCCL_ERROR("RecursiveDoublingAllGather: local buffer too small"), HCCL_E_INTERNAL);

    uint64_t processedBytes = 0;
    while (processedBytes < totalDataSize) {
        uint64_t chunkBytes = std::min(maxChunkBytes, totalDataSize - processedBytes);

        // 将自己的分片拷贝到输出
        void *dst = AddOffset(param.outputPtr, static_cast<uint64_t>(param.myRank) * totalDataSize + processedBytes);
        const void *src = AddOffset(param.inputPtr, processedBytes);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, chunkBytes)));

        // 递归加倍：每轮与距离为 delta 的伙伴交换已聚合数据
        for (uint32_t delta = 1; delta < param.rankSize; delta <<= 1) {
            uint32_t partner = param.myRank ^ delta;
            const ChannelInfo *channel = GetChannelByRank(resCtx, partner);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("RecursiveDoublingAllGather: missing channel to rank[%u]", partner), HCCL_E_INTERNAL);

            uint32_t groupBase = (param.myRank / (delta << 1)) * (delta << 1);
            bool isLowerHalf = ((param.myRank / delta) % 2) == 0;
            uint32_t sendBaseRank = isLowerHalf ? groupBase : (groupBase + delta);
            uint32_t recvBaseRank = isLowerHalf ? (groupBase + delta) : groupBase;
            uint64_t exchangeBytes = static_cast<uint64_t>(delta) * chunkBytes;

            const void *sendSrc =
                AddOffset(param.outputPtr, static_cast<uint64_t>(sendBaseRank) * totalDataSize + processedBytes);
            void *recvDst =
                AddOffset(param.outputPtr, static_cast<uint64_t>(recvBaseRank) * totalDataSize + processedBytes);
            CHK_RET(WriteExchange(thread, *channel, sendSrc, channel->remoteCclMem.addr, exchangeBytes));
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, recvDst, resCtx.localBuffer.addr, exchangeBytes)));
        }
        processedBytes += chunkBytes;
    }
    return HCCL_SUCCESS;
}

} // namespace

// 融合策略：小包走递归加倍（O(log₂N) 轮次），大包走双缓冲直读，中包走并发直写
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    uint64_t dataBytes = 0;
    CHK_RET(ValidateAndGetSize(param, resCtx, dataBytes));
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    const ThreadHandle mainThread = resCtx.aicpuThread;
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(
            mainThread, param.outputPtr, param.inputPtr, dataBytes));
    }

    // 融合策略：
    // 1）小包 ≤1MB + 2的幂：递归加倍（O(log₂N) 轮次，延迟极低）
    if (dataBytes <= SMALL_MESSAGE_THRESHOLD && IsPowerOfTwo(param.rankSize)) {
        HCCL_INFO("ExecOp select [recursive_doubling] small msg rank=%u size=%llu",
            param.myRank, static_cast<unsigned long long>(dataBytes));
        return RecursiveDoublingAllGather(mainThread, param, resCtx, dataBytes);
    }

    // 2）大包 >1MB：双缓冲直读
    if (dataBytes > DIRECT_READ_THRESHOLD) {
        return ExecDirectRead(param, resCtx, dataBytes);
    }

    // 3）兜底：并发直写
    return ExecDirectWrite(param, resCtx, dataBytes);
}
} // namespace ops_hccl
