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
#include <cstddef>
#include <cstdint>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_SLICE_BYTES = 256ULL * 1024 * 1024;
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t TOPO_RANK_SIZE = 2 * SERVER_RANK_SIZE;
constexpr uint32_t SMALL_CHANNEL_NUM = 4;
constexpr uint64_t SMALL_MESSAGE_BYTES = 512ULL * 1024;
constexpr uint64_t LARGE_TWO_SLICE_BYTES = 2 * MAX_SLICE_BYTES;

// VM Checker 要求所有通信 thread 由主 thread 显式放行；这组边同时
// 构成每轮跨 rank Record/Wait 匹配的本地起点，不可省略。
HcclResult SyncThreadsBefore(const std::vector<ThreadHandle> &threads)
{
    for (std::size_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[i], 0)));
    }
    for (std::size_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[i], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

// 主 thread 等到所有通信 thread 完成本轮交换及输出拷贝后才进入下一轮。
HcclResult SyncThreadsAfter(const std::vector<ThreadHandle> &threads)
{
    for (std::size_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[0], static_cast<uint32_t>(i - 1), CUSTOM_TIMEOUT)));
    }
    for (std::size_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[i], threads[0], static_cast<uint32_t>(i - 1))));
    }
    return HCCL_SUCCESS;
}

// 小包目标是已注册的最终output，没有槽位复用，因而不需要写前ACK。
HcclResult PostSliceSmall(ThreadHandle thread, const ChannelInfo &channel, void *remoteDst,
    const void *localSrc, uint64_t bytes)
{
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle, remoteDst, localSrc, bytes)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult WaitSliceSmall(ThreadHandle thread, const ChannelInfo &channel)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ExchangeSliceSmall(ThreadHandle thread, const ChannelInfo &channel, void *remoteDst,
    const void *localSrc, uint64_t bytes)
{
    CHK_RET(PostSliceSmall(thread, channel, remoteDst, localSrc, bytes));
    return WaitSliceSmall(thread, channel);
}

// 大包已经直接写入建链时注册并交换的远端最终output，每个切片写入独立
// 地址，不再复用CCL槽位，因此写前ACK没有数据依赖。保留兼容性更好的
// Write + DATA Record/Wait，确保每个切片返回前对端数据已经完成。
HcclResult ExchangeSliceLarge(ThreadHandle thread, const ChannelInfo &channel, void *remoteDst,
    const void *localSrc, uint64_t bytes)
{
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle, remoteDst, localSrc, bytes)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}
} // namespace

// 两级 AllGather：先在每个 8-rank Server 内 Full-Mesh 聚合，再由相同
// local-rank 的两个 NPU 通过高带宽 Clos 链路交换完整的 8-rank 数据块。
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const auto typeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type"), HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = typeIt->second;
    CHK_PRT_RET(param.count > UINT64_MAX / typeSize, HCCL_ERROR("Input byte size overflow"), HCCL_E_PARA);
    const uint64_t inputBytes = param.count * typeSize;

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No AICPU thread resource"), HCCL_E_INTERNAL);
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank[%u]/rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(inputBytes > UINT64_MAX / param.rankSize, HCCL_ERROR("Output byte size overflow"), HCCL_E_PARA);

    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, inputBytes));
    }
    CHK_PRT_RET(param.rankSize != TOPO_RANK_SIZE,
        HCCL_ERROR("Hierarchical AllGather requires rankSize[%u], got[%u]", TOPO_RANK_SIZE, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    const bool useSmallAlgorithm = inputBytes <= SMALL_MESSAGE_BYTES;
    const std::size_t expectedChannelNum = useSmallAlgorithm ? SMALL_CHANNEL_NUM : param.rankSize - 1;
    const std::size_t expectedThreadNum = useSmallAlgorithm ? 1 : param.rankSize - 1;
    CHK_PRT_RET(resCtx.channels.size() != expectedChannelNum || resCtx.threads.size() != expectedThreadNum,
        HCCL_ERROR("Resource mismatch: channels[%zu]/expected[%zu], threads[%zu]/expected[%zu]",
            resCtx.channels.size(), expectedChannelNum, resCtx.threads.size(), expectedThreadNum),
        HCCL_E_INTERNAL);

    const uint32_t serverBase = param.myRank / SERVER_RANK_SIZE * SERVER_RANK_SIZE;
    if (useSmallAlgorithm) {
        const uint32_t crossPeer = param.myRank < SERVER_RANK_SIZE ? param.myRank + SERVER_RANK_SIZE
                                                                   : param.myRank - SERVER_RANK_SIZE;
        CHK_PRT_RET(resCtx.channels[0].remoteRank != crossPeer,
            HCCL_ERROR("Invalid cross-server peer[%u], expected[%u]", resCtx.channels[0].remoteRank, crossPeer),
            HCCL_E_INTERNAL);
    }

    // 大消息直接写入每个 peer 的最终 rank-major output 槽。这样既消除了
    // CCL -> recvBuf 的第二次整包搬运，也不再受 CCL/rankSize 分槽容量限制；
    // 512MB继续使用两次256MB逻辑Write；不足512MB的大消息合并成一次
    // 逻辑Write，由底层UB连接在内部按硬件maxWriteSize自动拆分WQE。
    if (!useSmallAlgorithm) {
        const bool useSingleLogicalWrite = inputBytes < LARGE_TWO_SLICE_BYTES;
        const uint64_t maxSliceBytes = useSingleLogicalWrite
                                           ? inputBytes
                                           : MAX_SLICE_BYTES / typeSize * typeSize;
        const uint64_t maxSliceCount = maxSliceBytes / typeSize;
        const uint64_t outputBytes = inputBytes * param.rankSize;
        CHK_RET(SyncThreadsBefore(resCtx.threads));
        for (std::size_t i = 0; i < resCtx.channels.size(); ++i) {
            const ChannelInfo &channel = resCtx.channels[i];
            CHK_PRT_RET(channel.remoteRank >= param.rankSize,
                HCCL_ERROR("Invalid remote rank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
            CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size < outputBytes,
                HCCL_ERROR("Remote output buffer[%llu] is smaller than required[%llu] for rank[%u]",
                    static_cast<unsigned long long>(channel.remoteCclMem.size),
                    static_cast<unsigned long long>(outputBytes), channel.remoteRank),
                HCCL_E_INTERNAL);

            uint64_t peerProcessedCount = 0;
            while (peerProcessedCount < param.count) {
                const uint64_t sliceCount = std::min(maxSliceCount, param.count - peerProcessedCount);
                const uint64_t sliceBytes = sliceCount * typeSize;
                const uint64_t inputOffset = peerProcessedCount * typeSize;
                uint8_t *input = static_cast<uint8_t *>(param.inputPtr) + inputOffset;
                uint8_t *remoteOutput = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                                        static_cast<uint64_t>(param.myRank) * inputBytes + inputOffset;

                CHK_RET(ExchangeSliceLarge(resCtx.threads[i], channel, remoteOutput, input, sliceBytes));
                peerProcessedCount += sliceCount;
            }

            // 400MB+4B时main thread还承担全部从thread的放行与回收，
            // 不再给它追加self copy；14条slave thread各复制1/14，
            // 使main在peer交换完成后立即进入结束汇合。512MB仍严格
            // 保留原来的15-thread平均分配任务图。
            if (!useSingleLogicalWrite || i != 0) {
                const uint64_t selfThreadCount =
                    useSingleLogicalWrite ? resCtx.threads.size() - 1 : resCtx.threads.size();
                const uint64_t selfThreadIndex = useSingleLogicalWrite ? i - 1 : i;
                const uint64_t selfBaseCount = param.count / selfThreadCount;
                const uint64_t selfRemainder = param.count % selfThreadCount;
                const uint64_t selfBeginCount =
                    selfBaseCount * selfThreadIndex + std::min<uint64_t>(selfRemainder, selfThreadIndex);
                const uint64_t selfCount =
                    selfBaseCount + (selfThreadIndex < selfRemainder ? 1 : 0);
                if (selfCount != 0) {
                    const uint64_t selfOffset = selfBeginCount * typeSize;
                    uint8_t *selfInput = static_cast<uint8_t *>(param.inputPtr) + selfOffset;
                    uint8_t *selfOutput = static_cast<uint8_t *>(param.outputPtr) +
                                          static_cast<uint64_t>(param.myRank) * inputBytes + selfOffset;
                    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                        resCtx.threads[i], selfOutput, selfInput, selfCount * typeSize)));
                }
            }
        }
        CHK_RET(SyncThreadsAfter(resCtx.threads));
        return HCCL_SUCCESS;
    }

    // 小消息使用机内递归倍增：xor1交换1个rank块、xor2交换2个rank块、
    // xor4交换4个rank块；随后对应local-rank通过Clos交换完整8-rank块。
    // 只需4条channel和1个thread，不产生跨thread同步任务。
    const uint64_t outputBytes = inputBytes * param.rankSize;
    CHK_RET(SyncThreadsBefore(resCtx.threads));

    const uint32_t localRank = param.myRank - serverBase;
    const ChannelInfo &firstChannel = resCtx.channels[1];
    const uint32_t firstPeer = serverBase + (localRank ^ 1U);
    CHK_PRT_RET(firstChannel.remoteRank != firstPeer,
        HCCL_ERROR("Invalid xor-1 peer[%u], expected[%u]", firstChannel.remoteRank, firstPeer),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(firstChannel.remoteCclMem.addr == nullptr || firstChannel.remoteCclMem.size < outputBytes,
        HCCL_ERROR("Remote output buffer is invalid for xor-1 peer[%u]", firstChannel.remoteRank),
        HCCL_E_INTERNAL);

    // 先启动xor-1写，再做self copy，最后等待对端数据，使本地copy与第一步
    // 网络传输重叠。两端Wait完成后，各rank已拥有连续的2-rank块。
    uint8_t *remoteSelfSlot = static_cast<uint8_t *>(firstChannel.remoteCclMem.addr) +
                              static_cast<uint64_t>(param.myRank) * inputBytes;
    CHK_RET(PostSliceSmall(
        resCtx.threads[0], firstChannel, remoteSelfSlot, param.inputPtr, inputBytes));
    uint8_t *selfOutput = static_cast<uint8_t *>(param.outputPtr) +
                          static_cast<uint64_t>(param.myRank) * inputBytes;
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(resCtx.threads[0], selfOutput, param.inputPtr, inputBytes)));
    CHK_RET(WaitSliceSmall(resCtx.threads[0], firstChannel));

    std::size_t channelIdx = 2;
    for (uint32_t distance = 2; distance < SERVER_RANK_SIZE; distance <<= 1, ++channelIdx) {
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        const uint32_t expectedPeer = serverBase + (localRank ^ distance);
        CHK_PRT_RET(channel.remoteRank != expectedPeer,
            HCCL_ERROR("Invalid xor peer[%u], expected[%u]", channel.remoteRank, expectedPeer),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size < outputBytes,
            HCCL_ERROR("Remote output buffer[%llu] is smaller than required[%llu] for rank[%u]",
                static_cast<unsigned long long>(channel.remoteCclMem.size),
                static_cast<unsigned long long>(outputBytes), channel.remoteRank),
            HCCL_E_INTERNAL);

        const uint32_t groupStartRank = serverBase + localRank / distance * distance;
        const uint64_t groupBytes = static_cast<uint64_t>(distance) * inputBytes;
        uint8_t *localGroup = static_cast<uint8_t *>(param.outputPtr) +
                              static_cast<uint64_t>(groupStartRank) * inputBytes;
        uint8_t *remoteGroup = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                               static_cast<uint64_t>(groupStartRank) * inputBytes;
        CHK_RET(ExchangeSliceSmall(
            resCtx.threads[0], channel, remoteGroup, localGroup, groupBytes));
    }
    CHK_RET(SyncThreadsAfter(resCtx.threads));

    const ChannelInfo &crossChannel = resCtx.channels[0];
    CHK_PRT_RET(crossChannel.remoteCclMem.addr == nullptr || crossChannel.remoteCclMem.size < outputBytes,
        HCCL_ERROR("Remote cross output buffer[%llu] is smaller than required[%llu]",
            static_cast<unsigned long long>(crossChannel.remoteCclMem.size),
            static_cast<unsigned long long>(outputBytes)),
        HCCL_E_INTERNAL);
    const uint64_t groupBytes = static_cast<uint64_t>(SERVER_RANK_SIZE) * inputBytes;
    uint8_t *localGroup = static_cast<uint8_t *>(param.outputPtr) +
                          static_cast<uint64_t>(serverBase) * inputBytes;
    uint8_t *remoteGroup = static_cast<uint8_t *>(crossChannel.remoteCclMem.addr) +
                           static_cast<uint64_t>(serverBase) * inputBytes;
    CHK_RET(ExchangeSliceSmall(resCtx.threads[0], crossChannel, remoteGroup, localGroup, groupBytes));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
