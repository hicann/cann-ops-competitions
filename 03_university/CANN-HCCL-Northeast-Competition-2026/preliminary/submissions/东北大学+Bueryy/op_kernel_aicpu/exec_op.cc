/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <limits>
#include <vector>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace {
// 分摊路径的叶子主动读取root部分，只等待邻居后段通知；非分摊路径保留root通知。
constexpr uint32_t NOTIFY_FROM_ROOT = 0;   // root 投递「对端自己那份」完成
constexpr uint32_t NOTIFY_FROM_RELAY = 2;  // 邻居代发转发完成

void *AddOffset(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
}

const void *AddOffsetConst(const void *base, uint64_t offset)
{
    return static_cast<const void *>(static_cast<const uint8_t *>(base) + offset);
}

HcclResult GetDataTypeSize(HcclDataType dataType, uint32_t &dataTypeSize)
{
    auto iter = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(iter == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type[%d]", static_cast<int32_t>(dataType)), HCCL_E_NOT_SUPPORT);
    dataTypeSize = iter->second;
    return HCCL_SUCCESS;
}

// 主线程唤醒其余从线程，再由所有从线程等待各自的 Notify 0。
// 这条本地 Record/Wait 依赖必须保留：缺失后 Checker 建图阶段会因 Wait 无法解除而失败。
HcclResult ThreadSyncBefore(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t idx = 1; idx < threads.size(); idx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[idx], 0)));
    }
    for (uint32_t idx = 1; idx < threads.size(); idx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[idx], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

// 主线程的 Notify 0 由 Host/Device 同步占用，因此通信线程完成信号使用 Notify 1..N-1。
HcclResult ThreadSyncAfter(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t idx = 1; idx < threads.size(); idx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], idx, CUSTOM_TIMEOUT)));
    }
    for (uint32_t idx = 1; idx < threads.size(); idx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[idx], threads[0], idx)));
    }
    return HCCL_SUCCESS;
}

// ---- root：朴素直发（小数据 / 非规整拓扑）----
HcclResult RunRootDirect(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvSize)
{
    uint32_t commThreadNum = static_cast<uint32_t>(resCtx.threads.size());
    for (uint32_t idx = 0; idx < resCtx.channels.size(); idx++) {
        const ChannelInfo &channel = resCtx.channels[idx];
        ThreadHandle thread = resCtx.threads[idx % commThreadNum];
        const void *src = AddOffsetConst(param.inputPtr,
            static_cast<uint64_t>(channel.remoteRank) * recvSize);
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
            thread, channel.handle, channel.remoteOutputMem.addr, src, recvSize, NOTIFY_FROM_ROOT)));
    }
    return HCCL_SUCCESS;
}

// 小数据诊断路径：外层保证root输入就绪，且在所有读取完成前保持有效。
// 本轮不增加跨rank的READY/DONE；读任务及本rank完成通知仍按原通信线程编排。
HcclResult RunTinyRead(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvSize)
{
    CHK_PRT_RET(resCtx.rootChannelIdx >= resCtx.channels.size(),
        HCCL_ERROR("root通道下标[%u]越界，通道数[%zu]", resCtx.rootChannelIdx, resCtx.channels.size()),
        HCCL_E_INTERNAL);
    const ChannelInfo &rootChannel = resCtx.channels[resCtx.rootChannelIdx];
    CHK_PTR_NULL(rootChannel.remoteInputMem.addr);
    const uint64_t offset = static_cast<uint64_t>(param.myRank) * recvSize;
    CHK_PRT_RET(offset > rootChannel.remoteInputMem.size || recvSize > rootChannel.remoteInputMem.size - offset,
        HCCL_ERROR("Read分片超出root输入范围，rank[%u]", param.myRank), HCCL_E_PARA);
    const void *src = AddOffsetConst(rootChannel.remoteInputMem.addr, offset);
    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
        resCtx.threads[0], rootChannel.handle, param.outputPtr, src, recvSize)));
    return HCCL_SUCCESS;
}

// ---- relay：先Read代发段，再并行转发与Read自身数据 ----
HcclResult RunRelay(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvSize, uint64_t directSize)
{
    const uint64_t relaySize = recvSize - directSize;
    CHK_PRT_RET(resCtx.threads.size() != 2,
        HCCL_ERROR("中转Read需要两条线程，实际[%zu]", resCtx.threads.size()), HCCL_E_INTERNAL);
    // 全连接建链后 channels[] 是「跳过自己的全体 rank 升序」，root 与代发搭档的下标
    // 都由 host 侧算好写进 ctx；这里绝不能再假设 channels[0] 是 root。
    CHK_PRT_RET(resCtx.rootChannelIdx >= resCtx.channels.size() ||
                resCtx.relayChannelIdx >= resCtx.channels.size(),
        HCCL_ERROR("Relay channel index out of range: root[%u] peer[%u] channels[%zu]",
            resCtx.rootChannelIdx, resCtx.relayChannelIdx, resCtx.channels.size()),
        HCCL_E_INTERNAL);
    const ChannelInfo &fromRoot = resCtx.channels[resCtx.rootChannelIdx];
    const ChannelInfo &toPeer = resCtx.channels[resCtx.relayChannelIdx];
    const ThreadHandle readThread = resCtx.threads[0];
    const ThreadHandle forwardThread = resCtx.threads[1];
    CHK_PTR_NULL(fromRoot.remoteInputMem.addr);
    const uint64_t relayOffset = static_cast<uint64_t>(toPeer.remoteRank) * recvSize + directSize;
    const uint64_t selfOffset = static_cast<uint64_t>(param.myRank) * recvSize;
    CHK_PRT_RET(relayOffset > fromRoot.remoteInputMem.size ||
                    relaySize > fromRoot.remoteInputMem.size - relayOffset ||
                    selfOffset > fromRoot.remoteInputMem.size ||
                    recvSize > fromRoot.remoteInputMem.size - selfOffset,
        HCCL_ERROR("中转Read分片超出root输入范围，rank[%u]", param.myRank), HCCL_E_PARA);

    // 同一拉取线程先读代发段，再读自身数据，固定root到本rank的Mesh传输顺序。
    if (relaySize != 0) {
        CHK_PTR_NULL(resCtx.localBuffer.addr);
        CHK_PRT_RET(resCtx.localBuffer.size < relaySize,
            HCCL_ERROR("中转CCL缓冲区不足，rank[%u]", param.myRank), HCCL_E_INTERNAL);
        const void *relaySrc = AddOffsetConst(fromRoot.remoteInputMem.addr, relayOffset);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
            readThread, fromRoot.handle, resCtx.localBuffer.addr, relaySrc, relaySize)));
    }

    // 复用线程启动的Record/Wait：代发段Read完成后才唤醒转发线程，不另加READY通知。
    CHK_RET(ThreadSyncBefore(resCtx.threads));

    if (relaySize != 0) {
        // 转发到搭档output的后段位置，搭档同时从root读取不重叠的前段。
        CHK_PTR_NULL(toPeer.remoteOutputMem.addr);
        void *dst = AddOffset(toPeer.remoteOutputMem.addr, directSize);
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
            forwardThread, toPeer.handle, dst, resCtx.localBuffer.addr, relaySize, NOTIFY_FROM_RELAY)));
    }

    // 该Read与另一线程的Clos转发重叠，最后统一汇合，保证两份数据均处理完毕。
    const void *selfSrc = AddOffsetConst(fromRoot.remoteInputMem.addr, selfOffset);
    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
        readThread, fromRoot.handle, param.outputPtr, selfSrc, recvSize)));
    return HCCL_SUCCESS;
}

// 分摊路径从root读取前段或整份，后段仍等待中转通知；非分摊路径等待root直写。
HcclResult RunRemoteRecv(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvSize, uint64_t directSize, bool expectRelay)
{
    ThreadHandle thread = resCtx.threads[0];
    CHK_PRT_RET(resCtx.rootChannelIdx >= resCtx.channels.size(),
        HCCL_ERROR("Root channel index [%u] out of range, channels[%zu]",
            resCtx.rootChannelIdx, resCtx.channels.size()),
        HCCL_E_INTERNAL);
    const ChannelInfo &fromRoot = resCtx.channels[resCtx.rootChannelIdx];
    const bool readRoot = resCtx.role == SCATTER_ROLE_RELAY_RECV ||
                          resCtx.role == SCATTER_ROLE_FULL_READ;
    if (readRoot) {
        const uint64_t readSize = expectRelay ? directSize : recvSize;
        const uint64_t offset = static_cast<uint64_t>(param.myRank) * recvSize;
        CHK_PRT_RET(readSize > recvSize || offset > fromRoot.remoteInputMem.size ||
                        readSize > fromRoot.remoteInputMem.size - offset,
            HCCL_ERROR("叶子Read超出root输入范围，rank[%u]", param.myRank), HCCL_E_PARA);
        if (readSize != 0) {
            CHK_PTR_NULL(fromRoot.remoteInputMem.addr);
            const void *src = AddOffsetConst(fromRoot.remoteInputMem.addr, offset);
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
                thread, fromRoot.handle, param.outputPtr, src, readSize)));
        }
    } else {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, fromRoot.handle, NOTIFY_FROM_ROOT, CUSTOM_TIMEOUT)));
    }
    if (expectRelay && recvSize != directSize) {
        // 代发段由邻居写入，Notify 落在与该邻居的 channel 上（下标由 host 侧算好）。
        CHK_PRT_RET(resCtx.relayChannelIdx >= resCtx.channels.size(),
            HCCL_ERROR("Relay channel index [%u] out of range, channels[%zu]",
                resCtx.relayChannelIdx, resCtx.channels.size()),
            HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, resCtx.channels[resCtx.relayChannelIdx].handle, NOTIFY_FROM_RELAY, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    CHK_PTR_NULL(param.outputPtr);
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank information: myRank[%u], rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("Invalid root[%u] for rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA);

    uint32_t dataTypeSize = 0;
    CHK_RET(GetDataTypeSize(param.dataType, dataTypeSize));
    uint64_t recvSize = param.count * dataTypeSize;
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No AICPU communication thread was acquired"), HCCL_E_INTERNAL);

    const bool isRoot = (param.myRank == param.root);
    const bool tinyRead = ScatterUseTinyRead(param.rankSize, recvSize);

    if (param.rankSize == 1) {
        CHK_PTR_NULL(param.inputPtr);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, recvSize)));
        return HCCL_SUCCESS;
    }

    // 分摊路径由 host 侧建链时确定并写进 role，这里不再重新推导拓扑条件。
    // 各角色使用相同的分段计算，保证叶子Read与中转Write的边界一致。
    const bool relayOn = (resCtx.role == SCATTER_ROLE_RELAY) ||
                         (resCtx.role == SCATTER_ROLE_RELAY_RECV) ||
                         (resCtx.role == SCATTER_ROLE_FULL_READ) ||
                         (isRoot && resCtx.closDirectFullRank != INVALID_VALUE_RANKID);
    const uint64_t directSize = relayOn ? ScatterDirectSize(recvSize, dataTypeSize) : recvSize;

    HCCL_INFO("Scatter叶子Read v11：rank[%u/%u] root[%u] role[%u] recvSize[%llu] "
              "directSize[%llu] relayOn[%d] tinyRead[%d] threads[%zu] channels[%zu]",
        param.myRank, param.rankSize, param.root, resCtx.role,
        static_cast<unsigned long long>(recvSize), static_cast<unsigned long long>(directSize),
        static_cast<int32_t>(relayOn), static_cast<int32_t>(tinyRead), resCtx.threads.size(), resCtx.channels.size());

    // 中转节点在代发段Read完成后才启动转发线程，其他角色沿用原启动顺序。
    if (resCtx.role != SCATTER_ROLE_RELAY) {
        CHK_RET(ThreadSyncBefore(resCtx.threads));
    }

    if (isRoot) {
        CHK_PTR_NULL(param.inputPtr);
        CHK_PRT_RET(resCtx.channels.size() != param.rankSize - 1,
            HCCL_ERROR("Root expects [%u] channels, actual [%zu]", param.rankSize - 1, resCtx.channels.size()),
            HCCL_E_INTERNAL);

        if (!tinyRead && !relayOn) {
            CHK_RET(RunRootDirect(param, resCtx, recvSize));
        }
        // 小数据Read和大数据分摊路径都由接收方拉取，root仅完成自身拷贝。

        // root 自己那份不过网络。
        const void *selfSrc = AddOffsetConst(param.inputPtr,
            static_cast<uint64_t>(param.myRank) * recvSize);
        if (param.outputPtr != selfSrc) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, selfSrc, recvSize)));
        }
    } else if (tinyRead) {
        CHK_RET(RunTinyRead(param, resCtx, recvSize));
    } else if (resCtx.role == SCATTER_ROLE_RELAY) {
        CHK_RET(RunRelay(param, resCtx, recvSize, directSize));
    } else if (resCtx.role == SCATTER_ROLE_RELAY_RECV || resCtx.role == SCATTER_ROLE_FULL_READ) {
        // 有搭档时读取前段并等后段，无搭档时读取整份；均不等待root的远端通知。
        CHK_RET(RunRemoteRecv(param, resCtx, recvSize, directSize,
            resCtx.role == SCATTER_ROLE_RELAY_RECV));
    } else {
        // 非分摊路径仍由root整份直写，沿用原通知等待。
        CHK_RET(RunRemoteRecv(param, resCtx, recvSize, directSize, false));
    }

    CHK_RET(ThreadSyncAfter(resCtx.threads));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
