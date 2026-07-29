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
#include <unordered_map>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {

// 单次通信允许的最大字节数：256 MB。与本端 ccl buffer 容量取小，作为单槽复用的分段粒度。
constexpr uint64_t MAX_DATA_SIZE = 256ULL * 1024 * 1024;

// 数据面一条"有向边"传输：把 srcRank 上 buf[offset,+bytes) 送到 dstRank 的同段 buf。
// 语义完全复用参考 alltoall 的 write→notify→localcopy 节拍，但为单向：
//   - 发送端(本 rank==src)：写到对端 ccl buffer(偏移0，单槽) -> record 通知对端数据到达
//   - 接收端(本 rank==dst)：wait 数据到达 -> 从本端 ccl buffer 拷到 recvBuf
// 每一分段前用一次就绪屏障(record+wait)，确保对端 ccl buffer 空槽可写；
// 分段后用一次回执屏障(record+wait)，确保对端读完再复用单槽。
//
// 分段粒度 maxBytesPerLoop 在收发两端由各自 own local buffer 派生（均为同一 400MB 配置），
// 因此两端计算出的 subBytes 序列完全一致，分段边界严格对齐，不会错位或死锁。
class EdgeTransfer {
public:
    EdgeTransfer(ThreadHandle thread, const ChannelInfo &ch, void *localCclAddr, uint64_t maxBytesPerLoop)
        : thread_(thread), ch_(ch), localCcl_(localCclAddr), maxBytesPerLoop_(maxBytesPerLoop)
    {}

    // 本 rank 作为发送端：把 sendBuf[0,totalBytes) 推给对端。
    HcclResult Send(const uint8_t *sendBuf, uint64_t totalBytes)
    {
        return Run(sendBuf, nullptr, totalBytes, true);
    }

    // 本 rank 作为接收端：从对端接收 totalBytes 并落到 recvBuf[0,totalBytes)。
    HcclResult Recv(uint8_t *recvBuf, uint64_t totalBytes)
    {
        return Run(nullptr, recvBuf, totalBytes, false);
    }

private:
    HcclResult Run(const uint8_t *sendBuf, uint8_t *recvBuf, uint64_t totalBytes, bool isSender)
    {
        // 就绪屏障：双方互报到达交换点（对端 ccl buffer 已就位可写）。
        CHK_RET(NotifyRecord());
        CHK_RET(NotifyWait());

        uint64_t off = 0;
        while (off < totalBytes) {
            uint64_t subBytes = std::min(maxBytesPerLoop_, totalBytes - off);
            if (isSender) {
                // 写本地 sendBuf 本段 -> 对端 ccl buffer 偏移 0（单槽复用）。
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                    thread_, ch_.handle, ch_.remoteCclMem.addr, const_cast<uint8_t *>(sendBuf) + off, subBytes)));
                // 通知对端本段已写达；等对端回执（已读完，单槽可复用）。
                CHK_RET(NotifyRecord());
                CHK_RET(NotifyWait());
            } else {
                // 等对端把本段写到我的 ccl buffer；从本端 ccl buffer 拷到 recvBuf 本段。
                CHK_RET(NotifyWait());
                CHK_RET(static_cast<HcclResult>(
                    HcommLocalCopyOnThread(thread_, recvBuf + off, localCcl_, subBytes)));
                // 通知对端我已读完本段（可写下一段）。
                CHK_RET(NotifyRecord());
            }
            off += subBytes;
        }
        return HCCL_SUCCESS;
    }

    HcclResult NotifyRecord()
    {
        return static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread_, ch_.handle, NOTIFY_IDX_DATA_SIGNAL));
    }
    HcclResult NotifyWait()
    {
        return static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread_, ch_.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    }

    ThreadHandle thread_;
    const ChannelInfo &ch_;
    void *localCcl_;
    uint64_t maxBytesPerLoop_;
};

} // namespace

// =====================================================================================
// Broadcast 数据面编排
// -------------------------------------------------------------------------------------
// 算法：二项树（binomial tree）。以 root 为虚拟 0 号编号，
//   - 每个非根节点在其最低置位比特处，从父节点 (vrank - lowbit) 接收整包；
//   - 收到后按比特从高到低，依次向子节点 (vrank + 2^k) 转发整包。
// 步数 = ceil(log2(rankSize))，每步一条单向边。在 2*8 拓扑上跨 Server 边至多出现一次，
// 其余为 Server 内 layer-0 Full-Mesh 高带宽直连，兼顾时延与带宽。
//
// 大消息由 EdgeTransfer 内部按 MAX_DATA_SIZE(256MB) 分段搬运，收发两端分段边界一致。
//
// 数据落位：root 的 inputPtr 即源；非根节点最终把数据写入 outputPtr。
// 本算子 inputPtr==outputPtr==buf（见 broadcast.cc），故发送端直接从 buf 读、接收端写回 buf。
// =====================================================================================
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Broadcast on AICPU: myRank=%u root=%u rankSize=%u count=%lu", param.myRank, param.root,
        param.rankSize, param.count);

    if (param.count == 0 || param.rankSize <= 1) {
        return HCCL_SUCCESS; // 无数据或单 rank，无需搬运
    }

    uint32_t dataTypeSize = SIZE_TABLE.at(param.dataType);
    uint64_t totalBytes = param.count * dataTypeSize;

    ThreadHandle thread = resCtx.aicpuThread;
    void *cclBuffAddr = resCtx.localBuffer.addr;
    CHK_PTR_NULL(cclBuffAddr);
    uint64_t maxBytesPerLoop = std::min(MAX_DATA_SIZE, resCtx.localBuffer.size);
    CHK_PRT_RET(maxBytesPerLoop == 0, HCCL_ERROR("ExecOp: ccl buffer size is 0"), HCCL_E_INTERNAL);

    uint8_t *buf = static_cast<uint8_t *>(param.inputPtr); // inputPtr==outputPtr
    CHK_PTR_NULL(buf);

    // 构建 remoteRank -> ChannelInfo 映射，供二项树按需选边。
    std::unordered_map<uint32_t, ChannelInfo> channelMap;
    for (const auto &ch : resCtx.channels) {
        channelMap.emplace(ch.remoteRank, ch);
    }

    const uint32_t rankSize = param.rankSize;
    const uint32_t vrank = (param.myRank + rankSize - param.root) % rankSize;
    auto realRank = [&](uint32_t v) -> uint32_t { return (v + param.root) % rankSize; };

    // 找到本节点最低置位比特：在该轮从父节点接收整包。根节点无父。
    uint32_t mask = 1;
    while (mask < rankSize) {
        if (vrank & mask) {
            uint32_t parentReal = realRank(vrank - mask);
            auto it = channelMap.find(parentReal);
            CHK_PRT_RET(it == channelMap.end(),
                HCCL_ERROR("ExecOp: parent channel to rank[%u] not found", parentReal), HCCL_E_INTERNAL);
            EdgeTransfer edge(thread, it->second, cclBuffAddr, maxBytesPerLoop);
            CHK_RET(edge.Recv(buf, totalBytes));
            break;
        }
        mask <<= 1;
    }

    // 向子节点转发整包：mask 右移，childV = vrank + mask（低位全 0，保证树无重叠边）。
    mask >>= 1;
    while (mask > 0) {
        uint32_t childV = vrank + mask;
        if (childV < rankSize) {
            uint32_t childReal = realRank(childV);
            auto it = channelMap.find(childReal);
            CHK_PRT_RET(it == channelMap.end(),
                HCCL_ERROR("ExecOp: child channel to rank[%u] not found", childReal), HCCL_E_INTERNAL);
            EdgeTransfer edge(thread, it->second, cclBuffAddr, maxBytesPerLoop);
            CHK_RET(edge.Send(buf, totalBytes));
        }
        mask >>= 1;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
