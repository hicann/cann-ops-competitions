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
#include <vector>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
// 查表获取数据类型宽度。common.h 的 SIZE_TABLE 只注册了 FP32，
// 这里用 find() 而非 at()，避免不支持的类型在 AICPU 侧抛 std::out_of_range。
HcclResult GetDataTypeSize(HcclDataType dataType, uint32_t &dataTypeSize)
{
    auto it = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(it == SIZE_TABLE.end(),
        HCCL_ERROR("ExecOp: unsupported dataType[%d]", static_cast<int32_t>(dataType)), HCCL_E_NOT_SUPPORT);
    dataTypeSize = it->second;
    return HCCL_SUCCESS;
}

// 在资源上下文里找到通往 remoteRank 的 channel
HcclResult FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank, ChannelInfo &channel)
{
    for (const ChannelInfo &ch : resCtx.channels) {
        if (ch.remoteRank == remoteRank) {
            channel = ch;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("ExecOp: channel to rank[%u] not found", remoteRank);
    return HCCL_E_NOT_FOUND;
}

// 分段计划：每段搬运的字节数按元素对齐，且不超过任一端 ccl buffer 的容量
struct LoopPlan {
    uint64_t stepBytes;  ///< 每段（除尾段）的字节数
    uint64_t totalBytes; ///< 单个 rank 的总字节数
    uint64_t loopCount;  ///< 分段数
    uint64_t maxPerLoop; ///< 单段上限 = min(256MB, 本端与所有对端 ccl buffer)
};

HcclResult MakeLoopPlan(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize, LoopPlan &plan)
{
    // 单段上限 = min(256MB, 本端 ccl buffer, 所有对端 ccl buffer)
    uint64_t maxDataSizePerLoop = std::min(MAX_DATA_SIZE, resCtx.localBuffer.size);
    for (const ChannelInfo &ch : resCtx.channels) {
        maxDataSizePerLoop = std::min(maxDataSizePerLoop, ch.remoteCclMem.size);
    }
    // 向下取整到整数个元素，保证段边界不会切开一个元素
    uint64_t maxCountPerLoop = maxDataSizePerLoop / dataTypeSize;
    CHK_PRT_RET(maxCountPerLoop == 0,
        HCCL_ERROR("ExecOp: ccl buffer too small, maxDataSizePerLoop[%llu] dataTypeSize[%u]",
            static_cast<unsigned long long>(maxDataSizePerLoop), dataTypeSize),
        HCCL_E_INTERNAL);

    plan.maxPerLoop = maxDataSizePerLoop;
    plan.stepBytes = maxCountPerLoop * dataTypeSize;
    plan.totalBytes = param.count * dataTypeSize;
    plan.loopCount = param.count / maxCountPerLoop + (param.count % maxCountPerLoop != 0 ? 1 : 0);
    HCCL_INFO("ExecOp: rank[%u] root[%u] totalBytes[%llu] stepBytes[%llu] loopCount[%llu]", param.myRank, param.root,
        static_cast<unsigned long long>(plan.totalBytes), static_cast<unsigned long long>(plan.stepBytes),
        static_cast<unsigned long long>(plan.loopCount));
    return HCCL_SUCCESS;
}

// 末尾 join 用的从流收集器。
// aicpu_kernel.cc（不可改）在 ExecOp 返回后只对 aicpuThread 打"任务下发完成"notify，
// 框架据此认为算子结束。任何下发到 threads[1..] 的任务如果不 join 回 aicpuThread，
// Host 就会在数据还在飞的时候放行下一步 —— 小数据可能偶然通过，大数据必错。
struct Joiner {
    static constexpr uint32_t MAX_SUB_THREAD = 32;
    ThreadHandle threads[MAX_SUB_THREAD] = {};
    uint32_t num = 0;

    void Add(ThreadHandle handle)
    {
        for (uint32_t i = 0; i < num; i++) {
            if (threads[i] == handle) { // 去重：同一条流被多次使用只 join 一次
                return;
            }
        }
        if (num < MAX_SUB_THREAD) {
            threads[num++] = handle;
        }
    }
};

// 把所有用过的从流 join 回 aicpuThread。
// record 排在从流任务队列的末尾（流内保序 ⇒ 该流全部任务完成后才触发），
// wait 排在 aicpuThread 上，于是 aicpuThread 必然最后结束。
// thread notify 下标 0 被框架的 Host/Device 握手占用，所以 join 从 1 开始。
HcclResult JoinSubThreads(ThreadHandle mainThread, const Joiner &joiner)
{
    for (uint32_t i = 0; i < joiner.num; i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(joiner.threads[i], mainThread, i + 1)));
    }
    for (uint32_t i = 0; i < joiner.num; i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(mainThread, i + 1, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

// root 侧单条流上的发送序列，只负责 group 里这几个对端。
// notify 计数是 per-channel 的，(root, peer_j) 的三个下标是这一对 rank 的私有信号量，
// 所以把 peer 拆到不同流上不改变任何配对关系，节拍与 v1 逐字相同。
HcclResult EmitRootSendGroup(
    ThreadHandle thread, const std::vector<ChannelInfo> &group, const OpParam &param, const LoopPlan &plan)
{
    // 起始就绪屏障：等对端报告其 ccl buffer 槽位可写，之后首段写入才是安全的
    for (const ChannelInfo &ch : group) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, ch.handle, NOTIFY_IDX_BUF_READY, CUSTOM_TIMEOUT)));
    }

    const uint8_t *sendBase = static_cast<const uint8_t *>(param.inputPtr);
    for (uint64_t k = 0; k < plan.loopCount; k++) {
        uint64_t segOffset = k * plan.stepBytes;
        uint64_t segBytes = std::min(plan.stepBytes, plan.totalBytes - segOffset);

        // 把本段写进对端的 ccl buffer（偏移 0，单槽复用），写完立即通知对端
        for (const ChannelInfo &ch : group) {
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, ch.handle, ch.remoteCclMem.addr,
                sendBase + static_cast<uint64_t>(ch.remoteRank) * plan.totalBytes + segOffset, segBytes)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, ch.handle, NOTIFY_IDX_DATA_SIGNAL)));
        }
        // 收齐本段 ACK：对端已把本段取走，槽位才可以被下一段覆盖
        for (const ChannelInfo &ch : group) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, ch.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        }
    }
    return HCCL_SUCCESS;
}

// root 侧：自己那段在 aicpuThread 上本地拷贝，15 个对端分摊到从流上并行发送
HcclResult RunAsRoot(const OpParam &param, const AlgResourceCtx &resCtx, const LoopPlan &plan, Joiner &joiner)
{
    CHK_PTR_NULL(param.inputPtr);
    const ThreadHandle mainThread = resCtx.aicpuThread;

    // 实际可用从流数。host 侧申请失败会折半降级，所以这里按拿到的数量自适应；
    // subCount == 0 时全部落回 aicpuThread，等价于 v1 的单流串行。
    uint32_t subCount = resCtx.threads.size() > 1 ? static_cast<uint32_t>(resCtx.threads.size() - 1) : 0;
    if (subCount > Joiner::MAX_SUB_THREAD) {
        subCount = Joiner::MAX_SUB_THREAD;
    }
    const uint32_t peerCount = param.rankSize - 1;
    const uint32_t groupCount = (subCount == 0) ? 1 : std::min(subCount, peerCount);

    // 按 rank 升序轮转分组：从流数够时每条流恰好 1 个对端，不够时每条流带几个
    std::vector<std::vector<ChannelInfo>> groups(groupCount);
    uint32_t seq = 0;
    for (uint32_t j = 0; j < param.rankSize; j++) {
        if (j == param.myRank) {
            continue;
        }
        ChannelInfo channel;
        CHK_RET(FindChannel(resCtx, j, channel));
        groups[seq % groupCount].push_back(channel);
        seq++;
    }

    // root 自己那一块：sendBuf 第 root 段 -> recvBuf，不走 channel，留在 aicpuThread 上
    const uint8_t *sendBase = static_cast<const uint8_t *>(param.inputPtr);
    uint8_t *recvBase = static_cast<uint8_t *>(param.outputPtr);
    for (uint64_t k = 0; k < plan.loopCount; k++) {
        uint64_t segOffset = k * plan.stepBytes;
        uint64_t segBytes = std::min(plan.stepBytes, plan.totalBytes - segOffset);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, recvBase + segOffset,
            sendBase + static_cast<uint64_t>(param.myRank) * plan.totalBytes + segOffset, segBytes)));
    }

    for (uint32_t g = 0; g < groupCount; g++) {
        ThreadHandle thread = (subCount == 0) ? mainThread : resCtx.threads[1 + g];
        CHK_RET(EmitRootSendGroup(thread, groups[g], param, plan));
        if (subCount != 0) {
            joiner.Add(thread);
        }
    }
    return HCCL_SUCCESS;
}

// ===========================================================================
// 大消息路径：同 Server helper 中继（路径 L）
//
// 拓扑：2 个等大 Server，Server 内 1DMesh 全互联（每条链路带宽 B），
//       跨 Server Clos，每个 rank 一条上行（带宽 W·B）。
// root 出口总容量 = (m-1)·B + W·B，要发 (n-1)·C，所以把每个远端 chunk 切成
//   direct（root 经 Clos 直发） + relay（root 先经框内给同列 helper，helper 再经它自己的 Clos 转投）
// 令框内与 Clos 两条路径同时打满即得 q = relay/C = (m-W)/(W+m-1)。
//
// 发送顺序技巧：root 在每条框内链路上**先发中继片、再发 helper 自己那整段**。
// helper 在 relay/B 时刻就能起转发（耗时 relay/(W·B)），而 root 要到 (1+q)C/B 才发完，
// 第二跳完整落在阴影里，不需要分段流水。
// ===========================================================================
constexpr uint32_t NL_READY = NOTIFY_IDX_ACK;         // 接收方 -> 写入方："我的 ccl buffer 槽位可写"
constexpr uint32_t NL_DATA = NOTIFY_IDX_DATA_SIGNAL;  // 写入方 -> 接收方："这一份已写入"
constexpr uint32_t NL_SELF = NOTIFY_IDX_BUF_READY;    // root -> helper："你自己那整段已写入"
constexpr uint32_t MAX_MASK_RANK = 32;                // intraPeerMask 是 uint32_t

struct TopoView {
    bool valid = false;
    bool rootInMyServer = false;
    uint32_t localSize = 0;                 ///< 每个 Server 的 rank 数 m
    uint32_t myCol = 0;                     ///< 我在本 Server 升序列表中的下标（列号）
    uint32_t rootCol = 0;                   ///< root 的列号
    uint32_t home[MAX_MASK_RANK] = {};      ///< root 所在 Server，升序
    uint32_t away[MAX_MASK_RANK] = {};      ///< 另一个 Server，升序
};

// 由 host 侧记录的 layer-0 邻接位掩码还原两个 Server 的 rank 列表。
// 两侧算出的 home/away 必然一致：都是同一个全局集合按 rank 升序排列，
// 「列号」= 在各自升序列表中的下标，因此同列的两个 rank 互为伙伴，无需任何协商。
void BuildTopoView(const OpParam &param, const AlgResourceCtx &resCtx, TopoView &topo)
{
    topo.valid = false;
    if (param.rankSize < 4 || param.rankSize > MAX_MASK_RANK || (param.rankSize % 2) != 0) {
        return;
    }
    uint32_t mine[MAX_MASK_RANK] = {};
    uint32_t other[MAX_MASK_RANK] = {};
    uint32_t mineNum = 0;
    uint32_t otherNum = 0;
    for (uint32_t j = 0; j < param.rankSize; j++) {
        bool sameServer = (j == param.myRank) || (((resCtx.intraPeerMask >> j) & 1U) != 0);
        if (sameServer) {
            mine[mineNum++] = j;
        } else {
            other[otherNum++] = j;
        }
    }
    // 只支持"两个等大 Server"，其它分层形态一律退回直发
    if (mineNum != otherNum || mineNum * 2 != param.rankSize || mineNum < 2) {
        return;
    }
    bool rootInMine = false;
    for (uint32_t i = 0; i < mineNum; i++) {
        if (mine[i] == param.root) {
            rootInMine = true;
        }
    }
    topo.rootInMyServer = rootInMine;
    topo.localSize = mineNum;
    for (uint32_t i = 0; i < mineNum; i++) {
        topo.home[i] = rootInMine ? mine[i] : other[i];
        topo.away[i] = rootInMine ? other[i] : mine[i];
    }
    const uint32_t *myList = rootInMine ? topo.home : topo.away;
    bool foundMe = false;
    bool foundRoot = false;
    for (uint32_t i = 0; i < mineNum; i++) {
        if (myList[i] == param.myRank) {
            topo.myCol = i;
            foundMe = true;
        }
        if (topo.home[i] == param.root) {
            topo.rootCol = i;
            foundRoot = true;
        }
    }
    topo.valid = foundMe && foundRoot;
}

struct SplitPlan {
    uint64_t direct = 0; ///< root 经 Clos 直发的前段（起始偏移 0，长度 512B 对齐）
    uint64_t relay = 0;  ///< 交给同列 helper 中继的后段（远端起始偏移 = direct）
};

// 按实测带宽系数现算切分比例，并做安全钳位。relay == 0 表示"不中继"。
//
// 记 W = bwCoeffInter / bwCoeffIntra，m = 每 Server rank 数：
//   平衡解 q = (m·Wd - Wn) / (Wn + (m-1)·Wd)，于是 1 - q = (2·Wn - Wd) / denom
// 安全上限（中继不得比朴素直发更慢）：1 + q <= max(1, m/W)，即 q <= (m-W)/W。
// W >= m 时上限为 0 —— 此时朴素直发本身已经均衡，任何中继都是把框内变成新瓶颈，净亏。
void MakeSplitPlan(const TopoView &topo, uint64_t chunkBytes, const AlgResourceCtx &resCtx, SplitPlan &sp)
{
    sp.direct = chunkBytes;
    sp.relay = 0;
    const uint64_t wn = resCtx.bwCoeffInter;
    const uint64_t wd = resCtx.bwCoeffIntra;
    const uint64_t m = topo.localSize;
    // 查询失败 / 取值不可信 -> q = 0
    if (wn == 0 || wd == 0 || chunkBytes == 0) {
        return;
    }
    // W >= m（本拓扑即 W >= 8）-> q 必须为 0，否则比朴素直发更慢
    if (wn >= m * wd) {
        return;
    }
    // 1 - q 的分子必须为正，否则整段都该走中继（不可能，m >= 2 时恒不成立，留作防御）
    if (2 * wn <= wd) {
        return;
    }
    const uint64_t denom = wn + (m - 1) * wd;
    uint64_t direct = chunkBytes * (2 * wn - wd) / denom;
    // ★ 对 direct 向上对齐到 512：既保证 helper 远端写入的**起始偏移**对齐
    // （点 11 的 C mod 512 = 4 会精确踩这个坑），又让取整只会让 relay 变小，
    // 从而安全上限在取整后一定仍然成立，不会因为凑整反向越界。
    direct = ((direct + SLICE_ALIGN - 1) / SLICE_ALIGN) * SLICE_ALIGN;
    if (direct >= chunkBytes) {
        return; // 中继片被凑没了，退回纯直发
    }
    const uint64_t relay = chunkBytes - direct;
    // 安全钳位复核：relay/C <= (m - W)/W  <=>  relay·Wn <= C·(m·Wd - Wn)
    if (relay * wn > chunkBytes * (m * wd - wn)) {
        return;
    }
    sp.direct = direct;
    sp.relay = relay;
}

// 轮转取一条从流；没有从流时全部落回 aicpuThread（正确性不变，只是没有并行）
ThreadHandle PickThread(const AlgResourceCtx &resCtx, uint32_t subCount, uint32_t seq, ThreadHandle fallback)
{
    if (subCount == 0) {
        return fallback;
    }
    return resCtx.threads[1 + (seq % subCount)];
}

// 路径 L 的编排。三种角色：root / HOME 非 root（helper 兼接收方）/ AWAY（纯接收方）。
HcclResult RunHybrid(const OpParam &param, const AlgResourceCtx &resCtx, const TopoView &topo, uint64_t chunkBytes,
    const SplitPlan &sp, Joiner &joiner)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    const uint32_t subCount = resCtx.threads.size() > 1 ? static_cast<uint32_t>(resCtx.threads.size() - 1) : 0;
    const uint8_t *sendBase = static_cast<const uint8_t *>(param.inputPtr);
    uint8_t *recvBase = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localCcl = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const uint32_t m = topo.localSize;
    // helper 暂存区起始偏移：C 向上对齐到 512。C mod 512 = 4（点 11）时若直接用 C，
    // root -> helper 的远端写入起始地址就不对齐，与 direct 对齐的初衷相悖，这里统一补齐。
    const uint64_t stagingOff = ((chunkBytes + SLICE_ALIGN - 1) / SLICE_ALIGN) * SLICE_ALIGN;
    uint32_t seq = 0;

    if (param.myRank == param.root) {
        CHK_PTR_NULL(param.inputPtr);
        // root 自己那段：本地拷贝，不走 channel
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, recvBase,
            sendBase + static_cast<uint64_t>(param.myRank) * chunkBytes, chunkBytes)));

        // --- 框内：m-1 条链路，每条一条独立从流 ---
        for (uint32_t i = 0; i < m; i++) {
            if (i == topo.rootCol) {
                continue; // 这一列在 HOME 侧就是 root 自己，没有 helper
            }
            const uint32_t helper = topo.home[i];
            const uint32_t mate = topo.away[i];
            ChannelInfo ch;
            CHK_RET(FindChannel(resCtx, helper, ch));
            ThreadHandle thread = PickThread(resCtx, subCount, seq++, mainThread);
            // 等 helper 报告槽位可写（跨算子调用复用同一块 ccl buffer，这一步不能省）
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, ch.handle, NL_READY, CUSTOM_TIMEOUT)));
            // ① 先发中继片 -> helper 暂存区 [C, C+relay)，write+notify 合成一条 SQE
            CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(thread, ch.handle,
                static_cast<uint8_t *>(ch.remoteCclMem.addr) + stagingOff,
                sendBase + static_cast<uint64_t>(mate) * chunkBytes + sp.direct, sp.relay, NL_DATA)));
            // ② 再发 helper 自己那整段 -> [0, C)
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, ch.handle, ch.remoteCclMem.addr,
                sendBase + static_cast<uint64_t>(helper) * chunkBytes, chunkBytes)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, ch.handle, NL_SELF)));
            if (subCount != 0) {
                joiner.Add(thread);
            }
        }

        // --- Clos：m 个远端，共享 root 那一条上行，分摊到多条从流以提高在途请求数 ---
        for (uint32_t i = 0; i < m; i++) {
            const uint32_t remote = topo.away[i];
            // root 那一列在 AWAY 侧没有 helper，整段直发
            const uint64_t len = (i == topo.rootCol) ? chunkBytes : sp.direct;
            ChannelInfo ch;
            CHK_RET(FindChannel(resCtx, remote, ch));
            ThreadHandle thread = PickThread(resCtx, subCount, seq++, mainThread);
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, ch.handle, NL_READY, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(thread, ch.handle, ch.remoteCclMem.addr,
                sendBase + static_cast<uint64_t>(remote) * chunkBytes, len, NL_DATA)));
            if (subCount != 0) {
                joiner.Add(thread);
            }
        }
        return HCCL_SUCCESS;
    }

    CHK_PTR_NULL(localCcl);
    ChannelInfo fromRoot;
    CHK_RET(FindChannel(resCtx, param.root, fromRoot));

    if (topo.rootInMyServer) {
        // HOME 非 root：既是 helper（转发同列远端的中继片）又是接收方
        const uint32_t mate = topo.away[topo.myCol];
        ChannelInfo toMate;
        CHK_RET(FindChannel(resCtx, mate, toMate));
        ThreadHandle fwd = (subCount > 0) ? resCtx.threads[1] : mainThread;

        // 主流第一件事：告诉 root 我的槽位可写（root 在等它）
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, fromRoot.handle, NL_READY)));
        // 转发流：等 mate 报告槽位可写 -> 等中继片 -> 转投。
        // ★ 必须是独立的一条流：转发只依赖 ①，不能被 ② 的整段 C 字节拖在后面，
        // 否则第二跳就从阴影里暴露出来了。
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(fwd, toMate.handle, NL_READY, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(fwd, fromRoot.handle, NL_DATA, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(fwd, toMate.handle,
            static_cast<uint8_t *>(toMate.remoteCclMem.addr) + sp.direct, localCcl + stagingOff, sp.relay,
            NL_DATA)));
        if (subCount != 0) {
            joiner.Add(fwd);
        }
        // 主流：等自己那整段 -> 拷进 recvBuf
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(mainThread, fromRoot.handle, NL_SELF, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, recvBase, localCcl, chunkBytes)));
        return HCCL_SUCCESS;
    }

    // AWAY：纯接收方。前段来自 root，后段来自同列 helper（root 那一列没有 helper）
    const bool hasMate = (topo.myCol != topo.rootCol);
    ChannelInfo fromMate;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, fromRoot.handle, NL_READY)));
    if (hasMate) {
        CHK_RET(FindChannel(resCtx, topo.home[topo.myCol], fromMate));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, fromMate.handle, NL_READY)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread, fromRoot.handle, NL_DATA, CUSTOM_TIMEOUT)));
    if (hasMate) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(mainThread, fromMate.handle, NL_DATA, CUSTOM_TIMEOUT)));
    }
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, recvBase, localCcl, chunkBytes)));
    return HCCL_SUCCESS;
}

// 非 root 侧：只与 root 交互，逐段等待 -> 本地拷出 -> 回 ACK
HcclResult RunAsPeer(const OpParam &param, const AlgResourceCtx &resCtx, const LoopPlan &plan)
{
    // 非 root 的 sendBuf 允许为空，全程不访问
    void *cclBuffAddr = resCtx.localBuffer.addr;
    CHK_PTR_NULL(cclBuffAddr);

    ChannelInfo rootChannel;
    CHK_RET(FindChannel(resCtx, param.root, rootChannel));

    ThreadHandle thread = resCtx.aicpuThread;
    // 起始就绪屏障：告知 root 我已进入本次算子，ccl buffer 槽位可写
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, NOTIFY_IDX_BUF_READY)));

    uint8_t *recvBase = static_cast<uint8_t *>(param.outputPtr);
    for (uint64_t k = 0; k < plan.loopCount; k++) {
        uint64_t segOffset = k * plan.stepBytes;
        uint64_t segBytes = std::min(plan.stepBytes, plan.totalBytes - segOffset);

        // 等 root 把本段写进我的 ccl buffer
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, rootChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        // 本地 ccl buffer -> recvBuf 本段（recvBuf 只有本 rank 的一份数据，偏移不含 rank 因子）
        CHK_RET(
            static_cast<HcclResult>(HcommLocalCopyOnThread(thread, recvBase + segOffset, cclBuffAddr, segBytes)));
        // 告知 root 本段已取走，槽位可复用
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, NOTIFY_IDX_ACK)));
    }
    return HCCL_SUCCESS;
}
} // namespace

// Scatter v2 阶段 A：root 直发全连接 + 多流并行
//   - 语义：param.count 为每个 rank 接收的元素数，root 的 sendBuf 含 rankSize 段，
//     rank i 取第 i 段写入自己的 recvBuf（recvBuf 只有一段，偏移不乘 rank）
//   - root 自己那段本地拷贝，不走 channel；其余每段经对端 ccl buffer 中转
//   - 并行：15 个对端分摊到 threads[1..] 上，一条流一个对端。单条流内任务顺序执行，
//     多条流在 aicpu_kernel.cc 的 HcommBatchModeEnd 处被 CommTaskLaunch 一起拉起 ⇒ 真并发
//   - 非 root 只和 root 交互，单流足够，不申请额外并行资源
//   - notify 节拍（每条 channel 独立计数，逐段严格配对，算子结束时各计数归零）：
//       root : wait(BUF_READY) -> [ write+record(DATA_SIGNAL) -> wait(ACK) ] x loopCount   // 每条从流
//       peer : record(BUF_READY) -> [ wait(DATA_SIGNAL) -> localCopy -> record(ACK) ] x loopCount
//   - 末尾把所有从流 join 回 aicpuThread（thread notify 下标 1..），见 JoinSubThreads
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    // 无数据可散布
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(param.outputPtr);
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("ExecOp: invalid root[%u], rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA);

    uint32_t dataTypeSize = 0;
    CHK_RET(GetDataTypeSize(param.dataType, dataTypeSize));

    // 单 rank 通信域：root 必为本 rank，退化为一次本地拷贝
    if (param.rankSize == 1) {
        CHK_PTR_NULL(param.inputPtr);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.aicpuThread, param.outputPtr, param.inputPtr,
            param.count * dataTypeSize)));
        return HCCL_SUCCESS;
    }

    LoopPlan plan;
    CHK_RET(MakeLoopPlan(param, resCtx, dataTypeSize, plan));

    // ---- 大消息路径准入判定 ----
    // 用到的每个量在所有 rank 上都必然相同：总量、分段计划（各端 ccl buffer 尺寸一致）、
    // 位掩码导出的两 Server 视图、以及锚定在固定全局 rank 上探测出的带宽系数。
    // 因此不会出现"有的 rank 走 L、有的走 S"的分裂。
    const uint64_t totalBytes = plan.totalBytes * param.rankSize;
    TopoView topo;
    SplitPlan split;
    bool useHybrid = false;
    if (totalBytes > LARGE_MSG_THRESHOLD && plan.loopCount == 1) {
        BuildTopoView(param, resCtx, topo);
        if (topo.valid) {
            MakeSplitPlan(topo, plan.totalBytes, resCtx, split);
            // helper 的 ccl buffer 还要额外放下 [C, C+relay) 的中继暂存区
            // 暂存区起始偏移与 RunHybrid 里的 stagingOff 同一公式：C 向上对齐到 512
            const uint64_t stagingOff = ((plan.totalBytes + SLICE_ALIGN - 1) / SLICE_ALIGN) * SLICE_ALIGN;
            useHybrid = (split.relay > 0) && (stagingOff + split.relay <= plan.maxPerLoop);
        }
    }

    Joiner joiner;
    if (useHybrid) {
        CHK_RET(RunHybrid(param, resCtx, topo, plan.totalBytes, split, joiner));
    } else if (param.myRank == param.root) {
        // split.relay == 0 时控制流走到这里，与阶段 A 逐字节相同：
        // 同一个 RunAsRoot/RunAsPeer、同一组入参，没有任何多余的原语或 notify 往返。
        CHK_RET(RunAsRoot(param, resCtx, plan, joiner));
    } else {
        CHK_RET(RunAsPeer(param, resCtx, plan));
    }
    // 必须在算法之后、ExecOp 返回之前：把从流收束回 aicpuThread
    CHK_RET(JoinSubThreads(resCtx.aicpuThread, joiner));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
