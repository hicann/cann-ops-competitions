/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom.h"
#include "log.h"
#include "exec_op.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_WRITE_SLICE_BYTES = 256ULL * 1024 * 1024; // AICPU UB 单次搬移上限
constexpr uint64_t MIN_STRIPE_BYTES = 1ULL * 1024 * 1024;        // 条带最小粒度（消息太小不拆条带）
constexpr uint64_t UPLINK_LINK_UNITS = 4; // 每 NPU Clos 上行带宽 = 4 条机内单链路（赛题给定）

// 主线程 -> 子线程同步（子线程等待主线程信号后才开始执行各自任务）
HcclResult ThreadSyncBefore(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[i], 0)));
    }
    for (uint32_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[i], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

// 子线程 -> 主线程同步（主线程等待所有子线程完成任务）
HcclResult ThreadSyncAfter(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], i - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[i], threads[0], i - 1)));
    }
    return HCCL_SUCCESS;
}

// 对端通道按 rank 归组：groups[rank] = 该对端通道列表（每对端 1 条，[0] 即唯一通道）
void BuildChannelGroups(
    const AlgResourceCtx &resCtx, uint32_t rankSize, std::vector<std::vector<const ChannelInfo *>> &groups)
{
    groups.assign(rankSize, {});
    for (const auto &ch : resCtx.channels) {
        groups[ch.remoteRank].push_back(&ch);
    }
}

// 生效条带数：通道数与消息大小共同决定。通道恒为 1（赛题合规约束），保留公式备用。
uint32_t EffectiveStripes(uint64_t chunkBytes, size_t channelCount)
{
    if (channelCount <= 1 || chunkBytes < MIN_STRIPE_BYTES) {
        return 1;
    }
    const uint64_t bySize = chunkBytes / MIN_STRIPE_BYTES;
    return static_cast<uint32_t>(std::min<uint64_t>(channelCount, std::max<uint64_t>(bySize, 1)));
}

// ================= 跨机中继分流 =================
// 拓扑（赛题给定）：机内 Full-Mesh 共 M 条单链路（带宽各 B），每 NPU 的 Clos 上行 = 4B。
// 基线（全部直写）：T = max(M·chunk/(M·B), numCross·chunk/(4B)) —— 上行口瓶颈，机内链路闲置。
// 配平（本方案）：root 出口 = M 条 mesh + 4B 上行；令上行直发 direct 字节、mesh 转发其余跨机字节，
//   解 direct/4 = (M·chunk + totalCross - direct)/M 得 direct = 4·(M·chunk + totalCross)/(M+4)，
//   T 收敛到 (M + numCross)·chunk / ((M+4)·B)（512MB 例：64MB/B → 43.6MB/B，约 1.47×）。
// 三侧（root / root 同 server 的中继 peer / 跨机接收方）调用同一 BuildRelayPlan 保证公式一致。

struct RelayPiece {
    uint32_t crossRank; // 目标跨机 rank
    uint64_t dstOff;    // 在目标 recvBuf 中本 chunk 内的偏移
    uint64_t cclOff;    // 在中继 peer cclBuffer 中的打包偏移
    uint64_t len;       // 字节数
};

// r 是否与 root 同 server：root 自身为真；否则与本 rank 视角下 root 的同区性标记比较
// （root 侧 peerIsLocal[root]=1；跨机接收侧 peerIsLocal[root]=0，两种视角算出的集合一致）
inline bool SameServerAsRoot(const AlgResourceCtx &resCtx, uint32_t myRank, uint32_t root, uint32_t r)
{
    if (r == root) {
        return true;
    }
    const uint8_t rootLocal = (root == myRank) ? 1 : resCtx.peerIsLocal[root];
    return resCtx.peerIsLocal[r] == rootLocal;
}

// 计算 root 直发字节数（4:7 配平）与每个本地 peer 的转发载荷（跨机 chunk 后缀按字节均摊，
// 交错切分保证每个被中继的 chunk 由全部本地 peer 各发一片，接收方等待集合简单确定）
void BuildRelayPlan(const AlgResourceCtx &resCtx, uint32_t myRank, uint32_t root, uint32_t rankSize,
    uint64_t chunkBytes, uint64_t &directBytes, std::vector<uint32_t> &locals, std::vector<uint32_t> &crosses,
    std::vector<std::vector<RelayPiece>> &peerPieces)
{
    locals.clear();
    crosses.clear();
    for (uint32_t r = 0; r < rankSize; r++) {
        if (r == root) {
            continue;
        }
        (SameServerAsRoot(resCtx, myRank, root, r) ? locals : crosses).push_back(r);
    }
    const uint64_t meshLinks = locals.size();
    const uint64_t totalCross = static_cast<uint64_t>(crosses.size()) * chunkBytes;
    directBytes = totalCross;
    if (meshLinks > 0 && totalCross > 0) {
        directBytes = UPLINK_LINK_UNITS * (meshLinks * chunkBytes + totalCross) / (meshLinks + UPLINK_LINK_UNITS);
        directBytes = std::min<uint64_t>(directBytes, totalCross);
    }
    peerPieces.assign(meshLinks, {});
    std::vector<uint64_t> cursor(meshLinks, 0);
    uint64_t off = 0;
    for (const uint32_t c : crosses) {
        uint64_t prefix = (off < directBytes) ? std::min<uint64_t>(directBytes - off, chunkBytes) : 0;
        uint64_t suffix = chunkBytes - prefix;
        if (suffix > 0 && suffix < meshLinks) {
            // 极小后缀无法均摊给全部本地 peer（会出现空片 → 接收方等待集合不一致）：整体转直发
            prefix = chunkBytes;
            suffix = 0;
        }
        for (uint64_t j = 0; j < meshLinks; j++) {
            const uint64_t begin = suffix * j / meshLinks;
            const uint64_t end = suffix * (j + 1) / meshLinks;
            if (end > begin) {
                // dstOff 必须是 chunk 内绝对偏移（prefix + 后缀内偏移）：
                // 边界 chunk 前半由 root 直发，peer 补后半，两者拼起来才是完整 chunk
                peerPieces[j].push_back(RelayPiece{c, prefix + begin, cursor[j], end - begin});
                cursor[j] += end - begin;
            }
        }
        off += chunkBytes;
    }
}

// 中继 root 侧：Phase A 全并行——本地 peer（中继载荷 + final chunk，同链路同线程顺序执行，先载荷后
// final 让 peer 提前进入上行转发）；跨机 chunk 的直发前缀走 root 上行口；自留块多线程并行拷贝。
HcclResult ExecRootRelay(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<std::vector<const ChannelInfo *>> &channelGroups, uint64_t chunkBytes)
{
    const uint32_t myRank = param.myRank;
    const uint32_t root = param.root;
    uint8_t *inBase = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *outBase = static_cast<uint8_t *>(param.outputPtr);

    uint64_t directBytes = 0;
    std::vector<uint32_t> locals;
    std::vector<uint32_t> crosses;
    std::vector<std::vector<RelayPiece>> peerPieces;
    BuildRelayPlan(resCtx, myRank, root, param.rankSize, chunkBytes, directBytes, locals, crosses, peerPieces);

    CHK_RET(ThreadSyncBefore(resCtx.threads));

    // root 自留块：copyThreadNum 个拷贝线程均分，与网络全重叠
    const uint64_t copySlice = (chunkBytes + resCtx.copyThreadNum - 1) / resCtx.copyThreadNum;
    for (uint32_t t = 0; t < resCtx.copyThreadNum; t++) {
        const uint64_t begin = std::min<uint64_t>(static_cast<uint64_t>(t) * copySlice, chunkBytes);
        const uint64_t end = std::min<uint64_t>(begin + copySlice, chunkBytes);
        if (begin >= end) {
            break;
        }
        ThreadHandle copyThread = resCtx.threads[resCtx.jobThreadNum + 1 + t];
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(copyThread, outBase + begin, inBase + root * chunkBytes + begin, end - begin)));
    }

    // 本地 peer：先写中继载荷（落其 cclBuffer，末片融合 RELAY 通知），后写 final chunk（DATA_SIGNAL）
    for (uint32_t j = 0; j < locals.size(); j++) {
        CHK_PRT_RET(1 + j >= resCtx.threads.size(), HCCL_ERROR("[ExecOp] relay thread index overflow"), HCCL_E_INTERNAL);
        ThreadHandle th = resCtx.threads[1 + j];
        const ChannelInfo *ch = channelGroups[locals[j]][0];
        const auto &pieces = peerPieces[j];
        if (pieces.empty()) {
            // 无载荷也要放行对端的 RELAY 等待
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(th, ch->handle, NOTIFY_IDX_RELAY)));
        } else {
            CHK_PRT_RET((pieces.back().cclOff + pieces.back().len) > ch->remoteCclMem.size,
                HCCL_ERROR("[ExecOp] relay payload exceeds peer[%u] ccl buffer", locals[j]), HCCL_E_INTERNAL);
            for (size_t k = 0; k < pieces.size(); k++) {
                const RelayPiece &pc = pieces[k];
                uint8_t *dst = static_cast<uint8_t *>(ch->remoteCclMem.addr) + pc.cclOff;
                const uint8_t *src = inBase + pc.crossRank * chunkBytes + pc.dstOff;
                if (k + 1 == pieces.size()) {
                    // 同线程顺序执行保证末片通知在前面切片全部落盘之后发出
                    CHK_RET(static_cast<HcclResult>(
                        HcommWriteWithNotifyOnThread(th, ch->handle, dst, src, pc.len, NOTIFY_IDX_RELAY)));
                } else {
                    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(th, ch->handle, dst, src, pc.len)));
                }
            }
        }
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(th, ch->handle, ch->remoteOutput,
            inBase + locals[j] * chunkBytes, chunkBytes, NOTIFY_IDX_DATA_SIGNAL)));
    }

    // 跨机 chunk 直发前缀：走 root 上行口，每目标一个线程
    uint32_t jobIdx = locals.size();
    uint64_t off = 0;
    for (const uint32_t c : crosses) {
        const uint64_t prefix = (off < directBytes) ? std::min<uint64_t>(directBytes - off, chunkBytes) : 0;
        if (prefix > 0) {
            CHK_PRT_RET(1 + jobIdx >= resCtx.threads.size(),
                HCCL_ERROR("[ExecOp] direct thread index overflow"), HCCL_E_INTERNAL);
            ThreadHandle th = resCtx.threads[1 + jobIdx];
            jobIdx++;
            const ChannelInfo *ch = channelGroups[c][0];
            CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(th, ch->handle, ch->remoteOutput,
                inBase + c * chunkBytes, prefix, NOTIFY_IDX_DATA_SIGNAL)));
        }
        off += chunkBytes;
    }

    CHK_RET(ThreadSyncAfter(resCtx.threads));
    return HCCL_SUCCESS;
}

// 中继 peer 侧（root 同 server 的非 root rank）：等 RELAY 载荷落地本端 cclBuffer →
// 逐片转发到对应跨机 rank 的 recvBuf（走本 rank 自己的上行口）→ 收自己的 final chunk。
HcclResult ExecPeerRelay(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<std::vector<const ChannelInfo *>> &channelGroups, uint64_t chunkBytes)
{
    const uint32_t myRank = param.myRank;
    const uint32_t root = param.root;

    uint64_t directBytes = 0;
    std::vector<uint32_t> locals;
    std::vector<uint32_t> crosses;
    std::vector<std::vector<RelayPiece>> peerPieces;
    BuildRelayPlan(resCtx, myRank, root, param.rankSize, chunkBytes, directBytes, locals, crosses, peerPieces);

    // 本 rank 在 root 视角本地 peer 列表中的下标
    int64_t myJ = -1;
    for (uint32_t j = 0; j < locals.size(); j++) {
        if (locals[j] == myRank) {
            myJ = static_cast<int64_t>(j);
            break;
        }
    }
    CHK_PRT_RET(myJ < 0, HCCL_ERROR("[ExecOp] rank[%u] not in root-local list", myRank), HCCL_E_INTERNAL);

    const ChannelInfo *rootCh = channelGroups[root][0];
    // 等 root 的中继载荷落地
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(resCtx.threads[0], rootCh->handle, NOTIFY_IDX_RELAY, CUSTOM_TIMEOUT)));

    // 转发：本端 cclBuffer → 跨机 rank recvBuf（同步写在主线程顺序执行，总载荷约 chunk 总量/peer 数，
    // 走本 rank 独立上行口，不在 root 临界路径上）
    for (const RelayPiece &pc : peerPieces[static_cast<size_t>(myJ)]) {
        const ChannelInfo *ch = channelGroups[pc.crossRank][0];
        CHK_PRT_RET((pc.cclOff + pc.len) > resCtx.localBuffer.size,
            HCCL_ERROR("[ExecOp] relay piece exceeds local ccl buffer"), HCCL_E_INTERNAL);
        uint8_t *src = static_cast<uint8_t *>(resCtx.localBuffer.addr) + pc.cclOff;
        uint8_t *dst = static_cast<uint8_t *>(ch->remoteOutput) + pc.dstOff;
        CHK_RET(static_cast<HcclResult>(
            HcommWriteWithNotifyOnThread(resCtx.threads[0], ch->handle, dst, src, pc.len, NOTIFY_IDX_DATA_SIGNAL)));
    }

    // 收自己的 final chunk（root 直写本端 recvBuf）
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(resCtx.threads[0], rootCh->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

// 中继接收侧（与 root 不同 server 的 rank）：按同一公式确定本 chunk 的发送方集合——
// 直发前缀来自 root，中继后缀来自 root 的全部本地 peer（交错切分保证集合简单确定）。
HcclResult ExecCrossRecvRelay(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<std::vector<const ChannelInfo *>> &channelGroups, uint64_t chunkBytes)
{
    const uint32_t myRank = param.myRank;
    const uint32_t root = param.root;

    uint64_t directBytes = 0;
    std::vector<uint32_t> locals;
    std::vector<uint32_t> crosses;
    std::vector<std::vector<RelayPiece>> peerPieces;
    BuildRelayPlan(resCtx, myRank, root, param.rankSize, chunkBytes, directBytes, locals, crosses, peerPieces);

    int64_t myIdx = -1;
    for (uint32_t i = 0; i < crosses.size(); i++) {
        if (crosses[i] == myRank) {
            myIdx = static_cast<int64_t>(i);
            break;
        }
    }
    CHK_PRT_RET(myIdx < 0, HCCL_ERROR("[ExecOp] rank[%u] not in cross list", myRank), HCCL_E_INTERNAL);

    const uint64_t off = static_cast<uint64_t>(myIdx) * chunkBytes;
    const uint64_t prefix = (off < directBytes) ? std::min<uint64_t>(directBytes - off, chunkBytes) : 0;

    if (prefix > 0) {
        const ChannelInfo *rootCh = channelGroups[root][0];
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            resCtx.threads[0], rootCh->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    if (prefix < chunkBytes) {
        for (const uint32_t r : locals) {
            const ChannelInfo *ch = channelGroups[r][0];
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                resCtx.threads[0], ch->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }
    }
    return HCCL_SUCCESS;
}
} // namespace

// Scatter 通信算法编排（AICPU + TS 模式）
// 语义：root 的 sendBuf 按 rank 均分成 rankSize 个连续分块，第 i 块发送给 rank i。
// 大消息（> threshold）两种模式（env HCCL_SCATTER_RELAY 切换，全网一致）：
//   直写模式：每对端一个专属线程并发单边直写对端 recvBuf，跨机对端优先发出；root 自留块并行拷贝。
//   中继模式：4:7 配平分流——root 上行直发前缀，其余跨机字节经 mesh 摊给本地 peer、由各 peer 独立
//             上行口转发（T 从上行瓶颈 64MB/B 收敛到全出口配平 43.6MB/B，约 1.47×）。
// 小消息（≤ threshold）：NBI PUSH（默认）——主线程无句柄直发 15 路融合写（数据+通知一足到达），
//   对端回 ACK 保证完成语义；NBI 关闭时回退 PULL（peer 远端读 root sendBuf，root 只发通知）。
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const uint32_t dataTypeSize = SIZE_TABLE.at(param.dataType);
    const uint64_t chunkBytes = param.count * dataTypeSize; // 每个 rank 收到的字节数

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, chunkBytes)));
        return HCCL_SUCCESS;
    }

    const uint32_t myRank = param.myRank;
    const uint32_t root = param.root;
    const bool isRoot = (myRank == root);

    // 对端通道按 rank 归组（O(1) 查找）
    std::vector<std::vector<const ChannelInfo *>> channelGroups;
    BuildChannelGroups(resCtx, param.rankSize, channelGroups);
    CHK_PRT_RET(channelGroups[root].empty() && !isRoot,
        HCCL_ERROR("[ExecOp] myRank[%u] cannot find channel to root[%u]", myRank, root), HCCL_E_INTERNAL);

    // ================= 大消息路径 =================
    if (chunkBytes > resCtx.smallMsgThresholdBytes) {
        if (resCtx.relayEnabled != 0) {
            // 中继分流模式：三侧分支，公式同源（BuildRelayPlan）
            if (isRoot) {
                return ExecRootRelay(param, resCtx, channelGroups, chunkBytes);
            }
            if (SameServerAsRoot(resCtx, myRank, root, myRank)) {
                return ExecPeerRelay(param, resCtx, channelGroups, chunkBytes);
            }
            return ExecCrossRecvRelay(param, resCtx, channelGroups, chunkBytes);
        }

        // 直写模式（基线）
        uint8_t *inBase = static_cast<uint8_t *>(param.inputPtr);
        uint8_t *outBase = static_cast<uint8_t *>(param.outputPtr);
        if (isRoot) {
            CHK_RET(ThreadSyncBefore(resCtx.threads));

            // root 自留块：copyThreadNum 个拷贝线程均分，与全部网络写重叠
            const uint64_t copySlice = (chunkBytes + resCtx.copyThreadNum - 1) / resCtx.copyThreadNum;
            for (uint32_t t = 0; t < resCtx.copyThreadNum; t++) {
                const uint64_t begin = std::min<uint64_t>(static_cast<uint64_t>(t) * copySlice, chunkBytes);
                const uint64_t end = std::min<uint64_t>(begin + copySlice, chunkBytes);
                if (begin >= end) {
                    break;
                }
                ThreadHandle copyThread = resCtx.threads[resCtx.jobThreadNum + 1 + t];
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(copyThread, outBase + begin,
                    inBase + root * chunkBytes + begin, end - begin)));
            }

            // 网络写任务排序：跨机对端（peerIsLocal==0）优先——Clos 上行带宽最紧张，先把最深的流量推出去
            std::vector<uint32_t> peerRanks;
            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
                if (remoteRank != myRank && !channelGroups[remoteRank].empty()) {
                    peerRanks.push_back(remoteRank);
                }
            }
            std::stable_sort(peerRanks.begin(), peerRanks.end(),
                [&resCtx](uint32_t a, uint32_t b) { return resCtx.peerIsLocal[a] < resCtx.peerIsLocal[b]; });

            // 每个写任务独占一个线程：同步写原语会阻塞所在线程直至传输完成，写并发度由线程数决定
            //（线程内串行多任务 = 大消息成倍变慢，已实测踩坑）。
            uint32_t jobSerial = 0;
            for (const uint32_t remoteRank : peerRanks) {
                const auto &channels = channelGroups[remoteRank];
                const uint32_t stripes = EffectiveStripes(chunkBytes, channels.size());
                const uint8_t *srcBase = inBase + remoteRank * chunkBytes;
                for (uint32_t s = 0; s < stripes; s++) {
                    const ChannelInfo *ch = channels[s];
                    uint8_t *dstBase = static_cast<uint8_t *>(ch->remoteOutput);
                    CHK_PRT_RET(jobSerial >= resCtx.jobThreadNum,
                        HCCL_ERROR("[ExecOp] write job serial[%u] exceeds job threads[%u]", jobSerial,
                            resCtx.jobThreadNum),
                        HCCL_E_INTERNAL);
                    ThreadHandle th = resCtx.threads[1 + jobSerial];
                    jobSerial++;
                    const uint64_t begin = chunkBytes * s / stripes;
                    const uint64_t end = chunkBytes * (s + 1) / stripes;
                    for (uint64_t off = begin; off < end; off += MAX_WRITE_SLICE_BYTES) {
                        const uint64_t len = std::min(MAX_WRITE_SLICE_BYTES, end - off);
                        if (off + len >= end) {
                            // 每通道最后一个切片：融合 DATA_SIGNAL 通知
                            CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
                                th, ch->handle, dstBase + off, srcBase + off, len, NOTIFY_IDX_DATA_SIGNAL)));
                        } else {
                            CHK_RET(static_cast<HcclResult>(
                                HcommWriteOnThread(th, ch->handle, dstBase + off, srcBase + off, len)));
                        }
                    }
                }
            }
            CHK_RET(ThreadSyncAfter(resCtx.threads));
        } else {
            // 非 root：对 root 的前 stripes 条通道各等一次 DATA_SIGNAL（与 root 侧切片规则对称）
            const auto &rootChannels = channelGroups[root];
            const uint32_t stripes = EffectiveStripes(chunkBytes, rootChannels.size());
            for (uint32_t s = 0; s < stripes; s++) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    resCtx.threads[0], rootChannels[s]->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }
        }
        return HCCL_SUCCESS;
    }

    // ================= 小消息路径 =================
    uint8_t *inBase = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *outBase = static_cast<uint8_t *>(param.outputPtr);
    if (resCtx.nbiEnabled != 0) {
        // NBI PUSH：主线程单线程 NBI 直发 15 路融合写——数据与就绪通知一足到达（PULL 需
        // 通知去 + 读请求回 + 数据到三足），免去线程调度与 barrier。Fence 收尾等 sendBuf 读走
        // 即安全返回，不引入对端 ACK 往返；对端正确性由其 DATA_SIGNAL 等待保证。
        if (isRoot) {
            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
                if (remoteRank == myRank || channelGroups[remoteRank].empty()) {
                    continue;
                }
                const ChannelInfo *ch = channelGroups[remoteRank][0];
                CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyNbiOnThread(
                    resCtx.threads[0], ch->handle, ch->remoteOutput, inBase + remoteRank * chunkBytes, chunkBytes,
                    NOTIFY_IDX_DATA_SIGNAL)));
            }
            // 自留块本地拷贝（与 NBI 网络写重叠）
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(resCtx.threads[0], outBase, inBase + root * chunkBytes, chunkBytes)));
            // Fence 收尾：等所有 NBI 写把 sendBuf 读走（可安全复用），不引入对端 ACK 往返
            CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(resCtx.threads[0])));
        } else {
            // 非 root：等就绪通知（数据已直接落在本端 recvBuf），无需回 ACK
            const ChannelInfo *rootCh = channelGroups[root][0];
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                resCtx.threads[0], rootCh->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }
        return HCCL_SUCCESS;
    }

    // 小消息 PULL 路径（NBI 关闭时回退）：peer 直接远端读 root sendBuf，root 只发通知
    if (isRoot) {
        // 通知先行：peer 收到即可远端读 root 的 sendBuf（数据就绪由算子流序保证），与本地自留块拷贝重叠
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
            if (remoteRank == myRank || channelGroups[remoteRank].empty()) {
                continue;
            }
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                resCtx.threads[0], channelGroups[remoteRank][0]->handle, NOTIFY_IDX_DATA_SIGNAL)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], outBase, inBase + root * chunkBytes, chunkBytes)));
    } else {
        // 非 root：等通知后直接从 root 的 sendBuf 远端读自己的分块（省 ccl buffer 暂存拷贝）
        const ChannelInfo *rootCh = channelGroups[root][0];
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            resCtx.threads[0], rootCh->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(resCtx.threads[0], rootCh->handle, outBase,
            static_cast<uint8_t *>(rootCh->remoteInput) + myRank * chunkBytes, chunkBytes)));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
