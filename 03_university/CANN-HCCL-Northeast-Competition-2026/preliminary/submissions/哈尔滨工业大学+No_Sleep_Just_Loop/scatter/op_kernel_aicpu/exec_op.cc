/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <algorithm>
#include <cstdint>
#include <unordered_map>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint32_t SERVER_SIZE = 8;
// Worker 阶段号(线程间 notify 槽)。NOTIFY_NUM_PER_THREAD=4，至少够 2 个阶段。
constexpr uint32_t WORKER_PHASE = 0;
// 本 Server / 对端 block 在本地 CCL 中的 slot（0 块）。对端 CCL 起始 slot 复用同号。
constexpr uint32_t LOCAL_SLOT = 0;
// big 部分代理：mesh channel 上 proxy 轮的独立 notify 槽(第 2 轮握手)。
// self 轮复用 NOTIFY_IDX_ACK(0)/NOTIFY_IDX_DATA_SIGNAL(1)；proxy 轮必须独立 idx(2/3)。
constexpr uint32_t PROXY_NOTIFY_ACK = 2;
constexpr uint32_t PROXY_NOTIFY_DATA = 3;
// small 路径本 Server Mesh 分发使用的 worker 数(< 实际 7 条 mesh channel)。
// 单 chunk 下 7 Record + 7 Wait = 14 次握手；减为 3 个 worker(每个轮转发 2~3 条 channel)则只需 6 次。
// 代价：单 worker 内多笔 Write 串行，最长 worker 的 mesh 耗时从 1 块(≈0.76us)增至 3 块(≈2.3us)。
// 依据见 docs/scatter-version-evolution.md §7(同步原语≈0.7~1.4us/个)与 §7.2：预期净省 4~10us。
// 注意：线程仍按 7 个申请(一次性、缓存在 EngineCtx，per-call 无开销)，big 路径仍用全部 7 个。
constexpr uint32_t SMALL_MESH_WORKERS = 3;
// 分路径阈值(字节)：≤ 阈值走两级分层(subRoot 中转)；> 阈值走部分代理(多 clos 口)。
// ⚠️ 比较对象是 totalBytes = param.count * elementBytes，即【每 rank】字节数，**不是 -b 的值**。
//    hccl_test 的 -b 是全部 rank 总字节，用例内先 /rank_size 再作 param.count
//    (hccl_scatter_rootinfo_test.cc:126-130)。换算表见 docs/scatter-version-evolution.md §4.2。
//    官方 case 实际 totalBytes：64B→4B, 512KB→32KB, 400M+4B→25MB, 512MB→32MB。
// 2026-09-10 A/B 已定论：设 1<<14(16KB) 让 512KB(32KB) 走 big，实测 512KB **45us→71us(+58%)**，
//    明确更差。原因：big 路径 root 侧启动 7 mesh + 8 clos 共 15 个 worker(30 个 ThreadNotify)且
//    每轮 Write 前有 WaitACK 往返；small 路径仅 7 worker 且 v8 起已去掉 ACK。32KB 数据搬运本身
//    仅约 2us，故 big 的控制面开销压倒一切。**512KB 必须留在 small 两级，勿再挪 big**。
//    勿误用 -b 值设阈值(曾踩坑：设 256KB 结果是 no-op)。
constexpr uint64_t SMALL_PATH_BYTES = 1ULL << 20;  // 1MB
// big 单 chunk 上限(单槽，撤销双槽后)：min(256MB, localBuffer)。
constexpr uint64_t BIG_CHUNK_BYTES = 256ULL * 1024 * 1024;
// 部分代理比例 p = PROXY_NUM / PROXY_DEN：跨服块后 p 部分经本 Server peer 转发，前 (1-p) root 直发。
// 关键路径 = max(root Clos 负载, root Mesh 负载)，k = Clos单口/Mesh单链路 带宽比：
//   root Clos 负载 = (8-7p)b/k      root Mesh 负载 = (1+p)b      (b = 总输入/16)
// 实测 sweep(512MB)：p=0 1.44ms(悬崖) / p=1/4 1.16ms / p=1/3 1.07ms(最优) / p=4/11 1.09ms。
// 反推 k≈4；p 过小则 root Clos 成瓶颈(p=1/4 时 6.25b/4=1.56 > Mesh 1.25b)，p 过大则 wire 反噬。
// 峰顶在 p≈1/3，已扫定，勿再下探或上调。sweep 改这里。
constexpr uint32_t PROXY_NUM = 1;
constexpr uint32_t PROXY_DEN = 3;
// 在 channel 列表按对端 rank 找通道
const ChannelInfo *FindChannelByRank(const std::vector<ChannelInfo> &chans, uint32_t remoteRank)
{
    for (const auto &ch : chans) {
        if (ch.remoteRank == remoteRank) return &ch;
    }
    return nullptr;
}
} // namespace

// 按数据规模分路径：
//   - small(≤1MB, 如 512KB)：两级分层 + subRoot 中转 + 7 mesh worker。小数据往返主导，最优。
//   - big(>1MB, 如 512MB / 400M+4B)：部分代理 p=PROXY_NUM/PROXY_DEN，单槽、chunk 上限 256MB。
//     跨服块(除与 root 同号那块)切两段：前 (1-p) 段 root 经 8 clos worker 直发对端；
//     后 p 段 root 经 7 mesh worker 送本 Server 对号 peer，peer 用自身 clos 口转发对端；
//     对端收两段拼 recvBuf。root 从用户 input 直接发块(零 staging)。非 root 从 root/peer 收。
// 资源由 Host 一次性申请(7 mesh + 8 clos channel + main + 7 mesh worker + 8 clos worker)。
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PTR_NULL(param.outputPtr);
    const auto typeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", param.dataType), HCCL_E_PARA);
    if (param.count == 0) return HCCL_SUCCESS;
    const uint64_t elementBytes = typeIt->second;
    // big 单槽 chunk 上限：min(256MB, localBuffer)。部分代理下 peer 需 self(整块)+proxy(前段复用 base)，
    // 对端 B 需整块区 bytes，均 ≤ cap。
    const uint64_t maxChunkCount = std::max<uint64_t>(1,
        std::min<uint64_t>(BIG_CHUNK_BYTES, resCtx.localBuffer.size) / elementBytes);

    // 分路径：按总字节数切，小数据走两级分层，大数据走 root 直发。
    const uint64_t totalBytes = param.count * elementBytes;
    const bool smallPath = (totalBytes <= SMALL_PATH_BYTES);

    const uint64_t myRank = param.myRank;
    const uint64_t root = param.root;
    const uint64_t serverSize = SERVER_SIZE;
    const uint64_t myServer = myRank / serverSize;
    const uint64_t rankInServer = myRank % serverSize;
    const uint64_t rootServer = root / serverSize;
    const uint64_t rootRankInServer = root % serverSize;
    // 两级分层(小路径)需要 subRoot 角色：本 Server 的"分发改代"，root 在本 Server 的对应 rank。
    const uint64_t mySubRoot = (myServer == rootServer) ? root
        : (rootServer == 0 ? root + serverSize : root - serverSize);
    const bool isRoot = (myRank == root);
    const bool isSubRoot = (myRank == mySubRoot);
    const bool isOtherServerSubRoot = isSubRoot && (myServer != rootServer);

    CHK_PRT_RET(resCtx.meshThreads.size() < SERVER_SIZE - 1,
        HCCL_ERROR("need 7 mesh worker threads"), HCCL_E_INTERNAL);

    const ThreadHandle main = resCtx.aicpuThread;
    auto *localBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    auto *input = static_cast<uint8_t *>(param.inputPtr);

    for (uint64_t offset = 0; offset < param.count; offset += maxChunkCount) {
        const uint64_t count = std::min(maxChunkCount, param.count - offset);
        const uint64_t bytes = count * elementBytes;
        const uint64_t offsetBytes = offset * elementBytes;

        if (smallPath) {
            // ---------------- 小数据：两级分层 + subRoot 中转，任务数最小化 ----------------
            // ≤1MB 单 chunk、每个接收槽只用一次 → 全链路去掉 ACK(RecordACK/WaitACK)，
            // 每传输只留 sender Write+RecordDATA / receiver WaitDATA；root 本 Server 8 块不再 staging
            // 到 localBase(mesh worker 直接以 input 为源，同 big 能力)，root self 直接 input。
            // 对端 subRoot 收 root 聚合块落 localBase，其 worker 从 localBase 分发。
            if (isRoot) {
                // Level1：root 把对端 Server 8 block 一次 Write 到对端 subRoot(不 WaitACK)
                const uint64_t otherServer = (rootServer == 0) ? 1 : 0;
                const uint64_t otherSubRoot = (rootServer == 0) ? root + serverSize : root - serverSize;
                for (const auto &c : resCtx.closChannels) {
                    if (c.remoteRank != otherSubRoot) continue;
                    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(main, c.handle,
                        static_cast<uint8_t *>(c.remoteCclMem.addr) + (uint64_t)LOCAL_SLOT * bytes,
                        input + (static_cast<uint64_t>(otherServer) * serverSize * param.count + offset) * elementBytes,
                        serverSize * bytes)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(main, c.handle,
                        NOTIFY_IDX_DATA_SIGNAL)));
                }
            } else if (isOtherServerSubRoot) {
                // 对端 subRoot：只收 root 发来那条(不 RecordACK)
                for (const auto &c : resCtx.closChannels) {
                    if (c.remoteRank != root) continue;
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(main, c.handle,
                        NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                }
            }

            // Level0(Server 内 Mesh)：subRoot 并发发本 Server 7 peer(不 WaitACK)
            if (isSubRoot) {
                // 只用 SMALL_MESH_WORKERS 个 worker 承担 7 条 channel(轮转)，握手从 14 降到 6。
                // main Record WORKER_PHASE 唤醒 worker，必须由 main 侧 Wait 消费(配对)；
                // 只 Record 不 Wait 会致 checker unconsumed Record(已验证)，勿省。
                for (uint32_t w = 0; w < SMALL_MESH_WORKERS; ++w)
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(main,
                        resCtx.meshThreads[w], WORKER_PHASE)));
                for (uint32_t w = 0; w < SMALL_MESH_WORKERS; ++w)
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.meshThreads[w],
                        WORKER_PHASE, CUSTOM_TIMEOUT)));

                for (uint32_t i = 0; i < resCtx.meshChannels.size(); ++i) {
                    const ChannelInfo &ch = resCtx.meshChannels[i];
                    const uint32_t peerRankInServer = ch.remoteRank % serverSize;
                    if (peerRankInServer == rankInServer) continue;
                    // 轮转分配到 SMALL_MESH_WORKERS 个 worker 上(每 worker 内该 2~3 笔 Write 串行)
                    const ThreadHandle worker = resCtx.meshThreads[i % SMALL_MESH_WORKERS];
                    // root(本 Server subRoot)直接以 input 为源；对端 subRoot 以收下的 localBase 为源
                    const uint8_t *source = isRoot
                        ? input +
                            ((static_cast<uint64_t>(rootServer) * serverSize + peerRankInServer) * param.count
                                + offset) * elementBytes
                        : localBase + (uint64_t)peerRankInServer * bytes;
                    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(worker, ch.handle,
                        static_cast<uint8_t *>(ch.remoteCclMem.addr) + (uint64_t)LOCAL_SLOT * bytes,
                        source, bytes)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(worker, ch.handle,
                        NOTIFY_IDX_DATA_SIGNAL)));
                }
                // subRoot 自己那块：root 直接 input；对端 subRoot 从收下的 localBase 拷
                const uint8_t *selfSrc = isRoot
                    ? input + (root * param.count + offset) * elementBytes
                    : localBase + (uint64_t)rankInServer * bytes;
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(main,
                    output + offsetBytes, selfSrc, bytes)));
            } else {
                // 普通 rank：等本 Server subRoot 经 Mesh 发来自己那块(不 RecordACK)
                const ChannelInfo *chFound = nullptr;
                for (const auto &ch : resCtx.meshChannels) {
                    if (ch.remoteRank == mySubRoot) { chFound = &ch; break; }
                }
                CHK_PRT_RET(chFound == nullptr, HCCL_ERROR("no mesh channel to subRoot[%u]",
                    static_cast<uint32_t>(mySubRoot)), HCCL_E_INTERNAL);
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(main, chFound->handle,
                    NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                // 数据已由 subRoot 的 worker Write 到本端 CCL buffer，落盘到 recvBuf
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(main,
                    output + offsetBytes, localBase, bytes)));
            }
        } else {
            // ---------------- 大数据：部分代理(p=PROXY_NUM/PROXY_DEN) + 多 clos worker 直发 ----------------
            // 单槽、chunk 上限 256MB。跨服块(除与 root 同号那块)切两段：前 (1-p) 段 root 经 clos 直发到对端；
            // 后 p 段 root 经 mesh 送本 Server 对号 peer，peer 用自身 clos 口转发到对端；对端收两段拼 recvBuf。
            // 与 root 同号那块对端无对应 peer，整块由 root 直发。方向一律 otherServer(勿用 ±8)。
            const uint64_t otherServer = (rootServer == 0) ? 1 : 0;
            // count/2 为防御上限(保证 proxy 段不超过半块)，p <= 1/2 时恒不触发；
            // max(1,..) 保证极小 chunk 下 proxy 段非空(否则对端分段逻辑退化)。
            const uint64_t proxyCount = std::min<uint64_t>(count / 2,
                std::max<uint64_t>(1, (count * PROXY_NUM) / PROXY_DEN));
            const uint64_t directCount = count - proxyCount;
            const uint64_t proxyBytes = proxyCount * elementBytes;
            const uint64_t directBytes = directCount * elementBytes;
            if (isRoot) {
                // 启动 7 mesh + 8 clos worker(主/从流握手)
                CHK_PRT_RET(resCtx.closThreads.size() < resCtx.closChannels.size(),
                    HCCL_ERROR("need %u clos worker threads",
                        static_cast<uint32_t>(resCtx.closChannels.size())), HCCL_E_INTERNAL);
                for (uint32_t i = 0; i < resCtx.meshThreads.size(); ++i)
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(main,
                        resCtx.meshThreads[i], WORKER_PHASE)));
                for (uint32_t i = 0; i < resCtx.meshThreads.size(); ++i)
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.meshThreads[i],
                        WORKER_PHASE, CUSTOM_TIMEOUT)));
                for (uint32_t j = 0; j < resCtx.closThreads.size(); ++j)
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(main,
                        resCtx.closThreads[j], WORKER_PHASE)));
                for (uint32_t j = 0; j < resCtx.closThreads.size(); ++j)
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.closThreads[j],
                        WORKER_PHASE, CUSTOM_TIMEOUT)));

                // mesh worker：本 Server 7 peer 的 self 整块 + 其要转发的跨服块 proxy 段(两轮独立 notify)
                for (uint32_t i = 0; i < resCtx.meshChannels.size(); ++i) {
                    const ChannelInfo &ch = resCtx.meshChannels[i];
                    const ThreadHandle worker = resCtx.meshThreads[i];
                    const uint64_t peerIdx = ch.remoteRank % serverSize;
                    const uint64_t proxyRank = otherServer * serverSize + peerIdx;
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(worker, ch.handle,
                        NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(worker, ch.handle,
                        ch.remoteCclMem.addr,
                        input + (ch.remoteRank * param.count + offset) * elementBytes, bytes)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(worker, ch.handle,
                        NOTIFY_IDX_DATA_SIGNAL)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(worker, ch.handle,
                        PROXY_NOTIFY_ACK, CUSTOM_TIMEOUT)));
                    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(worker, ch.handle,
                        ch.remoteCclMem.addr,
                        input + (proxyRank * param.count + offset + directCount) * elementBytes,
                        proxyBytes)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(worker, ch.handle,
                        PROXY_NOTIFY_DATA)));
                }
                // clos worker：root 直发跨 Server 8 rank(与 root 同号那块整块，其余只前段)
                for (uint32_t j = 0; j < resCtx.closChannels.size(); ++j) {
                    const ChannelInfo &ch = resCtx.closChannels[j];
                    const ThreadHandle worker = resCtx.closThreads[j];
                    const bool fullBlock = ((ch.remoteRank % serverSize) == rootRankInServer);
                    const uint64_t sendBytes = fullBlock ? bytes : directBytes;
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(worker, ch.handle,
                        NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(worker, ch.handle,
                        ch.remoteCclMem.addr,
                        input + (ch.remoteRank * param.count + offset) * elementBytes, sendBytes)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(worker, ch.handle,
                        NOTIFY_IDX_DATA_SIGNAL)));
                }
                // root 自留本 chunk
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(main,
                    output + offsetBytes, input + (root * param.count + offset) * elementBytes, bytes)));
            } else if (myServer == rootServer) {
                // 本 Server peer：收 self(整块，落 base->recvBuf) + 收 proxy 段(复用 base)后 clos 转发对端同号
                const ChannelInfo *chMesh = FindChannelByRank(resCtx.meshChannels, static_cast<uint32_t>(root));
                CHK_PRT_RET(chMesh == nullptr, HCCL_ERROR("peer no mesh channel to root[%u]",
                    static_cast<uint32_t>(root)), HCCL_E_INTERNAL);
                // self 轮
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(main, chMesh->handle,
                    NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(main, chMesh->handle,
                    NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(main,
                    output + offsetBytes, localBase, bytes)));
                // proxy 轮(base 已空)
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(main, chMesh->handle,
                    PROXY_NOTIFY_ACK)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(main, chMesh->handle,
                    PROXY_NOTIFY_DATA, CUSTOM_TIMEOUT)));
                // 经自身 clos 转发 proxy(base 前段)到对端 [directBytes, bytes)
                const uint64_t proxyTarget = otherServer * serverSize + rankInServer;
                const ChannelInfo *chClos = FindChannelByRank(resCtx.closChannels,
                    static_cast<uint32_t>(proxyTarget));
                CHK_PRT_RET(chClos == nullptr, HCCL_ERROR("peer no clos channel to %u",
                    static_cast<uint32_t>(proxyTarget)), HCCL_E_INTERNAL);
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(main, chClos->handle,
                    NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(main, chClos->handle,
                    static_cast<uint8_t *>(chClos->remoteCclMem.addr) + directBytes,
                    localBase, proxyBytes)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(main, chClos->handle,
                    NOTIFY_IDX_DATA_SIGNAL)));
            } else {
                // 对端 rank：收两段拼 recvBuf；与 root 同号那块只收 root 整块
                const bool hasPeer = (rankInServer != rootRankInServer);
                const ChannelInfo *chRoot = FindChannelByRank(resCtx.closChannels, static_cast<uint32_t>(root));
                CHK_PRT_RET(chRoot == nullptr, HCCL_ERROR("remote no clos channel to root[%u]",
                    static_cast<uint32_t>(root)), HCCL_E_INTERNAL);
                if (!hasPeer) {
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(main, chRoot->handle,
                        NOTIFY_IDX_ACK)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(main, chRoot->handle,
                        NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(main,
                        output + offsetBytes, localBase, bytes)));
                } else {
                    const ChannelInfo *chPeer = FindChannelByRank(resCtx.closChannels,
                        static_cast<uint32_t>(rootServer * serverSize + rankInServer));
                    CHK_PRT_RET(chPeer == nullptr, HCCL_ERROR("remote no peer clos channel[%u]",
                        static_cast<uint32_t>(rootServer * serverSize + rankInServer)), HCCL_E_INTERNAL);
                    // root 直发段 [0,directBytes)
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(main, chRoot->handle,
                        NOTIFY_IDX_ACK)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(main, chRoot->handle,
                        NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(main,
                        output + offsetBytes, localBase, directBytes)));
                    // peer 转发段 [directBytes, bytes)
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(main, chPeer->handle,
                        NOTIFY_IDX_ACK)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(main, chPeer->handle,
                        NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(main,
                        output + offsetBytes + directBytes, localBase + directBytes, proxyBytes)));
                }
            }
        }
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
