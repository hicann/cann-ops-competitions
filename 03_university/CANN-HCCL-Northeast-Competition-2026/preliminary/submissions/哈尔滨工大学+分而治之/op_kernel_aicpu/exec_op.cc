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
#include <limits>
#include <map>
#include <vector>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
    constexpr uint64_t MAX_DATA_SIZE = 256ULL * 1024 * 1024;            // 单次传输任务最大字节数
    constexpr uint64_t SMALL_MSG_MAX_TOTAL_BYTES = 16ULL * 1024 * 1024; // 小消息拉取路径阈值（输入总字节）
    constexpr uint32_t CHANNEL_NOTIFY_NUM = 8;                          // 与 Host 侧申请的 channel Notify 数量一致

    // 双路径中继（2 Server x 8 NPU）：d0 = NUM/DEN * m 走机内扇出-跨机中继，d1 走 root 机间口直发。
    // NUM/DEN=8/22 时机内 Mesh 与 root 机间口在 15m/11 处配平（推导见 design_relay_20260908.md）；
    // 当前 9/22 为实验 H2 配平扫描值（试探真实 mesh:clos 带宽比是否偏离 1:4）。
    constexpr uint32_t RELAY_RANK_SIZE = 16;
    constexpr uint32_t RELAY_SERVER_RANKS = 8;
    constexpr uint32_t RELAY_THREAD_NUM = 9;        // 主线程 + 7 机内 worker + 1 机间 worker
    constexpr uint32_t RELAY_CLOS_WORKER = 8;       // 机间直发 worker 的线程下标
    constexpr uint32_t THREAD_NOTIFY_IDX_START = 0; // worker 起跑枪（worker 本线程 0 号 notify）
    constexpr uint64_t RELAY_SPLIT_ALIGN = 128;     // d0/d1 切分对齐
    constexpr uint64_t RELAY_SPLIT_NUM = 9; // d0 = sliceBytes * NUM / DEN（实验 H2 配平扫描：4/11=8/22，本版 9/22）
    constexpr uint64_t RELAY_SPLIT_DEN = 22;

    using ChannelMap = std::unordered_map<uint32_t, const ChannelInfo *>;

    HcclResult ValidateTensorShape(const OpParam &param, uint32_t dataTypeSize, uint64_t &sliceBytes)
    {
        CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
            HCCL_ERROR("Invalid rank metadata, myRank[%u], rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
        CHK_PRT_RET(param.root >= param.rankSize,
            HCCL_ERROR("Invalid root[%u], rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA);
        CHK_PRT_RET(dataTypeSize == 0, HCCL_ERROR("Invalid data type size[0]"), HCCL_E_PARA);

        constexpr uint64_t maxValue = std::numeric_limits<uint64_t>::max();
        CHK_PRT_RET(param.count > maxValue / dataTypeSize,
            HCCL_ERROR("Slice byte size overflow, count[%llu], dataTypeSize[%u]",
                static_cast<unsigned long long>(param.count), dataTypeSize),
            HCCL_E_PARA);
        sliceBytes = param.count * dataTypeSize;
        CHK_PRT_RET(sliceBytes != 0 && static_cast<uint64_t>(param.rankSize) > maxValue / sliceBytes,
            HCCL_ERROR("Input byte size overflow, rankSize[%u], sliceBytes[%llu]", param.rankSize,
                static_cast<unsigned long long>(sliceBytes)),
            HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    HcclResult BuildChannelMap(
        const OpParam &param, const AlgResourceCtx &resCtx, ChannelMap &channelMap, uint64_t &globalMinBuffer)
    {
        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0,
            HCCL_ERROR("Invalid local HCCL buffer, addr[%p], size[%llu]", resCtx.localBuffer.addr,
                static_cast<unsigned long long>(resCtx.localBuffer.size)),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.channels.size() != static_cast<size_t>(param.rankSize - 1),
            HCCL_ERROR("Channel count mismatch, actual[%llu], expected[%u]",
                static_cast<unsigned long long>(resCtx.channels.size()), param.rankSize - 1),
            HCCL_E_INTERNAL);

        globalMinBuffer = resCtx.localBuffer.size;
        channelMap.reserve(resCtx.channels.size());
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank,
                HCCL_ERROR("Invalid channel remoteRank[%u], myRank[%u], rankSize[%u]", channel.remoteRank, param.myRank,
                    param.rankSize),
                HCCL_E_INTERNAL);
            CHK_PRT_RET(channel.notifyNum < CHANNEL_NOTIFY_NUM,
                HCCL_ERROR("Insufficient channel Notify, remoteRank[%u], actual[%u], required[%u]", channel.remoteRank,
                    channel.notifyNum, CHANNEL_NOTIFY_NUM),
                HCCL_E_INTERNAL);
            CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size == 0,
                HCCL_ERROR("Invalid remote HCCL buffer, remoteRank[%u], addr[%p], size[%llu]", channel.remoteRank,
                    channel.remoteCclMem.addr, static_cast<unsigned long long>(channel.remoteCclMem.size)),
                HCCL_E_INTERNAL);

            auto insertResult = channelMap.emplace(channel.remoteRank, &channel);
            CHK_PRT_RET(!insertResult.second, HCCL_ERROR("Duplicate channel for remoteRank[%u]", channel.remoteRank),
                HCCL_E_INTERNAL);
            globalMinBuffer = std::min(globalMinBuffer, channel.remoteCclMem.size);
        }
        CHK_PRT_RET(channelMap.size() != static_cast<size_t>(param.rankSize - 1),
            HCCL_ERROR("Peer channel map is incomplete, actual[%llu], expected[%u]",
                static_cast<unsigned long long>(channelMap.size()), param.rankSize - 1),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    // 本地拷贝，按 MAX_DATA_SIZE 分块（dst/src 均为本端地址）
    HcclResult LocalCopyChunked(ThreadHandle thread, void *dst, const void *src, uint64_t totalBytes)
    {
        uint64_t copied = 0;
        while (copied < totalBytes) {
            const uint64_t subBytes = std::min(MAX_DATA_SIZE, totalBytes - copied);
            void *subDst = static_cast<uint8_t *>(dst) + copied;
            const void *subSrc = static_cast<const uint8_t *>(src) + copied;
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, subDst, subSrc, subBytes)));
            copied += subBytes;
        }
        return HCCL_SUCCESS;
    }

    // 小消息拉取路径：root 将全量输入铺入本端 CCL Buffer 并广播 DATA 信号；
    // 各 rank 直接从 root 的 CCL Buffer 读取属于自己的分片到输出，完成后回 ACK。
    // 优势：任务数最少、对端直读用户输出免去中转拷贝、15 路读天然链路并发。
    // 实验 C4（2 跳 4 组长树，归因自 R/C1/C2/C3/C3b：root Record 约 2us/个，每加一跳
    // 树深代价约 6us）：连续 4 卡一组，负责人一律为组内最小 rank——root 不再担任自己组的
    // 负责人，只做纯数据源，DATA 只发给各组组长（通常 4 个 Record），组长转发给成员，
    // ACK 沿同树反向汇总。深度统一 2 跳（C3 为 3 跳）。root 恰好是组 base 时兼任该组组长。
    // 完成依赖一条不删。成员↔组长的 channel 在本路径原本闲置，notify
    // idx 沿用 NOTIFY_IDX_ACK / NOTIFY_IDX_DATA_SIGNAL，每对 channel 上仍严格 1:1。
    constexpr uint32_t PULL_GROUP_SIZE = 4;

    struct PullGroup {
        uint32_t base = 0;   // 本组最小 rank（即组长）
        uint32_t end = 0;    // 本组上界（不含）
        uint32_t leader = 0; // 本组负责人（恒为 base，root 不例外）
    };

    PullGroup GroupOf(uint32_t rankSize, uint32_t root, uint32_t rank)
    {
        (void)root; // C4 起 root 不担任组长
        PullGroup group;
        group.base = (rank / PULL_GROUP_SIZE) * PULL_GROUP_SIZE;
        group.end = std::min(group.base + PULL_GROUP_SIZE, rankSize);
        group.leader = group.base;
        return group;
    }

    HcclResult ExecPullPath(const OpParam &param, const AlgResourceCtx &resCtx, const ChannelMap &channelMap,
        ThreadHandle thread, uint64_t sliceBytes, uint64_t totalBytes)
    {
        auto requireChannel = [&](uint32_t peer, const ChannelInfo *&channel) -> HcclResult {
            auto channelIt = channelMap.find(peer);
            CHK_PRT_RET(channelIt == channelMap.end(),
                HCCL_ERROR("Missing channel to rank[%u], myRank[%u]", peer, param.myRank), HCCL_E_INTERNAL);
            channel = channelIt->second;
            return HCCL_SUCCESS;
        };
        if (param.myRank == param.root) {
            // 全量输入连续铺入 CCL：分片 i 位于 i*sliceBytes，供各 rank 按自身编号寻址
            CHK_RET(LocalCopyChunked(thread, resCtx.localBuffer.addr, param.inputPtr, totalBytes));
            // 实验 C4：root 只通知各组组长（通常 4 个 Record）；
            // root 恰好是组 base 时兼任该组组长，直接通知本组成员。
            const PullGroup rootGroup = GroupOf(param.rankSize, param.root, param.myRank);
            for (uint32_t base = 0; base < param.rankSize; base += PULL_GROUP_SIZE) {
                if (base == param.root) {
                    continue; // root 兼任该组组长，无需通知自己
                }
                const ChannelInfo *leaderChannel = nullptr;
                CHK_RET(requireChannel(base, leaderChannel));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, leaderChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
            }
            if (param.root == rootGroup.base) {
                for (uint32_t peer = rootGroup.base; peer < rootGroup.end; ++peer) {
                    if (peer == param.root) {
                        continue;
                    }
                    const ChannelInfo *peerChannel = nullptr;
                    CHK_RET(requireChannel(peer, peerChannel));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyRecordOnThread(thread, peerChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
                }
            }
            // root 自己的分片直接由输入拷到输出，与对端拉取并发
            const uint8_t *selfSrc
                = static_cast<const uint8_t *>(param.inputPtr) + static_cast<uint64_t>(param.root) * sliceBytes;
            CHK_RET(LocalCopyChunked(thread, param.outputPtr, selfSrc, sliceBytes));
            // 收尾：root 兼任组长时收本组成员 ACK；再等所有组组长的组 ACK，
            // 全部到齐才能保证下一次调用可以安全覆写 CCL Buffer
            if (param.root == rootGroup.base) {
                for (uint32_t peer = rootGroup.base; peer < rootGroup.end; ++peer) {
                    if (peer == param.root) {
                        continue;
                    }
                    const ChannelInfo *peerChannel = nullptr;
                    CHK_RET(requireChannel(peer, peerChannel));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(thread, peerChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                }
            }
            for (uint32_t base = 0; base < param.rankSize; base += PULL_GROUP_SIZE) {
                if (base == param.root) {
                    continue;
                }
                const ChannelInfo *leaderChannel = nullptr;
                CHK_RET(requireChannel(base, leaderChannel));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, leaderChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            }
        } else {
            const ChannelInfo *rootChannelPtr = nullptr;
            CHK_RET(requireChannel(param.root, rootChannelPtr));
            const ChannelInfo &rootChannel = *rootChannelPtr;

            const PullGroup myGroup = GroupOf(param.rankSize, param.root, param.myRank);
            const bool isLeader = (myGroup.leader == param.myRank);
            if (isLeader) {
                // 组长：收 root 的 DATA 后先向本组成员转发，再读自己的分片
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    thread, rootChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                for (uint32_t peer = myGroup.base; peer < myGroup.end; ++peer) {
                    if (peer == param.myRank || peer == param.root) {
                        continue;
                    }
                    const ChannelInfo *peerChannel = nullptr;
                    CHK_RET(requireChannel(peer, peerChannel));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyRecordOnThread(thread, peerChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
                }
            } else {
                // 成员：DATA 由本组组长转发
                const ChannelInfo *leaderChannel = nullptr;
                CHK_RET(requireChannel(myGroup.leader, leaderChannel));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    thread, leaderChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }
            const uint8_t *remoteSrc = static_cast<const uint8_t *>(rootChannel.remoteCclMem.addr)
                                       + static_cast<uint64_t>(param.myRank) * sliceBytes;
            uint64_t fetched = 0;
            while (fetched < sliceBytes) {
                const uint64_t subBytes = std::min(MAX_DATA_SIZE, sliceBytes - fetched);
                void *subDst = static_cast<uint8_t *>(param.outputPtr) + fetched;
                CHK_RET(static_cast<HcclResult>(
                    HcommReadOnThread(thread, rootChannel.handle, subDst, remoteSrc + fetched, subBytes)));
                fetched += subBytes;
            }
            if (isLeader) {
                // 组长：收齐本组成员 ACK 后向 root 回组 ACK
                for (uint32_t peer = myGroup.base; peer < myGroup.end; ++peer) {
                    if (peer == param.myRank || peer == param.root) {
                        continue;
                    }
                    const ChannelInfo *peerChannel = nullptr;
                    CHK_RET(requireChannel(peer, peerChannel));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(thread, peerChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, NOTIFY_IDX_ACK)));
            } else {
                const ChannelInfo *leaderChannel = nullptr;
                CHK_RET(requireChannel(myGroup.leader, leaderChannel));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, leaderChannel->handle, NOTIFY_IDX_ACK)));
            }
        }
        return HCCL_SUCCESS;
    }

    // ---------------- 双路径中继（2x8 拓扑专用，16 rank 大消息） ----------------
    // 结构对齐官方 ParallelMeshNHR 的多线程范式：每个 worker 队列以"起跑枪"
    // （主线程 record、worker wait 本线程 THREAD_NOTIFY_IDX_START）开头，以 record 主线程
    // 完成信号收尾，保证 worker 流在 checker 任务图中从主流程可达。
    // 每对 rank 只用一条 channel；同一 channel 上多条逻辑流各占独立 notify 对；
    // 每条流都是"预发 N 个 ACK + 每 chunk 两拍握手 + 尾随 N 个 Wait ACK"的多槽流水协议：
    // 槽位区域切 RELAY_SUB_SLOT_NUM 个子槽交替使用（N = 子槽数）。
    // 实验 G：单子槽（N=1，严格乒乓，chunk=整个槽位区域）对照双子槽流水。
    constexpr uint64_t RELAY_SMALL_SLOT_MAX = 128ULL * 1024 * 1024;
    constexpr uint32_t RELAY_SUB_SLOT_NUM = 1; // 每槽位区域的子槽数（1=单槽乒乓，2=双 buffer 流水）
    // 中继路径 notify 布局：中继/直发流 ACK base 0、DATA base 2；本地片流 ACK base 4、DATA base 6。
    // 每个子槽用独立 id（base + k % RELAY_SUB_SLOT_NUM），保证每个 id 上 Record/Wait 严格一一交替，
    // 满足 checker 的 1:1 匹配（同 id 多个未消费 Record 会被判 many-to-one 冲突）。
    constexpr uint32_t RELAY_ACK_BASE = 0;
    constexpr uint32_t RELAY_DATA_BASE = 2;
    constexpr uint32_t LOCAL_ACK_BASE = 4;
    constexpr uint32_t LOCAL_DATA_BASE = 6;

    struct RelayPlan {
        uint32_t myBase = 0;     // 本 Server 连续块最小 rank
        uint32_t myPos = 0;      // 我在本 Server 内的位置
        uint32_t srcBase = 0;    // root 所在 Server 的最小 rank
        uint32_t rootPos = 0;    // root 在其 Server 内的位置
        uint32_t peerBase = 0;   // 对端 Server 连续块最小 rank
        bool sourceSide = false; // 我与 root 同 Server
    };

    struct RelayChunks {
        uint64_t d0 = 0;          // 中继部分字节数（分片的 [d1, m) 区间）
        uint64_t d1 = 0;          // 直发部分字节数（分片的 [0, d1) 区间）
        uint64_t cBig = 0;        // 本地片/直发槽位区域大小（含全部子槽）
        uint64_t cSmall = 0;      // 中继槽位区域大小（含全部子槽）
        uint64_t cBigSub = 0;     // 大槽子槽大小（= 单 chunk 字节数）
        uint64_t cSmallSub = 0;   // 小槽子槽大小（= 单 chunk 字节数）
        uint64_t localLoops = 0;  // 本地片流轮数
        uint64_t relayLoops = 0;  // 中继流轮数
        uint64_t directLoops = 0; // 对位卡（j==rootPos）整片直发轮数
        uint64_t partLoops = 0;   // 非对位卡 d1 直发轮数
    };

    HcclResult CheckContiguousBlock(std::vector<uint32_t> &ranks, const char *groupName, uint32_t &base)
    {
        std::sort(ranks.begin(), ranks.end());
        base = ranks.front();
        for (size_t idx = 0; idx < ranks.size(); ++idx) {
            CHK_PRT_RET(ranks[idx] != base + idx,
                HCCL_ERROR("Relay topology group[%s] is not contiguous, idx[%llu], rank[%u], base[%u]", groupName,
                    static_cast<unsigned long long>(idx), ranks[idx], base),
                HCCL_E_NOT_SUPPORT);
        }
        return HCCL_SUCCESS;
    }

    // 按 channel 的 netLayer 把对端分两组：7 个机内 + 8 个机间；两组 rank 必须各自连续。
    // 校验失败显式报错，不回退其他算法（保证线上结果可归因）。
    HcclResult BuildRelayPlan(const OpParam &param, const AlgResourceCtx &resCtx, RelayPlan &plan)
    {
        CHK_PRT_RET(resCtx.threads.size() < RELAY_THREAD_NUM,
            HCCL_ERROR("Insufficient AICPU threads, actual[%llu], required[%u]",
                static_cast<unsigned long long>(resCtx.threads.size()), RELAY_THREAD_NUM),
            HCCL_E_INTERNAL);

        std::map<uint32_t, std::vector<uint32_t>> peersByLayer;
        for (const ChannelInfo &channel : resCtx.channels) {
            peersByLayer[channel.netLayer].push_back(channel.remoteRank);
        }
        CHK_PRT_RET(peersByLayer.size() != 2,
            HCCL_ERROR(
                "Relay expects 2 topology layers, actual[%llu]", static_cast<unsigned long long>(peersByLayer.size())),
            HCCL_E_NOT_SUPPORT);

        auto firstIt = peersByLayer.begin();
        auto secondIt = std::next(firstIt);
        const std::vector<uint32_t> *intraPeers
            = firstIt->second.size() == RELAY_SERVER_RANKS - 1 ? &firstIt->second : &secondIt->second;
        const std::vector<uint32_t> *interPeers
            = firstIt->second.size() == RELAY_SERVER_RANKS ? &firstIt->second : &secondIt->second;
        CHK_PRT_RET(intraPeers->size() != RELAY_SERVER_RANKS - 1 || interPeers->size() != RELAY_SERVER_RANKS,
            HCCL_ERROR("Relay expects 7 intra + 8 inter peers, actual[%llu]/[%llu]",
                static_cast<unsigned long long>(intraPeers->size()),
                static_cast<unsigned long long>(interPeers->size())),
            HCCL_E_NOT_SUPPORT);

        std::vector<uint32_t> localRanks(intraPeers->begin(), intraPeers->end());
        localRanks.push_back(param.myRank);
        uint32_t myBase = 0;
        CHK_RET(CheckContiguousBlock(localRanks, "intra", myBase));
        std::vector<uint32_t> remoteRanks(interPeers->begin(), interPeers->end());
        uint32_t peerBase = 0;
        CHK_RET(CheckContiguousBlock(remoteRanks, "inter", peerBase));

        plan.myBase = myBase;
        plan.myPos = param.myRank - myBase;
        plan.peerBase = peerBase;
        if (param.root >= myBase && param.root < myBase + RELAY_SERVER_RANKS) {
            plan.srcBase = myBase;
            plan.rootPos = param.root - myBase;
            plan.sourceSide = true;
        } else if (param.root >= peerBase && param.root < peerBase + RELAY_SERVER_RANKS) {
            plan.srcBase = peerBase;
            plan.rootPos = param.root - peerBase;
            plan.sourceSide = false;
        } else {
            HCCL_ERROR("Root[%u] outside both topology blocks", param.root);
            return HCCL_E_NOT_SUPPORT;
        }
        return HCCL_SUCCESS;
    }

    HcclResult CalcRelayChunks(uint64_t sliceBytes, uint64_t globalMinBuffer, RelayChunks &chunks)
    {
        // 本端 CCL 需同时容纳大槽（本地片/直发）与小槽（中继），两边都对 globalMinBuffer 取 min
        chunks.cBig = std::min(MAX_DATA_SIZE, globalMinBuffer - globalMinBuffer / 3);
        CHK_PRT_RET(globalMinBuffer <= chunks.cBig,
            HCCL_ERROR("CCL buffer too small for relay, size[%llu]", static_cast<unsigned long long>(globalMinBuffer)),
            HCCL_E_NOT_SUPPORT);
        chunks.cSmall = std::min(RELAY_SMALL_SLOT_MAX, globalMinBuffer - chunks.cBig);

        // d0 = m * NUM/DEN 向下 128B 对齐，d1 为剩余部分（含非对齐尾字节）
        const uint64_t d0raw = (sliceBytes / RELAY_SPLIT_DEN) * RELAY_SPLIT_NUM
                               + (sliceBytes % RELAY_SPLIT_DEN) * RELAY_SPLIT_NUM / RELAY_SPLIT_DEN;
        chunks.d0 = std::min(sliceBytes, d0raw & ~(RELAY_SPLIT_ALIGN - 1));
        chunks.d1 = sliceBytes - chunks.d0;

        // 子槽交替流水：chunk = 子槽大小；接收端预发 RELAY_SUB_SLOT_NUM 个 ACK 表示子槽均空闲
        chunks.cBigSub = chunks.cBig / RELAY_SUB_SLOT_NUM;
        chunks.cSmallSub = chunks.cSmall / RELAY_SUB_SLOT_NUM;
        chunks.localLoops = (sliceBytes + chunks.cBigSub - 1) / chunks.cBigSub;
        chunks.relayLoops = chunks.d0 == 0 ? 0 : (chunks.d0 + chunks.cSmallSub - 1) / chunks.cSmallSub;
        chunks.directLoops = chunks.localLoops;
        chunks.partLoops = (chunks.d1 + chunks.cBigSub - 1) / chunks.cBigSub;
        return HCCL_SUCCESS;
    }

    // root：起跑枪 -> 自片 -> 7 个机内 worker（中继流 + 本地片流）-> 1 个机间 worker（直发流）-> 汇合
    HcclResult ExecRelayRoot(const OpParam &param, const AlgResourceCtx &resCtx, const ChannelMap &channelMap,
        const RelayPlan &plan, const RelayChunks &chunks, uint64_t sliceBytes)
    {
        const ThreadHandle mainThread = resCtx.aicpuThread;
        const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr);

        // 起跑枪：主线程向所有 worker 发 thread-notify，worker 队列第一个任务就是等它
        for (uint32_t w = 1; w < RELAY_THREAD_NUM; ++w) {
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(mainThread, resCtx.threads[w], THREAD_NOTIFY_IDX_START)));
        }

        // 自片：输入直接拷到输出
        CHK_RET(LocalCopyChunked(
            mainThread, param.outputPtr, input + static_cast<uint64_t>(param.root) * sliceBytes, sliceBytes));

        // 机内 worker w=1..7 按序对应 Server 内位置 pos=(rootPos+w)%8
        for (uint32_t w = 1; w < RELAY_SERVER_RANKS; ++w) {
            const uint32_t pos = (plan.rootPos + w) % RELAY_SERVER_RANKS;
            const uint32_t peerRank = plan.myBase + pos;
            auto channelIt = channelMap.find(peerRank);
            CHK_PRT_RET(channelIt == channelMap.end(),
                HCCL_ERROR("Missing channel to intra peer[%u], myRank[%u]", peerRank, param.myRank), HCCL_E_INTERNAL);
            const ChannelInfo &channel = *channelIt->second;
            const ThreadHandle worker = resCtx.threads[w];

            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(worker, THREAD_NOTIFY_IDX_START, CUSTOM_TIMEOUT)));

            // 中继流：peer 的 slot1（偏移 cBig），内容为 R(pos) 分片的 [d1, m) 区间
            // 中继流的尾随 ACK Wait 统一后移到本地片流之后：对端取走最后 N 块的确认
            // 不阻塞本地片流启动（与机间 worker 尾随 ACK 后移同理）。
            if (chunks.relayLoops > 0) {
                const uint8_t *relaySrc = input + static_cast<uint64_t>(plan.peerBase + pos) * sliceBytes + chunks.d1;
                uint8_t *slot1 = static_cast<uint8_t *>(channel.remoteCclMem.addr) + chunks.cBig;
                for (uint64_t k = 0; k < chunks.relayLoops; ++k) {
                    const uint64_t cur = std::min(chunks.cSmallSub, chunks.d0 - k * chunks.cSmallSub);
                    const uint32_t sub = static_cast<uint32_t>(k % RELAY_SUB_SLOT_NUM);
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(worker, channel.handle, RELAY_ACK_BASE + sub, CUSTOM_TIMEOUT)));
                    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                        worker, channel.handle, slot1 + sub * chunks.cSmallSub, relaySrc + k * chunks.cSmallSub, cur)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyRecordOnThread(worker, channel.handle, RELAY_DATA_BASE + sub)));
                }
            }

            // 本地片流：peer 的 slot0（偏移 0），内容为 peerRank 分片全量
            const uint8_t *localSrc = input + static_cast<uint64_t>(peerRank) * sliceBytes;
            uint8_t *slot0 = static_cast<uint8_t *>(channel.remoteCclMem.addr);
            for (uint64_t k = 0; k < chunks.localLoops; ++k) {
                const uint64_t cur = std::min(chunks.cBigSub, sliceBytes - k * chunks.cBigSub);
                const uint32_t sub = static_cast<uint32_t>(k % RELAY_SUB_SLOT_NUM);
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(worker, channel.handle, LOCAL_ACK_BASE + sub, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                    worker, channel.handle, slot0 + sub * chunks.cBigSub, localSrc + k * chunks.cBigSub, cur)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(worker, channel.handle, LOCAL_DATA_BASE + sub)));
            }
            for (uint32_t s = 0; s < RELAY_SUB_SLOT_NUM; ++s) {
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(worker, channel.handle, LOCAL_ACK_BASE + s, CUSTOM_TIMEOUT)));
            }

            // 中继流尾随 Wait：每个子槽 id 各等一次，确认对端已取走最后 N 块（后移至此）
            if (chunks.relayLoops > 0) {
                for (uint32_t s = 0; s < RELAY_SUB_SLOT_NUM; ++s) {
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(worker, channel.handle, RELAY_ACK_BASE + s, CUSTOM_TIMEOUT)));
                }
            }

            // 汇合：worker w 完成信号发往主线程 w 号 notify
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(worker, mainThread, w)));
        }

        // 机间 worker：逐目的串行直发（共用 root 一个机间口，串行不损失带宽）
        const ThreadHandle closWorker = resCtx.threads[RELAY_CLOS_WORKER];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(closWorker, THREAD_NOTIFY_IDX_START, CUSTOM_TIMEOUT)));
        for (uint32_t pos = 0; pos < RELAY_SERVER_RANKS; ++pos) {
            const uint32_t remoteRank = plan.peerBase + pos;
            const uint64_t bytes = (pos == plan.rootPos) ? sliceBytes : chunks.d1;
            const uint64_t loops = (pos == plan.rootPos) ? chunks.directLoops : chunks.partLoops;
            auto channelIt = channelMap.find(remoteRank);
            CHK_PRT_RET(channelIt == channelMap.end(),
                HCCL_ERROR("Missing channel to inter peer[%u], myRank[%u]", remoteRank, param.myRank), HCCL_E_INTERNAL);
            const ChannelInfo &channel = *channelIt->second;
            const uint8_t *src = input + static_cast<uint64_t>(remoteRank) * sliceBytes;
            uint8_t *slotB = static_cast<uint8_t *>(channel.remoteCclMem.addr);
            for (uint64_t k = 0; k < loops; ++k) {
                const uint64_t cur = std::min(chunks.cBigSub, bytes - k * chunks.cBigSub);
                const uint32_t sub = static_cast<uint32_t>(k % RELAY_SUB_SLOT_NUM);
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(closWorker, channel.handle, RELAY_ACK_BASE + sub, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                    closWorker, channel.handle, slotB + sub * chunks.cBigSub, src + k * chunks.cBigSub, cur)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(closWorker, channel.handle, RELAY_DATA_BASE + sub)));
            }
        }
        // 尾随 Wait 统一后移：某目的地的最终消费确认不阻塞向其他目的地发送
        for (uint32_t pos = 0; pos < RELAY_SERVER_RANKS; ++pos) {
            const uint32_t remoteRank = plan.peerBase + pos;
            auto channelIt = channelMap.find(remoteRank);
            CHK_PRT_RET(channelIt == channelMap.end(),
                HCCL_ERROR("Missing channel to inter peer[%u], myRank[%u]", remoteRank, param.myRank), HCCL_E_INTERNAL);
            for (uint32_t s = 0; s < RELAY_SUB_SLOT_NUM; ++s) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    closWorker, channelIt->second->handle, RELAY_ACK_BASE + s, CUSTOM_TIMEOUT)));
            }
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(closWorker, mainThread, RELAY_CLOS_WORKER)));

        // 主线程汇合：等全部 worker 完成
        for (uint32_t w = 1; w < RELAY_THREAD_NUM; ++w) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(mainThread, w, CUSTOM_TIMEOUT)));
        }
        return HCCL_SUCCESS;
    }

    // 源侧中继卡（位置 myPos）：worker 转发中继包（slot1 -> 对端 slotA），主线程收本地片（slot0）
    HcclResult ExecRelaySourceCard(const OpParam &param, const AlgResourceCtx &resCtx, const ChannelMap &channelMap,
        const RelayPlan &plan, const RelayChunks &chunks, uint64_t sliceBytes)
    {
        const ThreadHandle mainThread = resCtx.aicpuThread;
        const ThreadHandle forwarder = resCtx.threads[1];
        auto rootIt = channelMap.find(param.root);
        CHK_PRT_RET(rootIt == channelMap.end(),
            HCCL_ERROR("Missing channel to root[%u], myRank[%u]", param.root, param.myRank), HCCL_E_INTERNAL);
        const ChannelInfo &rootChannel = *rootIt->second;
        const uint32_t remoteRank = plan.peerBase + plan.myPos;
        auto remoteIt = channelMap.find(remoteRank);
        CHK_PRT_RET(remoteIt == channelMap.end(),
            HCCL_ERROR("Missing channel to inter pair[%u], myRank[%u]", remoteRank, param.myRank), HCCL_E_INTERNAL);
        const ChannelInfo &remoteChannel = *remoteIt->second;
        uint8_t *localBuf = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        uint8_t *out = static_cast<uint8_t *>(param.outputPtr);

        // 起跑枪必须先于主线程任何阻塞等待下发
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(mainThread, forwarder, THREAD_NOTIFY_IDX_START)));

        // 转发 worker：起跑 wait 是其队列第一个任务
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(forwarder, THREAD_NOTIFY_IDX_START, CUSTOM_TIMEOUT)));
        if (chunks.relayLoops > 0) {
            uint8_t *slotA = static_cast<uint8_t *>(remoteChannel.remoteCclMem.addr) + chunks.cBig;
            for (uint32_t s = 0; s < RELAY_SUB_SLOT_NUM; ++s) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    forwarder, rootChannel.handle, RELAY_ACK_BASE + s))); // slot1 子槽初始空闲
            }
            for (uint64_t k = 0; k < chunks.relayLoops; ++k) {
                const uint64_t cur = std::min(chunks.cSmallSub, chunks.d0 - k * chunks.cSmallSub);
                const uint32_t sub = static_cast<uint32_t>(k % RELAY_SUB_SLOT_NUM);
                const uint64_t subOff = sub * chunks.cSmallSub;
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    forwarder, rootChannel.handle, RELAY_DATA_BASE + sub, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    forwarder, remoteChannel.handle, RELAY_ACK_BASE + sub, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                    forwarder, remoteChannel.handle, slotA + subOff, localBuf + chunks.cBig + subOff, cur)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(forwarder, remoteChannel.handle, RELAY_DATA_BASE + sub)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(forwarder, rootChannel.handle, RELAY_ACK_BASE + sub)));
            }
            // 尾随 Wait：每个子槽 id 各等一次，对端取走最后 N 块
            for (uint32_t s = 0; s < RELAY_SUB_SLOT_NUM; ++s) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    forwarder, remoteChannel.handle, RELAY_ACK_BASE + s, CUSTOM_TIMEOUT)));
            }
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(forwarder, mainThread, 1)));

        // 主线程：本地片接收流（slot0 @0）
        for (uint32_t s = 0; s < RELAY_SUB_SLOT_NUM; ++s) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                mainThread, rootChannel.handle, LOCAL_ACK_BASE + s))); // slot0 子槽初始空闲
        }
        for (uint64_t k = 0; k < chunks.localLoops; ++k) {
            const uint64_t cur = std::min(chunks.cBigSub, sliceBytes - k * chunks.cBigSub);
            const uint32_t sub = static_cast<uint32_t>(k % RELAY_SUB_SLOT_NUM);
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(mainThread, rootChannel.handle, LOCAL_DATA_BASE + sub, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(mainThread, out + k * chunks.cBigSub, localBuf + sub * chunks.cBigSub, cur)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(mainThread, rootChannel.handle, LOCAL_ACK_BASE + sub)));
        }

        // 主线程汇合
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(mainThread, 1, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    // 接收侧卡（位置 myPos）：主线程收两条流——直发进 slotB(@0)，中继进 slotA(@cBig)。
    // 初始空闲信号全部前置，避免发送端先等。
    HcclResult ExecRelayRemoteCard(const OpParam &param, const AlgResourceCtx &resCtx, const ChannelMap &channelMap,
        const RelayPlan &plan, const RelayChunks &chunks, uint64_t sliceBytes)
    {
        const ThreadHandle mainThread = resCtx.aicpuThread;
        auto rootIt = channelMap.find(param.root);
        CHK_PRT_RET(rootIt == channelMap.end(),
            HCCL_ERROR("Missing channel to root[%u], myRank[%u]", param.root, param.myRank), HCCL_E_INTERNAL);
        const ChannelInfo &rootChannel = *rootIt->second;
        const bool isRootPair = (plan.myPos == plan.rootPos);
        const uint64_t directBytes = isRootPair ? sliceBytes : chunks.d1;
        const uint64_t directLoops = isRootPair ? chunks.directLoops : chunks.partLoops;
        uint8_t *localBuf = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        uint8_t *out = static_cast<uint8_t *>(param.outputPtr);

        for (uint32_t s = 0; s < RELAY_SUB_SLOT_NUM; ++s) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                mainThread, rootChannel.handle, RELAY_ACK_BASE + s))); // slotB 子槽初始空闲
        }

        // 中继接收流（slotA @cBig），仅非对位卡且有中继数据时参与
        if (!isRootPair && chunks.relayLoops > 0) {
            const uint32_t srcCardRank = plan.srcBase + plan.myPos;
            auto srcIt = channelMap.find(srcCardRank);
            CHK_PRT_RET(srcIt == channelMap.end(),
                HCCL_ERROR("Missing channel to relay card[%u], myRank[%u]", srcCardRank, param.myRank),
                HCCL_E_INTERNAL);
            const ChannelInfo &srcChannel = *srcIt->second;
            for (uint32_t s = 0; s < RELAY_SUB_SLOT_NUM; ++s) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    mainThread, srcChannel.handle, RELAY_ACK_BASE + s))); // slotA 子槽初始空闲
            }
            for (uint64_t k = 0; k < chunks.relayLoops; ++k) {
                const uint64_t cur = std::min(chunks.cSmallSub, chunks.d0 - k * chunks.cSmallSub);
                const uint32_t sub = static_cast<uint32_t>(k % RELAY_SUB_SLOT_NUM);
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    mainThread, srcChannel.handle, RELAY_DATA_BASE + sub, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
                    out + chunks.d1 + k * chunks.cSmallSub, localBuf + chunks.cBig + sub * chunks.cSmallSub, cur)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(mainThread, srcChannel.handle, RELAY_ACK_BASE + sub)));
            }
        }

        // 直发接收流（slotB @0）
        for (uint64_t k = 0; k < directLoops; ++k) {
            const uint64_t cur = std::min(chunks.cBigSub, directBytes - k * chunks.cBigSub);
            const uint32_t sub = static_cast<uint32_t>(k % RELAY_SUB_SLOT_NUM);
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(mainThread, rootChannel.handle, RELAY_DATA_BASE + sub, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(mainThread, out + k * chunks.cBigSub, localBuf + sub * chunks.cBigSub, cur)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(mainThread, rootChannel.handle, RELAY_ACK_BASE + sub)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult ExecRelayPath(const OpParam &param, const AlgResourceCtx &resCtx, const ChannelMap &channelMap,
        uint64_t sliceBytes, uint64_t globalMinBuffer)
    {
        RelayPlan plan;
        CHK_RET(BuildRelayPlan(param, resCtx, plan));
        RelayChunks chunks;
        CHK_RET(CalcRelayChunks(sliceBytes, globalMinBuffer, chunks));

        if (param.myRank == param.root) {
            return ExecRelayRoot(param, resCtx, channelMap, plan, chunks, sliceBytes);
        }
        if (plan.sourceSide) {
            return ExecRelaySourceCard(param, resCtx, channelMap, plan, chunks, sliceBytes);
        }
        return ExecRelayRemoteCard(param, resCtx, channelMap, plan, chunks, sliceBytes);
    }

    // 大消息直推路径：单线程串行，先确认接收方已到达本次调用、CCL 槽位可用。
    // root 逐 peer 把各分片从用户输入直推到对端 CCL Buffer 的 0 偏移单槽，每轮 Write 后
    // Record(DATA_SIGNAL) 通知、Wait(ACK) 等接收方拷走再复用槽位；接收方 Wait(DATA_SIGNAL)
    // 后从本地 CCL 拷到用户输出并 Record(ACK)，最后一段也必须等 ACK。
    HcclResult ExecPushPath(const OpParam &param, const AlgResourceCtx &resCtx, const ChannelMap &channelMap,
        ThreadHandle thread, uint64_t sliceBytes, uint64_t globalMinBuffer)
    {
        const uint64_t chunkBytes = std::min(globalMinBuffer, MAX_DATA_SIZE);
        const uint64_t loopNum = (sliceBytes + chunkBytes - 1) / chunkBytes;

        if (param.myRank == param.root) {
            // root 自己的分片直接由输入拷到输出
            const uint8_t *selfSrc
                = static_cast<const uint8_t *>(param.inputPtr) + static_cast<uint64_t>(param.root) * sliceBytes;
            CHK_RET(LocalCopyChunked(thread, param.outputPtr, selfSrc, sliceBytes));

            // 逐 peer 串行推送，每轮写入对端 CCL Buffer 单槽（偏移 0）
            for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
                if (peerRank == param.root) {
                    continue;
                }
                auto channelIt = channelMap.find(peerRank);
                CHK_PRT_RET(channelIt == channelMap.end(),
                    HCCL_ERROR("Missing channel to peer[%u], myRank[%u]", peerRank, param.myRank), HCCL_E_INTERNAL);
                const ChannelInfo &channel = *channelIt->second;
                // 初始 READY 与每段 ACK 共用索引，但每个 Record 都有独立的 Wait 消费。
                // 尤其在连续调用切换 root/路径时，不能提前覆写对端上一轮仍在使用的 CCL。
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                const uint8_t *peerSrcBase
                    = static_cast<const uint8_t *>(param.inputPtr) + static_cast<uint64_t>(peerRank) * sliceBytes;
                for (uint64_t loopIdx = 0; loopIdx < loopNum; ++loopIdx) {
                    const uint64_t curBytes = std::min(chunkBytes, sliceBytes - loopIdx * chunkBytes);
                    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle,
                        channel.remoteCclMem.addr, peerSrcBase + loopIdx * chunkBytes, curBytes)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
                }
            }
        } else {
            auto channelIt = channelMap.find(param.root);
            CHK_PRT_RET(channelIt == channelMap.end(),
                HCCL_ERROR("Missing channel to root[%u], myRank[%u]", param.root, param.myRank), HCCL_E_INTERNAL);
            const ChannelInfo &rootChannel = *channelIt->second;

            CHK_RET(
                static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, NOTIFY_IDX_ACK)));
            for (uint64_t loopIdx = 0; loopIdx < loopNum; ++loopIdx) {
                const uint64_t curBytes = std::min(chunkBytes, sliceBytes - loopIdx * chunkBytes);
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    thread, rootChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(
                    HcommLocalCopyOnThread(thread, static_cast<uint8_t *>(param.outputPtr) + loopIdx * chunkBytes,
                        resCtx.localBuffer.addr, curBytes)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, NOTIFY_IDX_ACK)));
            }
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Scatter Kernel on Ascend NPU");

    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank metadata, myRank[%u], rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    const auto dataTypeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(dataTypeIt == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type[%d]", static_cast<int32_t>(param.dataType)), HCCL_E_NOT_SUPPORT);
    const uint32_t dataTypeSize = dataTypeIt->second;

    uint64_t sliceBytes = 0;
    CHK_RET(ValidateTensorShape(param, dataTypeSize, sliceBytes));

    const ThreadHandle thread = resCtx.aicpuThread;
    if (param.rankSize == 1) {
        // 单卡通信域：root 即本卡，输入直接拷到输出
        return LocalCopyChunked(thread, param.outputPtr, param.inputPtr, sliceBytes);
    }

    ChannelMap channelMap;
    uint64_t globalMinBuffer = 0;
    CHK_RET(BuildChannelMap(param, resCtx, channelMap, globalMinBuffer));

    const uint64_t totalBytes = sliceBytes * param.rankSize;
    if (totalBytes <= SMALL_MSG_MAX_TOTAL_BYTES && totalBytes <= globalMinBuffer) {
        return ExecPullPath(param, resCtx, channelMap, thread, sliceBytes, totalBytes);
    }
    if (param.rankSize == RELAY_RANK_SIZE) {
        // 2x8 评测拓扑：双路径中继；拓扑校验失败显式报错，不回退（保证结果可归因）
        return ExecRelayPath(param, resCtx, channelMap, sliceBytes, globalMinBuffer);
    }
    // 非 16 卡通信域的通用兜底（评测不涉及）
    return ExecPushPath(param, resCtx, channelMap, thread, sliceBytes, globalMinBuffer);
}
} // namespace ops_hccl
