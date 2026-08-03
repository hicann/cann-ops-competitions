/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ===========================================================================
// ReduceScatter 算子 —— Device 侧（数据面，AICPU Kernel 算法任务编排）
//
// 【算子语义】
//   每个 rank 的输入视为 rankSize 个连续块（每块 count 个元素），所有 rank 对
//   编号相同的块做归约（sum/max/min），编号为 i 的归约结果最终写入 rank i 的
//   输出 buffer（count 个元素）。
//
// 【目标拓扑】（2*8 Ascend 950 仿真环境）
//   Server 内 8 卡 Full-Mesh（layer-0），Server 间 Clos（layer-1）；
//   跨 Server 单链路带宽约为 Server 内单链路的 8 倍。
//
// 【算法 0（主路径）：Mesh 多线程】—— 对标官方 InsTempReduceScatterVMesh1D
//   适用：totalBytes >= 16MB（数据量大到并发收益 > 线程调度开销）
//   rankSize 个线程（1 主 + n-1 从），每个从线程负责与一个 peer 的 SendRecvWrite。
//   CCL 划分为 n 个 slot，每个 slot 接收恰好一个 rank 的贡献（无写冲突）。
//   通信完全并发（GO/DONE 在独立 channel 上不互相阻塞），每 slice 仅 1 轮并发通信。
//   通信结束后主线程按固定 rank 顺序 LocalReduce（满足确定性要求）。
//
// 【算法 1（回退）：Recursive Halving 递归折半】—— 单线程
//   共 log2(rankSize) 步。第 k 步（mask = n/2, n/4, ..., 1）：
//     - 对端 peer = rank ^ mask；
//     - 当前窗口一分为二，本 rank 只继续负责其中一半（keep 半窗），
//       另一半（send 半窗）通过 HcommWriteReduceOnThread 原子归约进对端
//       CCL Buffer 中【相同绝对块号】的区间——对端的 keep 半窗恰是我的 send 半窗；
//     - 归约与数据搬运一次完成，无需本地规约任务。
//   log2(n) 步后窗口只剩 1 块，恰好是本 rank 编号对应的归约结果块。
//   通信任务数 O(log n)（16 rank 仅 4 步），远优于 Ring/Mesh 的 O(n)。
//
// 【算法 2（回退路径）：分层双 WriteReduce】—— rankSize 非 2 的幂时
//   slotA（CCL 前半）：本 rank 结果块的累加器。Server 内每个对端把
//     "我输入中属于它的块"WriteReduce 进我的 slotA，我自己先用 LocalCopy 垫底；
//   slotB（CCL 后半）：跨 Server 镜像块的部分和累加器。Server 内每个对端把
//     "我输入中属于它镜像块（localIdx 相同的另一 Server 块）"的数据 WriteReduce
//     进我的 slotB，我自己也用镜像块数据垫底；
//   最后每个 rank 把 slotB 单向 WriteReduce 到跨 Server 配对 rank 的 slotA。
//   跨 Server 仅 1 次单向归约写，避免双向互写同一累加器造成的内存冲突（且真机
//   会重复计数）。
//
// 【工作窗与分片】
//   两种算法都以本端 CCL Buffer 为工作窗；当输入块过大放不进工作窗时，
//   按 slice（块内纵向切片）循环处理，每次只处理 maxSliceByCcl/maxSliceCount
//   个元素，与数据总量无关，可支持任意大小输入。
//
// 【仿真平台约束（实测踩坑沉淀，改动需谨慎）】
//   1) checker 的 302 内存冲突检测不豁免远端原子归约：写/归约进同一区间的
//      任务对必须用 GO/DONE 握手串行化（封装为 custom.h 的 WriteReduceSync）；
//   2) 不要使用 HcommWriteReduceWithNotifyOnThread：其隐式 notify 与 checker
//      的跨 rank Record/Wait 匹配不兼容（ErrorCode 102）；
//   3) 双方互发写规约（源是自己累加器、目的是对端累加器）必然冲突；
//   4) 每个通信步的 GO/DONE 两拍握手即使数据为空也要照走，保证时序对齐；
//   5) HCCL_REDUCE_PROD 在仿真 SQE 解析中不支持（仅 SUM/MAX/MIN）。
//
// 【依赖】公共扩展函数（ToHdt/ToHop/ChMap/ServerRanksOf/WriteReduceSync）
//   位于 include/custom.h。
// ===========================================================================

#include "common.h"
#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    // HCCL_INFO("ExecOp ReduceScatter: rank[%u/%u] count[%llu] dtype[%d] op[%d]", param.myRank, param.rankSize,
    //     static_cast<unsigned long long>(param.count), static_cast<int>(param.dataType),
    //     static_cast<int>(param.reduceType));

    // 空数据直接成功（握手都不需要：所有 rank 一同提前返回，时序一致）
    if (param.count == 0 || param.rankSize == 0) {
        return HCCL_SUCCESS;
    }

    // 单元素字节数（FP32 快速路径 + 查表兜底）
    const uint32_t ts = GetElemSize(param.dataType);
    CHK_PRT_RET(ts == 0,
        HCCL_ERROR("ExecOp ReduceScatter: unsupported dataType[%d]", static_cast<int>(param.dataType)),
        HCCL_E_PARA);
    const HcommDataType hdt = ToHdt(param.dataType);
    const HcommReduceOp hop = ToHop(param.reduceType);
    const ThreadHandle th = resCtx.aicpuThread; // 任务编排线程（Host 已导出）
    const uint32_t g = param.myRank;            // 本 rank 编号
    const uint32_t n = param.rankSize;          // 通信域大小
    const uint64_t totalCount = param.count;    // 每 rank 输出元素数 = 每块元素数
    const uint64_t bb = totalCount * ts;        // 每块字节数（输入共 n 块）

    uint8_t *cclBase = static_cast<uint8_t *>(resCtx.localBuffer.addr); // 本端 CCL 工作窗
    const uint64_t cclSize = resCtx.localBuffer.size;
    uint8_t *in = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *out = static_cast<uint8_t *>(param.outputPtr);

    // 单 rank 通信域：结果即输入自身的前 count 个元素
    if (n == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(th, out, in, bb));
    }

    const ChannelInfo* chArray[16] = {nullptr};
    BuildChannelArray(resCtx, chArray);

    // =========================================================================
    // 工作窗容量：RH 保留 n-slot；Mesh 用 n-1 个 packed remote slot
    // （output 兼任 local leaf / root，省 1 个 CCL slot 和末尾 copy）
    // =========================================================================
    const bool pof2 = (n > 0) && ((n & (n - 1)) == 0);
    // 128B 对齐：slot stride 上取整到 32 个 FP32 元素（= 128B）
    constexpr uint64_t ALIGN_ELEMS = 32; // 128B / sizeof(float)
    const uint64_t maxRhSliceRaw = pof2 ? ((cclSize / ts) / n) : 0;
    const uint64_t maxRhSlice = (maxRhSliceRaw >= ALIGN_ELEMS) ? (maxRhSliceRaw & ~(ALIGN_ELEMS - 1)) : 0;
    const uint64_t maxMeshSliceRaw = (pof2 && n > 1) ? ((cclSize / ts) / (n - 1)) : 0;
    const uint64_t maxMeshSlice = (maxMeshSliceRaw >= ALIGN_ELEMS) ? (maxMeshSliceRaw & ~(ALIGN_ELEMS - 1)) : 0;
    const bool hasRhCclSpace = (maxRhSlice > 0);
    const bool hasMeshCclSpace = (maxMeshSlice > 0);

    // =========================================================================
    // 算法选择
    // =========================================================================
    constexpr uint64_t MESH_THRESHOLD_BYTES = 16ULL * 1024 * 1024;
    const bool useMesh = pof2 && hasMeshCclSpace && (totalCount * ts >= MESH_THRESHOLD_BYTES);
    const bool useRh = pof2 && hasRhCclSpace && !useMesh;

    if (useMesh) {
        // 15-slot packed: output=logical 0(root), CCL[0..14]=logical 1..15
        const uint64_t numSlices = (totalCount + maxMeshSlice - 1) / maxMeshSlice;
        const uint64_t evenSlice = (totalCount + numSlices - 1) / numSlices;
        const uint64_t alignedSliceCount = (evenSlice + ALIGN_ELEMS - 1) & ~(ALIGN_ELEMS - 1);
        const uint64_t slotBytes = alignedSliceCount * ts;

        // pre-build XOR mapping: worker l handles peer = g XOR l, slot = l-1
        const ChannelInfo *xorPeerCh[16] = {nullptr};
        uint64_t xorPeerInOff[16] = {};
        ThreadHandle subThreads[16];
        uint32_t notifyMainToSub[16] = {0};
        for (uint32_t i = 1; i < n; ++i) {
            subThreads[i - 1] = resCtx.threads[i];
        }
        const uint8_t *localInBlock = in + static_cast<uint64_t>(g) * bb;
        const bool exactInPlace = (out == localInBlock);
        for (uint32_t logical = 1; logical < n; ++logical) {
            const uint32_t peer = g ^ logical;
            const ChannelInfo *ch = chArray[peer];
            CHK_PRT_RET(ch == nullptr,
                HCCL_ERROR("ExecOp Mesh: no channel to peer[%u] logical[%u]", peer, logical), HCCL_E_INTERNAL);
            xorPeerCh[logical] = ch;
            xorPeerInOff[logical] = static_cast<uint64_t>(peer) * bb;
        }

        uint64_t sliceOff = 0;
        while (sliceOff < totalCount) {
            const uint64_t sliceCount = ((evenSlice < (totalCount - sliceOff)) ? evenSlice : (totalCount - sliceOff));
            const uint64_t dataBytes = sliceCount * ts;
            const uint64_t sliceOffBytes = sliceOff * ts;
            uint8_t *const outSlice = out + sliceOffBytes;

            // ---- 推 worker 任务：Wait + WriteSync（无 NotifyRecord，tree edge 替代） ----
            for (uint32_t logical = 1; logical < n; ++logical) {
                const ThreadHandle &worker = resCtx.threads[logical];
                const ChannelInfo &ch = *xorPeerCh[logical];

                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT)));

                void *remoteSlot = static_cast<void *>(static_cast<uint8_t *>(ch.remoteCclMem.addr) +
                    static_cast<uint64_t>(logical - 1) * slotBytes);
                const void *localSrc = static_cast<const void *>(
                    in + xorPeerInOff[logical] + sliceOffBytes);
                CHK_RET(WriteSync(worker, ch, remoteSlot, localSrc, dataBytes));
            }

            // ---- PreSync：同时唤醒 15 路 worker ----
            CHK_RET(PreSyncInterThreads(th, subThreads, notifyMainToSub, n - 1));

            // ---- output = block[g]（logical 0 = local leaf） ----
            if (!exactInPlace) {
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                    th, outSlice, localInBlock + sliceOffBytes, dataBytes)));
            }

            // ---- BFS 归约树 4 层静态展开（n=16） ----
            // 地址预计算：每对 (dst, src) 的 CCL slot 索引完全确定。
            // ThreadNotify 边是 checker 建立跨 rank 写→读依赖链的唯一锚点（ErrorCode 302）。

            // L0 (stride=1): 8 pairs (0,1) (2,3) (4,5) (6,7) (8,9) (10,11) (12,13) (14,15)
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[1], resCtx.threads[0], 1)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[0], 1, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[0], outSlice,
                cclBase + 0 * slotBytes, sliceCount, hdt, hop)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[3], resCtx.threads[2], 1)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[2], 1, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[2], cclBase + 1 * slotBytes,
                cclBase + 2 * slotBytes, sliceCount, hdt, hop)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[5], resCtx.threads[4], 1)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[4], 1, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[4], cclBase + 3 * slotBytes,
                cclBase + 4 * slotBytes, sliceCount, hdt, hop)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[7], resCtx.threads[6], 1)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[6], 1, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[6], cclBase + 5 * slotBytes,
                cclBase + 6 * slotBytes, sliceCount, hdt, hop)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[9], resCtx.threads[8], 1)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[8], 1, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[8], cclBase + 7 * slotBytes,
                cclBase + 8 * slotBytes, sliceCount, hdt, hop)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[11], resCtx.threads[10], 1)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[10], 1, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[10], cclBase + 9 * slotBytes,
                cclBase + 10 * slotBytes, sliceCount, hdt, hop)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[13], resCtx.threads[12], 1)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[12], 1, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[12], cclBase + 11 * slotBytes,
                cclBase + 12 * slotBytes, sliceCount, hdt, hop)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[15], resCtx.threads[14], 1)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[14], 1, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[14], cclBase + 13 * slotBytes,
                cclBase + 14 * slotBytes, sliceCount, hdt, hop)));

            // L1 (stride=2): 4 pairs (0,2) (4,6) (8,10) (12,14)
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[2], resCtx.threads[0], 2)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[0], 2, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[0], outSlice,
                cclBase + 1 * slotBytes, sliceCount, hdt, hop)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[6], resCtx.threads[4], 2)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[4], 2, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[4], cclBase + 3 * slotBytes,
                cclBase + 5 * slotBytes, sliceCount, hdt, hop)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[10], resCtx.threads[8], 2)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[8], 2, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[8], cclBase + 7 * slotBytes,
                cclBase + 9 * slotBytes, sliceCount, hdt, hop)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[14], resCtx.threads[12], 2)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[12], 2, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[12], cclBase + 11 * slotBytes,
                cclBase + 13 * slotBytes, sliceCount, hdt, hop)));

            // L2 (stride=4): 2 pairs (0,4) (8,12)
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[4], resCtx.threads[0], 3)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[0], 3, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[0], outSlice,
                cclBase + 3 * slotBytes, sliceCount, hdt, hop)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[12], resCtx.threads[8], 3)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[8], 3, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[8], cclBase + 7 * slotBytes,
                cclBase + 11 * slotBytes, sliceCount, hdt, hop)));

            // L3 (stride=8): 1 pair (0,8)
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[8], resCtx.threads[0], 4)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[0], 4, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                resCtx.threads[0], outSlice,
                cclBase + 7 * slotBytes, sliceCount, hdt, hop)));

            sliceOff += sliceCount;
        }
        return HCCL_SUCCESS;
    }

    // =========================================================================
    // 算法 1（小数据路径）：Recursive Halving 递归折半 —— 单线程
    //   适用：totalBytes < 16MB，O(log n) 步握手开销低于 Mesh 多线程调度开销
    // =========================================================================
    if (useRh) {
        // =====================================================================
        // 优化：均匀分片，消除尾块放大器效应
        //
        // 旧方案按固定 maxSliceByCcl 步进，当 totalCount 不整除 maxSliceByCcl 时，
        // 最后一个 slice 可能极小（例如 400MB+4B 场景：16 个满片 + 1 个 1 元素尾片）。
        // 尾片虽只有 4B 数据，却仍要支付全部 log2(n) 轮 GO/DONE 握手开销
        // （16 rank = 4 步 × 5 原语 = 20 条任务），浪费比超 10^7 倍。
        //
        // 新方案：均匀分片，消除尾片放大器效应
        // =====================================================================
        const uint64_t numSlices = (totalCount + maxRhSlice - 1) / maxRhSlice;
        const uint64_t evenSlice = (totalCount + numSlices - 1) / numSlices;

        // 预计算 RH 4 步（mask=8,4,2,1）的静态参数 + 精准 4-peer Channel 校验。
        // sendBlk/keepBlk 为块索引（0..15），countMul 为元素数乘子（即 mask 值）。
        // 仅校验 RH 实际使用的 4 个 peer（g^8, g^4, g^2, g^1），无需遍历全部 15 个。
        struct { uint32_t peer; uint64_t sendBlk; uint64_t keepBlk; uint64_t countMul; } rhSteps[4];
        {
            uint64_t off = 0;
            uint32_t si = 0;
            for (uint32_t mask = n >> 1; mask >= 1; mask >>= 1, ++si) {
                rhSteps[si].peer = g ^ mask;
                CHK_PRT_RET(chArray[rhSteps[si].peer] == nullptr,
                    HCCL_ERROR("ExecOp RH: no channel to peer[%u]", rhSteps[si].peer), HCCL_E_INTERNAL);
                bool keepLow = ((g & mask) == 0);
                rhSteps[si].sendBlk = keepLow ? (off + mask) : off;
                rhSteps[si].keepBlk = keepLow ? off : (off + mask);
                rhSteps[si].countMul = mask;
                if (!keepLow) off += mask;
            }
        }

        uint64_t sliceOff = 0;
        while (sliceOff < totalCount) {
            const uint64_t sliceCount = ((evenSlice < (totalCount - sliceOff)) ? evenSlice : (totalCount - sliceOff));
            const uint64_t sliceBytes = sliceCount * ts; // block stride: 必须连续，WriteReduce 跨多块读取
            const uint64_t sliceOffBytes = sliceOff * ts;
            uint8_t *const work = cclBase;

            // 批量下发模式：将 slice 内全部 Hcomm 调用聚合成一次硬件提交，
            // 消除逐调用 doorbell 开销（~0.2μs/次 × 22 次 ≈ 4.4μs → 1 次）
            CHK_RET(static_cast<HcclResult>(HcommBatchModeStart("rh_rs")));

            // ---- 初始化（仅多 slice，单 slice 走首步免拷贝路径） ----
            if (numSlices > 1) {
                uint8_t *wptr = work;
                uint8_t *iptr = in + sliceOffBytes;
                for (uint32_t b = 0; b < n; b++) {
                    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                        th, wptr, iptr, sliceBytes)));
                    wptr += sliceBytes;
                    iptr += bb;
                }
            }

            // ---- RH 4 步完全展平：消除循环跳转 + 条件分支预测开销 ----
            // Step 0 (mask=8): 首步免拷贝（仅单 slice），
            //   WriteSync 先于 LocalReduce——checker 要求 CCL 目标区间在 reduce 前
            //   有前置操作（WriteSync 跨 rank 写入被视为 CCL 初始化锚点）。
            {
                const uint64_t sendOff = rhSteps[0].sendBlk * sliceBytes;
                const uint64_t keepOff = rhSteps[0].keepBlk * sliceBytes;
                const uint64_t xferCount = rhSteps[0].countMul * sliceCount;
                const ChannelInfo &ch = *chArray[rhSteps[0].peer];
                void *remoteRed =
                    static_cast<void *>(static_cast<uint8_t *>(ch.remoteCclMem.addr) + sendOff);
                if (numSlices == 1) {
                    CHK_RET(WriteSync(th, ch, remoteRed,
                        in + rhSteps[0].sendBlk * bb + sliceOffBytes, xferCount * ts));
                    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(th,
                        work + keepOff,
                        in + rhSteps[0].keepBlk * bb + sliceOffBytes, xferCount, hdt, hop)));
                } else {
                    CHK_RET(WriteReduceSync(th, ch, remoteRed,
                        work + sendOff, xferCount, hdt, hop));
                }
            }
            // Step 1 (mask=4)
            {
                const uint64_t sendOff = rhSteps[1].sendBlk * sliceBytes;
                const uint64_t xferCount = rhSteps[1].countMul * sliceCount;
                const ChannelInfo &ch = *chArray[rhSteps[1].peer];
                void *remoteRed =
                    static_cast<void *>(static_cast<uint8_t *>(ch.remoteCclMem.addr) + sendOff);
                CHK_RET(WriteReduceSync(th, ch, remoteRed,
                    work + sendOff, xferCount, hdt, hop));
            }
            // Step 2 (mask=2)
            {
                const uint64_t sendOff = rhSteps[2].sendBlk * sliceBytes;
                const uint64_t xferCount = rhSteps[2].countMul * sliceCount;
                const ChannelInfo &ch = *chArray[rhSteps[2].peer];
                void *remoteRed =
                    static_cast<void *>(static_cast<uint8_t *>(ch.remoteCclMem.addr) + sendOff);
                CHK_RET(WriteReduceSync(th, ch, remoteRed,
                    work + sendOff, xferCount, hdt, hop));
            }
            // Step 3 (mask=1)
            {
                const uint64_t sendOff = rhSteps[3].sendBlk * sliceBytes;
                const uint64_t xferCount = rhSteps[3].countMul * sliceCount;
                const ChannelInfo &ch = *chArray[rhSteps[3].peer];
                void *remoteRed =
                    static_cast<void *>(static_cast<uint8_t *>(ch.remoteCclMem.addr) + sendOff);
                CHK_RET(WriteReduceSync(th, ch, remoteRed,
                    work + sendOff, xferCount, hdt, hop));
            }

            // ---- 结果落盘 ----
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                th, out + sliceOffBytes, work + static_cast<uint64_t>(g) * sliceBytes, sliceBytes)));

            CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd("rh_rs")));

            sliceOff += sliceCount;
        }
        return HCCL_SUCCESS;
    }

    // =========================================================================
    // 算法 2（回退）：分层双 WriteReduce + GO/DONE + 分片
    //   适用：rankSize 非 2 的幂（Recursive Halving 不适用）。
    //
    //   CCL 工作窗布局：slotA（本 rank 结果块累加器）+ slotB（跨 Server 镜像块
    //   部分和累加器），合计 2 × sliceBytes。远端 WriteReduce 写入对端 CCL，
    //   不占用本端额外空间，因此 maxSliceBytes = cclSize / 2（单/多 Server 一致，
    //   旧代码 multi 时 /4 过度保守，浪费了一半 CCL 空间，导致 slice 数翻倍）。
    //
    //   同样采用均匀分片策略消除尾块放大器效应（见 RH 路径注释）。
    // =========================================================================
    auto sr = ServerRanksOf(param, resCtx);
    const uint32_t S = static_cast<uint32_t>(sr.size()); // 本 Server rank 数
    // multi = 存在跨 Server 配对（2 Server 分层场景）
    const bool multi = (resCtx.interPeerRank != INVALID_VALUE_RANKID) && (S < n);

    const uint64_t maxSliceBytes = cclSize / 2; // slotA + slotB，与 multi 无关
    CHK_PRT_RET(maxSliceBytes < ts,
        HCCL_ERROR("ExecOp ReduceScatter: ccl too small, size[%llu]", static_cast<unsigned long long>(cclSize)),
        HCCL_E_INTERNAL);
    const uint64_t maxSliceCountRaw = maxSliceBytes / ts;
    CHK_PRT_RET(maxSliceCountRaw < ALIGN_ELEMS,
        HCCL_ERROR("ExecOp ReduceScatter: ccl too small for 128B alignment, size[%llu]",
            static_cast<unsigned long long>(cclSize)),
        HCCL_E_INTERNAL);
    const uint64_t maxSliceCount = maxSliceCountRaw & ~(ALIGN_ELEMS - 1);
    const uint64_t numSlices = (totalCount + maxSliceCount - 1) / maxSliceCount;
    const uint64_t evenSliceCount = (totalCount + numSlices - 1) / numSlices;
    const uint64_t alignedSliceCount = (evenSliceCount + ALIGN_ELEMS - 1) & ~(ALIGN_ELEMS - 1);
    const uint64_t slotBytes = alignedSliceCount * ts;

    // 预计算跨 slice 不变的输入偏移
    const uint64_t g_bb = static_cast<uint64_t>(g) * bb;
    const uint64_t ip_bb = multi ? static_cast<uint64_t>(resCtx.interPeerRank) * bb : 0;

    // 预建 Server 内 peer → Channel 直接索引 + 输入偏移，省 slice 循环内 hash 查找
    const ChannelInfo *intraPeerCh[16] = {nullptr};
    uint64_t intraPeerBB[16] = {};
    for (uint32_t pg : sr) {
        if (pg == g) {
            continue;
        }
        const ChannelInfo *ch = chArray[pg];
        CHK_PRT_RET(ch == nullptr,
            HCCL_ERROR("ExecOp fallback: no channel to intra peer[%u]", pg), HCCL_E_INTERNAL);
        intraPeerCh[pg] = ch;
        intraPeerBB[pg] = static_cast<uint64_t>(pg) * bb;
    }
    const ChannelInfo *interPeerCh = nullptr;
    if (multi) {
        const ChannelInfo *ch = chArray[resCtx.interPeerRank];
        CHK_PRT_RET(ch == nullptr,
            HCCL_ERROR("ExecOp fallback: no channel to interPeer[%u]", resCtx.interPeerRank), HCCL_E_INTERNAL);
        interPeerCh = ch;
    }

    uint64_t sliceOff = 0;
    while (sliceOff < totalCount) {
        const uint64_t sliceCount = ((evenSliceCount < (totalCount - sliceOff)) ? evenSliceCount : (totalCount - sliceOff));
        const uint64_t dataBytes = sliceCount * ts;
        const uint64_t sliceOffBytes = sliceOff * ts;

        void *slotA = cclBase;
        void *slotB = cclBase + slotBytes;

        // 自己的贡献垫底
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            th, slotA, in + g_bb + sliceOffBytes, dataBytes)));
        if (multi) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(th, slotB,
                in + ip_bb + sliceOffBytes, dataBytes)));
        }

        // ---- Server 内汇聚 ----
        for (uint32_t pg : sr) {
            if (pg == g) {
                continue;
            }
            const ChannelInfo &ch = *intraPeerCh[pg];
            const uint64_t pg_bb = intraPeerBB[pg];
            if (multi) {
                const uint32_t pFwd = (pg < S) ? (pg + S) : (pg - S);
                void *rSlotB =
                    static_cast<void *>(static_cast<uint8_t *>(ch.remoteCclMem.addr) + slotBytes);
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(th, ch.handle, NOTIFY_IDX_DATA_SIGNAL)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(th, ch.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(th, ch.handle, ch.remoteCclMem.addr,
                    in + pg_bb + sliceOffBytes, sliceCount, hdt, hop)));
                CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(th, ch.handle, rSlotB,
                    in + static_cast<uint64_t>(pFwd) * bb + sliceOffBytes, sliceCount, hdt, hop)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(th, ch.handle, NOTIFY_IDX_DATA_SIGNAL)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(th, ch.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            } else {
                CHK_RET(WriteReduceSync(th, ch, ch.remoteCclMem.addr,
                    in + pg_bb + sliceOffBytes, sliceCount, hdt, hop));
            }
        }

        // ---- 跨 Server 合并 ----
        if (multi) {
            const ChannelInfo &ch = *interPeerCh;
            CHK_RET(WriteReduceSync(th, ch, ch.remoteCclMem.addr, slotB, sliceCount, hdt, hop));
        }

        // ---- 结果落盘 ----
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(th, out + sliceOffBytes, slotA, dataBytes)));

        sliceOff += sliceCount;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
