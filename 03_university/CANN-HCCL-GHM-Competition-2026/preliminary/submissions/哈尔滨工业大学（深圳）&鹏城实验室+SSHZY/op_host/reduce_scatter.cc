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
// ReduceScatter 算子 —— Host 侧（控制面）
//
// 职责：
//   1) 校验入参并构造 OpParam（算子参数包，随 AICPU Kernel 下发到 Device）；
//   2) 解析通信域拓扑（layer-0 Server 内 Full-Mesh / layer-1 跨 Server Clos）；
//   3) 申请通信资源：Thread（Host/Device 同步 + AICPU 任务线程）、Channel（点对点链路）；
//   4) 把资源上下文 AlgResourceCtx 序列化后注册到通信引擎上下文（按 tag 缓存，重复调用复用）；
//   5) 通过 LaunchAICPUKernel 把算法任务编排下发到 Device 侧 AICPU 执行（见 exec_op.cc）。
//
// Channel 申请策略：全连接（Full-Mesh）。
//   Device 侧主算法为 Recursive Halving（递归折半），第 k 步的通信对端是
//   peer = rank ^ (rankSize/2^k)，可能落在任意 Server 的任意 rank 上；
//   回退算法（分层双 WriteReduce）也只需要全连接的子集。
//   因此对其余每个 rank 各申请一条 Channel，建链时 layer-0 优先、layer-1 兜底。
//
// 大数据量优化（>400MB）：
//   - 均匀分片（Device 侧 exec_op.cc）：消除固定步进导致的极小尾片；
//   - 回退算法 CCL 利用率翻倍：multi-server 场景 maxSliceBytes 从 cclSize/4 → cclSize/2；
//   - 所有 slice 尺寸一致（除最后一片可能微调），尾片不再支付不成比例的握手开销。
// ===========================================================================

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
// 每条 Channel 申请的 Notify 数量：
// 索引 0 = NOTIFY_IDX_ACK（就绪/许可信号），索引 1 = NOTIFY_IDX_DATA_SIGNAL（数据完成信号），预留 1 个。
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

// ========== 提交人: ss | 提交时间: 2026-07-20 10:59:04 BEGIN ==========
// ---------------------------------------------------------------------------
// AcquireDescAnyLayer：为 srcRank -> dstRank 填充一条 Channel 描述符
//
// 选路规则：先查 layer-0（Server 内 Full-Mesh 直连）链路；若两层 rank 之间
// 不存在 layer-0 链路（必然分属两个 Server），回退查 layer-1（跨 Server Clos）。
// 描述符中的协议、端点地址直接取自 RankGraph 返回的链路属性，保证与仿真拓扑一致。
// ---------------------------------------------------------------------------
HcclResult AcquireDescAnyLayer(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc *desc)
{
    uint32_t listSize = 0;
    CommLink *linkList = nullptr;
    uint32_t netLayer = 0; // 0 = Server 内 Full-Mesh 层
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize));
    if (listSize == 0) {
        netLayer = 1; // 1 = 跨 Server Clos 层
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize));
    }
    CHK_PRT_RET(listSize == 0,
        HCCL_ERROR("AcquireDescAnyLayer: no link between rank[%u] and rank[%u]", srcRank, dstRank), HCCL_E_INTERNAL);

    CHK_RET(HcclChannelDescInit(desc, 1));
    CommLink link = linkList[0]; // 同层多条等价链路取第一条即可
    desc->remoteRank = dstRank;
    desc->notifyNum = CHANNEL_NOTIFY_NUM;
    desc->channelProtocol = link.linkAttr.linkProtocol;
    desc->localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc->localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc->localEndpoint.loc = link.srcEndpointDesc.loc;
    desc->remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc->remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc->remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

// ---------------------------------------------------------------------------
// BuildTopology：解析分层拓扑，结果写入 AlgResourceCtx 供 Device 侧算法使用
//
//   serverRanks   : 本 rank 所在 Server 的 rank 列表（layer-0 直连域成员）；
//   myLocalIdx    : 本 rank 在 serverRanks 中的下标（Server 内编号）；
//   interPeerRank : 跨 Server 配对 rank = 另一 Server 中 localIdx 相同的 rank
//                   （2*8 拓扑下即 rank ± 8）；单 Server 通信域时置 INVALID_VALUE_RANKID。
//
// 依赖假设：同一 Server 的 rank 编号连续（本仿真平台 ranktable 满足该假设）。
// ---------------------------------------------------------------------------
HcclResult BuildTopology(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    uint32_t *ranks = nullptr;
    uint32_t rankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &ranks, &rankNum));
    CHK_PRT_RET(rankNum == 0, HCCL_ERROR("BuildTopology: empty layer-0 rank list"), HCCL_E_INTERNAL);

    resCtxHost.serverRanks.assign(ranks, ranks + rankNum);
    resCtxHost.myLocalIdx = INVALID_VALUE_RANKID;
    for (uint32_t i = 0; i < rankNum; i++) {
        if (resCtxHost.serverRanks[i] == param.myRank) {
            resCtxHost.myLocalIdx = i;
            break;
        }
    }
    CHK_PRT_RET(resCtxHost.myLocalIdx == INVALID_VALUE_RANKID,
        HCCL_ERROR("BuildTopology: myRank[%u] not in its layer-0 rank list", param.myRank), HCCL_E_INTERNAL);

    if (rankNum >= param.rankSize) {
        // 整个通信域都在一个 Server 内：无跨 Server 对端
        resCtxHost.interPeerRank = INVALID_VALUE_RANKID;
        return HCCL_SUCCESS;
    }
    // 2 Server 场景：按相同 localIdx 配对（myRank ± rankNum）
    uint32_t myServer = param.myRank / rankNum;
    resCtxHost.interPeerRank = (myServer % 2 == 0) ? (param.myRank + rankNum) : (param.myRank - rankNum);
    return HCCL_SUCCESS;
}

// ---------------------------------------------------------------------------
// AcquireChannelFullMesh：对其余每个 rank 各申请一条 Channel（全连接）
//
// 每条 Channel 顺带记录对端 CCL Buffer 地址（remoteCclMem），Device 侧
// Write/WriteReduce 的目的地址都基于该地址推算。
// ---------------------------------------------------------------------------
HcclResult AcquireChannelFullMesh(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    uint32_t channelNum = param.rankSize - 1;
    std::vector<HcclChannelDesc> desc(channelNum);
    std::vector<ChannelHandle> channels(channelNum);
    uint32_t idx = 0;
    for (uint32_t j = 0; j < param.rankSize; j++) {
        if (j == param.myRank) {
            continue;
        }
        CHK_RET(AcquireDescAnyLayer(comm, param.myRank, j, &desc[idx]));
        idx++;
    }
    // 批量申请：desc[i] 与 channels[i] 一一对应
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, desc.data(), channelNum, channels.data()));
    for (uint32_t i = 0; i < channelNum; i++) {
        ChannelInfo channel;
        channel.remoteRank = desc[i].remoteRank;
        channel.handle = channels[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        void *cclBuf = nullptr;
        uint64_t cclBufSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channels[i], &cclBuf, &cclBufSize));
        channel.remoteCclMem = CommBuffer{cclBuf, cclBufSize};
        resCtxHost.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}
// ========== 提交人: ss | 提交时间: 2026-07-20 10:59:04 END ==========
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    // ========== 提交人: ss | 提交时间: 2026-07-20 10:59:04 BEGIN ==========
    CHK_PTR_NULL(recvBuf); // 骨架遗漏的入参校验
    // ========== 提交人: ss | 提交时间: 2026-07-20 10:59:04 END ==========
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_reduce_scatter");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount; // 每个 rank 的输出元素数（输入为 rankSize * recvCount）
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    // Dfx 诊断信息注册
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 thread，并申请 Notify；同时导出为 AICPU 上可用的 thread
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        // HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // ---- 申请 Thread 资源 ----
        // 多线程 Mesh 算法：1 个主线程 + (rankSize-1) 个从线程。
        // 主线程负责 LocalCopy/LocalReduce + PreSync/PostSync；
        // 每个从线程负责与一个 peer 的 SendRecvWrite（对标官方 InsTempReduceScatterVMesh1D）。
        // Notify 分配：主线程需要 rankSize-1 个 notify（从→主 PostSync），
        // 每个从线程需要 1 个 notify（主→从 PreSync）。
        // 取 max(rankSize, CHANNEL_NOTIFY_NUM) 保证 channel notify 槽位匹配（checker 排序依赖）。
        uint32_t threadNum = param.rankSize;
        uint32_t notifyNumPerThread = std::max(param.rankSize, CHANNEL_NOTIFY_NUM);

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // ---- 申请 Channel 资源 ----
        // ========== 提交人: ss | 提交时间: 2026-07-20 10:59:04 BEGIN ==========
        // 先解析分层拓扑（serverRanks / myLocalIdx / interPeerRank，Device 侧回退算法使用），
        // 再按全连接申请 Channel：Recursive Halving 第 k 步对端为 rank ^ (rankSize/2^k)，
        // 可能位于任意 Server，因此对其余每个 rank 各建一条 Channel（layer-0 优先，layer-1 兜底）。
        CHK_RET(BuildTopology(comm, param, resCtxHost));
        CHK_RET(AcquireChannelFullMesh(comm, param, resCtxHost));
        // ========== 提交人: ss | 提交时间: 2026-07-20 10:59:04 END ==========

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));
        // 申请 CPU 通信引擎上下文，存放 aicpuThread 句柄
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
