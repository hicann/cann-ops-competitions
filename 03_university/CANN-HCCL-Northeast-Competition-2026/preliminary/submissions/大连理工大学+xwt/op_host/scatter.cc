/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
// 在指定 netLayer 上查询 srcRank -> dstRank 的首条链路。
// 该层不存在链路时（查询失败或链路数为 0）返回 false，由调用者决定是否回退到下一层。
bool TryGetLink(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink &link)
{
    uint32_t listSize = 0;
    CommLink *linkList = nullptr;
    if (HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize) != HCCL_SUCCESS) {
        return false;
    }
    if (linkList == nullptr || listSize == 0) {
        return false;
    }
    link = linkList[0]; // links 内存由库内管理，立即拷出
    return true;
}

// 构造本 rank 到 dstRank 的 channel 描述符。
// 通信域跨 2 个 Server，框内 peer 在 layer-0 有直连、跨框 peer 只在 layer-1 有链路，
// 因此这里先试 layer-0 再回退 layer-1，协议与两端 endpoint 全部取实际命中层的 CommLink。
HcclResult AcquireDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc *desc, uint32_t &hitLayer)
{
    CommLink link = {};
    hitLayer = NET_LAYER_INTRA;
    bool found = TryGetLink(comm, NET_LAYER_INTRA, srcRank, dstRank, link);
    if (!found) {
        hitLayer = NET_LAYER_INTER;
        found = TryGetLink(comm, NET_LAYER_INTER, srcRank, dstRank, link);
    }
    CHK_PRT_RET(!found,
        HCCL_ERROR("AcquireDesc: no link between rank[%u] and rank[%u] on layer-0 or layer-1", srcRank, dstRank),
        HCCL_E_INTERNAL);
    HCCL_INFO("AcquireDesc: rank[%u] -> rank[%u] hit netLayer[%u]", srcRank, dstRank, hitLayer);

    CHK_RET(HcclChannelDescInit(desc, 1));
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

// 查询某条链路源端 endpoint 的带宽系数。任何一步失败都返回 0（= 不可信）。
uint32_t ProbeBwCoeff(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank)
{
    CommLink link = {};
    if (!TryGetLink(comm, netLayer, srcRank, dstRank, link)) {
        return 0;
    }
    EndpointAttrBwCoeff coeff = 0;
    if (HcclRankGraphGetEndpointInfo(comm, srcRank, &link.srcEndpointDesc, ENDPOINT_ATTR_BW_COEFF,
            static_cast<uint32_t>(sizeof(coeff)), &coeff) != HCCL_SUCCESS) {
        return 0;
    }
    if (coeff < BW_COEFF_MIN || coeff > BW_COEFF_MAX) {
        return 0; // 取值不在合理区间，视为不可信
    }
    return coeff;
}

// 探测框内 / 跨框的带宽系数，供 device 侧现算中继比例 q。
//
// ★ 关键：这里刻意锚定在 rank 0 的两条链路上，而不是"我自己的链路"。
// q 必须在所有 rank 上算出同一个值（root 按它切分、helper 按它决定远端写入偏移、
// 接收方按它决定等几个 notify），任何不一致都会变成数据错或挂死。
// 用固定的全局 rank 编号做查询输入，所有 rank 的输入完全相同，结果自然一致。
void ProbeBwCoeffs(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    uint32_t intraPeerOfRank0 = INVALID_VALUE_RANKID;
    uint32_t interPeerOfRank0 = INVALID_VALUE_RANKID;
    CommLink probe = {};
    for (uint32_t j = 1; j < param.rankSize; j++) {
        if (intraPeerOfRank0 == INVALID_VALUE_RANKID && TryGetLink(comm, NET_LAYER_INTRA, 0, j, probe)) {
            intraPeerOfRank0 = j;
        }
        if (interPeerOfRank0 == INVALID_VALUE_RANKID && !TryGetLink(comm, NET_LAYER_INTRA, 0, j, probe) &&
            TryGetLink(comm, NET_LAYER_INTER, 0, j, probe)) {
            interPeerOfRank0 = j;
        }
        if (intraPeerOfRank0 != INVALID_VALUE_RANKID && interPeerOfRank0 != INVALID_VALUE_RANKID) {
            break;
        }
    }
    if (intraPeerOfRank0 == INVALID_VALUE_RANKID || interPeerOfRank0 == INVALID_VALUE_RANKID) {
        HCCL_WARNING("HcclScatter: bw coeff probe skipped (intraPeer[%u] interPeer[%u])", intraPeerOfRank0,
            interPeerOfRank0);
        return; // 两个系数保持 0 -> device 侧 q = 0 -> 纯直发
    }
    resCtxHost.bwCoeffIntra = ProbeBwCoeff(comm, NET_LAYER_INTRA, 0, intraPeerOfRank0);
    resCtxHost.bwCoeffInter = ProbeBwCoeff(comm, NET_LAYER_INTER, 0, interPeerOfRank0);
    if (resCtxHost.bwCoeffIntra == 0 || resCtxHost.bwCoeffInter == 0) {
        // 任一不可信就一起清零，避免出现"一半可信"的中间状态
        resCtxHost.bwCoeffIntra = 0;
        resCtxHost.bwCoeffInter = 0;
    }
    HCCL_INFO("HcclScatter: bwCoeff intra[%u] inter[%u] (probe edges 0->%u / 0->%u)", resCtxHost.bwCoeffIntra,
        resCtxHost.bwCoeffInter, intraPeerOfRank0, interPeerOfRank0);
}

// 全连接：对通信域内其余每个 rank 各建 1 条 channel（共 rankSize-1 条），
// 恰好满足"同一对端最多申请 1 条 channel"的约束。
// 顺带把"哪些对端在 layer-0 上有直连"记成位掩码，device 侧据此还原 Server 归属。
HcclResult AcquireChannel(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
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
        uint32_t hitLayer = NET_LAYER_INTRA;
        CHK_RET(AcquireDesc(comm, param.myRank, j, &desc[idx], hitLayer));
        if (hitLayer == NET_LAYER_INTRA && j < 32) {
            resCtxHost.intraPeerMask |= (1U << j);
        }
        idx++;
    }
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, desc.data(), channelNum, channels.data()));
    for (uint32_t i = 0; i < channelNum; i++) {
        ChannelInfo channel;
        channel.remoteRank = desc[i].remoteRank;
        channel.handle = channels[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        // 该 channel 对应的对端 HCCL Buffer 地址，作为 HcommWriteOnThread 的写入目标
        void *cclBuf = nullptr;
        uint64_t cclBufSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channels[i], &cclBuf, &cclBufSize));
        channel.remoteCclMem = CommBuffer{cclBuf, cclBufSize};
        resCtxHost.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_scatter");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("HcclScatter: invalid root[%u], rankSize[%u]", root, param.rankSize), HCCL_E_PARA);
    // Scatter 语义：非 root 的 sendBuf 允许为空且不会被访问，仅 root 的 sendBuf 必须有效。
    // 因此 sendBuf 的校验必须放在拿到 myRank 之后，不能像模板那样在函数入口无条件校验。
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }

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
        HCCL_INFO("Engine context already exists");
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

        // 创建 AICPU_TS 通信引擎上的 thread 资源（见 custom.h 的 MAX_THREAD_NUM 说明）。
        // threads[0] = aicpuThread，承担框架 Host/Device 握手（thread notify 下标 0）；
        // threads[1..] 供 root 并行向 rankSize-1 个对端发送，一个对端一条流。
        // notifyNumPerThread 与 threadNum 同值：下标 0 给框架，1..threadNum-1 给末尾 join。
        uint32_t wantThreadNum = param.rankSize < MAX_THREAD_NUM ? param.rankSize : MAX_THREAD_NUM;
        if (wantThreadNum == 0) {
            wantThreadNum = 1;
        }
        // 申请失败时逐级折半降级。device 侧按 resCtx.threads.size() 自适应并行度，
        // 退到 1 条流时行为与 v1 单流串行完全一致，所以降级只损失性能、不影响功能。
        uint32_t threadNum = 0;
        HcclResult acquireRet = HCCL_E_INTERNAL;
        for (uint32_t n = wantThreadNum; n >= 1; n /= 2) {
            resCtxHost.threads.assign(n, 0);
            acquireRet = HcclThreadAcquire(comm, aicpuTsEngine, n, n, resCtxHost.threads.data());
            if (acquireRet == HCCL_SUCCESS) {
                threadNum = n;
                break;
            }
            HCCL_WARNING("HcclScatter: HcclThreadAcquire failed with threadNum[%u], ret[%d], degrading", n,
                static_cast<int32_t>(acquireRet));
            if (n == 1) {
                break;
            }
        }
        CHK_PRT_RET(threadNum == 0,
            HCCL_ERROR("HcclScatter: HcclThreadAcquire failed for all threadNum candidates, ret[%d]",
                static_cast<int32_t>(acquireRet)),
            acquireRet);
        resCtxHost.threads.resize(threadNum);
        HCCL_INFO("HcclScatter: acquired threadNum[%u] notifyNumPerThread[%u]", threadNum, threadNum);
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 申请 Channel 资源：对每个对端各 1 条，layer-0 优先、layer-1 回退
        CHK_RET(AcquireChannel(comm, param, resCtxHost));

        // 探测带宽系数，供 device 侧自校准中继比例。失败不算错误，只是退回纯直发。
        ProbeBwCoeffs(comm, param, resCtxHost);

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
