/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
// AICPU+TS 通信协议（Ascend950 / A5）
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3; // ACK + DATA_SIGNAL + RELAY（RELAY 供中继载荷就绪通知）

// 算法可调参数（均可用环境变量覆盖，免改码实测调优）
constexpr uint64_t DEFAULT_COPY_THREADS = 4;             // 自留块本地拷贝并行线程数
constexpr uint64_t DEFAULT_THRESHOLD_BYTES = 1ULL << 20; // 大小消息路径切换阈值（每 rank 字节）

uint64_t GetEnvU64(const char *name, uint64_t defVal)
{
    const char *val = std::getenv(name);
    return (val == nullptr || val[0] == '\0') ? defVal : std::strtoull(val, nullptr, 0);
}

// 根据 rank 图，收集本端到远端 dstRank 的 channel 描述。
// 先定位首个含目标协议链路的层（机内 mesh 为 layer0，机间 Clos 为 layer1），再仅在该层内收集
// 最多 maxStripes 条链路建条带通道：CTP 支持多 Transport Channel 多路径（白皮书 4.6.1）。
// 层内收集是硬约束——跨层混采会把机内直连流量拆到 Clos 绕行路径上，反而大幅劣化。
HcclResult FillChannelDescs(
    HcclComm comm, uint32_t srcRank, uint32_t dstRank, uint32_t maxStripes, std::vector<HcclChannelDesc> &descs)
{
    // 获取全部网络层：机内 mesh 为 layer0，机间 Clos 为 layer1（跨机链路只存在于更高层）
    uint32_t *netLayersRaw = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayersRaw, &netLayerNum));

    const CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    // pass1：定位首个含目标协议链路的层（主链路层）
    int32_t primaryLayer = -1;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum && primaryLayer < 0; layerIdx++) {
        uint32_t listSize = 0;
        CommLink *linkList = nullptr;
        if (HcclRankGraphGetLinks(comm, netLayersRaw[layerIdx], srcRank, dstRank, &linkList, &listSize)
            != HCCL_SUCCESS) {
            continue;
        }
        for (uint32_t idx = 0; idx < listSize; idx++) {
            if (linkList[idx].linkAttr.linkProtocol == protocol) {
                primaryLayer = static_cast<int32_t>(netLayersRaw[layerIdx]);
                break;
            }
        }
    }
    if (primaryLayer < 0) {
        return HCCL_SUCCESS; // 无目标协议链路，由调用方统一报错
    }
    // pass2：仅在主链路层内收集条带链路（GetLinks 返回内存在下次调用后失效，逐字段即时拷贝）
    uint32_t listSize = 0;
    CommLink *linkList = nullptr;
    CHK_RET(HcclRankGraphGetLinks(comm, static_cast<uint32_t>(primaryLayer), srcRank, dstRank, &linkList, &listSize));
    uint32_t ctpLinkTotal = 0;
    for (uint32_t idx = 0; idx < listSize; idx++) {
        if (linkList[idx].linkAttr.linkProtocol == protocol) {
            ctpLinkTotal++;
        }
    }
    for (uint32_t idx = 0; idx < listSize && descs.size() < maxStripes; idx++) {
        CommLink link = linkList[idx];
        if (link.linkAttr.linkProtocol != protocol) {
            continue;
        }
        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = dstRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link.srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
        descs.push_back(desc);
    }
    // 测试资源紧张：把该对端可用链路总数落到日志，单次运行即可判断条带上限与中继价值
    HCCL_INFO("[FillChannelDescs] rank %u->%u primaryLayer %d ctpLinks %u collected %lu", srcRank, dstRank,
        primaryLayer, ctpLinkTotal, static_cast<unsigned long>(descs.size()));
    return HCCL_SUCCESS;
}

// 解析拓扑：layer-0（机内 mesh）直连域 → peerIsLocal 标记，供 device 侧"跨机对端优先发送"排序。
HcclResult BuildTopology(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    uint32_t *ranks = nullptr;
    uint32_t rankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &ranks, &rankNum));
    CHK_PRT_RET(rankNum == 0, HCCL_ERROR("[BuildTopology] empty layer-0 rank list"), HCCL_E_INTERNAL);

    resCtxHost.peerIsLocal.assign(param.rankSize, 0);
    bool myRankFound = false;
    for (uint32_t i = 0; i < rankNum; i++) {
        if (ranks[i] >= param.rankSize) {
            HCCL_ERROR("[BuildTopology] rank[%u] out of range, rankSize=%u", ranks[i], param.rankSize);
            return HCCL_E_INTERNAL;
        }
        resCtxHost.peerIsLocal[ranks[i]] = 1;
        myRankFound = myRankFound || (ranks[i] == param.myRank);
    }
    CHK_PRT_RET(!myRankFound, HCCL_ERROR("[BuildTopology] myRank[%u] not in layer-0 rank list", param.myRank),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    int tagLen = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_scatter");
    CHK_PRT_RET(tagLen < 0 || static_cast<size_t>(tagLen) >= sizeof(param.tag),
        HCCL_ERROR("Failed to build op tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
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

        // 算法参数（环境变量覆盖默认值，免改码实测调优）
        resCtxHost.smallMsgThresholdBytes = GetEnvU64("HCCL_SCATTER_THRESHOLD_BYTES", DEFAULT_THRESHOLD_BYTES);
        resCtxHost.relayEnabled = GetEnvU64("HCCL_SCATTER_RELAY", 1) != 0 ? 1 : 0; // 跨机中继分流（默认开，env 置 0 关闭）
        resCtxHost.nbiEnabled = GetEnvU64("HCCL_SCATTER_NBI", 0) != 0 ? 1 : 0;    // 小消息 NBI 直发（默认关走 PULL，env 置 1 切 NBI PUSH）

        // 合规约束（赛题 2.2）：针对一个对端仅能申请 1 个 channel——多通道条带被规则禁止，锁死为 1。
        // 跨机多路径带宽改由"中继分流"实现（数据面分流复用既有 channel，不新增 channel）。
        constexpr uint32_t stripePerPeer = 1;

        // threads[0]=主线程；[1..jobThreadNum]=写任务专属线程；[jobThreadNum+1..+copyThreadNum]=拷贝线程。
        // 同步写原语（HcommWrite*OnThread）会阻塞所在线程直至传输完成，写并发度=线程数：
        // 每个写任务（对端×条带）必须独占一个线程，否则任务在线程内串行，大消息下成倍变慢。
        // 自留块拷贝同样按线程并行（AICPU 单线程拷 32MB 需数百 μs，与网络写同量级，串行会成临界路径）。
        resCtxHost.jobThreadNum = param.rankSize > 1 ? (param.rankSize - 1) * stripePerPeer : 0;
        resCtxHost.copyThreadNum = param.rankSize > 1
            ? static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(1,
                  GetEnvU64("HCCL_SCATTER_COPY_THREADS", DEFAULT_COPY_THREADS)), 16))
            : 1;
        const uint32_t threadNum = param.rankSize > 1 ? resCtxHost.jobThreadNum + resCtxHost.copyThreadNum + 1 : 1;
        const uint32_t notifyNumPerThread = param.rankSize > 1 ? resCtxHost.jobThreadNum + resCtxHost.copyThreadNum
                                                               : 1;
        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 解析 server 分组（peerIsLocal），供 device 侧跨机优先排序/中继分流
        CHK_RET(BuildTopology(comm, param, resCtxHost));

        // 取本端 cclBuffer（中继节点的转发载荷读取源）
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // 注册 recvBuf 与 sendBuf：
        //   output（recvBuf）——root 单边直写对端 recvBuf，省 ccl buffer 中转 + 落地拷贝
        //   input（sendBuf）——小消息路径 peer 直接远端读 root sendBuf，省 root 暂存拷贝
        char outputMemTag[TAG_LENGTH] = {};
        char inputMemTag[TAG_LENGTH] = {};
        int outputTagLen = std::snprintf(outputMemTag, sizeof(outputMemTag), "%s_output", param.tag);
        int inputTagLen = std::snprintf(inputMemTag, sizeof(inputMemTag), "%s_input", param.tag);
        CHK_PRT_RET(outputTagLen < 0 || static_cast<size_t>(outputTagLen) >= sizeof(outputMemTag)
                || inputTagLen < 0 || static_cast<size_t>(inputTagLen) >= sizeof(inputMemTag),
            HCCL_ERROR("Failed to build memory tag"), HCCL_E_INTERNAL);

        const uint32_t dataTypeSize = SIZE_TABLE.at(param.dataType);
        const uint64_t recvBytes = param.count * dataTypeSize;
        CommMem outputMem{COMM_MEM_TYPE_DEVICE, recvBuf, recvBytes};
        CommMem inputMem{COMM_MEM_TYPE_DEVICE, sendBuf, recvBytes * param.rankSize};
        HcclMemHandle memHandles[2] = {nullptr, nullptr};
        CHK_RET(HcclCommMemReg(comm, outputMemTag, &outputMem, &memHandles[0]));
        CHK_RET(HcclCommMemReg(comm, inputMemTag, &inputMem, &memHandles[1]));
        CHK_PTR_NULL(memHandles[0]);
        CHK_PTR_NULL(memHandles[1]);

        // 每对端取最多 stripePerPeer 条 CTP 链路建条带通道（多路径摊 Port 带宽）。
        // 注意：条带上限按"每对端"计，先收集到独立临时 vector 再汇总，避免累积计数误判。
        std::vector<HcclChannelDesc> allDescs;
        std::vector<uint32_t> descRanks;
        std::vector<HcclChannelDesc> peerDescs;
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
            if (remoteRank == param.myRank) {
                continue;
            }
            peerDescs.clear();
            CHK_RET(FillChannelDescs(comm, param.myRank, remoteRank, stripePerPeer, peerDescs));
            CHK_PRT_RET(peerDescs.empty(),
                HCCL_ERROR("[HcclScatter] no CTP channel between rank %u and rank %u", param.myRank, remoteRank),
                HCCL_E_NOT_FOUND);
            for (auto &desc : peerDescs) {
                desc.memHandles = memHandles;
                desc.memHandleNum = 2; // 交换 recvBuf + sendBuf 地址
            }
            allDescs.insert(allDescs.end(), peerDescs.begin(), peerDescs.end());
            descRanks.insert(descRanks.end(), peerDescs.size(), remoteRank);
        }

        std::vector<ChannelHandle> channelHandles(allDescs.size());
        if (!allDescs.empty()) {
            CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, allDescs.data(), allDescs.size(), channelHandles.data()));
        }

        std::vector<uint32_t> stripeIdxOf(param.rankSize, 0);
        for (size_t i = 0; i < allDescs.size(); i++) {
            ChannelInfo channel;
            channel.remoteRank = descRanks[i];
            channel.stripeIdx = stripeIdxOf[channel.remoteRank]++;
            channel.handle = channelHandles[i];
            channel.notifyNum = CHANNEL_NOTIFY_NUM;
            // 取回对端注册的 recvBuf / sendBuf 地址（按 tag 匹配）
            uint32_t remoteMemNum = 0;
            CommMem *remoteMems = nullptr;
            char **remoteMemTags = nullptr;
            CHK_RET(HcclChannelGetRemoteMems(comm, channelHandles[i], &remoteMemNum, &remoteMems, &remoteMemTags));
            bool foundOutput = false;
            bool foundInput = false;
            for (uint32_t memIdx = 0; memIdx < remoteMemNum; memIdx++) {
                if (remoteMemTags[memIdx] != nullptr && std::strcmp(remoteMemTags[memIdx], outputMemTag) == 0) {
                    channel.remoteOutput = remoteMems[memIdx].addr;
                    foundOutput = true;
                } else if (remoteMemTags[memIdx] != nullptr && std::strcmp(remoteMemTags[memIdx], inputMemTag) == 0) {
                    channel.remoteInput = remoteMems[memIdx].addr;
                    foundInput = true;
                }
            }
            CHK_PRT_RET(!foundOutput || !foundInput || channel.remoteOutput == nullptr
                    || channel.remoteInput == nullptr,
                HCCL_ERROR("Remote memory not exchanged for rank[%u]", channel.remoteRank), HCCL_E_NOT_FOUND);
            // 取回对端 cclBuffer 地址（中继载荷落点）
            void *remoteCclBuf = nullptr;
            uint64_t remoteCclBufSize = 0;
            CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[i], &remoteCclBuf, &remoteCclBufSize));
            channel.remoteCclMem = CommBuffer{remoteCclBuf, remoteCclBufSize};
            resCtxHost.channels.push_back(channel);
        }

        // 测试资源紧张：参数与拓扑汇总落日志，单次运行带回尽量多的决策信息
        uint32_t layer0PeerNum = 0;
        for (uint32_t r = 0; r < param.rankSize; r++) {
            layer0PeerNum += (r != param.myRank) ? resCtxHost.peerIsLocal[r] : 0;
        }
        HCCL_INFO("[HcclScatter] rank %u rankSize %u relay %u nbi %u jobThreads %u copyThreads %u threshold %lu "
                  "layer0Peers %u channels %lu",
            param.myRank, param.rankSize, resCtxHost.relayEnabled, resCtxHost.nbiEnabled, resCtxHost.jobThreadNum,
            resCtxHost.copyThreadNum, static_cast<unsigned long>(resCtxHost.smallMsgThresholdBytes), layer0PeerNum,
            static_cast<unsigned long>(resCtxHost.channels.size()));

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
