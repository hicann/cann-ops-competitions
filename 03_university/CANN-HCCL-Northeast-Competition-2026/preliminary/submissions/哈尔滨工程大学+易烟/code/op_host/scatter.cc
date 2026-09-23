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
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
int ReadProtocolPriority(CommProtocol protocol)
{
    // 950 的 ReadOnThread 支持 UB_CTP / UBOE。UBC_CTP 是兼容旧 SDK 的同值名称。
    // ChannelAcquire 支持的协议范围更大，不能据此选择不能执行 Read 的 Channel。
    if (protocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
        return 0;
    }
    if (protocol == CommProtocol::COMM_PROTOCOL_UBOE) {
        return 1;
    }
    return 2;
}

int CompareAddress(const CommAddr &left, const CommAddr &right)
{
    if (left.type != right.type) {
        return left.type < right.type ? -1 : 1;
    }
    // 只比较地址类型实际使用的字段，不能比较 union 未使用字节或结构体 padding。
    switch (left.type) {
        case CommAddrType::COMM_ADDR_TYPE_IP_V4:
            return std::memcmp(&left.addr, &right.addr, sizeof(left.addr));
        case CommAddrType::COMM_ADDR_TYPE_IP_V6:
            return std::memcmp(&left.addr6, &right.addr6, sizeof(left.addr6));
        case CommAddrType::COMM_ADDR_TYPE_ID:
            return left.id == right.id ? 0 : (left.id < right.id ? -1 : 1);
        case CommAddrType::COMM_ADDR_TYPE_EID:
            return std::memcmp(left.eid, right.eid, sizeof(left.eid));
        default:
            return 0;
    }
}

bool LinkComesFirst(const CommLink &left, const CommLink &right)
{
    const int leftPriority = ReadProtocolPriority(left.linkAttr.linkProtocol);
    const int rightPriority = ReadProtocolPriority(right.linkAttr.linkProtocol);
    if (leftPriority != rightPriority) {
        return leftPriority < rightPriority;
    }
    const int sourceOrder = CompareAddress(left.srcEndpointDesc.commAddr, right.srcEndpointDesc.commAddr);
    if (sourceOrder != 0) {
        return sourceOrder < 0;
    }
    return CompareAddress(left.dstEndpointDesc.commAddr, right.dstEndpointDesc.commAddr) < 0;
}

HcclResult SelectChannel(HcclComm comm, uint32_t myRank, uint32_t peer, const std::vector<uint32_t> &layers,
    HcclChannelDesc &desc)
{
    // 两端都查询同一个方向，使多链路情况下的端点选择保持一致。
    const uint32_t sourceRank = std::min(myRank, peer);
    const uint32_t destinationRank = std::max(myRank, peer);
    for (uint32_t layer : layers) {
        CommLink *links = nullptr;
        uint32_t linkCount = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layer, sourceRank, destinationRank, &links, &linkCount));
        if (linkCount == 0) {
            continue;
        }
        CHK_PTR_NULL(links);
        bool found = false;
        CommLink selected{};
        for (uint32_t index = 0; index < linkCount; ++index) {
            if (ReadProtocolPriority(links[index].linkAttr.linkProtocol) >= 2) {
                continue;
            }
            if (!found || LinkComesFirst(links[index], selected)) {
                selected = links[index];
                found = true;
            }
        }
        if (!found) {
            continue;
        }

        // 按真实网络层次优先选择 Mesh，跨 Server 才进入 Clos，不使用 rank/8 推测归属。
        // 必须复制端点：下一次 GetLinks 查询可能使 links 指向的库内存失效。
        desc.remoteRank = peer;
        desc.channelProtocol = selected.linkAttr.linkProtocol;
        desc.localEndpoint = myRank == sourceRank ? selected.srcEndpointDesc : selected.dstEndpointDesc;
        desc.remoteEndpoint = myRank == sourceRank ? selected.dstEndpointDesc : selected.srcEndpointDesc;
        desc.notifyNum = ops_hccl::SCATTER_CHANNEL_NOTIFY_NUM;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("No AICPU_TS Read channel between ranks %u and %u", myRank, peer);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    // 按 rank 索引缓存，不依赖本次 root 或消息大小；每个 peer 只建一个 Channel。
    resCtx.channels.resize(param.rankSize);
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    uint32_t *layerList = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerCount));
    CHK_PRT_RET(layerCount == 0, HCCL_ERROR("Scatter topology has no network layer"), HCCL_E_NOT_SUPPORT);
    CHK_PTR_NULL(layerList);
    std::vector<uint32_t> layers(layerList, layerList + layerCount);
    std::sort(layers.begin(), layers.end());
    layers.erase(std::unique(layers.begin(), layers.end()), layers.end());

    const uint32_t channelCount = param.rankSize - 1;
    std::vector<HcclChannelDesc> descriptors(channelCount);
    std::vector<ChannelHandle> handles(channelCount);
    CHK_RET(HcclChannelDescInit(descriptors.data(), channelCount));
    uint32_t index = 0;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        CHK_RET(SelectChannel(comm, param.myRank, peer, layers, descriptors[index]));
        ++index;
    }
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, descriptors.data(), channelCount,
        handles.data()));

    for (uint32_t channelIndex = 0; channelIndex < channelCount; ++channelIndex) {
        const uint32_t peer = descriptors[channelIndex].remoteRank;
        ChannelInfo &channel = resCtx.channels[peer];
        channel.remoteRank = peer;
        channel.notifyNum = ops_hccl::SCATTER_CHANNEL_NOTIFY_NUM;
        channel.handle = handles[channelIndex];
        CHK_RET(HcclChannelGetHcclBuffer(comm, channel.handle, &channel.remoteCclMem.addr,
            &channel.remoteCclMem.size));
        CHK_PTR_NULL(channel.remoteCclMem.addr);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    // V18 保留五个通知和原资源布局，使用独立 tag 隔离版本上下文。
    std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_scatter_suffix_v18");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize || root >= param.rankSize,
        HCCL_ERROR("Invalid Scatter rank information"), HCCL_E_PARA);
    const auto type = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(type == SIZE_TABLE.end(), HCCL_ERROR("Scatter supports float32 only"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / type->second / param.rankSize,
        HCCL_ERROR("Scatter input size overflows uint64_t"), HCCL_E_PARA);
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(recvBuf);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;
    // CPU_TS 仅保留模板的 Host/Device 启停同步；全部数据搬运由 AICPU_TS 编排。
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        CHK_PTR_NULL(ctx);
        CHK_PRT_RET(size == 0, HCCL_ERROR("Empty Scatter device context"), HCCL_E_INTERNAL);
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PTR_NULL(hostCtx);
        CHK_PRT_RET(hostCtxSize != sizeof(ThreadHandle),
            HCCL_ERROR("Invalid Scatter host context size"), HCCL_E_INTERNAL);
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;
        CHK_RET(HcclGetHcclBuffer(comm, &resCtxHost.localBuffer.addr, &resCtxHost.localBuffer.size));
        CHK_PTR_NULL(resCtxHost.localBuffer.addr);
        CHK_PRT_RET(resCtxHost.localBuffer.size < sizeof(float),
            HCCL_ERROR("Scatter HCCL Buffer is too small"), HCCL_E_MEMORY);

        resCtxHost.threads.resize(param.rankSize);
        // 主线程 notify[0] 用于 Host 同步，notify[1..N-1] 分别接收各 worker 的完成通知。
        // 一次申请全部线程：多次 Acquire 可能复用线程池前缀，不能假定每次返回新句柄。
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, param.rankSize, param.rankSize, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine,
            &param.aicpuThreadOnCpu));
        CHK_RET(AcquireChannels(comm, param, resCtxHost));

        std::vector<char> sequence = resCtxHost.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, sequence.data(), param.ctxSize, 0));

        void *hostCtx = nullptr;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, sizeof(ThreadHandle), &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, &resCtxHost.aicpuThread, sizeof(ThreadHandle), 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
