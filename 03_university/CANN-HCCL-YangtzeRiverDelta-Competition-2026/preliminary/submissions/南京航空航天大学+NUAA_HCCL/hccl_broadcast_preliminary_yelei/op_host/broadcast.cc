/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
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
HcclResult BuildChannelDesc(
    HcclComm comm, uint32_t myRank, uint32_t remoteRank, const std::vector<uint32_t> &netLayers, HcclChannelDesc &desc)
{
    constexpr CommProtocol PROTOCOL_PRIORITY[] = {COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UBC_TP, COMM_PROTOCOL_UBOE};

    for (CommProtocol protocol : PROTOCOL_PRIORITY) {
        for (uint32_t layer : netLayers) {
            CommLink *links = nullptr;
            uint32_t linkNum = 0;
            HcclResult ret = HcclRankGraphGetLinks(comm, layer, myRank, remoteRank, &links, &linkNum);
            if (ret != HCCL_SUCCESS || links == nullptr) {
                continue;
            }

            for (uint32_t index = 0; index < linkNum; ++index) {
                const CommLink &link = links[index];
                if (link.linkAttr.linkProtocol != protocol) {
                    continue;
                }

                CHK_RET(HcclChannelDescInit(&desc, 1));
                desc.remoteRank = remoteRank;
                desc.notifyNum = CUSTOM_CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = protocol;
                desc.localEndpoint = link.srcEndpointDesc;
                desc.remoteEndpoint = link.dstEndpointDesc;
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[BuildChannelDesc] No AICPU protocol from rank[%u] to rank[%u]", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    (void)std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_broadcast_v7");
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.root >= param.rankSize,
        HCCL_ERROR("[HcclBroadcast] Invalid root[%u], rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA);

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtxSize != sizeof(ThreadHandle),
            HCCL_ERROR(
                "[HcclBroadcast] Invalid cached host context size[%llu]", static_cast<unsigned long long>(hostCtxSize)),
            HCCL_E_INTERNAL);
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        CHK_PRT_RET(cclBufferAddr == nullptr || cclBufferSize == 0,
            HCCL_ERROR("[HcclBroadcast] Invalid local HCCL buffer"), HCCL_E_MEMORY);
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        resCtxHost.threads.resize(CUSTOM_ALGORITHM_THREAD_NUM);
        CHK_RET(HcclThreadAcquire(
            comm, aicpuTsEngine, CUSTOM_ALGORITHM_THREAD_NUM, CUSTOM_THREAD_NOTIFY_NUM, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        if (param.rankSize > 1) {
            uint32_t *netLayerData = nullptr;
            uint32_t netLayerNum = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &netLayerData, &netLayerNum));
            CHK_PRT_RET(netLayerData == nullptr || netLayerNum == 0, HCCL_ERROR("[HcclBroadcast] Empty rank graph"),
                HCCL_E_NOT_FOUND);
            std::vector<uint32_t> netLayers(netLayerData, netLayerData + netLayerNum);

            std::vector<HcclChannelDesc> channelDescs;
            std::vector<uint32_t> remoteRanks;
            channelDescs.reserve(param.rankSize - 1);
            remoteRanks.reserve(param.rankSize - 1);
            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                if (remoteRank == param.myRank) {
                    continue;
                }
                HcclChannelDesc desc;
                CHK_RET(BuildChannelDesc(comm, param.myRank, remoteRank, netLayers, desc));
                channelDescs.push_back(desc);
                remoteRanks.push_back(remoteRank);
            }

            const uint32_t channelNum = static_cast<uint32_t>(channelDescs.size());
            std::vector<ChannelHandle> channelHandles(channelNum);
            CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, channelDescs.data(), channelNum, channelHandles.data()));

            resCtxHost.channels.resize(channelNum);
            for (uint32_t index = 0; index < channelNum; ++index) {
                ChannelInfo &channel = resCtxHost.channels[index];
                channel.remoteRank = remoteRanks[index];
                channel.notifyNum = CUSTOM_CHANNEL_NOTIFY_NUM;
                channel.handle = channelHandles[index];
                CHK_RET(HcclChannelGetHcclBuffer(
                    comm, channel.handle, &channel.remoteCclMem.addr, &channel.remoteCclMem.size));
                CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size == 0,
                    HCCL_ERROR("[HcclBroadcast] Invalid remote buffer for rank[%u]", channel.remoteRank),
                    HCCL_E_MEMORY);
            }
        }

        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
