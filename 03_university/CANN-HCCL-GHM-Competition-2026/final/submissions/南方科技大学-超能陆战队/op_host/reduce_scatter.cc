/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t MAX_DIE_NUM = 2;

struct ChannelGroup {
    uint32_t dieId;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> remoteRanks;
};

// 用以持久化持有所有注册算子的参数指针，防止函数退出时被析构
static std::vector<std::shared_ptr<void>> g_persistentKernelArgs;

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

HcclResult AcquirePeerChannels(
    HcclComm comm, CommEngine engine, uint32_t myRank, uint32_t rankSize, std::vector<ChannelGroup> &groups)
{
    uint32_t *networkLayers = nullptr;
    uint32_t networkLayerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &networkLayers, &networkLayerCount));
    if (networkLayers == nullptr || networkLayerCount == 0) {
        HCCL_ERROR("[ReduceScatter] No topology layer is available for rank %u", myRank);
        return HCCL_E_NOT_FOUND;
    }
    const std::vector<uint32_t> networkLayerIds(networkLayers, networkLayers + networkLayerCount);

    std::vector<HcclChannelDesc> channelDescs;
    std::vector<uint32_t> remoteRanks;
    std::vector<uint32_t> channelDieIds;
    channelDescs.reserve(rankSize - 1);
    remoteRanks.reserve(rankSize - 1);
    channelDieIds.reserve(rankSize - 1);

    for (uint32_t remoteRank = 0; remoteRank < rankSize; ++remoteRank) {
        if (remoteRank == myRank) {
            continue;
        }

        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        bool protocolFound = false;

        for (uint32_t layerIndex = 0; layerIndex < networkLayerCount && !protocolFound; ++layerIndex) {
            uint32_t linkCount = 0;
            CommLink *links = nullptr;
            HcclResult linkResult
                = HcclRankGraphGetLinks(comm, networkLayerIds[layerIndex], myRank, remoteRank, &links, &linkCount);
            if (linkResult != HCCL_SUCCESS || links == nullptr) {
                continue;
            }

            for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
                const CommLink &link = links[linkIndex];
                if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }

                desc.remoteRank = remoteRank;
                desc.notifyNum = CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                protocolFound = true;
                break;
            }
        }

        if (!protocolFound) {
            HCCL_ERROR("[ReduceScatter] UBC_CTP link not found between rank %u and rank %u", myRank, remoteRank);
            return HCCL_E_NOT_FOUND;
        }

        EndpointAttrDieId dieId = 0;
        uint32_t infoLength = sizeof(dieId);
        CHK_RET(HcclRankGraphGetEndpointInfo(
            comm, myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, infoLength, &dieId));
        if (dieId >= MAX_DIE_NUM) {
            HCCL_ERROR("[ReduceScatter] Invalid die id %u for channel from rank %u to rank %u", dieId, myRank,
                remoteRank);
            return HCCL_E_INTERNAL;
        }

        channelDescs.push_back(desc);
        remoteRanks.push_back(remoteRank);
        channelDieIds.push_back(dieId);
    }

    std::vector<ChannelHandle> channels(channelDescs.size());
    CHK_RET(HcclChannelAcquire(
        comm, engine, channelDescs.data(), static_cast<uint32_t>(channelDescs.size()), channels.data()));

    std::array<ChannelGroup, MAX_DIE_NUM> groupsByDie;
    for (uint32_t dieId = 0; dieId < MAX_DIE_NUM; ++dieId) {
        groupsByDie[dieId].dieId = dieId;
    }
    for (uint32_t channelIndex = 0; channelIndex < channels.size(); ++channelIndex) {
        ChannelGroup &group = groupsByDie[channelDieIds[channelIndex]];
        group.channels.push_back(channels[channelIndex]);
        group.remoteRanks.push_back(remoteRanks[channelIndex]);
    }

    groups.clear();
    for (uint32_t dieId = 0; dieId < MAX_DIE_NUM; ++dieId) {
        if (groupsByDie[dieId].channels.empty()) {
            continue;
        }
        groups.push_back(std::move(groupsByDie[dieId]));
    }
    if (groups.size() == MAX_DIE_NUM && groups[0].channels.size() > groups[1].channels.size()) {
        // The primary group also reduces the local rank. Put it on the die
        // with fewer peer channels to balance the number of sources.
        std::swap(groups[0], groups[1]);
    }

    return HCCL_SUCCESS;
}

HcclResult RegisterCcuKernels(
    HcclComm comm, const OpParam &param, const std::vector<ChannelGroup> &groups, AlgResourceCtx &resource)
{
    CcuInsHandle instructionHandle{0};
    uint32_t instructionCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instructionHandle, &instructionCount));
    if (instructionCount != 1) {
        HCCL_ERROR("[ReduceScatter] Expected one CCU instruction instance, got %u", instructionCount);
        return HCCL_E_INTERNAL;
    }

    CcuResult ccuResult = HcommCcuKernelRegisterStart(instructionHandle);
    if (ccuResult != CCU_SUCCESS) {
        return ConvertCcuResult(ccuResult);
    }

    std::vector<std::shared_ptr<CcuReduceScatterKernelArg>> kernelArguments;
    kernelArguments.reserve(groups.size());
    const bool needsCombineKernel = groups.size() == MAX_DIE_NUM;
    resource.ccuKernels.resize(groups.size() + (needsCombineKernel ? groups.size() : 0));

    for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const ChannelGroup &group = groups[groupIndex];
        auto kernelArg = std::make_shared<CcuReduceScatterKernelArg>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->includeLocal = groupIndex == 0;
        kernelArg->scratchSlotBase
            = needsCombineKernel && groupIndex == 0 ? static_cast<uint32_t>(groups[1].channels.size()) : 0;
        kernelArg->dataType = param.dataType;
        kernelArg->reduceOp = param.reduceType;
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        for (uint32_t channelIndex = 0; channelIndex < group.channels.size(); ++channelIndex) {
            kernelArg->channels[channelIndex] = group.channels[channelIndex];
            kernelArg->remoteRanks[channelIndex] = group.remoteRanks[channelIndex];
        }

        const void *kernelArgs[] = {kernelArg.get()};
        constexpr uint32_t registerDieId = 0; // Reserved by the registration API; channels select the actual die.
        ccuResult = HcommCcuKernelRegister(instructionHandle, registerDieId, "CcuReduceScatterKernel",
            reinterpret_cast<void *>(ops_hccl::CcuReduceScatterKernel), kernelArgs, 1,
            &resource.ccuKernels[groupIndex]);
        if (ccuResult != CCU_SUCCESS) {
            return ConvertCcuResult(ccuResult);
        }
        g_persistentKernelArgs.push_back(kernelArg);
        kernelArguments.push_back(std::move(kernelArg));
    }

    if (needsCombineKernel) {
        for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            auto combineArg = std::make_shared<CcuCombineKernelArg>();
            combineArg->channelCount = 1;
            combineArg->channels[0] = groups[groupIndex].channels[0];
            const void *combineArgs[] = {combineArg.get()};
            ccuResult = HcommCcuKernelRegister(instructionHandle, 0, "CcuCombineKernel",
                reinterpret_cast<void *>(ops_hccl::CcuCombineKernel), combineArgs, 1,
                &resource.ccuKernels[groups.size() + groupIndex]);
            if (ccuResult != CCU_SUCCESS) {
                return ConvertCcuResult(ccuResult);
            }
            g_persistentKernelArgs.push_back(combineArg);
        }
    }

    ccuResult = HcommCcuKernelRegisterEnd(instructionHandle);
    if (ccuResult != CCU_SUCCESS) {
        return ConvertCcuResult(ccuResult);
    }

    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("[ReduceScatter] Only float32 sum is supported");
        return HCCL_E_NOT_SUPPORT;
    }

    // 构造算子参数
    OpParam param;
    int tagLength = std::snprintf(
        param.tag, sizeof(param.tag), "%s", "hccl_custom_reducescatter_ccu_buffered_reduce");
    if (tagLength <= 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag)) {
        return HCCL_E_INTERNAL;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    param.reduceType = op;

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
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // CCU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost{};
        resCtxHost.ccuThread = param.cpuThread;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        if (param.rankSize > 1) {
            std::vector<ChannelGroup> channelGroups;
            CHK_RET(AcquirePeerChannels(comm, ccuEngine, param.myRank, param.rankSize, channelGroups));
            if (channelGroups.empty() || channelGroups.size() > MAX_DIE_NUM) {
                return HCCL_E_INTERNAL;
            }

            resCtxHost.threads.resize(channelGroups.size());
            resCtxHost.threads[0] = param.cpuThread;
            if (channelGroups.size() > 1) {
                CHK_RET(HcclThreadAcquire(
                    comm, ccuEngine, static_cast<uint32_t>(channelGroups.size() - 1), 1, &resCtxHost.threads[1]));
            }
            CHK_RET(RegisterCcuKernels(comm, param, channelGroups, resCtxHost));
        } else {
            resCtxHost.threads = {param.cpuThread};
        }

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 CCU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
