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
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t THREAD_NOTIFY_NUM = 1;
constexpr uint32_t DIE_NUM = 2;
constexpr uint32_t CCU_DIE_ID = 0;
constexpr uint32_t COMBINED_TWO_SHOT_RANK_SIZE = 4;

struct ChannelGroup {
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peerRanks;
};

HcclResult FillChannelDesc(HcclComm comm, const std::vector<uint32_t> &netLayers, uint32_t localRank,
    uint32_t remoteRank, HcclChannelDesc &desc)
{
    for (uint32_t netLayer : netLayers) {
        CommLink *linkList = nullptr;
        uint32_t linkCount = 0;
        if (HcclRankGraphGetLinks(comm, netLayer, localRank, remoteRank, &linkList, &linkCount) != HCCL_SUCCESS) {
            continue;
        }
        for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
            const CommLink &link = linkList[linkIndex];
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
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("No CCU channel from rank[%u] to rank[%u]", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannelGroups(HcclComm comm, const OpParam &param, std::array<ChannelGroup, DIE_NUM> &groups)
{
    uint32_t *netLayerData = nullptr;
    uint32_t netLayerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayerData, &netLayerCount));
    CHK_PRT_RET(
        netLayerData == nullptr || netLayerCount == 0, HCCL_ERROR("Rank graph has no network layer"), HCCL_E_INTERNAL);
    const std::vector<uint32_t> netLayers(netLayerData, netLayerData + netLayerCount);

    const uint32_t channelCount = param.rankSize - 1;
    std::vector<HcclChannelDesc> channelDescs(channelCount);
    std::vector<uint32_t> channelDieIds(channelCount);
    CHK_RET(HcclChannelDescInit(channelDescs.data(), channelCount));
    uint32_t channelIndex = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        auto &desc = channelDescs[channelIndex];
        CHK_RET(FillChannelDesc(comm, netLayers, param.myRank, remoteRank, desc));

        EndpointAttrDieId dieId = 0;
        CHK_RET(HcclRankGraphGetEndpointInfo(
            comm, param.myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("Invalid die[%u] for channel from rank[%u] to rank[%u]", dieId, param.myRank, remoteRank),
            HCCL_E_INTERNAL);
        channelDieIds[channelIndex] = dieId;
        ++channelIndex;
    }

    std::vector<ChannelHandle> channels(channelCount);
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, channelDescs.data(), channelCount, channels.data()));
    for (uint32_t index = 0; index < channelCount; ++index) {
        const uint32_t dieId = channelDieIds[index];
        groups[dieId].channels.push_back(channels[index]);
        groups[dieId].peerRanks.push_back(channelDescs[index].remoteRank);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterKernelRound(CcuInsHandle insHandle,
    const std::vector<std::shared_ptr<ops_hccl::CcuKernelArgBroadcast>> &kernelArgs, const char *kernelName,
    const void *kernelFunction, std::vector<CcuKernelHandle> &kernelHandles)
{
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    kernelHandles.reserve(kernelArgs.size());
    for (const auto &kernelArg : kernelArgs) {
        CcuKernelHandle kernelHandle{0};
        const void *rawKernelArgs[] = {static_cast<const void *>(kernelArg.get())};
        const CcuResult result = HcommCcuKernelRegister(
            insHandle, CCU_DIE_ID, kernelName, kernelFunction, rawKernelArgs, 1, &kernelHandle);
        if (result != CCU_SUCCESS) {
            (void)HcommCcuKernelRegisterEnd(insHandle);
            HCCL_ERROR("Register CCU kernel[%s] failed, result[%d]", kernelName, result);
            return ConvertCcuToHccl(result);
        }
        kernelHandles.push_back(kernelHandle);
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(
    HcclComm comm, const OpParam &param, const std::array<ChannelGroup, DIE_NUM> &groups, AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle{0};
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    CHK_PRT_RET(insCount != 1, HCCL_ERROR("Expected one CCU instance, actual[%u]", insCount), HCCL_E_INTERNAL);

    std::vector<std::shared_ptr<ops_hccl::CcuKernelArgBroadcast>> kernelArgs;
    for (uint32_t dieId = 0; dieId < DIE_NUM; ++dieId) {
        const auto &group = groups[dieId];
        if (group.channels.empty()) {
            continue;
        }

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgBroadcast>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->root = param.root;
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
            kernelArg->channels[index] = group.channels[index];
            kernelArg->peerRanks[index] = group.peerRanks[index];
        }
        kernelArgs.push_back(kernelArg);
    }
    if (kernelArgs.empty()) {
        HCCL_ERROR("No CCU broadcast kernel was registered");
        return HCCL_E_INTERNAL;
    }

    CHK_RET(RegisterKernelRound(insHandle, kernelArgs, "CcuBroadcastDirectKernel",
        reinterpret_cast<const void *>(ops_hccl::CcuBroadcastDirectKernel), resCtx.directKernels));
    if (param.rankSize == COMBINED_TWO_SHOT_RANK_SIZE && kernelArgs.size() == 1) {
        CHK_RET(RegisterKernelRound(insHandle, kernelArgs, "CcuBroadcastTwoShotKernel",
            reinterpret_cast<const void *>(ops_hccl::CcuBroadcastTwoShotKernel), resCtx.twoShotKernels));
        return HCCL_SUCCESS;
    }

    CHK_RET(RegisterKernelRound(insHandle, kernelArgs, "CcuBroadcastScatterKernel",
        reinterpret_cast<const void *>(ops_hccl::CcuBroadcastScatterKernel), resCtx.scatterKernels));
    CHK_RET(RegisterKernelRound(insHandle, kernelArgs, "CcuBroadcastAllGatherKernel",
        reinterpret_cast<const void *>(ops_hccl::CcuBroadcastAllGatherKernel), resCtx.allGatherKernels));
    return HCCL_SUCCESS;
}

HcclResult AcquireThreads(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    resCtx.threads.push_back(param.cpuThread);
    if (resCtx.directKernels.size() == 1) {
        return HCCL_SUCCESS;
    }

    const uint32_t slaveThreadNum = static_cast<uint32_t>(resCtx.directKernels.size() - 1);
    std::vector<ThreadHandle> slaveThreads(slaveThreadNum);
    CHK_RET(
        HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, slaveThreadNum, THREAD_NOTIFY_NUM, slaveThreads.data()));
    resCtx.threads.insert(resCtx.threads.end(), slaveThreads.begin(), slaveThreads.end());
    return HCCL_SUCCESS;
}

HcclResult CreateResource(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    std::array<ChannelGroup, DIE_NUM> groups;
    CHK_RET(AcquireChannelGroups(comm, param, groups));
    CHK_RET(RegisterKernels(comm, param, groups, resCtx));
    CHK_RET(AcquireThreads(comm, param, resCtx));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    const auto sizeIt = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", dataType), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize
                    || root >= param.rankSize,
        HCCL_ERROR("Invalid rank info: rank[%u], root[%u], rankSize[%u]", param.myRank, root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeIt->second,
        HCCL_ERROR("Broadcast data size overflow"), HCCL_E_PARA);

    const int tagLength = std::snprintf(param.tag, sizeof(param.tag), "hccl_custom_broadcast_%u", root);
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("Failed to build broadcast tag"), HCCL_E_INTERNAL);

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    if (count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, THREAD_NOTIFY_NUM, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx resCtxHost;
        CHK_RET(CreateResource(comm, param, resCtxHost));
        const std::vector<char> sequence = resCtxHost.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, sequence.data(), sequence.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
