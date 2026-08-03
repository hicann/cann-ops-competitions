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
#include <limits>
#include <memory>
#include <vector>

#include <ccu_launch.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include <hccl_ccu_res.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
// Notify 0 is reserved for stream entry/exit synchronization.  The remaining
// notifies connect cross-die pipeline stages to their local relay stage.
constexpr uint32_t BASE_THREAD_NOTIFY_NUM = 1;
constexpr uint32_t RELAY_THREAD_NOTIFY_NUM = 5;
constexpr size_t MAX_DIE_GROUP_COUNT = 2;

struct ChannelGroup {
    EndpointAttrDieId dieId = 0;
    std::vector<uint32_t> remoteRanks;
    std::vector<ChannelHandle> channels;
};

HcclResult BuildChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    CHK_PRT_RET(
        layers == nullptr || layerCount == 0, HCCL_ERROR("Rank graph contains no network layers"), HCCL_E_INTERNAL);

    CommLink selected{};
    bool found = false;
    for (uint32_t layerIndex = 0; layerIndex < layerCount && !found; ++layerIndex) {
        CommLink *links = nullptr;
        uint32_t linkCount = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layers[layerIndex], srcRank, dstRank, &links, &linkCount));
        for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
            if (links[linkIndex].linkAttr.linkProtocol == COMM_PROTOCOL_UBC_CTP) {
                selected = links[linkIndex];
                found = true;
                break;
            }
        }
    }
    CHK_PRT_RET(!found, HCCL_ERROR("No UBC_CTP link from rank[%u] to rank[%u]", srcRank, dstRank), HCCL_E_NOT_FOUND);

    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = dstRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = selected.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = selected.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = selected.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = selected.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = selected.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = selected.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = selected.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult AcquireChannelGroups(HcclComm comm, const OpParam &param, std::vector<ChannelGroup> &groups)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    const uint32_t peerCount = param.rankSize - 1;
    std::vector<HcclChannelDesc> descriptions(peerCount);
    std::vector<EndpointAttrDieId> dieIds(peerCount);
    std::vector<ChannelHandle> channels(peerCount);

    uint32_t channelIndex = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CHK_RET(BuildChannelDesc(comm, param.myRank, remoteRank, descriptions[channelIndex]));
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &descriptions[channelIndex].localEndpoint,
            ENDPOINT_ATTR_DIE_ID, sizeof(EndpointAttrDieId), &dieIds[channelIndex]));
        ++channelIndex;
    }

    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, descriptions.data(), peerCount, channels.data()));

    for (uint32_t i = 0; i < peerCount; ++i) {
        auto group = std::find_if(groups.begin(), groups.end(), [dieId = dieIds[i]](const ChannelGroup &candidate) {
            return candidate.dieId == dieId;
        });
        if (group == groups.end()) {
            groups.push_back(ChannelGroup{});
            groups.back().dieId = dieIds[i];
            group = groups.end() - 1;
        }
        group->remoteRanks.push_back(descriptions[i].remoteRank);
        group->channels.push_back(channels[i]);
    }
    CHK_PRT_RET(groups.empty() || groups.size() > MAX_DIE_GROUP_COUNT,
        HCCL_ERROR("Unsupported CCU die group count[%zu]", groups.size()), HCCL_E_NOT_SUPPORT);
    std::sort(groups.begin(), groups.end(), [](const ChannelGroup &left, const ChannelGroup &right) {
        return left.dieId < right.dieId;
    });
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(
    HcclComm comm, const OpParam &param, const std::vector<ChannelGroup> &groups, AlgResourceCtx &resource)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(groups.empty() || groups.size() > MAX_DIE_GROUP_COUNT,
        HCCL_ERROR("Invalid CCU die group count[%zu]", groups.size()), HCCL_E_INTERNAL);

    // GroupCopy 始终放最高 die (die1=4×带宽) 避免 8+4 不对称性
    size_t localCopyGroup = groups.size() - 1;

    constexpr uint32_t RANKS_PER_SERVER = 8;
    uint32_t relayPeerRank = INVALID_VALUE_RANKID;
    size_t relayGroup = groups.size();
    bool relaySource = false;
    bool forwardRelayData = false;
    // 半中继：2×8 全覆盖，8+4 rank 0-3↔8-11 走 relay，4-7 直连
    const bool relayEligible = (param.rankSize == 16 || param.rankSize == 12) && groups.size() == MAX_DIE_GROUP_COUNT;
    if (relayEligible) {
        const uint32_t localServer = param.myRank / RANKS_PER_SERVER;
        const uint32_t serverRankCount = (localServer == 0)
            ? std::min(RANKS_PER_SERVER, param.rankSize)
            : param.rankSize - RANKS_PER_SERVER;
        const uint32_t localPeerCount = serverRankCount - 1;
        for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            const ChannelGroup &group = groups[groupIndex];
            const bool containsOnlyLocalPeers = group.remoteRanks.size() == localPeerCount
                && std::all_of(group.remoteRanks.begin(), group.remoteRanks.end(), [localServer](uint32_t rank) {
                       return rank / RANKS_PER_SERVER == localServer;
                   });
            if (containsOnlyLocalPeers) {
                relayGroup = groupIndex;
                break;
            }
        }

        if (relayGroup != groups.size()) {
            if (param.myRank < RANKS_PER_SERVER && param.myRank < param.rankSize - RANKS_PER_SERVER) {
                relayPeerRank = param.myRank + RANKS_PER_SERVER;
            } else if (param.myRank >= RANKS_PER_SERVER) {
                relayPeerRank = param.myRank - RANKS_PER_SERVER;
            }
            relaySource = (relayPeerRank != INVALID_VALUE_RANKID);
            forwardRelayData = relaySource;

            const bool hasRelayPeer = relayPeerRank == INVALID_VALUE_RANKID
                || std::any_of(groups.begin(), groups.end(), [relayPeerRank](const ChannelGroup &group) {
                       return std::find(group.remoteRanks.begin(), group.remoteRanks.end(), relayPeerRank)
                           != group.remoteRanks.end();
                   });
            if (hasRelayPeer) {
                resource.relayKernelIndex = static_cast<uint32_t>(relayGroup);
            } else {
                relayGroup = groups.size();
                relayPeerRank = INVALID_VALUE_RANKID;
                relaySource = false;
                forwardRelayData = false;
            }
        }
    }

    const bool relayEnabled = relayGroup != groups.size();
    std::vector<CcuKernelInfo> kernelInfos(groups.size());
    std::vector<CcuKernelInfo> continuationKernelInfos(relayEnabled ? 0 : groups.size());
    std::vector<CcuKernelInfo> relayKernelInfos(relayEnabled ? groups.size() : 0);
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        CcuKernelInfo &kernelInfo = kernelInfos[groupIndex];
        const int nameLength = std::snprintf(
            kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s", "CcuAllGatherDirectKernel");
        CHK_PRT_RET(nameLength <= 0 || static_cast<size_t>(nameLength) >= sizeof(kernelInfo.kernelFuncName),
            HCCL_ERROR("Failed to set CCU kernel name"), HCCL_E_INTERNAL);
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);

        const ChannelGroup &group = groups[groupIndex];
        CHK_PRT_RET(group.channels.empty() || group.channels.size() != group.remoteRanks.size()
                || group.channels.size() > MAX_RANK_SIZE,
            HCCL_ERROR("Invalid channel group[%zu] size[%zu]", groupIndex, group.channels.size()), HCCL_E_INTERNAL);

        auto kernelArg = std::make_shared<AllGatherKernelArg>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        kernelArg->handleLocalCopy = groupIndex == localCopyGroup;
        kernelArg->relaySource = relaySource;
        kernelArg->fullLocalTransfer = relaySource && groupIndex == relayGroup;
        kernelArg->forwardRelayData = forwardRelayData && groupIndex == relayGroup;
        kernelArg->relayPeerRank = relayPeerRank;
        for (size_t i = 0; i < group.channels.size(); ++i) {
            kernelArg->channels[i] = group.channels[i];
            kernelArg->remoteRanks[i] = group.remoteRanks[i];
            if (group.remoteRanks[i] == relayPeerRank) {
                kernelArg->relayPeerChannelIndex = static_cast<uint32_t>(i);
            }
        }
        kernelInfo.setKernelArg(kernelArg);

        if (!relayEnabled) {
            CcuKernelInfo &continuationKernelInfo = continuationKernelInfos[groupIndex];
            const int continuationNameLength = std::snprintf(continuationKernelInfo.kernelFuncName,
                sizeof(continuationKernelInfo.kernelFuncName), "%s", "CcuAllGatherContinuationKernel");
            CHK_PRT_RET(continuationNameLength <= 0
                    || static_cast<size_t>(continuationNameLength) >= sizeof(continuationKernelInfo.kernelFuncName),
                HCCL_ERROR("Failed to set CCU continuation kernel name"), HCCL_E_INTERNAL);
            continuationKernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuContinuationKernel);
            continuationKernelInfo.setKernelArg(kernelArg);
        }

        if (relayEnabled) {
            CcuKernelInfo &relayKernelInfo = relayKernelInfos[groupIndex];
            const int relayNameLength = std::snprintf(relayKernelInfo.kernelFuncName,
                sizeof(relayKernelInfo.kernelFuncName), "%s", "CcuAllGatherRelayKernel");
            CHK_PRT_RET(relayNameLength <= 0
                    || static_cast<size_t>(relayNameLength) >= sizeof(relayKernelInfo.kernelFuncName),
                HCCL_ERROR("Failed to set CCU relay kernel name"), HCCL_E_INTERNAL);
            relayKernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuRelayKernel);
            relayKernelInfo.setKernelArg(kernelArg);
        }
    }

    CcuInsHandle instructionHandle{0};
    uint32_t instructionCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instructionHandle, &instructionCount));
    CHK_PRT_RET(instructionCount != 1, HCCL_ERROR("Expected one CCU instruction instance, got[%u]", instructionCount),
        HCCL_E_INTERNAL);

    CcuResult ccuResult = HcommCcuKernelRegisterStart(instructionHandle);
    CHK_PRT_RET(ccuResult != CCU_SUCCESS, HCCL_ERROR("CCU kernel register start failed[%d]", ccuResult),
        ConvertCcuToHccl(ccuResult));

    resource.ccuKernels.resize(kernelInfos.size());
    constexpr uint32_t DIE_ID = 0;
    for (size_t i = 0; i < kernelInfos.size(); ++i) {
        const void *kernelArgs[] = {kernelInfos[i].kernelArg};
        ccuResult = HcommCcuKernelRegister(instructionHandle, DIE_ID, kernelInfos[i].kernelFuncName,
            kernelInfos[i].kernelFunc, kernelArgs, 1, &resource.ccuKernels[i]);
        CHK_PRT_RET(ccuResult != CCU_SUCCESS,
            HCCL_ERROR("CCU kernel[%zu] register failed[%d]", i, ccuResult), ConvertCcuToHccl(ccuResult));
    }

    resource.continuationKernels.resize(continuationKernelInfos.size());
    for (size_t i = 0; i < continuationKernelInfos.size(); ++i) {
        const void *kernelArgs[] = {continuationKernelInfos[i].kernelArg};
        ccuResult = HcommCcuKernelRegister(instructionHandle, DIE_ID,
            continuationKernelInfos[i].kernelFuncName, continuationKernelInfos[i].kernelFunc, kernelArgs, 1,
            &resource.continuationKernels[i]);
        CHK_PRT_RET(ccuResult != CCU_SUCCESS,
            HCCL_ERROR("CCU continuation kernel[%zu] register failed[%d]", i, ccuResult),
            ConvertCcuToHccl(ccuResult));
    }

    resource.relayKernels.resize(relayKernelInfos.size());
    for (size_t i = 0; i < relayKernelInfos.size(); ++i) {
        const void *kernelArgs[] = {relayKernelInfos[i].kernelArg};
        ccuResult = HcommCcuKernelRegister(instructionHandle, DIE_ID, relayKernelInfos[i].kernelFuncName,
            relayKernelInfos[i].kernelFunc, kernelArgs, 1, &resource.relayKernels[i]);
        CHK_PRT_RET(ccuResult != CCU_SUCCESS,
            HCCL_ERROR("CCU relay kernel[%zu] register failed[%d]", i, ccuResult), ConvertCcuToHccl(ccuResult));
    }

    ccuResult = HcommCcuKernelRegisterEnd(instructionHandle);
    CHK_PRT_RET(ccuResult != CCU_SUCCESS, HCCL_ERROR("CCU kernel register end failed[%d]", ccuResult),
        ConvertCcuToHccl(ccuResult));
    return HCCL_SUCCESS;
}

HcclResult ValidateParameters(const OpParam &param)
{
    const auto typeSize = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeSize == SIZE_TABLE.end() || typeSize->second == 0,
        HCCL_ERROR("Unsupported AllGather data type[%d]", static_cast<int>(param.dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank[%u] rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize->second,
        HCCL_ERROR("AllGather input size overflow"), HCCL_E_PARA);
    const uint64_t inputBytes = param.count * typeSize->second;
    CHK_PRT_RET(inputBytes != 0 && param.rankSize > std::numeric_limits<uint64_t>::max() / inputBytes,
        HCCL_ERROR("AllGather output size overflow"), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    const int tagLength = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allgather");
    CHK_PRT_RET(tagLength <= 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("Failed to set operator tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_RET(ValidateParameters(param));

    constexpr CommEngine CCU_ENGINE = CommEngine::COMM_ENGINE_CCU;
    const uint32_t threadNotifyNum
        = (param.rankSize == 16 || param.rankSize == 12) ? RELAY_THREAD_NOTIFY_NUM : BASE_THREAD_NOTIFY_NUM;
    CHK_RET(HcclThreadAcquireWithStream(comm, CCU_ENGINE, stream, threadNotifyNum, &param.cpuThread));

    void *context = nullptr;
    uint64_t contextSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, CCU_ENGINE, &context, &contextSize) == HCCL_SUCCESS) {
        CHK_PRT_RET(
            context == nullptr || contextSize == 0, HCCL_ERROR("Invalid cached CCU engine context"), HCCL_E_INTERNAL);
        param.resCtx = context;
        param.ctxSize = contextSize;
    } else {
        AlgResourceCtx resource{};
        void *cclBuffer = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
        resource.localBuffer = CommBuffer{cclBuffer, cclBufferSize};
        resource.ccuThread = param.cpuThread;

        std::vector<ChannelGroup> channelGroups;
        CHK_RET(AcquireChannelGroups(comm, param, channelGroups));
        resource.threads.resize(std::max<size_t>(1, channelGroups.size()));
        resource.threads[0] = param.cpuThread;
        if (resource.threads.size() > 1) {
            CHK_RET(HcclThreadAcquire(comm, CCU_ENGINE, static_cast<uint32_t>(resource.threads.size() - 1),
                threadNotifyNum, resource.threads.data() + 1));
        }
        CHK_RET(RegisterKernels(comm, param, channelGroups, resource));

        std::vector<char> serialized = resource.Serialize();
        param.ctxSize = serialized.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CCU_ENGINE, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, CCU_ENGINE, param.tag, serialized.data(), serialized.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
