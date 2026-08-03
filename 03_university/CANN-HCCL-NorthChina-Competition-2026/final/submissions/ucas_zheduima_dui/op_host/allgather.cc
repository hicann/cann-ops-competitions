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
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "custom.h"
#include "ccu_kernel.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t MAX_KERNEL_NUM = 2;
constexpr uint32_t KERNEL_ARG_NUM = 1;
constexpr uint32_t RESERVED_DIE_ID = 0;

struct ChannelPlan {
    HcclChannelDesc desc{};
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t netLayer = 0;
    EndpointAttrDieId dieId = 0;
};

struct ChannelGroup {
    uint32_t netLayer = 0;
    EndpointAttrDieId dieId = 0;
    std::vector<ChannelPlan> plans;
    std::vector<ChannelHandle> handles;
};

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_INTERNAL:
            return HCCL_E_INTERNAL;
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

HcclResult SelectLink(
    HcclComm comm, const OpParam &param, uint32_t remoteRank, const std::vector<uint32_t> &netLayers, ChannelPlan &plan)
{
    for (uint32_t netLayer : netLayers) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &links, &linkNum));

        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            const CommLink link = links[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            CHK_RET(HcclChannelDescInit(&plan.desc, 1));
            plan.desc.remoteRank = remoteRank;
            plan.desc.notifyNum = CHANNEL_NOTIFY_NUM;
            plan.desc.channelProtocol = link.linkAttr.linkProtocol;
            plan.desc.localEndpoint = link.srcEndpointDesc;
            plan.desc.remoteEndpoint = link.dstEndpointDesc;
            plan.remoteRank = remoteRank;
            plan.netLayer = netLayer;

            CHK_RET(HcclRankGraphGetEndpointInfo(
                comm, param.myRank, &plan.desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(plan.dieId), &plan.dieId));
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[AllGather] No UBC_CTP link from rank %u to rank %u", param.myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult BuildChannelGroups(HcclComm comm, const OpParam &param, std::vector<ChannelGroup> &groups)
{
    uint32_t *netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayerList, &netLayerNum));
    CHK_PRT_RET(netLayerList == nullptr || netLayerNum == 0, HCCL_ERROR("[AllGather] Rank graph has no network layer"),
        HCCL_E_INTERNAL);

    std::vector<uint32_t> netLayers(netLayerList, netLayerList + netLayerNum);
    std::sort(netLayers.begin(), netLayers.end());

    std::map<std::pair<uint32_t, EndpointAttrDieId>, std::vector<ChannelPlan>> groupedPlans;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        ChannelPlan plan;
        CHK_RET(SelectLink(comm, param, remoteRank, netLayers, plan));
        groupedPlans[{plan.netLayer, plan.dieId}].push_back(plan);
    }

    CHK_PRT_RET(groupedPlans.empty(),
        HCCL_ERROR("[AllGather] No channel group was generated for rank %u", param.myRank), HCCL_E_INTERNAL);
    CHK_PRT_RET(groupedPlans.size() > MAX_KERNEL_NUM,
        HCCL_ERROR("[AllGather] Need %zu CCU kernels, but final topology supports at most %u", groupedPlans.size(),
            MAX_KERNEL_NUM),
        HCCL_E_NOT_SUPPORT);

    groups.reserve(groupedPlans.size());
    for (auto &entry : groupedPlans) {
        ChannelGroup group;
        group.netLayer = entry.first.first;
        group.dieId = entry.first.second;
        group.plans = std::move(entry.second);

        std::vector<HcclChannelDesc> descs;
        descs.reserve(group.plans.size());
        for (const ChannelPlan &plan : group.plans) {
            descs.push_back(plan.desc);
        }

        group.handles.resize(descs.size());
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, descs.data(), static_cast<uint32_t>(descs.size()),
            group.handles.data()));
        HCCL_INFO("[AllGather] rank %u: layer %u, die %u, channel count %zu", param.myRank, group.netLayer, group.dieId,
            group.handles.size());
        groups.push_back(std::move(group));
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(
    HcclComm comm, const OpParam &param, const std::vector<ChannelGroup> &groups, AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("[AllGather] Expected one CCU instance, got %u", insNum), HCCL_E_INTERNAL);

    size_t copyKernelIdx = 0;
    for (size_t idx = 1; idx < groups.size(); ++idx) {
        if (groups[idx].handles.size() < groups[copyKernelIdx].handles.size()) {
            copyKernelIdx = idx;
        }
    }

    size_t localGroupIdx = groups.size();
    size_t remoteGroupIdx = groups.size();
    std::vector<uint32_t> localRanks{param.myRank};
    std::vector<uint32_t> remoteRanks;
    for (size_t idx = 0; idx < groups.size(); ++idx) {
        if (groups[idx].netLayer == 0) {
            localGroupIdx = idx;
            for (const ChannelPlan &plan : groups[idx].plans) {
                localRanks.push_back(plan.remoteRank);
            }
        } else if (groups[idx].netLayer == 1) {
            remoteGroupIdx = idx;
            for (const ChannelPlan &plan : groups[idx].plans) {
                remoteRanks.push_back(plan.remoteRank);
            }
        }
    }
    std::sort(localRanks.begin(), localRanks.end());
    std::sort(remoteRanks.begin(), remoteRanks.end());

    const bool hasTwoLayers = localGroupIdx < groups.size() && remoteGroupIdx < groups.size();
    const bool isTwoByEight = param.rankSize == 16 && hasTwoLayers && localRanks.size() == 8 && remoteRanks.size() == 8;
    const bool isEightPlusFour = param.rankSize == 12 && hasTwoLayers
                                 && ((localRanks.size() == 8 && remoteRanks.size() == 4)
                                     || (localRanks.size() == 4 && remoteRanks.size() == 8));
    const bool hierarchicalEnabled = isTwoByEight || isEightPlusFour;
    uint32_t pairRank = param.myRank;
    uint32_t pairChannelIdx = 0;
    std::vector<uint32_t> ownedRemoteRanks;
    std::vector<uint32_t> ownedRemoteChannelIdx;
    std::vector<uint32_t> stageTwoTargetChannelIdx;
    if (hierarchicalEnabled) {
        const auto localRankIt = std::find(localRanks.begin(), localRanks.end(), param.myRank);
        CHK_PRT_RET(localRankIt == localRanks.end(),
            HCCL_ERROR("[AllGather] rank %u is absent from its local rank group", param.myRank), HCCL_E_INTERNAL);
        const size_t localRankIdx = static_cast<size_t>(localRankIt - localRanks.begin());
        pairRank = remoteRanks[localRankIdx % remoteRanks.size()];

        const ChannelGroup &remoteGroup = groups[remoteGroupIdx];
        const auto getRemoteChannelIdx = [&remoteGroup](uint32_t remoteRank) {
            const auto channelIt = std::find_if(
                remoteGroup.plans.begin(), remoteGroup.plans.end(), [remoteRank](const ChannelPlan &plan) {
                    return plan.remoteRank == remoteRank;
                });
            return static_cast<size_t>(channelIt - remoteGroup.plans.begin());
        };
        const size_t pairChannel = getRemoteChannelIdx(pairRank);
        CHK_PRT_RET(pairChannel >= remoteGroup.plans.size(),
            HCCL_ERROR("[AllGather] no layer-1 channel from rank %u to pair rank %u", param.myRank, pairRank),
            HCCL_E_INTERNAL);
        pairChannelIdx = static_cast<uint32_t>(pairChannel);

        for (size_t remoteRankIdx = 0; remoteRankIdx < remoteRanks.size(); ++remoteRankIdx) {
            if (remoteRankIdx % localRanks.size() == localRankIdx) {
                ownedRemoteRanks.push_back(remoteRanks[remoteRankIdx]);
                ownedRemoteChannelIdx.push_back(static_cast<uint32_t>(getRemoteChannelIdx(remoteRanks[remoteRankIdx])));
            }
        }

        for (size_t targetRankIdx = localRankIdx; targetRankIdx < remoteRanks.size();
            targetRankIdx += localRanks.size()) {
            const size_t targetChannel = getRemoteChannelIdx(remoteRanks[targetRankIdx]);
            CHK_PRT_RET(targetChannel >= remoteGroup.plans.size(),
                HCCL_ERROR(
                    "[AllGather] no stage-2 channel from rank %u to rank %u", param.myRank, remoteRanks[targetRankIdx]),
                HCCL_E_INTERNAL);
            stageTwoTargetChannelIdx.push_back(static_cast<uint32_t>(targetChannel));
        }
        HCCL_INFO("[AllGather] rank %u enables hierarchical path: local %zu, remote %zu, stage-1 pair %u, "
                  "owned remote %zu, stage-2 targets %zu",
            param.myRank, localRanks.size(), remoteRanks.size(), pairRank, ownedRemoteRanks.size(),
            stageTwoTargetChannelIdx.size());
    }

    // 4*1 递归倍增配对：round-1 = myRank^1，round-2 = myRank^2（rank id 在通信域内为 0..3）
    uint32_t pairChannelIdxR2 = 0;
    const bool recursiveEnabled = param.rankSize == 4 && remoteGroupIdx < groups.size() && remoteRanks.size() == 3;
    if (recursiveEnabled) {
        const ChannelGroup &remoteGroup = groups[remoteGroupIdx];
        const auto getRemoteChannelIdx = [&remoteGroup](uint32_t remoteRank) {
            const auto channelIt = std::find_if(
                remoteGroup.plans.begin(), remoteGroup.plans.end(), [remoteRank](const ChannelPlan &plan) {
                    return plan.remoteRank == remoteRank;
                });
            return static_cast<size_t>(channelIt - remoteGroup.plans.begin());
        };
        pairRank = param.myRank ^ 1U;
        const size_t pairChannel = getRemoteChannelIdx(pairRank);
        CHK_PRT_RET(pairChannel >= remoteGroup.plans.size(),
            HCCL_ERROR("[AllGather] no layer-1 channel from rank %u to round-1 pair %u", param.myRank, pairRank),
            HCCL_E_INTERNAL);
        pairChannelIdx = static_cast<uint32_t>(pairChannel);
        const uint32_t pairRankR2 = param.myRank ^ 2U;
        const size_t pairChannelR2 = getRemoteChannelIdx(pairRankR2);
        CHK_PRT_RET(pairChannelR2 >= remoteGroup.plans.size(),
            HCCL_ERROR("[AllGather] no layer-1 channel from rank %u to round-2 pair %u", param.myRank, pairRankR2),
            HCCL_E_INTERNAL);
        pairChannelIdxR2 = static_cast<uint32_t>(pairChannelR2);
        HCCL_INFO("[AllGather] rank %u enables recursive path: round-1 pair %u, round-2 pair %u",
            param.myRank, pairRank, pairRankR2);
    }

    std::vector<CcuKernelInfo> kernelInfos(groups.size());
    std::vector<CcuKernelInfo> flatKernelInfos(groups.size());
    for (size_t idx = 0; idx < groups.size(); ++idx) {
        CcuKernelInfo &kernelInfo = kernelInfos[idx];
        const int nameRet = std::snprintf(
            kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllGatherDirectPush_%zu", idx);
        CHK_PRT_RET(nameRet <= 0 || static_cast<size_t>(nameRet) >= sizeof(kernelInfo.kernelFuncName),
            HCCL_ERROR("[AllGather] Failed to create CCU kernel name"), HCCL_E_INTERNAL);
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuAllGatherDirectPush);

        auto kernelArg = std::make_shared<CcuKernelArgAllGather>();
        kernelArg->channelCount = static_cast<uint32_t>(groups[idx].handles.size());
        kernelArg->copySelf = (idx == copyKernelIdx) ? 1U : 0U;
        kernelArg->isLocalLayer = (idx == localGroupIdx) ? 1U : 0U;
        kernelArg->hierarchicalEnabled = hierarchicalEnabled ? 1U : 0U;
        kernelArg->myRank = param.myRank;
        kernelArg->pairRank = pairRank;
        kernelArg->pairChannelIdx = pairChannelIdx;
        kernelArg->pairChannelIdxR2 = pairChannelIdxR2;
        kernelArg->localRankCount = static_cast<uint32_t>(localRanks.size());
        kernelArg->remoteRankCount = static_cast<uint32_t>(remoteRanks.size());
        for (size_t rankIdx = 0; rankIdx < localRanks.size(); ++rankIdx) {
            kernelArg->localRanks[rankIdx] = localRanks[rankIdx];
        }
        kernelArg->ownedRemoteCount = static_cast<uint32_t>(ownedRemoteRanks.size());
        for (size_t rankIdx = 0; rankIdx < ownedRemoteRanks.size(); ++rankIdx) {
            kernelArg->ownedRemoteRanks[rankIdx] = ownedRemoteRanks[rankIdx];
            kernelArg->ownedRemoteChannelIdx[rankIdx] = ownedRemoteChannelIdx[rankIdx];
        }
        kernelArg->stageTwoTargetCount = static_cast<uint32_t>(stageTwoTargetChannelIdx.size());
        for (size_t targetIdx = 0; targetIdx < stageTwoTargetChannelIdx.size(); ++targetIdx) {
            kernelArg->stageTwoTargetChannelIdx[targetIdx] = stageTwoTargetChannelIdx[targetIdx];
        }
        for (size_t channelIdx = 0; channelIdx < groups[idx].handles.size(); ++channelIdx) {
            kernelArg->channels[channelIdx] = groups[idx].handles[channelIdx];
        }
        kernelInfo.setKernelArg(kernelArg);

        // 同一份 kernelArg 再注册一个极简扁平 kernel（小消息/4*1 走它，指令数最小化）
        CcuKernelInfo &flatKernelInfo = flatKernelInfos[idx];
        const int flatNameRet = std::snprintf(
            flatKernelInfo.kernelFuncName, sizeof(flatKernelInfo.kernelFuncName), "CcuAllGatherFlat_%zu", idx);
        CHK_PRT_RET(flatNameRet <= 0 || static_cast<size_t>(flatNameRet) >= sizeof(flatKernelInfo.kernelFuncName),
            HCCL_ERROR("[AllGather] Failed to create CCU kernel name"), HCCL_E_INTERNAL);
        flatKernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuAllGatherFlat);
        flatKernelInfo.setKernelArg(kernelArg);
    }

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("[AllGather] CCU kernel register start failed: %d", ccuRet);
        return ConvertCcuResult(ccuRet);
    }

    // 每条 channel 只能属于一个 kernel：分层拓扑只注册分层 kernel（内含扁平整写分支），
    // 非分层拓扑只注册极简扁平 kernel（含 4*1 递归两轮）
    const bool registerHier = hierarchicalEnabled;
    if (registerHier) {
        resCtx.ccuKernels.resize(kernelInfos.size());
    } else {
        resCtx.flatKernels.resize(flatKernelInfos.size());
    }
    for (size_t idx = 0; idx < kernelInfos.size(); ++idx) {
        if (registerHier) {
            const void *kernelArgs[] = {kernelInfos[idx].kernelArg};
            ccuRet = HcommCcuKernelRegister(insHandle, RESERVED_DIE_ID, kernelInfos[idx].kernelFuncName,
                kernelInfos[idx].kernelFunc, kernelArgs, KERNEL_ARG_NUM, &resCtx.ccuKernels[idx]);
            if (ccuRet != CCU_SUCCESS) {
                HCCL_ERROR("[AllGather] CCU kernel %zu register failed: %d", idx, ccuRet);
                return ConvertCcuResult(ccuRet);
            }
        } else {
            const void *flatKernelArgs[] = {flatKernelInfos[idx].kernelArg};
            ccuRet = HcommCcuKernelRegister(insHandle, RESERVED_DIE_ID, flatKernelInfos[idx].kernelFuncName,
                flatKernelInfos[idx].kernelFunc, flatKernelArgs, KERNEL_ARG_NUM, &resCtx.flatKernels[idx]);
            if (ccuRet != CCU_SUCCESS) {
                HCCL_ERROR("[AllGather] CCU flat kernel %zu register failed: %d", idx, ccuRet);
                return ConvertCcuResult(ccuRet);
            }
        }
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("[AllGather] CCU kernel register end failed: %d", ccuRet);
        return ConvertCcuResult(ccuRet);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    resCtx.ccuThread = param.cpuThread;
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    resCtx.threads.push_back(param.cpuThread);

    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    std::vector<ChannelGroup> groups;
    CHK_RET(BuildChannelGroups(comm, param, groups));
    if (groups.size() == MAX_KERNEL_NUM) {
        resCtx.threads.resize(MAX_KERNEL_NUM);
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, 1, 1, &resCtx.threads[1]));
    }
    CHK_RET(RegisterKernels(comm, param, groups, resCtx));
    return HCCL_SUCCESS;
}

// 每个通信域的静态信息缓存：comm 创建后 rank/dfx 信息不变，避免每 call 重复 HCCL API 查询
struct CommStaticInfo {
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    aclrtStream stream = nullptr;
    ThreadHandle cpuThread = 0;
};

std::mutex g_commInfoMtx;
std::map<HcclComm, CommStaticInfo> g_commInfoCache;

HcclResult GetCommStaticInfo(HcclComm comm, aclrtStream stream, CommEngine engine, CommStaticInfo &info)
{
    {
        std::lock_guard<std::mutex> lock(g_commInfoMtx);
        const auto it = g_commInfoCache.find(comm);
        if (it != g_commInfoCache.end() && it->second.stream == stream) {
            info = it->second;
            return HCCL_SUCCESS;
        }
    }
    // 注册算子信息（每通信域一次）
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CommStaticInfo fresh;
    CHK_RET(HcclGetRankId(comm, &fresh.myRank));
    CHK_RET(HcclGetRankSize(comm, &fresh.rankSize));
    fresh.stream = stream;
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread，并申请 1 个 notify
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream, 1, &fresh.cpuThread));

    std::lock_guard<std::mutex> lock(g_commInfoMtx);
    g_commInfoCache[comm] = fresh;
    info = fresh;
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

    // 构造算子参数
    OpParam param;
    const int tagRet = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allgather");
    CHK_PRT_RET(tagRet <= 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("[AllGather] Failed to fill operation tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    // ==============================================
    // STEP 1: 解析拓扑信息（每通信域缓存）
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CommStaticInfo commInfo;
    CHK_RET(GetCommStaticInfo(comm, stream, ccuEngine, commInfo));
    param.myRank = commInfo.myRank;
    param.rankSize = commInfo.rankSize;
    param.cpuThread = commInfo.cpuThread;
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("[AllGather] Invalid rank: myRank %u, rankSize %u", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(),
        HCCL_ERROR("[AllGather] Unsupported data type %d", dataType), HCCL_E_NOT_SUPPORT);

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
        CHK_RET(CreateResources(comm, param, resCtxHost));

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
