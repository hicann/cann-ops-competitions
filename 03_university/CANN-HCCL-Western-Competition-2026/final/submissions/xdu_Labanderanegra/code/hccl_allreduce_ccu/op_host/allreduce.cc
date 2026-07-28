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

constexpr uint32_t DIE_NUM = 2;
constexpr uint32_t LEVEL_NUM = 2;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t ROOT_RANK = 0;

enum KernelLevel : uint32_t {
    LOCAL_MESH_LEVEL = 0,
    COLUMN_TREE_LEVEL = 1,
};

struct GcdPlan {
    uint32_t localGroupSize = 1;
    uint32_t localIndex = 0;
    uint32_t subgroupBase = 0;
    bool columnParent = false;
    std::vector<uint32_t> localPeers;
    std::vector<uint32_t> columnPeers;
};

struct PeerChannel {
    uint32_t dieId = 0;
    ChannelHandle channel{};
};

struct ChannelGroup {
    std::vector<ChannelHandle> channels;
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
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        case CCU_E_INTERNAL:
        default:
            return HCCL_E_INTERNAL;
    }
}

uint32_t Gcd(uint32_t left, uint32_t right)
{
    while (right != 0) {
        const uint32_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

void BuildRootFallbackPlan(const OpParam &param, GcdPlan &plan)
{
    plan = {};
    plan.columnParent = param.myRank == ROOT_RANK;
    if (plan.columnParent) {
        for (uint32_t rank = 1; rank < param.rankSize; ++rank) {
            plan.columnPeers.push_back(rank);
        }
    } else {
        plan.columnPeers.push_back(ROOT_RANK);
    }
}

HcclResult BuildGcdPlan(HcclComm comm, const OpParam &param, GcdPlan &plan)
{
    uint32_t *instanceSizes = nullptr;
    uint32_t serverNum = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &instanceSizes, &serverNum));
    if (serverNum == 0) {
        HCCL_WARNING("Layer 0 has no instance sizes; use root fallback");
        BuildRootFallbackPlan(param, plan);
        return HCCL_SUCCESS;
    }

    uint32_t groupSize = instanceSizes[0];
    uint32_t rankTotal = 0;
    uint32_t myServerBase = 0;
    uint32_t myServerSize = 0;
    std::vector<uint32_t> serverBases;
    for (uint32_t serverIdx = 0; serverIdx < serverNum; ++serverIdx) {
        const uint32_t serverSize = instanceSizes[serverIdx];
        if (serverSize == 0 || serverSize > param.rankSize || rankTotal > param.rankSize - serverSize) {
            HCCL_WARNING("Invalid layer 0 instance sizes; use root fallback");
            BuildRootFallbackPlan(param, plan);
            return HCCL_SUCCESS;
        }
        serverBases.push_back(rankTotal);
        if (param.myRank >= rankTotal && param.myRank < rankTotal + serverSize) {
            myServerBase = rankTotal;
            myServerSize = serverSize;
        }
        rankTotal += serverSize;
        groupSize = Gcd(groupSize, serverSize);
    }
    if (rankTotal != param.rankSize || myServerSize == 0 || groupSize == 0) {
        HCCL_WARNING("Layer 0 sizes do not cover the communication domain; use root fallback");
        BuildRootFallbackPlan(param, plan);
        return HCCL_SUCCESS;
    }

    plan = {};
    plan.localGroupSize = groupSize;
    const uint32_t indexInServer = param.myRank - myServerBase;
    plan.subgroupBase = myServerBase + (indexInServer / groupSize) * groupSize;
    plan.localIndex = param.myRank - plan.subgroupBase;

    for (uint32_t index = 0; index < groupSize; ++index) {
        const uint32_t rank = plan.subgroupBase + index;
        if (rank != param.myRank) {
            plan.localPeers.push_back(rank);
        }
    }

    std::vector<uint32_t> columnRanks;
    for (uint32_t serverIdx = 0; serverIdx < serverNum; ++serverIdx) {
        const uint32_t serverSize = instanceSizes[serverIdx];
        for (uint32_t subgroupOffset = 0; subgroupOffset < serverSize; subgroupOffset += groupSize) {
            columnRanks.push_back(serverBases[serverIdx] + subgroupOffset + plan.localIndex);
        }
    }
    const uint32_t columnRoot = columnRanks.front();
    plan.columnParent = param.myRank == columnRoot;
    if (plan.columnParent) {
        for (uint32_t rank : columnRanks) {
            if (rank != columnRoot) {
                plan.columnPeers.push_back(rank);
            }
        }
    } else {
        plan.columnPeers.push_back(columnRoot);
    }

    HCCL_INFO("GCD plan rank[%u] groupSize[%u] subgroupBase[%u] localIndex[%u] localPeers[%zu] "
              "columnParent[%d] columnPeers[%zu]",
        param.myRank, plan.localGroupSize, plan.subgroupBase, plan.localIndex, plan.localPeers.size(),
        plan.columnParent, plan.columnPeers.size());
    return HCCL_SUCCESS;
}

HcclResult FindCcuChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));

    constexpr CommProtocol PROTOCOLS[] = {COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UBC_TP};
    for (CommProtocol protocol : PROTOCOLS) {
        for (uint32_t layerIdx = 0; layerIdx < layerNum; ++layerIdx) {
            CommLink *links = nullptr;
            uint32_t linkNum = 0;
            if (HcclRankGraphGetLinks(comm, layers[layerIdx], srcRank, dstRank, &links, &linkNum) != HCCL_SUCCESS) {
                continue;
            }
            for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                const CommLink &link = links[linkIdx];
                if (link.linkAttr.linkProtocol != protocol) {
                    continue;
                }

                CHK_RET(HcclChannelDescInit(&desc, CHANNEL_NOTIFY_NUM));
                desc.remoteRank = dstRank;
                desc.notifyNum = CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = protocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("No CCU channel from rank %u to peer %u", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquirePeerChannels(
    HcclComm comm, const OpParam &param, const std::vector<uint32_t> &peers, std::vector<PeerChannel> &channels)
{
    for (uint32_t peer : peers) {
        HcclChannelDesc desc{};
        CHK_RET(FindCcuChannelDesc(comm, param.myRank, peer, desc));

        EndpointAttrDieId dieId = 0;
        CHK_RET(HcclRankGraphGetEndpointInfo(
            comm, param.myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
        if (dieId >= DIE_NUM) {
            HCCL_ERROR("Invalid die id %u for rank %u channel to rank %u", dieId, param.myRank, peer);
            return HCCL_E_INTERNAL;
        }

        ChannelHandle channel{};
        CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, &desc, 1, &channel));
        channels.push_back(PeerChannel{dieId, channel});
    }
    return HCCL_SUCCESS;
}

HcclResult BuildLevelGroups(const GcdPlan &plan, const std::vector<PeerChannel> &localChannels,
    const std::vector<PeerChannel> &columnChannels, uint32_t level,
    std::array<ChannelGroup, DIE_NUM> &groups, bool &isParent)
{
    groups = {};
    if (level == LOCAL_MESH_LEVEL) {
        isParent = false;
        for (const PeerChannel &channel : localChannels) {
            groups[channel.dieId].channels.push_back(channel.channel);
        }
    } else {
        isParent = plan.columnParent;
        for (const PeerChannel &channel : columnChannels) {
            groups[channel.dieId].channels.push_back(channel.channel);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterGcdKernels(HcclComm comm, const OpParam &param, const GcdPlan &plan,
    const std::vector<PeerChannel> &localChannels, const std::vector<PeerChannel> &columnChannels,
    AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle{};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    if (insNum != 1) {
        HCCL_ERROR("Expected one CCU instance, got %u", insNum);
        return HCCL_E_INTERNAL;
    }

    resCtx.localGroupSize = plan.localGroupSize;
    resCtx.localIndex = plan.localIndex;
    resCtx.ccuKernels.assign(LEVEL_NUM * DIE_NUM, CcuKernelHandle{});
    resCtx.kernelValid.assign(LEVEL_NUM * DIE_NUM, 0);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuResult(ccuRet);
    }

    uint32_t registeredKernelNum = 0;
    for (uint32_t level = 0; level < LEVEL_NUM; ++level) {
        std::array<ChannelGroup, DIE_NUM> groups{};
        bool isParent = false;
        CHK_RET(BuildLevelGroups(plan, localChannels, columnChannels, level, groups, isParent));

        for (uint32_t dieId = 0; dieId < DIE_NUM; ++dieId) {
            const ChannelGroup &group = groups[dieId];
            if (group.channels.empty()) {
                continue;
            }
            if (group.channels.size() >= MAX_RANK_SIZE) {
                HCCL_ERROR("Too many channels on level %u die %u: %zu", level, dieId, group.channels.size());
                return HCCL_E_INTERNAL;
            }

            CcuKernelInfo kernelInfo{};
            int nameLen = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
                "CcuGcdRsag_%u_%u", level, dieId);
            if (nameLen <= 0 || static_cast<size_t>(nameLen) >= sizeof(kernelInfo.kernelFuncName)) {
                return HCCL_E_INTERNAL;
            }

            if (level == LOCAL_MESH_LEVEL) {
                kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuLocalMeshKernel);
                auto kernelArg = std::make_shared<ops_hccl::LocalMeshKernelArg>();
                kernelArg->rankId = param.myRank;
                kernelArg->rankSize = param.rankSize;
                kernelArg->dataType = param.dataType;
                kernelArg->reduceOp = param.reduceType;
                kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
                for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
                    kernelArg->channels[channelIdx] = group.channels[channelIdx];
                }
                kernelInfo.setKernelArg(kernelArg);
            } else {
                kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuTreeAllReduceKernel);
                auto kernelArg = std::make_shared<ops_hccl::TreeAllReduceKernelArg>();
                kernelArg->rankId = param.myRank;
                kernelArg->rankSize = param.rankSize;
                kernelArg->dataType = param.dataType;
                kernelArg->reduceOp = param.reduceType;
                kernelArg->isParent = isParent;
                kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
                for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
                    kernelArg->channels[channelIdx] = group.channels[channelIdx];
                }
                kernelInfo.setKernelArg(kernelArg);
            }

            const void *kernelArgs[] = {kernelInfo.kernelArg};
            CcuKernelHandle kernelHandle{};
            ccuRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName, kernelInfo.kernelFunc,
                kernelArgs, 1, &kernelHandle);
            if (ccuRet != CCU_SUCCESS) {
                HCCL_ERROR("Register GCD kernel failed: rank[%u] level[%u] die[%u] channels[%u] ccuRet[%d]",
                    param.myRank, level, dieId,
                    static_cast<CcuKernelArgBase *>(kernelInfo.kernelArg)->channelCount, ccuRet);
                return ConvertCcuResult(ccuRet);
            }

            const uint32_t kernelIdx = level * DIE_NUM + dieId;
            resCtx.ccuKernels[kernelIdx] = kernelHandle;
            resCtx.kernelValid[kernelIdx] = 1;
            ++registeredKernelNum;
        }
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuResult(ccuRet);
    }
    if (registeredKernelNum == 0) {
        HCCL_ERROR("No GCD kernel was registered for rank %u", param.myRank);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    int tagLen = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allreduce_gcd_rsag_v2");
    if (tagLen <= 0 || static_cast<size_t>(tagLen) >= sizeof(param.tag)) {
        return HCCL_E_INTERNAL;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.reduceType = op;

    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("The GCD RSAG implementation supports FP32 SUM only, dataType[%d] reduceOp[%d]", dataType, op);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE) {
        HCCL_ERROR("Unsupported rank size %u", param.rankSize);
        return HCCL_E_PARA;
    }
    if (count == 0) {
        return HCCL_SUCCESS;
    }

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("Reuse GCD RSAG allreduce engine context");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost{};
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        const uint32_t threadNum = param.rankSize == 1 ? 1 : DIE_NUM;
        resCtxHost.threads.resize(threadNum);
        const uint32_t mainNotifyNum = param.rankSize == 1 ? 0 : 1;
        CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, mainNotifyNum, &resCtxHost.threads[0]));
        if (threadNum == DIE_NUM) {
            CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, 1, &resCtxHost.threads[1]));
        }
        resCtxHost.ccuThread = resCtxHost.threads[0];
        param.cpuThread = resCtxHost.threads[0];

        if (param.rankSize > 1) {
            GcdPlan plan;
            std::vector<PeerChannel> localChannels;
            std::vector<PeerChannel> columnChannels;
            CHK_RET(BuildGcdPlan(comm, param, plan));
            CHK_RET(AcquirePeerChannels(comm, param, plan.localPeers, localChannels));
            CHK_RET(AcquirePeerChannels(comm, param, plan.columnPeers, columnChannels));
            CHK_RET(RegisterGcdKernels(comm, param, plan, localChannels, columnChannels, resCtxHost));
        }

        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
