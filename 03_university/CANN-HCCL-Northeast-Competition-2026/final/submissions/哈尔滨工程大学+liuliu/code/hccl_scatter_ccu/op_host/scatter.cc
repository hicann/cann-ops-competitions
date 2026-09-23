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
#include <map>
#include <limits>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_launch.h>
#include "log.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
struct RelayPlan {
    bool enabled = false;
    uint32_t roles[MAX_RANK_SIZE]{};
    uint32_t partners[MAX_RANK_SIZE]{};
};

RelayPlan MakeRelayPlan(HcclComm comm, const OpParam &param, uint64_t localBufferSize)
{
    RelayPlan plan{};
    std::fill(std::begin(plan.roles), std::end(plan.roles), SCATTER_ROLE_DIRECT);
    std::fill(std::begin(plan.partners), std::end(plan.partners), INVALID_VALUE_RANKID);
    plan.roles[param.root] = SCATTER_ROLE_ROOT;
    const uint64_t bytes = param.count * sizeof(float);
    if (bytes < SCATTER_RELAY_THRESHOLD || localBufferSize < SCATTER_CCU_CHUNK) {
        return plan;
    }
    uint32_t *sizes = nullptr, groupCount = 0;
    if (HcclRankGraphGetInstSizeListByLayer(comm, 0, &sizes, &groupCount) != HCCL_SUCCESS || sizes == nullptr) {
        return plan;
    }
    const bool isTwoByEight = param.rankSize == 16 && groupCount == 2 && sizes[0] == 8 && sizes[1] == 8;
    const bool isFourByThree
        = param.rankSize == 12 && groupCount == 4 && sizes[0] == 3 && sizes[1] == 3 && sizes[2] == 3 && sizes[3] == 3;
    if (!isTwoByEight && !isFourByThree) {
        return plan;
    }
    uint32_t *localRanks = nullptr, localCount = 0;
    if (HcclRankGraphGetRanksByLayer(comm, 0, &localRanks, &localCount) != HCCL_SUCCESS || localRanks == nullptr) {
        return plan;
    }
    uint32_t begin = 0, rootBegin = 0, rootEnd = 0, myBegin = 0, myEnd = 0;
    for (uint32_t group = 0; group < groupCount; ++group) {
        const uint32_t end = begin + sizes[group];
        if (param.root >= begin && param.root < end) {
            rootBegin = begin;
            rootEnd = end;
        }
        if (param.myRank >= begin && param.myRank < end) {
            myBegin = begin;
            myEnd = end;
        }
        begin = end;
    }
    if (begin != param.rankSize || rootEnd <= rootBegin || myEnd <= myBegin || localCount != myEnd - myBegin) {
        return plan;
    }
    uint64_t actualMask = 0, expectedMask = 0;
    for (uint32_t i = 0; i < localCount; ++i) {
        if (localRanks[i] >= param.rankSize) {
            return plan;
        }
        actualMask |= 1ULL << localRanks[i];
    }
    for (uint32_t rank = myBegin; rank < myEnd; ++rank) {
        expectedMask |= 1ULL << rank;
    }
    if (actualMask != expectedMask) {
        return plan;
    }
    std::vector<uint32_t> helpers;
    std::vector<uint32_t> targets;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank >= rootBegin && rank < rootEnd) {
            if (rank != param.root) {
                helpers.push_back(rank);
            }
        } else {
            targets.push_back(rank);
        }
    }
    if (helpers.empty() || targets.size() <= 4 || helpers.size() >= targets.size()) {
        return plan;
    }
    for (uint32_t helper : helpers) {
        plan.roles[helper] = SCATTER_ROLE_HELPER;
    }
    for (size_t i = 0; i < helpers.size(); ++i) {
        const uint32_t helper = helpers[i];
        const uint32_t target = targets[i];
        plan.roles[target] = SCATTER_ROLE_TARGET;
        plan.partners[helper] = target;
        plan.partners[target] = helper;
    }
    plan.enabled = true;
    return plan;
}

HcclResult CreateResources(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx)
{
    CHK_RET(HcclGetHcclBuffer(comm, &ctx.localBuffer.addr, &ctx.localBuffer.size));
    const RelayPlan plan = MakeRelayPlan(comm, param, ctx.localBuffer.size);
    ctx.relayEnabled = plan.enabled;
    CcuKernelArgBase kernelArg{};
    kernelArg.rankSize = param.rankSize;
    kernelArg.myRank = param.myRank;
    std::vector<HcclChannelDesc> descs;
    std::map<uint32_t, std::vector<uint32_t>> dieGroups;
    if (param.rankSize > 1) {
        uint32_t *layerIds = nullptr, layerCount = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &layerIds, &layerCount));
        if (layerIds == nullptr || layerCount == 0) {
            return HCCL_E_PARA;
        }
        std::vector<uint32_t> layers(layerIds, layerIds + layerCount);
        std::sort(layers.begin(), layers.end());
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.myRank) {
                continue;
            }
            HcclChannelDesc desc{};
            CHK_RET(HcclChannelDescInit(&desc, 1));
            bool found = false;
            for (uint32_t layer : layers) {
                CommLink *links = nullptr;
                uint32_t count = 0;
                CHK_RET(HcclRankGraphGetLinks(comm, layer, param.myRank, peer, &links, &count));
                for (uint32_t i = 0; links != nullptr && i < count; ++i) {
                    // Use the topology's protocol, including cross-server layers.
                    desc.remoteRank = peer;
                    desc.channelProtocol = links[i].linkAttr.linkProtocol;
                    desc.localEndpoint = links[i].srcEndpointDesc;
                    desc.remoteEndpoint = links[i].dstEndpointDesc;
                    desc.notifyNum = 3;
                    found = true;
                    break;
                }
                if (found) {
                    break;
                }
            }
            if (!found) {
                HCCL_ERROR("CCU Scatter: no link rank=%u peer=%u", param.myRank, peer);
                return HCCL_E_NOT_SUPPORT;
            }
            EndpointAttrDieId die = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(
                comm, param.myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(die), &die));
            if (die == std::numeric_limits<uint32_t>::max()) {
                return HCCL_E_PARA;
            }
            dieGroups[die].push_back(static_cast<uint32_t>(descs.size()));
            descs.push_back(desc);
        }
        kernelArg.channelCount = static_cast<uint32_t>(descs.size());
        CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, descs.data(), kernelArg.channelCount, kernelArg.channels));
    }
    CcuInsHandle ins = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &ins, &insNum));
    if (insNum != 1) {
        HCCL_ERROR("CCU Scatter: expected one instance, got %u", insNum);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_RET(ConvertCcuToHccl(HcommCcuKernelRegisterStart(ins)));
    if (dieGroups.empty()) {
        dieGroups[0] = {};
    }
    std::vector<uint32_t> peerGroups(param.rankSize, INVALID_VALUE_RANKID);
    uint32_t mappedGroup = 0;
    for (const auto &group : dieGroups) {
        for (uint32_t index : group.second) {
            peerGroups[descs[index].remoteRank] = mappedGroup;
        }
        ++mappedGroup;
    }
    ctx.workers.resize(dieGroups.size() - 1);
    if (!ctx.workers.empty()) {
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU, static_cast<uint32_t>(ctx.workers.size()),
            plan.enabled ? 2 : 1, ctx.workers.data()));
    }
    ctx.ccuKernels.resize(dieGroups.size() + 1);
    uint32_t groupIndex = 0;
    for (const auto &group : dieGroups) {
        CcuKernelArgBase groupArg{};
        groupArg.rankSize = param.rankSize;
        groupArg.myRank = param.myRank;
        groupArg.rootRank = param.root;
        groupArg.singleChunk = param.count > 0 && param.count <= SCATTER_CCU_CHUNK / sizeof(float);
        groupArg.role = plan.roles[param.myRank];
        groupArg.partnerRank = plan.partners[param.myRank];
        groupArg.groupIndex = groupIndex;
        groupArg.rootGroup = param.myRank == param.root ? INVALID_VALUE_RANKID : peerGroups[param.root];
        groupArg.partnerGroup
            = groupArg.partnerRank == INVALID_VALUE_RANKID ? INVALID_VALUE_RANKID : peerGroups[groupArg.partnerRank];
        std::copy(std::begin(plan.roles), std::end(plan.roles), std::begin(groupArg.peerRoles));
        std::copy(std::begin(plan.partners), std::end(plan.partners), std::begin(groupArg.partners));
        uint64_t peerMask = 0;
        for (uint32_t index : group.second) {
            const uint32_t slot = groupArg.channelCount++;
            groupArg.channels[slot] = kernelArg.channels[index];
            groupArg.peerRanks[slot] = descs[index].remoteRank;
            peerMask |= 1ULL << groupArg.peerRanks[slot];
        }
        ctx.groupMasks.push_back(peerMask);
        char name[64]{};
        std::snprintf(name, sizeof(name), "FinalScatterV4Die%uRank%u", group.first, param.myRank);
        const void *args[] = {&groupArg};
        HCCL_INFO("CCU Scatter: rank=%u die=%u channels=%u", param.myRank, group.first, groupArg.channelCount);
        const void *kernel = plan.enabled ? reinterpret_cast<const void *>(ops_hccl::CcuRelayKernel)
                                          : reinterpret_cast<const void *>(ops_hccl::CcuKernel);
        const CcuResult registered
            = HcommCcuKernelRegister(ins, group.first, name, kernel, args, 1, &ctx.ccuKernels[groupIndex]);
        if (registered != CCU_SUCCESS) {
            HCCL_ERROR("CCU Scatter: register failed rank=%u die=%u channels=%u ret=%d", param.myRank, group.first,
                groupArg.channelCount, static_cast<int>(registered));
            return ConvertCcuToHccl(registered);
        }
        ++groupIndex;
    }
    if (plan.enabled) {
        // The relay main mission also handles deferred local copies. No extra mission is registered.
        ctx.ccuKernels.back() = ctx.ccuKernels.front();
    } else {
        const void *copyArgs[] = {&kernelArg};
        CHK_RET(ConvertCcuToHccl(HcommCcuKernelRegister(ins, dieGroups.begin()->first, "FinalScatterCopyV4",
            reinterpret_cast<const void *>(ops_hccl::CcuCopyKernel), copyArgs, 1, &ctx.ccuKernels.back())));
    }
    CHK_RET(ConvertCcuToHccl(HcommCcuKernelRegisterEnd(ins)));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    OpParam param{};
    std::snprintf(param.tag, sizeof(param.tag), "final_scatter_v4_15_r%u_large%u_short%u", root,
        recvCount >= SCATTER_RELAY_THRESHOLD / sizeof(float) ? 1U : 0U,
        recvCount <= SCATTER_CCU_CHUNK / sizeof(float) ? 1U : 0U);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize
        || root >= param.rankSize || dataType != HCCL_DATA_TYPE_FP32
        || recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize) {
        return HCCL_E_PARA;
    }
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(recvBuf);
    const uint64_t bytes = recvCount * sizeof(float);
    if (reinterpret_cast<uint64_t>(recvBuf) > std::numeric_limits<uint64_t>::max() - bytes) {
        return HCCL_E_PARA;
    }
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
        if (reinterpret_cast<uint64_t>(sendBuf) > std::numeric_limits<uint64_t>::max() - bytes * param.rankSize) {
            return HCCL_E_PARA;
        }
    }
    HcclDfxOpInfo dfx{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfx)));
    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, MAX_RANK_SIZE, &param.cpuThread));
    if (HcclEngineCtxGet(comm, param.tag, COMM_ENGINE_CCU, &param.resCtx, &param.ctxSize) != HCCL_SUCCESS) {
        AlgResourceCtx resources;
        CHK_RET(CreateResources(comm, param, resources));
        std::vector<char> data = resources.Serialize();
        param.ctxSize = data.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, COMM_ENGINE_CCU, param.tag, data.data(), data.size(), 0));
    }
    CHK_PTR_NULL(param.resCtx);
    return ops_hccl::ExecOp(param);
}
