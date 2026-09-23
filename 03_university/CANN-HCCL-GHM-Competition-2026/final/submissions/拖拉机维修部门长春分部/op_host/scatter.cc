/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
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
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;

bool ReceiverPullSizeCandidate(const OpParam &param)
{
    const uint64_t bytes = param.count * sizeof(float);
    return bytes > 0 && bytes <= 512ULL * 1024ULL &&
        (param.rankSize == 4 || param.rankSize == 12);
}

bool OwnCopyCandidate(const OpParam &param, bool receiverPull = false)
{
    const uint64_t bytes = param.count * sizeof(float);
    if (param.myRank != param.root ||
        (param.rankSize != 4 && param.rankSize != 12 && param.rankSize != 16) ||
        (bytes < 8ULL * 1024ULL * 1024ULL &&
            !(receiverPull && ReceiverPullSizeCandidate(param))) || bytes > MAX_DATA_SIZE) {
        return false;
    }
    const uint64_t input = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t output = reinterpret_cast<uint64_t>(param.outputPtr);
    // Require the output to be disjoint from the entire input, including
    // remote slices still being read. Subtraction avoids end-address overflow.
    return output >= input ? output - input >= bytes * param.rankSize :
        input - output >= bytes;
}

struct PeerChannel {
    uint32_t rank;
    ChannelHandle handle;
};

struct RelayPlan {
    uint32_t helperFor[MAX_RANK_SIZE]{};
    uint32_t targetFor[MAX_RANK_SIZE]{};
    uint32_t numerator = 0;
    uint32_t denominator = 1;
    bool relayPrefixFirst = false;
};

bool ReceiverPullCandidate(HcclComm comm, const OpParam &param)
{
    // Use the actual receive size range, without assuming how a benchmark
    // input-size label is converted to recvCount.
    if (!ReceiverPullSizeCandidate(param)) {
        return false;
    }
    uint32_t *serverSizes = nullptr;
    uint32_t serverCount = 0;
    if (HcclRankGraphGetInstSizeListByLayer(comm, 0, &serverSizes, &serverCount) != HCCL_SUCCESS ||
        serverSizes == nullptr || serverCount != 4) {
        return false;
    }
    const uint32_t expectedSize = param.rankSize == 4 ? 1U : 3U;
    for (uint32_t server = 0; server < serverCount; ++server) {
        if (serverSizes[server] != expectedSize) {
            return false;
        }
    }
    return true;
}

HcclResult BuildRelayPlan(HcclComm comm, const OpParam &param, RelayPlan &plan)
{
    std::fill_n(plan.helperFor, MAX_RANK_SIZE, MAX_RANK_SIZE);
    std::fill_n(plan.targetFor, MAX_RANK_SIZE, MAX_RANK_SIZE);
    if (param.count * sizeof(float) < 8ULL * 1024ULL * 1024ULL ||
        param.count * sizeof(float) > MAX_DATA_SIZE ||
        (param.rankSize != 16 && param.rankSize != 12)) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 12) {
        // Twelve ranks can also be 8+4. Only enable this route for four
        // consecutive three-rank servers, using communicator-wide sizes.
        uint32_t *serverSizes = nullptr;
        uint32_t serverCount = 0;
        CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &serverSizes, &serverCount));
        if (serverCount != 4) {
            return HCCL_SUCCESS;
        }
        CHK_PTR_NULL(serverSizes);
        for (uint32_t server = 0; server < serverCount; ++server) {
            if (serverSizes[server] != 3) {
                return HCCL_SUCCESS;
            }
        }
        const uint32_t serverBase = (param.root / 3U) * 3U;
        const uint32_t targetBase = ((param.root / 3U + 1U) % 4U) * 3U;
        for (uint32_t slot = 0; slot < 3U; ++slot) {
            const uint32_t helper = serverBase + slot;
            if (helper == param.root) {
                continue;
            }
            const uint32_t target = targetBase + slot;
            plan.targetFor[helper] = target;
            plan.helperFor[target] = helper;
        }
        // With prefix-ready helper release, balance (9-2f)/4 against 1+f.
        // The resulting 5/6 ratio hides the second hop behind the helper's
        // own output receive; the old 5/7 ratio was balanced for the
        // whole-block completion schedule.
        plan.numerator = 5;
        plan.denominator = 6;
        plan.relayPrefixFirst = true;
        return HCCL_SUCCESS;
    }
    // 2x8 uses consecutive rank slots on each server. Derive the same
    // matched channel graph on every rank without querying the root's
    // local Mesh links from another rank's process.
    const uint32_t serverBase = (param.root / 8U) * 8U;
    for (uint32_t slot = 0; slot < 8U; ++slot) {
        const uint32_t helper = serverBase + slot;
        if (helper == param.root) {
            continue;
        }
        const uint32_t target = helper ^ 8U;
        plan.targetFor[helper] = target;
        plan.helperFor[target] = helper;
    }
    // Candidate C combines 2x8 OwnCopy overlap with the prefix-ready ratio
    // retune validated independently by candidate A.
    plan.numerator = 4;
    plan.denominator = 11;
    plan.relayPrefixFirst = true;
    return HCCL_SUCCESS;
}

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS: return HCCL_SUCCESS;
        case CCU_E_PARA: return HCCL_E_PARA;
        case CCU_E_PTR: return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT: return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND: return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL: return HCCL_E_UNAVAIL;
        default: return HCCL_E_INTERNAL;
    }
}

HcclResult BuildChannelDesc(HcclComm comm, uint32_t myRank, uint32_t peer, HcclChannelDesc &desc)
{
    for (uint32_t layer = 0; layer <= 1; ++layer) {
        CommLink *links = nullptr;
        uint32_t count = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layer, myRank, peer, &links, &count);
        if (ret != HCCL_SUCCESS || links == nullptr) {
            continue;
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = peer;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = links[i].linkAttr.linkProtocol;
            desc.localEndpoint.protocol = links[i].srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = links[i].srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = links[i].srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = links[i].dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = links[i].dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = links[i].dstEndpointDesc.loc;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("[Scatter] no UBC_CTP link between rank %u and rank %u", myRank, peer);
    return HCCL_E_NOT_FOUND;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param,
    const std::map<uint32_t, std::vector<PeerChannel>> &groups,
    const RelayPlan &plan, bool receiverPull, AlgResourceCtx &ctx)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("[Scatter] expected one CCU instance, got %u", insNum),
        HCCL_E_INTERNAL);
    CcuResult result = HcommCcuKernelRegisterStart(insHandle);
    if (result != CCU_SUCCESS) {
        return ConvertCcuResult(result);
    }
    for (const auto &group : groups) {
        auto arg = std::make_shared<CcuKernelArgScatter>();
        arg->myRank = param.myRank;
        arg->root = param.root;
        arg->rankSize = param.rankSize;
        arg->relayHelper = plan.helperFor[param.myRank];
        arg->relayTarget = plan.targetFor[param.myRank];
        arg->handleOwnCopy = ctx.overlapOwnCopy != 0 && group.first == groups.begin()->first;
        arg->relayPrefixFirst = plan.relayPrefixFirst;
        arg->relayTilePipeline = false;
        arg->receiverPull = receiverPull;
        arg->channelCount = static_cast<uint32_t>(group.second.size());
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            arg->channels[i] = group.second[i].handle;
            arg->peerRanks[i] = group.second[i].rank;
            arg->relayTargets[i] = plan.targetFor[group.second[i].rank];
            arg->relayHelpers[i] = plan.helperFor[group.second[i].rank];
        }
        const void *args[] = {arg.get()};
        CcuKernelHandle handle = 0;
        const char *kernelName = receiverPull ?
            (group.first == 0 ? "CcuScatterPullDie0" : "CcuScatterPullDie1") :
            plan.numerator == 0 ?
            (group.first == 0 ? "CcuScatterRootPushDie0" : "CcuScatterRootPushDie1") :
            (group.first == 0 ? "CcuScatterMeshRelayDie0" : "CcuScatterMeshRelayDie1");
        void *kernel = reinterpret_cast<void *>(receiverPull ? ops_hccl::CcuKernelPull : plan.numerator == 0 ?
            ops_hccl::CcuKernelDirect : ops_hccl::CcuKernel);
        result = HcommCcuKernelRegister(insHandle, group.first, kernelName,
            kernel, args, 1, &handle);
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("[Scatter] CCU kernel registration failed on die %u: %d", group.first, result);
            return ConvertCcuResult(result);
        }
        ctx.ccuKernels.push_back(handle);
        uint32_t role = 0;
        bool hasRoot = false;
        if (plan.targetFor[param.myRank] < param.rankSize) {
            for (const PeerChannel &peer : group.second) {
                hasRoot |= peer.rank == param.root;
                if (peer.rank == plan.targetFor[param.myRank]) {
                    role = 1;
                }
            }
        }
        ctx.kernelRoles.push_back(hasRoot ? 0 : role);
    }
    result = HcommCcuKernelRegisterEnd(insHandle);
    return ConvertCcuResult(result);
}

HcclResult CreateResources(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx)
{
    RelayPlan plan;
    CHK_RET(BuildRelayPlan(comm, param, plan));
    const bool receiverPull = plan.numerator == 0 && ReceiverPullCandidate(comm, param);
    ctx.relayHelper = plan.helperFor[param.myRank];
    ctx.relayTarget = plan.targetFor[param.myRank];
    ctx.relayNumerator = plan.numerator;
    ctx.relayDenominator = plan.denominator;
    if (OwnCopyCandidate(param, receiverPull)) {
        if (receiverPull || plan.numerator != 0) {
            // Exactly one root kernel overlaps a disjoint own-slice copy.
            ctx.overlapOwnCopy = 1;
        } else {
            uint32_t *sizes = nullptr;
            uint32_t count = 0;
            CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &sizes, &count));
            if (sizes != nullptr) {
                const bool fourByOne = param.rankSize == 4 && count == 4 &&
                    sizes[0] == 1 && sizes[1] == 1 && sizes[2] == 1 && sizes[3] == 1;
                const bool eightPlusFour = param.rankSize == 12 && count == 2 &&
                    ((sizes[0] == 8 && sizes[1] == 4) || (sizes[0] == 4 && sizes[1] == 8));
                ctx.overlapOwnCopy = fourByOne || eightPlusFour;
            }
        }
    }
    void *buffer = nullptr;
    uint64_t bufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &buffer, &bufferSize));
    ctx.localBuffer = CommBuffer{buffer, bufferSize};
    ctx.ccuThread = param.cpuThread;
    ctx.threads.push_back(param.cpuThread);

    std::map<uint32_t, std::vector<PeerChannel>> groups;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank || (param.myRank != param.root && peer != param.root)) {
            continue;
        }
        HcclChannelDesc desc{};
        CHK_RET(BuildChannelDesc(comm, param.myRank, peer, desc));
        EndpointAttrDieId endpointDie = 0;
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint,
            ENDPOINT_ATTR_DIE_ID, sizeof(endpointDie), &endpointDie));
        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        groups[static_cast<uint32_t>(endpointDie)].push_back(PeerChannel{peer, channel});
    }
    if (plan.numerator != 0 && param.myRank != param.root) {
        const uint32_t peer = plan.helperFor[param.myRank] < param.rankSize ?
            plan.helperFor[param.myRank] : plan.targetFor[param.myRank];
        if (peer < param.rankSize) {
            HcclChannelDesc desc{};
            CHK_RET(BuildChannelDesc(comm, param.myRank, peer, desc));
            EndpointAttrDieId endpointDie = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint,
                ENDPOINT_ATTR_DIE_ID, sizeof(endpointDie), &endpointDie));
            ChannelHandle channel = 0;
            CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
            groups[static_cast<uint32_t>(endpointDie)].push_back(PeerChannel{peer, channel});
        }
    }
    CHK_PRT_RET(groups.empty() || groups.size() > 2,
        HCCL_ERROR("[Scatter] invalid number of CCU die groups: %zu", groups.size()), HCCL_E_INTERNAL);
    if (groups.size() == 2) {
        ThreadHandle worker = 0;
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, 1,
            plan.numerator == 0 ? 1U : 2U, &worker));
        ctx.threads.push_back(worker);
    }
    return RegisterKernels(comm, param, groups, plan, receiverPull, ctx);
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || root >= param.rankSize,
        HCCL_ERROR("[Scatter] invalid rank size %u or root %u", param.rankSize, root), HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[Scatter] unsupported data type %d", dataType), HCCL_E_NOT_SUPPORT);
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(recvBuf);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() /
        (static_cast<uint64_t>(param.rankSize) * sizeof(float)),
        HCCL_ERROR("[Scatter] input byte count overflow"), HCCL_E_PARA);
    const bool relayCandidate = recvCount * sizeof(float) >= 8ULL * 1024ULL * 1024ULL &&
        recvCount * sizeof(float) <= MAX_DATA_SIZE &&
        (param.rankSize == 16 || param.rankSize == 12);
    // The size class separates pull from push even when relay=0. Topology
    // is fixed per communicator; the copy bit also separates disjoint and
    // overlapping buffers when addresses change between calls.
    const bool receiverPullSizeCandidate = ReceiverPullSizeCandidate(param);
    std::snprintf(param.tag, sizeof(param.tag),
        "hccl_scatter_v4h_root_%u_relay_%u_copy_%u_pull_%u",
        root, static_cast<uint32_t>(relayCandidate),
        static_cast<uint32_t>(OwnCopyCandidate(param, receiverPullSizeCandidate)),
        static_cast<uint32_t>(receiverPullSizeCandidate));

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream,
        relayCandidate ? 2U : 1U, &param.cpuThread));
    if (param.rankSize == 1) {
        if (sendBuf != recvBuf) {
            const uint64_t bytes = recvCount * sizeof(float);
            for (uint64_t offset = 0; offset < bytes;) {
                const uint64_t chunk = std::min<uint64_t>(MAX_DATA_SIZE, bytes - offset);
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(param.cpuThread,
                    static_cast<uint8_t *>(recvBuf) + offset,
                    static_cast<const uint8_t *>(sendBuf) + offset, chunk)));
                offset += chunk;
            }
        }
        return HCCL_SUCCESS;
    }
    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resources;
        CHK_RET(CreateResources(comm, param, resources));
        std::vector<char> serialized = resources.Serialize();
        param.ctxSize = serialized.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, serialized.data(), serialized.size(), 0));
    }
    return ops_hccl::ExecOp(param);
}
