/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>
#include <unistd.h>

#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_kernel.h"
#include "ccu_launch.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
constexpr CommProtocol REQUIRED_PROTOCOL = CommProtocol::COMM_PROTOCOL_UBC_CTP;
constexpr const char *LOG_TAG = "SCATTER_CCU_FINAL";

struct ChannelGroup {
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> indices;
};

HcclResult MakeRelayPlan(HcclComm comm, const OpParam &param, RelayPlan &plan)
{
    if (param.count * sizeof(float) < 1024U * 1024U || param.rankSize == 4U) {
        return HCCL_SUCCESS;
    }
    uint32_t *localRanks = nullptr;
    uint32_t localCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &localRanks, &localCount));
    uint32_t rootBegin = 0;
    uint32_t rootCount = 0;
    // The four documented subcommunicators number ranks server by server.
    // Layer-0 queries only expose the caller's server, not a remote root's.
    if (param.rankSize == 16U && localCount == 8U) {
        rootCount = 8U;
        rootBegin = param.root / 8U * 8U;
    } else if (param.rankSize == 12U && localCount == 3U) {
        rootCount = 3U;
        rootBegin = param.root / 3U * 3U;
    } else if (param.rankSize == 12U && (localCount == 8U || localCount == 4U)) {
        rootCount = param.root < 8U ? 8U : 4U;
        rootBegin = param.root < 8U ? 0U : 8U;
    } else {
        return HCCL_SUCCESS;
    }
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.root) {
            continue;
        }
        const bool local = peer >= rootBegin && peer < rootBegin + rootCount;
        (local ? plan.relays : plan.remotes).push_back(peer);
    }
    // Only offload when the Clos egress is slower than the root's local links.
    if (plan.relays.empty() || plan.remotes.size() <= 4U) {
        plan.relays.clear();
        plan.remotes.clear();
        return HCCL_SUCCESS;
    }
    plan.enabled = 1;
    plan.persistent = param.count * sizeof(float) <= MAX_DATA_SIZE ? 1U : 0U;
    const uint64_t coarseLimit = param.rankSize == 16U ? COARSE_SLICE_LIMIT : EXTENDED_COARSE_SLICE_LIMIT;
    if (param.count * sizeof(float) <= coarseLimit) {
        // Query the communicator buffer on every rank so the scheduling mode
        // is chosen from the same communicator-wide capacity configuration.
        void *buffer = nullptr;
        uint64_t capacity = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &buffer, &capacity));
        // The total offloaded tails per relay are less than one slice.
        // Require a full coarseLimit buffer consistently on every rank.
        plan.coarse = capacity >= coarseLimit ? 1U : 0U;
    }
    if (param.myRank == param.root) {
        return HCCL_SUCCESS;
    }
    plan.role = 2;
    for (uint32_t i = 0; i < plan.remotes.size(); ++i) {
        if (plan.remotes[i] == param.myRank) {
            plan.index = i;
        }
    }
    for (uint32_t i = 0; i < plan.relays.size(); ++i) {
        if (plan.relays[i] == param.myRank) {
            plan.role = 1;
            plan.index = i;
            void *scratch = nullptr;
            uint64_t bytes = 0;
            CHK_RET(HcclGetHcclBuffer(comm, &scratch, &bytes));
            CHK_PRT_RET(bytes < 2U * RELAY_BANK_BYTES, HCCL_ERROR("relay scratch too small"), HCCL_E_MEMORY);
            plan.scratch = reinterpret_cast<uint64_t>(scratch);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult FindLink(HcclComm comm, uint32_t localRank, uint32_t remoteRank, CommLink &selectedLink)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layers[layerIndex], localRank, remoteRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr) {
            continue;
        }
        for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
            if (links[linkIndex].linkAttr.linkProtocol == REQUIRED_PROTOCOL) {
                selectedLink = links[linkIndex];
                return HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("[%s] no UBC_CTP link between rank %u and rank %u", LOG_TAG, localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult BuildChannelDesc(
    HcclComm comm, const OpParam &param, uint32_t peer, HcclChannelDesc &desc, uint32_t &ioDieId)
{
    CommLink link{};
    CHK_RET(FindLink(comm, param.myRank, peer, link));
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = peer;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint = link.srcEndpointDesc;
    desc.remoteEndpoint = link.dstEndpointDesc;
    CHK_RET(HcclRankGraphGetEndpointInfo(
        comm, param.myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(ioDieId), &ioDieId));
    CHK_PRT_RET(ioDieId >= IO_DIE_NUM, HCCL_ERROR("[%s] invalid IO Die %u for peer %u", LOG_TAG, ioDieId, peer),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(
    HcclComm comm, const OpParam &param, const std::array<ChannelGroup, IO_DIE_NUM> &groups,
    const std::array<ChannelGroup, IO_DIE_NUM> &forward, AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle{};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("[%s] expected one CCU instance, got %u", LOG_TAG, insNum), HCCL_E_INTERNAL);

    static std::vector<std::shared_ptr<ScatterCcuKernelArg>> kernelArgKeeper;
    bool localCopyAssigned = false;
    uint32_t largestPushGroup = 0;
    std::vector<uint32_t> rootLocalRanks;
    if (resCtx.directPush != 0U) {
        uint32_t *ranks = nullptr;
        uint32_t count = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &ranks, &count));
        if (count != 0U) {
            rootLocalRanks.assign(ranks, ranks + count);
        }
    }
    const bool asymmetricDirect = resCtx.directPush != 0U && param.rankSize == 12U
        && param.count * sizeof(float) <= MAX_DATA_SIZE
        && (rootLocalRanks.size() == 8U || rootLocalRanks.size() == 4U);
    uint32_t copyDie = INVALID_IO_DIE;
    if (param.myRank == param.root && param.rankSize != 16U
        && (asymmetricDirect || resCtx.relay.coarse != 0U)) {
        uint32_t leastPush = MAX_RANK_SIZE;
        size_t leastPeers = MAX_RANK_SIZE;
        for (uint32_t die = 0; die < IO_DIE_NUM; ++die) {
            if (groups[die].channels.empty()) { continue; }
            uint32_t pushCount = 0;
            for (uint32_t peer : groups[die].indices) {
                const bool pushed = asymmetricDirect
                    ? std::find(rootLocalRanks.begin(), rootLocalRanks.end(), peer) == rootLocalRanks.end()
                    : std::find(resCtx.relay.remotes.begin(), resCtx.relay.remotes.end(), peer)
                        != resCtx.relay.remotes.end();
                pushCount += pushed ? 1U : 0U;
            }
            if (pushCount < leastPush || (pushCount == leastPush && groups[die].channels.size() < leastPeers)) {
                copyDie = die;
                leastPush = pushCount;
                leastPeers = groups[die].channels.size();
            }
        }
    }
    for (uint32_t die = 0; die < IO_DIE_NUM; ++die) {
        if (groups[die].channels.empty()) {
            continue;
        }
        auto kernelArg = std::make_shared<ScatterCcuKernelArg>();
        kernelArg->legacySchedule = param.rankSize == 16U ? 1U : 0U;
        kernelArg->persistent = resCtx.relay.persistent;
        // v16: plain small pulls no longer owe the root a completion receipt;
        // the framework's per-round rendezvous protects the input lifetime.
        kernelArg->noRootAck = resCtx.directPush == 0U && resCtx.relay.enabled == 0U
            && param.count * sizeof(float) < 1024U * 1024U ? 1U : 0U;
        kernelArg->isRoot = param.myRank == param.root ? 1U : 0U;
        kernelArg->copyLocalSlice = kernelArg->isRoot != 0U
            && (copyDie != INVALID_IO_DIE ? die == copyDie : !localCopyAssigned) ? 1U : 0U;
        if (kernelArg->copyLocalSlice != 0U) {
            if (param.rankSize != 16U) {
                resCtx.directPullMask |= static_cast<uint32_t>(resCtx.ccuKernels.size()) << ROOT_COPY_INDEX_SHIFT;
            }
        }
        if (param.myRank != param.root) {
            resCtx.relay.receiveDie = die;
        }
        if (resCtx.relay.enabled != 0U && resCtx.relay.role == 2U) {
            kernelArg->receiveRelayCount = forward[die].channels.size();
            std::copy(forward[die].channels.begin(), forward[die].channels.end(), kernelArg->receiveRelayChannels);
            std::copy(forward[die].indices.begin(), forward[die].indices.end(), kernelArg->receiveRelayIndices);
            resCtx.relay.fusedIndices = forward[die].indices;
            kernelArg->remoteIndex = resCtx.relay.index;
            // A resident mission can own the complete receive schedule when
            // every source channel belongs to this die. Mixed-die ranks keep
            // the independently ordered two-queue path.
            resCtx.relay.residentReceiver = resCtx.relay.persistent != 0U && resCtx.relay.coarse == 0U
                && forward[die].channels.size() == resCtx.relay.relays.size() ? 1U : 0U;
        }
        if (resCtx.relay.enabled != 0U && resCtx.relay.role == 1U) {
            kernelArg->gatherCount = resCtx.relay.remotes.size();
            std::copy(resCtx.relay.remotes.begin(), resCtx.relay.remotes.end(), kernelArg->gatherRanks);
        }
        kernelArg->channelCount = static_cast<uint32_t>(groups[die].channels.size());
        for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
            kernelArg->channels[index] = groups[die].channels[index];
            kernelArg->peerRanks[index] = groups[die].indices[index];
            if (kernelArg->isRoot != 0U && resCtx.relay.coarse != 0U
                && std::find(resCtx.relay.remotes.begin(), resCtx.relay.remotes.end(),
                       groups[die].indices[index]) != resCtx.relay.remotes.end()) {
                kernelArg->pushMask |= 1U << index;
            }
            if (kernelArg->isRoot != 0U && asymmetricDirect
                && std::find(rootLocalRanks.begin(), rootLocalRanks.end(), groups[die].indices[index])
                    == rootLocalRanks.end()) {
                kernelArg->pushMask |= 1U << index;
            }
        }
        uint32_t pushCount = static_cast<uint32_t>(__builtin_popcount(kernelArg->pushMask));
        if (param.myRank == param.root && resCtx.directPush != 0U && !asymmetricDirect && !rootLocalRanks.empty()) {
            for (uint32_t peer : groups[die].indices) {
                if (std::find(rootLocalRanks.begin(), rootLocalRanks.end(), peer) == rootLocalRanks.end()) {
                    ++pushCount;
                }
            }
        }
        if (pushCount > largestPushGroup) {
            largestPushGroup = pushCount;
            resCtx.rootMainIndex = resCtx.ccuKernels.size();
        }

        char kernelName[64]{};
        int written = std::snprintf(kernelName, sizeof(kernelName), "CcuScatterPull_d%u_r%u", die, param.root);
        CHK_PRT_RET(written <= 0 || static_cast<size_t>(written) >= sizeof(kernelName),
            HCCL_ERROR("[%s] failed to build kernel name", LOG_TAG), HCCL_E_INTERNAL);
        const void *kernelArgs[] = {kernelArg.get()};
        CcuKernelHandle handle{};
        void *entry = resCtx.directPush != 0U ? reinterpret_cast<void *>(ops_hccl::CcuScatterDirectPush)
                                             : reinterpret_cast<void *>(ops_hccl::CcuScatterKernel);
        // The 8+4 direct route has independent Mesh links. Let local receivers
        // issue their Reads concurrently after a compact capability exchange;
        // retain root-driven Writes on the shared Clos egress.
        const bool sameServerGroup = asymmetricDirect && std::all_of(groups[die].indices.begin(),
            groups[die].indices.end(), [&](uint32_t peer) {
                return std::find(rootLocalRanks.begin(), rootLocalRanks.end(), peer) != rootLocalRanks.end();
            });
        if (asymmetricDirect && param.myRank == param.root) {
            // Mesh and Clos channels can share one IO Die. Select the protocol
            // per peer; never infer a peer's protocol from its receiver die.
            resCtx.directPullMask |= 1U << resCtx.ccuKernels.size();
            entry = reinterpret_cast<void *>(ops_hccl::CcuScatterHybridRoot);
        } else if (sameServerGroup) {
            resCtx.directPullMask |= 1U << resCtx.ccuKernels.size();
            entry = reinterpret_cast<void *>(ops_hccl::CcuScatterKernel);
        }
        if (resCtx.relay.residentReceiver != 0U) {
            entry = reinterpret_cast<void *>(ops_hccl::CcuScatterResidentReceiver);
        }
        if (resCtx.relay.coarse != 0U && resCtx.relay.role == 1U) {
            entry = reinterpret_cast<void *>(ops_hccl::CcuScatterCoarseRelay);
        }
        if (resCtx.relay.coarse != 0U && kernelArg->isRoot != 0U) {
            entry = reinterpret_cast<void *>(ops_hccl::CcuScatterHybridRoot);
        } else if (resCtx.relay.coarse != 0U && resCtx.relay.role == 2U) {
            entry = reinterpret_cast<void *>(ops_hccl::CcuScatterHybridReceiver);
        }
        CcuResult ret = HcommCcuKernelRegister(insHandle, die, kernelName, entry, kernelArgs, 1, &handle);
        CHK_PRT_RET(ret != CCU_SUCCESS,
            HCCL_ERROR("[%s] register failed on die %u with %u channels, ret %d", LOG_TAG, die, kernelArg->channelCount,
                static_cast<int32_t>(ret)),
            ConvertCcuToHccl(ret));
        resCtx.ccuKernels.push_back(handle);
        resCtx.directPeers.push_back(groups[die].indices);
        resCtx.directWriteMasks.push_back(kernelArg->pushMask);
        kernelArgKeeper.push_back(kernelArg);
        localCopyAssigned = localCopyAssigned || kernelArg->copyLocalSlice != 0U;
    }
    CHK_PRT_RET(resCtx.ccuKernels.empty(), HCCL_ERROR("[%s] no CCU kernel registered", LOG_TAG), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RegisterForward(HcclComm comm, AlgResourceCtx &ctx, const std::array<ChannelGroup, IO_DIE_NUM> &groups)
{
    if (ctx.relay.enabled == 0U || ctx.relay.role == 0U) {
        return HCCL_SUCCESS;
    }
    CcuInsHandle ins{};
    uint32_t count = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &ins, &count));
    static std::vector<std::shared_ptr<ScatterCcuKernelArg>> keeper;
    for (uint32_t die = 0; die < IO_DIE_NUM; ++die) {
        if (groups[die].channels.empty() || (ctx.relay.role == 2U && die == ctx.relay.receiveDie)) {
            continue;
        }
        auto arg = std::make_shared<ScatterCcuKernelArg>();
        arg->legacySchedule = ctx.relay.relays.size() == 7U && ctx.relay.remotes.size() == 8U ? 1U : 0U;
        arg->isRoot = ctx.relay.role == 1U ? 1U : 0U;
        arg->channelCount = groups[die].channels.size();
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            arg->channels[i] = groups[die].channels[i];
        }
        const void *args[] = {arg.get()};
        CcuKernelHandle handle{};
        void *fn = ctx.relay.role == 1U ? reinterpret_cast<void *>(ops_hccl::CcuScatterKernel)
                                        : reinterpret_cast<void *>(ops_hccl::CcuScatterRelayRead);
        CHK_RET_CCU(HcommCcuKernelRegister(ins, die, "ScatterForward", fn, args, 1, &handle));
        ctx.relay.forwardKernels.push_back(handle);
        ctx.relay.forwardIndices.push_back(groups[die].indices);
        // Keep one ordered queue per IO Die. A different die can forward the
        // previous bank while the receiving die gathers the following tile.
        const bool onMain = die == ctx.relay.receiveDie && ctx.relay.coarse == 0U;
        ThreadHandle thread{};
        if (!onMain) {
            CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, 1, 2, &thread));
            ctx.relay.pipeline = 1U;
        }
        ctx.relay.forwardThreads.push_back(thread);
        ctx.relay.forwardOnMain.push_back(onMain ? 1U : 0U);
        keeper.push_back(arg);
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireAll(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx,
    std::array<ChannelGroup, IO_DIE_NUM> &direct, std::array<ChannelGroup, IO_DIE_NUM> &forward)
{
    std::vector<HcclChannelDesc> descs;
    std::vector<uint32_t> dies, indices, stages;
    const auto &extra = ctx.relay.role == 1U ? ctx.relay.remotes : ctx.relay.relays;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        const bool isDirect = param.myRank == param.root || peer == param.root;
        const auto found = std::find(extra.begin(), extra.end(), peer);
        if (!isDirect && (ctx.relay.enabled == 0U || found == extra.end())) {
            continue;
        }
        HcclChannelDesc desc{};
        uint32_t die = INVALID_IO_DIE;
        CHK_RET(BuildChannelDesc(comm, param, peer, desc, die));
        descs.push_back(desc);
        dies.push_back(die);
        stages.push_back(isDirect ? 0U : 1U);
        indices.push_back(isDirect ? peer : static_cast<uint32_t>(found - extra.begin()));
    }
    std::vector<ChannelHandle> handles(descs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, descs.data(), descs.size(), handles.data()));
    for (size_t i = 0; i < handles.size(); ++i) {
        auto &group = stages[i] == 0U ? direct[dies[i]] : forward[dies[i]];
        group.channels.push_back(handles[i]);
        group.indices.push_back(indices[i]);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(
        dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("[%s] only float32 is supported", LOG_TAG), HCCL_E_NOT_SUPPORT);

    OpParam param{};
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || root >= param.rankSize,
        HCCL_ERROR("[%s] invalid rankSize %u or root %u", LOG_TAG, param.rankSize, root), HCCL_E_PARA);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize,
        HCCL_ERROR("[%s] recvCount is too large: %lu", LOG_TAG, recvCount), HCCL_E_PARA);

    const uint64_t sliceBytes = recvCount * sizeof(float);
    const uint64_t coarseLimit = param.rankSize == 16U ? COARSE_SLICE_LIMIT : EXTENDED_COARSE_SLICE_LIMIT;
    const uint32_t sizeClass = sliceBytes > MAX_DATA_SIZE ? 3U :
        (sliceBytes > coarseLimit ? 2U : (sliceBytes >= 1024U * 1024U ? 1U : 0U));
    // v17: sizeClass 0 pulls additionally leave the reader's transfer event
    // unwaited, so they need a fresh resource identity again.
    // v18: large non-16-rank roots (DirectPush and asymmetric HybridRoot)
    // likewise leave their writes in flight, so the identity moves to v25.
    const char *tagFormat = sizeClass == 0U ? "scatter_smallpull_v17_r%u_s%u"
        : (param.rankSize == 16U ? "scatter_optimized_v8_r%u_s%u" : "scatter_cursor_v25_r%u_s%u");
    int tagRet = std::snprintf(param.tag, sizeof(param.tag), tagFormat, root, sizeClass);
    CHK_PRT_RET(tagRet <= 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("[%s] failed to build resource tag", LOG_TAG), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    HcclDfxOpInfo dfxInfo{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;
    for (uint32_t retry = 0; retry < 300U; ++retry) {
        CHK_RET(HcclCommGetStatus(commName, &commStatus));
        if (commStatus == HCCL_COMM_STATUS_READY) {
            break;
        }
        usleep(100U * 1000U);
    }
    CHK_PRT_RET(commStatus != HCCL_COMM_STATUS_READY,
        HCCL_ERROR("[%s] communicator is not ready, status %d", LOG_TAG, static_cast<int32_t>(commStatus)),
        HCCL_E_INTERNAL);

    constexpr CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    const uint32_t mainNotifyNum = param.rankSize != 4U ? 2U * IO_DIE_NUM : 0U;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream, mainNotifyNum, &param.cpuThread));
    if (HcclEngineCtxGet(comm, param.tag, engine, &param.resCtx, &param.ctxSize) != HCCL_SUCCESS) {
        AlgResourceCtx resCtx;
        CHK_RET(MakeRelayPlan(comm, param, resCtx.relay));
        resCtx.directPush = resCtx.relay.enabled == 0U && recvCount * sizeof(float) >= 1024U * 1024U ? 1U : 0U;
        if (param.rankSize > 1U) {
            std::array<ChannelGroup, IO_DIE_NUM> groups{};
            std::array<ChannelGroup, IO_DIE_NUM> forwardGroups{};
            CHK_RET(AcquireAll(comm, param, resCtx, groups, forwardGroups));
            if (param.myRank == param.root && !groups[0].channels.empty() && !groups[1].channels.empty()) {
                CHK_RET(HcclThreadAcquire(comm, engine, 1, 1, &resCtx.workerThread));
            }
            CcuInsHandle ins{};
            uint32_t insCount = 0;
            CHK_RET(HcclCommQueryCcuIns(comm, &ins, &insCount));
            CHK_RET_CCU(HcommCcuKernelRegisterStart(ins));
            CHK_RET(RegisterKernels(comm, param, groups, forwardGroups, resCtx));
            CHK_RET(RegisterForward(comm, resCtx, forwardGroups));
            CHK_RET_CCU(HcommCcuKernelRegisterEnd(ins));
        }
        std::vector<char> serialized = resCtx.Serialize();
        CHK_PRT_RET(serialized.empty(), HCCL_ERROR("resource cache exceeds supported bounds"), HCCL_E_INTERNAL);
        param.ctxSize = serialized.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, serialized.data(), serialized.size(), 0));
    }
    return param.rankSize == 16U ? ops_hccl::ExecOpLegacy16(param) : ops_hccl::ExecOp(param);
}
