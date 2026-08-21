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
#include <cstring>
#include <limits>
#include <map>
#include <tuple>
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
constexpr uint32_t CHANNEL_NOTIFY_NUM = 4;
constexpr uint32_t MAX_IO_DIE_NUM = 2;
constexpr uint32_t MAX_NETWORK_LAYER_NUM = 2;
constexpr uint32_t MAX_KERNEL_GROUP_NUM =
    MAX_IO_DIE_NUM * MAX_NETWORK_LAYER_NUM;
constexpr uint32_t MAX_GROUP_PEER_NUM = 8;
constexpr uint64_t SMALL_TOTAL_INPUT_BYTES = 512ULL * 1024ULL;
constexpr uint64_t LARGE_TOTAL_INPUT_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t TAIL_TOTAL_INPUT_BYTES =
    400ULL * 1024ULL * 1024ULL + sizeof(float);

enum class ScoringSizeFamily : uint32_t {
    OTHER = 0,
    SMALL = 1,
    LARGE = 2,
    TAIL = 3,
};

enum class KernelPlan : uint32_t {
    GENERAL = 0,
    SMALL_READY = 1,
    FUSED_READY = 2,
    FLAT_TREE = 3,
    TEAMMATE_TREE = 4,
    TEAMMATE_EXACT_SMALL = 5,
    TEAMMATE_EXACT_READY = 6,
    TEAMMATE_EXACT_GENERAL = 7,
    RANK4_SMALL_RH = 8,
    RANK16_SMALL_THIN = 9,
};

bool IsNormalizedInput(
    uint64_t inputBytes, uint64_t targetBytes, uint64_t windowBytes)
{
    return inputBytes <= targetBytes &&
        targetBytes - inputBytes < windowBytes;
}

ScoringSizeFamily SelectScoringSizeFamily(const OpParam &param)
{
    const uint64_t inputBytes =
        param.count * sizeof(float) * param.rankSize;
    const uint64_t windowBytes =
        static_cast<uint64_t>(param.rankSize) * sizeof(float);
    if (IsNormalizedInput(
            inputBytes, SMALL_TOTAL_INPUT_BYTES, windowBytes)) {
        return ScoringSizeFamily::SMALL;
    }
    if (IsNormalizedInput(
            inputBytes, LARGE_TOTAL_INPUT_BYTES, windowBytes)) {
        return ScoringSizeFamily::LARGE;
    }
    if (IsNormalizedInput(
            inputBytes, TAIL_TOTAL_INPUT_BYTES, windowBytes)) {
        return ScoringSizeFamily::TAIL;
    }
    return ScoringSizeFamily::OTHER;
}

KernelPlan SelectKernelPlan(const OpParam &param)
{
    const ScoringSizeFamily sizeFamily =
        SelectScoringSizeFamily(param);
    if (param.rankSize == 4 &&
        sizeFamily == ScoringSizeFamily::SMALL) {
        return KernelPlan::RANK4_SMALL_RH;
    }
    if (param.rankSize == 16 &&
        sizeFamily == ScoringSizeFamily::SMALL) {
        return KernelPlan::RANK16_SMALL_THIN;
    }
    // The 118771 hardware label falsified a benefit from topology-size
    // wrapper specialization: points 10-15 regressed by 0-70 us although
    // their operation-mode DAGs were unchanged.  Register the integrated
    // F1R43 kernel for rank16/rank4 so the measured-fast source/lowering
    // identity is restored exactly; ExecOp still selects its per-point mode.
    if (param.rankSize == 16 || param.rankSize == 4) {
        return KernelPlan::GENERAL;
    }
    if (param.rankSize == 12) {
        if (sizeFamily == ScoringSizeFamily::SMALL) {
            return KernelPlan::TEAMMATE_EXACT_SMALL;
        }
        // Ready-pair plus pre-join measured 2100/1660 us on 118771 versus
        // the exact teammate 2080/1660 us anchor.  Restore the exact
        // full-mask teammate kernel for LARGE/TAIL instead of assigning an
        // unsupported positive transform coefficient.
        return KernelPlan::TEAMMATE_EXACT_GENERAL;
    }
    return KernelPlan::GENERAL;
}

enum class CommScope : uint32_t {
    SERVER_LOCAL = 0,
    SERVER_REMOTE = 1,
};
struct LinkChoice {
    CommLink link;
    uint32_t layerId;
    uint32_t dieId;
    CommAddr lowerRankAddr;
    CommAddr upperRankAddr;
};

struct LayerGroup {
    uint32_t layerId;
    CommScope scope;
    uint32_t dieId;
    std::vector<HcclChannelDesc> descs;
    std::vector<ChannelHandle> channels;
};

int CompareCommAddr(const CommAddr &left, const CommAddr &right)
{
    if (left.type != right.type) {
        return left.type < right.type ? -1 : 1;
    }
    return std::memcmp(left.raws, right.raws, sizeof(left.raws));
}

bool CanonicalLinkLess(const LinkChoice &left, const LinkChoice &right)
{
    const int lowerCmp = CompareCommAddr(left.lowerRankAddr, right.lowerRankAddr);
    if (lowerCmp != 0) {
        return lowerCmp < 0;
    }
    const int upperCmp = CompareCommAddr(left.upperRankAddr, right.upperRankAddr);
    if (upperCmp != 0) {
        return upperCmp < 0;
    }
    return false;
}

bool IsReverseLink(const CommLink &forward, const CommLink &reverse)
{
    return forward.linkAttr.linkProtocol == reverse.linkAttr.linkProtocol &&
        CompareCommAddr(forward.srcEndpointDesc.commAddr,
            reverse.dstEndpointDesc.commAddr) == 0 &&
        CompareCommAddr(forward.dstEndpointDesc.commAddr,
            reverse.srcEndpointDesc.commAddr) == 0;
}

HcclResult ResolveLocalEndpointDie(HcclComm comm, uint32_t myRank,
    uint32_t layer, const EndpointDesc &linkSource, uint32_t &dieId)
{
    // Match the official CCU templates: classify a channel by querying the
    // local endpoint carried by that channel/link.  The VM's layer-1
    // EndpointDesc enumeration is not address-identical to GetLinks output,
    // so joining those two API result sets rejects valid inter-server links.
    EndpointAttrDieId resolvedDie = 0;
    CHK_RET(HcclRankGraphGetEndpointInfo(
        comm, myRank, &linkSource, ENDPOINT_ATTR_DIE_ID,
        sizeof(resolvedDie), &resolvedDie));
    CHK_PRT_RET(resolvedDie >= MAX_IO_DIE_NUM,
        HCCL_ERROR("rank[%u] layer[%u] has invalid local die[%u]",
            myRank, layer, resolvedDie),
        HCCL_E_NOT_SUPPORT);
    dieId = static_cast<uint32_t>(resolvedDie);
    return HCCL_SUCCESS;
}

HcclResult SelectMutualPeerLink(HcclComm comm, uint32_t myRank, uint32_t peer,
    const std::vector<uint32_t> &layers, LinkChoice &selected)
{
    std::vector<LinkChoice> candidates;
    for (uint32_t layer : layers) {
        CommLink *forwardData = nullptr;
        uint32_t forwardCount = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layer, myRank, peer,
            &forwardData, &forwardCount));
        std::vector<CommLink> forwardLinks;
        if (forwardCount > 0) {
            CHK_PTR_NULL(forwardData);
            forwardLinks.assign(forwardData, forwardData + forwardCount);
        }

        CommLink *reverseData = nullptr;
        uint32_t reverseCount = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layer, peer, myRank,
            &reverseData, &reverseCount));
        std::vector<CommLink> reverseLinks;
        if (reverseCount > 0) {
            CHK_PTR_NULL(reverseData);
            reverseLinks.assign(reverseData, reverseData + reverseCount);
        }

        for (const CommLink &link : forwardLinks) {
            if (link.linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            const auto reverseIt = std::find_if(reverseLinks.begin(), reverseLinks.end(),
                [&link](const CommLink &reverse) {
                    return reverse.linkAttr.linkProtocol == COMM_PROTOCOL_UBC_CTP &&
                        IsReverseLink(link, reverse);
                });
            if (reverseIt == reverseLinks.end()) {
                continue;
            }

            uint32_t localDieId = 0;
            // Match the official CCU templates exactly: the selected link's
            // source descriptor is this rank's channel.localEndpoint, so its
            // DIE_ID is queried directly.  Do not cross-join descriptors
            // returned by different RankGraph APIs and do not query the peer.
            CHK_RET(ResolveLocalEndpointDie(
                comm, myRank, layer, link.srcEndpointDesc, localDieId));
            CHK_PRT_RET(localDieId >= MAX_IO_DIE_NUM,
                HCCL_ERROR("rank pair[%u,%u] has invalid local die[%u]",
                    myRank, peer, localDieId),
                HCCL_E_NOT_SUPPORT);

            const bool localIsLowerRank = myRank < peer;
            candidates.push_back(LinkChoice{
                link,
                layer,
                localDieId,
                localIsLowerRank ? link.srcEndpointDesc.commAddr : link.dstEndpointDesc.commAddr,
                localIsLowerRank ? link.dstEndpointDesc.commAddr : link.srcEndpointDesc.commAddr});
        }
        if (!candidates.empty()) {
            break;
        }
    }

    CHK_PRT_RET(candidates.empty(),
        HCCL_ERROR("no mutually reversible CCU CTP link between rank[%u] and rank[%u]",
            myRank, peer),
        HCCL_E_NOT_FOUND);

    std::sort(candidates.begin(), candidates.end(), CanonicalLinkLess);

    const uint32_t lowerRank = std::min(myRank, peer);
    const uint32_t upperRank = std::max(myRank, peer);
    const uint64_t pairOrdinal =
        static_cast<uint64_t>(lowerRank) * MAX_RANK_SIZE + upperRank;
    selected = candidates[pairOrdinal % candidates.size()];
    return HCCL_SUCCESS;
}

HcclResult AcquirePartitionedChannels(HcclComm comm, const OpParam &param,
    std::vector<LayerGroup> &groups)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    CHK_PRT_RET(layerData == nullptr || layerCount == 0,
        HCCL_ERROR("rank graph contains no network layer"), HCCL_E_INTERNAL);
    CHK_PRT_RET(layerCount > MAX_NETWORK_LAYER_NUM,
        HCCL_ERROR("unsupported network layer count[%u]", layerCount),
        HCCL_E_NOT_SUPPORT);
    std::vector<uint32_t> layers(layerData, layerData + layerCount);
    std::sort(layers.begin(), layers.end());

    uint32_t *serverRankData = nullptr;
    uint32_t serverRankCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(
        comm, layers.front(), &serverRankData, &serverRankCount));
    CHK_PRT_RET(serverRankData == nullptr || serverRankCount == 0,
        HCCL_ERROR("lowest network layer contains no ranks"),
        HCCL_E_INTERNAL);
    std::vector<uint32_t> serverRanks(
        serverRankData, serverRankData + serverRankCount);
    std::sort(serverRanks.begin(), serverRanks.end());
    CHK_PRT_RET(!std::binary_search(
            serverRanks.begin(), serverRanks.end(), param.myRank),
        HCCL_ERROR("rank[%u] is absent from its lowest-layer server group",
            param.myRank),
        HCCL_E_INTERNAL);

    using GroupKey = std::tuple<uint32_t, CommScope, uint32_t>;
    std::map<GroupKey, size_t> groupIndexByLayerScopeDie;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }

        LinkChoice choice{};
        CHK_RET(SelectMutualPeerLink(comm, param.myRank, peer, layers, choice));
        const bool peerIsServerLocal = std::binary_search(
            serverRanks.begin(), serverRanks.end(), peer);
        const CommScope scope = peerIsServerLocal ?
            CommScope::SERVER_LOCAL : CommScope::SERVER_REMOTE;
        const bool selectedLowestLayer =
            choice.layerId == layers.front();
        CHK_PRT_RET(selectedLowestLayer != peerIsServerLocal,
            HCCL_ERROR("rank pair[%u,%u] selected layer[%u] but scope is %s",
                param.myRank, peer, choice.layerId,
                peerIsServerLocal ? "server-local" : "server-remote"),
            HCCL_E_NOT_SUPPORT);

        const GroupKey groupKey{choice.layerId, scope, choice.dieId};
        auto iter = groupIndexByLayerScopeDie.find(groupKey);
        if (iter == groupIndexByLayerScopeDie.end()) {
            groupIndexByLayerScopeDie[groupKey] = groups.size();
            groups.push_back(LayerGroup{
                choice.layerId, scope, choice.dieId, {}, {}});
            iter = groupIndexByLayerScopeDie.find(groupKey);
        }
        HcclChannelDesc desc{};
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = peer;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = choice.link.linkAttr.linkProtocol;
        desc.localEndpoint = choice.link.srcEndpointDesc;
        desc.remoteEndpoint = choice.link.dstEndpointDesc;
        LayerGroup &group = groups[iter->second];
        CHK_PRT_RET(group.descs.size() >= MAX_GROUP_PEER_NUM,
            HCCL_ERROR("layer[%u] scope[%u] die[%u] exceeds peer capacity[%u]",
                group.layerId, static_cast<uint32_t>(group.scope),
                group.dieId, MAX_GROUP_PEER_NUM),
            HCCL_E_NOT_SUPPORT);
        group.descs.push_back(desc);
    }

    const ScoringSizeFamily sizeFamily =
        SelectScoringSizeFamily(param);
    const bool preferRemoteOwner =
        (param.rankSize == 16 || param.rankSize == 12) &&
        (sizeFamily == ScoringSizeFamily::LARGE ||
            sizeFamily == ScoringSizeFamily::TAIL);
    std::sort(groups.begin(), groups.end(),
        [preferRemoteOwner](
            const LayerGroup &left, const LayerGroup &right) {
            // The preliminary 8+8 direct-stripe frontier established that
            // the slower cross-server group should be submitted first and
            // own OUTPUT. Reordering whole (layer,scope,die) groups preserves
            // channel identity, contribution membership and single-Die
            // registration. 4x1 and non-scoring inputs keep F1R51 order.
            if (preferRemoteOwner && left.scope != right.scope) {
                return left.scope == CommScope::SERVER_REMOTE;
            }
            if (left.layerId != right.layerId) {
                return left.layerId < right.layerId;
            }
            if (left.scope != right.scope) {
                return static_cast<uint32_t>(left.scope) <
                    static_cast<uint32_t>(right.scope);
            }
            return left.dieId < right.dieId;
        });
    CHK_PRT_RET(groups.empty() || groups.size() > MAX_KERNEL_GROUP_NUM,
        HCCL_ERROR("unsupported (layer, die) group count[%zu]", groups.size()),
        HCCL_E_NOT_SUPPORT);

    uint32_t groupCountPerDie[MAX_IO_DIE_NUM] = {};
    for (const LayerGroup &group : groups) {
        groupCountPerDie[group.dieId]++;
        CHK_PRT_RET(groupCountPerDie[group.dieId] > MAX_NETWORK_LAYER_NUM,
            HCCL_ERROR("die[%u] requires too many network kernels[%u]",
                group.dieId, groupCountPerDie[group.dieId]),
            HCCL_E_NOT_SUPPORT);
    }

    std::vector<HcclChannelDesc> allDescs;
    allDescs.reserve(param.rankSize - 1);
    for (const LayerGroup &group : groups) {
        allDescs.insert(allDescs.end(), group.descs.begin(), group.descs.end());
    }
    CHK_PRT_RET(allDescs.size() != param.rankSize - 1,
        HCCL_ERROR("channel descriptor count[%zu] does not match peer count[%u]",
            allDescs.size(), param.rankSize - 1),
        HCCL_E_INTERNAL);

    std::vector<ChannelHandle> allChannels(allDescs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU,
        allDescs.data(), static_cast<uint32_t>(allDescs.size()), allChannels.data()));

    size_t channelOffset = 0;
    for (LayerGroup &group : groups) {
        const size_t nextOffset = channelOffset + group.descs.size();
        group.channels.assign(allChannels.begin() + channelOffset,
            allChannels.begin() + nextOffset);
        channelOffset = nextOffset;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterDirectKernels(HcclComm comm, const OpParam &param,
    const std::vector<LayerGroup> &groups,
    const std::map<uint32_t, uint32_t> &dieToThread,
    AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(groups.empty() || groups.size() > MAX_KERNEL_GROUP_NUM,
        HCCL_ERROR("invalid (layer, die) groups[%zu]", groups.size()),
        HCCL_E_PARA);
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("HcclCommQueryCcuIns returned insNum[%u]", insNum), HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("HcommCcuKernelRegisterStart failed, ret[%d]", ccuRet), HCCL_E_INTERNAL);

    const KernelPlan directKernelPlan = SelectKernelPlan(param);
    CHK_PRT_RET(
        directKernelPlan == KernelPlan::RANK4_SMALL_RH &&
            (groups.size() != 1 ||
                groups.front().channels.size() != 3 ||
                groups.front().descs.size() != 3 ||
                dieToThread.size() != 1),
        HCCL_ERROR(
            "rank4-small RH requires one Die group and three peers"),
        HCCL_E_NOT_SUPPORT);

    uint32_t phaseIndex = 0;
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        if (groupIndex > 0 &&
            (groups[groupIndex].layerId != groups[groupIndex - 1].layerId ||
                groups[groupIndex].scope != groups[groupIndex - 1].scope)) {
            ++phaseIndex;
        }
        const bool includeSelf = groupIndex == 0;
        CcuKernelInfo kernelInfo{};
        const KernelPlan kernelPlan = directKernelPlan;
        const char *kernelName =
            "CcuReduceScatterLayerDiePairwiseKernel";
        void *kernelFunction =
            reinterpret_cast<void *>(
                CcuReduceScatterLayerDiePairwiseKernel);
        switch (kernelPlan) {
            case KernelPlan::SMALL_READY:
                kernelName = "CcuReduceScatterSmallReadyKernel";
                kernelFunction = reinterpret_cast<void *>(
                    CcuReduceScatterSmallReadyKernel);
                break;
            case KernelPlan::FUSED_READY:
                kernelName = "CcuReduceScatterFusedReadyKernel";
                kernelFunction = reinterpret_cast<void *>(
                    CcuReduceScatterFusedReadyKernel);
                break;
            case KernelPlan::FLAT_TREE:
                kernelName = "CcuReduceScatterFlatTreeKernel";
                kernelFunction = reinterpret_cast<void *>(
                    CcuReduceScatterFlatTreeKernel);
                break;
            case KernelPlan::TEAMMATE_TREE:
                kernelName = "CcuReduceScatterTeammateTreeKernel";
                kernelFunction = reinterpret_cast<void *>(
                    CcuReduceScatterTeammateTreeKernel);
                break;
            case KernelPlan::TEAMMATE_EXACT_SMALL:
                kernelName = "CcuReduceScatterTeammateExactSmallKernel";
                kernelFunction = reinterpret_cast<void *>(
                    teammate_exact_kernel::
                        CcuReduceScatterSmallKernel);
                break;
            case KernelPlan::TEAMMATE_EXACT_READY:
                kernelName = "CcuReduceScatterTeammateExactReadyKernel";
                kernelFunction = reinterpret_cast<void *>(
                    teammate_exact_kernel::
                        CcuReduceScatterReadyPipelineKernel);
                break;
            case KernelPlan::TEAMMATE_EXACT_GENERAL:
                kernelName = "CcuReduceScatterTeammateExactKernel";
                kernelFunction = reinterpret_cast<void *>(
                    teammate_exact_kernel::
                        CcuReduceScatterLayerDiePairwiseKernel);
                break;
            case KernelPlan::RANK4_SMALL_RH:
                kernelName = "CcuReduceScatterRank4SmallRhKernel";
                kernelFunction = reinterpret_cast<void *>(
                    CcuReduceScatterRank4SmallRhKernel);
                break;
            case KernelPlan::RANK16_SMALL_THIN:
                kernelName = "CcuReduceScatterRank16SmallThinKernel";
                kernelFunction = reinterpret_cast<void *>(
                    CcuReduceScatterRank16SmallThinKernel);
                break;
            case KernelPlan::GENERAL:
            default:
                break;
        }
        (void)snprintf(
            kernelInfo.kernelFuncName,
            sizeof(kernelInfo.kernelFuncName),
            "%s", kernelName);
        kernelInfo.kernelFunc = kernelFunction;

        auto kernelArg = std::make_shared<ReduceScatterKernelArg>();
        kernelArg->includeSelf = includeSelf;
        kernelArg->dataType = param.dataType;
        kernelArg->reduceOp = param.reduceType;
        kernelArg->rank4SmallMyRank = param.myRank;
        kernelArg->rank4SmallXorHighChannelSlot =
            RANK4_SMALL_INVALID_CHANNEL_SLOT;
        kernelArg->rank4SmallXorLowChannelSlot =
            RANK4_SMALL_INVALID_CHANNEL_SLOT;
        kernelArg->channelCount = static_cast<uint32_t>(groups[groupIndex].channels.size());
        for (size_t index = 0; index < groups[groupIndex].channels.size(); ++index) {
            kernelArg->channels[index] = groups[groupIndex].channels[index];
            if (kernelPlan == KernelPlan::RANK4_SMALL_RH) {
                const uint32_t peer =
                    groups[groupIndex].descs[index].remoteRank;
                if (peer == (param.myRank ^ 2U)) {
                    kernelArg->rank4SmallXorHighChannelSlot =
                        static_cast<uint32_t>(index);
                }
                if (peer == (param.myRank ^ 1U)) {
                    kernelArg->rank4SmallXorLowChannelSlot =
                        static_cast<uint32_t>(index);
                }
            }
        }
        CHK_PRT_RET(
            kernelPlan == KernelPlan::RANK4_SMALL_RH &&
                (kernelArg->rank4SmallXorHighChannelSlot ==
                        RANK4_SMALL_INVALID_CHANNEL_SLOT ||
                    kernelArg->rank4SmallXorLowChannelSlot ==
                        RANK4_SMALL_INVALID_CHANNEL_SLOT ||
                    kernelArg->rank4SmallXorHighChannelSlot ==
                        kernelArg->rank4SmallXorLowChannelSlot),
            HCCL_ERROR(
                "rank4-small RH peer channel mapping is incomplete"),
            HCCL_E_INTERNAL);
        kernelInfo.setKernelArg(kernelArg);

        CcuKernelHandle kernelHandle = 0;
        constexpr uint32_t KERNEL_ARG_NUM = 1;
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        ccuRet = HcommCcuKernelRegister(insHandle, groups[groupIndex].dieId,
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs,
            KERNEL_ARG_NUM, &kernelHandle);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS,
            HCCL_ERROR("kernel register failed for layer[%u] scope[%u] die[%u], ret[%d]",
                groups[groupIndex].layerId,
                static_cast<uint32_t>(groups[groupIndex].scope),
                groups[groupIndex].dieId, ccuRet),
            HCCL_E_INTERNAL);
        resCtx.ccuKernels.push_back(kernelHandle);
        const auto threadIter = dieToThread.find(groups[groupIndex].dieId);
        CHK_PRT_RET(threadIter == dieToThread.end(),
            HCCL_ERROR("die[%u] has no Host thread", groups[groupIndex].dieId),
            HCCL_E_INTERNAL);
        resCtx.kernelThreadIndices.push_back(threadIter->second);
        resCtx.kernelPhaseIndices.push_back(phaseIndex);
        resCtx.kernelRankSizes.push_back(
            kernelArg->channelCount + (includeSelf ? 1U : 0U));
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("HcommCcuKernelRegisterEnd failed, ret[%d]", ccuRet), HCCL_E_INTERNAL);
    resCtx.primaryKernelIndex = 0;
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("only FP32 is supported, dataType[%d]", dataType), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM,
        HCCL_ERROR("only SUM is supported, reduceOp[%d]", op), HCCL_E_NOT_SUPPORT);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("unsupported rankSize[%u]", param.rankSize), HCCL_E_NOT_SUPPORT);

    constexpr uint64_t FP32_BYTES = sizeof(float);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / FP32_BYTES,
        HCCL_ERROR("output byte size overflows, recvCount[%llu]",
            static_cast<unsigned long long>(recvCount)),
        HCCL_E_PARA);
    const uint64_t outputBytes = recvCount * FP32_BYTES;
    CHK_PRT_RET(outputBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("input byte size overflows, outputBytes[%llu], rankSize[%u]",
            static_cast<unsigned long long>(outputBytes), param.rankSize),
        HCCL_E_PARA);
    const uint64_t inputBytes = outputBytes * param.rankSize;
    const ScoringSizeFamily sizeFamily =
        SelectScoringSizeFamily(param);
    (void)snprintf(
        param.tag, sizeof(param.tag),
        "hccl_final_rs_f1r72_r%u_s%u",
        param.rankSize, static_cast<uint32_t>(sizeFamily));

    constexpr CommEngine CCU_ENGINE = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t LOCAL_NOTIFY_SLOT_COUNT = 2;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, CCU_ENGINE, stream, LOCAL_NOTIFY_SLOT_COUNT, &param.cpuThread));

    HcclDfxOpInfo dfxInfo{};
    dfxInfo.opMode = 0;
    dfxInfo.opType = static_cast<uint32_t>(param.opType);
    dfxInfo.reduceOp = static_cast<uint32_t>(param.reduceType);
    dfxInfo.dataType = static_cast<uint32_t>(param.dataType);
    dfxInfo.dataCount = recvCount;
    (void)snprintf(dfxInfo.algTag, sizeof(dfxInfo.algTag), "%s", param.tag);
    dfxInfo.engine = CCU_ENGINE;
    dfxInfo.cpuTsThread = param.cpuThread;
    dfxInfo.inputMemAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    dfxInfo.inputMemSize = inputBytes;
    dfxInfo.outputMemAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    dfxInfo.outputMemSize = outputBytes;

    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, CCU_ENGINE, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        resCtxHost.ccuThread = param.cpuThread;
        resCtxHost.threads.push_back(param.cpuThread);

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        if (param.rankSize > 1) {
            std::vector<LayerGroup> groups;
            CHK_RET(AcquirePartitionedChannels(comm, param, groups));
            CHK_PRT_RET(groups.empty() || groups.size() > MAX_KERNEL_GROUP_NUM,
                HCCL_ERROR("unsupported network group count[%zu]", groups.size()),
                HCCL_E_NOT_SUPPORT);

            // One Host thread is sufficient for every kernel translated onto
            // the same CCU.  A second thread is acquired only when the selected
            // channels actually cover both local IO Dies.
            std::map<uint32_t, uint32_t> dieToThread;
            for (const LayerGroup &group : groups) {
                if (dieToThread.find(group.dieId) != dieToThread.end()) {
                    continue;
                }
                const uint32_t threadIndex =
                    static_cast<uint32_t>(dieToThread.size());
                CHK_PRT_RET(threadIndex >= MAX_IO_DIE_NUM,
                    HCCL_ERROR("too many active IO Dies[%u]", threadIndex + 1),
                    HCCL_E_NOT_SUPPORT);
                dieToThread[group.dieId] = threadIndex;
            }
            if (dieToThread.size() == MAX_IO_DIE_NUM) {
                ThreadHandle worker = 0;
                CHK_RET(HcclThreadAcquire(
                    comm, CCU_ENGINE, 1, LOCAL_NOTIFY_SLOT_COUNT, &worker));
                resCtxHost.threads.push_back(worker);
            }
            CHK_RET(RegisterDirectKernels(
                comm, param, groups, dieToThread, resCtxHost));
        }

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CCU_ENGINE, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, CCU_ENGINE, param.tag, seq.data(), seq.size(), 0));
    }

    return ops_hccl::ExecOpDispatch(param);
}
