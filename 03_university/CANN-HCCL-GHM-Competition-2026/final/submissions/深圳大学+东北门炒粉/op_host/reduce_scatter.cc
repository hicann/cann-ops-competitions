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
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
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
constexpr uint32_t SUB_THREAD_NOTIFY_NUM = 1;
constexpr uint64_t LARGE_MESSAGE_COUNT = (1ULL << 20) / sizeof(float);
constexpr char REDUCE_SCATTER_SMALL_TAG[] = "hccl_custom_reducescatter_inline_barrier_v43";
constexpr char REDUCE_SCATTER_LARGE_TAG[] = "hccl_custom_reducescatter_direct_stripe_v43";
static_assert(sizeof(REDUCE_SCATTER_SMALL_TAG) <= TAG_LENGTH, "Small ReduceScatter tag is too long");
static_assert(sizeof(REDUCE_SCATTER_LARGE_TAG) <= TAG_LENGTH, "Large ReduceScatter tag is too long");

struct ChannelGroup {
    uint32_t dieId = 0;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> remoteRanks;
};

HcclResult ValidateOpParam(const OpParam &param)
{
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[HcclReduceScatter] invalid rankSize[%u]", param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.myRank >= param.rankSize,
        HCCL_ERROR("[HcclReduceScatter] myRank[%u] is outside rankSize[%u]", param.myRank, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[HcclReduceScatter] only FP32 is supported, dataType[%d]", param.dataType), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("[HcclReduceScatter] only SUM is supported, reduceType[%d]", param.reduceType),
        HCCL_E_NOT_SUPPORT);

    constexpr uint64_t dataTypeSize = sizeof(float);
    const uint64_t maxValue = std::numeric_limits<uint64_t>::max();
    CHK_PRT_RET(param.count > maxValue / dataTypeSize,
        HCCL_ERROR("[HcclReduceScatter] recvCount overflows byte size"), HCCL_E_PARA);
    const uint64_t rankSliceBytes = param.count * dataTypeSize;
    CHK_PRT_RET(rankSliceBytes > maxValue / param.rankSize,
        HCCL_ERROR("[HcclReduceScatter] input byte size overflows"), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult MakeChannelDesc(const CommLink &link, uint32_t remoteRank, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint = link.srcEndpointDesc;
    desc.remoteEndpoint = link.dstEndpointDesc;
    return HCCL_SUCCESS;
}

bool SelectDualDieRank4(const OpParam &param,
    const std::vector<std::map<uint32_t, HcclChannelDesc>> &candidates,
    uint32_t &primaryDieId, std::vector<HcclChannelDesc> &descs,
    std::vector<uint32_t> &dieIds)
{
    if (param.rankSize != 4 || param.count <= LARGE_MESSAGE_COUNT) {
        return false;
    }

    std::vector<uint32_t> peers;
    peers.reserve(3);
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.myRank) {
            peers.push_back(rank);
        }
    }

    std::vector<uint32_t> commonDies;
    for (const auto &candidate : candidates[peers.front()]) {
        bool availableToEveryPeer = true;
        for (uint32_t peer : peers) {
            if (candidates[peer].find(candidate.first) == candidates[peer].end()) {
                availableToEveryPeer = false;
                break;
            }
        }
        if (availableToEveryPeer) {
            commonDies.push_back(candidate.first);
        }
    }
    if (commonDies.size() < 2) {
        return false;
    }

    auto primary = std::find(commonDies.begin(), commonDies.end(), primaryDieId);
    if (primary == commonDies.end()) {
        primaryDieId = commonDies.front();
        primary = commonDies.begin();
    }
    const uint32_t secondaryDieId =
        primary == commonDies.begin() ? commonDies[1] : commonDies.front();

    descs.clear();
    dieIds.clear();
    descs.reserve(peers.size() * 2);
    dieIds.reserve(peers.size() * 2);
    for (uint32_t dieId : {primaryDieId, secondaryDieId}) {
        for (uint32_t peer : peers) {
            dieIds.push_back(dieId);
            descs.push_back(candidates[peer].at(dieId));
        }
    }
    return true;
}

HcclResult SelectOneChannelPerPeer(HcclComm comm, const OpParam &param, std::vector<HcclChannelDesc> &descs,
    std::vector<uint32_t> &dieIds, uint32_t &primaryDieId)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("[SelectOneChannelPerPeer] rank graph has no layer"), HCCL_E_NOT_FOUND);
    const std::vector<uint32_t> layers(netLayers, netLayers + netLayerNum);

    std::vector<std::map<uint32_t, HcclChannelDesc>> candidates(param.rankSize);
    std::map<uint32_t, uint32_t> singleChannelCountByDie;
    bool hasMultiDiePeer = false;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        auto &peerCandidates = candidates[remoteRank];
        for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
            CommLink *linkList = nullptr;
            uint32_t linkNum = 0;
            CHK_RET(HcclRankGraphGetLinks(
                comm, layers[layerIdx], param.myRank, remoteRank, &linkList, &linkNum));
            CHK_PRT_RET(linkNum != 0 && linkList == nullptr,
                HCCL_ERROR("[SelectOneChannelPerPeer] invalid link list for remoteRank[%u]", remoteRank),
                HCCL_E_INTERNAL);
            for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                const CommLink &link = linkList[linkIdx];
                if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }

                EndpointAttrDieId dieId = 0;
                CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &link.srcEndpointDesc,
                    ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
                if (peerCandidates.find(dieId) == peerCandidates.end()) {
                    HcclChannelDesc desc;
                    CHK_RET(MakeChannelDesc(link, remoteRank, desc));
                    peerCandidates.emplace(dieId, desc);
                }
            }
        }

        CHK_PRT_RET(peerCandidates.empty(),
            HCCL_ERROR("[SelectOneChannelPerPeer] no UBC_CTP link to remoteRank[%u]", remoteRank),
            HCCL_E_NOT_FOUND);
        if (peerCandidates.size() == 1) {
            ++singleChannelCountByDie[peerCandidates.begin()->first];
        } else {
            hasMultiDiePeer = true;
        }
    }

    if (singleChannelCountByDie.empty()) {
        primaryDieId = candidates[param.myRank == 0 ? 1 : 0].begin()->first;
    } else {
        auto selected = singleChannelCountByDie.begin();
        for (auto it = std::next(singleChannelCountByDie.begin()); it != singleChannelCountByDie.end(); ++it) {
            const bool preferMore = hasMultiDiePeer && it->second > selected->second;
            const bool preferLess = !hasMultiDiePeer && it->second < selected->second;
            if (preferMore || preferLess) {
                selected = it;
            }
        }
        primaryDieId = selected->first;
    }

    if (SelectDualDieRank4(param, candidates, primaryDieId, descs, dieIds)) {
        return HCCL_SUCCESS;
    }

    descs.clear();
    dieIds.clear();
    descs.reserve(param.rankSize - 1);
    dieIds.reserve(param.rankSize - 1);
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        const auto &peerCandidates = candidates[remoteRank];
        auto selected = peerCandidates.begin();
        if (peerCandidates.size() > 1) {
            if (singleChannelCountByDie.empty()) {
                auto primary = peerCandidates.find(primaryDieId);
                if (primary != peerCandidates.end()) {
                    selected = primary;
                }
            } else {
                for (auto it = peerCandidates.begin(); it != peerCandidates.end(); ++it) {
                    if (it->first != primaryDieId) {
                        selected = it;
                        break;
                    }
                }
            }
        }
        dieIds.push_back(selected->first);
        descs.push_back(selected->second);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterOneKernel(CcuInsHandle insHandle, uint32_t dieId, const char *name, const void *function,
    const void *kernelArg, CcuKernelHandle &kernelHandle)
{
    const void *kernelArgs[] = {kernelArg};
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, name, function, kernelArgs, 1, &kernelHandle));
    return HCCL_SUCCESS;
}

void ResetKernelResource(AlgResourceCtx &resCtx)
{
    resCtx.channelGroupCount = 0;
    std::fill(std::begin(resCtx.channelGroupSizes), std::end(resCtx.channelGroupSizes), 0);
    resCtx.pullKernelCount = 0;
    resCtx.stripeKernelCount = 0;
    resCtx.releaseKernelCount = 0;
    resCtx.fusedKernel = 0;
    resCtx.reduceKernel = 0;
    resCtx.stripeReduceKernel = 0;
}

void FillBaseChannels(const ChannelGroup &group, CcuKernelArgBase &kernelArg)
{
    kernelArg.channelCount = static_cast<uint32_t>(group.channels.size());
    for (uint32_t channelIdx = 0; channelIdx < kernelArg.channelCount; ++channelIdx) {
        kernelArg.channels[channelIdx] = group.channels[channelIdx];
    }
}

template <typename T> void FillChannels(const ChannelGroup &group, T &kernelArg)
{
    FillBaseChannels(group, kernelArg);
    for (uint32_t channelIdx = 0; channelIdx < kernelArg.channelCount; ++channelIdx) {
        kernelArg.remoteRanks[channelIdx] = group.remoteRanks[channelIdx];
    }
}

HcclResult ValidateChannelGroups(const OpParam &param, const std::vector<ChannelGroup> &groups)
{
    const bool validGroupCount = param.rankSize == 4 ?
        (groups.size() == 1 || groups.size() == 2) : groups.size() == 2;
    CHK_PRT_RET(!validGroupCount,
        HCCL_ERROR("[ValidateChannelGroups] invalid group count[%zu]", groups.size()),
        HCCL_E_NOT_SUPPORT);

    const bool dualRank4 = param.rankSize == 4 && groups.size() == 2;
    uint32_t rankOccurrences[MAX_RANK_SIZE]{};
    uint32_t peerCount = 0;
    std::vector<size_t> groupSizes;
    groupSizes.reserve(groups.size());
    for (const ChannelGroup &group : groups) {
        CHK_PRT_RET(group.channels.empty() || group.channels.size() != group.remoteRanks.size(),
            HCCL_ERROR("[ValidateChannelGroups] invalid channel group"), HCCL_E_INTERNAL);
        groupSizes.push_back(group.channels.size());
        for (uint32_t channelIdx = 0; channelIdx < group.remoteRanks.size(); ++channelIdx) {
            const uint32_t remoteRank = group.remoteRanks[channelIdx];
            CHK_PRT_RET(remoteRank >= param.rankSize || remoteRank == param.myRank,
                HCCL_ERROR("[ValidateChannelGroups] invalid remoteRank[%u]", remoteRank),
                HCCL_E_INTERNAL);
            CHK_PRT_RET(channelIdx != 0 && group.remoteRanks[channelIdx - 1] >= remoteRank,
                HCCL_ERROR("[ValidateChannelGroups] remote ranks are not ordered"),
                HCCL_E_INTERNAL);
            ++rankOccurrences[remoteRank];
            ++peerCount;
        }
    }
    const uint32_t expectedOccurrences = dualRank4 ? 2 : 1;
    CHK_PRT_RET(peerCount != (param.rankSize - 1) * expectedOccurrences,
        HCCL_ERROR("[ValidateChannelGroups] peer count[%u] does not match rankSize[%u]",
            peerCount, param.rankSize),
        HCCL_E_INTERNAL);
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        const uint32_t expected = rank == param.myRank ? 0 : expectedOccurrences;
        CHK_PRT_RET(rankOccurrences[rank] != expected,
            HCCL_ERROR("[ValidateChannelGroups] rank[%u] occurrence[%u], expected[%u]",
                rank, rankOccurrences[rank], expected),
            HCCL_E_INTERNAL);
    }

    std::sort(groupSizes.begin(), groupSizes.end());
    const bool validRank4 = param.rankSize == 4 &&
        (groupSizes == std::vector<size_t>{3} || groupSizes == std::vector<size_t>{3, 3});
    const bool validRank16 = param.rankSize == MAX_RANK_SIZE &&
        groupSizes == std::vector<size_t>{7, 8};
    const bool validRank12 = param.rankSize == 12 &&
        (groupSizes == std::vector<size_t>{3, 8} || groupSizes == std::vector<size_t>{4, 7});
    CHK_PRT_RET(!validRank4 && !validRank12 && !validRank16,
        HCCL_ERROR("[ValidateChannelGroups] unsupported topology for rankSize[%u]", param.rankSize),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

std::shared_ptr<ops_hccl::CcuKernelArgReduceScatterPull> MakePullArg(const OpParam &param,
    const ChannelGroup &group, uint32_t phase, uint32_t groupIndex)
{
    auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgReduceScatterPull>();
    kernelArg->rankId = param.myRank;
    kernelArg->rankSize = param.rankSize;
    // Adjacent chunks alternate CKE bits so an early signal cannot coalesce with an unconsumed one.
    kernelArg->addrMask = static_cast<uint16_t>(uint32_t{1} << (phase * 2));
    kernelArg->tokenMask = static_cast<uint16_t>(uint32_t{1} << (phase * 2 + 1));
    kernelArg->prepareOwnContribution = groupIndex == 0;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    FillChannels(group, *kernelArg);
    return kernelArg;
}

std::shared_ptr<ops_hccl::CcuKernelArgReduceScatterStripe> MakeStripeArg(const OpParam &param,
    const ChannelGroup &group, uint32_t phase, uint32_t groupIndex, bool finalBarrier)
{
    auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgReduceScatterStripe>();
    kernelArg->rankId = param.myRank;
    kernelArg->rankSize = param.rankSize;
    kernelArg->addrMask = static_cast<uint16_t>(uint32_t{1} << (phase * 2));
    kernelArg->tokenMask = static_cast<uint16_t>(uint32_t{1} << (phase * 2 + 1));
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    kernelArg->groupIndex = groupIndex;
    const bool dataParallelRank4 = param.rankSize == 4 && !finalBarrier;
    kernelArg->includeOwnContribution = dataParallelRank4 || groupIndex == 0;
    kernelArg->writeOutput = dataParallelRank4 || groupIndex == 0;
    kernelArg->finalBarrier = finalBarrier;
    kernelArg->directStripe =
        (param.rankSize == 12 || param.rankSize == MAX_RANK_SIZE) && !finalBarrier;
    FillChannels(group, *kernelArg);
    return kernelArg;
}

HcclResult RegisterFusedKernel(CcuInsHandle insHandle, const OpParam &param,
    const std::vector<ChannelGroup> &groups, AlgResourceCtx &resCtx)
{
    auto kernelArg = MakePullArg(param, groups.front(), 0, 0);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    CHK_RET(RegisterOneKernel(insHandle, groups.front().dieId, "CcuReduceScatterFused",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterFusedKernel), kernelArg.get(),
        resCtx.fusedKernel));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));

    constexpr uint32_t phaseCount = MAX_KERNEL_PHASE_COUNT;
    std::vector<std::shared_ptr<ops_hccl::CcuKernelArgReduceScatterStripe>> stripeArgs;
    stripeArgs.reserve(phaseCount);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    for (uint32_t phase = 0; phase < phaseCount; ++phase) {
        auto stripeArg = MakeStripeArg(param, groups.front(), phase, 0, true);
        char kernelName[64]{};
        const int nameRet = std::snprintf(
            kernelName, sizeof(kernelName), "CcuReduceScatterStripeP%u", phase);
        CHK_PRT_RET(nameRet <= 0 || static_cast<size_t>(nameRet) >= sizeof(kernelName),
            HCCL_ERROR("[RegisterFusedKernel] failed to set stripe kernel name"),
            HCCL_E_INTERNAL);

        CcuKernelHandle kernel = 0;
        CHK_RET(RegisterOneKernel(insHandle, groups.front().dieId, kernelName,
            reinterpret_cast<void *>(ops_hccl::CcuReduceScatterStripeKernel),
            stripeArg.get(), kernel));
        stripeArgs.push_back(stripeArg);
        resCtx.stripeKernels[resCtx.stripeKernelCount++] = kernel;
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    resCtx.channelGroupCount = 1;
    resCtx.channelGroupSizes[0] = static_cast<uint32_t>(groups.front().channels.size());
    return HCCL_SUCCESS;
}

HcclResult RegisterSplitKernels(CcuInsHandle insHandle, const OpParam &param,
    const std::vector<ChannelGroup> &groups, AlgResourceCtx &resCtx)
{
    constexpr uint32_t phaseCount = MAX_KERNEL_PHASE_COUNT;
    const uint32_t groupCount = static_cast<uint32_t>(groups.size());
    std::vector<std::shared_ptr<ops_hccl::CcuKernelArgReduceScatterPull>> pullArgs;
    if (param.rankSize == 4) {
        pullArgs.reserve(groupCount);
        for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
            pullArgs.push_back(MakePullArg(param, groups[groupIdx], 0, groupIdx));
        }
    } else {
        pullArgs.reserve(phaseCount * groupCount);
        CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
        for (uint32_t phase = 0; phase < phaseCount; ++phase) {
            for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
                auto kernelArg = MakePullArg(param, groups[groupIdx], phase, groupIdx);
                char kernelName[64]{};
                const int nameRet = std::snprintf(kernelName, sizeof(kernelName),
                    "CcuReduceScatterPullP%uG%u", phase, groupIdx);
                CHK_PRT_RET(nameRet <= 0 || static_cast<size_t>(nameRet) >= sizeof(kernelName),
                    HCCL_ERROR("[RegisterSplitKernels] failed to set pull kernel name"),
                    HCCL_E_INTERNAL);

                CcuKernelHandle kernel = 0;
                CHK_RET(RegisterOneKernel(insHandle, groups[groupIdx].dieId, kernelName,
                    reinterpret_cast<void *>(ops_hccl::CcuReduceScatterPullKernel),
                    kernelArg.get(), kernel));
                pullArgs.push_back(kernelArg);
                resCtx.pullKernels[resCtx.pullKernelCount++] = kernel;
            }
        }
        CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    }

    std::vector<std::shared_ptr<ops_hccl::CcuKernelArgReduceScatterStripe>> stripeArgs;
    stripeArgs.reserve(phaseCount * groupCount);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    for (uint32_t phase = 0; phase < phaseCount; ++phase) {
        for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
            auto stripeArg = MakeStripeArg(param, groups[groupIdx], phase, groupIdx, false);
            char kernelName[64]{};
            const int nameRet = std::snprintf(kernelName, sizeof(kernelName),
                "CcuReduceScatterStripeP%uG%u", phase, groupIdx);
            CHK_PRT_RET(nameRet <= 0 || static_cast<size_t>(nameRet) >= sizeof(kernelName),
                HCCL_ERROR("[RegisterSplitKernels] failed to set stripe kernel name"),
                HCCL_E_INTERNAL);

            CcuKernelHandle kernel = 0;
            void *stripeFunction = (param.rankSize == 12 || param.rankSize == MAX_RANK_SIZE) ?
                reinterpret_cast<void *>(ops_hccl::CcuReduceScatterDirectStripeKernel) :
                reinterpret_cast<void *>(ops_hccl::CcuReduceScatterStripeKernel);
            CHK_RET(RegisterOneKernel(insHandle, groups[groupIdx].dieId, kernelName,
                stripeFunction, stripeArg.get(), kernel));
            stripeArgs.push_back(stripeArg);
            resCtx.stripeKernels[resCtx.stripeKernelCount++] = kernel;
        }
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));

    if (param.rankSize != 4) {
        auto reduceArg = std::make_shared<ops_hccl::CcuKernelArgReduceScatterReduce>();
        reduceArg->rankId = param.myRank;
        reduceArg->rankSize = param.rankSize;
        reduceArg->dataType = param.dataType;
        reduceArg->reduceOp = param.reduceType;
        CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
        CHK_RET(RegisterOneKernel(insHandle, groups.front().dieId, "CcuReduceScatterReduce",
            reinterpret_cast<void *>(ops_hccl::CcuReduceScatterReduceKernel), reduceArg.get(),
            resCtx.reduceKernel));
        void *stripeReduceFunction =
            (param.rankSize == 12 || param.rankSize == MAX_RANK_SIZE) ?
            reinterpret_cast<void *>(ops_hccl::CcuReduceScatterDirectMergeKernel) :
            reinterpret_cast<void *>(ops_hccl::CcuReduceScatterStripeReduceKernel);
        CHK_RET(RegisterOneKernel(insHandle, groups.front().dieId, "CcuReduceScatterStripeReduce",
            stripeReduceFunction, reduceArg.get(), resCtx.stripeReduceKernel));
        CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    }

    resCtx.releaseKernelCount = 0;
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
        char kernelName[64]{};
        const int nameRet = std::snprintf(
            kernelName, sizeof(kernelName), "CcuReduceScatterReleaseG%u", groupIdx);
        CHK_PRT_RET(nameRet <= 0 || static_cast<size_t>(nameRet) >= sizeof(kernelName),
            HCCL_ERROR("[RegisterSplitKernels] failed to set release kernel name"),
            HCCL_E_INTERNAL);
        CcuKernelHandle releaseKernel = 0;
        CHK_RET(RegisterOneKernel(insHandle, groups[groupIdx].dieId, kernelName,
            reinterpret_cast<void *>(ops_hccl::CcuReduceScatterReleaseKernel),
            pullArgs[groupIdx].get(), releaseKernel));
        resCtx.releaseKernels[resCtx.releaseKernelCount++] = releaseKernel;
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    resCtx.channelGroupCount = groupCount;
    for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
        resCtx.channelGroupSizes[groupIdx] =
            static_cast<uint32_t>(groups[groupIdx].channels.size());
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelGroup> &groups,
    AlgResourceCtx &resCtx)
{
    ResetKernelResource(resCtx);
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(param.rankSize != 4 && param.rankSize != 12 && param.rankSize != MAX_RANK_SIZE,
        HCCL_ERROR("[RegisterKernels] unsupported rankSize[%u]", param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_RET(ValidateChannelGroups(param, groups));

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[RegisterKernels] expected one CCU instance, got[%u]", insNum), HCCL_E_INTERNAL);
    if (param.rankSize == 4 && groups.size() == 1) {
        return RegisterFusedKernel(insHandle, param, groups, resCtx);
    }
    return RegisterSplitKernels(insHandle, param, groups, resCtx);
}

HcclResult AllocateResource(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    resCtx.rankId = param.myRank;
    resCtx.rankSize = param.rankSize;
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    void *bufferAddr = nullptr;
    uint64_t bufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &bufferAddr, &bufferSize));
    CHK_PRT_RET(bufferAddr == nullptr || bufferSize == 0,
        HCCL_ERROR("[AllocateResource] invalid local HCCL buffer"), HCCL_E_INTERNAL);
    resCtx.localBuffer = {bufferAddr, bufferSize};
    resCtx.minBufferSize = bufferSize;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(bufferAddr), bufferSize,
        &resCtx.scratchToken));

    std::vector<HcclChannelDesc> channelDescs;
    std::vector<uint32_t> channelDieIds;
    uint32_t primaryDieId = 0;
    CHK_RET(SelectOneChannelPerPeer(comm, param, channelDescs, channelDieIds, primaryDieId));

    std::vector<ChannelHandle> channels(channelDescs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, channelDescs.data(),
        static_cast<uint32_t>(channelDescs.size()), channels.data()));

    std::map<uint32_t, ChannelGroup> groupsByDie;
    const uint32_t channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t channelIdx = 0; channelIdx < channelCount; ++channelIdx) {
        void *remoteBufferAddr = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(
            comm, channels[channelIdx], &remoteBufferAddr, &remoteBufferSize));
        CHK_PRT_RET(remoteBufferAddr == nullptr || remoteBufferSize == 0,
            HCCL_ERROR("[AllocateResource] invalid remote HCCL buffer"), HCCL_E_INTERNAL);
        resCtx.minBufferSize = std::min(resCtx.minBufferSize, remoteBufferSize);

        const uint32_t dieId = channelDieIds[channelIdx];
        ChannelGroup &group = groupsByDie[dieId];
        group.dieId = dieId;
        group.channels.push_back(channels[channelIdx]);
        group.remoteRanks.push_back(channelDescs[channelIdx].remoteRank);
    }

    std::vector<ChannelGroup> groups;
    groups.reserve(groupsByDie.size());
    auto primaryGroup = groupsByDie.find(primaryDieId);
    CHK_PRT_RET(primaryGroup == groupsByDie.end(),
        HCCL_ERROR("[AllocateResource] primary die[%u] has no channel", primaryDieId), HCCL_E_INTERNAL);
    groups.push_back(std::move(primaryGroup->second));
    for (auto &entry : groupsByDie) {
        if (entry.first != primaryDieId) {
            groups.push_back(std::move(entry.second));
        }
    }
    CHK_PRT_RET(groups.empty(), HCCL_ERROR("[AllocateResource] no channel group"), HCCL_E_INTERNAL);

    const uint32_t subThreadNum = static_cast<uint32_t>(groups.size() - 1);
    if (subThreadNum != 0) {
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, subThreadNum, SUB_THREAD_NOTIFY_NUM,
            resCtx.subThreads));
    }
    resCtx.subThreadCount = subThreadNum;
    CHK_RET(RegisterKernels(comm, param, groups, resCtx));
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

    OpParam param{};
    const char *selectedTag = recvCount > LARGE_MESSAGE_COUNT ?
        REDUCE_SCATTER_LARGE_TAG : REDUCE_SCATTER_SMALL_TAG;
    const size_t selectedTagSize = std::strlen(selectedTag) + 1;
    std::memcpy(param.tag, selectedTag, selectedTagSize);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    constexpr CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    void *ctx = nullptr;
    uint64_t size = 0;
    const bool hasCachedCtx =
        HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS;
    AlgResourceCtx cachedResCtx;
    if (hasCachedCtx) {
        CHK_PRT_RET(ctx == nullptr || size != sizeof(AlgResourceCtx),
            HCCL_ERROR("[HcclReduceScatter] invalid cached engine context"), HCCL_E_INTERNAL);
        std::memcpy(&cachedResCtx, ctx, sizeof(cachedResCtx));
        param.myRank = cachedResCtx.rankId;
        param.rankSize = cachedResCtx.rankSize;
    } else {
        CHK_RET(HcclGetRankId(comm, &param.myRank));
        CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    }
    CHK_RET(ValidateOpParam(param));
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    uint32_t mainThreadNotifyNum = 0;
    if (hasCachedCtx) {
        CHK_PRT_RET(cachedResCtx.subThreadCount > MAX_CHANNEL_GROUP_COUNT - 1,
            HCCL_ERROR("[HcclReduceScatter] invalid cached sub-thread count[%u]",
                cachedResCtx.subThreadCount),
            HCCL_E_INTERNAL);
        mainThreadNotifyNum = cachedResCtx.subThreadCount;
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        CHK_RET(AllocateResource(comm, param, resCtxHost));
        mainThreadNotifyNum = resCtxHost.subThreadCount;
        param.ctxSize = sizeof(resCtxHost);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(
            comm, ccuEngine, param.tag, &resCtxHost, sizeof(resCtxHost), 0));
    }

    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, mainThreadNotifyNum, &param.cpuThread));
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
