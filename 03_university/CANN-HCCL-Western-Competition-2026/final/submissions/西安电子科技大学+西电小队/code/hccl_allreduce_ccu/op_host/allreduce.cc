/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_launch.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t MAX_ACTIVE_DIES = 2;
constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t DUAL_GROUP_PUSH_MAX_BYTES = 512U * 1024U;
constexpr uint64_t EIGHT_PLUS_FOUR_LAYOUT_SAFE_TREE_P16_BYTES = 512U * 1024U;
constexpr uint64_t EIGHT_PLUS_FOUR_EXACT_P16_COMPACT_PRESYNC_BYTES = 512U * 1024U;
constexpr uint64_t EIGHT_PLUS_FOUR_EXACT_P17_TREE_PUSH_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t EIGHT_PLUS_FOUR_EXACT_P18_TREE_BYTES = 400ULL * 1024ULL * 1024ULL + FP32_BYTES;
constexpr uint64_t TWO_BY_EIGHT_EXACT_P11_TREE_PUSH_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t TWO_BY_EIGHT_EXACT_P12_TREE_PUSH_BYTES = 400ULL * 1024ULL * 1024ULL + FP32_BYTES;
constexpr uint64_t FOUR_BY_ONE_EXACT_P14_NHR_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t FOUR_BY_ONE_EXACT_P13_COMPACT_BYTES = 512ULL * 1024ULL;
constexpr uint64_t FOUR_BY_ONE_EXACT_P15_NHR_BYTES = 400ULL * 1024ULL * 1024ULL + FP32_BYTES;
constexpr uint32_t DUAL_GROUP_PUSH_MIN_ELEMENTS_PER_RANK = 2U;
constexpr uint64_t SINGLE_GROUP_PUSH_MAX_BYTES = 512U * 1024U;
constexpr uint32_t SINGLE_GROUP_PUSH_MIN_ELEMENTS_PER_RANK = 2U;
constexpr uint32_t GROUP_REDUCE_INTERLEAVE = 8;
constexpr uint32_t ISOLATED_DUAL_GROUP_RESOURCE_FLAG = 1U << 31U;
constexpr uint32_t ISOLATED_EXACT_P14_NHR_RESOURCE_FLAG = 1U << 30U;
constexpr uint32_t ISOLATED_EXACT_P13_COMPACT_RESOURCE_FLAG = 1U << 29U;
constexpr uint32_t ISOLATED_EXACT_P15_NHR_RESOURCE_FLAG = 1U << 28U;
constexpr uint32_t ISOLATED_EXACT_P17_TREE_PUSH_RESOURCE_FLAG = 1U << 27U;
constexpr uint32_t ISOLATED_EXACT_P16_COMPACT_PRESYNC_RESOURCE_FLAG = 1U << 26U;
constexpr uint32_t ISOLATED_EXACT_P18_TREE_RESOURCE_FLAG = 1U << 25U;
constexpr uint32_t ISOLATED_EXACT_P11_TREE_PUSH_RESOURCE_FLAG = 1U << 24U;
constexpr uint32_t ISOLATED_EXACT_P12_TREE_PUSH_RESOURCE_FLAG = 1U << 23U;

using ChannelGroups = std::map<uint32_t, std::vector<HcclChannelDesc>>;

uint32_t ScratchSlotForPeer(uint32_t myRank, uint32_t peerRank)
{
    return peerRank < myRank ? peerRank : peerRank - 1U;
}

HcclResult DetectTopology(HcclComm comm, uint32_t rankSize, TopologyKind &kind)
{
    if (rankSize == 1U) {
        kind = TopologyKind::SINGLE_RANK;
        return HCCL_SUCCESS;
    }

    uint32_t *instanceSizes = nullptr;
    uint32_t instanceCount = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &instanceSizes, &instanceCount));
    std::vector<uint32_t> sizes(instanceSizes, instanceSizes + instanceCount);
    std::sort(sizes.begin(), sizes.end(), std::greater<uint32_t>());

    if (rankSize == 16U && sizes == std::vector<uint32_t>({8U, 8U})) {
        kind = TopologyKind::TOPO_2X8;
    } else if (rankSize == 4U && sizes == std::vector<uint32_t>({1U, 1U, 1U, 1U})) {
        kind = TopologyKind::TOPO_4X1;
    } else if (rankSize == 12U && sizes == std::vector<uint32_t>({8U, 4U})) {
        kind = TopologyKind::TOPO_8_PLUS_4;
    } else {
        HCCL_ERROR("[CCU_V001] unsupported topology, rankSize[%u], instanceCount[%u]", rankSize, instanceCount);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult SelectChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank, const std::vector<uint32_t> &layers,
    HcclChannelDesc &desc, uint32_t &dieId)
{
    constexpr std::array<CommProtocol, 2> protocols = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
    };

    bool found = false;
    CommLink selectedLink{};
    for (uint32_t layer : layers) {
        CommLink *linkList = nullptr;
        uint32_t linkCount = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layer, myRank, remoteRank, &linkList, &linkCount));
        for (CommProtocol protocol : protocols) {
            for (uint32_t i = 0; i < linkCount; ++i) {
                if (linkList[i].linkAttr.linkProtocol == protocol) {
                    selectedLink = linkList[i];
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            }
        }
        if (found) {
            break;
        }
    }
    CHK_PRT_RET(!found, HCCL_ERROR("[CCU_V001] no CCU link between rank[%u] and rank[%u]", myRank, remoteRank),
        HCCL_E_NOT_FOUND);

    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = selectedLink.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = selectedLink.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = selectedLink.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = selectedLink.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = selectedLink.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = selectedLink.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = selectedLink.dstEndpointDesc.loc;

    EndpointAttrDieId endpointDie{};
    CHK_RET(HcclRankGraphGetEndpointInfo(
        comm, myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(endpointDie), &endpointDie));
    dieId = endpointDie;
    return HCCL_SUCCESS;
}

HcclResult BuildChannelGroups(HcclComm comm, const OpParam &param, ChannelGroups &groups)
{
    uint32_t *layerList = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerCount));
    std::vector<uint32_t> layers(layerList, layerList + layerCount);
    std::sort(layers.begin(), layers.end());

    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        HcclChannelDesc desc;
        uint32_t dieId = 0;
        CHK_RET(SelectChannelDesc(comm, param.myRank, peer, layers, desc, dieId));
        groups[dieId].push_back(desc);
    }
    CHK_PRT_RET(groups.empty() || groups.size() > MAX_ACTIVE_DIES,
        HCCL_ERROR("[CCU_V001] unsupported active die count[%zu]", groups.size()), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult AcquireThreads(HcclComm comm, const OpParam &param, uint32_t kernelCount, AlgResourceCtx &resCtx)
{
    resCtx.ccuThread = param.cpuThread;
    resCtx.threads.push_back(param.cpuThread);
    if (kernelCount == 2U) {
        ThreadHandle slave = 0;
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, 1, 1, &slave));
        resCtx.threads.push_back(slave);
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannelsAndRegisterKernels(
    HcclComm comm, const OpParam &param, const ChannelGroups &groups, AlgResourceCtx &resCtx)
{
    const uint32_t kernelCount = static_cast<uint32_t>(groups.size());
    CHK_RET(AcquireThreads(comm, param, kernelCount, resCtx));

    std::vector<std::vector<HcclChannelDesc>> descriptors;
    std::vector<uint32_t> dieIds;
    descriptors.reserve(kernelCount);
    dieIds.reserve(kernelCount);
    for (const auto &entry : groups) {
        dieIds.push_back(entry.first);
        descriptors.push_back(entry.second);
    }

    // Channel creation must be symmetric at the rank-pair level.  A link can
    // terminate on different local dies at its two ends, so acquiring one die
    // group at a time can make the peers enter different acquire batches and
    // deadlock during socket setup.  A single mixed-die bulk acquire avoids
    // that first failure mode, but its local descriptor order is not the same
    // at the two ends and can pair the reverse CCU resources inconsistently.
    // Use an even-rank round-robin matching: every rank talks to exactly one
    // peer in a round, and both ends enter HcclChannelAcquire in that same
    // round.  Resource creation is cached, so this correctness-first control
    // plane cost is outside the steady-state CCU kernel path.
    CHK_PRT_RET((param.rankSize & 1U) != 0U,
        HCCL_ERROR("[CCU_V001] direct channel schedule requires an even rank size"), HCCL_E_NOT_SUPPORT);

    std::vector<const HcclChannelDesc *> descriptorByPeer(param.rankSize, nullptr);
    for (const auto &group : descriptors) {
        for (const auto &desc : group) {
            descriptorByPeer[desc.remoteRank] = &desc;
        }
    }

    struct ChannelAcquireExchangeInfo {
        uint32_t rankSize;
        uint32_t round;
    };

    std::vector<ChannelHandle> handleByPeer(param.rankSize, 0);
    const uint32_t rotatingSize = param.rankSize - 1U;
    for (uint32_t round = 0; round < rotatingSize; ++round) {
        uint32_t peer = 0;
        if (param.myRank == param.rankSize - 1U) {
            peer = round;
        } else if (param.myRank == round) {
            peer = param.rankSize - 1U;
        } else {
            peer = (2U * round + rotatingSize - param.myRank) % rotatingSize;
        }

        CHK_PRT_RET(descriptorByPeer[peer] == nullptr,
            HCCL_ERROR("[CCU_V001] channel descriptor for peer[%u] is missing", peer), HCCL_E_INTERNAL);
        const ChannelAcquireExchangeInfo exchangeInfo{param.rankSize, round};
        CHK_RET(HcclCommAddExchangeInfo(comm, &exchangeInfo, sizeof(exchangeInfo)));
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, descriptorByPeer[peer], 1, &handleByPeer[peer]));
    }

    uint32_t includeLocalOrdinal = 0;
    if (kernelCount == 2U && descriptors[1].size() < descriptors[0].size()) {
        includeLocalOrdinal = 1;
    }
    uint32_t otherPartialSlot = 0;
    if (kernelCount == 2U) {
        const uint32_t nonLocalOrdinal = 1U - includeLocalOrdinal;
        otherPartialSlot = ScratchSlotForPeer(param.myRank, descriptors[nonLocalOrdinal][0].remoteRank);
    }

    std::vector<std::shared_ptr<CcuKernelArgDirect>> baseKernelArgs;
    baseKernelArgs.reserve(kernelCount);
    for (uint32_t ordinal = 0; ordinal < kernelCount; ++ordinal) {
        auto kernelArg = std::make_shared<CcuKernelArgDirect>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->myRank = param.myRank;
        kernelArg->dieOrdinal = dieIds[ordinal];
        kernelArg->activeDieCount = kernelCount;
        kernelArg->includeLocal = (kernelCount == 1U || ordinal == includeLocalOrdinal);
        kernelArg->otherPartialScratchSlot = otherPartialSlot;
        kernelArg->channelCount = static_cast<uint32_t>(descriptors[ordinal].size());
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            kernelArg->channels[i] = handleByPeer[descriptors[ordinal][i].remoteRank];
            kernelArg->peerRanks[i] = descriptors[ordinal][i].remoteRank;
            kernelArg->scratchSlots[i] = ScratchSlotForPeer(param.myRank, descriptors[ordinal][i].remoteRank);
        }
        baseKernelArgs.push_back(kernelArg);
    }

    CcuInsHandle insHandle{0};
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    CHK_PRT_RET(insCount != 1U, HCCL_ERROR("[CCU_V001] expected one CCU instance, got[%u]", insCount), HCCL_E_INTERNAL);

    bool groupInputsFitOneBatch = kernelCount == 2U;
    bool groupInputsFitTwoBatches = kernelCount == 2U;
    for (uint32_t ordinal = 0; ordinal < kernelCount; ++ordinal) {
        const uint32_t inputCount
            = baseKernelArgs[ordinal]->channelCount + (baseKernelArgs[ordinal]->includeLocal ? 1U : 0U);
        groupInputsFitOneBatch
            = groupInputsFitOneBatch && inputCount > 0U && inputCount <= GROUP_REDUCE_INTERLEAVE;
        groupInputsFitTwoBatches
            = groupInputsFitTwoBatches && inputCount > 0U && inputCount <= 2U * GROUP_REDUCE_INTERLEAVE;
    }
    const bool topologyInputsFit
        = (resCtx.topologyKind == TopologyKind::TOPO_2X8 && groupInputsFitOneBatch)
          || (resCtx.topologyKind == TopologyKind::TOPO_8_PLUS_4 && groupInputsFitTwoBatches);
    const bool useDualGroupPushSmall = topologyInputsFit
                                       && param.inputPtr != param.outputPtr
                                       && param.count >= param.rankSize * DUAL_GROUP_PUSH_MIN_ELEMENTS_PER_RANK
                                       && param.count <= DUAL_GROUP_PUSH_MAX_BYTES / FP32_BYTES;
    const bool useEightPlusFourLayoutSafeTreeP16
        = resCtx.topologyKind == TopologyKind::TOPO_8_PLUS_4 && param.inputPtr != param.outputPtr
          && param.count == EIGHT_PLUS_FOUR_LAYOUT_SAFE_TREE_P16_BYTES / FP32_BYTES;
    const bool useExactP16CompactPreSync
        = resCtx.topologyKind == TopologyKind::TOPO_8_PLUS_4 && param.rankSize == 12U && kernelCount == 2U
          && param.inputPtr != param.outputPtr
          && param.count == EIGHT_PLUS_FOUR_EXACT_P16_COMPACT_PRESYNC_BYTES / FP32_BYTES;
    const bool useExactP17TreePush
        = resCtx.topologyKind == TopologyKind::TOPO_8_PLUS_4 && param.rankSize == 12U && kernelCount == 2U
          && param.inputPtr != param.outputPtr
          && param.count == EIGHT_PLUS_FOUR_EXACT_P17_TREE_PUSH_BYTES / FP32_BYTES;
    const bool useExactP18Tree
        = resCtx.topologyKind == TopologyKind::TOPO_8_PLUS_4 && param.rankSize == 12U && kernelCount == 2U
          && param.inputPtr != param.outputPtr
          && param.count == EIGHT_PLUS_FOUR_EXACT_P18_TREE_BYTES / FP32_BYTES;
    const bool useExactP11TreePush
        = resCtx.topologyKind == TopologyKind::TOPO_2X8 && param.rankSize == 16U && kernelCount == 2U
          && groupInputsFitOneBatch && param.inputPtr != param.outputPtr
          && param.count == TWO_BY_EIGHT_EXACT_P11_TREE_PUSH_BYTES / FP32_BYTES;
    const bool useExactP12TreePush
        = resCtx.topologyKind == TopologyKind::TOPO_2X8 && param.rankSize == 16U && kernelCount == 2U
          && groupInputsFitOneBatch && param.inputPtr != param.outputPtr
          && param.count == TWO_BY_EIGHT_EXACT_P12_TREE_PUSH_BYTES / FP32_BYTES;
    const bool useExactP14Nhr = kernelCount == 1U && resCtx.topologyKind == TopologyKind::TOPO_4X1
                                && param.rankSize == 4U && param.inputPtr != param.outputPtr
                                && param.count == FOUR_BY_ONE_EXACT_P14_NHR_BYTES / FP32_BYTES;
    const bool useExactP13Compact = kernelCount == 1U && resCtx.topologyKind == TopologyKind::TOPO_4X1
                                    && param.rankSize == 4U && param.inputPtr != param.outputPtr
                                    && param.count == FOUR_BY_ONE_EXACT_P13_COMPACT_BYTES / FP32_BYTES;
    const bool useExactP15Nhr = kernelCount == 1U && resCtx.topologyKind == TopologyKind::TOPO_4X1
                                && param.rankSize == 4U && param.inputPtr != param.outputPtr
                                && param.count == FOUR_BY_ONE_EXACT_P15_NHR_BYTES / FP32_BYTES;
    const bool useSingleGroupPushSmall = kernelCount == 1U
                                         && resCtx.topologyKind == TopologyKind::TOPO_4X1
                                         && param.rankSize == 4U
                                         && param.inputPtr != param.outputPtr
                                         && param.count >= param.rankSize * SINGLE_GROUP_PUSH_MIN_ELEMENTS_PER_RANK
                                         && param.count <= SINGLE_GROUP_PUSH_MAX_BYTES / FP32_BYTES;

    CcuResult ret = HcommCcuKernelRegisterStart(insHandle);
    if (ret != CCU_SUCCESS) {
        return ConvertCcuResult(ret);
    }
    // The resource context is cached under a size-independent HCCL tag.  Keep
    // both one-die algorithms registered so each invocation can select the
    // correct handle from its runtime payload size.
    const uint32_t registeredKernelCount
        = (useExactP14Nhr || useExactP13Compact || useExactP15Nhr) ? 1U : ((kernelCount == 1U) ? 2U : kernelCount + 1U);
    resCtx.ccuKernels.resize(registeredKernelCount);
    std::vector<std::shared_ptr<CcuKernelArgDirect>> registeredArgs;
    registeredArgs.reserve(registeredKernelCount);

    auto registerKernelEntry = [&](uint32_t handleIndex, uint32_t ordinal, DirectKernelPhase phase,
                                   const char *kernelName, void *kernelEntry) -> HcclResult {
        auto phaseArg = std::make_shared<CcuKernelArgDirect>(*baseKernelArgs[ordinal]);
        phaseArg->phase = phase;
        registeredArgs.push_back(phaseArg);
        const void *args[] = {phaseArg.get()};
        const CcuResult registerRet = HcommCcuKernelRegister(
            insHandle, dieIds[ordinal], kernelName, kernelEntry, args, 1, &resCtx.ccuKernels[handleIndex]);
        if (registerRet != CCU_SUCCESS) {
            HCCL_ERROR("[CCU_V002] kernel[%u] phase[%u] register failed, ret[%d]", ordinal,
                static_cast<uint32_t>(phase), registerRet);
            return ConvertCcuResult(registerRet);
        }
        return HCCL_SUCCESS;
    };
    auto registerKernel
        = [&](uint32_t handleIndex, uint32_t ordinal, DirectKernelPhase phase, const char *kernelName) -> HcclResult {
        return registerKernelEntry(handleIndex, ordinal, phase, kernelName,
            reinterpret_cast<void *>(ops_hccl::CcuDirectAllReduceKernel));
    };

    if (useExactP14Nhr) {
        CHK_RET(registerKernelEntry(0U, 0U, DirectKernelPhase::DIRECT_SINGLE_DIE,
            "CcuExactP14NhrIsolatedV042",
            reinterpret_cast<void *>(ops_hccl::CcuExactP14NhrAllReduceKernelV042)));
    } else if (useExactP13Compact) {
        CHK_RET(registerKernelEntry(0U, 0U, DirectKernelPhase::FUSED_SINGLE_DIE,
            "CcuExactP13CompactHandshakeV051",
            reinterpret_cast<void *>(ops_hccl::CcuExactP13CompactHandshakeAllReduceKernelV051)));
    } else if (useExactP15Nhr) {
        CHK_RET(registerKernelEntry(0U, 0U, DirectKernelPhase::DIRECT_SINGLE_DIE,
            "CcuExactP15NhrIsolatedV063",
            reinterpret_cast<void *>(ops_hccl::CcuExactP14NhrAllReduceKernelV042)));
    } else if (useSingleGroupPushSmall) {
        CHK_RET(registerKernelEntry(0U, 0U, DirectKernelPhase::FUSED_SINGLE_DIE,
            "CcuSingleGroupDirectPushIsolated",
            reinterpret_cast<void *>(ops_hccl::CcuSingleGroupDirectPushAllReduceKernel)));
        CHK_RET(registerKernel(1U, 0U, DirectKernelPhase::DIRECT_SINGLE_DIE, "CcuDirectSingleDieBulk"));
    } else if (useExactP16CompactPreSync) {
        for (uint32_t ordinal = 0; ordinal < kernelCount; ++ordinal) {
            CHK_RET(registerKernelEntry(ordinal, ordinal, DirectKernelPhase::PHASED,
                "CcuEightPlusFourTreeDirectPushP16CompactPreSyncV065",
                reinterpret_cast<void *>(ops_hccl::CcuEightPlusFourTreeDirectPushCompactPreSyncAllReduceKernelV065)));
        }
        CHK_RET(registerKernel(kernelCount, includeLocalOrdinal, DirectKernelPhase::MERGE, "CcuDirectMerge"));
    } else if (useExactP11TreePush || useExactP12TreePush || useExactP17TreePush || useExactP18Tree
               || useDualGroupPushSmall) {
        void *isolatedKernel = reinterpret_cast<void *>(ops_hccl::CcuDualGroupPushAllReduceKernel);
        const char *isolatedKernelName = "CcuDualGroupPushIsolated";
        if (useExactP11TreePush) {
            isolatedKernel = reinterpret_cast<void *>(ops_hccl::CcuTwoByEightTreeDirectPushAllReduceKernelV067);
            isolatedKernelName = "CcuTwoByEightTreeDirectPushExactP11V067";
        } else if (useExactP12TreePush) {
            isolatedKernel = reinterpret_cast<void *>(ops_hccl::CcuTwoByEightTreeDirectPushAllReduceKernelV068);
            isolatedKernelName = "CcuTwoByEightTreeDirectPushExactP12V068";
        } else if (resCtx.topologyKind == TopologyKind::TOPO_8_PLUS_4) {
            if (useExactP17TreePush || useExactP18Tree || useEightPlusFourLayoutSafeTreeP16) {
                isolatedKernel
                    = reinterpret_cast<void *>(ops_hccl::CcuEightPlusFourTreeDirectPushAllReduceKernelV041);
                isolatedKernelName = useExactP17TreePush ? "CcuEightPlusFourTreeDirectPushP17V064"
                                   : (useExactP18Tree ? "CcuEightPlusFourTreeDirectPushP18V066"
                                                       : "CcuEightPlusFourTreeDirectPushP16LayoutSafeV041");
            } else {
                isolatedKernel = reinterpret_cast<void *>(ops_hccl::CcuEightPlusFourDirectPushAllReduceKernel);
                isolatedKernelName = "CcuEightPlusFourDirectPushIsolated";
            }
        }
        for (uint32_t ordinal = 0; ordinal < kernelCount; ++ordinal) {
            CHK_RET(registerKernelEntry(
                ordinal, ordinal, DirectKernelPhase::PHASED, isolatedKernelName, isolatedKernel));
        }
        CHK_RET(registerKernel(kernelCount, includeLocalOrdinal, DirectKernelPhase::MERGE, "CcuDirectMerge"));
    } else if (kernelCount == 1U) {
        CHK_RET(registerKernel(0U, 0U, DirectKernelPhase::FUSED_SINGLE_DIE, "CcuDirectSingleDieGroup"));
        CHK_RET(registerKernel(1U, 0U, DirectKernelPhase::DIRECT_SINGLE_DIE, "CcuDirectSingleDieBulk"));
    } else {
        for (uint32_t ordinal = 0; ordinal < kernelCount; ++ordinal) {
            CHK_RET(registerKernel(ordinal, ordinal, DirectKernelPhase::PHASED, "CcuDirectPhasedCompact"));
        }
        CHK_RET(registerKernel(kernelCount, includeLocalOrdinal, DirectKernelPhase::MERGE, "CcuDirectMerge"));
    }
    ret = HcommCcuKernelRegisterEnd(insHandle);
    if (ret != CCU_SUCCESS) {
        return ConvertCcuResult(ret);
    }
    // Keep the registration decision in the serialized resource context.  A
    // A mock rank graph can still map more than eight peers to one local die,
    // in which case the one-group kernel is not registered.  The execution
    // side must not infer the registered kernel ABI from topology and payload
    // size alone.
    resCtx.activeKernelCount
        = kernelCount | ((useDualGroupPushSmall && !useExactP16CompactPreSync) ? ISOLATED_DUAL_GROUP_RESOURCE_FLAG : 0U)
          | (useExactP14Nhr ? ISOLATED_EXACT_P14_NHR_RESOURCE_FLAG : 0U)
          | (useExactP13Compact ? ISOLATED_EXACT_P13_COMPACT_RESOURCE_FLAG : 0U)
          | (useExactP15Nhr ? ISOLATED_EXACT_P15_NHR_RESOURCE_FLAG : 0U)
          | (useExactP17TreePush ? ISOLATED_EXACT_P17_TREE_PUSH_RESOURCE_FLAG : 0U)
          | (useExactP16CompactPreSync ? ISOLATED_EXACT_P16_COMPACT_PRESYNC_RESOURCE_FLAG : 0U)
          | (useExactP18Tree ? ISOLATED_EXACT_P18_TREE_RESOURCE_FLAG : 0U)
          | (useExactP11TreePush ? ISOLATED_EXACT_P11_TREE_PUSH_RESOURCE_FLAG : 0U)
          | (useExactP12TreePush ? ISOLATED_EXACT_P12_TREE_PUSH_RESOURCE_FLAG : 0U);
    return HCCL_SUCCESS;
}

HcclResult BuildResourceContext(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    void *buffer = nullptr;
    uint64_t bufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &buffer, &bufferSize));
    resCtx.localBuffer = CommBuffer{buffer, bufferSize};
    CHK_RET(DetectTopology(comm, param.rankSize, resCtx.topologyKind));

    if (param.rankSize == 1U) {
        resCtx.ccuThread = param.cpuThread;
        resCtx.threads.push_back(param.cpuThread);
        return HCCL_SUCCESS;
    }

    ChannelGroups groups;
    CHK_RET(BuildChannelGroups(comm, param, groups));
    return AcquireChannelsAndRegisterKernels(comm, param, groups, resCtx);
}

} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("[CCU_V001] only FP32 is supported by the finals path"),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(
        op != HCCL_REDUCE_SUM, HCCL_ERROR("[CCU_V001] only SUM is supported by the finals path"), HCCL_E_NOT_SUPPORT);

    // 构造算子参数
    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.reduceType = op;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0U || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[CCU_V001] invalid rank size[%u]", param.rankSize), HCCL_E_NOT_SUPPORT);
    const bool useEightPlusFourLayoutSafeTreeP16Tag
        = param.rankSize == 12U && param.inputPtr != param.outputPtr
          && param.count == EIGHT_PLUS_FOUR_LAYOUT_SAFE_TREE_P16_BYTES / FP32_BYTES;
    const bool useExactP16CompactPreSyncTag
        = param.rankSize == 12U && param.inputPtr != param.outputPtr
          && param.count == EIGHT_PLUS_FOUR_EXACT_P16_COMPACT_PRESYNC_BYTES / FP32_BYTES;
    const bool useExactP17TreePushTag
        = param.rankSize == 12U && param.inputPtr != param.outputPtr
          && param.count == EIGHT_PLUS_FOUR_EXACT_P17_TREE_PUSH_BYTES / FP32_BYTES;
    const bool useExactP18TreeTag
        = param.rankSize == 12U && param.inputPtr != param.outputPtr
          && param.count == EIGHT_PLUS_FOUR_EXACT_P18_TREE_BYTES / FP32_BYTES;
    const bool useExactP11TreePushTag
        = param.rankSize == 16U && param.inputPtr != param.outputPtr
          && param.count == TWO_BY_EIGHT_EXACT_P11_TREE_PUSH_BYTES / FP32_BYTES;
    const bool useExactP12TreePushTag
        = param.rankSize == 16U && param.inputPtr != param.outputPtr
          && param.count == TWO_BY_EIGHT_EXACT_P12_TREE_PUSH_BYTES / FP32_BYTES;
    const bool useExactP14NhrTag = param.rankSize == 4U && param.inputPtr != param.outputPtr
                                   && param.count == FOUR_BY_ONE_EXACT_P14_NHR_BYTES / FP32_BYTES;
    const bool useExactP13CompactTag = param.rankSize == 4U && param.inputPtr != param.outputPtr
                                       && param.count == FOUR_BY_ONE_EXACT_P13_COMPACT_BYTES / FP32_BYTES;
    const bool useExactP15NhrTag = param.rankSize == 4U && param.inputPtr != param.outputPtr
                                   && param.count == FOUR_BY_ONE_EXACT_P15_NHR_BYTES / FP32_BYTES;
    const bool useDualGroupPushSmallTag = (param.rankSize == 16U || param.rankSize == 12U)
                                          && param.inputPtr != param.outputPtr
                                          && param.count >= param.rankSize * DUAL_GROUP_PUSH_MIN_ELEMENTS_PER_RANK
                                          && param.count <= DUAL_GROUP_PUSH_MAX_BYTES / FP32_BYTES;
    const bool useSingleGroupPushSmallTag = param.rankSize == 4U
                                            && param.inputPtr != param.outputPtr
                                            && param.count >= param.rankSize * SINGLE_GROUP_PUSH_MIN_ELEMENTS_PER_RANK
                                            && param.count <= SINGLE_GROUP_PUSH_MAX_BYTES / FP32_BYTES;
    const char *resourceTag = "hccl_custom_allreduce_ccu_v001";
    if (useExactP14NhrTag) {
        resourceTag = "hccl_custom_allreduce_ccu_v042_exact_4x1_nhr_p14";
    } else if (useExactP11TreePushTag) {
        resourceTag = "hccl_custom_allreduce_ccu_v067_exact_2x8_tree_push_p11";
    } else if (useExactP12TreePushTag) {
        resourceTag = "hccl_custom_allreduce_ccu_v068_exact_2x8_tree_push_p12";
    } else if (useExactP17TreePushTag) {
        resourceTag = "hccl_custom_allreduce_ccu_v064_exact_8plus4_two_window_tree_push_p17";
    } else if (useExactP18TreeTag) {
        resourceTag = "hccl_custom_allreduce_ccu_v066_exact_8plus4_p18_tree";
    } else if (useExactP13CompactTag) {
        resourceTag = "hccl_custom_allreduce_ccu_v051_exact_4x1_p13_compact";
    } else if (useExactP15NhrTag) {
        resourceTag = "hccl_custom_allreduce_ccu_v063_exact_4x1_nhr_p15";
    } else if (useExactP16CompactPreSyncTag) {
        resourceTag = "hccl_custom_allreduce_ccu_v065_exact_8plus4_p16_compact_presync";
    } else if (useSingleGroupPushSmallTag) {
        resourceTag = "hccl_custom_allreduce_ccu_v030_isolated_4x1_direct_push_small";
    } else if (useEightPlusFourLayoutSafeTreeP16Tag) {
        resourceTag = "hccl_custom_allreduce_ccu_v041_layout_safe_8plus4_tree_p16";
    } else if (useDualGroupPushSmallTag) {
        resourceTag = param.rankSize == 16U
                          ? "hccl_custom_allreduce_ccu_v027_isolated_dual_group_push_small"
                          : "hccl_custom_allreduce_ccu_v028_isolated_8plus4_direct_push_small";
    }
    const int tagRet = sprintf_s(param.tag, sizeof(param.tag), "%s", resourceTag);
    CHK_PRT_RET(tagRet <= 0, HCCL_ERROR("[CCU_V028] failed to build resource tag"), HCCL_E_INTERNAL);

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread，并申请 1 个 notify
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
        AlgResourceCtx resCtxHost;
        CHK_RET(BuildResourceContext(comm, param, resCtxHost));

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
