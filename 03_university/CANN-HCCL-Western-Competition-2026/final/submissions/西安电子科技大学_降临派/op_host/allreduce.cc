/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include <hcomm/ccu/ccu_launch.h>
#include <hcomm/ccu/ccu_res.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <vector>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
// 不同模式的 Kernel 和任务参数 ABI 不兼容，必须使用独立的执行资源缓存标识。
constexpr char SMALL_OWNER_CONTEXT_TAG[] = "hccl_custom_allreduce_final_small_owner_pull_v1";
constexpr char SMALL_4X1_CONTEXT_TAG[] = "hccl_custom_allreduce_final_small_4x1_oneshot_v1";
constexpr char LARGE_LATIN_CONTEXT_TAG[] = "hccl_custom_allreduce_final_latin_v2_1";
constexpr char LARGE_PACKED_CONTEXT_TAG[] = "hccl_custom_allreduce_final_2x8_packed_pull_v1";
constexpr char LARGE_4X1_CONTEXT_TAG[] = "hccl_custom_allreduce_final_4x1_fused_rsag_v1";
constexpr uint32_t SMALL_CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t LARGE_CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t DIE_NUM = 2;
constexpr uint64_t EXACT_512M_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t EXACT_400M4_BYTES = 400ULL * 1024ULL * 1024ULL + sizeof(float);

bool Contains(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

HcclResult QueryTopology(HcclComm comm, uint32_t myRank, uint32_t rankSize, uint32_t &meshLayer, uint32_t &closLayer,
    TopologyKind &kind, std::vector<uint32_t> &localGroup, std::vector<uint32_t> &otherGroup)
{
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    bool meshFound = false;
    bool closFound = false;
    for (uint32_t index = 0; index < layerCount; ++index) {
        CommTopo topo = COMM_TOPO_RESERVED;
        CHK_RET(HcclRankGraphGetTopoTypeByLayer(comm, layers[index], &topo));
        if (!meshFound && topo == COMM_TOPO_1DMESH) {
            meshLayer = layers[index];
            meshFound = true;
        }
        if (!closFound && topo == COMM_TOPO_CLOS) {
            closLayer = layers[index];
            closFound = true;
        }
    }

    if (rankSize == 4U) {
        localGroup = {myRank};
        meshFound = true;
    } else if (!meshFound) {
        uint32_t bestSize = rankSize;
        for (uint32_t index = 0; index < layerCount; ++index) {
            uint32_t *ranks = nullptr;
            uint32_t rankCount = 0;
            CHK_RET(HcclRankGraphGetRanksByLayer(comm, layers[index], &ranks, &rankCount));
            std::vector<uint32_t> candidate(ranks, ranks + rankCount);
            if (rankCount > 0U && rankCount < bestSize && Contains(candidate, myRank)) {
                meshLayer = layers[index];
                bestSize = rankCount;
                meshFound = true;
            }
        }
    }
    CHK_PRT_RET(!meshFound, HCCL_ERROR("required Mesh layer was not found"), HCCL_E_NOT_SUPPORT);

    if (localGroup.empty()) {
        uint32_t *ranks = nullptr;
        uint32_t rankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, meshLayer, &ranks, &rankCount));
        localGroup.assign(ranks, ranks + rankCount);
    }
    std::sort(localGroup.begin(), localGroup.end());
    CHK_PRT_RET(
        !Contains(localGroup, myRank), HCCL_ERROR("rank %u is absent from its local group", myRank), HCCL_E_INTERNAL);

    if (!closFound) {
        for (uint32_t index = 0; index < layerCount && !closFound; ++index) {
            uint32_t *ranks = nullptr;
            uint32_t rankCount = 0;
            CHK_RET(HcclRankGraphGetRanksByLayer(comm, layers[index], &ranks, &rankCount));
            std::vector<uint32_t> candidate(ranks, ranks + rankCount);
            if (rankCount == rankSize && Contains(candidate, myRank)) {
                closLayer = layers[index];
                closFound = true;
            }
        }
    }
    if (!closFound) {
        for (uint32_t index = 0; index < layerCount && !closFound; ++index) {
            for (uint32_t rank = 0; rank < rankSize; ++rank) {
                if (Contains(localGroup, rank)) {
                    continue;
                }
                CommLink *links = nullptr;
                uint32_t linkCount = 0;
                CHK_RET(HcclRankGraphGetLinks(comm, layers[index], myRank, rank, &links, &linkCount));
                if (linkCount != 0U) {
                    closLayer = layers[index];
                    closFound = true;
                    break;
                }
            }
        }
    }
    CHK_PRT_RET(!closFound, HCCL_ERROR("required Clos layer was not found"), HCCL_E_NOT_SUPPORT);

    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (!Contains(localGroup, rank)) {
            otherGroup.push_back(rank);
        }
    }
    std::sort(otherGroup.begin(), otherGroup.end());
    if (rankSize == 16U && localGroup.size() == 8U && otherGroup.size() == 8U) {
        kind = TopologyKind::TWO_BY_EIGHT;
    } else if (rankSize == 4U && localGroup.size() == 1U) {
        kind = TopologyKind::FOUR_BY_ONE;
    } else if (rankSize == 12U
               && ((localGroup.size() == 8U && otherGroup.size() == 4U)
                   || (localGroup.size() == 4U && otherGroup.size() == 8U))) {
        kind = TopologyKind::EIGHT_PLUS_FOUR;
    } else {
        HCCL_ERROR("unsupported topology: rankSize=%u localGroup=%zu otherGroup=%zu", rankSize, localGroup.size(),
            otherGroup.size());
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

KernelMode SelectKernelMode(const OpParam &param)
{
    const uint64_t totalBytes = param.count * sizeof(float);
    if (totalBytes == small_data::TOTAL_BYTES) {
        return param.rankSize == fused_4x1::RANK_SIZE ? KernelMode::SMALL_4X1_ONESHOT : KernelMode::SMALL_OWNER_PULL;
    }
    if (param.rankSize == fused_4x1::RANK_SIZE) {
        return KernelMode::LARGE_4X1_FUSED_RSAG;
    }
    // Packed endpoint-pull 依赖独立的输入、输出区，仅对已验证的数据量注册专用资源。
    if (param.rankSize == 16U && param.inputPtr != param.outputPtr
        && (totalBytes == EXACT_512M_BYTES || totalBytes == EXACT_400M4_BYTES)) {
        return KernelMode::LARGE_2X8_PACKED_PULL;
    }
    return KernelMode::LARGE_LATIN;
}

const char *ContextTag(KernelMode mode)
{
    switch (mode) {
        case KernelMode::SMALL_OWNER_PULL:
            return SMALL_OWNER_CONTEXT_TAG;
        case KernelMode::SMALL_4X1_ONESHOT:
            return SMALL_4X1_CONTEXT_TAG;
        case KernelMode::LARGE_2X8_PACKED_PULL:
            return LARGE_PACKED_CONTEXT_TAG;
        case KernelMode::LARGE_4X1_FUSED_RSAG:
            return LARGE_4X1_CONTEXT_TAG;
        default:
            return LARGE_LATIN_CONTEXT_TAG;
    }
}

void BuildAllPeers(uint32_t myRank, uint32_t rankSize, bool cyclic, std::vector<uint32_t> &peers)
{
    if (cyclic) {
        for (uint32_t step = 1U; step < rankSize; ++step) {
            peers.push_back((myRank + step) % rankSize);
        }
        return;
    }
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (rank != myRank) {
            peers.push_back(rank);
        }
    }
}

HcclResult SelectChannelLink(HcclComm comm, uint32_t myRank, uint32_t peer, uint32_t layer, uint32_t requiredDie,
    CommLink &selected, bool &found)
{
    CommLink *links = nullptr;
    uint32_t linkCount = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, layer, myRank, peer, &links, &linkCount));
    found = false;
    for (CommProtocol protocol : {COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UBC_TP}) {
        for (uint32_t index = 0; index < linkCount; ++index) {
            if (links[index].linkAttr.linkProtocol != protocol) {
                continue;
            }
            if (requiredDie < DIE_NUM) {
                EndpointAttrDieId dieId = 0;
                CHK_RET(HcclRankGraphGetEndpointInfo(
                    comm, myRank, &links[index].srcEndpointDesc, ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
                if (dieId != requiredDie) {
                    continue;
                }
            }
            selected = links[index];
            found = true;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult FillChannelDesc(uint32_t peer, const CommLink &link, uint32_t notifyNum, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = peer;
    desc.notifyNum = notifyNum;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint = link.srcEndpointDesc;
    desc.remoteEndpoint = link.dstEndpointDesc;
    return HCCL_SUCCESS;
}

HcclResult BuildLargeChannelDesc(HcclComm comm, uint32_t myRank, uint32_t peer, uint32_t layer, HcclChannelDesc &desc)
{
    CommLink selected{};
    bool found = false;
    CHK_RET(SelectChannelLink(comm, myRank, peer, layer, DIE_NUM, selected, found));
    CHK_PRT_RET(!found, HCCL_ERROR("no CCU channel from rank %u to %u", myRank, peer), HCCL_E_NOT_FOUND);
    return FillChannelDesc(peer, selected, LARGE_CHANNEL_NOTIFY_NUM, desc);
}

HcclResult BuildCommonClosDescs(HcclComm comm, uint32_t myRank, const std::vector<uint32_t> &peers, uint32_t closLayer,
    uint32_t notifyNum, std::vector<HcclChannelDesc> &descs, uint32_t &selectedDie)
{
    // 单个 CCU Kernel 只能使用同一 Die 的 Channel，因此所有 Peer 必须能从同一 Die 到达。
    for (uint32_t die = 0; die < DIE_NUM; ++die) {
        std::vector<HcclChannelDesc> candidates(peers.size());
        bool complete = true;
        for (size_t index = 0; index < peers.size(); ++index) {
            CommLink selected{};
            bool found = false;
            CHK_RET(SelectChannelLink(comm, myRank, peers[index], closLayer, die, selected, found));
            if (!found) {
                complete = false;
                break;
            }
            CHK_RET(FillChannelDesc(peers[index], selected, notifyNum, candidates[index]));
        }
        if (complete) {
            descs = std::move(candidates);
            selectedDie = die;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("rank %u has no common-die Clos channel set", myRank);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult GetChannelDieId(HcclComm comm, uint32_t myRank, const HcclChannelDesc &desc, uint32_t &dieId)
{
    EndpointAttrDieId endpointDieId = 0;
    CHK_RET(HcclRankGraphGetEndpointInfo(
        comm, myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId), &endpointDieId));
    CHK_PRT_RET(endpointDieId >= DIE_NUM, HCCL_ERROR("invalid CCU die id %u", endpointDieId), HCCL_E_INTERNAL);
    dieId = endpointDieId;
    return HCCL_SUCCESS;
}

HcclResult QueryCcuInstance(HcclComm comm, CcuInsHandle &insHandle)
{
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    CHK_PRT_RET(insCount != 1U, HCCL_ERROR("expected one CCU instance, got %u", insCount), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RegisterSmallKernel(HcclComm comm, const OpParam &param, KernelMode mode, uint32_t registrationDie,
    const std::vector<uint32_t> &peers, const std::vector<ChannelHandle> &channels, AlgResourceCtx &resources)
{
    SmallKernelArg kernelArg{};
    kernelArg.channelCount = static_cast<uint32_t>(channels.size());
    kernelArg.myRank = param.myRank;
    kernelArg.rankSize = param.rankSize;
    kernelArg.mode = mode;
    for (size_t index = 0; index < channels.size(); ++index) {
        kernelArg.channels[index] = channels[index];
        kernelArg.peers[index] = peers[index];
    }

    uint64_t requiredScratch = 0;
    if (mode == KernelMode::SMALL_OWNER_PULL) {
        const uint64_t normalElements = param.count / param.rankSize;
        const uint64_t lastElements = param.count - normalElements * (param.rankSize - 1U);
        const uint64_t normalBytes = normalElements * sizeof(float);
        const uint64_t lastBytes = lastElements * sizeof(float);
        kernelArg.ownerOffset = normalBytes * param.myRank;
        kernelArg.ownerBytes = param.myRank + 1U == param.rankSize ? lastBytes : normalBytes;
        requiredScratch = lastBytes * param.rankSize;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            kernelArg.localScratch[rank] = resources.scratchAddr + lastBytes * rank;
        }
    } else {
        CHK_PRT_RET(
            param.rankSize != fused_4x1::RANK_SIZE, HCCL_ERROR("4x1 OneShot requires four ranks"), HCCL_E_INTERNAL);
        kernelArg.smallBytes = param.count * sizeof(float);
        requiredScratch = kernelArg.smallBytes * fused_4x1::RANK_SIZE;
        for (uint32_t rank = 0; rank < fused_4x1::RANK_SIZE; ++rank) {
            kernelArg.localScratch[rank] = resources.scratchAddr + kernelArg.smallBytes * rank;
        }
        for (size_t index = 0; index < channels.size(); ++index) {
            void *remoteScratch = nullptr;
            uint64_t remoteScratchSize = 0;
            CHK_RET(HcclChannelGetHcclBuffer(comm, channels[index], &remoteScratch, &remoteScratchSize));
            CHK_PTR_NULL(remoteScratch);
            CHK_PRT_RET(remoteScratchSize < requiredScratch,
                HCCL_ERROR("remote 4x1 scratch is too small for rank %u", peers[index]), HCCL_E_UNAVAIL);
            kernelArg.remoteScratch[index]
                = reinterpret_cast<uint64_t>(remoteScratch) + kernelArg.smallBytes * param.myRank;
        }
    }
    CHK_PRT_RET(resources.scratchAddr == 0 || resources.scratchSize < requiredScratch,
        HCCL_ERROR("small scratch is too small: required=%lu available=%lu",
            static_cast<unsigned long>(requiredScratch), static_cast<unsigned long>(resources.scratchSize)),
        HCCL_E_UNAVAIL);
    CHK_RET_CCU(HcommCcuGetMemToken(resources.scratchAddr, requiredScratch, &resources.scratchToken));

    CcuInsHandle insHandle = 0;
    CHK_RET(QueryCcuInstance(comm, insHandle));
    resources.kernels.assign(1U, 0);
    char name[64];
    int length = std::snprintf(
        name, sizeof(name), "FinalSmall_%u_%u_%u", static_cast<uint32_t>(mode), param.myRank, registrationDie);
    CHK_PRT_RET(length < 0 || static_cast<size_t>(length) >= sizeof(name),
        HCCL_ERROR("failed to construct small kernel name"), HCCL_E_INTERNAL);
    const void *kernelArgs[] = {&kernelArg};
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, registrationDie, name,
        reinterpret_cast<const void *>(ops_hccl::SmallAllReduceKernel), kernelArgs, 1, &resources.kernels[0]));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

void FillLargeKernelArg(uint32_t myRank, uint32_t rankSize, uint32_t die, uint32_t primaryDie, KernelMode mode,
    const std::vector<uint32_t> &peers, const std::vector<ChannelHandle> &channels, LargeKernelArg &arg)
{
    arg.channelCount = static_cast<uint32_t>(channels.size());
    arg.dieId = die;
    arg.myRank = myRank;
    arg.rankSize = rankSize;
    arg.primaryDie = primaryDie;
    arg.mode = mode;
    for (size_t index = 0; index < channels.size(); ++index) {
        arg.channels[index] = channels[index];
        arg.peers[index] = peers[index];
    }
}

HcclResult RegisterLargeKernels(HcclComm comm, uint32_t myRank, uint32_t rankSize, uint32_t primaryDie, KernelMode mode,
    const std::array<std::vector<uint32_t>, DIE_NUM> &peers,
    const std::array<std::vector<ChannelHandle>, DIE_NUM> &channels, AlgResourceCtx &resources)
{
    std::array<LargeKernelArg, DIE_NUM> kernelArgs{};
    for (uint32_t die = 0; die < DIE_NUM; ++die) {
        if (!channels[die].empty()) {
            FillLargeKernelArg(myRank, rankSize, die, primaryDie, mode, peers[die], channels[die], kernelArgs[die]);
        }
    }
    CcuInsHandle insHandle = 0;
    CHK_RET(QueryCcuInstance(comm, insHandle));
    resources.kernels.assign(DIE_NUM, 0);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    for (uint32_t die = 0; die < DIE_NUM; ++die) {
        if (channels[die].empty()) {
            continue;
        }
        char name[64];
        int length = std::snprintf(name, sizeof(name), "FinalLarge_%u_%u_%u", static_cast<uint32_t>(mode), myRank, die);
        CHK_PRT_RET(length < 0 || static_cast<size_t>(length) >= sizeof(name),
            HCCL_ERROR("failed to construct large kernel name"), HCCL_E_INTERNAL);
        const void *args[] = {&kernelArgs[die]};
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, die, name,
            reinterpret_cast<const void *>(ops_hccl::LargeAllReduceKernel), args, 1, &resources.kernels[die]));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult CreateSmallResources(
    HcclComm comm, const OpParam &param, KernelMode mode, uint32_t closLayer, AlgResourceCtx &resources)
{
    std::vector<uint32_t> peers;
    BuildAllPeers(param.myRank, param.rankSize, mode == KernelMode::SMALL_OWNER_PULL, peers);
    std::vector<HcclChannelDesc> descs(peers.size());
    uint32_t registrationDie = 0;
    CHK_RET(
        BuildCommonClosDescs(comm, param.myRank, peers, closLayer, SMALL_CHANNEL_NOTIFY_NUM, descs, registrationDie));
    std::vector<ChannelHandle> channels(peers.size());
    CHK_RET(
        HcclChannelAcquire(comm, COMM_ENGINE_CCU, descs.data(), static_cast<uint32_t>(descs.size()), channels.data()));
    return RegisterSmallKernel(comm, param, mode, registrationDie, peers, channels, resources);
}

HcclResult CreateLarge4x1Resources(HcclComm comm, const OpParam &param, uint32_t closLayer, AlgResourceCtx &resources)
{
    std::vector<uint32_t> peers;
    BuildAllPeers(param.myRank, param.rankSize, false, peers);
    std::vector<HcclChannelDesc> descs(peers.size());
    uint32_t registrationDie = 0;
    CHK_RET(
        BuildCommonClosDescs(comm, param.myRank, peers, closLayer, LARGE_CHANNEL_NOTIFY_NUM, descs, registrationDie));
    std::vector<ChannelHandle> channels(peers.size());
    CHK_RET(
        HcclChannelAcquire(comm, COMM_ENGINE_CCU, descs.data(), static_cast<uint32_t>(descs.size()), channels.data()));
    std::array<std::vector<uint32_t>, DIE_NUM> peersByDie;
    std::array<std::vector<ChannelHandle>, DIE_NUM> channelsByDie;
    peersByDie[registrationDie] = peers;
    channelsByDie[registrationDie] = channels;
    CHK_RET(RegisterLargeKernels(comm, param.myRank, param.rankSize, registrationDie, KernelMode::LARGE_4X1_FUSED_RSAG,
        peersByDie, channelsByDie, resources));
    const CcuKernelHandle kernel = resources.kernels[registrationDie];
    resources.kernels.assign(1U, kernel);
    return HCCL_SUCCESS;
}

HcclResult CreateDualDieLargeResources(HcclComm comm, const OpParam &param, KernelMode mode,
    const std::vector<uint32_t> &localGroup, uint32_t meshLayer, uint32_t closLayer, AlgResourceCtx &resources)
{
    std::vector<uint32_t> peers;
    BuildAllPeers(param.myRank, param.rankSize, false, peers);
    std::vector<HcclChannelDesc> descs(peers.size());
    for (size_t index = 0; index < peers.size(); ++index) {
        uint32_t layer = Contains(localGroup, peers[index]) ? meshLayer : closLayer;
        CHK_RET(BuildLargeChannelDesc(comm, param.myRank, peers[index], layer, descs[index]));
    }
    std::vector<ChannelHandle> channels(peers.size());
    CHK_RET(
        HcclChannelAcquire(comm, COMM_ENGINE_CCU, descs.data(), static_cast<uint32_t>(descs.size()), channels.data()));

    std::array<std::vector<uint32_t>, DIE_NUM> peersByDie;
    std::array<std::vector<ChannelHandle>, DIE_NUM> channelsByDie;
    std::vector<uint32_t> peerDie(param.rankSize, DIE_NUM);
    std::array<uint32_t, DIE_NUM> closCounts{};
    // 按物理端点 Die 拆分已获取链路，每个子集注册为独立的 CCU Kernel。
    for (size_t index = 0; index < peers.size(); ++index) {
        uint32_t die = 0;
        CHK_RET(GetChannelDieId(comm, param.myRank, descs[index], die));
        peerDie[peers[index]] = die;
        peersByDie[die].push_back(peers[index]);
        channelsByDie[die].push_back(channels[index]);
        closCounts[die] += Contains(localGroup, peers[index]) ? 0U : 1U;
    }

    uint32_t primaryDie = !channelsByDie[0].empty() ? 0U : 1U;
    if (param.rankSize == 16U) {
        // 配对的跨 Server 端点负责本地结果分片的初始化和最终合并。
        CHK_PRT_RET(peerDie[param.myRank ^ 8U] >= DIE_NUM, HCCL_ERROR("missing paired Clos channel"), HCCL_E_INTERNAL);
        primaryDie = peerDie[param.myRank ^ 8U];
    } else if (param.rankSize == 12U) {
        primaryDie = channelsByDie[1].size() > channelsByDie[0].size() ? 1U : 0U;
    }
    resources.primaryDie = primaryDie;
    resources.launchFirstDie = closCounts[1] > closCounts[0] ? 1U : 0U;
    for (uint32_t die = 0; die < DIE_NUM; ++die) {
        resources.channelCounts[die] = static_cast<uint32_t>(channelsByDie[die].size());
    }
    return RegisterLargeKernels(
        comm, param.myRank, param.rankSize, primaryDie, mode, peersByDie, channelsByDie, resources);
}

HcclResult CreateResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resources)
{
    uint32_t meshLayer = 0;
    uint32_t closLayer = 0;
    TopologyKind kind = TopologyKind::TWO_BY_EIGHT;
    std::vector<uint32_t> localGroup;
    std::vector<uint32_t> otherGroup;
    CHK_RET(QueryTopology(comm, param.myRank, param.rankSize, meshLayer, closLayer, kind, localGroup, otherGroup));
    CHK_PRT_RET((kind == TopologyKind::TWO_BY_EIGHT && param.rankSize != 16U)
                    || (kind == TopologyKind::FOUR_BY_ONE && param.rankSize != 4U)
                    || (kind == TopologyKind::EIGHT_PLUS_FOUR && param.rankSize != 12U),
        HCCL_ERROR("rank size does not match topology"), HCCL_E_INTERNAL);
    resources.mode = SelectKernelMode(param);
    if (resources.mode == KernelMode::SMALL_OWNER_PULL || resources.mode == KernelMode::SMALL_4X1_ONESHOT) {
        return CreateSmallResources(comm, param, resources.mode, closLayer, resources);
    }
    if (resources.mode == KernelMode::LARGE_4X1_FUSED_RSAG) {
        return CreateLarge4x1Resources(comm, param, closLayer, resources);
    }
    return CreateDualDieLargeResources(comm, param, resources.mode, localGroup, meshLayer, closLayer, resources);
}
} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM, HCCL_ERROR("only FP32 SUM is supported"),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count == 0U || count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("invalid FP32 count %lu", static_cast<unsigned long>(count)), HCCL_E_PARA);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.reduceType = op;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != 4U && param.rankSize != 12U && param.rankSize != 16U,
        HCCL_ERROR("unsupported rank size %u", param.rankSize), HCCL_E_NOT_SUPPORT);
    const KernelMode mode = SelectKernelMode(param);
    const int tagLength = std::snprintf(param.tag, sizeof(param.tag), "%s", ContextTag(mode));
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("failed to construct resource tag"), HCCL_E_INTERNAL);

    CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    void *context = nullptr;
    uint64_t contextSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, engine, &context, &contextSize) == HCCL_SUCCESS) {
        param.resCtx = context;
        param.ctxSize = contextSize;
    } else {
        // Channel、Thread 和 Kernel 注册结果按模式缓存，后续调用直接复用。
        AlgResourceCtx resources{};
        resources.mode = mode;
        void *scratch = nullptr;
        CHK_RET(HcclGetHcclBuffer(comm, &scratch, &resources.scratchSize));
        CHK_PTR_NULL(scratch);
        resources.scratchAddr = reinterpret_cast<uint64_t>(scratch);
        resources.scratchToken = 0;
        const bool dualDie = mode == KernelMode::LARGE_LATIN || mode == KernelMode::LARGE_2X8_PACKED_PULL;
        resources.ccuThreads.resize(dualDie ? DIE_NUM : 1U);
        CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream, 1, &resources.ccuThreads[0]));
        if (dualDie) {
            ThreadConfig slaveConfig;
            CHK_RET(static_cast<HcclResult>(ThreadConfigInit(&slaveConfig, 1)));
            slaveConfig.notifyNumPerThread = 1;
            CHK_RET(HcclThreadAcquireWithConfig(
                comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &slaveConfig, &resources.ccuThreads[1]));
        }
        CHK_RET(CreateResources(comm, param, resources));
        std::vector<char> sequence = resources.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, sequence.data(), sequence.size(), 0));
    }
    return ops_hccl::ExecOp(param);
}
