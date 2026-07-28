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
#include <cstdint>
#include <cstdio>
#include <limits>
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
constexpr uint32_t THREAD_NOTIFY_NUM = 1;
constexpr uint32_t ASCEND950_DIE_NUM = 2;
constexpr uint32_t ASYMMETRIC_RANK_SIZE = 12;
constexpr uint32_t SERVER_RANK_NUM = 8;
constexpr uint32_t SMALL_SERVER_RANK_NUM =
    ASYMMETRIC_RANK_SIZE - SERVER_RANK_NUM;
constexpr uint32_t MIXED_NET_LAYER = std::numeric_limits<uint32_t>::max();

struct PairChannelGroup {
    uint32_t dieId = 0;
    uint32_t netLayer = MIXED_NET_LAYER;
    std::vector<uint32_t> peers;
    std::vector<ChannelHandle> channels;
};

bool FindRank12ClosGroup(const OpParam &param,
    const std::vector<PairChannelGroup> &groups, size_t &closGroupIndex)
{
    if (param.rankSize != ASYMMETRIC_RANK_SIZE ||
        groups.size() != ASCEND950_DIE_NUM) {
        return false;
    }

    size_t meshGroupIndex = groups.size();
    closGroupIndex = groups.size();
    bool onLargeServer = param.myRank < SERVER_RANK_NUM;
    size_t expectedMeshPeers = onLargeServer ?
        SERVER_RANK_NUM - 1 : SMALL_SERVER_RANK_NUM - 1;
    size_t expectedClosPeers = onLargeServer ?
        SMALL_SERVER_RANK_NUM : SERVER_RANK_NUM;

    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const PairChannelGroup &group = groups[groupIndex];
        bool allLocal = std::all_of(group.peers.begin(), group.peers.end(),
            [&](uint32_t peer) {
                return (peer < SERVER_RANK_NUM) == onLargeServer;
            });
        bool allRemote = std::all_of(group.peers.begin(), group.peers.end(),
            [&](uint32_t peer) {
                return (peer < SERVER_RANK_NUM) != onLargeServer;
            });
        if (group.peers.size() == expectedMeshPeers && allLocal) {
            meshGroupIndex = groupIndex;
        }
        if (group.peers.size() == expectedClosPeers && allRemote) {
            bool exactRemoteSet = true;
            uint32_t firstRemoteRank =
                onLargeServer ? SERVER_RANK_NUM : 0;
            for (size_t peerIndex = 0;
                peerIndex < group.peers.size(); ++peerIndex) {
                if (group.peers[peerIndex] != firstRemoteRank +
                    static_cast<uint32_t>(peerIndex)) {
                    exactRemoteSet = false;
                    break;
                }
            }
            if (exactRemoteSet) {
                closGroupIndex = groupIndex;
            }
        }
    }
    return meshGroupIndex < groups.size() &&
        closGroupIndex < groups.size() && meshGroupIndex != closGroupIndex;
}

bool FindRank16ClosGroup(const OpParam &param,
    const std::vector<PairChannelGroup> &groups, size_t &closGroupIndex)
{
    if (param.rankSize != MAX_RANK_SIZE ||
        groups.size() != ASCEND950_DIE_NUM) {
        return false;
    }

    size_t meshGroupIndex = groups.size();
    closGroupIndex = groups.size();
    bool onLowServer = param.myRank < SERVER_RANK_NUM;
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const PairChannelGroup &group = groups[groupIndex];
        bool allLocal = std::all_of(group.peers.begin(), group.peers.end(),
            [&](uint32_t peer) {
                return (peer < SERVER_RANK_NUM) == onLowServer;
            });
        bool allRemote = std::all_of(group.peers.begin(), group.peers.end(),
            [&](uint32_t peer) {
                return (peer < SERVER_RANK_NUM) != onLowServer;
            });
        if (group.peers.size() == SERVER_RANK_NUM - 1 && allLocal) {
            meshGroupIndex = groupIndex;
        }
        if (group.peers.size() == SERVER_RANK_NUM && allRemote) {
            closGroupIndex = groupIndex;
        }
    }
    return meshGroupIndex < groups.size() &&
        closGroupIndex < groups.size() && meshGroupIndex != closGroupIndex;
}

HcclResult BuildPairChannelDesc(
    HcclComm comm, uint32_t srcRank, uint32_t dstRank,
    HcclChannelDesc &desc, uint32_t &localDieId, uint32_t &selectedNetLayer)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("Rank graph contains no network layer"), HCCL_E_INTERNAL);
    // Rank-graph query buffers are owned by HCCL and may be reused by later
    // graph queries. Preserve the layer IDs before asking for link lists.
    std::vector<uint32_t> layers(netLayers, netLayers + netLayerNum);

    for (uint32_t layerIndex = 0; layerIndex < netLayerNum; ++layerIndex) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult result =
            HcclRankGraphGetLinks(comm, layers[layerIndex], srcRank, dstRank, &links, &linkNum);
        if (result != HCCL_SUCCESS || links == nullptr) {
            continue;
        }

        for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
            const CommLink &link = links[linkIndex];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = dstRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
            EndpointAttrDieId endpointDieId = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, srcRank, &desc.localEndpoint,
                ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId), &endpointDieId));
            CHK_PRT_RET(endpointDieId >= ASCEND950_DIE_NUM,
                HCCL_ERROR("Invalid local endpoint die[%u]", endpointDieId), HCCL_E_INTERNAL);
            localDieId = endpointDieId;
            selectedNetLayer = layers[layerIndex];

            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("No UBC_CTP link between rank[%u] and rank[%u]", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

void BuildAllPeerList(const OpParam &param, std::vector<uint32_t> &peers)
{
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank) {
            peers.push_back(peer);
        }
    }
}

HcclResult AcquirePairChannels(HcclComm comm, const OpParam &param,
    std::vector<uint32_t> &peers, std::vector<ChannelHandle> &channels,
    std::vector<uint32_t> &channelDieIds, std::vector<uint32_t> &channelNetLayers)
{
    BuildAllPeerList(param, peers);
    if (peers.empty()) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> descs(peers.size());
    channelDieIds.resize(peers.size());
    channelNetLayers.resize(peers.size());
    for (size_t i = 0; i < peers.size(); ++i) {
        CHK_RET(BuildPairChannelDesc(
            comm, param.myRank, peers[i], descs[i], channelDieIds[i], channelNetLayers[i]));
    }

    channels.resize(peers.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, descs.data(),
        static_cast<uint32_t>(descs.size()), channels.data()));
    return HCCL_SUCCESS;
}

HcclResult RegisterProbeRoute(CcuInsHandle insHandle, const OpParam &param,
    const std::vector<PairChannelGroup> &groups,
    const std::vector<std::shared_ptr<ops_hccl::CcuKernelArgGroup>> &groupKernelArgs,
    const std::vector<size_t> &groupIndices, const char *routeCode,
    void *kernelFunc, std::vector<CcuKernelHandle> &handles)
{
    CHK_PTR_NULL(routeCode);
    CHK_PTR_NULL(kernelFunc);
    CHK_PRT_RET(groupIndices.empty(),
        HCCL_ERROR("F081 probe route has no registration group"),
        HCCL_E_INTERNAL);

    std::vector<CcuKernelInfo> infos(groupIndices.size());
    handles.resize(groupIndices.size());
    for (size_t routeIndex = 0; routeIndex < groupIndices.size(); ++routeIndex) {
        size_t groupIndex = groupIndices[routeIndex];
        CHK_PRT_RET(groupIndex >= groups.size() ||
                groupIndex >= groupKernelArgs.size(),
            HCCL_ERROR("F081 invalid route group index[%llu]",
                static_cast<unsigned long long>(groupIndex)),
            HCCL_E_INTERNAL);
        CcuKernelInfo &info = infos[routeIndex];
        int nameLength = std::snprintf(info.kernelFuncName,
            sizeof(info.kernelFuncName), "F081%sR%uD%u", routeCode,
            param.myRank, groups[groupIndex].dieId);
        CHK_PRT_RET(nameLength <= 0 ||
                static_cast<size_t>(nameLength) >= sizeof(info.kernelFuncName),
            HCCL_ERROR("Failed to construct F081 probe kernel name"),
            HCCL_E_INTERNAL);
        info.kernelFunc = kernelFunc;
        info.SetKernelArg(groupKernelArgs[groupIndex]);
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    for (size_t routeIndex = 0; routeIndex < groupIndices.size(); ++routeIndex) {
        size_t groupIndex = groupIndices[routeIndex];
        const void *kernelArgs[] = {infos[routeIndex].kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle,
            groups[groupIndex].dieId, infos[routeIndex].kernelFuncName,
            infos[routeIndex].kernelFunc, kernelArgs, 1,
            &handles[routeIndex]));
        CHK_PRT_RET(handles[routeIndex] == 0,
            HCCL_ERROR("F081 probe registered a zero handle"),
            HCCL_E_INTERNAL);
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    HCCL_INFO("F081 probe route[%s] registered[%llu] rank[%u]",
        routeCode, static_cast<unsigned long long>(handles.size()),
        param.myRank);
    return HCCL_SUCCESS;
}

HcclResult RegisterExactProbeKernels(CcuInsHandle insHandle,
    const OpParam &param, const std::vector<PairChannelGroup> &groups,
    const std::vector<std::shared_ptr<ops_hccl::CcuKernelArgGroup>> &groupKernelArgs,
    AlgResourceCtx &resource)
{
    std::vector<size_t> smallGroups;
    std::vector<size_t> largeGroups;
    void *smallFunc = nullptr;
    void *large512Func = nullptr;
    void *large400MiB4BFunc = nullptr;

    if (param.rankSize == 4 && groups.size() == 1 &&
        groups[0].channels.size() == 3) {
        smallGroups = {0};
        largeGroups = {0};
        smallFunc = reinterpret_cast<void *>(
            ops_hccl::CcuRank4SmallProbeKernel);
        large512Func = reinterpret_cast<void *>(
            ops_hccl::CcuRank4Large512ProbeKernel);
        large400MiB4BFunc = reinterpret_cast<void *>(
            ops_hccl::CcuRank4Large400MiB4BProbeKernel);
    } else if (param.rankSize == ASYMMETRIC_RANK_SIZE) {
        size_t closGroupIndex = groups.size();
        CHK_PRT_RET(!FindRank12ClosGroup(
                param, groups, closGroupIndex),
            HCCL_ERROR("F081 rank12 probe topology mismatch"),
            HCCL_E_INTERNAL);
        smallGroups = {closGroupIndex};
        largeGroups = {0, 1};
        smallFunc = reinterpret_cast<void *>(
            ops_hccl::CcuRank12SmallProbeKernel);
        large512Func = reinterpret_cast<void *>(
            ops_hccl::CcuRank12Large512ProbeKernel);
        large400MiB4BFunc = reinterpret_cast<void *>(
            ops_hccl::CcuRank12Large400MiB4BProbeKernel);
    } else if (param.rankSize == MAX_RANK_SIZE) {
        size_t closGroupIndex = groups.size();
        CHK_PRT_RET(!FindRank16ClosGroup(
                param, groups, closGroupIndex),
            HCCL_ERROR("F081 rank16 probe topology mismatch"),
            HCCL_E_INTERNAL);
        smallGroups = {closGroupIndex};
        largeGroups = {0, 1};
        smallFunc = reinterpret_cast<void *>(
            ops_hccl::CcuRank16SmallProbeKernel);
        large512Func = reinterpret_cast<void *>(
            ops_hccl::CcuRank16Large512ProbeKernel);
        large400MiB4BFunc = reinterpret_cast<void *>(
            ops_hccl::CcuRank16Large400MiB4BProbeKernel);
    } else {
        return HCCL_SUCCESS;
    }

    CHK_RET(RegisterProbeRoute(insHandle, param, groups, groupKernelArgs,
        smallGroups, "S", smallFunc, resource.perf512KiBKernels));
    CHK_RET(RegisterProbeRoute(insHandle, param, groups, groupKernelArgs,
        largeGroups, "L", large512Func, resource.perf512MiBKernels));
    CHK_RET(RegisterProbeRoute(insHandle, param, groups, groupKernelArgs,
        largeGroups, "T", large400MiB4BFunc,
        resource.perf400MiB4BKernels));
    return HCCL_SUCCESS;
}

HcclResult RegisterGroupedPairKernels(HcclComm comm, const OpParam &param,
    const std::vector<uint32_t> &peers, const std::vector<ChannelHandle> &channels,
    const std::vector<uint32_t> &channelDieIds, const std::vector<uint32_t> &channelNetLayers,
    AlgResourceCtx &resource)
{
    CHK_PRT_RET(peers.size() != channels.size() || peers.size() != channelDieIds.size() ||
            peers.size() != channelNetLayers.size(),
        HCCL_ERROR("Pair channel metadata sizes do not match"), HCCL_E_INTERNAL);
    if (peers.empty()) {
        return HCCL_SUCCESS;
    }

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1 || insHandle == 0,
        HCCL_ERROR("Expected one CCU instruction instance, got[%u]", insNum), HCCL_E_INTERNAL);

    std::vector<PairChannelGroup> groups;
    for (size_t i = 0; i < peers.size(); ++i) {
        auto group = std::find_if(groups.begin(), groups.end(), [&](const PairChannelGroup &candidate) {
            return candidate.dieId == channelDieIds[i];
        });
        if (group == groups.end()) {
            groups.push_back(PairChannelGroup{});
            groups.back().dieId = channelDieIds[i];
            groups.back().netLayer = channelNetLayers[i];
            group = groups.end() - 1;
        } else if (group->netLayer != channelNetLayers[i]) {
            // Keep F003's one-kernel-per-die resource layout. A die that
            // contains channels from multiple network layers is marked as
            // mixed and will use the unchanged serial AllGather fallback.
            group->netLayer = MIXED_NET_LAYER;
        }
        group->peers.push_back(peers[i]);
        group->channels.push_back(channels[i]);
    }
    std::sort(groups.begin(), groups.end(), [](const PairChannelGroup &left, const PairChannelGroup &right) {
        return left.dieId < right.dieId;
    });

    std::vector<CcuKernelInfo> kernelInfos(groups.size());
    std::vector<std::shared_ptr<ops_hccl::CcuKernelArgGroup>> groupKernelArgs(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        CHK_PRT_RET(groups[i].channels.empty() || groups[i].channels.size() > MAX_RANK_SIZE,
            HCCL_ERROR("Invalid channel group size[%llu] on die[%u]",
                static_cast<unsigned long long>(groups[i].channels.size()), groups[i].dieId),
            HCCL_E_INTERNAL);

        CcuKernelInfo &info = kernelInfos[i];
        int nameLength = std::snprintf(info.kernelFuncName, sizeof(info.kernelFuncName),
            "F031RsR%uD%u", param.myRank, groups[i].dieId);
        CHK_PRT_RET(nameLength <= 0 || static_cast<size_t>(nameLength) >= sizeof(info.kernelFuncName),
            HCCL_ERROR("Failed to construct grouped kernel name"), HCCL_E_INTERNAL);
        info.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuGroupedPairKernel);

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgGroup>();
        kernelArg->myRank = param.myRank;
        kernelArg->rankSize = param.rankSize;
        kernelArg->groupIndex = static_cast<uint32_t>(i);
        kernelArg->groupCount = static_cast<uint32_t>(groups.size());
        kernelArg->channelCount = static_cast<uint32_t>(groups[i].channels.size());
        for (size_t channelIndex = 0; channelIndex < groups[i].channels.size(); ++channelIndex) {
            kernelArg->channels[channelIndex] = groups[i].channels[channelIndex];
            kernelArg->remoteRanks[channelIndex] = groups[i].peers[channelIndex];
        }
        info.SetKernelArg(kernelArg);
        groupKernelArgs[i] = kernelArg;
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    resource.ccuKernels.resize(groups.size());
    resource.kernelNetLayers.resize(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        const void *kernelArgs[] = {kernelInfos[i].kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, groups[i].dieId,
            kernelInfos[i].kernelFuncName, kernelInfos[i].kernelFunc,
            kernelArgs, 1, &resource.ccuKernels[i]));
        resource.kernelNetLayers[i] = groups[i].netLayer;
        HCCL_INFO("F031 group[%llu] die[%u] layer[%u] channels[%llu]",
            static_cast<unsigned long long>(i), groups[i].dieId, groups[i].netLayer,
            static_cast<unsigned long long>(groups[i].channels.size()));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));

    // F081 keeps the stable F031 generic registration first, then appends
    // three exact-size handle sets that reuse the same Channel arguments.
    // No additional Channel or Thread is acquired for any route.
    CHK_RET(RegisterExactProbeKernels(
        insHandle, param, groups, groupKernelArgs, resource));
    return HCCL_SUCCESS;
}

HcclResult BuildResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    CHK_PRT_RET(cclBufferAddr == nullptr || cclBufferSize < sizeof(float),
        HCCL_ERROR("Invalid local HCCL buffer"), HCCL_E_INTERNAL);
    resource.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    resource.ccuThread = param.cpuThread;

    std::vector<uint32_t> peers;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> channelDieIds;
    std::vector<uint32_t> channelNetLayers;
    CHK_RET(AcquirePairChannels(
        comm, param, peers, channels, channelDieIds, channelNetLayers));
    CHK_RET(RegisterGroupedPairKernels(
        comm, param, peers, channels, channelDieIds, channelNetLayers, resource));

    size_t registeredHandleCount = resource.ccuKernels.size() +
        resource.perf512KiBKernels.size() +
        resource.perf512MiBKernels.size() +
        resource.perf400MiB4BKernels.size();
    size_t expectedHandleCount = param.rankSize == 4 ? 4 :
        ((param.rankSize == ASYMMETRIC_RANK_SIZE ||
            param.rankSize == MAX_RANK_SIZE) ? 7 :
            resource.ccuKernels.size());
    CHK_PRT_RET(registeredHandleCount != expectedHandleCount,
        HCCL_ERROR("F081 probe handle count mismatch rankSize[%u] actual[%llu] expected[%llu]",
            param.rankSize,
            static_cast<unsigned long long>(registeredHandleCount),
            static_cast<unsigned long long>(expectedHandleCount)),
        HCCL_E_INTERNAL);

    std::vector<CcuKernelHandle> allRegisteredHandles;
    allRegisteredHandles.reserve(registeredHandleCount);
    auto appendHandles = [&allRegisteredHandles](
        const std::vector<CcuKernelHandle> &handles) {
        allRegisteredHandles.insert(
            allRegisteredHandles.end(), handles.begin(), handles.end());
    };
    appendHandles(resource.ccuKernels);
    appendHandles(resource.perf512KiBKernels);
    appendHandles(resource.perf512MiBKernels);
    appendHandles(resource.perf400MiB4BKernels);
    CHK_PRT_RET(allRegisteredHandles.size() != registeredHandleCount,
        HCCL_ERROR("F081 probe failed to collect every registered handle"),
        HCCL_E_INTERNAL);
    for (size_t i = 0; i < allRegisteredHandles.size(); ++i) {
        CHK_PRT_RET(allRegisteredHandles[i] == 0,
            HCCL_ERROR("F081 probe zero handle at flat index[%llu]",
                static_cast<unsigned long long>(i)),
            HCCL_E_INTERNAL);
        for (size_t j = i + 1; j < allRegisteredHandles.size(); ++j) {
            CHK_PRT_RET(allRegisteredHandles[i] == allRegisteredHandles[j],
                HCCL_ERROR("F081 probe duplicate handles at flat indices[%llu,%llu]",
                    static_cast<unsigned long long>(i),
                    static_cast<unsigned long long>(j)),
                HCCL_E_INTERNAL);
        }
    }
    HCCL_INFO("F081 probe unique nonzero handles rankSize[%u] total[%llu] status[PASS]",
        param.rankSize,
        static_cast<unsigned long long>(allRegisteredHandles.size()));
    HCCL_INFO("F081 probe handle layout rankSize[%u] generic[%llu] small[%llu] "
        "large512[%llu] large400p4[%llu] total[%llu] channels[%llu]",
        param.rankSize,
        static_cast<unsigned long long>(resource.ccuKernels.size()),
        static_cast<unsigned long long>(resource.perf512KiBKernels.size()),
        static_cast<unsigned long long>(resource.perf512MiBKernels.size()),
        static_cast<unsigned long long>(resource.perf400MiB4BKernels.size()),
        static_cast<unsigned long long>(registeredHandleCount),
        static_cast<unsigned long long>(channels.size()));

    resource.threads.push_back(param.cpuThread);
    bool hasIndependentLayers = resource.kernelNetLayers.size() == ASCEND950_DIE_NUM &&
        resource.kernelNetLayers[0] != MIXED_NET_LAYER &&
        resource.kernelNetLayers[1] != MIXED_NET_LAYER &&
        resource.kernelNetLayers[0] != resource.kernelNetLayers[1];
    if (hasIndependentLayers) {
        resource.threads.resize(ASCEND950_DIE_NUM);
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, 1,
            THREAD_NOTIFY_NUM, &resource.threads[1]));
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
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("F031 supports float32 only"), HCCL_E_PARA);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM,
        HCCL_ERROR("F031 supports sum only"), HCCL_E_PARA);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("AllReduce byte size overflows uint64"), HCCL_E_PARA);
    if (count == 0) {
        return HCCL_SUCCESS;
    }

    OpParam param{};
    int tagLength = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allreduce_f031");
    CHK_PRT_RET(tagLength <= 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("Failed to construct operation tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank[%u] rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);

    constexpr CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, THREAD_NOTIFY_NUM, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx resource;
        CHK_RET(BuildResources(comm, param, resource));

        std::vector<char> sequence = resource.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag,
            sequence.data(), sequence.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
