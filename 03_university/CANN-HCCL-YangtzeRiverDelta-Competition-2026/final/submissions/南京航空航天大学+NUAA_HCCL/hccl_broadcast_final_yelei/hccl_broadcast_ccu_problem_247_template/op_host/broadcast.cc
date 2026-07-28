/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <vector>

#include <ccu_launch.h>
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

struct ChannelGroup {
    uint32_t dieId = 0;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peerRanks;
};

bool IsCrossServer(uint32_t rankA, uint32_t rankB)
{
    return rankA / 8U != rankB / 8U;
}

bool IsMixedForestCandidate(const OpParam &param)
{
    (void)param;
    // The single-kernel forest can deadlock on the official 8+4 channel-to-Die map.
    return false;
}

bool UseMeshScatterAllgather(const OpParam &param)
{
    return param.rankSize > 4;
}

bool UseClosPipeline(const OpParam &param, bool useMixedForest)
{
    const uint64_t dataSize = param.count * sizeof(float);
    return !useMixedForest && param.rankSize > 4 && dataSize > F2_SMALL_DIRECT_LIMIT &&
        !(param.rankSize == 12 && param.root >= 8);
}

uint32_t ChannelNotifyCount(bool useMesh, bool useMixedForest, bool useClosPipeline)
{
    if (useMesh) {
        return F6_MESH_CHANNEL_NOTIFY_NUM;
    }
    if (useMixedForest) {
        return F4_FOREST_CHANNEL_NOTIFY_NUM;
    }
    return useClosPipeline ? F2_PIPELINE_CHANNEL_NOTIFY_NUM : F2_CHANNEL_NOTIFY_NUM;
}

HcclResult BuildChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    uint32_t notifyNum, const std::vector<uint32_t> &netLayers,
    HcclChannelDesc &desc, uint32_t &dieId)
{
    constexpr std::array<CommProtocol, 3> protocols = {
        COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UBC_TP, COMM_PROTOCOL_UBOE};
    for (CommProtocol protocol : protocols) {
        for (uint32_t layer : netLayers) {
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            HcclResult ret = HcclRankGraphGetLinks(comm, layer, myRank, remoteRank, &links, &linkCount);
            if (ret != HCCL_SUCCESS || links == nullptr) {
                continue;
            }
            for (uint32_t index = 0; index < linkCount; ++index) {
                const CommLink &link = links[index];
                if (link.linkAttr.linkProtocol != protocol) {
                    continue;
                }
                EndpointAttrDieId endpointDie = 0;
                CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank, &link.srcEndpointDesc,
                    ENDPOINT_ATTR_DIE_ID, sizeof(endpointDie), &endpointDie));
                CHK_PRT_RET(endpointDie >= F2_DIE_COUNT,
                    HCCL_ERROR("[F2BuildChannelDesc] invalid die %u for rank %u to %u",
                        endpointDie, myRank, remoteRank), HCCL_E_INTERNAL);
                CHK_RET(HcclChannelDescInit(&desc, 1));
                desc.remoteRank = remoteRank;
                desc.notifyNum = notifyNum;
                desc.channelProtocol = protocol;
                desc.localEndpoint = link.srcEndpointDesc;
                desc.remoteEndpoint = link.dstEndpointDesc;
                dieId = endpointDie;
                return HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("[F2BuildChannelDesc] no supported link from rank %u to rank %u", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

void FillTopology(const OpParam &param, ops_hccl::CcuKernelArgBroadcast &arg)
{
    if (param.rankSize == 4) {
        arg.topologyMode = F2_TOPO_FLAT_FOUR;
        arg.ownerCount = param.rankSize;
        uint32_t owner = 0;
        arg.sourceOwners[owner++] = param.root;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.root) {
                arg.sourceOwners[owner++] = rank;
            }
        }
        for (uint32_t index = 0; index < arg.ownerCount; ++index) {
            arg.destinationOwners[index] = arg.sourceOwners[index];
        }
    } else {
        arg.topologyMode = F2_TOPO_TWO_SERVER;
        const uint32_t rootBegin = param.root < 8 ? 0 : 8;
        const uint32_t rootEnd = param.root < 8 ? std::min<uint32_t>(8, param.rankSize) : param.rankSize;
        const uint32_t remoteBegin = param.root < 8 ? 8 : 0;
        const uint32_t remoteEnd = param.root < 8 ? param.rankSize : 8;

        for (uint32_t rank = rootBegin; rank < rootEnd; ++rank) {
            arg.rootServerRanks[arg.rootServerSize++] = rank;
        }
        for (uint32_t rank = remoteBegin; rank < remoteEnd; ++rank) {
            arg.remoteServerRanks[arg.remoteServerSize++] = rank;
        }
        arg.ownerCount = std::min(arg.rootServerSize, arg.remoteServerSize);

        uint32_t sourceIndex = 0;
        arg.sourceOwners[sourceIndex++] = param.root;
        for (uint32_t index = 0; index < arg.rootServerSize && sourceIndex < arg.ownerCount; ++index) {
            const uint32_t rank = arg.rootServerRanks[index];
            if (rank != param.root) {
                arg.sourceOwners[sourceIndex++] = rank;
            }
        }
        for (uint32_t index = 0; index < arg.ownerCount; ++index) {
            arg.destinationOwners[index] = arg.remoteServerRanks[index];
        }
    }

    const uint64_t stripeBase = param.count / arg.ownerCount;
    const uint64_t stripeRemainder = param.count % arg.ownerCount;
    uint64_t elementOffset = 0;
    for (uint32_t stripe = 0; stripe < arg.ownerCount; ++stripe) {
        const uint64_t stripeElements = stripeBase + static_cast<uint64_t>(stripe < stripeRemainder);
        arg.stripeOffsets[stripe] = elementOffset * sizeof(float);
        arg.stripeSizes[stripe] = stripeElements * sizeof(float);
        elementOffset += stripeElements;
    }
}

void FillPipeline(const OpParam &param, ops_hccl::CcuKernelArgBroadcast &arg)
{
    const uint32_t rootBegin = param.root < 8 ? 0 : 8;
    const uint32_t rootEnd = param.root < 8 ? std::min<uint32_t>(8, param.rankSize) : param.rankSize;
    const uint32_t relayBegin = param.root < 8 ? 8 : 0;
    const uint32_t relayEnd = param.root < 8 ? param.rankSize : std::min<uint32_t>(8, param.rankSize);

    arg.pipelineRootServerRanks[arg.pipelineRootServerSize++] = param.root;
    for (uint32_t rank = rootBegin; rank < rootEnd; ++rank) {
        if (rank != param.root) {
            arg.pipelineRootServerRanks[arg.pipelineRootServerSize++] = rank;
        }
    }
    for (uint32_t rank = relayBegin; rank < relayEnd; ++rank) {
        arg.pipelineRelayServerRanks[arg.pipelineRelayServerSize++] = rank;
    }

    const uint64_t stripeBase = param.count / arg.pipelineRelayServerSize;
    const uint64_t stripeRemainder = param.count % arg.pipelineRelayServerSize;
    uint64_t firstElement = 0;
    for (uint32_t stripe = 0; stripe < arg.pipelineRelayServerSize; ++stripe) {
        const uint64_t stripeElements =
            stripeBase + static_cast<uint64_t>(stripe < stripeRemainder);
        const uint64_t chunkBase = stripeElements / F2_PIPELINE_CHUNK_COUNT;
        const uint64_t chunkRemainder = stripeElements % F2_PIPELINE_CHUNK_COUNT;
        uint64_t consumed = 0;
        for (uint32_t chunk = 0; chunk < F2_PIPELINE_CHUNK_COUNT; ++chunk) {
            const uint64_t chunkElements =
                chunkBase + static_cast<uint64_t>(chunk < chunkRemainder);
            arg.pipelineOffsets[stripe][chunk] = (firstElement + consumed) * sizeof(float);
            arg.pipelineSizes[stripe][chunk] = chunkElements * sizeof(float);
            consumed += chunkElements;
        }
        firstElement += stripeElements;
    }
}

struct ForestEdge {
    uint32_t tree;
    uint32_t source;
    uint32_t destination;
};

struct ForestPlan {
    uint32_t treeCount = 0;
    std::array<std::vector<ForestEdge>, F4_FOREST_MAX_TREE_COUNT> edges;
};

bool ValidateForestPlan(const OpParam &param, const ForestPlan &plan)
{
    if (plan.treeCount == 0 || plan.treeCount > F4_FOREST_MAX_TREE_COUNT) {
        return false;
    }
    for (uint32_t tree = 0; tree < plan.treeCount; ++tree) {
        if (plan.edges[tree].size() != param.rankSize - 1U) {
            return false;
        }
        std::array<uint32_t, MAX_RANK_SIZE> parentCounts = {};
        for (const ForestEdge &edge : plan.edges[tree]) {
            if (edge.tree != tree || edge.source >= param.rankSize ||
                edge.destination >= param.rankSize || edge.source == edge.destination) {
                return false;
            }
            ++parentCounts[edge.destination];
        }
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            const uint32_t expected = rank == param.root ? 0U : 1U;
            if (parentCounts[rank] != expected) {
                return false;
            }
        }
        std::array<bool, MAX_RANK_SIZE> reached = {};
        reached[param.root] = true;
        for (uint32_t round = 0; round < param.rankSize; ++round) {
            for (const ForestEdge &edge : plan.edges[tree]) {
                if (reached[edge.source]) {
                    reached[edge.destination] = true;
                }
            }
        }
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (!reached[rank]) {
                return false;
            }
        }
    }
    return true;
}

bool BuildPipelinedForestPlan(const OpParam &param, ForestPlan &plan)
{
    if (!IsMixedForestCandidate(param)) {
        return false;
    }
    plan.treeCount = 7;
    constexpr std::array<ForestEdge, 30> fixedEdges = {{
        {0, 0, 2}, {0, 2, 3}, {0, 2, 5}, {0, 5, 7}, {0, 5, 4}, {0, 4, 6}, {0, 7, 1},
        {1, 0, 4}, {1, 4, 2}, {1, 4, 5}, {1, 4, 6}, {1, 4, 7}, {1, 2, 1}, {1, 8, 3},
        {2, 0, 6}, {2, 6, 2}, {2, 6, 4}, {2, 6, 7}, {2, 2, 1}, {2, 4, 5}, {2, 8, 3},
        {3, 0, 8}, {3, 8, 10}, {3, 10, 11}, {3, 1, 9},
        {4, 0, 9}, {4, 9, 10}, {5, 0, 10}, {5, 10, 9}, {6, 0, 11}
    }};
    for (const ForestEdge &edge : fixedEdges) {
        plan.edges[edge.tree].push_back(edge);
    }
    for (uint32_t remote = 8; remote < 12; ++remote) {
        plan.edges[0].push_back(ForestEdge{0, 7, remote});
        plan.edges[1].push_back(ForestEdge{1, 2, remote});
        plan.edges[2].push_back(ForestEdge{2, 2, remote});
    }
    for (uint32_t rank = 1; rank < 8; ++rank) {
        plan.edges[3].push_back(ForestEdge{3, 8, rank});
        plan.edges[4].push_back(ForestEdge{4, 9, rank});
        plan.edges[5].push_back(ForestEdge{5, 10, rank});
        plan.edges[6].push_back(ForestEdge{6, 11, rank});
    }
    plan.edges[4].push_back(ForestEdge{4, 1, 8});
    plan.edges[4].push_back(ForestEdge{4, 1, 11});
    plan.edges[5].push_back(ForestEdge{5, 1, 8});
    plan.edges[5].push_back(ForestEdge{5, 1, 11});
    plan.edges[6].push_back(ForestEdge{6, 11, 9});
    plan.edges[6].push_back(ForestEdge{6, 9, 8});
    plan.edges[6].push_back(ForestEdge{6, 1, 10});
    return ValidateForestPlan(param, plan);
}

bool ForestPeerAvailable(const ops_hccl::CcuKernelArgBroadcast &arg, uint32_t peer)
{
    for (uint32_t channel = 0; channel < arg.channelCount; ++channel) {
        if (arg.peerRanks[channel] == peer) {
            return true;
        }
    }
    return false;
}

uint32_t FindPeerDie(
    const std::array<ChannelGroup, F2_DIE_COUNT> &groups, uint32_t peer)
{
    uint32_t result = F2_DIE_COUNT;
    for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
        if (std::find(groups[die].peerRanks.begin(), groups[die].peerRanks.end(), peer) ==
            groups[die].peerRanks.end()) {
            continue;
        }
        if (result != F2_DIE_COUNT) {
            return F2_DIE_COUNT;
        }
        result = die;
    }
    return result;
}

void InitForestStripes(const OpParam &param, const ForestPlan &plan,
    ops_hccl::CcuKernelArgBroadcast &arg)
{
    arg.forestTreeCount = plan.treeCount;
    for (uint32_t tree = 0; tree < F4_FOREST_MAX_TREE_COUNT; ++tree) {
        arg.forestParents[tree] = F4_INVALID_RANK;
    }

    const uint64_t stripeBase = param.count / plan.treeCount;
    const uint64_t stripeRemainder = param.count % plan.treeCount;
    uint64_t firstElement = 0;
    for (uint32_t tree = 0; tree < plan.treeCount; ++tree) {
        const uint64_t stripeElements =
            stripeBase + static_cast<uint64_t>(tree < stripeRemainder);
        const uint64_t chunkBase = stripeElements / F4_FOREST_CHUNK_COUNT;
        const uint64_t chunkRemainder = stripeElements % F4_FOREST_CHUNK_COUNT;
        uint64_t consumed = 0;
        for (uint32_t chunk = 0; chunk < F4_FOREST_CHUNK_COUNT; ++chunk) {
            const uint64_t chunkElements =
                chunkBase + static_cast<uint64_t>(chunk < chunkRemainder);
            arg.forestOffsets[tree][chunk] =
                (firstElement + consumed) * sizeof(float);
            arg.forestSizes[tree][chunk] = chunkElements * sizeof(float);
            consumed += chunkElements;
        }
        firstElement += stripeElements;
    }
}

bool FillPipelinedForest(const OpParam &param, const ForestPlan &plan,
    const std::array<ChannelGroup, F2_DIE_COUNT> &groups,
    std::array<std::shared_ptr<ops_hccl::CcuKernelArgBroadcast>, F2_DIE_COUNT> &args)
{
    for (const auto &arg : args) {
        if (arg != nullptr) {
            InitForestStripes(param, plan, *arg);
        }
    }

    std::array<uint32_t, F4_FOREST_MAX_TREE_COUNT> childDieMasks = {};
    for (uint32_t tree = 0; tree < plan.treeCount; ++tree) {
        for (const ForestEdge &edge : plan.edges[tree]) {
            if (param.myRank == edge.source) {
                const uint32_t childDie = FindPeerDie(groups, edge.destination);
                if (childDie >= F2_DIE_COUNT || args[childDie] == nullptr) {
                    return false;
                }
                auto &arg = *args[childDie];
                if (arg.forestChildCounts[tree] >= MAX_RANK_SIZE) {
                    return false;
                }
                const uint32_t child = arg.forestChildCounts[tree]++;
                arg.forestChildren[tree][child] = edge.destination;
                arg.forestActive[tree] = 1;
                childDieMasks[tree] |= 1U << childDie;
            }
            if (param.myRank == edge.destination) {
                const uint32_t parentDie = FindPeerDie(groups, edge.source);
                if (parentDie >= F2_DIE_COUNT || args[parentDie] == nullptr ||
                    args[parentDie]->forestParents[tree] != F4_INVALID_RANK) {
                    return false;
                }
                args[parentDie]->forestParents[tree] = edge.source;
                args[parentDie]->forestActive[tree] = 1;
            }
        }
    }
    for (const auto &arg : args) {
        if (arg == nullptr) {
            continue;
        }
        for (uint32_t tree = 0; tree < plan.treeCount; ++tree) {
            arg->forestChildDieMasks[tree] = childDieMasks[tree];
        }
    }
    return true;
}

bool ForestMapComplete(const OpParam &param, const ForestPlan &plan,
    const std::array<std::shared_ptr<ops_hccl::CcuKernelArgBroadcast>, F2_DIE_COUNT> &args)
{
    for (uint32_t tree = 0; tree < plan.treeCount; ++tree) {
        uint32_t parentCount = 0;
        uint32_t childCount = 0;
        uint32_t childDieMask = 0;
        for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
            if (args[die] == nullptr) {
                continue;
            }
            const auto &arg = *args[die];
            const uint32_t parent = arg.forestParents[tree];
            const uint32_t localChildCount = arg.forestChildCounts[tree];
            const bool active = parent != F4_INVALID_RANK || localChildCount != 0;
            if (arg.forestTreeCount != plan.treeCount ||
                static_cast<bool>(arg.forestActive[tree]) != active) {
                return false;
            }
            if (parent != F4_INVALID_RANK) {
                if (parent >= param.rankSize || !ForestPeerAvailable(arg, parent)) {
                    return false;
                }
                ++parentCount;
            }
            if (localChildCount != 0) {
                childDieMask |= 1U << die;
            }
            childCount += localChildCount;
            for (uint32_t child = 0; child < localChildCount; ++child) {
                if (arg.forestChildren[tree][child] >= param.rankSize ||
                    !ForestPeerAvailable(arg, arg.forestChildren[tree][child])) {
                    return false;
                }
            }
        }

        uint32_t expectedChildren = 0;
        for (const ForestEdge &edge : plan.edges[tree]) {
            expectedChildren += static_cast<uint32_t>(edge.source == param.myRank);
        }
        const uint32_t expectedParents = param.myRank == param.root ? 0U : 1U;
        if (parentCount != expectedParents || childCount != expectedChildren) {
            return false;
        }
        for (const auto &arg : args) {
            if (arg != nullptr && arg->forestChildDieMasks[tree] != childDieMask) {
                return false;
            }
        }
    }
    return true;
}

std::shared_ptr<ops_hccl::CcuKernelArgBroadcast> MakeKernelArg(
    const OpParam &param, const ChannelGroup &group, uint32_t notifyIndex, uint32_t sliceIndex,
    bool useClosPipeline)
{
    auto arg = std::make_shared<ops_hccl::CcuKernelArgBroadcast>();
    arg->rankSize = param.rankSize;
    arg->rankId = param.myRank;
    arg->root = param.root;
    arg->dieId = group.dieId;
    arg->notifyIndex = notifyIndex;
    arg->channelCount = static_cast<uint32_t>(group.channels.size());
    for (uint32_t index = 0; index < arg->channelCount; ++index) {
        arg->channels[index] = group.channels[index];
        arg->peerRanks[index] = group.peerRanks[index];
    }

    const uint64_t totalBytes = param.count * sizeof(float);
    arg->offset = std::min<uint64_t>(
        static_cast<uint64_t>(sliceIndex) * MAX_DATA_SIZE, totalBytes);
    arg->dataSize = std::min<uint64_t>(MAX_DATA_SIZE, totalBytes - arg->offset);
    FillTopology(param, *arg);
    if (useClosPipeline) {
        FillPipeline(param, *arg);
    }

    const uint64_t stripeBase = param.count / F2_PATH_COUNT;
    const uint64_t stripeRemainder = param.count % F2_PATH_COUNT;
    uint64_t firstElement = 0;
    for (uint32_t path = 0; path < F2_PATH_COUNT; ++path) {
        const uint64_t stripeElements =
            stripeBase + static_cast<uint64_t>(path < stripeRemainder);
        uint64_t consumed = 0;
        for (uint32_t chunk = 0; chunk < F2_CHUNK_COUNT; ++chunk) {
            const uint64_t chunkElements = stripeElements / F2_CHUNK_COUNT +
                static_cast<uint64_t>(chunk < stripeElements % F2_CHUNK_COUNT);
            arg->chunkOffsets[path][chunk] =
                (firstElement + consumed) * sizeof(float);
            arg->chunkSizes[path][chunk] = chunkElements * sizeof(float);
            consumed += chunkElements;
        }
        firstElement += stripeElements;
    }
    return arg;
}
HcclResult RegisterKernel(CcuInsHandle insHandle, uint32_t dieId, uint32_t root, uint64_t count,
    const char *baseName, void *kernelFunc, const void *kernelArg, CcuKernelHandle &kernelHandle)
{
    char kernelName[64] = {};
    (void)std::snprintf(kernelName, sizeof(kernelName), "%sD%uRoot%uCount%llu", baseName, dieId, root,
        static_cast<unsigned long long>(count));
    const void *kernelArgs[] = {kernelArg};
    CcuResult result = HcommCcuKernelRegisterStart(insHandle);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("[F2RegisterKernel] register start failed, ret %d", result);
        return ConvertCcuToHccl(result);
    }
    result = HcommCcuKernelRegister(insHandle, dieId, kernelName, kernelFunc, kernelArgs, 1, &kernelHandle);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("[F2RegisterKernel] register %s failed, ret %d", kernelName, result);
        return ConvertCcuToHccl(result);
    }
    result = HcommCcuKernelRegisterEnd(insHandle);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("[F2RegisterKernel] register end failed, ret %d", result);
        return ConvertCcuToHccl(result);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterForestKernels(CcuInsHandle insHandle, const OpParam &param,
    const std::array<std::shared_ptr<ops_hccl::CcuKernelArgBroadcast>, F2_DIE_COUNT> &args,
    AlgResourceCtx &resource)
{
    CcuResult result = HcommCcuKernelRegisterStart(insHandle);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("[F4RegisterForest] register start failed, ret %d", result);
        return ConvertCcuToHccl(result);
    }

    uint32_t registered = 0;
    for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
        if (args[die] == nullptr) {
            continue;
        }
        char kernelName[64] = {};
        (void)std::snprintf(kernelName, sizeof(kernelName),
            "F4PipelinedForestD%uRoot%uCount%llu", die, param.root,
            static_cast<unsigned long long>(param.count));
        const void *kernelArgs[] = {args[die].get()};
        const uint32_t handleIndex =
            die * F2_KERNEL_PER_DIE + F4_KERNEL_PIPELINED_FOREST;
        result = HcommCcuKernelRegister(insHandle, die, kernelName,
            reinterpret_cast<void *>(ops_hccl::CcuMixedForestKernel),
            kernelArgs, 1, &resource.ccuKernels[handleIndex]);
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("[F4RegisterForest] register %s failed, ret %d",
                kernelName, result);
            return ConvertCcuToHccl(result);
        }
        ++registered;
    }
    CHK_PRT_RET(registered == 0,
        HCCL_ERROR("[F4RegisterForest] no active die on rank %u", param.myRank),
        HCCL_E_INTERNAL);

    result = HcommCcuKernelRegisterEnd(insHandle);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("[F4RegisterForest] register end failed, ret %d", result);
        return ConvertCcuToHccl(result);
    }
    return HCCL_SUCCESS;
}

HcclResult AllocateKernels(HcclComm comm, const OpParam &param,
    const std::array<ChannelGroup, F2_DIE_COUNT> &groups, const ForestPlan *forestPlan,
    bool useClosPipeline, bool useMesh, AlgResourceCtx &resource)
{
    CcuInsHandle insHandle{0};
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    CHK_PRT_RET(insCount != 1,
        HCCL_ERROR("[F2AllocateKernels] invalid CCU instance count %u", insCount),
        HCCL_E_INTERNAL);

    resource.ccuKernels.resize(F2_KERNEL_COUNT);
    resource.dieChannelCounts.resize(F2_DIE_COUNT, 0);
    for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
        resource.dieChannelCounts[die] =
            static_cast<uint32_t>(groups[die].channels.size());
    }

    if (forestPlan != nullptr) {
        std::array<std::shared_ptr<ops_hccl::CcuKernelArgBroadcast>,
            F2_DIE_COUNT> forestArgs;
        for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
            if (!groups[die].channels.empty()) {
                forestArgs[die] = MakeKernelArg(
                    param, groups[die], F2_SCATTER_NOTIFY, 0, false);
            }
        }
        CHK_PRT_RET(!FillPipelinedForest(
            param, *forestPlan, groups, forestArgs),
            HCCL_ERROR("[F4AllocateKernels] cannot map forest channels on rank %u",
                param.myRank),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(!ForestMapComplete(param, *forestPlan, forestArgs),
            HCCL_ERROR("[F4AllocateKernels] incomplete pipelined forest on rank %u",
                param.myRank),
            HCCL_E_INTERNAL);
        return RegisterForestKernels(insHandle, param, forestArgs, resource);
    }

    constexpr std::array<const char *, F2_KERNEL_PER_DIE> names = {
        "F2Prepare", "F2Direct0", "F2Direct1", "F2Scatter", "F2Cross",
        "F2Local", "F2Packed", "F2ClosPipeline", "F4PipelinedForest",
        "F6Mesh"};

    for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
        const ChannelGroup &group = groups[die];
        if (group.channels.empty()) {
            continue;
        }

        std::array<CcuKernelInfo, F2_KERNEL_PER_DIE> infos;
        if (useMesh) {
            infos[F6_KERNEL_MESH].kernelFunc =
                reinterpret_cast<void *>(ops_hccl::CcuMeshKernel);
            infos[F6_KERNEL_MESH].SetKernelArg(
                MakeKernelArg(param, group, F2_SCATTER_NOTIFY, 0, false));
        } else if (useClosPipeline) {
            infos[F2_KERNEL_CLOS_PIPELINE].kernelFunc =
                reinterpret_cast<void *>(ops_hccl::CcuClosPipelineKernel);
            infos[F2_KERNEL_CLOS_PIPELINE].SetKernelArg(
                MakeKernelArg(param, group, F2_SCATTER_NOTIFY, 0, true));
        } else {
            infos[F2_KERNEL_DIRECT_0].kernelFunc =
                reinterpret_cast<void *>(ops_hccl::CcuDirectKernel0);
            infos[F2_KERNEL_DIRECT_0].SetKernelArg(
                MakeKernelArg(param, group, F2_SCATTER_NOTIFY, 0, false));
            if (param.rankSize == 4) {
                infos[F2_KERNEL_PACKED].kernelFunc =
                    reinterpret_cast<void *>(ops_hccl::CcuPackedThreePathKernel);
                infos[F2_KERNEL_PACKED].SetKernelArg(
                    MakeKernelArg(param, group, F2_PATH_NOTIFY_BASE, 0, false));
            } else {
                infos[F2_KERNEL_SCATTER].kernelFunc =
                    reinterpret_cast<void *>(ops_hccl::CcuScatterKernel);
                infos[F2_KERNEL_SCATTER].SetKernelArg(
                    MakeKernelArg(param, group, F2_SCATTER_NOTIFY, 0, false));
                infos[F2_KERNEL_LOCAL].kernelFunc =
                    reinterpret_cast<void *>(ops_hccl::CcuLocalAllgatherKernel);
                infos[F2_KERNEL_LOCAL].SetKernelArg(
                    MakeKernelArg(param, group, F2_LOCAL_NOTIFY, 0, false));
            }
        }

        const uint32_t base = die * F2_KERNEL_PER_DIE;
        for (uint32_t index = 0; index < F2_KERNEL_PER_DIE; ++index) {
            if (infos[index].kernelFunc == nullptr) {
                continue;
            }
            CHK_RET(RegisterKernel(insHandle, die, param.root, param.count,
                names[index], infos[index].kernelFunc, infos[index].kernelArg,
                resource.ccuKernels[base + index]));
        }
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    (void)std::snprintf(param.tag, sizeof(param.tag), "hccl_f6_broadcast_root_%u_count_%llu", root,
        static_cast<unsigned long long>(count));
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || root >= param.rankSize,
        HCCL_ERROR("[F2Broadcast] invalid rankSize %u or root %u", param.rankSize, root), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize != 1 && param.rankSize != 4 && param.rankSize != 12 && param.rankSize != 16,
        HCCL_ERROR("[F2Broadcast] unsupported competition rankSize %u", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[F2Broadcast] only fp32 is supported, dataType %d", dataType), HCCL_E_NOT_SUPPORT);

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream, F2_THREAD_NOTIFY_NUM, &param.cpuThread));

    void *context = nullptr;
    uint64_t contextSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, engine, &context, &contextSize) == HCCL_SUCCESS) {
        param.resCtx = context;
        param.ctxSize = contextSize;
    } else {
        AlgResourceCtx resource;
        void *cclBuffer = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
        resource.localBuffer = CommBuffer{cclBuffer, cclBufferSize};
        resource.ccuThread = param.cpuThread;
        resource.threads.resize(F2_WORKER_COUNT + 1);
        resource.threads[0] = param.cpuThread;
        CHK_RET(HcclThreadAcquire(
            comm, engine, F2_WORKER_COUNT, F2_THREAD_NOTIFY_NUM, &resource.threads[1]));

        if (param.rankSize > 1) {
            uint32_t *layerData = nullptr;
            uint32_t layerCount = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
            CHK_PRT_RET(layerData == nullptr || layerCount == 0,
                HCCL_ERROR("[F2Broadcast] empty rank graph"), HCCL_E_NOT_FOUND);
            const std::vector<uint32_t> layers(layerData, layerData + layerCount);

            ForestPlan forestPlan;
            const bool useMesh = UseMeshScatterAllgather(param);
            const bool useMixedForest = !useMesh &&
                BuildPipelinedForestPlan(param, forestPlan);
            const bool useClosPipeline = !useMesh &&
                UseClosPipeline(param, useMixedForest);
            resource.algorithmMode = useMesh ? F6_ALG_MESH_SCATTER_ALLGATHER :
                (useMixedForest ? F4_ALG_PIPELINED_FOREST :
                    (useClosPipeline ? F4_ALG_CLOS_PIPELINE : F4_ALG_F2_DEFAULT));
            std::vector<HcclChannelDesc> descriptors;
            std::vector<uint32_t> descriptorDies;
            std::vector<uint32_t> peerRanks;
            descriptors.reserve(param.rankSize - 1);
            descriptorDies.reserve(param.rankSize - 1);
            peerRanks.reserve(param.rankSize - 1);
            for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
                if (peer == param.myRank || (useClosPipeline && !IsCrossServer(param.myRank, peer))) {
                    continue;
                }
                HcclChannelDesc descriptor;
                uint32_t dieId = 0;
                CHK_RET(BuildChannelDesc(comm, param.myRank, peer,
                    ChannelNotifyCount(useMesh, useMixedForest, useClosPipeline),
                    layers, descriptor, dieId));
                descriptors.push_back(descriptor);
                descriptorDies.push_back(dieId);
                peerRanks.push_back(peer);
            }

            std::vector<ChannelHandle> channels(descriptors.size());
            CHK_RET(HcclChannelAcquire(
                comm, engine, descriptors.data(), static_cast<uint32_t>(descriptors.size()), channels.data()));

            std::array<ChannelGroup, F2_DIE_COUNT> groups;
            for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
                groups[die].dieId = die;
            }
            for (uint32_t index = 0; index < channels.size(); ++index) {
                const uint32_t die = descriptorDies[index];
                groups[die].channels.push_back(channels[index]);
                groups[die].peerRanks.push_back(peerRanks[index]);
            }
            CHK_RET(AllocateKernels(comm, param, groups,
                useMixedForest ? &forestPlan : nullptr, useClosPipeline, useMesh, resource));
        }

        std::vector<char> serialized = resource.Serialize();
        param.ctxSize = serialized.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, serialized.data(), serialized.size(), 0));
    }
    return ops_hccl::ExecOp(param);
}
