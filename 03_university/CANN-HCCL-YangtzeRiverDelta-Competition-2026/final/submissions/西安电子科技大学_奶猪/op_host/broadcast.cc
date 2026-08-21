/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_COUNT = 4;
constexpr uint32_t THREAD_NOTIFY_COUNT = 2;
constexpr char FOREST_TAG[] = "hccl_broadcast_multiring_forest_v6_r2";

struct ChannelGroup {
    uint32_t dieId{0};
    std::vector<uint32_t> peerRanks;
    std::vector<ChannelHandle> channels;
};

HcclResult DescribePeer(HcclComm comm, uint32_t localRank, uint32_t peerRank,
    HcclChannelDesc &descriptor, uint32_t &dieId)
{
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    CHK_PRT_RET(layers == nullptr || layerCount == 0,
        HCCL_ERROR("[ForestV4] rank graph contains no link layers"), HCCL_E_INTERNAL);

    constexpr CommProtocol preferred[] = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
    };
    for (CommProtocol protocol : preferred) {
        for (uint32_t layerIndex = 0; layerIndex < layerCount; ++layerIndex) {
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            HcclResult query = HcclRankGraphGetLinks(
                comm, layers[layerIndex], localRank, peerRank, &links, &linkCount);
            if (query != HCCL_SUCCESS || links == nullptr) {
                continue;
            }
            for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
                const CommLink &link = links[linkIndex];
                if (link.linkAttr.linkProtocol != protocol) {
                    continue;
                }
                CHK_RET(HcclChannelDescInit(&descriptor, 1));
                descriptor.remoteRank = peerRank;
                descriptor.notifyNum = CHANNEL_NOTIFY_COUNT;
                descriptor.channelProtocol = protocol;
                descriptor.localEndpoint = link.srcEndpointDesc;
                descriptor.remoteEndpoint = link.dstEndpointDesc;

                EndpointAttrDieId endpointDie = 0;
                CHK_RET(HcclRankGraphGetEndpointInfo(comm, localRank, &descriptor.localEndpoint,
                    EndpointAttr::ENDPOINT_ATTR_DIE_ID, sizeof(endpointDie), &endpointDie));
                dieId = endpointDie;
                return HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("[ForestV4] no CCU channel from rank %u to rank %u", localRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

ChannelGroup &GetGroup(std::vector<ChannelGroup> &groups, uint32_t dieId)
{
    auto position = std::find_if(groups.begin(), groups.end(),
        [dieId](const ChannelGroup &group) { return group.dieId == dieId; });
    if (position == groups.end()) {
        groups.push_back(ChannelGroup{});
        groups.back().dieId = dieId;
        return groups.back();
    }
    return *position;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param,
    std::vector<ChannelGroup> &groups)
{
    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }
        HcclChannelDesc descriptor{};
        uint32_t dieId = 0;
        CHK_RET(DescribePeer(comm, param.myRank, peerRank, descriptor, dieId));

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &descriptor, 1, &channel));
        ChannelGroup &group = GetGroup(groups, dieId);
        group.peerRanks.push_back(peerRank);
        group.channels.push_back(channel);
    }
    std::sort(groups.begin(), groups.end(),
        [](const ChannelGroup &left, const ChannelGroup &right) { return left.dieId < right.dieId; });
    CHK_PRT_RET(groups.empty() || groups.size() > 2,
        HCCL_ERROR("[ForestV4] unsupported CCU DIE group count %zu", groups.size()), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult DiscoverServerBuckets(HcclComm comm, uint32_t rankSize,
    std::vector<uint32_t> &serverByRank)
{
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    CHK_PRT_RET(layers == nullptr || layerCount == 0,
        HCCL_ERROR("[ForestV4] rank graph contains no topology layers"), HCCL_E_INTERNAL);

    std::vector<uint32_t> bestSizes;
    for (uint32_t layerIndex = 0; layerIndex < layerCount; ++layerIndex) {
        uint32_t *sizes = nullptr;
        uint32_t instanceCount = 0;
        CHK_RET(HcclRankGraphGetInstSizeListByLayer(
            comm, layers[layerIndex], &sizes, &instanceCount));
        CHK_PRT_RET(sizes == nullptr || instanceCount == 0,
            HCCL_ERROR("[ForestV4] layer %u contains no topology instances", layers[layerIndex]),
            HCCL_E_INTERNAL);

        uint64_t coveredRanks = 0;
        for (uint32_t instance = 0; instance < instanceCount; ++instance) {
            CHK_PRT_RET(sizes[instance] == 0,
                HCCL_ERROR("[ForestV4] layer %u has an empty topology instance", layers[layerIndex]),
                HCCL_E_INTERNAL);
            coveredRanks += sizes[instance];
        }
        CHK_PRT_RET(coveredRanks != rankSize,
            HCCL_ERROR("[ForestV4] layer %u covers %llu ranks instead of %u", layers[layerIndex],
                static_cast<unsigned long long>(coveredRanks), rankSize), HCCL_E_INTERNAL);

        // The layer with the most instances is the finest partition and maps
        // directly to the per-server rank groups used by HCCL's group planner.
        if (instanceCount > bestSizes.size()) {
            bestSizes.assign(sizes, sizes + instanceCount);
        }
    }

    CHK_PRT_RET(bestSizes.empty(),
        HCCL_ERROR("[ForestV4] failed to discover server topology"), HCCL_E_INTERNAL);
    serverByRank.resize(rankSize);
    uint32_t rank = 0;
    for (uint32_t server = 0; server < bestSizes.size(); ++server) {
        for (uint32_t localRank = 0; localRank < bestSizes[server]; ++localRank) {
            serverByRank[rank++] = server;
        }
    }
    CHK_PRT_RET(rank != rankSize,
        HCCL_ERROR("[ForestV4] server partition produced %u ranks instead of %u", rank, rankSize),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void DivideIntoLanes(uint64_t totalBytes, uint32_t rankSize, ForestKernelConfig &config)
{
    const uint64_t elementCount = totalBytes / sizeof(float);
    const uint64_t quotient = elementCount / rankSize;
    const uint64_t remainder = elementCount % rankSize;
    uint64_t elementOffset = 0;
    for (uint32_t lane = 0; lane < rankSize; ++lane) {
        const uint64_t laneElements = quotient + (lane < remainder ? 1ULL : 0ULL);
        config.laneOffsets[lane] = elementOffset * sizeof(float);
        config.laneBytes[lane] = laneElements * sizeof(float);
        elementOffset += laneElements;
    }
}

struct ForestEdge {
    uint32_t parent;
    uint32_t child;
    uint32_t stage;
};

constexpr uint32_t LOCAL_PORT_CLASS_COUNT = 7;
constexpr uint32_t LOCAL_PORT_CLASS[8][8] = {
    {0, 3, 0, 4, 1, 5, 2, 6},
    {3, 0, 0, 4, 1, 5, 2, 6},
    {3, 0, 0, 4, 1, 2, 0, 6},
    {3, 0, 4, 0, 1, 5, 6, 3},
    {3, 0, 4, 1, 0, 5, 2, 3},
    {3, 0, 5, 2, 6, 0, 0, 3},
    {3, 0, 4, 2, 6, 0, 0, 3},
    {3, 0, 4, 1, 5, 2, 6, 0},
};

struct ScheduleLoads {
    uint32_t ports[MAX_RANK_SIZE][LOCAL_PORT_CLASS_COUNT]{};
    uint32_t pairs[MAX_RANK_SIZE][MAX_RANK_SIZE]{};
    uint32_t crossEndpoints[MAX_RANK_SIZE]{};
};

struct ScheduleScore {
    uint32_t maxPort{0};
    uint64_t portSquares{0};
    uint32_t maxPair{0};
    uint64_t pairSquares{0};
    uint32_t maxCrossEndpoint{0};
    uint64_t crossSquares{0};
};

bool BetterScore(const ScheduleScore &left, const ScheduleScore &right)
{
    if (left.maxPort != right.maxPort) {
        return left.maxPort < right.maxPort;
    }
    if (left.portSquares != right.portSquares) {
        return left.portSquares < right.portSquares;
    }
    if (left.maxPair != right.maxPair) {
        return left.maxPair < right.maxPair;
    }
    if (left.pairSquares != right.pairSquares) {
        return left.pairSquares < right.pairSquares;
    }
    if (left.maxCrossEndpoint != right.maxCrossEndpoint) {
        return left.maxCrossEndpoint < right.maxCrossEndpoint;
    }
    return left.crossSquares < right.crossSquares;
}

uint32_t GreatestCommonDivisor(uint32_t left, uint32_t right)
{
    while (right != 0) {
        const uint32_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

void BuildRingOrders(const std::vector<uint32_t> &serverByRank, std::vector<uint32_t> &orders)
{
    std::map<uint32_t, std::vector<uint32_t>> ranksByServer;
    for (uint32_t rank = 0; rank < serverByRank.size(); ++rank) {
        ranksByServer[serverByRank[rank]].push_back(rank);
    }
    orders.clear();
    orders.reserve(FOREST_RING_COUNT * serverByRank.size());
    for (uint32_t ring = 0; ring < FOREST_RING_COUNT; ++ring) {
        for (const auto &entry : ranksByServer) {
            const std::vector<uint32_t> &ranks = entry.second;
            std::vector<uint32_t> steps;
            for (uint32_t step = 1; step < ranks.size(); ++step) {
                if (GreatestCommonDivisor(step, ranks.size()) == 1) {
                    steps.push_back(step);
                }
            }
            if (ranks.size() == 1) {
                orders.push_back(ranks.front());
                continue;
            }
            const uint32_t step = steps[ring % steps.size()];
            const uint32_t start = (ring * (ring + 1) / 2) % ranks.size();
            for (uint32_t index = 0; index < ranks.size(); ++index) {
                orders.push_back(ranks[(start + index * step) % ranks.size()]);
            }
        }
    }
}

std::vector<std::vector<uint32_t>> BuildAffineOrders(const std::vector<uint32_t> &ranks)
{
    if (ranks.size() <= 1) {
        std::vector<std::vector<uint32_t>> singleOrder;
        singleOrder.push_back(ranks);
        return singleOrder;
    }
    std::vector<std::vector<uint32_t>> orders;
    for (uint32_t step = 1; step < ranks.size(); ++step) {
        if (GreatestCommonDivisor(step, ranks.size()) != 1) {
            continue;
        }
        for (uint32_t start = 0; start < ranks.size(); ++start) {
            std::vector<uint32_t> order;
            order.reserve(ranks.size());
            for (uint32_t index = 0; index < ranks.size(); ++index) {
                order.push_back(ranks[(start + index * step) % ranks.size()]);
            }
            orders.push_back(std::move(order));
        }
    }
    return orders;
}

void AppendRadixTree(uint32_t source, const std::vector<uint32_t> &order,
    uint32_t firstStage, uint32_t radix, std::vector<ForestEdge> &edges)
{
    std::vector<uint32_t> tree{source};
    tree.insert(tree.end(), order.begin(), order.end());
    for (uint32_t position = 1; position < tree.size(); ++position) {
        uint32_t depth = 0;
        for (uint32_t cursor = position; cursor != 0; cursor = (cursor - 1) / radix) {
            ++depth;
        }
        edges.push_back(ForestEdge{
            tree[(position - 1) / radix], tree[position], firstStage + depth - 1});
    }
}

void EnumerateBridgeSets(const std::vector<uint32_t> &targets, uint32_t bridgeCount,
    uint32_t start, std::vector<uint32_t> &selection,
    std::vector<std::vector<uint32_t>> &bridgeSets)
{
    if (selection.size() == bridgeCount) {
        bridgeSets.push_back(selection);
        return;
    }
    const uint32_t missing = bridgeCount - selection.size();
    for (uint32_t index = start; index + missing <= targets.size(); ++index) {
        selection.push_back(targets[index]);
        EnumerateBridgeSets(targets, bridgeCount, index + 1, selection, bridgeSets);
        selection.pop_back();
    }
}

std::vector<std::vector<ForestEdge>> BuildCandidates(const std::vector<uint32_t> &serverByRank,
    uint32_t leader, const std::vector<uint32_t> &targets)
{
    std::vector<std::vector<ForestEdge>> candidates;
    if (targets.empty()) {
        candidates.push_back({});
        return candidates;
    }
    if (serverByRank[leader] == serverByRank[targets.front()]) {
        for (const std::vector<uint32_t> &order : BuildAffineOrders(targets)) {
            std::vector<ForestEdge> edges;
            AppendRadixTree(leader, order, 0, FOREST_RADIX - 1, edges);
            candidates.push_back(std::move(edges));
        }
        return candidates;
    }

    const uint32_t bridgeCount = std::min<uint32_t>(2U, targets.size());
    std::vector<std::vector<uint32_t>> bridgeSets;
    std::vector<uint32_t> selection;
    EnumerateBridgeSets(targets, bridgeCount, 0, selection, bridgeSets);
    for (const std::vector<uint32_t> &bridges : bridgeSets) {
        std::vector<uint32_t> remainder;
        for (uint32_t rank : targets) {
            if (std::find(bridges.begin(), bridges.end(), rank) == bridges.end()) {
                remainder.push_back(rank);
            }
        }
        for (const std::vector<uint32_t> &order : BuildAffineOrders(remainder)) {
            std::vector<ForestEdge> edges;
            for (uint32_t bridge : bridges) {
                edges.push_back(ForestEdge{leader, bridge, 0});
            }
            for (uint32_t index = 0; index < order.size(); ++index) {
                edges.push_back(ForestEdge{
                    bridges[index % bridges.size()], order[index], 1});
            }
            candidates.push_back(std::move(edges));
        }
    }
    return candidates;
}

void AccountEdge(const ForestEdge &edge, const std::vector<uint32_t> &serverByRank,
    const std::vector<uint32_t> &localByRank, ScheduleLoads &loads)
{
    if (serverByRank[edge.parent] != serverByRank[edge.child]) {
        ++loads.crossEndpoints[edge.parent];
        ++loads.crossEndpoints[edge.child];
        return;
    }
    const uint32_t lowRank = std::min(edge.parent, edge.child);
    const uint32_t highRank = std::max(edge.parent, edge.child);
    ++loads.pairs[lowRank][highRank];
    const uint32_t parentLocal = localByRank[edge.parent];
    const uint32_t childLocal = localByRank[edge.child];
    ++loads.ports[edge.parent][LOCAL_PORT_CLASS[parentLocal][childLocal]];
    ++loads.ports[edge.child][LOCAL_PORT_CLASS[childLocal][parentLocal]];
}

ScheduleScore MeasureLoads(const ScheduleLoads &loads, uint32_t rankSize)
{
    ScheduleScore score;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        for (uint32_t port = 0; port < LOCAL_PORT_CLASS_COUNT; ++port) {
            const uint32_t load = loads.ports[rank][port];
            score.maxPort = std::max(score.maxPort, load);
            score.portSquares += static_cast<uint64_t>(load) * load;
        }
        score.maxCrossEndpoint = std::max(score.maxCrossEndpoint, loads.crossEndpoints[rank]);
        score.crossSquares += static_cast<uint64_t>(loads.crossEndpoints[rank])
            * loads.crossEndpoints[rank];
        for (uint32_t peer = rank + 1; peer < rankSize; ++peer) {
            const uint32_t load = loads.pairs[rank][peer];
            score.maxPair = std::max(score.maxPair, load);
            score.pairSquares += static_cast<uint64_t>(load) * load;
        }
    }
    return score;
}

HcclResult BuildForestEdges(const std::vector<uint32_t> &serverByRank,
    uint32_t root, std::vector<std::vector<ForestEdge>> &edgesByLane)
{
    std::map<uint32_t, std::vector<uint32_t>> ranksByServer;
    for (uint32_t rank = 0; rank < serverByRank.size(); ++rank) {
        ranksByServer[serverByRank[rank]].push_back(rank);
    }
    std::vector<uint32_t> localByRank(serverByRank.size(), 0);
    for (const auto &entry : ranksByServer) {
        CHK_PRT_RET(entry.second.size() > 8,
            HCCL_ERROR("[ForestV4] server group %u exceeds eight ranks", entry.first),
            HCCL_E_NOT_SUPPORT);
        for (uint32_t local = 0; local < entry.second.size(); ++local) {
            localByRank[entry.second[local]] = local;
        }
    }

    ScheduleLoads committedLoads;
    edgesByLane.assign(serverByRank.size(), {});
    // Seed transfers are fixed by the striped layout.  Include them in the
    // load ledger so tree selection does not accidentally reuse their hottest
    // local port or cross-server endpoint.
    for (uint32_t lane = 0; lane < serverByRank.size(); ++lane) {
        if (lane != root) {
            AccountEdge(ForestEdge{root, lane, 0}, serverByRank, localByRank, committedLoads);
        }
    }
    for (uint32_t lane = 0; lane < serverByRank.size(); ++lane) {
        for (const auto &entry : ranksByServer) {
            std::vector<uint32_t> targets;
            for (uint32_t rank : entry.second) {
                if (rank != lane && !(lane != root && rank == root)) {
                    targets.push_back(rank);
                }
            }
            if (targets.empty()) {
                continue;
            }

            const std::vector<std::vector<ForestEdge>> candidates =
                BuildCandidates(serverByRank, lane, targets);
            bool found = false;
            ScheduleScore bestScore;
            ScheduleLoads bestLoads;
            const std::vector<ForestEdge> *bestEdges = nullptr;
            for (const std::vector<ForestEdge> &candidate : candidates) {
                ScheduleLoads trialLoads = committedLoads;
                for (const ForestEdge &edge : candidate) {
                    AccountEdge(edge, serverByRank, localByRank, trialLoads);
                }
                const ScheduleScore trialScore = MeasureLoads(trialLoads, serverByRank.size());
                if (!found || BetterScore(trialScore, bestScore)) {
                    found = true;
                    bestScore = trialScore;
                    bestLoads = trialLoads;
                    bestEdges = &candidate;
                }
            }
            CHK_PRT_RET(!found || bestEdges == nullptr,
                HCCL_ERROR("[ForestV4] no schedule candidate for lane %u", lane), HCCL_E_INTERNAL);
            committedLoads = bestLoads;
            edgesByLane[lane].insert(
                edgesByLane[lane].end(), bestEdges->begin(), bestEdges->end());
        }
    }
    return HCCL_SUCCESS;
}

HcclResult FillLocalSchedule(const std::vector<uint32_t> &serverByRank, ForestKernelConfig &config)
{
    std::fill(&config.parentRanks[0][0],
        &config.parentRanks[0][0] + FOREST_STAGE_COUNT * MAX_RANK_SIZE, INVALID_VALUE_RANKID);
    std::fill(&config.childRankMasks[0][0],
        &config.childRankMasks[0][0] + FOREST_STAGE_COUNT * MAX_RANK_SIZE, 0U);
    std::vector<std::vector<ForestEdge>> edgesByLane;
    CHK_RET(BuildForestEdges(serverByRank, config.rootRank, edgesByLane));
    for (uint32_t lane = 0; lane < config.rankSize; ++lane) {
        for (const ForestEdge &edge : edgesByLane[lane]) {
            CHK_PRT_RET(edge.stage >= FOREST_STAGE_COUNT,
                HCCL_ERROR("[ForestV4] tree stage %u exceeds compiled stage count", edge.stage),
                HCCL_E_INTERNAL);
            if (edge.child == config.rankId) {
                config.parentRanks[edge.stage][lane] = edge.parent;
            }
            if (edge.parent == config.rankId) {
                config.childRankMasks[edge.stage][lane] |= 1U << edge.child;
            }
        }
    }
    return HCCL_SUCCESS;
}

bool ContainsRank(const ChannelGroup &group, uint32_t rank)
{
    return std::find(group.peerRanks.begin(), group.peerRanks.end(), rank) != group.peerRanks.end();
}

HcclResult RegisterForestKernels(HcclComm comm, const OpParam &param, uint64_t totalBytes,
    const std::vector<uint32_t> &serverByRank, const std::vector<ChannelGroup> &groups,
    ForestResourceContext &resource)
{
    CcuInsHandle instruction = 0;
    uint32_t instructionCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instruction, &instructionCount));
    CHK_PRT_RET(instructionCount != 1,
        HCCL_ERROR("[ForestV4] expected one CCU instruction engine, got %u", instructionCount),
        HCCL_E_INTERNAL);

    resource.kernels.resize(groups.size());
    resource.directActive.resize(groups.size(), 0);
    std::vector<ForestKernelRecord> records(groups.size());
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instruction));
    for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const ChannelGroup &group = groups[groupIndex];
        auto config = std::make_shared<ForestKernelConfig>();
        config->channelCount = group.channels.size();
        config->rankSize = param.rankSize;
        config->rankId = param.myRank;
        config->rootRank = param.root;
        config->directMode = totalBytes <= DIRECT_MESSAGE_LIMIT ? 1U : 0U;
        config->totalBytes = totalBytes;
        for (uint32_t channel = 0; channel < group.channels.size(); ++channel) {
            config->channels[channel] = group.channels[channel];
            config->peerRanks[channel] = group.peerRanks[channel];
        }
        DivideIntoLanes(totalBytes, param.rankSize, *config);
        CHK_RET(FillLocalSchedule(serverByRank, *config));
        CHK_PRT_RET(resource.ringOrders.size() != FOREST_RING_COUNT * param.rankSize,
            HCCL_ERROR("[ForestV4] malformed ring schedule"), HCCL_E_INTERNAL);
        for (uint32_t ring = 0; ring < FOREST_RING_COUNT; ++ring) {
            const auto begin = resource.ringOrders.begin() + ring * param.rankSize;
            const auto end = begin + param.rankSize;
            const auto position = std::find(begin, end, param.myRank);
            CHK_PRT_RET(position == end,
                HCCL_ERROR("[ForestV4] rank %u is absent from ring %u", param.myRank, ring),
                HCCL_E_INTERNAL);
            const uint32_t index = static_cast<uint32_t>(position - begin);
            config->ringNextRanks[ring] = *(begin + (index + 1) % param.rankSize);
            config->ringPreviousRanks[ring] = *(begin + (index + param.rankSize - 1) % param.rankSize);
        }

        ForestKernelRecord &record = records[groupIndex];
        const int written = std::snprintf(record.name, sizeof(record.name),
            "ForestV6Rank%uRoot%uDie%u", param.myRank, param.root, group.dieId);
        CHK_PRT_RET(written < 0 || static_cast<size_t>(written) >= sizeof(record.name),
            HCCL_ERROR("[ForestV4] kernel name does not fit"), HCCL_E_INTERNAL);
        record.function = reinterpret_cast<void *>(ops_hccl::StripedForestKernel);
        record.Keep(config);
        const void *arguments[] = {record.argument};
        CHK_RET_CCU(HcommCcuKernelRegister(instruction, group.dieId, record.name,
            record.function, arguments, 1, &resource.kernels[groupIndex]));
        resource.directActive[groupIndex] =
            (param.myRank == param.root || ContainsRank(group, param.root)) ? 1U : 0U;
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(instruction));
    return HCCL_SUCCESS;
}

HcclResult BuildResources(HcclComm comm, const OpParam &param, uint64_t totalBytes,
    ForestResourceContext &resource)
{
    std::vector<ChannelGroup> groups;
    std::vector<uint32_t> serverByRank;
    CHK_RET(DiscoverServerBuckets(comm, param.rankSize, serverByRank));
    BuildRingOrders(serverByRank, resource.ringOrders);
    CHK_RET(AcquireChannels(comm, param, groups));
    if (groups.size() > 1) {
        resource.auxiliaryThreads.resize(groups.size() - 1);
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU,
            resource.auxiliaryThreads.size(), THREAD_NOTIFY_COUNT, resource.auxiliaryThreads.data()));
    }
    CHK_RET(RegisterForestKernels(comm, param, totalBytes, serverByRank, groups, resource));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[ForestV4] finals implementation expects float32"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[ForestV4] byte count overflows uint64"), HCCL_E_PARA);

    OpParam param{};
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[ForestV4] unsupported rank size %u", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[ForestV4] root %u is outside rank size %u", root, param.rankSize), HCCL_E_PARA);
    if (count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    const uint64_t totalBytes = count * sizeof(float);

    const int tagLength = std::snprintf(param.tag, sizeof(param.tag), "%s_n%u_r%u_b%llu",
        FOREST_TAG, param.rankSize, root, static_cast<unsigned long long>(totalBytes));
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("[ForestV4] resource tag does not fit"), HCCL_E_INTERNAL);

    constexpr CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream, THREAD_NOTIFY_COUNT, &param.cpuThread));

    void *context = nullptr;
    uint64_t contextSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, engine, &context, &contextSize) == HCCL_SUCCESS) {
        param.resCtx = context;
        param.ctxSize = contextSize;
    } else {
        ForestResourceContext hostContext;
        CHK_RET(BuildResources(comm, param, totalBytes, hostContext));
        std::vector<char> encoded = hostContext.Encode();
        param.ctxSize = encoded.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, encoded.data(), encoded.size(), 0));
    }
    return ops_hccl::ExecOp(param);
}
