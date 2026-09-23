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
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_launch.h>

// Initialization and newly cached plans are the only topology diagnostics.
// Make them visible with the template's default build; an explicit -DLOG_LEVEL
// still takes precedence. Other translation units retain their own defaults.
#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_INFO
#endif
#include "log.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr CommEngine SCATTER_ENGINE = COMM_ENGINE_CCU;
constexpr uint64_t METADATA_PREFIX_BYTES = 8;
constexpr uint32_t RESOURCE_READY = 4;
#define SCATTER_STRINGIFY_INNER(value) #value
#define SCATTER_STRINGIFY(value) SCATTER_STRINGIFY_INNER(value)
constexpr char RESOURCE_TAG[] = "hccl_scatter_v025_f" SCATTER_STRINGIFY(SCATTER_V016_FEATURE_MASK) "_o" SCATTER_STRINGIFY(SCATTER_V016_OPT_MASK) "_t" SCATTER_STRINGIFY(SCATTER_V017_TUNE_MASK) "_p" SCATTER_STRINGIFY(SCATTER_V018_OPT_MASK) "_q" SCATTER_STRINGIFY(SCATTER_V019_OPT_MASK) "_v" SCATTER_STRINGIFY(SCATTER_V020_OPT_MASK) "_a" SCATTER_STRINGIFY(SCATTER_V021_OPT_MASK) "_z" SCATTER_STRINGIFY(SCATTER_V022_OPT_MASK) "_w" SCATTER_STRINGIFY(SCATTER_V023_OPT_MASK) "_y" SCATTER_STRINGIFY(SCATTER_V024_OPT_MASK) "_k" SCATTER_STRINGIFY(SCATTER_V025_OPT_MASK) "_resources";
// One schema prefix for registration, filtering and parsing. Keep this
// independent of build/cache tags: a partial rename prevents initialization.
#define SCATTER_METADATA_SCHEMA_PREFIX "sc8_"
constexpr char METADATA_TAG_PREFIX[] = SCATTER_METADATA_SCHEMA_PREFIX;
constexpr char METADATA_TAG_FORMAT[] = SCATTER_METADATA_SCHEMA_PREFIX "r%u_n%u_k%u_m%08x_c%016llx";
constexpr char METADATA_TAG_SCAN[] = SCATTER_METADATA_SCHEMA_PREFIX "r%u_n%u_k%u_m%x_c%llx%n";
#undef SCATTER_METADATA_SCHEMA_PREFIX
constexpr char BUILD_ID[] = "V025_CREDIT_PUSH_F" SCATTER_STRINGIFY(SCATTER_V016_FEATURE_MASK) "_o" SCATTER_STRINGIFY(SCATTER_V016_OPT_MASK) "_t" SCATTER_STRINGIFY(SCATTER_V017_TUNE_MASK) "_p" SCATTER_STRINGIFY(SCATTER_V018_OPT_MASK) "_q" SCATTER_STRINGIFY(SCATTER_V019_OPT_MASK) "_v" SCATTER_STRINGIFY(SCATTER_V020_OPT_MASK) "_a" SCATTER_STRINGIFY(SCATTER_V021_OPT_MASK) "_z" SCATTER_STRINGIFY(SCATTER_V022_OPT_MASK) "_w" SCATTER_STRINGIFY(SCATTER_V023_OPT_MASK) "_y" SCATTER_STRINGIFY(SCATTER_V024_OPT_MASK) "_k" SCATTER_STRINGIFY(SCATTER_V025_OPT_MASK);
using AddressKey = std::array<uint8_t, COMM_ADDR_EID_LEN + 1>;
using LinkKey = std::pair<AddressKey, AddressKey>;

struct PeerResource {
    uint32_t rank = 0;
    uint32_t dieId = SCATTER_UNKNOWN_DIE;
    ChannelHandle channel = 0;
};

bool MakeAddressKey(const CommAddr &addr, AddressKey &key)
{
    key.fill(0);
    key[0] = static_cast<uint8_t>(addr.type);
    switch (addr.type) {
        case COMM_ADDR_TYPE_EID:
            std::memcpy(key.data() + 1, addr.eid, COMM_ADDR_EID_LEN);
            return true;
        case COMM_ADDR_TYPE_IP_V4:
            std::memcpy(key.data() + 1, &addr.addr, sizeof(addr.addr));
            return true;
        case COMM_ADDR_TYPE_IP_V6:
            std::memcpy(key.data() + 1, &addr.addr6, sizeof(addr.addr6));
            return true;
        case COMM_ADDR_TYPE_ID:
            for (uint32_t i = 0; i < sizeof(addr.id); ++i) {
                key[i + 1] = static_cast<uint8_t>(addr.id >> (8 * (sizeof(addr.id) - i - 1)));
            }
            return true;
        default:
            return false;
    }
}

HcclResult MakeChannelDesc(HcclComm comm, const OpParam &param, uint32_t peer,
    const std::vector<uint32_t> &layers, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    const uint32_t first = std::min(param.myRank, peer);
    const uint32_t second = std::max(param.myRank, peer);
    for (const uint32_t layer : layers) {
        uint32_t *ranks = nullptr;
        uint32_t rankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &ranks, &rankCount));
        if (rankCount == 0) {
            continue;
        }
        CHK_PTR_NULL(ranks);
        const std::vector<uint32_t> layerRanks(ranks, ranks + rankCount);
        if (std::find(layerRanks.begin(), layerRanks.end(), peer) == layerRanks.end()) {
            continue;
        }

        // Both sides query the same direction and select the same address key.
        // Never compare ABI padding or depend on the graph's link list order.
        CommLink *links = nullptr;
        uint32_t linkCount = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layer, first, second, &links, &linkCount));
        if (linkCount == 0) {
            continue;
        }
        CHK_PTR_NULL(links);
        const std::vector<CommLink> peerLinks(links, links + linkCount);
        bool found = false;
        CommLink selected{};
        LinkKey selectedKey{};
        for (const CommLink &link : peerLinks) {
            if (link.linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            LinkKey key{};
            if (!MakeAddressKey(link.srcEndpointDesc.commAddr, key.first) ||
                !MakeAddressKey(link.dstEndpointDesc.commAddr, key.second)) {
                continue;
            }
            if (!found || key < selectedKey) {
                selected = link;
                selectedKey = key;
                found = true;
            }
        }
        if (found) {
            desc.remoteRank = peer;
            desc.channelProtocol = selected.linkAttr.linkProtocol;
            desc.localEndpoint = param.myRank == first ? selected.srcEndpointDesc : selected.dstEndpointDesc;
            desc.remoteEndpoint = param.myRank == first ? selected.dstEndpointDesc : selected.srcEndpointDesc;
            // READY, DONE and ACK use 0..2; adaptive phase requests use slot 3.
            desc.notifyNum = SCATTER_CHANNEL_NOTIFY_COUNT;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("No CCU CTP link between rank %u and rank %u", param.myRank, peer);
    return HCCL_E_NOT_FOUND;
}

const char *TopologyStatusName(ScatterTopologyStatus status)
{
    switch (status) {
        case ScatterTopologyStatus::NOT_QUERIED: return "NOT_QUERIED";
        case ScatterTopologyStatus::FOUND: return "FOUND";
        case ScatterTopologyStatus::NO_MESH_INSTANCE: return "NO_MESH_INSTANCE";
        case ScatterTopologyStatus::QUERY_FAILED: return "QUERY_FAILED";
        case ScatterTopologyStatus::INVALID_MEMBERS: return "INVALID_MEMBERS";
        case ScatterTopologyStatus::CONFLICTING_MESH: return "CONFLICTING_MESH";
        case ScatterTopologyStatus::SINGLE_RANK: return "SINGLE_RANK";
        default: return "INVALID_STATUS";
    }
}

void QueryLocalTopology(HcclComm comm, const OpParam &param,
    const std::vector<uint32_t> &layers, ScatterCommResources &ctx)
{
    ctx.topologyKnown = 0;
    ctx.localRankMask = uint32_t{1} << param.myRank;
    ctx.topologyStatus = ScatterTopologyStatus::NO_MESH_INSTANCE;
    uint32_t candidateMask = 0;
    bool queryFailed = false;
    bool invalidMembers = false;
    bool conflictingMesh = false;
    // Layer identifiers and rank numbering do not encode Server membership.
    // A CUSTOM layer may contain MESH instances. Its type is diagnostic only.
    // Inspect every instance before committing a candidate: do not pick the
    // first Mesh, merge distinct memberships, or hide an incomplete query.
    for (const uint32_t layer : layers) {
        CommTopo layerType = COMM_TOPO_RESERVED;
        const HcclResult layerResult = HcclRankGraphGetTopoTypeByLayer(comm, layer, &layerType);
        HCCL_INFO("Scatter V023 fix1 rank %u layer=%u layerType=%u layerTypeResult=%d diagnosticOnly=1",
            param.myRank, layer, static_cast<uint32_t>(layerType), static_cast<int>(layerResult));
        uint32_t *instanceData = nullptr;
        uint32_t instanceCount = 0;
        const HcclResult instancesResult = HcclRankGraphGetTopoInstsByLayer(comm, layer,
            &instanceData, &instanceCount);
        if (instancesResult != HCCL_SUCCESS || (instanceCount != 0 && instanceData == nullptr)) {
            queryFailed = true;
            HCCL_WARNING("Scatter V023 fix1 rank %u layer=%u instance query incomplete result=%d count=%u",
                param.myRank, layer, static_cast<int>(instancesResult), instanceCount);
            continue;
        }
        if (instanceCount == 0) {
            HCCL_INFO("Scatter V023 fix1 rank %u layer=%u instanceCount=0", param.myRank, layer);
            continue;
        }
        // The library owns every returned list and can reuse it on the next
        // query. Copy each list before any other rank-graph API is called.
        const std::vector<uint32_t> instances(instanceData, instanceData + instanceCount);
        for (const uint32_t instance : instances) {
            CommTopo instanceType = COMM_TOPO_RESERVED;
            const HcclResult typeResult = HcclRankGraphGetTopoType(comm, layer, instance, &instanceType);
            if (typeResult != HCCL_SUCCESS) {
                queryFailed = true;
                HCCL_WARNING("Scatter V023 fix1 rank %u layer=%u instance=%u type query failed result=%d",
                    param.myRank, layer, instance, static_cast<int>(typeResult));
                continue;
            }
            HCCL_INFO("Scatter V023 fix1 rank %u layer=%u instance=%u instanceType=%u",
                param.myRank, layer, instance, static_cast<uint32_t>(instanceType));
            if (instanceType != COMM_TOPO_1DMESH) {
                continue;
            }
            uint32_t *rankData = nullptr;
            uint32_t rankCount = 0;
            const HcclResult ranksResult = HcclRankGraphGetRanksByTopoInst(comm, layer, instance,
                &rankData, &rankCount);
            if (ranksResult != HCCL_SUCCESS) {
                queryFailed = true;
                HCCL_WARNING("Scatter V023 fix1 rank %u layer=%u instance=%u Mesh member query failed result=%d",
                    param.myRank, layer, instance, static_cast<int>(ranksResult));
                continue;
            }
            if (rankData == nullptr || rankCount == 0 || rankCount > param.rankSize) {
                invalidMembers = true;
                HCCL_WARNING("Scatter V023 fix1 rank %u layer=%u instance=%u invalid Mesh member list count=%u",
                    param.myRank, layer, instance, rankCount);
                continue;
            }
            const std::vector<uint32_t> members(rankData, rankData + rankCount);
            uint32_t mask = 0;
            bool valid = true;
            for (const uint32_t member : members) {
                if (member >= param.rankSize || (mask & (uint32_t{1} << member)) != 0) {
                    valid = false;
                    break;
                }
                mask |= uint32_t{1} << member;
            }
            if (!valid) {
                invalidMembers = true;
                HCCL_WARNING("Scatter V023 fix1 rank %u layer=%u instance=%u invalid Mesh members count=%u mask=%08x",
                    param.myRank, layer, instance, rankCount, mask);
                continue;
            }
            if ((mask & (uint32_t{1} << param.myRank)) == 0) {
                // An instance list can include other groups in this network
                // instance. Only Mesh memberships containing this rank compete.
                HCCL_INFO("Scatter V023 fix1 rank %u layer=%u instance=%u Mesh mask=%08x ignored=notMember",
                    param.myRank, layer, instance, mask);
                continue;
            }
            HCCL_INFO("Scatter V023 fix1 rank %u layer=%u instance=%u Mesh count=%u mask=%08x",
                param.myRank, layer, instance, rankCount, mask);
            if (candidateMask != 0 && candidateMask != mask) {
                conflictingMesh = true;
                HCCL_WARNING("Scatter V023 fix1 rank %u conflicting Mesh masks=%08x,%08x layer=%u instance=%u",
                    param.myRank, candidateMask, mask, layer, instance);
            } else {
                candidateMask = mask;
            }
        }
    }
    if (conflictingMesh) {
        ctx.topologyStatus = ScatterTopologyStatus::CONFLICTING_MESH;
    } else if (invalidMembers) {
        ctx.topologyStatus = ScatterTopologyStatus::INVALID_MEMBERS;
    } else if (queryFailed) {
        ctx.topologyStatus = ScatterTopologyStatus::QUERY_FAILED;
    } else if (candidateMask != 0) {
        ctx.topologyStatus = ScatterTopologyStatus::FOUND;
        ctx.localRankMask = candidateMask;
        ctx.topologyKnown = 1;
    }
    if (ctx.topologyKnown != 0) {
        HCCL_INFO("Scatter V023 fix1 rank %u topologyStatus=%s(%u) localMask=%08x",
            param.myRank, TopologyStatusName(ctx.topologyStatus),
            static_cast<uint32_t>(ctx.topologyStatus), ctx.localRankMask);
    } else {
        HCCL_WARNING("Scatter V023 fix1 rank %u topologyStatus=%s(%u) localMask=%08x topologyKnown=0 "
            "queryFailed=%u invalidMembers=%u conflictingMesh=%u; advertise unknown membership",
            param.myRank, TopologyStatusName(ctx.topologyStatus), static_cast<uint32_t>(ctx.topologyStatus),
            ctx.localRankMask, static_cast<uint32_t>(queryFailed), static_cast<uint32_t>(invalidMembers),
            static_cast<uint32_t>(conflictingMesh));
    }
}

HcclResult SaveResources(HcclComm comm, const ScatterCommResources &resources)
{
    return HcclEngineCtxCopy(comm, SCATTER_ENGINE, RESOURCE_TAG,
        &resources, sizeof(resources), 0);
}

HcclResult ReadRemoteMetadata(HcclComm comm, uint32_t peer, ScatterCommResources &resources)
{
    uint32_t memCount = 0;
    CommMem *mems = nullptr;
    char **tags = nullptr;
    CHK_RET(HcclChannelGetRemoteMems(comm, resources.channels[peer], &memCount, &mems, &tags));
    if (memCount == 0 || mems == nullptr || tags == nullptr) {
        return HCCL_E_NOT_FOUND;
    }
    const std::vector<CommMem> remoteMems(mems, mems + memCount);
    std::vector<std::string> remoteTags;
    remoteTags.reserve(memCount);
    for (uint32_t i = 0; i < memCount; ++i) {
        remoteTags.emplace_back(tags[i] == nullptr ? "" : tags[i]);
    }
    bool found = false;
    for (uint32_t i = 0; i < memCount; ++i) {
        const char *tag = remoteTags[i].c_str();
        const CommMem &memory = remoteMems[i];
        if (std::strncmp(tag, METADATA_TAG_PREFIX,
            sizeof(METADATA_TAG_PREFIX) - 1) != 0) {
            continue;
        }
        uint32_t rank = 0;
        uint32_t size = 0;
        uint32_t known = 0;
        uint32_t mask = 0;
        unsigned long long capacity = 0;
        int consumed = 0;
        const int fields = std::sscanf(tag, METADATA_TAG_SCAN,
            &rank, &size, &known, &mask, &capacity, &consumed);
        const uint32_t validRanks = (uint32_t{1} << resources.rankSize) - 1;
        const uint64_t addr = reinterpret_cast<uintptr_t>(memory.addr);
        if (found || fields != 5 || consumed <= 0 || tag[consumed] != '\0' ||
            rank != peer || size != resources.rankSize || known > 1 ||
            (mask & ~validRanks) != 0 || (mask & (uint32_t{1} << peer)) == 0 ||
            (known == 0 && mask != (uint32_t{1} << peer)) ||
            memory.type != COMM_MEM_TYPE_DEVICE || memory.addr == nullptr ||
            memory.size != METADATA_PREFIX_BYTES || capacity < METADATA_PREFIX_BYTES ||
            capacity > std::numeric_limits<uintptr_t>::max() - addr) {
            HCCL_ERROR("Scatter V023 fix1 invalid resource metadata from peer %u", peer);
            return HCCL_E_PARA;
        }
        resources.serverMasks[peer] = known != 0 ? mask : 0;
        resources.capacities[peer] = static_cast<uint64_t>(capacity);
        found = true;
    }
    return found ? HCCL_SUCCESS : HCCL_E_NOT_FOUND;
}

HcclResult InitializeResources(HcclComm comm, const OpParam &param, ScatterCommResources &resources)
{
    HCCL_INFO("Scatter build=%s rank=%u rankSize=%u initialize communicator resources "
        "adaptiveArgCount=%u sqeArgCapacity=%u fullSqeArgs=1",
        BUILD_ID, param.myRank, param.rankSize, static_cast<uint32_t>(ops_hccl::ADAPTIVE_ARG_COUNT),
        ops_hccl::SCATTER_SQE_ARG_CAPACITY);
    HCCL_INFO("Scatter build=%s rank=%u adaptiveHandshake=request-ready notifySlots=%u "
        "requestSlot=%u dataRequestMask=%u ackRequestMask=%u defaultSendPhases=DATA,REQUEST_ACK,WAIT_ACK rootFusionOptIn=1",
        BUILD_ID, param.myRank, SCATTER_CHANNEL_NOTIFY_COUNT, ops_hccl::ADAPTIVE_REQUEST_NOTIFY,
        static_cast<uint32_t>(ops_hccl::ADAPTIVE_DATA_REQUEST),
        static_cast<uint32_t>(ops_hccl::ADAPTIVE_ACK_REQUEST));
    resources.myRank = param.myRank;
    resources.rankSize = param.rankSize;
    resources.localRankMask = uint32_t{1} << param.myRank;
    std::fill(std::begin(resources.dies), std::end(resources.dies), SCATTER_UNKNOWN_DIE);
    resources.initStage = 1;
    CHK_RET(SaveResources(comm, resources));
    if (param.rankSize == 1) {
        resources.topologyKnown = 1;
        resources.topologyStatus = ScatterTopologyStatus::SINGLE_RANK;
        resources.serverMasks[param.myRank] = resources.localRankMask;
        resources.initStage = RESOURCE_READY;
        return SaveResources(comm, resources);
    }

    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    if (layerCount == 0) {
        return HCCL_E_NOT_FOUND;
    }
    CHK_PTR_NULL(layerData);
    // The library owns and may reuse the returned storage.
    std::vector<uint32_t> layers(layerData, layerData + layerCount);
    std::sort(layers.begin(), layers.end());
    layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
    QueryLocalTopology(comm, param, layers, resources);

    void *scratch = nullptr;
    CHK_RET(HcclGetHcclBuffer(comm, &scratch, &resources.scratchBytes));
    resources.scratchAddress = reinterpret_cast<uintptr_t>(scratch);
    if (scratch == nullptr || resources.scratchBytes < METADATA_PREFIX_BYTES ||
        resources.scratchBytes > std::numeric_limits<uintptr_t>::max() - resources.scratchAddress) {
        return HCCL_E_PARA;
    }
    resources.serverMasks[param.myRank] = resources.topologyKnown != 0 ? resources.localRankMask : 0;
    resources.capacities[param.myRank] = resources.scratchBytes;
    char metadataTag[128]{};
    const int tagLength = std::snprintf(metadataTag, sizeof(metadataTag),
        METADATA_TAG_FORMAT, param.myRank, param.rankSize,
        resources.topologyKnown, resources.localRankMask,
        static_cast<unsigned long long>(resources.scratchBytes));
    if (tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(metadataTag)) {
        return HCCL_E_INTERNAL;
    }
    // Capacity and membership are control-plane tag metadata. Register only a
    // prefix because CCU serializes the exchanged memory size as uint32_t.
    // No Host payload read/write or bootstrap communication is performed.
    CommMem memory{COMM_MEM_TYPE_DEVICE, scratch, METADATA_PREFIX_BYTES};
    CHK_RET(HcclCommMemReg(comm, metadataTag, &memory, &resources.metadataHandle));
    CHK_PTR_NULL(resources.metadataHandle);
    resources.initStage = 2;
    CHK_RET(SaveResources(comm, resources));

    std::vector<HcclChannelDesc> descs;
    std::vector<uint32_t> ranks;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        HcclChannelDesc desc{};
        CHK_RET(MakeChannelDesc(comm, param, peer, layers, desc));
        desc.memHandles = &resources.metadataHandle;
        desc.memHandleNum = 1;
        EndpointAttrDieId dieId = SCATTER_UNKNOWN_DIE;
        const HcclResult dieResult = HcclRankGraphGetEndpointInfo(comm, param.myRank,
            &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId);
        if (dieResult == HCCL_SUCCESS && dieId < SCATTER_MAX_DIES) {
            resources.dies[peer] = dieId;
        } else {
            HCCL_WARNING("Scatter V023 fix1 rank %u peer %u has unknown CCU die; use sequential kernels",
                param.myRank, peer);
        }
        ranks.push_back(peer);
        descs.push_back(desc);
    }
    std::vector<ChannelHandle> handles(ranks.size());
    CHK_RET(HcclChannelAcquire(comm, SCATTER_ENGINE, descs.data(),
        static_cast<uint32_t>(descs.size()), handles.data()));
    for (size_t i = 0; i < ranks.size(); ++i) {
        if (handles[i] == 0) {
            return HCCL_E_INTERNAL;
        }
        resources.channels[ranks[i]] = handles[i];
    }
    resources.initStage = 3;
    CHK_RET(SaveResources(comm, resources));
    for (const uint32_t peer : ranks) {
        CHK_RET(ReadRemoteMetadata(comm, peer, resources));
    }
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (resources.serverMasks[rank] == 0) {
            HCCL_WARNING("Scatter V023 fix1 rank %u metadata rank=%u MeshMask=%08x capacity=%llu; "
                "unknown membership forces communicator-wide DIRECT fallback",
                param.myRank, rank, resources.serverMasks[rank],
                static_cast<unsigned long long>(resources.capacities[rank]));
        } else {
            HCCL_INFO("Scatter V023 fix1 rank %u metadata rank=%u MeshMask=%08x capacity=%llu",
                param.myRank, rank, resources.serverMasks[rank],
                static_cast<unsigned long long>(resources.capacities[rank]));
        }
    }
    // A pooled worker is optional. Ordered invocations join it before reuse;
    // if unavailable, DATA and ACK phases still execute safely in sequence.
    const HcclResult workerResult = HcclThreadAcquire(comm, SCATTER_ENGINE, 1, 1, &resources.worker);
    if (workerResult != HCCL_SUCCESS) {
        resources.worker = 0;
        HCCL_WARNING("Scatter V023 fix1 CCU worker unavailable (%d)", static_cast<int>(workerResult));
    }
    resources.initStage = RESOURCE_READY;
    return SaveResources(comm, resources);
}

HcclResult AcquireResources(HcclComm comm, const OpParam &param, ScatterCommResources &resources)
{
    void *cached = nullptr;
    uint64_t size = 0;
    const HcclResult getResult = HcclEngineCtxGet(comm, RESOURCE_TAG, SCATTER_ENGINE, &cached, &size);
    if (getResult == HCCL_SUCCESS) {
        if (cached == nullptr || size != sizeof(resources)) {
            return HCCL_E_PARA;
        }
        std::memcpy(&resources, cached, sizeof(resources));
        if (resources.magic != SCATTER_CTX_MAGIC || resources.version != SCATTER_CTX_VERSION ||
            resources.myRank != param.myRank || resources.rankSize != param.rankSize ||
            resources.initStage != RESOURCE_READY) {
            // Keep incomplete state: a retry must not duplicate registration
            // tags or peer channels after a partially successful allocation.
            HCCL_ERROR("Scatter V023 fix1 resource initialization incomplete or incompatible");
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }
    if (getResult != HCCL_E_NOT_FOUND) {
        return getResult;
    }
    CHK_RET(HcclEngineCtxCreate(comm, RESOURCE_TAG, SCATTER_ENGINE, sizeof(resources), &cached));
    return InitializeResources(comm, param, resources);
}

uint32_t PeerCount(uint32_t mask)
{
    uint32_t count = 0;
    for (; mask != 0; mask >>= 1) {
        count += mask & 1U;
    }
    return count;
}

bool IsRemotePeer(const AlgResourceCtx &ctx, uint32_t rank)
{
    return ctx.topologyKnown != 0 && (ctx.localRankMask & (uint32_t{1} << rank)) == 0;
}

void BuildGroups(const OpParam &param, const std::vector<PeerResource> &peers,
    AlgResourceCtx &ctx, std::vector<ops_hccl::ScatterKernelArg> &args)
{
    const bool allDiesKnown = std::all_of(peers.begin(), peers.end(), [](const PeerResource &peer) {
        return peer.dieId < SCATTER_MAX_DIES;
    });
    ctx.schedule = allDiesKnown ? ScatterSchedule::PARALLEL : ScatterSchedule::SEQUENTIAL;
    for (const PeerResource &peer : peers) {
        uint32_t group = ctx.groupCount;
        if (allDiesKnown) {
            for (uint32_t i = 0; i < ctx.groupCount; ++i) {
                if (ctx.groups[i].dieId == peer.dieId) {
                    group = i;
                    break;
                }
            }
        }
        if (group == ctx.groupCount) {
            ctx.groups[group].dieId = peer.dieId;
            ++ctx.groupCount;
        }
        ScatterKernelGroup &entry = ctx.groups[group];
        entry.peerMask |= uint32_t{1} << peer.rank;
        if (ctx.topologyKnown != 0) {
            if (IsRemotePeer(ctx, peer.rank)) {
                ++entry.remotePeerCount;
            } else {
                ++entry.localPeerCount;
            }
        }
    }
    // Place the group with the most Clos traffic first; peer count breaks ties.
    // This ordering comes from the actual Mesh membership, not a case number.
    std::stable_sort(ctx.groups, ctx.groups + ctx.groupCount,
        [](const ScatterKernelGroup &left, const ScatterKernelGroup &right) {
            if (left.remotePeerCount != right.remotePeerCount) {
                return left.remotePeerCount > right.remotePeerCount;
            }
            return PeerCount(left.peerMask) > PeerCount(right.peerMask);
        });
    args.resize(ctx.groupCount);
    for (uint32_t i = 0; i < ctx.groupCount; ++i) {
        auto &arg = args[i];
        arg.role = param.myRank == param.root
            ? ops_hccl::ScatterKernelRole::SEND : ops_hccl::ScatterKernelRole::RECEIVE;
        arg.dataPath = ctx.dataPath;
        arg.copyOwner = ctx.directFusion != 0 && param.myRank == param.root && i + 1 == ctx.groupCount ? 1 : 0;
        std::vector<PeerResource> members;
        for (const PeerResource &peer : peers) {
            if ((ctx.groups[i].peerMask & (uint32_t{1} << peer.rank)) != 0) {
                members.push_back(peer);
            }
        }
        std::stable_sort(members.begin(), members.end(), [&ctx](const PeerResource &left,
            const PeerResource &right) {
            return IsRemotePeer(ctx, left.rank) && !IsRemotePeer(ctx, right.rank);
        });
        for (const PeerResource &member : members) {
            arg.channels[arg.peerCount] = member.channel;
            arg.peers[arg.peerCount] = member.rank;
            ++arg.peerCount;
        }
        arg.mergeEvents = arg.role == ops_hccl::ScatterKernelRole::SEND && arg.peerCount > 1 &&
            ctx.dataPath == ScatterDataPath::SINGLE_WAVE ? ops_hccl::SelectScatterEventMerge(ctx.sceneId) : 0;
        arg.readyWrite = arg.role == ops_hccl::ScatterKernelRole::SEND && arg.peerCount > 1 &&
            ctx.dataPath == ScatterDataPath::SINGLE_WAVE ? ops_hccl::SelectScatterReadyWrite(ctx.sceneId) : 0;
        arg.earlyCopy = arg.copyOwner != 0 && ctx.dataPath == ScatterDataPath::SINGLE_WAVE ?
            ops_hccl::SelectScatterEarlyCopy(ctx.sceneId) : 0;
        arg.pullRead = ctx.dataPath == ScatterDataPath::SINGLE_WAVE ?
            ops_hccl::SelectScatterPullRead(ctx.algorithm, ctx.sceneId, ctx.recvBytes) : 0;
        arg.pullRead |= ops_hccl::SelectScatterExtendedPull(ctx);
        arg.shortPush = ops_hccl::SelectScatterShortPush(ctx);
        if (arg.shortPush != 0) { arg.pullRead = 0; }
        if (arg.role == ops_hccl::ScatterKernelRole::SEND && arg.peerCount > 1) {
            arg.readyWrite |= ops_hccl::SelectScatterLargeReadyWrite(ctx);
        }
        const uint32_t smallPipeline = ctx.topologyKnown != 0 ?
            ops_hccl::SelectScatterSmallPipeline(ctx.algorithm, ctx.topology,
                ctx.sceneId, param.rankSize, ctx.dataPath, ctx.recvBytes) : 0;
        if (arg.role == ops_hccl::ScatterKernelRole::SEND && smallPipeline != 0) {
            if (arg.peerCount > 1) {
                arg.readyWrite |= (smallPipeline & 1U) != 0 ? 1U : 0U;
                arg.mergeEvents |= (smallPipeline & 2U) != 0 ? 1U : 0U;
            }
            arg.deferCopyWait = (smallPipeline & 4U) != 0 && arg.copyOwner != 0 &&
                arg.earlyCopy == 0 && arg.pullRead == 0 ? 1U : 0U;
        }
        if (arg.pullRead != 0) {
            // Pull replaces READY/Write, not the invocation-level Copy/alias
            // decision. No old Write event masks are allocated on this path.
            arg.mergeEvents = 0;
            arg.readyWrite = 0;
            arg.earlyCopy = 0;
            arg.deferCopyWait = 0;
            arg.pullAddressMode = ops_hccl::SelectScatterEffectivePullAddressMode(ctx);
            arg.pullRoot = arg.pullAddressMode != ScatterPullAddressMode::SLICE ? param.root : 0;
            arg.pullPeerOrder = ctx.topologyKnown != 0 && arg.role == ops_hccl::ScatterKernelRole::SEND ?
                ops_hccl::SelectScatterPullPeerOrder(ctx.algorithm, ctx.topology,
                    ctx.sceneId, param.rankSize, ctx.dataPath, ctx.recvBytes) : 0U;
            arg.pullEarlyCopy = (SCATTER_V020_OPTIONS & 8U) != 0 && ctx.sceneId == 4 &&
                arg.role == ops_hccl::ScatterKernelRole::SEND && arg.copyOwner != 0 ? 1U : 0U;
            arg.pullMsBytes = ops_hccl::SelectScatterPullMsBytes(ctx);
            arg.pullMsTail = ops_hccl::SelectScatterPullMsTail(ctx);
        }
    }
}

HcclResult RegisterOneKernel(CcuInsHandle instance, const OpParam &param, uint32_t index,
    const ops_hccl::ScatterKernelArg &arg, CcuKernelHandle &kernel)
{
    char name[128]{};
    const int length = std::snprintf(name, sizeof(name), "CcuS25_f" SCATTER_STRINGIFY(SCATTER_V016_FEATURE_MASK) "_o" SCATTER_STRINGIFY(SCATTER_V016_OPT_MASK) "_t" SCATTER_STRINGIFY(SCATTER_V017_TUNE_MASK) "_p" SCATTER_STRINGIFY(SCATTER_V018_OPT_MASK) "_q" SCATTER_STRINGIFY(SCATTER_V019_OPT_MASK) "_v" SCATTER_STRINGIFY(SCATTER_V020_OPT_MASK) "_a" SCATTER_STRINGIFY(SCATTER_V021_OPT_MASK) "_z" SCATTER_STRINGIFY(SCATTER_V022_OPT_MASK) "_w" SCATTER_STRINGIFY(SCATTER_V023_OPT_MASK) "_y" SCATTER_STRINGIFY(SCATTER_V024_OPT_MASK) "_k" SCATTER_STRINGIFY(SCATTER_V025_OPT_MASK) "_r%u_m%u_n%llu_g%u_x%u",
        param.root, param.myRank, static_cast<unsigned long long>(param.count), index, arg.shortPush != 0 ? 3U : (arg.pullMsTail != 0 ? 2U : (arg.pullMsBytes != 0 ? 1U : 0U)));
    if (length < 0 || static_cast<size_t>(length) >= sizeof(name)) {
        return HCCL_E_INTERNAL;
    }
    const void *kernelArgs[] = {&arg};
    // dieId is reserved in 9.1: the referenced channels determine the die.
    return ConvertCcuToHccl(HcommCcuKernelRegister(instance, 0, name,
        reinterpret_cast<const void *>(ops_hccl::CcuKernel), kernelArgs, 1, &kernel));
}

HcclResult RegisterOneRound(CcuInsHandle instance, const OpParam &param, uint32_t index,
    const ops_hccl::ScatterKernelArg &arg, CcuKernelHandle &kernel)
{
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
    const HcclResult result = RegisterOneKernel(instance, param, index, arg, kernel);
    const CcuResult endResult = HcommCcuKernelRegisterEnd(instance);
    if (result != HCCL_SUCCESS) {
        return result;
    }
    return ConvertCcuToHccl(endResult);
}

HcclResult QueryInstance(HcclComm comm, CcuInsHandle &instance)
{
    uint32_t instanceCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    if (instanceCount != 1) {
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterPullMsReceiver(CcuInsHandle instance, const OpParam &param,
    ops_hccl::ScatterKernelArg arg, CcuKernelHandle &kernel)
{
    // MS is receiver-only: the complete local plan is this single Kernel,
    // with no root Copy or other groups in its registration transaction.
    kernel = 0;
    for (uint32_t attempt = 0; attempt < 2; ++attempt) {
        CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
        CcuKernelHandle candidate = 0;
        const HcclResult result = RegisterOneKernel(instance, param, 0, arg, candidate);
        const CcuResult endResult = HcommCcuKernelRegisterEnd(instance);
        // End/translation failure cannot be retried: the SDK may already
        // have loaded part of the instruction graph, with no public rollback.
        if (endResult != CCU_SUCCESS) { return ConvertCcuToHccl(endResult); }
        if (result == HCCL_SUCCESS) {
            kernel = candidate;
            return HCCL_SUCCESS;
        }
        if (attempt != 0 || result != HCCL_E_UNAVAIL || arg.pullMsBytes == 0) { return result; }
        // Register failure aborts and invalidates the WHOLE untranslated
        // round. Its successful End has now closed it. Rebuild the complete
        // one-Kernel baseline in a fresh round; never reuse candidate.
        HCCL_WARNING("Scatter V025 receiver MS unavailable; register direct Read in a fresh round");
        arg.pullMsBytes = 0;
        arg.pullMsTail = 0;
    }
    return HCCL_E_INTERNAL;
}

HcclResult RegisterDirectKernels(HcclComm comm, const OpParam &param,
    const ScatterCommResources &common, AlgResourceCtx &ctx)
{
    CcuInsHandle instance = 0;
    CHK_RET(QueryInstance(comm, instance));
    std::vector<PeerResource> peers;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank && (param.myRank == param.root || peer == param.root)) {
            peers.push_back(PeerResource{peer, common.dies[peer], common.channels[peer]});
        }
    }

    std::vector<ops_hccl::ScatterKernelArg> args;
    BuildGroups(param, peers, ctx, args);
    const bool useMs = std::any_of(args.begin(), args.end(), [](const ops_hccl::ScatterKernelArg &arg) {
        return arg.pullMsBytes != 0;
    });
    if (useMs) {
        if (param.myRank == param.root || ctx.groupCount != 1 || args.size() != 1 ||
            args[0].role != ops_hccl::ScatterKernelRole::RECEIVE || args[0].peerCount != 1) {
            return HCCL_E_INTERNAL;
        }
        return RegisterPullMsReceiver(instance, param, args[0], ctx.groups[0].kernel);
    }
    ops_hccl::ScatterKernelArg copyArg{};
    copyArg.role = ops_hccl::ScatterKernelRole::LOCAL_COPY;
    copyArg.dataPath = ctx.dataPath;
    if (ctx.schedule == ScatterSchedule::PARALLEL) {
        // A single round gives all potentially concurrent groups disjoint
        // missions, variables and events. Each die has at most one sender and
        // one zero-channel Copy Kernel: this fits the default two missions/die.
        CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
        HcclResult result = HCCL_SUCCESS;
        for (uint32_t i = 0; i < ctx.groupCount && result == HCCL_SUCCESS; ++i) {
            result = RegisterOneKernel(instance, param, i, args[i], ctx.groups[i].kernel);
        }
        if (result == HCCL_SUCCESS && param.myRank == param.root) {
            result = RegisterOneKernel(instance, param, MAX_RANK_SIZE, copyArg, ctx.copyKernel);
        }
        const CcuResult endResult = HcommCcuKernelRegisterEnd(instance);
        if (result != HCCL_SUCCESS) {
            return result;
        }
        CHK_RET_CCU(endResult);
    } else {
        // If any die is unknown, use one peer per Kernel and one registration
        // round per Kernel. The reset reuses resources, so execution MUST stay
        // sequential for every group, even groups whose own die was known.
        for (uint32_t i = 0; i < ctx.groupCount; ++i) {
            CHK_RET(RegisterOneRound(instance, param, i, args[i], ctx.groups[i].kernel));
        }
        if (param.myRank == param.root) {
            CHK_RET(RegisterOneRound(instance, param, MAX_RANK_SIZE, copyArg, ctx.copyKernel));
        }
    }
    if (param.myRank == param.root && ctx.schedule == ScatterSchedule::PARALLEL && ctx.groupCount == 2) {
        ctx.worker = common.worker;
        if (ctx.worker == 0) {
            // Some installations do not expose the CCU worker path. All
            // resources are registered already, and no work has been launched;
            // executing these disjoint groups in order is a safe fallback.
            HCCL_WARNING("Scatter V023 fix1 direct groups use sequential worker fallback");
            ctx.schedule = ScatterSchedule::SEQUENTIAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterAdaptiveKernel(CcuInsHandle instance, const OpParam &param, uint32_t shape,
    uint32_t batch, uint32_t group, const ops_hccl::AdaptiveKernelArg &arg, CcuKernelHandle &kernel)
{
    char name[128]{};
    const int length = std::snprintf(name, sizeof(name), "CcuS25_f" SCATTER_STRINGIFY(SCATTER_V016_FEATURE_MASK) "_o" SCATTER_STRINGIFY(SCATTER_V016_OPT_MASK) "_t" SCATTER_STRINGIFY(SCATTER_V017_TUNE_MASK) "_p" SCATTER_STRINGIFY(SCATTER_V018_OPT_MASK) "_q" SCATTER_STRINGIFY(SCATTER_V019_OPT_MASK) "_v" SCATTER_STRINGIFY(SCATTER_V020_OPT_MASK) "_a" SCATTER_STRINGIFY(SCATTER_V021_OPT_MASK) "_z" SCATTER_STRINGIFY(SCATTER_V022_OPT_MASK) "_w" SCATTER_STRINGIFY(SCATTER_V023_OPT_MASK) "_y" SCATTER_STRINGIFY(SCATTER_V024_OPT_MASK) "_k" SCATTER_STRINGIFY(SCATTER_V025_OPT_MASK) "_r%u_m%u_n%llu_s%u_b%u_g%u",
        param.root, param.myRank, static_cast<unsigned long long>(param.count), shape, batch, group);
    if (length < 0 || static_cast<size_t>(length) >= sizeof(name)) {
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("Scatter build=%s registerAdaptive rank=%u root=%u shape=%u batch=%u group=%u "
        "role=%u fuseAckRequest=%u peers=%u transfers=%u", BUILD_ID, param.myRank, param.root,
        shape, batch, group, static_cast<uint32_t>(arg.role), arg.fuseAckRequest, arg.peerCount, arg.transferCount);
    const void *args[] = {&arg};
    return ConvertCcuToHccl(HcommCcuKernelRegister(instance, 0, name,
        reinterpret_cast<const void *>(ops_hccl::AdaptiveCcuKernel), args, 1, &kernel));
}

HcclResult RegisterShape(CcuInsHandle instance, const OpParam &param,
    const ops_hccl::ScatterShapePlan &plan, uint32_t shapeIndex, ThreadHandle worker, ScatterShape &shape)
{
    if (plan.batches.size() > SCATTER_MAX_BATCHES) {
        return HCCL_E_PARA;
    }
    shape.batchCount = static_cast<uint32_t>(plan.batches.size());
    for (uint32_t batchIndex = 0; batchIndex < shape.batchCount; ++batchIndex) {
        const auto &batchPlan = plan.batches[batchIndex];
        auto &batch = shape.batches[batchIndex];
        if (batchPlan.groups.empty() || batchPlan.groups.size() > MAX_RANK_SIZE ||
            (batchPlan.parallel && batchPlan.groups.size() > SCATTER_MAX_DIES)) {
            return HCCL_E_PARA;
        }
        batch.kernelCount = static_cast<uint32_t>(batchPlan.groups.size());
        batch.parallel = batchPlan.parallel && batch.kernelCount == 2 && worker != 0 ? 1 : 0;
        // Each batch joins before the next batch/wave. Different batches can
        // reuse resources; concurrent groups within a batch must not do so.
        if (batchPlan.parallel) {
            CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
            HcclResult result = HCCL_SUCCESS;
            for (uint32_t group = 0; group < batch.kernelCount && result == HCCL_SUCCESS; ++group) {
                result = RegisterAdaptiveKernel(instance, param, shapeIndex, batchIndex, group,
                    batchPlan.groups[group], batch.kernels[group]);
            }
            const CcuResult endResult = HcommCcuKernelRegisterEnd(instance);
            if (result != HCCL_SUCCESS) {
                return result;
            }
            CHK_RET_CCU(endResult);
        } else {
            for (uint32_t group = 0; group < batch.kernelCount; ++group) {
                CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
                const HcclResult result = RegisterAdaptiveKernel(instance, param, shapeIndex,
                    batchIndex, group, batchPlan.groups[group], batch.kernels[group]);
                const CcuResult endResult = HcommCcuKernelRegisterEnd(instance);
                if (result != HCCL_SUCCESS) {
                    return result;
                }
                CHK_RET_CCU(endResult);
            }
        }
    }
    return HCCL_SUCCESS;
}

bool ShapeUsesWorker(const ScatterShape &shape)
{
    for (uint32_t i = 0; i < shape.batchCount; ++i) {
        if (shape.batches[i].parallel != 0) {
            return true;
        }
    }
    return false;
}

HcclResult RegisterAdaptiveKernels(HcclComm comm, const OpParam &param,
    const ScatterCommResources &common, const ops_hccl::ScatterPlan &plan, AlgResourceCtx &ctx)
{
    CcuInsHandle instance = 0;
    CHK_RET(QueryInstance(comm, instance));
    CHK_RET(RegisterShape(instance, param, plan.full, 0, common.worker, ctx.fullShape));
    CHK_RET(RegisterShape(instance, param, plan.tail, 1, common.worker, ctx.tailShape));
    ctx.worker = ShapeUsesWorker(ctx.fullShape) || ShapeUsesWorker(ctx.tailShape) ? common.worker : 0;
    if (param.myRank == param.root) {
        ops_hccl::ScatterKernelArg copyArg{};
        copyArg.role = ops_hccl::ScatterKernelRole::LOCAL_COPY;
        copyArg.dataPath = ScatterDataPath::CHUNKED;
        // The final root copy starts after all waves and all downstream ACKs.
        CHK_RET(RegisterOneRound(instance, param, MAX_RANK_SIZE, copyArg, ctx.copyKernel));
    }
    return HCCL_SUCCESS;
}

void LogSelectedPlan(const OpParam &param, const ScatterCommResources &common,
    const ops_hccl::ScatterPlan &plan, ScatterDataPath dataPath)
{
    const bool directPolicy = plan.routeReason == ScatterRouteReason::DIRECT_POLICY ||
        plan.routeReason == ScatterRouteReason::REMOTE_FANOUT_AT_MOST_FOUR ||
        plan.routeReason == ScatterRouteReason::SINGLE_RANK;
    const bool fallback = plan.algorithm == ScatterAlgorithm::DIRECT && !directPolicy;
    uint64_t minScratch = common.scratchBytes;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        minScratch = std::min(minScratch, common.capacities[rank]);
    }
    uint32_t fineTail = 0, fineForward = 0, interleaved = 0;
    for (const auto &batch : plan.full.batches) {
        for (const auto &arg : batch.groups) {
            fineTail += arg.earlyTail == 2 ? 1U : 0U;
            interleaved += arg.interleaveTails;
            fineForward += arg.tailChannel != 0 ? 1U : 0U;
        }
    }
    char description[1024]{};
    const int length = std::snprintf(description, sizeof(description),
        "Scatter build=%s newPlan rank=%u root=%u rankSize=%u recvCount=%llu B=%llu "
        "topology=%u scene=%u algorithm=%u route=%s reason=%s(%u) "
        "topologyStatus=%s(%u) localMask=%08x directFusion=%u dataPath=%s(%u) "
        "chunk=%llu full=%llu tail=%llu scratchUsed=%llu minScratch=%llu "
        "rootAckFusion=%u localAckFusion=%u fullBatches=%u tailBatches=%u "
        "features=%u eventMerge=%u compactScratch=%u relayOverlap=%u tailFirst=%u "
        "opt=%u readyWrite=%u relayCopy=%u fineTail=%u fineForward=%u tune=%u interleavedGroups=%u earlyCopyEligible=%u "
        "v022=%u smallPipelineEligible=%u pullPeerOrderEligible=%u",
        BUILD_ID, param.myRank, param.root, param.rankSize, static_cast<unsigned long long>(param.count),
        static_cast<unsigned long long>(plan.recvBytes), static_cast<uint32_t>(plan.topology), plan.sceneId,
        static_cast<uint32_t>(plan.algorithm), fallback ? "DIRECT_FALLBACK" : (directPolicy ? "DIRECT_POLICY" : "SCENARIO"),
        ops_hccl::ScatterRouteReasonName(plan.routeReason), static_cast<uint32_t>(plan.routeReason),
        TopologyStatusName(common.topologyStatus), static_cast<uint32_t>(common.topologyStatus),
        common.localRankMask, plan.directFusion,
        plan.algorithm == ScatterAlgorithm::DIRECT
            ? (dataPath == ScatterDataPath::SINGLE_WAVE ? "SINGLE_WAVE" : "CHUNKED") : "ADAPTIVE",
        static_cast<uint32_t>(dataPath), static_cast<unsigned long long>(plan.chunkBytes),
        static_cast<unsigned long long>(plan.fullSteps), static_cast<unsigned long long>(plan.tailBytes),
        static_cast<unsigned long long>(plan.scratchUsedBytes),
        static_cast<unsigned long long>(minScratch), plan.rootAckFusion,
        param.myRank == param.root ? plan.rootAckFusion : 0,
        static_cast<uint32_t>(plan.full.batches.size()), static_cast<uint32_t>(plan.tail.batches.size()),
        SCATTER_FEATURES, ops_hccl::SelectScatterEventMerge(plan.sceneId), plan.compactScratch, plan.relayOverlap,
        param.myRank == param.root && plan.algorithm == ScatterAlgorithm::RELAY && plan.relayOverlap != 0 ? 1U : 0U,
        SCATTER_OPTIMIZATIONS, ops_hccl::SelectScatterReadyWrite(plan.sceneId), plan.relayCopy,
        fineTail, fineForward, SCATTER_TUNING, interleaved,
        param.myRank == param.root && plan.algorithm == ScatterAlgorithm::DIRECT && plan.directFusion != 0 &&
            dataPath == ScatterDataPath::SINGLE_WAVE ? ops_hccl::SelectScatterEarlyCopy(plan.sceneId) : 0U,
        SCATTER_V022_OPTIONS,
        common.topologyKnown != 0 && param.myRank == param.root ?
            ops_hccl::SelectScatterSmallPipeline(plan.algorithm, plan.topology,
                plan.sceneId, param.rankSize, dataPath, plan.recvBytes) : 0U,
        common.topologyKnown != 0 && param.myRank == param.root ?
            ops_hccl::SelectScatterPullPeerOrder(plan.algorithm, plan.topology,
                plan.sceneId, param.rankSize, dataPath, plan.recvBytes) : 0U);
    if (length < 0 || static_cast<size_t>(length) >= sizeof(description)) {
        HCCL_ERROR("Scatter V023 fix1 plan diagnostic formatting failed");
        return;
    }
    if (fallback) {
        HCCL_WARNING("%s", description);
    } else {
        HCCL_INFO("%s", description);
    }
}

HcclResult AcquireContext(HcclComm comm, OpParam &param, uint64_t recvBytes)
{
    void *ctx = nullptr;
    uint64_t size = 0;
    const HcclResult getResult = HcclEngineCtxGet(comm, param.tag, SCATTER_ENGINE, &ctx, &size);
    if (getResult == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
        return HCCL_SUCCESS;
    }
    if (getResult != HCCL_E_NOT_FOUND) {
        return getResult;
    }

    AlgResourceCtx resources{};
    resources.root = param.root;
    resources.myRank = param.myRank;
    resources.rankSize = param.rankSize;
    ScatterCommResources common{};
    CHK_RET(AcquireResources(comm, param, common));
    ops_hccl::ScatterPlan plan{};
    CHK_RET(ops_hccl::BuildScatterPlan(param, common, recvBytes, plan));
    resources.localRankMask = common.localRankMask;
    resources.topologyKnown = common.topologyKnown;
    resources.algorithm = plan.algorithm;
    resources.topology = plan.topology;
    resources.sceneId = plan.sceneId;
    resources.routeReason = plan.routeReason;
    resources.directFusion = plan.directFusion;
    resources.rootAckFusion = plan.rootAckFusion;
    resources.compactScratch = plan.compactScratch;
    resources.relayOverlap = plan.relayOverlap;
    resources.relayCopy = plan.relayCopy;
    resources.recvBytes = recvBytes;
    resources.chunkBytes = plan.chunkBytes;
    resources.fullSteps = plan.fullSteps;
    resources.tailBytes = plan.tailBytes;
    resources.scratchAddress = common.scratchAddress;
    resources.scratchBytes = common.scratchBytes;
    resources.scratchUsedBytes = plan.scratchUsedBytes;
    resources.dataPath = ops_hccl::SelectScatterDataPath(
        plan.algorithm, plan.sceneId, plan.directFusion, recvBytes);
    LogSelectedPlan(param, common, plan, resources.dataPath);
    if (plan.algorithm == ScatterAlgorithm::DIRECT) {
        CHK_RET(RegisterDirectKernels(comm, param, common, resources));
    } else {
        CHK_RET(RegisterAdaptiveKernels(comm, param, common, plan, resources));
    }
    param.ctxSize = sizeof(resources);
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, SCATTER_ENGINE, param.ctxSize, &param.resCtx));
    const HcclResult copyResult = HcclEngineCtxCopy(comm, SCATTER_ENGINE, param.tag,
        &resources, sizeof(resources), 0);
    if (copyResult != HCCL_SUCCESS) {
        (void)HcclEngineCtxDestroy(comm, param.tag, SCATTER_ENGINE);
    }
    return copyResult;
}

HcclResult RegisterOpInfo(HcclComm comm, const OpParam &param, uint64_t recvBytes, uint64_t inputBytes)
{
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    HcclDfxOpInfo info{};
    info.opType = static_cast<uint32_t>(param.opType);
    info.dataType = static_cast<uint32_t>(param.dataType);
    info.outputType = static_cast<uint32_t>(param.dataType);
    info.dataCount = param.count;
    info.root = param.root;
    info.engine = SCATTER_ENGINE;
    info.cpuTsThread = param.cpuThread;
    info.inputMemAddr = param.myRank == param.root ? reinterpret_cast<uintptr_t>(param.inputPtr) : 0;
    info.inputMemSize = param.myRank == param.root ? inputBytes : 0;
    info.outputMemAddr = reinterpret_cast<uintptr_t>(param.outputPtr);
    info.outputMemSize = recvBytes;
    const int tagLength = std::snprintf(info.algTag, sizeof(info.algTag), "%s", param.tag);
    if (tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(info.algTag)) {
        return HCCL_E_INTERNAL;
    }
    return HcclDfxRegOpInfoByCommId(commName, &info);
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    uint64_t recvBytes = 0;
    uint64_t inputBytes = 0;
    CHK_RET(ops_hccl::ValidateScatterParams(param, recvBytes, inputBytes));
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }
    // Reuse one communicator channel set across all roots and sizes. Adaptive
    // transfer offsets/counts are fixed in the registered root+B plan, while
    // user pointers and tokens are always supplied by this invocation.
    const int tagLength = std::snprintf(param.tag, sizeof(param.tag), "hccl_scatter_v025_f" SCATTER_STRINGIFY(SCATTER_V016_FEATURE_MASK) "_o" SCATTER_STRINGIFY(SCATTER_V016_OPT_MASK) "_t" SCATTER_STRINGIFY(SCATTER_V017_TUNE_MASK) "_p" SCATTER_STRINGIFY(SCATTER_V018_OPT_MASK) "_q" SCATTER_STRINGIFY(SCATTER_V019_OPT_MASK) "_v" SCATTER_STRINGIFY(SCATTER_V020_OPT_MASK) "_a" SCATTER_STRINGIFY(SCATTER_V021_OPT_MASK) "_z" SCATTER_STRINGIFY(SCATTER_V022_OPT_MASK) "_w" SCATTER_STRINGIFY(SCATTER_V023_OPT_MASK) "_y" SCATTER_STRINGIFY(SCATTER_V024_OPT_MASK) "_k" SCATTER_STRINGIFY(SCATTER_V025_OPT_MASK) "_r%u_b%llu",
        root, static_cast<unsigned long long>(recvBytes));
    if (tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag)) {
        return HCCL_E_INTERNAL;
    }

    // Acquire the thread for THIS invocation's stream, including cache hits.
    CHK_RET(HcclThreadAcquireWithStream(comm, SCATTER_ENGINE, stream, 1, &param.cpuThread));
    CHK_RET(RegisterOpInfo(comm, param, recvBytes, inputBytes));
    CHK_RET(AcquireContext(comm, param, recvBytes));
    return ops_hccl::ExecOp(param);
}
