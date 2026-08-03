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
#include <cstdio>
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

#include "common.h"
#include "custom.h"
#include "ccu_kernel.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t WIDE_CHANNEL_NOTIFY_NUM = 4;
constexpr uint32_t KERNEL_ARG_NUM = 1;
constexpr uint32_t MAX_DIE_GROUPS = 2;
constexpr uint32_t MAX_CHANNELS_PER_DIE = 8;
constexpr uint32_t THREAD_NOTIFY_NUM = 2;
constexpr uint32_t OUTPUT_512_DYNAMIC_TASK_ARGS = 6;
constexpr uint32_t ALGORITHM_V10_PULL = 0;
constexpr uint32_t ALGORITHM_4X1_COMMON_DIE_PUSH = 1;
constexpr uint32_t ALGORITHM_8P4_DUAL_SEED = 2;
constexpr uint32_t ALGORITHM_4X1_DUAL_PLANE_PUSH = 3;
constexpr uint32_t ALGORITHM_2X8_V22_HYBRID = 4;
constexpr uint32_t ALGORITHM_8P4_LATENCY_DIRECT = 5;
constexpr uint32_t ALGORITHM_8P4_WIDE_PIPELINE = 6;
constexpr uint32_t ALGORITHM_OUTPUT_512_STATIC = 7;
constexpr uint32_t V22_CHANNEL_NOTIFY_NUM = 7;
constexpr uint32_t V22_TOPOLOGY_LAYER_NUM = 3;
constexpr uint32_t V22_RANKS_PER_SERVER = 8;
constexpr uint32_t V22_TWO_BY_EIGHT_RANK_SIZE = 16;
constexpr uint32_t V22_GENERIC_INTRA_KERNEL = 0;
constexpr uint32_t V22_GENERIC_CROSS_KERNEL = 1;
constexpr uint32_t V22_PIPELINE_INTRA_ORIGINAL_KERNEL = 2;
constexpr uint32_t V22_INTRA_RELAY_FINAL_KERNEL = 3;
constexpr uint32_t V22_CROSS_HIERARCHICAL_KERNEL = 4;
constexpr uint32_t V22_CROSS_DIRECT_SINGLE_FINAL_KERNEL = 5;
constexpr uint32_t V22_CROSS_DIRECT_DOUBLE_FINAL_KERNEL = 6;
constexpr uint32_t V22_PIPELINE_KERNEL_NUM = 7;
constexpr uint64_t EXACT_512_PAYLOAD_BYTES = 512ULL * 1024ULL;
constexpr uint64_t OUTPUT_512_TARGET_BYTES = 512ULL * 1024ULL;
constexpr const char *ENGINE_CONTEXT_TAG = "hccl_custom_allgather_v23";
constexpr const char *ENGINE_CONTEXT_TAG_4X1_LATENCY =
    "hccl_custom_allgather_v27_4x1_triple10_fast";
constexpr const char *ENGINE_CONTEXT_TAG_4X1_EXACT_512 =
    "hccl_custom_allgather_v35_4x1_registered_warm_push";
constexpr const char *ENGINE_CONTEXT_TAG_4X1_BANDWIDTH =
    "hccl_custom_allgather_v23_4x1_v20_bandwidth";
constexpr const char *ENGINE_CONTEXT_TAG_2X8_LATENCY =
    "hccl_custom_allgather_v27_2x8_triple10_fast";
constexpr const char *ENGINE_CONTEXT_TAG_2X8_BANDWIDTH =
    "hccl_custom_allgather_v23_2x8_v20_bandwidth";
constexpr const char *ENGINE_CONTEXT_TAG_8P4_LATENCY =
    "hccl_custom_allgather_v27_8p4_triple10_fast";
constexpr const char *ENGINE_CONTEXT_TAG_8P4_BANDWIDTH =
    "hccl_custom_allgather_v23_8p4_v20_bandwidth";
constexpr const char *ENGINE_CONTEXT_TAG_OUTPUT_512_RANK4 =
    "hccl_custom_allgather_v36b_output512_rank4_prebiased";
constexpr const char *ENGINE_CONTEXT_TAG_OUTPUT_512_RANK12 =
    "hccl_custom_allgather_v36b_output512_rank12_prebiased";
constexpr const char *ENGINE_CONTEXT_TAG_OUTPUT_512_RANK16 =
    "hccl_custom_allgather_v36b_output512_rank16_prebiased";
constexpr const char *KERNEL_NAME_PREFIX = "CcuAllGatherPeerLanePullDie";
constexpr const char *ROTATING_PUSH_KERNEL_NAME = "CcuV13RotatingDirectPush";
constexpr const char *COMMON_DIE_LATENCY_KERNEL_NAME = "CcuV16Clos4x1LatencyPush";
constexpr const char *COMMON_DIE_BANDWIDTH_KERNEL_NAME = "CcuV16Clos4x1FourChunkPush";
constexpr const char *DUAL_PLANE_MAIN_KERNEL_NAME = "CcuV16DualPlaneMainPush";
constexpr const char *DUAL_PLANE_WORKER_KERNEL_NAME = "CcuV16DualPlaneWorkerPush";
constexpr const char *DUAL_SEED_CROSS_KERNEL_NAME = "CcuV13DualSeedCross";
constexpr const char *DUAL_SEED_RELAY_KERNEL_NAME = "CcuV13DualSeedRelay";
constexpr const char *LATENCY_4X1_KERNEL_NAME = "CcuV23Latency4x1Direct";
constexpr const char *REGISTERED_4X1_COLD_KERNEL_NAME =
    "CcuV35Registered4x1ColdPush";
constexpr const char *REGISTERED_4X1_WARM_KERNEL_NAME =
    "CcuV35Registered4x1WarmPush";
constexpr const char *LATENCY_8P4_INTRA_KERNEL_NAME = "CcuV23Latency8p4Intra";
constexpr const char *LATENCY_8P4_CROSS_KERNEL_NAME = "CcuV23Latency8p4Cross";
constexpr const char *WIDE_INTRA_KERNEL_NAME = "CcuV34WideIntraOriginal";
constexpr const char *WIDE_CROSS_SEED_KERNEL_NAME = "CcuV34WideCrossSeed";
constexpr const char *WIDE_CROSS_DIRECT_KERNEL_NAME = "CcuV34WideCrossDirect";
constexpr const char *WIDE_RELAY_KERNEL_NAME = "CcuV34WideIntraRelay";
constexpr const char *OUTPUT_512_DYNAMIC_KERNEL_NAMES[3][2] = {
    {"CcuV36BOutput512R4Dynamic0", "CcuV36BOutput512R4Dynamic1"},
    {"CcuV36BOutput512R12Dynamic0", "CcuV36BOutput512R12Dynamic1"},
    {"CcuV36BOutput512R16Dynamic0", "CcuV36BOutput512R16Dynamic1"}};

struct ChannelGroup {
    uint32_t dieId = 0;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peerRanks;
};

struct PendingChannel {
    HcclChannelDesc desc{};
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t dieId = 0;
    uint32_t layerOrder = 0;
};

struct FourRankLinkCandidate {
    HcclChannelDesc desc{};
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t netLayer = 0;
    uint32_t layerOrder = 0;
    uint32_t localDieId = 0;
    uint32_t bandwidthCoeff = 0;
    uint8_t hop = 0;
    bool isClos = false;
};

struct FourRankPlaneKey {
    uint32_t netLayer = 0;
    uint32_t localDieId = 0;
    bool isClos = false;
};

struct V22PeerChannel {
    ChannelHandle handle{};
    EndpointAttrDieId dieId{};
    uint32_t remoteRank = INVALID_VALUE_RANKID;
};

struct V22ChannelGroup {
    EndpointAttrDieId dieId{};
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peerRanks;
    bool handleLocalCopy = false;
    bool isIntraServer = false;
};

struct FourByOneExact512FastCache {
    HcclComm comm{};
    aclrtStream stream{};
    ThreadHandle mainThread{};
    void *context = nullptr;
    uint64_t contextSize = 0U;
    uint32_t rankId = INVALID_VALUE_RANKID;
    void *inputPtr = nullptr;
    void *outputPtr = nullptr;
};

thread_local FourByOneExact512FastCache g_4x1Exact512FastCache{};

struct Output512FastCache {
    HcclComm comm{};
    aclrtStream stream{};
    void *inputPtr = nullptr;
    void *outputPtr = nullptr;
    uint64_t sendCount = 0U;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0U;
    uint32_t groupCount = 0U;
    void *context = nullptr;
    ThreadHandle threads[2]{};
    CcuKernelHandle warmKernels[2]{};
    uint64_t inputAddr = 0U;
    uint64_t outputAddr = 0U;
    uint64_t inputToken = 0U;
    uint64_t outputToken = 0U;
    uint64_t rankOutputOffset = 0U;
    uint64_t transferBytes = 0U;
    bool valid = false;
};

thread_local Output512FastCache g_output512FastCache{};

bool IsOutput512Target(uint64_t rankBytes, uint32_t rankSize)
{
    if (rankBytes == 0U
        || (rankSize != 4U && rankSize != 12U && rankSize != 16U)
        || rankBytes > std::numeric_limits<uint64_t>::max() / rankSize) {
        return false;
    }
    const uint64_t outputBytes = rankBytes * rankSize;
    const uint64_t delta = outputBytes >= OUTPUT_512_TARGET_BYTES
        ? outputBytes - OUTPUT_512_TARGET_BYTES
        : OUTPUT_512_TARGET_BYTES - outputBytes;
    return delta < static_cast<uint64_t>(rankSize) * sizeof(float);
}

HcclResult LaunchOutput512WarmFast(const Output512FastCache &cache)
{
    CHK_PRT_RET(!cache.valid || cache.groupCount == 0U
            || cache.groupCount > 2U,
        HCCL_ERROR("[LaunchOutput512WarmFast] Invalid cache"),
        HCCL_E_INTERNAL);
    const uint64_t taskArgs[OUTPUT_512_DYNAMIC_TASK_ARGS] = {
        cache.inputAddr,
        cache.outputAddr,
        cache.inputToken,
        cache.outputToken,
        cache.rankOutputOffset,
        cache.transferBytes,
    };
    if (cache.groupCount == 1U) {
        const CcuResult ret = HcommCcuKernelLaunch(
            cache.threads[0], cache.warmKernels[0], taskArgs,
            OUTPUT_512_DYNAMIC_TASK_ARGS);
        return ret == CCU_SUCCESS ? HCCL_SUCCESS : ConvertCcuToHccl(ret);
    }

    constexpr uint32_t workerNotifyId = 0U;
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        cache.threads[0], cache.threads[1], workerNotifyId)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        cache.threads[1], workerNotifyId, CUSTOM_TIMEOUT)));

    // Preserve the registration order (main/intra, then worker/cross).  Some
    // CCU runtimes pair missions in this order even though the two IO Dies run
    // concurrently.
    CcuResult ret = HcommCcuKernelLaunch(
        cache.threads[0], cache.warmKernels[0], taskArgs,
        OUTPUT_512_DYNAMIC_TASK_ARGS);
    if (ret != CCU_SUCCESS) {
        return ConvertCcuToHccl(ret);
    }
    ret = HcommCcuKernelLaunch(
        cache.threads[1], cache.warmKernels[1], taskArgs,
        OUTPUT_512_DYNAMIC_TASK_ARGS);
    if (ret != CCU_SUCCESS) {
        return ConvertCcuToHccl(ret);
    }
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        cache.threads[0], workerNotifyId, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        cache.threads[1], cache.threads[0], workerNotifyId)));
    return HCCL_SUCCESS;
}

HcclResult V22FindPeerLink(
    HcclComm comm, uint32_t myRank, uint32_t remoteRank, CommLink &selectedLink)
{
    for (uint32_t netLayer = 0; netLayer < V22_TOPOLOGY_LAYER_NUM; ++netLayer) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        const HcclResult queryResult = HcclRankGraphGetLinks(
            comm, netLayer, myRank, remoteRank, &linkList, &listSize);
        if (queryResult != HCCL_SUCCESS || linkList == nullptr || listSize == 0U) {
            continue;
        }

        uint32_t selectedIndex = 0;
        for (uint32_t linkIndex = 0; linkIndex < listSize; ++linkIndex) {
            if (linkList[linkIndex].linkAttr.linkProtocol
                == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                selectedIndex = linkIndex;
                break;
            }
        }
        selectedLink = linkList[selectedIndex];
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[V22FindPeerLink] No link between rank %u and rank %u",
        myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult V22AcquirePeerChannels(HcclComm comm, CommEngine engine,
    uint32_t myRank, std::vector<V22PeerChannel> &peerChannels)
{
    peerChannels.clear();
    peerChannels.reserve(V22_TWO_BY_EIGHT_RANK_SIZE - 1U);
    for (uint32_t remoteRank = 0;
         remoteRank < V22_TWO_BY_EIGHT_RANK_SIZE;
         ++remoteRank) {
        if (remoteRank == myRank) {
            continue;
        }

        CommLink selectedLink{};
        CHK_RET(V22FindPeerLink(comm, myRank, remoteRank, selectedLink));

        HcclChannelDesc desc{};
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = remoteRank;
        desc.notifyNum = V22_CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = selectedLink.linkAttr.linkProtocol;
        desc.localEndpoint = selectedLink.srcEndpointDesc;
        desc.remoteEndpoint = selectedLink.dstEndpointDesc;

        EndpointAttrDieId dieId{};
        CHK_RET(HcclRankGraphGetEndpointInfo(
            comm, myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID,
            static_cast<uint32_t>(sizeof(dieId)), &dieId));

        ChannelHandle channel{};
        CHK_RET(HcclChannelAcquire(comm, engine, &desc, 1, &channel));
        peerChannels.push_back(V22PeerChannel{channel, dieId, remoteRank});
    }
    return HCCL_SUCCESS;
}

uint32_t V22GetCyclicIssueKey(uint32_t myRank, uint32_t remoteRank)
{
    const uint32_t myLocalRank = myRank % V22_RANKS_PER_SERVER;
    const uint32_t remoteLocalRank = remoteRank % V22_RANKS_PER_SERVER;
    return (remoteLocalRank + V22_RANKS_PER_SERVER - myLocalRank)
        % V22_RANKS_PER_SERVER;
}

HcclResult V22GroupChannelsByDie(
    const std::vector<V22PeerChannel> &peerChannels,
    uint32_t myRank,
    std::vector<V22ChannelGroup> &groups)
{
    CHK_PRT_RET(peerChannels.size() != V22_TWO_BY_EIGHT_RANK_SIZE - 1U,
        HCCL_ERROR("[V22GroupChannelsByDie] Expected 15 peer channels, got %zu",
            peerChannels.size()),
        HCCL_E_INTERNAL);

    std::map<EndpointAttrDieId, std::vector<V22PeerChannel>> channelsByDie;
    for (const V22PeerChannel &peerChannel : peerChannels) {
        channelsByDie[peerChannel.dieId].push_back(peerChannel);
    }
    CHK_PRT_RET(channelsByDie.size() != 2U,
        HCCL_ERROR("[V22GroupChannelsByDie] Expected two CCU Dies, got %zu",
            channelsByDie.size()),
        HCCL_E_INTERNAL);

    groups.clear();
    groups.reserve(channelsByDie.size());
    const uint32_t myServer = myRank / V22_RANKS_PER_SERVER;
    for (auto &entry : channelsByDie) {
        std::stable_sort(entry.second.begin(), entry.second.end(),
            [myRank](const V22PeerChannel &left, const V22PeerChannel &right) {
                const uint32_t leftKey =
                    V22GetCyclicIssueKey(myRank, left.remoteRank);
                const uint32_t rightKey =
                    V22GetCyclicIssueKey(myRank, right.remoteRank);
                if (leftKey != rightKey) {
                    return leftKey < rightKey;
                }
                return left.remoteRank < right.remoteRank;
            });

        bool isIntraServer = true;
        for (const V22PeerChannel &peer : entry.second) {
            if (peer.remoteRank / V22_RANKS_PER_SERVER != myServer) {
                isIntraServer = false;
                break;
            }
        }

        V22ChannelGroup group{};
        group.dieId = entry.first;
        group.isIntraServer = isIntraServer;
        group.channels.reserve(entry.second.size());
        group.peerRanks.reserve(entry.second.size());
        for (const V22PeerChannel &peer : entry.second) {
            group.channels.push_back(peer.handle);
            group.peerRanks.push_back(peer.remoteRank);
        }
        groups.push_back(std::move(group));
    }

    std::stable_sort(groups.begin(), groups.end(),
        [](const V22ChannelGroup &left, const V22ChannelGroup &right) {
            return left.isIntraServer && !right.isIntraServer;
        });

    CHK_PRT_RET(groups.size() != 2U
            || !groups[0].isIntraServer
            || groups[1].isIntraServer
            || groups[0].channels.size() != V22_RANKS_PER_SERVER - 1U
            || groups[1].channels.size() != V22_RANKS_PER_SERVER
            || groups[0].dieId == groups[1].dieId,
        HCCL_ERROR("[V22GroupChannelsByDie] Topology is not a strict 2x8 split"),
        HCCL_E_NOT_SUPPORT);

    for (uint32_t peerRank : groups[0].peerRanks) {
        CHK_PRT_RET(peerRank / V22_RANKS_PER_SERVER != myServer,
            HCCL_ERROR("[V22GroupChannelsByDie] Intra group contains rank %u",
                peerRank),
            HCCL_E_NOT_SUPPORT);
    }
    for (uint32_t peerRank : groups[1].peerRanks) {
        CHK_PRT_RET(peerRank / V22_RANKS_PER_SERVER == myServer,
            HCCL_ERROR("[V22GroupChannelsByDie] Cross group contains rank %u",
                peerRank),
            HCCL_E_NOT_SUPPORT);
    }

    groups[0].handleLocalCopy = true;
    groups[1].handleLocalCopy = false;
    return HCCL_SUCCESS;
}

HcclResult FillChannelDescFromLink(
    uint32_t remoteRank, const CommLink &link, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    // RankGraph返回的CommLink已经给出了与该物理链路严格匹配的两端
    // EndpointDesc。完整复制描述符（包括其扩展区），不要再根据远端rank
    // 二次查询/重建Endpoint；远端rank的EndpointInfo在当前仿真接口中并不
    // 保证可查询，且部分字段丢失会使Channel被错误识别成跨Die loop channel。
    desc.localEndpoint = link.srcEndpointDesc;
    desc.remoteEndpoint = link.dstEndpointDesc;
    return HCCL_SUCCESS;
}

HcclResult BuildChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    const uint32_t *netLayers, uint32_t netLayerNum, uint32_t notifyNum, HcclChannelDesc &desc, uint32_t &dieId,
    uint32_t &layerOrder)
{
    for (uint32_t layerIndex = 0; layerIndex < netLayerNum; ++layerIndex) {
        CommLink *linkList = nullptr;
        uint32_t linkNum = 0;
        const HcclResult linkRet = HcclRankGraphGetLinks(
            comm, netLayers[layerIndex], myRank, remoteRank, &linkList, &linkNum);
        if (linkRet == HCCL_E_NOT_FOUND) {
            continue;
        }
        if (linkRet != HCCL_SUCCESS) {
            HCCL_ERROR("[BuildChannelDesc] Failed to query layer %u, ret=%d", netLayers[layerIndex], linkRet);
            return linkRet;
        }
        if (linkNum == 0) {
            continue;
        }
        CHK_PRT_RET(linkList == nullptr,
            HCCL_ERROR("[BuildChannelDesc] Layer %u returned a null link list", netLayers[layerIndex]),
            HCCL_E_INTERNAL);

        for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
            const CommLink &link = linkList[linkIndex];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            // 通用路径保持V13已经通过1～9功能点的描述符构造方式不变；
            // 完整CommLink Endpoint复制只用于下方4x1公共Die实验路径。
            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = remoteRank;
            desc.notifyNum = notifyNum;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;

            EndpointAttrDieId endpointDieId = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank, &link.srcEndpointDesc,
                ENDPOINT_ATTR_DIE_ID, static_cast<uint32_t>(sizeof(endpointDieId)), &endpointDieId));
            dieId = endpointDieId;
            layerOrder = layerIndex;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[BuildChannelDesc] UBC_CTP link not found between rank %u and rank %u", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

bool IsBetter4x1Candidate(
    const FourRankLinkCandidate &candidate, const FourRankLinkCandidate &current)
{
    if (candidate.bandwidthCoeff != current.bandwidthCoeff) {
        return candidate.bandwidthCoeff > current.bandwidthCoeff;
    }
    if (candidate.hop != current.hop) {
        return candidate.hop < current.hop;
    }
    return candidate.layerOrder > current.layerOrder;
}

const FourRankLinkCandidate *Find4x1Candidate(
    const std::vector<FourRankLinkCandidate> &candidates, uint32_t netLayer, uint32_t dieId)
{
    const FourRankLinkCandidate *best = nullptr;
    for (const FourRankLinkCandidate &candidate : candidates) {
        // 单Kernel约束只作用于本端网络设备。合法链路可能是
        // localDie0 -> remoteDie1，不能用远端设备的Die编号排除它。
        if (candidate.netLayer != netLayer || candidate.localDieId != dieId) {
            continue;
        }
        if (best == nullptr || IsBetter4x1Candidate(candidate, *best)) {
            best = &candidate;
        }
    }
    return best;
}

HcclResult Collect4x1Candidates(HcclComm comm, uint32_t myRank,
    const std::vector<uint32_t> &netLayers,
    std::vector<std::vector<FourRankLinkCandidate>> &candidatesByPeer)
{
    candidatesByPeer.clear();
    candidatesByPeer.resize(4U);
    for (uint32_t layerOrder = 0; layerOrder < netLayers.size(); ++layerOrder) {
        const uint32_t netLayer = netLayers[layerOrder];
        CommTopo topoType = COMM_TOPO_RESERVED;
        const HcclResult topoRet =
            HcclRankGraphGetTopoTypeByLayer(comm, netLayer, &topoType);
        const bool isClos = topoRet == HCCL_SUCCESS && topoType == COMM_TOPO_CLOS;

        for (uint32_t remoteRank = 0; remoteRank < 4U; ++remoteRank) {
            if (remoteRank == myRank) {
                continue;
            }
            CommLink *links = nullptr;
            uint32_t linkNum = 0;
            const HcclResult linkRet =
                HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &links, &linkNum);
            if (linkRet == HCCL_E_NOT_FOUND) {
                continue;
            }
            if (linkRet != HCCL_SUCCESS) {
                HCCL_ERROR("[Collect4x1Candidates] Layer %u peer %u query failed, ret=%d",
                    netLayer, remoteRank, linkRet);
                return linkRet;
            }
            if (linkNum == 0U) {
                continue;
            }
            CHK_PRT_RET(links == nullptr,
                HCCL_ERROR("[Collect4x1Candidates] Layer %u returned null links", netLayer),
                HCCL_E_INTERNAL);

            for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
                // RankGraph返回的Link列表由库管理；本端属性查询前复制当前项。
                // CommLink内的dstEndpointDesc已经是建立Channel所需的完整远端
                // 描述，禁止再以remoteRank调用GetEndpointInfo。
                const CommLink link = links[linkIndex];
                if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }
                EndpointAttrDieId localDieId = 0;
                const HcclResult localDieRet = HcclRankGraphGetEndpointInfo(comm, myRank,
                    &link.srcEndpointDesc, ENDPOINT_ATTR_DIE_ID,
                    static_cast<uint32_t>(sizeof(localDieId)), &localDieId);
                if (localDieRet != HCCL_SUCCESS) {
                    continue;
                }

                EndpointAttrBwCoeff localBw = 0;
                (void)HcclRankGraphGetEndpointInfo(comm, myRank, &link.srcEndpointDesc,
                    ENDPOINT_ATTR_BW_COEFF, static_cast<uint32_t>(sizeof(localBw)), &localBw);

                FourRankLinkCandidate candidate{};
                CHK_RET(FillChannelDescFromLink(remoteRank, link, candidate.desc));
                candidate.remoteRank = remoteRank;
                candidate.netLayer = netLayer;
                candidate.layerOrder = layerOrder;
                candidate.localDieId = localDieId;
                candidate.bandwidthCoeff = localBw;
                candidate.hop = link.linkAttr.hop;
                candidate.isClos = isClos;
                candidatesByPeer[remoteRank].push_back(candidate);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TryAcquire4x1CommonDieChannels(HcclComm comm, CommEngine engine,
    uint32_t myRank, std::vector<ChannelGroup> &groups, bool &selected)
{
    selected = false;
    uint32_t *netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayerList, &netLayerNum));
    CHK_PRT_RET(netLayerList == nullptr || netLayerNum == 0U,
        HCCL_ERROR("[TryAcquire4x1CommonDieChannels] No topology layer"), HCCL_E_NOT_FOUND);

    // GetLayers返回的内存由库管理；后续拓扑查询前先复制，避免重复查询使指针失效。
    const std::vector<uint32_t> netLayers(netLayerList, netLayerList + netLayerNum);
    std::vector<std::vector<FourRankLinkCandidate>> candidatesByPeer;
    CHK_RET(Collect4x1Candidates(comm, myRank, netLayers, candidatesByPeer));

    const FourRankLinkCandidate *bestSeed = nullptr;
    uint64_t bestBandwidthSum = 0U;
    bool bestIsClos = false;
    uint32_t bestLayerOrder = 0U;
    for (uint32_t seedRank = 0; seedRank < 4U; ++seedRank) {
        if (seedRank == myRank) {
            continue;
        }
        for (const FourRankLinkCandidate &seed : candidatesByPeer[seedRank]) {
            uint64_t bandwidthSum = 0U;
            bool coversAllPeers = true;
            for (uint32_t remoteRank = 0; remoteRank < 4U; ++remoteRank) {
                if (remoteRank == myRank) {
                    continue;
                }
                const FourRankLinkCandidate *candidate =
                    Find4x1Candidate(candidatesByPeer[remoteRank], seed.netLayer, seed.localDieId);
                if (candidate == nullptr) {
                    coversAllPeers = false;
                    break;
                }
                bandwidthSum += candidate->bandwidthCoeff;
            }
            if (!coversAllPeers) {
                continue;
            }
            const bool better = bestSeed == nullptr
                || (seed.isClos && !bestIsClos)
                || (seed.isClos == bestIsClos && bandwidthSum > bestBandwidthSum)
                || (seed.isClos == bestIsClos && bandwidthSum == bestBandwidthSum
                    && seed.layerOrder > bestLayerOrder);
            if (better) {
                bestSeed = &seed;
                bestBandwidthSum = bandwidthSum;
                bestIsClos = seed.isClos;
                bestLayerOrder = seed.layerOrder;
            }
        }
    }
    if (bestSeed == nullptr) {
        HCCL_INFO("[TryAcquire4x1CommonDieChannels] No common layer/Die, use generic path");
        return HCCL_SUCCESS;
    }

    std::vector<FourRankLinkCandidate> selectedCandidates;
    selectedCandidates.reserve(3U);
    for (uint32_t remoteRank = 0; remoteRank < 4U; ++remoteRank) {
        if (remoteRank == myRank) {
            continue;
        }
        const FourRankLinkCandidate *candidate =
            Find4x1Candidate(candidatesByPeer[remoteRank], bestSeed->netLayer, bestSeed->localDieId);
        CHK_PRT_RET(candidate == nullptr,
            HCCL_ERROR("[TryAcquire4x1CommonDieChannels] Candidate intersection changed"),
            HCCL_E_INTERNAL);
        selectedCandidates.push_back(*candidate);
    }
    std::sort(selectedCandidates.begin(), selectedCandidates.end(),
        [myRank](const FourRankLinkCandidate &left, const FourRankLinkCandidate &right) {
            const uint32_t leftDistance = (left.remoteRank + 4U - myRank) % 4U;
            const uint32_t rightDistance = (right.remoteRank + 4U - myRank) % 4U;
            return leftDistance < rightDistance;
        });

    std::vector<HcclChannelDesc> descs;
    descs.reserve(selectedCandidates.size());
    for (const FourRankLinkCandidate &candidate : selectedCandidates) {
        descs.push_back(candidate.desc);
    }
    std::vector<ChannelHandle> channels(descs.size());
    CHK_RET(HcclChannelAcquire(
        comm, engine, descs.data(), static_cast<uint32_t>(descs.size()), channels.data()));

    ChannelGroup group{};
    group.dieId = bestSeed->localDieId;
    group.channels = std::move(channels);
    for (const FourRankLinkCandidate &candidate : selectedCandidates) {
        group.peerRanks.push_back(candidate.remoteRank);
    }
    groups = {std::move(group)};
    selected = true;
    HCCL_INFO("[TryAcquire4x1CommonDieChannels] Selected layer=%u die=%u clos=%u",
        bestSeed->netLayer, bestSeed->localDieId, static_cast<uint32_t>(bestIsClos));
    return HCCL_SUCCESS;
}

bool Is4x1MatchingEdge(uint32_t leftRank, uint32_t rightRank)
{
    const uint32_t lowRank = std::min(leftRank, rightRank);
    const uint32_t highRank = std::max(leftRank, rightRank);
    // K4的固定完美匹配：(0,1)与(2,3)。该判断只依赖无向rank pair，
    // 因而一条边的两端一定把它放到相同逻辑平面。
    return (lowRank == 0U && highRank == 1U)
        || (lowRank == 2U && highRank == 3U);
}

bool Same4x1Plane(const FourRankPlaneKey &left, const FourRankPlaneKey &right)
{
    return left.netLayer == right.netLayer && left.localDieId == right.localDieId;
}

const FourRankLinkCandidate *Find4x1PlaneCandidate(
    const std::vector<FourRankLinkCandidate> &candidates, const FourRankPlaneKey &key)
{
    return Find4x1Candidate(candidates, key.netLayer, key.localDieId);
}

std::vector<FourRankPlaneKey> Build4x1CoveringPlanes(
    const std::vector<std::vector<FourRankLinkCandidate>> &candidatesByPeer,
    const std::vector<uint32_t> &peerRanks)
{
    std::vector<FourRankPlaneKey> planes;
    if (peerRanks.empty()) {
        return planes;
    }
    for (const FourRankLinkCandidate &seed : candidatesByPeer[peerRanks[0]]) {
        FourRankPlaneKey key{seed.netLayer, seed.localDieId, seed.isClos};
        bool duplicate = false;
        for (const FourRankPlaneKey &existing : planes) {
            if (Same4x1Plane(existing, key)) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        bool coversAllPeers = true;
        for (uint32_t peerRank : peerRanks) {
            if (Find4x1PlaneCandidate(candidatesByPeer[peerRank], key) == nullptr) {
                coversAllPeers = false;
                break;
            }
        }
        if (coversAllPeers) {
            planes.push_back(key);
        }
    }
    std::sort(planes.begin(), planes.end(),
        [](const FourRankPlaneKey &left, const FourRankPlaneKey &right) {
            if (left.isClos != right.isClos) {
                return left.isClos > right.isClos;
            }
            if (left.localDieId != right.localDieId) {
                return left.localDieId < right.localDieId;
            }
            return left.netLayer < right.netLayer;
        });
    return planes;
}

bool IsBetter4x1PlanePair(const FourRankPlaneKey &matchingPlane,
    const FourRankPlaneKey &crossPlane, const FourRankPlaneKey *bestMatchingPlane,
    const FourRankPlaneKey *bestCrossPlane)
{
    if (bestMatchingPlane == nullptr || bestCrossPlane == nullptr) {
        return true;
    }
    const uint32_t closScore =
        static_cast<uint32_t>(matchingPlane.isClos) + static_cast<uint32_t>(crossPlane.isClos);
    const uint32_t bestClosScore =
        static_cast<uint32_t>(bestMatchingPlane->isClos)
        + static_cast<uint32_t>(bestCrossPlane->isClos);
    if (closScore != bestClosScore) {
        return closScore > bestClosScore;
    }
    // 优先让固定matching走较小Die、其余两条边走较大Die。该次序完全由
    // rank pair、layer id和Die id确定，不使用本端测得的带宽，避免四个
    // Rank独立贪心后对同一条无向边作出不一致选择。
    const bool ordered = matchingPlane.localDieId < crossPlane.localDieId;
    const bool bestOrdered = bestMatchingPlane->localDieId < bestCrossPlane->localDieId;
    if (ordered != bestOrdered) {
        return ordered;
    }
    if (matchingPlane.localDieId != bestMatchingPlane->localDieId) {
        return matchingPlane.localDieId < bestMatchingPlane->localDieId;
    }
    if (crossPlane.localDieId != bestCrossPlane->localDieId) {
        return crossPlane.localDieId < bestCrossPlane->localDieId;
    }
    if (matchingPlane.netLayer != bestMatchingPlane->netLayer) {
        return matchingPlane.netLayer < bestMatchingPlane->netLayer;
    }
    return crossPlane.netLayer < bestCrossPlane->netLayer;
}

HcclResult TryAcquire4x1DualPlaneChannels(HcclComm comm, CommEngine engine,
    uint32_t myRank, std::vector<ChannelGroup> &groups, bool &selected)
{
    selected = false;
    uint32_t *netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayerList, &netLayerNum));
    CHK_PRT_RET(netLayerList == nullptr || netLayerNum == 0U,
        HCCL_ERROR("[TryAcquire4x1DualPlaneChannels] No topology layer"),
        HCCL_E_NOT_FOUND);

    const std::vector<uint32_t> netLayers(netLayerList, netLayerList + netLayerNum);
    std::vector<std::vector<FourRankLinkCandidate>> candidatesByPeer;
    CHK_RET(Collect4x1Candidates(comm, myRank, netLayers, candidatesByPeer));

    std::vector<uint32_t> matchingPeers;
    std::vector<uint32_t> crossPeers;
    for (uint32_t remoteRank = 0U; remoteRank < 4U; ++remoteRank) {
        if (remoteRank == myRank) {
            continue;
        }
        if (Is4x1MatchingEdge(myRank, remoteRank)) {
            matchingPeers.push_back(remoteRank);
        } else {
            crossPeers.push_back(remoteRank);
        }
    }
    CHK_PRT_RET(matchingPeers.size() != 1U || crossPeers.size() != 2U,
        HCCL_ERROR("[TryAcquire4x1DualPlaneChannels] Invalid K4 edge partition"),
        HCCL_E_INTERNAL);

    const std::vector<FourRankPlaneKey> matchingPlanes =
        Build4x1CoveringPlanes(candidatesByPeer, matchingPeers);
    const std::vector<FourRankPlaneKey> crossPlanes =
        Build4x1CoveringPlanes(candidatesByPeer, crossPeers);
    const FourRankPlaneKey *bestMatchingPlane = nullptr;
    const FourRankPlaneKey *bestCrossPlane = nullptr;
    for (const FourRankPlaneKey &matchingPlane : matchingPlanes) {
        for (const FourRankPlaneKey &crossPlane : crossPlanes) {
            if (matchingPlane.localDieId == crossPlane.localDieId) {
                continue;
            }
            if (IsBetter4x1PlanePair(matchingPlane, crossPlane,
                bestMatchingPlane, bestCrossPlane)) {
                bestMatchingPlane = &matchingPlane;
                bestCrossPlane = &crossPlane;
            }
        }
    }
    if (bestMatchingPlane == nullptr || bestCrossPlane == nullptr) {
        HCCL_INFO("[TryAcquire4x1DualPlaneChannels] No two-Die edge partition; use common-Die path");
        return HCCL_SUCCESS;
    }

    // 两Peer组先放入主线程，单Peer matching组放入worker线程。
    const std::vector<std::vector<uint32_t>> peerGroups{crossPeers, matchingPeers};
    const FourRankPlaneKey planeKeys[2] = {*bestCrossPlane, *bestMatchingPlane};
    std::vector<HcclChannelDesc> descs;
    std::vector<FourRankLinkCandidate> selectedCandidates;
    for (uint32_t groupIndex = 0U; groupIndex < peerGroups.size(); ++groupIndex) {
        std::vector<FourRankLinkCandidate> groupCandidates;
        for (uint32_t peerRank : peerGroups[groupIndex]) {
            const FourRankLinkCandidate *candidate =
                Find4x1PlaneCandidate(candidatesByPeer[peerRank], planeKeys[groupIndex]);
            CHK_PRT_RET(candidate == nullptr,
                HCCL_ERROR("[TryAcquire4x1DualPlaneChannels] Plane intersection changed"),
                HCCL_E_INTERNAL);
            groupCandidates.push_back(*candidate);
        }
        std::sort(groupCandidates.begin(), groupCandidates.end(),
            [myRank](const FourRankLinkCandidate &left, const FourRankLinkCandidate &right) {
                const uint32_t leftDistance = (left.remoteRank + 4U - myRank) % 4U;
                const uint32_t rightDistance = (right.remoteRank + 4U - myRank) % 4U;
                return leftDistance < rightDistance;
            });
        for (const FourRankLinkCandidate &candidate : groupCandidates) {
            descs.push_back(candidate.desc);
            selectedCandidates.push_back(candidate);
        }
    }

    std::vector<ChannelHandle> channels(descs.size());
    CHK_RET(HcclChannelAcquire(
        comm, engine, descs.data(), static_cast<uint32_t>(descs.size()), channels.data()));

    groups.clear();
    uint32_t channelOffset = 0U;
    for (uint32_t groupIndex = 0U; groupIndex < peerGroups.size(); ++groupIndex) {
        ChannelGroup group{};
        group.dieId = planeKeys[groupIndex].localDieId;
        for (uint32_t peerIndex = 0U; peerIndex < peerGroups[groupIndex].size(); ++peerIndex) {
            group.channels.push_back(channels[channelOffset]);
            group.peerRanks.push_back(selectedCandidates[channelOffset].remoteRank);
            ++channelOffset;
        }
        groups.push_back(std::move(group));
    }
    selected = true;
    HCCL_INFO("[TryAcquire4x1DualPlaneChannels] Cross plane layer=%u die=%u peers=2; "
        "matching plane layer=%u die=%u peers=1",
        bestCrossPlane->netLayer, bestCrossPlane->localDieId,
        bestMatchingPlane->netLayer, bestMatchingPlane->localDieId);
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(
    HcclComm comm, CommEngine engine, uint32_t myRank, uint32_t rankSize, uint32_t notifyNum, std::vector<ChannelGroup> &groups)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("[AcquireChannels] No topology layer is available"), HCCL_E_NOT_FOUND);

    std::vector<PendingChannel> pendingChannels;
    pendingChannels.reserve(rankSize - 1);
    for (uint32_t remoteRank = 0; remoteRank < rankSize; ++remoteRank) {
        if (remoteRank == myRank) {
            continue;
        }
        PendingChannel pending{};
        pending.remoteRank = remoteRank;
        CHK_RET(BuildChannelDesc(comm, myRank, remoteRank, netLayers, netLayerNum, notifyNum, pending.desc, pending.dieId,
            pending.layerOrder));
        pendingChannels.push_back(pending);
    }

    // 同一Die内采用rank旋转顺序，使所有rank的第k个Write构成置换，减少同步同序提交造成的瞬时incast。
    std::sort(pendingChannels.begin(), pendingChannels.end(),
        [myRank, rankSize](const PendingChannel &left, const PendingChannel &right) {
            if (left.dieId != right.dieId) {
                return left.dieId < right.dieId;
            }
            if (left.layerOrder != right.layerOrder) {
                return left.layerOrder > right.layerOrder;
            }
            const uint32_t leftDistance = (left.remoteRank + rankSize - myRank) % rankSize;
            const uint32_t rightDistance = (right.remoteRank + rankSize - myRank) % rankSize;
            return leftDistance < rightDistance;
        });

    std::vector<HcclChannelDesc> channelDescs;
    channelDescs.reserve(pendingChannels.size());
    for (const PendingChannel &pending : pendingChannels) {
        channelDescs.push_back(pending.desc);
    }
    std::vector<ChannelHandle> channels(channelDescs.size());
    if (!channelDescs.empty()) {
        CHK_RET(HcclChannelAcquire(
            comm, engine, channelDescs.data(), static_cast<uint32_t>(channelDescs.size()), channels.data()));
    }

    groups.clear();
    for (uint32_t channelIndex = 0; channelIndex < channels.size(); ++channelIndex) {
        const PendingChannel &pending = pendingChannels[channelIndex];
        const uint32_t dieId = pending.dieId;
        const auto groupIt = std::find_if(groups.begin(), groups.end(), [dieId](const ChannelGroup &group) {
            return group.dieId == dieId;
        });
        if (groupIt == groups.end()) {
            ChannelGroup group{};
            group.dieId = dieId;
            group.channels.push_back(channels[channelIndex]);
            group.peerRanks.push_back(pending.remoteRank);
            groups.push_back(std::move(group));
        } else {
            groupIt->channels.push_back(channels[channelIndex]);
            groupIt->peerRanks.push_back(pending.remoteRank);
        }
    }
    // 主线程承担更重的Die组，减少关键组的一次Thread启动等待。
    std::sort(groups.begin(), groups.end(), [](const ChannelGroup &left, const ChannelGroup &right) {
        if (left.channels.size() != right.channels.size()) {
            return left.channels.size() > right.channels.size();
        }
        return left.dieId < right.dieId;
    });
    CHK_PRT_RET(rankSize > 1 && (groups.empty() || groups.size() > MAX_DIE_GROUPS),
        HCCL_ERROR("[AcquireChannels] Expected one or two local Die groups, got %zu", groups.size()),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

ops_hccl::TopologyType DetectTopology(const OpParam &param, const std::vector<ChannelGroup> &groups)
{
    size_t totalPeerCount = 0;
    for (const ChannelGroup &group : groups) {
        totalPeerCount += group.channels.size();
    }
    if (param.rankSize == 4 && totalPeerCount == 3) {
        return ops_hccl::TopologyType::MESH_4X1;
    }
    if (param.rankSize == 16 && totalPeerCount == 15 && groups.size() <= MAX_DIE_GROUPS) {
        return ops_hccl::TopologyType::DUAL_SERVER_2X8;
    }
    if (param.rankSize == 12 && totalPeerCount == 11 && groups.size() <= MAX_DIE_GROUPS) {
        return ops_hccl::TopologyType::ASYMMETRIC_8P4;
    }
    return ops_hccl::TopologyType::GENERIC;
}

bool PeerSetEquals(const ChannelGroup &group, const std::vector<uint32_t> &expectedPeers)
{
    if (group.channels.size() != group.peerRanks.size()
        || group.peerRanks.size() != expectedPeers.size()) {
        return false;
    }
    std::vector<uint32_t> actual = group.peerRanks;
    std::vector<uint32_t> expected = expectedPeers;
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    return actual == expected;
}

std::vector<uint32_t> BuildRankRange(uint32_t begin, uint32_t end, uint32_t excludedRank)
{
    std::vector<uint32_t> result;
    for (uint32_t rank = begin; rank < end; ++rank) {
        if (rank != excludedRank) {
            result.push_back(rank);
        }
    }
    return result;
}

bool Resolve4x1Group(const OpParam &param, const std::vector<ChannelGroup> &groups, uint32_t &groupIndex)
{
    if (param.rankSize != 4U || groups.size() != 1U) {
        return false;
    }
    const std::vector<uint32_t> expected = BuildRankRange(0U, 4U, param.myRank);
    if (!PeerSetEquals(groups[0], expected)) {
        return false;
    }
    groupIndex = 0U;
    return true;
}

bool Resolve8p4Groups(const OpParam &param, const std::vector<ChannelGroup> &groups,
    uint32_t &intraGroupIndex, uint32_t &crossGroupIndex)
{
    if (param.rankSize != 12U || groups.size() != 2U) {
        return false;
    }

    const bool onLargeServer = param.myRank < 8U;
    const std::vector<uint32_t> expectedIntra = onLargeServer
        ? BuildRankRange(0U, 8U, param.myRank)
        : BuildRankRange(8U, 12U, param.myRank);
    const std::vector<uint32_t> expectedCross = onLargeServer
        ? BuildRankRange(8U, 12U, INVALID_VALUE_RANKID)
        : BuildRankRange(0U, 8U, INVALID_VALUE_RANKID);

    bool foundIntra = false;
    bool foundCross = false;
    for (uint32_t index = 0; index < groups.size(); ++index) {
        if (PeerSetEquals(groups[index], expectedIntra)) {
            intraGroupIndex = index;
            foundIntra = true;
        }
        if (PeerSetEquals(groups[index], expectedCross)) {
            crossGroupIndex = index;
            foundCross = true;
        }
    }
    return foundIntra && foundCross && intraGroupIndex != crossGroupIndex
        && groups[intraGroupIndex].dieId != groups[crossGroupIndex].dieId;
}

template <typename T>
void FillStaticPeerMap(T &kernelArg, const OpParam &param, const ChannelGroup &group)
{
    kernelArg.rankSize = param.rankSize;
    kernelArg.rankId = param.myRank;
    kernelArg.channelCount = static_cast<uint32_t>(group.channels.size());
    for (uint32_t i = 0; i < kernelArg.channelCount; ++i) {
        kernelArg.channels[i] = group.channels[i];
        kernelArg.peerRanks[i] = group.peerRanks[i];
    }
}

template <typename T>
void FillRotated4x1PeerMap(T &kernelArg, const OpParam &param, const ChannelGroup &group)
{
    std::vector<uint32_t> order(group.peerRanks.size());
    for (uint32_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](uint32_t left, uint32_t right) {
        const uint32_t leftDistance =
            (group.peerRanks[left] + param.rankSize - param.myRank) % param.rankSize;
        const uint32_t rightDistance =
            (group.peerRanks[right] + param.rankSize - param.myRank) % param.rankSize;
        return leftDistance < rightDistance;
    });

    kernelArg.rankSize = param.rankSize;
    kernelArg.rankId = param.myRank;
    kernelArg.channelCount = static_cast<uint32_t>(order.size());
    for (uint32_t issueIndex = 0; issueIndex < order.size(); ++issueIndex) {
        const uint32_t originalIndex = order[issueIndex];
        kernelArg.channels[issueIndex] = group.channels[originalIndex];
        kernelArg.peerRanks[issueIndex] = group.peerRanks[originalIndex];
    }
}

template <typename T>
void FillRotatedCrossPeerMap(T &kernelArg, const OpParam &param, const ChannelGroup &group)
{
    std::vector<uint32_t> order(group.peerRanks.size());
    for (uint32_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    const bool sourceIsSmallServer = param.myRank >= 8U;
    const uint32_t sourceLocalRank =
        sourceIsSmallServer ? param.myRank - 8U : param.myRank % 4U;
    const uint32_t remoteServerSize = sourceIsSmallServer ? 8U : 4U;
    std::sort(order.begin(), order.end(), [&](uint32_t left, uint32_t right) {
        const uint32_t leftLocalRank =
            sourceIsSmallServer ? group.peerRanks[left] : group.peerRanks[left] - 8U;
        const uint32_t rightLocalRank =
            sourceIsSmallServer ? group.peerRanks[right] : group.peerRanks[right] - 8U;
        const uint32_t leftDistance =
            (leftLocalRank + remoteServerSize - sourceLocalRank) % remoteServerSize;
        const uint32_t rightDistance =
            (rightLocalRank + remoteServerSize - sourceLocalRank) % remoteServerSize;
        return leftDistance < rightDistance;
    });

    kernelArg.rankSize = param.rankSize;
    kernelArg.rankId = param.myRank;
    kernelArg.channelCount = static_cast<uint32_t>(order.size());
    for (uint32_t issueIndex = 0; issueIndex < order.size(); ++issueIndex) {
        const uint32_t originalIndex = order[issueIndex];
        kernelArg.channels[issueIndex] = group.channels[originalIndex];
        kernelArg.peerRanks[issueIndex] = group.peerRanks[originalIndex];
    }
}

std::shared_ptr<ops_hccl::V22CcuKernelArgAllGatherGroup>
V22BuildGenericKernelArg(const V22ChannelGroup &group,
    bool exchangeResources = true, bool finalSync = true,
    bool forceLocalCopy = false, bool latencyPipeline = false,
    bool progressiveAddressReady = false,
    bool rotateSecondSlice = false, bool combineSliceWaits = false)
{
    auto arg = std::make_shared<ops_hccl::V22CcuKernelArgAllGatherGroup>();
    arg->channelCount = static_cast<uint32_t>(group.channels.size());
    arg->handleLocalCopy = group.handleLocalCopy || forceLocalCopy;
    arg->exchangeResources = exchangeResources;
    arg->finalSync = finalSync;
    arg->latencyPipeline = latencyPipeline;
    arg->progressiveAddressReady = progressiveAddressReady;
    arg->rotateSecondSlice = rotateSecondSlice;
    arg->combineSliceWaits = combineSliceWaits;
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        arg->channels[i] = group.channels[i];
    }
    return arg;
}

std::shared_ptr<ops_hccl::V22CcuKernelArgAllGatherPhase>
V22BuildPhaseKernelArg(const V22ChannelGroup &group,
    uint32_t rankId,
    ops_hccl::V22CcuPhaseKind phaseKind,
    uint32_t patternSize,
    uint32_t incomingPatternSize,
    uint32_t seedCount,
    uint32_t incomingSeedCount,
    uint32_t chunkCount,
    const std::vector<ops_hccl::V22CcuTransferPlan> &plans,
    bool exchangeResources,
    bool seedRecord,
    bool seedWait,
    bool finalSync,
    bool progressiveAddressReady = false,
    bool rotateSecondChunk = false)
{
    auto arg = std::make_shared<ops_hccl::V22CcuKernelArgAllGatherPhase>();
    arg->channelCount = static_cast<uint32_t>(group.channels.size());
    arg->rankId = rankId;
    arg->phaseKind = phaseKind;
    arg->patternSize = patternSize;
    arg->incomingPatternSize = incomingPatternSize;
    arg->seedCount = seedCount;
    arg->incomingSeedCount = incomingSeedCount;
    arg->chunkCount = chunkCount;
    arg->operationCount = static_cast<uint32_t>(plans.size());
    arg->exchangeResources = exchangeResources;
    arg->seedRecord = seedRecord;
    arg->seedWait = seedWait;
    arg->finalSync = finalSync;
    arg->progressiveAddressReady = progressiveAddressReady;
    arg->rotateSecondChunk = rotateSecondChunk;
    for (uint32_t i = 0; i < arg->operationCount; ++i) {
        arg->plans[i] = plans[i];
    }
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        arg->channels[i] = group.channels[i];
        arg->peerRanks[i] = group.peerRanks[i];
    }
    return arg;
}

HcclResult V22RegisterOneKernel(CcuInsHandle insHandle,
    EndpointAttrDieId dieId,
    const char *name,
    void *kernelFunc,
    const std::shared_ptr<CcuKernelArgBase> &kernelArg,
    CcuKernelHandle &kernelHandle)
{
    CcuResult result = HcommCcuKernelRegisterStart(insHandle);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("[V22RegisterOneKernel] RegisterStart failed for %s, ccuRet=%d",
            name, static_cast<int>(result));
        return ConvertCcuToHccl(result);
    }

    const void *registerArgs[] = {static_cast<void *>(kernelArg.get())};
    result = HcommCcuKernelRegister(insHandle, dieId, name, kernelFunc,
        registerArgs, KERNEL_ARG_NUM, &kernelHandle);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("[V22RegisterOneKernel] Register failed for %s on Die %u, ccuRet=%d",
            name, static_cast<uint32_t>(dieId), static_cast<int>(result));
        (void)HcommCcuKernelRegisterEnd(insHandle);
        return ConvertCcuToHccl(result);
    }

    result = HcommCcuKernelRegisterEnd(insHandle);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("[V22RegisterOneKernel] RegisterEnd failed for %s, ccuRet=%d",
            name, static_cast<int>(result));
        return ConvertCcuToHccl(result);
    }
    return HCCL_SUCCESS;
}

HcclResult Register2x8V22Kernels(HcclComm comm,
    const OpParam &param,
    const std::vector<V22ChannelGroup> &groups,
    bool latencyMode,
    AlgResourceCtx &resource)
{
    CHK_PRT_RET(param.rankSize != V22_TWO_BY_EIGHT_RANK_SIZE
            || groups.size() != 2U
            || !groups[0].isIntraServer
            || groups[1].isIntraServer
            || groups[0].channels.size() != V22_RANKS_PER_SERVER - 1U
            || groups[1].channels.size() != V22_RANKS_PER_SERVER
            || groups[0].dieId == groups[1].dieId,
        HCCL_ERROR("[Register2x8V22Kernels] Invalid strict 2x8 groups"),
        HCCL_E_NOT_SUPPORT);

    CcuInsHandle insHandle{};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1U,
        HCCL_ERROR("[Register2x8V22Kernels] Expected one CCU instance, got %u",
            insNum),
        HCCL_E_INTERNAL);

    resource.ccuKernels.resize(V22_PIPELINE_KERNEL_NUM);
    // latencyMode下数据事件即完成条件，关闭逐Peer FullSync尾部。
    auto genericIntra =
        V22BuildGenericKernelArg(groups[0], true, !latencyMode, false, latencyMode);
    auto genericCross =
        V22BuildGenericKernelArg(groups[1], true, !latencyMode, false, latencyMode);
    auto pipelineIntraOriginal =
        V22BuildGenericKernelArg(groups[0], true, false, false, false);

    const std::vector<ops_hccl::V22CcuTransferPlan>
        hierarchicalPlans = {{0U, 0U, 0U, 0U, 0U}};
    const std::vector<ops_hccl::V22CcuTransferPlan>
        directSinglePlans = {{0U, 0U, 0U, 0U, 0U}};
    const std::vector<ops_hccl::V22CcuTransferPlan>
        directDoublePlans = {
            {0U, 0U, 0U, 0U, 0U},
            {1U, 0U, 0U, 0U, 0U},
        };
    const uint32_t remoteSourceIndex =
        param.myRank % V22_RANKS_PER_SERVER;
    const std::vector<ops_hccl::V22CcuTransferPlan>
        relayPlans = {
            {0U, 0U, 0U, 0U, remoteSourceIndex},
            {0U, 0U, 1U, 0U, remoteSourceIndex},
            {0U, 0U, 2U, 0U, remoteSourceIndex},
            {0U, 0U, 3U, 0U, remoteSourceIndex},
            {0U, 0U, 4U, 0U, remoteSourceIndex},
            {0U, 0U, 5U, 0U, remoteSourceIndex},
            {0U, 0U, 6U, 0U, remoteSourceIndex},
        };

    auto intraRelay = V22BuildPhaseKernelArg(
        groups[0], param.myRank,
        ops_hccl::V22CcuPhaseKind::INTRA_RELAY_BATCH,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        1U, 1U, 1U, relayPlans,
        false, false, false, true);
    auto crossHierarchical = V22BuildPhaseKernelArg(
        groups[1], param.myRank,
        ops_hccl::V22CcuPhaseKind::CROSS_DIRECT_BATCH,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        1U, 1U, 1U, hierarchicalPlans,
        true, true, true, false);
    auto crossDirectSingle = V22BuildPhaseKernelArg(
        groups[1], param.myRank,
        ops_hccl::V22CcuPhaseKind::CROSS_DIRECT_BATCH,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        1U, directSinglePlans,
        false, false, false, true);
    auto crossDirectDouble = V22BuildPhaseKernelArg(
        groups[1], param.myRank,
        ops_hccl::V22CcuPhaseKind::CROSS_DIRECT_BATCH,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        2U, directDoublePlans,
        false, false, false, true);

    // The handle order is part of the V22 execution ABI. Keep each
    // Start/Register/End transaction independent so kernels on different Dies
    // are never selected as one mixed-Die registration unit.
    CHK_RET(V22RegisterOneKernel(insHandle, groups[0].dieId,
        "CcuFusionV23GenericIntra",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherGroupKernel),
        genericIntra,
        resource.ccuKernels[V22_GENERIC_INTRA_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[1].dieId,
        "CcuFusionV23GenericCross",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherGroupKernel),
        genericCross,
        resource.ccuKernels[V22_GENERIC_CROSS_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[0].dieId,
        "CcuFusionV23IntraOriginalNoFinal",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherGroupKernel),
        pipelineIntraOriginal,
        resource.ccuKernels[V22_PIPELINE_INTRA_ORIGINAL_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[0].dieId,
        "CcuFusionV23IntraRelayFinal",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherPhaseKernel),
        intraRelay,
        resource.ccuKernels[V22_INTRA_RELAY_FINAL_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[1].dieId,
        "CcuFusionV23CrossHierarchical",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherPhaseKernel),
        crossHierarchical,
        resource.ccuKernels[V22_CROSS_HIERARCHICAL_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[1].dieId,
        "CcuFusionV23CrossDirectSingleFinal",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherPhaseKernel),
        crossDirectSingle,
        resource.ccuKernels[V22_CROSS_DIRECT_SINGLE_FINAL_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[1].dieId,
        "CcuFusionV23CrossDirectDoubleFinal",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherPhaseKernel),
        crossDirectDouble,
        resource.ccuKernels[V22_CROSS_DIRECT_DOUBLE_FINAL_KERNEL]));

    resource.peerRanksByGroup = {
        groups[0].peerRanks,
        groups[1].peerRanks,
    };
    resource.topologyType =
        static_cast<uint32_t>(ops_hccl::TopologyType::DUAL_SERVER_2X8);
    resource.algorithmMode = ALGORITHM_2X8_V22_HYBRID;
    return HCCL_SUCCESS;
}

HcclResult Register2x8BandwidthKernels(HcclComm comm,
    const OpParam &param,
    const std::vector<V22ChannelGroup> &groups,
    bool bandwidthOptimized,
    bool inlineTailCopy,
    AlgResourceCtx &resource)
{
    CHK_PRT_RET(param.rankSize != V22_TWO_BY_EIGHT_RANK_SIZE
            || groups.size() != 2U
            || !groups[0].isIntraServer
            || groups[1].isIntraServer
            || groups[0].channels.size() != V22_RANKS_PER_SERVER - 1U
            || groups[1].channels.size() != V22_RANKS_PER_SERVER
            || groups[0].dieId == groups[1].dieId,
        HCCL_ERROR("[Register2x8V22Kernels] Invalid strict 2x8 groups"),
        HCCL_E_NOT_SUPPORT);

    CcuInsHandle insHandle{};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1U,
        HCCL_ERROR("[Register2x8V22Kernels] Expected one CCU instance, got %u",
            insNum),
        HCCL_E_INTERNAL);

    resource.ccuKernels.resize(V22_PIPELINE_KERNEL_NUM);
    auto genericIntra = V22BuildGenericKernelArg(groups[0]);
    auto genericCross = V22BuildGenericKernelArg(groups[1]);
    auto pipelineIntraOriginal =
        V22BuildGenericKernelArg(groups[0], true, false, false,
            false, bandwidthOptimized, bandwidthOptimized, inlineTailCopy);
    pipelineIntraOriginal->handleLocalCopy =
        !bandwidthOptimized || inlineTailCopy;

    const std::vector<ops_hccl::V22CcuTransferPlan>
        hierarchicalPlans = {{0U, 0U, 0U, 0U, 0U}};
    const std::vector<ops_hccl::V22CcuTransferPlan>
        directSinglePlans = {{0U, 0U, 0U, 0U, 0U}};
    const std::vector<ops_hccl::V22CcuTransferPlan>
        directDoublePlans = {
            {0U, 0U, 0U, 0U, 0U},
            {1U, 0U, 0U, 0U, 0U},
        };
    const uint32_t remoteSourceIndex =
        param.myRank % V22_RANKS_PER_SERVER;
    const std::vector<ops_hccl::V22CcuTransferPlan>
        relayPlans = {
            {0U, 0U, 0U, 0U, remoteSourceIndex},
            {0U, 0U, 1U, 0U, remoteSourceIndex},
            {0U, 0U, 2U, 0U, remoteSourceIndex},
            {0U, 0U, 3U, 0U, remoteSourceIndex},
            {0U, 0U, 4U, 0U, remoteSourceIndex},
            {0U, 0U, 5U, 0U, remoteSourceIndex},
            {0U, 0U, 6U, 0U, remoteSourceIndex},
        };

    auto intraRelay = V22BuildPhaseKernelArg(
        groups[0], param.myRank,
        ops_hccl::V22CcuPhaseKind::INTRA_RELAY_BATCH,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        1U, 1U, 1U, relayPlans,
        false, false, false, true);
    auto crossHierarchical = V22BuildPhaseKernelArg(
        groups[1], param.myRank,
        ops_hccl::V22CcuPhaseKind::CROSS_DIRECT_BATCH,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        1U, 1U, 1U, hierarchicalPlans,
        true, true, true, false, bandwidthOptimized, false);
    auto crossDirectSingle = V22BuildPhaseKernelArg(
        groups[1], param.myRank,
        ops_hccl::V22CcuPhaseKind::CROSS_DIRECT_BATCH,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        1U, directSinglePlans,
        false, false, false, true);
    auto crossDirectDouble = V22BuildPhaseKernelArg(
        groups[1], param.myRank,
        ops_hccl::V22CcuPhaseKind::CROSS_DIRECT_BATCH,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        V22_RANKS_PER_SERVER, V22_RANKS_PER_SERVER,
        2U, directDoublePlans,
        false, false, false, true, false, bandwidthOptimized);

    // The handle order is part of the V22 execution ABI. Keep each
    // Start/Register/End transaction independent so kernels on different Dies
    // are never selected as one mixed-Die registration unit.
    CHK_RET(V22RegisterOneKernel(insHandle, groups[0].dieId,
        inlineTailCopy ? "CcuV25TGenericIntra"
            : bandwidthOptimized ? "CcuCrossApplyV24BGenericIntra"
                                 : "CcuCrossApplyV24GGenericIntra",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherGroupKernel),
        genericIntra,
        resource.ccuKernels[V22_GENERIC_INTRA_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[1].dieId,
        inlineTailCopy ? "CcuV25TGenericCross"
            : bandwidthOptimized ? "CcuCrossApplyV24BGenericCross"
                                 : "CcuCrossApplyV24GGenericCross",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherGroupKernel),
        genericCross,
        resource.ccuKernels[V22_GENERIC_CROSS_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[0].dieId,
        inlineTailCopy ? "CcuV25TIntraOriginal"
            : bandwidthOptimized ? "CcuCrossApplyV24BIntraOriginal"
                                 : "CcuCrossApplyV24GIntraOriginal",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherGroupKernel),
        pipelineIntraOriginal,
        resource.ccuKernels[V22_PIPELINE_INTRA_ORIGINAL_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[0].dieId,
        inlineTailCopy ? "CcuV25TIntraRelay"
            : bandwidthOptimized ? "CcuCrossApplyV24BIntraRelay"
                                 : "CcuCrossApplyV24GIntraRelay",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherPhaseKernel),
        intraRelay,
        resource.ccuKernels[V22_INTRA_RELAY_FINAL_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[1].dieId,
        inlineTailCopy ? "CcuV25TCrossSeed"
            : bandwidthOptimized ? "CcuCrossApplyV24BCrossSeed"
                                 : "CcuCrossApplyV24GCrossSeed",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherPhaseKernel),
        crossHierarchical,
        resource.ccuKernels[V22_CROSS_HIERARCHICAL_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[1].dieId,
        inlineTailCopy ? "CcuV25TDirectSingle"
            : bandwidthOptimized ? "CcuCrossApplyV24BDirectSingle"
                                 : "CcuCrossApplyV24GDirectSingle",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherPhaseKernel),
        crossDirectSingle,
        resource.ccuKernels[V22_CROSS_DIRECT_SINGLE_FINAL_KERNEL]));
    CHK_RET(V22RegisterOneKernel(insHandle, groups[1].dieId,
        inlineTailCopy ? "CcuV25TDirectDouble"
            : bandwidthOptimized ? "CcuCrossApplyV24BDirectDouble"
                                 : "CcuCrossApplyV24GDirectDouble",
        reinterpret_cast<void *>(ops_hccl::CcuV22AllGatherPhaseKernel),
        crossDirectDouble,
        resource.ccuKernels[V22_CROSS_DIRECT_DOUBLE_FINAL_KERNEL]));

    resource.peerRanksByGroup = {
        groups[0].peerRanks,
        groups[1].peerRanks,
    };
    resource.topologyType =
        static_cast<uint32_t>(ops_hccl::TopologyType::DUAL_SERVER_2X8);
    resource.algorithmMode = ALGORITHM_2X8_V22_HYBRID;
    return HCCL_SUCCESS;
}

template <typename T>
CcuResult RegisterOneSpecializedKernel(CcuInsHandle insHandle, uint32_t dieId, const char *kernelName,
    void *kernelFunc, const std::shared_ptr<T> &kernelArg, CcuKernelHandle &kernelHandle)
{
    CcuKernelInfo kernelInfo{};
    const int nameLength =
        std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s", kernelName);
    if (nameLength < 0 || static_cast<size_t>(nameLength) >= sizeof(kernelInfo.kernelFuncName)) {
        return CCU_E_PARA;
    }
    kernelInfo.kernelFunc = kernelFunc;
    kernelInfo.setKernelArg(kernelArg);
    const void *kernelArgs[] = {kernelInfo.kernelArg};
    return HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
        kernelInfo.kernelFunc, kernelArgs, KERNEL_ARG_NUM, &kernelHandle);
}

template <typename T>
void FillOutput512GroupArg(T &kernelArg, const OpParam &param,
    const ChannelGroup &group, bool handleLocalCopy)
{
    kernelArg.rankSize = param.rankSize;
    kernelArg.rankId = param.myRank;
    kernelArg.channelCount = static_cast<uint32_t>(group.channels.size());
    kernelArg.handleLocalCopy = handleLocalCopy;
    for (uint32_t i = 0U; i < kernelArg.channelCount; ++i) {
        kernelArg.channels[i] = group.channels[i];
        kernelArg.peerRanks[i] = group.peerRanks[i];
    }
}

HcclResult RegisterOutput512StaticKernels(HcclComm comm,
    const OpParam &param, const std::vector<ChannelGroup> &groups,
    AlgResourceCtx &resource)
{
    const uint32_t groupCount = static_cast<uint32_t>(groups.size());
    CHK_PRT_RET(groupCount == 0U || groupCount > 2U,
        HCCL_ERROR("[RegisterOutput512StaticKernels] Invalid group count %u",
            groupCount),
        HCCL_E_NOT_SUPPORT);
    for (uint32_t i = 0U; i < groupCount; ++i) {
        CHK_PRT_RET(groups[i].channels.empty()
                || groups[i].channels.size() > MAX_CHANNELS_PER_DIE
                || groups[i].channels.size() != groups[i].peerRanks.size(),
            HCCL_ERROR("[RegisterOutput512StaticKernels] Invalid group %u", i),
            HCCL_E_NOT_SUPPORT);
    }

    const uint64_t rankBytes = param.count * sizeof(float);
    CHK_PRT_RET(!IsOutput512Target(rankBytes, param.rankSize),
        HCCL_ERROR("[RegisterOutput512StaticKernels] Invalid geometry"),
        HCCL_E_NOT_SUPPORT);
    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0U;
    uint64_t outputToken = 0U;
    CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, rankBytes, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(
        outputAddr, rankBytes * param.rankSize, &outputToken));
    const uint64_t rankOutputOffset =
        static_cast<uint64_t>(param.myRank) * rankBytes;
    const bool needsOwnCopy = inputAddr != outputAddr + rankOutputOffset;

    std::shared_ptr<CcuLatencyPushKernelArg> dynamicArgs[2];
    const uint32_t ownCopyGroup =
        param.rankSize == 12U && param.myRank >= 8U && groupCount == 2U
        ? 1U : 0U;
    for (uint32_t i = 0U; i < groupCount; ++i) {
        const bool handleLocalCopy = i == ownCopyGroup && needsOwnCopy;
        dynamicArgs[i] = std::make_shared<CcuLatencyPushKernelArg>();
        FillOutput512GroupArg(
            *dynamicArgs[i], param, groups[i], handleLocalCopy);

    }

    CcuInsHandle insHandle{};
    uint32_t insNum = 0U;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1U,
        HCCL_ERROR("[RegisterOutput512StaticKernels] Expected one CCU instance, got %u",
            insNum),
        HCCL_E_INTERNAL);

    // Retain B's rank-12 scheduling order, but install only the proven
    // dynamic graph on each Die so the microcode loader cannot exhaust slots.
    resource.ccuKernels.resize(groupCount);
    const uint32_t topologyIndex =
        param.rankSize == 4U ? 0U : param.rankSize == 12U ? 1U : 2U;
    CcuResult ret = HcommCcuKernelRegisterStart(insHandle);
    if (ret != CCU_SUCCESS) {
        return ConvertCcuToHccl(ret);
    }
    for (uint32_t i = 0U; i < groupCount && ret == CCU_SUCCESS; ++i) {
        ret = RegisterOneSpecializedKernel(insHandle, groups[i].dieId,
            OUTPUT_512_DYNAMIC_KERNEL_NAMES[topologyIndex][i],
            reinterpret_cast<void *>(ops_hccl::CcuLatencyDirectPushKernel),
            dynamicArgs[i], resource.ccuKernels[i]);
    }
    const CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ret != CCU_SUCCESS) {
        return ConvertCcuToHccl(ret);
    }
    if (endRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(endRet);
    }

    resource.peerRanksByGroup.clear();
    for (const ChannelGroup &group : groups) {
        resource.peerRanksByGroup.push_back(group.peerRanks);
    }
    resource.topologyType = static_cast<uint32_t>(
        param.rankSize == 4U ? ops_hccl::TopologyType::MESH_4X1
        : param.rankSize == 12U ? ops_hccl::TopologyType::ASYMMETRIC_8P4
                                : ops_hccl::TopologyType::DUAL_SERVER_2X8);
    resource.algorithmMode = ALGORITHM_OUTPUT_512_STATIC;
    resource.hasOutput512StaticSession = 1U;
    resource.output512InputAddr = inputAddr;
    resource.output512OutputAddr = outputAddr;
    resource.output512InputToken = inputToken;
    resource.output512OutputToken = outputToken;
    resource.output512RankBytes = rankBytes;
    return HCCL_SUCCESS;
}

HcclResult Register4x1CommonDieKernel(HcclComm comm, const OpParam &param,
    const ChannelGroup &group, bool latencyMode, AlgResourceCtx &resource)
{
    CcuInsHandle insHandle{};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1U,
        HCCL_ERROR("[Register4x1CommonDieKernel] Expected one CCU instance, got %u", insNum),
        HCCL_E_INTERNAL);

    const uint64_t rankBytes = param.count * sizeof(float);
    const bool exact512 = latencyMode && rankBytes == EXACT_512_PAYLOAD_BYTES;

    auto latencyArg = std::make_shared<CcuLatencyPushKernelArg>();
    auto bandwidthArg = std::make_shared<CcuClos4x1PushKernelArg>();
    std::shared_ptr<CcuRegistered4x1PushKernelArg> coldArg;
    std::shared_ptr<CcuRegistered4x1PushKernelArg> warmArg;

    if (latencyMode) {
        FillRotated4x1PeerMap(*latencyArg, param, group);
        latencyArg->handleLocalCopy = true;
    } else {
        FillRotated4x1PeerMap(*bandwidthArg, param, group);
        bandwidthArg->chunkCount = 4U;
        bandwidthArg->handleLocalCopy = false;
        bandwidthArg->latencyPipeline = false;
    }

    if (exact512) {
        uint64_t inputToken = 0U;
        uint64_t outputToken = 0U;
        const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
        const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
        CHK_RET_CCU(HcommCcuGetMemToken(
            inputAddr, EXACT_512_PAYLOAD_BYTES, &inputToken));
        CHK_RET_CCU(HcommCcuGetMemToken(outputAddr,
            EXACT_512_PAYLOAD_BYTES * 4U, &outputToken));

        coldArg = std::make_shared<CcuRegistered4x1PushKernelArg>();
        FillRotated4x1PeerMap(*coldArg, param, group);
        coldArg->inputAddr = inputAddr;
        coldArg->outputAddr = outputAddr;
        coldArg->inputToken = inputToken;
        coldArg->outputToken = outputToken;
        coldArg->rankOutputOffset =
            static_cast<uint64_t>(param.myRank) * EXACT_512_PAYLOAD_BYTES;
        coldArg->transferBytes = EXACT_512_PAYLOAD_BYTES;
        coldArg->publishResources = true;
        const uint64_t ownOutputAddr =
            outputAddr + coldArg->rankOutputOffset;
        coldArg->handleLocalCopy = inputAddr != ownOutputAddr;
        warmArg = std::make_shared<CcuRegistered4x1PushKernelArg>(*coldArg);
        warmArg->publishResources = false;

        resource.hasRegistered4x1Exact512 = 1U;
        resource.registered4x1InputAddr = inputAddr;
        resource.registered4x1OutputAddr = outputAddr;
        resource.registered4x1InputToken = inputToken;
        resource.registered4x1OutputToken = outputToken;
    }

    const CcuResult startRet = HcommCcuKernelRegisterStart(insHandle);
    if (startRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(startRet);
    }

    CcuResult registerRet = CCU_SUCCESS;
    CcuKernelHandle dynamicKernel{};
    CcuKernelHandle coldKernel{};
    CcuKernelHandle warmKernel{};
    if (latencyMode) {
        registerRet = RegisterOneSpecializedKernel(
            insHandle, group.dieId, LATENCY_4X1_KERNEL_NAME,
            reinterpret_cast<void *>(ops_hccl::CcuLatencyDirectPushKernel),
            latencyArg, dynamicKernel);
        if (registerRet == CCU_SUCCESS && exact512) {
            registerRet = RegisterOneSpecializedKernel(
                insHandle, group.dieId, REGISTERED_4X1_COLD_KERNEL_NAME,
                reinterpret_cast<void *>(ops_hccl::CcuRegistered4x1DirectPushKernel),
                coldArg, coldKernel);
        }
        if (registerRet == CCU_SUCCESS && exact512) {
            registerRet = RegisterOneSpecializedKernel(
                insHandle, group.dieId, REGISTERED_4X1_WARM_KERNEL_NAME,
                reinterpret_cast<void *>(ops_hccl::CcuRegistered4x1DirectPushKernel),
                warmArg, warmKernel);
        }
    } else {
        registerRet = RegisterOneSpecializedKernel(
            insHandle, group.dieId, COMMON_DIE_BANDWIDTH_KERNEL_NAME,
            reinterpret_cast<void *>(ops_hccl::CcuClos4x1DirectPushKernel),
            bandwidthArg, dynamicKernel);
    }

    const CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (registerRet != CCU_SUCCESS) {
        HCCL_ERROR("[Register4x1CommonDieKernel] Kernel registration failed, ccuRet=%d",
            registerRet);
        return ConvertCcuToHccl(registerRet);
    }
    if (endRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(endRet);
    }

    resource.ccuKernels = exact512
        ? std::vector<CcuKernelHandle>{dynamicKernel, coldKernel, warmKernel}
        : std::vector<CcuKernelHandle>{dynamicKernel};
    resource.peerRanksByGroup = {group.peerRanks};
    resource.topologyType = static_cast<uint32_t>(ops_hccl::TopologyType::MESH_4X1);
    resource.algorithmMode = ALGORITHM_4X1_COMMON_DIE_PUSH;
    return HCCL_SUCCESS;
}

HcclResult Register4x1DualPlaneKernels(HcclComm comm, const OpParam &param,
    const std::vector<ChannelGroup> &groups, AlgResourceCtx &resource)
{
    CHK_PRT_RET(groups.size() != 2U || groups[0].dieId == groups[1].dieId
            || groups[0].channels.size() != 2U || groups[1].channels.size() != 1U,
        HCCL_ERROR("[Register4x1DualPlaneKernels] Invalid two-Die groups"),
        HCCL_E_INTERNAL);

    CcuInsHandle insHandle{};
    uint32_t insNum = 0U;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1U,
        HCCL_ERROR("[Register4x1DualPlaneKernels] Expected one CCU instance, got %u", insNum),
        HCCL_E_INTERNAL);

    auto mainArg = std::make_shared<CcuRotatingPushKernelArg>();
    FillRotated4x1PeerMap(*mainArg, param, groups[0]);
    mainArg->handleLocalCopy = false;
    auto workerArg = std::make_shared<CcuRotatingPushKernelArg>();
    FillRotated4x1PeerMap(*workerArg, param, groups[1]);
    workerArg->handleLocalCopy = false;

    const CcuResult startRet = HcommCcuKernelRegisterStart(insHandle);
    if (startRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(startRet);
    }
    CcuKernelHandle mainKernel{};
    CcuKernelHandle workerKernel{};
    CcuResult registerRet = RegisterOneSpecializedKernel(insHandle, groups[0].dieId,
        DUAL_PLANE_MAIN_KERNEL_NAME,
        reinterpret_cast<void *>(ops_hccl::CcuRotatingDirectPushKernel),
        mainArg, mainKernel);
    if (registerRet == CCU_SUCCESS) {
        registerRet = RegisterOneSpecializedKernel(insHandle, groups[1].dieId,
            DUAL_PLANE_WORKER_KERNEL_NAME,
            reinterpret_cast<void *>(ops_hccl::CcuRotatingDirectPushKernel),
            workerArg, workerKernel);
    }
    const CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (registerRet != CCU_SUCCESS) {
        HCCL_ERROR("[Register4x1DualPlaneKernels] Registration failed, ccuRet=%d", registerRet);
        return ConvertCcuToHccl(registerRet);
    }
    if (endRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(endRet);
    }

    resource.ccuKernels = {mainKernel, workerKernel};
    resource.peerRanksByGroup = {groups[0].peerRanks, groups[1].peerRanks};
    resource.topologyType = static_cast<uint32_t>(ops_hccl::TopologyType::MESH_4X1);
    resource.algorithmMode = ALGORITHM_4X1_DUAL_PLANE_PUSH;
    return HCCL_SUCCESS;
}

HcclResult Register8p4LatencyKernels(HcclComm comm, const OpParam &param,
    const ChannelGroup &intraGroup, const ChannelGroup &crossGroup,
    AlgResourceCtx &resource)
{
    CcuInsHandle insHandle{};
    uint32_t insNum = 0U;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1U || intraGroup.dieId == crossGroup.dieId,
        HCCL_ERROR("[Register8p4LatencyKernels] Invalid CCU/Die layout"),
        HCCL_E_INTERNAL);

    auto intraArg = std::make_shared<CcuLatencyPushKernelArg>();
    FillStaticPeerMap(*intraArg, param, intraGroup);
    intraArg->handleLocalCopy = true;
    auto crossArg = std::make_shared<CcuLatencyPushKernelArg>();
    FillRotatedCrossPeerMap(*crossArg, param, crossGroup);
    crossArg->handleLocalCopy = false;

    const CcuResult startRet = HcommCcuKernelRegisterStart(insHandle);
    if (startRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(startRet);
    }
    CcuKernelHandle intraKernel{};
    CcuKernelHandle crossKernel{};
    CcuResult registerRet = RegisterOneSpecializedKernel(
        insHandle, intraGroup.dieId, LATENCY_8P4_INTRA_KERNEL_NAME,
        reinterpret_cast<void *>(ops_hccl::CcuLatencyDirectPushKernel),
        intraArg, intraKernel);
    if (registerRet == CCU_SUCCESS) {
        registerRet = RegisterOneSpecializedKernel(
            insHandle, crossGroup.dieId, LATENCY_8P4_CROSS_KERNEL_NAME,
            reinterpret_cast<void *>(ops_hccl::CcuLatencyDirectPushKernel),
            crossArg, crossKernel);
    }
    const CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (registerRet != CCU_SUCCESS) {
        HCCL_ERROR("[Register8p4LatencyKernels] Registration failed, ccuRet=%d",
            registerRet);
        return ConvertCcuToHccl(registerRet);
    }
    if (endRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(endRet);
    }

    resource.ccuKernels = {intraKernel, crossKernel};
    resource.peerRanksByGroup = {intraGroup.peerRanks, crossGroup.peerRanks};
    resource.topologyType =
        static_cast<uint32_t>(ops_hccl::TopologyType::ASYMMETRIC_8P4);
    resource.algorithmMode = ALGORITHM_8P4_LATENCY_DIRECT;
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult Register8p4Kernels(HcclComm comm, const OpParam &param,
    const ChannelGroup &intraGroup, const ChannelGroup &crossGroup,
    bool latencyMode, AlgResourceCtx &resource)
{
    CcuInsHandle insHandle{};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1U,
        HCCL_ERROR("[Register8p4Kernels] Expected one CCU instance, got %u", insNum), HCCL_E_INTERNAL);

    auto intraArg = std::make_shared<CcuRotatingPushKernelArg>();
    FillStaticPeerMap(*intraArg, param, intraGroup);
    intraArg->handleLocalCopy = true;
    intraArg->latencyPipeline = latencyMode;

    auto crossArg = std::make_shared<CcuDualSeedCrossKernelArg>();
    FillRotatedCrossPeerMap(*crossArg, param, crossGroup);
    crossArg->sourceIsSmallServer = param.myRank >= 8U;
    crossArg->latencyPipeline = latencyMode;
    if (crossArg->sourceIsSmallServer) {
        const uint32_t smallIndex = param.myRank - 8U;
        crossArg->seedRanks[0] = 2U * smallIndex;
        crossArg->seedRanks[1] = 2U * smallIndex + 1U;
    } else {
        crossArg->seedRanks[0] = 8U + (param.myRank % 4U);
        crossArg->seedRanks[1] = crossArg->seedRanks[0];
    }

    auto relayArg = std::make_shared<CcuDualSeedRelayKernelArg>();
    FillStaticPeerMap(*relayArg, param, intraGroup);
    relayArg->enabled = true;
    relayArg->relayOnLargeServer = param.myRank < 8U;
    if (relayArg->relayOnLargeServer) {
        relayArg->sourceRankCount = 1U;
        relayArg->sourceRanks[0] = 8U + param.myRank / 2U;
        relayArg->stripeIndices[0] = param.myRank % 2U;
    } else {
        const uint32_t smallIndex = param.myRank - 8U;
        relayArg->sourceRankCount = 2U;
        relayArg->sourceRanks[0] = smallIndex;
        relayArg->sourceRanks[1] = smallIndex + 4U;
        relayArg->stripeIndices[0] = 0U;
        relayArg->stripeIndices[1] = 0U;
    }

    const CcuResult startRet = HcommCcuKernelRegisterStart(insHandle);
    if (startRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(startRet);
    }

    CcuKernelHandle intraKernel{};
    CcuKernelHandle crossKernel{};
    CcuKernelHandle relayKernel{};
    CcuResult registerRet = RegisterOneSpecializedKernel(insHandle, intraGroup.dieId,
        ROTATING_PUSH_KERNEL_NAME, reinterpret_cast<void *>(ops_hccl::CcuRotatingDirectPushKernel),
        intraArg, intraKernel);
    if (registerRet == CCU_SUCCESS) {
        registerRet = RegisterOneSpecializedKernel(insHandle, crossGroup.dieId,
            DUAL_SEED_CROSS_KERNEL_NAME, reinterpret_cast<void *>(ops_hccl::CcuDualSeedCrossKernel),
            crossArg, crossKernel);
    }
    if (registerRet == CCU_SUCCESS) {
        registerRet = RegisterOneSpecializedKernel(insHandle, intraGroup.dieId,
            DUAL_SEED_RELAY_KERNEL_NAME, reinterpret_cast<void *>(ops_hccl::CcuDualSeedRelayKernel),
            relayArg, relayKernel);
    }
    const CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (registerRet != CCU_SUCCESS) {
        HCCL_ERROR("[Register8p4Kernels] Kernel registration failed, ccuRet=%d", registerRet);
        return ConvertCcuToHccl(registerRet);
    }
    if (endRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(endRet);
    }

    // 固定顺序：[0]机内Push，[1]动态Seed/Direct Cross，[2]机内Relay。
    resource.ccuKernels = {intraKernel, crossKernel, relayKernel};
    resource.peerRanksByGroup = {intraGroup.peerRanks,
        std::vector<uint32_t>(crossArg->peerRanks,
            crossArg->peerRanks + crossArg->channelCount)};
    resource.topologyType = static_cast<uint32_t>(ops_hccl::TopologyType::ASYMMETRIC_8P4);
    resource.algorithmMode = ALGORITHM_8P4_DUAL_SEED;
    return HCCL_SUCCESS;
}

HcclResult Register8p4WideKernels(HcclComm comm, const OpParam &param,
    const ChannelGroup &intraGroup, const ChannelGroup &crossGroup,
    AlgResourceCtx &resource)
{
    CcuInsHandle insHandle{};
    uint32_t insNum = 0U;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1U,
        HCCL_ERROR("[Register8p4WideKernels] Expected one CCU instance, got %u", insNum),
        HCCL_E_INTERNAL);

    auto intraArg = std::make_shared<CcuRotatingPushKernelArg>();
    FillStaticPeerMap(*intraArg, param, intraGroup);
    intraArg->handleLocalCopy = false;
    intraArg->exchangeResources = true;
    intraArg->progressiveAddressReady = true;
    intraArg->finalSync = false;

    auto fillCrossArg = [&](CcuWideDualSeedCrossKernelArg &arg) {
        FillRotatedCrossPeerMap(arg, param, crossGroup);
        arg.sourceIsSmallServer = param.myRank >= 8U;
        if (arg.sourceIsSmallServer) {
            const uint32_t smallIndex = param.myRank - 8U;
            arg.seedRanks[0] = 2U * smallIndex;
            arg.seedRanks[1] = 2U * smallIndex + 1U;
            arg.incomingSeedRankCount = 2U;
            arg.incomingSeedRanks[0] = smallIndex;
            arg.incomingSeedRanks[1] = smallIndex + 4U;
        } else {
            arg.seedRanks[0] = 8U + (param.myRank % 4U);
            arg.seedRanks[1] = arg.seedRanks[0];
            arg.incomingSeedRankCount = 1U;
            arg.incomingSeedRanks[0] = 8U + param.myRank / 2U;
        }
    };

    auto crossSeedArg = std::make_shared<CcuWideDualSeedCrossKernelArg>();
    fillCrossArg(*crossSeedArg);
    crossSeedArg->phaseKind = CcuDualSeedPhaseKind::SEED;
    crossSeedArg->exchangeResources = true;
    crossSeedArg->finalSync = false;

    auto crossDirectArg = std::make_shared<CcuWideDualSeedCrossKernelArg>();
    fillCrossArg(*crossDirectArg);
    crossDirectArg->phaseKind = CcuDualSeedPhaseKind::DIRECT;
    crossDirectArg->exchangeResources = false;
    crossDirectArg->finalSync = true;

    auto relayArg = std::make_shared<CcuDualSeedRelayKernelArg>();
    FillStaticPeerMap(*relayArg, param, intraGroup);
    relayArg->enabled = true;
    relayArg->relayOnLargeServer = param.myRank < 8U;
    if (relayArg->relayOnLargeServer) {
        relayArg->sourceRankCount = 1U;
        relayArg->sourceRanks[0] = 8U + param.myRank / 2U;
        relayArg->stripeIndices[0] = param.myRank % 2U;
    } else {
        const uint32_t smallIndex = param.myRank - 8U;
        relayArg->sourceRankCount = 2U;
        relayArg->sourceRanks[0] = smallIndex;
        relayArg->sourceRanks[1] = smallIndex + 4U;
        relayArg->stripeIndices[0] = 0U;
        relayArg->stripeIndices[1] = 0U;
    }

    resource.ccuKernels.resize(4U);
    // 每个Kernel独立Start/Register/End，避免把两个IO Die放进同一注册选择单元。
    CHK_RET(V22RegisterOneKernel(insHandle,
        static_cast<EndpointAttrDieId>(intraGroup.dieId),
        WIDE_INTRA_KERNEL_NAME,
        reinterpret_cast<void *>(ops_hccl::CcuWideRotatingDirectPushKernel),
        intraArg, resource.ccuKernels[0]));
    CHK_RET(V22RegisterOneKernel(insHandle,
        static_cast<EndpointAttrDieId>(crossGroup.dieId),
        WIDE_CROSS_SEED_KERNEL_NAME,
        reinterpret_cast<void *>(ops_hccl::CcuWideDualSeedCrossKernel),
        crossSeedArg, resource.ccuKernels[1]));
    CHK_RET(V22RegisterOneKernel(insHandle,
        static_cast<EndpointAttrDieId>(crossGroup.dieId),
        WIDE_CROSS_DIRECT_KERNEL_NAME,
        reinterpret_cast<void *>(ops_hccl::CcuWideDualSeedCrossKernel),
        crossDirectArg, resource.ccuKernels[2]));
    CHK_RET(V22RegisterOneKernel(insHandle,
        static_cast<EndpointAttrDieId>(intraGroup.dieId),
        WIDE_RELAY_KERNEL_NAME,
        reinterpret_cast<void *>(ops_hccl::CcuWideDualSeedRelayKernel),
        relayArg, resource.ccuKernels[3]));

    resource.peerRanksByGroup = {
        intraGroup.peerRanks,
        std::vector<uint32_t>(crossSeedArg->peerRanks,
            crossSeedArg->peerRanks + crossSeedArg->channelCount),
    };
    resource.topologyType =
        static_cast<uint32_t>(ops_hccl::TopologyType::ASYMMETRIC_8P4);
    resource.algorithmMode = ALGORITHM_8P4_WIDE_PIPELINE;
    return HCCL_SUCCESS;
}

HcclResult RegisterDirectKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelGroup> &groups,
    ops_hccl::TopologyType topology, AlgResourceCtx &resource)
{
    CcuInsHandle insHandle{};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[RegisterDirectKernels] Expected one CCU instance, but got %u", insNum), HCCL_E_INTERNAL);

    const CcuResult registerStartRet = HcommCcuKernelRegisterStart(insHandle);
    if (registerStartRet != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterDirectKernels] Kernel registration start failed, ccuRet=%d", registerStartRet);
        return ConvertCcuToHccl(registerStartRet);
    }

    std::vector<CcuKernelHandle> kernelHandles;
    std::vector<std::vector<uint32_t>> peerRanksByGroup;
    std::vector<std::shared_ptr<CcuDirectKernelArg>> retainedArgs;
    kernelHandles.reserve(groups.size());
    peerRanksByGroup.reserve(groups.size());
    retainedArgs.reserve(groups.size());
    HcclResult hostFailure = HCCL_SUCCESS;
    CcuResult registerFailure = CCU_SUCCESS;

    for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const ChannelGroup &group = groups[groupIndex];
        if (group.channels.empty() || group.channels.size() != group.peerRanks.size()
            || group.channels.size() >= param.rankSize
            || group.channels.size() > MAX_CHANNELS_PER_DIE) {
            HCCL_ERROR("[RegisterDirectKernels] Invalid channel group %u", groupIndex);
            hostFailure = HCCL_E_NOT_SUPPORT;
            break;
        }

        CcuKernelInfo kernelInfo{};
        const int nameLength = std::snprintf(
            kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s%u", KERNEL_NAME_PREFIX, group.dieId);
        if (nameLength < 0 || static_cast<size_t>(nameLength) >= sizeof(kernelInfo.kernelFuncName)) {
            HCCL_ERROR("[RegisterDirectKernels] Failed to set kernel name");
            hostFailure = HCCL_E_INTERNAL;
            break;
        }
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuPeerLanePullKernel);

        auto kernelArg = std::make_shared<CcuDirectKernelArg>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
            kernelArg->channels[channelIndex] = group.channels[channelIndex];
            kernelArg->peerRanks[channelIndex] = group.peerRanks[channelIndex];
        }
        retainedArgs.push_back(kernelArg);
        kernelInfo.setKernelArg(kernelArg);

        CcuKernelHandle kernelHandle{};
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        const CcuResult registerRet = HcommCcuKernelRegister(insHandle, group.dieId, kernelInfo.kernelFuncName,
            kernelInfo.kernelFunc, kernelArgs, KERNEL_ARG_NUM, &kernelHandle);
        if (registerRet != CCU_SUCCESS) {
            HCCL_ERROR("[RegisterDirectKernels] Die %u registration failed, ccuRet=%d", group.dieId, registerRet);
            registerFailure = registerRet;
            break;
        }
        kernelHandles.push_back(kernelHandle);
        peerRanksByGroup.push_back(group.peerRanks);
    }

    const CcuResult registerEndRet = HcommCcuKernelRegisterEnd(insHandle);
    if (hostFailure != HCCL_SUCCESS) {
        return hostFailure;
    }
    if (registerFailure != CCU_SUCCESS) {
        return ConvertCcuToHccl(registerFailure);
    }
    if (registerEndRet != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterDirectKernels] Kernel registration end failed, ccuRet=%d", registerEndRet);
        return ConvertCcuToHccl(registerEndRet);
    }

    resource.ccuKernels = std::move(kernelHandles);
    resource.peerRanksByGroup = std::move(peerRanksByGroup);
    resource.topologyType = static_cast<uint32_t>(topology);
    resource.algorithmMode = ALGORITHM_V10_PULL;
    return HCCL_SUCCESS;
}

HcclResult BuildResource(HcclComm comm, CommEngine engine, const OpParam &param, AlgResourceCtx &resource)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resource.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    resource.ccuThread = param.cpuThread;
    resource.threads = {param.cpuThread};

    if (param.rankSize == 1) {
        resource.topologyType = static_cast<uint32_t>(ops_hccl::TopologyType::SINGLE_RANK);
        return HCCL_SUCCESS;
    }

    std::vector<ChannelGroup> groups;
    const uint64_t rankBytes = param.count * sizeof(float);
    const bool output512 = IsOutput512Target(rankBytes, param.rankSize);

    if (param.rankSize == V22_TWO_BY_EIGHT_RANK_SIZE) {
        std::vector<V22PeerChannel> peerChannels;
        CHK_RET(V22AcquirePeerChannels(comm, engine, param.myRank, peerChannels));
        std::vector<V22ChannelGroup> v22Groups;
        CHK_RET(V22GroupChannelsByDie(peerChannels, param.myRank, v22Groups));

        if (output512) {
            CHK_PRT_RET(v22Groups.size() != 2U,
                HCCL_ERROR("[BuildResource] Invalid output-512 rank16 groups"),
                HCCL_E_INTERNAL);
            std::vector<ChannelGroup> exactGroups(2U);
            for (uint32_t i = 0U; i < 2U; ++i) {
                exactGroups[i].dieId =
                    static_cast<uint32_t>(v22Groups[i].dieId);
                exactGroups[i].channels = v22Groups[i].channels;
                exactGroups[i].peerRanks = v22Groups[i].peerRanks;
            }
            ThreadHandle crossThread{};
            CHK_RET(HcclThreadAcquire(
                comm, engine, 1U, THREAD_NOTIFY_NUM, &crossThread));
            resource.threads = {param.cpuThread, crossThread};
            CHK_RET(RegisterOutput512StaticKernels(
                comm, param, exactGroups, resource));
            return HCCL_SUCCESS;
        }

        if (rankBytes <= ops_hccl::INLINE_COPY_THRESHOLD_BYTES) {
            ThreadHandle crossThread{};
            CHK_RET(HcclThreadAcquire(
                comm, engine, 1U, THREAD_NOTIFY_NUM, &crossThread));
            resource.threads = {param.cpuThread, crossThread};
            CHK_RET(Register2x8V22Kernels(
                comm, param, v22Groups, true, resource));
            return HCCL_SUCCESS;
        }

        const bool bandwidthOptimized =
            rankBytes <= 2ULL * static_cast<uint64_t>(MAX_DATA_SIZE);
        const bool inlineTailCopy = bandwidthOptimized
            && rankBytes % ops_hccl::DMA_ALIGNMENT_BYTES != 0U;
        if (bandwidthOptimized && !inlineTailCopy) {
            ThreadHandle extraThreads[2]{};
            CHK_RET(HcclThreadAcquire(
                comm, engine, 2U, THREAD_NOTIFY_NUM + 1U, extraThreads));
            resource.threads = {param.cpuThread, extraThreads[0]};
            resource.ownCopyThread = extraThreads[1];
        } else {
            ThreadHandle crossThread{};
            CHK_RET(HcclThreadAcquire(
                comm, engine, 1U, THREAD_NOTIFY_NUM, &crossThread));
            resource.threads = {param.cpuThread, crossThread};
        }
        CHK_RET(Register2x8BandwidthKernels(
            comm, param, v22Groups, bandwidthOptimized, inlineTailCopy, resource));
        return HCCL_SUCCESS;
    }

    const bool latency4x1 = param.rankSize == 4U
        && rankBytes <= ops_hccl::INLINE_COPY_THRESHOLD_BYTES;

    if (param.rankSize == 4U) {
        if (!latency4x1) {
            bool dualPlaneSelected = false;
            CHK_RET(TryAcquire4x1DualPlaneChannels(
                comm, engine, param.myRank, groups, dualPlaneSelected));
            if (dualPlaneSelected) {
                ThreadHandle extraThreads[2]{};
                CHK_RET(HcclThreadAcquire(
                    comm, engine, 2U, THREAD_NOTIFY_NUM, extraThreads));
                resource.threads = {param.cpuThread, extraThreads[0]};
                resource.ownCopyThread = extraThreads[1];
                CHK_RET(Register4x1DualPlaneKernels(comm, param, groups, resource));
                return HCCL_SUCCESS;
            }
            groups.clear();
        }

        bool commonDieSelected = false;
        CHK_RET(TryAcquire4x1CommonDieChannels(
            comm, engine, param.myRank, groups, commonDieSelected));
        if (commonDieSelected) {
            uint32_t meshGroupIndex = 0U;
            CHK_PRT_RET(!Resolve4x1Group(param, groups, meshGroupIndex),
                HCCL_ERROR("[BuildResource] Invalid 4x1 common-Die group"), HCCL_E_INTERNAL);
            resource.threads = {param.cpuThread};
            if (output512) {
                const std::vector<ChannelGroup> exactGroups = {
                    groups[meshGroupIndex]};
                CHK_RET(RegisterOutput512StaticKernels(
                    comm, param, exactGroups, resource));
                return HCCL_SUCCESS;
            }
            if (!latency4x1) {
                ThreadHandle ownCopyThread{};
                CHK_RET(HcclThreadAcquire(
                    comm, engine, 1U, THREAD_NOTIFY_NUM, &ownCopyThread));
                resource.ownCopyThread = ownCopyThread;
            }
            CHK_RET(Register4x1CommonDieKernel(
                comm, param, groups[meshGroupIndex], latency4x1, resource));
            return HCCL_SUCCESS;
        }
        groups.clear();
    }

    const bool wide8p4 = param.rankSize == 12U
        && rankBytes > ops_hccl::INLINE_COPY_THRESHOLD_BYTES;
    CHK_RET(AcquireChannels(comm, engine, param.myRank, param.rankSize,
        wide8p4 ? WIDE_CHANNEL_NOTIFY_NUM : CHANNEL_NOTIFY_NUM, groups));
    const ops_hccl::TopologyType topology = DetectTopology(param, groups);

    uint32_t meshGroupIndex = 0U;
    if (topology == ops_hccl::TopologyType::MESH_4X1
        && Resolve4x1Group(param, groups, meshGroupIndex)) {
        resource.threads = {param.cpuThread};
        if (output512) {
            const std::vector<ChannelGroup> exactGroups = {
                groups[meshGroupIndex]};
            CHK_RET(RegisterOutput512StaticKernels(
                comm, param, exactGroups, resource));
            return HCCL_SUCCESS;
        }
        if (!latency4x1) {
            ThreadHandle ownCopyThread{};
            CHK_RET(HcclThreadAcquire(
                comm, engine, 1U, THREAD_NOTIFY_NUM, &ownCopyThread));
            resource.ownCopyThread = ownCopyThread;
        }
        CHK_RET(Register4x1CommonDieKernel(
            comm, param, groups[meshGroupIndex], latency4x1, resource));
        return HCCL_SUCCESS;
    }

    uint32_t intraGroupIndex = 0U;
    uint32_t crossGroupIndex = 0U;
    if (topology == ops_hccl::TopologyType::ASYMMETRIC_8P4
        && Resolve8p4Groups(param, groups, intraGroupIndex, crossGroupIndex)) {
        if (output512) {
            ThreadHandle crossThread{};
            CHK_RET(HcclThreadAcquire(
                comm, engine, 1U, THREAD_NOTIFY_NUM, &crossThread));
            resource.threads = {param.cpuThread, crossThread};
            const std::vector<ChannelGroup> exactGroups = param.myRank >= 8U
                ? std::vector<ChannelGroup>{
                    groups[crossGroupIndex], groups[intraGroupIndex]}
                : std::vector<ChannelGroup>{
                    groups[intraGroupIndex], groups[crossGroupIndex]};
            CHK_RET(RegisterOutput512StaticKernels(
                comm, param, exactGroups, resource));
            return HCCL_SUCCESS;
        }
        if (wide8p4) {
            ThreadHandle extraThreads[2]{};
            CHK_RET(HcclThreadAcquire(
                comm, engine, 2U, THREAD_NOTIFY_NUM + 1U, extraThreads));
            resource.threads = {param.cpuThread, extraThreads[0]};
            resource.ownCopyThread = extraThreads[1];
            CHK_RET(Register8p4WideKernels(
                comm, param, groups[intraGroupIndex], groups[crossGroupIndex], resource));
        } else {
            ThreadHandle crossThread{};
            CHK_RET(HcclThreadAcquire(
                comm, engine, 1U, THREAD_NOTIFY_NUM, &crossThread));
            resource.threads = {param.cpuThread, crossThread};
            CHK_RET(Register8p4LatencyKernels(
                comm, param, groups[intraGroupIndex], groups[crossGroupIndex], resource));
        }
        return HCCL_SUCCESS;
    }

    std::vector<ThreadHandle> acquiredThreads(groups.size());
    CHK_RET(HcclThreadAcquire(
        comm, engine, static_cast<uint32_t>(acquiredThreads.size()),
        THREAD_NOTIFY_NUM, acquiredThreads.data()));
    resource.threads.resize(groups.size());
    resource.threads[0] = param.cpuThread;
    for (uint32_t groupIndex = 1; groupIndex < groups.size(); ++groupIndex) {
        resource.threads[groupIndex] = acquiredThreads[groupIndex - 1];
    }
    resource.ownCopyThread = acquiredThreads.back();

    const ops_hccl::TopologyType fallbackTopology =
        topology == ops_hccl::TopologyType::MESH_4X1
            || topology == ops_hccl::TopologyType::ASYMMETRIC_8P4
        ? ops_hccl::TopologyType::GENERIC
        : topology;
    CHK_RET(RegisterDirectKernels(comm, param, groups, fallbackTopology, resource));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    // Every valid cache entry was created only after the public API checks
    // below.  Match the full session signature first and keep those invariant
    // checks off the steady-state 512 KiB launch path.
    const bool output512FastHit = g_output512FastCache.valid
        && g_output512FastCache.comm == comm
        && g_output512FastCache.stream == stream
        && g_output512FastCache.inputPtr == sendBuf
        && g_output512FastCache.outputPtr == recvBuf
        && g_output512FastCache.sendCount == sendCount
        && g_output512FastCache.dataType == dataType;
    if (output512FastHit) {
        return LaunchOutput512WarmFast(g_output512FastCache);
    }

    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[HcclAllGather] Only HCCL_DATA_TYPE_FP32 is supported"), HCCL_E_NOT_SUPPORT);

    if (g_output512FastCache.valid
        && g_output512FastCache.comm == comm) {
        g_output512FastCache = Output512FastCache{};
    }

    // 构造算子参数
    OpParam param{};
    const int tagLength = std::snprintf(param.tag, sizeof(param.tag), "%s", ENGINE_CONTEXT_TAG);
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("[HcclAllGather] Failed to set operation tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    const bool exact512Count =
        sendCount == EXACT_512_PAYLOAD_BYTES / sizeof(float);
    const bool exact512FastHit = exact512Count
        && g_4x1Exact512FastCache.comm == comm
        && g_4x1Exact512FastCache.stream == stream
        && g_4x1Exact512FastCache.inputPtr == sendBuf
        && g_4x1Exact512FastCache.outputPtr == recvBuf
        && g_4x1Exact512FastCache.context != nullptr;
    if (exact512FastHit) {
        param.myRank = g_4x1Exact512FastCache.rankId;
        param.rankSize = 4U;
        param.cpuThread = g_4x1Exact512FastCache.mainThread;
        param.resCtx = g_4x1Exact512FastCache.context;
        param.ctxSize = g_4x1Exact512FastCache.contextSize;
        return ops_hccl::Exec4x1Exact512WarmOp(param);
    }
    // A different buffer or stream may republish the channel XN variables.
    // Invalidate the old warm identity so it can never be reused after that.
    if (g_4x1Exact512FastCache.context != nullptr
        && g_4x1Exact512FastCache.comm == comm) {
        g_4x1Exact512FastCache = FourByOneExact512FastCache{};
    }

    // 注册算子信息
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[HcclAllGather] Invalid rank size %u", param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.myRank >= param.rankSize,
        HCCL_ERROR("[HcclAllGather] Invalid rank id %u for rank size %u", param.myRank, param.rankSize), HCCL_E_PARA);

    constexpr uint64_t dataTypeSize = sizeof(float);
    CHK_PRT_RET(sendCount > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("[HcclAllGather] Input byte size overflows uint64_t"), HCCL_E_PARA);
    const uint64_t rankBytes = sendCount * dataTypeSize;
    CHK_PRT_RET(rankBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("[HcclAllGather] Output byte size overflows uint64_t"), HCCL_E_PARA);
    const bool output512 = IsOutput512Target(rankBytes, param.rankSize);

    if (param.rankSize == V22_TWO_BY_EIGHT_RANK_SIZE) {
        const char *twoByEightTag =
            output512 ? ENGINE_CONTEXT_TAG_OUTPUT_512_RANK16
            : rankBytes <= ops_hccl::INLINE_COPY_THRESHOLD_BYTES
            ? ENGINE_CONTEXT_TAG_2X8_LATENCY
            : ENGINE_CONTEXT_TAG_2X8_BANDWIDTH;
        const int twoByEightTagLength = std::snprintf(
            param.tag, sizeof(param.tag), "%s", twoByEightTag);
        CHK_PRT_RET(twoByEightTagLength < 0
                || static_cast<size_t>(twoByEightTagLength) >= sizeof(param.tag),
            HCCL_ERROR("[HcclAllGather] Failed to set 2x8 V22 operation tag"),
            HCCL_E_INTERNAL);
    }

    // 4x1低时延和带宽路径分别缓存Engine上下文。这样每个Peer在任一上下文中
    // 仍只申请一条Channel，同时低时延上下文不需要额外注册两个双Die Kernel。
    // 同一通信域后续切换消息档位时会命中各自稳定的资源集合。
    if (param.rankSize == 4U) {
        const char *fourRankTag =
            output512 ? ENGINE_CONTEXT_TAG_OUTPUT_512_RANK4
            : rankBytes == EXACT_512_PAYLOAD_BYTES
            ? ENGINE_CONTEXT_TAG_4X1_EXACT_512
            : rankBytes <= ops_hccl::INLINE_COPY_THRESHOLD_BYTES
                ? ENGINE_CONTEXT_TAG_4X1_LATENCY
                : ENGINE_CONTEXT_TAG_4X1_BANDWIDTH;
        const int fourRankTagLength =
            std::snprintf(param.tag, sizeof(param.tag), "%s", fourRankTag);
        CHK_PRT_RET(fourRankTagLength < 0
                || static_cast<size_t>(fourRankTagLength) >= sizeof(param.tag),
            HCCL_ERROR("[HcclAllGather] Failed to set 4x1 operation tag"), HCCL_E_INTERNAL);
    }

    if (param.rankSize == 12U) {
        const char *eightPlusFourTag =
            output512 ? ENGINE_CONTEXT_TAG_OUTPUT_512_RANK12
            : rankBytes <= ops_hccl::INLINE_COPY_THRESHOLD_BYTES
            ? ENGINE_CONTEXT_TAG_8P4_LATENCY
            : ENGINE_CONTEXT_TAG_8P4_BANDWIDTH;
        const int eightPlusFourTagLength =
            std::snprintf(param.tag, sizeof(param.tag), "%s", eightPlusFourTag);
        CHK_PRT_RET(eightPlusFourTagLength < 0
                || static_cast<size_t>(eightPlusFourTagLength) >= sizeof(param.tag),
            HCCL_ERROR("[HcclAllGather] Failed to set 8+4 operation tag"),
            HCCL_E_INTERNAL);
    }

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 每次调用都绑定当前 stream；EngineCtx 中只缓存通信域生命周期内稳定的资源。
    // 2x8带宽路径和8+4带宽路径会把OwnCopy完成信号记录到
    // DEFERRED_OWN_COPY_DONE_NOTIFY_ID(2)，因此主stream thread必须拥有
    // notify索引0、1、2。低时延路径继续只申请2个notify，避免改变
    // 点10、13、16已验证的V27执行资源。
    const bool externalCopy2x8 =
        param.rankSize == V22_TWO_BY_EIGHT_RANK_SIZE
        && rankBytes >= ops_hccl::INLINE_COPY_THRESHOLD_BYTES
        && rankBytes <= 2ULL * static_cast<uint64_t>(MAX_DATA_SIZE)
        && rankBytes % ops_hccl::DMA_ALIGNMENT_BYTES == 0U;
    const bool needsDeferredCopyDoneNotify =
        externalCopy2x8
        || (param.rankSize == 12U
            && rankBytes > ops_hccl::INLINE_COPY_THRESHOLD_BYTES);
    const uint32_t streamThreadNotifyNum =
        needsDeferredCopyDoneNotify ? THREAD_NOTIFY_NUM + 1U : THREAD_NOTIFY_NUM;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, ccuEngine, stream, streamThreadNotifyNum, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    bool createdContext = false;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // CCU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost{};
        CHK_RET(BuildResource(comm, ccuEngine, param, resCtxHost));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 CCU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
        createdContext = true;
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param));
    if (output512) {
        std::vector<char> cachedSeq(
            static_cast<char *>(param.resCtx),
            static_cast<char *>(param.resCtx) + param.ctxSize);
        AlgResourceCtx cachedResource{};
        cachedResource.DeSerialize(cachedSeq);
        const uint32_t groupCount =
            static_cast<uint32_t>(cachedResource.threads.size());
        uint64_t liveInputToken = 0U;
        uint64_t liveOutputToken = 0U;
        if (createdContext) {
            liveInputToken = cachedResource.output512InputToken;
            liveOutputToken = cachedResource.output512OutputToken;
        } else {
            CHK_RET_CCU(HcommCcuGetMemToken(
                reinterpret_cast<uint64_t>(sendBuf), rankBytes,
                &liveInputToken));
            CHK_RET_CCU(HcommCcuGetMemToken(
                reinterpret_cast<uint64_t>(recvBuf),
                rankBytes * param.rankSize, &liveOutputToken));
        }
        const bool cacheable =
            cachedResource.algorithmMode == ALGORITHM_OUTPUT_512_STATIC
            && cachedResource.hasOutput512StaticSession != 0U
            && groupCount > 0U && groupCount <= 2U
            && cachedResource.ccuKernels.size() == groupCount
            && cachedResource.output512InputAddr
                == reinterpret_cast<uint64_t>(sendBuf)
            && cachedResource.output512OutputAddr
                == reinterpret_cast<uint64_t>(recvBuf)
            && cachedResource.output512InputToken == liveInputToken
            && cachedResource.output512OutputToken == liveOutputToken
            && cachedResource.output512RankBytes == rankBytes;
        if (cacheable) {
            Output512FastCache cache{};
            cache.comm = comm;
            cache.stream = stream;
            cache.inputPtr = sendBuf;
            cache.outputPtr = recvBuf;
            cache.sendCount = sendCount;
            cache.dataType = dataType;
            cache.rankId = param.myRank;
            cache.rankSize = param.rankSize;
            cache.groupCount = groupCount;
            cache.context = param.resCtx;
            cache.inputAddr = reinterpret_cast<uint64_t>(sendBuf);
            cache.outputAddr = reinterpret_cast<uint64_t>(recvBuf);
            cache.inputToken = liveInputToken;
            cache.outputToken = liveOutputToken;
            cache.rankOutputOffset =
                static_cast<uint64_t>(param.myRank) * rankBytes;
            cache.transferBytes = rankBytes;
            for (uint32_t i = 0U; i < groupCount; ++i) {
                cache.threads[i] =
                    i == 0U ? param.cpuThread : cachedResource.threads[i];
                cache.warmKernels[i] =
                    cachedResource.ccuKernels[i];
            }
            cache.valid = true;
            g_output512FastCache = cache;
        }
    }
    if (createdContext && param.rankSize == 4U
        && rankBytes == EXACT_512_PAYLOAD_BYTES) {
        g_4x1Exact512FastCache.comm = comm;
        g_4x1Exact512FastCache.stream = stream;
        g_4x1Exact512FastCache.mainThread = param.cpuThread;
        g_4x1Exact512FastCache.context = param.resCtx;
        g_4x1Exact512FastCache.contextSize = param.ctxSize;
        g_4x1Exact512FastCache.rankId = param.myRank;
        g_4x1Exact512FastCache.inputPtr = sendBuf;
        g_4x1Exact512FastCache.outputPtr = recvBuf;
    }
    return HCCL_SUCCESS;
}
