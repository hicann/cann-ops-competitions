/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
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
constexpr uint32_t NET_LAYER_INTRA_SERVER = 0;
constexpr uint32_t NET_LAYER_INTER_SERVER = 1;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;

void AppendUnique(std::vector<uint32_t> &ranks, uint32_t rank)
{
    for (uint32_t current : ranks) {
        if (current == rank) {
            return;
        }
    }
    ranks.push_back(rank);
}

uint32_t LargestPowerOfTwo(uint32_t value)
{
    uint32_t result = 1;
    while ((result << 1U) <= value) {
        result <<= 1U;
    }
    return result;
}

uint32_t ActiveRankToOriginal(uint32_t activeRank, uint32_t remainder)
{
    return activeRank < remainder ? activeRank * 2U + 1U : activeRank + remainder;
}

HcclResult BuildRecursiveDoublingPeers(const OpParam &param, std::vector<uint32_t> &peers)
{
    const uint32_t powerOfTwo = LargestPowerOfTwo(param.rankSize);
    const uint32_t remainder = param.rankSize - powerOfTwo;
    uint32_t activeRank = 0;

    if (remainder != 0 && param.myRank < 2U * remainder) {
        const uint32_t pairRank = (param.myRank & 1U) == 0 ? param.myRank + 1U : param.myRank - 1U;
        AppendUnique(peers, pairRank);
        if ((param.myRank & 1U) == 0) {
            return HCCL_SUCCESS;
        }
        activeRank = param.myRank / 2U;
    } else {
        activeRank = param.myRank - remainder;
    }

    for (uint32_t distance = 1; distance < powerOfTwo; distance <<= 1U) {
        const uint32_t peerActiveRank = activeRank ^ distance;
        AppendUnique(peers, ActiveRankToOriginal(peerActiveRank, remainder));
    }
    return HCCL_SUCCESS;
}

HcclResult BuildHighRadixPeers(const OpParam &param, std::vector<uint32_t> &peers)
{
    if (param.rankSize % HIGH_RADIX != 0) {
        return HCCL_E_PARA;
    }

    const uint32_t localDigit = param.myRank % HIGH_RADIX;
    const uint32_t groupBase = param.myRank - localDigit;
    for (uint32_t digit = 0; digit < HIGH_RADIX; ++digit) {
        const uint32_t peerRank = groupBase + digit;
        if (peerRank != param.myRank) {
            AppendUnique(peers, peerRank);
        }
    }

    const uint32_t highGroupCount = param.rankSize / HIGH_RADIX;
    for (uint32_t highDigit = 0; highDigit < highGroupCount; ++highDigit) {
        const uint32_t peerRank = highDigit * HIGH_RADIX + localDigit;
        if (peerRank != param.myRank) {
            AppendUnique(peers, peerRank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BuildRequiredPeers(const OpParam &param, AllReduceAlgorithm algorithm, std::vector<uint32_t> &peers)
{
    if (algorithm == AllReduceAlgorithm::CLOS_FULL_LANE_FLAT_OWNER_RSAG) {
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.myRank) {
                peers.push_back(rank);
            }
        }
        return HCCL_SUCCESS;
    }
    if (algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_DOUBLING ||
        algorithm == AllReduceAlgorithm::CLOS_RADIX4X3_RSAG ||
        algorithm == AllReduceAlgorithm::CLOS_RADIX4_RSAG) {
        return BuildHighRadixPeers(param, peers);
    }
    if (algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_HALVING_DOUBLING) {
        return BuildRecursiveDoublingPeers(param, peers);
    }

    const uint32_t previousRank = (param.myRank + param.rankSize - 1U) % param.rankSize;
    const uint32_t nextRank = (param.myRank + 1U) % param.rankSize;
    AppendUnique(peers, previousRank);
    AppendUnique(peers, nextRank);
    return HCCL_SUCCESS;
}

HcclResult FillChannelDesc(const CommLink &link, uint32_t remoteRank, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult GetChannelDieId(HcclComm comm, uint32_t rankId, const HcclChannelDesc &desc, uint32_t &dieId)
{
    EndpointAttrDieId endpointDieId = 0;
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, rankId, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID,
        sizeof(endpointDieId), &endpointDieId));
    dieId = endpointDieId;
    return HCCL_SUCCESS;
}

struct LayerChannelCandidate {
    uint32_t dieId = 0;
    HcclChannelDesc desc {};
};

HcclResult GetLayerChannelCandidates(HcclComm comm, uint32_t netLayer, CommProtocol protocol,
    uint32_t srcRank, uint32_t dstRank, std::vector<LayerChannelCandidate> &candidates)
{
    CommLink *linkList = nullptr;
    uint32_t listSize = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize));
    for (uint32_t idx = 0; idx < listSize; ++idx) {
        if (linkList[idx].linkAttr.linkProtocol != protocol) {
            continue;
        }
        LayerChannelCandidate candidate;
        CHK_RET(FillChannelDesc(linkList[idx], dstRank, candidate.desc));
        CHK_RET(GetChannelDieId(comm, srcRank, candidate.desc, candidate.dieId));
        candidates.push_back(candidate);
    }
    CHK_PRT_RET(candidates.empty(),
        HCCL_ERROR("No layer[%u] protocol[%u] link between rank[%u] and rank[%u]",
            netLayer, static_cast<uint32_t>(protocol), srcRank, dstRank),
        HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

HcclResult BuildLayerChannels(HcclComm comm, const OpParam &param, uint32_t netLayer, CommProtocol protocol,
    const std::vector<uint32_t> &peers, std::vector<HcclChannelDesc> &descs, uint32_t &dieId)
{
    CHK_PRT_RET(peers.empty() || peers.size() > MAX_ALG_CHANNEL_NUM,
        HCCL_ERROR("Invalid layer[%u] peer count[%zu]", netLayer, peers.size()), HCCL_E_INTERNAL);

    std::vector<std::vector<LayerChannelCandidate>> candidates(peers.size());
    for (uint32_t idx = 0; idx < peers.size(); ++idx) {
        CHK_RET(GetLayerChannelCandidates(comm, netLayer, protocol, param.myRank, peers[idx], candidates[idx]));
    }

    bool commonDieFound = false;
    for (const auto &firstCandidate : candidates[0]) {
        bool availableForAllPeers = true;
        for (uint32_t peerIndex = 1; peerIndex < candidates.size(); ++peerIndex) {
            bool availableForPeer = false;
            for (const auto &candidate : candidates[peerIndex]) {
                if (candidate.dieId == firstCandidate.dieId) {
                    availableForPeer = true;
                    break;
                }
            }
            if (!availableForPeer) {
                availableForAllPeers = false;
                break;
            }
        }
        if (availableForAllPeers) {
            dieId = firstCandidate.dieId;
            commonDieFound = true;
            break;
        }
    }
    CHK_PRT_RET(!commonDieFound,
        HCCL_ERROR("No common CCU die for layer[%u] channels of rank[%u]", netLayer, param.myRank),
        HCCL_E_INTERNAL);

    descs.resize(peers.size());
    for (uint32_t peerIndex = 0; peerIndex < peers.size(); ++peerIndex) {
        bool descSelected = false;
        for (const auto &candidate : candidates[peerIndex]) {
            if (candidate.dieId == dieId) {
                descs[peerIndex] = candidate.desc;
                descSelected = true;
                break;
            }
        }
        CHK_PRT_RET(!descSelected,
            HCCL_ERROR("Failed to select layer[%u] channel on die[%u] for peerRank[%u]",
                netLayer, dieId, peers[peerIndex]),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult GetLocalRankMask(HcclComm comm, const OpParam &param,
    std::vector<bool> &isLocal, uint32_t &localRankCount)
{
    uint32_t *layerRanks = nullptr;
    localRankCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, NET_LAYER_INTRA_SERVER, &layerRanks, &localRankCount));
    CHK_PRT_RET(layerRanks == nullptr || localRankCount == 0U,
        HCCL_ERROR("Layer 0 rank instance is empty for rank[%u]", param.myRank), HCCL_E_INTERNAL);

    isLocal.assign(param.rankSize, false);
    for (uint32_t idx = 0; idx < localRankCount; ++idx) {
        const uint32_t rank = layerRanks[idx];
        CHK_PRT_RET(rank >= param.rankSize,
            HCCL_ERROR("Invalid Layer 0 rank[%u] for rankSize[%u]", rank, param.rankSize), HCCL_E_INTERNAL);
        isLocal[rank] = true;
    }
    CHK_PRT_RET(!isLocal[param.myRank],
        HCCL_ERROR("Layer 0 rank instance does not contain myRank[%u]", param.myRank), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

bool IsTopologyAware(AllReduceAlgorithm algorithm, uint32_t rankSize)
{
    return (algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_HALVING_DOUBLING && rankSize == 16U) ||
        (algorithm == AllReduceAlgorithm::CLOS_RADIX4X3_RSAG && rankSize == 12U) ||
        (algorithm == AllReduceAlgorithm::CLOS_FULL_LANE_FLAT_OWNER_RSAG &&
            (rankSize == 12U || rankSize == 16U));
}

HcclResult BuildTopologyPhasePeers(HcclComm comm, const OpParam &param, AllReduceAlgorithm algorithm,
    std::vector<uint32_t> &localPeers, std::vector<uint32_t> &crossPeers)
{
    std::vector<bool> isLocal;
    uint32_t localRankCount = 0;
    CHK_RET(GetLocalRankMask(comm, param, isLocal, localRankCount));

    if (algorithm == AllReduceAlgorithm::CLOS_FULL_LANE_FLAT_OWNER_RSAG) {
        const bool validLocalRankCount = param.rankSize == 16U ? localRankCount == 8U :
            (param.rankSize == 12U && (localRankCount == 4U || localRankCount == 8U));
        CHK_PRT_RET(!validLocalRankCount,
            HCCL_ERROR("Unexpected Flat Owner topology, rankSize[%u], localRankCount[%u]",
                param.rankSize, localRankCount), HCCL_E_INTERNAL);
        for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
            if (peerRank == param.myRank) {
                continue;
            }
            if (isLocal[peerRank]) {
                localPeers.push_back(peerRank);
            } else {
                crossPeers.push_back(peerRank);
            }
        }
        CHK_PRT_RET(localPeers.empty() || crossPeers.empty() ||
            localPeers.size() + crossPeers.size() != param.rankSize - 1U,
            HCCL_ERROR("Invalid Flat Owner peer split for rank[%u]: local[%zu], cross[%zu], rankSize[%u]",
                param.myRank, localPeers.size(), crossPeers.size(), param.rankSize), HCCL_E_INTERNAL);
        if (param.rankSize == 12U) {
            const bool isEightPlusFour =
                (localPeers.size() == 7U && crossPeers.size() == 4U) ||
                (localPeers.size() == 3U && crossPeers.size() == 8U);
            CHK_PRT_RET(!isEightPlusFour,
                HCCL_ERROR("12-rank Flat Owner requires an 8+4 topology, rank[%u], local[%zu], cross[%zu]",
                    param.myRank, localPeers.size(), crossPeers.size()), HCCL_E_INTERNAL);
        }
        return HCCL_SUCCESS;
    }

    if (algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_HALVING_DOUBLING) {
        CHK_PRT_RET(param.rankSize != 16U || localRankCount != 8U,
            HCCL_ERROR("Unexpected RH+D topology, rankSize[%u], localRankCount[%u]",
                param.rankSize, localRankCount), HCCL_E_INTERNAL);
        for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
            if (peerRank != param.myRank && isLocal[peerRank]) {
                localPeers.push_back(peerRank);
            }
        }
        CHK_PRT_RET(localPeers.size() != 7U,
            HCCL_ERROR("Unexpected RH+D local peer count[%zu] for rank[%u]", localPeers.size(), param.myRank),
            HCCL_E_INTERNAL);
        const uint32_t crossRank = param.myRank ^ 8U;
        CHK_PRT_RET(isLocal[crossRank],
            HCCL_ERROR("RH+D cross peer[%u] unexpectedly belongs to rank[%u] Layer 0 instance",
                crossRank, param.myRank), HCCL_E_INTERNAL);
        crossPeers.push_back(crossRank);
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(algorithm != AllReduceAlgorithm::CLOS_RADIX4X3_RSAG || param.rankSize != 12U ||
        (localRankCount != 8U && localRankCount != 4U),
        HCCL_ERROR("Unexpected radix-4x3 topology, rankSize[%u], localRankCount[%u]",
            param.rankSize, localRankCount), HCCL_E_INTERNAL);

    const uint32_t lowPosition = param.myRank % HIGH_RADIX;
    const uint32_t lowGroupBase = param.myRank - lowPosition;
    for (uint32_t position = 0; position < HIGH_RADIX; ++position) {
        const uint32_t peerRank = lowGroupBase + position;
        if (peerRank == param.myRank) {
            continue;
        }
        CHK_PRT_RET(!isLocal[peerRank],
            HCCL_ERROR("Radix-4 local-group peer[%u] is not in rank[%u] Layer 0 instance",
                peerRank, param.myRank), HCCL_E_INTERNAL);
        AppendUnique(localPeers, peerRank);
    }

    for (uint32_t highPosition = 0; highPosition < 3U; ++highPosition) {
        const uint32_t peerRank = highPosition * HIGH_RADIX + lowPosition;
        if (peerRank == param.myRank) {
            continue;
        }
        if (isLocal[peerRank]) {
            AppendUnique(localPeers, peerRank);
        } else {
            AppendUnique(crossPeers, peerRank);
        }
    }
    const uint32_t expectedCrossPeers = localRankCount == 8U ? 1U : 2U;
    CHK_PRT_RET(crossPeers.size() != expectedCrossPeers,
        HCCL_ERROR("Unexpected radix-4x3 cross peer count[%zu] for rank[%u] localRankCount[%u]",
            crossPeers.size(), param.myRank, localRankCount), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult BuildClosChannels(HcclComm comm, const OpParam &param, AllReduceAlgorithm algorithm,
    std::vector<uint32_t> &peers, std::vector<HcclChannelDesc> &descs, uint32_t &dieId)
{
    CHK_RET(BuildRequiredPeers(param, algorithm, peers));
    return BuildLayerChannels(comm, param, NET_LAYER_INTER_SERVER,
        CommProtocol::COMM_PROTOCOL_UBC_CTP, peers, descs, dieId);
}

void BuildSegmentLayout(const OpParam &param, uint32_t segmentCount, AllReduceKernelArg &arg)
{
    const uint64_t baseElements = param.count / segmentCount;
    const uint64_t remainder = param.count % segmentCount;
    uint64_t offsetElements = 0;
    for (uint32_t segment = 0; segment < segmentCount; ++segment) {
        const uint64_t elements = baseElements + (segment < remainder ? 1ULL : 0ULL);
        arg.segmentOffsets[segment] = offsetElements * sizeof(float);
        arg.segmentBytes[segment] = elements * sizeof(float);
        offsetElements += elements;
    }
}

std::shared_ptr<AllReduceKernelArg> BuildKernelArg(const OpParam &param, AllReduceAlgorithm algorithm,
    TopologyPhase topologyPhase, uint64_t dataBytes, const std::vector<uint32_t> &peers,
    const std::vector<ChannelHandle> &channels)
{
    auto kernelArg = std::make_shared<AllReduceKernelArg>();
    kernelArg->rankId = param.myRank;
    kernelArg->rankSize = param.rankSize;
    kernelArg->algorithm = algorithm;
    kernelArg->topologyPhase = topologyPhase;
    kernelArg->pipelineDepth = SelectEqualPipelineDepth(param.rankSize, dataBytes);
    kernelArg->dataBytes = dataBytes;
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        kernelArg->channels[idx] = channels[idx];
        kernelArg->remoteRanks[idx] = peers[idx];
    }
    if (algorithm == AllReduceAlgorithm::CLOS_RING_RSAG ||
        algorithm == AllReduceAlgorithm::CLOS_RADIX4X3_RSAG ||
        algorithm == AllReduceAlgorithm::CLOS_RADIX4_RSAG ||
        algorithm == AllReduceAlgorithm::CLOS_FULL_LANE_FLAT_OWNER_RSAG) {
        BuildSegmentLayout(param, param.rankSize, *kernelArg);
    } else if (algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_HALVING_DOUBLING) {
        BuildSegmentLayout(param, LargestPowerOfTwo(param.rankSize), *kernelArg);
    }
    return kernelArg;
}

HcclResult GetCcuInstructionHandle(HcclComm comm, CcuInsHandle &insHandle)
{
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Expected one CCU instruction instance, got[%u]", insNum),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RegisterOneKernel(CcuInsHandle insHandle, uint32_t dieId, const char *name,
    const std::shared_ptr<AllReduceKernelArg> &kernelArg, CcuKernelHandle &kernel)
{
    const void *registerArgs[] = {kernelArg.get()};
    const CcuResult ccuRet = HcommCcuKernelRegister(insHandle, dieId, name,
        reinterpret_cast<const void *>(ops_hccl::CcuKernel), registerArgs, 1, &kernel);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("Register CCU kernel[%s] on die[%u] failed, ret[%d]", name, dieId, ccuRet),
        ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult RegisterKernel(HcclComm comm, const OpParam &param, AllReduceAlgorithm algorithm,
    uint64_t dataBytes, uint32_t dieId, const std::vector<uint32_t> &peers,
    const std::vector<ChannelHandle> &channels, AlgResourceCtx &resource)
{
    CcuInsHandle insHandle = 0;
    CHK_RET(GetCcuInstructionHandle(comm, insHandle));
    auto kernelArg = BuildKernelArg(param, algorithm, TopologyPhase::FULL_CLOS, dataBytes, peers, channels);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register start failed, ret[%d]", ccuRet),
        ConvertCcuToHccl(ccuRet));
    CHK_RET(RegisterOneKernel(insHandle, dieId, "CcuKernel", kernelArg, resource.kernel));
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register end failed, ret[%d]", ccuRet),
        ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult RegisterTopologyAwareKernels(HcclComm comm, const OpParam &param,
    AllReduceAlgorithm algorithm, uint64_t dataBytes, uint32_t localDieId,
    const std::vector<uint32_t> &localPeers, const std::vector<ChannelHandle> &localChannels,
    uint32_t crossDieId, const std::vector<uint32_t> &crossPeers,
    const std::vector<ChannelHandle> &crossChannels, AlgResourceCtx &resource)
{
    const TopologyPhase crossPhase = algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_HALVING_DOUBLING ?
        TopologyPhase::RHD_CROSS_REDUCE : TopologyPhase::RADIX_CROSS_REDUCE_GATHER;
    auto crossArg = BuildKernelArg(param, algorithm, crossPhase, dataBytes, crossPeers, crossChannels);

    CcuInsHandle insHandle = 0;
    CHK_RET(GetCcuInstructionHandle(comm, insHandle));
    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register start failed, ret[%d]", ccuRet),
        ConvertCcuToHccl(ccuRet));

    if (algorithm == AllReduceAlgorithm::CLOS_RECURSIVE_HALVING_DOUBLING) {
        auto reduceArg = BuildKernelArg(param, algorithm, TopologyPhase::RHD_LOCAL_REDUCE,
            dataBytes, localPeers, localChannels);
        auto gatherArg = BuildKernelArg(param, algorithm, TopologyPhase::RHD_LOCAL_GATHER,
            dataBytes, localPeers, localChannels);

        CHK_RET(RegisterOneKernel(insHandle, localDieId, "CcuTopologyLocalLoopReduce", reduceArg,
            resource.phase0Kernel));
        CHK_RET(RegisterOneKernel(insHandle, crossDieId, "CcuTopologyCross", crossArg,
            resource.phase1Kernel));
        CHK_RET(RegisterOneKernel(insHandle, localDieId, "CcuTopologyLocalGather", gatherArg,
            resource.phase2Kernel));
    } else {
        auto localArg = BuildKernelArg(param, algorithm, TopologyPhase::RADIX_LOCAL_SHARED,
            dataBytes, localPeers, localChannels);
        CHK_RET(RegisterOneKernel(insHandle, localDieId, "CcuTopologyLocalShared", localArg,
            resource.phase0Kernel));
        CHK_RET(RegisterOneKernel(insHandle, crossDieId, "CcuTopologyCross", crossArg,
            resource.phase1Kernel));
        resource.phase2Kernel = resource.phase0Kernel;
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register end failed, ret[%d]", ccuRet),
        ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult CreateResources(HcclComm comm, aclrtStream stream, const OpParam &param,
    AllReduceAlgorithm algorithm, uint64_t dataBytes, AlgResourceCtx &resource)
{
    void *buffer = nullptr;
    uint64_t bufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &buffer, &bufferSize));
    resource.localBuffer = {buffer, bufferSize};
    resource.algorithm = algorithm;
    resource.rankSize = param.rankSize;
    resource.dataBytes = dataBytes;

    const uint32_t pipelineDepth = SelectEqualPipelineDepth(param.rankSize, dataBytes);
    const bool pipelineFlatOwner =
        algorithm == AllReduceAlgorithm::CLOS_FULL_LANE_FLAT_OWNER_RSAG &&
        (param.rankSize == 12U || param.rankSize == 16U) && pipelineDepth > 1U;
    const uint32_t threadNotifyNum = pipelineFlatOwner ? pipelineDepth : 1U;
    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, threadNotifyNum,
        &resource.thread));
    if (IsTopologyAware(algorithm, param.rankSize)) {
        std::vector<uint32_t> localPeers;
        std::vector<uint32_t> crossPeers;
        CHK_RET(BuildTopologyPhasePeers(comm, param, algorithm, localPeers, crossPeers));

        std::vector<HcclChannelDesc> localDescs;
        std::vector<HcclChannelDesc> crossDescs;
        uint32_t localDieId = 0;
        uint32_t crossDieId = 0;
        CHK_RET(BuildLayerChannels(comm, param, NET_LAYER_INTRA_SERVER,
            CommProtocol::COMM_PROTOCOL_UBC_CTP, localPeers, localDescs, localDieId));
        CHK_RET(BuildLayerChannels(comm, param, NET_LAYER_INTER_SERVER,
            CommProtocol::COMM_PROTOCOL_UBC_CTP, crossPeers, crossDescs, crossDieId));

        ThreadHandle slaveThreads[2] {};
        const uint32_t slaveThreadNum = pipelineFlatOwner ? 2U : 1U;
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, slaveThreadNum,
            threadNotifyNum, slaveThreads));
        resource.slaveThread = slaveThreads[0];
        if (pipelineFlatOwner) {
            resource.gatherThread = slaveThreads[1];
        }
        std::vector<ChannelHandle> localChannels(localDescs.size());
        std::vector<ChannelHandle> crossChannels(crossDescs.size());
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, localDescs.data(),
            static_cast<uint32_t>(localDescs.size()), localChannels.data()));
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, crossDescs.data(),
            static_cast<uint32_t>(crossDescs.size()), crossChannels.data()));
        CHK_RET(RegisterTopologyAwareKernels(comm, param, algorithm, dataBytes, localDieId,
            localPeers, localChannels, crossDieId, crossPeers, crossChannels, resource));
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> peers;
    std::vector<HcclChannelDesc> descs;
    uint32_t dieId = 0;
    CHK_RET(BuildClosChannels(comm, param, algorithm, peers, descs, dieId));
    std::vector<ChannelHandle> channels(descs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, descs.data(),
        static_cast<uint32_t>(descs.size()), channels.data()));
    CHK_RET(RegisterKernel(comm, param, algorithm, dataBytes, dieId, peers, channels, resource));
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateResources(HcclComm comm, aclrtStream stream, const OpParam &param,
    AllReduceAlgorithm algorithm, uint64_t dataBytes, AlgResourceCtx &resource)
{
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize) == HCCL_SUCCESS) {
        CHK_PTR_NULL(ctx);
        CHK_PRT_RET(ctxSize == 0, HCCL_ERROR("Invalid cached CCU resource context"), HCCL_E_INTERNAL);
        auto *ctxBytes = static_cast<char *>(ctx);
        std::vector<char> serialized(ctxBytes, ctxBytes + ctxSize);
        resource.DeSerialize(serialized);
        const bool topologyAware = IsTopologyAware(algorithm, param.rankSize);
        const uint32_t pipelineDepth = SelectEqualPipelineDepth(param.rankSize, dataBytes);
        const bool pipelineFlatOwner =
            algorithm == AllReduceAlgorithm::CLOS_FULL_LANE_FLAT_OWNER_RSAG &&
            (param.rankSize == 12U || param.rankSize == 16U) && pipelineDepth > 1U;
        const bool invalidKernel = topologyAware ?
            resource.slaveThread == 0 || (pipelineFlatOwner && resource.gatherThread == 0) ||
                resource.phase0Kernel == 0 || resource.phase1Kernel == 0 || resource.phase2Kernel == 0 :
            resource.kernel == 0;
        CHK_PRT_RET(invalidKernel || resource.rankSize != param.rankSize ||
            resource.dataBytes != dataBytes || resource.algorithm != algorithm,
            HCCL_ERROR("Cached CCU resource does not match this AllReduce"), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    CHK_RET(CreateResources(comm, stream, param, algorithm, dataBytes, resource));
    std::vector<char> serialized = resource.Serialize();
    CHK_PRT_RET(serialized.empty(), HCCL_ERROR("Failed to serialize CCU resources"), HCCL_E_INTERNAL);
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU,
        static_cast<uint64_t>(serialized.size()), &ctx));
    CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag, serialized.data(),
        static_cast<uint64_t>(serialized.size()), 0));
    return HCCL_SUCCESS;
}
} // namespace

extern "C" HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("Only float32 is supported"), HCCL_E_PARA);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM, HCCL_ERROR("Only sum is supported"), HCCL_E_PARA);
    CHK_PRT_RET(count > UINT64_MAX / sizeof(float), HCCL_ERROR("AllReduce byte size overflow"), HCCL_E_PARA);

    OpParam param {};
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    HcclDfxOpInfo dfxInfo {};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rankSize[%u]", param.rankSize), HCCL_E_PARA);

    const uint64_t dataBytes = count * sizeof(float);
    AllReduceAlgorithm algorithm = AllReduceAlgorithm::CLOS_RECURSIVE_DOUBLING;
    if (dataBytes > SMALL_DATA_BYTES) {
        if (param.rankSize == 12U || param.rankSize == 16U) {
            algorithm = AllReduceAlgorithm::CLOS_FULL_LANE_FLAT_OWNER_RSAG;
        } else {
            algorithm = AllReduceAlgorithm::CLOS_RECURSIVE_HALVING_DOUBLING;
        }
    }
    std::snprintf(param.tag, sizeof(param.tag), "hccl_custom_ar_v67_barrier_cleanup_r%u_c%llu_a%u",
        param.rankSize, static_cast<unsigned long long>(count), static_cast<uint32_t>(algorithm));

    AlgResourceCtx resource {};
    if (param.rankSize == 1) {
        CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, 0, &resource.thread));
    } else {
        CHK_RET(GetOrCreateResources(comm, stream, param, algorithm, dataBytes, resource));
    }
    CHK_RET(ops_hccl::ExecOp(param, resource));
    return HCCL_SUCCESS;
}
