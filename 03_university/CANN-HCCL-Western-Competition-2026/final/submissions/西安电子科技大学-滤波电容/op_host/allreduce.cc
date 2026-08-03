/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AllReduce host-side entry point.
//
// Algorithm: topology-aware sharded AllReduce with a proven tree fallback.
//
// Key differences from baseline (v4 tree):
//   - Small messages and non-power-of-two groups use the aligned WriteReduce tree.
//   - Four-rank large messages use bidirectional recursive halving/doubling.
//   - Twelve-rank and sixteen-rank large messages use a dual-die direct mesh.
//   - The algorithm variant is part of the engine-context tag.

#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "hccl_ccu_res.h"

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint64_t SMALL_MESSAGE_BYTES = 1024ULL * 1024ULL;
constexpr uint64_t FAST_MESH_MIN_BYTES = 512ULL * 1024ULL;
constexpr uint32_t FAST_MESH_DIE_NUM = 2;

struct PeerChannel {
    uint32_t remoteRank;
    ChannelHandle handle;
    uint32_t dieId;
};

HcclResult AcquirePeerChannels(HcclComm comm, const OpParam &param,
    bool queryDie, std::vector<PeerChannel> &channels)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("No network layer is available"), HCCL_E_INTERNAL);
    const std::vector<uint32_t> netLayerList(netLayers, netLayers + netLayerNum);

    channels.clear();
    channels.reserve(param.rankSize - 1);
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        bool channelFound = false;
        for (const uint32_t netLayer : netLayerList) {
            if (channelFound) {
                break;
            }
            CommLink *links = nullptr;
            uint32_t linkNum = 0;
            CHK_RET(HcclRankGraphGetLinks(
                comm, netLayer, param.myRank, remoteRank, &links, &linkNum));

            for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
                const CommLink &link = links[linkIndex];
                if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }

                HcclChannelDesc desc;
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

                ChannelHandle channel = 0;
                CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
                uint32_t dieId = 0;
                if (queryDie) {
                    EndpointDesc localEndpoint = desc.localEndpoint;
                    CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank,
                        &localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
                    CHK_PRT_RET(dieId >= FAST_MESH_DIE_NUM,
                        HCCL_ERROR("Invalid die id %u for peer %u", dieId, remoteRank),
                        HCCL_E_INTERNAL);
                }
                channels.push_back(PeerChannel{remoteRank, channel, dieId});
                channelFound = true;
                break;
            }
        }

        CHK_PRT_RET(!channelFound,
            HCCL_ERROR("No UBC_CTP link from rank %u to rank %u", param.myRank, remoteRank),
            HCCL_E_NOT_FOUND);
    }
    return HCCL_SUCCESS;
}

HcclResult ResolveLocalServerRanks(HcclComm comm, const OpParam &param,
    std::vector<uint32_t> &localServerRanks)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("No network layer is available"), HCCL_E_INTERNAL);

    localServerRanks.clear();
    for (uint32_t layerIndex = 0; layerIndex < netLayerNum; ++layerIndex) {
        uint32_t *layerRanks = nullptr;
        uint32_t layerRankNum = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(
            comm, netLayers[layerIndex], &layerRanks, &layerRankNum));
        if (layerRanks == nullptr || layerRankNum >= param.rankSize ||
            layerRankNum <= localServerRanks.size()) {
            continue;
        }
        const bool containsLocalRank =
            std::find(layerRanks, layerRanks + layerRankNum, param.myRank) !=
            layerRanks + layerRankNum;
        if (containsLocalRank) {
            localServerRanks.assign(layerRanks, layerRanks + layerRankNum);
        }
    }

    std::sort(localServerRanks.begin(), localServerRanks.end());
    CHK_PRT_RET(localServerRanks.size() != 4 && localServerRanks.size() != 8,
        HCCL_ERROR("Unable to resolve a four-rank or eight-rank local topology instance"),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

bool IsTreeNeighbor(uint32_t rank, uint32_t remoteRank, uint32_t rankSize)
{
    for (uint32_t stride = 1; stride < rankSize; stride <<= 1) {
        const uint32_t position = rank % (stride << 1);
        if (position == 0) {
            const uint32_t child = rank + stride;
            if (child < rankSize && remoteRank == child) {
                return true;
            }
        } else if (position == stride) {
            return remoteRank == rank - stride;
        }
    }
    return false;
}

bool IsPowerOfTwo(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

bool UseShardedAlgorithm(uint32_t rankSize, uint64_t count)
{
    constexpr uint64_t FP32_BYTES = sizeof(float);
    const bool usePowerOfTwoRhd =
        (rankSize == 4 || rankSize == 16) && IsPowerOfTwo(rankSize);
    return (rankSize == 12 || usePowerOfTwoRhd) &&
        count > SMALL_MESSAGE_BYTES / FP32_BYTES;
}

bool UseFastMeshAlgorithm(uint32_t rankSize, uint64_t count)
{
    constexpr uint64_t FP32_BYTES = sizeof(float);
    const bool useRank12SmallMesh = rankSize == 12 &&
        count >= FAST_MESH_MIN_BYTES / FP32_BYTES &&
        count <= SMALL_MESSAGE_BYTES / FP32_BYTES;
    const bool useRank16SmallMesh = rankSize == 16 &&
        count >= FAST_MESH_MIN_BYTES / FP32_BYTES &&
        count <= SMALL_MESSAGE_BYTES / FP32_BYTES;
    return useRank12SmallMesh || useRank16SmallMesh;
}

bool UseHierarchicalAlgorithm(uint32_t rankSize, uint64_t count)
{
    constexpr uint64_t FP32_BYTES = sizeof(float);
    return (rankSize == 12 || rankSize == 16) &&
        count > SMALL_MESSAGE_BYTES / FP32_BYTES;
}

bool IsRequiredNeighbor(uint32_t rank, uint32_t remoteRank, uint32_t rankSize,
    bool useShardedAlgorithm)
{
    if (!useShardedAlgorithm) {
        return IsTreeNeighbor(rank, remoteRank, rankSize);
    }
    return rank != remoteRank;
}

HcclResult RegisterAllReduceKernel(HcclComm comm, const OpParam &param,
    const std::vector<PeerChannel> &channels, bool useShardedAlgorithm,
    AlgResourceCtx &resource, uint32_t kernelOffset = 0)
{
    CHK_PRT_RET(channels.size() != static_cast<size_t>(param.rankSize - 1),
        HCCL_ERROR("Unexpected channel count: %zu", channels.size()), HCCL_E_INTERNAL);

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("Unexpected CCU instance count: %u", insNum), HCCL_E_INTERNAL);

    constexpr uint32_t DIE_ID = 0;
    if (kernelOffset == 0) {
        resource.ccuKernels.assign(param.rankSize, 0);
    } else {
        resource.ccuKernels.resize(kernelOffset + param.rankSize, 0);
    }
    uint32_t registeredKernelCount = 0;
    for (uint32_t remoteRank = 0, channelIndex = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        const PeerChannel &peerChannel = channels[channelIndex++];
        const ChannelHandle channel = peerChannel.handle;
        if (!IsRequiredNeighbor(param.myRank, remoteRank, param.rankSize,
                useShardedAlgorithm)) {
            continue;
        }

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgAllReduce>();
        kernelArg->rankId = param.myRank;
        kernelArg->peerRank = remoteRank;
        kernelArg->dataType = param.dataType;
        kernelArg->reduceType = param.reduceType;
        kernelArg->channelCount = 1;
        kernelArg->channels[0] = channel;

        CcuKernelInfo kernelInfo{};
        const int nameResult = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "CcuPairAllReduce_%u_%u", param.myRank, remoteRank);
        CHK_PRT_RET(nameResult < 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName),
            HCCL_ERROR("Failed to set CCU kernel name"), HCCL_E_INTERNAL);
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);
        kernelInfo.setKernelArg(kernelArg);

        CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        const uint32_t kernelDie = useShardedAlgorithm &&
                (param.rankSize == 12 || param.rankSize == 16)
            ? peerChannel.dieId
            : DIE_ID;
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, kernelDie, kernelInfo.kernelFuncName,
            kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle));
        CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
        resource.ccuKernels[kernelOffset + remoteRank] = kernelHandle;
        ++registeredKernelCount;
    }

    CHK_PRT_RET(registeredKernelCount == 0,
        HCCL_ERROR("No pair channel was registered for rank %u", param.myRank), HCCL_E_INTERNAL);
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        CHK_PRT_RET(IsRequiredNeighbor(param.myRank, remoteRank, param.rankSize,
                useShardedAlgorithm) &&
                resource.ccuKernels[kernelOffset + remoteRank] == 0,
            HCCL_ERROR("Missing pair kernel from rank %u to rank %u", param.myRank, remoteRank),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterFastMeshKernels(HcclComm comm, const OpParam &param,
    const std::vector<PeerChannel> &channels, AlgResourceCtx &resource)
{
    CHK_PRT_RET(channels.size() < 3 || channels.size() >= 16,
        HCCL_ERROR("Unexpected local-mesh channel count: %zu", channels.size()),
        HCCL_E_INTERNAL);

    std::vector<PeerChannel> dieChannels[FAST_MESH_DIE_NUM];
    for (const PeerChannel &peer : channels) {
        CHK_PRT_RET(peer.dieId >= FAST_MESH_DIE_NUM,
            HCCL_ERROR("Invalid channel die id: %u", peer.dieId), HCCL_E_INTERNAL);
        dieChannels[peer.dieId].push_back(peer);
    }
    const bool dualDie =
        !dieChannels[0].empty() && !dieChannels[1].empty();
    uint32_t primaryDie = 0;
    if (dieChannels[0].empty()) {
        primaryDie = 1;
    } else if (dualDie && dieChannels[1].size() < dieChannels[0].size()) {
        primaryDie = 1;
    }
    const uint32_t secondaryDie = 1 - primaryDie;
    uint32_t secondaryPartialLane = 0;
    if (dualDie) {
        const uint32_t secondaryPartialRank =
            dieChannels[secondaryDie][0].remoteRank;
        const auto secondaryIter = std::find_if(channels.begin(), channels.end(),
            [secondaryPartialRank](const PeerChannel &peer) {
                return peer.remoteRank == secondaryPartialRank;
            });
        CHK_PRT_RET(secondaryIter == channels.end(),
            HCCL_ERROR("Unable to assign the secondary partial lane"),
            HCCL_E_INTERNAL);
        secondaryPartialLane =
            static_cast<uint32_t>(secondaryIter - channels.begin());
        CHK_PRT_RET(secondaryPartialLane >= channels.size(),
            HCCL_ERROR("Invalid secondary partial lane: %u",
                secondaryPartialLane), HCCL_E_INTERNAL);
    }

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("Unexpected CCU instance count: %u", insNum), HCCL_E_INTERNAL);

    resource.ccuKernels.assign(FAST_MESH_DIE_NUM * 2, 0);
    const uint32_t orderedDies[FAST_MESH_DIE_NUM] = {primaryDie, secondaryDie};
    const uint32_t activeDieCount = dualDie ? FAST_MESH_DIE_NUM : 1;
    std::vector<std::shared_ptr<ops_hccl::CcuKernelArgShardedMesh>> kernelArgHolders;
    kernelArgHolders.reserve(activeDieCount);
    std::vector<CcuKernelInfo> computeInfos(FAST_MESH_DIE_NUM);
    std::vector<CcuKernelHandle> computeHandles(FAST_MESH_DIE_NUM, 0);
    for (uint32_t role = 0; role < activeDieCount; ++role) {
        const uint32_t dieId = orderedDies[role];
        const std::vector<PeerChannel> &group = dieChannels[dieId];
        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgShardedMesh>();
        kernelArg->rankId = param.myRank;
        kernelArg->rankSize = param.rankSize;
        kernelArg->dataType = param.dataType;
        kernelArg->reduceType = param.reduceType;
        kernelArg->secondaryPartialLane = secondaryPartialLane;
        kernelArg->primary = role == 0;
        kernelArg->channelCount = static_cast<uint32_t>(group.size());
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            kernelArg->channels[i] = group[i].handle;
            kernelArg->peerRanks[i] = group[i].remoteRank;
            const auto laneIter = std::find_if(channels.begin(), channels.end(),
                [&group, i](const PeerChannel &peer) {
                    return peer.remoteRank == group[i].remoteRank;
                });
            CHK_PRT_RET(laneIter == channels.end(),
                HCCL_ERROR("Unable to assign a local scratch lane"), HCCL_E_INTERNAL);
            kernelArg->scratchLanes[i] =
                static_cast<uint32_t>(laneIter - channels.begin());
        }

        CcuKernelInfo bootstrapInfo{};
        const int bootstrapNameResult = std::snprintf(bootstrapInfo.kernelFuncName,
            sizeof(bootstrapInfo.kernelFuncName), "CcuMeshBootstrap_%u_%u_%u",
            param.myRank, dieId, role);
        CHK_PRT_RET(bootstrapNameResult < 0 ||
                static_cast<size_t>(bootstrapNameResult) >=
                    sizeof(bootstrapInfo.kernelFuncName),
            HCCL_ERROR("Failed to set mesh-bootstrap kernel name"), HCCL_E_INTERNAL);
        bootstrapInfo.kernelFunc =
            reinterpret_cast<void *>(ops_hccl::CcuMeshBootstrapKernel);
        bootstrapInfo.setKernelArg(kernelArg);

        CcuKernelHandle bootstrapHandle = 0;
        const void *bootstrapArgs[] = {bootstrapInfo.kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId,
            bootstrapInfo.kernelFuncName, bootstrapInfo.kernelFunc,
            bootstrapArgs, 1, &bootstrapHandle));
        CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
        resource.ccuKernels[role] = bootstrapHandle;

        CcuKernelInfo &computeInfo = computeInfos[role];
        const int computeNameResult = std::snprintf(computeInfo.kernelFuncName,
            sizeof(computeInfo.kernelFuncName), "CcuShardedMesh_%u_%u_%u",
            param.myRank, dieId, role);
        CHK_PRT_RET(computeNameResult < 0 ||
                static_cast<size_t>(computeNameResult) >=
                    sizeof(computeInfo.kernelFuncName),
            HCCL_ERROR("Failed to set fast-mesh kernel name"), HCCL_E_INTERNAL);
        computeInfo.kernelFunc =
            reinterpret_cast<void *>(ops_hccl::CcuShardedMeshKernel);
        computeInfo.setKernelArg(kernelArg);
        kernelArgHolders.push_back(kernelArg);
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    for (uint32_t role = 0; role < activeDieCount; ++role) {
        CcuKernelInfo &computeInfo = computeInfos[role];
        const void *computeArgs[] = {computeInfo.kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, orderedDies[role],
            computeInfo.kernelFuncName, computeInfo.kernelFunc,
            computeArgs, 1, &computeHandles[role]));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    for (uint32_t role = 0; role < activeDieCount; ++role) {
        resource.ccuKernels[FAST_MESH_DIE_NUM + role] = computeHandles[role];
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterHierarchicalKernels(HcclComm comm, const OpParam &param,
    const std::vector<PeerChannel> &channels, AlgResourceCtx &resource)
{
    std::vector<PeerChannel> localChannels;
    for (const PeerChannel &peer : channels) {
        if (std::find(resource.localServerRanks.begin(),
                resource.localServerRanks.end(), peer.remoteRank) !=
            resource.localServerRanks.end()) {
            localChannels.push_back(peer);
        }
    }
    CHK_PRT_RET(localChannels.size() + 1 != resource.localServerRanks.size(),
        HCCL_ERROR("Invalid local mesh channel count: %zu for group %zu",
            localChannels.size(), resource.localServerRanks.size()),
        HCCL_E_INTERNAL);

    CHK_RET(RegisterFastMeshKernels(comm, param, localChannels, resource));
    constexpr uint32_t MESH_KERNEL_COUNT = FAST_MESH_DIE_NUM * 2;
    CHK_RET(RegisterAllReduceKernel(comm, param, channels, true,
        resource, MESH_KERNEL_COUNT));
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
        HCCL_ERROR("Only FP32 is supported, dataType=%d", static_cast<int>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only SUM is supported, reduceOp=%d", static_cast<int>(op)),
        HCCL_E_NOT_SUPPORT);

    // Build the operator parameters.
    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    // Register operator diagnostics information.
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: Resolve topology information.
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rank size: %u", param.rankSize), HCCL_E_NOT_SUPPORT);

    const bool useShardedAlgorithm = UseShardedAlgorithm(param.rankSize, param.count);
    const bool useHierarchicalAlgorithm =
        UseHierarchicalAlgorithm(param.rankSize, param.count);
    const bool useFastMeshAlgorithm = UseFastMeshAlgorithm(param.rankSize, param.count) &&
        sendBuf != recvBuf;
    const uint32_t algorithmKind = useHierarchicalAlgorithm ? 3u :
        (useFastMeshAlgorithm ? 2u :
        (useShardedAlgorithm ? 1u : 0u));

    // Separate tree and sharded contexts because their peer sets differ.
    const int tagResult = std::snprintf(param.tag, sizeof(param.tag),
        "hccl_custom_allreduce_v18_parallel_%u_%u_%u_%u",
        static_cast<uint32_t>(dataType),
        static_cast<uint32_t>(op), param.rankSize, algorithmKind);
    CHK_PRT_RET(tagResult < 0 || static_cast<size_t>(tagResult) >= sizeof(param.tag),
        HCCL_ERROR("Failed to construct operator tag"), HCCL_E_INTERNAL);

    // ==============================================
    // STEP 2: Create communication resources.
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: Acquire resources for host/device synchronization.
    // ==============================================
    // Convert the user stream into a CCU thread and acquire one notify resource.
    const uint32_t mainNotifyCount =
        useHierarchicalAlgorithm && param.rankSize == 16 ? 2 : 1;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, ccuEngine, stream, mainNotifyCount, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // Reuse an existing CCU engine context.
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        // Build device resources when no engine context exists.
        AlgResourceCtx resCtxHost;

        // Obtain the device-side HCCL communication buffer.
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: Acquire the thread, channels, and CCU kernel.
        // ==============================================

        resCtxHost.ccuThread = param.cpuThread;
        if (useFastMeshAlgorithm || useHierarchicalAlgorithm) {
            ThreadHandle meshThread = 0;
            CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, 1, &meshThread));
            resCtxHost.threads = {param.cpuThread, meshThread};
            if (useHierarchicalAlgorithm && param.rankSize == 16) {
                ThreadHandle closThread = 0;
                CHK_RET(HcclThreadAcquire(
                    comm, ccuEngine, 1, 1, &closThread));
                resCtxHost.threads.push_back(closThread);
            }
        } else {
            resCtxHost.threads = {param.cpuThread};
        }

        if (param.rankSize > 1) {
            std::vector<PeerChannel> channels;
            if (useHierarchicalAlgorithm) {
                CHK_RET(ResolveLocalServerRanks(comm, param, resCtxHost.localServerRanks));
            }
            const bool queryChannelDie = useFastMeshAlgorithm ||
                useHierarchicalAlgorithm ||
                (useShardedAlgorithm &&
                    (param.rankSize == 12 || param.rankSize == 16));
            CHK_RET(AcquirePeerChannels(comm, param, queryChannelDie, channels));
            if (useHierarchicalAlgorithm) {
                CHK_RET(RegisterHierarchicalKernels(
                    comm, param, channels, resCtxHost));
            } else if (useFastMeshAlgorithm) {
                CHK_RET(RegisterFastMeshKernels(comm, param, channels, resCtxHost));
            } else {
                CHK_RET(RegisterAllReduceKernel(comm, param, channels,
                    useShardedAlgorithm, resCtxHost));
            }
        }

        // ==============================================
        // STEP 2.3: Acquire the communication engine context.
        // ==============================================
        // Serialize AlgResourceCtx into the CCU communication engine context.
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    // ==============================================
    // STEP 3: Launch the CCU kernel.
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
