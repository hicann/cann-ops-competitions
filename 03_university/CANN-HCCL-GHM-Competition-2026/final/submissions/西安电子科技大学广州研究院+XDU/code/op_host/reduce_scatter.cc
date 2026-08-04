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
constexpr uint32_t LAYER_MESH = 0;
constexpr uint32_t LAYER_CLOS = 1;
constexpr uint32_t MAX_REDUCE_SOURCES = 8;
constexpr uint64_t CCL_ALIGNMENT = 4096;
constexpr uint64_t MAX_CCU_TRANSFER_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t SMALL_INPUT_BYTES = 512ULL * 1024ULL;
constexpr uint64_t LARGE_512M_INPUT_BYTES = 512ULL * 1024ULL * 1024ULL;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (value == 0) {
        return 0;
    }
    return (value - 1U) / alignment * alignment + alignment;
}

bool HasLayer(const std::vector<uint32_t> &layers, uint32_t layer)
{
    return std::find(layers.begin(), layers.end(), layer) != layers.end();
}

HcclResult GetLayerRanks(HcclComm comm, uint32_t layer, std::vector<uint32_t> &result)
{
    uint32_t *ranks = nullptr;
    uint32_t rankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &ranks, &rankNum));
    CHK_PRT_RET(rankNum == 0 || ranks == nullptr,
        HCCL_ERROR("Layer[%u] returned an empty rank list", layer), HCCL_E_INTERNAL);
    result.assign(ranks, ranks + rankNum);
    std::sort(result.begin(), result.end());
    return HCCL_SUCCESS;
}

HcclResult BuildChannel(HcclComm comm, uint32_t localRank, uint32_t remoteRank,
    uint32_t layer, ChannelHandle &channel)
{
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, layer, localRank, remoteRank, &links, &linkNum));
    CHK_PRT_RET(linkNum == 0 || links == nullptr,
        HCCL_ERROR("No link on layer[%u] from rank[%u] to rank[%u]",
            layer, localRank, remoteRank),
        HCCL_E_NOT_FOUND);

    // Ascend 950 CCU prefers the UBC transport.  The remaining entries are
    // safe fallbacks for compatible rank tables.
    constexpr std::array<CommProtocol, 4> PROTOCOL_PREFERENCE = {
        COMM_PROTOCOL_UBC_CTP,
        COMM_PROTOCOL_UBC_TP,
        COMM_PROTOCOL_UB_MEM,
        COMM_PROTOCOL_UBOE,
    };

    const CommLink *selected = nullptr;
    for (CommProtocol protocol : PROTOCOL_PREFERENCE) {
        for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
            if (links[linkIndex].linkAttr.linkProtocol == protocol) {
                selected = &links[linkIndex];
                break;
            }
        }
        if (selected != nullptr) {
            break;
        }
    }
    CHK_PRT_RET(selected == nullptr,
        HCCL_ERROR("No CCU protocol on layer[%u] from rank[%u] to rank[%u]",
            layer, localRank, remoteRank),
        HCCL_E_NOT_SUPPORT);

    HcclChannelDesc desc;
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = selected->linkAttr.linkProtocol;
    desc.localEndpoint.protocol = selected->srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = selected->srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = selected->srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = selected->dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = selected->dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = selected->dstEndpointDesc.loc;
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
    return HCCL_SUCCESS;
}

HcclResult FillLayerKernelArg(HcclComm comm, uint32_t localRank,
    const std::vector<uint32_t> &peerRanks, uint32_t layer, bool includeSelf,
    std::shared_ptr<CcuKernelArgReduceLayer> &kernelArg)
{
    const uint32_t sourceCount =
        static_cast<uint32_t>(peerRanks.size()) + static_cast<uint32_t>(includeSelf);
    CHK_PRT_RET(sourceCount == 0 || sourceCount > MAX_REDUCE_SOURCES,
        HCCL_ERROR("Layer[%u] source count[%u] is unsupported", layer, sourceCount),
        HCCL_E_NOT_SUPPORT);

    kernelArg = std::make_shared<CcuKernelArgReduceLayer>();
    kernelArg->rankId = localRank;
    kernelArg->peerCount = static_cast<uint32_t>(peerRanks.size());
    kernelArg->includeSelf = includeSelf ? 1U : 0U;
    kernelArg->layer = layer;
    kernelArg->channelCount = kernelArg->peerCount;

    for (uint32_t index = 0; index < kernelArg->peerCount; ++index) {
        kernelArg->peerRanks[index] = peerRanks[index];
        CHK_RET(BuildChannel(
            comm, localRank, peerRanks[index], layer, kernelArg->channels[index]));
    }
    return HCCL_SUCCESS;
}

void SetSmallMetadata(CcuKernelArgReduceLayer &arg, uint32_t topology,
    uint32_t retainedGroupIndex, uint32_t groupRankIndex)
{
    arg.smallTopology = topology;
    arg.retainedGroupIndex = retainedGroupIndex;
    arg.groupRankIndex = groupRankIndex;
}

HcclResult ConfigureSmallTopology(uint32_t myRank,
    CcuKernelArgReduceLayer &meshArg, CcuKernelArgReduceLayer &closArg,
    AlgResourceCtx &resCtx)
{
    const bool twoByEight = resCtx.localRankSize == 8U &&
        resCtx.remoteRankSize == 8U;
    const bool mixedLarge = resCtx.localRankSize == 8U &&
        resCtx.remoteRankSize == 4U;
    const bool mixedSmall = resCtx.localRankSize == 4U &&
        resCtx.remoteRankSize == 8U;

    if (twoByEight) {
        resCtx.smallTopology = SMALL_TOPO_TWO_BY_EIGHT;
        SetSmallMetadata(meshArg, resCtx.smallTopology, myRank / 8U, myRank % 8U);
        SetSmallMetadata(closArg, resCtx.smallTopology, myRank / 8U, myRank % 8U);

        for (uint32_t peerIndex = 0; peerIndex < closArg.peerCount; ++peerIndex) {
            if (closArg.peerRanks[peerIndex] % 8U == myRank % 8U) {
                closArg.selectedPeerIndices[closArg.selectedPeerCount++] = peerIndex;
            }
        }
        for (uint32_t step = 1; step < 8U; ++step) {
            const uint32_t target = (myRank % 8U + step) % 8U;
            for (uint32_t peerIndex = 0; peerIndex < meshArg.peerCount; ++peerIndex) {
                if (meshArg.peerRanks[peerIndex] % 8U == target) {
                    meshArg.groupPeerIndices[meshArg.groupPeerCount++] = peerIndex;
                    break;
                }
            }
        }
        CHK_PRT_RET(closArg.selectedPeerCount != 1U || meshArg.groupPeerCount != 7U,
            HCCL_ERROR("Invalid 2x8 mixed-radix peer mapping on rank[%u]", myRank),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    if (!mixedLarge && !mixedSmall) {
        return HCCL_SUCCESS;
    }

    resCtx.smallTopology = mixedLarge ?
        SMALL_TOPO_EIGHT_PLUS_FOUR_LARGE :
        SMALL_TOPO_EIGHT_PLUS_FOUR_SMALL;
    const uint32_t retainedGroup = myRank / 4U;
    const uint32_t groupRank = myRank % 4U;
    SetSmallMetadata(meshArg, resCtx.smallTopology, retainedGroup, groupRank);
    SetSmallMetadata(closArg, resCtx.smallTopology, retainedGroup, groupRank);

    for (uint32_t peerIndex = 0; peerIndex < closArg.peerCount; ++peerIndex) {
        if (closArg.peerRanks[peerIndex] % 4U == groupRank) {
            closArg.selectedPeerIndices[closArg.selectedPeerCount++] = peerIndex;
        }
    }
    for (uint32_t peerIndex = 0; peerIndex < meshArg.peerCount; ++peerIndex) {
        const uint32_t peerRank = meshArg.peerRanks[peerIndex];
        if (peerRank / 4U != retainedGroup && peerRank % 4U == groupRank) {
            meshArg.selectedPeerIndices[meshArg.selectedPeerCount++] = peerIndex;
        }
    }
    for (uint32_t step = 1; step < 4U; ++step) {
        const uint32_t target = (groupRank + step) % 4U;
        for (uint32_t peerIndex = 0; peerIndex < meshArg.peerCount; ++peerIndex) {
            const uint32_t peerRank = meshArg.peerRanks[peerIndex];
            if (peerRank / 4U == retainedGroup && peerRank % 4U == target) {
                meshArg.groupPeerIndices[meshArg.groupPeerCount++] = peerIndex;
                break;
            }
        }
    }

    const uint32_t expectedClosPeers = mixedLarge ? 1U : 2U;
    const uint32_t expectedMeshStagePeers = mixedLarge ? 1U : 0U;
    CHK_PRT_RET(closArg.selectedPeerCount != expectedClosPeers ||
            meshArg.selectedPeerCount != expectedMeshStagePeers ||
            meshArg.groupPeerCount != 3U,
        HCCL_ERROR("Invalid 8+4 mixed-radix peer mapping on rank[%u]", myRank),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RegisterKernel(CcuInsHandle insHandle, const char *name, const void *function,
    CcuKernelArgBase *kernelArg, CcuKernelHandle &kernelHandle)
{
    const void *kernelArgs[] = {kernelArg};
    constexpr uint32_t RESERVED_DIE_ID = 0;
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, RESERVED_DIE_ID, name, function,
        kernelArgs, 1, &kernelHandle));
    return HCCL_SUCCESS;
}

HcclResult AllocateResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    void *cclBuffer = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
    CHK_PRT_RET(cclBuffer == nullptr || cclBufferSize < 8192,
        HCCL_ERROR("Invalid CCL buffer, addr[%p], size[%llu]", cclBuffer,
            static_cast<unsigned long long>(cclBufferSize)),
        HCCL_E_INTERNAL);
    resCtx.localBuffer = CommBuffer{cclBuffer, cclBufferSize};

    uint32_t *layerData = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerNum));
    CHK_PRT_RET(layerNum == 0 || layerData == nullptr,
        HCCL_ERROR("Rank[%u] has no topology layers", param.myRank), HCCL_E_INTERNAL);
    const std::vector<uint32_t> layers(layerData, layerData + layerNum);
    CHK_PRT_RET(!HasLayer(layers, LAYER_CLOS),
        HCCL_ERROR("Rank[%u] has no layer-1 Clos network", param.myRank),
        HCCL_E_NOT_SUPPORT);

    std::vector<uint32_t> localRanks;
    if (HasLayer(layers, LAYER_MESH)) {
        CHK_RET(GetLayerRanks(comm, LAYER_MESH, localRanks));
    } else {
        localRanks.push_back(param.myRank);
    }
    CHK_PRT_RET(std::find(localRanks.begin(), localRanks.end(), param.myRank) == localRanks.end(),
        HCCL_ERROR("Layer-0 rank list does not contain rank[%u]", param.myRank),
        HCCL_E_INTERNAL);
    resCtx.localRankSize = static_cast<uint32_t>(localRanks.size());
    resCtx.localRankIndex = static_cast<uint32_t>(
        std::distance(localRanks.begin(),
            std::find(localRanks.begin(), localRanks.end(), param.myRank)));

    std::vector<uint32_t> meshPeers;
    std::vector<uint32_t> closPeers;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.myRank) {
            continue;
        }
        if (std::binary_search(localRanks.begin(), localRanks.end(), rank)) {
            meshPeers.push_back(rank);
        } else {
            closPeers.push_back(rank);
        }
    }

    // One-server topologies use layer-1 for the complete reduction. Two-layer
    // topologies use one independent kernel per IO die.
    const bool twoLayer = !meshPeers.empty() && !closPeers.empty();
    resCtx.remoteRankSize = static_cast<uint32_t>(closPeers.size());
    if (twoLayer) {
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, 1, 1,
            &resCtx.auxThread));
    }

    std::vector<std::shared_ptr<CcuKernelArgBase>> kernelArgLifetime;
    std::shared_ptr<CcuKernelArgReduceLayer> meshArg;
    std::shared_ptr<CcuKernelArgReduceLayer> closArg;
    std::shared_ptr<CcuKernelArgFinalize> finalizeArg;

    if (twoLayer) {
        CHK_RET(FillLayerKernelArg(
            comm, param.myRank, meshPeers, LAYER_MESH, true, meshArg));
        CHK_RET(FillLayerKernelArg(
            comm, param.myRank, closPeers, LAYER_CLOS, false, closArg));
        CHK_RET(ConfigureSmallTopology(
            param.myRank, *meshArg, *closArg, resCtx));

        const uint64_t outputBytes = param.count * sizeof(float);
        const uint64_t inputBytes = outputBytes * param.rankSize;
        const bool eightPlusFour =
            resCtx.smallTopology == SMALL_TOPO_EIGHT_PLUS_FOUR_LARGE ||
            resCtx.smallTopology == SMALL_TOPO_EIGHT_PLUS_FOUR_SMALL;
        const uint64_t smallPaddingBytes = eightPlusFour ?
            static_cast<uint64_t>(param.rankSize) * sizeof(float) : 0;
        const bool groupedSourceCounts =
            (resCtx.localRankSize == 4U || resCtx.localRankSize == 8U) &&
            (resCtx.remoteRankSize == 4U || resCtx.remoteRankSize == 8U);
        const uint64_t partialStride = AlignUp(outputBytes, CCL_ALIGNMENT);
        const uint64_t meshGroupCount = resCtx.localRankSize / 2U;
        const uint64_t closGroupCount = resCtx.remoteRankSize / 2U;
        const bool layoutSafe = groupedSourceCounts && partialStride != 0 &&
            meshGroupCount <= std::numeric_limits<uint64_t>::max() / partialStride &&
            closGroupCount <= std::numeric_limits<uint64_t>::max() / partialStride &&
            (meshGroupCount - 1U) * partialStride <= cclBufferSize &&
            closGroupCount * partialStride <=
                cclBufferSize - (meshGroupCount - 1U) * partialStride;
        if (inputBytes > SMALL_INPUT_BYTES + smallPaddingBytes &&
            outputBytes <= MAX_CCU_TRANSFER_BYTES && layoutSafe) {
            resCtx.groupedLargeKernel = LARGE_ALGORITHM_V10;
            // hccl_test may round a multi-rank request down by one 4 KiB
            // page when the nominal input does not divide evenly by rankSize.
            const bool point512M =
                inputBytes >= LARGE_512M_INPUT_BYTES - CCL_ALIGNMENT;
            if (resCtx.localRankSize == 8U && resCtx.remoteRankSize == 8U) {
                // V23 moves two remote sources per output byte from the
                // four-link Clos bottleneck onto the seven-link local Mesh.
                // The same dedicated graph handles both 512 MiB and the
                // padded 400 MiB+4B case without size-label assumptions.
                resCtx.groupedLargeKernel = LARGE_ALGORITHM_V23_TWO_BY_EIGHT;
            } else if (point512M &&
                resCtx.localRankSize != resCtx.remoteRankSize) {
                resCtx.groupedLargeKernel = LARGE_ALGORITHM_V13_P17;
            }
        }
    } else {
        std::vector<uint32_t> allPeers;
        allPeers.reserve(param.rankSize - 1);
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.myRank) {
                allPeers.push_back(rank);
            }
        }
        CHK_RET(FillLayerKernelArg(
            comm, param.myRank, allPeers, LAYER_CLOS, true, closArg));
    }

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("Expected one CCU instruction instance, got[%u]", insNum),
        HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    if (twoLayer) {
        CHK_PRT_RET(resCtx.kernelCount >= MAX_CCU_KERNELS,
            HCCL_ERROR("CCU kernel capacity exceeded"), HCCL_E_INTERNAL);
        resCtx.meshKernelIndex = resCtx.kernelCount++;
        const bool groupedLarge =
            resCtx.groupedLargeKernel != LARGE_ALGORITHM_NONE;
        const bool groupedEarly =
            resCtx.groupedLargeKernel == LARGE_ALGORITHM_V12_P11;
        const bool groupedV13 =
            resCtx.groupedLargeKernel == LARGE_ALGORITHM_V13_P17;
        const bool groupedV23 =
            resCtx.groupedLargeKernel == LARGE_ALGORITHM_V23_TWO_BY_EIGHT;
        const char *kernelName = groupedV23 ?
            "CcuReduceLayer0GroupedV23Kernel" : groupedEarly ?
            "CcuReduceLayer0GroupedEarlyKernel" : groupedV13 ?
            "CcuReduceLayer0GroupedV13Kernel" : groupedLarge ?
            "CcuReduceLayer0GroupedKernel" : "CcuReduceLayer0Kernel";
        const void *kernelFunction = groupedV23 ?
            reinterpret_cast<const void *>(
                ops_hccl::CcuReduceLayer0GroupedV23Kernel) : groupedEarly ?
            reinterpret_cast<const void *>(
                ops_hccl::CcuReduceLayer0GroupedEarlyKernel) : groupedV13 ?
            reinterpret_cast<const void *>(
                ops_hccl::CcuReduceLayer0GroupedV13Kernel) : groupedLarge ?
            reinterpret_cast<const void *>(
                ops_hccl::CcuReduceLayer0GroupedKernel) :
            reinterpret_cast<const void *>(ops_hccl::CcuReduceLayer0Kernel);
        CHK_RET(RegisterKernel(insHandle,
            kernelName, kernelFunction,
            meshArg.get(), resCtx.ccuKernels[resCtx.meshKernelIndex]));
        kernelArgLifetime.push_back(meshArg);
    }

    CHK_PRT_RET(resCtx.kernelCount >= MAX_CCU_KERNELS,
        HCCL_ERROR("CCU kernel capacity exceeded"), HCCL_E_INTERNAL);
    resCtx.closKernelIndex = resCtx.kernelCount++;
    const bool groupedLarge =
        resCtx.groupedLargeKernel != LARGE_ALGORITHM_NONE;
    const bool groupedEarly =
        resCtx.groupedLargeKernel == LARGE_ALGORITHM_V12_P11;
    const bool groupedV13 =
        resCtx.groupedLargeKernel == LARGE_ALGORITHM_V13_P17;
    const bool groupedV23 =
        resCtx.groupedLargeKernel == LARGE_ALGORITHM_V23_TWO_BY_EIGHT;
    const char *kernelName = groupedV23 ?
        "CcuReduceLayer1GroupedV23Kernel" : groupedEarly ?
        "CcuReduceLayer1GroupedEarlyKernel" : groupedV13 ?
        "CcuReduceLayer1GroupedV13Kernel" : groupedLarge ?
        "CcuReduceLayer1GroupedKernel" : "CcuReduceLayer1Kernel";
    const void *kernelFunction = groupedV23 ?
        reinterpret_cast<const void *>(
            ops_hccl::CcuReduceLayer1GroupedV23Kernel) : groupedEarly ?
        reinterpret_cast<const void *>(
            ops_hccl::CcuReduceLayer1GroupedEarlyKernel) : groupedV13 ?
        reinterpret_cast<const void *>(
            ops_hccl::CcuReduceLayer1GroupedV13Kernel) : groupedLarge ?
        reinterpret_cast<const void *>(
            ops_hccl::CcuReduceLayer1GroupedKernel) :
        reinterpret_cast<const void *>(ops_hccl::CcuReduceLayer1Kernel);
    CHK_RET(RegisterKernel(insHandle,
        kernelName, kernelFunction,
        closArg.get(), resCtx.ccuKernels[resCtx.closKernelIndex]));
    kernelArgLifetime.push_back(closArg);

    if (twoLayer) {
        finalizeArg = std::make_shared<CcuKernelArgFinalize>();
        CHK_PRT_RET(resCtx.kernelCount >= MAX_CCU_KERNELS,
            HCCL_ERROR("CCU kernel capacity exceeded"), HCCL_E_INTERNAL);
        resCtx.finalizeKernelIndex = resCtx.kernelCount++;
        CHK_RET(RegisterKernel(insHandle, "CcuReduceFinalizeKernel",
            reinterpret_cast<const void *>(ops_hccl::CcuReduceFinalizeKernel),
            finalizeArg.get(), resCtx.ccuKernels[resCtx.finalizeKernelIndex]));
        kernelArgLifetime.push_back(finalizeArg);
    }

    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Only FP32 is supported, dataType[%d]", dataType),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only SUM is supported, reduceOp[%d]", op),
        HCCL_E_NOT_SUPPORT);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(
        commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE ||
            param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank information: rank[%u], rankSize[%u]",
            param.myRank, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("recvCount overflows output byte size"), HCCL_E_PARA);
    const uint64_t outputBytes = recvCount * sizeof(float);
    CHK_PRT_RET(outputBytes != 0 &&
            param.rankSize > std::numeric_limits<uint64_t>::max() / outputBytes,
        HCCL_ERROR("Input byte size overflows"), HCCL_E_PARA);
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    (void)std::snprintf(param.tag, sizeof(param.tag),
        "hccl_custom_rs_v22_%llu",
        static_cast<unsigned long long>(recvCount));

    constexpr CommEngine CCU_ENGINE = CommEngine::COMM_ENGINE_CCU;
    // The user stream is acquired on every invocation. It is intentionally not
    // serialized in the reusable engine context.
    CHK_RET(HcclThreadAcquireWithStream(
        comm, CCU_ENGINE, stream, 1, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, CCU_ENGINE, &ctx, &size) == HCCL_SUCCESS) {
        CHK_PRT_RET(size != sizeof(AlgResourceCtx),
            HCCL_ERROR("Unexpected reusable context size[%llu]",
                static_cast<unsigned long long>(size)),
            HCCL_E_INTERNAL);
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        CHK_RET(AllocateResources(comm, param, resCtxHost));
        param.ctxSize = sizeof(resCtxHost);
        CHK_RET(HcclEngineCtxCreate(
            comm, param.tag, CCU_ENGINE, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, CCU_ENGINE, param.tag,
            &resCtxHost, sizeof(resCtxHost), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
