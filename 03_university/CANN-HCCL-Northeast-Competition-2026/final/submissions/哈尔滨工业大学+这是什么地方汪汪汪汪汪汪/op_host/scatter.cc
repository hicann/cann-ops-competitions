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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t MAX_CCU_DIES = 2;
constexpr uint64_t HELPER_THRESHOLD_BYTES = 1024ULL * 1024ULL;
constexpr char SMALL_CONTEXT_TAG[] = "hccl_custom_scatter_ccu_v24_lt1m";
constexpr char LARGE_CONTEXT_TAG[] = "hccl_custom_scatter_ccu_v24_ge1m";

using ChannelGroup = std::vector<std::pair<uint32_t, ChannelHandle>>;
using ChannelGroups = std::map<uint32_t, ChannelGroup>;

bool IsEightPlusFourTopology(const AlgResourceCtx &resCtx)
{
    if (resCtx.rankSize != 12U || resCtx.instanceCount != 2U) {
        return false;
    }
    const uint32_t first = resCtx.instanceSizes[0];
    const uint32_t second = resCtx.instanceSizes[1];
    return (first == 8U && second == 4U) ||
        (first == 4U && second == 8U);
}

bool IsFourByThreeTopology(const AlgResourceCtx &resCtx)
{
    if (resCtx.rankSize != 12U || resCtx.instanceCount != 4U) {
        return false;
    }
    for (uint32_t i = 0; i < resCtx.instanceCount; ++i) {
        if (resCtx.instanceSizes[i] != 3U) {
            return false;
        }
    }
    return true;
}

HcclResult BuildChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    const std::vector<uint32_t> &layers, HcclChannelDesc &desc, uint32_t &dieId,
    uint32_t &selectedLayer)
{
    const std::array<CommProtocol, 2> protocols = {
        COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UBC_TP};

    for (CommProtocol wanted : protocols) {
        for (uint32_t layer : layers) {
            CommLink *links = nullptr;
            uint32_t linkNum = 0;
            const HcclResult ret = HcclRankGraphGetLinks(
                comm, layer, myRank, remoteRank, &links, &linkNum);
            if (ret != HCCL_SUCCESS || links == nullptr) {
                continue;
            }
            for (uint32_t i = 0; i < linkNum; ++i) {
                if (links[i].linkAttr.linkProtocol != wanted) {
                    continue;
                }
                CHK_RET(HcclChannelDescInit(&desc, 1));
                desc.remoteRank = remoteRank;
                desc.channelProtocol = wanted;
                desc.notifyNum = CHANNEL_NOTIFY_NUM;
                desc.localEndpoint = links[i].srcEndpointDesc;
                desc.remoteEndpoint = links[i].dstEndpointDesc;
                EndpointAttrDieId endpointDie{};
                CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank, &desc.localEndpoint,
                    ENDPOINT_ATTR_DIE_ID, sizeof(endpointDie), &endpointDie));
                dieId = endpointDie;
                selectedLayer = layer;
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("No CCU-capable link from rank %u to rank %u", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult InvalidateSmallDescriptorCache(HcclComm comm)
{
    void *smallCtx = nullptr;
    uint64_t smallCtxSize = 0;
    const HcclResult ret = HcclEngineCtxGet(comm, SMALL_CONTEXT_TAG,
        CommEngine::COMM_ENGINE_CCU, &smallCtx, &smallCtxSize);
    if (ret == HCCL_E_NOT_FOUND) {
        return HCCL_SUCCESS;
    }
    CHK_RET(ret);
    CHK_PRT_RET(smallCtx == nullptr || smallCtxSize != sizeof(AlgResourceCtx),
        HCCL_ERROR("Invalid Small descriptor context size %llu",
            static_cast<unsigned long long>(smallCtxSize)), HCCL_E_INTERNAL);
    const DescriptorCacheState empty{};
    return HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU,
        SMALL_CONTEXT_TAG, &empty, sizeof(empty),
        offsetof(AlgResourceCtx, lastDescriptorCache));
}

HcclResult DiscoverLocalGroups(HcclComm comm, uint32_t myRank, uint32_t rankSize,
    uint32_t localLayer, std::vector<uint32_t> &groupByRank,
    std::vector<uint32_t> &allInstanceSizes)
{
    // This API intentionally returns the instance containing myRank.  Keep that
    // exact membership; remote receivers learn their dynamic sender from root
    // inside the CCU protocol instead of inventing remote-instance membership.
    uint32_t *instanceRanks = nullptr;
    uint32_t instanceRankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, localLayer, &instanceRanks,
        &instanceRankNum));
    CHK_PRT_RET(instanceRanks == nullptr || instanceRankNum == 0,
        HCCL_ERROR("RankGraph returned an empty local instance"), HCCL_E_INTERNAL);
    uint32_t canonicalLocalRank = rankSize;
    std::vector<uint8_t> authoritative(rankSize, 0);
    for (uint32_t i = 0; i < instanceRankNum; ++i) {
        CHK_PRT_RET(instanceRanks[i] >= rankSize,
            HCCL_ERROR("Invalid instance rank %u", instanceRanks[i]), HCCL_E_INTERNAL);
        authoritative[instanceRanks[i]] = 1;
        canonicalLocalRank = std::min(canonicalLocalRank, instanceRanks[i]);
    }
    CHK_PRT_RET(authoritative[myRank] == 0,
        HCCL_ERROR("Local rank missing from its layer instance"), HCCL_E_INTERNAL);
    groupByRank.resize(rankSize);
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        groupByRank[rank] = authoritative[rank] != 0 ? canonicalLocalRank : rank;
    }

    uint32_t *instanceSizes = nullptr;
    uint32_t instanceNum = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, localLayer, &instanceSizes,
        &instanceNum));
    CHK_PRT_RET(instanceSizes == nullptr || instanceNum == 0,
        HCCL_ERROR("RankGraph returned no instance sizes"), HCCL_E_INTERNAL);
    allInstanceSizes.assign(instanceSizes, instanceSizes + instanceNum);
    uint64_t rankSum = 0;
    for (uint32_t size : allInstanceSizes) {
        rankSum += size;
    }
    CHK_PRT_RET(rankSum != rankSize,
        HCCL_ERROR("Layer instance sizes do not cover the communication domain"),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param,
    const std::vector<uint32_t> &layers, ChannelGroups &groups,
    uint32_t localLayer, const std::vector<uint32_t> &groupByRank,
    uint32_t &helperTopologyValid)
{
    helperTopologyValid = layers.size() >= 2 ? 1U : 0U;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        HcclChannelDesc desc;
        uint32_t dieId = 0;
        uint32_t selectedLayer = 0;
        CHK_RET(BuildChannelDesc(comm, param.myRank, remoteRank, layers, desc, dieId,
            selectedLayer));
        const bool sameLocalGroup = groupByRank[param.myRank] == groupByRank[remoteRank];
        if ((selectedLayer == localLayer) != sameLocalGroup) {
            helperTopologyValid = 0;
        }
        CHK_PRT_RET(dieId >= MAX_CCU_DIES,
            HCCL_ERROR("Invalid CCU die %u for peer %u", dieId, remoteRank), HCCL_E_PARA);
        ChannelHandle channel{};
        CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, &desc, 1, &channel));
        groups[dieId].emplace_back(remoteRank, channel);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterScatterKernels(HcclComm comm, const OpParam &param,
    const ChannelGroups &groups, AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle{};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("Expected one CCU instance, got %u", insNum), HCCL_E_INTERNAL);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    uint32_t groupIndex = 0;
    for (const auto &entry : groups) {
        CcuKernelInfo kernelInfo{};
        const bool specializedSmallRoot = resCtx.rank12SmallSpecialized != 0 ||
            resCtx.rank4SmallSpecialized != 0;
        const char *primaryName = specializedSmallRoot ?
            "ScatterSpecializedSmallRootCcuKernelDie%u" :
            "ScatterDirectCcuKernelDie%u";
        const int written = std::snprintf(kernelInfo.kernelFuncName,
            sizeof(kernelInfo.kernelFuncName), primaryName, entry.first);
        if (written < 0 || static_cast<size_t>(written) >= sizeof(kernelInfo.kernelFuncName)) {
            (void)HcommCcuKernelRegisterEnd(insHandle);
            HCCL_ERROR("CCU kernel name does not fit");
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = specializedSmallRoot ?
            reinterpret_cast<void *>(ops_hccl::CcuRank12SmallRootKernel) :
            reinterpret_cast<void *>(ops_hccl::CcuKernel);

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgScatter>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->peerCount = static_cast<uint32_t>(entry.second.size());
        kernelArg->channelCount = kernelArg->peerCount;
        kernelArg->includeLocal = groupIndex == 0;
        for (uint32_t i = 0; i < kernelArg->peerCount; ++i) {
            kernelArg->peerRanks[i] = entry.second[i].first;
            kernelArg->channels[i] = entry.second[i].second;
        }
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            kernelArg->localGroupByRank[rank] = resCtx.localGroupByRank[rank];
        }
        kernelArg->smallSingleReady = specializedSmallRoot ? 1U : 0U;
        kernelArg->rank12FourByThree = IsFourByThreeTopology(resCtx) ? 1U : 0U;
        kernelArg->largeBlock = resCtx.largeBlock;
        kernelInfo.setKernelArg(kernelArg);

        CcuKernelHandle kernelHandle{};
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        const CcuResult registerRet = HcommCcuKernelRegister(insHandle, entry.first,
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
        if (registerRet != CCU_SUCCESS) {
            (void)HcommCcuKernelRegisterEnd(insHandle);
            HCCL_ERROR("CCU kernel register failed for die %u: %d", entry.first, registerRet);
            return ConvertCcuToHccl(registerRet);
        }
        resCtx.ccuKernels[groupIndex] = kernelHandle;
        ++resCtx.ccuKernelCount;

        if (param.rankSize == 4U || resCtx.rank12SmallSpecialized != 0) {
            const int receiverWritten = std::snprintf(kernelInfo.kernelFuncName,
                sizeof(kernelInfo.kernelFuncName), "ScatterSmallReceiverCcuKernelDie%u",
                entry.first);
            if (receiverWritten < 0 || static_cast<size_t>(receiverWritten) >=
                sizeof(kernelInfo.kernelFuncName)) {
                (void)HcommCcuKernelRegisterEnd(insHandle);
                HCCL_ERROR("CCU small receiver kernel name does not fit");
                return HCCL_E_INTERNAL;
            }
            kernelInfo.kernelFunc =
                reinterpret_cast<void *>(ops_hccl::CcuSmallReceiverKernel);
            CcuKernelHandle receiverKernelHandle{};
            const CcuResult receiverRegisterRet = HcommCcuKernelRegister(insHandle,
                entry.first, kernelInfo.kernelFuncName, kernelInfo.kernelFunc,
                kernelArgs, 1, &receiverKernelHandle);
            if (receiverRegisterRet != CCU_SUCCESS) {
                (void)HcommCcuKernelRegisterEnd(insHandle);
                HCCL_ERROR("CCU small receiver kernel register failed for die %u: %d",
                    entry.first, receiverRegisterRet);
                return ConvertCcuToHccl(receiverRegisterRet);
            }
            resCtx.smallReceiverKernels[groupIndex] = receiverKernelHandle;
            ++resCtx.smallReceiverKernelCount;
        }

        if (resCtx.helperTopologyValid != 0 &&
            resCtx.rank12SmallSpecialized == 0 &&
            !IsEightPlusFourTopology(resCtx)) {
            const int helperWritten = std::snprintf(kernelInfo.kernelFuncName,
                sizeof(kernelInfo.kernelFuncName), "ScatterHelperCcuKernelDie%u",
                entry.first);
            if (helperWritten < 0 || static_cast<size_t>(helperWritten) >=
                sizeof(kernelInfo.kernelFuncName)) {
                (void)HcommCcuKernelRegisterEnd(insHandle);
                HCCL_ERROR("CCU helper kernel name does not fit");
                return HCCL_E_INTERNAL;
            }
            kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuHelperKernel);
            CcuKernelHandle helperKernelHandle{};
            const CcuResult helperRegisterRet = HcommCcuKernelRegister(insHandle,
                entry.first, kernelInfo.kernelFuncName, kernelInfo.kernelFunc,
                kernelArgs, 1, &helperKernelHandle);
            if (helperRegisterRet != CCU_SUCCESS) {
                (void)HcommCcuKernelRegisterEnd(insHandle);
                HCCL_ERROR("CCU helper kernel register failed for die %u: %d",
                    entry.first, helperRegisterRet);
                return ConvertCcuToHccl(helperRegisterRet);
            }
            resCtx.helperKernels[groupIndex] = helperKernelHandle;
            ++resCtx.helperKernelCount;
        }

        if (IsEightPlusFourTopology(resCtx) &&
            resCtx.rank12SmallSpecialized == 0) {
            // The second mission keeps the v18 Push data direction while
            // removing the generic Helper phase/route branches.  It is
            // registered only for the exact 8+4 topology, so all other
            // topologies retain their v21/v22 algorithms.
            const int widePushWritten = std::snprintf(kernelInfo.kernelFuncName,
                sizeof(kernelInfo.kernelFuncName), "ScatterWidePushCcuKernelDie%u",
                entry.first);
            if (widePushWritten < 0 || static_cast<size_t>(widePushWritten) >=
                sizeof(kernelInfo.kernelFuncName)) {
                (void)HcommCcuKernelRegisterEnd(insHandle);
                HCCL_ERROR("CCU wide-push kernel name does not fit");
                return HCCL_E_INTERNAL;
            }
            kernelInfo.kernelFunc =
                reinterpret_cast<void *>(ops_hccl::CcuWidePushKernel);
            CcuKernelHandle widePushKernelHandle{};
            const CcuResult widePushRegisterRet = HcommCcuKernelRegister(insHandle,
                entry.first, kernelInfo.kernelFuncName, kernelInfo.kernelFunc,
                kernelArgs, 1, &widePushKernelHandle);
            if (widePushRegisterRet != CCU_SUCCESS) {
                (void)HcommCcuKernelRegisterEnd(insHandle);
                HCCL_ERROR("CCU wide-push kernel register failed for die %u: %d",
                    entry.first, widePushRegisterRet);
                return ConvertCcuToHccl(widePushRegisterRet);
            }
            resCtx.widePushKernels[groupIndex] = widePushKernelHandle;
            ++resCtx.widePushKernelCount;
        }

        ++groupIndex;
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult BuildResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    uint32_t *layerList = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerNum));
    CHK_PRT_RET(layerNum == 0 || layerList == nullptr,
        HCCL_ERROR("RankGraph returned no network layers"), HCCL_E_NOT_FOUND);
    const std::vector<uint32_t> layers(layerList, layerList + layerNum);
    uint32_t localLayer = layers.front();
    uint32_t greatestInstanceCount = 0;
    for (uint32_t layer : layers) {
        uint32_t *instanceSizes = nullptr;
        uint32_t instanceCount = 0;
        CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, layer, &instanceSizes,
            &instanceCount));
        if (instanceCount > greatestInstanceCount) {
            greatestInstanceCount = instanceCount;
            localLayer = layer;
        }
    }
    std::vector<uint32_t> groupByRank;
    std::vector<uint32_t> instanceSizes;
    CHK_RET(DiscoverLocalGroups(comm, param.myRank, param.rankSize, localLayer,
        groupByRank, instanceSizes));
    CHK_PRT_RET(groupByRank.size() != param.rankSize ||
        instanceSizes.size() > MAX_RANK_SIZE,
        HCCL_ERROR("Invalid fixed context topology sizes"), HCCL_E_INTERNAL);
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        resCtx.localGroupByRank[rank] = groupByRank[rank];
    }
    resCtx.instanceCount = static_cast<uint32_t>(instanceSizes.size());
    for (uint32_t i = 0; i < resCtx.instanceCount; ++i) {
        resCtx.instanceSizes[i] = instanceSizes[i];
    }
    const uint64_t blockBytes = param.count * sizeof(float);
    resCtx.largeBlock = blockBytes >= HELPER_THRESHOLD_BYTES ? 1U : 0U;
    resCtx.rank12SmallSpecialized =
        (IsFourByThreeTopology(resCtx) || IsEightPlusFourTopology(resCtx)) &&
        blockBytes < HELPER_THRESHOLD_BYTES ? 1U : 0U;
    resCtx.rank4SmallSpecialized =
        param.rankSize == 4U && blockBytes == 128ULL * 1024ULL ? 1U : 0U;
    ChannelGroups groups;
    CHK_RET(AcquireChannels(comm, param, layers, groups, localLayer,
        groupByRank,
        resCtx.helperTopologyValid));
    bool hasUsefulHelperInstance = false;
    for (uint32_t i = 0; i < resCtx.instanceCount; ++i) {
        const uint32_t instanceSize = resCtx.instanceSizes[i];
        const uint32_t localPeers = instanceSize == 0 ? 0 : instanceSize - 1U;
        const uint32_t remotePeers = instanceSize > param.rankSize ? 0 :
            param.rankSize - instanceSize;
        hasUsefulHelperInstance = hasUsefulHelperInstance ||
            (localPeers > 0 && remotePeers > 4 && localPeers <= remotePeers);
    }
    if (!hasUsefulHelperInstance) {
        resCtx.helperTopologyValid = 0;
    }
    CHK_PRT_RET(groups.empty() || groups.size() > MAX_CCU_DIES,
        HCCL_ERROR("Invalid CCU die group count %zu", groups.size()), HCCL_E_INTERNAL);

    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        resCtx.peerKernelGroups[rank] = std::numeric_limits<uint32_t>::max();
    }
    uint32_t groupIndex = 0;
    for (const auto &entry : groups) {
        for (const auto &peerChannel : entry.second) {
            resCtx.peerKernelGroups[peerChannel.first] = groupIndex;
        }
        ++groupIndex;
    }

    resCtx.ccuThread = param.cpuThread;
    resCtx.threads[0] = param.cpuThread;
    resCtx.threadCount = 1;
    if (groups.size() > 1) {
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU,
            static_cast<uint32_t>(groups.size() - 1U), 1, &resCtx.threads[1]));
        resCtx.threadCount = static_cast<uint32_t>(groups.size());
    }
    CHK_RET(RegisterScatterKernels(comm, param, groups, resCtx));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Only FP32 is supported, datatype=%d", static_cast<int>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("recvCount byte calculation overflows"), HCCL_E_PARA);
    const uint64_t blockBytes = recvCount * sizeof(float);
    const char *sizeTag = blockBytes < HELPER_THRESHOLD_BYTES ?
        SMALL_CONTEXT_TAG : LARGE_CONTEXT_TAG;
    const int tagWritten = std::snprintf(param.tag, sizeof(param.tag), "%s", sizeTag);
    CHK_PRT_RET(tagWritten < 0 || static_cast<size_t>(tagWritten) >= sizeof(param.tag),
        HCCL_ERROR("Scatter tag does not fit"), HCCL_E_INTERNAL);

    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    if (blockBytes >= HELPER_THRESHOLD_BYTES) {
        CHK_RET(InvalidateSmallDescriptorCache(comm));
    }
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    const HcclResult ctxRet = HcclEngineCtxGet(comm, param.tag, ccuEngine,
        &ctx, &ctxSize);
    const AlgResourceCtx *cachedResCtx = nullptr;
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    if (ctxRet == HCCL_SUCCESS) {
        CHK_PRT_RET(ctx == nullptr || ctxSize != sizeof(AlgResourceCtx),
            HCCL_ERROR("Invalid fixed engine context size %llu",
                static_cast<unsigned long long>(ctxSize)), HCCL_E_INTERNAL);
        cachedResCtx = static_cast<const AlgResourceCtx *>(ctx);
        CHK_PRT_RET(cachedResCtx->magic != AlgResourceCtx::MAGIC ||
            cachedResCtx->version != AlgResourceCtx::VERSION,
            HCCL_ERROR("Invalid fixed engine context header"), HCCL_E_INTERNAL);
        param.myRank = cachedResCtx->myRank;
        param.rankSize = cachedResCtx->rankSize;
        const int copied = std::snprintf(commName, sizeof(commName), "%s",
            cachedResCtx->commName);
        CHK_PRT_RET(copied < 0 || static_cast<size_t>(copied) >= sizeof(commName),
            HCCL_ERROR("Cached communicator name is invalid"), HCCL_E_INTERNAL);
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        CHK_PRT_RET(ctxRet != HCCL_E_NOT_FOUND,
            HCCL_ERROR("Engine context lookup failed: %d", ctxRet), ctxRet);
        CHK_RET(HcclGetRankId(comm, &param.myRank));
        CHK_RET(HcclGetRankSize(comm, &param.rankSize));
        CHK_RET(HcclGetCommName(comm, commName));
    }
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rank size %u", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.myRank >= param.rankSize || root >= param.rankSize,
        HCCL_ERROR("Invalid rank/root: myRank=%u root=%u rankSize=%u",
            param.myRank, root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(blockBytes > 0 && blockBytes >
        std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("Scatter total input byte calculation overflows"), HCCL_E_PARA);
    if (blockBytes > 0) {
        CHK_PTR_NULL(recvBuf);
        if (param.myRank == root) {
            CHK_PTR_NULL(sendBuf);
        }
    }

    // 注册本轮 DFX 元数据；不记录 token。
    HcclDfxOpInfo dfxInfo{};
    dfxInfo.opType = static_cast<uint32_t>(param.opType);
    dfxInfo.dataType = static_cast<uint32_t>(param.dataType);
    dfxInfo.dataCount = recvCount;
    dfxInfo.root = root;
    dfxInfo.engine = COMM_ENGINE_CCU;
    dfxInfo.inputMemAddr = reinterpret_cast<uint64_t>(sendBuf);
    dfxInfo.inputMemSize = blockBytes * param.rankSize;
    dfxInfo.outputMemAddr = reinterpret_cast<uint64_t>(recvBuf);
    dfxInfo.outputMemSize = blockBytes;
    (void)std::snprintf(dfxInfo.algTag, sizeof(dfxInfo.algTag), "%s", param.tag);
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    if (blockBytes == 0) {
        return HCCL_SUCCESS;
    }

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread，并申请 1 个 notify
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    if (cachedResCtx == nullptr) {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;
        resCtxHost.myRank = param.myRank;
        resCtxHost.rankSize = param.rankSize;
        const int nameWritten = std::snprintf(resCtxHost.commName,
            sizeof(resCtxHost.commName), "%s", commName);
        CHK_PRT_RET(nameWritten < 0 || static_cast<size_t>(nameWritten) >=
            sizeof(resCtxHost.commName), HCCL_ERROR("Communicator name does not fit"),
            HCCL_E_INTERNAL);
        CHK_RET(BuildResources(comm, param, resCtxHost));

        param.ctxSize = sizeof(resCtxHost);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, &resCtxHost,
            sizeof(resCtxHost), 0));
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(comm, param, blockBytes));
    return HCCL_SUCCESS;
}
