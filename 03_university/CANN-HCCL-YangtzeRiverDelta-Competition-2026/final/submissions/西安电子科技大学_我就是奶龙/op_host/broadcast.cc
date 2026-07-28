/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include <ccu/ccu_launch.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t THREAD_NOTIFY_NUM = 2;
constexpr uint64_t ONE_SHOT_MAX_BYTES = 512ULL * 1024ULL;
constexpr char ALG_TAG[] = "hccl_custom_broadcast_allpeer_v3_r2";

struct ChannelGroup {
    uint32_t dieId{0};
    std::vector<uint32_t> peerRanks;
    std::vector<ChannelHandle> channels;
};

void GetRankSlice(uint64_t bytes, uint32_t rankSize, uint32_t rank, uint64_t &offset, uint64_t &sliceBytes)
{
    const uint64_t elements = bytes / sizeof(float);
    const uint64_t baseElements = elements / rankSize;
    const uint64_t remainder = elements % rankSize;
    const uint64_t offsetElements = static_cast<uint64_t>(rank) * baseElements + std::min<uint64_t>(rank, remainder);
    const uint64_t rankElements = baseElements + (rank < remainder ? 1ULL : 0ULL);
    offset = offsetElements * sizeof(float);
    sliceBytes = rankElements * sizeof(float);
}

std::vector<BroadcastWindow> BuildWindows(uint64_t totalBytes)
{
    std::vector<BroadcastWindow> windows;
    if (totalBytes <= ONE_SHOT_MAX_BYTES) {
        windows.push_back(BroadcastWindow{0, totalBytes});
        return windows;
    }
    for (uint64_t offset = 0; offset < totalBytes; offset += BROADCAST_WINDOW_BYTES) {
        windows.push_back(BroadcastWindow{offset, std::min(BROADCAST_WINDOW_BYTES, totalBytes - offset)});
    }
    return windows;
}

HcclResult FillChannelDesc(uint32_t remoteRank, const CommLink &link, HcclChannelDesc &desc)
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

HcclResult SelectChannelDesc(
    HcclComm comm, uint32_t localRank, uint32_t remoteRank, HcclChannelDesc &desc, uint32_t &dieId)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    CHK_PRT_RET(layers == nullptr || layerNum == 0,
        HCCL_ERROR("[BroadcastV3] rank graph has no network layer"), HCCL_E_INTERNAL);

    constexpr CommProtocol protocols[] = {COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UBC_TP};
    for (uint32_t layerIdx = 0; layerIdx < layerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layers[layerIdx], localRank, remoteRank, &links, &linkNum));
        for (const CommProtocol protocol : protocols) {
            for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                if (links[linkIdx].linkAttr.linkProtocol != protocol) {
                    continue;
                }
                CHK_RET(FillChannelDesc(remoteRank, links[linkIdx], desc));
                EndpointAttrDieId endpointDieId = 0;
                CHK_RET(HcclRankGraphGetEndpointInfo(comm, localRank, &desc.localEndpoint,
                    ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId), &endpointDieId));
                dieId = endpointDieId;
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[BroadcastV3] no CCU-capable link from rank[%u] to rank[%u]", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireAllPeerChannels(HcclComm comm, const OpParam &param, std::vector<ChannelGroup> &groups)
{
    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }
        HcclChannelDesc desc{};
        uint32_t dieId = 0;
        CHK_RET(SelectChannelDesc(comm, param.myRank, peerRank, desc, dieId));
        ChannelHandle channel{};
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));

        auto groupIt = std::find_if(groups.begin(), groups.end(),
            [dieId](const ChannelGroup &group) { return group.dieId == dieId; });
        if (groupIt == groups.end()) {
            groups.push_back(ChannelGroup{dieId, {}, {}});
            groupIt = groups.end() - 1;
        }
        groupIt->peerRanks.push_back(peerRank);
        groupIt->channels.push_back(channel);
    }
    std::sort(groups.begin(), groups.end(),
        [](const ChannelGroup &lhs, const ChannelGroup &rhs) { return lhs.dieId < rhs.dieId; });
    CHK_PRT_RET(groups.empty() || groups.size() > 2,
        HCCL_ERROR("[BroadcastV3] expected one or two DIE groups, got[%llu]",
            static_cast<unsigned long long>(groups.size())), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelGroup> &groups,
    AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[BroadcastV3] expected one CCU instruction engine, got[%u]", insNum), HCCL_E_INTERNAL);

    uint32_t rankKernelIndices[MAX_RANK_SIZE]{};
    std::fill(rankKernelIndices, rankKernelIndices + MAX_RANK_SIZE, INVALID_VALUE_RANKID);
    for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        for (uint32_t peerRank : groups[groupIndex].peerRanks) {
            rankKernelIndices[peerRank] = groupIndex;
        }
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    std::vector<CcuKernelInfo> kernelInfos(groups.size());
    resCtx.ccuKernels.resize(groups.size());
    resCtx.kernelMeta.resize(groups.size());
    for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const ChannelGroup &group = groups[groupIndex];
        auto kernelArg = std::make_shared<CcuKernelArgBroadcast>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->kernelIndex = groupIndex;
        kernelArg->kernelCount = groups.size();
        kernelArg->mode = resCtx.mode;
        kernelArg->windowCount = resCtx.windows.size();
        kernelArg->channelCount = group.channels.size();
        std::copy(rankKernelIndices, rankKernelIndices + MAX_RANK_SIZE, kernelArg->rankKernelIndices);

        BroadcastKernelMeta &meta = resCtx.kernelMeta[groupIndex];
        meta.peerCount = group.peerRanks.size();
        for (uint32_t channelIndex = 0; channelIndex < group.channels.size(); ++channelIndex) {
            kernelArg->channels[channelIndex] = group.channels[channelIndex];
            kernelArg->peerRanks[channelIndex] = group.peerRanks[channelIndex];
            meta.peerRanks[channelIndex] = group.peerRanks[channelIndex];
        }
        for (uint32_t window = 0; window < resCtx.windows.size(); ++window) {
            kernelArg->windowBytes[window] = resCtx.windows[window].bytes;
            GetRankSlice(resCtx.windows[window].bytes, param.rankSize, param.myRank,
                kernelArg->ownerSliceOffsets[window], kernelArg->ownerSliceBytes[window]);
            for (uint32_t channelIndex = 0; channelIndex < group.peerRanks.size(); ++channelIndex) {
                GetRankSlice(resCtx.windows[window].bytes, param.rankSize, group.peerRanks[channelIndex],
                    kernelArg->peerSliceOffsets[window][channelIndex],
                    kernelArg->peerSliceBytes[window][channelIndex]);
            }
        }

        CcuKernelInfo &kernelInfo = kernelInfos[groupIndex];
        const int nameLen = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "CcuBroadcastV3_r%u_d%u_m%u", param.myRank, group.dieId, resCtx.mode);
        CHK_PRT_RET(nameLen < 0 || static_cast<size_t>(nameLen) >= sizeof(kernelInfo.kernelFuncName),
            HCCL_ERROR("[BroadcastV3] failed to initialize CCU kernel name"), HCCL_E_INTERNAL);
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);
        kernelInfo.setKernelArg(kernelArg);
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, group.dieId, kernelInfo.kernelFuncName,
            kernelInfo.kernelFunc, kernelArgs, 1, &resCtx.ccuKernels[groupIndex]));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult BuildResources(HcclComm comm, const OpParam &param, uint64_t totalBytes, AlgResourceCtx &resCtx)
{
    resCtx.mode = static_cast<uint32_t>(totalBytes <= ONE_SHOT_MAX_BYTES
        ? BroadcastMode::ONE_SHOT : BroadcastMode::SCATTER_ALLGATHER);
    resCtx.windows = BuildWindows(totalBytes);
    CHK_PRT_RET(resCtx.windows.empty() || resCtx.windows.size() > BROADCAST_MAX_WINDOWS,
        HCCL_ERROR("[BroadcastV3] message requires unsupported window count[%llu]",
            static_cast<unsigned long long>(resCtx.windows.size())), HCCL_E_NOT_SUPPORT);

    std::vector<ChannelGroup> groups;
    CHK_RET(AcquireAllPeerChannels(comm, param, groups));
    if (groups.size() > 1) {
        resCtx.extraThreads.resize(groups.size() - 1);
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU,
            resCtx.extraThreads.size(), THREAD_NOTIFY_NUM, resCtx.extraThreads.data()));
    }
    CHK_RET(RegisterKernels(comm, param, groups, resCtx));
    HCCL_INFO("[BroadcastV3] resources ranks[%u] bytes[%llu] mode[%u] kernels[%llu] windows[%llu]",
        param.rankSize, static_cast<unsigned long long>(totalBytes), resCtx.mode,
        static_cast<unsigned long long>(resCtx.ccuKernels.size()),
        static_cast<unsigned long long>(resCtx.windows.size()));
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
        HCCL_ERROR("[BroadcastV3] unsupported data type[%d]", static_cast<int32_t>(dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[BroadcastV3] element count overflows byte calculation"), HCCL_E_PARA);
    if (count == 0) {
        return HCCL_SUCCESS;
    }
    const uint64_t totalBytes = count * sizeof(float);
    CHK_PRT_RET(totalBytes > BROADCAST_MAX_WINDOWS * BROADCAST_WINDOW_BYTES,
        HCCL_ERROR("[BroadcastV3] message[%llu] exceeds finals maximum[%llu]",
            static_cast<unsigned long long>(totalBytes),
            static_cast<unsigned long long>(BROADCAST_MAX_WINDOWS * BROADCAST_WINDOW_BYTES)), HCCL_E_NOT_SUPPORT);

    OpParam param{};
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[BroadcastV3] unsupported rank size[%u]", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[BroadcastV3] root[%u] is outside rank size[%u]", root, param.rankSize), HCCL_E_PARA);
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    const int tagLen = std::snprintf(param.tag, sizeof(param.tag), "%s_p%u_b%llu", ALG_TAG, param.rankSize,
        static_cast<unsigned long long>(totalBytes));
    CHK_PRT_RET(tagLen < 0 || static_cast<size_t>(tagLen) >= sizeof(param.tag),
        HCCL_ERROR("[BroadcastV3] failed to initialize operation tag"), HCCL_E_INTERNAL);

    constexpr CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, THREAD_NOTIFY_NUM, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx resCtxHost{};
        CHK_RET(BuildResources(comm, param, totalBytes, resCtxHost));
        const std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seq.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
