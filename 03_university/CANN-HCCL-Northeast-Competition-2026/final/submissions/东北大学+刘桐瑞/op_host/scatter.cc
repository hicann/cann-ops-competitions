/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>

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
#if SCATTER_DIAG_LOG
#define SCATTER_DIAG(fmt, ...) HCCL_ERROR(fmt, ##__VA_ARGS__)
#else
#define SCATTER_DIAG(fmt, ...) HCCL_INFO(fmt, ##__VA_ARGS__)
#endif

bool IsPeerInMyServer(HcclComm comm, uint32_t layer, uint32_t myRank, uint32_t peer, bool &queryOk)
{
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    if (HcclRankGraphGetLinks(comm, layer, myRank, peer, &links, &linkNum) != HCCL_SUCCESS) {
        queryOk = false;
        return false;
    }
    return (links != nullptr) && (linkNum > 0);
}

void BuildServerId(HcclComm comm, uint32_t myRank, uint32_t rankSize, std::vector<uint32_t> &serverId)
{
    serverId.assign(rankSize, 0U);

    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    if (HcclRankGraphGetLayers(comm, &layers, &layerNum) != HCCL_SUCCESS || layers == nullptr || layerNum == 0) {
        SCATTER_DIAG("[BuildServerId] GetLayers unavailable, fallback to single-server view (relay off)");
        return;
    }
    const uint32_t level0 = layers[0];

    uint32_t *sizeList = nullptr;
    uint32_t listSize = 0;
    if (HcclRankGraphGetInstSizeListByLayer(comm, level0, &sizeList, &listSize) != HCCL_SUCCESS || sizeList == nullptr
        || listSize == 0) {
        SCATTER_DIAG("[BuildServerId] GetInstSizeList unavailable, fallback to single-server view (relay off)");
        return;
    }
    const std::vector<uint32_t> sizes(sizeList, sizeList + listSize);
    uint64_t total = 0;
    for (uint32_t idx = 0; idx < sizes.size(); idx++) {
        total += sizes[idx];
    }
    if (total != static_cast<uint64_t>(rankSize)) {
        SCATTER_DIAG("[BuildServerId] inst size sum[%llu] != rankSize[%u], fallback to single-server (relay off)",
            static_cast<unsigned long long>(total), rankSize);
        return;
    }
    if (listSize < 2) {
        SCATTER_DIAG("[BuildServerId] single layer0 instance, no cross-server peer (relay off)");
        return;
    }

    if (listSize == 2) {
        bool queryOk = true;
        std::vector<uint32_t> label(rankSize, 1U);
        label[myRank] = 0U;
        uint32_t mySize = 1;
        for (uint32_t r = 0; r < rankSize; r++) {
            if (r == myRank) {
                continue;
            }
            if (IsPeerInMyServer(comm, level0, myRank, r, queryOk)) {
                label[r] = 0U;
                mySize++;
            }
            if (!queryOk) {
                SCATTER_DIAG("[BuildServerId] GetLinks(%u,%u,%u) failed, fallback to single-server (relay off)",
                    level0, myRank, r);
                return;
            }
        }

        if (mySize != sizes[0] && mySize != sizes[1]) {
            SCATTER_DIAG("[BuildServerId] mySize[%u] not in instSizes[%u,%u], fallback to single-server (relay off)",
                mySize, sizes[0], sizes[1]);
            return;
        }
        serverId = label;
        SCATTER_DIAG("[BuildServerId] two-instance exact split: layer0[%u] myRank[%u] mySize[%u]", level0, myRank,
            mySize);
        return;
    }

    for (uint32_t idx = 1; idx < sizes.size(); idx++) {
        if (sizes[idx] != sizes[0]) {
            SCATTER_DIAG("[BuildServerId] %u non-uniform instances, fallback to single-server (relay off)", listSize);
            return;
        }
    }
    const uint32_t groupSize = sizes[0];
    if (groupSize == 0) {
        return;
    }
    for (uint32_t r = 0; r < rankSize; r++) {
        serverId[r] = r / groupSize;
    }
    SCATTER_DIAG("[BuildServerId] uniform block split: layer0[%u] instNum[%u] instSize[%u]", level0, listSize,
        groupSize);
}

#if SCATTER_SPREAD_CLOS_DIES

int CmpEndpointDesc(const EndpointDesc &a, const EndpointDesc &b)
{
    if (a.loc.device.devPhyId != b.loc.device.devPhyId) {
        return (a.loc.device.devPhyId < b.loc.device.devPhyId) ? -1 : 1;
    }
    if (a.commAddr.type != b.commAddr.type) {
        return (static_cast<int>(a.commAddr.type) < static_cast<int>(b.commAddr.type)) ? -1 : 1;
    }
    return std::memcmp(a.commAddr.raws, b.commAddr.raws, sizeof(a.commAddr.raws));
}

bool UndirectedLinkLess(const CommLink &x, const CommLink &y)
{
    const EndpointDesc *xLo = &x.srcEndpointDesc;
    const EndpointDesc *xHi = &x.dstEndpointDesc;
    if (CmpEndpointDesc(*xLo, *xHi) > 0) {
        const EndpointDesc *tmp = xLo;
        xLo = xHi;
        xHi = tmp;
    }
    const EndpointDesc *yLo = &y.srcEndpointDesc;
    const EndpointDesc *yHi = &y.dstEndpointDesc;
    if (CmpEndpointDesc(*yLo, *yHi) > 0) {
        const EndpointDesc *tmp = yLo;
        yLo = yHi;
        yHi = tmp;
    }
    const int cLo = CmpEndpointDesc(*xLo, *yLo);
    if (cLo != 0) {
        return cLo < 0;
    }
    return CmpEndpointDesc(*xHi, *yHi) < 0;
}
#endif

void ApplyLinkToDesc(uint32_t remoteRank, const CommLink &link, HcclChannelDesc &desc)
{
    desc.remoteRank = remoteRank;
    desc.notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
}

HcclResult FillChannelDesc(HcclComm comm, const std::vector<uint32_t> &layers, uint32_t myRank, uint32_t remoteRank,
    HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    for (uint32_t li = 0; li < layers.size(); li++) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        if (HcclRankGraphGetLinks(comm, layers[li], myRank, remoteRank, &linkList, &listSize) != HCCL_SUCCESS) {
            continue;
        }
        if (linkList == nullptr || listSize == 0) {
            continue;
        }

        std::vector<CommLink> candidates;
        candidates.reserve(listSize);
        for (uint32_t idx = 0; idx < listSize; idx++) {
            if (linkList[idx].linkAttr.linkProtocol == SCATTER_CCU_PROTOCOL) {
                candidates.push_back(linkList[idx]);
            }
        }
        if (candidates.empty()) {
            continue;
        }
        uint32_t pick = 0;
#if SCATTER_SPREAD_CLOS_DIES
        if (candidates.size() > 1) {
            std::sort(candidates.begin(), candidates.end(), UndirectedLinkLess);
            pick = (myRank + remoteRank) % static_cast<uint32_t>(candidates.size());
        }
#endif
        ApplyLinkToDesc(remoteRank, candidates[pick], desc);
        SCATTER_DIAG("[FillChannelDesc] rank %u -> %u layer[%u] ubCtp[%u] pick[%u/%u]", myRank, remoteRank, layers[li],
            static_cast<uint32_t>(candidates.size()), pick, static_cast<uint32_t>(candidates.size()));
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[FillChannelDesc] no UB_CTP link between rank %u and rank %u", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult GetChannelDieId(HcclComm comm, uint32_t myRank, const HcclChannelDesc &desc, uint32_t &dieId)
{
    EndpointAttrDieId tmpDieId{};
    const uint32_t infoLen = sizeof(EndpointAttrDieId);
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank, &(desc.localEndpoint), ENDPOINT_ATTR_DIE_ID, infoLen,
        &tmpDieId));
    dieId = tmpDieId;
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, uint32_t myRank, uint32_t rankSize, std::vector<ChannelHandle> &channels,
    std::vector<uint32_t> &channelDie)
{
    uint32_t *layerPtr = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerPtr, &layerNum));
    CHK_PTR_NULL(layerPtr);
    const std::vector<uint32_t> layers(layerPtr, layerPtr + layerNum);

    std::vector<HcclChannelDesc> descs;
    descs.resize(rankSize - 1);
    for (uint32_t c = 0; c + 1 < rankSize; c++) {
        const uint32_t remoteRank = ScatterPeerRankOf(c, myRank);
        CHK_RET(FillChannelDesc(comm, layers, myRank, remoteRank, descs[c]));
    }

    channels.assign(descs.size(), static_cast<ChannelHandle>(0));
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, descs.data(), static_cast<uint32_t>(descs.size()), channels.data()));

    channelDie.assign(descs.size(), 0U);
    for (uint32_t c = 0; c < descs.size(); c++) {
        uint32_t dieId = 0;
        CHK_RET(GetChannelDieId(comm, myRank, descs[c], dieId));
        CHK_PRT_RET(dieId >= SCATTER_DIE_NUM,
            HCCL_ERROR("[AcquireChannels] channel[%u] dieId[%u] out of range", c, dieId), HCCL_E_INTERNAL);
        channelDie[c] = dieId;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterCcuKernels(HcclComm comm, uint32_t myRank, uint32_t rankSize, AlgResourceCtx &ctx)
{
    const std::vector<ChannelHandle> &channels = ctx.channels;
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("[RegisterCcuKernels] insNum[%u] invalid", insNum), HCCL_E_INTERNAL);

    const char *kernelNames[SCATTER_KIND_NUM] = {"ScatterCcuPullRoot", "ScatterCcuPush"};
    void *kernelFuncs[SCATTER_KIND_NUM]
        = {reinterpret_cast<void *>(ops_hccl::CcuKernelPullRoot), reinterpret_cast<void *>(ops_hccl::CcuKernelPush)};

    const size_t kernelSlotNum = static_cast<size_t>(SCATTER_KIND_NUM) * SCATTER_DIE_NUM;
    ctx.ccuKernels.assign(kernelSlotNum, static_cast<CcuKernelHandle>(0));
    ctx.argNum.assign(kernelSlotNum, 0U);

    std::vector<std::vector<uint32_t>> dieChannelIdx;
    ScatterGroupChannelsByDie(myRank, ctx.channelDie, ctx.serverId, dieChannelIdx);

    const uint32_t ownCopyDie = ScatterOwnCopyDie(dieChannelIdx);

    std::vector<std::vector<ChannelHandle>> dieChannels(SCATTER_DIE_NUM);
    for (uint32_t die = 0; die < SCATTER_DIE_NUM; die++) {
        for (uint32_t idx : dieChannelIdx[die]) {
            dieChannels[die].push_back(channels[idx]);
        }
    }

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterCcuKernels] register start failed, ccuRet[%d]", ccuRet);
        return ConvertCcuToHccl(ccuRet);
    }

    constexpr uint32_t kernelArgNum = 1;

    std::vector<CcuKernelInfo> kernelInfos(kernelSlotNum);
    for (uint32_t die = 0; die < SCATTER_DIE_NUM && ccuRet == CCU_SUCCESS; die++) {
        if (dieChannels[die].empty()) {
            continue;
        }
        for (uint32_t kind = 0; kind < SCATTER_KIND_NUM && ccuRet == CCU_SUCCESS; kind++) {
            const size_t slot = static_cast<size_t>(kind) * SCATTER_DIE_NUM + die;

            auto kernelArg = std::make_shared<ScatterKernelArg>();
            kernelArg->rankSize = rankSize;
            kernelArg->rankId = myRank;
            kernelArg->handleSelfRank = (die == ownCopyDie) ? 1U : 0U;
            kernelArg->channelCount = static_cast<uint32_t>(dieChannels[die].size());
            for (uint32_t i = 0; i < dieChannels[die].size(); i++) {
                kernelArg->channels[i] = dieChannels[die][i];
            }

            kernelArg->pubChannelCount = 0;
            const uint32_t mySid = (myRank < ctx.serverId.size()) ? ctx.serverId[myRank] : 0U;

            uint32_t localNum = 0;
            for (uint32_t r = 0; r < rankSize; r++) {
                const uint32_t sid = (r < ctx.serverId.size()) ? ctx.serverId[r] : 0U;
                if (sid == mySid) {
                    localNum++;
                }
            }
            const uint32_t relayCap = (localNum >= 1U) ? (localNum - 1U) : 0U;

            uint32_t remoteSeq[MAX_RANK_SIZE] = {0};
            uint32_t remoteCnt = 0;
            for (uint32_t r = 0; r < rankSize; r++) {
                const uint32_t sid = (r < ctx.serverId.size()) ? ctx.serverId[r] : 0U;
                if (sid != mySid) {
                    remoteSeq[r] = remoteCnt++;
                }
            }

            kernelArg->crossNum = 0;
            for (uint32_t i = 0; i < dieChannelIdx[die].size(); i++) {
                const uint32_t peer = ScatterPeerRankOf(dieChannelIdx[die][i], myRank);
                const uint32_t sid = (peer < ctx.serverId.size()) ? ctx.serverId[peer] : 0U;
                if (sid == mySid) {
                    kernelArg->pubChannelIdx[kernelArg->pubChannelCount++] = i;
                    kernelArg->wrUseRelaySize[i] = 0U;
                } else {
                    kernelArg->crossNum++;

                    kernelArg->wrUseRelaySize[i] = (remoteSeq[peer] < relayCap) ? 1U : 0U;
                }
            }
            kernelInfos[slot].setKernelArg(kernelArg);
            const void *kernelArgs[] = {kernelInfos[slot].kernelArg};

            CcuKernelHandle handle = 0;
            ccuRet = HcommCcuKernelRegister(
                insHandle, die, kernelNames[kind], kernelFuncs[kind], kernelArgs, kernelArgNum, &handle);
            if (ccuRet != CCU_SUCCESS) {
                HCCL_ERROR("[RegisterCcuKernels] register kind[%u] die[%u] failed, ccuRet[%d]"
                           " (7 = CCU_E_UNAVAIL, at most %u kernels per die)",
                    kind, die, ccuRet, SCATTER_MAX_KERNEL_PER_DIE);
                break;
            }
            ctx.ccuKernels[slot] = handle;
        }
    }

    const CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuRet);
    }
    if (endRet != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterCcuKernels] register end failed, ccuRet[%d]", endRet);
        return ConvertCcuToHccl(endRet);
    }
    return HCCL_SUCCESS;
}

HcclResult BuildAlgResource(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    resCtxHost.localBufferToken = 0;
    if (cclBufferAddr != nullptr && cclBufferSize > 0) {
        const uint64_t bufVa = reinterpret_cast<uint64_t>(cclBufferAddr);
        const CcuResult tokenRet = HcommCcuGetMemToken(bufVa, cclBufferSize, &resCtxHost.localBufferToken);
        if (tokenRet != CCU_SUCCESS) {
            HCCL_ERROR("[BuildAlgResource] hccl buffer token unavailable, ccuRet[%d]", tokenRet);
            resCtxHost.localBufferToken = 0;
        }
    } else {
        resCtxHost.localBuffer.size = 0;
    }

    resCtxHost.ccuThread = param.cpuThread;
    resCtxHost.threads.assign(static_cast<size_t>(SCATTER_DIE_NUM), static_cast<ThreadHandle>(0));
    resCtxHost.threads[0] = param.cpuThread;
    if (SCATTER_DIE_NUM > 1) {
        constexpr uint32_t notifyNumPerThread = 1;
        CHK_RET(HcclThreadAcquire(
            comm, CommEngine::COMM_ENGINE_CCU, SCATTER_DIE_NUM - 1, notifyNumPerThread, &resCtxHost.threads[1]));

    }

    resCtxHost.myRank = param.myRank;
    resCtxHost.rankSize = param.rankSize;

    BuildServerId(comm, param.myRank, param.rankSize, resCtxHost.serverId);
    CHK_RET(AcquireChannels(comm, param.myRank, param.rankSize, resCtxHost.channels, resCtxHost.channelDie));

    CHK_RET(RegisterCcuKernels(comm, param.myRank, param.rankSize, resCtxHost));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    static constexpr char SCATTER_OP_TAG[] = "hccl_custom_scatter";
    static_assert(sizeof(SCATTER_OP_TAG) <= TAG_LENGTH, "op tag too long");
    OpParam param;
    memcpy(param.tag, SCATTER_OP_TAG, sizeof(SCATTER_OP_TAG));
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    static thread_local HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    bool ctxHit = false;
#if SCATTER_CACHE_COMM_INFO

    ctxHit = (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &ctxSize) == HCCL_SUCCESS);
    const bool rankInfoHit = ctxHit && ops_hccl::TryGetCachedRankInfo(ctx, ctxSize, param.myRank, param.rankSize);
#else
    const bool rankInfoHit = false;
#endif
    if (!rankInfoHit) {
        CHK_RET(HcclGetRankId(comm, &param.myRank));
        CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    }

    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("[HcclScatter] invalid root[%u], rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(static_cast<uint64_t>(param.rankSize) > MAX_RANK_SIZE,
        HCCL_ERROR("[HcclScatter] rankSize[%u] exceeds limit", param.rankSize), HCCL_E_NOT_SUPPORT);

    const uint32_t perDataSize = ScatterDataTypeSize(param.dataType);
    CHK_PRT_RET(perDataSize == 0, HCCL_ERROR("[HcclScatter] unsupported dataType[%d]", param.dataType),
        HCCL_E_NOT_SUPPORT);

    if (param.rankSize == 1) {
        const uint64_t dataSize = recvCount * perDataSize;
        if (dataSize == 0 || sendBuf == recvBuf) {
            return HCCL_SUCCESS;
        }
        ACLCHECK(aclrtMemcpyAsync(recvBuf, dataSize, sendBuf, dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        return HCCL_SUCCESS;
    }

    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

#if !SCATTER_CACHE_COMM_INFO

    ctxHit = (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &ctxSize) == HCCL_SUCCESS);
#endif

    if (ctxHit) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx resCtxHost;

        CHK_RET(BuildAlgResource(comm, param, resCtxHost));

        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
