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
#include <memory>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"

namespace ops_hccl {
CcuResult CcuKernel(CcuKernelArg arg);
}

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

HcclResult AcquireAllReduceChannels(HcclComm comm, const OpParam &param,
    std::vector<ChannelHandle> &channels)
{
    channels.resize(param.rankSize - 1);
    uint32_t channelIndex = 0;
    uint32_t *layerList = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerCount));
    CHK_PRT_RET(layerList == nullptr || layerCount == 0,
        HCCL_ERROR("Communication domain contains no network layer"), HCCL_E_NOT_FOUND);
    const std::vector<uint32_t> netLayers(layerList, layerList + layerCount);
    std::vector<uint32_t> orderedLayers;
    std::vector<uint32_t> fallbackLayers;
    orderedLayers.reserve(netLayers.size());
    fallbackLayers.reserve(netLayers.size());

    // A Clos layer spans the complete communication domain and all of its
    // local endpoints are attached to the same high-bandwidth CCU die.  If
    // Mesh links are selected first for local peers while Clos links are used
    // for remote-server peers, a single kernel contains channels from two
    // dies and registration fails.  Prefer Clos for every peer, then retain
    // the remaining layers as a fallback for communication domains without a
    // usable Clos link.
    for (const uint32_t netLayer : netLayers) {
        CommTopo topoType = CommTopo::COMM_TOPO_RESERVED;
        if (HcclRankGraphGetTopoTypeByLayer(comm, netLayer, &topoType) == HCCL_SUCCESS &&
            topoType == CommTopo::COMM_TOPO_CLOS) {
            orderedLayers.emplace_back(netLayer);
        } else {
            fallbackLayers.emplace_back(netLayer);
        }
    }
    orderedLayers.insert(orderedLayers.end(), fallbackLayers.begin(), fallbackLayers.end());

    // Select one die that has a CTP path to every peer.  A Clos endpoint is
    // normally common to the whole domain; querying the endpoint attribute
    // also makes the choice independent of link-list ordering.
    std::vector<uint32_t> commonDieIds;
    bool firstRemoteRank = true;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        std::vector<uint32_t> peerDieIds;
        for (const uint32_t netLayer : orderedLayers) {
            uint32_t linkCount = 0;
            CommLink *links = nullptr;
            if (HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &links, &linkCount) != HCCL_SUCCESS) {
                continue;
            }
            for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
                const CommLink &link = links[linkIndex];
                if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }
                EndpointAttrDieId dieId = 0;
                if (HcclRankGraphGetEndpointInfo(comm, param.myRank, &link.srcEndpointDesc,
                        EndpointAttr::ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId) != HCCL_SUCCESS) {
                    continue;
                }
                if (std::find(peerDieIds.begin(), peerDieIds.end(), dieId) == peerDieIds.end()) {
                    peerDieIds.emplace_back(dieId);
                }
            }
        }
        CHK_PRT_RET(peerDieIds.empty(),
            HCCL_ERROR("No CCU die has a UBC_CTP path from rank %u to rank %u", param.myRank, remoteRank),
            HCCL_E_NOT_FOUND);

        if (firstRemoteRank) {
            commonDieIds = peerDieIds;
            firstRemoteRank = false;
        } else {
            commonDieIds.erase(std::remove_if(commonDieIds.begin(), commonDieIds.end(),
                [&peerDieIds](uint32_t dieId) {
                    return std::find(peerDieIds.begin(), peerDieIds.end(), dieId) == peerDieIds.end();
                }), commonDieIds.end());
        }
    }
    CHK_PRT_RET(commonDieIds.empty(),
        HCCL_ERROR("No single CCU die can reach every peer in the communication domain"), HCCL_E_NOT_FOUND);
    const uint32_t selectedDieId = commonDieIds.front();

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        bool foundCtp = false;
        for (const uint32_t netLayer : orderedLayers) {
            uint32_t linkCount = 0;
            CommLink *links = nullptr;
            if (HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &links, &linkCount) != HCCL_SUCCESS) {
                continue;
            }
            for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
                const CommLink &link = links[linkIndex];
                if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }
                EndpointAttrDieId dieId = 0;
                if (HcclRankGraphGetEndpointInfo(comm, param.myRank, &link.srcEndpointDesc,
                        EndpointAttr::ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId) != HCCL_SUCCESS ||
                    dieId != selectedDieId) {
                    continue;
                }

                desc.remoteRank = remoteRank;
                desc.notifyNum = CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                foundCtp = true;
                break;
            }
            if (foundCtp) {
                break;
            }
        }

        CHK_PRT_RET(!foundCtp,
            HCCL_ERROR("UBC_CTP link is unavailable between rank %u and rank %u", param.myRank, remoteRank),
            HCCL_E_NOT_FOUND);
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channels[channelIndex]));
        ++channelIndex;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterAllReduceKernel(HcclComm comm, const OpParam &param,
    const std::vector<ChannelHandle> &channels, AlgResourceCtx &resCtx)
{
    CcuKernelInfo kernelInfo{};
    const int nameLength = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
        "%s", "CcuAllReduceKernel");
    CHK_PRT_RET(nameLength <= 0 || static_cast<uint64_t>(nameLength) >= sizeof(kernelInfo.kernelFuncName),
        HCCL_ERROR("Failed to set CCU kernel name"), HCCL_E_INTERNAL);
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);

    auto kernelArg = std::make_shared<CcuKernelArgAllReduce>();
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceType = param.reduceType;
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        kernelArg->channels[index] = channels[index];
    }
    kernelInfo.setKernelArg(kernelArg);

    CcuInsHandle insHandle = 0;
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    CHK_PRT_RET(insCount != 1,
        HCCL_ERROR("Expected one CCU instruction engine, got %u", insCount), HCCL_E_INTERNAL);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    constexpr uint32_t dieId = 0;
    const void *kernelArgs[] = {kernelInfo.kernelArg};
    CcuKernelHandle kernelHandle = 0;
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
        kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));

    resCtx.ccuKernels = {kernelHandle};
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

    // 构造算子参数
    OpParam param;
    const int tagLength = std::snprintf(param.tag, sizeof(param.tag), "hccl_custom_allreduce_%u_%u",
        static_cast<uint32_t>(dataType), static_cast<uint32_t>(op));
    CHK_PRT_RET(tagLength <= 0 || static_cast<uint64_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("Failed to build AllReduce tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.reduceType = op;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid communication domain: rank %u, rank size %u", param.myRank, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type %u", static_cast<uint32_t>(dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM && op != HCCL_REDUCE_PROD && op != HCCL_REDUCE_MAX &&
            op != HCCL_REDUCE_MIN,
        HCCL_ERROR("Unsupported reduction operation %u", static_cast<uint32_t>(op)), HCCL_E_NOT_SUPPORT);

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread。Kernel 仅使用 channel notify。
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 0, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // CCU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;
        resCtxHost.ccuThread = param.cpuThread;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源：Thread、Channel、CCU Kernel
        // ==============================================

        resCtxHost.threads = {param.cpuThread};

        if (param.rankSize > 1) {
            std::vector<ChannelHandle> kernelChannels;
            CHK_RET(AcquireAllReduceChannels(comm, param, kernelChannels));
            CHK_RET(RegisterAllReduceKernel(comm, param, kernelChannels, resCtxHost));
        }

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 CCU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
