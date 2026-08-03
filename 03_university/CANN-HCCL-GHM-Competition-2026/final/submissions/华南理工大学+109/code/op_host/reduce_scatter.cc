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
#include <memory>
#include <vector>

#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hcomm/ccu/ccu_launch.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "ccu_kernel.h"
#include "exec_op.h"

namespace {
constexpr uint32_t kChannelNotifyNum = 1;
constexpr uint32_t kKernelArgNum = 1;

HcclResult FillChannelDesc(
    HcclComm comm, uint32_t layer, uint32_t localRank, uint32_t remoteRank, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, layer, localRank, remoteRank, &links, &linkNum));
    for (uint32_t i = 0; i < linkNum; ++i) {
        if (links[i].linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            continue;
        }
        desc.remoteRank = remoteRank;
        desc.notifyNum = kChannelNotifyNum;
        desc.channelProtocol = links[i].linkAttr.linkProtocol;
        desc.localEndpoint.protocol = links[i].srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = links[i].srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = links[i].srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = links[i].dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = links[i].dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = links[i].dstEndpointDesc.loc;
        return HCCL_SUCCESS;
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult RegisterKernel(HcclComm comm, const std::vector<ChannelHandle> &channels, HcclDataType dataType,
    HcclReduceOp reduceOp, bool initializeOutput, bool striped, CcuKernelHandle *kernel)
{
    auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgReduceScatter>();
    CHK_PRT_RET(channels.empty() || channels.size() >= MAX_RANK_SIZE, HCCL_ERROR("Invalid kernel channel resources"),
        HCCL_E_INTERNAL);
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    kernelArg->dataType = dataType;
    kernelArg->reduceOp = reduceOp;
    kernelArg->initializeOutput = initializeOutput;
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        kernelArg->channels[i] = channels[i];
    }

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum == 0, HCCL_ERROR("No CCU instruction resource is available"), HCCL_E_INTERNAL);

    const void *kernelArgs[kKernelArgNum] = {kernelArg.get()};
    CcuResult ret = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ret != CCU_SUCCESS, HCCL_ERROR("CCU kernel registration start failed: %d", ret), ConvertCcuToHccl(ret));
    const char *kernelName = striped ? "CcuReduceScatterStripedKernel" : "CcuReduceScatterReadKernel";
    const void *kernelFunc = striped ? reinterpret_cast<const void *>(ops_hccl::CcuReduceScatterStripedKernel)
                                     : reinterpret_cast<const void *>(ops_hccl::CcuReduceScatterReadKernel);
    ret = HcommCcuKernelRegister(insHandle, 0, kernelName, kernelFunc, kernelArgs, kKernelArgNum, kernel);
    CHK_PRT_RET(ret != CCU_SUCCESS, HCCL_ERROR("CCU kernel registration failed: %d", ret), ConvertCcuToHccl(ret));
    ret = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ret != CCU_SUCCESS, HCCL_ERROR("CCU kernel registration end failed: %d", ret), ConvertCcuToHccl(ret));
    return HCCL_SUCCESS;
}

HcclResult RegisterMergeKernel(HcclComm comm, ChannelHandle deploymentChannel, HcclDataType dataType,
    HcclReduceOp reduceOp, CcuKernelHandle *kernel)
{
    auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgReduceScatterMerge>();
    kernelArg->channelCount = 1;
    kernelArg->channels[0] = deploymentChannel;
    kernelArg->dataType = dataType;
    kernelArg->reduceOp = reduceOp;

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum == 0, HCCL_ERROR("No CCU instruction resource is available"), HCCL_E_INTERNAL);

    const void *kernelArgs[kKernelArgNum] = {kernelArg.get()};
    CcuResult ret = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ret != CCU_SUCCESS, HCCL_ERROR("CCU merge registration start failed: %d", ret), ConvertCcuToHccl(ret));
    ret = HcommCcuKernelRegister(insHandle, 0, "CcuReduceScatterMergeKernel",
        reinterpret_cast<const void *>(ops_hccl::CcuReduceScatterMergeKernel), kernelArgs, kKernelArgNum, kernel);
    CHK_PRT_RET(ret != CCU_SUCCESS, HCCL_ERROR("CCU merge kernel registration failed: %d", ret), ConvertCcuToHccl(ret));
    ret = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ret != CCU_SUCCESS, HCCL_ERROR("CCU merge registration end failed: %d", ret), ConvertCcuToHccl(ret));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_reducescatter_striped_v8");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

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
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rank size %u", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type %d", dataType),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op == HCCL_REDUCE_RESERVED, HCCL_ERROR("Invalid reduction operation"), HCCL_E_PARA);

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread，并申请 1 个 notify
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

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

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源：Thread、Channel、CCU Kernel
        // ==============================================

        resCtxHost.ccuThread = param.cpuThread;
        resCtxHost.threads.push_back(param.cpuThread);

        if (param.rankSize > 1) {
            uint32_t *layerList = nullptr;
            uint32_t layerNum = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerNum));
            CHK_PRT_RET(layerNum == 0, HCCL_ERROR("Communication graph has no network layer"), HCCL_E_INTERNAL);
            // Rank-graph 查询结果由 HCCL 持有，后续的查询可能会使其失效。
            // 在请求任何对端链路或层属性之前，先保留一份稳定的副本
            const std::vector<uint32_t> layers(layerList, layerList + layerNum);

            // 一个 kernel 只能包含来自同一个 IO Die 的通道。
            // 在该拓扑中，层与 IO Die 一一对应，因此为每一层保留独立的描述符组，
            // 并对每个对端使用首个可用的层
            std::vector<std::vector<HcclChannelDesc>> descGroups(layerNum);
            for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
                if (peer == param.myRank) {
                    continue;
                }
                bool found = false;
                for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
                    HcclChannelDesc desc;
                    const HcclResult result = FillChannelDesc(comm, layers[layerIndex], param.myRank, peer, desc);
                    if (result == HCCL_E_NOT_FOUND) {
                        continue;
                    }
                    CHK_RET(result);
                    descGroups[layerIndex].push_back(desc);
                    found = true;
                    break;
                }
                CHK_PRT_RET(!found, HCCL_ERROR("No UBC_CTP link from rank %u to rank %u", param.myRank, peer),
                    HCCL_E_NOT_FOUND);
            }

            std::vector<HcclChannelDesc> descs;
            std::vector<uint32_t> descGroup;
            for (uint32_t group = 0; group < layerNum; ++group) {
                for (const HcclChannelDesc &desc : descGroups[group]) {
                    descs.push_back(desc);
                    descGroup.push_back(group);
                }
            }
            std::vector<ChannelHandle> handles(descs.size());
            CHK_RET(
                HcclChannelAcquire(comm, ccuEngine, descs.data(), static_cast<uint32_t>(descs.size()), handles.data()));

            std::vector<std::vector<ChannelHandle>> channelGroups(layerNum);
            for (uint32_t i = 0; i < handles.size(); ++i) {
                channelGroups[descGroup[i]].push_back(handles[i]);
            }
            const uint32_t activeGroupCount = static_cast<uint32_t>(
                std::count_if(channelGroups.begin(), channelGroups.end(), [](const std::vector<ChannelHandle> &group) {
                    return !group.empty();
                }));
            CHK_PRT_RET(activeGroupCount > 2, HCCL_ERROR("Communication graph uses more than two CCU network groups"),
                HCCL_E_NOT_SUPPORT);
            if (activeGroupCount == 2) {
                ThreadHandle slaveThread{};
                CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, 1, &slaveThread));
                resCtxHost.threads.push_back(slaveThread);
            }
            for (uint32_t group = 0; group < layerNum; ++group) {
                if (channelGroups[group].empty()) {
                    continue;
                }
                CcuKernelHandle kernel{};
                const bool initializeOutput = resCtxHost.ccuKernels.empty();
                CHK_RET(RegisterKernel(comm, channelGroups[group], dataType, op, initializeOutput, false, &kernel));
                resCtxHost.ccuKernels.push_back(kernel);
                CcuKernelHandle stripedKernel{};
                CHK_RET(
                    RegisterKernel(comm, channelGroups[group], dataType, op, initializeOutput, true, &stripedKernel));
                resCtxHost.stripedKernels.push_back(stripedKernel);
                resCtxHost.kernelChannelCounts.push_back(static_cast<uint32_t>(channelGroups[group].size()));
                resCtxHost.kernelThreadIndices.push_back(static_cast<uint32_t>(resCtxHost.kernelThreadIndices.size()));
            }
            if (resCtxHost.ccuKernels.size() == 2) {
                const auto firstGroup = std::find_if(
                    channelGroups.begin(), channelGroups.end(), [](const std::vector<ChannelHandle> &group) {
                        return !group.empty();
                    });
                CHK_PRT_RET(firstGroup == channelGroups.end(), HCCL_ERROR("CCU merge kernel has no deployment channel"),
                    HCCL_E_INTERNAL);
                CHK_RET(RegisterMergeKernel(comm, firstGroup->front(), dataType, op, &resCtxHost.mergeKernel));
            }
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
