/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <ccu/ccu_res.h>
#include <ccu_launch.h>
#include <hccl_ccu_res.h>

#include <algorithm>
#include <utility>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t NET_LAYER_COUNT = 2;
constexpr uint32_t DEFAULT_PIPELINE_STEPS = 4;
constexpr uint32_t FINE_PIPELINE_STEPS = 8;
constexpr uint32_t DEEP_PIPELINE_STEPS = 12;

enum class BroadcastKernelType {
    DIRECT,
    MESH,
    SCATTER,
    GATHER
};

HcclResult AcquireChannel(HcclComm comm, CommEngine engine, uint32_t localRank, uint32_t remoteRank,
    const std::vector<uint32_t> &netLayers, uint32_t &selectedLayer, ChannelHandle &channel)
{
    constexpr CommProtocol requiredProtocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;

    for (const uint32_t netLayer : netLayers) {
        uint32_t linkCount = 0;
        CommLink *links = nullptr;

        const HcclResult linkRet =
            HcclRankGraphGetLinks(comm, netLayer, localRank, remoteRank, &links, &linkCount);
        if (linkRet != HCCL_SUCCESS || links == nullptr || linkCount == 0) {
            continue;
        }

        for (uint32_t i = 0; i < linkCount; ++i) {
            const CommLink &link = links[i];
            if (link.linkAttr.linkProtocol != requiredProtocol) {
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

            CHK_RET(HcclChannelAcquire(comm, engine, &desc, 1, &channel));

            selectedLayer = netLayer;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR(
        "[AcquireChannel] UBC CTP link not found between rank %u and rank %u",
        localRank,
        remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult RegisterBroadcastKernel(HcclComm comm, const OpParam &param, uint32_t netLayer,
    const std::vector<std::pair<uint32_t, ChannelHandle>> &rankChannels, BroadcastKernelType kernelType,
    AlgResourceCtx &resCtx, bool treeEnabled = false,
    uint32_t pipelineSteps = DEFAULT_PIPELINE_STEPS)
{
    std::vector<CcuKernelHandle> *kernels = nullptr;
    const char *kernelTypeName = nullptr;
    void *kernelFunc = nullptr;

    switch (kernelType) {
        case BroadcastKernelType::DIRECT:
            kernels = &resCtx.directKernels;
            kernelTypeName = "Direct";
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuBroadcastKernel);
            break;
        case BroadcastKernelType::MESH:
            kernels = &resCtx.meshKernels;
            kernelTypeName = "Mesh";
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuBroadcastMeshKernel);
            break;
        case BroadcastKernelType::SCATTER:
            kernels = &resCtx.scatterKernels;
            kernelTypeName = "Scatter";
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuBroadcastScatterKernel);
            break;
        case BroadcastKernelType::GATHER:
            kernels = &resCtx.gatherKernels;
            kernelTypeName = "Gather";
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuBroadcastGatherKernel);
            break;
    }

    CHK_PTR_NULL(kernels);
    CHK_PRT_RET(netLayer >= kernels->size(),
        HCCL_ERROR("[RegisterBroadcastKernel] invalid net layer: %u", netLayer),
        HCCL_E_PARA);

    CcuKernelInfo kernelInfo;
    const int32_t nameResult =
        snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "CcuBroadcast%sL%uR%u", kernelTypeName, netLayer, param.root);

    CHK_PRT_RET(
        nameResult <= 0 ||
            static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName),
        HCCL_ERROR("[RegisterBroadcastKernel] failed to set kernel name"),
        HCCL_E_INTERNAL);

    kernelInfo.kernelFunc = kernelFunc;

    auto kernelArg =
        std::make_shared<ops_hccl::CcuKernelArgBroadcast>();

    kernelArg->rankId = param.myRank;
    kernelArg->rankSize = param.rankSize;
    kernelArg->root = param.root;
    kernelArg->pipelineSteps = pipelineSteps;
    kernelArg->treeEnabled = treeEnabled;
    kernelArg->channelCount = static_cast<uint32_t>(rankChannels.size());

    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        kernelArg->remoteRanks[i] = rankChannels[i].first;
        kernelArg->channels[i] = rankChannels[i].second;
    }

    kernelInfo.setKernelArg(kernelArg);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));

    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR(
            "[RegisterBroadcastKernel] unexpected CCU instance count: %u",
            insNum),
        HCCL_E_INTERNAL);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    CcuKernelHandle kernelHandle{};
    const void *kernelArgs[] = {kernelInfo.kernelArg};

    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;

    CHK_RET_CCU(HcommCcuKernelRegister(
        insHandle,
        dieId,
        kernelInfo.kernelFuncName,
        kernelInfo.kernelFunc,
        kernelArgs,
        kernelArgNum,
        &kernelHandle));

    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));

    (*kernels)[netLayer] = kernelHandle;
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t root,
    HcclComm comm,
    aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};

    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(
        commName,
        reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    CHK_PRT_RET(
        param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR(
            "[HcclBroadcast] unsupported rank size: %u",
            param.rankSize),
        HCCL_E_NOT_SUPPORT);

    CHK_PRT_RET(
        root >= param.rankSize,
        HCCL_ERROR(
            "[HcclBroadcast] root %u is outside rank size %u",
            root,
            param.rankSize),
        HCCL_E_PARA);

    const auto sizeIt = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(
        sizeIt == SIZE_TABLE.end(),
        HCCL_ERROR(
            "[HcclBroadcast] unsupported data type: %d",
            static_cast<int32_t>(dataType)),
        HCCL_E_NOT_SUPPORT);

    CHK_PRT_RET(
        count > UINT64_MAX / sizeIt->second,
        HCCL_ERROR("[HcclBroadcast] data size overflow"),
        HCCL_E_PARA);

    constexpr uint64_t smallDataThreshold = 1024 * 1024;
    const uint64_t dataSize = count * sizeIt->second;
    const bool treeEnabled =
        dataSize <= smallDataThreshold && param.rankSize >= 12;
    const char *algorithmTag =
        dataSize > smallDataThreshold ? "cursor_v1" : "single_v6";
    uint32_t pipelineSteps = DEFAULT_PIPELINE_STEPS;
    if (dataSize > smallDataThreshold) {
        if (param.rankSize == MAX_RANK_SIZE) {
            pipelineSteps = DEEP_PIPELINE_STEPS;
        } else if (param.rankSize == 4 || param.rankSize == 12) {
            pipelineSteps = FINE_PIPELINE_STEPS;
        }
    }

    std::vector<uint32_t> netLayers;

    if (param.rankSize > 1) {
        uint32_t *netLayerData = nullptr;
        uint32_t netLayerNum = 0;

        CHK_RET(HcclRankGraphGetLayers(
            comm,
            &netLayerData,
            &netLayerNum));

        CHK_PRT_RET(
            netLayerData == nullptr || netLayerNum == 0,
            HCCL_ERROR("[HcclBroadcast] no network layer found"),
            HCCL_E_NOT_FOUND);

        netLayers.assign(
            netLayerData,
            netLayerData + netLayerNum);

        for (const uint32_t netLayer : netLayers) {
            CHK_PRT_RET(
                netLayer >= NET_LAYER_COUNT,
                HCCL_ERROR(
                    "[HcclBroadcast] unsupported net layer: %u",
                    netLayer),
                HCCL_E_NOT_SUPPORT);
        }
    }

    /*
     * Kernel注册参数中包含root和固定的数据规模策略。
     * 不同参数必须使用不同的Engine Context，避免复用错误的Kernel参数。
     */
    const int32_t tagResult =
        snprintf(
            param.tag,
            sizeof(param.tag),
            "hccl_custom_broadcast_%s_root_%u_tree_%u_p%u",
            algorithmTag,
            root,
            treeEnabled ? 1U : 0U,
            pipelineSteps);

    CHK_PRT_RET(
        tagResult <= 0 ||
            static_cast<size_t>(tagResult) >= sizeof(param.tag),
        HCCL_ERROR("[HcclBroadcast] failed to set resource tag"),
        HCCL_E_INTERNAL);

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    void *ctx = nullptr;
    uint64_t size = 0;

    if (HcclEngineCtxGet(
            comm,
            param.tag,
            ccuEngine,
            &ctx,
            &size) == HCCL_SUCCESS) {
        HCCL_INFO("Engine context already exists");

        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;

        CHK_RET(HcclGetHcclBuffer(
            comm,
            &cclBufferAddr,
            &cclBufferSize));

        resCtxHost.localBuffer =
            CommBuffer{cclBufferAddr, cclBufferSize};
        resCtxHost.primaryStream = stream;

        /*
         * 主Thread绑定调用方stream；跨层拓扑会在资源注册完成后再申请辅助Thread。
         * Kernel内部会向同一层的不同对端发起异步Write。
         */
        constexpr uint32_t notifyNumOnThread = 0;

        CHK_RET(HcclThreadAcquireWithStream(
            comm,
            ccuEngine,
            stream,
            notifyNumOnThread,
            &param.cpuThread));

        resCtxHost.ccuThread = param.cpuThread;
        resCtxHost.threads.push_back(param.cpuThread);

        if (param.rankSize > 1) {
            /*
             * layer-0和layer-1位于不同IO Die。
             * 两层的channel必须注册到不同的CCU Kernel。
             */
            std::vector<std::vector<std::pair<uint32_t, ChannelHandle>>>
                channelsByLayer(NET_LAYER_COUNT);

            std::vector<uint32_t> preferredLayers;
            /* Clos带宽更高；当两rank同时存在多层链路时优先选择layer-1。 */
            for (const uint32_t preferredLayer : {1U, 0U}) {
                if (std::find(netLayers.begin(), netLayers.end(), preferredLayer) != netLayers.end()) {
                    preferredLayers.push_back(preferredLayer);
                }
            }

            uint32_t commonLayer = NET_LAYER_COUNT;
            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                if (remoteRank == param.myRank) {
                    continue;
                }

                ChannelHandle channel{};
                uint32_t selectedLayer = NET_LAYER_COUNT;

                CHK_RET(AcquireChannel(
                    comm,
                    ccuEngine,
                    param.myRank,
                    remoteRank,
                    preferredLayers,
                    selectedLayer,
                    channel));

                channelsByLayer[selectedLayer].emplace_back(remoteRank, channel);
                if (commonLayer == NET_LAYER_COUNT) {
                    commonLayer = selectedLayer;
                } else if (commonLayer != selectedLayer) {
                    commonLayer = NET_LAYER_COUNT + 1;
                }
            }

            /*
             * 下标0保存layer-0 Kernel。
             * 下标1保存layer-1 Kernel。
             * 当前rank不使用某一层时，对应handle保持为0。
             */
            resCtxHost.directKernels.assign(NET_LAYER_COUNT, CcuKernelHandle{});
            resCtxHost.meshKernels.assign(NET_LAYER_COUNT, CcuKernelHandle{});
            resCtxHost.scatterKernels.assign(NET_LAYER_COUNT, CcuKernelHandle{});
            resCtxHost.gatherKernels.assign(NET_LAYER_COUNT, CcuKernelHandle{});
            resCtxHost.meshEnabled = param.rankSize > 2 && commonLayer < NET_LAYER_COUNT;
            resCtxHost.stagedEnabled = param.rankSize > 2 && !resCtxHost.meshEnabled;

            for (uint32_t netLayer = 0;
                 netLayer < NET_LAYER_COUNT;
                 ++netLayer) {
                if (channelsByLayer[netLayer].empty()) {
                    continue;
                }

                const bool rootLayer = param.myRank == param.root ||
                    std::any_of(channelsByLayer[netLayer].begin(), channelsByLayer[netLayer].end(),
                        [&param](const std::pair<uint32_t, ChannelHandle> &rankChannel) {
                            return rankChannel.first == param.root;
                        });

                std::vector<std::pair<uint32_t, ChannelHandle>> rootChannels;
                if (rootLayer) {
                    if (param.myRank == param.root) {
                        rootChannels = channelsByLayer[netLayer];
                    } else {
                        const auto rootChannel = std::find_if(
                            channelsByLayer[netLayer].begin(), channelsByLayer[netLayer].end(),
                            [&param](const std::pair<uint32_t, ChannelHandle> &rankChannel) {
                                return rankChannel.first == param.root;
                            });
                        CHK_PRT_RET(
                            rootChannel == channelsByLayer[netLayer].end(),
                            HCCL_ERROR("[HcclBroadcast] root channel is missing on layer %u", netLayer),
                            HCCL_E_INTERNAL);
                        rootChannels.push_back(*rootChannel);
                    }

                    const bool useTree =
                        treeEnabled && resCtxHost.meshEnabled;
                    const auto &directChannels =
                        useTree ? channelsByLayer[netLayer] : rootChannels;
                    CHK_RET(RegisterBroadcastKernel(
                        comm,
                        param,
                        netLayer,
                        directChannels,
                        BroadcastKernelType::DIRECT,
                        resCtxHost,
                        useTree));
                }

                if (resCtxHost.meshEnabled &&
                    dataSize > smallDataThreshold) {
                    CHK_RET(RegisterBroadcastKernel(
                        comm,
                        param,
                        netLayer,
                        channelsByLayer[netLayer],
                        BroadcastKernelType::MESH,
                        resCtxHost,
                        false,
                        pipelineSteps));
                } else if (resCtxHost.stagedEnabled) {
                    if (rootLayer) {
                        CHK_RET(RegisterBroadcastKernel(
                            comm,
                            param,
                            netLayer,
                            rootChannels,
                            BroadcastKernelType::SCATTER,
                            resCtxHost));
                    }
                    CHK_RET(RegisterBroadcastKernel(
                        comm,
                        param,
                        netLayer,
                        channelsByLayer[netLayer],
                        BroadcastKernelType::GATHER,
                        resCtxHost));
                }
            }

            if (resCtxHost.stagedEnabled) {
                /*
                 * layer-0与layer-1位于不同IO Die。使用独立runtime stream，
                 * 使两个CCU Kernel能够并发执行；跨stream依赖在ExecOp中
                 * 由Event和HBM就绪代次共同维护。
                 */
                ACLCHECK(aclrtCreateStream(&resCtxHost.auxiliaryStream));
                ACLCHECK(aclrtCreateEventExWithFlag(&resCtxHost.startEvent, ACL_EVENT_SYNC));
                ACLCHECK(aclrtCreateEventExWithFlag(&resCtxHost.finishEvent, ACL_EVENT_SYNC));

                ThreadHandle auxiliaryThread{};
                CHK_RET(HcclThreadAcquireWithStream(
                    comm,
                    ccuEngine,
                    resCtxHost.auxiliaryStream,
                    notifyNumOnThread,
                    &auxiliaryThread));
                resCtxHost.threads.push_back(auxiliaryThread);
            }
        }

        std::vector<char> seq = resCtxHost.Serialize();
        const uint64_t seqSize = seq.size();

        param.ctxSize = seqSize;

        CHK_RET(HcclEngineCtxCreate(
            comm,
            param.tag,
            ccuEngine,
            param.ctxSize,
            &param.resCtx));

        CHK_RET(HcclEngineCtxCopy(
            comm,
            ccuEngine,
            param.tag,
            seq.data(),
            seqSize,
            0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}