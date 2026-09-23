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
#include <cstring>
#include <limits>
#include <vector>
#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    uint32_t *layerList = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerCount));
    CHK_PRT_RET(layerList == nullptr || layerCount == 0,
        HCCL_ERROR("Missing Scatter topology layers"), HCCL_E_INTERNAL);
    // Library-owned query results may be invalidated by the next query.
    std::vector<uint32_t> layers(layerList, layerList + layerCount);
    std::sort(layers.begin(), layers.end());
    std::vector<HcclChannelDesc> descriptions(param.rankSize - 1);
    std::vector<ChannelHandle> handles(param.rankSize - 1);
    CHK_RET(HcclChannelDescInit(descriptions.data(), descriptions.size()));
    uint32_t index = 0;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        HcclChannelDesc &desc = descriptions[index++];
        bool found = false;
        for (uint32_t layer : layers) {
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layer, param.myRank, peer, &links, &linkCount));
            std::vector<CommLink> copiedLinks;
            if (links != nullptr) {
                copiedLinks.assign(links, links + linkCount);
            }
            for (const CommLink &link : copiedLinks) {
                if (link.linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP
                    && link.linkAttr.linkProtocol != COMM_PROTOCOL_UBOE) {
                    continue;
                }
                desc.remoteRank = peer;
                desc.notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                found = true;
                break;
            }
            if (found) {
                break;
            }
        }
        CHK_PRT_RET(!found, HCCL_ERROR("No CCU link from rank %u to rank %u", param.myRank, peer),
            HCCL_E_NOT_SUPPORT);
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID,
            sizeof(resource.channelDie[peer]), &resource.channelDie[peer]));
    }
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, descriptions.data(), descriptions.size(), handles.data()));
    for (uint32_t i = 0; i < handles.size(); ++i) {
        resource.channels[descriptions[i].remoteRank] = handles[i];
    }
    return HCCL_SUCCESS;
}

HcclResult QueryLocalMask(HcclComm comm, const OpParam &param, uint64_t &mask)
{
    mask = uint64_t(1) << param.myRank;
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    uint32_t *layerList = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerCount));
    CHK_PRT_RET(layerList == nullptr || layerCount == 0,
        HCCL_ERROR("Missing local topology layers"), HCCL_E_INTERNAL);
    std::vector<uint32_t> layers(layerList, layerList + layerCount);
    std::sort(layers.begin(), layers.end());
    const uint32_t localLayer = layers.front();

    uint32_t *members = nullptr;
    uint32_t memberCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, localLayer, &members, &memberCount));
    CHK_PRT_RET(members == nullptr || memberCount == 0 || memberCount > param.rankSize,
        HCCL_ERROR("Invalid local instance member array"), HCCL_E_NOT_SUPPORT);
    // Library-owned query results may be invalidated by the next query.
    const std::vector<uint32_t> copiedMembers(members, members + memberCount);
    uint64_t found = 0;
    for (uint32_t member : copiedMembers) {
        CHK_PRT_RET(member >= param.rankSize || (found & (uint64_t(1) << member)) != 0,
            HCCL_ERROR("Invalid or duplicate local instance member"), HCCL_E_NOT_SUPPORT);
        found |= uint64_t(1) << member;
    }
    CHK_PRT_RET((found & (uint64_t(1) << param.myRank)) == 0,
        HCCL_ERROR("Rank absent from its local instance"), HCCL_E_NOT_SUPPORT);

    uint32_t *sizeList = nullptr;
    uint32_t instanceCount = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, localLayer, &sizeList, &instanceCount));
    CHK_PRT_RET(sizeList == nullptr || instanceCount == 0 || instanceCount > param.rankSize,
        HCCL_ERROR("Invalid local instance size list"), HCCL_E_NOT_SUPPORT);
    const std::vector<uint32_t> instanceSizes(sizeList, sizeList + instanceCount);
    uint64_t totalSize = 0;
    bool ownSizePresent = false;
    for (uint32_t size : instanceSizes) {
        CHK_PRT_RET(size == 0 || size > param.rankSize,
            HCCL_ERROR("Invalid local instance size"), HCCL_E_NOT_SUPPORT);
        totalSize += size;
        ownSizePresent = ownSizePresent || size == memberCount;
    }
    CHK_PRT_RET(totalSize != param.rankSize || !ownSizePresent,
        HCCL_ERROR("Instance sizes do not describe the local partition"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(instanceCount == 1 && (memberCount != param.rankSize || instanceSizes[0] != param.rankSize),
        HCCL_ERROR("Single instance does not cover the communicator"), HCCL_E_NOT_SUPPORT);
    mask = found;
    return HCCL_SUCCESS;
}

HcclResult RegisterKernel(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    uint64_t queriedMask = 0;
    CHK_RET(QueryLocalMask(comm, param, queriedMask));
    const uint32_t localMask = static_cast<uint32_t>(queriedMask);
    resource.localMask = localMask;
    const uint32_t localCount = ScatterLocalCount(localMask, param.rankSize);
    const bool relayTopology = localCount > 1 && param.rankSize - localCount > 4;
    const bool helper = param.myRank != param.root && ScatterIsLocal(localMask, param.root);
    if (relayTopology && helper && resource.stagingBase == 0) {
        void *buffer = nullptr;
        CHK_RET(HcclGetHcclBuffer(comm, &buffer, &resource.stagingSize));
        resource.stagingBase = reinterpret_cast<uint64_t>(buffer);
        CHK_PRT_RET(buffer == nullptr || resource.stagingSize == 0,
            HCCL_ERROR("Missing helper staging buffer"), HCCL_E_NOT_SUPPORT);
    }
    std::vector<uint32_t> dies;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank && std::find(dies.begin(), dies.end(), resource.channelDie[peer]) == dies.end()) {
            dies.push_back(resource.channelDie[peer]);
        }
    }
    if (dies.empty()) {
        dies.push_back(0); // Single-rank Scatter still needs the local copy kernel.
    }
    CcuInsHandle instance = 0;
    uint32_t instanceCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    CHK_PRT_RET(instanceCount != 1, HCCL_ERROR("Expected one CCU instance"), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
    CcuResult registerResult = CCU_SUCCESS;
    for (uint32_t group = 0; group < dies.size(); ++group) {
        ScatterKernelArg arg{};
        arg.rankId = param.myRank;
        arg.rankSize = param.rankSize;
        arg.root = param.root;
        arg.localMask = localMask;
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer != param.myRank && resource.channelDie[peer] == dies[group]) {
                arg.channels[peer] = resource.channels[peer];
                if (peer == param.root) {
                    resource.rootChannelGroup[param.root] = group;
                }

            }
        }
        const void *args[] = {&arg};
        // Reuse one kernel per die: additional phase kernels exhaust the instance's resources.
        registerResult = HcommCcuKernelRegister(instance, dies[group], "ScatterMetadataFanoutV14",
            reinterpret_cast<const void *>(ops_hccl::CcuKernel), args, 1, &resource.kernels[param.root][group]);
        if (registerResult != CCU_SUCCESS) {
            HCCL_ERROR("Scatter kernel registration failed: rank=%u root=%u die=%u ret=%u",
                param.myRank, param.root, dies[group], static_cast<uint32_t>(registerResult));
            break;
        }
    }
    // Close the registration session even when this kernel fails to register.
    const CcuResult endResult = HcommCcuKernelRegisterEnd(instance);
    CHK_RET_CCU(registerResult);
    CHK_RET_CCU(endResult);
    resource.kernelCount[param.root] = dies.size();
    if (param.myRank == param.root) {
        std::vector<uint32_t> order(dies.size()), remotePeers(dies.size(), 0);
        for (uint32_t group = 0; group < dies.size(); ++group) {
            order[group] = group;
            for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
                if (!ScatterIsLocal(localMask, peer) && resource.channelDie[peer] == dies[group]) {
                    ++remotePeers[group];
                }
            }
        }
        // Start cross-group traffic earlier without changing registered kernels or channels.
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return remotePeers[a] > remotePeers[b];
        });
        for (uint32_t group = 0; group < dies.size(); ++group) {
            resource.smallGroupOrder[group] = order[group];
        }
    }
    resource.rankId = param.myRank;
    resource.rankSize = param.rankSize;
    return HCCL_SUCCESS;
}

HcclResult ValidateBuffers(const OpParam &param)
{
    const uint64_t bytes = param.count * sizeof(float);
    const uint64_t output = reinterpret_cast<uint64_t>(param.outputPtr);
    CHK_PRT_RET(output > std::numeric_limits<uint64_t>::max() - bytes,
        HCCL_ERROR("Output address range overflows"), HCCL_E_PARA);
    if (param.myRank == param.root) {
        const uint64_t input = reinterpret_cast<uint64_t>(param.inputPtr);
        CHK_PRT_RET(input > std::numeric_limits<uint64_t>::max() - bytes * param.rankSize,
            HCCL_ERROR("Input address range overflows"), HCCL_E_PARA);
        const uint64_t source = input + bytes * param.root;
        // LocalCopy requires disjoint ranges. Exact in-place needs no copy.
        const bool partialOverlap = source != output && source < output + bytes && output < source + bytes;
        CHK_PRT_RET(partialOverlap, HCCL_ERROR("Root source and output partially overlap"), HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    OpParam param{};
    snprintf(param.tag, sizeof(param.tag), "%s", "hccl_scatter_ccu_metadata_fanout_v14");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > SCATTER_MAX_RANKS
            || param.myRank >= param.rankSize || root >= param.rankSize,
        HCCL_ERROR("Invalid Scatter rank or root"), HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("Scatter supports float32"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize,
        HCCL_ERROR("Scatter input size overflows"), HCCL_E_PARA);
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(recvBuf);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }
    CHK_RET(ValidateBuffers(param));

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, 1, &param.cpuThread));
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    AlgResourceCtx resource{};
    if (HcclEngineCtxGet(comm, param.tag, COMM_ENGINE_CCU, &ctx, &ctxSize) == HCCL_SUCCESS) {
        CHK_PRT_RET(ctx == nullptr || ctxSize != sizeof(AlgResourceCtx),
            HCCL_ERROR("Invalid Scatter resource context"), HCCL_E_INTERNAL);
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
        std::memcpy(&resource, ctx, sizeof(resource));
        CHK_PRT_RET(resource.rankId != param.myRank || resource.rankSize != param.rankSize,
            HCCL_ERROR("Scatter context does not match communicator"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resource.failed,
            HCCL_ERROR("Scatter communicator requires coordinated recovery after initialization failure"), HCCL_E_INTERNAL);
    } else {
        CHK_RET(AcquireChannels(comm, param, resource));
        resource.rankId = param.myRank;
        resource.rankSize = param.rankSize;
        param.ctxSize = sizeof(resource);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
    }
    if (resource.kernelCount[root] == 0) {
        // Persist failure before any initialization side effects. If publishing
        // the final context fails, even an older root cache must refuse retries.
        resource.failed = true;
        CHK_RET(HcclEngineCtxCopy(comm, COMM_ENGINE_CCU, param.tag, &resource, sizeof(resource), 0));
        const HcclResult result = RegisterKernel(comm, param, resource);
        resource.failed = result != HCCL_SUCCESS;
        const HcclResult copyResult = HcclEngineCtxCopy(comm, COMM_ENGINE_CCU, param.tag, &resource, sizeof(resource), 0);
        // Preserve the first error. There is no distributed failure commit: any
        // rank failure requires coordinated recovery, never independent retries.
        CHK_RET(result);
        CHK_RET(copyResult);
    }
    return ops_hccl::ExecOp(param);
}
