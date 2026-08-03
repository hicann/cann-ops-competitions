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

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr char ALLGATHER_TAG[] = "hccl_custom_allgather";
constexpr char ALLGATHER_OUTPUT_MEM_TAG[] = "hccl_custom_allgather_output";

struct TwoServerTopologyPlan {
    bool valid = false;
    uint32_t localLayer = INVALID_VALUE_RANKID;
    uint32_t globalLayer = INVALID_VALUE_RANKID;
    uint32_t localGroupBaseRank = INVALID_VALUE_RANKID;
    uint32_t remoteGroupBaseRank = INVALID_VALUE_RANKID;
    uint32_t localGroupIndex = INVALID_VALUE_RANKID;
    uint32_t proxyRank = INVALID_VALUE_RANKID;
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> remoteRanks;
    std::vector<uint32_t> recursiveDoublingPartners;
};

HcclResult GetAllGatherDataTypeSize(HcclDataType dataType, uint64_t &dataTypeSize)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            dataTypeSize = 1;
            break;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_BFP16:
            dataTypeSize = 2;
            break;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
        case HCCL_DATA_TYPE_FP32:
            dataTypeSize = 4;
            break;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            dataTypeSize = 8;
            break;
        case HCCL_DATA_TYPE_INT128:
            dataTypeSize = 16;
            break;
        default:
            HCCL_ERROR("Unsupported data type[%d]", static_cast<int32_t>(dataType));
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

bool IsAicpuProtocol(CommProtocol protocol)
{
    return protocol == CommProtocol::COMM_PROTOCOL_UBC_CTP || protocol == CommProtocol::COMM_PROTOCOL_UBC_TP
           || protocol == CommProtocol::COMM_PROTOCOL_UBOE;
}

HcclResult GetNetLayers(HcclComm comm, std::vector<uint32_t> &netLayers)
{
    uint32_t *netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayerList, &netLayerNum));
    CHK_PRT_RET(netLayerNum == 0 || netLayerList == nullptr,
        HCCL_ERROR("Rank graph does not contain any network layer"), HCCL_E_NOT_SUPPORT);
    netLayers.assign(netLayerList, netLayerList + netLayerNum);
    return HCCL_SUCCESS;
}

bool IsContiguousRankGroup(const std::vector<uint32_t> &ranks)
{
    if (ranks.size() != OPTIMIZED_GROUP_SIZE) {
        return false;
    }
    for (uint32_t index = 1; index < OPTIMIZED_GROUP_SIZE; ++index) {
        if (ranks[index] != ranks[0] + index) {
            return false;
        }
    }
    return true;
}

HcclResult BuildTwoServerTopologyPlan(HcclComm comm, uint32_t myRank, uint32_t rankSize,
    const std::vector<uint32_t> &netLayers, TwoServerTopologyPlan &plan)
{
    if (rankSize != OPTIMIZED_RANK_NUM) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> globalRanks;
    for (uint32_t netLayer : netLayers) {
        uint32_t *rankList = nullptr;
        uint32_t rankNum = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, netLayer, &rankList, &rankNum));
        CHK_PRT_RET(rankNum != 0 && rankList == nullptr, HCCL_ERROR("Layer[%u] returned a null rank list", netLayer),
            HCCL_E_INTERNAL);

        // 接口返回的内存可能被后续拓扑查询覆盖，必须立即复制。
        std::vector<uint32_t> ranks;
        if (rankNum != 0) {
            ranks.assign(rankList, rankList + rankNum);
        }
        std::sort(ranks.begin(), ranks.end());
        if (std::adjacent_find(ranks.begin(), ranks.end()) != ranks.end()) {
            continue;
        }
        if (ranks.size() == OPTIMIZED_GROUP_SIZE && std::binary_search(ranks.begin(), ranks.end(), myRank)
            && plan.localRanks.empty()) {
            plan.localLayer = netLayer;
            plan.localRanks = ranks;
        }
        if (ranks.size() == OPTIMIZED_RANK_NUM && globalRanks.empty()) {
            plan.globalLayer = netLayer;
            globalRanks = ranks;
        }
    }

    if (plan.localRanks.size() != OPTIMIZED_GROUP_SIZE || globalRanks.size() != OPTIMIZED_RANK_NUM
        || plan.localLayer == plan.globalLayer) {
        return HCCL_SUCCESS;
    }
    for (uint32_t rank = 0; rank < OPTIMIZED_RANK_NUM; ++rank) {
        if (globalRanks[rank] != rank) {
            return HCCL_SUCCESS;
        }
    }

    for (uint32_t rank : globalRanks) {
        if (!std::binary_search(plan.localRanks.begin(), plan.localRanks.end(), rank)) {
            plan.remoteRanks.push_back(rank);
        }
    }
    if (!IsContiguousRankGroup(plan.localRanks) || !IsContiguousRankGroup(plan.remoteRanks)) {
        return HCCL_SUCCESS;
    }

    auto localRankIter = std::lower_bound(plan.localRanks.begin(), plan.localRanks.end(), myRank);
    if (localRankIter == plan.localRanks.end() || *localRankIter != myRank) {
        return HCCL_SUCCESS;
    }
    plan.localGroupIndex = static_cast<uint32_t>(localRankIter - plan.localRanks.begin());
    plan.localGroupBaseRank = plan.localRanks[0];
    plan.remoteGroupBaseRank = plan.remoteRanks[0];
    plan.proxyRank = plan.remoteRanks[plan.localGroupIndex];

    for (uint32_t mask = 1; mask < OPTIMIZED_GROUP_SIZE; mask <<= 1) {
        plan.recursiveDoublingPartners.push_back(plan.localRanks[plan.localGroupIndex ^ mask]);
    }
    plan.recursiveDoublingPartners.push_back(plan.proxyRank);
    plan.valid = plan.recursiveDoublingPartners.size() == RECURSIVE_DOUBLING_ROUND_NUM;
    return HCCL_SUCCESS;
}

bool IsLocalGroupRank(const TwoServerTopologyPlan &plan, uint32_t rank)
{
    return std::binary_search(plan.localRanks.begin(), plan.localRanks.end(), rank);
}

uint32_t GetRepresentedRank(const TwoServerTopologyPlan &plan, uint32_t cclOwnerRank)
{
    if (IsLocalGroupRank(plan, cclOwnerRank)) {
        return plan.remoteRanks[cclOwnerRank - plan.localGroupBaseRank];
    }
    return plan.localRanks[cclOwnerRank - plan.remoteGroupBaseRank];
}

HcclResult SelectLinkForPeer(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    const std::vector<uint32_t> &netLayers, uint32_t preferredLayer, CommLink &selectedLink, uint32_t &selectedLayer)
{
    std::vector<uint32_t> orderedLayers;
    if (preferredLayer != INVALID_VALUE_RANKID) {
        orderedLayers.push_back(preferredLayer);
    }
    for (uint32_t netLayer : netLayers) {
        if (netLayer != preferredLayer) {
            orderedLayers.push_back(netLayer);
        }
    }

    for (uint32_t netLayer : orderedLayers) {
        CommLink *linkList = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &linkNum));
        for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
            if (!IsAicpuProtocol(linkList[linkIndex].linkAttr.linkProtocol)) {
                continue;
            }
            selectedLink = linkList[linkIndex];
            selectedLayer = netLayer;
            if (linkNum > 1) {
                HCCL_WARNING("Rank[%u] has %u links to rank[%u] on layer[%u]; only one channel is used", myRank,
                    linkNum, remoteRank, netLayer);
            }
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("No AICPU-readable link from rank[%u] to rank[%u]", myRank, remoteRank);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult BuildChannelDescs(HcclComm comm, uint32_t myRank, uint32_t rankSize, const std::vector<uint32_t> &netLayers,
    const TwoServerTopologyPlan &plan, HcclMemHandle *memHandles, uint32_t memHandleNum,
    std::vector<HcclChannelDesc> &channelDescs)
{
    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }

    channelDescs.reserve(rankSize - 1);
    for (uint32_t remoteRank = 0; remoteRank < rankSize; ++remoteRank) {
        if (remoteRank == myRank) {
            continue;
        }

        CommLink selectedLink;
        uint32_t preferredLayer = INVALID_VALUE_RANKID;
        if (plan.valid) {
            preferredLayer = IsLocalGroupRank(plan, remoteRank) ? plan.localLayer : plan.globalLayer;
        }
        uint32_t selectedLayer = INVALID_VALUE_RANKID;
        CHK_RET(SelectLinkForPeer(comm, myRank, remoteRank, netLayers, preferredLayer, selectedLink, selectedLayer));
        if (plan.valid && selectedLayer != preferredLayer) {
            HCCL_WARNING("Rank[%u] uses fallback layer[%u] instead of preferred layer[%u] for rank[%u]", myRank,
                selectedLayer, preferredLayer, remoteRank);
        }

        HcclChannelDesc channelDesc;
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        channelDesc.remoteRank = remoteRank;
        channelDesc.channelProtocol = selectedLink.linkAttr.linkProtocol;
        channelDesc.localEndpoint = selectedLink.srcEndpointDesc;
        channelDesc.remoteEndpoint = selectedLink.dstEndpointDesc;
        channelDesc.notifyNum = CHANNEL_NOTIFY_NUM;
        channelDesc.memHandles = memHandles;
        channelDesc.memHandleNum = memHandleNum;
        channelDescs.push_back(channelDesc);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    int32_t tagLength = std::snprintf(param.tag, sizeof(param.tag), "%s", ALLGATHER_TAG);
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("Failed to build allgather tag"), HCCL_E_INTERNAL);

    // 注册算子信息
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank information: rank[%u], rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);

    uint64_t dataTypeSize = 0;
    CHK_RET(GetAllGatherDataTypeSize(dataType, dataTypeSize));
    CHK_PRT_RET(sendCount > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("Input size overflows: count[%llu], dataTypeSize[%llu]", static_cast<unsigned long long>(sendCount),
            static_cast<unsigned long long>(dataTypeSize)),
        HCCL_E_PARA);
    uint64_t inputSize = sendCount * dataTypeSize;
    CHK_PRT_RET(inputSize != 0 && param.rankSize > std::numeric_limits<uint64_t>::max() / inputSize,
        HCCL_ERROR("Output size overflows: inputSize[%llu], rankSize[%u]", static_cast<unsigned long long>(inputSize),
            param.rankSize),
        HCCL_E_PARA);
    uint64_t outputSize = inputSize * param.rankSize;

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 thread，并申请 Notify；同时导出为 AICPU 上可用的 thread
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取实际配置的HCCL Buffer，算法不得假设固定容量。
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 每个对端只创建一条Channel，由一个独立Worker Thread处理；另有COPIER_THREAD_NUM个拷贝Thread。
        // mate流水路径中前两个拷贝Thread各独占7条READY Channel，第三个与main并行拷self输出。
        std::vector<uint32_t> netLayers;
        if (param.rankSize > 1) {
            CHK_RET(GetNetLayers(comm, netLayers));
        }
        TwoServerTopologyPlan topologyPlan;
        CHK_RET(BuildTwoServerTopologyPlan(comm, param.myRank, param.rankSize, netLayers, topologyPlan));

        // 仅为大消息注册用户输出，保持512KiB小消息路径的建链行为不变。若当前资源版本不支持
        // 注册内存交换，则保留空句柄，Device会继续使用原有CCL Buffer路径。
        std::vector<HcclMemHandle> outputMemHandles;
        if (param.rankSize > 1 && inputSize > SERIAL_SINGLE_SHOT_MAX_SIZE) {
            CommMem outputMem{COMM_MEM_TYPE_DEVICE, recvBuf, outputSize};
            HcclMemHandle outputMemHandle = nullptr;
            HcclResult registerRet = HcclCommMemReg(comm, ALLGATHER_OUTPUT_MEM_TAG, &outputMem, &outputMemHandle);
            if (registerRet == HCCL_SUCCESS && outputMemHandle != nullptr) {
                outputMemHandles.push_back(outputMemHandle);
                resCtxHost.registeredOutput = CommBuffer{recvBuf, outputSize};
            } else {
                HCCL_WARNING("Direct-output registration is unavailable, ret[%d]", registerRet);
            }
        }

        std::vector<HcclChannelDesc> channelDescs;
        CHK_RET(BuildChannelDescs(comm, param.myRank, param.rankSize, netLayers, topologyPlan,
            outputMemHandles.empty() ? nullptr : outputMemHandles.data(),
            static_cast<uint32_t>(outputMemHandles.size()), channelDescs));
        CHK_PRT_RET(channelDescs.size() != static_cast<size_t>(param.rankSize - 1),
            HCCL_ERROR("Unexpected channel count[%zu], rankSize[%u]", channelDescs.size(), param.rankSize),
            HCCL_E_INTERNAL);

        uint32_t channelNum = static_cast<uint32_t>(channelDescs.size());
        uint32_t workerNum = channelNum;
        uint32_t threadNum = workerNum + 1 + COPIER_THREAD_NUM;
        // 主Thread的0号Notify用于Host/AICPU同步，其余分别回收Worker staged/完成与拷贝Thread完成通知。
        // 从Thread默认只用0号Notify接收启动信号；mate流水的copier0/1额外接收完成/逐chunk转发。
        uint32_t mainNotifyNum = 2 * workerNum + 1 + COPIER_THREAD_NUM;

        resCtxHost.threads.resize(threadNum);
        std::vector<ThreadConfig> threadConfigs(threadNum);
        CHK_PRT_RET(ThreadConfigInit(threadConfigs.data(), threadNum) != 0,
            HCCL_ERROR("Failed to initialize thread configurations"), HCCL_E_INTERNAL);
        CHK_PRT_RET(mainNotifyNum > std::numeric_limits<uint16_t>::max(),
            HCCL_ERROR("Main thread notify count[%u] exceeds uint16_t", mainNotifyNum), HCCL_E_INTERNAL);
        threadConfigs[0].notifyNumPerThread = static_cast<uint16_t>(mainNotifyNum);
        for (uint32_t threadIndex = 1; threadIndex < threadNum; ++threadIndex) {
            threadConfigs[threadIndex].notifyNumPerThread = 1;
        }
        threadConfigs[workerNum + 1].notifyNumPerThread = 2;
        threadConfigs[workerNum + 2].notifyNumPerThread = static_cast<uint16_t>(THREAD_NOTIFY_MATE_RELAY_NUM);
        // 一次性申请全部Thread，避免部分资源管理版本的后续申请返回已有Thread前缀。
        CHK_RET(HcclThreadAcquireWithConfig(comm, CommEngine::COMM_ENGINE_AICPU, threadNum, THREAD_TYPE_TS,
            threadConfigs.data(), resCtxHost.threads.data()));
        for (uint32_t threadIndex = 0; threadIndex < threadNum; ++threadIndex) {
            for (uint32_t previousIndex = 0; previousIndex < threadIndex; ++previousIndex) {
                CHK_PRT_RET(resCtxHost.threads[threadIndex] == resCtxHost.threads[previousIndex],
                    HCCL_ERROR("Thread[%u] unexpectedly reuses thread[%u]", threadIndex, previousIndex),
                    HCCL_E_INTERNAL);
            }
        }
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        if (channelNum > 0) {
            std::vector<ChannelHandle> channelHandles(channelNum);
            CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, channelDescs.data(), channelNum, channelHandles.data()));

            resCtxHost.channels.resize(channelNum);
            bool allRemoteOutputsFound = !outputMemHandles.empty();
            if (allRemoteOutputsFound) {
                resCtxHost.remoteOutputs.resize(channelNum);
            }
            for (uint32_t channelIndex = 0; channelIndex < channelNum; ++channelIndex) {
                ChannelInfo &channelInfo = resCtxHost.channels[channelIndex];
                channelInfo.remoteRank = channelDescs[channelIndex].remoteRank;
                if (topologyPlan.valid) {
                    channelInfo.representedRank = GetRepresentedRank(topologyPlan, channelInfo.remoteRank);
                }
                channelInfo.notifyNum = channelDescs[channelIndex].notifyNum;
                channelInfo.handle = channelHandles[channelIndex];
                CHK_RET(HcclChannelGetHcclBuffer(
                    comm, channelInfo.handle, &channelInfo.remoteCclMem.addr, &channelInfo.remoteCclMem.size));
                CHK_PRT_RET(channelInfo.remoteCclMem.addr == nullptr || channelInfo.remoteCclMem.size == 0,
                    HCCL_ERROR("Invalid remote CCL buffer for rank[%u]", channelInfo.remoteRank), HCCL_E_INTERNAL);

                if (allRemoteOutputsFound) {
                    uint32_t remoteMemNum = 0;
                    CommMem *remoteMems = nullptr;
                    char **remoteMemTags = nullptr;
                    HcclResult remoteMemRet = HcclChannelGetRemoteMems(
                        comm, channelInfo.handle, &remoteMemNum, &remoteMems, &remoteMemTags);
                    bool found = false;
                    if (remoteMemRet == HCCL_SUCCESS && remoteMems != nullptr && remoteMemTags != nullptr) {
                        for (uint32_t memIndex = 0; memIndex < remoteMemNum; ++memIndex) {
                            if (remoteMemTags[memIndex] != nullptr
                                && std::strcmp(remoteMemTags[memIndex], ALLGATHER_OUTPUT_MEM_TAG) == 0
                                && remoteMems[memIndex].type == COMM_MEM_TYPE_DEVICE
                                && remoteMems[memIndex].addr != nullptr
                                && remoteMems[memIndex].size >= outputSize) {
                                resCtxHost.remoteOutputs[channelIndex] =
                                    CommBuffer{remoteMems[memIndex].addr, remoteMems[memIndex].size};
                                found = true;
                                break;
                            }
                        }
                    }
                    if (!found) {
                        HCCL_WARNING("Registered output was not exchanged with rank[%u], ret[%d]",
                            channelInfo.remoteRank, remoteMemRet);
                        allRemoteOutputsFound = false;
                    }
                }
            }
            if (!allRemoteOutputsFound) {
                resCtxHost.registeredOutput = CommBuffer{};
                resCtxHost.remoteOutputs.clear();
            }
        }

        if (topologyPlan.valid) {
            resCtxHost.topologyPlanValid = 1;
            resCtxHost.localGroupBaseRank = topologyPlan.localGroupBaseRank;
            resCtxHost.remoteGroupBaseRank = topologyPlan.remoteGroupBaseRank;
            resCtxHost.localGroupIndex = topologyPlan.localGroupIndex;
            resCtxHost.proxyRank = topologyPlan.proxyRank;
            for (uint32_t partnerRank : topologyPlan.recursiveDoublingPartners) {
                uint32_t channelIndex = 0;
                while (channelIndex < channelNum && resCtxHost.channels[channelIndex].remoteRank != partnerRank) {
                    ++channelIndex;
                }
                CHK_PRT_RET(channelIndex == channelNum,
                    HCCL_ERROR("No channel was created for recursive-doubling partner rank[%u]", partnerRank),
                    HCCL_E_INTERNAL);
                resCtxHost.recursiveDoublingChannelIndices.push_back(channelIndex);
            }
        }

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));
        // 申请 CPU 通信引擎上下文，存放 aicpuThread 句柄
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
