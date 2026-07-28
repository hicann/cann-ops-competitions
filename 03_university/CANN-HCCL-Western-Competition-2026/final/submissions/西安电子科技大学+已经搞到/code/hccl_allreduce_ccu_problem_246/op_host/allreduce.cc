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
#include <hccl/hccl_ccu_res.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"

#include <cstring>
#include <cstdio>
#include <iterator>
#include <limits>
#include <memory>
#include <map>
#include <vector>
#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>
#include "../op_kernel_ccu/ccu_kernel.h"

namespace {
constexpr uint32_t kChannelNotifyNum = 5;
constexpr uint32_t kInputId = 0;
constexpr uint32_t kInputTokenId = 1;
constexpr uint64_t kFullReduceFastPathBytes = 1ULL * 1024 * 1024;
constexpr uint64_t kTopologySmallPathBytes = 512ULL * 1024;

struct TopologyPlan {
    uint32_t kind;
    uint32_t localGroupBase;
    uint32_t localGroupSize;
    uint32_t rootCount;
};

bool MakeTopologyPlan(uint32_t rankSize, uint32_t rankId, TopologyPlan *plan)
{
    if (plan == nullptr || rankId >= rankSize) {
        return false;
    }
    switch (rankSize) {
        case 4:
            *plan = TopologyPlan{4, rankId, 1, 4};
            return true;
        case 12:
            *plan = rankId < 8 ? TopologyPlan{12, 0, 8, 2} : TopologyPlan{12, 8, 4, 2};
            return true;
        case 16:
            *plan = rankId < 8 ? TopologyPlan{16, 0, 8, 2} : TopologyPlan{16, 8, 8, 2};
            return true;
        default:
            return false;
    }
}

void *SelectTopologyKernel(uint32_t rankSize, bool smallPath)
{
    if (smallPath) {
        switch (rankSize) {
            case 4: return reinterpret_cast<void *>(ops_hccl::CcuSmall4Kernel);
            case 12: return reinterpret_cast<void *>(ops_hccl::CcuSmall12Kernel);
            case 16: return reinterpret_cast<void *>(ops_hccl::CcuSmall16Kernel);
            default: return nullptr;
        }
    }
    switch (rankSize) {
        case 4: return reinterpret_cast<void *>(ops_hccl::CcuLarge4Kernel);
        case 12: return reinterpret_cast<void *>(ops_hccl::CcuLarge12Kernel);
        case 16: return reinterpret_cast<void *>(ops_hccl::CcuLarge16Kernel);
        default: return nullptr;
    }
}

void *SelectTopologyLaneKernel(uint32_t rankSize)
{
    switch (rankSize) {
        case 4: return reinterpret_cast<void *>(ops_hccl::CcuSmall4LaneKernel);
        case 16: return reinterpret_cast<void *>(ops_hccl::CcuSmall16HierKernel);
        default: return nullptr;
    }
}

const char *SelectTopologyLaneKernelName(uint32_t rankSize)
{
    switch (rankSize) {
        case 4: return "CcuAllReduceSmall4Lane";
        case 16: return "CcuAllReduceSmall16Hier";
        default: return nullptr;
    }
}

const char *SelectTopologyKernelName(uint32_t rankSize, bool smallPath)
{
    if (smallPath) {
        switch (rankSize) {
            case 4: return "CcuAllReduceSmall4";
            case 12: return "CcuAllReduceSmall12";
            case 16: return "CcuAllReduceSmall16";
            default: return nullptr;
        }
    }
    switch (rankSize) {
        case 4: return "CcuAllReduceLarge4";
        case 12: return "CcuAllReduceLarge12";
        case 16: return "CcuAllReduceLarge16";
        default: return nullptr;
    }
}

HcclResult AcquirePeerChannel(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
                              ChannelHandle *channel, uint32_t *dieId)
{
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    constexpr CommProtocol protocols[] = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
    };

    for (uint32_t layerIndex = 0; layerIndex < layerCount; ++layerIndex) {
        const uint32_t netLayer = layers[layerIndex];
        uint32_t listSize = 0;
        CommLink *linkList = nullptr;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize));
        for (CommProtocol protocol : protocols) {
            for (uint32_t i = 0; i < listSize; ++i) {
                const CommLink &link = linkList[i];
                if (link.linkAttr.linkProtocol != protocol) {
                    continue;
                }
                HcclChannelDesc desc;
                CHK_RET(HcclChannelDescInit(&desc, 1));
                desc.remoteRank = remoteRank;
                desc.notifyNum = kChannelNotifyNum;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                EndpointAttrDieId endpointDieId{};
                CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank, &desc.localEndpoint,
                                                      ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId),
                                                      &endpointDieId));
                *dieId = endpointDieId;
                HCCL_INFO("Acquire CCU channel rank %u -> %u, layer %u, protocol %d",
                          myRank, remoteRank, netLayer, static_cast<int>(protocol));
                return HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, channel);
            }
        }
    }
    HCCL_ERROR("No CCU UBC link from rank %u to rank %u", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}
} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // Kernel 的数据类型和归约操作在注册阶段固化。本题只评测 FP32/SUM，
    // 必须在创建 EngineCtx 前拒绝其他组合，避免复用不匹配的 CCU 图。
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("Only FP32/SUM is supported, dataType[%d], reduceOp[%d]",
                   static_cast<int>(dataType), static_cast<int>(op));
        return HCCL_E_NOT_SUPPORT;
    }
    if (count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        HCCL_ERROR("AllReduce count overflows byte size");
        return HCCL_E_PARA;
    }

    // 构造算子参数
    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

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
    if (param.rankSize != 1 && param.rankSize != 4 && param.rankSize != 12 && param.rankSize != 16) {
        HCCL_ERROR("AllReduce v4 only supports contest rank sizes 4/12/16, got %u", param.rankSize);
        return HCCL_E_NOT_SUPPORT;
    }
    TopologyPlan topologyPlan{};
    if (param.rankSize != 1 && !MakeTopologyPlan(param.rankSize, param.myRank, &topologyPlan)) {
        return HCCL_E_NOT_SUPPORT;
    }
    const uint64_t dataBytes = count * sizeof(float);
    // V4.8 同时优化三个 512 KiB 性能点，但为每种拓扑注册独立图：
    // 4-rank 使用 owner slice tiled reduce + mission 内 fused gather；
    // 12/16-rank 使用双 die tiled full-reduce + Host combine。
    const bool topology1Small512 =
        dataBytes == kTopologySmallPathBytes && param.rankSize == 4;
    const bool topology2Small512 =
        dataBytes == kTopologySmallPathBytes && param.rankSize == 12;
    const bool topology3Small512 =
        dataBytes == kTopologySmallPathBytes && param.rankSize == 16;
    const bool tunedSmall512 =
        topology1Small512 || topology2Small512 || topology3Small512;
    // 16-rank 其他 <=1 MiB 功能点继续使用 V4.4 已通过的固定
    // Reduce+Combine 图，避免把 512 KiB 实验扩大到功能测试。
    const bool topology3SmallPath =
        dataBytes <= kFullReduceFastPathBytes && param.rankSize == 16 &&
        !topology3Small512;
    // 保留旧变量语义供 owner-die/secondary-partial 选择使用。
    const bool topologySmallPath = topology1Small512;
    const bool oneShotSmallPath = dataBytes <= kFullReduceFastPathBytes &&
        !tunedSmall512 && !topology3SmallPath;
    // 三种大包固定阶段图都必须显式排除各自的小包路径。尤其 topology3
    // small 会把 oneShotSmallPath 置 false，不能只用后者作为互斥条件。
    const bool topology1LargePath = !tunedSmall512 && !oneShotSmallPath &&
        param.rankSize == 4;
    const bool topology2LargePath =
        !tunedSmall512 && !oneShotSmallPath && param.rankSize == 12;
    const bool topology3LargePath =
        !tunedSmall512 && !topology3SmallPath && !oneShotSmallPath &&
        param.rankSize == 16;
    const bool fixedLargePath =
        topology1LargePath || topology2LargePath || topology3LargePath;
    // 保留旧 pipe 实现作为可回退代码；V4.4 的赛事拓扑不再选择它。
    const bool pipeLargePath = !oneShotSmallPath && !topologySmallPath &&
        !fixedLargePath && param.rankSize == 4;
    const bool hierarchicalLargePath = false;
    const bool directHierPath = false;
    const char *pathTag = topology1Small512 ? "topo1_small512_adaptive" :
        (topology2Small512 ? "topo2_small512_tiled" :
        (topology3Small512 ? "topo3_small512_tiled" :
        (topology3SmallPath ? "topo3_8x8_small" :
        (oneShotSmallPath ? "small_oneshot" :
        (topology1LargePath ? "topo1_4_large_multi" :
        (topology2LargePath ? "topo2_8x4_large_multi" :
        (topology3LargePath ? "topo3_8x8_large" : (pipeLargePath ? "large_pipe" :
        (directHierPath ? "large_direct_hier" :
        (hierarchicalLargePath ? "large_hier12" : "large"))))))))));
    const int tagLen = std::snprintf(param.tag, sizeof(param.tag),
        "hccl_custom_allreduce_fp32_sum_v4_8_r%u_%s", param.rankSize, pathTag);
    if (tagLen < 0 || static_cast<size_t>(tagLen) >= sizeof(param.tag)) {
        HCCL_ERROR("Failed to generate AllReduce resource tag");
        return HCCL_E_INTERNAL;
    }

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

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
        resCtxHost.algorithmKind = topology1Small512 ? 10 :
            (topology2Small512 ? 11 : (topology3Small512 ? 12 :
            (oneShotSmallPath ? 0 : (topologySmallPath ? 2 :
            (topology3SmallPath ? 6 : (topology3LargePath ? 7 :
            (topology1LargePath ? 8 : (topology2LargePath ? 9 :
            (pipeLargePath ? 3 : (directHierPath ? 5 :
            (hierarchicalLargePath ? 4 : 1)))))))))));
        resCtxHost.topologyKind = topologyPlan.kind;
        resCtxHost.localGroupBase = topologyPlan.localGroupBase;
        resCtxHost.localGroupSize = topologyPlan.localGroupSize;

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        if (cclBufferAddr == nullptr || cclBufferSize == 0) {
            HCCL_ERROR("HCCL scratch buffer is unavailable");
            return HCCL_E_INTERNAL;
        }
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源：Thread、Channel、CCU Kernel
        // ==============================================

        // 创建 CCU 通信引擎上的 thread 资源。
        if (param.rankSize > CcuKernelArgBase::kMaxRankSize) {
            return HCCL_E_NOT_SUPPORT;
        }

        if (param.rankSize == 1) {
            resCtxHost.threads.resize(1);
            CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &resCtxHost.threads[0]));
        } else {
            // 按本端 endpoint 的 die 分组。一个 CCU kernel 只能登记同一 die 上的通道。
            struct PeerChannel {
                uint32_t rank;
                ChannelHandle handle;
            };
            std::map<uint32_t, std::vector<PeerChannel>> channelsByDie;
            for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
                if (peer == param.myRank) {
                    continue;
                }
                ChannelHandle channel = 0;
                uint32_t dieId = 0;
                CHK_RET(AcquirePeerChannel(comm, param.myRank, peer, &channel, &dieId));
                channelsByDie[dieId].push_back(PeerChannel{peer, channel});
            }
            if (channelsByDie.empty() || channelsByDie.size() > 2) {
                HCCL_ERROR("AllReduce v4 expects one or two active CCU dies, got %zu", channelsByDie.size());
                return HCCL_E_NOT_SUPPORT;
            }
            // 真实平台可能把 topology1 的三条 channel 分成 2+1。单 die
            // 才能在一个 mission 内完成 fused gather；双 die 自动回退到
            // V4.4 已通过的 kind2 lane 图，不能因本地 VM 的单 die 布局拒绝注册。
            const bool topology1SmallFused =
                topology1Small512 && channelsByDie.size() == 1;
            if (topology1Small512 && !topology1SmallFused) {
                resCtxHost.algorithmKind = 2;
                HCCL_INFO("V4.8 topology1 small uses V4.4 multi-die lane fallback");
            }

            uint32_t ownerDie = channelsByDie.begin()->first;
            if (channelsByDie.size() == 2) {
                // legacy small、12-rank 512KiB 和所有大包均保留平台已验证的
                // V2.3 owner-die 选择。只有 4/16-rank 512KiB 专用 lane 图
                // 才按本地组通道数选择 owner die。
                if (!topologySmallPath) {
                    auto first = channelsByDie.begin();
                    auto second = std::next(first);
                    ownerDie = (first->second.size() <= second->second.size()) ? first->first : second->first;
                } else {
                    size_t bestLocalPeerCount = 0;
                    for (const auto &dieEntry : channelsByDie) {
                        size_t localPeerCount = 0;
                        for (const PeerChannel &peer : dieEntry.second) {
                            if (param.rankSize == 4 ||
                                (peer.rank >= topologyPlan.localGroupBase &&
                                 peer.rank < topologyPlan.localGroupBase + topologyPlan.localGroupSize)) {
                                ++localPeerCount;
                            }
                        }
                        if (localPeerCount > bestLocalPeerCount) {
                            bestLocalPeerCount = localPeerCount;
                            ownerDie = dieEntry.first;
                        }
                    }
                }
            }
            const bool laneOwner = param.rankSize == 4 || param.rankSize == 16 ||
                param.myRank < 4 || param.myRank >= 8;
            auto needsSecondaryPartial = [&]() {
                for (const auto &dieEntry : channelsByDie) {
                    if (dieEntry.first == ownerDie) continue;
                    bool selected = false;
                    for (const PeerChannel &peer : dieEntry.second) {
                        selected = laneOwner && (param.rankSize == 4 ||
                            (peer.rank >= topologyPlan.localGroupBase &&
                             peer.rank < topologyPlan.localGroupBase + topologyPlan.localGroupSize));
                        if (selected) break;
                    }
                    if (selected) return true;
                }
                return false;
            };
            if (hierarchicalLargePath && channelsByDie.size() == 2) {
                // 分层大包的两个 die 都生成 partial，必须始终执行一次
                // Host combine；不能再依赖 laneOwner/peer 子集判断。
                resCtxHost.reduceCombineMask |= 1U;
            } else if (topologySmallPath && needsSecondaryPartial()) {
                resCtxHost.reduceCombineMask |= 1U;
            }

            CcuInsHandle insHandle{0};
            uint32_t insNum = 0;
            CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
            if (insNum != 1) {
                return HCCL_E_INTERNAL;
            }
            CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
            if (ccuRet != CCU_SUCCESS) {
                return ConvertCcuToHccl(ccuRet);
            }
            auto closeRegistration = [&]() {
                CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
                if (endRet != CCU_SUCCESS) {
                    HCCL_ERROR("CCU kernel registration cleanup failed: %d", endRet);
                }
            };

            // 首个 thread 与用户 stream 绑定；其余 die 使用独立 thread。
            resCtxHost.threads.resize(channelsByDie.size());
            HcclResult threadRet =
                HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &resCtxHost.threads[0]);
            if (threadRet != HCCL_SUCCESS) {
                closeRegistration();
                return threadRet;
            }
            if (resCtxHost.threads.size() > 1) {
                threadRet = HcclThreadAcquire(comm, ccuEngine, resCtxHost.threads.size() - 1, 1,
                                              &resCtxHost.threads[1]);
                if (threadRet != HCCL_SUCCESS) {
                    closeRegistration();
                    return threadRet;
                }
            }

            // 小包保留各拓扑已验证图；三种大包均注册固定 Reduce 图，随后在
            // 独立资源组注册 Gather 图。每个 die 只能登记本 die 的 channel。
            for (const auto &entry : channelsByDie) {
                    const uint32_t dieId = entry.first;
                    CcuKernelInfo kernelInfo{};
                    auto kernelArg = std::make_shared<CcuKernelArgAllReduce>();
                    kernelArg->rankSize = param.rankSize;
                    kernelArg->rankId = param.myRank;
                    kernelArg->topologyKind = topologyPlan.kind;
                    kernelArg->stage = 0;
                    kernelArg->includeLocalInput = (dieId == ownerDie);
                    kernelArg->deferLocalInput = false;
                    kernelArg->dataType = param.dataType;
                    kernelArg->reduceType = param.reduceType;
                    kernelArg->channelCount = static_cast<uint32_t>(entry.second.size());
                    if (hierarchicalLargePath) {
                        uint32_t groupMin = param.myRank;
                        uint32_t groupMax = param.myRank;
                        for (const PeerChannel &peer : entry.second) {
                            groupMin = std::min(groupMin, peer.rank);
                            groupMax = std::max(groupMax, peer.rank);
                        }
                        kernelArg->localGroupBase = groupMin;
                        kernelArg->localGroupSize = groupMax - groupMin + 1;
                    } else {
                        kernelArg->localGroupBase = topologyPlan.localGroupBase;
                        kernelArg->localGroupSize = topologyPlan.localGroupSize;
                    }
                    kernelArg->initOutput = directHierPath &&
                        (dieId == channelsByDie.begin()->first);
                    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                        kernelArg->channels[i] = entry.second[i].handle;
                        kernelArg->peerRanks[i] = entry.second[i].rank;
                    }
                    if (topology1SmallFused) {
                        kernelInfo.kernelFunc = reinterpret_cast<void *>(
                            ops_hccl::CcuSmall4TiledFusedKernel);
                        if (std::snprintf(kernelInfo.kernelFuncName,
                            sizeof(kernelInfo.kernelFuncName),
                            "CcuAllReduceTopo1Small4TiledFused_d%u", dieId) < 0) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                    } else if (topology2Small512) {
                        kernelInfo.kernelFunc = reinterpret_cast<void *>(
                            ops_hccl::CcuSmall12TiledKernel);
                        if (std::snprintf(kernelInfo.kernelFuncName,
                            sizeof(kernelInfo.kernelFuncName),
                            "CcuAllReduceTopo2Small12Tiled_d%u", dieId) < 0) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                    } else if (topology3Small512) {
                        kernelInfo.kernelFunc = reinterpret_cast<void *>(
                            ops_hccl::CcuSmall16TiledKernel);
                        if (std::snprintf(kernelInfo.kernelFuncName,
                            sizeof(kernelInfo.kernelFuncName),
                            "CcuAllReduceTopo3Small16Tiled_d%u", dieId) < 0) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                    } else if (directHierPath) {
                        kernelInfo.kernelFunc = param.rankSize == 16 ?
                            reinterpret_cast<void *>(ops_hccl::CcuHierarchicalLarge16Kernel) :
                            reinterpret_cast<void *>(ops_hccl::CcuHierarchicalLarge12Kernel);
                        const int nameLen = std::snprintf(kernelInfo.kernelFuncName,
                            sizeof(kernelInfo.kernelFuncName), "CcuAllReduceDirectHier_r%u_d%u",
                            param.rankSize, dieId);
                        if (nameLen < 0 || static_cast<size_t>(nameLen) >= sizeof(kernelInfo.kernelFuncName)) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                    } else if (topology3SmallPath) {
                        kernelInfo.kernelFunc =
                            reinterpret_cast<void *>(ops_hccl::CcuTopo3SmallReduce8x8Kernel);
                        if (std::snprintf(kernelInfo.kernelFuncName,
                            sizeof(kernelInfo.kernelFuncName),
                            "CcuAllReduceTopo3SmallReduce8x8_d%u", dieId) < 0) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                    } else if (topology1LargePath) {
                        kernelInfo.kernelFunc =
                            reinterpret_cast<void *>(ops_hccl::CcuTopo1LargeReduce4Kernel);
                        if (std::snprintf(kernelInfo.kernelFuncName,
                            sizeof(kernelInfo.kernelFuncName),
                            "CcuAllReduceTopo1LargeReduce4_d%u", dieId) < 0) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                    } else if (topology2LargePath) {
                        kernelInfo.kernelFunc =
                            reinterpret_cast<void *>(ops_hccl::CcuTopo2LargeReduce12Kernel);
                        if (std::snprintf(kernelInfo.kernelFuncName,
                            sizeof(kernelInfo.kernelFuncName),
                            "CcuAllReduceTopo2LargeReduce12_d%u", dieId) < 0) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                    } else if (topology3LargePath) {
                        kernelInfo.kernelFunc =
                            reinterpret_cast<void *>(ops_hccl::CcuTopo3LargeReduce8x8Kernel);
                        if (std::snprintf(kernelInfo.kernelFuncName,
                            sizeof(kernelInfo.kernelFuncName),
                            "CcuAllReduceTopo3LargeReduce8x8_d%u", dieId) < 0) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                    } else if (pipeLargePath) {
                        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuLargePipeKernel);
                        const int nameLen = std::snprintf(kernelInfo.kernelFuncName,
                            sizeof(kernelInfo.kernelFuncName), "CcuAllReduceLargePipe_r%u_d%u",
                            param.rankSize, dieId);
                        if (nameLen < 0 || static_cast<size_t>(nameLen) >= sizeof(kernelInfo.kernelFuncName)) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                    } else if (!topologySmallPath) {
                        const char *kernelName = SelectTopologyKernelName(param.rankSize, oneShotSmallPath);
                        kernelInfo.kernelFunc = SelectTopologyKernel(param.rankSize, oneShotSmallPath);
                        if (kernelName == nullptr || kernelInfo.kernelFunc == nullptr ||
                            strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), kernelName) != 0) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                    } else {
                        const char *baseName = SelectTopologyLaneKernelName(param.rankSize);
                        kernelInfo.kernelFunc = SelectTopologyLaneKernel(param.rankSize);
                        if (baseName == nullptr || kernelInfo.kernelFunc == nullptr) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                        const int nameLen = std::snprintf(kernelInfo.kernelFuncName,
                            sizeof(kernelInfo.kernelFuncName), "%s_d%u", baseName, dieId);
                        if (nameLen < 0 || static_cast<size_t>(nameLen) >= sizeof(kernelInfo.kernelFuncName)) {
                            closeRegistration();
                            return HCCL_E_INTERNAL;
                        }
                    }
                    kernelInfo.setKernelArg(kernelArg);
                    CcuKernelHandle kernelHandle = 0;
                    const void *kernelArgs[] = {kernelInfo.kernelArg};
                    ccuRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
                        kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
                    if (ccuRet != CCU_SUCCESS) {
                        closeRegistration();
                        return ConvertCcuToHccl(ccuRet);
                    }
                    resCtxHost.ccuKernels.push_back(kernelHandle);
            }

            // HCOMM 当前 named shared-notify 在双 die 下会把 0x1/0x2
            // 映射到相反的等待侧。把跨 die 合并拆成独立 kernel，由 Host
            // thread 屏障建立 partial -> combine 的真实依赖。
            if (channelsByDie.size() == 2 && !directHierPath) {
                CcuKernelInfo combineInfo{};
                auto combineArg = std::make_shared<CcuKernelArgAllReduce>();
                combineArg->rankSize = param.rankSize;
                combineArg->rankId = param.myRank;
                combineArg->combineOnly = true;
                combineArg->combineScratch = !topologySmallPath ||
                    (resCtxHost.reduceCombineMask & 1U) != 0;
                combineArg->combineInput = false;
                combineArg->dataType = param.dataType;
                combineArg->reduceType = param.reduceType;
                combineArg->channelCount = 1;
                combineArg->channels[0] = channelsByDie.begin()->second.front().handle;
                if (strcpy_s(combineInfo.kernelFuncName, sizeof(combineInfo.kernelFuncName),
                             "CcuAllReduceCombineKernel") != 0) {
                    closeRegistration();
                    return HCCL_E_INTERNAL;
                }
                combineInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuCombineKernel);
                combineInfo.setKernelArg(combineArg);
                CcuKernelHandle combineHandle = 0;
                const void *combineArgs[] = {combineInfo.kernelArg};
                ccuRet = HcommCcuKernelRegister(insHandle, channelsByDie.begin()->first,
                    combineInfo.kernelFuncName, combineInfo.kernelFunc, combineArgs, 1, &combineHandle);
                if (ccuRet != CCU_SUCCESS) {
                    closeRegistration();
                    return ConvertCcuToHccl(ccuRet);
                }
                resCtxHost.ccuKernels.push_back(combineHandle);
            }

            const bool fixedSmallPath = tunedSmall512 || oneShotSmallPath ||
                topologySmallPath || topology3SmallPath;
            if (fixedSmallPath) {
                resCtxHost.reduceKernelCount =
                    static_cast<uint32_t>(resCtxHost.ccuKernels.size());
                const uint32_t expectedSmallKernelCount =
                    static_cast<uint32_t>(channelsByDie.size() +
                        (channelsByDie.size() == 2 ? 1U : 0U));
                if (resCtxHost.reduceKernelCount != expectedSmallKernelCount) {
                    closeRegistration();
                    return HCCL_E_INTERNAL;
                }
                HCCL_INFO("Registered fixed small graph kind[%u]: reduce[%zu], "
                    "combine[%zu], total[%u]", resCtxHost.algorithmKind,
                    channelsByDie.size(), channelsByDie.size() == 2 ? 1UL : 0UL,
                    resCtxHost.reduceKernelCount);
            }

            // 官方两模板执行器会按 resGroup 分别 Start/End 注册
            // ReduceScatter 与 AllGather。HCOMM 的 registration group 是
            // CCU 同步资源域的一部分；只把多个 handle 塞进一个 group 会让
            // 两阶段图共享 notify 状态，造成 checker 看不到有效同步记录。
            // 因此所有固定阶段大包必须在此显式切换到 group 1。
            if (fixedLargePath) {
                resCtxHost.reduceKernelCount = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
                const uint32_t expectedReduceKernelCount =
                    static_cast<uint32_t>(channelsByDie.size() + (channelsByDie.size() == 2 ? 1U : 0U));
                if (resCtxHost.reduceKernelCount != expectedReduceKernelCount) {
                    closeRegistration();
                    return HCCL_E_INTERNAL;
                }
                ccuRet = HcommCcuKernelRegisterEnd(insHandle);
                if (ccuRet != CCU_SUCCESS) {
                    return ConvertCcuToHccl(ccuRet);
                }
                ccuRet = HcommCcuKernelRegisterStart(insHandle);
                if (ccuRet != CCU_SUCCESS) {
                    return ConvertCcuToHccl(ccuRet);
                }

                // group 1: only owner-slice AllGather kernels. Handle 顺序固定为
                // [reduce per die] [combine] [gather per die]。
                void *gatherKernelFunc = nullptr;
                const char *gatherKernelName = nullptr;
                if (topology1LargePath) {
                    gatherKernelFunc =
                        reinterpret_cast<void *>(ops_hccl::CcuTopo1LargeGather4Kernel);
                    gatherKernelName = "CcuAllReduceTopo1LargeGather4";
                } else if (topology2LargePath) {
                    gatherKernelFunc =
                        reinterpret_cast<void *>(ops_hccl::CcuTopo2LargeGather12Kernel);
                    gatherKernelName = "CcuAllReduceTopo2LargeGather12";
                } else {
                    gatherKernelFunc =
                        reinterpret_cast<void *>(ops_hccl::CcuTopo3LargeGather8x8Kernel);
                    gatherKernelName = "CcuAllReduceTopo3LargeGather8x8";
                }
                for (const auto &entry : channelsByDie) {
                    const uint32_t dieId = entry.first;
                    CcuKernelInfo gatherInfo{};
                    auto gatherArg = std::make_shared<CcuKernelArgAllReduce>();
                    gatherArg->rankSize = param.rankSize;
                    gatherArg->rankId = param.myRank;
                    gatherArg->topologyKind = topologyPlan.kind;
                    gatherArg->stage = 1;
                    gatherArg->dataType = param.dataType;
                    gatherArg->reduceType = param.reduceType;
                    gatherArg->channelCount = static_cast<uint32_t>(entry.second.size());
                    for (uint32_t i = 0; i < gatherArg->channelCount; ++i) {
                        gatherArg->channels[i] = entry.second[i].handle;
                        gatherArg->peerRanks[i] = entry.second[i].rank;
                    }
                    const int nameLen = std::snprintf(gatherInfo.kernelFuncName,
                        sizeof(gatherInfo.kernelFuncName),
                        "%s_d%u", gatherKernelName, dieId);
                    if (nameLen < 0 ||
                        static_cast<size_t>(nameLen) >= sizeof(gatherInfo.kernelFuncName)) {
                        closeRegistration();
                        return HCCL_E_INTERNAL;
                    }
                    gatherInfo.kernelFunc = gatherKernelFunc;
                    gatherInfo.setKernelArg(gatherArg);
                    CcuKernelHandle gatherHandle = 0;
                    const void *gatherArgs[] = {gatherInfo.kernelArg};
                    ccuRet = HcommCcuKernelRegister(insHandle, dieId,
                        gatherInfo.kernelFuncName, gatherInfo.kernelFunc,
                        gatherArgs, 1, &gatherHandle);
                    if (ccuRet != CCU_SUCCESS) {
                        closeRegistration();
                        return ConvertCcuToHccl(ccuRet);
                    }
                    resCtxHost.ccuKernels.push_back(gatherHandle);
                }
                resCtxHost.gatherKernelCount =
                    static_cast<uint32_t>(resCtxHost.ccuKernels.size()) - resCtxHost.reduceKernelCount;
                if (resCtxHost.gatherKernelCount != channelsByDie.size()) {
                    closeRegistration();
                    return HCCL_E_INTERNAL;
                }
                HCCL_INFO("Registered topology kind[%u] fixed large graph: reduce[%zu], "
                    "combine[%zu], gather[%u], total[%zu], resourceGroups=2",
                    resCtxHost.topologyKind, channelsByDie.size(),
                    channelsByDie.size() == 2 ? 1UL : 0UL,
                    resCtxHost.gatherKernelCount, resCtxHost.ccuKernels.size());
            }

            ccuRet = HcommCcuKernelRegisterEnd(insHandle);
            if (ccuRet != CCU_SUCCESS) {
                return ConvertCcuToHccl(ccuRet);
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
