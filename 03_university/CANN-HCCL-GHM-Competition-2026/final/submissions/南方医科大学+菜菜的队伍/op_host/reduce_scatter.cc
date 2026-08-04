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
#include <ccu/ccu_launch.h>
#include <acl/acl_rt.h>

#include <map>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace ops_hccl {
CcuResult CcuInboxExchange(CcuKernelArg arg);
CcuResult CcuInboxFold(CcuKernelArg arg);
CcuResult CcuQuartetDeposit(CcuKernelArg arg);
CcuResult CcuTinyTileGraph(CcuKernelArg arg);
CcuResult CcuCapacityInbox(CcuKernelArg arg);
CcuResult CcuRankOrderFold(CcuKernelArg arg);
} // namespace ops_hccl

namespace {
constexpr uint32_t kChannelNotifyNum = 3;
constexpr uint32_t kP4OutputChannelNotifyNum = 4;
constexpr uint64_t kSlotAlignment = 512U;
HcclResult FindCtpLink(HcclComm comm, uint32_t selfRank, uint32_t peerRank, CommLink *selectedLink,
    EndpointAttrDieId *selectedLocalDie, uint32_t *localDieMask)
{
    CHK_PTR_NULL(selectedLink);
    CHK_PTR_NULL(selectedLocalDie);
    CHK_PTR_NULL(localDieMask);
    bool found = false;
    *localDieMask = 0;
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    for (uint32_t layerIdx = 0; layerIdx < layerCount; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkCount = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layers[layerIdx], selfRank, peerRank, &links, &linkCount));
        for (uint32_t linkIdx = 0; linkIdx < linkCount; ++linkIdx) {
            if (links[linkIdx].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            EndpointAttrDieId localDie = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, selfRank, &links[linkIdx].srcEndpointDesc,
                ENDPOINT_ATTR_DIE_ID, sizeof(localDie), &localDie));
            if (localDie < 32) {
                *localDieMask |= 1U << localDie;
            }
            if (!found) {
                *selectedLink = links[linkIdx];
                *selectedLocalDie = localDie;
                found = true;
            }
        }
    }
    if (found) {
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[FindCtpLink] no CTP link from rank %u to rank %u", selfRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult ConvertCcuResult(CcuResult ret)
{
    if (ret == CCU_SUCCESS) {
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("CCU operation failed, ccuRet=%d", static_cast<int32_t>(ret));
    return HCCL_E_INTERNAL;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType scalarType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = scalarType;
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
    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM ||
        (param.rankSize != 4 && param.rankSize != 12 && param.rankSize != 16)) {
        HCCL_ERROR("[HcclReduceScatter] only FP32 SUM on the 4, 12, and 16-rank competition topologies is supported");
        return HCCL_E_PARA;
    }
    const WorkloadCell cell = ClassifyWorkload(param.rankSize, param.count);
    const char *cellTag = WorkloadKey(cell);
    if (cell == WorkloadCell::INVALID || cellTag == nullptr) {
        HCCL_ERROR("[HcclReduceScatter] unsupported competition cell worldSize=%u recvCount=%lu", param.rankSize,
            param.count);
        return HCCL_E_PARA;
    }
    snprintf(param.tag, sizeof(param.tag), "%s", cellTag);

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
        DispatchContext blueprint;
        blueprint.version = WorkloadEpoch(cell);
        blueprint.cell = cell;
        const bool p4Large = cell == WorkloadCell::P4_512M || cell == WorkloadCell::P4_400M4B;
        const bool directMeshSmall = cell == WorkloadCell::P4_512K || cell == WorkloadCell::P16_512K;
        const bool capacityAwarePull = cell == WorkloadCell::P12_512K;
        const bool v1SlotGather = cell == WorkloadCell::P16_512M ||
            cell == WorkloadCell::P16_400M4B || cell == WorkloadCell::P12_512M ||
            cell == WorkloadCell::P12_400M4B;
        blueprint.algorithm = directMeshSmall ? ScheduleKind::DIRECT_MESH_SMALL
            : (p4Large ? ScheduleKind::P4_OUTPUT_ACCUMULATE
                : (capacityAwarePull ? ScheduleKind::CAPACITY_AWARE_PULL
                                     : ScheduleKind::SLOT_GATHER));

        // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread。EngineCtx 命中时复用已绑定的 Thread，
        // 不重复申请一个随后不会使用的句柄。
        CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

        // ==============================================
        // STEP 2.2: 申请资源：Thread、Channel、CCU Kernel
        // ==============================================
        blueprint.workers.push_back(param.cpuThread);

        struct ChannelInfo {
            uint32_t peerRank;
            ChannelHandle channel;
        };
        std::map<EndpointAttrDieId, std::vector<ChannelInfo>> dieGroups;
        uint32_t commonLocalDieMask = UINT32_MAX;
        uint32_t unionLocalDieMask = 0;
        for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
            if (peerRank == param.myRank) {
                continue;
            }
            CommLink link;
            EndpointAttrDieId dieId = 0;
            uint32_t localDieMask = 0;
            CHK_RET(FindCtpLink(comm, param.myRank, peerRank, &link, &dieId, &localDieMask));
            commonLocalDieMask &= localDieMask;
            unionLocalDieMask |= localDieMask;
            HcclChannelDesc desc;
            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = peerRank;
            desc.notifyNum = p4Large ? kP4OutputChannelNotifyNum : (v1SlotGather ? kChannelNotifyNum : 1U);
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint = link.srcEndpointDesc;
            desc.remoteEndpoint = link.dstEndpointDesc;
            ChannelHandle channel = 0;
            CHK_RET(HcclChannelAcquire(comm, ccuEngine, &desc, 1, &channel));
            dieGroups[dieId].push_back({peerRank, channel});
        }
        HCCL_INFO("[CtpCandidateRankSummary] selfRank=%u commonLocalDieMask=0x%x unionLocalDieMask=0x%x",
            param.myRank, commonLocalDieMask, unionLocalDieMask);
        const uint32_t expectedGroupCount = param.rankSize == 4 ? 1U : 2U;
        if (dieGroups.size() != expectedGroupCount) {
            HCCL_ERROR("[HcclReduceScatter] rank size %u requires %u non-empty CCU Die groups, got %zu",
                param.rankSize, expectedGroupCount, dieGroups.size());
            return HCCL_E_PARA;
        }
        if (dieGroups.empty()) {
            HCCL_ERROR("[HcclReduceScatter] no CCU lanes were acquired");
            return HCCL_E_PARA;
        }
        std::vector<EndpointAttrDieId> orderedDieIds;
        for (const auto &[dieId, group] : dieGroups) {
            (void)group;
            orderedDieIds.push_back(dieId);
        }
        if ((directMeshSmall || capacityAwarePull) && orderedDieIds.size() == 2U &&
            dieGroups.at(orderedDieIds[1]).size() < dieGroups.at(orderedDieIds[0]).size()) {
            std::swap(orderedDieIds[0], orderedDieIds[1]);
        }

        if (expectedGroupCount == 2) {
            void *scratch = nullptr;
            CHK_RET(HcclGetHcclBuffer(comm, &scratch, &blueprint.stagingBytes));
            blueprint.stagingBase = reinterpret_cast<uint64_t>(scratch);
        }
        if (expectedGroupCount == 2) {
            aclrtStream secondaryStream = nullptr;
            if (aclrtCreateStream(&secondaryStream) != ACL_SUCCESS) {
                HCCL_ERROR("[HcclReduceScatter] failed to create dual-Die stream synchronization resources");
                return HCCL_E_INTERNAL;
            }
            ThreadHandle secondaryThread = 0;
            CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, secondaryStream, 1, &secondaryThread));
            blueprint.workers.push_back(secondaryThread);
        }

        std::vector<uint32_t> slotForRank(param.rankSize, MAX_RANK_SIZE);
        std::vector<uint32_t> directPositionBySource(param.rankSize, MAX_RANK_SIZE);
        uint32_t slotTotal = 0U;
        uint32_t spillCount = 0U;
        uint64_t slotStride = 0U;
        if (capacityAwarePull) {
            const uint64_t resultBytes = param.count * sizeof(float);
            slotStride = (resultBytes + kSlotAlignment - 1U) & ~(kSlotAlignment - 1U);
            slotTotal = std::min<uint32_t>(param.rankSize - 1U,
                static_cast<uint32_t>(blueprint.stagingBytes / slotStride));
            spillCount = param.rankSize - 1U - slotTotal;
            if (slotTotal == 0U || spillCount > 3U) {
                HCCL_ERROR("[HcclReduceScatter] unsupported hybrid-spill layout slots=%u spills=%u",
                    slotTotal, spillCount);
                return HCCL_E_PARA;
            }
            const auto &coordinatorGroup = dieGroups.at(orderedDieIds.front());
            if (coordinatorGroup.size() < spillCount) {
                HCCL_ERROR("[HcclReduceScatter] primary Die cannot host %u spill pulls", spillCount);
                return HCCL_E_PARA;
            }
            for (uint32_t position = 0U; position < spillCount; ++position) {
                directPositionBySource[coordinatorGroup[position].peerRank] = position;
            }
            uint32_t nextSlot = 0U;
            for (uint32_t sourceRank = 0U; sourceRank < param.rankSize; ++sourceRank) {
                if (sourceRank == param.myRank) {
                    continue;
                }
                if (directPositionBySource[sourceRank] == MAX_RANK_SIZE) {
                    slotForRank[sourceRank] = nextSlot++;
                }
            }
            if (nextSlot != slotTotal) {
                HCCL_ERROR("[HcclReduceScatter] invalid capacity-aware mapping slots=%u spills=%u",
                    slotTotal, spillCount);
                return HCCL_E_PARA;
            }
        }

        CcuInsHandle insHandle = 0;
        uint32_t insNum = 0;
        CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
        if (insNum != 1) {
            HCCL_ERROR("[HcclReduceScatter] expected one CCU instance, got %u", insNum);
            return HCCL_E_INTERNAL;
        }
        CHK_RET(ConvertCcuResult(HcommCcuKernelRegisterStart(insHandle)));
        blueprint.groupCuts.push_back(0);
        for (const EndpointAttrDieId dieId : orderedDieIds) {
            const auto &group = dieGroups.at(dieId);
            std::shared_ptr<ExchangeKernelSpec> kernelArg;
            if (directMeshSmall) {
                auto smallArg = std::make_shared<TinyMeshKernelSpec>();
                smallArg->laneCount = static_cast<uint32_t>(group.size());
                smallArg->selfRank = param.myRank;
                smallArg->worldSize = param.rankSize;
                smallArg->isLeader = dieId == orderedDieIds.front() ? 1U : 0U;
                smallArg->resultBytes = param.count * sizeof(float);
                smallArg->blockBytes = 4096U;
                smallArg->tileParallelism = param.rankSize == 4U ? 16U :
                    static_cast<uint32_t>(smallArg->resultBytes / smallArg->blockBytes);
                smallArg->scalarType = param.dataType;
                smallArg->combineOp = param.reduceType;
                for (uint32_t channelIdx = 0; channelIdx < smallArg->laneCount; ++channelIdx) {
                    smallArg->lanes[channelIdx] = group[channelIdx].channel;
                    blueprint.groupRanks.push_back(group[channelIdx].peerRank);
                }
                blueprint.groupCuts.push_back(static_cast<uint32_t>(blueprint.groupRanks.size()));
                const void *kernelArgsArray[] = {smallArg.get()};
                CcuKernelHandle kernelHandle = 0;
                CHK_RET(ConvertCcuResult(HcommCcuKernelRegister(insHandle, dieId, "CcuTinyTileGraph",
                    reinterpret_cast<void *>(ops_hccl::CcuTinyTileGraph), kernelArgsArray, 1, &kernelHandle)));
                blueprint.programs.push_back(kernelHandle);
                continue;
            } else if (capacityAwarePull) {
                auto pullArg = std::make_shared<CapacityInboxSpec>();
                pullArg->laneCount = static_cast<uint32_t>(group.size());
                pullArg->selfRank = param.myRank;
                pullArg->worldSize = param.rankSize;
                pullArg->resultBytes = param.count * sizeof(float);
                pullArg->slotPitch = slotStride;
                pullArg->slotTotal = slotTotal;
                pullArg->scalarType = param.dataType;
                pullArg->combineOp = param.reduceType;
                for (uint32_t channelIdx = 0; channelIdx < pullArg->laneCount; ++channelIdx) {
                    pullArg->lanes[channelIdx] = group[channelIdx].channel;
                    pullArg->remoteRanks[channelIdx] = group[channelIdx].peerRank;
                    pullArg->fusedOrder[channelIdx] =
                        directPositionBySource[group[channelIdx].peerRank];
                    pullArg->inboxSlots[channelIdx] = slotForRank[group[channelIdx].peerRank];
                    blueprint.groupRanks.push_back(group[channelIdx].peerRank);
                }
                blueprint.groupCuts.push_back(static_cast<uint32_t>(blueprint.groupRanks.size()));
                const void *kernelArgsArray[] = {pullArg.get()};
                CcuKernelHandle pullKernel = 0;
                CHK_RET(ConvertCcuResult(HcommCcuKernelRegister(insHandle, dieId,
                    "CcuCapacityInbox", reinterpret_cast<void *>(ops_hccl::CcuCapacityInbox),
                    kernelArgsArray, 1, &pullKernel)));
                blueprint.programs.push_back(pullKernel);
                continue;
            } else {
                kernelArg = std::make_shared<ExchangeKernelSpec>();
            }
            kernelArg->laneCount = static_cast<uint32_t>(group.size());
            kernelArg->selfRank = param.myRank;
            kernelArg->worldSize = param.rankSize;
            kernelArg->resultBytes = param.count * sizeof(float);
            const uint64_t stripeBytes = (kernelArg->resultBytes / 3U) & ~3ULL;
            kernelArg->stripeOffsets[0] = 0;
            kernelArg->stripeOffsets[1] = stripeBytes;
            kernelArg->stripeOffsets[2] = 2U * stripeBytes;
            kernelArg->stripeBytes[0] = kernelArg->stripeBytes[1] = stripeBytes;
            kernelArg->stripeBytes[2] = kernelArg->resultBytes - 2U * stripeBytes;
            for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
                kernelArg->lanes[channelIdx] = group[channelIdx].channel;
                kernelArg->remoteRanks[channelIdx] = group[channelIdx].peerRank;
                blueprint.groupRanks.push_back(group[channelIdx].peerRank);
            }
            blueprint.groupCuts.push_back(static_cast<uint32_t>(blueprint.groupRanks.size()));
            const void *kernelArgsArray[] = {kernelArg.get()};
            CcuKernelHandle kernelHandle = 0;
            const bool p4Output = blueprint.algorithm == ScheduleKind::P4_OUTPUT_ACCUMULATE;
            const char *kernelName = p4Output ? "CcuQuartetDeposit" : "CcuInboxExchange";
            void *kernelFunc = p4Output ? reinterpret_cast<void *>(ops_hccl::CcuQuartetDeposit)
                                        : reinterpret_cast<void *>(ops_hccl::CcuInboxExchange);
            CHK_RET(ConvertCcuResult(HcommCcuKernelRegister(insHandle, dieId, kernelName,
                kernelFunc, kernelArgsArray, 1, &kernelHandle)));
            blueprint.programs.push_back(kernelHandle);
        }
        if (capacityAwarePull) {
            auto foldArg = std::make_shared<CapacityFoldSpec>();
            foldArg->laneCount = 0U;
            foldArg->selfRank = param.myRank;
            foldArg->worldSize = param.rankSize;
            std::copy(slotForRank.begin(), slotForRank.end(), foldArg->slotForRank);
            foldArg->resultBytes = param.count * sizeof(float);
            foldArg->slotPitch = slotStride;
            foldArg->slotTotal = slotTotal;
            foldArg->scalarType = param.dataType;
            foldArg->combineOp = param.reduceType;
            const void *kernelArgsArray[] = {foldArg.get()};
            CcuKernelHandle foldKernel = 0;
            CHK_RET(ConvertCcuResult(HcommCcuKernelRegister(insHandle, orderedDieIds.front(),
                "CcuRankOrderFold", reinterpret_cast<void *>(ops_hccl::CcuRankOrderFold),
                kernelArgsArray, 1, &foldKernel)));
            blueprint.programs.push_back(foldKernel);
        }
        if (expectedGroupCount == 2 && !directMeshSmall && !capacityAwarePull) {
            auto mergeArg = std::make_shared<MergeKernelSpec>();
            mergeArg->laneCount = 0;
            mergeArg->remoteCount = param.rankSize - 1;
            const void *mergeArgsArray[] = {mergeArg.get()};
            CcuKernelHandle mergeKernel = 0;
            const EndpointAttrDieId mergeDieId = orderedDieIds.front();
            CHK_RET(ConvertCcuResult(HcommCcuKernelRegister(insHandle, mergeDieId, "CcuInboxFold",
                reinterpret_cast<void *>(ops_hccl::CcuInboxFold), mergeArgsArray, 1, &mergeKernel)));
            blueprint.programs.push_back(mergeKernel);
        }
        CHK_RET(ConvertCcuResult(HcommCcuKernelRegisterEnd(insHandle)));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 CCU 通信引擎上下文，存放 DispatchContext 信息
        std::vector<char> seq = blueprint.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::DispatchReduceScatter(param, stream));
    return HCCL_SUCCESS;
}
