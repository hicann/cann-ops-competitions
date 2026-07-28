/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <vector>

#include <ccu/ccu_control_flow_macro.h>
#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "log.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {
namespace die_group_allreduce {

constexpr uint32_t ROOT_RANK = 0;

constexpr uint32_t REMOTE_BUFFER_ADDR_SLOT = 0;
constexpr uint32_t REMOTE_BUFFER_TOKEN_SLOT = 1;
constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;

constexpr uint16_t BUFFER_ADDR_READY_MASK = 1U << 0;
constexpr uint16_t BUFFER_TOKEN_READY_MASK = 1U << 1;
constexpr uint16_t TRANSFER_DONE_MASK = 1U << 2;
constexpr uint16_t BUFFER_READY_MASK =
    BUFFER_ADDR_READY_MASK | BUFFER_TOKEN_READY_MASK;

constexpr uint64_t PHASE_REDUCE = 0;
constexpr uint64_t PHASE_BROADCAST = 1;

constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;

struct DieGroupKernelArg : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t peerRanks[MAX_RANK_SIZE];
    bool isRoot;
    HcclDataType dataType;
    HcclReduceOp reduceType;
};

#ifndef CCU_CHK_RET
#define OPS_HCCL_LOCAL_CCU_CHK_RET
#define CCU_CHK_RET(call)                \
    do {                                 \
        CcuResult result = (call);       \
        if (result != CCU_SUCCESS) {     \
            return result;               \
        }                                \
    } while (0)
#endif

CcuResult DieGroupKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<DieGroupKernelArg *>(arg);
    if (kernelArg == nullptr ||
        kernelArg->rankSize <= 1 ||
        kernelArg->rankSize > MAX_RANK_SIZE ||
        kernelArg->rankId >= kernelArg->rankSize ||
        kernelArg->channelCount == 0 ||
        kernelArg->channelCount >= kernelArg->rankSize) {
        return CCU_E_PARA;
    }

    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        if (kernelArg->peerRanks[index] >= kernelArg->rankSize ||
            kernelArg->peerRanks[index] == kernelArg->rankId ||
            (index > 0 && kernelArg->peerRanks[index - 1] >= kernelArg->peerRanks[index])) {
            return CCU_E_PARA;
        }
    }

    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable cclBufferAddr;
    ccu::Variable cclBufferToken;
    ccu::Variable tileBytes;
    ccu::Variable phase;

    ccu::Variable remoteBufferAddr[MAX_RANK_SIZE];
    ccu::Variable remoteBufferToken[MAX_RANK_SIZE];

    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        const ChannelHandle channel = kernelArg->channels[index];
        remoteBufferAddr[index] = ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_BUFFER_ADDR_SLOT);
        remoteBufferToken[index] = ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_BUFFER_TOKEN_SLOT);
    }

    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(outputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(cclBufferAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(cclBufferToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(tileBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(phase, argId++));

    ccu::LocalAddr output;
    output.addr = outputAddr;
    output.token = outputToken;

    ccu::Event event;

    CCU_IF(phase == PHASE_REDUCE) {
        if (kernelArg->isRoot) {
            if (kernelArg->channelCount == 1) {
                const ChannelHandle channel = kernelArg->channels[0];
                CCU_CHK_RET(ccu::NotifyWait(channel, CHANNEL_NOTIFY_INDEX, BUFFER_READY_MASK));

                ccu::RemoteAddr remoteSource;
                remoteSource.addr = remoteBufferAddr[0];
                remoteSource.token = remoteBufferToken[0];

                CCU_CHK_RET(ccu::ReadReduce(
                    channel, output, remoteSource, tileBytes,
                    kernelArg->dataType, kernelArg->reduceType, event));
                CCU_CHK_RET(ccu::EventWait(event));
                CCU_CHK_RET(ccu::NotifyRecord(channel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
            } else {
                ccu::Event readEvents[MAX_RANK_SIZE];
                std::vector<ccu::LocalAddr> reduceNodes(kernelArg->channelCount + 1);
                reduceNodes[0].addr = outputAddr;
                reduceNodes[0].token = outputToken;
            
                ccu::Variable currentScratchAddr;
                currentScratchAddr = cclBufferAddr;

                for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
                    const ChannelHandle channel = kernelArg->channels[index];

                    CCU_CHK_RET(ccu::NotifyWait(channel, CHANNEL_NOTIFY_INDEX, BUFFER_READY_MASK));

                    ccu::RemoteAddr remoteSource;
                    remoteSource.addr = remoteBufferAddr[index];
                    remoteSource.token = remoteBufferToken[index];

                    reduceNodes[index + 1].addr = currentScratchAddr;
                    reduceNodes[index + 1].token = cclBufferToken;

                    CCU_CHK_RET(ccu::Read(
                        channel, reduceNodes[index + 1], remoteSource, tileBytes, readEvents[index]));
                    currentScratchAddr += tileBytes;
                }

                ccu::Event reduceEvent;
                const uint32_t operandCount = kernelArg->channelCount + 1;
                uint16_t roundMask = 0;
                uint32_t pairIndex = 0;

                for (uint32_t left = 0; left + 1 < operandCount; left += 2) {
                    if (left == 0) {
                        const ChannelHandle channel = kernelArg->channels[0];
                        CCU_CHK_RET(ccu::EventWait(readEvents[0]));
                        CCU_CHK_RET(ccu::NotifyRecord(
                            channel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
                    } else {
                        const uint32_t firstPeer = left - 1;
                        const uint32_t secondPeer = left;
                        const ChannelHandle firstChannel = kernelArg->channels[firstPeer];
                        const ChannelHandle secondChannel = kernelArg->channels[secondPeer];

                        CCU_CHK_RET(ccu::EventWait(readEvents[firstPeer]));
                        CCU_CHK_RET(ccu::NotifyRecord(
                            firstChannel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
                        CCU_CHK_RET(ccu::EventWait(readEvents[secondPeer]));
                        CCU_CHK_RET(ccu::NotifyRecord(
                            secondChannel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
                    }

                    const uint16_t pairMask = static_cast<uint16_t>(1U << pairIndex++);
                    roundMask = static_cast<uint16_t>(roundMask | pairMask);
                    CCU_CHK_RET(ccu::LocalReduce(
                        reduceNodes[left], reduceNodes[left + 1], tileBytes,
                        kernelArg->dataType, kernelArg->reduceType, reduceEvent, pairMask));
                }

                if ((operandCount & 1U) != 0) {
                    const uint32_t lastPeer = operandCount - 2;
                    const ChannelHandle channel = kernelArg->channels[lastPeer];
                    CCU_CHK_RET(ccu::EventWait(readEvents[lastPeer]));
                    CCU_CHK_RET(ccu::NotifyRecord(
                        channel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
                }

                CCU_CHK_RET(ccu::EventWait(reduceEvent, roundMask));

                for (uint32_t stride = 2; stride < operandCount; stride *= 2) {
                    roundMask = 0;
                    pairIndex = 0;

                    for (uint32_t left = 0; left + stride < operandCount; left += stride * 2) {
                        const uint16_t pairMask = static_cast<uint16_t>(1U << pairIndex++);
                        roundMask = static_cast<uint16_t>(roundMask | pairMask);
                        CCU_CHK_RET(ccu::LocalReduce(
                            reduceNodes[left], reduceNodes[left + stride], tileBytes,
                            kernelArg->dataType, kernelArg->reduceType, reduceEvent, pairMask));
                    }

                    CCU_CHK_RET(ccu::EventWait(reduceEvent, roundMask));
                }
            }
        } else {
            const ChannelHandle channel = kernelArg->channels[0];

            CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, cclBufferAddr, REMOTE_BUFFER_ADDR_SLOT, CHANNEL_NOTIFY_INDEX, BUFFER_ADDR_READY_MASK));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, cclBufferToken, REMOTE_BUFFER_TOKEN_SLOT, CHANNEL_NOTIFY_INDEX, BUFFER_TOKEN_READY_MASK));
            CCU_CHK_RET(ccu::NotifyWait(channel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
        }
    } CCU_ELSE {
        if (kernelArg->isRoot) {
            for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
                const ChannelHandle channel = kernelArg->channels[index];

                CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, outputAddr, REMOTE_BUFFER_ADDR_SLOT, CHANNEL_NOTIFY_INDEX, BUFFER_ADDR_READY_MASK));
                CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, outputToken, REMOTE_BUFFER_TOKEN_SLOT, CHANNEL_NOTIFY_INDEX, BUFFER_TOKEN_READY_MASK));
            }
            
            for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
                const ChannelHandle channel = kernelArg->channels[index];
                CCU_CHK_RET(ccu::NotifyWait(channel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
            }
        } else {
            const ChannelHandle channel = kernelArg->channels[0];

            CCU_CHK_RET(ccu::NotifyWait(channel, CHANNEL_NOTIFY_INDEX, BUFFER_READY_MASK));

            ccu::RemoteAddr remoteSource;
            remoteSource.addr = remoteBufferAddr[0];
            remoteSource.token = remoteBufferToken[0];

            CCU_CHK_RET(ccu::Read(channel, output, remoteSource, tileBytes, event));
            CCU_CHK_RET(ccu::EventWait(event));

            CCU_CHK_RET(ccu::NotifyRecord(channel, CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
        }
    }

    return CCU_SUCCESS;
}

#ifdef OPS_HCCL_LOCAL_CCU_CHK_RET
#undef CCU_CHK_RET
#undef OPS_HCCL_LOCAL_CCU_CHK_RET
#endif

HcclResult Execute(const OpParam &param)
{
    CHK_PTR_NULL(param.resCtx);

    CHK_PRT_RET(
        param.ctxSize == 0,
        HCCL_ERROR("Empty CCU resource context"),
        HCCL_E_INTERNAL);

    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> serialized(ctx, ctx + param.ctxSize);

    AlgResourceCtx resCtx;
    resCtx.DeSerialize(serialized);

    constexpr uint64_t dataTypeSize = sizeof(float);

    CHK_PRT_RET(
        param.count > UINT64_MAX / dataTypeSize,
        HCCL_ERROR("AllReduce byte size overflow"),
        HCCL_E_PARA);

    const uint64_t totalBytes = param.count * dataTypeSize;

    if (totalBytes == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        resCtx.threads.empty(),
        HCCL_ERROR("No CCU thread"),
        HCCL_E_INTERNAL);

    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(
                param.cpuThread, param.outputPtr, param.inputPtr, totalBytes));
    }

    CHK_PTR_NULL(resCtx.localBuffer.addr);

    CHK_PRT_RET(
        resCtx.localBuffer.size < dataTypeSize,
        HCCL_ERROR("HCCL Buffer is too small"),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(
        resCtx.ccuKernels.empty(),
        HCCL_ERROR("No CCU kernel was registered"),
        HCCL_E_INTERNAL);

    const uint32_t scratchSlots = param.rankSize - 1;
    uint64_t maxTileBytes = resCtx.localBuffer.size / scratchSlots;

    // 绝对安全的硬件级 DMA 内存对齐 (Cache-Line 友好)
    constexpr uint64_t ALIGN_BYTES = 256;
    if (maxTileBytes >= ALIGN_BYTES) {
        maxTileBytes = (maxTileBytes / ALIGN_BYTES) * ALIGN_BYTES;
    } else {
        // 极端边缘情况下的安全退路
        maxTileBytes = (maxTileBytes / dataTypeSize) * dataTypeSize;
    }

    CHK_PRT_RET(
        maxTileBytes == 0,
        HCCL_ERROR("Invalid tile size after alignment"),
        HCCL_E_INTERNAL);

    uint64_t outputToken = 0;
    uint64_t inputToken = 0;
    uint64_t cclBufferToken = 0;

    const uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t cclBufferAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    const bool isRoot = (param.myRank == ROOT_RANK);

    CcuResult ccuRet = HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.outputPtr), totalBytes, &outputToken);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("Failed to get output memory token: %d", ccuRet), HCCL_E_INTERNAL);

    if (!isRoot) {
        ccuRet = HcommCcuGetMemToken(baseInputAddr, totalBytes, &inputToken);
        CHK_PRT_RET(
            ccuRet != CCU_SUCCESS, HCCL_ERROR("Failed to get input memory token: %d", ccuRet), HCCL_E_INTERNAL);
    }

    if (isRoot) {
        ccuRet = HcommCcuGetMemToken(cclBufferAddr, resCtx.localBuffer.size, &cclBufferToken);
        CHK_PRT_RET(
            ccuRet != CCU_SUCCESS, HCCL_ERROR("Failed to get HCCL Buffer token: %d", ccuRet), HCCL_E_INTERNAL);
    }

    for (uint64_t offset = 0; offset < totalBytes;) {
        const uint64_t remaining = totalBytes - offset;
        const uint64_t tileBytes = std::min<uint64_t>(remaining, maxTileBytes);
        const uint64_t tileInputAddr = baseInputAddr + offset;
        const uint64_t tileOutputAddr = baseOutputAddr + offset;

        if (isRoot) {
            const int32_t copyRet = HcommLocalCopyOnThread(
                param.cpuThread, reinterpret_cast<void *>(tileOutputAddr),
                reinterpret_cast<const void *>(tileInputAddr), tileBytes);
            CHK_PRT_RET(
                copyRet != HCCL_SUCCESS, HCCL_ERROR("Failed to seed AllReduce output: %d", copyRet), HCCL_E_INTERNAL);
        }

        uint64_t taskArgs[] = {
            tileOutputAddr,
            outputToken,
            isRoot ? cclBufferAddr : tileInputAddr,
            isRoot ? cclBufferToken : inputToken,
            tileBytes,
            PHASE_REDUCE,
        };

        for (CcuKernelHandle kernel : resCtx.ccuKernels) {
            ccuRet = HcommCcuKernelLaunch(
                param.cpuThread, kernel, taskArgs, sizeof(taskArgs) / sizeof(taskArgs[0]));
            CHK_PRT_RET(
                ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU reduce launch failed: %d", ccuRet), HCCL_E_INTERNAL);
        }

        taskArgs[5] = PHASE_BROADCAST;
        for (CcuKernelHandle kernel : resCtx.ccuKernels) {
            ccuRet = HcommCcuKernelLaunch(
                param.cpuThread, kernel, taskArgs, sizeof(taskArgs) / sizeof(taskArgs[0]));
            CHK_PRT_RET(
                ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU broadcast launch failed: %d", ccuRet), HCCL_E_INTERNAL);
        }

        offset += tileBytes;
    }

    return HCCL_SUCCESS;
}

} // namespace die_group_allreduce
} // namespace ops_hccl

HcclResult HcclAllReduce(
    void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    CHK_PRT_RET(
        dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only FP32 SUM is supported"), HCCL_E_NOT_SUPPORT);

    OpParam param{};
    (void)snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allreduce_die_group_v10_aligned");

    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};

    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    CHK_PRT_RET(
        param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank information"), HCCL_E_PARA);

    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;

    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx resCtxHost{};
        resCtxHost.ccuThread = param.cpuThread;
        resCtxHost.threads.push_back(param.cpuThread);

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;

        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        std::vector<ChannelHandle> channels;
        std::vector<uint32_t> channelPeerRanks;
        std::vector<uint32_t> channelDieIds;

        if (param.rankSize > 1) {
            uint32_t *layers = nullptr;
            uint32_t layerNum = 0;

            CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
            CHK_PRT_RET(layers == nullptr || layerNum == 0, HCCL_ERROR("No rank-graph layer"), HCCL_E_INTERNAL);

            const uint32_t channelNum = param.rankSize - 1;
            std::vector<HcclChannelDesc> channelDescs(channelNum);

            CHK_RET(HcclChannelDescInit(channelDescs.data(), channelNum));

            channelPeerRanks.reserve(channelNum);
            channelDieIds.reserve(channelNum);

            const auto protocolPriority = [](CommProtocol protocol) -> uint32_t {
                if (protocol == COMM_PROTOCOL_UBC_CTP) { return 0; }
                if (protocol == COMM_PROTOCOL_UBC_TP) { return 1; }
                return UINT32_MAX;
            };

            uint32_t channelIndex = 0;

            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                if (remoteRank == param.myRank) { continue; }

                bool found = false;
                CommLink selectedLink{};
                uint32_t selectedPriority = UINT32_MAX;
                uint8_t selectedHop = UINT8_MAX;

                for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
                    CommLink *links = nullptr;
                    uint32_t linkNum = 0;

                    CHK_RET(HcclRankGraphGetLinks(
                        comm, layers[layerIndex], param.myRank, remoteRank, &links, &linkNum));

                    for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
                        const uint32_t priority = protocolPriority(links[linkIndex].linkAttr.linkProtocol);
                        const uint8_t hop = links[linkIndex].linkAttr.hop;

                        if (priority == UINT32_MAX) { continue; }

                        if (!found || priority < selectedPriority ||
                            (priority == selectedPriority && hop < selectedHop)) {
                            selectedLink = links[linkIndex];
                            selectedPriority = priority;
                            selectedHop = hop;
                            found = true;
                        }
                    }
                }

                CHK_PRT_RET(
                    !found, HCCL_ERROR("No supported link from rank[%u] to rank[%u]", param.myRank, remoteRank),
                    HCCL_E_NOT_FOUND);

                HcclChannelDesc &desc = channelDescs[channelIndex++];
                desc.remoteRank = remoteRank;
                desc.notifyNum = ops_hccl::die_group_allreduce::CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = selectedLink.linkAttr.linkProtocol;
                desc.localEndpoint = selectedLink.srcEndpointDesc;
                desc.remoteEndpoint = selectedLink.dstEndpointDesc;

                EndpointAttrDieId dieId = UINT32_MAX;

                if (param.myRank == ops_hccl::die_group_allreduce::ROOT_RANK ||
                    remoteRank == ops_hccl::die_group_allreduce::ROOT_RANK) {
                    CHK_RET(HcclRankGraphGetEndpointInfo(
                        comm, param.myRank, &selectedLink.srcEndpointDesc,
                        ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
                }

                channelPeerRanks.push_back(remoteRank);
                channelDieIds.push_back(dieId);
            }

            channels.resize(channelNum);
            CHK_RET(HcclChannelAcquire(comm, ccuEngine, channelDescs.data(), channelNum, channels.data()));
        }

        if (param.rankSize > 1) {
            std::map<uint32_t, std::vector<uint32_t>> dieGroups;

            if (param.myRank == ops_hccl::die_group_allreduce::ROOT_RANK) {
                for (uint32_t index = 0; index < channels.size(); ++index) {
                    CHK_PRT_RET(
                        channelDieIds[index] == UINT32_MAX,
                        HCCL_ERROR("Missing Die id for channel[%u]", index), HCCL_E_INTERNAL);
                    dieGroups[channelDieIds[index]].push_back(index);
                }
            } else {
                auto rootIt = std::find(
                    channelPeerRanks.begin(), channelPeerRanks.end(), ops_hccl::die_group_allreduce::ROOT_RANK);
                CHK_PRT_RET(
                    rootIt == channelPeerRanks.end(),
                    HCCL_ERROR("Channel to root rank is missing"), HCCL_E_INTERNAL);

                const uint32_t rootChannelIndex = static_cast<uint32_t>(rootIt - channelPeerRanks.begin());
                CHK_PRT_RET(
                    channelDieIds[rootChannelIndex] == UINT32_MAX,
                    HCCL_ERROR("Missing Die id for root channel"), HCCL_E_INTERNAL);

                dieGroups[channelDieIds[rootChannelIndex]].push_back(rootChannelIndex);
            }

            CcuInsHandle insHandle = 0;
            uint32_t insNum = 0;

            CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
            CHK_PRT_RET(
                insNum != 1, HCCL_ERROR("Expected one CCU instance, got %u", insNum), HCCL_E_INTERNAL);

            CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
            CHK_PRT_RET(
                ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU register start failed: %d", ccuRet), HCCL_E_INTERNAL);

            std::vector<std::shared_ptr<ops_hccl::die_group_allreduce::DieGroupKernelArg>> keepAlive;
            keepAlive.reserve(dieGroups.size());

            for (const auto &group : dieGroups) {
                const uint32_t dieId = group.first;
                const std::vector<uint32_t> &channelIndices = group.second;
                CcuKernelInfo kernelInfo{};

                (void)snprintf(
                    kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
                    "DieGroupAllReduce_r%u_d%u", param.myRank, dieId);

                kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::die_group_allreduce::DieGroupKernel);
                auto kernelArg = std::make_shared<ops_hccl::die_group_allreduce::DieGroupKernelArg>();

                kernelArg->rankSize = param.rankSize;
                kernelArg->rankId = param.myRank;
                kernelArg->isRoot = param.myRank == ops_hccl::die_group_allreduce::ROOT_RANK;
                kernelArg->dataType = param.dataType;
                kernelArg->reduceType = param.reduceType;
                kernelArg->channelCount = static_cast<uint32_t>(channelIndices.size());

                CHK_PRT_RET(
                    kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE,
                    HCCL_ERROR("Invalid channel count in Die group"), HCCL_E_INTERNAL);

                for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
                    const uint32_t globalIndex = channelIndices[index];
                    kernelArg->channels[index] = channels[globalIndex];
                    kernelArg->peerRanks[index] = channelPeerRanks[globalIndex];
                }

                kernelInfo.setKernelArg(kernelArg);
                keepAlive.push_back(kernelArg);
                const void *kernelArgs[] = {kernelInfo.kernelArg};
                CcuKernelHandle kernelHandle = 0;

                ccuRet = HcommCcuKernelRegister(
                    insHandle, dieId, kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
                CHK_PRT_RET(
                    ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register failed for Die[%u]: %d", dieId, ccuRet),
                    HCCL_E_INTERNAL);

                resCtxHost.ccuKernels.push_back(kernelHandle);
            }

            ccuRet = HcommCcuKernelRegisterEnd(insHandle);
            CHK_PRT_RET(
                ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU register end failed: %d", ccuRet), HCCL_E_INTERNAL);

            (void)keepAlive;
        }

        std::vector<char> serialized = resCtxHost.Serialize();
        param.ctxSize = serialized.size();

        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, serialized.data(), serialized.size(), 0));
    }

    return ops_hccl::die_group_allreduce::Execute(param);
}