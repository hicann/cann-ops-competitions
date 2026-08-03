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
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t SERVER_SIZE = 8;
constexpr uint32_t RELAY_COPIES = 5;

bool UseSmallMessageFastPath(const OpParam &param)
{
    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    return sizeIt != SIZE_TABLE.end() && sizeIt->second != 0 &&
        param.count <= ops_hccl::DIRECT_FAST_PATH_MAX_BYTES / sizeIt->second;
}

enum class ScheduleNetwork : uint8_t {
    MESH = 0,
    CLOS = 1,
};

struct ScheduleAction {
    std::string id;
    ScheduleNetwork network = ScheduleNetwork::MESH;
    ops_hccl::TransferKind kind = ops_hccl::TransferKind::NATIVE;
    uint32_t src = 0;
    uint32_t dst = 0;
    uint32_t owner = 0;
    uint32_t chunk = 0;
    int32_t prerequisite = -1;
    uint32_t dependentCount = 0;
};

std::string RankName(uint32_t rank)
{
    const char server = rank < SERVER_SIZE ? 'A' : 'B';
    return std::string(1, server) + std::to_string(rank % SERVER_SIZE);
}

HcclResult BuildRelaySchedule(uint32_t myRank, ops_hccl::CcuKernelArgRelay2x8 &kernelArg)
{
    using ops_hccl::TransferDesc;
    using ops_hccl::TransferKind;

    std::vector<ScheduleAction> actions;
    actions.reserve(1920);
    std::array<std::array<std::array<int32_t, MAX_RANK_SIZE>, MAX_RANK_SIZE>,
        ops_hccl::RELAY_CHUNK_COUNT>
        crossAction{};
    for (auto &perChunk : crossAction) {
        for (auto &perOwner : perChunk) {
            perOwner.fill(-1);
        }
    }
    std::array<std::array<uint32_t, MAX_RANK_SIZE>, MAX_RANK_SIZE> forwardLoad{};

    auto addAction = [&actions](const std::string &id, ScheduleNetwork network, TransferKind kind, uint32_t src,
                         uint32_t dst, uint32_t owner, uint32_t chunk, int32_t prerequisite = -1) {
        actions.push_back(ScheduleAction{id, network, kind, src, dst, owner, chunk, prerequisite, 0});
        return static_cast<int32_t>(actions.size() - 1);
    };

    for (uint32_t chunk = 0; chunk < ops_hccl::RELAY_CHUNK_COUNT; ++chunk) {
        for (uint32_t owner = 0; owner < MAX_RANK_SIZE; ++owner) {
            const uint32_t localBase = owner < SERVER_SIZE ? 0 : SERVER_SIZE;
            const uint32_t remoteBase = owner < SERVER_SIZE ? SERVER_SIZE : 0;
            const uint32_t ownerIndex = owner % SERVER_SIZE;

            for (uint32_t localIndex = 0; localIndex < SERVER_SIZE; ++localIndex) {
                const uint32_t dst = localBase + localIndex;
                if (dst == owner) {
                    continue;
                }
                const std::string id = "mesh-native-c" + std::to_string(chunk) + "-" + RankName(owner) + "-" +
                                       RankName(dst);
                addAction(id, ScheduleNetwork::MESH, TransferKind::NATIVE, owner, dst, owner, chunk);
            }

            std::array<uint32_t, RELAY_COPIES> ingress{};
            const uint32_t ingressStart = (ownerIndex + chunk) % SERVER_SIZE;
            for (uint32_t index = 0; index < RELAY_COPIES; ++index) {
                const uint32_t remoteIndex = (ingressStart + index) % SERVER_SIZE;
                const uint32_t dst = remoteBase + remoteIndex;
                ingress[index] = dst;
                const std::string id =
                    "clos-c" + std::to_string(chunk) + "-" + RankName(owner) + "-" + RankName(dst);
                const int32_t actionIndex =
                    addAction(id, ScheduleNetwork::CLOS, TransferKind::CLOS, owner, dst, owner, chunk);
                crossAction[chunk][owner][dst] = actionIndex;
            }

            for (uint32_t targetIndex = 0; targetIndex < SERVER_SIZE; ++targetIndex) {
                const uint32_t dst = remoteBase + targetIndex;
                if (std::find(ingress.begin(), ingress.end(), dst) != ingress.end()) {
                    continue;
                }

                uint32_t bestRelay = ingress[0];
                auto bestKey = std::make_tuple(forwardLoad[bestRelay][dst],
                    (targetIndex + ownerIndex + chunk + bestRelay % SERVER_SIZE) % RELAY_COPIES);
                for (uint32_t candidate : ingress) {
                    const auto key = std::make_tuple(forwardLoad[candidate][dst],
                        (targetIndex + ownerIndex + chunk + candidate % SERVER_SIZE) % RELAY_COPIES);
                    if (key < bestKey) {
                        bestKey = key;
                        bestRelay = candidate;
                    }
                }
                ++forwardLoad[bestRelay][dst];
                const int32_t prerequisite = crossAction[chunk][owner][bestRelay];
                CHK_PRT_RET(prerequisite < 0, HCCL_ERROR("relay prerequisite was not created"), HCCL_E_INTERNAL);
                const std::string id = "mesh-forward-c" + std::to_string(chunk) + "-" + RankName(owner) + "-" +
                                       RankName(bestRelay) + "-" + RankName(dst);
                addAction(id, ScheduleNetwork::MESH, TransferKind::RELAY, bestRelay, dst, owner, chunk,
                    prerequisite);
            }
        }
    }

    for (const ScheduleAction &action : actions) {
        if (action.prerequisite >= 0) {
            ++actions[static_cast<std::size_t>(action.prerequisite)].dependentCount;
        }
    }

    std::vector<uint8_t> completed(actions.size(), 0);
    std::size_t remaining = actions.size();
    std::array<std::vector<uint32_t>, ops_hccl::RELAY_SLOT_COUNT> slotActions;
    uint32_t slot = 0;
    while (remaining != 0) {
        CHK_PRT_RET(slot >= ops_hccl::RELAY_SLOT_COUNT,
            HCCL_ERROR("relay schedule exceeds %u slots", ops_hccl::RELAY_SLOT_COUNT), HCCL_E_INTERNAL);
        std::vector<uint32_t> ready;
        ready.reserve(remaining);
        for (std::size_t index = 0; index < actions.size(); ++index) {
            if (completed[index] != 0) {
                continue;
            }
            const int32_t prerequisite = actions[index].prerequisite;
            if (prerequisite < 0 || completed[static_cast<std::size_t>(prerequisite)] != 0) {
                ready.push_back(static_cast<uint32_t>(index));
            }
        }
        std::sort(ready.begin(), ready.end(), [&actions](uint32_t lhsIndex, uint32_t rhsIndex) {
            const ScheduleAction &lhs = actions[lhsIndex];
            const ScheduleAction &rhs = actions[rhsIndex];
            return std::make_tuple(lhs.network != ScheduleNetwork::CLOS,
                       -static_cast<int32_t>(lhs.dependentCount), lhs.id) <
                   std::make_tuple(rhs.network != ScheduleNetwork::CLOS,
                       -static_cast<int32_t>(rhs.dependentCount), rhs.id);
        });

        std::array<std::array<uint32_t, MAX_RANK_SIZE>, 2> sendCount{};
        std::array<std::array<uint32_t, MAX_RANK_SIZE>, 2> receiveCount{};
        std::array<std::array<std::array<uint8_t, MAX_RANK_SIZE>, MAX_RANK_SIZE>, 2> usedLink{};
        for (uint32_t actionIndex : ready) {
            const ScheduleAction &action = actions[actionIndex];
            const uint32_t networkIndex = static_cast<uint32_t>(action.network);
            const uint32_t capacity = action.network == ScheduleNetwork::CLOS ?
                                          ops_hccl::MAX_CLOS_OPS_PER_SLOT :
                                          ops_hccl::MAX_MESH_OPS_PER_SLOT;
            if (sendCount[networkIndex][action.src] >= capacity ||
                receiveCount[networkIndex][action.dst] >= capacity ||
                usedLink[networkIndex][action.src][action.dst] != 0) {
                continue;
            }
            ++sendCount[networkIndex][action.src];
            ++receiveCount[networkIndex][action.dst];
            usedLink[networkIndex][action.src][action.dst] = 1;
            slotActions[slot].push_back(actionIndex);
        }
        CHK_PRT_RET(slotActions[slot].empty(), HCCL_ERROR("relay scheduler made no progress"), HCCL_E_INTERNAL);
        for (uint32_t actionIndex : slotActions[slot]) {
            completed[actionIndex] = 1;
            --remaining;
        }
        ++slot;
    }

    std::array<std::array<uint8_t, MAX_RANK_SIZE>, ops_hccl::RELAY_CHUNK_COUNT> relayWaitInserted{};
    std::array<uint8_t, ops_hccl::RELAY_CHUNK_COUNT> localCopyInserted{};
    for (uint32_t slotIndex = 0; slotIndex < ops_hccl::RELAY_SLOT_COUNT; ++slotIndex) {
        ops_hccl::RelaySlotSchedule &localSlot = kernelArg.slots[slotIndex];
        for (uint32_t actionIndex : slotActions[slotIndex]) {
            const ScheduleAction &action = actions[actionIndex];
            if (action.src != myRank) {
                continue;
            }
            TransferDesc desc;
            desc.kind = action.kind;
            desc.chunkId = static_cast<uint8_t>(action.chunk);
            desc.ownerRank = static_cast<uint8_t>(action.owner);
            desc.peerRank = static_cast<uint8_t>(action.dst);
            desc.signalReady = static_cast<uint8_t>(
                action.network == ScheduleNetwork::CLOS && action.dependentCount != 0);
            if (action.kind == TransferKind::RELAY && relayWaitInserted[action.chunk][action.owner] == 0) {
                desc.waitReady = 1;
                relayWaitInserted[action.chunk][action.owner] = 1;
            }

            if (action.network == ScheduleNetwork::CLOS) {
                CHK_PRT_RET(localSlot.closCount >= ops_hccl::MAX_CLOS_OPS_PER_SLOT,
                    HCCL_ERROR("local Clos schedule overflow"), HCCL_E_INTERNAL);
                localSlot.closOps[localSlot.closCount++] = desc;
            } else {
                CHK_PRT_RET(localSlot.meshCount >= ops_hccl::MAX_MESH_OPS_PER_SLOT,
                    HCCL_ERROR("local Mesh schedule overflow"), HCCL_E_INTERNAL);
                localSlot.meshOps[localSlot.meshCount++] = desc;
                if (action.kind == TransferKind::NATIVE && localCopyInserted[action.chunk] == 0) {
                    localSlot.localCopyMask |= static_cast<uint8_t>(1U << action.chunk);
                    localCopyInserted[action.chunk] = 1;
                }
            }
        }
    }
    for (uint32_t chunk = 0; chunk < ops_hccl::RELAY_CHUNK_COUNT; ++chunk) {
        CHK_PRT_RET(localCopyInserted[chunk] == 0,
            HCCL_ERROR("chunk %u has no local-copy slot", chunk), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

struct ChannelGroup {
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peerRanks;
};

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, std::map<uint32_t, ChannelGroup> &groups,
    bool &relaySupported)
{
    relaySupported = param.rankSize == MAX_RANK_SIZE;
    uint32_t *layerList = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerCount));
    CHK_PRT_RET(layerList == nullptr || layerCount == 0,
        HCCL_ERROR("rank graph contains no network layer"), HCCL_E_NOT_FOUND);

    const std::vector<uint32_t> layers(layerList, layerList + layerCount);
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        bool found = false;
        const bool expectedLocal = (param.myRank < SERVER_SIZE) == (remoteRank < SERVER_SIZE);
        const CommProtocol preferred = expectedLocal ? CommProtocol::COMM_PROTOCOL_UBC_TP
                                                     : CommProtocol::COMM_PROTOCOL_UBC_CTP;
        const CommProtocol protocols[] = {
            preferred,
            preferred == CommProtocol::COMM_PROTOCOL_UBC_TP ? CommProtocol::COMM_PROTOCOL_UBC_CTP
                                                              : CommProtocol::COMM_PROTOCOL_UBC_TP,
        };
        bool foundPreferred = false;
        for (CommProtocol protocol : protocols) {
            for (uint32_t netLayer : layers) {
                CommLink *linkList = nullptr;
                uint32_t listSize = 0;
                if (HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &linkList, &listSize) !=
                    HCCL_SUCCESS) {
                    continue;
                }
                for (uint32_t index = 0; index < listSize; ++index) {
                    const CommLink &link = linkList[index];
                    if (link.linkAttr.linkProtocol != protocol) {
                        continue;
                    }
                    desc.remoteRank = remoteRank;
                    desc.notifyNum = CHANNEL_NOTIFY_NUM;
                    desc.channelProtocol = link.linkAttr.linkProtocol;
                    desc.localEndpoint = link.srcEndpointDesc;
                    desc.remoteEndpoint = link.dstEndpointDesc;
                    found = true;
                    foundPreferred = protocol == preferred;
                    break;
                }
                if (found) {
                    break;
                }
            }
            if (found) {
                break;
            }
        }
        CHK_PRT_RET(!found,
            HCCL_ERROR("CCU-compatible link not found between rank %u and rank %u", param.myRank, remoteRank),
            HCCL_E_NOT_FOUND);
        if (!foundPreferred) {
            relaySupported = false;
        }
        EndpointAttrDieId dieId{};
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID,
            sizeof(dieId), static_cast<void *>(&dieId)));
        ChannelHandle channel{};
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        groups[dieId].channels.push_back(channel);
        groups[dieId].peerRanks.push_back(remoteRank);
    }
    relaySupported = relaySupported && groups.size() == 1;
    return HCCL_SUCCESS;
}

template <typename T> void FillKernelChannels(T &kernelArg, const ChannelGroup &group)
{
    kernelArg.channelCount = static_cast<uint32_t>(group.channels.size());
    for (std::size_t index = 0; index < group.channels.size(); ++index) {
        kernelArg.channels[index] = group.channels[index];
        kernelArg.peerRanks[index] = group.peerRanks[index];
    }
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelGroup> &groups,
    bool relaySupported, AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("expected one CCU instance, got %u", insNum), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    resCtx.directKernels.resize(groups.size());
    resCtx.relayEnabled = relaySupported ? 1U : 0U;
    constexpr uint32_t dieId = 0;
    for (std::size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const ChannelGroup &group = groups[groupIndex];
        CHK_PRT_RET(group.channels.empty() || group.channels.size() != group.peerRanks.size(),
            HCCL_ERROR("invalid channel group %zu", groupIndex), HCCL_E_INTERNAL);
        auto directArg = std::make_shared<ops_hccl::CcuKernelArgDirect>();
        directArg->rankSize = param.rankSize;
        directArg->rankId = param.myRank;
        directArg->ownsLocalCopy = groupIndex == 0 ? 1U : 0U;
        directArg->useSmallMessageFastPath = UseSmallMessageFastPath(param) ? 1U : 0U;
        FillKernelChannels(*directArg, group);
        const void *directArgs[] = {directArg.get()};
        char directName[] = "DirectAllGatherKernel";
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, directName,
            reinterpret_cast<void *>(ops_hccl::DirectAllGatherKernel), directArgs, 1,
            &resCtx.directKernels[groupIndex]));
    }

    if (resCtx.relayEnabled != 0) {
        const ChannelGroup &group = groups[0];
        auto relayArg = std::make_shared<ops_hccl::CcuKernelArgRelay2x8>();
        relayArg->rankSize = param.rankSize;
        relayArg->rankId = param.myRank;
        FillKernelChannels(*relayArg, group);
        CHK_RET(BuildRelaySchedule(param.myRank, *relayArg));
        const void *relayArgs[] = {relayArg.get()};
        char relayName[] = "RelayAllGather2x8Kernel";
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, relayName,
            reinterpret_cast<void *>(ops_hccl::RelayAllGather2x8Kernel), relayArgs, 1,
            &resCtx.relayKernel));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult CreateResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    resCtx.threads = {param.cpuThread};

    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    std::map<uint32_t, ChannelGroup> groupedChannels;
    bool relaySupported = false;
    CHK_RET(AcquireChannels(comm, param, groupedChannels, relaySupported));
    CHK_PRT_RET(groupedChannels.empty(), HCCL_ERROR("no channel group was created"), HCCL_E_INTERNAL);
    std::vector<ChannelGroup> groups;
    groups.reserve(groupedChannels.size());
    for (auto &entry : groupedChannels) {
        groups.push_back(std::move(entry.second));
    }
    if (groups.size() > 1) {
        const uint32_t slaveCount = static_cast<uint32_t>(groups.size() - 1);
        const std::size_t oldSize = resCtx.threads.size();
        resCtx.threads.resize(oldSize + slaveCount);
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, slaveCount, THREAD_NOTIFY_NUM,
            resCtx.threads.data() + oldSize));
    }
    return RegisterKernels(comm, param, groups, relaySupported, resCtx);
}

} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    constexpr char smallOpTag[] = "hccl_custom_allgather_md_small_v3";
    constexpr char regularOpTag[] = "hccl_custom_allgather_md_v2";
    static_assert(sizeof(smallOpTag) <= sizeof(param.tag));
    static_assert(sizeof(regularOpTag) <= sizeof(param.tag));
    if (UseSmallMessageFastPath(param)) {
        std::copy_n(smallOpTag, sizeof(smallOpTag), param.tag);
    } else {
        std::copy_n(regularOpTag, sizeof(regularOpTag), param.tag);
    }

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("unsupported rank size: %u", param.rankSize), HCCL_E_NOT_SUPPORT);

    constexpr CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream, THREAD_NOTIFY_NUM, &param.cpuThread));
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx resCtx;
        CHK_RET(CreateResources(comm, param, resCtx));
        const std::vector<char> sequence = resCtx.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, sequence.data(), sequence.size(), 0));
    }

    return ops_hccl::ExecOp(param);
}
