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
#include <cstring>
#include <limits>
#include <map>
#include <utility>
#include <vector>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include "hccl.h"
#include "custom.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
using Groups = std::map<uint32_t, std::vector<std::pair<ChannelHandle, uint32_t>>>;

HcclResult RegisterUserBuffer(HcclComm comm, const OpParam &param,
                             std::vector<HcclMemHandle> &handles, AlgResourceCtx &ctx)
{
    const uint64_t rankBytes = param.count * sizeof(float);
    const uint64_t totalBytes = rankBytes * param.rankSize;
    // 十二 rank 的两种拓扑还需要大消息输入预交换；其他大消息保持原内存集合。
    if (totalBytes > scatter_plan::SMALL_TOTAL_BYTES &&
        !(SCATTER_CANDIDATE == 0 && param.rankSize == 12)) {
        return HCCL_SUCCESS;
    }
    const bool isRoot = param.myRank == param.root;
    CommMem mem{COMM_MEM_TYPE_DEVICE, isRoot ? param.inputPtr : param.outputPtr,
                isRoot ? totalBytes : rankBytes};
    char memTag[HCCL_RES_TAG_MAX_LEN + 1]{};
    const int length = std::snprintf(memTag, sizeof(memTag), "%s_%s", param.tag, isRoot ? "input" : "output");
    CHK_PRT_RET(length < 0 || static_cast<size_t>(length) >= sizeof(memTag),
                HCCL_ERROR("Scatter 用户内存标签过长"), HCCL_E_INTERNAL);
    HcclMemHandle handle = nullptr;
    // root 只注册完整输入，避免 recvBuf 别名到输入分片时重复注册重叠区域。
    CHK_RET(HcclCommMemReg(comm, memTag, &mem, &handle));
    CHK_PTR_NULL(handle);
    handles.push_back(handle);
    ctx.registeredRoot = param.root;
    if (isRoot) {
        ctx.registeredInput = CommBuffer{param.inputPtr, totalBytes};
    }
    return HCCL_SUCCESS;
}

HcclResult GetRegisteredInput(HcclComm comm, ChannelHandle channel,
                              const OpParam &param, AlgResourceCtx &ctx)
{
    uint32_t count = 0;
    CommMem *mems = nullptr;
    char **tags = nullptr;
    CHK_RET(HcclChannelGetRemoteMems(comm, channel, &count, &mems, &tags));
    CHK_PTR_NULL(mems);
    CHK_PTR_NULL(tags);
    char inputTag[HCCL_RES_TAG_MAX_LEN + 1]{};
    std::snprintf(inputTag, sizeof(inputTag), "%s_input", param.tag);
    for (uint32_t i = 0; i < count; ++i) {
        if (tags[i] != nullptr && std::strcmp(tags[i], inputTag) == 0) {
            CHK_PRT_RET(mems[i].type != COMM_MEM_TYPE_DEVICE || mems[i].addr == nullptr ||
                        mems[i].size < param.count * sizeof(float) * param.rankSize,
                        HCCL_ERROR("Scatter 建链交换的 root 输入无效"), HCCL_E_PARA);
            ctx.registeredInput = CommBuffer{mems[i].addr, mems[i].size};
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("Scatter 建链未返回 root 用户输入");
    return HCCL_E_NOT_FOUND;
}

// 为每个对端 rank 申请一条 channel（跳过自己，按 remoteRank 升序处理），并按 channel 所在 die 分组
HcclResult GetChannelForCcu(HcclComm comm, const OpParam &param, CommEngine ccuEngine,
                            std::vector<HcclMemHandle> &memHandles, Groups &groups, AlgResourceCtx &ctx,
                            std::array<RoleLinkDescriptor, scatter_plan::MAX_RANKS> &roleLinks)
{
    const uint64_t rankBytes = param.count * sizeof(float);
    const bool exchangeRoleOutput = SCATTER_CANDIDATE == 0;
    const bool exchangeInput = exchangeRoleOutput &&
        (rankBytes * param.rankSize <= scatter_plan::SMALL_TOTAL_BYTES || param.rankSize == 12) &&
        (param.rankSize == 4 || param.rankSize == 12 || param.rankSize == 16);
    LinkExchangeDescriptor localExchange;
    auto &local = localExchange.output;
    if (exchangeRoleOutput) {
        local.rank = param.myRank;
        local.address = reinterpret_cast<uint64_t>(param.outputPtr);
        local.bytes = rankBytes;
        CHK_PRT_RET(HcommCcuGetMemToken(local.address, local.bytes, &local.token) != CCU_SUCCESS,
                    HCCL_ERROR("建链输出 Token 查询失败"), HCCL_E_INTERNAL);
        local.scratchAddress = reinterpret_cast<uint64_t>(ctx.scratch.addr);
        local.scratchBytes = ctx.scratch.size;
        CHK_PRT_RET(local.scratchAddress == 0 || local.scratchBytes == 0,
                    HCCL_ERROR("建链暂存区为空"), HCCL_E_PARA);
        CHK_PRT_RET(HcommCcuGetMemToken(local.scratchAddress, local.scratchBytes, &local.scratchToken) != CCU_SUCCESS,
                    HCCL_ERROR("建链暂存区 Token 查询失败"), HCCL_E_INTERNAL);
        roleLinks[param.myRank] = local;
        if (exchangeInput && param.myRank == param.root) {
            auto &input = localExchange.input;
            input.address = reinterpret_cast<uint64_t>(param.inputPtr);
            input.bytes = rankBytes * param.rankSize;
            if (input.bytes != 0) {
                CHK_PRT_RET(HcommCcuGetMemToken(input.address, input.bytes, &input.token) != CCU_SUCCESS,
                    HCCL_ERROR("建链输入 Token 查询失败"), HCCL_E_INTERNAL);
            }
            ctx.linkedInput = input;
        }
    }
    // RankGraph 分层（如 layer0=Server内Mesh、layer1=跨Server Clos），不同层覆盖不同的 rank 对
    uint32_t *netLayersRaw = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayersRaw, &layerNum));
    CHK_PRT_RET(netLayersRaw == nullptr || layerNum == 0,
        HCCL_ERROR("[GetChannelForCcu] no netLayer found"), HCCL_E_NOT_FOUND);
    // 库文档要求立即复制：后续调用 HcclRankGraphGetLinks 可能使返回的 netLayers 内存失效
    std::vector<uint32_t> netLayers(netLayersRaw, netLayersRaw + layerNum);

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
        if (remoteRank == param.myRank) {
            continue;
        }

        // 遍历所有 netLayer，找到第一个存在本 rank 到 remoteRank 链路的层
        uint32_t linkNum = 0;
        CommLink *links = nullptr;
        for (uint32_t layer = 0; layer < layerNum; layer++) {
            if (HcclRankGraphGetLinks(comm, netLayers[layer], param.myRank, remoteRank,
                                      &links, &linkNum) == HCCL_SUCCESS &&
                links != nullptr && linkNum > 0) {
                break;
            }
            links = nullptr;
            linkNum = 0;
        }
        CHK_PRT_RET(links == nullptr || linkNum == 0,
            HCCL_ERROR("[GetChannelForCcu] no link found between rank %u and rank %u on any netLayer",
                       param.myRank, remoteRank),
            HCCL_E_NOT_FOUND);

        // 优先选择 UBC_CTP 协议链路；跨 Server 的 Clos 链路协议可能不同，兜底取第一条可用 link
        const CommLink *chosenLink = &links[0];
        for (uint32_t idx = 0; idx < linkNum; idx++) {
            if (links[idx].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                chosenLink = &links[idx];
                break;
            }
        }

        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = remoteRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.memHandles = memHandles.empty() ? nullptr : memHandles.data();
        desc.memHandleNum = memHandles.size();
        desc.channelProtocol = chosenLink->linkAttr.linkProtocol;
        desc.localEndpoint.protocol = chosenLink->srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = chosenLink->srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = chosenLink->srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = chosenLink->dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = chosenLink->dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = chosenLink->dstEndpointDesc.loc;

        // 查询该 channel 本端端点所在 die（channel acquire 之前基于 desc 即可查询）
        EndpointAttrDieId dieId{};
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint,
            ENDPOINT_ATTR_DIE_ID, sizeof(EndpointAttrDieId), &dieId));

        ChannelHandle channel{0};
        // 每次 Acquire 后库会清空本端待交换信息，所有 rank 都逐 channel 重新注入。
        if (exchangeRoleOutput) {
            CHK_RET(HcclCommAddExchangeInfo(comm, &localExchange, sizeof(localExchange)));
        }
        CHK_RET(HcclChannelAcquire(comm, ccuEngine, &desc, 1, &channel));
        if (exchangeRoleOutput) {
            LinkExchangeDescriptor remoteExchange;
            auto &remote = remoteExchange.output;
            uint32_t actualLength = 0;
            CHK_RET(HcclCommGetExchangeInfo(comm, remoteRank, sizeof(remoteExchange), &remoteExchange, &actualLength));
            CHK_PRT_RET(actualLength != sizeof(remoteExchange) || remote.magic != ROLE_LINK_MAGIC ||
                        remote.rank != remoteRank || remote.address == 0 || remote.bytes != rankBytes ||
                        remote.scratchAddress == 0 || remote.scratchBytes != local.scratchBytes,
                        HCCL_ERROR("建链接收描述无效，对端=%u，实际长度=%u", remoteRank, actualLength),
                        HCCL_E_PARA);
            roleLinks[remoteRank] = remote;
            if (exchangeInput && remoteRank == param.root) {
                CHK_PRT_RET(remoteExchange.input.address == 0 ||
                    remoteExchange.input.bytes != rankBytes * param.rankSize,
                    HCCL_ERROR("建链 root 输入描述无效"), HCCL_E_PARA);
                ctx.linkedInput = remoteExchange.input;
            }
        }
        groups[dieId].emplace_back(channel, remoteRank);
        if (remoteRank == ctx.registeredRoot) {
            CHK_RET(GetRegisteredInput(comm, channel, param, ctx));
        }
    }

    return HCCL_SUCCESS;
}

// 基于 RankGraph 层信息识别拓扑：layer0（机内 Mesh）的实例大小签名区分四种通信域，
// 并校验本 rank 的实例成员与 SameServer 的分块公式一致；任何不符都回退 Generic（直发，功能总是正确）。
scatter_plan::Topology DetectTopology(HcclComm comm, const OpParam &param, const Groups &groups)
{
    using scatter_plan::Topology;
    uint32_t *layersRaw = nullptr;
    uint32_t layerNum = 0;
    if (HcclRankGraphGetLayers(comm, &layersRaw, &layerNum) != HCCL_SUCCESS ||
        layersRaw == nullptr || layerNum == 0) {
        return Topology::Generic;
    }
    const uint32_t layer0 = layersRaw[0];
    uint32_t *sizeListRaw = nullptr;
    uint32_t listSize = 0;
    if (HcclRankGraphGetInstSizeListByLayer(comm, layer0, &sizeListRaw, &listSize) != HCCL_SUCCESS ||
        sizeListRaw == nullptr || listSize == 0) {
        return Topology::Generic;
    }
    // 返回内存由库管理，后续调用可能使其失效，立即复制
    std::vector<uint32_t> instSizes(sizeListRaw, sizeListRaw + listSize);
    // 实例列表来自哈希容器，8+4 可能返回 {4,8}；大小签名不能依赖遍历顺序。
    std::sort(instSizes.begin(), instSizes.end());
    uint32_t *membersRaw = nullptr;
    uint32_t memberNum = 0;
    if (HcclRankGraphGetRanksByLayer(comm, layer0, &membersRaw, &memberNum) != HCCL_SUCCESS ||
        membersRaw == nullptr || memberNum == 0) {
        return Topology::Generic;
    }
    std::vector<uint32_t> members(membersRaw, membersRaw + memberNum);
    std::sort(members.begin(), members.end());

    Topology candidate = Topology::Generic;
    if (param.rankSize == 16 && instSizes == std::vector<uint32_t>({8, 8})) {
        candidate = Topology::TwoByEight;
    } else if (param.rankSize == 12 && instSizes == std::vector<uint32_t>({4, 8})) {
        candidate = Topology::EightPlusFour;
    } else if (param.rankSize == 12 && instSizes == std::vector<uint32_t>({3, 3, 3, 3})) {
        candidate = Topology::FourByThree;
    } else if (param.rankSize == 4 &&
               (instSizes == std::vector<uint32_t>({1, 1, 1, 1}) || groups.size() == 1)) {
        candidate = Topology::FourByOne;
    }
    if (candidate == Topology::Generic) {
        return Topology::Generic;
    }

    // 本 rank 的 layer0 实例成员必须恰好是 SameServer 分块公式认定的同机 rank 集合
    std::vector<uint32_t> expected;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (scatter_plan::SameServer(candidate, param.myRank, rank)) {
            expected.push_back(rank);
        }
    }
    if (members != expected) {
        return Topology::Generic;
    }

    if (candidate == Topology::FourByOne) {
        return groups.size() == 1 ? Topology::FourByOne : Topology::Generic;
    }
    if (groups.size() != 2) {
        return Topology::Generic;
    }
    bool mesh = false;
    bool clos = false;
    for (const auto &group : groups) {
        const bool same = scatter_plan::SameServer(candidate, param.myRank, group.second.front().second);
        for (const auto &channel : group.second) {
            if (scatter_plan::SameServer(candidate, param.myRank, channel.second) != same) {
                return Topology::Generic;
            }
        }
        mesh = mesh || same;
        clos = clos || !same;
    }
    return mesh && clos ? candidate : Topology::Generic;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param, const Groups &groups, AlgResourceCtx &ctx,
                          const std::array<RoleLinkDescriptor, scatter_plan::MAX_RANKS> &roleLinks)
{
    CcuInsHandle instance{};
    uint32_t count = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &count));
    CHK_PRT_RET(count != 1, HCCL_ERROR("CCU 实例数不是 1"), HCCL_E_INTERNAL);
    CHK_PRT_RET(groups.empty() || groups.size() > 2,
                HCCL_ERROR("不支持的 IO Die 分组数量"), HCCL_E_NOT_SUPPORT);
    ctx.topology = DetectTopology(comm, param, groups);
    if (ctx.topology != scatter_plan::Topology::Generic) ctx.linkedOutput = roleLinks[param.myRank];
    const bool relayTopology = ctx.topology == scatter_plan::Topology::TwoByEight ||
        ctx.topology == scatter_plan::Topology::EightPlusFour ||
        ctx.topology == scatter_plan::Topology::FourByThree;
    ctx.kernelArgs.reserve(groups.size());
    uint32_t selfGroup = 0;
    uint32_t groupIndex = 0;
    size_t fewest = groups.begin()->second.size();
    for (const auto &group : groups) {
        if (group.second.size() < fewest) {
            fewest = group.second.size();
            selfGroup = groupIndex;
        }
        ++groupIndex;
    }
    ctx.selfGroup = selfGroup;
    groupIndex = 0;
    CHK_PRT_RET(HcommCcuKernelRegisterStart(instance) != CCU_SUCCESS,
                HCCL_ERROR("CCU 注册开始失败"), HCCL_E_INTERNAL);
    for (const auto &group : groups) {
        ScatterKernelArg arg;
        arg.rank = param.myRank;
        arg.registeredRoot = ctx.registeredRoot;
        // 8+4 大机两个组都具备复制能力，实际启用的组由每轮 host 参数决定。
        arg.selfCopy = groupIndex == selfGroup ||
            (ctx.topology == scatter_plan::Topology::EightPlusFour && param.myRank < 8);
        arg.roleWriteOnly = scatter_plan::UseRoleWritePath(ctx.topology, SCATTER_CANDIDATE);
        const bool mesh = scatter_plan::SameServer(ctx.topology, param.myRank, group.second.front().second);
        arg.meshReadWriteOnly = scatter_plan::UseMeshReadWriteOnly(ctx.topology, param.myRank,
            mesh, SCATTER_CANDIDATE);
        arg.directedWrite = ctx.topology == scatter_plan::Topology::FourByOne ||
            (ctx.topology == scatter_plan::Topology::EightPlusFour && !mesh);
        if (ctx.topology == scatter_plan::Topology::TwoByEight) {
            arg.optimization = WriteOptimization::FusedMetadata;
        } else if (ctx.topology == scatter_plan::Topology::EightPlusFour) {
            arg.optimization = WriteOptimization::ChannelReady;
        } else if (ctx.topology == scatter_plan::Topology::FourByThree) {
            arg.optimization = WriteOptimization::Compact;
        }
        // 默认一槽覆盖 2×8 的各阶段；仅下列 Mesh 组存在同 channel 多个目标分片。
        if (ctx.topology == scatter_plan::Topology::EightPlusFour && param.myRank >= 8 &&
            group.second.front().second >= 8) {
            arg.slots = 8;
        } else if (ctx.topology == scatter_plan::Topology::FourByThree &&
                   scatter_plan::SameServer(scatter_plan::Topology::FourByThree,
                                            param.myRank, group.second.front().second)) {
            // 4×3 root 的 Mesh channel 在阶段 0 承载 9 个跨机目标的中继分片。
            arg.slots = 9;
        }
        for (const auto &channel : group.second) {
            arg.channels[arg.channelCount] = channel.first;
            arg.ranks[arg.channelCount++] = channel.second;
        }
        if (ctx.linkedOutput.bytes != 0) {
            arg.roleLinkBytes = roleLinks[param.myRank].bytes;
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                arg.roleLinkTargets[i] = roleLinks[arg.ranks[i]];
            }
        }
        ctx.kernelArgs.push_back(arg);
        const void *arguments[] = {&ctx.kernelArgs.back()};
        CcuKernelHandle write{}, read{}, relay{};
        // die 参数当前为预留值，实际部署由该 Kernel 使用的 channel 端点决定。
        // 每个 die 的默认资源包只有两个 mission；直写与中继必须共用一个写 Kernel。
        // 不能只注册当前算法，否则同一通信域切换 root 或数据量后会复用到缺失的句柄。
        CcuResult status;
        if (relayTopology) {
            status = HcommCcuKernelRegister(instance, 0, "ScatterRelayKernel",
                reinterpret_cast<void *>(ops_hccl::ScatterRelayKernel), arguments, 1, &relay);
            write = relay;
        } else {
            status = HcommCcuKernelRegister(instance, 0, "ScatterWriteKernel",
                reinterpret_cast<void *>(ops_hccl::ScatterWriteKernel), arguments, 1, &write);
        }
        CHK_PRT_RET(status != CCU_SUCCESS,
            HCCL_ERROR("Scatter 写 Kernel 注册失败，组=%u，中继布局=%u，CCU返回码=%d",
                groupIndex, static_cast<uint32_t>(relayTopology), static_cast<int>(status)), HCCL_E_INTERNAL);
        status = HcommCcuKernelRegister(instance, 0, "ScatterReadKernel",
            reinterpret_cast<void *>(ops_hccl::ScatterReadKernel), arguments, 1, &read);
        CHK_PRT_RET(status != CCU_SUCCESS,
            HCCL_ERROR("Scatter 读 Kernel 注册失败，组=%u，CCU返回码=%d",
                groupIndex, static_cast<int>(status)), HCCL_E_INTERNAL);
        ctx.writeKernels.push_back(write);
        ctx.readKernels.push_back(read);
        ctx.relayKernels.push_back(relay);
        ++groupIndex;
    }
    CHK_PRT_RET(HcommCcuKernelRegisterEnd(instance) != CCU_SUCCESS,
                HCCL_ERROR("CCU 注册结束失败"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}
}

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                       uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    OpParam param;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || root >= param.rankSize,
                HCCL_ERROR("通信域大小或 root 非法"), HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
                HCCL_ERROR("预研工程当前仅支持 FP32"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize,
                HCCL_ERROR("输入总长度溢出"), HCCL_E_PARA);
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(recvBuf);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    // 固定上下文和角色协议使用独立标签，隔离旧版资源。
    std::snprintf(param.tag, sizeof(param.tag), "hccl_scatter_final_ccu_v25_%u", SCATTER_CANDIDATE);
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream, 2, &param.cpuThread));
    if (param.rankSize == 1) {
        if (sendBuf != recvBuf) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                param.cpuThread, recvBuf, sendBuf, recvCount * sizeof(float))));
        }
        return HCCL_SUCCESS;
    }
    if (HcclEngineCtxGet(comm, param.tag, engine, &param.resCtx, &param.ctxSize) != HCCL_SUCCESS) {
        AlgResourceCtx ctx;
        CHK_RET(HcclGetHcclBuffer(comm, &ctx.scratch.addr, &ctx.scratch.size));
        std::vector<HcclMemHandle> memHandles;
        CHK_RET(RegisterUserBuffer(comm, param, memHandles, ctx));
        Groups groups;
        std::array<RoleLinkDescriptor, scatter_plan::MAX_RANKS> roleLinks{};
        CHK_RET(GetChannelForCcu(comm, param, engine, memHandles, groups, ctx, roleLinks));
        CHK_RET(RegisterKernels(comm, param, groups, ctx, roleLinks));
        ctx.threads.resize(groups.size());
        ctx.threads[0] = param.cpuThread;
        if (ctx.threads.size() > 1) {
            CHK_RET(HcclThreadAcquire(comm, engine, ctx.threads.size() - 1, 2, &ctx.threads[1]));
        }
        std::vector<char> serialized;
        if (scatter_plan::UseRoleWritePath(ctx.topology, SCATTER_CANDIDATE)) {
            RoleResourceCtx fixed;
            fixed.mainThread = ctx.threads[0];
            fixed.sender = ctx.writeKernels[0];
            fixed.receiver = ctx.readKernels[0];
            std::copy_n(ctx.kernelArgs[0].ranks, 3, fixed.peers);
            fixed.linkedOutput = roleLinks[param.myRank];
            fixed.linkedInput = ctx.linkedInput;
            fixed.registeredRoot = ctx.registeredRoot;
            const auto *begin = reinterpret_cast<const char *>(&fixed);
            serialized.assign(begin, begin + sizeof(fixed));
        } else {
            serialized = ctx.Serialize();
            if (SCATTER_CANDIDATE == 0 && ctx.topology != scatter_plan::Topology::Generic) {
                CHK_PRT_RET(ctx.kernelArgs.size() != 2 || ctx.threads.size() != 2 || ctx.readKernels.size() != 2,
                    HCCL_ERROR("固定上下文要求完整的两个 die 分组"), HCCL_E_INTERNAL);
                const auto fixed = MakeFastReadContext(ctx, serialized.size());
                const auto *begin = reinterpret_cast<const char *>(&fixed);
                serialized.insert(serialized.begin(), begin, begin + sizeof(fixed));
            }
        }
        param.ctxSize = serialized.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, serialized.data(), param.ctxSize, 0));
    }
    return ops_hccl::ExecOp(param, comm);
}
