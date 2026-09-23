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
#include <map>
#include <memory>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr uint32_t DEFAULT_CHANNEL_NOTIFY_NUM = 3; // PRE_SYNC(0) + POST_SYNC(1) + RELAY(2)
constexpr uint32_t AUX_THREAD_NOTIFY_NUM = 1;
constexpr uint64_t PULL_THRESHOLD = 1ULL * 1024ULL * 1024ULL; // 小消息（≤1MB）走拉模型
constexpr uint64_t CLOS_TO_MESH_BW_RATIO = 4;

// 从 selfRank 到 peerRank 的所有网络层中选一条 UBC_CTP 链路，返回其本地 Endpoint 的 IO Die ID 与所在 layer。
// 机内 Mesh（低 layer）优先于机间 Clos；同一层内按带宽系数择优（相同时取小 Die）。
// 与 hccl 官方库 SelectOneLinkPerPeer 的选路策略一致，避免选到无效/低质链路导致 channel 句柄异常。
HcclResult FindCtpLink(HcclComm comm, uint32_t selfRank, uint32_t peerRank, CommLink *selectedLink, uint32_t *dieId,
    uint32_t *layer)
{
    CHK_PTR_NULL(selectedLink);
    CHK_PTR_NULL(dieId);
    CHK_PTR_NULL(layer);

    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    CHK_PRT_RET(layerNum == 0 || layers == nullptr, HCCL_ERROR("No network layer found"), HCCL_E_NOT_FOUND);

    for (uint32_t layerIdx = 0; layerIdx < layerNum; layerIdx++) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layers[layerIdx], selfRank, peerRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
            continue;
        }

        const CommLink *best = nullptr;
        uint32_t bestBw = 0;
        uint32_t bestDie = 0;
        for (uint32_t linkIdx = 0; linkIdx < linkNum; linkIdx++) {
            if (links[linkIdx].linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            EndpointAttrDieId localDie = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(
                comm, selfRank, &links[linkIdx].srcEndpointDesc, ENDPOINT_ATTR_DIE_ID, sizeof(localDie), &localDie));
            EndpointAttrBwCoeff bwCoeff = 0;
            HcclResult bwRet = HcclRankGraphGetEndpointInfo(
                comm, selfRank, &links[linkIdx].srcEndpointDesc, ENDPOINT_ATTR_BW_COEFF, sizeof(bwCoeff), &bwCoeff);
            if (bwRet != HCCL_SUCCESS) {
                bwCoeff = 0;
            }
            if (best == nullptr || bwCoeff > bestBw || (bwCoeff == bestBw && localDie < bestDie)) {
                best = &links[linkIdx];
                bestBw = bwCoeff;
                bestDie = static_cast<uint32_t>(localDie);
            }
        }
        if (best != nullptr) {
            *selectedLink = *best;
            *dieId = bestDie;
            *layer = layers[layerIdx];
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("No UBC_CTP link from rank %u to rank %u", selfRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

// 构建直发星形资源：root 向每个对端申请 1 条通道，非 root 只向 root 申请 1 条通道；
// 通道按本地 IO Die 分组，每组注册一个 CCU Kernel。
HcclResult BuildResources(HcclComm comm, CommEngine engine, const OpParam &param, AlgResourceCtx &resCtx)
{
    const uint64_t chunkBytes = param.count * sizeof(float);
    // 4×1（单机单 IO Die，无 mesh/Clos）小消息改用直发星形(PUSH)，避免 PULL 的「发布→读」往返；其余拓扑仍走 PULL。
    const bool usePull = (chunkBytes <= PULL_THRESHOLD) && (param.rankSize != 4);
    const bool useLarge = chunkBytes > MAX_DATA_SIZE;
    // 4×1 小消息（≤1MB）走专用 kernel：root 自留拷贝先提交 + 逐 channel 交错写；非 root 只加载 2 个地址参数尽早发布。
    const bool is4x1Small = (param.rankSize == 4) && (chunkBytes <= PULL_THRESHOLD);
    // 探测本机（layer0）的卡数：中继分组大小 + 区分 4×3（每服务器 3 卡）与 8+4/2×8（每服务器 8 卡）。
    uint32_t localRankNum = 0;
    {
        uint32_t *layer0Ranks = nullptr;
        uint32_t layer0Num = 0;
        HcclResult ret = HcclRankGraphGetRanksByLayer(comm, 0, &layer0Ranks, &layer0Num);
        if (ret == HCCL_SUCCESS && layer0Ranks != nullptr) {
            localRankNum = layer0Num;
        }
    }
    const bool is2x8 = param.rankSize == 16;
    const bool is4x3 = param.rankSize == 12 && localRankNum == 3;
    // 中继：2×8 / 4×3（>1MB），root 把部分跨机块交给 root 同 server 的 relay 转发（支持任意 root）。单片。
    const bool useRelay = !usePull && (is2x8 || is4x3);
    uint32_t relayRole = 0; // 0=非中继, 1=relay(本地中转), 2=target(跨机目标)
    uint32_t relayPeerRank = INVALID_VALUE_RANKID;
    std::vector<uint32_t> relays;   // root 同 server 的非 root rank（按 rank 升序）
    std::vector<uint32_t> targets;  // 跨 server 的 rank（按 rank 升序）
    if (useRelay) {
        // 组大小 = 每 server 卡数（2×8=8，4×3=3）。rank 按组连续编号：组号 = rank/组大小。
        const uint32_t groupSize = localRankNum;
        const uint32_t rootGroup = param.root / groupSize;
        for (uint32_t r = rootGroup * groupSize; r < (rootGroup + 1) * groupSize && r < param.rankSize; r++) {
            if (r != param.root) {
                relays.push_back(r);
            }
        }
        for (uint32_t r = 0; r < param.rankSize; r++) {
            if (r / groupSize != rootGroup) {
                targets.push_back(r);
            }
        }
        if (param.myRank != param.root) {
            const uint32_t myGroup = param.myRank / groupSize;
            if (myGroup == rootGroup) {
                // relay：转发给 targets[自己序号]
                relayRole = 1;
                for (uint32_t i = 0; i < relays.size(); i++) {
                    if (relays[i] == param.myRank) {
                        relayPeerRank = (i < targets.size()) ? targets[i] : INVALID_VALUE_RANKID;
                        break;
                    }
                }
            } else {
                // target：序号 < relays.size() 的被 relay 转发，否则整块直收
                for (uint32_t i = 0; i < targets.size(); i++) {
                    if (targets[i] == param.myRank) {
                        if (i < relays.size()) {
                            relayRole = 2;
                            relayPeerRank = relays[i];
                        }
                        break;
                    }
                }
            }
        }
    }
    resCtx.usePull = usePull ? 1 : 0;
    resCtx.useLarge = useLarge ? 1 : 0;
    resCtx.useRelay = useRelay ? 1 : 0;
    resCtx.relayRole = relayRole;
    resCtx.relayPeerRank = relayPeerRank;
    // 平衡 root 的 Clos 出流与 relay 的 Mesh 入流：
    //   chunk + relayBytes = (targetNum * chunk - relayNum * relayBytes) / ClosMeshRatio
    // 2×8 得 4/11，4×3 得 5/6；再受实际 CCL 中转缓冲容量约束。
    if (useRelay) {
        void *stagingBuffer = nullptr;
        uint64_t stagingCapacity = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &stagingBuffer, &stagingCapacity));

        const uint64_t relayNum = static_cast<uint64_t>(relays.size());
        const uint64_t targetNum = static_cast<uint64_t>(targets.size());
        CHK_PRT_RET(relayNum == 0 || targetNum <= CLOS_TO_MESH_BW_RATIO || stagingCapacity == 0,
            HCCL_ERROR("Invalid relay layout: relays=%llu targets=%llu staging=%llu",
                static_cast<unsigned long long>(relayNum), static_cast<unsigned long long>(targetNum),
                static_cast<unsigned long long>(stagingCapacity)),
            HCCL_E_INTERNAL);

        uint64_t balancedBytes =
            chunkBytes * (targetNum - CLOS_TO_MESH_BW_RATIO) / (CLOS_TO_MESH_BW_RATIO + relayNum);
        // 4×3 的 400MB+4B 按 CCU 实测 Clos/Mesh 带宽和 4 个 Clos jetty 配平：
        // (40 / 11) * (clos + relay) = 9 * clos - 2 * relay，得到 relay = 59 / 62 * clos。
        // 512MB 保持原带宽模型，避免改变已持平的性能点。
        if (is4x3 && useLarge && chunkBytes < 2 * MAX_DATA_SIZE) {
            balancedBytes = chunkBytes * 59 / 62;
        }
        // relayBytes 是配平后的总转发量（不再被 staging 钳制）；relaySliceBytes 是单次中转上限（= staging 容量）。
        // relayBytes > relaySliceBytes 时，relay 的 gather/forward 按 relaySliceBytes 切片循环转发。
        resCtx.relayBytes = (balancedBytes & ~3ULL);
        resCtx.relaySliceBytes = (std::min<uint64_t>(resCtx.relayBytes, stagingCapacity) & ~3ULL);
        CHK_PRT_RET(resCtx.relayBytes == 0 || resCtx.relayBytes >= chunkBytes || resCtx.relaySliceBytes == 0,
            HCCL_ERROR("Invalid relay bytes %llu (slice %llu) for chunk %llu",
                static_cast<unsigned long long>(resCtx.relayBytes),
                static_cast<unsigned long long>(resCtx.relaySliceBytes),
                static_cast<unsigned long long>(chunkBytes)),
            HCCL_E_INTERNAL);

        if (relayRole == 1) {
            resCtx.relayStagingAddr = reinterpret_cast<uint64_t>(stagingBuffer);
        }
    }
    // relay→target 映射（按 rank 索引），供 root mesh kernel 计算 payload 源偏移。
    resCtx.relayTargetOf.assign(param.rankSize, INVALID_VALUE_RANKID);
    for (uint32_t i = 0; i < relays.size() && i < targets.size(); i++) {
        resCtx.relayTargetOf[relays[i]] = targets[i];
    }

    std::vector<uint32_t> peerRanks;
    if (relayRole == 1 || relayRole == 2) {
        peerRanks.push_back(param.root);
        peerRanks.push_back(relayPeerRank);
    } else if (param.myRank == param.root) {
        peerRanks.reserve(param.rankSize - 1);
        for (uint32_t peerRank = 0; peerRank < param.rankSize; peerRank++) {
            if (peerRank != param.myRank) {
                peerRanks.push_back(peerRank);
            }
        }
    } else {
        peerRanks.push_back(param.root);
    }

    struct ChannelEntry {
        uint32_t peerRank;
        ChannelHandle channel;
    };
    // 按 (layer, dieId) 分组：机内 mesh（低 layer）与机间 clos 拆到独立 kernel，
    // 机内先完成先通知、机间慢慢传，大消息下可重叠。
    std::map<std::pair<uint32_t, uint32_t>, std::vector<ChannelEntry>> groups;
    for (const uint32_t peerRank : peerRanks) {
        CommLink link;
        uint32_t dieId = 0;
        uint32_t layer = 0;
        CHK_RET(FindCtpLink(comm, param.myRank, peerRank, &link, &dieId, &layer));

        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = peerRank;
        desc.notifyNum = DEFAULT_CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link.srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, engine, &desc, 1, &channel));
        // relay(1)/target(2) 的多通道合并进同一 kernel（忽略 layer，按实际 die 归组），避免跨 kernel 同步。
        const auto groupKey = (relayRole != 0) ? std::make_pair(0U, dieId) : std::make_pair(layer, dieId);
        groups[groupKey].push_back(ChannelEntry{peerRank, channel});
    }
    CHK_PRT_RET(groups.empty(), HCCL_ERROR("No CCU channel acquired"), HCCL_E_INTERNAL);

    // relay root 先启动完整长传输，再启动 relay 短尾，尽早把主链路灌满。
    if (useRelay && param.myRank == param.root) {
        const uint32_t relayedTargetNum = std::min<uint32_t>(static_cast<uint32_t>(relays.size()),
            static_cast<uint32_t>(targets.size()));
        auto isRelayedTarget = [&](uint32_t rank) {
            return std::find(targets.begin(), targets.begin() + relayedTargetNum, rank) !=
                targets.begin() + relayedTargetNum;
        };
        for (auto &groupPair : groups) {
            std::vector<ChannelEntry> &group = groupPair.second;
            std::stable_sort(group.begin(), group.end(), [&](const ChannelEntry &lhs, const ChannelEntry &rhs) {
                return isRelayedTarget(lhs.peerRank) < isRelayedTarget(rhs.peerRank);
            });
        }
    }

    // 按 (layer, dieId) 顺序确定分组顺序，保证资源布局确定性。
    std::vector<std::pair<uint32_t, uint32_t>> groupKeys;
    groupKeys.reserve(groups.size());
    for (const auto &group : groups) {
        groupKeys.push_back(group.first);
    }
    std::sort(groupKeys.begin(), groupKeys.end());

    // 8+4 大消息把 root 本地拷贝放到通道更少的 Clos kernel，减轻 7 路 Mesh kernel 的负担。
    uint32_t localCopyGroupIdx = 0;
    if (param.myRank == param.root && param.rankSize == 12 && localRankNum == 8 && useLarge) {
        for (uint32_t groupIdx = 1; groupIdx < static_cast<uint32_t>(groupKeys.size()); groupIdx++) {
            if (groups[groupKeys[groupIdx]].size() < groups[groupKeys[localCopyGroupIdx]].size()) {
                localCopyGroupIdx = groupIdx;
            }
        }
    }

    const uint32_t groupCount = static_cast<uint32_t>(groups.size());
    resCtx.threads.resize(groupCount);
    resCtx.threads[0] = param.cpuThread;
    if (groupCount > 1) {
        CHK_RET(HcclThreadAcquire(comm, engine, groupCount - 1, AUX_THREAD_NOTIFY_NUM, &resCtx.threads[1]));
    }

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Expected one CCU instance, but got %u", insNum), HCCL_E_INTERNAL);

    // KernelArg 需在 RegisterEnd 前保持存活，注册完成后其内容已固化为 CCU 图。
    std::vector<std::shared_ptr<CcuKernelArgScatter>> kernelArgs(groupCount);
    std::vector<std::shared_ptr<CcuKernelArgScatter>> extraKernelArgs; // 切片时 forward 非末片(doNotify=0)的 kernel 参数
    resCtx.ccuKernels.resize(groupCount);
    resCtx.peersPerKernel.resize(groupCount);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    for (uint32_t groupIdx = 0; groupIdx < groupCount; groupIdx++) {
        const std::vector<ChannelEntry> &group = groups[groupKeys[groupIdx]];
        CHK_PRT_RET(group.empty(), HCCL_ERROR("Empty CCU layer/die group"), HCCL_E_INTERNAL);

        kernelArgs[groupIdx] = std::make_shared<CcuKernelArgScatter>();
        CcuKernelArgScatter *arg = kernelArgs[groupIdx].get();
        arg->rankId = param.myRank;
        arg->rankSize = param.rankSize;
        arg->rootId = param.root;
        arg->doLocalCopy = (groupIdx == localCopyGroupIdx) ? 1 : 0;
        arg->relayRole = relayRole;
        arg->relayPeerRank = relayPeerRank;
        arg->relayIsMesh = (groupKeys[groupIdx].first == 0) ? 1 : 0;
        arg->doNotify = 1; // forward(Clos) 默认最后一片发 DONE（非切片恒 1；切片非末片用另一个 doNotify=0 的 kernel）
        arg->channelCount = static_cast<uint32_t>(group.size());
        resCtx.peersPerKernel[groupIdx].reserve(group.size());
        for (uint32_t channelIdx = 0; channelIdx < group.size(); channelIdx++) {
            arg->channels[channelIdx] = group[channelIdx].channel;
            arg->peerRanks[channelIdx] = group[channelIdx].peerRank;
            // root 的 clos 组：peer 是否被中继（relay 转前块，root 只发尾块）。
            arg->relayIsRelayed[channelIdx] = 0;
            if (useRelay && param.myRank == param.root && arg->relayIsMesh == 0) {
                const uint32_t peer = group[channelIdx].peerRank;
                for (uint32_t i = 0; i < relays.size() && i < targets.size(); i++) {
                    if (targets[i] == peer) {
                        arg->relayIsRelayed[channelIdx] = 1;
                        break;
                    }
                }
            }
            resCtx.peersPerKernel[groupIdx].push_back(group[channelIdx].peerRank);
        }

        char kernelName[64];
        int ret = snprintf(kernelName, sizeof(kernelName), "CcuScatterL%uD%u", groupKeys[groupIdx].first,
            groupKeys[groupIdx].second);
        CHK_PRT_RET(ret <= 0 || static_cast<size_t>(ret) >= sizeof(kernelName),
            HCCL_ERROR("Failed to build CCU kernel name"), HCCL_E_INTERNAL);

        const void *registerArgs[] = {arg};
        const void *kernelFunc = nullptr;
        bool isRelayMesh = false; // relay 的 mesh 组（对端=root），需再注册一个 own kernel 与 gather 并行
        if (useRelay) {
            if (param.myRank == param.root) {
                kernelFunc = reinterpret_cast<const void *>(ops_hccl::CcuScatterRelayRootKernel);
            } else if (relayRole == 1) {
                // relay 拆 mesh(die1, 对端=root)/Clos(die0, 对端=target)，mesh 组再拆 gather+own 两 kernel，
                // 跨 die 先后由 host 线程 notify 串。
                isRelayMesh = (!group.empty() && group[0].peerRank == param.root);
                kernelFunc = reinterpret_cast<const void *>(
                    isRelayMesh ? ops_hccl::CcuScatterRelayRelayMeshKernel : ops_hccl::CcuScatterRelayRelayClosKernel);
            } else if (relayRole == 2) {
                kernelFunc = reinterpret_cast<const void *>(ops_hccl::CcuScatterRelayTargetKernel);
            } else {
                // rank15 最后一个 target：直收整块，走 direct non-root。
                kernelFunc = reinterpret_cast<const void *>(ops_hccl::CcuScatterNonRootKernel);
            }
        } else if (param.myRank == param.root) {
            if (usePull) {
                kernelFunc = reinterpret_cast<const void *>(ops_hccl::CcuScatterPullRootKernel);
            } else if (useLarge) {
                kernelFunc = reinterpret_cast<const void *>(ops_hccl::CcuScatterLargeRootKernel);
            } else if (is4x1Small) {
                kernelFunc = reinterpret_cast<const void *>(ops_hccl::CcuScatter4x1RootKernel);
            } else {
                kernelFunc = reinterpret_cast<const void *>(ops_hccl::CcuScatterRootKernel);
            }
        } else {
            if (usePull) {
                kernelFunc = reinterpret_cast<const void *>(ops_hccl::CcuScatterPullNonRootKernel);
            } else if (useLarge) {
                kernelFunc = reinterpret_cast<const void *>(ops_hccl::CcuScatterLargeNonRootKernel);
            } else if (is4x1Small) {
                kernelFunc = reinterpret_cast<const void *>(ops_hccl::CcuScatter4x1NonRootKernel);
            } else {
                kernelFunc = reinterpret_cast<const void *>(ops_hccl::CcuScatterNonRootKernel);
            }
        }
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, groupKeys[groupIdx].second, kernelName, kernelFunc, registerArgs,
            1, &resCtx.ccuKernels[groupIdx]));

        // relay 的 clos 组（forward）在切片时再注册一个 doNotify=0 的 kernel，非末片不发 DONE。
        if (relayRole == 1 && !isRelayMesh && resCtx.relaySliceBytes < resCtx.relayBytes) {
            extraKernelArgs.emplace_back(std::make_shared<CcuKernelArgScatter>(*arg));
            extraKernelArgs.back()->doNotify = 0;
            const void *noDoneArgs[] = {extraKernelArgs.back().get()};
            char noDoneName[64];
            int noDoneRet = snprintf(noDoneName, sizeof(noDoneName), "CcuScatterRelayClosNoDoneL%uD%u",
                groupKeys[groupIdx].first, groupKeys[groupIdx].second);
            CHK_PRT_RET(noDoneRet <= 0 || static_cast<size_t>(noDoneRet) >= sizeof(noDoneName),
                HCCL_ERROR("Failed to build relay clos no-done kernel name"), HCCL_E_INTERNAL);
            CHK_RET_CCU(HcommCcuKernelRegister(insHandle, groupKeys[groupIdx].second, noDoneName,
                reinterpret_cast<const void *>(ops_hccl::CcuScatterRelayRelayClosKernel), noDoneArgs, 1,
                &resCtx.relayClosNoDoneKernel));
        }

        // relay 的 mesh 组再注册 own kernel（读自己块），与 gather 同 thread/die/channel，让 forward(Clos)
        // 在 gather 完成后即启动、与 own 读并行。
        if (isRelayMesh) {
            char ownName[64];
            int ownRet = snprintf(ownName, sizeof(ownName), "CcuScatterRelayOwnL%uD%u", groupKeys[groupIdx].first,
                groupKeys[groupIdx].second);
            CHK_PRT_RET(ownRet <= 0 || static_cast<size_t>(ownRet) >= sizeof(ownName),
                HCCL_ERROR("Failed to build relay own kernel name"), HCCL_E_INTERNAL);
            CHK_RET_CCU(HcommCcuKernelRegister(insHandle, groupKeys[groupIdx].second, ownName,
                reinterpret_cast<const void *>(ops_hccl::CcuScatterRelayRelayOwnKernel), registerArgs, 1,
                &resCtx.relayOwnKernel));
        }
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
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
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

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
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rank size %u", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(
        root >= param.rankSize, HCCL_ERROR("Invalid root %u for rank size %u", root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("Unsupported data type %d", dataType), HCCL_E_NOT_SUPPORT);

    // 空数据或单 rank 无对端，直接返回（赛题拓扑为 4/12/16 rank，此处仅兜底）。
    if (recvCount == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    // 直发星形：root 向每个对端写各自的 chunk。
    // 小消息（≤1MB）走拉模型，2×8/4×3 大消息走中继（relay），其余大消息分片下放 kernel 内，中段走单片推模型。
    const uint64_t chunkBytesTag = param.count * sizeof(float);
    bool relayTopo = false;
    if (chunkBytesTag > PULL_THRESHOLD) {
        if (param.rankSize == 16) {
            relayTopo = true;
        } else if (param.rankSize == 12) {
            uint32_t *layer0Ranks = nullptr;
            uint32_t layer0Num = 0;
            relayTopo = (HcclRankGraphGetRanksByLayer(comm, 0, &layer0Ranks, &layer0Num) == HCCL_SUCCESS
                && layer0Ranks != nullptr && layer0Num == 3);
        }
    }
    const char *mode = nullptr;
    if (chunkBytesTag <= PULL_THRESHOLD) {
        mode = "pull";
    } else if (relayTopo) {
        mode = "relay";
    } else if (chunkBytesTag > MAX_DATA_SIZE) {
        mode = "large";
    } else {
        mode = "direct";
    }
    int tagLen = snprintf(param.tag, sizeof(param.tag), "hccl_custom_scatter_%s_r%u", mode, root);
    CHK_PRT_RET(tagLen < 0 || static_cast<size_t>(tagLen) >= sizeof(param.tag), HCCL_ERROR("Failed to build op tag"),
        HCCL_E_INTERNAL);

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 2, &param.cpuThread));

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
        CHK_RET(BuildResources(comm, ccuEngine, param, resCtxHost));

        // ==============================================
        // STEP 2.2: 申请通信引擎上下文
        // ==============================================
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
