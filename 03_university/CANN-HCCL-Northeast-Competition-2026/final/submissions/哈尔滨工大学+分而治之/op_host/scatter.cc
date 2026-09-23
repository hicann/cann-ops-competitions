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

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"

namespace ops_hccl {
CcuResult CcuKernel(CcuKernelArg arg);
CcuResult CcuKernelSmall(CcuKernelArg arg);
} // namespace ops_hccl

namespace {

struct PeerLinkChoice {
    uint32_t peer;
    CommLink link;
    uint32_t layer;
    uint32_t localDie;
};

std::string EndpointKey(const EndpointDesc &e)
{
    std::ostringstream s;
    s << static_cast<int>(e.protocol) << ':' << static_cast<int>(e.commAddr.type) << ':';
    size_t bytes = 0;
    switch (e.commAddr.type) {
        case COMM_ADDR_TYPE_IP_V4:
            bytes = sizeof(e.commAddr.addr);
            break;
        case COMM_ADDR_TYPE_IP_V6:
            bytes = sizeof(e.commAddr.addr6);
            break;
        case COMM_ADDR_TYPE_ID:
            bytes = sizeof(e.commAddr.id);
            break;
        case COMM_ADDR_TYPE_EID:
            bytes = sizeof(e.commAddr.eid);
            break;
        default:
            return "unsupported";
    }
    for (size_t i = 0; i < bytes; ++i) {
        s << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(e.commAddr.raws[i]);
    }
    s << std::dec << ':' << static_cast<int>(e.loc.locType);
    if (e.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
        s << ':' << e.loc.device.devPhyId << ':' << e.loc.device.superDevId << ':' << e.loc.device.serverIdx << ':'
          << e.loc.device.superPodIdx;
    } else if (e.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
        s << ':' << e.loc.host.id;
    } else {
        return "unsupported";
    }
    return s.str();
}

uint32_t QueryLocalDie(HcclComm comm, uint32_t rank, const EndpointDesc &endpoint)
{
    uint32_t value = UINT32_MAX;
    const HcclResult ret
        = HcclRankGraphGetEndpointInfo(comm, rank, &endpoint, ENDPOINT_ATTR_DIE_ID, sizeof(value), &value);
    return ret == HCCL_SUCCESS ? value : UINT32_MAX;
}

// 两端统一按 (minRank,maxRank) 方向查询并排序，保证两端选中同一条物理链路；
// 本端为 maxRank 时交换端点得到本地视角。同机 pair 在 layer0(mesh) 有候选，
// 跨机 pair 只有 layer1(Clos)，按层升序首选即得"同机 mesh、跨机 Clos"。
HcclResult SelectPeerLink(
    HcclComm comm, uint32_t rank, uint32_t peer, const std::vector<uint32_t> &layers, PeerLinkChoice &choice)
{
    const uint32_t a = std::min(rank, peer);
    const uint32_t b = std::max(rank, peer);
    for (const uint32_t layer : layers) {
        CommLink *links = nullptr;
        uint32_t linkCount = 0;
        if (HcclRankGraphGetLinks(comm, layer, a, b, &links, &linkCount) != HCCL_SUCCESS || links == nullptr
            || linkCount == 0) {
            continue;
        }
        std::vector<CommLink> forward(links, links + linkCount);
        links = nullptr;
        linkCount = 0;
        if (HcclRankGraphGetLinks(comm, layer, b, a, &links, &linkCount) != HCCL_SUCCESS || links == nullptr
            || linkCount == 0) {
            continue;
        }
        const std::vector<CommLink> reverse(links, links + linkCount);
        auto key = [](const CommLink &l) {
            return std::to_string(static_cast<int>(l.linkAttr.linkProtocol)) + "/" + EndpointKey(l.srcEndpointDesc)
                   + "/" + EndpointKey(l.dstEndpointDesc);
        };
        std::sort(forward.begin(), forward.end(), [&](const CommLink &x, const CommLink &y) {
            return key(x) < key(y);
        });
        for (const auto &candidate : forward) {
            if (candidate.linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            if (EndpointKey(candidate.srcEndpointDesc) == "unsupported"
                || EndpointKey(candidate.dstEndpointDesc) == "unsupported") {
                continue;
            }
            bool matched = false;
            for (const auto &rev : reverse) {
                matched |= rev.linkAttr.linkProtocol == candidate.linkAttr.linkProtocol
                           && EndpointKey(rev.srcEndpointDesc) == EndpointKey(candidate.dstEndpointDesc)
                           && EndpointKey(rev.dstEndpointDesc) == EndpointKey(candidate.srcEndpointDesc);
            }
            if (!matched) {
                continue;
            }
            CommLink selected = candidate;
            if (rank == b) {
                std::swap(selected.srcEndpointDesc, selected.dstEndpointDesc);
            }
            const uint32_t die = QueryLocalDie(comm, rank, selected.srcEndpointDesc);
            CHK_PRT_RET(
                die > 1, HCCL_ERROR("[P2Probe] endpoint die %u out of range for peer %u", die, peer), HCCL_E_PARA);
            choice.peer = peer;
            choice.link = selected;
            choice.layer = layer;
            choice.localDie = die;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("[P2Probe] no rank-graph link to peer %u", peer);
    return HCCL_E_NOT_FOUND;
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
    // E5.4 L1-L4 改变了注册 kernel 的指令流；换身份避免进程复用旧上下文或旧指令。
    snprintf(param.tag, sizeof(param.tag), "hccl_custom_scatter_e54_large_v2_r%u", root);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    int32_t deviceId = -1;
    CHK_PRT_RET(aclrtGetDevice(&deviceId) != ACL_SUCCESS, HCCL_ERROR("[P2Probe] unable to obtain current device"),
        HCCL_E_INTERNAL);
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[P2Probe] unsupported rank size %u on device %d", param.rankSize, deviceId), HCCL_E_PARA);
    CHK_PRT_RET(root >= param.rankSize, HCCL_ERROR("[P2Probe] invalid root %u for rank size %u", root, param.rankSize),
        HCCL_E_PARA);

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
        AlgResourceCtx resCtxHost{};
        resCtxHost.ccuThread = param.cpuThread;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 选链 → 批量建链 → 按本地 die 分组注册 CCU Kernel
        // ==============================================
        uint32_t *layerList = nullptr;
        uint32_t layerCount = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerCount));
        CHK_PRT_RET(
            layerList == nullptr || layerCount == 0, HCCL_ERROR("[P2Probe] no rank-graph layers"), HCCL_E_NOT_FOUND);
        std::vector<uint32_t> layers(layerList, layerList + layerCount);
        std::sort(layers.begin(), layers.end());

        std::vector<PeerLinkChoice> peerLinks;
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.myRank) {
                continue;
            }
            PeerLinkChoice choice{};
            CHK_RET(SelectPeerLink(comm, param.myRank, peer, layers, choice));
            peerLinks.push_back(choice);
        }

        // ==============================================
        // layer 0 is server-local, layer 1 is Clos in the competition topology.
        // GetRanksByLayer is authoritative locally; instance sizes are an unordered multiset.
        CHK_PRT_RET(!std::binary_search(layers.begin(), layers.end(), 0U)
                        || !std::binary_search(layers.begin(), layers.end(), 1U),
            HCCL_ERROR("[RelayTopo] required layer 0/1 missing"), HCCL_E_NOT_SUPPORT);
        uint32_t *serverRanks = nullptr;
        uint32_t serverSize = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &serverRanks, &serverSize));
        CHK_PRT_RET(serverRanks == nullptr || serverSize == 0 || serverSize > param.rankSize,
            HCCL_ERROR("[RelayTopo] invalid local member list"), HCCL_E_PARA);
        std::vector<uint32_t> ownServer(serverRanks, serverRanks + serverSize);
        std::sort(ownServer.begin(), ownServer.end());
        CHK_PRT_RET(ownServer.back() >= param.rankSize
                        || std::adjacent_find(ownServer.begin(), ownServer.end()) != ownServer.end()
                        || !std::binary_search(ownServer.begin(), ownServer.end(), param.myRank),
            HCCL_ERROR("[RelayTopo] invalid local membership"), HCCL_E_PARA);
        const bool rootInOwn = std::binary_search(ownServer.begin(), ownServer.end(), param.root);
        for (const auto &choice : peerLinks) {
            const bool inOwn = std::binary_search(ownServer.begin(), ownServer.end(), choice.peer);
            CHK_PRT_RET((choice.layer == 0) != inOwn || choice.layer > 1,
                HCCL_ERROR("[RelayTopo] membership/link mismatch peer=%u", choice.peer), HCCL_E_PARA);
        }
        uint32_t *instSizes = nullptr;
        uint32_t instCount = 0;
        CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizes, &instCount));
        CHK_PRT_RET(instSizes == nullptr || instCount == 0 || instCount > param.rankSize,
            HCCL_ERROR("[RelayTopo] invalid instance sizes"), HCCL_E_PARA);
        const std::vector<uint32_t> sizes(instSizes, instSizes + instCount);
        uint64_t sum = 0;
        bool uniform = true;
        for (const auto n : sizes) {
            CHK_PRT_RET(n == 0 || n > param.rankSize, HCCL_ERROR("[RelayTopo] invalid instance size"), HCCL_E_PARA);
            sum += n;
            uniform = uniform && n == serverSize;
        }
        CHK_PRT_RET(sum != param.rankSize || std::find(sizes.begin(), sizes.end(), serverSize) == sizes.end(),
            HCCL_ERROR("[RelayTopo] inconsistent instance sizes"), HCCL_E_PARA);

        // Two servers retain E4.2's point-to-point route. With uniform >2 servers,
        // everyone knows k, but only root-local ranks need root membership identity.
        // Remote ranks participate in a static Clos-only completion phase instead.
        const bool passiveP2 = instCount > 2 && uniform;
        const bool routeSupported = instCount == 2 || passiveP2;
        uint32_t kRoot = rootInOwn ? serverSize : (instCount == 2 ? param.rankSize - serverSize : sizes.front());
        const bool relayEligible
            = routeSupported && kRoot >= 2 && param.rankSize >= kRoot + 5 && param.rankSize < 2 * kRoot + 7;
        uint32_t myDestRank = UINT32_MAX;
        uint32_t myHelperRank = UINT32_MAX;
        std::vector<uint32_t> helpers;
        std::vector<uint32_t> remotes;
        if (relayEligible && (rootInOwn || !passiveP2)) {
            for (uint32_t r = 0; r < param.rankSize; ++r) {
                const bool inOwn = std::binary_search(ownServer.begin(), ownServer.end(), r);
                const bool inRootServer = rootInOwn ? inOwn : !inOwn;
                if (inRootServer && r != param.root) {
                    helpers.push_back(r);
                } else if (!inRootServer) {
                    remotes.push_back(r);
                }
            }
            const size_t pairs = std::min(helpers.size(), remotes.size());
            for (size_t i = 0; i < pairs; ++i) {
                if (helpers[i] == param.myRank) {
                    myDestRank = remotes[i];
                }
                if (remotes[i] == param.myRank) {
                    myHelperRank = helpers[i];
                }
            }
        }
        HCCL_INFO("[RelayTopo] rank=%u root=%u local=%u instances=%u k=%u eligible=%u passive=%u dest=%u helper=%u",
            param.myRank, param.root, serverSize, instCount, kRoot, relayEligible ? 1U : 0U, passiveP2 ? 1U : 0U,
            myDestRank, myHelperRank);
        const size_t serveCount = std::min(helpers.size(), remotes.size());
        resCtxHost.kRoot = kRoot;
        resCtxHost.relayEligible = relayEligible ? (passiveP2 ? 2 : 1) : 0;
        HCCL_INFO("[P2Probe] rank %u relay eligible=%u k=%u helpers=%zu remotes=%zu myDest=%u myHelper=%u",
            param.myRank, relayEligible ? 1U : 0U, kRoot, helpers.size(), remotes.size(), myDestRank, myHelperRank);

        // 一次性批量申请全部 channel：逐条阻塞申请会把建链串行化并产生互等
        std::vector<HcclChannelDesc> descriptions(peerLinks.size());
        std::vector<ChannelHandle> channels(peerLinks.size());
        for (size_t i = 0; i < peerLinks.size(); ++i) {
            auto &desc = descriptions[i];
            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = peerLinks[i].peer;
            desc.notifyNum = 3;
            desc.channelProtocol = peerLinks[i].link.linkAttr.linkProtocol;
            desc.localEndpoint = peerLinks[i].link.srcEndpointDesc;
            desc.remoteEndpoint = peerLinks[i].link.dstEndpointDesc;
            HCCL_INFO("[P2Probe] rank %u -> %u: layer=%u protocol=%d localDie=%u", param.myRank, peerLinks[i].peer,
                peerLinks[i].layer, static_cast<int>(desc.channelProtocol), peerLinks[i].localDie);
        }
        if (!descriptions.empty()) {
            CHK_RET(HcclChannelAcquire(comm, ccuEngine, descriptions.data(), descriptions.size(), channels.data()));
        }

        CcuInsHandle insHandle{0};
        uint32_t insCount = 0;
        CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
        CHK_PRT_RET(
            insCount != 1, HCCL_ERROR("[P2Probe] expected one CCU instruction, got %u", insCount), HCCL_E_INTERNAL);
        const CcuResult startResult = HcommCcuKernelRegisterStart(insHandle);
        CHK_PRT_RET(startResult != CCU_SUCCESS,
            HCCL_ERROR("[P2Probe] CCU kernel registration start failed: %d", startResult), HCCL_E_INTERNAL);

        resCtxHost.threads.push_back(param.cpuThread);
        // 一个 kernel 的全部 channel 必须同属一个 IO die：按 channel 实际本地 die 分组，
        // 每个非空 die 组注册一个自包含 kernel；root 自拷贝只交给第一个非空组。
        // P1/P2 keep their original die placement. A small entry may occupy only
        // the second mission on a die with no P2 reservation.
        // Reserve the original P2 mission before choosing small-entry coverage.
        uint32_t relayDie = UINT32_MAX;
        const uint32_t smallP2Peer = myDestRank != UINT32_MAX ? myDestRank : myHelperRank;
        if (relayEligible) {
            for (const auto &link : peerLinks) {
                if ((passiveP2 && link.layer == 1) || (!passiveP2 && link.peer == smallP2Peer)) {
                    relayDie = link.localDie;
                    break;
                }
            }
        }
        uint32_t groupIndex = 0;
        for (uint32_t groupDie = 0; groupDie < 2; ++groupDie) {
            auto kernelArg = std::make_shared<CcuKernelArgDirectProbe>();
            kernelArg->rankSize = param.rankSize;
            kernelArg->rankId = param.myRank;
            kernelArg->rootId = param.root;
            kernelArg->groupDie = groupDie;
            kernelArg->phase = 1;
            kernelArg->relayVariant = relayEligible ? (passiveP2 ? 2 : 1) : 0;
            kernelArg->relayRole = 0;
            kernelArg->publishRootIdx = UINT32_MAX;
            kernelArg->smallRootIdx = UINT32_MAX;
            kernelArg->ingressCount = 0;
            kernelArg->channelCount = 0;
            for (size_t i = 0; i < peerLinks.size(); ++i) {
                if (peerLinks[i].localDie != groupDie) {
                    continue;
                }
                const uint32_t idx = kernelArg->channelCount;
                kernelArg->channels[idx] = channels[i];
                kernelArg->peerRanks[idx] = peerLinks[i].peer;
                if (peerLinks[i].peer == param.root) {
                    kernelArg->smallRootIdx = idx;
                }
                ++kernelArg->channelCount;
            }
            // E5.4 root-star leaves a nonroot group without its root channel entirely empty.
            // Do not register it: fixed-resource SDKs infer the die from used channels and
            // place channel-free kernels on the first enabled die, consuming P1/P2 missions.
            // Skip before assigning compact group indices; helper prefix uses the root group too.
            if (kernelArg->channelCount == 0 || (param.myRank != param.root && kernelArg->smallRootIdx == UINT32_MAX)) {
                continue;
            }
            // E4.5：非 root 的小消息只经 root channel；记录其所在 die 组（root 恒无此标记）
            if (kernelArg->smallRootIdx != UINT32_MAX) {
                resCtxHost.smallRootGroup = groupIndex;
            }
            kernelArg->selfCopy = groupIndex == 0 ? 1 : 0;
            if (kernelArg->relayVariant != 0) {
                // serving helper：在持有 root channel 的组 kernel 上发布本端 scratch
                if (myDestRank != UINT32_MAX) {
                    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
                        if (kernelArg->peerRanks[idx] == param.root) {
                            kernelArg->publishRootIdx = idx;
                            break;
                        }
                    }
                }
                // root：本组 helper channel 的 ingress 表 + 被服务 remote 标记
                if (param.myRank == param.root) {
                    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
                        const uint32_t peer = kernelArg->peerRanks[idx];
                        const auto hIt = std::lower_bound(helpers.begin(), helpers.end(), peer);
                        if (hIt != helpers.end() && *hIt == peer) {
                            const size_t hIdx = static_cast<size_t>(hIt - helpers.begin());
                            if (hIdx < serveCount) {
                                kernelArg->ingressHelperIdx[kernelArg->ingressCount] = idx;
                                kernelArg->ingressRemoteRank[kernelArg->ingressCount] = remotes[hIdx];
                                ++kernelArg->ingressCount;
                            }
                        }
                        const auto rIt = std::lower_bound(remotes.begin(), remotes.end(), peer);
                        if (rIt != remotes.end() && *rIt == peer
                            && static_cast<size_t>(rIt - remotes.begin()) < serveCount) {
                            kernelArg->servedRemote[idx] = 1;
                        }
                    }
                }
            }
            const bool prefixGroup = kernelArg->ingressCount != 0 || kernelArg->publishRootIdx != UINT32_MAX;
            resCtxHost.prefixGroups.push_back(prefixGroup ? 1U : 0U);
            if (kernelArg->publishRootIdx != UINT32_MAX) {
                resCtxHost.helperPrefixGroup = groupIndex;
            }
            CcuKernelInfo kernelInfo{};
            snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernel_g%u_die%u", groupIndex,
                groupDie);
            kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);
            kernelInfo.setKernelArg(kernelArg);
            const void *kernelArgs[] = {kernelInfo.kernelArg};
            CcuKernelHandle kernelHandle{0};
            const CcuResult registerResult = HcommCcuKernelRegister(
                insHandle, groupDie, kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
            CHK_PRT_RET(registerResult != CCU_SUCCESS,
                HCCL_ERROR("[P2Probe] CCU kernel registration failed: %d", registerResult), HCCL_E_INTERNAL);
            HCCL_INFO("[P2Probe] rank %u registered group=%u die=%u variant=%u channels=%u", param.myRank, groupIndex,
                groupDie, kernelArg->relayVariant, kernelArg->channelCount);
            resCtxHost.ccuKernels.push_back(kernelHandle);
            // Existing instance has two missions per die. P1+P2 dies remain generic.
            const bool smallEntry
                = groupDie != relayDie && (param.myRank == param.root || kernelArg->smallRootIdx != UINT32_MAX);
            CcuKernelHandle smallHandle{0};
            if (smallEntry) {
                char name[64];
                snprintf(name, sizeof(name), "CcuKernelSmall_g%u_die%u", groupIndex, groupDie);
                const CcuResult smallResult = HcommCcuKernelRegister(insHandle, groupDie, name,
                    reinterpret_cast<void *>(ops_hccl::CcuKernelSmall), kernelArgs, 1, &smallHandle);
                CHK_PRT_RET(smallResult != CCU_SUCCESS, HCCL_ERROR("[SmallEntry] registration failed: %d", smallResult),
                    HCCL_E_INTERNAL);
            }
            resCtxHost.smallKernels.push_back(smallHandle);
            resCtxHost.smallDedicated.push_back(smallEntry ? 1U : 0U);
            HCCL_INFO("[SmallEntry] rank=%u root=%u group=%u die=%u dedicated=%u relayDie=%u args=%u", param.myRank,
                param.root, groupIndex, groupDie, smallEntry ? 1U : 0U, relayDie,
                smallEntry ? (param.myRank == param.root ? 5U : 2U) : (param.myRank == param.root ? 12U : 7U));
            ++groupIndex;
        }
        CHK_PRT_RET(groupIndex == 0 && param.rankSize > 1,
            HCCL_ERROR("[P2Probe] no kernel registered for rank size %u", param.rankSize), HCCL_E_INTERNAL);
        // Resource-blocked small groups retain the generic P1 entry.
        // 非 root 必须在一个 die 组内持有 root channel；root 的各组天然覆盖全部 peer。
        CHK_PRT_RET(param.myRank != param.root && resCtxHost.smallRootGroup == UINT32_MAX,
            HCCL_ERROR("[SmallRead] no die group holds the root channel"), HCCL_E_INTERNAL);

        // Uniform multi-server route: every rank uses one Clos-only P2 kernel.
        // Each peer publishes READY(address+token) before waiting for DONE(epoch).
        // Only root-local helpers write payload; all remote ranks remain role-agnostic.
        if (relayEligible && passiveP2) {
            auto p2Arg = std::make_shared<CcuKernelArgDirectProbe>();
            p2Arg->rankSize = param.rankSize;
            p2Arg->rankId = param.myRank;
            p2Arg->rootId = param.root;
            p2Arg->phase = 3;
            p2Arg->relayDestIdx = UINT32_MAX;
            uint32_t closDie = UINT32_MAX;
            for (size_t i = 0; i < peerLinks.size(); ++i) {
                if (peerLinks[i].layer != 1) {
                    continue;
                }
                if (closDie == UINT32_MAX) {
                    closDie = peerLinks[i].localDie;
                }
                CHK_PRT_RET(closDie != peerLinks[i].localDie,
                    HCCL_ERROR("[RelayTopo] Clos channels span multiple dies"), HCCL_E_NOT_SUPPORT);
                const uint32_t idx = p2Arg->channelCount++;
                p2Arg->channels[idx] = channels[i];
                p2Arg->peerRanks[idx] = peerLinks[i].peer;
                if (peerLinks[i].peer == myDestRank) {
                    p2Arg->relayDestIdx = idx;
                }
            }
            CHK_PRT_RET(p2Arg->channelCount == 0 || (myDestRank != UINT32_MAX && p2Arg->relayDestIdx == UINT32_MAX),
                HCCL_ERROR("[RelayTopo] missing Clos relay channels"), HCCL_E_INTERNAL);
            p2Arg->groupDie = closDie;
            CcuKernelInfo info{};
            snprintf(info.kernelFuncName, sizeof(info.kernelFuncName), "CcuKernelPassiveRelay");
            info.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);
            info.setKernelArg(p2Arg);
            const void *args[] = {info.kernelArg};
            CcuKernelHandle handle{0};
            const CcuResult result
                = HcommCcuKernelRegister(insHandle, closDie, info.kernelFuncName, info.kernelFunc, args, 1, &handle);
            CHK_PRT_RET(result != CCU_SUCCESS, HCCL_ERROR("[RelayTopo] passive relay registration failed: %d", result),
                HCCL_E_INTERNAL);
            resCtxHost.relayKernels.push_back(handle);
        }

        // E4：相位2 kernel（serving helper 转发后 Record epoch / 被服务 dest Wait epoch），
        // 单 channel（helper<->dest 一一对应）；地址/token 由相位2 自交换（专用槽/bit，
        // epoch 消费链保证跨 chunk 复用安全），不依赖相位1 的槽位，故无需奇偶多实例
        const uint32_t p2Peer = myDestRank != UINT32_MAX ? myDestRank : myHelperRank;
        if (relayEligible && !passiveP2 && p2Peer != UINT32_MAX) {
            for (size_t i = 0; i < peerLinks.size(); ++i) {
                if (peerLinks[i].peer != p2Peer) {
                    continue;
                }
                auto p2Arg = std::make_shared<CcuKernelArgDirectProbe>();
                p2Arg->rankSize = param.rankSize;
                p2Arg->rankId = param.myRank;
                p2Arg->rootId = param.root;
                p2Arg->groupDie = peerLinks[i].localDie;
                p2Arg->selfCopy = 0;
                p2Arg->phase = 2;
                p2Arg->relayVariant = 0;
                p2Arg->relayRole = myDestRank != UINT32_MAX ? 1 : 0;
                p2Arg->publishRootIdx = UINT32_MAX;
                p2Arg->ingressCount = 0;
                p2Arg->channelCount = 1;
                p2Arg->channels[0] = channels[i];
                p2Arg->peerRanks[0] = p2Peer;
                CcuKernelInfo p2Info{};
                snprintf(p2Info.kernelFuncName, sizeof(p2Info.kernelFuncName), "CcuKernelRelay");
                p2Info.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);
                p2Info.setKernelArg(p2Arg);
                const void *p2Args[] = {p2Info.kernelArg};
                CcuKernelHandle p2Handle{0};
                const CcuResult p2Result = HcommCcuKernelRegister(
                    insHandle, peerLinks[i].localDie, p2Info.kernelFuncName, p2Info.kernelFunc, p2Args, 1, &p2Handle);
                CHK_PRT_RET(p2Result != CCU_SUCCESS,
                    HCCL_ERROR("[P2Probe] P2 kernel registration failed: %d", p2Result), HCCL_E_INTERNAL);
                HCCL_INFO("[P2Probe] rank %u registered P2 peer=%u role=%u die=%u", param.myRank, p2Peer,
                    p2Arg->relayRole, peerLinks[i].localDie);
                resCtxHost.relayKernels.push_back(p2Handle);
            }
        }
        const CcuResult endResult = HcommCcuKernelRegisterEnd(insHandle);
        CHK_PRT_RET(endResult != CCU_SUCCESS, HCCL_ERROR("[P2Probe] CCU kernel registration end failed: %d", endResult),
            HCCL_E_INTERNAL);

        // 双 die 分组时，两个 kernel 各跑在独立 stream 的 thread 上；
        // 同 stream 背靠背 launch 会被按 stream 前驱关系展开成跨 rank 依赖环
        if (groupIndex > 1) {
            CHK_PRT_RET(aclrtCreateStream(&resCtxHost.auxiliaryStream) != ACL_SUCCESS,
                HCCL_ERROR("[P2Probe] auxiliary stream creation failed"), HCCL_E_INTERNAL);
            CHK_PRT_RET(aclrtCreateNotify(&resCtxHost.startNotify, 0) != ACL_SUCCESS
                            || aclrtCreateNotify(&resCtxHost.doneNotify, 0) != ACL_SUCCESS,
                HCCL_ERROR("[P2Probe] fork/join notify creation failed"), HCCL_E_INTERNAL);
            ThreadHandle auxiliaryThread{};
            CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, resCtxHost.auxiliaryStream, 0, &auxiliaryThread));
            CHK_PRT_RET(auxiliaryThread == param.cpuThread,
                HCCL_ERROR("[P2Probe] distinct streams unexpectedly share thread"), HCCL_E_INTERNAL);
            resCtxHost.threads.push_back(auxiliaryThread);
        }

        // E4：P2 流水跑在独立 stream 的 thread 上，与 P1 流组并行（chunk 级重叠）；
        // fork 按 chunk 奇偶各用一个 notify，避免连续 Record 合并丢失
        if (!resCtxHost.relayKernels.empty()) {
            CHK_PRT_RET(aclrtCreateStream(&resCtxHost.relayStream) != ACL_SUCCESS,
                HCCL_ERROR("[P2Probe] relay stream creation failed"), HCCL_E_INTERNAL);
            CHK_PRT_RET(aclrtCreateNotify(&resCtxHost.relayForkParity0, 0) != ACL_SUCCESS
                            || aclrtCreateNotify(&resCtxHost.relayForkParity1, 0) != ACL_SUCCESS
                            || aclrtCreateNotify(&resCtxHost.relayDoneNotify, 0) != ACL_SUCCESS,
                HCCL_ERROR("[P2Probe] relay fork/join notify creation failed"), HCCL_E_INTERNAL);
            ThreadHandle relayThread{};
            CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, resCtxHost.relayStream, 0, &relayThread));
            CHK_PRT_RET(relayThread == param.cpuThread
                            || (resCtxHost.threads.size() > 1 && relayThread == resCtxHost.threads[1]),
                HCCL_ERROR("[P2Probe] relay stream unexpectedly shares thread"), HCCL_E_INTERNAL);
            resCtxHost.relayThread = relayThread;
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
    CHK_RET(ops_hccl::ExecOp(param, stream));
    return HCCL_SUCCESS;
}
