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
 #include <iterator>
 #include <limits>
 #include <memory>
 #include <utility>
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
 constexpr uint32_t MAIN_THREAD_NOTIFY_NUM = ALLGATHER_PARALLEL_MAX_KERNEL_NUM - 1;
 constexpr uint32_t SLAVE_THREAD_NOTIFY_NUM = 1;
 constexpr uint32_t MAX_CCU_DIE_NUM = 2;
 
 struct KernelChannelGroup {
     uint32_t dieId = 0;
     std::vector<uint32_t> peerRanks;
     std::vector<ChannelHandle> channels;
 };
 
 struct ChannelCandidate {
     HcclChannelDesc desc{};
     uint32_t dieId = 0;
 };
 
 struct RankChannelCandidates {
     uint32_t peerRank = 0;
     ChannelCandidate primary;
 };
 
 struct AcquiredChannel {
     uint32_t peerRank = 0;
     uint32_t dieId = 0;
     ChannelHandle channel{};
 };
 
 struct ChannelPlan {
     std::vector<KernelChannelGroup> directGroups;
     bool parallel2x8Enabled = false;
     bool asym8x4Enabled = false;
     uint32_t parallelCounterpartRank = MAX_RANK_SIZE;
     uint32_t parallelInterChannelCount = 0;
     std::vector<uint32_t> parallelLocalRanks;
     std::vector<KernelChannelGroup> parallelIntraGroups;
     std::vector<KernelChannelGroup> parallelInterGroups;
     std::vector<KernelChannelGroup> asymGatewayInterGroups;
 };
 
 bool IsValidUbcLink(const CommLink &link)
 {
     return link.linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP
            && link.srcEndpointDesc.protocol != CommProtocol::COMM_PROTOCOL_RESERVED
            && link.dstEndpointDesc.protocol != CommProtocol::COMM_PROTOCOL_RESERVED
            && link.srcEndpointDesc.loc.locType != EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED
            && link.dstEndpointDesc.loc.locType != EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED;
 }
 
 HcclResult MakeChannelCandidate(
     HcclComm comm, uint32_t myRank, uint32_t remoteRank, const CommLink &link, ChannelCandidate &candidate)
 {
     CHK_RET(HcclChannelDescInit(&candidate.desc, 1));
     candidate.desc.remoteRank = remoteRank;
     candidate.desc.notifyNum = CHANNEL_NOTIFY_NUM;
     candidate.desc.channelProtocol = link.linkAttr.linkProtocol;
     candidate.desc.localEndpoint = link.srcEndpointDesc;
     candidate.desc.remoteEndpoint = link.dstEndpointDesc;
     CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank, &candidate.desc.localEndpoint, ENDPOINT_ATTR_DIE_ID,
         sizeof(EndpointAttrDieId), &candidate.dieId));
     return HCCL_SUCCESS;
 }
 
 HcclResult CollectChannelCandidates(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
     const std::vector<uint32_t> &netLayers, RankChannelCandidates &candidates)
 {
     candidates = RankChannelCandidates{};
     candidates.peerRank = remoteRank;
     for (uint32_t netLayer : netLayers) {
         CommLink *links = nullptr;
         uint32_t linkNum = 0;
         CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &links, &linkNum));
         if (links == nullptr || linkNum == 0) {
             continue;
         }
 
         // EndpointInfo查询可能使RankGraph内部返回区失效，先复制整层链路。
         const std::vector<CommLink> layerLinks(links, links + linkNum);
         const auto primaryIt = std::find_if(layerLinks.begin(), layerLinks.end(), IsValidUbcLink);
         if (primaryIt != layerLinks.end()) {
             CHK_RET(MakeChannelCandidate(comm, myRank, remoteRank, *primaryIt, candidates.primary));
             return HCCL_SUCCESS;
         }
     }
 
     HCCL_ERROR("No UBC_CTP link from rank %u to rank %u", myRank, remoteRank);
     return HCCL_E_NOT_FOUND;
 }
 
 void AddChannelToGroups(std::vector<KernelChannelGroup> &groups, const AcquiredChannel &acquiredChannel)
 {
     auto groupIt
         = std::find_if(groups.begin(), groups.end(), [dieId = acquiredChannel.dieId](const KernelChannelGroup &group) {
               return group.dieId == dieId;
           });
     if (groupIt == groups.end()) {
         groups.push_back(KernelChannelGroup{});
         groupIt = std::prev(groups.end());
         groupIt->dieId = acquiredChannel.dieId;
     }
     groupIt->peerRanks.push_back(acquiredChannel.peerRank);
     groupIt->channels.push_back(acquiredChannel.channel);
 }
 
 void SortChannelGroups(std::vector<KernelChannelGroup> &groups)
 {
     std::sort(groups.begin(), groups.end(), [](const KernelChannelGroup &lhs, const KernelChannelGroup &rhs) {
         return lhs.dieId < rhs.dieId;
     });
 }
 
 void BuildParallel2x8Plan(const OpParam &param, const std::vector<AcquiredChannel> &acquiredChannels, ChannelPlan &plan)
 {
     if (param.rankSize != 2 * ALLGATHER_PARALLEL_LOCAL_RANK_NUM) {
         return;
     }
 
     // 官方2x8拓扑按Server连续编号：0~7、8~15；Clos子通信域由相同机内序号的两个Rank组成。
     const uint32_t localRankBase = param.myRank / ALLGATHER_PARALLEL_LOCAL_RANK_NUM * ALLGATHER_PARALLEL_LOCAL_RANK_NUM;
     const uint32_t counterpartRank = param.myRank ^ ALLGATHER_PARALLEL_LOCAL_RANK_NUM;
     plan.parallelIntraGroups.clear();
     plan.parallelInterGroups.clear();
     for (const AcquiredChannel &acquiredChannel : acquiredChannels) {
         if (acquiredChannel.peerRank >= localRankBase
             && acquiredChannel.peerRank < localRankBase + ALLGATHER_PARALLEL_LOCAL_RANK_NUM) {
             AddChannelToGroups(plan.parallelIntraGroups, acquiredChannel);
         } else if (acquiredChannel.peerRank == counterpartRank) {
             AddChannelToGroups(plan.parallelInterGroups, acquiredChannel);
         }
     }
     SortChannelGroups(plan.parallelIntraGroups);
     SortChannelGroups(plan.parallelInterGroups);
 
     plan.parallelLocalRanks.reserve(ALLGATHER_PARALLEL_LOCAL_RANK_NUM);
     for (uint32_t localRank = localRankBase; localRank < localRankBase + ALLGATHER_PARALLEL_LOCAL_RANK_NUM;
         ++localRank) {
         plan.parallelLocalRanks.push_back(localRank);
     }
     plan.parallel2x8Enabled = true;
     plan.parallelCounterpartRank = counterpartRank;
     plan.parallelInterChannelCount = 1;
     HCCL_INFO("Enabled 2x8 OmniPipe AllGather for rank %u: counterpart %u, intra kernels %zu", param.myRank,
         counterpartRank, plan.parallelIntraGroups.size());
 }
 
 bool IsAsym8x4LocalPeer(uint32_t rank, uint32_t peerRank)
 {
     if (rank < ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM) {
         return peerRank < ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM;
     }
     return peerRank >= ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM;
 }
 
 bool IsAsym8x4GatewayPeer(uint32_t rank, uint32_t peerRank)
 {
     if (rank < ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM) {
         return peerRank == ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM + rank / 2;
     }
     const uint32_t firstLargeRank = 2 * (rank - ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM);
     return peerRank == firstLargeRank || peerRank == firstLargeRank + 1;
 }
 
 void BuildAsym8x4Plan(const OpParam &param, const std::vector<AcquiredChannel> &acquiredChannels, ChannelPlan &plan)
 {
     if (param.rankSize != ALLGATHER_ASYM_8X4_RANK_NUM) {
         return;
     }
 
     for (const AcquiredChannel &acquiredChannel : acquiredChannels) {
         if (IsAsym8x4LocalPeer(param.myRank, acquiredChannel.peerRank)) {
             AddChannelToGroups(plan.parallelIntraGroups, acquiredChannel);
         } else {
             AddChannelToGroups(plan.parallelInterGroups, acquiredChannel);
             if (IsAsym8x4GatewayPeer(param.myRank, acquiredChannel.peerRank)) {
                 AddChannelToGroups(plan.asymGatewayInterGroups, acquiredChannel);
             }
         }
     }
     SortChannelGroups(plan.parallelIntraGroups);
     SortChannelGroups(plan.parallelInterGroups);
     SortChannelGroups(plan.asymGatewayInterGroups);
     plan.asym8x4Enabled = true;
     HCCL_INFO("Enabled asymmetric 8+4 OmniPipe AllGather for rank %u", param.myRank);
 }
 
 HcclResult AcquireChannelPlan(HcclComm comm, const OpParam &param, CommEngine engine, ChannelPlan &plan)
 {
     if (param.rankSize <= 1) {
         plan = ChannelPlan{};
         return HCCL_SUCCESS;
     }
 
     uint32_t *netLayerList = nullptr;
     uint32_t netLayerNum = 0;
     CHK_RET(HcclRankGraphGetLayers(comm, &netLayerList, &netLayerNum));
     CHK_PRT_RET(netLayerList == nullptr || netLayerNum == 0, HCCL_ERROR("Rank graph contains no network layer"),
         HCCL_E_NOT_FOUND);
 
     // 后续拓扑查询可能使库内返回指针失效，因此先复制网络层列表。
     std::vector<uint32_t> netLayers(netLayerList, netLayerList + netLayerNum);
     std::vector<RankChannelCandidates> rankCandidates;
     rankCandidates.reserve(param.rankSize - 1);
     for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
         if (remoteRank == param.myRank) {
             continue;
         }
         RankChannelCandidates candidates;
         CHK_RET(CollectChannelCandidates(comm, param.myRank, remoteRank, netLayers, candidates));
         rankCandidates.push_back(std::move(candidates));
     }
 
     const uint32_t channelNum = static_cast<uint32_t>(rankCandidates.size());
     std::vector<HcclChannelDesc> channelDescs;
     channelDescs.reserve(channelNum);
     for (const RankChannelCandidates &candidates : rankCandidates) {
         channelDescs.push_back(candidates.primary.desc);
     }
     std::vector<ChannelHandle> channels(channelNum);
     CHK_RET(HcclChannelAcquire(comm, engine, channelDescs.data(), channelNum, channels.data()));
 
     std::vector<AcquiredChannel> acquiredChannels;
     acquiredChannels.reserve(channelNum);
     plan = ChannelPlan{};
     for (uint32_t idx = 0; idx < channelNum; ++idx) {
         const RankChannelCandidates &candidates = rankCandidates[idx];
         AcquiredChannel acquiredChannel;
         acquiredChannel.peerRank = candidates.peerRank;
         acquiredChannel.dieId = candidates.primary.dieId;
         acquiredChannel.channel = channels[idx];
         acquiredChannels.push_back(acquiredChannel);
         AddChannelToGroups(plan.directGroups, acquiredChannel);
     }
 
     SortChannelGroups(plan.directGroups);
     CHK_PRT_RET(plan.directGroups.empty() || plan.directGroups.size() > MAX_CCU_DIE_NUM,
         HCCL_ERROR("Unsupported CCU die count %zu", plan.directGroups.size()), HCCL_E_NOT_SUPPORT);
     BuildParallel2x8Plan(param, acquiredChannels, plan);
     BuildAsym8x4Plan(param, acquiredChannels, plan);
     return HCCL_SUCCESS;
 }
 
 HcclResult RegisterCcuKernelVariant(CcuInsHandle insHandle, const char *kernelNamePrefix, void *kernelFunc,
     uint32_t groupDieId, const std::shared_ptr<ops_hccl::CcuKernelArgAllGather> &kernelArg,
     CcuKernelHandle &kernelHandle)
 {
     CcuKernelInfo kernelInfo;
     int nameRet = std::snprintf(
         kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%sDie%u", kernelNamePrefix, groupDieId);
     CHK_PRT_RET(nameRet <= 0 || static_cast<size_t>(nameRet) >= sizeof(kernelInfo.kernelFuncName),
         HCCL_ERROR("Failed to set CCU kernel name"), HCCL_E_INTERNAL);
     kernelInfo.kernelFunc = kernelFunc;
     kernelInfo.setKernelArg(kernelArg);
 
     constexpr uint32_t dieId = 0;
     constexpr uint32_t kernelArgNum = 1;
     const void *kernelArgs[] = {kernelInfo.kernelArg};
     CcuResult registerRet = HcommCcuKernelRegister(
         insHandle, dieId, kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
     if (registerRet != CCU_SUCCESS) {
         HCCL_ERROR("%s registration failed on die %u: %d", kernelNamePrefix, groupDieId, registerRet);
         return ConvertCcuToHccl(registerRet);
     }
     return HCCL_SUCCESS;
 }
 
 HcclResult RegisterCcuKernelFamily(CcuInsHandle insHandle, const char *kernelNamePrefix, void *kernelFunc,
     const std::vector<KernelChannelGroup> &channelGroups,
     const std::vector<std::shared_ptr<ops_hccl::CcuKernelArgAllGather>> &kernelArgs,
     std::vector<CcuKernelHandle> &kernelHandles)
 {
     CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
     for (size_t groupIdx = 0; groupIdx < channelGroups.size(); ++groupIdx) {
         HcclResult registerRet = RegisterCcuKernelVariant(insHandle, kernelNamePrefix, kernelFunc,
             channelGroups[groupIdx].dieId, kernelArgs[groupIdx], kernelHandles[groupIdx]);
         if (registerRet != HCCL_SUCCESS) {
             (void)HcommCcuKernelRegisterEnd(insHandle);
             return registerRet;
         }
     }
 
     CcuResult registerEndRet = HcommCcuKernelRegisterEnd(insHandle);
     if (registerEndRet != CCU_SUCCESS) {
         HCCL_ERROR("%s registration finalization failed: %d", kernelNamePrefix, registerEndRet);
         return ConvertCcuToHccl(registerEndRet);
     }
     return HCCL_SUCCESS;
 }
 
 HcclResult RegisterConcurrentCcuKernelFamilies(CcuInsHandle insHandle, const char *firstKernelNamePrefix,
     void *firstKernelFunc, const std::vector<KernelChannelGroup> &firstGroups,
     const std::vector<std::shared_ptr<ops_hccl::CcuKernelArgAllGather>> &firstKernelArgs,
     std::vector<CcuKernelHandle> &firstKernelHandles, const char *secondKernelNamePrefix, void *secondKernelFunc,
     const std::vector<KernelChannelGroup> &secondGroups,
     const std::vector<std::shared_ptr<ops_hccl::CcuKernelArgAllGather>> &secondKernelArgs,
     std::vector<CcuKernelHandle> &secondKernelHandles)
 {
     CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
     for (size_t groupIdx = 0; groupIdx < firstGroups.size(); ++groupIdx) {
         HcclResult registerRet = RegisterCcuKernelVariant(insHandle, firstKernelNamePrefix, firstKernelFunc,
             firstGroups[groupIdx].dieId, firstKernelArgs[groupIdx], firstKernelHandles[groupIdx]);
         if (registerRet != HCCL_SUCCESS) {
             (void)HcommCcuKernelRegisterEnd(insHandle);
             return registerRet;
         }
     }
     for (size_t groupIdx = 0; groupIdx < secondGroups.size(); ++groupIdx) {
         HcclResult registerRet = RegisterCcuKernelVariant(insHandle, secondKernelNamePrefix, secondKernelFunc,
             secondGroups[groupIdx].dieId, secondKernelArgs[groupIdx], secondKernelHandles[groupIdx]);
         if (registerRet != HCCL_SUCCESS) {
             (void)HcommCcuKernelRegisterEnd(insHandle);
             return registerRet;
         }
     }
 
     CcuResult registerEndRet = HcommCcuKernelRegisterEnd(insHandle);
     if (registerEndRet != CCU_SUCCESS) {
         HCCL_ERROR("Concurrent CCU kernel registration finalization failed: %d", registerEndRet);
         return ConvertCcuToHccl(registerEndRet);
     }
     return HCCL_SUCCESS;
 }
 
 HcclResult ValidateChannelGroups(const OpParam &param, const std::vector<KernelChannelGroup> &channelGroups,
     size_t maxGroupCount, size_t expectedChannelCount, const char *groupName)
 {
     size_t totalChannelCount = 0;
     uint16_t peerMask = 0;
     for (size_t groupIdx = 0; groupIdx < channelGroups.size(); ++groupIdx) {
         const KernelChannelGroup &group = channelGroups[groupIdx];
         CHK_PRT_RET(group.channels.empty() || group.channels.size() != group.peerRanks.size()
                         || group.channels.size() >= MAX_RANK_SIZE,
             HCCL_ERROR("Invalid %s channel group on die %u", groupName, group.dieId), HCCL_E_PARA);
         if (groupIdx != 0) {
             CHK_PRT_RET(channelGroups[groupIdx - 1].dieId == group.dieId,
                 HCCL_ERROR("Duplicate %s group on die %u", groupName, group.dieId), HCCL_E_PARA);
         }
         for (uint32_t peerRank : group.peerRanks) {
             CHK_PRT_RET(peerRank >= param.rankSize || peerRank == param.myRank,
                 HCCL_ERROR("Invalid %s peer rank %u", groupName, peerRank), HCCL_E_PARA);
             const uint16_t peerBit = static_cast<uint16_t>(uint32_t{1} << peerRank);
             CHK_PRT_RET(
                 (peerMask & peerBit) != 0, HCCL_ERROR("Duplicate %s peer rank %u", groupName, peerRank), HCCL_E_PARA);
             peerMask = static_cast<uint16_t>(peerMask | peerBit);
         }
         totalChannelCount += group.channels.size();
     }
     CHK_PRT_RET(
         channelGroups.empty() || channelGroups.size() > maxGroupCount || totalChannelCount != expectedChannelCount,
         HCCL_ERROR("Invalid %s groups: group count %zu, channel count %zu, expected %zu", groupName,
             channelGroups.size(), totalChannelCount, expectedChannelCount),
         HCCL_E_PARA);
     return HCCL_SUCCESS;
 }
 
 std::vector<std::shared_ptr<ops_hccl::CcuKernelArgAllGather>> BuildKernelArgs(const OpParam &param,
     const std::vector<KernelChannelGroup> &channelGroups, bool assignLocalCopy, uint32_t blockCount)
 {
     size_t localCopyGroup = channelGroups.size();
     if (assignLocalCopy) {
         localCopyGroup = 0;
         for (size_t groupIdx = 1; groupIdx < channelGroups.size(); ++groupIdx) {
             if (channelGroups[groupIdx].channels.size() < channelGroups[localCopyGroup].channels.size()) {
                 localCopyGroup = groupIdx;
             }
         }
     }
 
     std::vector<std::shared_ptr<ops_hccl::CcuKernelArgAllGather>> kernelArgs(channelGroups.size());
     for (size_t groupIdx = 0; groupIdx < channelGroups.size(); ++groupIdx) {
         const KernelChannelGroup &group = channelGroups[groupIdx];
         std::shared_ptr<ops_hccl::CcuKernelArgAllGather> &kernelArg = kernelArgs[groupIdx];
         kernelArg = std::make_shared<ops_hccl::CcuKernelArgAllGather>();
         kernelArg->rankSize = param.rankSize;
         kernelArg->rankId = param.myRank;
         kernelArg->handleLocalCopy = groupIdx == localCopyGroup ? 1U : 0U;
         kernelArg->blockCount = blockCount;
         kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
         for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
             kernelArg->channels[channelIdx] = group.channels[channelIdx];
             kernelArg->peerRanks[channelIdx] = group.peerRanks[channelIdx];
         }
     }
     return kernelArgs;
 }
 
 HcclResult RegisterCcuKernels(
     HcclComm comm, const OpParam &param, const ChannelPlan &channelPlan, AlgResourceCtx &resCtx)
 {
     CHK_RET(ValidateChannelGroups(param, channelPlan.directGroups, MAX_CCU_DIE_NUM, param.rankSize - 1, "direct"));
     CHK_PRT_RET(channelPlan.parallel2x8Enabled && channelPlan.asym8x4Enabled,
         HCCL_ERROR("Conflicting topology-specific AllGather plans"), HCCL_E_PARA);
     if (channelPlan.parallel2x8Enabled) {
         CHK_PRT_RET(channelPlan.parallelCounterpartRank >= param.rankSize || channelPlan.parallelInterChannelCount != 1
                         || channelPlan.parallelLocalRanks.size() != ALLGATHER_PARALLEL_LOCAL_RANK_NUM,
             HCCL_ERROR("Invalid 2x8 parallel rank metadata"), HCCL_E_PARA);
         CHK_RET(ValidateChannelGroups(param, channelPlan.parallelIntraGroups, MAX_CCU_DIE_NUM,
             ALLGATHER_PARALLEL_LOCAL_RANK_NUM - 1, "parallel intra"));
         CHK_RET(ValidateChannelGroups(param, channelPlan.parallelInterGroups, MAX_CCU_DIE_NUM,
             channelPlan.parallelInterChannelCount, "parallel inter"));
         for (const KernelChannelGroup &group : channelPlan.parallelInterGroups) {
             for (uint32_t peerRank : group.peerRanks) {
                 CHK_PRT_RET(peerRank != channelPlan.parallelCounterpartRank,
                     HCCL_ERROR(
                         "Invalid parallel peer rank %u, expected %u", peerRank, channelPlan.parallelCounterpartRank),
                     HCCL_E_PARA);
             }
         }
     }
     if (channelPlan.asym8x4Enabled) {
         const uint32_t localRankCount = param.myRank < ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM
                                             ? ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM
                                             : ALLGATHER_ASYM_SMALL_SERVER_RANK_NUM;
         const uint32_t gatewayPeerCount = param.myRank < ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM ? 1U : 2U;
         CHK_RET(ValidateChannelGroups(
             param, channelPlan.parallelIntraGroups, MAX_CCU_DIE_NUM, localRankCount - 1, "8+4 intra"));
         CHK_RET(ValidateChannelGroups(
             param, channelPlan.parallelInterGroups, MAX_CCU_DIE_NUM, param.rankSize - localRankCount, "8+4 inter"));
         CHK_RET(ValidateChannelGroups(
             param, channelPlan.asymGatewayInterGroups, MAX_CCU_DIE_NUM, gatewayPeerCount, "8+4 gateway inter"));
         for (const KernelChannelGroup &group : channelPlan.asymGatewayInterGroups) {
             for (uint32_t peerRank : group.peerRanks) {
                 CHK_PRT_RET(!IsAsym8x4GatewayPeer(param.myRank, peerRank),
                     HCCL_ERROR("Invalid 8+4 gateway peer rank %u", peerRank), HCCL_E_PARA);
             }
         }
     }
 
     CcuInsHandle insHandle{};
     uint32_t insNum = 0;
     CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
     CHK_PRT_RET(insNum != 1, HCCL_ERROR("Expected one CCU instance, got %u", insNum), HCCL_E_INTERNAL);
 
     const auto directKernelArgs = BuildKernelArgs(param, channelPlan.directGroups, true, 0);
     std::vector<CcuKernelHandle> kernelHandles(channelPlan.directGroups.size());
     std::vector<CcuKernelHandle> dualSliceKernelHandles(channelPlan.directGroups.size());
     std::vector<CcuKernelHandle> groupCopyKernelHandles;
 
     // 各类Kernel不会同时执行，分资源组注册以复用CCU资源；
     // 每个资源组内的双Die Kernel可以并行。
     CHK_RET(RegisterCcuKernelFamily(insHandle, "CcuAllGatherDirectKernel",
         reinterpret_cast<void *>(ops_hccl::CcuKernel), channelPlan.directGroups, directKernelArgs, kernelHandles));
     CHK_RET(RegisterCcuKernelFamily(insHandle, "CcuAllGatherDualSliceKernel",
         reinterpret_cast<void *>(ops_hccl::CcuDualSliceKernel), channelPlan.directGroups, directKernelArgs,
         dualSliceKernelHandles));
     if (param.rankSize == 4) {
         groupCopyKernelHandles.resize(channelPlan.directGroups.size());
         CHK_RET(RegisterCcuKernelFamily(insHandle, "CcuAllGatherGroupCopyKernel",
             reinterpret_cast<void *>(ops_hccl::CcuGroupCopyKernel), channelPlan.directGroups, directKernelArgs,
             groupCopyKernelHandles));
     }

     resCtx.ccuKernels = std::move(kernelHandles);
     resCtx.groupCopyCcuKernels = std::move(groupCopyKernelHandles);
     resCtx.dualSliceCcuKernels = std::move(dualSliceKernelHandles);
     if (channelPlan.asym8x4Enabled) {
         const uint32_t forwardBlockCount = param.myRank < ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM ? 1U : 2U;
         const auto stage0IntraKernelArgs = BuildKernelArgs(param, channelPlan.parallelIntraGroups, false, 0);
         const auto stage0InterKernelArgs = BuildKernelArgs(param, channelPlan.asymGatewayInterGroups, false, 0);
         const auto stage1IntraKernelArgs
             = BuildKernelArgs(param, channelPlan.parallelIntraGroups, true, forwardBlockCount);
         const auto stage1InterKernelArgs = BuildKernelArgs(param, channelPlan.parallelInterGroups, false, 0);
 
         std::vector<CcuKernelHandle> stage0IntraKernelHandles(channelPlan.parallelIntraGroups.size());
         std::vector<CcuKernelHandle> stage0InterKernelHandles(channelPlan.asymGatewayInterGroups.size());
         std::vector<CcuKernelHandle> stage1IntraKernelHandles(channelPlan.parallelIntraGroups.size());
         std::vector<CcuKernelHandle> stage1InterKernelHandles(channelPlan.parallelInterGroups.size());
 
         CHK_RET(RegisterConcurrentCcuKernelFamilies(insHandle, "CcuAllGatherAsym8x4Stage0Intra",
             reinterpret_cast<void *>(ops_hccl::CcuKernel), channelPlan.parallelIntraGroups, stage0IntraKernelArgs,
             stage0IntraKernelHandles, "CcuAllGatherAsym8x4Stage0Inter",
             reinterpret_cast<void *>(ops_hccl::CcuAsym8x4Stage0InterKernel), channelPlan.asymGatewayInterGroups,
             stage0InterKernelArgs, stage0InterKernelHandles));
         CHK_RET(RegisterConcurrentCcuKernelFamilies(insHandle, "CcuAllGatherAsym8x4Stage1Intra",
             reinterpret_cast<void *>(ops_hccl::CcuAsym8x4Stage1IntraKernel), channelPlan.parallelIntraGroups,
             stage1IntraKernelArgs, stage1IntraKernelHandles, "CcuAllGatherAsym8x4Stage1Inter",
             reinterpret_cast<void *>(ops_hccl::CcuKernel), channelPlan.parallelInterGroups, stage1InterKernelArgs,
             stage1InterKernelHandles));
 
         resCtx.asym8x4Enabled = 1;
         resCtx.parallelPhase1IntraCcuKernels = std::move(stage0IntraKernelHandles);
         resCtx.parallelPhase1InterCcuKernels = std::move(stage0InterKernelHandles);
         resCtx.parallelPhase2IntraCcuKernels = std::move(stage1IntraKernelHandles);
         resCtx.parallelPhase2InterCcuKernels = std::move(stage1InterKernelHandles);
         return HCCL_SUCCESS;
     }
     if (!channelPlan.parallel2x8Enabled) {
         return HCCL_SUCCESS;
     }
 
     const auto phase1IntraKernelArgs = BuildKernelArgs(param, channelPlan.parallelIntraGroups, false, 0);
     const auto phase1InterKernelArgs = BuildKernelArgs(param, channelPlan.parallelInterGroups, true, 0);
     const auto phase2IntraKernelArgs = BuildKernelArgs(param, channelPlan.parallelIntraGroups, false, 0);
     const auto phase2InterKernelArgs
         = BuildKernelArgs(param, channelPlan.parallelInterGroups, false, ALLGATHER_PARALLEL_LOCAL_RANK_NUM - 1);
 
     std::vector<CcuKernelHandle> phase1IntraKernelHandles(channelPlan.parallelIntraGroups.size());
     std::vector<CcuKernelHandle> phase1InterKernelHandles(channelPlan.parallelInterGroups.size());
     std::vector<CcuKernelHandle> phase2IntraKernelHandles(channelPlan.parallelIntraGroups.size());
     std::vector<CcuKernelHandle> phase2InterKernelHandles(channelPlan.parallelInterGroups.size());
 
     // Stage 0同时执行本Rank全量跨Server发送和首段Server内发送。
     CHK_RET(RegisterConcurrentCcuKernelFamilies(insHandle, "CcuAllGatherOmniStage0Intra",
         reinterpret_cast<void *>(ops_hccl::CcuKernel), channelPlan.parallelIntraGroups, phase1IntraKernelArgs,
         phase1IntraKernelHandles, "CcuAllGatherOmniStage0Inter", reinterpret_cast<void *>(ops_hccl::CcuKernel),
         channelPlan.parallelInterGroups, phase1InterKernelArgs, phase1InterKernelHandles));
     // 后续三个Stage复用同一资源组：慢轴发送一个连续片，快轴转发7个本Server数据块。
     CHK_RET(RegisterConcurrentCcuKernelFamilies(insHandle, "CcuAllGatherOmniPipelineIntra",
         reinterpret_cast<void *>(ops_hccl::CcuKernel), channelPlan.parallelIntraGroups, phase2IntraKernelArgs,
         phase2IntraKernelHandles, "CcuAllGatherOmniPipelineInter",
         reinterpret_cast<void *>(ops_hccl::CcuParallelRepeatKernel), channelPlan.parallelInterGroups,
         phase2InterKernelArgs, phase2InterKernelHandles));
 
     resCtx.parallel2x8Enabled = 1;
     resCtx.parallelCounterpartRank = channelPlan.parallelCounterpartRank;
     resCtx.parallelInterChannelCount = channelPlan.parallelInterChannelCount;
     resCtx.parallelLocalRanks = channelPlan.parallelLocalRanks;
     resCtx.parallelPhase1IntraCcuKernels = std::move(phase1IntraKernelHandles);
     resCtx.parallelPhase1InterCcuKernels = std::move(phase1InterKernelHandles);
     resCtx.parallelPhase2IntraCcuKernels = std::move(phase2IntraKernelHandles);
     resCtx.parallelPhase2InterCcuKernels = std::move(phase2InterKernelHandles);
     return HCCL_SUCCESS;
 }
 
 HcclResult CreateAlgResources(HcclComm comm, const OpParam &param, CommEngine engine, AlgResourceCtx &resCtx)
 {
     void *cclBufferAddr = nullptr;
     uint64_t cclBufferSize = 0;
     CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
     CHK_PTR_NULL(cclBufferAddr);
     resCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
     resCtx.ccuThread = param.cpuThread;
     resCtx.threads = {param.cpuThread};
 
     if (param.rankSize == 1) {
         return HCCL_SUCCESS;
     }
 
     ChannelPlan channelPlan;
     CHK_RET(AcquireChannelPlan(comm, param, engine, channelPlan));
     size_t maxConcurrentKernelCount = channelPlan.directGroups.size();
     if (channelPlan.parallel2x8Enabled) {
         maxConcurrentKernelCount = std::max(
             maxConcurrentKernelCount, channelPlan.parallelIntraGroups.size() + channelPlan.parallelInterGroups.size());
     }
     if (channelPlan.asym8x4Enabled) {
         maxConcurrentKernelCount = std::max(maxConcurrentKernelCount,
             channelPlan.parallelIntraGroups.size() + channelPlan.asymGatewayInterGroups.size());
         maxConcurrentKernelCount = std::max(
             maxConcurrentKernelCount, channelPlan.parallelIntraGroups.size() + channelPlan.parallelInterGroups.size());
     }
     CHK_PRT_RET(maxConcurrentKernelCount == 0 || maxConcurrentKernelCount > ALLGATHER_PARALLEL_MAX_KERNEL_NUM,
         HCCL_ERROR("Invalid concurrent CCU kernel count %zu", maxConcurrentKernelCount), HCCL_E_NOT_SUPPORT);
 
     for (size_t threadIdx = 1; threadIdx < maxConcurrentKernelCount; ++threadIdx) {
         ThreadHandle slaveThread{};
         CHK_RET(HcclThreadAcquire(comm, engine, 1, SLAVE_THREAD_NOTIFY_NUM, &slaveThread));
         resCtx.threads.push_back(slaveThread);
     }
     CHK_RET(RegisterCcuKernels(comm, param, channelPlan, resCtx));
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
     int tagRet = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allgather");
     CHK_PRT_RET(tagRet <= 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag),
         HCCL_ERROR("Failed to set operator tag"), HCCL_E_INTERNAL);
     param.inputPtr = sendBuf;
     param.outputPtr = recvBuf;
     param.count = sendCount;
     param.dataType = dataType;
     param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
 
     // 注册算子信息
     HcclDfxOpInfo dfxInfo{};
     char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
     CHK_RET(HcclGetCommName(comm, commName));
     CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
 
     // ==============================================
     // STEP 1: 解析拓扑信息
     // ==============================================
     CHK_RET(HcclGetRankId(comm, &param.myRank));
     CHK_RET(HcclGetRankSize(comm, &param.rankSize));
     CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
         HCCL_ERROR("Invalid rank %u or rank size %u", param.myRank, param.rankSize), HCCL_E_PARA);
 
     auto typeSize = SIZE_TABLE.find(dataType);
     CHK_PRT_RET(typeSize == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type %u", static_cast<uint32_t>(dataType)),
         HCCL_E_NOT_SUPPORT);
     CHK_PRT_RET(sendCount > std::numeric_limits<uint64_t>::max() / typeSize->second, HCCL_ERROR("Input size overflow"),
         HCCL_E_PARA);
     uint64_t inputSize = sendCount * typeSize->second;
     if (inputSize == 0) {
         return HCCL_SUCCESS;
     }
     CHK_PRT_RET(inputSize > std::numeric_limits<uint64_t>::max() / param.rankSize, HCCL_ERROR("Output size overflow"),
         HCCL_E_PARA);
 
     // ==============================================
     // STEP 2: 创建资源
     // ==============================================
     CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
 
     // ==============================================
     // STEP 2.1: 申请用于 Host/Device 同步的通信资源
     // ==============================================
     // 当前调用的stream每次都转换为thread，Kernel句柄和Channel则按通信域缓存复用。
     const bool needsParallelThreads
         = param.rankSize == 2 * ALLGATHER_PARALLEL_LOCAL_RANK_NUM || param.rankSize == ALLGATHER_ASYM_8X4_RANK_NUM;
     const uint32_t mainThreadNotifyNum
         = needsParallelThreads ? MAIN_THREAD_NOTIFY_NUM : (param.rankSize > 1 ? SLAVE_THREAD_NOTIFY_NUM : 0);
     CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, mainThreadNotifyNum, &param.cpuThread));
 
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
         CHK_RET(CreateAlgResources(comm, param, ccuEngine, resCtxHost));
 
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
