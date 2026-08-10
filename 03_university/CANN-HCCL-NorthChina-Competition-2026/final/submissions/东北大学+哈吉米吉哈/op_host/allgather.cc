/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_launch.h>

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
// 每个 channel 上使用的 notify 数量（与 kernel 内使用的 notify bit 位数匹配）
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t MAX_CCU_DIE_NUM = 2;
constexpr const char *CCU_KERNEL_FUNC_NAME = "CcuKernel";
constexpr const char *CCU_SMALL_READ_KERNEL_FUNC_NAME = "CcuKernelSmallRead";

// dieId -> [(channel, remoteRank)]，组内按 remoteRank 升序；一个 CCU kernel 的所有 channel 必须在同一个 die 上
using DieChannelGroups = std::map<uint32_t, std::vector<std::pair<ChannelHandle, uint32_t>>>;

// 2×8 的 rank 按两台 Server 连续编号：同一行 8 个 rank 走 Mesh，另一行 8 个 rank 走 Clos。
// 额外检查每个 die 分组的成员，避免在其他 16 卡拓扑上误启用专用算法。
uint32_t Detect2x8Axis(const OpParam &param,
                       const std::vector<std::pair<ChannelHandle, uint32_t>> &channels)
{
    constexpr uint32_t ranksPerServer = 8;
    if (param.rankSize != 2 * ranksPerServer) {
        return CUSTOM_AXIS_NONE;
    }

    if (channels.size() == ranksPerServer - 1) {
        for (const auto &channel : channels) {
            if (channel.second / ranksPerServer != param.myRank / ranksPerServer) {
                return CUSTOM_AXIS_NONE;
            }
        }
        return CUSTOM_AXIS_MESH;
    }

    if (channels.size() == ranksPerServer) {
        for (const auto &channel : channels) {
            if (channel.second / ranksPerServer == param.myRank / ranksPerServer) {
                return CUSTOM_AXIS_NONE;
            }
        }
        return CUSTOM_AXIS_CLOS;
    }

    return CUSTOM_AXIS_NONE;
}

// 8+4 的两台 Server 分别为 ranks 0~7 和 8~11。每个 rank 的 Mesh
// channel 只连接本 Server，Clos channel 连接另一台 Server。
uint32_t Detect8x4Axis(const OpParam &param,
                       const std::vector<std::pair<ChannelHandle, uint32_t>> &channels)
{
    constexpr uint32_t largeServerSize = 8;
    constexpr uint32_t rankSize = 12;
    if (param.rankSize != rankSize) {
        return CUSTOM_AXIS_NONE;
    }

    const bool inLargeServer = param.myRank < largeServerSize;
    const uint32_t meshChannelCount = inLargeServer ? 7U : 3U;
    const uint32_t closChannelCount = inLargeServer ? 4U : 8U;

    if (channels.size() == meshChannelCount) {
        for (const auto &channel : channels) {
            if ((channel.second < largeServerSize) != inLargeServer) {
                return CUSTOM_AXIS_NONE;
            }
        }
        return CUSTOM_AXIS_MESH;
    }

    if (channels.size() == closChannelCount) {
        for (const auto &channel : channels) {
            if ((channel.second < largeServerSize) == inLargeServer) {
                return CUSTOM_AXIS_NONE;
            }
        }
        return CUSTOM_AXIS_CLOS;
    }

    return CUSTOM_AXIS_NONE;
}

HcclResult GetNetworkLayers(HcclComm comm, std::vector<uint32_t> &netLayers)
{
    uint32_t *netLayersRaw = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayersRaw, &layerNum));
    CHK_PRT_RET(netLayersRaw == nullptr || layerNum == 0,
        HCCL_ERROR("[GetChannelForCcu] no netLayer found"),
        HCCL_E_NOT_FOUND);
    netLayers.assign(netLayersRaw, netLayersRaw + layerNum);
    return HCCL_SUCCESS;
}

HcclResult FindPeerLinks(HcclComm comm, const OpParam &param,
                         const std::vector<uint32_t> &netLayers,
                         uint32_t remoteRank, CommLink *&links,
                         uint32_t &linkNum)
{
    links = nullptr;
    linkNum = 0;
    for (uint32_t netLayer : netLayers) {
        if (HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank,
                                  &links, &linkNum) == HCCL_SUCCESS &&
            links != nullptr && linkNum > 0) {
            return HCCL_SUCCESS;
        }
        links = nullptr;
        linkNum = 0;
    }
    HCCL_ERROR(
        "[GetChannelForCcu] no link found between rank %u and rank %u on any netLayer",
        param.myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

const CommLink &SelectPeerLink(const CommLink *links, uint32_t linkNum)
{
    for (uint32_t linkIndex = 0; linkIndex < linkNum; linkIndex++) {
        if (links[linkIndex].linkAttr.linkProtocol ==
            CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            return links[linkIndex];
        }
    }
    return links[0];
}

HcclResult BuildChannelDesc(uint32_t remoteRank, const CommLink &link,
                            HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult AcquirePeerChannel(
    HcclComm comm, const OpParam &param, CommEngine ccuEngine,
    const std::vector<uint32_t> &netLayers, uint32_t remoteRank,
    DieChannelGroups &dieChannelGroups)
{
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(FindPeerLinks(
        comm, param, netLayers, remoteRank, links, linkNum));
    HcclChannelDesc desc;
    CHK_RET(BuildChannelDesc(
        remoteRank, SelectPeerLink(links, linkNum), desc));
    EndpointAttrDieId dieId{};
    CHK_RET(HcclRankGraphGetEndpointInfo(
        comm, param.myRank, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID,
        sizeof(EndpointAttrDieId), &dieId));
    ChannelHandle channel{0};
    CHK_RET(HcclChannelAcquire(comm, ccuEngine, &desc, 1, &channel));
    dieChannelGroups[dieId].emplace_back(channel, remoteRank);
    return HCCL_SUCCESS;
}

// 为每个对端 rank 申请一条 channel（跳过自己，按 remoteRank 升序处理），并按 channel 所在 die 分组
HcclResult GetChannelForCcu(HcclComm comm, const OpParam &param, CommEngine ccuEngine,
                            DieChannelGroups &dieChannelGroups)
{
    std::vector<uint32_t> netLayers;
    CHK_RET(GetNetworkLayers(comm, netLayers));
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CHK_RET(AcquirePeerChannel(
            comm, param, ccuEngine, netLayers, remoteRank, dieChannelGroups));
    }
    return HCCL_SUCCESS;
}

struct AxisPlan {
    std::vector<uint32_t> axisIds;
    bool enable2x8Axis = false;
    bool enableSpecialAxis = false;
};

AxisPlan BuildAxisPlan(const OpParam &param,
                       const DieChannelGroups &dieChannelGroups)
{
    AxisPlan plan;
    plan.axisIds.reserve(dieChannelGroups.size());
    bool hasMeshAxis = false;
    bool hasClosAxis = false;
    for (const auto &group : dieChannelGroups) {
        uint32_t axisId = Detect2x8Axis(param, group.second);
        if (axisId == CUSTOM_AXIS_NONE) {
            axisId = Detect8x4Axis(param, group.second);
        }
        plan.axisIds.push_back(axisId);
        hasMeshAxis = hasMeshAxis || axisId == CUSTOM_AXIS_MESH;
        hasClosAxis = hasClosAxis || axisId == CUSTOM_AXIS_CLOS;
    }
    const bool hasTwoAxes =
        dieChannelGroups.size() == MAX_CCU_DIE_NUM &&
        hasMeshAxis && hasClosAxis;
    plan.enable2x8Axis = param.rankSize == 16 && hasTwoAxes;
    plan.enableSpecialAxis =
        plan.enable2x8Axis || (param.rankSize == 12 && hasTwoAxes);
    return plan;
}

uint32_t FindSelfCopyGroup(const DieChannelGroups &dieChannelGroups)
{
    uint32_t bestGroupIndex = 0;
    uint32_t groupIndex = 0;
    size_t minChannelNum = dieChannelGroups.begin()->second.size();
    for (const auto &group : dieChannelGroups) {
        if (group.second.size() < minChannelNum) {
            minChannelNum = group.second.size();
            bestGroupIndex = groupIndex;
        }
        groupIndex++;
    }
    return bestGroupIndex;
}

std::shared_ptr<CcuKernelArgAllGatherMesh1DMem2Mem> BuildKernelArgument(
    const OpParam &param,
    const std::vector<std::pair<ChannelHandle, uint32_t>> &channels,
    const AxisPlan &axisPlan,
    uint32_t groupIndex,
    uint32_t selfCopyGroupIndex)
{
    auto kernelArg =
        std::make_shared<CcuKernelArgAllGatherMesh1DMem2Mem>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->ifHandleSelfRank =
        groupIndex == selfCopyGroupIndex ? 1U : 0U;
    kernelArg->axisId = axisPlan.enableSpecialAxis
        ? axisPlan.axisIds[groupIndex] : CUSTOM_AXIS_NONE;
    kernelArg->pairedRankId = axisPlan.enable2x8Axis
        ? ((param.myRank + 8U) % 16U) : param.myRank;
    uint32_t channelIndex = 0;
    for (const auto &channelInfo : channels) {
        kernelArg->channels[channelIndex] = channelInfo.first;
        kernelArg->rankIds[channelIndex] = channelInfo.second;
        channelIndex++;
    }
    kernelArg->channelCount = channelIndex;
    return kernelArg;
}

HcclResult FillKernelInfo(
    const std::shared_ptr<CcuKernelArgAllGatherMesh1DMem2Mem> &kernelArg,
    CcuKernelInfo &kernelInfo)
{
    int ret = sprintf_s(
        kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
        "%s", CCU_KERNEL_FUNC_NAME);
    CHK_PRT_RET(
        ret < 0 || static_cast<size_t>(ret) >= sizeof(kernelInfo.kernelFuncName),
        HCCL_ERROR("[RegisterCcuKernels] failed to fill kernelFuncName"),
        HCCL_E_INTERNAL);
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);
    kernelInfo.setKernelArg(kernelArg);
    return HCCL_SUCCESS;
}

HcclResult RegisterKernelPair(
    CcuInsHandle insHandle,
    const std::shared_ptr<CcuKernelArgAllGatherMesh1DMem2Mem> &kernelArg,
    AlgResourceCtx &resCtxHost,
    uint32_t groupIndex)
{
    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    CcuKernelInfo kernelInfo{};
    CHK_RET(FillKernelInfo(kernelArg, kernelInfo));
    resCtxHost.kernelArgs[groupIndex] = *kernelArg;
    const void *registerArgs[] = {kernelInfo.kernelArg};
    CcuKernelHandle kernelHandle{0};
    CHK_RET_CCU(HcommCcuKernelRegister(
        insHandle, dieId, kernelInfo.kernelFuncName, kernelInfo.kernelFunc,
        registerArgs, kernelArgNum, &kernelHandle));
    resCtxHost.ccuKernels[groupIndex] = kernelHandle;
    CcuKernelHandle smallReadKernelHandle{0};
    CHK_RET_CCU(HcommCcuKernelRegister(
        insHandle, dieId, CCU_SMALL_READ_KERNEL_FUNC_NAME,
        reinterpret_cast<void *>(ops_hccl::CcuKernelSmallRead),
        registerArgs, kernelArgNum, &smallReadKernelHandle));
    resCtxHost.smallReadCcuKernels[groupIndex] = smallReadKernelHandle;
    return HCCL_SUCCESS;
}

// 按 die 分组注册 CCU Kernel（同一个 kernel 函数，每组一个 kernelArg），kernelArg 含 rankSize/rankId/channels/rankIds
HcclResult RegisterCcuKernels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost,
                              const DieChannelGroups &dieChannelGroups)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[RegisterCcuKernels] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
        HCCL_E_INTERNAL);

    const uint32_t groupNum = static_cast<uint32_t>(dieChannelGroups.size());
    CHK_PRT_RET(groupNum == 0 || groupNum > MAX_CCU_DIE_NUM,
        HCCL_ERROR("[RegisterCcuKernels] invalid die group num [%u]", groupNum),
        HCCL_E_INTERNAL);
    resCtxHost.ccuKernels.resize(groupNum);
    resCtxHost.smallReadCcuKernels.resize(groupNum);
    resCtxHost.kernelArgs.resize(groupNum);

    const AxisPlan axisPlan = BuildAxisPlan(param, dieChannelGroups);
    const uint32_t selfCopyGroupIndex =
        FindSelfCopyGroup(dieChannelGroups);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    uint32_t groupIndex = 0;
    for (const auto &group : dieChannelGroups) {
        auto kernelArg = BuildKernelArgument(
            param, group.second, axisPlan, groupIndex, selfCopyGroupIndex);
        CHK_RET(RegisterKernelPair(
            insHandle, kernelArg, resCtxHost, groupIndex));
        groupIndex++;
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult InitializeOpParam(
    void *sendBuf, void *recvBuf, uint64_t sendCount,
    HcclDataType dataType, HcclComm comm, OpParam &param)
{
    int32_t tagLength = sprintf_s(
        param.tag, sizeof(param.tag), "%s", "hccl_custom_allgather");
    CHK_PRT_RET(
        tagLength < 0 ||
            static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("[HcclAllGather] failed to fill operator tag"),
        HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    return HCCL_SUCCESS;
}

HcclResult RegisterDfxInfo(HcclComm comm, HcclDfxOpInfo &dfxInfo)
{
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    return HcclDfxRegOpInfoByCommId(
        commName, reinterpret_cast<void *>(&dfxInfo));
}

HcclResult BuildMultiRankResources(
    HcclComm comm, const OpParam &param, CommEngine ccuEngine,
    AlgResourceCtx &resCtxHost)
{
    DieChannelGroups dieChannelGroups;
    CHK_RET(GetChannelForCcu(
        comm, param, ccuEngine, dieChannelGroups));
    const uint32_t threadNum =
        static_cast<uint32_t>(dieChannelGroups.size());
    CHK_PRT_RET(
        threadNum == 0 || threadNum > MAX_CCU_DIE_NUM,
        HCCL_ERROR(
            "[HcclAllGather] invalid CCU thread num [%u]", threadNum),
        HCCL_E_INTERNAL);
    resCtxHost.threads.resize(threadNum);
    if (threadNum > 1) {
        constexpr uint32_t notifyNumPerThread = 1;
        CHK_RET(HcclThreadAcquire(
            comm, ccuEngine, threadNum - 1, notifyNumPerThread,
            &resCtxHost.threads[1]));
    }
    return RegisterCcuKernels(
        comm, param, resCtxHost, dieChannelGroups);
}

HcclResult StoreResourceContext(
    HcclComm comm, CommEngine ccuEngine,
    AlgResourceCtx &resCtxHost, OpParam &param)
{
    std::vector<char> sequence = resCtxHost.Serialize();
    param.ctxSize = sequence.size();
    CHK_RET(HcclEngineCtxCreate(
        comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
    return HcclEngineCtxCopy(
        comm, ccuEngine, param.tag, sequence.data(), sequence.size(), 0);
}

HcclResult BuildResourceContext(
    HcclComm comm, CommEngine ccuEngine, OpParam &param)
{
    AlgResourceCtx resCtxHost;
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(
        comm, &cclBufferAddr, &cclBufferSize));
    resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    resCtxHost.threads.resize(1);
    resCtxHost.threads[0] = param.cpuThread;
    resCtxHost.ccuThread = param.cpuThread;
    if (param.rankSize > 1) {
        CHK_RET(BuildMultiRankResources(
            comm, param, ccuEngine, resCtxHost));
    }
    return StoreResourceContext(
        comm, ccuEngine, resCtxHost, param);
}

HcclResult PrepareExecutionContext(
    HcclComm comm, aclrtStream stream, OpParam &param)
{
    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, ccuEngine, stream, 1, &param.cpuThread));
    void *context = nullptr;
    uint64_t contextSize = 0;
    if (HcclEngineCtxGet(
            comm, param.tag, ccuEngine, &context, &contextSize) ==
        HCCL_SUCCESS) {
        HCCL_INFO("Engine context already exists");
        param.resCtx = context;
        param.ctxSize = contextSize;
        return HCCL_SUCCESS;
    }
    return BuildResourceContext(comm, ccuEngine, param);
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    OpParam param{};
    CHK_RET(InitializeOpParam(
        sendBuf, recvBuf, sendCount, dataType, comm, param));
    HcclDfxOpInfo dfxInfo;
    CHK_RET(RegisterDfxInfo(comm, dfxInfo));
    CHK_RET(PrepareExecutionContext(comm, stream, param));
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
