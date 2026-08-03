/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ===========================================================================
// ReduceScatter 算子 —— Host 侧（控制面）
//
// 职责：
//   1) 校验入参并构造 OpParam
//   2) 申请通信资源：Thread（Host/Device 同步 + CCU Kernel 执行）、
//      Channel（全连接 Mesh，每对端一条）
//   3) 注册 CCU Kernel
//   4) 序列化资源上下文到通信引擎上下文（按 tag 缓存，重复调用复用）
//   5) 调用 ExecOp 完成数据分片与 Kernel 下发
//
// Channel 申请策略：
//   - 全连接（Full-Mesh）：对其余每个 rank 各申请一条 Channel
//   - 选路规则：layer-0（Server 内 Full-Mesh）优先，layer-1（跨 Server Clos）兜底
//   - 协议过滤：CCU 引擎要求 UBC_CTP 协议
//   - 分组规则：按本地端点真实 IO die 分组，不能把 netLayer 等同于 die
//
// 多拓扑兼容：
//   - 拓扑无关的 Mesh 1D Read + LocalReduce 算法适用于 2*8 / 4*1 / 8+4
//     所有三种子拓扑，无需拓扑感知代码
// ===========================================================================

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {

// 每条 Channel 申请的 Notify 数量：
// RS_INPUT_XN(0) + RS_TOKEN_XN(1) + RS_POST_SYNC(2) + 1 预留
constexpr uint32_t CHANNEL_NOTIFY_NUM = 4;

// ===========================================================================
// AcquireDescAnyLayer —— 为 srcRank -> dstRank 填充 Channel 描述符
//
// 选路：先查 layer-0（Server 内 Full-Mesh），若无则查 layer-1（跨 Server Clos）
// 协议：过滤出 UBC_CTP（CCU 引擎要求）
// die：查询本地端点真实 IO die；netLayer 只表示网络层，不能作为 CCU die 分组依据
// ===========================================================================
HcclResult AcquireDescAnyLayer(
    HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc *desc, uint32_t &outLayer, uint32_t &outDie)
{
    uint32_t listSize = 0;
    CommLink *linkList = nullptr;
    uint32_t netLayer = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize));
    if (listSize == 0) {
        netLayer = 1;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize));
    }
    CHK_PRT_RET(listSize == 0,
        HCCL_ERROR("AcquireDescAnyLayer: no link between rank[%u] and rank[%u]", srcRank, dstRank), HCCL_E_INTERNAL);

    // 查找 UBC_CTP 链路（CCU 引擎必需）
    CommLink *selectedLink = nullptr;
    for (uint32_t i = 0; i < listSize; i++) {
        if (linkList[i].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            selectedLink = &linkList[i];
            break;
        }
    }
    CHK_PRT_RET(selectedLink == nullptr,
        HCCL_ERROR("AcquireDescAnyLayer: no UBC_CTP link between rank[%u] and rank[%u]", srcRank, dstRank),
        HCCL_E_INTERNAL);

    CHK_RET(HcclChannelDescInit(desc, 1));
    desc->remoteRank = dstRank;
    desc->notifyNum = CHANNEL_NOTIFY_NUM;
    desc->channelProtocol = selectedLink->linkAttr.linkProtocol;
    desc->localEndpoint.protocol = selectedLink->srcEndpointDesc.protocol;
    desc->localEndpoint.commAddr = selectedLink->srcEndpointDesc.commAddr;
    desc->localEndpoint.loc = selectedLink->srcEndpointDesc.loc;
    desc->remoteEndpoint.protocol = selectedLink->dstEndpointDesc.protocol;
    desc->remoteEndpoint.commAddr = selectedLink->dstEndpointDesc.commAddr;
    desc->remoteEndpoint.loc = selectedLink->dstEndpointDesc.loc;

    EndpointAttrDieId dieId = netLayer;
    HcclResult dieRet = HcclRankGraphGetEndpointInfo(
        comm, srcRank, &selectedLink->srcEndpointDesc, ENDPOINT_ATTR_DIE_ID, sizeof(EndpointAttrDieId), &dieId);
    if (dieRet != HCCL_SUCCESS) {
        dieId = netLayer;
    }

    outLayer = netLayer;
    outDie = (dieId < 2) ? dieId : netLayer;
    return HCCL_SUCCESS;
}

// ===========================================================================
// AcquireChannelsFullMesh —— 对其余每个 rank 申请一条 Channel，记录所在层
// ===========================================================================
HcclResult AcquireChannelsFullMesh(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    uint32_t channelNum = param.rankSize - 1;
    std::vector<HcclChannelDesc> desc(channelNum);
    std::vector<ChannelHandle> handles(channelNum);
    // 记录每个 channel 的 network layer 与真实 IO die
    std::vector<uint32_t> chLayer(channelNum, 0);
    std::vector<uint32_t> chDie(channelNum, 0);
    uint32_t idx = 0;
    for (uint32_t j = 0; j < param.rankSize; j++) {
        if (j == param.myRank) {
            continue;
        }
        CHK_RET(AcquireDescAnyLayer(comm, param.myRank, j, &desc[idx], chLayer[idx], chDie[idx]));
        idx++;
    }
    // 批量申请 Channel
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclChannelAcquire(comm, ccuEngine, desc.data(), channelNum, handles.data()));

    // 记录 Channel 元数据
    for (uint32_t i = 0; i < channelNum; i++) {
        ChannelInfo channel;
        channel.remoteRank = desc[i].remoteRank;
        channel.handle = handles[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.netLayer = chLayer[i];
        channel.localDie = chDie[i];
        resCtxHost.channels.push_back(channel);
    }

    // 按真实 IO die 分组到 dieChannels[0] 和 dieChannels[1]
    for (const auto &ch : resCtxHost.channels) {
        resCtxHost.dieChannels[ch.localDie].push_back(ch);
    }

    return HCCL_SUCCESS;
}

// ===========================================================================
// RegisterCcuKernelPerDieRH —— RH 专用注册，增加 stepOffset
// ===========================================================================
static HcclResult RegisterCcuKernelPerDieRH(HcclComm comm, const OpParam &param, const std::vector<ChannelInfo> &dieChs,
    CcuKernelHandle &kernelHandle, bool initOutput, uint32_t stepOffset, bool useSliced, uint32_t dieId)
{
    if (dieChs.empty())
        return HCCL_SUCCESS;

    auto kernelArg = std::make_shared<ReduceScatterKernelArgHost>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    kernelArg->channelCount = static_cast<uint32_t>(dieChs.size());
    kernelArg->initOutput = initOutput;
    kernelArg->reduceToOutput = false;
    kernelArg->skipOutput = false;
    kernelArg->stepOffset = stepOffset;
    for (uint32_t i = 0; i < kernelArg->channelCount; i++) {
        kernelArg->channels[i] = dieChs[i].handle;
    }

    CcuKernelInfo kernelInfo;
    snprintf(
        kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), useSliced ? "CcuKernelRHSliced" : "CcuKernelRH");
    kernelInfo.kernelFunc = useSliced ? reinterpret_cast<void *>(ops_hccl::CcuKernelRHSliced)
                                      : reinterpret_cast<void *>(ops_hccl::CcuKernelRH);
    kernelInfo.setKernelArg(kernelArg);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("RegisterCcuKernelPerDieRH: HcclCommQueryCcuIns failed, insNum[%u]", insNum),
        HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("HcommCcuKernelRegisterStart failed, ccuRet[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);

    const void *kernelArgs[1] = {kernelInfo.kernelArg};
    ccuRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
        reinterpret_cast<const void *>(kernelInfo.kernelFunc), kernelArgs, 1, &kernelHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("HcommCcuKernelRegister failed, ccuRet[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("HcommCcuKernelRegisterEnd failed, ccuRet[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

// ===========================================================================
// RegisterCcuKernelPerDie —— 为一个 die 上的 channel 集合注册一个 CCU Kernel（Mesh）
// ===========================================================================
static HcclResult RegisterCcuKernelPerDie(HcclComm comm, const OpParam &param, const std::vector<ChannelInfo> &dieChs,
    CcuKernelHandle &kernelHandle, bool includeSelf, bool reduceToOutput, bool skipOutput, uint32_t dieId,
    bool usePipelineKernel)
{
    if (dieChs.empty()) {
        return HCCL_SUCCESS;
    }

    auto kernelArg = std::make_shared<ReduceScatterKernelArgHost>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    kernelArg->channelCount = static_cast<uint32_t>(dieChs.size());
    kernelArg->initOutput = includeSelf;
    kernelArg->reduceToOutput = reduceToOutput;
    kernelArg->skipOutput = skipOutput;
    kernelArg->stepOffset = 0;
    for (uint32_t i = 0; i < kernelArg->channelCount; i++) {
        kernelArg->channels[i] = dieChs[i].handle;
    }

    CcuKernelInfo kernelInfo;
    snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
        usePipelineKernel ? "CcuKernelPipeline" : "CcuKernel");
    kernelInfo.kernelFunc = usePipelineKernel ? reinterpret_cast<void *>(ops_hccl::CcuKernelPipeline)
                                              : reinterpret_cast<void *>(ops_hccl::CcuKernel);
    kernelInfo.setKernelArg(kernelArg);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("RegisterCcuKernelPerDie: HcclCommQueryCcuIns failed, insNum[%u]", insNum),
        HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("HcommCcuKernelRegisterStart failed, ccuRet[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);

    const void *kernelArgs[1] = {kernelInfo.kernelArg};
    ccuRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
        reinterpret_cast<const void *>(kernelInfo.kernelFunc), kernelArgs, 1, &kernelHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("HcommCcuKernelRegister failed, ccuRet[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("HcommCcuKernelRegisterEnd failed, ccuRet[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

static HcclResult RegisterSmall4x1Kernel(HcclComm comm, const OpParam &param,
    const std::vector<ChannelInfo> &dieChs, CcuKernelHandle &kernelHandle, uint32_t dieId)
{
    CHK_PRT_RET(dieChs.size() != 3U,
        HCCL_ERROR("RegisterSmall4x1Kernel: invalid channel count[%llu]",
            static_cast<unsigned long long>(dieChs.size())),
        HCCL_E_INTERNAL);

    auto kernelArg = std::make_shared<ReduceScatterKernelArgHost>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    kernelArg->channelCount = static_cast<uint32_t>(dieChs.size());
    kernelArg->initOutput = true;
    kernelArg->reduceToOutput = false;
    kernelArg->skipOutput = false;
    kernelArg->stepOffset = 0;
    for (uint32_t i = 0; i < kernelArg->channelCount; i++) {
        kernelArg->channels[i] = dieChs[i].handle;
    }

    CcuKernelInfo kernelInfo;
    snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelSmall4x1");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernelSmall4x1);
    kernelInfo.setKernelArg(kernelArg);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("RegisterSmall4x1Kernel: invalid CCU insNum[%u]", insNum), HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("RegisterSmall4x1Kernel: register start failed, ret[%d]", static_cast<int>(ccuRet)),
        HCCL_E_INTERNAL);

    const void *kernelArgs[1] = {kernelInfo.kernelArg};
    ccuRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
        reinterpret_cast<const void *>(kernelInfo.kernelFunc), kernelArgs, 1, &kernelHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("RegisterSmall4x1Kernel: register failed, ret[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("RegisterSmall4x1Kernel: register end failed, ret[%d]", static_cast<int>(ccuRet)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

static HcclResult RegisterCcuKernelMergeScratch(HcclComm comm, const OpParam &param,
    const std::vector<ChannelInfo> &dieChs, CcuKernelHandle &kernelHandle, uint32_t dieId, bool postSyncAllChannels)
{
    CHK_PRT_RET(dieChs.empty(), HCCL_ERROR("RegisterCcuKernelMergeScratch: die[%u] has no anchor channel", dieId),
        HCCL_E_INTERNAL);

    auto kernelArg = std::make_shared<ReduceScatterKernelArgHost>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    kernelArg->channelCount = postSyncAllChannels ? static_cast<uint32_t>(dieChs.size()) : 1U;
    for (uint32_t i = 0; i < kernelArg->channelCount; i++) {
        kernelArg->channels[i] = dieChs[i].handle;
    }
    kernelArg->initOutput = false;
    kernelArg->reduceToOutput = false;
    kernelArg->skipOutput = false;
    kernelArg->stepOffset = 0;

    CcuKernelInfo kernelInfo;
    snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelMergeScratchDie%u", dieId);
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernelMergeScratch);
    kernelInfo.setKernelArg(kernelArg);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("RegisterCcuKernelMergeScratch: HcclCommQueryCcuIns failed, insNum[%u]", insNum), HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("HcommCcuKernelRegisterStart failed, ccuRet[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);

    const void *kernelArgs[1] = {kernelInfo.kernelArg};
    ccuRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
        reinterpret_cast<const void *>(kernelInfo.kernelFunc), kernelArgs, 1, &kernelHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("HcommCcuKernelRegister failed, ccuRet[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("HcommCcuKernelRegisterEnd failed, ccuRet[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

static HcclResult RegisterGroupDrrKernel(HcclComm comm, const OpParam &param, const std::vector<ChannelInfo> &channels,
    CcuKernelHandle &kernelHandle, uint32_t dieId, bool includeSelf, bool directOutput, uint32_t maxGroupSize)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("RegisterGroupDrrKernel: die[%u] has no channels", dieId),
        HCCL_E_INTERNAL);

    auto kernelArg = std::make_shared<ReduceScatterKernelArgHost>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    kernelArg->initOutput = includeSelf;
    kernelArg->reduceToOutput = directOutput;
    kernelArg->skipOutput = false;
    kernelArg->stepOffset = maxGroupSize;
    for (uint32_t i = 0; i < kernelArg->channelCount; i++) {
        kernelArg->channels[i] = channels[i].handle;
    }

    CcuKernelInfo kernelInfo;
    snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelGroupDrrDie%u", dieId);
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernelGroupDrr);
    kernelInfo.setKernelArg(kernelArg);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("RegisterGroupDrrKernel: invalid CCU insNum[%u]", insNum), HCCL_E_INTERNAL);
    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("RegisterGroupDrrKernel: register start failed, ret[%d]", static_cast<int>(ccuRet)),
        HCCL_E_INTERNAL);
    const void *kernelArgs[1] = {kernelInfo.kernelArg};
    ccuRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
        reinterpret_cast<const void *>(kernelInfo.kernelFunc), kernelArgs, 1, &kernelHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("RegisterGroupDrrKernel: register failed, ret[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("RegisterGroupDrrKernel: register end failed, ret[%d]", static_cast<int>(ccuRet)), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
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
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    int tagLen = snprintf(param.tag, sizeof(param.tag), "hccl_custom_reducescatter_v50_%u_%llu_%u_%u", param.rankSize,
        static_cast<unsigned long long>(param.count), static_cast<uint32_t>(param.dataType),
        static_cast<uint32_t>(param.reduceType));
    CHK_PRT_RET(tagLen <= 0 || static_cast<size_t>(tagLen) >= sizeof(param.tag),
        HCCL_ERROR("HcclReduceScatter: failed to build resource tag"), HCCL_E_INTERNAL);

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // 单 rank 通信域：直接 LocalCopy 后返回（无需 Channel/Kernel）
    if (param.rankSize <= 1) {
        uint32_t ts = ops_hccl::GetElemSize(param.dataType);
        CHK_PRT_RET(ts == 0,
            HCCL_ERROR("HcclReduceScatter: unsupported dataType[%d]", static_cast<int>(param.dataType)), HCCL_E_PARA);
        CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(param.cpuThread, recvBuf, sendBuf, param.count * ts)));
        return HCCL_SUCCESS;
    }

    // Dfx 诊断信息注册
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================

    // ==============================================
    // STEP 2.1: 申请 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread，申请 1 个 notify
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // CCU 资源已存在，复用
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device 上的内存，默认 400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请 Thread 资源
        // ==============================================
        // 主 thread 绑定用户 stream；双 die mesh 会额外申请 slave thread 并通过 notify 回收依赖。
        resCtxHost.threads.resize(1);
        resCtxHost.threads[0] = param.cpuThread;
        resCtxHost.ccuThread = param.cpuThread;

        // ==============================================
        // STEP 2.3: 申请 Channel 资源（全连接）
        // ==============================================
        CHK_RET(AcquireChannelsFullMesh(comm, param, resCtxHost));

        // ==============================================
        // STEP 2.4: 计算 CCL Buffer Token
        // ==============================================
        uint64_t cclToken = 0;
        CHK_RET(static_cast<HcclResult>(
            HcommCcuGetMemToken(reinterpret_cast<uint64_t>(cclBufferAddr), cclBufferSize, &cclToken)));
        resCtxHost.localCclToken = cclToken;

        // ==============================================
        // STEP 2.5: 算法选择 + 注册 CCU Kernel
        // ==============================================
        // RH 条件：pof2；跨 die 时按连续 die 段拆成多个 RH kernel 顺序执行。
        // 大数据逐 block 传输当前 slice，非 2 的幂 rank 继续走双 die 并行 mesh。
        const bool pof2 = (param.rankSize > 0) && ((param.rankSize & (param.rankSize - 1)) == 0);
        const uint32_t ts = ops_hccl::GetElemSize(param.dataType);
        constexpr uint64_t RH_THRESHOLD = 4ULL * 1024 * 1024; // 4MB
        const uint64_t opBytes = param.count * ts;
        const bool rank4 = param.rankSize == 4;
        const uint32_t dieChannelCount[2] = {static_cast<uint32_t>(resCtxHost.dieChannels[0].size()),
            static_cast<uint32_t>(resCtxHost.dieChannels[1].size())};
        const bool singleDie = (dieChannelCount[0] == 0) != (dieChannelCount[1] == 0);
        const bool useSmall4x1
            = rank4 && opBytes < RH_THRESHOLD && singleDie && dieChannelCount[0] + dieChannelCount[1] == 3;

        struct RhStepInfo {
            ChannelInfo channel;
            uint32_t globalStep;
        };
        std::vector<RhStepInfo> rhSteps;
        bool rhStepsReady = rank4 && pof2 && opBytes >= RH_THRESHOLD;
        uint32_t rhDieMask = 0;
        if (rhStepsReady) {
            uint32_t globalStep = 0;
            for (uint32_t mask = param.rankSize >> 1; mask >= 1; mask >>= 1, globalStep++) {
                uint32_t peer = param.myRank ^ mask;
                const ChannelInfo *peerChannel = nullptr;
                for (const auto &ch : resCtxHost.channels) {
                    if (ch.remoteRank == peer) {
                        peerChannel = &ch;
                        rhDieMask |= 1U << ch.localDie;
                        break;
                    }
                }
                if (peerChannel == nullptr) {
                    rhStepsReady = false;
                    break;
                }
                rhSteps.push_back(RhStepInfo{*peerChannel, globalStep});
            }
        }
        const bool rhSingleDie = rhStepsReady && ((rhDieMask & (rhDieMask - 1U)) == 0);
        const bool useRhSliced = rhStepsReady && rhSingleDie;
        const bool useRh = useRhSliced;
        const bool gDrrCandidate = !useSmall4x1 && !useRh && opBytes >= RH_THRESHOLD;
        bool groupDrr2x8Topo = gDrrCandidate && param.rankSize == 16 && resCtxHost.dieChannels[0].size() == 8
            && resCtxHost.dieChannels[1].size() == 7;
        if (groupDrr2x8Topo) {
            for (const auto &ch : resCtxHost.dieChannels[0]) {
                if (ch.netLayer != 1) {
                    groupDrr2x8Topo = false;
                    break;
                }
            }
        }
        if (groupDrr2x8Topo) {
            for (const auto &ch : resCtxHost.dieChannels[1]) {
                if (ch.netLayer != 0) {
                    groupDrr2x8Topo = false;
                    break;
                }
            }
        }
        const uint32_t minDieChannelCount
            = (dieChannelCount[0] < dieChannelCount[1]) ? dieChannelCount[0] : dieChannelCount[1];
        const uint32_t maxDieChannelCount
            = (dieChannelCount[0] > dieChannelCount[1]) ? dieChannelCount[0] : dieChannelCount[1];
        const bool groupDrr8p4Topo = gDrrCandidate && param.rankSize == 12
            && dieChannelCount[0] + dieChannelCount[1] == 11 && maxDieChannelCount >= 7
            && minDieChannelCount <= 4 && minDieChannelCount > 0;
        uint32_t groupDrrMaxGroupSize[2] = {0, 0};
        uint32_t groupDrrGroupCount[2] = {0, 0};
        const bool groupDrrTopo = groupDrr2x8Topo || groupDrr8p4Topo;
        if (groupDrrTopo) {
            for (uint32_t d = 0; d < 2; d++) {
                groupDrrMaxGroupSize[d] = ops_hccl::GetGroupDrrMaxGroupSize(dieChannelCount[d]);
                groupDrrGroupCount[d]
                    = ops_hccl::GetGroupDrrGroupCount(dieChannelCount[d], groupDrrMaxGroupSize[d]);
            }
        }
        const uint32_t groupDrrRequiredBlocks
            = (groupDrrGroupCount[0] == 0 || groupDrrGroupCount[1] == 0)
                  ? 0U
                  : (groupDrrGroupCount[0] + groupDrrGroupCount[1] - 1U);
        const bool useGroupDrr = groupDrrTopo
            && groupDrrRequiredBlocks > 0 && opBytes <= resCtxHost.localBuffer.size / groupDrrRequiredBlocks;
        const bool meshTwoDie = !useSmall4x1 && !useRh && !useGroupDrr && !resCtxHost.dieChannels[0].empty()
            && !resCtxHost.dieChannels[1].empty();
        resCtxHost.algoType = useSmall4x1 ? RS_ALGO_SMALL_4X1
                                           : (useRh ? RS_ALGO_RH
                                                    : (useGroupDrr ? RS_ALGO_GROUP_DRR
                                                                   : RS_ALGO_MESH));

        if (meshTwoDie || useGroupDrr) {
            ThreadHandle slaveThread = 0;
            CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, 1, &slaveThread));
            resCtxHost.threads.push_back(slaveThread);
        }

        if (useSmall4x1) {
            const uint32_t die = resCtxHost.dieChannels[0].empty() ? 1U : 0U;
            CcuKernelHandle kHandle;
            CHK_RET(RegisterSmall4x1Kernel(comm, param, resCtxHost.dieChannels[die], kHandle, die));
            resCtxHost.ccuKernels.push_back(kHandle);
        } else if (useRh) {
            // ---- RH: WriteReduce(my_INPUT → peer_INPUT), 原地累积 ----
            struct RhSegment {
                std::vector<ChannelInfo> channels;
                uint32_t stepOffset;
                uint32_t die;
            };
            std::vector<RhSegment> rhSegments;
            for (const auto &step : rhSteps) {
                const uint32_t d = step.channel.localDie;
                if (rhSegments.empty() || rhSegments.back().die != d) {
                    RhSegment segment;
                    segment.stepOffset = step.globalStep;
                    segment.die = d;
                    rhSegments.push_back(segment);
                }
                rhSegments.back().channels.push_back(step.channel);
            }

            for (uint32_t i = 0; i < rhSegments.size(); i++) {
                CcuKernelHandle kHandle;
                CHK_RET(RegisterCcuKernelPerDieRH(comm, param, rhSegments[i].channels, kHandle,
                    i + 1 == rhSegments.size(), rhSegments[i].stepOffset, useRhSliced, rhSegments[i].die));
                resCtxHost.ccuKernels.push_back(kHandle);
            }
        } else if (useGroupDrr) {
            CcuKernelHandle kernel;
            const uint32_t outputDie = ops_hccl::SelectGroupDrrOutputDie(resCtxHost);
            CHK_RET(RegisterGroupDrrKernel(comm, param, resCtxHost.dieChannels[0], kernel, 0, outputDie != 0,
                outputDie == 0, groupDrrMaxGroupSize[0]));
            resCtxHost.ccuKernels.push_back(kernel);
            CHK_RET(RegisterGroupDrrKernel(comm, param, resCtxHost.dieChannels[1], kernel, 1, outputDie != 1,
                outputDie == 1, groupDrrMaxGroupSize[1]));
            resCtxHost.ccuKernels.push_back(kernel);
            CHK_RET(RegisterCcuKernelMergeScratch(comm, param, resCtxHost.dieChannels[0], kernel, 0, false));
            resCtxHost.ccuKernels.push_back(kernel);
        } else {
            // ---- Mesh: Read(peer_INPUT → CCL) + tree LocalReduce ----
            const uint32_t outputDie = ops_hccl::SelectMeshOutputDie(resCtxHost);
            bool firstKernel = true;
            const bool usePipelineKernel = meshTwoDie && (opBytes >= RH_THRESHOLD);
            for (uint32_t d = 0; d < 2; d++) {
                if (!resCtxHost.dieChannels[d].empty()) {
                    CcuKernelHandle kHandle;
                    const bool includeSelf = meshTwoDie ? (d == outputDie) : firstKernel;
                    const bool reduceToOutput = !firstKernel && !meshTwoDie;
                    const bool skipOutput = meshTwoDie && (d != outputDie);
                    CHK_RET(RegisterCcuKernelPerDie(comm, param, resCtxHost.dieChannels[d], kHandle, includeSelf,
                        reduceToOutput, skipOutput, d, usePipelineKernel));
                    resCtxHost.ccuKernels.push_back(kHandle);
                    firstKernel = false;
                }
            }
            if (meshTwoDie) {
                const uint32_t mergeKernelCount = (opBytes >= RH_THRESHOLD) ? 2U : 1U;
                for (uint32_t d = 0; d < mergeKernelCount; d++) {
                    CcuKernelHandle kHandle;
                    CHK_RET(RegisterCcuKernelMergeScratch(
                        comm, param, resCtxHost.dieChannels[d], kHandle, d, opBytes >= RH_THRESHOLD));
                    resCtxHost.ccuKernels.push_back(kHandle);
                }
            }
        }
        // ==============================================
        // STEP 2.6: 申请通信引擎上下文
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
