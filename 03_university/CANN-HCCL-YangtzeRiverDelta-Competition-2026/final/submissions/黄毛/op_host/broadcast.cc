#include <cstdio>
#include <memory>
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
constexpr char RESOURCE_TAG[] = "hccl_custom_broadcast_ccu_v2_pod";
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

struct ChannelResource {
    ChannelHandle handle;
    uint32_t remoteRank;
    uint32_t dieId;
};

HcclResult ValidateParam(const OpParam &param)
{
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize ||
        param.root >= param.rankSize) {
        HCCL_ERROR("invalid rank info, rank=%u root=%u rankSize=%u", param.myRank, param.root, param.rankSize);
        return HCCL_E_PARA;
    }
    auto iter = SIZE_TABLE.find(param.dataType);
    if (iter == SIZE_TABLE.end() || iter->second == 0 || param.count > UINT64_MAX / iter->second) {
        HCCL_ERROR("unsupported data type or data size overflow");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult FindLink(HcclComm comm, uint32_t localRank, uint32_t remoteRank, CommLink &selected, uint32_t &dieId)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    HcclResult lastRet = HCCL_E_NOT_FOUND;
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(
            comm, layers[layerIndex], localRank, remoteRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS) {
            lastRet = ret;
            continue;
        }
        for (uint32_t index = 0; index < linkNum; ++index) {
            if (links[index].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                selected = links[index];
                CHK_RET(HcclRankGraphGetEndpointInfo(comm, localRank, &selected.srcEndpointDesc,
                    ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
                return HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("no link between rank %u and rank %u", localRank, remoteRank);
    return lastRet;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, std::vector<ChannelResource> &channels)
{
    channels.resize(param.rankSize - 1);
    std::vector<HcclChannelDesc> descs(param.rankSize - 1);
    std::vector<uint32_t> dieIds(param.rankSize - 1);
    CHK_RET(HcclChannelDescInit(descs.data(), static_cast<uint32_t>(descs.size())));

    uint32_t index = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CommLink link;
        CHK_RET(FindLink(comm, param.myRank, remoteRank, link, dieIds[index]));
        HcclChannelDesc &desc = descs[index++];
        desc.remoteRank = remoteRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint = link.srcEndpointDesc;
        desc.remoteEndpoint = link.dstEndpointDesc;
    }
    std::vector<ChannelHandle> handles(descs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, descs.data(),
        static_cast<uint32_t>(descs.size()), handles.data()));
    for (uint32_t channelIndex = 0; channelIndex < handles.size(); ++channelIndex) {
        channels[channelIndex] = ChannelResource{handles[channelIndex], descs[channelIndex].remoteRank,
            dieIds[channelIndex]};
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterKernel(HcclComm comm, const OpParam &param, const std::vector<ChannelResource> &channels,
    uint32_t dieId, uint32_t phaseMask, AlgResourceCtx &resource)
{
    auto kernelArg = std::make_shared<BroadcastKernelArg>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->phaseMask = phaseMask;
    kernelArg->channelCount = 0;
    for (const ChannelResource &channel : channels) {
        if (channel.dieId != dieId) {
            continue;
        }
        uint32_t index = kernelArg->channelCount++;
        kernelArg->channels[index] = channel.handle;
        kernelArg->remoteRanks[index] = channel.remoteRank;
    }
    if (kernelArg->channelCount == 0) {
        return HCCL_SUCCESS;
    }

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("unexpected CCU instance count=%u", insNum), HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuResult(ccuRet);
    }
    CcuKernelHandle kernelHandle = 0;
    const void *kernelArgs[] = {kernelArg.get()};
    char kernelName[64] = {0};
    int nameRet = std::snprintf(kernelName, sizeof(kernelName), "CcuBroadcast_%u_%u", phaseMask, dieId);
    if (nameRet < 0 || static_cast<uint32_t>(nameRet) >= sizeof(kernelName)) {
        (void)HcommCcuKernelRegisterEnd(insHandle);
        return HCCL_E_INTERNAL;
    }
    ccuRet = HcommCcuKernelRegister(insHandle, dieId, kernelName,
        reinterpret_cast<const void *>(ops_hccl::CcuKernel), kernelArgs, 1, &kernelHandle);
    if (ccuRet != CCU_SUCCESS) {
        (void)HcommCcuKernelRegisterEnd(insHandle);
        return ConvertCcuResult(ccuRet);
    }
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuResult(ccuRet);
    }
    return resource.AddKernel(kernelHandle, dieId);
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    int tagRet = std::snprintf(param.tag, TAG_LENGTH, "%s", RESOURCE_TAG);
    CHK_PRT_RET(tagRet < 0 || static_cast<uint32_t>(tagRet) >= TAG_LENGTH,
        HCCL_ERROR("failed to format resource tag"), HCCL_E_PARA);
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo {};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_RET(ValidateParam(param));
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    constexpr CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream, 1, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resource;
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resource.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        resource.ccuThread = param.cpuThread;
        CHK_RET(resource.AddThread(param.cpuThread));
        ThreadHandle workerThread = 0;
        CHK_RET(HcclThreadAcquire(comm, engine, 1, 1, &workerThread));
        CHK_RET(resource.AddThread(workerThread));

        std::vector<ChannelResource> channels;
        CHK_RET(AcquireChannels(comm, param, channels));
        for (uint32_t dieId = 0; dieId < 2; ++dieId) {
            CHK_RET(RegisterKernel(comm, param, channels, dieId, 0, resource));
        }
        resource.directKernelCount = resource.kernelCount;
        for (uint32_t phaseMask = 1; phaseMask <= 2; ++phaseMask) {
            for (uint32_t dieId = 0; dieId < 2; ++dieId) {
                CHK_RET(RegisterKernel(comm, param, channels, dieId, phaseMask, resource));
            }
        }

        std::vector<char> sequence = resource.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, sequence.data(), sequence.size(), 0));
    }

    return ops_hccl::ExecOp(param);
}
