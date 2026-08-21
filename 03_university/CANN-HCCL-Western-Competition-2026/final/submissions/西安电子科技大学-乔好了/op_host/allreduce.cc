#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_launch.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"
#include "../op_kernel_ccu/ccu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t DIE_NUM = 2;
constexpr uint64_t LATENCY_MAX_ELEMENTS = (512ULL * 1024ULL) / sizeof(float);

struct ChannelGroup {
    uint32_t dieId{0};
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> ranks;
};

HcclResult BuildChannels(HcclComm comm, const OpParam &param, std::vector<ChannelGroup> &groups)
{
    uint32_t *netLayersRaw = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayersRaw, &netLayerNum));
    CHK_PRT_RET(netLayerNum == 0 || netLayersRaw == nullptr,
        HCCL_ERROR("[BuildChannels] empty RankGraph layer list"), HCCL_E_INTERNAL);
    // RankGraph returned memory is invalidated by later RankGraph calls.
    const std::vector<uint32_t> netLayers(netLayersRaw, netLayersRaw + netLayerNum);

    std::vector<ChannelGroup> byDie(DIE_NUM);
    for (uint32_t die = 0; die < DIE_NUM; ++die) {
        byDie[die].dieId = die;
    }

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CommLink selectedLink{};
        bool found = false;
        for (uint32_t layer : netLayers) {
            CommLink *links = nullptr;
            uint32_t linkNum = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layer, param.myRank, remoteRank, &links, &linkNum));
            for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
                if (links[linkIndex].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    // The official sample uses the first UBC_CTP link.  Keeping that
                    // symmetric choice is required for the two endpoints to connect.
                    selectedLink = links[linkIndex];
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            }
        }
        CHK_PRT_RET(!found,
            HCCL_ERROR("[BuildChannels] UBC_CTP link not found, localRank[%u], remoteRank[%u]",
                param.myRank, remoteRank), HCCL_E_NOT_FOUND);

        EndpointAttrDieId dieId = 0;
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &selectedLink.srcEndpointDesc,
            ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[BuildChannels] invalid endpoint die[%u]", dieId), HCCL_E_INTERNAL);

        HcclChannelDesc desc{};
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = remoteRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = selectedLink.linkAttr.linkProtocol;
        desc.localEndpoint = selectedLink.srcEndpointDesc;
        desc.remoteEndpoint = selectedLink.dstEndpointDesc;
        ChannelHandle channel{};
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        byDie[dieId].channels.push_back(channel);
        byDie[dieId].ranks.push_back(remoteRank);
    }

    for (ChannelGroup &group : byDie) {
        if (!group.channels.empty()) {
            groups.push_back(group);
        }
    }
    CHK_PRT_RET(groups.empty() || groups.size() > DIE_NUM,
        HCCL_ERROR("[BuildChannels] invalid non-empty die group count[%zu]", groups.size()),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param,
    const std::vector<ChannelGroup> &groups, AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(groups.empty() || groups.size() > DIE_NUM,
        HCCL_ERROR("[RegisterKernels] invalid die group count[%zu]", groups.size()), HCCL_E_INTERNAL);

    bool seenRank[MAX_RANK_SIZE]{};
    uint32_t remoteRankCount = 0;
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const ChannelGroup &group = groups[groupIndex];
        CHK_PRT_RET(group.dieId >= DIE_NUM || group.channels.size() != group.ranks.size() ||
                group.channels.empty() || group.channels.size() >= param.rankSize,
            HCCL_ERROR("[RegisterKernels] invalid group[%zu], die[%u], channelNum[%zu], rankNum[%zu]",
                groupIndex, group.dieId, group.channels.size(), group.ranks.size()), HCCL_E_INTERNAL);
        CHK_PRT_RET(groupIndex != 0 && groups[groupIndex - 1].dieId >= group.dieId,
            HCCL_ERROR("[RegisterKernels] die groups are not strictly ordered"), HCCL_E_INTERNAL);
        for (size_t channel = 0; channel < group.ranks.size(); ++channel) {
            const uint32_t remoteRank = group.ranks[channel];
            CHK_PRT_RET(remoteRank >= param.rankSize || remoteRank == param.myRank || seenRank[remoteRank],
                HCCL_ERROR("[RegisterKernels] duplicate/invalid source rank[%u] in group[%zu]",
                    remoteRank, groupIndex), HCCL_E_INTERNAL);
            CHK_PRT_RET(channel != 0 && group.ranks[channel - 1] >= remoteRank,
                HCCL_ERROR("[RegisterKernels] group[%zu] source order is not canonical", groupIndex),
                HCCL_E_INTERNAL);
            seenRank[remoteRank] = true;
            ++remoteRankCount;
        }
    }
    CHK_PRT_RET(remoteRankCount != param.rankSize - 1,
        HCCL_ERROR("[RegisterKernels] source partition incomplete, remoteNum[%u], expected[%u]",
            remoteRankCount, param.rankSize - 1), HCCL_E_INTERNAL);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[RegisterKernels] expected one CCU instance, got[%u]", insNum), HCCL_E_INTERNAL);

    // Keep the primary role aligned with thread/group zero.  Finalize uses this
    // stable ordering to serialize the cross-die merge before peer distribution.
    constexpr size_t primaryIndex = 0;
    size_t localSourceIndex = primaryIndex;
    if (param.rankSize == 16 && groups.size() == DIE_NUM &&
        groups[1].ranks.size() < groups[0].ranks.size()) {
        localSourceIndex = 1;
    }
    std::vector<std::shared_ptr<ops_hccl::CcuKernelArgAllReduce>> kernelArgs;
    kernelArgs.reserve(groups.size());
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const ChannelGroup &group = groups[groupIndex];
        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgAllReduce>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->dieId = group.dieId;
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        kernelArg->reduceWithLocal = groupIndex == primaryIndex;
        kernelArg->ownsLocalSource = groupIndex == localSourceIndex;
        kernelArg->useP16DualChain =
            param.rankSize == 16 && param.count > LATENCY_MAX_ELEMENTS ?
            (param.inputPtr != param.outputPtr ? 2U : 1U) : 0U;
        kernelArg->kernelCount = static_cast<uint32_t>(groups.size());
        kernelArg->dataType = param.dataType;
        kernelArg->reduceOp = param.reduceType;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            kernelArg->channels[i] = group.channels[i];
            kernelArg->channelRanks[i] = group.ranks[i];
        }

        kernelArgs.push_back(kernelArg);
    }

    if (kernelArgs.size() == DIE_NUM) {
        CHK_PRT_RET(kernelArgs[0]->dieId == kernelArgs[1]->dieId,
            HCCL_ERROR("[RegisterKernels] two-die path received identical die ids[%u]",
                kernelArgs[0]->dieId), HCCL_E_INTERNAL);
        CHK_PRT_RET(kernelArgs[0]->reduceWithLocal == kernelArgs[1]->reduceWithLocal,
            HCCL_ERROR("[RegisterKernels] two CCU kernels have the same role[%u]",
                kernelArgs[0]->reduceWithLocal), HCCL_E_INTERNAL);
        CHK_PRT_RET(kernelArgs[0]->ownsLocalSource == kernelArgs[1]->ownsLocalSource,
            HCCL_ERROR("[RegisterKernels] two CCU kernels have the same local-source role[%u]",
                kernelArgs[0]->ownsLocalSource), HCCL_E_INTERNAL);
    }

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterKernels] register start failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));

    const bool useP4FusedKernel = param.rankSize == 4 && groups.size() == 1;
    if (useP4FusedKernel) {
        const void *args[] = {kernelArgs[0].get()};
        CcuKernelHandle kernelHandle{};
        ccuRet = HcommCcuKernelRegister(insHandle, groups[0].dieId,
            "CcuKernelAllReduceP4Fused", reinterpret_cast<void *>(ops_hccl::CcuKernelP4Fused),
            args, 1, &kernelHandle);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS,
            HCCL_ERROR("[RegisterKernels] P4 fused kernel register failed[%d]", ccuRet),
            ConvertCcuToHccl(ccuRet));
        resCtx.ccuKernels.push_back(kernelHandle);
    } else {
        // Stage-major immutable order: reduce one owner shard, then finalize it
        // and gather completed peer shards.  CCU exposes two kernel slots per die.
        constexpr uint32_t stageCount = 2;
        const bool useCombinedLatencyKernels =
            (param.rankSize == 12 || param.rankSize == 16) &&
            param.count <= LATENCY_MAX_ELEMENTS;
        const char *kernelNames[stageCount] = {
            useCombinedLatencyKernels ? "CcuKernelAllReducePartialCombinedExchange" :
                "CcuKernelAllReducePartial",
            useCombinedLatencyKernels ? "CcuKernelAllReduceFinalizeReuseExchange" :
                "CcuKernelAllReduceFinalize"};
        const void *kernelFuncs[stageCount] = {
            reinterpret_cast<void *>(useCombinedLatencyKernels ?
                ops_hccl::CcuKernelPartialCombinedExchange : ops_hccl::CcuKernelPartial),
            reinterpret_cast<void *>(useCombinedLatencyKernels ?
                ops_hccl::CcuKernelFinalizeReuseExchange : ops_hccl::CcuKernelFinalize)};
        for (uint32_t stage = 0; stage < stageCount; ++stage) {
            for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
                const ChannelGroup &group = groups[groupIndex];
                const void *args[] = {kernelArgs[groupIndex].get()};
                CcuKernelHandle kernelHandle{};
                ccuRet = HcommCcuKernelRegister(insHandle, group.dieId,
                    kernelNames[stage], kernelFuncs[stage], args, 1, &kernelHandle);
                CHK_PRT_RET(ccuRet != CCU_SUCCESS,
                    HCCL_ERROR("[RegisterKernels] stage[%u] die[%u] register failed[%d]",
                        stage, group.dieId, ccuRet), ConvertCcuToHccl(ccuRet));
                resCtx.ccuKernels.push_back(kernelHandle);
            }
        }
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterKernels] register end failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.reduceType = op;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[HcclAllReduce] rankSize[%u] exceeds fixed capacity[%llu]", param.rankSize,
            static_cast<unsigned long long>(MAX_RANK_SIZE)), HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM,
        HCCL_ERROR("[HcclAllReduce] only FP32 SUM is supported"), HCCL_E_NOT_SUPPORT);

    // P4/P16 have distinct registered partial graphs and scratch contracts.
    // Keep the proven B0 cache key only for the unchanged P12 specialization.
    const bool useP12LatencyKernel = param.rankSize == 12 && param.count <= LATENCY_MAX_ELEMENTS;
    const bool useP16LatencyKernel = param.rankSize == 16 && param.count <= LATENCY_MAX_ELEMENTS;
    const bool useP16Preseed = param.rankSize == 16 &&
        param.count > LATENCY_MAX_ELEMENTS && sendBuf != recvBuf;
    const char *engineCtxTag = param.rankSize == 4 ?
        "hccl_custom_allreduce_p4scratchfanin_e3aabi_v1" :
        (useP12LatencyKernel ? "hccl_custom_allreduce_p12latency512k_e3aabi_v1" :
        (useP16LatencyKernel ? "hccl_custom_allreduce_p16latency512k_e3aabi_v1" :
        (param.rankSize == 16 ? (useP16Preseed ?
            "hccl_custom_allreduce_p16dualhbm4x4_slotcap_readrot_ooppreseed_v1" :
            "hccl_custom_allreduce_p16dualhbm4x4_slotcap_readrot_v1") :
            "hccl_custom_allreduce_staged_e3aabi_v1")));
    (void)snprintf(param.tag, sizeof(param.tag), "%s", engineCtxTag);

    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost{};
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        resCtxHost.ccuThread = param.cpuThread;
        resCtxHost.threads.push_back(param.cpuThread);

        if (param.rankSize > 1) {
            std::vector<ChannelGroup> groups;
            CHK_RET(BuildChannels(comm, param, groups));
            if (groups.size() == DIE_NUM) {
                ThreadHandle slaveThread{};
                CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, 1, &slaveThread));
                resCtxHost.threads.push_back(slaveThread);
            }
            CHK_RET(RegisterKernels(comm, param, groups, resCtxHost));
        }

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seq.size(), 0));
    }
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
