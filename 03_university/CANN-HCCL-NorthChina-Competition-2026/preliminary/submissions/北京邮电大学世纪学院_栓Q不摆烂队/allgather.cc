#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <string>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;

HcclResult FillChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    for (uint32_t i = 0; i < netLayerNum; i++) {
        uint32_t netLayer = netLayers[i];
        uint32_t listSize = 0;
        CommLink *linkList = nullptr;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize));

        if (listSize == 0) {
            continue;
        }

        CHK_RET(HcclChannelDescInit(&desc, 1));
        CommLink link = linkList[0];
        desc.remoteRank = dstRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link.srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
        HCCL_INFO("[FillChannelDesc] Found link between rank %u and rank %u at netLayer %u, protocol %d",
            srcRank, dstRank, netLayer, link.linkAttr.linkProtocol);
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[FillChannelDesc] No link found between rank %u and rank %u",
        srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult HcclAllocAlgResourceAICPU(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    void *cclBufferAddr;
    uint64_t cclBufferSize;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

    uint32_t threadNum = param.rankSize > 1 ? param.rankSize - 1 : 1;
    uint32_t notifyNumOnMainThread = threadNum;
    resCtxHost.threads.resize(threadNum);
    CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, threadNum, notifyNumOnMainThread + 1, resCtxHost.threads.data()));
    resCtxHost.aicpuThread = resCtxHost.threads[0];

    if (param.rankSize > 1) {
        const uint32_t channelNum = param.rankSize - 1;
        std::vector<HcclChannelDesc> descs(channelNum);
        std::vector<ChannelHandle> channels(channelNum);

        uint32_t idx = 0;
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
            if (remoteRank == param.myRank) {
                continue;
            }
            CHK_RET(FillChannelDesc(comm, param.myRank, remoteRank, descs[idx]));
            idx++;
        }

        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), channelNum, channels.data()));

        resCtxHost.channels.reserve(channelNum);
        idx = 0;
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
            if (remoteRank == param.myRank) {
                continue;
            }
            ChannelInfo channel;
            channel.remoteRank = remoteRank;
            channel.handle = channels[idx];
            channel.notifyNum = CHANNEL_NOTIFY_NUM;
            void *cclBuf;
            uint64_t cclBufSize;
            CHK_RET(HcclChannelGetHcclBuffer(comm, channels[idx], &cclBuf, &cclBufSize));
            channel.remoteCclMem = CommBuffer{cclBuf, cclBufSize};
            resCtxHost.channels.push_back(channel);
            idx++;
        }
    }
    return HCCL_SUCCESS;
}
}

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    static constexpr const char TAG[] = "hccl_custom_allgather";
    static_assert(sizeof(TAG) <= sizeof(param.tag), "TAG exceeds param.tag size");
    memcpy(param.tag, TAG, sizeof(TAG));
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        HCCL_INFO("Creating engine context");
        AlgResourceCtx resCtxHost;

        CHK_RET(HcclAllocAlgResourceAICPU(comm, param, resCtxHost));

        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}