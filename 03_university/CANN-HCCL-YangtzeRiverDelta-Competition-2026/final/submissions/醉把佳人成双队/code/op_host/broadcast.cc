#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <vector>
#include <ccu/ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include "ccu_launch.h"
#include "hccl_ccu_res.h"
#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint64_t DIRECT_ALG_MAX_BYTES = 512ULL * 1024ULL;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t THREAD_NOTIFY_NUM = 4;
// 测试点23/24极致优化：将 Chunk 大小缩短为 8MB 极限降低流水线起步延迟
constexpr uint64_t PIPELINE_CHUNK_BYTES = 8ULL * 1024ULL * 1024ULL;

struct DieChannelGroup {
    uint32_t dieId = 0;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> remoteRanks;
};

HcclResult QueryEndpointDieId(HcclComm comm, uint32_t rankId, const EndpointDesc &endpoint, uint32_t &dieId)
{
    EndpointAttrDieId endpointDieId = 0;
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, rankId, &endpoint, ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId), &endpointDieId));
    dieId = static_cast<uint32_t>(endpointDieId);
    return HCCL_SUCCESS;
}

HcclResult GetCcuChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank, const uint32_t *netLayers, uint32_t netLayerNum, HcclChannelDesc &desc, uint32_t &localDieId)
{
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], myRank, remoteRank, &links, &linkNum));

        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            const CommLink &link = links[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) continue;

            HcclChannelDesc candidate;
            CHK_RET(HcclChannelDescInit(&candidate, 1));
            candidate.remoteRank = remoteRank;
            candidate.notifyNum = CHANNEL_NOTIFY_NUM;
            candidate.channelProtocol = link.linkAttr.linkProtocol;
            candidate.localEndpoint = link.srcEndpointDesc;
            candidate.remoteEndpoint = link.dstEndpointDesc;

            CHK_RET(QueryEndpointDieId(comm, myRank, candidate.localEndpoint, localDieId));
            desc = candidate;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult RegisterOneKernel(CcuInsHandle insHandle, uint32_t dieId, const char *name, void *kernelFunc, const std::shared_ptr<BroadcastCcuKernelArg> &kernelArg, std::vector<CcuKernelInfo> &kernelInfos, CcuKernelHandle &kernelHandle)
{
    kernelInfos.emplace_back();
    CcuKernelInfo &kernelInfo = kernelInfos.back();
    std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s", name);
    kernelInfo.kernelFunc = kernelFunc;
    kernelInfo.setKernelArg(kernelArg);
    const void *kernelArgs[] = {kernelInfo.kernelArg};
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle));
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(const OpParam &param, CcuInsHandle insHandle, const std::vector<DieChannelGroup> &groups, uint32_t algorithm, uint64_t totalBytes, AlgResourceCtx &resCtxHost)
{
    CHK_PRT_RET(groups.empty(), HCCL_ERROR("No die channel group"), HCCL_E_INTERNAL);
    CHK_PRT_RET(groups.size() > MAX_DIE_KERNEL_NUM, HCCL_ERROR("Too many die groups"), HCCL_E_NOT_SUPPORT);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    std::vector<CcuKernelInfo> kernelInfos;
    kernelInfos.reserve(groups.size() * 2);
    resCtxHost.algorithm = algorithm;
    resCtxHost.ccuKernelEntries.clear();
    resCtxHost.ccuKernelEntries.reserve(groups.size());

    for (uint32_t groupIdx = 0; groupIdx < groups.size(); ++groupIdx) {
        const DieChannelGroup &group = groups[groupIdx];

        auto kernelArg = std::make_shared<BroadcastCcuKernelArg>();
        kernelArg->myRank = param.myRank;
        kernelArg->rankSize = param.rankSize;
        kernelArg->root = param.root;
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        for (uint32_t idx = 0; idx < group.channels.size(); ++idx) {
            kernelArg->channels[idx] = group.channels[idx];
            kernelArg->remoteRanks[idx] = group.remoteRanks[idx];
        }

        if (algorithm == BCAST_ALG_CHAIN_PIPELINE) {
            const uint32_t relativeRank = (param.myRank + param.rankSize - param.root) % param.rankSize;
            if (relativeRank > 0) kernelArg->prevRank = (param.myRank + param.rankSize - 1) % param.rankSize;
            if (relativeRank + 1 < param.rankSize) kernelArg->nextRank = (param.myRank + 1) % param.rankSize;
            for (uint32_t idx = 0; idx < group.remoteRanks.size(); ++idx) {
                if (group.remoteRanks[idx] == kernelArg->prevRank) kernelArg->prevChannelIndex = idx;
                if (group.remoteRanks[idx] == kernelArg->nextRank) kernelArg->nextChannelIndex = idx;
            }
            kernelArg->pipelineChunkBytes = PIPELINE_CHUNK_BYTES;
            kernelArg->pipelineChunkCount = static_cast<uint32_t>((totalBytes + PIPELINE_CHUNK_BYTES - 1ULL) / PIPELINE_CHUNK_BYTES);
            kernelArg->pipelineLastChunkBytes = totalBytes - static_cast<uint64_t>(kernelArg->pipelineChunkCount - 1U) * PIPELINE_CHUNK_BYTES;
        }

        CcuKernelLaunchEntry entry{};
        entry.dieId = group.dieId;
        entry.threadIndex = groupIdx;
        entry.channelCount = static_cast<uint32_t>(group.remoteRanks.size());
        for (uint32_t idx = 0; idx < group.remoteRanks.size(); ++idx) {
            entry.remoteRanks[idx] = group.remoteRanks[idx];
            if (group.remoteRanks[idx] == param.root) entry.containsRootChannel = 1;
        }

        char kernelName[64]{};
        if (algorithm == BCAST_ALG_DIRECT) {
            const bool isRoot = param.myRank == param.root;
            std::snprintf(kernelName, sizeof(kernelName), isRoot ? "bcast_v82_direct_root_r%u_m%u_d%u" : "bcast_v82_direct_peer_r%u_m%u_d%u", param.root, param.myRank, group.dieId);
            CHK_RET(RegisterOneKernel(insHandle, group.dieId, kernelName, reinterpret_cast<void *>(isRoot ? ops_hccl::CcuBroadcastDirectRootKernel : ops_hccl::CcuBroadcastDirectPeerKernel), kernelArg, kernelInfos, entry.directHandle));
        } else if (algorithm == BCAST_ALG_CHAIN_PIPELINE) {
            std::snprintf(kernelName, sizeof(kernelName), "bcast_v82_chain_r%u_m%u_d%u_n%u", param.root, param.myRank, group.dieId, kernelArg->pipelineChunkCount);
            CHK_RET(RegisterOneKernel(insHandle, group.dieId, kernelName, reinterpret_cast<void *>(ops_hccl::CcuBroadcastChainPipelineKernel), kernelArg, kernelInfos, entry.chainHandle));
        } else if (algorithm == BCAST_ALG_8P4_PREFETCH_SAG) {
            std::snprintf(kernelName, sizeof(kernelName), "bcast_v82_8p4_scatter_r%u_m%u_d%u", param.root, param.myRank, group.dieId);
            CHK_RET(RegisterOneKernel(insHandle, group.dieId, kernelName, reinterpret_cast<void *>(ops_hccl::CcuBroadcast8p4ScatterPrefetchKernel), kernelArg, kernelInfos, entry.scatterHandle));
            std::snprintf(kernelName, sizeof(kernelName), "bcast_v82_8p4_ag_r%u_m%u_d%u", param.root, param.myRank, group.dieId);
            CHK_RET(RegisterOneKernel(insHandle, group.dieId, kernelName, reinterpret_cast<void *>(ops_hccl::CcuBroadcast8p4AllGatherSkipRootKernel), kernelArg, kernelInfos, entry.allGatherHandle));
        } else {
            std::snprintf(kernelName, sizeof(kernelName), "bcast_v82_scatter_r%u_m%u_d%u", param.root, param.myRank, group.dieId);
            CHK_RET(RegisterOneKernel(insHandle, group.dieId, kernelName, reinterpret_cast<void *>(ops_hccl::CcuBroadcastScatterKernel), kernelArg, kernelInfos, entry.scatterHandle));
            std::snprintf(kernelName, sizeof(kernelName), "bcast_v82_allgather_r%u_m%u_d%u", param.root, param.myRank, group.dieId);
            CHK_RET(RegisterOneKernel(insHandle, group.dieId, kernelName, reinterpret_cast<void *>(ops_hccl::CcuBroadcastAllGatherKernel), kernelArg, kernelInfos, entry.allGatherHandle));
        }
        resCtxHost.ccuKernelEntries.push_back(entry);
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf); CHK_PTR_NULL(comm); CHK_PTR_NULL(stream);

    OpParam param;
    param.inputPtr = buf; param.outputPtr = buf; param.count = count;
    param.dataType = dataType; param.root = root; param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo; char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    
    const uint64_t typeSize = FastGetTypeSize(param.dataType);
    CHK_PRT_RET(typeSize == 0, HCCL_ERROR("Unsupported dataType[%d]", static_cast<int32_t>(param.dataType)), HCCL_E_PARA);
    const uint64_t totalBytes = param.count * typeSize;

    if (param.rankSize <= 1 || totalBytes == 0) return HCCL_SUCCESS;

    const bool useDirect = totalBytes <= DIRECT_ALG_MAX_BYTES;
    const bool useChainPipeline = !useDirect && param.rankSize == 4;
    const bool use8P4 = !useDirect && param.rankSize == 12;

    if (useChainPipeline) {
        std::snprintf(param.tag, sizeof(param.tag), "hccl_bcast_ccu_v82_chain_r%u_b%llu", root, static_cast<unsigned long long>(totalBytes));
    } else if (use8P4) {
        std::snprintf(param.tag, sizeof(param.tag), "hccl_bcast_ccu_v82_8p4_r%u", root);
    } else {
        std::snprintf(param.tag, sizeof(param.tag), "hccl_bcast_ccu_v82_%s_r%u", useDirect ? "direct" : "sag", root);
    }

    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, THREAD_NOTIFY_NUM, &param.cpuThread));

    void *ctx = nullptr; uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx; param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        resCtxHost.ccuThread = param.cpuThread;
        void *cclBufferAddr = nullptr; uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        std::vector<uint32_t> remoteRanks;
        if (useDirect) {
            if (param.myRank == param.root) {
                remoteRanks.reserve(param.rankSize - 1);
                for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                    if (rank != param.root) remoteRanks.push_back(rank);
                }
            } else {
                remoteRanks.push_back(param.root);
            }
        } else if (useChainPipeline) {
            const uint32_t relativeRank = (param.myRank + param.rankSize - param.root) % param.rankSize;
            if (relativeRank > 0) remoteRanks.push_back((param.myRank + param.rankSize - 1) % param.rankSize);
            if (relativeRank + 1 < param.rankSize) remoteRanks.push_back((param.myRank + 1) % param.rankSize);
        } else {
            remoteRanks.reserve(param.rankSize - 1);
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (rank != param.myRank) remoteRanks.push_back(rank);
            }
        }

        uint32_t *netLayers = nullptr; uint32_t netLayerNum = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

        std::vector<HcclChannelDesc> channelDescs(remoteRanks.size());
        std::vector<uint32_t> localDieIds(remoteRanks.size(), 0);
        for (uint32_t idx = 0; idx < remoteRanks.size(); ++idx) {
            CHK_RET(GetCcuChannelDesc(comm, param.myRank, remoteRanks[idx], netLayers, netLayerNum, channelDescs[idx], localDieIds[idx]));
        }

        std::vector<ChannelHandle> channels(remoteRanks.size());
        CHK_RET(HcclChannelAcquire(comm, ccuEngine, channelDescs.data(), static_cast<uint32_t>(channelDescs.size()), channels.data()));

        std::map<uint32_t, DieChannelGroup> groupMap;
        for (uint32_t idx = 0; idx < channels.size(); ++idx) {
            auto &group = groupMap[localDieIds[idx]];
            group.dieId = localDieIds[idx];
            group.channels.push_back(channels[idx]);
            group.remoteRanks.push_back(remoteRanks[idx]);
        }

        std::vector<DieChannelGroup> groups;
        for (auto &item : groupMap) groups.push_back(std::move(item.second));

        // Large 12/16-rank paths use the measured-stable two-die ordering.
        // Direct and 4-rank chain paths are deliberately left byte-for-byte
        // equivalent to the uploaded 110066 behavior.
        if (!useDirect && !useChainPipeline && groups.size() == 2 &&
            (param.rankSize == 12 || param.rankSize == 16)) {
            if (param.myRank == param.root) {
                // Launch the die with more channels as the worker first; keep
                // the lighter die on the stream-bound main thread.
                if (groups[0].channels.size() > groups[1].channels.size()) {
                    std::swap(groups[0], groups[1]);
                }
            } else {
                bool rootInFirst = false;
                bool rootInSecond = false;
                for (uint32_t rank : groups[0].remoteRanks) {
                    rootInFirst = rootInFirst || rank == param.root;
                }
                for (uint32_t rank : groups[1].remoteRanks) {
                    rootInSecond = rootInSecond || rank == param.root;
                }
                CHK_PRT_RET(rootInFirst == rootInSecond,
                    HCCL_ERROR("Root-link die group is not unique, rankSize[%u]",
                        param.rankSize),
                    HCCL_E_INTERNAL);
                if (!rootInFirst) {
                    std::swap(groups[0], groups[1]);
                }
            }
        }

        resCtxHost.threads.resize(groups.size());
        resCtxHost.threads[0] = param.cpuThread;
        if (groups.size() > 1) {
            CHK_RET(HcclThreadAcquire(comm, ccuEngine, static_cast<uint32_t>(groups.size() - 1), THREAD_NOTIFY_NUM, &resCtxHost.threads[1]));
        }

        CcuInsHandle insHandle = 0; uint32_t insNum = 0;
        CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));

        uint32_t algorithm = BCAST_ALG_DIRECT;
        if (useChainPipeline) {
            algorithm = BCAST_ALG_CHAIN_PIPELINE;
        } else if (use8P4) {
            algorithm = BCAST_ALG_8P4_PREFETCH_SAG;
        } else if (!useDirect) {
            algorithm = BCAST_ALG_SAG_SPLIT;
        }
        
        CHK_RET(RegisterKernels(param, insHandle, groups, algorithm, totalBytes, resCtxHost));

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seq.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}