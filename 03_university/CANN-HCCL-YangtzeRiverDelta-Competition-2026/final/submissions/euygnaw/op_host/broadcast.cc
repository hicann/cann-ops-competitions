#include <algorithm>
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
constexpr uint32_t DEFAULT_CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t CHAIN_CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t THREAD_NOTIFY_NUM = 4;
constexpr uint32_t MAX_DIE_KERNEL_NUM = 4;
constexpr uint32_t PIPELINE_CHUNK_COUNT = 26;
// 110463真实评测最优的4x1流水参数：8MiB块、四槽ACK回压。
constexpr uint64_t ACK_CHAIN_CHUNK_BYTES = 8ULL * 1024ULL * 1024ULL;
constexpr uint64_t SIZE_512M = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t SIZE_400M4 = 400ULL * 1024ULL * 1024ULL + 4ULL;

struct DieChannelGroup {
    uint32_t dieId = 0;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> remoteRanks;
};

HcclResult QueryEndpointDieId(HcclComm comm,
    uint32_t rankId,
    const EndpointDesc &endpoint,
    uint32_t &dieId)
{
    EndpointAttrDieId endpointDieId = 0;
    CHK_RET(HcclRankGraphGetEndpointInfo(comm,
        rankId,
        &endpoint,
        ENDPOINT_ATTR_DIE_ID,
        sizeof(endpointDieId),
        &endpointDieId));
    dieId = static_cast<uint32_t>(endpointDieId);
    return HCCL_SUCCESS;
}

HcclResult GetCcuChannelDesc(HcclComm comm,
    uint32_t myRank,
    uint32_t remoteRank,
    const uint32_t *netLayers,
    uint32_t netLayerNum,
    uint32_t notifyNum,
    HcclChannelDesc &desc,
    uint32_t &localDieId)
{
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(
            comm, netLayers[layerIdx], myRank, remoteRank, &links, &linkNum));

        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            const CommLink &link = links[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            HcclChannelDesc candidate{};
            CHK_RET(HcclChannelDescInit(&candidate, 1));
            candidate.remoteRank = remoteRank;
            candidate.notifyNum = notifyNum;
            candidate.channelProtocol = link.linkAttr.linkProtocol;
            candidate.localEndpoint = link.srcEndpointDesc;
            candidate.remoteEndpoint = link.dstEndpointDesc;

            CHK_RET(QueryEndpointDieId(comm, myRank, candidate.localEndpoint, localDieId));
            desc = candidate;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("No UBC_CTP link found from rank[%u] to rank[%u]", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult RegisterOneKernel(CcuInsHandle insHandle,
    uint32_t dieId,
    const char *name,
    void *kernelFunc,
    const std::shared_ptr<BroadcastCcuKernelArg> &kernelArg,
    std::vector<CcuKernelInfo> &kernelInfos,
    CcuKernelHandle &kernelHandle)
{
    kernelInfos.emplace_back();
    CcuKernelInfo &kernelInfo = kernelInfos.back();
    std::snprintf(kernelInfo.kernelFuncName,
        sizeof(kernelInfo.kernelFuncName),
        "%s",
        name);
    kernelInfo.kernelFunc = kernelFunc;
    kernelInfo.setKernelArg(kernelArg);

    constexpr uint32_t kernelArgNum = 1;
    const void *kernelArgs[] = {kernelInfo.kernelArg};
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle,
        dieId,
        kernelInfo.kernelFuncName,
        kernelInfo.kernelFunc,
        kernelArgs,
        kernelArgNum,
        &kernelHandle));
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(const OpParam &param,
    CcuInsHandle insHandle,
    const std::vector<DieChannelGroup> &groups,
    uint32_t algorithm,
    uint64_t totalBytes,
    AlgResourceCtx &resCtxHost)
{
    CHK_PRT_RET(groups.empty(), HCCL_ERROR("No die channel group"), HCCL_E_INTERNAL);
    CHK_PRT_RET(groups.size() > MAX_DIE_KERNEL_NUM,
        HCCL_ERROR("Too many die groups[%zu], max[%u]", groups.size(), MAX_DIE_KERNEL_NUM),
        HCCL_E_NOT_SUPPORT);

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
        kernelArg->directDataBytes = totalBytes;
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        for (uint32_t idx = 0; idx < group.channels.size(); ++idx) {
            kernelArg->channels[idx] = group.channels[idx];
            kernelArg->remoteRanks[idx] = group.remoteRanks[idx];
        }

        if (algorithm == BCAST_ALG_CHAIN_PIPELINE ||
            algorithm == BCAST_ALG_CHAIN_ACK_512) {
            const uint32_t relativeRank =
                (param.myRank + param.rankSize - param.root) % param.rankSize;
            if (relativeRank > 0) {
                kernelArg->prevRank = (param.myRank + param.rankSize - 1) % param.rankSize;
            }
            if (relativeRank + 1 < param.rankSize) {
                kernelArg->nextRank = (param.myRank + 1) % param.rankSize;
            }
            for (uint32_t idx = 0; idx < group.remoteRanks.size(); ++idx) {
                if (group.remoteRanks[idx] == kernelArg->prevRank) {
                    kernelArg->prevChannelIndex = idx;
                }
                if (group.remoteRanks[idx] == kernelArg->nextRank) {
                    kernelArg->nextChannelIndex = idx;
                }
            }
            if (algorithm == BCAST_ALG_CHAIN_ACK_512) {
                kernelArg->pipelineChunkBytes = ACK_CHAIN_CHUNK_BYTES;
                kernelArg->pipelineChunkCount = static_cast<uint32_t>(
                    (totalBytes + ACK_CHAIN_CHUNK_BYTES - 1ULL) / ACK_CHAIN_CHUNK_BYTES);
                kernelArg->pipelineLastChunkBytes =
                    totalBytes - static_cast<uint64_t>(kernelArg->pipelineChunkCount - 1U) *
                        ACK_CHAIN_CHUNK_BYTES;
            } else {
                kernelArg->pipelineChunkCount = PIPELINE_CHUNK_COUNT;
                kernelArg->pipelineChunkBytes =
                    ((totalBytes / PIPELINE_CHUNK_COUNT) / sizeof(float)) * sizeof(float);
                kernelArg->pipelineLastChunkBytes = 0;
            }
        }

        CcuKernelLaunchEntry entry{};
        entry.dieId = group.dieId;
        entry.threadIndex = groupIdx;
        entry.containsRootChannel = 0;
        entry.channelCount = static_cast<uint32_t>(group.remoteRanks.size());
        for (uint32_t idx = 0; idx < group.remoteRanks.size(); ++idx) {
            entry.remoteRanks[idx] = group.remoteRanks[idx];
            if (group.remoteRanks[idx] == param.root) {
                entry.containsRootChannel = 1;
            }
        }

        char kernelName[64]{};
        if (algorithm == BCAST_ALG_DIRECT) {
            const bool isRoot = param.myRank == param.root;
            std::snprintf(kernelName,
                sizeof(kernelName),
                isRoot ? "bcast_v50_direct_root_r%u_m%u_d%u"
                       : "bcast_v50_direct_peer_r%u_m%u_d%u",
                param.root,
                param.myRank,
                group.dieId);
            void *directKernel = nullptr;
            if (!isRoot) {
                directKernel = reinterpret_cast<void *>(ops_hccl::CcuBroadcastDirectPeerKernel);
            } else if (param.rankSize == 4) {
                directKernel = reinterpret_cast<void *>(ops_hccl::CcuBroadcastDirectRoot4Kernel);
            } else {
                directKernel = reinterpret_cast<void *>(ops_hccl::CcuBroadcastDirectRootKernel);
            }
            CHK_RET(RegisterOneKernel(insHandle,
                group.dieId,
                kernelName,
                directKernel,
                kernelArg,
                kernelInfos,
                entry.directHandle));
        } else if (algorithm == BCAST_ALG_SAG_FUSED) {
            std::snprintf(kernelName,
                sizeof(kernelName),
                "bcast_sag_fused_r%u_m%u_d%u",
                param.root,
                param.myRank,
                group.dieId);
            CHK_RET(RegisterOneKernel(insHandle,
                group.dieId,
                kernelName,
                reinterpret_cast<void *>(ops_hccl::CcuBroadcastScatterAllGatherKernel),
                kernelArg,
                kernelInfos,
                entry.fusedHandle));
        } else if (algorithm == BCAST_ALG_CHAIN_PIPELINE ||
            algorithm == BCAST_ALG_CHAIN_ACK_512) {
            const bool useAckChain = algorithm == BCAST_ALG_CHAIN_ACK_512;
            std::snprintf(kernelName,
                sizeof(kernelName),
                useAckChain ? "bcast_final_chain_async8m_r%u_m%u_d%u_n%u"
                            : "bcast_v90_chain_noack400_r%u_m%u_d%u_n%u",
                param.root,
                param.myRank,
                group.dieId,
                kernelArg->pipelineChunkCount);
            CHK_RET(RegisterOneKernel(insHandle,
                group.dieId,
                kernelName,
                reinterpret_cast<void *>(useAckChain
                    ? ops_hccl::CcuBroadcastChainAckKernel
                    : ops_hccl::CcuBroadcastChainNoAckKernel),
                kernelArg,
                kernelInfos,
                entry.chainHandle));
        } else if (algorithm == BCAST_ALG_8P4_PREFETCH_SAG) {
            std::snprintf(kernelName,
                sizeof(kernelName),
                "bcast_8p4_prefetch_scatter_r%u_m%u_d%u",
                param.root,
                param.myRank,
                group.dieId);
            CHK_RET(RegisterOneKernel(insHandle,
                group.dieId,
                kernelName,
                reinterpret_cast<void *>(ops_hccl::CcuBroadcast8p4ScatterPrefetchKernel),
                kernelArg,
                kernelInfos,
                entry.scatterHandle));

            std::snprintf(kernelName,
                sizeof(kernelName),
                "bcast_8p4_skiproot_ag_r%u_m%u_d%u",
                param.root,
                param.myRank,
                group.dieId);
            CHK_RET(RegisterOneKernel(insHandle,
                group.dieId,
                kernelName,
                reinterpret_cast<void *>(ops_hccl::CcuBroadcast8p4AllGatherSkipRootKernel),
                kernelArg,
                kernelInfos,
                entry.allGatherHandle));
        } else {
            std::snprintf(kernelName,
                sizeof(kernelName),
                "bcast_scatter_r%u_m%u_d%u",
                param.root,
                param.myRank,
                group.dieId);
            CHK_RET(RegisterOneKernel(insHandle,
                group.dieId,
                kernelName,
                reinterpret_cast<void *>(ops_hccl::CcuBroadcastScatterKernel),
                kernelArg,
                kernelInfos,
                entry.scatterHandle));

            std::snprintf(kernelName,
                sizeof(kernelName),
                "bcast_allgather_r%u_m%u_d%u",
                param.root,
                param.myRank,
                group.dieId);
            CHK_RET(RegisterOneKernel(insHandle,
                group.dieId,
                kernelName,
                reinterpret_cast<void *>(ops_hccl::CcuBroadcastAllGatherKernel),
                kernelArg,
                kernelInfos,
                entry.allGatherHandle));
        }

        resCtxHost.ccuKernelEntries.push_back(entry);
    }

    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rankSize[%u]", param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("Invalid root[%u], rankSize[%u]", param.root, param.rankSize),
        HCCL_E_PARA);

    const auto typeSizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeSizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported dataType[%d]", static_cast<int32_t>(param.dataType)),
        HCCL_E_PARA);
    const uint64_t typeSize = typeSizeIter->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("Data size overflow, count[%llu]", static_cast<unsigned long long>(param.count)),
        HCCL_E_PARA);
    const uint64_t totalBytes = param.count * typeSize;

    if (param.rankSize <= 1 || totalBytes == 0) {
        return HCCL_SUCCESS;
    }

    const bool useDirect = totalBytes <= DIRECT_ALG_MAX_BYTES;
    const bool useChainAck512 = param.rankSize == 4 && totalBytes == SIZE_512M;
    const bool useChainNoAck400 = param.rankSize == 4 && totalBytes == SIZE_400M4;
    const bool useChainPipeline = useChainAck512 || useChainNoAck400;
    if (useChainPipeline) {
        std::snprintf(param.tag,
            sizeof(param.tag),
            "hccl_bcast_ccu_final_chain_async8m_r%u_b%llu",
            root,
            static_cast<unsigned long long>(totalBytes));
    } else if (useDirect) {
        std::snprintf(param.tag,
            sizeof(param.tag),
            "hccl_bcast_ccu_v90_direct_r%u_b%llu",
            root,
            static_cast<unsigned long long>(totalBytes));
    } else {
        // 大数据2x8/8+4保持100865稳定版的SAG任务图与资源标签。
        std::snprintf(param.tag,
            sizeof(param.tag),
            "hccl_bcast_ccu_v34_sag_r%u",
            root);
    }

    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, ccuEngine, stream, THREAD_NOTIFY_NUM, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        resCtxHost.ccuThread = param.cpuThread;

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // 小数据保持已经拿到最优延迟的root直接扇出；大数据建立全Mesh，
        // 用Scatter + AllGather消除root的N-1倍完整数据出口瓶颈。
        std::vector<uint32_t> remoteRanks;
        if (useDirect) {
            if (param.myRank == param.root) {
                remoteRanks.reserve(param.rankSize - 1);
                for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                    if (rank != param.root) {
                        remoteRanks.push_back(rank);
                    }
                }
            } else {
                remoteRanks.push_back(param.root);
            }
        } else if (useChainPipeline) {
            // 4x1全为layer-1 Clos链路，按root起点构造单向链：
            // root -> root+1 -> root+2 -> root+3（逻辑rank取模）。
            const uint32_t relativeRank =
                (param.myRank + param.rankSize - param.root) % param.rankSize;
            if (relativeRank > 0) {
                remoteRanks.push_back((param.myRank + param.rankSize - 1) % param.rankSize);
            }
            if (relativeRank + 1 < param.rankSize) {
                remoteRanks.push_back((param.myRank + 1) % param.rankSize);
            }
        } else {
            remoteRanks.reserve(param.rankSize - 1);
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (rank != param.myRank) {
                    remoteRanks.push_back(rank);
                }
            }
        }

        uint32_t *netLayers = nullptr;
        uint32_t netLayerNum = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        CHK_PRT_RET(netLayerNum == 0 || netLayers == nullptr,
            HCCL_ERROR("No rank graph layer found"),
            HCCL_E_NOT_FOUND);

        std::vector<HcclChannelDesc> channelDescs(remoteRanks.size());
        std::vector<uint32_t> localDieIds(remoteRanks.size(), 0);
        for (uint32_t idx = 0; idx < remoteRanks.size(); ++idx) {
            CHK_RET(GetCcuChannelDesc(comm,
                param.myRank,
                remoteRanks[idx],
                netLayers,
                netLayerNum,
                useChainPipeline ? CHAIN_CHANNEL_NOTIFY_NUM : DEFAULT_CHANNEL_NOTIFY_NUM,
                channelDescs[idx],
                localDieIds[idx]));
        }

        std::vector<ChannelHandle> channels(remoteRanks.size());
        CHK_RET(HcclChannelAcquire(comm,
            ccuEngine,
            channelDescs.data(),
            static_cast<uint32_t>(channelDescs.size()),
            channels.data()));

        std::map<uint32_t, DieChannelGroup> groupMap;
        for (uint32_t idx = 0; idx < channels.size(); ++idx) {
            auto &group = groupMap[localDieIds[idx]];
            group.dieId = localDieIds[idx];
            group.channels.push_back(channels[idx]);
            group.remoteRanks.push_back(remoteRanks[idx]);
        }

        std::vector<DieChannelGroup> groups;
        groups.reserve(groupMap.size());
        for (auto &item : groupMap) {
            groups.push_back(std::move(item.second));
        }
        // Direct多Die路径：较大Channel组放worker先启动，较小组留给main最后下发。
        if (useDirect && groups.size() > 1) {
            std::stable_sort(groups.begin(),
                groups.end(),
                [](const DieChannelGroup &lhs, const DieChannelGroup &rhs) {
                    return lhs.channels.size() < rhs.channels.size();
                });
        }
        CHK_PRT_RET(groups.empty() || groups.size() > MAX_DIE_KERNEL_NUM,
            HCCL_ERROR("Invalid die group count[%zu]", groups.size()),
            HCCL_E_NOT_SUPPORT);

        resCtxHost.threads.resize(groups.size());
        resCtxHost.threads[0] = param.cpuThread;
        if (groups.size() > 1) {
            CHK_RET(HcclThreadAcquire(comm,
                ccuEngine,
                static_cast<uint32_t>(groups.size() - 1),
                THREAD_NOTIFY_NUM,
                &resCtxHost.threads[1]));
        }

        CcuInsHandle insHandle = 0;
        uint32_t insNum = 0;
        CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
        CHK_PRT_RET(insNum != 1 || insHandle == 0,
            HCCL_ERROR("Expected exactly one CCU instance, insNum[%u]", insNum),
            HCCL_E_INTERNAL);

        uint32_t algorithm = BCAST_ALG_DIRECT;
        if (useChainPipeline) {
            CHK_PRT_RET(groups.size() != 1,
                HCCL_ERROR("4x1 chain pipeline requires one die group, got[%zu]", groups.size()),
                HCCL_E_NOT_SUPPORT);
            // 512MiB和400MiB+4B都使用110463已实测的8MiB异步四槽链。
            algorithm = BCAST_ALG_CHAIN_ACK_512;
        } else if (!useDirect && param.rankSize == 12) {
            CHK_PRT_RET(groups.size() != 2,
                HCCL_ERROR("8+4 prefetch SAG requires two die groups, got[%zu]", groups.size()),
                HCCL_E_NOT_SUPPORT);
            algorithm = BCAST_ALG_8P4_PREFETCH_SAG;
        } else if (!useDirect) {
            algorithm = BCAST_ALG_SAG_SPLIT;
        }
        CHK_RET(RegisterKernels(
            param, insHandle, groups, algorithm, totalBytes, resCtxHost));

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seq.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
