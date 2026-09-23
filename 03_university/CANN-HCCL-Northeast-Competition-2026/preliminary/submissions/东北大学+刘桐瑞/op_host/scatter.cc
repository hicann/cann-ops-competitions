

#include <vector>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <cstring>

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr CommProtocol PROTOCOL_UB_CTP = static_cast<CommProtocol>(4);

const CommProtocol AICPU_TS_PROTOCOLS[] = {
    PROTOCOL_UB_CTP,
    CommProtocol::COMM_PROTOCOL_UBOE,
};
constexpr uint32_t AICPU_TS_PROTOCOL_NUM = sizeof(AICPU_TS_PROTOCOLS) / sizeof(AICPU_TS_PROTOCOLS[0]);

struct RelayPlan {
    bool enabled = false;
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> remoteRanks;
    uint32_t rootPos = 0;
};

HcclResult BuildRelayPlan(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &layers, RelayPlan &plan)
{
    plan.enabled = false;
    if (SCATTER_RELAY_DISABLE != 0 || param.rankSize <= 2 || layers.empty()) {
        return HCCL_SUCCESS;
    }

    uint32_t *ranks = nullptr;
    uint32_t num = 0;
    if (HcclRankGraphGetRanksByLayer(comm, layers[0], &ranks, &num) != HCCL_SUCCESS || ranks == nullptr || num == 0) {
        HCCL_WARNING("[BuildRelayPlan] cannot query layer[%u] ranks, fall back to direct scatter", layers[0]);
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> myServer(ranks, ranks + num);
    std::sort(myServer.begin(), myServer.end());

    if (static_cast<uint64_t>(num) * 2 != static_cast<uint64_t>(param.rankSize)) {
        HCCL_INFO("[BuildRelayPlan] not a 2-server topology (serverSize[%u] rankSize[%u]), use direct scatter", num,
            param.rankSize);
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> otherServer;
    otherServer.reserve(num);
    for (uint32_t r = 0; r < param.rankSize; r++) {
        if (!std::binary_search(myServer.begin(), myServer.end(), r)) {
            otherServer.push_back(r);
        }
    }
    if (otherServer.size() != num) {
        return HCCL_SUCCESS;
    }

    const bool rootInMyServer = std::binary_search(myServer.begin(), myServer.end(), param.root);
    plan.localRanks = rootInMyServer ? myServer : otherServer;
    plan.remoteRanks = rootInMyServer ? otherServer : myServer;
    const auto it = std::find(plan.localRanks.begin(), plan.localRanks.end(), param.root);
    if (it == plan.localRanks.end()) {
        return HCCL_SUCCESS;
    }
    plan.rootPos = static_cast<uint32_t>(std::distance(plan.localRanks.begin(), it));
    plan.enabled = true;
    HCCL_INFO("[BuildRelayPlan] relay enabled: serverSize[%u] rootPos[%u]", num, plan.rootPos);
    return HCCL_SUCCESS;
}

uint32_t FindRelayPeer(const RelayPlan &plan, const OpParam &param, uint32_t &role)
{
    role = SCATTER_ROLE_NONE;
    if (!plan.enabled || param.myRank == param.root) {
        return INVALID_VALUE_RANKID;
    }
    const auto lit = std::find(plan.localRanks.begin(), plan.localRanks.end(), param.myRank);
    if (lit != plan.localRanks.end()) {
        const uint32_t pos = static_cast<uint32_t>(std::distance(plan.localRanks.begin(), lit));
        if (pos == plan.rootPos) {
            return INVALID_VALUE_RANKID;
        }
        role = SCATTER_ROLE_FORWARD;
        return plan.remoteRanks[pos];
    }
    const auto rit = std::find(plan.remoteRanks.begin(), plan.remoteRanks.end(), param.myRank);
    if (rit != plan.remoteRanks.end()) {
        const uint32_t pos = static_cast<uint32_t>(std::distance(plan.remoteRanks.begin(), rit));
        if (pos == plan.rootPos) {
            return INVALID_VALUE_RANKID;
        }
        role = SCATTER_ROLE_RECV_RELAY;
        return plan.localRanks[pos];
    }
    return INVALID_VALUE_RANKID;
}

void FillRootSendPlan(const RelayPlan &plan, ChannelInfo &channel)
{
    channel.sendFullShare = 1;
    channel.relayForRank = INVALID_VALUE_RANKID;
    if (!plan.enabled) {
        return;
    }
    const auto lit = std::find(plan.localRanks.begin(), plan.localRanks.end(), channel.remoteRank);
    if (lit != plan.localRanks.end()) {
        const uint32_t pos = static_cast<uint32_t>(std::distance(plan.localRanks.begin(), lit));
        if (pos != plan.rootPos) {
            channel.relayForRank = plan.remoteRanks[pos];
        }
        return;
    }
    const auto rit = std::find(plan.remoteRanks.begin(), plan.remoteRanks.end(), channel.remoteRank);
    if (rit != plan.remoteRanks.end()) {
        const uint32_t pos = static_cast<uint32_t>(std::distance(plan.remoteRanks.begin(), rit));
        channel.sendFullShare = (pos == plan.rootPos) ? 1U : 0U;
    }
}

uint32_t CalcThreadNum(const OpParam &param, const RelayPlan &plan, uint32_t relayRole)
{
    if (param.rankSize <= 1) {
        return 1;
    }
    if (param.myRank != param.root) {
        return (relayRole == SCATTER_ROLE_NONE) ? 1 : 2;
    }

    if (!plan.enabled || plan.localRanks.empty()) {
        return param.rankSize - 1;
    }
    const uint32_t meshNum = static_cast<uint32_t>(plan.localRanks.size()) - 1;
    uint32_t closThreads = static_cast<uint32_t>(plan.remoteRanks.size());
    if (closThreads > SCATTER_CLOS_THREAD_NUM) {
        closThreads = SCATTER_CLOS_THREAD_NUM;
    }
    const uint32_t total = meshNum + closThreads;
    return (total == 0) ? 1 : total;
}

void AssignChannelThreads(
    const OpParam &param, const RelayPlan &plan, uint32_t threadNum, std::vector<ChannelInfo> &channels)
{
    const uint32_t chNum = static_cast<uint32_t>(channels.size());
    if (param.myRank != param.root) {
        for (uint32_t idx = 0; idx < chNum; idx++) {
            channels[idx].threadIdx = (idx == 0) ? 0U : 1U;
        }
        return;
    }
    if (!plan.enabled || plan.localRanks.empty() || threadNum == 0) {
        for (uint32_t idx = 0; idx < chNum; idx++) {
            channels[idx].threadIdx = (threadNum == 0) ? 0U : (idx % threadNum);
        }
        return;
    }
    const uint32_t meshNum = static_cast<uint32_t>(plan.localRanks.size()) - 1;
    const uint32_t closThreads = (threadNum > meshNum) ? (threadNum - meshNum) : 1U;

    const uint32_t fullThread = meshNum;
    const uint32_t rrBase = (closThreads > 1) ? (meshNum + 1) : meshNum;
    const uint32_t rrNum = (closThreads > 1) ? (closThreads - 1) : 1U;
    uint32_t meshSeen = 0;
    uint32_t closSeen = 0;
    for (uint32_t idx = 0; idx < chNum; idx++) {
        const bool sameServer
            = std::binary_search(plan.localRanks.begin(), plan.localRanks.end(), channels[idx].remoteRank);
        if (sameServer && meshSeen < meshNum) {
            channels[idx].threadIdx = meshSeen;
            meshSeen++;
        } else if (channels[idx].sendFullShare != 0) {
            channels[idx].threadIdx = fullThread;
        } else {
            channels[idx].threadIdx = rrBase + (closSeen % rrNum);
            closSeen++;
        }
    }
    HCCL_INFO("[AssignChannelThreads] threadNum[%u] meshNum[%u] closThreads[%u] mesh[%u] closRR[%u]", threadNum,
        meshNum, closThreads, meshSeen, closSeen);
}

HcclResult GetNetLayers(HcclComm comm, std::vector<uint32_t> &layers)
{
    uint32_t *layerPtr = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerPtr, &layerNum));
    CHK_PTR_NULL(layerPtr);
    CHK_PRT_RET(
        layerNum == 0, HCCL_ERROR("[GetNetLayers] net layer list is empty, layerNum[%u]", layerNum), HCCL_E_INTERNAL);

    layers.assign(layerPtr, layerPtr + layerNum);

    std::sort(layers.begin(), layers.end());
    return HCCL_SUCCESS;
}

HcclResult FillChannelDesc(HcclComm comm, const std::vector<uint32_t> &layers, uint32_t myRank, uint32_t dstRank,
    uint32_t rotate, HcclChannelDesc &desc)
{
    for (const uint32_t netLayer : layers) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, dstRank, &linkList, &listSize));
        if (linkList == nullptr || listSize == 0) {
            continue;
        }

        const std::vector<CommLink> links(linkList, linkList + listSize);

        for (uint32_t protoIdx = 0; protoIdx < AICPU_TS_PROTOCOL_NUM; protoIdx++) {
            const CommProtocol wantProtocol = AICPU_TS_PROTOCOLS[protoIdx];

            uint32_t matchNum = 0;
            for (const CommLink &link : links) {
                if (link.linkAttr.linkProtocol == wantProtocol) {
                    matchNum++;
                }
            }
            if (matchNum == 0) {
                continue;
            }
            const uint32_t pick = SCATTER_LINK_ROTATE ? (rotate % matchNum) : 0;
            uint32_t seen = 0;
            for (const CommLink &link : links) {
                if (link.linkAttr.linkProtocol != wantProtocol) {
                    continue;
                }
                if (seen++ != pick) {
                    continue;
                }
                CHK_RET(HcclChannelDescInit(&desc, 1));
                desc.remoteRank = dstRank;
                desc.notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                HCCL_INFO(
                    "[FillChannelDesc] rank[%u]->rank[%u] netLayer[%u] protocol[%d] links[%u] match[%u] pick[%u]",
                    myRank, dstRank, netLayer, static_cast<int32_t>(wantProtocol), listSize, matchNum, pick);
                return HCCL_SUCCESS;
            }
        }
        HCCL_WARNING("[FillChannelDesc] rank[%u]->rank[%u] netLayer[%u] has %u link(s) but none uses an "
                     "AICPU_TS-capable protocol, fall through to next layer",
            myRank, dstRank, netLayer, listSize);
    }

    HCCL_ERROR("[FillChannelDesc] no usable link between rank[%u] and rank[%u]", myRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

void BuildPeerList(
    const OpParam &param, const RelayPlan &plan, std::vector<uint32_t> &peers, uint32_t &relayPeer, uint32_t &role)
{
    peers.clear();
    relayPeer = INVALID_VALUE_RANKID;
    role = SCATTER_ROLE_NONE;
    if (param.rankSize <= 1) {
        return;
    }
    if (param.myRank == param.root) {
        peers.reserve(param.rankSize - 1);

        for (uint32_t step = 1; step < param.rankSize; step++) {
            peers.push_back((param.root + step) % param.rankSize);
        }
        return;
    }
    peers.push_back(param.root);
    relayPeer = FindRelayPeer(plan, param, role);
    if (relayPeer != INVALID_VALUE_RANKID) {
        peers.push_back(relayPeer);
    }
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const RelayPlan &plan,
    const std::vector<uint32_t> &peers, const std::vector<uint32_t> &layers, AlgResourceCtx &resCtxHost)
{
    if (peers.empty()) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> descs(peers.size());
    for (uint32_t idx = 0; idx < peers.size(); idx++) {
        CHK_RET(FillChannelDesc(comm, layers, param.myRank, peers[idx], peers[idx], descs[idx]));
    }

    std::vector<ChannelHandle> handles(peers.size(), 0);
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), static_cast<uint32_t>(descs.size()), handles.data()));

    const uint64_t needBytes = ScatterLandingBytes(resCtxHost.slotBytes);
    resCtxHost.channels.resize(peers.size());
    for (uint32_t idx = 0; idx < peers.size(); idx++) {
        ChannelInfo &channel = resCtxHost.channels[idx];
        channel.remoteRank = peers[idx];
        channel.notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
        channel.handle = handles[idx];

        void *remoteBuf = nullptr;
        uint64_t remoteSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[idx], &remoteBuf, &remoteSize));
        CHK_PTR_NULL(remoteBuf);

        CHK_PRT_RET(remoteSize < needBytes,
            HCCL_ERROR("[AcquireChannels] remote rank[%u] hccl buffer too small, size[%llu] need[%llu]", peers[idx],
                static_cast<unsigned long long>(remoteSize), static_cast<unsigned long long>(needBytes)),
            HCCL_E_INTERNAL);
        channel.remoteCclMem = CommBuffer{remoteBuf, remoteSize};
        if (param.myRank == param.root) {
            FillRootSendPlan(plan, channel);
        }
    }
    return HCCL_SUCCESS;
}

uint64_t CalcSlotBytes(uint64_t cclBufferSize)
{
    uint64_t perSlot = cclBufferSize / (SCATTER_REGION_NUM * SCATTER_SLOT_NUM);
    perSlot = std::min(perSlot, SCATTER_SLOT_MAX_BYTES);
    perSlot = perSlot / SCATTER_SLOT_ALIGN * SCATTER_SLOT_ALIGN;
    return perSlot;
}

struct HostCallCache {
    HcclComm comm = nullptr;
    uint32_t root = INVALID_VALUE_RANKID;
    aclrtStream stream = nullptr;

    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};

    void *resCtx = nullptr;
    uint64_t ctxSize = 0;
    ThreadHandle aicpuThread{};
    ThreadHandle aicpuThreadOnCpu{};

    ThreadHandle cpuThread{};
    ThreadHandle cpuThreadOnAicpu{};
};
thread_local HostCallCache g_hostCache;

inline bool HostCacheHit(HcclComm comm, uint32_t root, aclrtStream stream)
{
#if SCATTER_CACHE_HOST_THREADS
    return g_hostCache.resCtx != nullptr && g_hostCache.comm == comm && g_hostCache.root == root
        && g_hostCache.stream == stream;
#else
    (void)comm;
    (void)root;
    (void)stream;
    return false;
#endif
}

struct HostTagCache {
    uint32_t root = INVALID_VALUE_RANKID;
    uint32_t len = 0;
    char text[TAG_LENGTH] = {0};
};
thread_local HostTagCache g_tagCache;

HcclResult BuildOpTag(OpParam &param)
{
    if (g_tagCache.len > 0 && g_tagCache.root == param.root) {
        (void)memcpy(param.tag, g_tagCache.text, g_tagCache.len + 1);
        return HCCL_SUCCESS;
    }
    const int32_t tagRet = snprintf(param.tag, sizeof(param.tag), "hccl_custom_scatter_root%u", param.root);
    CHK_PRT_RET(tagRet <= 0 || static_cast<uint32_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("[HcclScatter] failed to build tag, ret[%d]", tagRet), HCCL_E_INTERNAL);
    g_tagCache.root = param.root;
    g_tagCache.len = static_cast<uint32_t>(tagRet);
    (void)memcpy(g_tagCache.text, param.tag, g_tagCache.len + 1);
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

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    HcclDfxOpInfo dfxInfo;

    if (HostCacheHit(comm, param.root, stream)) {
        CHK_RET(HcclDfxRegOpInfoByCommId(g_hostCache.commName, reinterpret_cast<void *>(&dfxInfo)));
        param.myRank = g_hostCache.myRank;
        param.rankSize = g_hostCache.rankSize;
        param.cpuThread = g_hostCache.cpuThread;
        param.cpuThreadOnAicpu = g_hostCache.cpuThreadOnAicpu;
        param.aicpuThreadOnCpu = g_hostCache.aicpuThreadOnCpu;
        param.resCtx = g_hostCache.resCtx;
        param.ctxSize = g_hostCache.ctxSize;

        CHK_PRT_RET(param.root >= param.rankSize,
            HCCL_ERROR("[HcclScatter] root[%u] out of range, rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA);
        CHK_PRT_RET(ScatterDataTypeSize(param.dataType) == 0,
            HCCL_ERROR("[HcclScatter] unsupported dataType[%d]", static_cast<int32_t>(param.dataType)),
            HCCL_E_NOT_SUPPORT);
        CHK_RET(BuildOpTag(param));
        return ops_hccl::LaunchAICPUKernel(param, stream);
    }

    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("[HcclScatter] root[%u] out of range, rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA);

    CHK_PRT_RET(ScatterDataTypeSize(param.dataType) == 0,
        HCCL_ERROR("[HcclScatter] unsupported dataType[%d]", static_cast<int32_t>(param.dataType)), HCCL_E_NOT_SUPPORT);

    CHK_RET(BuildOpTag(param));

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    ThreadHandle cachedAicpuThread{};
    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
        cachedAicpuThread = *aicpuThread;
    } else {
        AlgResourceCtx resCtxHost;

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));

        CHK_PTR_NULL(cclBufferAddr);
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        resCtxHost.slotBytes = CalcSlotBytes(cclBufferSize);
        CHK_PRT_RET(resCtxHost.slotBytes == 0,
            HCCL_ERROR("[HcclScatter] hccl buffer too small, size[%llu]",
                static_cast<unsigned long long>(cclBufferSize)),
            HCCL_E_INTERNAL);

        std::vector<uint32_t> layers;
        CHK_RET(GetNetLayers(comm, layers));

        RelayPlan plan;
        CHK_RET(BuildRelayPlan(comm, param, layers, plan));

        std::vector<uint32_t> peers;
        uint32_t relayPeer = INVALID_VALUE_RANKID;
        uint32_t relayRole = SCATTER_ROLE_NONE;
        BuildPeerList(param, plan, peers, relayPeer, relayRole);

        resCtxHost.relayRole = relayRole;
        resCtxHost.rootChannelIdx = 0;
        resCtxHost.relayChannelIdx = (relayRole == SCATTER_ROLE_NONE) ? 0 : 1;
        resCtxHost.recvFromRootFull = (relayRole == SCATTER_ROLE_RECV_RELAY) ? 0U : 1U;

        uint32_t threadNum = CalcThreadNum(param, plan, relayRole);
        uint32_t notifyNumPerThread = std::max(threadNum, 2U);

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));

        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquireChannels(comm, param, plan, peers, layers, resCtxHost));

        AssignChannelThreads(param, plan, threadNum, resCtxHost.channels);

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
        cachedAicpuThread = resCtxHost.aicpuThread;
    }

    g_hostCache.comm = comm;
    g_hostCache.root = param.root;
    g_hostCache.stream = stream;
    g_hostCache.myRank = param.myRank;
    g_hostCache.rankSize = param.rankSize;
    (void)memcpy(g_hostCache.commName, commName, sizeof(g_hostCache.commName));
    g_hostCache.resCtx = param.resCtx;
    g_hostCache.ctxSize = param.ctxSize;
    g_hostCache.aicpuThread = cachedAicpuThread;
    g_hostCache.aicpuThreadOnCpu = param.aicpuThreadOnCpu;
    g_hostCache.cpuThread = param.cpuThread;
    g_hostCache.cpuThreadOnAicpu = param.cpuThreadOnAicpu;

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
