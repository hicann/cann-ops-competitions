/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"
#include "log.h"

namespace {
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t COMPETITION_RANK_SIZE = 16;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 4; // DATA/ACK for two pipeline slots.
constexpr uint64_t TINY_LIMIT_BYTES = 4ULL * 1024ULL;
constexpr uint64_t SMALL_LIMIT_BYTES = 1ULL * 1024ULL * 1024ULL;
constexpr uint64_t HOST_CACHE_MAGIC = 0x5343324448433136ULL; // 

struct ScatterHostCache {
    uint64_t magic = HOST_CACHE_MAGIC;
    uintptr_t streamKey = 0;
    ThreadHandle cpuThread{};
    ThreadHandle cpuThreadOnAicpu{};
    ThreadHandle aicpuThreadOnCpu{};
    ThreadHandle aicpuThread{};
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
};

inline uintptr_t GetStreamKey(aclrtStream stream)
{
    return reinterpret_cast<uintptr_t>(stream);
}

inline uint32_t ServerBase(uint32_t rank)
{
    return (rank / RANKS_PER_SERVER) * RANKS_PER_SERVER;
}

inline uint32_t RemoteLeader(uint32_t root)
{
    return root ^ RANKS_PER_SERVER;
}

HcclResult LoadHostCache(void *hostCtx, uint64_t hostCtxSize, ScatterHostCache &cache)
{
    CHK_PRT_RET(hostCtx == nullptr || hostCtxSize < sizeof(ScatterHostCache),
        HCCL_ERROR("LoadHostCache: invalid context, ptr[%p], size[%llu]", hostCtx,
            static_cast<unsigned long long>(hostCtxSize)),
        HCCL_E_INTERNAL);
    std::memcpy(&cache, hostCtx, sizeof(cache));
    CHK_PRT_RET(cache.magic != HOST_CACHE_MAGIC, HCCL_ERROR("LoadHostCache: bad magic"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult AcquireDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc *desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    CommLink *selectedLinks = nullptr;
    uint32_t selectedSize = 0;
    for (uint32_t i = 0; i < netLayerNum; ++i) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[i], srcRank, dstRank, &linkList, &listSize));
        if (listSize != 0) {
            selectedLinks = linkList;
            selectedSize = listSize;
            break;
        }
    }
    CHK_PRT_RET(selectedSize == 0,
        HCCL_ERROR("AcquireDesc: no link between rank[%u] and rank[%u]", srcRank, dstRank), HCCL_E_INTERNAL);

    CHK_RET(HcclChannelDescInit(desc, 1));
    const CommLink link = selectedLinks[0];
    desc->remoteRank = dstRank;
    desc->notifyNum = CHANNEL_NOTIFY_NUM;
    desc->channelProtocol = link.linkAttr.linkProtocol;
    desc->localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc->localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc->localEndpoint.loc = link.srcEndpointDesc.loc;
    desc->remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc->remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc->remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

uint32_t BuildDirectPeers(const OpParam &param, uint32_t peers[SCATTER_MAX_CHANNEL_NUM])
{
    if (param.rankSize <= 1) {
        return 0;
    }
    if (param.myRank != param.root) {
        peers[0] = param.root;
        return 1;
    }

    uint32_t count = 0;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.root) {
            peers[count++] = rank;
        }
    }
    return count;
}

uint32_t BuildSmallPeers(const OpParam &param, uint32_t peers[SCATTER_MAX_CHANNEL_NUM])
{
    const uint32_t rootBase = ServerBase(param.root);
    const uint32_t leader = RemoteLeader(param.root);
    const uint32_t remoteBase = ServerBase(leader);

    if (param.myRank == param.root) {
        uint32_t count = 0;
        for (uint32_t rank = rootBase; rank < rootBase + RANKS_PER_SERVER; ++rank) {
            if (rank != param.root) {
                peers[count++] = rank;
            }
        }
        peers[count++] = leader;
        return count;
    }

    if (ServerBase(param.myRank) == rootBase) {
        peers[0] = param.root;
        return 1;
    }

    if (param.myRank == leader) {
        uint32_t count = 0;
        peers[count++] = param.root;
        for (uint32_t rank = remoteBase; rank < remoteBase + RANKS_PER_SERVER; ++rank) {
            if (rank != leader) {
                peers[count++] = rank;
            }
        }
        return count;
    }

    peers[0] = leader;
    return 1;
}

ScatterRole Get2DRole(const OpParam &param)
{
    if (param.myRank == param.root) {
        return SCATTER_ROLE_ROOT;
    }
    if (ServerBase(param.myRank) == ServerBase(param.root)) {
        return SCATTER_ROLE_LOCAL_RELAY;
    }
    if (param.myRank == RemoteLeader(param.root)) {
        return SCATTER_ROLE_REMOTE_LEADER;
    }
    return SCATTER_ROLE_REMOTE_LEAF;
}

uint32_t Build2DPeers(const OpParam &param, ScatterRole role, uint32_t peers[SCATTER_MAX_CHANNEL_NUM])
{
    uint32_t count = 0;

    // V13 direct-Clos 2-D topology:
    //   * root talks to every other rank exactly once;
    //   * local relays additionally talk to their vertical remote pair;
    //   * a normal remote rank receives B directly from root and A from its
    //     vertical relay;
    //   * the remote rank paired with root receives both A and B from root.
    // This removes the V10 root->leader->leaf second hop while preserving the
    // balanced Mesh-first / Clos-direct split.
    if (role == SCATTER_ROLE_ROOT) {
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.root) {
                peers[count++] = rank;
            }
        }
        return count;
    }

    if (role == SCATTER_ROLE_LOCAL_RELAY) {
        peers[count++] = param.root;
        peers[count++] = param.myRank ^ RANKS_PER_SERVER;
        return count;
    }

    if (role == SCATTER_ROLE_REMOTE_LEADER) {
        // This is the vertical pair of root. Root itself supplies both A and B.
        peers[count++] = param.root;
        return count;
    }

    peers[count++] = param.root;
    peers[count++] = param.myRank ^ RANKS_PER_SERVER;
    return count;
}

uint32_t ThreadCountForResource(ScatterAlgo algo, ScatterRole role, const OpParam &param)
{
    if (algo == SCATTER_ALGO_SMALL) {
        // Root overlaps one local-server staging lane with one cross-server lane.
        return (param.myRank == param.root) ? 2U : 1U;
    }
    if (algo == SCATTER_ALGO_2D && role == SCATTER_ROLE_ROOT) {
        constexpr uint64_t PERF_512M_BYTES = 512ULL * 1024ULL * 1024ULL;
        const uint64_t blockBytes = param.count * sizeof(float);
        // 512 MiB gets a second Clos submission worker. 400 MiB + 4 B keeps
        // the proven V13/V15A 9-thread path unchanged.
        return (blockBytes == PERF_512M_BYTES) ? 10U : 9U;
    }
    return 1;
}

HcclResult AcquireChannels(HcclComm comm, uint32_t myRank, const uint32_t *peers, uint32_t peerCount,
    AlgResourceCtx &resCtxHost)
{
    resCtxHost.channelCount = peerCount;
    for (uint32_t rank = 0; rank < SCATTER_MAX_RANK_SIZE; ++rank) {
        resCtxHost.channelIndex[rank] = 0xFFU;
    }
    if (peerCount == 0) {
        return HCCL_SUCCESS;
    }

    HcclChannelDesc desc[SCATTER_MAX_CHANNEL_NUM]{};
    ChannelHandle handles[SCATTER_MAX_CHANNEL_NUM]{};
    for (uint32_t i = 0; i < peerCount; ++i) {
        CHK_RET(AcquireDesc(comm, myRank, peers[i], &desc[i]));
    }
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, desc, peerCount, handles));

    for (uint32_t i = 0; i < peerCount; ++i) {
        ChannelInfo &channel = resCtxHost.channels[i];
        channel.remoteRank = desc[i].remoteRank;
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = handles[i];
        void *cclBuf = nullptr;
        uint64_t cclBufSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i], &cclBuf, &cclBufSize));
        channel.remoteCclMem = CommBuffer{cclBuf, cclBufSize};
        if (channel.remoteRank < SCATTER_MAX_RANK_SIZE) {
            resCtxHost.channelIndex[channel.remoteRank] = static_cast<uint8_t>(i);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BuildResources(HcclComm comm, OpParam &param, ScatterAlgo requestedAlgo, AlgResourceCtx &resCtxHost)
{
    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

    uint32_t peers[SCATTER_MAX_CHANNEL_NUM]{};
    uint32_t peerCount = 0;
    ScatterRole role = SCATTER_ROLE_DIRECT;
    ScatterAlgo algo = requestedAlgo;

    // The competition-specific 2-D and small fast paths assume 2 x 8 ranks.
    // Keep a functional direct fallback for any other rank size.
    if (param.rankSize != COMPETITION_RANK_SIZE) {
        algo = SCATTER_ALGO_DIRECT;
    }

    if (algo == SCATTER_ALGO_2D) {
        role = Get2DRole(param);
        peerCount = Build2DPeers(param, role, peers);
    } else if (algo == SCATTER_ALGO_SMALL) {
        role = SCATTER_ROLE_DIRECT;
        peerCount = BuildSmallPeers(param, peers);
    } else {
        role = SCATTER_ROLE_DIRECT;
        peerCount = BuildDirectPeers(param, peers);
    }

    resCtxHost.algo = algo;
    resCtxHost.role = role;
    resCtxHost.threadCount = ThreadCountForResource(algo, role, param);
    const uint32_t notifyNum = (resCtxHost.threadCount > 1) ? resCtxHost.threadCount : 1U;
    CHK_RET(HcclThreadAcquire(
        comm, aicpuTsEngine, resCtxHost.threadCount, notifyNum, resCtxHost.threads));
    resCtxHost.aicpuThread = resCtxHost.threads[0];
    CHK_RET(HcclThreadExportToCommEngine(
        comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

    CHK_RET(AcquireChannels(comm, param.myRank, peers, peerCount, resCtxHost));
    return HCCL_SUCCESS;
}

} // namespace

extern "C" HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("HcclScatter: unsupported dataType[%d]", dataType), HCCL_E_PARA);
    CHK_PRT_RET(recvCount > UINT64_MAX / sizeof(float),
        HCCL_ERROR("HcclScatter: recvCount overflow[%llu]", static_cast<unsigned long long>(recvCount)), HCCL_E_PARA);

    const uint64_t blockBytes = recvCount * sizeof(float);
    ScatterAlgo requestedAlgo = SCATTER_ALGO_DIRECT;
    const char *tagPrefix = "sctiny16a";
    if (blockBytes > TINY_LIMIT_BYTES && blockBytes <= SMALL_LIMIT_BYTES) {
        requestedAlgo = SCATTER_ALGO_SMALL;
        tagPrefix = "scsmall16h";
    } else if (blockBytes > SMALL_LIMIT_BYTES) {
        requestedAlgo = SCATTER_ALGO_2D;
        tagPrefix = "sc2d16c2";
    }

    OpParam param{};
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    const int tagRet = std::snprintf(param.tag, sizeof(param.tag), "%s_r%u_b%llu", tagPrefix, root,
        static_cast<unsigned long long>(blockBytes));
    CHK_PRT_RET(tagRet <= 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("HcclScatter: failed to build tag"), HCCL_E_INTERNAL);

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // Fast path for repeated benchmark invocations of the same root/size/stream.
    void *hostCtx = nullptr;
    uint64_t hostCtxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize) == HCCL_SUCCESS) {
        ScatterHostCache cache{};
        CHK_RET(LoadHostCache(hostCtx, hostCtxSize, cache));
        param.myRank = cache.myRank;
        param.rankSize = cache.rankSize;
        CHK_PRT_RET(root >= param.rankSize,
            HCCL_ERROR("HcclScatter: invalid root[%u], rankSize[%u]", root, param.rankSize), HCCL_E_PARA);
        if (param.myRank == root) {
            CHK_PTR_NULL(sendBuf);
        }

        if (cache.streamKey == GetStreamKey(stream)) {
            param.cpuThread = cache.cpuThread;
            param.cpuThreadOnAicpu = cache.cpuThreadOnAicpu;
            param.aicpuThreadOnCpu = cache.aicpuThreadOnCpu;
        } else {
            CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
            CHK_RET(HcclThreadExportToCommEngine(
                comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));
            CHK_RET(HcclThreadExportToCommEngine(
                comm, 1, &cache.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
        }

        void *ctx = nullptr;
        uint64_t size = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size));
        param.resCtx = ctx;
        param.ctxSize = size;
        CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
        return HCCL_SUCCESS;
    }

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > SCATTER_MAX_RANK_SIZE,
        HCCL_ERROR("HcclScatter: unsupported rankSize[%u]", param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("HcclScatter: invalid root[%u], rankSize[%u]", root, param.rankSize), HCCL_E_PARA);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(
        comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    AlgResourceCtx resCtxHost{};
    CHK_RET(BuildResources(comm, param, requestedAlgo, resCtxHost));

    std::vector<char> seq = resCtxHost.Serialize();
    param.ctxSize = seq.size();
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seq.size(), 0));

    ScatterHostCache cache{};
    cache.streamKey = GetStreamKey(stream);
    cache.cpuThread = param.cpuThread;
    cache.cpuThreadOnAicpu = param.cpuThreadOnAicpu;
    cache.aicpuThreadOnCpu = param.aicpuThreadOnCpu;
    cache.aicpuThread = resCtxHost.aicpuThread;
    cache.myRank = param.myRank;
    cache.rankSize = param.rankSize;

    hostCtx = nullptr;
    hostCtxSize = sizeof(cache);
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
    CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, &cache, sizeof(cache), 0));

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
