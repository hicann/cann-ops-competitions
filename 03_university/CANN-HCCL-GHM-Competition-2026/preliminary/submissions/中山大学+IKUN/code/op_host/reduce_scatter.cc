/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <set>
#include <vector>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"
#include "log.h"

namespace {
// 标准大包的三条 DATA 通知和最终 ACK 使用四个独立索引。
// packed 大包沿用已经通过官方 VM 的 V30 ping-pong 协议，只需 DATA/ACK 两槽。
constexpr uint32_t SMALL_CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t EAGER_CHANNEL_NOTIFY_NUM = 4;
#if HCCL_RS_PERSISTENT_PACKED_HANDOFF
constexpr uint32_t PACKED_CHANNEL_NOTIFY_NUM = 4;
#else
constexpr uint32_t PACKED_CHANNEL_NOTIFY_NUM = 2;
#endif
constexpr uint32_t COMPETITION_RANK_SIZE = 16;
constexpr uint32_t RANKS_PER_SERVER = 8;

static_assert(HCCL_RS_CANDIDATE_MODE >= RS_CANDIDATE_FLATTEN &&
    HCCL_RS_CANDIDATE_MODE <= RS_CANDIDATE_DUAL_PATH,
    "invalid HCCL_RS_CANDIDATE_MODE");

uint64_t GetInputBytes(const OpParam &param)
{
    return param.count * sizeof(float) * static_cast<uint64_t>(param.rankSize);
}

bool UseNhrResources(const OpParam &param)
{
    return param.rankSize != COMPETITION_RANK_SIZE;
}

bool UseTopologyXorResources(const OpParam &param)
{
    const uint64_t inputBytes = GetInputBytes(param);
    return param.rankSize == COMPETITION_RANK_SIZE &&
        inputBytes < RS_NHR_INPUT_LIMIT_BYTES;
}

bool UseTinySpecialResources(const OpParam &param)
{
    return HCCL_RS_TINY_MODE != RS_TINY_MODE_XOR_HALF_COPY &&
        param.rankSize == COMPETITION_RANK_SIZE && param.count == 1;
}

bool UsePackedStripeResources(const OpParam &param)
{
    return param.rankSize == COMPETITION_RANK_SIZE &&
        GetInputBytes(param) > RS_PACKED_STRIPE_INPUT_MIN_BYTES;
}

bool UseReferenceV2Resources(const OpParam &param)
{
    const uint64_t inputBytes = GetInputBytes(param);
    return param.rankSize == COMPETITION_RANK_SIZE &&
        inputBytes >= RS_REFERENCE_V2_INPUT_MIN_BYTES &&
        inputBytes <= RS_PACKED_STRIPE_INPUT_MIN_BYTES;
}

const char *GetAlgorithmTag(const OpParam &param)
{
#if HCCL_RS_CANDIDATE_MODE == RS_CANDIDATE_FLATTEN
    if (UseTinySpecialResources(param)) {
#if HCCL_RS_TINY_MODE == RS_TINY_MODE_DIRECT_15_WORKER
        return "hccl_rs_" HCCL_RS_VARIANT_TAG "_tiny_direct15";
#elif HCCL_RS_TINY_MODE == RS_TINY_MODE_HIERARCHY_SEQUENTIAL
        return "hccl_rs_" HCCL_RS_VARIANT_TAG "_tiny_hierarchy_seq";
#else
        return "hccl_rs_" HCCL_RS_VARIANT_TAG "_tiny_hierarchy_tree";
#endif
    }
    if (UseTopologyXorResources(param)) {
        return "hccl_rs_" HCCL_RS_VARIANT_TAG "_halfcopy_mailbox";
    }
    if (UseNhrResources(param)) {
        return "hccl_rs_v17_fallback_batched_nhr";
    }
    if (UsePackedStripeResources(param)) {
#if HCCL_RS_DEDICATED_PACKED_REDUCE_PIPELINE
        return "hccl_rs_" HCCL_RS_VARIANT_TAG "_point6_dedicated_reduce";
#else
#if HCCL_RS_PERSISTENT_PACKED_HANDOFF
        return "hccl_rs_" HCCL_RS_VARIANT_TAG "_point6_persistent_handoff";
#else
        return "hccl_rs_" HCCL_RS_VARIANT_TAG "_point6_v110_handoff";
#endif
#endif
    }
    if (UseReferenceV2Resources(param)) {
#if HCCL_RS_DEDICATED_REFERENCE_REDUCE_PIPELINE
        return "hccl_rs_" HCCL_RS_VARIANT_TAG "_point7_dedicated_reduce";
#else
#if HCCL_RS_PERSISTENT_REFERENCE_HANDOFF
        return "hccl_rs_" HCCL_RS_VARIANT_TAG "_point7_persistent_handoff";
#else
        return "hccl_rs_" HCCL_RS_VARIANT_TAG "_point7_v110_handoff";
#endif
#endif
    }
    return "hccl_rs_v35_large_4_11_17_eager_pipeline";
#elif HCCL_RS_CANDIDATE_MODE == RS_CANDIDATE_PAIR_MESH
    return "hccl_rs_adaptive_pair_mesh_v1";
#else
    return "hccl_rs_adaptive_dual_path_v1";
#endif
}

HcclResult FillChannelDesc(
    HcclComm comm, uint32_t srcRank, uint32_t dstRank, uint32_t notifyNum,
    const std::vector<uint32_t> &layers, HcclChannelDesc &desc)
{
    for (const uint32_t layer : layers) {
        CommLink *linkData = nullptr;
        uint32_t linkCount = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layer, srcRank, dstRank, &linkData, &linkCount));
        if (linkData == nullptr || linkCount == 0) {
            continue;
        }

        // 赛题约束为一个对端只申请一个 channel。若图接口返回多个候选，
        // 只选择优先层上的第一个物理 Link。
        const CommLink link = linkData[0];
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = dstRank;
        desc.notifyNum = notifyNum;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link.srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("no link between rank[%u] and rank[%u]", srcRank, dstRank);
    return HCCL_E_INTERNAL;
}

[[maybe_unused]] void AddNhrPeers(const OpParam &param, std::set<uint32_t> &peerSet)
{
    for (uint32_t delta = 1; delta < param.rankSize; delta <<= 1) {
        peerSet.insert((param.myRank + delta) % param.rankSize);
        peerSet.insert((param.myRank + param.rankSize - delta) % param.rankSize);
    }
}

[[maybe_unused]] void AddCompetitionHierarchyPeers(
    const OpParam &param, std::set<uint32_t> &peerSet)
{
    if (param.rankSize != COMPETITION_RANK_SIZE) {
        return;
    }
    const uint32_t serverBase = (param.myRank / RANKS_PER_SERVER) * RANKS_PER_SERVER;
    for (uint32_t localRank = 0; localRank < RANKS_PER_SERVER; ++localRank) {
        const uint32_t peer = serverBase + localRank;
        if (peer != param.myRank) {
            peerSet.insert(peer);
        }
    }
    peerSet.insert((param.myRank + RANKS_PER_SERVER) % COMPETITION_RANK_SIZE);
}

HcclResult AcquireCandidateChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> peers;
#if HCCL_RS_CANDIDATE_MODE == RS_CANDIDATE_FLATTEN
    if (UseTinySpecialResources(param)) {
#if HCCL_RS_TINY_MODE == RS_TINY_MODE_DIRECT_15_WORKER
        peers.reserve(param.rankSize - 1);
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer != param.myRank) {
                peers.push_back(peer);
            }
        }
#else
        std::set<uint32_t> peerSet;
        AddCompetitionHierarchyPeers(param, peerSet);
        peers.assign(peerSet.begin(), peerSet.end());
#endif
    } else if (UseTopologyXorResources(param)) {
        // 16 Rank 只需要 XOR 维度 1/2/4/8 的四个对端；第一轮 delta=8
        // 对齐高带宽 Clos，其余三轮完全留在 server 内 Mesh。固定四项向量
        // 避免点5路径构造std::set及四个树节点。
        peers.reserve(4);
        for (uint32_t delta = 1; delta < param.rankSize; delta <<= 1) {
            peers.push_back(param.myRank ^ delta);
        }
    } else {
        std::set<uint32_t> peerSet;
        if (UseNhrResources(param)) {
            AddNhrPeers(param, peerSet);
        } else {
            for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
                if (peer != param.myRank) {
                    peerSet.insert(peer);
                }
            }
        }
        peers.assign(peerSet.begin(), peerSet.end());
    }
#else
    // 小包继续使用 NHR；大包只额外连接 7 个机内对端和同号跨机对端。
    std::set<uint32_t> peerSet;
    AddNhrPeers(param, peerSet);
    AddCompetitionHierarchyPeers(param, peerSet);
    peers.assign(peerSet.begin(), peerSet.end());
#endif

    const uint32_t channelCount = static_cast<uint32_t>(peers.size());
    const bool useSmallResources =
        UseTopologyXorResources(param) || UseNhrResources(param);
    const uint32_t channelNotifyNum = useSmallResources ?
        SMALL_CHANNEL_NOTIFY_NUM :
        (UsePackedStripeResources(param) ?
            PACKED_CHANNEL_NOTIFY_NUM : EAGER_CHANNEL_NOTIFY_NUM);
    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    CHK_PRT_RET(layerData == nullptr || layerCount == 0,
        HCCL_ERROR("rank graph contains no network layer"), HCCL_E_INTERNAL);
    // RankGraph查询结果由HCCL管理；只复制一次，所有peer共用同一快照。
    const std::vector<uint32_t> layers(layerData, layerData + layerCount);

    std::vector<HcclChannelDesc> descs(channelCount);
    std::vector<ChannelHandle> handles(channelCount);
    for (uint32_t index = 0; index < channelCount; ++index) {
        CHK_RET(FillChannelDesc(
            comm, param.myRank, peers[index], channelNotifyNum, layers, descs[index]));
    }

    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), channelCount, handles.data()));

    resCtxHost.channels.reserve(channelCount);
    for (uint32_t index = 0; index < channelCount; ++index) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[index], &remoteBuffer, &remoteBufferSize));

        ChannelInfo channel;
        channel.remoteRank = peers[index];
        channel.notifyNum = channelNotifyNum;
        channel.handle = handles[index];
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        resCtxHost.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}

uint32_t GetThreadCount(const OpParam &param)
{
    if (param.rankSize <= 1) {
        return 1;
    }
#if HCCL_RS_CANDIDATE_MODE == RS_CANDIDATE_FLATTEN
    if (UseTinySpecialResources(param)) {
#if HCCL_RS_TINY_MODE == RS_TINY_MODE_DIRECT_15_WORKER
        return param.rankSize;
#else
        return RANKS_PER_SERVER;
#endif
    }
    if (UseTopologyXorResources(param) || UseNhrResources(param)) {
        return 1;
    }
    if (UsePackedStripeResources(param)) {
#if HCCL_RS_DEDICATED_PACKED_REDUCE_PIPELINE
        return param.rankSize + RS_DEDICATED_REDUCE_GROUPS;
#else
        // 点6只在末条带复用已经完成通信的前三个worker，不再申请专用规约线程。
        return param.rankSize;
#endif
    }
    if (UseReferenceV2Resources(param)) {
#if HCCL_RS_DEDICATED_REFERENCE_REDUCE_PIPELINE
        return param.rankSize + RS_DEDICATED_REDUCE_GROUPS;
#else
        return param.rankSize;
#endif
    }
    return param.rankSize;
#else
    // 仅赛题 2×8 大包路径需要 7 个 Mesh worker 和 1 个 Clos worker；
    // 其它通信域回退 NHR，只申请一个主 thread。
    return param.rankSize == COMPETITION_RANK_SIZE ? RANKS_PER_SERVER + 1 : 1;
#endif
}

uint32_t GetNotifyCountPerThread(const OpParam &param, uint32_t threadCount)
{
#if HCCL_RS_CANDIDATE_MODE == RS_CANDIDATE_FLATTEN
#if HCCL_RS_DEDICATED_PACKED_REDUCE_PIPELINE
    if (UsePackedStripeResources(param)) {
        return threadCount - 1 + 2 * RS_DEDICATED_REDUCE_MAX_STRIPES;
    }
#endif
#if HCCL_RS_DEDICATED_REFERENCE_REDUCE_PIPELINE
    if (UseReferenceV2Resources(param)) {
        return threadCount - 1 + 2 * RS_DEDICATED_REDUCE_MAX_STRIPES;
    }
#endif
#endif
    return threadCount > 1 ? threadCount - 1 : 1;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("only FP32 is supported, dataType[%d]", static_cast<int32_t>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM,
        HCCL_ERROR("only SUM is supported, reduceOp[%d]", static_cast<int32_t>(op)),
        HCCL_E_NOT_SUPPORT);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    sprintf(param.tag, "%s", GetAlgorithmTag(param));

    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(
        comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtx == nullptr || hostCtxSize != sizeof(ThreadHandle),
            HCCL_ERROR("invalid cached host context, size[%llu]",
                static_cast<unsigned long long>(hostCtxSize)),
            HCCL_E_INTERNAL);
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        const uint32_t threadCount = GetThreadCount(param);
        const uint32_t notifyCountPerThread =
            GetNotifyCountPerThread(param, threadCount);
        resCtxHost.threads.resize(threadCount);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadCount,
            notifyCountPerThread, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
        CHK_RET(AcquireCandidateChannels(comm, param, resCtxHost));

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seq.size(), 0));

        void *hostCtx = nullptr;
        constexpr uint64_t hostCtxSize = sizeof(ThreadHandle);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(
            comm, cpuTsEngine, param.tag, &resCtxHost.aicpuThread, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
