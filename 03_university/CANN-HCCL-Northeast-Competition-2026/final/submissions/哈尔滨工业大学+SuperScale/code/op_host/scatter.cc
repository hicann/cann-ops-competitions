/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of this repository.
 */
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>
#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "../op_kernel_ccu/ccu_kernel.h"

namespace {
uint32_t RootNearMask(HcclComm comm, const OpParam &param)
{
    // In the four competition topologies, layer 0 is the local Server
    // instance. Membership also handles singleton Servers with no Mesh edges.
    uint32_t *layers = nullptr, layerCount = 0;
    if (HcclRankGraphGetLayers(comm, &layers, &layerCount) == HCCL_SUCCESS &&
        layers != nullptr && std::find(layers, layers + layerCount, 0U) != layers + layerCount) {
        uint32_t *ranks = nullptr, rankCount = 0;
        if (HcclRankGraphGetRanksByLayer(comm, 0, &ranks, &rankCount) == HCCL_SUCCESS &&
            ranks != nullptr && rankCount > 0 && rankCount <= param.rankSize) {
            uint32_t members = 0;
            bool valid = true;
            for (uint32_t i = 0; i < rankCount; ++i) {
                if (ranks[i] >= param.rankSize || (members & (1U << ranks[i]))) {
                    valid = false;
                    break;
                }
                members |= 1U << ranks[i];
            }
            if (valid && (members & (1U << param.myRank)))
                return (members & (1U << param.root)) ? members & ~(1U << param.root) : 0;
        }
    }
    // Preserve the previously validated link-based query on SDKs that cannot
    // enumerate the local instance. Far ranks receive the plan from root.
    uint32_t mask = 0;
    for (uint32_t r = 0; r < param.rankSize; ++r) {
        if (r == param.root) continue;
        CommLink *links = nullptr;
        uint32_t count = 0;
        if (HcclRankGraphGetLinks(comm, 0, param.root, r, &links, &count) != HCCL_SUCCESS) continue;
        for (uint32_t i = 0; i < count; ++i) {
            if (links[i].linkAttr.linkProtocol == COMM_PROTOCOL_UBC_CTP) {
                mask |= 1U << r;
                break;
            }
        }
    }
    return mask;
}

uint32_t LocalMembers(HcclComm comm, const OpParam &param)
{
    uint32_t *ranks = nullptr, rankCount = 0;
    if (HcclRankGraphGetRanksByLayer(comm, 0, &ranks, &rankCount) != HCCL_SUCCESS || ranks == nullptr ||
        rankCount == 0 || rankCount > param.rankSize) return 0;
    uint32_t members = 0;
    for (uint32_t i = 0; i < rankCount; ++i) {
        if (ranks[i] >= param.rankSize || (members & (1U << ranks[i]))) return 0;
        members |= 1U << ranks[i];
    }
    return (members & (1U << param.myRank)) ? members : 0;
}

HcclResult ReadContext(HcclComm comm, const char *tag, AlgResourceCtx &resource)
{
    void *ctx = nullptr;
    uint64_t size = 0;
    HcclResult result = HcclEngineCtxGet(comm, tag, COMM_ENGINE_CCU, &ctx, &size);
    if (result != HCCL_SUCCESS) return result;
    if (ctx == nullptr || size == 0) return HCCL_E_INTERNAL;
    std::vector<char> data(static_cast<char *>(ctx), static_cast<char *>(ctx) + size);
    resource.DeSerialize(data);
    return HCCL_SUCCESS;
}

HcclResult WriteContext(HcclComm comm, const char *tag, AlgResourceCtx &resource)
{
    auto data = resource.Serialize();
    void *ctx = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, tag, COMM_ENGINE_CCU, data.size(), &ctx));
    return HcclEngineCtxCopy(comm, COMM_ENGINE_CCU, tag, data.data(), data.size(), 0);
}

// One channel per peer, shared by every count/root specialization on this communicator.
HcclResult AcquirePool(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    const char *tag = "superscale_scatter_ccu_channels_v7";
    if (ReadContext(comm, tag, resource) == HCCL_SUCCESS) return HCCL_SUCCESS;
    CHK_RET(HcclGetHcclBuffer(comm, &resource.localBuffer.addr, &resource.localBuffer.size));
    resource.channels.resize(param.rankSize, 0);
    resource.channelDie.resize(param.rankSize, 0);
    resource.minCclSize = resource.localBuffer.size;
    resource.threads.resize(1);
    CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU, 1, 1, resource.threads.data()));
    uint32_t *layerPtr = nullptr, layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerPtr, &layerCount));
    std::vector<uint32_t> layers(layerPtr, layerPtr + layerCount);
    std::sort(layers.begin(), layers.end());
    std::vector<HcclChannelDesc> descriptors;
    std::vector<uint32_t> peers;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) continue;
        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        bool found = false;
        for (uint32_t layer : layers) {
            CommLink *links = nullptr;
            uint32_t count = 0;
            HcclResult ret = HcclRankGraphGetLinks(comm, layer, param.myRank, peer, &links, &count);
            if (ret != HCCL_SUCCESS) continue;
            for (uint32_t i = 0; i < count; ++i) {
                const CommLink &link = links[i];
                if (link.linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) continue;
                desc.remoteRank = peer;
                // META, READY, DONE, ACK, then the deferred staged-read ack.
                desc.notifyNum = 6;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &link.srcEndpointDesc,
                    ENDPOINT_ATTR_DIE_ID, sizeof(uint32_t), &resource.channelDie[peer]));
                if (resource.channelDie[peer] > 1) return HCCL_E_NOT_SUPPORT;
                found = true;
                break;
            }
            if (found) break;
        }
        if (!found) return HCCL_E_NOT_FOUND;
        descriptors.push_back(desc);
        peers.push_back(peer);
    }
    std::vector<ChannelHandle> handles(peers.size());
    if (!peers.empty()) CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, descriptors.data(),
        descriptors.size(), handles.data()));
    for (size_t i = 0; i < peers.size(); ++i) {
        uint32_t peer = peers[i];
        resource.channels[peer] = handles[i];
        void *peerBuffer = nullptr;
        uint64_t peerSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, resource.channels[peer], &peerBuffer, &peerSize));
        resource.minCclSize = std::min(resource.minCclSize, peerSize);
    }
    return WriteContext(comm, tag, resource);
}
}

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    OpParam param{};
    param.root = root;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.opType = HCCL_CMD_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || root >= param.rankSize ||
        dataType != HCCL_DATA_TYPE_FP32 || recvCount > std::numeric_limits<uint64_t>::max() / 4 / param.rankSize)
        return HCCL_E_PARA;
    if (recvCount == 0) return HCCL_SUCCESS;
    CHK_PTR_NULL(recvBuf);
    if (param.myRank == root) { CHK_PTR_NULL(sendBuf); }
    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, 1, &param.cpuThread));
    HcclDfxOpInfo dfx{};
    dfx.opType = HCCL_CMD_SCATTER;
    dfx.dataType = dataType;
    dfx.dataCount = recvCount;
    dfx.root = root;
    dfx.engine = COMM_ENGINE_CCU;
    dfx.cpuTsThread = param.cpuThread;
    dfx.inputMemAddr = reinterpret_cast<uint64_t>(sendBuf);
    dfx.inputMemSize = param.myRank == root ? recvCount * 4 * param.rankSize : 0;
    dfx.outputMemAddr = reinterpret_cast<uint64_t>(recvBuf);
    dfx.outputMemSize = recvCount * 4;
    std::snprintf(dfx.algTag, sizeof(dfx.algTag), "Scatter_CCU_SuperScale_v12");
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, &dfx));

    // Specialize instruction graphs, never addresses; channels remain in the shared pool.
    std::snprintf(param.tag, sizeof(param.tag), "superscale_scatter_v12_r%u_b%llu", root,
        static_cast<unsigned long long>(recvCount * 4));
    // Hot path first: an existing kernel context also proves the shared channel
    // pool is present, so repeated calls skip the pool query and both context
    // deserializations; only the registration path ever acquires the pool.
    AlgResourceCtx kernelCtx;
    if (ReadContext(comm, param.tag, kernelCtx) == HCCL_SUCCESS)
        return ops_hccl::ExecOp(param, kernelCtx);
    AlgResourceCtx resources;
    CHK_RET(AcquirePool(comm, param, resources));
    {
        CcuKernelArgBase arg{};
        arg.rank = param.myRank;
        arg.rankSize = param.rankSize;
        arg.root = root;
        arg.bytes = recvCount * 4;
        arg.channelCount = param.rankSize - 1;
        for (uint32_t r = 0; r < param.rankSize; ++r) {
            arg.channels[r] = resources.channels[r];
            arg.channelDie[r] = resources.channelDie[r];
        }
        arg.nearMask = RootNearMask(comm, param);
        arg.localMask = LocalMembers(comm, param);
        // Root and helpers share their own Server's visible Mesh membership.
        // Never infer another rank's die from its endpoint metadata: use only
        // the die of this rank's acquired channel when scheduling each phase.
        arg.relayMask = arg.nearMask;
        arg.scratchCapacity = std::min<uint64_t>(resources.minCclSize, MAX_DATA_SIZE);
        CcuInsHandle instance = 0;
        // A prefix is staged once per invocation. The plan caps every helper's
        // total occupancy at the common actual capacity, with no slot reuse.
        kernelCtx.requiredScratch = 4ULL * 1024 * 1024;
        uint32_t instanceCount = 0;
        CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
        if (instanceCount != 1) return HCCL_E_INTERNAL;
        if (HcommCcuKernelRegisterStart(instance) != CCU_SUCCESS) return HCCL_E_INTERNAL;
        arg.relayAllowed = resources.minCclSize >= kernelCtx.requiredScratch;
        ops_hccl::ConfigureKernel(arg);
        HCCL_INFO("SuperScale CCU rank=%u root=%u bytes=%llu nearMask=%u activeDies=%u minCCL=%llu relay=%u relayDies=%u",
            arg.rank, arg.root, static_cast<unsigned long long>(arg.bytes), arg.nearMask, arg.activeDies,
            static_cast<unsigned long long>(resources.minCclSize), static_cast<unsigned>(arg.relayAllowed), arg.relayDies);
        CcuResult result = CCU_SUCCESS;
        std::vector<std::pair<uint32_t, uint32_t>> phases;
        const uint32_t rootDie = arg.channelDie[arg.root];
        if (arg.rank != arg.root && arg.relayDies != 0) {
            // Helper: root-die buffers/ready, far-die scratch service, root-die completion.
            if (arg.relayDies == (1U << rootDie)) {
                phases.push_back({4, rootDie});
            } else {
                phases.push_back({1, rootDie});
                phases.push_back({2, arg.relayDies == 1U ? 0U : 1U});
                phases.push_back({3, rootDie});
            }
        } else if (arg.rank != arg.root) {
            phases.push_back({0, rootDie});
        } else {
            for (uint32_t die = 0; die < 2; ++die)
                if (arg.activeDies & (1U << die)) phases.push_back({0, die});
        }
        kernelCtx.dataKernelCount = static_cast<uint32_t>(phases.size());
        for (const auto &phase : phases) {
            arg.phase = phase.first;
            arg.die = phase.second;
            CcuKernelHandle kernel = 0;
            const void *args[] = {&arg};
            char kernelName[128];
            std::snprintf(kernelName, sizeof(kernelName), "%.100s_p%u_d%u", param.tag, arg.phase, arg.die);
            result = HcommCcuKernelRegister(instance, 0, kernelName,
                reinterpret_cast<void *>(ops_hccl::CcuKernel), args, 1, &kernel);
            if (result != CCU_SUCCESS) break;
            kernelCtx.ccuKernels.push_back(kernel);
        }
        CcuResult endResult = HcommCcuKernelRegisterEnd(instance);
        if (result != CCU_SUCCESS || endResult != CCU_SUCCESS) return HCCL_E_INTERNAL;
        kernelCtx.localBuffer = resources.localBuffer;
        kernelCtx.threads = resources.threads;
        // The launch decides the argument count the same way the kernel decided
        // its Load instructions, so it needs the capacity the kernel was built
        // with, not just the local buffer.
        kernelCtx.minCclSize = resources.minCclSize;
        CHK_RET(WriteContext(comm, param.tag, kernelCtx));
    }
    return ops_hccl::ExecOp(param, kernelCtx);
}
