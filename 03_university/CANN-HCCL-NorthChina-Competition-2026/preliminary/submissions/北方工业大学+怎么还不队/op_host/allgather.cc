/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
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
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t TOPO_RANK_SIZE = 2 * SERVER_RANK_SIZE;
constexpr uint32_t SMALL_CHANNEL_NUM = 4;
constexpr uint64_t SMALL_MESSAGE_BYTES = 512ULL * 1024;
constexpr uint64_t LARGE_TWO_SLICE_BYTES = 512ULL * 1024 * 1024;

struct AicpuMainThreadCache {
    ThreadHandle thread;
    uint32_t hostSyncNotifyIdx;
};

// Server 内的 Full-Mesh 链路在 layer 0，跨 Server 的 Clos 链路位于更高层。
// 不假设具体层号，按 RankGraph 返回的层次选择到 peer 的第一条可用物理链路。
HcclResult BuildChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    CommLink *selectedLinks = nullptr;
    uint32_t selectedLinkNum = 0;
    for (uint32_t i = 0; i < netLayerNum; ++i) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayers[i], srcRank, dstRank, &links, &linkNum);
        if (ret == HCCL_SUCCESS && linkNum != 0) {
            selectedLinks = links;
            selectedLinkNum = linkNum;
            break;
        }
    }
    CHK_PRT_RET(selectedLinkNum == 0,
        HCCL_ERROR("No physical link between rank[%u] and rank[%u]", srcRank, dstRank), HCCL_E_NOT_FOUND);

    CHK_RET(HcclChannelDescInit(&desc, 1));
    const CommLink &link = selectedLinks[0];
    desc.remoteRank = dstRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

// 小包使用机内递归倍增：channels[1..3]依次连接localRank xor 1/2/4，
// channels[0]连接另一Server的对应localRank；大包仍连接全部15个peer。
HcclResult AcquirePeerResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.rankSize != 1 && param.rankSize != TOPO_RANK_SIZE,
        HCCL_ERROR("Hierarchical AllGather only supports rankSize 1 or %u, got %u", TOPO_RANK_SIZE, param.rankSize),
        HCCL_E_NOT_SUPPORT);

    const auto typeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type"), HCCL_E_NOT_SUPPORT);
    const uint64_t inputBytes = param.count * typeIt->second;
    const bool useSmallAlgorithm = inputBytes <= SMALL_MESSAGE_BYTES;
    // 小包递归倍增全部排在主thread，无跨thread同步任务；大包资源保持
    // 已验证的15 thread不变。
    const uint32_t threadNum = param.rankSize == 1
                                   ? 1
                                   : (useSmallAlgorithm ? 1 : param.rankSize - 1);
    resCtx.slaveThreadNum = threadNum - 1;
    resCtx.notifyNumOnMainThread = resCtx.slaveThreadNum;
    resCtx.threads.resize(threadNum);
    // 主 thread 的 [0, slaveThreadNum) 用于等待从 thread；最后一个
    // notify 专供 Host/Device 同步，避免两个同步域使用同一索引。
    const uint32_t notifyNumPerThread = resCtx.notifyNumOnMainThread + 1;
    CHK_RET(HcclThreadAcquire(
        comm, COMM_ENGINE_AICPU_TS, threadNum, notifyNumPerThread, resCtx.threads.data()));
    resCtx.aicpuThread = resCtx.threads[0];

    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    const uint32_t channelNum = useSmallAlgorithm ? SMALL_CHANNEL_NUM : param.rankSize - 1;
    std::vector<HcclChannelDesc> descs(channelNum);
    std::vector<ChannelHandle> handles(channelNum);

    uint32_t channelIdx = 0;
    if (useSmallAlgorithm) {
        const uint32_t serverBase = param.myRank / SERVER_RANK_SIZE * SERVER_RANK_SIZE;
        const uint32_t localRank = param.myRank - serverBase;
        const uint32_t crossPeer = param.myRank < SERVER_RANK_SIZE ? param.myRank + SERVER_RANK_SIZE
                                                                   : param.myRank - SERVER_RANK_SIZE;
        CHK_RET(BuildChannelDesc(comm, param.myRank, crossPeer, descs[channelIdx++]));
        for (uint32_t distance = 1; distance < SERVER_RANK_SIZE; distance <<= 1) {
            const uint32_t peer = serverBase + (localRank ^ distance);
            CHK_RET(BuildChannelDesc(comm, param.myRank, peer, descs[channelIdx++]));
        }
    } else {
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.myRank) {
                CHK_RET(BuildChannelDesc(comm, param.myRank, rank, descs[channelIdx++]));
            }
        }
    }

    // 直接写入对端最终输出，避免 CCL -> recvBuf 的第二次整包搬运。
    // 输出内存在建链前注册并通过 channel 交换。大小包使用独立 tag/context，
    // 小包只改变目标内存，大包 v6 的资源和任务图保持不变。
    HcclMemHandle outputMemHandle = nullptr;
    char outputMemTag[HCCL_RES_TAG_MAX_LEN + 1] = {};
    CHK_PRT_RET(inputBytes > UINT64_MAX / param.rankSize,
        HCCL_ERROR("Output byte size overflow"), HCCL_E_PARA);
    const int memTagLen = snprintf(outputMemTag, sizeof(outputMemTag), "%s_output", param.tag);
    CHK_PRT_RET(memTagLen <= 0 || static_cast<size_t>(memTagLen) >= sizeof(outputMemTag),
        HCCL_ERROR("Failed to set output memory tag"), HCCL_E_INTERNAL);

    CommMem outputMem{COMM_MEM_TYPE_DEVICE, param.outputPtr, inputBytes * param.rankSize};
    CHK_RET(HcclCommMemReg(comm, outputMemTag, &outputMem, &outputMemHandle));
    CHK_PTR_NULL(outputMemHandle);
    for (auto &desc : descs) {
        desc.memHandles = &outputMemHandle;
        desc.memHandleNum = 1;
    }
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, descs.data(), channelNum, handles.data()));

    resCtx.channels.reserve(channelNum);
    for (uint32_t i = 0; i < channelNum; ++i) {
        void *remoteCclBuffer = nullptr;
        uint64_t remoteCclBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i], &remoteCclBuffer, &remoteCclBufferSize));

        void *remoteBuffer = remoteCclBuffer;
        uint64_t remoteBufferSize = remoteCclBufferSize;
        uint32_t remoteMemNum = 0;
        CommMem *remoteMems = nullptr;
        char **remoteMemTags = nullptr;
        CHK_RET(HcclChannelGetRemoteMems(
            comm, handles[i], &remoteMemNum, &remoteMems, &remoteMemTags));
        bool foundOutput = false;
        for (uint32_t memIdx = 0; memIdx < remoteMemNum; ++memIdx) {
            if (remoteMemTags[memIdx] != nullptr && strcmp(remoteMemTags[memIdx], outputMemTag) == 0) {
                remoteBuffer = remoteMems[memIdx].addr;
                remoteBufferSize = remoteMems[memIdx].size;
                foundOutput = true;
                break;
            }
        }
        CHK_PRT_RET(!foundOutput || remoteBuffer == nullptr,
            HCCL_ERROR("Remote output memory was not exchanged for rank[%u]", descs[i].remoteRank),
            HCCL_E_NOT_FOUND);

        ChannelInfo channel;
        channel.remoteRank = descs[i].remoteRank;
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = handles[i];
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        resCtx.channels.push_back(channel);

        // 小包仍以 CCL buffer 容量决定安全切片；大包直接输出，不受用户
        // recvBuf 大小影响本地 CCL 容量计算。
        resCtx.localBuffer.size = std::min(resCtx.localBuffer.size, remoteCclBufferSize);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type[%d]", static_cast<int>(dataType)), HCCL_E_NOT_SUPPORT);

    OpParam param;
    const uint64_t typeSize = SIZE_TABLE.at(dataType);
    CHK_PRT_RET(sendCount > UINT64_MAX / typeSize, HCCL_ERROR("Input byte size overflow"), HCCL_E_PARA);
    const uint64_t inputBytes = sendCount * typeSize;
    const char *algorithmTag = inputBytes <= SMALL_MESSAGE_BYTES
                                   ? "hccl_custom_allgather_hier_direct_v8_recursive_doubling"
                                   : (inputBytes < LARGE_TWO_SLICE_BYTES
                                             ? "hccl_custom_allgather_large_direct_v11_skip_main_self_copy"
                                             : "hccl_custom_allgather_large_direct_v7_no_ack");
    const int tagLen = snprintf(param.tag, sizeof(param.tag), "%s", algorithmTag);
    CHK_PRT_RET(tagLen <= 0 || static_cast<size_t>(tagLen) >= sizeof(param.tag),
        HCCL_ERROR("Failed to set op tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank[%u]/rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize != 1 && param.rankSize != TOPO_RANK_SIZE,
        HCCL_ERROR("Unsupported rankSize[%u] for the fixed 2x8 topology", param.rankSize), HCCL_E_NOT_SUPPORT);

    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtxSize < sizeof(AicpuMainThreadCache),
            HCCL_ERROR("Invalid cached thread context"), HCCL_E_INTERNAL);
        auto *cache = static_cast<AicpuMainThreadCache *>(hostCtx);
        param.root = cache->hostSyncNotifyIdx;
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &cache->thread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;
        void *cclBuffer = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBuffer, cclBufferSize};

        CHK_RET(AcquirePeerResources(comm, param, resCtxHost));
        param.root = resCtxHost.notifyNumOnMainThread;
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        std::vector<char> sequence = resCtxHost.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, sequence.data(), sequence.size(), 0));

        void *hostCtx = nullptr;
        AicpuMainThreadCache cache{resCtxHost.aicpuThread, resCtxHost.notifyNumOnMainThread};
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, sizeof(cache), &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, &cache, sizeof(cache), 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
