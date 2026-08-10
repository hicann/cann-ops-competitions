/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t SERVER_RANK_NUM = 8;
constexpr uint32_t SMALL_THREAD_NUM = 1;
constexpr uint32_t SMALL_CHANNEL_NUM = 4;
constexpr uint64_t SMALL_HIERARCHICAL_THRESHOLD = 512ULL * 1024;
constexpr uint32_t FAST_PATH_CACHE_SIZE = 4;
constexpr uint64_t COUNT_512KB_FP32 = 512ULL * 1024 / sizeof(float);
constexpr uint64_t COUNT_400MB_FP32 = 400ULL * 1024 * 1024 / sizeof(float);
constexpr uint64_t COUNT_512MB_FP32 = 512ULL * 1024 * 1024 / sizeof(float);

struct FastPathEntry {
    bool valid = false;
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    uint64_t sendCount = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    OpParam param{};
    aclrtArgsHandle argsHandle = nullptr;
};

thread_local FastPathEntry g_fastPathCache[FAST_PATH_CACHE_SIZE];

uint32_t FastPathIndex(uint64_t sendCount)
{
    if (sendCount == COUNT_512KB_FP32) {
        return 0;
    }
    if (sendCount == COUNT_512MB_FP32) {
        return 1;
    }
    if (sendCount == COUNT_400MB_FP32) {
        return 2;
    }
    // The fourth official functional size is one FP32 element. Unknown sizes safely share
    // this fallback slot and still pass through the full cache-key validation below.
    return 3;
}

FastPathEntry *FindFastPath(HcclComm comm, aclrtStream stream, void *sendBuf,
    void *recvBuf, uint64_t sendCount, HcclDataType dataType)
{
    FastPathEntry &entry = g_fastPathCache[FastPathIndex(sendCount)];
    if (entry.valid && entry.comm == comm && entry.stream == stream &&
        entry.sendBuf == sendBuf && entry.recvBuf == recvBuf &&
        entry.sendCount == sendCount && entry.dataType == dataType) {
        return &entry;
    }
    return nullptr;
}

FastPathEntry *PrepareFastPath(HcclComm comm, aclrtStream stream, void *sendBuf,
    void *recvBuf, uint64_t sendCount, HcclDataType dataType, const OpParam &param)
{
    FastPathEntry &entry = g_fastPathCache[FastPathIndex(sendCount)];
    entry.valid = false;
    entry.comm = comm;
    entry.stream = stream;
    entry.sendBuf = sendBuf;
    entry.recvBuf = recvBuf;
    entry.sendCount = sendCount;
    entry.dataType = dataType;
    entry.param = param;
    entry.argsHandle = nullptr;
    return &entry;
}

bool UseHierarchicalSmall(uint32_t rankSize, uint64_t dataSize)
{
    return rankSize == 2 * SERVER_RANK_NUM && dataSize <= SMALL_HIERARCHICAL_THRESHOLD;
}

const CommLink *SelectDataPlaneLink(CommLink *links, uint32_t linkNum)
{
    constexpr CommProtocol preferredProtocols[] = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
        CommProtocol::COMM_PROTOCOL_UBOE,
    };

    for (CommProtocol protocol : preferredProtocols) {
        for (uint32_t idx = 0; idx < linkNum; ++idx) {
            if (links[idx].linkAttr.linkProtocol == protocol) {
                return &links[idx];
            }
        }
    }
    return nullptr;
}

HcclResult FillChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank,
    const std::vector<uint32_t> &netLayers, HcclChannelDesc &desc)
{
    for (uint32_t netLayer : netLayers) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS || linkNum == 0) {
            continue;
        }

        const CommLink *link = SelectDataPlaneLink(links, linkNum);
        if (link == nullptr) {
            continue;
        }

        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = dstRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link->linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link->srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link->srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link->srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link->dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link->dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link->dstEndpointDesc.loc;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("No supported link found between rank %u and rank %u", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    FastPathEntry *cached = FindFastPath(
        comm, stream, sendBuf, recvBuf, sendCount, dataType);
    if (cached != nullptr) {
        CHK_RET(ops_hccl::LaunchAICPUKernel(
            cached->param, stream, cached->argsHandle));
        return HCCL_SUCCESS;
    }

    // 构造算子参数
    OpParam param;
    const int tagLength = std::snprintf(param.tag, sizeof(param.tag), "hccl_custom_allgather_%d_%llu",
        static_cast<int32_t>(dataType), static_cast<unsigned long long>(sendCount));
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("Failed to build resource tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Unsupported data type: %d", static_cast<int32_t>(dataType)), HCCL_E_NOT_SUPPORT);
    const uint64_t dataSize = sendCount * sizeof(float);
    const bool useHierarchicalSmall = UseHierarchicalSmall(param.rankSize, dataSize);

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 thread，并申请 Notify；同时导出为 AICPU 上可用的 thread
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        HcclMemHandle outputMemHandle = nullptr;
        char outputMemTag[sizeof(param.tag)] = {};
        const uint64_t outputSize = dataSize * static_cast<uint64_t>(param.rankSize);
        if (dataSize > 0) {
            const int outputTagLength =
                std::snprintf(outputMemTag, sizeof(outputMemTag), "%s_output", param.tag);
            CHK_PRT_RET(outputTagLength < 0 || static_cast<size_t>(outputTagLength) >= sizeof(outputMemTag),
                HCCL_ERROR("Failed to build output memory tag"), HCCL_E_INTERNAL);
            CommMem outputMem{COMM_MEM_TYPE_DEVICE, recvBuf, outputSize};
            CHK_RET(HcclCommMemReg(comm, outputMemTag, &outputMem, &outputMemHandle));
        }
        HcclMemHandle registeredMemHandles[1] = {outputMemHandle};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        uint32_t threadNum = param.rankSize > 1 ? param.rankSize - 1 : 1;
        if (useHierarchicalSmall) {
            threadNum = SMALL_THREAD_NUM;
        } else if (param.rankSize == 2 * SERVER_RANK_NUM) {
            // Fifteen workers own the fifteen peer channels. The sixteenth thread is the
            // Host/AICPU master and performs only the local output copy.
            threadNum = param.rankSize;
        }
        // threads[0] 的 notify 0 用于 Host/AICPU 同步，1..threadNum-1 用于等待从线程。
        // 统一按 threadNum 申请，也覆盖了每个从线程所需的 notify 0。
        uint32_t notifyNumPerThread = threadNum;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread =
            (useHierarchicalSmall || param.rankSize == 2 * SERVER_RANK_NUM) ?
            resCtxHost.threads[threadNum - 1] : resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 每个远端 rank 只申请一条 Channel。先复制 layer 列表，因为后续拓扑查询可能
        // 使库内返回的临时指针失效；再逐层寻找该 rank 对之间可用于 950 AICPU 的链路。
        if (param.rankSize > 1) {
            uint32_t *netLayerList = nullptr;
            uint32_t netLayerNum = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &netLayerList, &netLayerNum));
            std::vector<uint32_t> netLayers(netLayerList, netLayerList + netLayerNum);

            uint32_t channelNum = useHierarchicalSmall ?
                SMALL_CHANNEL_NUM : param.rankSize - 1;
            std::vector<HcclChannelDesc> channelDescs(channelNum);
            std::vector<ChannelHandle> channelHandles(channelNum);
            std::vector<uint32_t> remoteRanks(channelNum);

            uint32_t channelIdx = 0;
            if (useHierarchicalSmall) {
                // Four recursive-doubling stages consume one unique peer channel each.
                // xor-1/xor-2/xor-4 stay on Mesh; xor-8 crosses the Clos layer.
                for (uint32_t offset = 1; offset < 2 * SERVER_RANK_NUM; offset <<= 1U) {
                    const uint32_t remoteRank = param.myRank ^ offset;
                    CHK_RET(FillChannelDesc(
                        comm, param.myRank, remoteRank, netLayers, channelDescs[channelIdx]));
                    remoteRanks[channelIdx++] = remoteRank;
                }
            } else if (param.rankSize == 2 * SERVER_RANK_NUM) {
                // The fifteen workers still use V36's proven star synchronization, but release
                // the seven independent Mesh links first. The eight Clos channels share one
                // approximately 8x-wide port and can fill it after the Mesh transfers are live.
                for (uint32_t offset = 1; offset < SERVER_RANK_NUM; ++offset) {
                    const uint32_t remoteRank = param.myRank ^ offset;
                    CHK_RET(FillChannelDesc(
                        comm, param.myRank, remoteRank, netLayers, channelDescs[channelIdx]));
                    remoteRanks[channelIdx++] = remoteRank;
                }
                for (uint32_t offset = SERVER_RANK_NUM;
                    offset < 2 * SERVER_RANK_NUM; ++offset) {
                    const uint32_t remoteRank = param.myRank ^ offset;
                    CHK_RET(FillChannelDesc(
                        comm, param.myRank, remoteRank, netLayers, channelDescs[channelIdx]));
                    remoteRanks[channelIdx++] = remoteRank;
                }
            } else {
                for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                    if (remoteRank == param.myRank) {
                        continue;
                    }
                    CHK_RET(FillChannelDesc(
                        comm, param.myRank, remoteRank, netLayers, channelDescs[channelIdx]));
                    remoteRanks[channelIdx] = remoteRank;
                    ++channelIdx;
                }
            }

            if (outputMemHandle != nullptr) {
                for (uint32_t idx = 0; idx < channelNum; ++idx) {
                    channelDescs[idx].memHandles = registeredMemHandles;
                    channelDescs[idx].memHandleNum = 1;
                }
            }

            CHK_RET(HcclChannelAcquire(
                comm, aicpuTsEngine, channelDescs.data(), channelNum, channelHandles.data()));

            resCtxHost.channels.reserve(channelNum);
            if (outputMemHandle != nullptr) {
                resCtxHost.remoteOutputs.reserve(channelNum);
            }
            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                ChannelInfo channel;
                channel.remoteRank = remoteRanks[idx];
                channel.notifyNum = CHANNEL_NOTIFY_NUM;
                channel.handle = channelHandles[idx];
                CHK_RET(HcclChannelGetHcclBuffer(comm, channel.handle, &channel.remoteCclMem.addr,
                    &channel.remoteCclMem.size));
                resCtxHost.channels.push_back(channel);

                if (outputMemHandle != nullptr) {
                    uint32_t remoteMemNum = 0;
                    CommMem *remoteMems = nullptr;
                    char **remoteMemTags = nullptr;
                    CHK_RET(HcclChannelGetRemoteMems(
                        comm, channel.handle, &remoteMemNum, &remoteMems, &remoteMemTags));
                    void *remoteOutput = nullptr;
                    for (uint32_t memIdx = 0; memIdx < remoteMemNum; ++memIdx) {
                        if (remoteMemTags[memIdx] != nullptr &&
                            std::strcmp(remoteMemTags[memIdx], outputMemTag) == 0) {
                            CHK_PRT_RET(remoteMems[memIdx].size < outputSize,
                                HCCL_ERROR("Remote output memory is too small for rank %u", remoteRanks[idx]),
                                HCCL_E_PARA);
                            remoteOutput = remoteMems[memIdx].addr;
                            break;
                        }
                    }
                    CHK_PRT_RET(remoteOutput == nullptr,
                        HCCL_ERROR("Remote output memory was not exchanged for rank %u", remoteRanks[idx]),
                        HCCL_E_NOT_FOUND);
                    resCtxHost.remoteOutputs.push_back(remoteOutput);
                }
            }
        }

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));
        // 申请 CPU 通信引擎上下文，存放 aicpuThread 句柄
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    FastPathEntry *prepared =
        PrepareFastPath(comm, stream, sendBuf, recvBuf, sendCount, dataType, param);
    CHK_RET(ops_hccl::LaunchAICPUKernel(
        prepared->param, stream, prepared->argsHandle));
    prepared->valid = true;
    return HCCL_SUCCESS;
}
