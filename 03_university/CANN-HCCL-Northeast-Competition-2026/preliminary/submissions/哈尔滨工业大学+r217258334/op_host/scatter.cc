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
#include <cstdint>
#include <cstdio>
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
constexpr uint32_t SCATTER_THREAD_NUM = 2;
constexpr uint32_t LARGE_ROOT_THREAD_NUM = 15;
constexpr uint32_t THREAD_NOTIFY_NUM = 2;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t EXPECTED_RANK_NUM = 16;
constexpr uint32_t EXPECTED_SERVER_RANK_NUM = 8;
constexpr uint64_t SMALL_SLICE_THRESHOLD = 1024ULL * 1024ULL;
constexpr char SCATTER_OUTPUT_MEM_TAG_PREFIX[] = "hccl_custom_scatter_v2_output_";
constexpr char SCATTER_INPUT_MEM_TAG_PREFIX[] = "hccl_custom_scatter_v3_input_";

HcclResult FillChannelDesc(const CommLink &link, uint32_t remoteRank, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
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

HcclResult BuildScatterChannels(HcclComm comm, uint32_t myRank, uint32_t rankSize,
    HcclMemHandle outputMemHandle, HcclMemHandle inputMemHandle, AlgResourceCtx &resCtx)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayerNum == 0, HCCL_ERROR("RankGraph contains no network layer"), HCCL_E_INTERNAL);

    std::vector<HcclChannelDesc> channelDescs;
    std::vector<uint32_t> remoteRankOrder;
    channelDescs.reserve(rankSize > 0 ? rankSize - 1 : 0);
    remoteRankOrder.reserve(rankSize > 0 ? rankSize - 1 : 0);

    // Channel建立时把本rank需要对端访问的用户内存一并交换。大包保持V2只交换recvBuf；
    // 小包root额外交换完整sendBuf，使接收端可以直接pull自己的slice，绕过root HCCL staging。
    HcclMemHandle localMemHandles[2] = {nullptr, nullptr};
    uint32_t localMemHandleNum = 0;
    if (outputMemHandle != nullptr) {
        localMemHandles[localMemHandleNum++] = outputMemHandle;
    }
    if (inputMemHandle != nullptr) {
        localMemHandles[localMemHandleNum++] = inputMemHandle;
    }

    resCtx.channels.resize(rankSize);
    resCtx.localRanks.clear();
    resCtx.remoteRanks.clear();
    resCtx.localRanks.push_back(myRank);

    for (uint32_t remoteRank = 0; remoteRank < rankSize; ++remoteRank) {
        if (remoteRank == myRank) {
            continue;
        }

        bool sameFirstLayer = false;
        bool foundLink = false;
        HcclChannelDesc selectedDesc;
        for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
            CommLink *linkList = nullptr;
            uint32_t listSize = 0;
            CHK_RET(HcclRankGraphGetLinks(
                comm, netLayers[layerIdx], myRank, remoteRank, &linkList, &listSize));
            if (layerIdx == 0 && listSize > 0) {
                sameFirstLayer = true;
            }
            if (!foundLink && listSize > 0) {
                // 不硬编码协议枚举。直接采用RankGraph返回的链路描述，避免CANN版本间协议名差异。
                CHK_RET(FillChannelDesc(linkList[0], remoteRank, selectedDesc));
                foundLink = true;
            }
        }

        CHK_PRT_RET(!foundLink,
            HCCL_ERROR("No communication link found, myRank[%u], remoteRank[%u]", myRank, remoteRank),
            HCCL_E_INTERNAL);

        if (sameFirstLayer) {
            resCtx.localRanks.push_back(remoteRank);
        } else {
            resCtx.remoteRanks.push_back(remoteRank);
        }
        if (localMemHandleNum != 0) {
            selectedDesc.memHandles = localMemHandles;
            selectedDesc.memHandleNum = localMemHandleNum;
        }
        channelDescs.push_back(selectedDesc);
        remoteRankOrder.push_back(remoteRank);
    }

    std::sort(resCtx.localRanks.begin(), resCtx.localRanks.end());
    std::sort(resCtx.remoteRanks.begin(), resCtx.remoteRanks.end());

    if (rankSize == EXPECTED_RANK_NUM &&
        (resCtx.localRanks.size() != EXPECTED_SERVER_RANK_NUM ||
            resCtx.remoteRanks.size() != EXPECTED_SERVER_RANK_NUM)) {
        HCCL_WARNING("Unexpected 2x8 topology split: local[%zu], remote[%zu]; device will use generic fallback",
            resCtx.localRanks.size(), resCtx.remoteRanks.size());
    }

    if (!channelDescs.empty()) {
        std::vector<ChannelHandle> handles(channelDescs.size());
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, channelDescs.data(), channelDescs.size(),
            handles.data()));
        for (size_t idx = 0; idx < handles.size(); ++idx) {
            uint32_t remoteRank = remoteRankOrder[idx];
            ChannelInfo &channel = resCtx.channels[remoteRank];
            channel.remoteRank = remoteRank;
            channel.notifyNum = CHANNEL_NOTIFY_NUM;
            channel.handle = handles[idx];
            CHK_RET(HcclChannelGetHcclBuffer(
                comm, channel.handle, &channel.remoteCclMem.addr, &channel.remoteCclMem.size));

            uint32_t remoteMemNum = 0;
            CommMem *remoteMems = nullptr;
            char **remoteMemTags = nullptr;
            CHK_RET(HcclChannelGetRemoteMems(
                comm, channel.handle, &remoteMemNum, &remoteMems, &remoteMemTags));
            CHK_PRT_RET((remoteMemNum > 0) && ((remoteMems == nullptr) || (remoteMemTags == nullptr)),
                HCCL_ERROR("Remote memory metadata is incomplete for rank[%u]", remoteRank), HCCL_E_INTERNAL);

            for (uint32_t memIdx = 0; memIdx < remoteMemNum; ++memIdx) {
                const char *memTag = remoteMemTags[memIdx];
                if (memTag == nullptr) {
                    continue;
                }
                if (std::strncmp(memTag, SCATTER_OUTPUT_MEM_TAG_PREFIX,
                        sizeof(SCATTER_OUTPUT_MEM_TAG_PREFIX) - 1) == 0) {
                    channel.remoteUserOutput = CommBuffer{remoteMems[memIdx].addr, remoteMems[memIdx].size};
                    channel.hasRemoteUserOutput = 1;
                    continue;
                }
                if (std::strncmp(memTag, SCATTER_INPUT_MEM_TAG_PREFIX,
                        sizeof(SCATTER_INPUT_MEM_TAG_PREFIX) - 1) == 0) {
                    channel.remoteUserInput = CommBuffer{remoteMems[memIdx].addr, remoteMems[memIdx].size};
                    channel.hasRemoteUserInput = 1;
                }
            }
        }
    }

    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    CHK_RET(HcclGetRankId(comm, &myRank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    CHK_PRT_RET(rankSize == 0 || root >= rankSize,
        HCCL_ERROR("Invalid scatter root[%u], rankSize[%u]", root, rankSize), HCCL_E_PARA);
    if (myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }

    auto sizeIt = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", dataType), HCCL_E_PARA);
    const uint32_t dataTypeSize = sizeIt->second;
    CHK_PRT_RET(recvCount > UINT64_MAX / dataTypeSize,
        HCCL_ERROR("recvCount overflows byte size"), HCCL_E_PARA);
    const uint64_t sliceBytes = recvCount * dataTypeSize;
    CHK_PRT_RET(rankSize != 0 && sliceBytes > UINT64_MAX / rankSize,
        HCCL_ERROR("Scatter input byte size overflows"), HCCL_E_PARA);

    OpParam param;
    int tagRet = snprintf(param.tag, sizeof(param.tag), "hccl_scatter_v3_%u_%u_%p_%p_%llu_%d",
        root, myRank, sendBuf, recvBuf, static_cast<unsigned long long>(recvCount), static_cast<int>(dataType));
    CHK_PRT_RET((tagRet < 0) || (static_cast<size_t>(tagRet) >= sizeof(param.tag)),
        HCCL_ERROR("Scatter resource tag is too long"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    param.myRank = myRank;
    param.rankSize = rankSize;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

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
    } else {
        AlgResourceCtx resCtxHost;
        HcclMemHandle outputMemHandle = nullptr;
        HcclMemHandle inputMemHandle = nullptr;

        // 大包保持V2：每个rank交换自己的recvBuf。小包root不需要暴露recvBuf，
        // 它把唯一的memHandle槽位留给完整sendBuf；其余rank仍交换一个recvBuf handle。
        // 这样V3小包每端仍只交换1份memHandle，避免为了零拷贝引入额外控制面复杂度。
        const bool smallRoot = sliceBytes != 0 && sliceBytes <= SMALL_SLICE_THRESHOLD && myRank == root;
        if (sliceBytes != 0 && !smallRoot) {
            char outputMemTag[HCCL_RES_TAG_MAX_LEN] = {};
            int memTagRet = snprintf(outputMemTag, sizeof(outputMemTag), "%s%u_%u_%p_%llu_%d",
                SCATTER_OUTPUT_MEM_TAG_PREFIX, root, myRank, recvBuf,
                static_cast<unsigned long long>(recvCount), static_cast<int>(dataType));
            CHK_PRT_RET((memTagRet < 0) || (static_cast<size_t>(memTagRet) >= sizeof(outputMemTag)),
                HCCL_ERROR("Scatter output memory tag is too long"), HCCL_E_INTERNAL);

            CommMem outputMem{COMM_MEM_TYPE_DEVICE, recvBuf, sliceBytes};
            CHK_RET(HcclCommMemReg(comm, outputMemTag, &outputMem, &outputMemHandle));
        }

        // V3只在<=1MiB的小包root注册完整sendBuf。大包资源与V2完全一致，避免扰动981/777us路径。
        if (smallRoot) {
            const uint64_t inputBytes = sliceBytes * rankSize;
            char inputMemTag[HCCL_RES_TAG_MAX_LEN] = {};
            int inputTagRet = snprintf(inputMemTag, sizeof(inputMemTag), "%s%u_%p_%llu_%d",
                SCATTER_INPUT_MEM_TAG_PREFIX, root, sendBuf,
                static_cast<unsigned long long>(recvCount), static_cast<int>(dataType));
            CHK_PRT_RET((inputTagRet < 0) || (static_cast<size_t>(inputTagRet) >= sizeof(inputMemTag)),
                HCCL_ERROR("Scatter input memory tag is too long"), HCCL_E_INTERNAL);

            CommMem inputMem{COMM_MEM_TYPE_DEVICE, sendBuf, inputBytes};
            CHK_RET(HcclCommMemReg(comm, inputMemTag, &inputMem, &inputMemHandle));
        }

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // 小包继续保持V1的2-thread资源；仅2x8大包root申请15条thread，
        // 对15个peer做到每peer一thread，避免root侧Write在同一AICPU_TS thread上串行。
        uint32_t threadNum = SCATTER_THREAD_NUM;
        if (rankSize == 1) {
            threadNum = 1;
        } else if (myRank == root && rankSize == EXPECTED_RANK_NUM && sliceBytes > SMALL_SLICE_THRESHOLD) {
            threadNum = LARGE_ROOT_THREAD_NUM;
        }
        const uint32_t notifyNumPerThread = std::max<uint32_t>(THREAD_NOTIFY_NUM, threadNum);
        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(
            comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(BuildScatterChannels(comm, param.myRank, param.rankSize, outputMemHandle, inputMemHandle, resCtxHost));

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
