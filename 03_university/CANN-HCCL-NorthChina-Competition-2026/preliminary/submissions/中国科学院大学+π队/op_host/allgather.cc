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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t LOCAL_LAYER = 0;
constexpr uint32_t NETWORK_LAYER = 1;
constexpr uint32_t RANKS_PER_SERVER = ALLGATHER_SERVER_RANK_SIZE;
constexpr uint32_t EXPECTED_RANK_SIZE = ALLGATHER_RANK_SIZE;
constexpr uint64_t SMALL_MESSAGE_THRESHOLD = 512ULL * 1024ULL;
constexpr char INPUT_MEM_TAG[] = "ag_send";
constexpr char OUTPUT_MEM_TAG[] = "ag_recv";

HcclResult GetChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank, uint32_t netLayer,
    HcclMemHandle *memHandles, uint32_t memHandleNum, HcclChannelDesc &desc)
{
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &links, &linkNum));
    if (links == nullptr || linkNum == 0) {
        HCCL_ERROR("No layer-%u link from rank %u to rank %u", netLayer, myRank, remoteRank);
        return HCCL_E_NOT_FOUND;
    }

    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.channelProtocol = links[0].linkAttr.linkProtocol;
    desc.localEndpoint = links[0].srcEndpointDesc;
    desc.remoteEndpoint = links[0].dstEndpointDesc;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    if (memHandleNum != 0) {
        desc.memHandles = memHandles;
        desc.memHandleNum = memHandleNum;
    }
    return HCCL_SUCCESS;
}

HcclResult InitTopology(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    CHK_PRT_RET(param.rankSize != EXPECTED_RANK_SIZE,
        HCCL_ERROR("This algorithm requires 2 servers x 8 ranks, got %u ranks", param.rankSize),
        HCCL_E_NOT_SUPPORT);

    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    bool hasLayer0 = false;
    bool hasLayer1 = false;
    for (uint32_t idx = 0; idx < layerNum; ++idx) {
        hasLayer0 = hasLayer0 || layers[idx] == LOCAL_LAYER;
        hasLayer1 = hasLayer1 || layers[idx] == NETWORK_LAYER;
    }
    CHK_PRT_RET(!hasLayer0 || !hasLayer1, HCCL_ERROR("Both layer-0 and layer-1 are required"),
        HCCL_E_NOT_SUPPORT);

    uint32_t *localRanks = nullptr;
    uint32_t localRankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, LOCAL_LAYER, &localRanks, &localRankNum));
    CHK_PRT_RET(localRanks == nullptr || localRankNum != RANKS_PER_SERVER,
        HCCL_ERROR("Expected 8 ranks in layer-0 instance, got %u", localRankNum), HCCL_E_NOT_SUPPORT);
    std::copy(localRanks, localRanks + localRankNum, resource.localRanks.begin());
    std::sort(resource.localRanks.begin(), resource.localRanks.end());

    CHK_PRT_RET(!std::binary_search(resource.localRanks.begin(), resource.localRanks.end(), param.myRank),
        HCCL_ERROR("Current rank is absent from layer-0"), HCCL_E_INTERNAL);

    uint32_t remoteIndex = 0;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!std::binary_search(resource.localRanks.begin(), resource.localRanks.end(), rank)) {
            CHK_PRT_RET(remoteIndex >= resource.remoteRanks.size(),
                HCCL_ERROR("Too many ranks outside the layer-0 instance"), HCCL_E_INTERNAL);
            resource.remoteRanks[remoteIndex++] = rank;
        }
    }
    CHK_PRT_RET(remoteIndex != RANKS_PER_SERVER,
        HCCL_ERROR("Expected 8 ranks outside the layer-0 instance, got %u", remoteIndex),
        HCCL_E_INTERNAL);
    resource.ownerRank = param.myRank;
    return HCCL_SUCCESS;
}

void InitSmallStageMetadata(const OpParam &param, AlgResourceCtx &resource)
{
    resource.smallStageValid = 0;
    for (uint32_t idx = 0; idx < RANKS_PER_SERVER; ++idx) {
        if (resource.localRanks[idx] != resource.localRanks[0] + idx ||
            resource.remoteRanks[idx] != resource.remoteRanks[0] + idx) {
            HCCL_WARNING("Rank numbering is not contiguous; small RD will use Flat fallback");
            return;
        }
    }

    const uint32_t localBase = resource.localRanks[0];
    if (param.myRank < localBase || param.myRank >= localBase + RANKS_PER_SERVER) {
        HCCL_WARNING("Rank %u is outside the contiguous local rank range", param.myRank);
        return;
    }
    resource.smallLocalIndex = param.myRank - localBase;
    const std::array<uint32_t, ALLGATHER_SMALL_STAGE_NUM> stagePeers = {
        resource.localRanks[resource.smallLocalIndex ^ 1U],
        resource.localRanks[resource.smallLocalIndex ^ 2U],
        resource.localRanks[resource.smallLocalIndex ^ 4U],
        resource.remoteRanks[resource.smallLocalIndex]
    };
    for (uint32_t stage = 0; stage < stagePeers.size(); ++stage) {
        bool found = false;
        for (uint32_t channelIdx = 0; channelIdx < resource.channels.size(); ++channelIdx) {
            if (resource.channels[channelIdx].remoteRank == stagePeers[stage]) {
                resource.smallStageChannels[stage] = channelIdx;
                found = true;
                break;
            }
        }
        if (!found) {
            HCCL_WARNING("No channel to small RD peer %u", stagePeers[stage]);
            return;
        }
    }

    // After xor-2 each rank owns one contiguous four-rank block. Besides the
    // same-index cross peer, fan that block out to the opposite four-rank
    // group on the remote server while xor-4 runs locally.
    const uint32_t crossMatePeer =
        resource.remoteRanks[resource.smallLocalIndex ^ (RANKS_PER_SERVER / 2U)];
    bool crossMateFound = false;
    for (uint32_t channelIdx = 0; channelIdx < resource.channels.size(); ++channelIdx) {
        if (resource.channels[channelIdx].remoteRank == crossMatePeer) {
            resource.smallCrossMateChannel = channelIdx;
            crossMateFound = true;
            break;
        }
    }
    if (!crossMateFound) {
        HCCL_WARNING("No channel to second small RD cross peer %u", crossMatePeer);
        return;
    }
    resource.smallStageValid = 1;
}

bool TryRegisterDirectOutput(HcclComm comm, void *output, uint64_t outputBytes,
    HcclMemHandle &outputHandle)
{
    CommMem outputMem{};
    outputMem.type = COMM_MEM_TYPE_DEVICE;
    outputMem.addr = output;
    outputMem.size = outputBytes;
    HcclResult ret = HcclCommMemReg(comm, OUTPUT_MEM_TAG, &outputMem, &outputHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("Registering direct OUTPUT failed (%d); use CCL fallback", static_cast<int32_t>(ret));
        return false;
    }
    return true;
}

bool TryRegisterDirectInput(HcclComm comm, void *input, uint64_t inputBytes,
    HcclMemHandle &inputHandle)
{
    CommMem inputMem{};
    inputMem.type = COMM_MEM_TYPE_DEVICE;
    inputMem.addr = input;
    inputMem.size = inputBytes;
    HcclResult ret = HcclCommMemReg(comm, INPUT_MEM_TAG, &inputMem, &inputHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("Registering direct INPUT failed (%d); disable copy/transfer overlap",
            static_cast<int32_t>(ret));
        return false;
    }
    return true;
}

bool TryGetRemoteOutput(HcclComm comm, ChannelHandle channel, uint64_t requiredBytes,
    CommBuffer &remoteOutput)
{
    uint32_t remoteMemNum = 0;
    CommMem *remoteMems = nullptr;
    char **remoteMemTags = nullptr;
    HcclResult ret = HcclChannelGetRemoteMems(
        comm, channel, &remoteMemNum, &remoteMems, &remoteMemTags);
    if (ret != HCCL_SUCCESS || remoteMems == nullptr || remoteMemTags == nullptr) {
        HCCL_WARNING("Getting registered remote memories failed (%d)", static_cast<int32_t>(ret));
        return false;
    }

    for (uint32_t idx = 0; idx < remoteMemNum; ++idx) {
        if (remoteMemTags[idx] != nullptr && std::strcmp(remoteMemTags[idx], OUTPUT_MEM_TAG) == 0) {
            remoteOutput.addr = remoteMems[idx].addr;
            remoteOutput.size = remoteMems[idx].size;
            if (remoteOutput.addr == nullptr || remoteOutput.size < requiredBytes) {
                HCCL_WARNING("Remote OUTPUT is invalid: size %lu, required %lu",
                    remoteOutput.size, requiredBytes);
                return false;
            }
            return true;
        }
    }
    HCCL_WARNING("The peer did not exchange memory tag %s", OUTPUT_MEM_TAG);
    return false;
}

bool IsSameDirectOutput(const HostResourceCtx &hostResource, void *output, uint64_t outputBytes)
{
    return hostResource.directEnabled != 0 && hostResource.outputPtr == output &&
        hostResource.outputBytes == outputBytes;
}

bool IsSameDirectInput(const HostResourceCtx &hostResource, const void *input, uint64_t inputBytes)
{
    return hostResource.inputDirectEnabled != 0 && hostResource.inputPtr == input &&
        hostResource.inputBytes == inputBytes;
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    (void)snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allgather");
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
    CHK_PRT_RET(param.rankSize != EXPECTED_RANK_SIZE,
        HCCL_ERROR("This algorithm requires exactly %u ranks", EXPECTED_RANK_SIZE), HCCL_E_NOT_SUPPORT);

    auto typeIt = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type %u", static_cast<uint32_t>(dataType)), HCCL_E_PARA);
    CHK_PRT_RET(sendCount > std::numeric_limits<uint64_t>::max() / typeIt->second,
        HCCL_ERROR("Input size overflows uint64"), HCCL_E_PARA);
    const uint64_t inputBytes = sendCount * typeIt->second;
    CHK_PRT_RET(inputBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("Output size overflows uint64"), HCCL_E_PARA);
    const uint64_t outputBytes = inputBytes * param.rankSize;
    if (inputBytes == 0) {
        return HCCL_SUCCESS;
    }

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(
        comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtx == nullptr || hostCtxSize < sizeof(HostResourceCtx),
            HCCL_ERROR("Invalid cached HostResourceCtx"), HCCL_E_INTERNAL);
        HostResourceCtx *hostResource = static_cast<HostResourceCtx *>(hostCtx);
        // AllGather has no root semantics. Reuse OpParam.root as registered
        // INPUT/OUTPUT validity bits for this exact pointer and size.
        param.root = 0;
        if (IsSameDirectInput(*hostResource, sendBuf, inputBytes)) {
            param.root |= DIRECT_INPUT_FLAG;
        }
        if (IsSameDirectOutput(*hostResource, recvBuf, outputBytes)) {
            param.root |= DIRECT_OUTPUT_FLAG;
        }
        if (inputBytes > SMALL_MESSAGE_THRESHOLD && (param.root & DIRECT_INPUT_FLAG) == 0) {
            HCCL_WARNING("Registered INPUT changed or is unavailable; disable copy/transfer overlap");
        }
        if ((param.root & DIRECT_OUTPUT_FLAG) == 0) {
            HCCL_WARNING("Registered OUTPUT changed or is unavailable; use the CCL fallback");
        }
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &hostResource->aicpuThread,
            cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        CHK_PRT_RET(cclBufferAddr == nullptr || cclBufferSize == 0,
            HCCL_ERROR("Local CCL buffer is invalid"), HCCL_E_MEMORY);
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        CHK_RET(InitTopology(comm, param, resCtxHost));

        HcclMemHandle inputHandle{};
        bool inputDirectEnabled = false;
        if (inputBytes > SMALL_MESSAGE_THRESHOLD) {
            inputDirectEnabled = TryRegisterDirectInput(
                comm, sendBuf, inputBytes, inputHandle);
        }
        if (inputDirectEnabled) {
            resCtxHost.localInput = CommBuffer{sendBuf, inputBytes};
            resCtxHost.inputDirectEnabled = 1U;
        }

        HcclMemHandle outputHandle{};
        const bool directEnabled = TryRegisterDirectOutput(
            comm, recvBuf, outputBytes, outputHandle);
        if (directEnabled) {
            resCtxHost.localOutput = CommBuffer{recvBuf, outputBytes};
        }

        std::array<HcclMemHandle, 2> registeredHandles{};
        uint32_t registeredHandleNum = 0;
        if (inputDirectEnabled) {
            registeredHandles[registeredHandleNum++] = inputHandle;
        }
        if (directEnabled) {
            registeredHandles[registeredHandleNum++] = outputHandle;
        }

        // thread[0] is main. thread[1..15] map one-to-one to channels[0..14].
        const uint32_t threadNum = param.rankSize;
        // main notify[0] is Host/AICPU control, notify[1..15] are worker DONE.
        // Each worker uses notify[1] for START.
        const uint32_t notifyNumPerThread = param.rankSize;
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum,
            notifyNumPerThread, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread,
            cpuTsEngine, &param.aicpuThreadOnCpu));

        std::vector<HcclChannelDesc> channelDescs;
        channelDescs.reserve(param.rankSize - 1);
        // Stable remote-rank order preserves the measured v0.4/v0.5 large-message
        // schedule and lets the small-message path derive channel indices in O(1).
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
            if (remoteRank == param.myRank) {
                continue;
            }
            bool isLocal = std::binary_search(
                resCtxHost.localRanks.begin(), resCtxHost.localRanks.end(), remoteRank);
            uint32_t netLayer = isLocal ? LOCAL_LAYER : NETWORK_LAYER;
            HcclChannelDesc desc{};
            CHK_RET(GetChannelDesc(comm, param.myRank, remoteRank, netLayer,
                registeredHandleNum == 0 ? nullptr : registeredHandles.data(),
                registeredHandleNum, desc));
            channelDescs.emplace_back(desc);
        }

        const uint32_t channelNum = static_cast<uint32_t>(channelDescs.size());
        CHK_PRT_RET(channelNum != resCtxHost.channels.size(),
            HCCL_ERROR("Expected %zu peer channels, got %u",
                resCtxHost.channels.size(), channelNum), HCCL_E_INTERNAL);
        std::vector<ChannelHandle> handles(channelNum);
        CHK_RET(HcclChannelAcquire(
            comm, aicpuTsEngine, channelDescs.data(), channelNum, handles.data()));
        bool allRemoteOutputsValid = directEnabled;
        uint64_t minRemoteCclBytes = std::numeric_limits<uint64_t>::max();
        for (uint32_t idx = 0; idx < channelNum; ++idx) {
            ChannelInfo info;
            info.remoteRank = channelDescs[idx].remoteRank;
            bool isLocal = std::binary_search(
                resCtxHost.localRanks.begin(), resCtxHost.localRanks.end(), info.remoteRank);
            info.netLayer = isLocal ? LOCAL_LAYER : NETWORK_LAYER;
            info.notifyNum = CHANNEL_NOTIFY_NUM;
            info.handle = handles[idx];
            CHK_RET(HcclChannelGetHcclBuffer(
                comm, info.handle, &info.remoteCclMem.addr, &info.remoteCclMem.size));
            CHK_PRT_RET(info.remoteCclMem.addr == nullptr || info.remoteCclMem.size == 0,
                HCCL_ERROR("Remote CCL buffer is invalid for rank %u", info.remoteRank),
                HCCL_E_MEMORY);
            minRemoteCclBytes = std::min(minRemoteCclBytes, info.remoteCclMem.size);
            if (directEnabled &&
                !TryGetRemoteOutput(comm, info.handle, outputBytes, info.remoteOutput)) {
                allRemoteOutputsValid = false;
            }
            resCtxHost.channels[idx] = info;
        }
        resCtxHost.minRemoteCclBytes = minRemoteCclBytes;
        InitSmallStageMetadata(param, resCtxHost);
        resCtxHost.directEnabled = allRemoteOutputsValid ? 1U : 0U;
        param.root = 0;
        if (resCtxHost.inputDirectEnabled != 0) {
            param.root |= DIRECT_INPUT_FLAG;
        }
        if (resCtxHost.directEnabled != 0) {
            param.root |= DIRECT_OUTPUT_FLAG;
        }
        if ((param.root & DIRECT_OUTPUT_FLAG) == 0) {
            HCCL_WARNING("Registered direct path is unavailable; use CCL fallback");
        }

        param.ctxSize = sizeof(AlgResourceCtx);
        CHK_RET(HcclEngineCtxCreate(
            comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(
            comm, aicpuTsEngine, param.tag, &resCtxHost, param.ctxSize, 0));

        HostResourceCtx hostResource{};
        hostResource.aicpuThread = resCtxHost.aicpuThread;
        hostResource.inputPtr = sendBuf;
        hostResource.inputBytes = inputBytes;
        hostResource.inputDirectEnabled = resCtxHost.inputDirectEnabled;
        hostResource.outputPtr = recvBuf;
        hostResource.outputBytes = outputBytes;
        hostResource.directEnabled = resCtxHost.directEnabled;
        void *hostCtx = nullptr;
        const uint64_t hostCtxSize = sizeof(HostResourceCtx);
        CHK_RET(HcclEngineCtxCreate(
            comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(
            comm, cpuTsEngine, param.tag, &hostResource, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
