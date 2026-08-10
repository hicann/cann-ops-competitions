/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include <algorithm>
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
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr char INPUT_MEM_SUFFIX[] = "_InputBuffer";
constexpr char OUTPUT_MEM_SUFFIX[] = "_OutputBuffer";

struct HostResourceCache {
    ThreadHandle aicpuThread;
    void *inputPtr;
    void *outputPtr;
    uint64_t inputSize;
    uint64_t outputSize;
};

HcclResult GetDataTypeSize(HcclDataType dataType, uint32_t &dataTypeSize)
{
    auto iter = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(iter == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type[%d]", static_cast<int32_t>(dataType)), HCCL_E_NOT_SUPPORT);
    dataTypeSize = iter->second;
    return HCCL_SUCCESS;
}

bool HasSuffix(const char *value, const char *suffix)
{
    if (value == nullptr || suffix == nullptr) {
        return false;
    }
    size_t valueLength = std::strlen(value);
    size_t suffixLength = std::strlen(suffix);
    return valueLength >= suffixLength && std::strcmp(value + valueLength - suffixLength, suffix) == 0;
}

HcclResult GetRemoteMemBySuffix(
    HcclComm comm, ChannelHandle channel, const char *memSuffix, CommBuffer &remoteMem)
{
    uint32_t memNum = 0;
    CommMem *remoteMems = nullptr;
    char **memTags = nullptr;
    CHK_RET(HcclChannelGetRemoteMems(comm, channel, &memNum, &remoteMems, &memTags));
    CHK_PRT_RET(memNum == 0 || remoteMems == nullptr || memTags == nullptr,
        HCCL_ERROR("Channel does not contain remote user memory"), HCCL_E_NOT_FOUND);

    for (uint32_t idx = 0; idx < memNum; idx++) {
        if (HasSuffix(memTags[idx], memSuffix)) {
            remoteMem = CommBuffer{remoteMems[idx].addr, remoteMems[idx].size};
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("Remote memory suffix[%s] was not found", memSuffix);
    return HCCL_E_NOT_FOUND;
}

uint32_t GetProtocolPriority(CommProtocol protocol)
{
    switch (protocol) {
        case CommProtocol::COMM_PROTOCOL_UBC_CTP:
            return 0;
        case CommProtocol::COMM_PROTOCOL_UBC_TP:
            return 1;
        case CommProtocol::COMM_PROTOCOL_PCIE:
            return 2;
        default:
            return std::numeric_limits<uint32_t>::max();
    }
}

HcclResult FillChannelDesc(HcclComm comm, uint32_t localRank, uint32_t remoteRank,
    const std::vector<uint32_t> &netLayers, HcclChannelDesc &channelDesc)
{
    for (uint32_t netLayer : netLayers) {
        CommLink *linkList = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, localRank, remoteRank, &linkList, &linkNum);
        if (ret == HCCL_E_NOT_FOUND) {
            continue;
        }
        CHK_RET(ret);
        CHK_PRT_RET(linkList == nullptr && linkNum != 0,
            HCCL_ERROR("Invalid link list between rank[%u] and rank[%u] on layer[%u]", localRank, remoteRank,
                netLayer),
            HCCL_E_INTERNAL);

        const CommLink *selectedLink = nullptr;
        uint32_t selectedPriority = std::numeric_limits<uint32_t>::max();
        for (uint32_t linkIdx = 0; linkIdx < linkNum; linkIdx++) {
            uint32_t priority = GetProtocolPriority(linkList[linkIdx].linkAttr.linkProtocol);
            if (priority < selectedPriority) {
                selectedLink = &linkList[linkIdx];
                selectedPriority = priority;
            }
        }

        // 网络层编号越小越接近本地直连；当前层没有可用链路时，
        // 再尝试更高层的Clos链路。
        if (selectedLink == nullptr) {
            continue;
        }

        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        channelDesc.remoteRank = remoteRank;
        channelDesc.channelProtocol = selectedLink->linkAttr.linkProtocol;
        channelDesc.localEndpoint = selectedLink->srcEndpointDesc;
        channelDesc.remoteEndpoint = selectedLink->dstEndpointDesc;
        channelDesc.notifyNum = CHANNEL_NOTIFY_NUM;
        HCCL_INFO("Select link: localRank[%u], remoteRank[%u], netLayer[%u], protocol[%d]", localRank,
            remoteRank, netLayer, static_cast<int32_t>(channelDesc.channelProtocol));
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("No AICPU_TS link found between rank[%u] and rank[%u]", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireAlgorithmThreads(
    HcclComm comm, CommEngine engine, uint32_t rankSize, uint64_t inputSize, AlgResourceCtx &resourceCtx)
{
    uint32_t channelNum = rankSize > 1 ? rankSize - 1 : 0;
    uint32_t communicationThreadNum = channelNum;
    if (inputSize <= DIRECT_SMALL_DATA_THRESHOLD && communicationThreadNum > DIRECT_SMALL_COMM_THREAD_NUM) {
        communicationThreadNum = DIRECT_SMALL_COMM_THREAD_NUM;
    }
    bool useDedicatedLocalCopyThread = rankSize > 1 && inputSize > DIRECT_LOCAL_COPY_THREAD_THRESHOLD;
    uint32_t threadNum =
        std::max(1U, communicationThreadNum + static_cast<uint32_t>(useDedicatedLocalCopyThread));

    // 小数据由两条通信线程轮询承担全部Channel，降低线程同步开销；其他数据仍然一条Channel对应一条线程。
    // threads[0]同时承担Host/Device同步。
    // 大数据额外使用最后一条线程执行本地拷贝，使本地拷贝与全部网络链路并行。
    // 主线程的Notify 0保留给Host/Device同步，Notify 1..threadNum-1用于等待其余线程结束；
    // 其他线程仅使用Notify 0等待主线程启动。
    uint32_t notifyNumPerThread = std::max(1U, threadNum);
    resourceCtx.threads.resize(threadNum);
    CHK_RET(HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, resourceCtx.threads.data()));
    resourceCtx.aicpuThread = resourceCtx.threads[0];
    return HCCL_SUCCESS;
}

HcclResult GetSortedNetworkLayers(
    HcclComm comm, std::vector<uint32_t> &netLayers)
{
    uint32_t *netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(
        comm, &netLayerList, &netLayerNum));
    CHK_PRT_RET(
        netLayerList == nullptr || netLayerNum == 0,
        HCCL_ERROR("Rank graph does not contain any network layer"),
        HCCL_E_NOT_FOUND);
    netLayers.assign(netLayerList, netLayerList + netLayerNum);
    std::sort(netLayers.begin(), netLayers.end());
    return HCCL_SUCCESS;
}

uint32_t GetRemoteRank(const OpParam &param, uint32_t channelIndex,
                       bool useSmallDataCyclicOrder)
{
    if (useSmallDataCyclicOrder) {
        uint32_t remoteRank = param.myRank + channelIndex + 1;
        if (remoteRank >= param.rankSize) {
            remoteRank -= param.rankSize;
        }
        return remoteRank;
    }
    return channelIndex >= param.myRank ? channelIndex + 1 : channelIndex;
}

HcclResult BuildChannelDescriptions(
    HcclComm comm, const OpParam &param,
    const std::vector<uint32_t> &netLayers,
    std::vector<HcclMemHandle> &memHandles,
    uint64_t inputSize,
    std::vector<HcclChannelDesc> &channelDescs,
    std::vector<uint32_t> &remoteRanks)
{
    const bool useCyclicOrder =
        inputSize <= DIRECT_SMALL_DATA_THRESHOLD;
    for (uint32_t channelIndex = 0;
         channelIndex < channelDescs.size(); channelIndex++) {
        const uint32_t remoteRank =
            GetRemoteRank(param, channelIndex, useCyclicOrder);
        CHK_RET(FillChannelDesc(
            comm, param.myRank, remoteRank, netLayers,
            channelDescs[channelIndex]));
        channelDescs[channelIndex].memHandles = memHandles.data();
        channelDescs[channelIndex].memHandleNum =
            static_cast<uint32_t>(memHandles.size());
        remoteRanks[channelIndex] = remoteRank;
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateRemoteBuffer(
    const CommBuffer &buffer, uint64_t requiredSize,
    uint32_t remoteRank, const char *bufferName)
{
    CHK_PRT_RET(
        buffer.addr == nullptr || buffer.size < requiredSize,
        HCCL_ERROR(
            "Invalid remote %s buffer for rank[%u]: size[%llu], required[%llu]",
            bufferName, remoteRank,
            static_cast<unsigned long long>(buffer.size),
            static_cast<unsigned long long>(requiredSize)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult PopulateChannelInfo(
    HcclComm comm, const std::vector<ChannelHandle> &channelHandles,
    const std::vector<uint32_t> &remoteRanks,
    const char *inputMemSuffix, uint64_t inputSize,
    const char *outputMemSuffix, uint64_t outputSize,
    AlgResourceCtx &resourceCtx)
{
    resourceCtx.channels.resize(channelHandles.size());
    for (uint32_t index = 0; index < channelHandles.size(); index++) {
        ChannelInfo &channel = resourceCtx.channels[index];
        channel.remoteRank = remoteRanks[index];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = channelHandles[index];
        CHK_RET(GetRemoteMemBySuffix(
            comm, channel.handle, inputMemSuffix, channel.remoteInputMem));
        CHK_RET(ValidateRemoteBuffer(
            channel.remoteInputMem, inputSize, channel.remoteRank, "input"));
        CHK_RET(GetRemoteMemBySuffix(
            comm, channel.handle, outputMemSuffix, channel.remoteOutputMem));
        CHK_RET(ValidateRemoteBuffer(
            channel.remoteOutputMem, outputSize, channel.remoteRank, "output"));
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireAlgorithmChannels(HcclComm comm, CommEngine engine, const OpParam &param,
    std::vector<HcclMemHandle> &memHandles, const char *inputMemSuffix, uint64_t inputSize,
    const char *outputMemSuffix, uint64_t outputSize, AlgResourceCtx &resourceCtx)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(memHandles.size() != 2,
        HCCL_ERROR("Direct AllGather requires input and output memory handles, actual[%zu]", memHandles.size()),
        HCCL_E_PARA);

    std::vector<uint32_t> netLayers;
    CHK_RET(GetSortedNetworkLayers(comm, netLayers));
    const uint32_t channelNum = param.rankSize - 1;
    std::vector<HcclChannelDesc> channelDescs(channelNum);
    std::vector<ChannelHandle> channelHandles(channelNum);
    std::vector<uint32_t> remoteRanks(channelNum);
    CHK_RET(BuildChannelDescriptions(
        comm, param, netLayers, memHandles, inputSize,
        channelDescs, remoteRanks));
    CHK_RET(HcclChannelAcquire(
        comm, engine, channelDescs.data(), channelNum,
        channelHandles.data()));
    return PopulateChannelInfo(
        comm, channelHandles, remoteRanks, inputMemSuffix, inputSize,
        outputMemSuffix, outputSize, resourceCtx);
}

HcclResult InitializeOpParam(
    void *sendBuf, void *recvBuf, uint64_t sendCount,
    HcclDataType dataType, OpParam &param, uint64_t &inputSize)
{
    uint32_t dataTypeSize = 0;
    CHK_RET(GetDataTypeSize(dataType, dataTypeSize));
    CHK_PRT_RET(
        sendCount > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR(
            "Input data size overflows: count[%llu], dataTypeSize[%u]",
            static_cast<unsigned long long>(sendCount), dataTypeSize),
        HCCL_E_PARA);
    inputSize = sendCount * dataTypeSize;
    int32_t tagLength = sprintf_s(
        param.tag, sizeof(param.tag),
        "hccl_custom_allgather_direct_%p_%p_%llu_%d",
        sendBuf, recvBuf, static_cast<unsigned long long>(sendCount),
        static_cast<int32_t>(dataType));
    CHK_PRT_RET(
        tagLength < 0 ||
            static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("Failed to build direct AllGather resource tag"),
        HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    return HCCL_SUCCESS;
}

HcclResult RegisterDfxInfo(HcclComm comm, HcclDfxOpInfo &dfxInfo)
{
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    return HcclDfxRegOpInfoByCommId(
        commName, reinterpret_cast<void *>(&dfxInfo));
}

HcclResult ResolveTopologyAndOutputSize(
    HcclComm comm, uint64_t inputSize,
    OpParam &param, uint64_t &outputSize)
{
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(
        param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR(
            "Invalid rank information: myRank[%u], rankSize[%u]",
            param.myRank, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(
        inputSize > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR(
            "Output data size overflows: inputSize[%llu], rankSize[%u]",
            static_cast<unsigned long long>(inputSize), param.rankSize),
        HCCL_E_PARA);
    outputSize = inputSize * param.rankSize;
    return HCCL_SUCCESS;
}

HcclResult BuildUserMemoryTags(
    const OpParam &param, char *inputMemTag, size_t inputMemTagSize,
    char *outputMemTag, size_t outputMemTagSize)
{
    int32_t inputTagLength = sprintf_s(
        inputMemTag, inputMemTagSize, "%s%s", param.tag, INPUT_MEM_SUFFIX);
    int32_t outputTagLength = sprintf_s(
        outputMemTag, outputMemTagSize, "%s%s", param.tag, OUTPUT_MEM_SUFFIX);
    CHK_PRT_RET(
        inputTagLength < 0 ||
            static_cast<size_t>(inputTagLength) >= inputMemTagSize ||
        outputTagLength < 0 ||
            static_cast<size_t>(outputTagLength) >= outputMemTagSize,
        HCCL_ERROR("Failed to build user memory tags"),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult AcquireSyncThreads(
    HcclComm comm, aclrtStream stream,
    CommEngine cpuTsEngine, CommEngine aicpuTsEngine,
    OpParam &param)
{
    CHK_RET(HcclThreadAcquireWithStream(
        comm, cpuTsEngine, stream, 1, &param.cpuThread));
    return HcclThreadExportToCommEngine(
        comm, 1, &param.cpuThread, aicpuTsEngine,
        &param.cpuThreadOnAicpu);
}

HcclResult ReuseResourceContext(
    HcclComm comm, CommEngine cpuTsEngine,
    void *context, uint64_t contextSize,
    void *sendBuf, void *recvBuf,
    uint64_t inputSize, uint64_t outputSize,
    OpParam &param)
{
    param.resCtx = context;
    param.ctxSize = contextSize;
    void *hostContext = nullptr;
    uint64_t hostContextSize = 0;
    CHK_RET(HcclEngineCtxGet(
        comm, param.tag, cpuTsEngine,
        &hostContext, &hostContextSize));
    CHK_PRT_RET(
        hostContext == nullptr ||
            hostContextSize != sizeof(HostResourceCache),
        HCCL_ERROR("Invalid direct AllGather Host resource cache"),
        HCCL_E_INTERNAL);
    auto *cache = static_cast<HostResourceCache *>(hostContext);
    CHK_PRT_RET(
        cache->inputPtr != sendBuf || cache->outputPtr != recvBuf ||
        cache->inputSize != inputSize || cache->outputSize != outputSize,
        HCCL_ERROR("Cached user buffers do not match current AllGather buffers"),
        HCCL_E_PARA);
    return HcclThreadExportToCommEngine(
        comm, 1, &cache->aicpuThread, cpuTsEngine,
        &param.aicpuThreadOnCpu);
}

HcclResult RegisterUserBuffers(
    HcclComm comm, const OpParam &param,
    const char *inputMemTag, const char *outputMemTag,
    uint64_t inputSize, uint64_t outputSize,
    std::vector<HcclMemHandle> &memHandles)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    CommMem inputMem{COMM_MEM_TYPE_DEVICE, param.inputPtr, inputSize};
    HcclMemHandle inputMemHandle = nullptr;
    CHK_RET(HcclCommMemReg(
        comm, inputMemTag, &inputMem, &inputMemHandle));
    CHK_PTR_NULL(inputMemHandle);
    memHandles.push_back(inputMemHandle);
    CommMem outputMem{COMM_MEM_TYPE_DEVICE, param.outputPtr, outputSize};
    HcclMemHandle outputMemHandle = nullptr;
    CHK_RET(HcclCommMemReg(
        comm, outputMemTag, &outputMem, &outputMemHandle));
    CHK_PTR_NULL(outputMemHandle);
    memHandles.push_back(outputMemHandle);
    return HCCL_SUCCESS;
}

HcclResult StoreResourceContexts(
    HcclComm comm, CommEngine aicpuTsEngine, CommEngine cpuTsEngine,
    AlgResourceCtx &resCtxHost, void *sendBuf, void *recvBuf,
    uint64_t inputSize, uint64_t outputSize, OpParam &param)
{
    std::vector<char> sequence = resCtxHost.Serialize();
    param.ctxSize = sequence.size();
    CHK_RET(HcclEngineCtxCreate(
        comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(
        comm, aicpuTsEngine, param.tag,
        sequence.data(), sequence.size(), 0));
    void *hostContext = nullptr;
    const uint64_t hostContextSize = sizeof(HostResourceCache);
    HostResourceCache cache{
        resCtxHost.aicpuThread, sendBuf, recvBuf, inputSize, outputSize};
    CHK_RET(HcclEngineCtxCreate(
        comm, param.tag, cpuTsEngine, hostContextSize, &hostContext));
    return HcclEngineCtxCopy(
        comm, cpuTsEngine, param.tag, &cache, hostContextSize, 0);
}

HcclResult BuildResourceContext(
    HcclComm comm, CommEngine aicpuTsEngine, CommEngine cpuTsEngine,
    const char *inputMemTag, const char *outputMemTag,
    uint64_t inputSize, uint64_t outputSize,
    void *sendBuf, void *recvBuf, OpParam &param)
{
    AlgResourceCtx resCtxHost;
    std::vector<HcclMemHandle> memHandles;
    CHK_RET(RegisterUserBuffers(
        comm, param, inputMemTag, outputMemTag,
        inputSize, outputSize, memHandles));
    CHK_RET(AcquireAlgorithmThreads(
        comm, aicpuTsEngine, param.rankSize, inputSize, resCtxHost));
    CHK_RET(HcclThreadExportToCommEngine(
        comm, 1, &resCtxHost.aicpuThread, cpuTsEngine,
        &param.aicpuThreadOnCpu));
    CHK_RET(AcquireAlgorithmChannels(
        comm, aicpuTsEngine, param, memHandles,
        INPUT_MEM_SUFFIX, inputSize, OUTPUT_MEM_SUFFIX,
        outputSize, resCtxHost));
    return StoreResourceContexts(
        comm, aicpuTsEngine, cpuTsEngine, resCtxHost,
        sendBuf, recvBuf, inputSize, outputSize, param);
}

HcclResult PrepareResourceContext(
    HcclComm comm, CommEngine aicpuTsEngine, CommEngine cpuTsEngine,
    const char *inputMemTag, const char *outputMemTag,
    uint64_t inputSize, uint64_t outputSize,
    void *sendBuf, void *recvBuf, OpParam &param)
{
    void *context = nullptr;
    uint64_t contextSize = 0;
    if (HcclEngineCtxGet(
            comm, param.tag, aicpuTsEngine,
            &context, &contextSize) == HCCL_SUCCESS) {
        HCCL_INFO("Engine context already exists");
        return ReuseResourceContext(
            comm, cpuTsEngine, context, contextSize,
            sendBuf, recvBuf, inputSize, outputSize, param);
    }
    return BuildResourceContext(
        comm, aicpuTsEngine, cpuTsEngine,
        inputMemTag, outputMemTag, inputSize, outputSize,
        sendBuf, recvBuf, param);
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount,
    HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    uint64_t inputSize = 0;
    CHK_RET(InitializeOpParam(
        sendBuf, recvBuf, sendCount, dataType, param, inputSize));
    HcclDfxOpInfo dfxInfo;
    CHK_RET(RegisterDfxInfo(comm, dfxInfo));
    uint64_t outputSize = 0;
    CHK_RET(ResolveTopologyAndOutputSize(
        comm, inputSize, param, outputSize));
    if (inputSize == 0) {
        return HCCL_SUCCESS;
    }

    char inputMemTag[HCCL_RES_TAG_MAX_LEN + 1] = {};
    char outputMemTag[HCCL_RES_TAG_MAX_LEN + 1] = {};
    CHK_RET(BuildUserMemoryTags(
        param, inputMemTag, sizeof(inputMemTag),
        outputMemTag, sizeof(outputMemTag)));
    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;
    CHK_RET(AcquireSyncThreads(
        comm, stream, cpuTsEngine, aicpuTsEngine, param));
    CHK_RET(PrepareResourceContext(
        comm, aicpuTsEngine, cpuTsEngine, inputMemTag, outputMemTag,
        inputSize, outputSize, sendBuf, recvBuf, param));
    return ops_hccl::LaunchAICPUKernel(param, stream);
}
