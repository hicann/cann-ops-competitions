/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>
#include <algorithm>
#include <array>
#include <limits>
#include "custom.h"
#include "exec_op.h"
#include "log.h"
namespace ops_hccl {
namespace {
constexpr uint64_t KILOBYTE = 1024;
constexpr uint64_t MEGABYTE = 1024 * KILOBYTE;
constexpr uint64_t SIZE_512KB_BYTES = 512 * KILOBYTE;
constexpr uint64_t SIZE_512MB_BYTES = 512 * MEGABYTE;
constexpr uint64_t SIZE_400MB_PLUS_4B_BYTES = 400 * MEGABYTE + sizeof(float);
TopologyKind ClassifyTopology(uint32_t rankSize)
{
    switch (rankSize) {
        case 4:
            return TopologyKind::TOPOLOGY_4X1;
        case 12:
            return TopologyKind::TOPOLOGY_8_PLUS_4;
        case 16:
            return TopologyKind::TOPOLOGY_2X8;
        default:
            return TopologyKind::UNSUPPORTED;
    }
}
MessageSizeKind ClassifyMessageSize(uint64_t dataBytes)
{
    switch (dataBytes) {
        case SIZE_512KB_BYTES:
            return MessageSizeKind::SIZE_512KB;
        case SIZE_512MB_BYTES:
            return MessageSizeKind::SIZE_512MB;
        case SIZE_400MB_PLUS_4B_BYTES:
            return MessageSizeKind::SIZE_400MB_PLUS_4B;
        default:
            return MessageSizeKind::UNSUPPORTED;
    }
}
constexpr bool IsDirect12MessageSize(MessageSizeKind messageSize)
{
    return messageSize == MessageSizeKind::SIZE_512MB
           || messageSize == MessageSizeKind::SIZE_400MB_PLUS_4B;
}
CcuKernelId LegacyKernelId(TopologyKind topology, MessageSizeKind messageSize)
{
    const uint32_t topologyOffset = static_cast<uint32_t>(topology) * 3;
    const uint32_t messageOffset = static_cast<uint32_t>(messageSize);
    return static_cast<CcuKernelId>(topologyOffset + messageOffset);
}
} // namespace
HcclResult SelectAlgorithmPlan(uint32_t rankSize, uint64_t dataBytes, AlgorithmPlan &plan)
{
TopologyKind topology = ClassifyTopology(rankSize);
MessageSizeKind messageSize = ClassifyMessageSize(dataBytes);
if (topology == TopologyKind::UNSUPPORTED || messageSize == MessageSizeKind::UNSUPPORTED) {
    HCCL_ERROR("[SelectAlgorithmPlan] Unsupported rankSize [%u] or dataBytes [%llu]", rankSize,
        static_cast<unsigned long long>(dataBytes));
    return HCCL_E_NOT_SUPPORT;
}
const CellDescriptor *cell = FindCellDescriptor(ACTIVE_PACKAGE, topology, messageSize);
if (cell == nullptr) {
    HCCL_ERROR("[SelectAlgorithmPlan] Cell is not configured for active package [%u]",
        static_cast<uint32_t>(ACTIVE_PACKAGE));
    return HCCL_E_NOT_FOUND;
}
plan.package = cell->key.package;
plan.id = cell->planId;
plan.topology = cell->key.topology;
plan.messageSize = cell->key.messageSize;
plan.family = cell->family;
plan.tailPolicy = cell->tailPolicy;
plan.target = cell->target;
plan.resourceProfile = cell->resourceProfile;
plan.kernelId = LegacyKernelId(topology, messageSize);
plan.kernelSlot = static_cast<uint32_t>(messageSize);
plan.streamCount = cell->streamCount;
plan.kernelHandleCount = cell->kernelHandleCount;
plan.tileBytes = cell->tileBytes;
plan.scratchBytes = cell->scratchBytes;
plan.implemented = cell->implemented;
plan.maxSliceBytes = MAX_DATA_SIZE;
return HCCL_SUCCESS;
}
HcclResult GetResourceProfile(ResourceProfileId profileId, ResourceProfile &profile)
{
switch (profileId) {
    case ResourceProfileId::PROFILE_4X1:
    case ResourceProfileId::PROFILE_8_PLUS_4:
    case ResourceProfileId::PROFILE_2X8:
        profile = MakeResourceProfile(profileId);
        return HCCL_SUCCESS;
    default:
        HCCL_ERROR("[GetResourceProfile] Invalid resource profile [%u]", static_cast<uint32_t>(profileId));
        return HCCL_E_PARA;
}
}
HcclResult BuildExecutionSlices(uint64_t dataBytes, uint64_t maxSliceBytes, std::vector<ExecutionSlice> &slices)
{
if (dataBytes == 0 || maxSliceBytes == 0) {
    HCCL_ERROR("[BuildExecutionSlices] Invalid dataBytes [%llu] or maxSliceBytes [%llu]",
        static_cast<unsigned long long>(dataBytes), static_cast<unsigned long long>(maxSliceBytes));
    return HCCL_E_PARA;
}
slices.clear();
for (uint64_t offset = 0; offset < dataBytes;) {
    uint64_t sliceBytes = std::min(maxSliceBytes, dataBytes - offset);
    slices.push_back(ExecutionSlice{offset, sliceBytes});
    offset += sliceBytes;
}
return HCCL_SUCCESS;
}
const char *GetTopologyName(TopologyKind topology)
{
switch (topology) {
    case TopologyKind::TOPOLOGY_4X1:
        return "4x1";
    case TopologyKind::TOPOLOGY_8_PLUS_4:
        return "8plus4";
    case TopologyKind::TOPOLOGY_2X8:
        return "2x8";
    default:
        return "unsupported";
}
}
const char *GetPlanName(PlanId planId)
{
switch (planId) {
    case PlanId::PLAN_4X1_512KB:
        return "4x1_512KB";
    case PlanId::PLAN_4X1_512MB:
        return "4x1_512MB";
    case PlanId::PLAN_4X1_400MB_PLUS_4B:
        return "4x1_400MB_PLUS_4B";
    case PlanId::PLAN_8_PLUS_4_512KB:
        return "8plus4_512KB";
    case PlanId::PLAN_8_PLUS_4_512MB:
        return "8plus4_512MB";
    case PlanId::PLAN_8_PLUS_4_400MB_PLUS_4B:
        return "8plus4_400MB_PLUS_4B";
    case PlanId::PLAN_2X8_512KB:
        return "2x8_512KB";
    case PlanId::PLAN_2X8_512MB:
        return "2x8_512MB";
    case PlanId::PLAN_2X8_400MB_PLUS_4B:
        return "2x8_400MB_PLUS_4B";
    default:
        return "invalid";
}
}
const char *GetKernelName(CcuKernelId kernelId)
{
switch (kernelId) {
    case CcuKernelId::KERNEL_4X1_512KB:
        return "CcuAllReduce4x1_512KbKernel";
    case CcuKernelId::KERNEL_4X1_512MB:
        return "CcuAllReduce4x1_512MbKernel";
    case CcuKernelId::KERNEL_4X1_400MB_PLUS_4B:
        return "CcuAllReduce4x1_400MbPlus4BKernel";
    case CcuKernelId::KERNEL_8_PLUS_4_512KB:
        return "CcuAllReduce8Plus4_512KbKernel";
    case CcuKernelId::KERNEL_8_PLUS_4_512MB:
        return "CcuAllReduce8Plus4_512MbKernel";
    case CcuKernelId::KERNEL_8_PLUS_4_400MB_PLUS_4B:
        return "CcuAllReduce8Plus4_400MbPlus4BKernel";
    case CcuKernelId::KERNEL_2X8_512KB:
        return "CcuAllReduce2x8_512KbKernel";
    case CcuKernelId::KERNEL_2X8_512MB:
        return "CcuAllReduce2x8_512MbKernel";
    case CcuKernelId::KERNEL_2X8_400MB_PLUS_4B:
        return "CcuAllReduce2x8_400MbPlus4BKernel";
    default:
        return "invalid";
}
}
namespace {
HcclResult Direct12DieHandle(const AlgResourceCtx &resourceContext, DieEndpoint die,
CcuStageKernelId &stage, CcuKernelHandle &handle)
{
CHK_PRT_RET(die != DieEndpoint::DIE_0 && die != DieEndpoint::DIE_1,
    HCCL_ERROR("[ExecOp] Direct12 stage has no concrete IO Die"), HCCL_E_PARA);
stage = die == DieEndpoint::DIE_0 ? CcuStageKernelId::EIGHT_PLUS_FOUR_DIRECT12_RS_L0
                                 : CcuStageKernelId::EIGHT_PLUS_FOUR_DIRECT12_RS_L0_DIE1;
for (uint32_t index = 0; index < resourceContext.registeredStageCount; ++index) {
    if (resourceContext.registeredStages[index].stageId == stage) {
        handle = resourceContext.registeredStages[index].handle;
        return handle == 0 ? HCCL_E_INTERNAL : HCCL_SUCCESS;
    }
}
return HCCL_E_NOT_FOUND;
}
HcclResult ResolveThreadForDie(
const AlgResourceCtx &resourceContext, DieEndpoint die, uint32_t worker, ThreadHandle &thread)
{
CHK_PRT_RET(die != DieEndpoint::DIE_0 && die != DieEndpoint::DIE_1,
    HCCL_ERROR("[ExecOp] Cell stage has no concrete IO Die"), HCCL_E_PARA);
bool hasDie0 = false;
bool hasDie1 = false;
for (uint32_t index = 0; index < resourceContext.peerChannelCount; ++index) {
    hasDie0 = hasDie0 || resourceContext.peerChannels[index].localDie == DieEndpoint::DIE_0;
    hasDie1 = hasDie1 || resourceContext.peerChannels[index].localDie == DieEndpoint::DIE_1;
}
if (resourceContext.workerCount == 1) {
    thread = resourceContext.mainThread;
} else if (!hasDie0 || !hasDie1) {
    CHK_PRT_RET(worker >= resourceContext.workerCount,
        HCCL_ERROR("[ExecOp] Single-Die cell worker is out of range"), HCCL_E_PARA);
    thread = worker == 0 ? resourceContext.mainThread : resourceContext.slaveThreads[0];
} else if (die == DieEndpoint::DIE_0) {
    thread = resourceContext.mainThread;
} else {
    CHK_PRT_RET(resourceContext.workerCount != 2 || resourceContext.slaveThreadCount != 1,
        HCCL_ERROR("[ExecOp] Die1 stage requires the second bound worker"), HCCL_E_INTERNAL);
    thread = resourceContext.slaveThreads[0];
}
return thread == 0 ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}
HcclResult CellThreadRecord(
ThreadHandle source, ThreadHandle destination, uint32_t notifyIndex, const char *name)
{
CHK_PRT_RET(notifyIndex >= CELL_THREAD_NOTIFY_COUNT,
    HCCL_ERROR("[ExecOp] Cell thread notify record index is out of range: %u", notifyIndex), HCCL_E_PARA);
const int32_t result = HcommThreadNotifyRecordOnThread(source, destination, notifyIndex);
CHK_PRT_RET(result != HCCL_SUCCESS,
    HCCL_ERROR("[ExecOp] Cell thread notify record failed: %s, result=%d", name, result),
    static_cast<HcclResult>(result));
return HCCL_SUCCESS;
}
HcclResult CellThreadWait(ThreadHandle thread, uint32_t notifyIndex, const char *name)
{
CHK_PRT_RET(notifyIndex >= CELL_THREAD_NOTIFY_COUNT,
    HCCL_ERROR("[ExecOp] Cell thread notify wait index is out of range: %u", notifyIndex), HCCL_E_PARA);
const int32_t result = HcommThreadNotifyWaitOnThreadWithDefaultTimeout(thread, notifyIndex);
CHK_PRT_RET(result != HCCL_SUCCESS,
    HCCL_ERROR("[ExecOp] Cell thread notify wait failed: %s, result=%d", name, result),
    static_cast<HcclResult>(result));
return HCCL_SUCCESS;
}
HcclResult CellCrossFence(const AlgResourceCtx &resourceContext, uint32_t notifyIndex)
{
CHK_PRT_RET(resourceContext.workerCount != 2 || resourceContext.slaveThreadCount != 1,
    HCCL_ERROR("[ExecOp] Cross-stream fence requires exactly two workers"), HCCL_E_INTERNAL);
const ThreadHandle worker0 = resourceContext.mainThread;
const ThreadHandle worker1 = resourceContext.slaveThreads[0];
CHK_RET(CellThreadRecord(worker0, worker1, notifyIndex, "worker0_to_worker1"));
CHK_RET(CellThreadRecord(worker1, worker0, notifyIndex, "worker1_to_worker0"));
CHK_RET(CellThreadWait(worker0, notifyIndex, "worker1_to_worker0"));
CHK_RET(CellThreadWait(worker1, notifyIndex, "worker0_to_worker1"));
return HCCL_SUCCESS;
}
HcclResult CellOneWayDependency(
const AlgResourceCtx &resourceContext, uint32_t notifyIndex, const char *name)
{
CHK_PRT_RET(resourceContext.workerCount != ACTIVE_WORKERS || resourceContext.slaveThreadCount != 1,
    HCCL_ERROR("[ExecOp] One-way worker dependency requires exactly two workers: %s", name),
    HCCL_E_INTERNAL);
CHK_RET(CellThreadRecord(
    resourceContext.slaveThreads[0], resourceContext.mainThread, notifyIndex, name));
return CellThreadWait(resourceContext.mainThread, notifyIndex, name);
}
HcclResult CellStageOneWayHandoff(const AlgResourceCtx &ctx, uint32_t producer,
uint32_t consumer, uint32_t notifyIndex, const char *name)
{
CHK_PRT_RET(producer >= ACTIVE_WORKERS || consumer >= ACTIVE_WORKERS
                || ctx.workerCount != ACTIVE_WORKERS || ctx.slaveThreadCount != 1
                || producer == consumer,
    HCCL_ERROR("[ExecOp] One-way stage handoff has invalid workers: %s", name), HCCL_E_INTERNAL);
const ThreadHandle workers[ACTIVE_WORKERS] = {ctx.mainThread, ctx.slaveThreads[0]};
CHK_RET(CellThreadRecord(workers[producer], workers[consumer], notifyIndex, name));
return CellThreadWait(workers[consumer], notifyIndex, name);
}
HcclResult LaunchCellStage(const AlgResourceCtx &resourceContext, CcuKernelHandle handle,
const StageLaunchSpec &launch, const TaskArgs &taskArgs, const char *planName, const char *stageName)
{
CHK_PRT_RET(handle == 0 || launch.worker >= resourceContext.workerCount
                || launch.creditSlot != Direct12CreditSlot(launch.worker, 0),
    HCCL_ERROR("[ExecOp] Invalid cell launch plan=%s stage=%s worker=%u slot=%u", planName, stageName,
        launch.worker, launch.creditSlot),
    HCCL_E_PARA);
ThreadHandle thread = 0;
CHK_RET(ResolveThreadForDie(resourceContext, launch.die, launch.worker, thread));
const CcuResult launchResult = HcommCcuKernelLaunch(thread, handle, taskArgs.Data(), taskArgs.Size());
CHK_PRT_RET(launchResult != CCU_SUCCESS,
    HCCL_ERROR("[ExecOp] Cell launch failed plan=%s stage=%s worker=%u slot=%u tile=%u "
               "tileOffset=%llu tileBytes=%llu result=%d",
        planName, stageName, launch.worker, launch.creditSlot, launch.tile,
        static_cast<unsigned long long>(taskArgs.tileOffsetBytes),
        static_cast<unsigned long long>(taskArgs.tileElementCount * sizeof(float)),
        static_cast<int32_t>(launchResult)),
    ConvertCcuToHccl(launchResult));
return HCCL_SUCCESS;
}
HcclResult LaunchDirect12DieStage(const AlgResourceCtx &resourceContext,
    const InvocationMemory &memory, uint64_t shardOffsetElements, uint64_t shardElements, uint32_t tile,
    uint32_t worker, uint32_t slot, DieEndpoint die, uint64_t flags, const char *stageName)
{
    CHK_PRT_RET(flags != DIRECT12_FLAG_GATHER_ALL_PUBLISH
                    && flags != DIRECT12_FLAG_GATHER_ALL
                    && flags != DIRECT12_FLAG_JOIN
                    && flags != DIRECT12_FLAG_BROADCAST_ALL
                    && flags != DIRECT12_FLAG_BROADCAST_GATHER_SCALAR
                    && flags != DIRECT12_FLAG_SCALAR_JOIN,
        HCCL_ERROR("[ExecOp] Direct12 stage has an unsupported flag: %llu",
            static_cast<unsigned long long>(flags)),
        HCCL_E_PARA);
    CcuStageKernelId stage = CcuStageKernelId::INVALID;
    CcuKernelHandle handle = 0;
    CHK_RET(Direct12DieHandle(resourceContext, die, stage, handle));
    const uint64_t tileElements = Direct12TileElementCount(shardElements, tile, resourceContext.tileBytes);
    const uint64_t tileBytes = tileElements * sizeof(float);
    CHK_PRT_RET(worker >= ACTIVE_WORKERS || slot >= ACTIVE_WORKERS * PIPELINE_GENERATIONS || tileElements == 0,
        HCCL_ERROR("[ExecOp] Invalid Direct12 task plan=%s stage=%s worker=%u slot=%u tileOffset=%llu tileBytes=%llu",
            "PACKAGE_A_8PLUS4_DIRECT12", stageName, worker, slot,
            static_cast<unsigned long long>(static_cast<uint64_t>(tile) * resourceContext.tileBytes),
            static_cast<unsigned long long>(tileBytes)),
        HCCL_E_PARA);

    TaskArgs taskArgs{};
    taskArgs.inputAddr = memory.input.address;
    taskArgs.inputToken = memory.input.token;
    taskArgs.outputAddr = memory.output.address;
    taskArgs.outputToken = memory.output.token;
    taskArgs.scratchAddr
        = memory.scratch.address + worker * Direct12ScratchBankBytes(resourceContext.tileBytes);
    taskArgs.scratchToken = memory.scratch.token;
    taskArgs.dataOffsetBytes = shardOffsetElements * sizeof(float);
    taskArgs.elementCount = shardElements;
    taskArgs.tileOffsetBytes = static_cast<uint64_t>(tile) * resourceContext.tileBytes;
    taskArgs.tileElementCount = tileElements;
    taskArgs.generation = slot;
    taskArgs.flags = flags;

    const StageLaunchSpec launch{stage, die, worker, slot, tile, taskArgs.flags};
    return LaunchCellStage(resourceContext, handle, launch, taskArgs, "PACKAGE_A_8PLUS4_DIRECT12", stageName);
}

HcclResult LaunchDirect12PairDieStage(const AlgResourceCtx &resourceContext,
    const InvocationMemory &memory, uint64_t shardOffsetElements, uint64_t shardElements,
    uint32_t firstTile, DieEndpoint die, uint64_t flags, const char *stageName)
{
    CHK_PRT_RET(flags != DIRECT12_FLAG_PAIR_GATHER_ALL_PUBLISH
                    && flags != DIRECT12_FLAG_PAIR_BROADCAST_ALL,
        HCCL_ERROR("[ExecOp] Direct12 pair stage has an unsupported flag: %llu",
            static_cast<unsigned long long>(flags)),
        HCCL_E_PARA);
    CcuStageKernelId stage = CcuStageKernelId::INVALID;
    CcuKernelHandle handle = 0;
    CHK_RET(Direct12DieHandle(resourceContext, die, stage, handle));

    const uint64_t firstTileElements
        = Direct12TileElementCount(shardElements, firstTile, resourceContext.tileBytes);
    const uint64_t secondTileElements
        = Direct12TileElementCount(shardElements, firstTile + 1, resourceContext.tileBytes);
    const uint64_t strideElements = Direct12TileElements(resourceContext.tileBytes);
    CHK_PRT_RET(firstTile % ACTIVE_WORKERS != 0 || firstTileElements != strideElements
                    || secondTileElements == 0 || secondTileElements > strideElements,
        HCCL_ERROR("[ExecOp] Invalid Direct12 pair task stage=%s firstTile=%u firstBytes=%llu secondBytes=%llu",
            stageName, firstTile, static_cast<unsigned long long>(firstTileElements * sizeof(float)),
            static_cast<unsigned long long>(secondTileElements * sizeof(float))),
        HCCL_E_PARA);

    constexpr uint32_t pairWorker = 0;
    constexpr uint32_t pairGeneration = 0;
    constexpr uint32_t pairSlot = Direct12CreditSlot(pairWorker, pairGeneration);
    TaskArgs taskArgs{};
    taskArgs.inputAddr = memory.input.address;
    taskArgs.inputToken = memory.input.token;
    taskArgs.outputAddr = memory.output.address;
    taskArgs.outputToken = memory.output.token;
    taskArgs.scratchAddr = memory.scratch.address;
    taskArgs.scratchToken = memory.scratch.token;
    taskArgs.dataOffsetBytes = shardOffsetElements * sizeof(float);
    taskArgs.elementCount = shardElements;
    taskArgs.tileOffsetBytes = static_cast<uint64_t>(firstTile) * resourceContext.tileBytes;
    // Pair kernels know the first tile is one full static stride; this field carries the second tile length.
    taskArgs.tileElementCount = secondTileElements;
    taskArgs.generation = pairSlot;
    taskArgs.flags = flags;

    const StageLaunchSpec launch{stage, die, pairWorker, pairSlot, firstTile, taskArgs.flags};
    CHK_PRT_RET(handle == 0 || launch.worker >= resourceContext.workerCount
                    || launch.creditSlot != Direct12CreditSlot(launch.worker, 0),
        HCCL_ERROR("[ExecOp] Invalid Direct12 pair launch stage=%s worker=%u slot=%u",
            stageName, launch.worker, launch.creditSlot),
        HCCL_E_PARA);
    ThreadHandle thread = 0;
    CHK_RET(ResolveThreadForDie(resourceContext, launch.die, launch.worker, thread));
    const CcuResult launchResult = HcommCcuKernelLaunch(thread, handle, taskArgs.Data(), taskArgs.Size());
    CHK_PRT_RET(launchResult != CCU_SUCCESS,
        HCCL_ERROR("[ExecOp] Direct12 pair launch failed stage=%s worker=%u slot=%u "
                   "firstTile=%u firstOffset=%llu firstBytes=%llu "
                   "secondTile=%u secondOffset=%llu secondBytes=%llu result=%d",
            stageName, launch.worker, launch.creditSlot, firstTile,
            static_cast<unsigned long long>(taskArgs.tileOffsetBytes),
            static_cast<unsigned long long>(firstTileElements * sizeof(float)), firstTile + 1,
            static_cast<unsigned long long>(taskArgs.tileOffsetBytes + resourceContext.tileBytes),
            static_cast<unsigned long long>(secondTileElements * sizeof(float)),
            static_cast<int32_t>(launchResult)),
        ConvertCcuToHccl(launchResult));
    return HCCL_SUCCESS;
}

HcclResult LaunchDirect12TransitionDieStage(const AlgResourceCtx &resourceContext,
    const InvocationMemory &memory, uint64_t shardOffsetElements, uint64_t shardElements,
    uint32_t currentFirstTile, DieEndpoint die, uint64_t flags, const char *stageName)
{
    CHK_PRT_RET(flags != DIRECT12_FLAG_PAIR_BROADCAST_GATHER_NEXT,
        HCCL_ERROR("[ExecOp] Direct12 transition has an unsupported flag: %llu",
            static_cast<unsigned long long>(flags)),
        HCCL_E_PARA);
    CcuStageKernelId stage = CcuStageKernelId::INVALID;
    CcuKernelHandle handle = 0;
    CHK_RET(Direct12DieHandle(resourceContext, die, stage, handle));

    const uint32_t nextFirstTile = currentFirstTile + ACTIVE_WORKERS;
    const uint64_t strideElements = Direct12TileElements(resourceContext.tileBytes);
    const uint64_t currentFirstElements
        = Direct12TileElementCount(shardElements, currentFirstTile, resourceContext.tileBytes);
    const uint64_t currentSecondElements
        = Direct12TileElementCount(shardElements, currentFirstTile + 1, resourceContext.tileBytes);
    const uint64_t nextFirstElements
        = Direct12TileElementCount(shardElements, nextFirstTile, resourceContext.tileBytes);
    const uint64_t nextSecondElements
        = Direct12TileElementCount(shardElements, nextFirstTile + 1, resourceContext.tileBytes);
    const bool nextIsPair = nextSecondElements != 0;
    CHK_PRT_RET(currentFirstTile % ACTIVE_WORKERS != 0
                    || currentFirstElements != strideElements || currentSecondElements != strideElements
                    || nextFirstElements == 0 || nextFirstElements > strideElements
                    || (nextIsPair && (nextFirstElements != strideElements
                                          || nextSecondElements == 0 || nextSecondElements > strideElements))
                    || (!nextIsPair && nextSecondElements != 0),
        HCCL_ERROR("[ExecOp] Invalid Direct12 transition stage=%s currentFirst=%u "
                   "currentFirstBytes=%llu currentSecondBytes=%llu nextFirst=%u "
                   "nextFirstBytes=%llu nextSecondBytes=%llu",
            stageName, currentFirstTile,
            static_cast<unsigned long long>(currentFirstElements * sizeof(float)),
            static_cast<unsigned long long>(currentSecondElements * sizeof(float)), nextFirstTile,
            static_cast<unsigned long long>(nextFirstElements * sizeof(float)),
            static_cast<unsigned long long>(nextSecondElements * sizeof(float))),
        HCCL_E_PARA);

    constexpr uint32_t transitionWorker = 0;
    constexpr uint32_t transitionSlot = Direct12CreditSlot(transitionWorker, 0);
    TaskArgs taskArgs{};
    taskArgs.inputAddr = memory.input.address;
    taskArgs.inputToken = memory.input.token;
    taskArgs.outputAddr = memory.output.address;
    taskArgs.outputToken = memory.output.token;
    taskArgs.scratchAddr = memory.scratch.address;
    taskArgs.scratchToken = memory.scratch.token;
    taskArgs.dataOffsetBytes = shardOffsetElements * sizeof(float);
    taskArgs.elementCount = shardElements;
    taskArgs.tileOffsetBytes = static_cast<uint64_t>(currentFirstTile) * resourceContext.tileBytes;
    // The next-first length drives the common Gather. This schedule does not consume runtime generations, so the
    // generation field carries the optional next-second length without growing the fixed 12-argument ABI.
    taskArgs.tileElementCount = nextFirstElements;
    taskArgs.generation = nextSecondElements;
    taskArgs.flags = flags;

    const StageLaunchSpec launch{stage, die, transitionWorker, transitionSlot, currentFirstTile, flags};
    return LaunchCellStage(
        resourceContext, handle, launch, taskArgs, "PACKAGE_A_8PLUS4_DIRECT12", stageName);
}

HcclResult FullExchangeStageHandle(
const AlgResourceCtx &resourceContext, DieEndpoint die, CcuKernelHandle &handle)
{
const CcuStageKernelId stage = die == DieEndpoint::DIE_0
                                   ? CcuStageKernelId::EIGHT_PLUS_FOUR_ALL_EXCHANGE_L0
                                   : CcuStageKernelId::EIGHT_PLUS_FOUR_ALL_EXCHANGE_L1;
CHK_PRT_RET(die != DieEndpoint::DIE_0 && die != DieEndpoint::DIE_1,
    HCCL_ERROR("[ExecOp] FullExchange stage has no concrete IO Die"), HCCL_E_PARA);
for (uint32_t index = 0; index < resourceContext.registeredStageCount; ++index) {
    if (resourceContext.registeredStages[index].stageId == stage) {
        handle = resourceContext.registeredStages[index].handle;
        return handle == 0 ? HCCL_E_INTERNAL : HCCL_SUCCESS;
    }
}
return HCCL_E_NOT_FOUND;
}
HcclResult FullExchangeJoinDie(const AlgResourceCtx &resourceContext, DieEndpoint &die)
{
die = DieEndpoint::NOT_APPLICABLE;
for (uint32_t index = 0; index < resourceContext.peerChannelCount; ++index) {
    const PeerChannelResource &peer = resourceContext.peerChannels[index];
    if (peer.layer != FabricLayer::L1) {
        continue;
    }
    CHK_PRT_RET(peer.localDie != DieEndpoint::DIE_0 && peer.localDie != DieEndpoint::DIE_1,
        HCCL_ERROR("[ExecOp] FullExchange L1 channel has no concrete IO Die"), HCCL_E_INTERNAL);
    CHK_PRT_RET(die != DieEndpoint::NOT_APPLICABLE && die != peer.localDie,
        HCCL_ERROR("[ExecOp] FullExchange L1 channels span both local IO Dies"), HCCL_E_NOT_SUPPORT);
    die = peer.localDie;
}
CHK_PRT_RET(die == DieEndpoint::NOT_APPLICABLE,
    HCCL_ERROR("[ExecOp] FullExchange has no L1 join-owner Die"), HCCL_E_INTERNAL);
return HCCL_SUCCESS;
}
HcclResult TwoXEightLayerResources(const AlgResourceCtx &resourceContext, FabricLayer layer,
const std::array<CcuStageKernelId, 2> &stageOrder, const char *familyName,
CcuKernelHandle &handle, DieEndpoint &die, uint32_t &worker)
{
CHK_PRT_RET(layer != FabricLayer::L0 && layer != FabricLayer::L1,
    HCCL_ERROR("[ExecOp] 2x8 %s requested an invalid layer", familyName), HCCL_E_PARA);
const CcuStageKernelId stage = layer == FabricLayer::L0
                                  ? stageOrder[0]
                                  : stageOrder[1];
handle = 0;
die = DieEndpoint::NOT_APPLICABLE;
for (uint32_t index = 0; index < resourceContext.registeredStageCount; ++index) {
    if (resourceContext.registeredStages[index].stageId == stage) {
        handle = resourceContext.registeredStages[index].handle;
        break;
    }
}
for (uint32_t index = 0; index < resourceContext.peerChannelCount; ++index) {
    const PeerChannelResource &peer = resourceContext.peerChannels[index];
    if (peer.layer != layer) {
        continue;
    }
    CHK_PRT_RET(peer.localDie != DieEndpoint::DIE_0 && peer.localDie != DieEndpoint::DIE_1,
        HCCL_ERROR("[ExecOp] 2x8 %s layer has no concrete local IO Die", familyName), HCCL_E_INTERNAL);
    CHK_PRT_RET(die != DieEndpoint::NOT_APPLICABLE && die != peer.localDie,
        HCCL_ERROR("[ExecOp] 2x8 %s layer spans both local IO Dies", familyName), HCCL_E_NOT_SUPPORT);
    die = peer.localDie;
}
CHK_PRT_RET(handle == 0 || die == DieEndpoint::NOT_APPLICABLE,
    HCCL_ERROR("[ExecOp] 2x8 %s layer launch resources are incomplete", familyName), HCCL_E_INTERNAL);
worker = die == DieEndpoint::DIE_0 ? 0 : 1;
return HCCL_SUCCESS;
}
HcclResult TwoXEightMultiRootResources(const AlgResourceCtx &resourceContext, FabricLayer layer,
CcuKernelHandle &handle, DieEndpoint &die, uint32_t &worker)
{
return TwoXEightLayerResources(resourceContext, layer, TWO_X_EIGHT_MULTI_ROOT_STAGE_ORDER,
    "MultiRoot", handle, die, worker);
}
HcclResult LaunchTwoXEightMultiRootStage(const AlgResourceCtx &resourceContext,
const InvocationMemory &memory, FabricLayer layer, uint64_t flags, const char *stageName,
uint32_t &launchedWorker)
{
CcuKernelHandle handle = 0;
DieEndpoint die = DieEndpoint::NOT_APPLICABLE;
CHK_RET(TwoXEightMultiRootResources(resourceContext, layer, handle, die, launchedWorker));
const CcuStageKernelId stage = layer == FabricLayer::L0
                                  ? TWO_X_EIGHT_MULTI_ROOT_STAGE_ORDER[0]
                                  : TWO_X_EIGHT_MULTI_ROOT_STAGE_ORDER[1];
TaskArgs taskArgs{};
taskArgs.inputAddr = memory.input.address;
taskArgs.inputToken = memory.input.token;
taskArgs.outputAddr = memory.output.address;
taskArgs.outputToken = memory.output.token;
taskArgs.scratchAddr = memory.scratch.address;
taskArgs.scratchToken = memory.scratch.token;
taskArgs.elementCount = TWO_X_EIGHT_MULTI_ROOT_MESSAGE_BYTES / sizeof(float);
taskArgs.tileElementCount = TWO_X_EIGHT_MULTI_ROOT_SHARD_BYTES / sizeof(float);
taskArgs.generation = Direct12CreditSlot(launchedWorker, 0);
taskArgs.flags = flags;
const StageLaunchSpec launch{stage, die, launchedWorker, Direct12CreditSlot(launchedWorker, 0), 0, flags};
return LaunchCellStage(resourceContext, handle, launch, taskArgs,
    "PACKAGE_A_2X8_512KB_TOPOLOGY_BALANCED_DIRECT_RSAG", stageName);
}
HcclResult TwoXEightDirectedDependency(const AlgResourceCtx &resourceContext,
uint32_t producerWorker, uint32_t consumerWorker, uint32_t notifyIndex, const char *name)
{
CHK_PRT_RET(resourceContext.workerCount != ACTIVE_WORKERS
                || resourceContext.slaveThreadCount != 1
                || producerWorker >= ACTIVE_WORKERS || consumerWorker >= ACTIVE_WORKERS
                || producerWorker == consumerWorker,
    HCCL_ERROR("[ExecOp] 2x8 directed dependency has invalid workers: %s", name),
    HCCL_E_INTERNAL);
const ThreadHandle workers[ACTIVE_WORKERS] = {
    resourceContext.mainThread, resourceContext.slaveThreads[0]};
CHK_RET(CellThreadRecord(
    workers[producerWorker], workers[consumerWorker], notifyIndex, name));
return CellThreadWait(workers[consumerWorker], notifyIndex, name);
}
HcclResult TwoXEightBulkResources(const AlgResourceCtx &resourceContext, FabricLayer layer,
CcuKernelHandle &handle, DieEndpoint &die, uint32_t &worker)
{
return TwoXEightLayerResources(resourceContext, layer, TWO_X_EIGHT_BULK_STAGE_ORDER,
    "Bulk", handle, die, worker);
}
HcclResult LaunchTwoXEightBulkStage(const AlgResourceCtx &resourceContext,
const InvocationMemory &memory, uint64_t messageBytes, FabricLayer layer,
uint32_t iteration, uint64_t dataOffsetBytes,
uint64_t tileBytes, uint64_t layout, uint64_t flags, const char *stageName,
uint32_t &launchedWorker)
{
CcuKernelHandle handle = 0;
DieEndpoint die = DieEndpoint::NOT_APPLICABLE;
CHK_RET(TwoXEightBulkResources(resourceContext, layer, handle, die, launchedWorker));
const CcuStageKernelId stage = layer == FabricLayer::L0
                                  ? TWO_X_EIGHT_BULK_STAGE_ORDER[0]
                                  : TWO_X_EIGHT_BULK_STAGE_ORDER[1];
TaskArgs taskArgs{};
taskArgs.inputAddr = memory.input.address;
taskArgs.inputToken = memory.input.token;
taskArgs.outputAddr = memory.output.address;
taskArgs.outputToken = memory.output.token;
taskArgs.scratchAddr = memory.scratch.address;
taskArgs.scratchToken = memory.scratch.token;
taskArgs.dataOffsetBytes = dataOffsetBytes;
taskArgs.elementCount = messageBytes / sizeof(float);
taskArgs.tileOffsetBytes = dataOffsetBytes;
taskArgs.tileElementCount = tileBytes / sizeof(float);
taskArgs.generation = layout;
taskArgs.flags = flags;
const StageLaunchSpec launch{stage, die, launchedWorker,
    Direct12CreditSlot(launchedWorker, 0), iteration, flags};
return LaunchCellStage(resourceContext, handle, launch, taskArgs,
    "PACKAGE_A_2X8_512MB_BULK_PUSH_STAGED_REDUCE", stageName);
}
struct TwoXEightBulkStageNames {
const char *pushL0;
const char *pushL1;
const char *join;
const char *allGatherL0;
const char *allGatherL1;
};
HcclResult PrepareTwoXEightBulkMemory(const OpParam &param, const AlgResourceCtx &resourceContext,
uint64_t messageBytes, const char *planName, InvocationMemory &memory)
{
CHK_PRT_RET(param.inputPtr == nullptr || param.outputPtr == nullptr
                || param.count != messageBytes / sizeof(float)
                || param.dataType != HCCL_DATA_TYPE_FP32 || param.rankSize != TWO_X_EIGHT_RANK_COUNT
                || resourceContext.workerCount != ACTIVE_WORKERS || resourceContext.slaveThreadCount != 1
                || resourceContext.hcclBuffer.addr == nullptr
                || resourceContext.hcclBuffer.size < resourceContext.scratchBytes
                || resourceContext.localCanonicalRank >= TWO_X_EIGHT_RANK_COUNT,
    HCCL_ERROR("[ExecOp] Invalid 2x8 Bulk invocation parameters plan=%s", planName), HCCL_E_PARA);
memory = InvocationMemory{};
memory.input.address = reinterpret_cast<uint64_t>(param.inputPtr);
memory.output.address = reinterpret_cast<uint64_t>(param.outputPtr);
memory.scratch.address = reinterpret_cast<uint64_t>(resourceContext.hcclBuffer.addr);
CcuResult tokenResult = HcommCcuGetMemToken(
    memory.input.address, messageBytes, &memory.input.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS,
    HCCL_ERROR("[ExecOp] 2x8 Bulk input token acquisition failed plan=%s", planName),
    ConvertCcuToHccl(tokenResult));
tokenResult = HcommCcuGetMemToken(memory.output.address, messageBytes, &memory.output.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS,
    HCCL_ERROR("[ExecOp] 2x8 Bulk output token acquisition failed plan=%s", planName),
    ConvertCcuToHccl(tokenResult));
tokenResult = HcommCcuGetMemToken(
    memory.scratch.address, resourceContext.scratchBytes, &memory.scratch.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS,
    HCCL_ERROR("[ExecOp] 2x8 Bulk scratch token acquisition failed plan=%s", planName),
    ConvertCcuToHccl(tokenResult));
return HCCL_SUCCESS;
}
HcclResult RunTwoXEightBulkTile(const AlgResourceCtx &resourceContext,
const InvocationMemory &memory, uint64_t messageBytes, uint32_t iteration,
uint64_t dataOffsetBytes, uint64_t tileBytes, uint64_t layout,
const TwoXEightBulkStageNames &names, uint32_t &syncPhase)
{
uint32_t l0Worker = 0;
uint32_t l1Worker = 0;
CHK_RET(LaunchTwoXEightBulkStage(resourceContext, memory, messageBytes, FabricLayer::L0,
    iteration, dataOffsetBytes, tileBytes, layout, TWO_X_EIGHT_BULK_FLAG_PUSH,
    names.pushL0, l0Worker));
CHK_RET(LaunchTwoXEightBulkStage(resourceContext, memory, messageBytes, FabricLayer::L1,
    iteration, dataOffsetBytes, tileBytes, layout, TWO_X_EIGHT_BULK_FLAG_PUSH,
    names.pushL1, l1Worker));
CHK_PRT_RET(l0Worker == l1Worker,
    HCCL_ERROR("[ExecOp] 2x8 Bulk L0 and L1 kernels resolved to the same stream"),
    HCCL_E_NOT_SUPPORT);
CHK_RET(TwoXEightDirectedDependency(resourceContext, l1Worker, l0Worker,
    CellInternalNotifyIndex(syncPhase), "2x8_bulk_push_l1_to_join"));
++syncPhase;
uint32_t joinWorker = 0;
CHK_RET(LaunchTwoXEightBulkStage(resourceContext, memory, messageBytes, FabricLayer::L0,
    iteration, dataOffsetBytes, tileBytes, layout, TWO_X_EIGHT_BULK_FLAG_JOIN,
    names.join, joinWorker));
CHK_PRT_RET(joinWorker != l0Worker,
    HCCL_ERROR("[ExecOp] 2x8 Bulk Join did not resolve to the L0 worker"), HCCL_E_INTERNAL);
CHK_RET(TwoXEightDirectedDependency(resourceContext, joinWorker, l1Worker,
    CellInternalNotifyIndex(syncPhase), "2x8_bulk_join_to_allgather"));
++syncPhase;
CHK_RET(LaunchTwoXEightBulkStage(resourceContext, memory, messageBytes, FabricLayer::L0,
    iteration, dataOffsetBytes, tileBytes, layout, TWO_X_EIGHT_BULK_FLAG_ALL_GATHER,
    names.allGatherL0, l0Worker));
CHK_RET(LaunchTwoXEightBulkStage(resourceContext, memory, messageBytes, FabricLayer::L1,
    iteration, dataOffsetBytes, tileBytes, layout, TWO_X_EIGHT_BULK_FLAG_ALL_GATHER,
    names.allGatherL1, l1Worker));
return HCCL_SUCCESS;
}
HcclResult RunTwoXEightBulkOneWave(const AlgResourceCtx &resourceContext,
const InvocationMemory &memory, uint64_t messageBytes, uint64_t tileBytes,
uint64_t layout, const TwoXEightBulkStageNames &names, uint32_t &syncPhase)
{
constexpr uint32_t iteration = 0;
constexpr uint64_t dataOffsetBytes = 0;
uint32_t l0Worker = 0;
uint32_t l1Worker = 0;
CHK_RET(LaunchTwoXEightBulkStage(resourceContext, memory, messageBytes, FabricLayer::L0,
    iteration, dataOffsetBytes, tileBytes, layout, TWO_X_EIGHT_BULK_FLAG_ONE_WAVE_PUSH,
    names.pushL0, l0Worker));
CHK_RET(LaunchTwoXEightBulkStage(resourceContext, memory, messageBytes, FabricLayer::L1,
    iteration, dataOffsetBytes, tileBytes, layout, TWO_X_EIGHT_BULK_FLAG_ONE_WAVE_PUSH,
    names.pushL1, l1Worker));
CHK_PRT_RET(l0Worker == l1Worker,
    HCCL_ERROR("[ExecOp] 2x8 Bulk L0 and L1 kernels resolved to the same stream"),
    HCCL_E_NOT_SUPPORT);
CHK_RET(TwoXEightDirectedDependency(resourceContext, l1Worker, l0Worker,
    CellInternalNotifyIndex(syncPhase), "2x8_bulk_push_l1_to_join"));
++syncPhase;
uint32_t joinWorker = 0;
CHK_RET(LaunchTwoXEightBulkStage(resourceContext, memory, messageBytes, FabricLayer::L0,
    iteration, dataOffsetBytes, tileBytes, layout, TWO_X_EIGHT_BULK_FLAG_ONE_WAVE_JOIN,
    names.join, joinWorker));
CHK_PRT_RET(joinWorker != l0Worker,
    HCCL_ERROR("[ExecOp] 2x8 Bulk Join did not resolve to the L0 worker"), HCCL_E_INTERNAL);
CHK_RET(TwoXEightDirectedDependency(resourceContext, joinWorker, l1Worker,
    CellInternalNotifyIndex(syncPhase), "2x8_bulk_join_to_allgather"));
++syncPhase;
CHK_RET(LaunchTwoXEightBulkStage(resourceContext, memory, messageBytes, FabricLayer::L0,
    iteration, dataOffsetBytes, tileBytes, layout, TWO_X_EIGHT_BULK_FLAG_ONE_WAVE_ALL_GATHER,
    names.allGatherL0, l0Worker));
CHK_RET(LaunchTwoXEightBulkStage(resourceContext, memory, messageBytes, FabricLayer::L1,
    iteration, dataOffsetBytes, tileBytes, layout, TWO_X_EIGHT_BULK_FLAG_ONE_WAVE_ALL_GATHER,
    names.allGatherL1, l1Worker));
return HCCL_SUCCESS;
}
bool UseTwoXEightBulkOneWave(const InvocationMemory &memory, uint64_t messageBytes)
{
return EqualSizedRangesDisjoint(memory.input.address, memory.output.address, messageBytes);
}
HcclResult LaunchFullExchangeStage(const AlgResourceCtx &resourceContext, const InvocationMemory &memory,
DieEndpoint die, uint64_t elementCount, uint64_t flags, const char *stageName)
{
CcuKernelHandle handle = 0;
CHK_RET(FullExchangeStageHandle(resourceContext, die, handle));
TaskArgs taskArgs{};
taskArgs.inputAddr = memory.input.address;
taskArgs.inputToken = memory.input.token;
taskArgs.outputAddr = memory.output.address;
taskArgs.outputToken = memory.output.token;
taskArgs.scratchAddr = memory.scratch.address;
taskArgs.scratchToken = memory.scratch.token;
taskArgs.elementCount = elementCount;
taskArgs.tileElementCount = elementCount;
taskArgs.generation = 0;
taskArgs.flags = flags;
const CcuStageKernelId stage = die == DieEndpoint::DIE_0
                                   ? CcuStageKernelId::EIGHT_PLUS_FOUR_ALL_EXCHANGE_L0
                                   : CcuStageKernelId::EIGHT_PLUS_FOUR_ALL_EXCHANGE_L1;
const StageLaunchSpec launch{stage, die, 0, Direct12CreditSlot(0, 0), 0, flags};
return LaunchCellStage(
    resourceContext, handle, launch, taskArgs, "PACKAGE_A_8PLUS4_512KB_FULL_EXCHANGE", stageName);
}
HcclResult FourXOneRingResources(
const AlgResourceCtx &resourceContext, uint32_t worker, CcuKernelHandle &handle, DieEndpoint &die)
{
CHK_PRT_RET(worker >= ACTIVE_WORKERS,
    HCCL_ERROR("[ExecOp] 4x1 Ring worker is out of range"), HCCL_E_PARA);
const CcuStageKernelId stage = FOUR_X_ONE_RING_STAGE_ORDER[worker];
handle = 0;
die = DieEndpoint::NOT_APPLICABLE;
for (uint32_t index = 0; index < resourceContext.registeredStageCount; ++index) {
    if (resourceContext.registeredStages[index].stageId == stage) {
        handle = resourceContext.registeredStages[index].handle;
        break;
    }
}
for (uint32_t index = 0; index < resourceContext.peerChannelCount; ++index) {
    const PeerChannelResource &peer = resourceContext.peerChannels[index];
    CHK_PRT_RET(peer.layer != FabricLayer::L1
                    || (peer.localDie != DieEndpoint::DIE_0 && peer.localDie != DieEndpoint::DIE_1),
        HCCL_ERROR("[ExecOp] 4x1 Ring requires concrete L1 peer channels"), HCCL_E_INTERNAL);
    CHK_PRT_RET(die != DieEndpoint::NOT_APPLICABLE && die != peer.localDie,
        HCCL_ERROR("[ExecOp] 4x1 Ring L1 channels span both IO Dies"), HCCL_E_NOT_SUPPORT);
    die = peer.localDie;
}
CHK_PRT_RET(handle == 0 || die == DieEndpoint::NOT_APPLICABLE,
    HCCL_ERROR("[ExecOp] 4x1 Ring launch resources are incomplete"), HCCL_E_INTERNAL);
return HCCL_SUCCESS;
}
HcclResult FourXOnePullResources(const AlgResourceCtx &resourceContext,
std::array<CcuKernelHandle, ACTIVE_WORKERS> &handles, DieEndpoint &die)
{
CHK_PRT_RET(resourceContext.workerCount != 1 || resourceContext.registeredStageCount != 1
                || resourceContext.peerChannelCount != FOUR_X_ONE_PEER_COUNT,
    HCCL_ERROR("[ExecOp] 4x1 pull resource cardinality is invalid"), HCCL_E_INTERNAL);
handles.fill(0);
die = DieEndpoint::NOT_APPLICABLE;
for (uint32_t index = 0; index < resourceContext.registeredStageCount; ++index) {
    if (resourceContext.registeredStages[index].stageId
        == FOUR_X_ONE_PULL_FULL_EXCHANGE_STAGE_ORDER[0]) {
        handles[0] = resourceContext.registeredStages[index].handle;
    }
}
for (uint32_t index = 0; index < resourceContext.peerChannelCount; ++index) {
    const PeerChannelResource &peer = resourceContext.peerChannels[index];
    CHK_PRT_RET(peer.layer != FabricLayer::L1
                    || (peer.localDie != DieEndpoint::DIE_0 && peer.localDie != DieEndpoint::DIE_1),
        HCCL_ERROR("[ExecOp] 4x1 pull requires concrete L1 peer channels"), HCCL_E_INTERNAL);
    CHK_PRT_RET(die != DieEndpoint::NOT_APPLICABLE && die != peer.localDie,
        HCCL_ERROR("[ExecOp] 4x1 pull channels span both IO Dies"), HCCL_E_NOT_SUPPORT);
    die = peer.localDie;
}
CHK_PRT_RET(handles[0] == 0,
    HCCL_ERROR("[ExecOp] 4x1 pull worker handle is missing"), HCCL_E_INTERNAL);
CHK_PRT_RET(die == DieEndpoint::NOT_APPLICABLE,
    HCCL_ERROR("[ExecOp] 4x1 pull launch Die is missing"), HCCL_E_INTERNAL);
return HCCL_SUCCESS;
}
HcclResult AcquireFourXOneDirectMemory(
const OpParam &param, const AlgResourceCtx &resourceContext, InvocationMemory &memory)
{
const uint64_t dataBytes = param.count * sizeof(float);
memory = InvocationMemory{};
memory.input.address = reinterpret_cast<uint64_t>(param.inputPtr);
memory.output.address = reinterpret_cast<uint64_t>(param.outputPtr);
memory.scratch.address = reinterpret_cast<uint64_t>(resourceContext.hcclBuffer.addr);
CcuResult result = HcommCcuGetMemToken(memory.input.address, dataBytes, &memory.input.token);
CHK_PRT_RET(result != CCU_SUCCESS, HCCL_ERROR("[ExecOp] 4x1 pull input token failed"),
    ConvertCcuToHccl(result));
result = HcommCcuGetMemToken(memory.output.address, dataBytes, &memory.output.token);
CHK_PRT_RET(result != CCU_SUCCESS, HCCL_ERROR("[ExecOp] 4x1 pull output token failed"),
    ConvertCcuToHccl(result));
result = HcommCcuGetMemToken(
    memory.scratch.address, resourceContext.scratchBytes, &memory.scratch.token);
CHK_PRT_RET(result != CCU_SUCCESS, HCCL_ERROR("[ExecOp] 4x1 pull scratch token failed"),
    ConvertCcuToHccl(result));
return HCCL_SUCCESS;
}
bool FourXOneBuffersAreDisjoint(const InvocationMemory &memory, uint64_t bytes)
{
return EqualSizedRangesDisjoint(memory.input.address, memory.output.address, bytes);
}
HcclResult LaunchFourXOnePullFullExchange(const AlgResourceCtx &resourceContext,
const InvocationMemory &memory, CcuKernelHandle handle, DieEndpoint die, uint64_t elementCount)
{
TaskArgs taskArgs{};
taskArgs.inputAddr = memory.input.address;
taskArgs.inputToken = memory.input.token;
taskArgs.outputAddr = memory.output.address;
taskArgs.outputToken = memory.output.token;
taskArgs.scratchAddr = memory.scratch.address;
taskArgs.scratchToken = memory.scratch.token;
taskArgs.elementCount = elementCount;
taskArgs.tileElementCount = elementCount;
taskArgs.flags = FourXOneBuffersAreDisjoint(
                     memory, elementCount * sizeof(float))
                     ? FOUR_X_ONE_PULL_FLAG_DISJOINT
                     : FOUR_X_ONE_PULL_FLAG_CONSERVATIVE;
const StageLaunchSpec launch{FOUR_X_ONE_PULL_FULL_EXCHANGE_STAGE_ORDER[0], die, 0,
    Direct12CreditSlot(0, 0), 0, taskArgs.flags};
return LaunchCellStage(resourceContext, handle, launch, taskArgs,
    "PACKAGE_A_4X1_512KB_PULL_FULL_EXCHANGE", "PullFullExchange");
}
HcclResult LaunchFourXOneRingTile(const AlgResourceCtx &resourceContext, const InvocationMemory &memory,
uint64_t chunkElements, uint32_t tile, uint32_t worker, bool nativeUnevenTail)
{
const uint64_t tileElements
    = FourXOneRingTileElementCount(chunkElements, resourceContext.tileBytes, tile);
CHK_PRT_RET(tileElements == 0,
    HCCL_ERROR("[ExecOp] 4x1 Ring tile has zero elements"), HCCL_E_PARA);
CcuKernelHandle handle = 0;
DieEndpoint die = DieEndpoint::NOT_APPLICABLE;
CHK_RET(FourXOneRingResources(resourceContext, worker, handle, die));
const uint32_t creditSlot = Direct12CreditSlot(worker, 0);
TaskArgs taskArgs{};
taskArgs.inputAddr = memory.input.address;
taskArgs.inputToken = memory.input.token;
taskArgs.outputAddr = memory.output.address;
taskArgs.outputToken = memory.output.token;
taskArgs.scratchAddr = memory.output.address;
taskArgs.scratchToken = memory.output.token;
taskArgs.dataOffsetBytes = chunkElements * sizeof(float);
taskArgs.elementCount = chunkElements * FOUR_X_ONE_RANK_COUNT;
taskArgs.tileOffsetBytes = static_cast<uint64_t>(tile) * resourceContext.tileBytes;
taskArgs.tileElementCount = tileElements;
taskArgs.generation = creditSlot;
taskArgs.flags = nativeUnevenTail && worker == 1 && tile == 1
                     ? FOUR_X_ONE_RING_FLAG_EXECUTE_UNEVEN_TAIL
                     : FOUR_X_ONE_RING_FLAG_EXECUTE;
const StageLaunchSpec launch{
    FOUR_X_ONE_RING_STAGE_ORDER[worker], die, worker, creditSlot, tile, taskArgs.flags};
return LaunchCellStage(resourceContext, handle, launch, taskArgs,
    nativeUnevenTail ? "PACKAGE_A_4X1_400MB_PLUS_4B_RING_RSAG" : "PACKAGE_A_4X1_512MB_RING_RSAG",
    worker == 0 ? "RingWorker0" : "RingWorker1");
}
HcclResult AcquireFourXOneRingMemory(const OpParam &param, InvocationMemory &memory)
{
const uint64_t dataBytes = param.count * sizeof(float);
memory = InvocationMemory{};
memory.input.address = reinterpret_cast<uint64_t>(param.inputPtr);
memory.output.address = reinterpret_cast<uint64_t>(param.outputPtr);
CcuResult tokenResult = HcommCcuGetMemToken(memory.input.address, dataBytes, &memory.input.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS, HCCL_ERROR("[ExecOp] 4x1 Ring input token acquisition failed"),
    ConvertCcuToHccl(tokenResult));
tokenResult = HcommCcuGetMemToken(memory.output.address, dataBytes, &memory.output.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS, HCCL_ERROR("[ExecOp] 4x1 Ring output token acquisition failed"),
    ConvertCcuToHccl(tokenResult));
return HCCL_SUCCESS;
}
HcclResult ScheduleFourXOneRingBulk(const AlgResourceCtx &resourceContext,
const InvocationMemory &memory, uint64_t bulkElements, bool nativeUnevenTail)
{
CHK_PRT_RET(bulkElements == 0 || bulkElements % FOUR_X_ONE_RANK_COUNT != 0,
    HCCL_ERROR("[ExecOp] 4x1 Ring bulk must divide evenly across four ranks"), HCCL_E_PARA);
const uint64_t chunkElements = FourXOneRingChunkElements(bulkElements);
const uint32_t tileCount = FourXOneRingTileCount(chunkElements, resourceContext.tileBytes);
CHK_PRT_RET(tileCount == 0,
    HCCL_ERROR("[ExecOp] 4x1 Ring produced no tiles"), HCCL_E_INTERNAL);
for (uint32_t pair = 0; pair * ACTIVE_WORKERS < tileCount; ++pair) {
    const uint32_t worker0Tile = pair * ACTIVE_WORKERS;
    const uint32_t worker1Tile = worker0Tile + 1;
    CHK_RET(LaunchFourXOneRingTile(
        resourceContext, memory, chunkElements, worker0Tile, 0, nativeUnevenTail));
    const bool hasWorker1Tile = worker1Tile < tileCount;
    if (hasWorker1Tile) {
        CHK_RET(LaunchFourXOneRingTile(
            resourceContext, memory, chunkElements, worker1Tile, 1, nativeUnevenTail));
    }
}
return HCCL_SUCCESS;
}
} // namespace
HcclResult ScheduleDirect12(const OpParam &param, const AlgResourceCtx &resourceContext)
{
CHK_PRT_RET(param.inputPtr == nullptr || param.outputPtr == nullptr || param.count == 0
                || param.dataType != HCCL_DATA_TYPE_FP32 || param.rankSize != DIRECT12_CANONICAL_LEAF_COUNT
                || resourceContext.hcclBuffer.addr == nullptr
                || resourceContext.tileBytes == 0
                || resourceContext.tileBytes > PACKAGE_A_DIRECT12_MAX_TILE_BYTES
                || resourceContext.scratchBytes != Direct12ScratchBytes(resourceContext.tileBytes)
                || resourceContext.hcclBuffer.size < resourceContext.scratchBytes
                || resourceContext.localCanonicalRank >= DIRECT12_CANONICAL_LEAF_COUNT,
    HCCL_ERROR("[ExecOp] Invalid Direct12 invocation parameters"), HCCL_E_PARA);
const uint64_t dataBytes = param.count * sizeof(float);
InvocationMemory memory{};
memory.input.address = reinterpret_cast<uint64_t>(param.inputPtr);
memory.output.address = reinterpret_cast<uint64_t>(param.outputPtr);
memory.scratch.address = reinterpret_cast<uint64_t>(resourceContext.hcclBuffer.addr);
CcuResult tokenResult = HcommCcuGetMemToken(memory.input.address, dataBytes, &memory.input.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS, HCCL_ERROR("[ExecOp] Direct12 input token acquisition failed"),
    ConvertCcuToHccl(tokenResult));
tokenResult = HcommCcuGetMemToken(memory.output.address, dataBytes, &memory.output.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS, HCCL_ERROR("[ExecOp] Direct12 output token acquisition failed"),
    ConvertCcuToHccl(tokenResult));
tokenResult = HcommCcuGetMemToken(
    memory.scratch.address, resourceContext.scratchBytes, &memory.scratch.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS, HCCL_ERROR("[ExecOp] Direct12 scratch token acquisition failed"),
    ConvertCcuToHccl(tokenResult));
const uint64_t shardOffsetElements
    = Direct12ShardElementOffset(param.count, resourceContext.localCanonicalRank);
const uint64_t shardElements
    = Direct12ShardElementCount(param.count, resourceContext.localCanonicalRank);
const uint32_t tileCount = Direct12TileCount(shardElements, resourceContext.tileBytes);
CHK_PRT_RET(tileCount == 0 || tileCount > ACTIVE_WORKERS * DIRECT12_MAX_TILE_PAIRS,
    HCCL_ERROR("[ExecOp] Direct12 local shard requires unsupported tile count [%u]", tileCount), HCCL_E_PARA);
CHK_RET(CellThreadRecord(resourceContext.mainThread, resourceContext.slaveThreads[0],
    CELL_ENTRY_NOTIFY_INDEX, "entry"));
CHK_RET(CellThreadWait(resourceContext.slaveThreads[0], CELL_ENTRY_NOTIFY_INDEX, "entry"));
uint32_t syncPhase = 0;
const bool transitionPipeline = dataBytes == 512ULL * MEBIBYTE;
for (uint32_t pair = 0; pair < DIRECT12_MAX_TILE_PAIRS && pair * ACTIVE_WORKERS < tileCount; ++pair) {
    const uint32_t worker0Tile = pair * ACTIVE_WORKERS;
    const uint32_t worker1Tile = worker0Tile + 1;
    const bool worker1Active = worker1Tile < tileCount;
    constexpr uint32_t pairGeneration = 0;
    const uint32_t worker0Slot = Direct12CreditSlot(0, pairGeneration);
    const uint32_t worker1Slot = Direct12CreditSlot(1, pairGeneration);
    if (!transitionPipeline) {
        const uint64_t worker0GatherFlag
            = pair == 0 ? DIRECT12_FLAG_GATHER_ALL_PUBLISH : DIRECT12_FLAG_GATHER_ALL;
        CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, 0, worker0Slot, DieEndpoint::DIE_0, worker0GatherFlag, "GatherDie0"));
        CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, 0, worker0Slot, DieEndpoint::DIE_1, worker0GatherFlag, "GatherDie1"));
        if (worker1Active) {
            CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
                worker1Tile, 1, worker1Slot, DieEndpoint::DIE_0,
                DIRECT12_FLAG_GATHER_ALL, "GatherDie0"));
            CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
                worker1Tile, 1, worker1Slot, DieEndpoint::DIE_1,
                DIRECT12_FLAG_GATHER_ALL, "GatherDie1"));
            CHK_RET(CellCrossFence(resourceContext, CellInternalNotifyIndex(syncPhase)));
        } else {
            CHK_RET(CellStageOneWayHandoff(resourceContext, 1, 0,
                CellInternalNotifyIndex(syncPhase), "direct12_gather_to_join"));
        }
        ++syncPhase;
        CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, 0, worker0Slot, DieEndpoint::DIE_0, DIRECT12_FLAG_JOIN, "JoinDie0"));
        if (worker1Active) {
            CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
                worker1Tile, 1, worker1Slot, DieEndpoint::DIE_1, DIRECT12_FLAG_JOIN, "JoinDie1"));
            CHK_RET(CellCrossFence(resourceContext, CellInternalNotifyIndex(syncPhase)));
        } else {
            CHK_RET(CellStageOneWayHandoff(resourceContext, 0, 1,
                CellInternalNotifyIndex(syncPhase), "direct12_join_to_broadcast"));
        }
        ++syncPhase;
        CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, 0, worker0Slot, DieEndpoint::DIE_0,
            DIRECT12_FLAG_BROADCAST_ALL, "BroadcastDie0"));
        CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, 0, worker0Slot, DieEndpoint::DIE_1,
            DIRECT12_FLAG_BROADCAST_ALL, "BroadcastDie1"));
        if (worker1Active) {
            CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
                worker1Tile, 1, worker1Slot, DieEndpoint::DIE_0,
                DIRECT12_FLAG_BROADCAST_ALL, "BroadcastDie0"));
            CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
                worker1Tile, 1, worker1Slot, DieEndpoint::DIE_1,
                DIRECT12_FLAG_BROADCAST_ALL, "BroadcastDie1"));
        }
        CHK_RET(CellCrossFence(resourceContext, CellInternalNotifyIndex(syncPhase)));
        ++syncPhase;
        continue;
    }
    if (pair == 0) {
        CHK_PRT_RET(!worker1Active,
            HCCL_ERROR("[ExecOp] Direct12 transition pipeline requires an initial tile pair"), HCCL_E_PARA);
        CHK_RET(LaunchDirect12PairDieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, DieEndpoint::DIE_0, DIRECT12_FLAG_PAIR_GATHER_ALL_PUBLISH, "PairGatherDie0"));
        CHK_RET(LaunchDirect12PairDieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, DieEndpoint::DIE_1, DIRECT12_FLAG_PAIR_GATHER_ALL_PUBLISH, "PairGatherDie1"));
    }
    CHK_RET(CellCrossFence(resourceContext, CellInternalNotifyIndex(syncPhase)));
    ++syncPhase;
    CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
        worker0Tile, 0, worker0Slot, DieEndpoint::DIE_0, DIRECT12_FLAG_JOIN, "JoinDie0"));
    if (worker1Active) {
        CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker1Tile, 1, worker1Slot, DieEndpoint::DIE_1, DIRECT12_FLAG_JOIN, "JoinDie1"));
    }
    CHK_RET(CellCrossFence(resourceContext, CellInternalNotifyIndex(syncPhase)));
    ++syncPhase;
    const uint32_t nextFirstTile = worker0Tile + ACTIVE_WORKERS;
    if (nextFirstTile < tileCount) {
        CHK_PRT_RET(!worker1Active,
            HCCL_ERROR("[ExecOp] Direct12 transition cannot broadcast an incomplete current pair"),
            HCCL_E_PARA);
        CHK_RET(LaunchDirect12TransitionDieStage(resourceContext, memory, shardOffsetElements,
            shardElements, worker0Tile, DieEndpoint::DIE_0,
            DIRECT12_FLAG_PAIR_BROADCAST_GATHER_NEXT, "TransitionDie0"));
        CHK_RET(LaunchDirect12TransitionDieStage(resourceContext, memory, shardOffsetElements,
            shardElements, worker0Tile, DieEndpoint::DIE_1,
            DIRECT12_FLAG_PAIR_BROADCAST_GATHER_NEXT, "TransitionDie1"));
    } else if (worker1Active) {
        CHK_RET(LaunchDirect12PairDieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, DieEndpoint::DIE_0, DIRECT12_FLAG_PAIR_BROADCAST_ALL, "PairBroadcastDie0"));
        CHK_RET(LaunchDirect12PairDieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, DieEndpoint::DIE_1, DIRECT12_FLAG_PAIR_BROADCAST_ALL, "PairBroadcastDie1"));
    } else {
        CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, 0, worker0Slot, DieEndpoint::DIE_0,
            DIRECT12_FLAG_BROADCAST_ALL, "BroadcastDie0"));
        CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, 0, worker0Slot, DieEndpoint::DIE_1,
            DIRECT12_FLAG_BROADCAST_ALL, "BroadcastDie1"));
    }
}
CHK_RET(CellThreadRecord(resourceContext.slaveThreads[0], resourceContext.mainThread,
    CELL_EXIT_NOTIFY_INDEX, "exit"));
CHK_RET(CellThreadWait(resourceContext.mainThread, CELL_EXIT_NOTIFY_INDEX, "exit"));
return HCCL_SUCCESS;
}

HcclResult ScheduleDirect12PrefixScalar(const OpParam &param, const AlgResourceCtx &resourceContext)
{
CHK_PRT_RET(param.inputPtr == nullptr || param.outputPtr == nullptr
                || param.count != PACKAGE_A_DIRECT12_TAIL_MESSAGE_ELEMENTS
                || param.dataType != HCCL_DATA_TYPE_FP32 || param.rankSize != DIRECT12_CANONICAL_LEAF_COUNT
                || resourceContext.messageSize != MessageSizeKind::SIZE_400MB_PLUS_4B
                || resourceContext.hcclBuffer.addr == nullptr
                || resourceContext.tileBytes != PACKAGE_A_DIRECT12_TAIL_TILE_BYTES
                || resourceContext.scratchBytes != PACKAGE_A_DIRECT12_TAIL_SCRATCH_BYTES
                || resourceContext.hcclBuffer.size < resourceContext.scratchBytes
                || resourceContext.localCanonicalRank >= DIRECT12_CANONICAL_LEAF_COUNT,
    HCCL_ERROR("[ExecOp] Invalid Direct12 Prefix-Scalar invocation parameters"), HCCL_E_PARA);
constexpr uint64_t dataBytes = 400ULL * MEBIBYTE + sizeof(float);
InvocationMemory memory{};
memory.input.address = reinterpret_cast<uint64_t>(param.inputPtr);
memory.output.address = reinterpret_cast<uint64_t>(param.outputPtr);
memory.scratch.address = reinterpret_cast<uint64_t>(resourceContext.hcclBuffer.addr);
CcuResult tokenResult = HcommCcuGetMemToken(memory.input.address, dataBytes, &memory.input.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS,
    HCCL_ERROR("[ExecOp] Direct12 Prefix-Scalar input token acquisition failed"),
    ConvertCcuToHccl(tokenResult));
tokenResult = HcommCcuGetMemToken(memory.output.address, dataBytes, &memory.output.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS,
    HCCL_ERROR("[ExecOp] Direct12 Prefix-Scalar output token acquisition failed"),
    ConvertCcuToHccl(tokenResult));
tokenResult = HcommCcuGetMemToken(
    memory.scratch.address, resourceContext.scratchBytes, &memory.scratch.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS,
    HCCL_ERROR("[ExecOp] Direct12 Prefix-Scalar scratch token acquisition failed"),
    ConvertCcuToHccl(tokenResult));
const uint64_t shardOffsetElements = Direct12ShardElementOffset(
    PACKAGE_A_DIRECT12_TAIL_PREFIX_ELEMENTS, resourceContext.localCanonicalRank);
const uint64_t shardElements = Direct12ShardElementCount(
    PACKAGE_A_DIRECT12_TAIL_PREFIX_ELEMENTS, resourceContext.localCanonicalRank);
const uint32_t tileCount = Direct12TileCount(shardElements, resourceContext.tileBytes);
CHK_PRT_RET(tileCount != 3,
    HCCL_ERROR("[ExecOp] Direct12 Prefix-Scalar requires exactly three prefix tiles, got [%u]", tileCount),
    HCCL_E_PARA);
CHK_RET(CellThreadRecord(resourceContext.mainThread, resourceContext.slaveThreads[0],
    CELL_ENTRY_NOTIFY_INDEX, "entry"));
CHK_RET(CellThreadWait(resourceContext.slaveThreads[0], CELL_ENTRY_NOTIFY_INDEX, "entry"));
uint32_t syncPhase = 0;
for (uint32_t pair = 0; pair * ACTIVE_WORKERS < tileCount; ++pair) {
    const uint32_t worker0Tile = pair * ACTIVE_WORKERS;
    const uint32_t worker1Tile = worker0Tile + 1;
    const bool worker1Active = worker1Tile < tileCount;
    const uint32_t worker0Slot = Direct12CreditSlot(0, 0);
    const uint32_t worker1Slot = Direct12CreditSlot(1, 0);
    if (pair == 0) {
        CHK_RET(LaunchDirect12PairDieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, DieEndpoint::DIE_0, DIRECT12_FLAG_PAIR_GATHER_ALL_PUBLISH, "PairGatherDie0"));
        CHK_RET(LaunchDirect12PairDieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, DieEndpoint::DIE_1, DIRECT12_FLAG_PAIR_GATHER_ALL_PUBLISH, "PairGatherDie1"));
    }
    CHK_RET(CellCrossFence(resourceContext, CellInternalNotifyIndex(syncPhase)));
    ++syncPhase;
    CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
        worker0Tile, 0, worker0Slot, DieEndpoint::DIE_0, DIRECT12_FLAG_JOIN, "JoinDie0"));
    if (worker1Active) {
        CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker1Tile, 1, worker1Slot, DieEndpoint::DIE_1, DIRECT12_FLAG_JOIN, "JoinDie1"));
    }
    CHK_RET(CellCrossFence(resourceContext, CellInternalNotifyIndex(syncPhase)));
    ++syncPhase;
    const uint32_t nextFirstTile = worker0Tile + ACTIVE_WORKERS;
    if (nextFirstTile < tileCount) {
        CHK_RET(LaunchDirect12TransitionDieStage(resourceContext, memory, shardOffsetElements,
            shardElements, worker0Tile, DieEndpoint::DIE_0,
            DIRECT12_FLAG_PAIR_BROADCAST_GATHER_NEXT, "TransitionDie0"));
        CHK_RET(LaunchDirect12TransitionDieStage(resourceContext, memory, shardOffsetElements,
            shardElements, worker0Tile, DieEndpoint::DIE_1,
            DIRECT12_FLAG_PAIR_BROADCAST_GATHER_NEXT, "TransitionDie1"));
    } else {
        CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, 0, worker0Slot, DieEndpoint::DIE_0,
            DIRECT12_FLAG_BROADCAST_GATHER_SCALAR, "BroadcastGatherScalarDie0"));
        CHK_RET(LaunchDirect12DieStage(resourceContext, memory, shardOffsetElements, shardElements,
            worker0Tile, 0, worker0Slot, DieEndpoint::DIE_1,
            DIRECT12_FLAG_BROADCAST_GATHER_SCALAR, "BroadcastGatherScalarDie1"));
    }
}
CHK_RET(CellStageOneWayHandoff(resourceContext, 1, 0,
    CellInternalNotifyIndex(syncPhase), "direct12_scalar_gather_to_join"));
CHK_RET(LaunchDirect12DieStage(resourceContext, memory,
    PACKAGE_A_DIRECT12_TAIL_PREFIX_ELEMENTS, PACKAGE_A_DIRECT12_TAIL_SUFFIX_ELEMENTS,
    0, 0, Direct12CreditSlot(0, 0), DieEndpoint::DIE_0,
    DIRECT12_FLAG_SCALAR_JOIN, "ScalarJoinDie0"));
CHK_RET(CellThreadRecord(resourceContext.slaveThreads[0], resourceContext.mainThread,
    CELL_EXIT_NOTIFY_INDEX, "exit"));
CHK_RET(CellThreadWait(resourceContext.mainThread, CELL_EXIT_NOTIFY_INDEX, "exit"));
return HCCL_SUCCESS;
}

HcclResult ScheduleFullExchange(const OpParam &param, const AlgResourceCtx &resourceContext)
{
constexpr uint64_t FULL_EXCHANGE_ELEMENT_COUNT = FULL_EXCHANGE_TILE_BYTES / sizeof(float);
CHK_PRT_RET(param.inputPtr == nullptr || param.outputPtr == nullptr
                || param.count != FULL_EXCHANGE_ELEMENT_COUNT || param.dataType != HCCL_DATA_TYPE_FP32
                || param.rankSize != DIRECT12_CANONICAL_LEAF_COUNT
                || resourceContext.hcclBuffer.addr == nullptr
                || resourceContext.hcclBuffer.size < FULL_EXCHANGE_SCRATCH_BYTES
                || resourceContext.localCanonicalRank >= DIRECT12_CANONICAL_LEAF_COUNT,
    HCCL_ERROR("[ExecOp] Invalid FullExchange invocation parameters"), HCCL_E_PARA);
InvocationMemory memory{};
memory.input.address = reinterpret_cast<uint64_t>(param.inputPtr);
memory.output.address = reinterpret_cast<uint64_t>(param.outputPtr);
memory.scratch.address = reinterpret_cast<uint64_t>(resourceContext.hcclBuffer.addr);
CcuResult tokenResult
    = HcommCcuGetMemToken(memory.input.address, FULL_EXCHANGE_TILE_BYTES, &memory.input.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS, HCCL_ERROR("[ExecOp] FullExchange input token acquisition failed"),
    ConvertCcuToHccl(tokenResult));
tokenResult = HcommCcuGetMemToken(memory.output.address, FULL_EXCHANGE_TILE_BYTES, &memory.output.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS, HCCL_ERROR("[ExecOp] FullExchange output token acquisition failed"),
    ConvertCcuToHccl(tokenResult));
tokenResult = HcommCcuGetMemToken(
    memory.scratch.address, FULL_EXCHANGE_SCRATCH_BYTES, &memory.scratch.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS, HCCL_ERROR("[ExecOp] FullExchange scratch token acquisition failed"),
    ConvertCcuToHccl(tokenResult));
CHK_RET(CellThreadRecord(resourceContext.mainThread, resourceContext.slaveThreads[0],
    CELL_ENTRY_NOTIFY_INDEX, "entry"));
CHK_RET(CellThreadWait(resourceContext.slaveThreads[0], CELL_ENTRY_NOTIFY_INDEX, "entry"));
uint32_t syncPhase = 0;
CHK_RET(LaunchFullExchangeStage(resourceContext, memory, DieEndpoint::DIE_0, param.count,
    FULL_EXCHANGE_FLAG_GATHER, "GatherDie0"));
CHK_RET(LaunchFullExchangeStage(resourceContext, memory, DieEndpoint::DIE_1, param.count,
    FULL_EXCHANGE_FLAG_GATHER, "GatherDie1"));
DieEndpoint joinDie = DieEndpoint::NOT_APPLICABLE;
CHK_RET(FullExchangeJoinDie(resourceContext, joinDie));
const uint32_t joinWorker = joinDie == DieEndpoint::DIE_0 ? 0U : 1U;
// The join owner's Gather is already ordered on its worker. Close the other worker's Gather before Join can
// overwrite output in an in-place invocation.
CHK_RET(CellStageOneWayHandoff(resourceContext, joinWorker ^ 1U, joinWorker,
    CellInternalNotifyIndex(syncPhase), "full_exchange_gather_to_join"));
++syncPhase;
CHK_RET(LaunchFullExchangeStage(
    resourceContext, memory, joinDie, param.count, FULL_EXCHANGE_FLAG_JOIN, "FixedTreeJoin"));
if (joinWorker == 1U) {
    CHK_RET(CellThreadRecord(resourceContext.slaveThreads[0], resourceContext.mainThread,
        CELL_EXIT_NOTIFY_INDEX, "exit"));
    CHK_RET(CellThreadWait(resourceContext.mainThread, CELL_EXIT_NOTIFY_INDEX, "exit"));
}
return HCCL_SUCCESS;
}
HcclResult ScheduleTwoXEightMultiRoot(const OpParam &param, const AlgResourceCtx &resourceContext)
{
constexpr uint64_t ELEMENT_COUNT = TWO_X_EIGHT_MULTI_ROOT_MESSAGE_BYTES / sizeof(float);
CHK_PRT_RET(param.inputPtr == nullptr || param.outputPtr == nullptr || param.count != ELEMENT_COUNT
                || param.dataType != HCCL_DATA_TYPE_FP32 || param.rankSize != TWO_X_EIGHT_RANK_COUNT
                || resourceContext.workerCount != ACTIVE_WORKERS || resourceContext.slaveThreadCount != 1
                || resourceContext.hcclBuffer.addr == nullptr
                || resourceContext.hcclBuffer.size < TWO_X_EIGHT_MULTI_ROOT_SCRATCH_BYTES
                || resourceContext.localCanonicalRank >= TWO_X_EIGHT_RANK_COUNT,
    HCCL_ERROR("[ExecOp] Invalid 2x8 MultiRoot invocation parameters"), HCCL_E_PARA);
InvocationMemory memory{};
memory.input.address = reinterpret_cast<uint64_t>(param.inputPtr);
memory.output.address = reinterpret_cast<uint64_t>(param.outputPtr);
memory.scratch.address = reinterpret_cast<uint64_t>(resourceContext.hcclBuffer.addr);
CcuResult tokenResult = HcommCcuGetMemToken(
    memory.input.address, TWO_X_EIGHT_MULTI_ROOT_MESSAGE_BYTES, &memory.input.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS,
    HCCL_ERROR("[ExecOp] 2x8 MultiRoot input token acquisition failed"), ConvertCcuToHccl(tokenResult));
tokenResult = HcommCcuGetMemToken(
    memory.output.address, TWO_X_EIGHT_MULTI_ROOT_MESSAGE_BYTES, &memory.output.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS,
    HCCL_ERROR("[ExecOp] 2x8 MultiRoot output token acquisition failed"), ConvertCcuToHccl(tokenResult));
tokenResult = HcommCcuGetMemToken(memory.scratch.address,
    TWO_X_EIGHT_MULTI_ROOT_SCRATCH_BYTES, &memory.scratch.token);
CHK_PRT_RET(tokenResult != CCU_SUCCESS,
    HCCL_ERROR("[ExecOp] 2x8 MultiRoot scratch token acquisition failed"), ConvertCcuToHccl(tokenResult));
CHK_RET(CellThreadRecord(resourceContext.mainThread, resourceContext.slaveThreads[0],
    CELL_ENTRY_NOTIFY_INDEX, "2x8_entry"));
CHK_RET(CellThreadWait(resourceContext.slaveThreads[0], CELL_ENTRY_NOTIFY_INDEX, "2x8_entry"));
uint32_t syncPhase = 0;
uint32_t l0Worker = 0;
uint32_t l1Worker = 0;
CHK_RET(LaunchTwoXEightMultiRootStage(resourceContext, memory, FabricLayer::L0,
    TWO_X_EIGHT_MULTI_ROOT_FLAG_DUAL_REPLICA_GATHER, "DualReplicaGatherL0", l0Worker));
CHK_RET(LaunchTwoXEightMultiRootStage(resourceContext, memory, FabricLayer::L1,
    TWO_X_EIGHT_MULTI_ROOT_FLAG_DUAL_REPLICA_GATHER, "DualReplicaGatherL1", l1Worker));
CHK_PRT_RET(l0Worker == l1Worker,
    HCCL_ERROR("[ExecOp] 2x8 MultiRoot L0 and L1 kernels resolved to the same stream"), HCCL_E_NOT_SUPPORT);
CHK_RET(CellCrossFence(resourceContext, CellInternalNotifyIndex(syncPhase)));
++syncPhase;

uint32_t l0JoinWorker = 0;
uint32_t l1JoinWorker = 0;
CHK_RET(LaunchTwoXEightMultiRootStage(resourceContext, memory, FabricLayer::L0,
    TWO_X_EIGHT_MULTI_ROOT_FLAG_DUAL_REPLICA_JOIN_BROADCAST,
    "DualReplicaJoinBroadcastL0", l0JoinWorker));
CHK_RET(LaunchTwoXEightMultiRootStage(resourceContext, memory, FabricLayer::L1,
    TWO_X_EIGHT_MULTI_ROOT_FLAG_DUAL_REPLICA_JOIN_BROADCAST,
    "DualReplicaJoinBroadcastL1", l1JoinWorker));
CHK_PRT_RET(l0JoinWorker != l0Worker || l1JoinWorker != l1Worker,
    HCCL_ERROR("[ExecOp] 2x8 dual-replica JoinBroadcast resolved to the wrong stream"),
    HCCL_E_INTERNAL);
return CellOneWayDependency(resourceContext, CELL_EXIT_NOTIFY_INDEX, "2x8_exit");
}
HcclResult ScheduleTwoXEightBulk(const OpParam &param, const AlgResourceCtx &resourceContext)
{
InvocationMemory memory{};
CHK_RET(PrepareTwoXEightBulkMemory(param, resourceContext, TWO_X_EIGHT_BULK_MESSAGE_BYTES,
    "PACKAGE_A_2X8_512MB_BULK_PUSH_STAGED_REDUCE", memory));
CHK_RET(CellThreadRecord(resourceContext.mainThread, resourceContext.slaveThreads[0],
    CELL_ENTRY_NOTIFY_INDEX, "2x8_bulk_entry"));
CHK_RET(CellThreadWait(resourceContext.slaveThreads[0], CELL_ENTRY_NOTIFY_INDEX, "2x8_bulk_entry"));
uint32_t syncPhase = 0;
if (UseTwoXEightBulkOneWave(memory, TWO_X_EIGHT_BULK_MESSAGE_BYTES)) {
    const TwoXEightBulkStageNames oneWaveNames{"BulkOneWavePushL0", "BulkOneWavePushL1",
        "BulkOneWaveFixedTreeJoin", "BulkOneWaveAllGatherL0", "BulkOneWaveAllGatherL1"};
    CHK_RET(RunTwoXEightBulkOneWave(resourceContext, memory, TWO_X_EIGHT_BULK_MESSAGE_BYTES,
        TwoXEightOneWaveOwnerBytes(resourceContext.localCanonicalRank, false),
        TWO_X_EIGHT_BULK_LAYOUT_ONE_WAVE_UNIFORM, oneWaveNames, syncPhase));
    return CellOneWayDependency(resourceContext, CELL_EXIT_NOTIFY_INDEX, "2x8_bulk_exit");
}
const TwoXEightBulkStageNames names{
    "BulkPushL0", "BulkPushL1", "BulkFixedTreeJoin", "BulkAllGatherL0", "BulkAllGatherL1"};
// Worker FIFOs and the directed chain complete each record/wait before notify indices 2/3 are reused.
for (uint32_t tile = 0; tile < PACKAGE_A_TWO_X_EIGHT_BULK_TILES_PER_SHARD; ++tile) {
    const uint64_t dataOffsetBytes = tile == 0 ? 0 : PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES;
    CHK_RET(RunTwoXEightBulkTile(resourceContext, memory, TWO_X_EIGHT_BULK_MESSAGE_BYTES,
        tile, dataOffsetBytes, TwoXEightBulkTileBytes(tile),
        TWO_X_EIGHT_BULK_LAYOUT_UNIFORM, names, syncPhase));
}
return CellOneWayDependency(resourceContext, CELL_EXIT_NOTIFY_INDEX, "2x8_bulk_exit");
}
HcclResult ScheduleTwoXEightBulkTail(const OpParam &param, const AlgResourceCtx &resourceContext)
{
InvocationMemory memory{};
CHK_RET(PrepareTwoXEightBulkMemory(param, resourceContext, TWO_X_EIGHT_BULK_TAIL_MESSAGE_BYTES,
    "PACKAGE_A_2X8_400MB_PLUS_4B_BULK_PUSH_STAGED_REDUCE", memory));
CHK_RET(CellThreadRecord(resourceContext.mainThread, resourceContext.slaveThreads[0],
    CELL_ENTRY_NOTIFY_INDEX, "2x8_bulk_tail_entry"));
CHK_RET(CellThreadWait(resourceContext.slaveThreads[0], CELL_ENTRY_NOTIFY_INDEX, "2x8_bulk_tail_entry"));
uint32_t syncPhase = 0;
if (UseTwoXEightBulkOneWave(memory, TWO_X_EIGHT_BULK_TAIL_MESSAGE_BYTES)) {
    const TwoXEightBulkStageNames oneWaveNames{"BulkTailOneWavePushL0", "BulkTailOneWavePushL1",
        "BulkTailOneWaveFixedTreeJoin", "BulkTailOneWaveAllGatherL0", "BulkTailOneWaveAllGatherL1"};
    CHK_RET(RunTwoXEightBulkOneWave(resourceContext, memory, TWO_X_EIGHT_BULK_TAIL_MESSAGE_BYTES,
        TwoXEightOneWaveOwnerBytes(resourceContext.localCanonicalRank, true),
        TWO_X_EIGHT_BULK_LAYOUT_ONE_WAVE_TAIL, oneWaveNames, syncPhase));
    return CellOneWayDependency(resourceContext, CELL_EXIT_NOTIFY_INDEX, "2x8_bulk_tail_exit");
}
const TwoXEightBulkStageNames names{"BulkTailPushL0", "BulkTailPushL1",
    "BulkTailFixedTreeJoin", "BulkTailAllGatherL0", "BulkTailAllGatherL1"};
for (uint32_t tile = 0; tile < PACKAGE_A_TWO_X_EIGHT_BULK_TILES_PER_SHARD; ++tile) {
    const uint64_t dataOffsetBytes
        = tile == 0 ? 0 : PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_TILE_BYTES;
    const uint64_t layout = tile == 0
                                ? TWO_X_EIGHT_BULK_LAYOUT_UNEVEN
                                : TWO_X_EIGHT_BULK_LAYOUT_UNEVEN_REMAINDER;
    CHK_RET(RunTwoXEightBulkTile(resourceContext, memory,
        TWO_X_EIGHT_BULK_TAIL_MESSAGE_BYTES, tile, dataOffsetBytes,
        TwoXEightBulkTailTileBytes(resourceContext.localCanonicalRank, tile),
        layout, names, syncPhase));
}
return CellOneWayDependency(resourceContext, CELL_EXIT_NOTIFY_INDEX, "2x8_bulk_tail_exit");
}
HcclResult ScheduleFourXOnePullFullExchange(
const OpParam &param, const AlgResourceCtx &resourceContext)
{
constexpr uint64_t ELEMENT_COUNT = FOUR_X_ONE_SMALL_TILE_BYTES / sizeof(float);
CHK_PRT_RET(param.inputPtr == nullptr || param.outputPtr == nullptr || param.count != ELEMENT_COUNT
                || param.dataType != HCCL_DATA_TYPE_FP32 || param.rankSize != FOUR_X_ONE_RANK_COUNT
                || resourceContext.workerCount != 1 || resourceContext.slaveThreadCount != 0
                || resourceContext.tileBytes != FOUR_X_ONE_SMALL_TILE_BYTES
                || resourceContext.scratchBytes != FOUR_X_ONE_SMALL_SCRATCH_BYTES
                || resourceContext.hcclBuffer.addr == nullptr
                || resourceContext.hcclBuffer.size < FOUR_X_ONE_SMALL_SCRATCH_BYTES
                || resourceContext.localCanonicalRank >= FOUR_X_ONE_RANK_COUNT,
    HCCL_ERROR("[ExecOp] Invalid 4x1 pull Full Exchange invocation parameters"), HCCL_E_PARA);
InvocationMemory memory{};
CHK_RET(AcquireFourXOneDirectMemory(param, resourceContext, memory));
std::array<CcuKernelHandle, ACTIVE_WORKERS> handles{};
DieEndpoint die = DieEndpoint::NOT_APPLICABLE;
CHK_RET(FourXOnePullResources(resourceContext, handles, die));
return LaunchFourXOnePullFullExchange(resourceContext, memory, handles[0], die, ELEMENT_COUNT);
}
HcclResult ScheduleFourXOneRing(const OpParam &param, const AlgResourceCtx &resourceContext)
{
constexpr uint64_t FOUR_X_ONE_512MB_ELEMENTS = 512ULL * MEGABYTE / sizeof(float);
CHK_PRT_RET(param.inputPtr == nullptr || param.outputPtr == nullptr
                || param.count != FOUR_X_ONE_512MB_ELEMENTS || param.dataType != HCCL_DATA_TYPE_FP32
                || param.rankSize != FOUR_X_ONE_RANK_COUNT || resourceContext.workerCount != ACTIVE_WORKERS
                || resourceContext.slaveThreadCount != 1
                || resourceContext.tileBytes != PACKAGE_A_FOUR_X_ONE_RING_512_TILE_BYTES,
    HCCL_ERROR("[ExecOp] Invalid 4x1 Ring invocation parameters"), HCCL_E_PARA);
InvocationMemory memory{};
CHK_RET(AcquireFourXOneRingMemory(param, memory));
CHK_RET(CellThreadRecord(resourceContext.mainThread, resourceContext.slaveThreads[0],
    CELL_ENTRY_NOTIFY_INDEX, "entry"));
CHK_RET(CellThreadWait(resourceContext.slaveThreads[0], CELL_ENTRY_NOTIFY_INDEX, "entry"));
CHK_RET(ScheduleFourXOneRingBulk(resourceContext, memory, param.count, false));
CHK_RET(CellThreadRecord(resourceContext.slaveThreads[0], resourceContext.mainThread,
    CELL_EXIT_NOTIFY_INDEX, "exit"));
CHK_RET(CellThreadWait(resourceContext.mainThread, CELL_EXIT_NOTIFY_INDEX, "exit"));
return HCCL_SUCCESS;
}
HcclResult ScheduleFourXOneRingTail(const OpParam &param, const AlgResourceCtx &resourceContext)
{
constexpr uint64_t FOUR_X_ONE_400MB_PLUS_4B_ELEMENTS
    = SIZE_400MB_PLUS_4B_BYTES / sizeof(float);
CHK_PRT_RET(param.inputPtr == nullptr || param.outputPtr == nullptr
                || param.count != FOUR_X_ONE_400MB_PLUS_4B_ELEMENTS
                || param.dataType != HCCL_DATA_TYPE_FP32 || param.rankSize != FOUR_X_ONE_RANK_COUNT
                || resourceContext.workerCount != ACTIVE_WORKERS || resourceContext.slaveThreadCount != 1
                || resourceContext.tileBytes != PACKAGE_A_FOUR_X_ONE_RING_TAIL_TILE_BYTES
                || FourXOneTailElementCount(param.count) != 1,
    HCCL_ERROR("[ExecOp] Invalid 4x1 Ring split-tail invocation parameters"), HCCL_E_PARA);
InvocationMemory memory{};
CHK_RET(AcquireFourXOneRingMemory(param, memory));
const uint64_t bulkElements = FourXOneTailBulkElements(param.count);
CHK_RET(CellThreadRecord(resourceContext.mainThread, resourceContext.slaveThreads[0],
    CELL_ENTRY_NOTIFY_INDEX, "entry"));
CHK_RET(CellThreadWait(resourceContext.slaveThreads[0], CELL_ENTRY_NOTIFY_INDEX, "entry"));
CHK_RET(ScheduleFourXOneRingBulk(resourceContext, memory, bulkElements, true));
CHK_RET(CellThreadRecord(resourceContext.slaveThreads[0], resourceContext.mainThread,
    CELL_EXIT_NOTIFY_INDEX, "exit"));
CHK_RET(CellThreadWait(resourceContext.mainThread, CELL_EXIT_NOTIFY_INDEX, "exit"));
return HCCL_SUCCESS;
}
HcclResult ExecOp(const OpParam &param)
{
CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0
                || param.ctxSize > MAX_SERIALIZED_RESOURCE_CONTEXT_BYTES,
    HCCL_ERROR("[ExecOp] Invalid serialized Direct12 context"), HCCL_E_PARA);
// 反序列化
char *ctx = static_cast<char *>(param.resCtx);
std::vector<char> seq(ctx, ctx + param.ctxSize);
AlgResourceCtx resCtx;
CHK_RET(resCtx.DeSerialize(seq));
auto sizeIter = SIZE_TABLE.find(param.dataType);
CHK_PRT_RET(sizeIter == SIZE_TABLE.end(),
    HCCL_ERROR("[ExecOp] Unsupported data type [%d]", static_cast<int32_t>(param.dataType)), HCCL_E_NOT_SUPPORT);
CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeIter->second,
    HCCL_ERROR("[ExecOp] Data size overflow, count [%llu]", static_cast<unsigned long long>(param.count)),
    HCCL_E_PARA);
uint64_t dataBytes = param.count * sizeIter->second;
AlgorithmPlan plan;
CHK_RET(SelectAlgorithmPlan(param.rankSize, dataBytes, plan));
CHK_PRT_RET(plan.determinism != DeterminismPolicy::FIXED_RANK_AND_SLICE_ORDER,
    HCCL_ERROR("[ExecOp] Plan [%s] does not satisfy deterministic execution", GetPlanName(plan.id)),
    HCCL_E_NOT_SUPPORT);
CHK_PRT_RET(resCtx.package != plan.package || resCtx.topology != plan.topology
                || resCtx.messageSize != plan.messageSize || resCtx.family != plan.family
                || resCtx.tailPolicy != plan.tailPolicy || resCtx.resourceProfile != plan.resourceProfile
                || resCtx.workerCount != plan.streamCount
                || resCtx.kernelHandleCount != plan.kernelHandleCount || resCtx.tileBytes != plan.tileBytes
                || resCtx.scratchBytes != plan.scratchBytes,
    HCCL_ERROR("[ExecOp] Cached resource profile does not match plan [%s]", GetPlanName(plan.id)), HCCL_E_INTERNAL);
CHK_PRT_RET(!plan.implemented || resCtx.package != AlgorithmPackage::PACKAGE_A,
    HCCL_ERROR("[ExecOp] Plan [%s] is explicitly unsupported", GetPlanName(plan.id)), HCCL_E_NOT_SUPPORT);
if (plan.family == AlgorithmFamily::DIRECT12_RSAG
    && plan.topology == TopologyKind::TOPOLOGY_8_PLUS_4
    && plan.messageSize == MessageSizeKind::SIZE_512MB) {
    return ScheduleDirect12(param, resCtx);
}
if (plan.family == AlgorithmFamily::DIRECT12_RSAG
    && plan.topology == TopologyKind::TOPOLOGY_8_PLUS_4
    && plan.messageSize == MessageSizeKind::SIZE_400MB_PLUS_4B) {
    return ScheduleDirect12PrefixScalar(param, resCtx);
}
if (plan.family == AlgorithmFamily::FULL_EXCHANGE_FIXED_TREE
    && plan.topology == TopologyKind::TOPOLOGY_8_PLUS_4
    && plan.messageSize == MessageSizeKind::SIZE_512KB) {
    return ScheduleFullExchange(param, resCtx);
}
if (plan.family == AlgorithmFamily::FULL_EXCHANGE_FIXED_TREE
    && plan.topology == TopologyKind::TOPOLOGY_4X1
    && plan.messageSize == MessageSizeKind::SIZE_512KB) {
    return ScheduleFourXOnePullFullExchange(param, resCtx);
}
if (plan.family == AlgorithmFamily::RING_RSAG && plan.topology == TopologyKind::TOPOLOGY_4X1
    && plan.messageSize == MessageSizeKind::SIZE_512MB) {
    return ScheduleFourXOneRing(param, resCtx);
}
if (plan.family == AlgorithmFamily::RING_RSAG && plan.topology == TopologyKind::TOPOLOGY_4X1
    && plan.messageSize == MessageSizeKind::SIZE_400MB_PLUS_4B) {
    return ScheduleFourXOneRingTail(param, resCtx);
}
if (plan.family == AlgorithmFamily::TOPOLOGY_BALANCED_DIRECT_RSAG
    && plan.topology == TopologyKind::TOPOLOGY_2X8
    && plan.messageSize == MessageSizeKind::SIZE_512KB) {
    return ScheduleTwoXEightMultiRoot(param, resCtx);
}
if (plan.family == AlgorithmFamily::BULK_PUSH_STAGED_REDUCE
    && plan.topology == TopologyKind::TOPOLOGY_2X8
    && plan.messageSize == MessageSizeKind::SIZE_512MB) {
    return ScheduleTwoXEightBulk(param, resCtx);
}
if (plan.family == AlgorithmFamily::BULK_PUSH_STAGED_REDUCE
    && plan.topology == TopologyKind::TOPOLOGY_2X8
    && plan.messageSize == MessageSizeKind::SIZE_400MB_PLUS_4B) {
    return ScheduleTwoXEightBulkTail(param, resCtx);
}
HCCL_ERROR("[ExecOp] Plan [%s] has no implemented family scheduler", GetPlanName(plan.id));
return HCCL_E_NOT_SUPPORT;
}
} // namespace ops_hccl
