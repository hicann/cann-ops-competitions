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
#include <array>
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

constexpr uint64_t DIRECT_WINDOW_SIZE = 2ULL * MAX_DATA_SIZE;
constexpr uint64_t SMALL_DIRECT_THRESHOLD = 1ULL * 1024ULL * 1024ULL;
constexpr uint64_t FP32_ALIGNMENT = sizeof(float);

constexpr uint32_t START_NOTIFY_INDEX = 0;
constexpr uint32_t READY0_NOTIFY_INDEX = 0;
constexpr uint32_t READY1_NOTIFY_INDEX = 1;
constexpr uint32_t READY2_NOTIFY_INDEX = 2;
constexpr uint32_t READY3_NOTIFY_INDEX = 3;
constexpr uint32_t L1_DONE_NOTIFY_INDEX = 4;
constexpr uint32_t A84_PREFIX_READY_NOTIFY_INDEX = 0;
constexpr uint32_t A84_L1_DONE_NOTIFY_INDEX = 1;

constexpr uint32_t DIRECT_TASK_ARG_COUNT = 11;
constexpr uint32_t SMALL_PULL_TASK_ARG_COUNT = 6;
constexpr uint32_t SMALL_PULL_WARM_TASK_ARG_COUNT = 3;
constexpr uint32_t STAGE_TASK_ARG_COUNT = 12;
constexpr uint32_t A84_TASK_ARG_COUNT = 12;

using DirectTaskArgs = std::array<uint64_t, DIRECT_TASK_ARG_COUNT>;
using SmallPullTaskArgs =
    std::array<uint64_t, SMALL_PULL_TASK_ARG_COUNT>;
using SmallPullWarmTaskArgs =
    std::array<uint64_t, SMALL_PULL_WARM_TASK_ARG_COUNT>;
using StageTaskArgs = std::array<uint64_t, STAGE_TASK_ARG_COUNT>;
using A84TaskArgs = std::array<uint64_t, A84_TASK_ARG_COUNT>;

struct PhysicalSlice {
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct MixedChunk {
    uint64_t offset = 0;
    uint64_t size = 0;
    uint64_t relaySize = 0;
    uint64_t directOffset = 0;
    uint64_t directSize = 0;
};

DirectTaskArgs MakeDirectTaskArgs(uint64_t inputAddress, uint64_t outputAddress,
    uint64_t inputToken, uint64_t outputToken, uint64_t rankOffset,
    uint64_t windowOffset, uint64_t windowSize)
{
    const uint64_t sliceSize0 = std::min<uint64_t>(MAX_DATA_SIZE, windowSize);
    const uint64_t sliceSize1 = windowSize - sliceSize0;
    return {
        inputAddress,
        outputAddress,
        inputToken,
        outputToken,
        rankOffset,
        rankOffset,
        windowOffset,
        sliceSize0,
        windowOffset + sliceSize0,
        sliceSize1,
        static_cast<uint64_t>(AllGatherKernelMode::DIRECT),
    };
}

SmallPullTaskArgs MakeSmallPullTaskArgs(
    uint64_t inputAddress, uint64_t outputAddress,
    uint64_t inputToken, uint64_t outputToken,
    uint64_t rankOffset, uint64_t dataSize)
{
    return {
        inputAddress,
        outputAddress,
        inputToken,
        outputToken,
        rankOffset,
        dataSize,
    };
}

SmallPullWarmTaskArgs MakeSmallPullWarmTaskArgs(
    uint64_t outputAddress, uint64_t outputToken, uint64_t dataSize)
{
    return {
        outputAddress,
        outputToken,
        dataSize,
    };
}

DirectTaskArgs MakeRd4TaskArgs(uint64_t inputAddress, uint64_t outputAddress,
    uint64_t inputToken, uint64_t outputToken, uint64_t rankOffset,
    uint64_t groupByteBase, uint64_t dataSize)
{
    return {
        inputAddress,
        outputAddress,
        inputToken,
        outputToken,
        rankOffset,
        groupByteBase,
        0,
        dataSize,
        groupByteBase,
        2ULL * dataSize,
        static_cast<uint64_t>(AllGatherKernelMode::RD4_XOR_TWO_STAGE),
    };
}

StageTaskArgs MakeLayer0StageArgs(uint64_t inputAddress, uint64_t outputAddress,
    uint64_t inputToken, uint64_t outputToken, uint64_t rankOffset,
    uint64_t partnerOffset, uint64_t ownOffset, uint64_t ownSize,
    uint64_t relayOffset, uint64_t relaySize, bool exchangeRemoteState,
    bool finalPostSync)
{
    return {
        inputAddress,
        outputAddress,
        inputToken,
        outputToken,
        rankOffset,
        partnerOffset,
        ownOffset,
        ownSize,
        relayOffset,
        relaySize,
        exchangeRemoteState ? 1ULL : 0ULL,
        finalPostSync ? 1ULL : 0ULL,
    };
}

StageTaskArgs MakeLayer1StageArgs(uint64_t inputAddress, uint64_t outputAddress,
    uint64_t inputToken, uint64_t outputToken, uint64_t rankOffset,
    uint64_t chunkOffset, uint64_t chunkSize, uint64_t directOffset,
    uint64_t directSize, bool exchangeRemoteState, bool partnerReady,
    bool finalPostSync)
{
    return {
        inputAddress,
        outputAddress,
        inputToken,
        outputToken,
        rankOffset,
        chunkOffset,
        chunkSize,
        directOffset,
        directSize,
        exchangeRemoteState ? 1ULL : 0ULL,
        partnerReady ? 1ULL : 0ULL,
        finalPostSync ? 1ULL : 0ULL,
    };
}

A84TaskArgs MakeA84TaskArgs(uint64_t inputAddress, uint64_t outputAddress,
    uint64_t inputToken, uint64_t outputToken, uint64_t dataSize,
    uint64_t rankOffset, uint64_t rangeOffset, uint64_t rangeSize,
    uint64_t prefixSize, bool exchangeRemoteState, bool finalPostSync,
    A84PrefixKernelMode mode)
{
    return {
        inputAddress,
        outputAddress,
        inputToken,
        outputToken,
        dataSize,
        rankOffset,
        rangeOffset,
        rangeSize,
        prefixSize,
        exchangeRemoteState ? 1ULL : 0ULL,
        finalPostSync ? 1ULL : 0ULL,
        static_cast<uint64_t>(mode),
    };
}

uint64_t ComputeA84PrefixBytes(uint64_t dataSize, uint32_t destinationGroupSize)
{
    uint64_t numerator = 0;
    uint64_t denominator = 0;
    if (destinationGroupSize == 4) {
        numerator = 2;
        denominator = 7;
    } else if (destinationGroupSize == 8) {
        numerator = 4;
        denominator = 11;
    } else {
        return 0;
    }
    const uint64_t quotientPart = (dataSize / denominator) * numerator;
    const uint64_t remainderPart = ((dataSize % denominator) * numerator) / denominator;
    return (quotientPart + remainderPart) & ~(FP32_ALIGNMENT - 1ULL);
}

HcclResult BuildAtMostTwoSlices(uint64_t offset, uint64_t size,
    std::array<PhysicalSlice, 2> &slices, uint32_t &sliceCount)
{
    CHK_PRT_RET(size == 0 || size > 2ULL * MAX_DATA_SIZE,
        HCCL_ERROR("A84 physical range is unsupported: offset=%llu size=%llu",
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(size)),
        HCCL_E_PARA);
    CHK_PRT_RET((offset & (FP32_ALIGNMENT - 1ULL)) != 0 ||
            (size & (FP32_ALIGNMENT - 1ULL)) != 0,
        HCCL_ERROR("A84 physical range is not FP32 aligned"), HCCL_E_PARA);

    slices[0].offset = offset;
    slices[0].size = std::min<uint64_t>(MAX_DATA_SIZE, size);
    sliceCount = 1;
    if (slices[0].size < size) {
        slices[1].offset = offset + slices[0].size;
        slices[1].size = size - slices[0].size;
        sliceCount = 2;
    }
    CHK_PRT_RET(slices[0].size == 0 || slices[0].size > MAX_DATA_SIZE ||
            (sliceCount == 2 &&
                (slices[1].size == 0 || slices[1].size > MAX_DATA_SIZE)) ||
            slices[0].offset + slices[0].size +
                (sliceCount == 2 ? slices[1].size : 0ULL) != offset + size,
        HCCL_ERROR("A84 physical slices do not cover their range"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

uint64_t ComputeRelayBytes(uint64_t chunkSize, uint32_t chunkIndex)
{
    // Measurement-calibrated per-generation relay schedule:
    // C0=12/25, C1=19/50, C2=19/50, C3=3/20.  Each product is
    // evaluated by quotient/remainder decomposition to avoid overflow.
    constexpr std::array<uint64_t, 4> numerators = {12ULL, 19ULL, 19ULL, 3ULL};
    constexpr std::array<uint64_t, 4> denominators = {25ULL, 50ULL, 50ULL, 20ULL};
    const uint64_t numerator = numerators[chunkIndex];
    const uint64_t denominator = denominators[chunkIndex];
    const uint64_t quotientPart = (chunkSize / denominator) * numerator;
    const uint64_t remainderPart = ((chunkSize % denominator) * numerator) / denominator;
    return (quotientPart + remainderPart) & ~(FP32_ALIGNMENT - 1ULL);
}

uint64_t ComputePrefixFirstRelayBytes(uint64_t dataSize)
{
    // Balance the two 2x8 IO-Die paths:
    //   layer-0 Mesh = 1 + r
    //   layer-1 Clos = (8 - 7r) / 4
    // which gives r = 4 / 11 and a common load of 15 / 11.
    constexpr uint64_t numerator = 4;
    constexpr uint64_t denominator = 11;
    const uint64_t quotientPart =
        (dataSize / denominator) * numerator;
    const uint64_t remainderPart =
        ((dataSize % denominator) * numerator) / denominator;
    return (quotientPart + remainderPart) & ~(FP32_ALIGNMENT - 1ULL);
}

HcclResult BuildSplitK4Chunks(uint64_t dataSize, std::array<MixedChunk, 4> &chunks)
{
    CHK_PRT_RET((dataSize & (FP32_ALIGNMENT - 1ULL)) != 0,
        HCCL_ERROR("FP32 data size %llu is not 4-byte aligned",
            static_cast<unsigned long long>(dataSize)),
        HCCL_E_PARA);

    const uint64_t base = (dataSize / 4ULL) & ~(FP32_ALIGNMENT - 1ULL);
    CHK_PRT_RET(base == 0 || base >= dataSize,
        HCCL_ERROR("Unable to split data size %llu into four non-empty chunks",
            static_cast<unsigned long long>(dataSize)),
        HCCL_E_PARA);

    chunks[0].offset = 0;
    chunks[0].size = base;
    chunks[1].offset = chunks[0].offset + chunks[0].size;
    chunks[1].size = base;
    chunks[2].offset = chunks[1].offset + chunks[1].size;
    chunks[2].size = base;
    chunks[3].offset = chunks[2].offset + chunks[2].size;
    chunks[3].size = dataSize - chunks[3].offset;

    for (uint32_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        MixedChunk &chunk = chunks[chunkIndex];
        chunk.relaySize = ComputeRelayBytes(chunk.size, chunkIndex);
        chunk.directOffset = chunk.offset + chunk.relaySize;
        chunk.directSize = chunk.size - chunk.relaySize;

        CHK_PRT_RET(chunk.relaySize == 0 || chunk.directSize == 0 ||
                chunk.size > MAX_DATA_SIZE || chunk.relaySize > MAX_DATA_SIZE ||
                chunk.directSize > MAX_DATA_SIZE,
            HCCL_ERROR("K4 chunk is outside the supported transfer limits: offset=%llu size=%llu relay=%llu direct=%llu",
                static_cast<unsigned long long>(chunk.offset),
                static_cast<unsigned long long>(chunk.size),
                static_cast<unsigned long long>(chunk.relaySize),
                static_cast<unsigned long long>(chunk.directSize)),
            HCCL_E_PARA);
        CHK_PRT_RET((chunk.offset & (FP32_ALIGNMENT - 1ULL)) != 0 ||
                (chunk.size & (FP32_ALIGNMENT - 1ULL)) != 0 ||
                (chunk.relaySize & (FP32_ALIGNMENT - 1ULL)) != 0 ||
                (chunk.directOffset & (FP32_ALIGNMENT - 1ULL)) != 0 ||
                (chunk.directSize & (FP32_ALIGNMENT - 1ULL)) != 0,
            HCCL_ERROR("K4 chunk partition is not FP32 aligned"), HCCL_E_INTERNAL);
        CHK_PRT_RET(chunk.directOffset + chunk.directSize != chunk.offset + chunk.size,
            HCCL_ERROR("K4 R/D partition does not cover its chunk"), HCCL_E_INTERNAL);
    }

    CHK_PRT_RET(chunks[0].offset != 0 ||
            chunks[0].offset + chunks[0].size != chunks[1].offset ||
            chunks[1].offset + chunks[1].size != chunks[2].offset ||
            chunks[2].offset + chunks[2].size != chunks[3].offset ||
            chunks[3].offset + chunks[3].size != dataSize,
        HCCL_ERROR("K4 chunk partition does not cover the input"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult QueueThreadRecord(ThreadHandle sourceThread, ThreadHandle destinationThread,
    uint32_t destinationNotifyIndex)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        sourceThread, destinationThread, destinationNotifyIndex)));
    return HCCL_SUCCESS;
}

HcclResult QueueThreadWait(ThreadHandle thread, uint32_t localNotifyIndex)
{
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(thread, localNotifyIndex, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult LaunchParallelDirect(const AlgResourceCtx &resourceCtx,
    const DirectTaskArgs &taskArgs)
{
    const ThreadHandle mainThread = resourceCtx.threads[0];
    const ThreadHandle slaveThread = resourceCtx.threads[1];

    CHK_RET(QueueThreadRecord(mainThread, slaveThread, START_NOTIFY_INDEX));
    CHK_RET(QueueThreadWait(slaveThread, START_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resourceCtx.ccuKernels[0],
        taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
    CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread, resourceCtx.ccuKernels[1],
        taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
    CHK_RET(QueueThreadRecord(slaveThread, mainThread, START_NOTIFY_INDEX));
    CHK_RET(QueueThreadWait(mainThread, START_NOTIFY_INDEX));
    return HCCL_SUCCESS;
}

HcclResult LaunchTwoStageSmallPull(const AlgResourceCtx &resourceCtx,
    const SmallPullTaskArgs &layer0Args,
    const SmallPullTaskArgs &layer1Args)
{
    const ThreadHandle mainThread = resourceCtx.threads[0];
    const ThreadHandle slaveThread = resourceCtx.threads[1];

    // Reuse notify 0 in two ordered host-thread generations. Each CCU kernel
    // performs token exchange plus local Read completion, without a second
    // cross-rank PostSync generation.
    // main : Record START -> L0 pull -> Wait DONE
    // slave: Wait START    -> L1 pull -> Record DONE
    CHK_RET(QueueThreadRecord(mainThread, slaveThread, START_NOTIFY_INDEX));
    CHK_RET(QueueThreadWait(slaveThread, START_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resourceCtx.ccuKernels[0],
        layer0Args.data(), static_cast<uint32_t>(layer0Args.size())));
    CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread, resourceCtx.ccuKernels[1],
        layer1Args.data(), static_cast<uint32_t>(layer1Args.size())));
    CHK_RET(QueueThreadRecord(slaveThread, mainThread, START_NOTIFY_INDEX));
    CHK_RET(QueueThreadWait(mainThread, START_NOTIFY_INDEX));
    return HCCL_SUCCESS;
}

HcclResult LaunchTwoStageSmallPullWarm(const AlgResourceCtx &resourceCtx,
    const SmallPullWarmTaskArgs &layer0Args,
    const SmallPullWarmTaskArgs &layer1Args)
{
    const ThreadHandle mainThread = resourceCtx.threads[0];
    const ThreadHandle slaveThread = resourceCtx.threads[1];

    CHK_RET(QueueThreadRecord(mainThread, slaveThread, START_NOTIFY_INDEX));
    CHK_RET(QueueThreadWait(slaveThread, START_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resourceCtx.ccuKernels[2],
        layer0Args.data(), static_cast<uint32_t>(layer0Args.size())));
    CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread, resourceCtx.ccuKernels[3],
        layer1Args.data(), static_cast<uint32_t>(layer1Args.size())));
    CHK_RET(QueueThreadRecord(slaveThread, mainThread, START_NOTIFY_INDEX));
    CHK_RET(QueueThreadWait(mainThread, START_NOTIFY_INDEX));
    return HCCL_SUCCESS;
}

HcclResult LaunchTwoStagePrefixFirst(const AlgResourceCtx &resourceCtx,
    const std::array<StageTaskArgs, 2> &layer0OwnArgs,
    uint32_t layer0OwnCount, const StageTaskArgs &layer0RelayArgs,
    const StageTaskArgs &layer1PrefixArgs,
    const std::array<StageTaskArgs, 2> &layer1SuffixArgs,
    uint32_t layer1SuffixCount)
{
    CHK_PRT_RET(layer0OwnCount == 0 || layer0OwnCount > layer0OwnArgs.size() ||
            layer1SuffixCount == 0 ||
            layer1SuffixCount > layer1SuffixArgs.size(),
        HCCL_ERROR("Invalid Prefix-first launch counts: own=%u suffix=%u",
            layer0OwnCount, layer1SuffixCount),
        HCCL_E_PARA);

    const ThreadHandle mainThread = resourceCtx.threads[0];
    const ThreadHandle slaveThread = resourceCtx.threads[1];

    // Main FIFO:
    //   START -> Own[0..1] -> Wait PREFIX_READY -> Relay -> L0 PostSync
    //   -> Wait L1_DONE.
    // Slave FIFO:
    //   Wait START -> Prefix-to-proxy -> PREFIX_READY -> Suffix[0..1]
    //   -> L1 PostSync -> L1_DONE.
    //
    // PREFIX_READY is queued before the suffix launches.  It releases the
    // layer-0 relay as soon as the disjoint prefix is visible, while the
    // layer-1 suffix continues independently on the other IO Die.
    CHK_RET(QueueThreadRecord(mainThread, slaveThread, START_NOTIFY_INDEX));
    CHK_RET(QueueThreadWait(slaveThread, START_NOTIFY_INDEX));

    for (uint32_t index = 0; index < layer0OwnCount; ++index) {
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            resourceCtx.ccuKernels[0], layer0OwnArgs[index].data(),
            static_cast<uint32_t>(layer0OwnArgs[index].size())));
    }

    CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread,
        resourceCtx.ccuKernels[1], layer1PrefixArgs.data(),
        static_cast<uint32_t>(layer1PrefixArgs.size())));
    CHK_RET(QueueThreadRecord(
        slaveThread, mainThread, READY0_NOTIFY_INDEX));
    for (uint32_t index = 0; index < layer1SuffixCount; ++index) {
        CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread,
            resourceCtx.ccuKernels[1], layer1SuffixArgs[index].data(),
            static_cast<uint32_t>(layer1SuffixArgs[index].size())));
    }
    CHK_RET(QueueThreadRecord(
        slaveThread, mainThread, L1_DONE_NOTIFY_INDEX));

    CHK_RET(QueueThreadWait(mainThread, READY0_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
        resourceCtx.ccuKernels[0], layer0RelayArgs.data(),
        static_cast<uint32_t>(layer0RelayArgs.size())));
    CHK_RET(QueueThreadWait(mainThread, L1_DONE_NOTIFY_INDEX));
    return HCCL_SUCCESS;
}

HcclResult LaunchTwoStageSplitK4(const AlgResourceCtx &resourceCtx,
    const std::array<StageTaskArgs, 5> &layer0Args,
    const std::array<StageTaskArgs, 5> &layer1Args)
{
    const ThreadHandle mainThread = resourceCtx.threads[0];
    const ThreadHandle slaveThread = resourceCtx.threads[1];

    // Main FIFO:
    //   Record START -> Own0 -> READY0 -> Relay0+Own1 -> READY1
    //   -> Relay1+Own2 -> READY2 -> Relay2+Own3 -> READY3
    //   -> Relay3+L0 final PostSync -> Wait L1_DONE.
    // Slave FIFO:
    //   Wait START -> L1 C0 -> READY0 -> L1 C1 -> READY1
    //   -> L1 C2 -> READY2 -> L1 C3 -> READY3
    //   -> finalize-only -> L1_DONE.
    CHK_RET(QueueThreadRecord(mainThread, slaveThread, START_NOTIFY_INDEX));
    CHK_RET(QueueThreadWait(slaveThread, START_NOTIFY_INDEX));

    // Own0 has no partner-data dependency and overlaps L1 C0.
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resourceCtx.ccuKernels[0],
        layer0Args[0].data(), static_cast<uint32_t>(layer0Args[0].size())));

    // Enqueue the complete slave FIFO. Each READY is recorded only after the
    // preceding L1 data kernel has completed EventWait and partner bit4 sync.
    CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread, resourceCtx.ccuKernels[1],
        layer1Args[0].data(), static_cast<uint32_t>(layer1Args[0].size())));
    CHK_RET(QueueThreadRecord(slaveThread, mainThread, READY0_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread, resourceCtx.ccuKernels[1],
        layer1Args[1].data(), static_cast<uint32_t>(layer1Args[1].size())));
    CHK_RET(QueueThreadRecord(slaveThread, mainThread, READY1_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread, resourceCtx.ccuKernels[1],
        layer1Args[2].data(), static_cast<uint32_t>(layer1Args[2].size())));
    CHK_RET(QueueThreadRecord(slaveThread, mainThread, READY2_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread, resourceCtx.ccuKernels[1],
        layer1Args[3].data(), static_cast<uint32_t>(layer1Args[3].size())));
    CHK_RET(QueueThreadRecord(slaveThread, mainThread, READY3_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread, resourceCtx.ccuKernels[1],
        layer1Args[4].data(), static_cast<uint32_t>(layer1Args[4].size())));
    CHK_RET(QueueThreadRecord(slaveThread, mainThread, L1_DONE_NOTIFY_INDEX));

    // READY[k] strictly gates Relay[k]. The next Own chunk shares the same L0
    // stage so it overlaps the following L1 data chunk on the other IO Die.
    CHK_RET(QueueThreadWait(mainThread, READY0_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resourceCtx.ccuKernels[0],
        layer0Args[1].data(), static_cast<uint32_t>(layer0Args[1].size())));
    CHK_RET(QueueThreadWait(mainThread, READY1_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resourceCtx.ccuKernels[0],
        layer0Args[2].data(), static_cast<uint32_t>(layer0Args[2].size())));
    CHK_RET(QueueThreadWait(mainThread, READY2_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resourceCtx.ccuKernels[0],
        layer0Args[3].data(), static_cast<uint32_t>(layer0Args[3].size())));
    CHK_RET(QueueThreadWait(mainThread, READY3_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resourceCtx.ccuKernels[0],
        layer0Args[4].data(), static_cast<uint32_t>(layer0Args[4].size())));
    CHK_RET(QueueThreadWait(mainThread, L1_DONE_NOTIFY_INDEX));
    return HCCL_SUCCESS;
}

HcclResult LaunchA84PrefixFirst(const AlgResourceCtx &resourceCtx,
    const std::array<A84TaskArgs, 2> &layer0OwnArgs, uint32_t layer0OwnCount,
    const A84TaskArgs &layer0RelayArgs, const A84TaskArgs &layer1PrefixArgs,
    const std::array<A84TaskArgs, 2> &layer1SuffixArgs, uint32_t layer1SuffixCount)
{
    CHK_PRT_RET(resourceCtx.ccuKernels.size() != 4 ||
            layer0OwnCount == 0 || layer0OwnCount > layer0OwnArgs.size() ||
            layer1SuffixCount == 0 || layer1SuffixCount > layer1SuffixArgs.size(),
        HCCL_ERROR("Invalid A84 Prefix-first launch resources/counts"), HCCL_E_INTERNAL);

    const ThreadHandle mainThread = resourceCtx.threads[0];
    const ThreadHandle slaveThread = resourceCtx.threads[1];
    constexpr uint32_t layer0KernelIndex = 2;
    constexpr uint32_t layer1KernelIndex = 3;

    // Main FIFO:
    //   START -> Own physical slices -> Wait PREFIX_READY
    //   -> Relay all proxy-owned prefixes + L0 PostSync -> Wait L1_DONE.
    // Slave FIFO:
    //   Wait START -> outgoing Prefix + incoming per-source bit4 waits
    //   -> PREFIX_READY -> Suffix physical slices + L1 PostSync -> L1_DONE.
    CHK_RET(QueueThreadRecord(mainThread, slaveThread, START_NOTIFY_INDEX));
    CHK_RET(QueueThreadWait(slaveThread, START_NOTIFY_INDEX));

    for (uint32_t index = 0; index < layer0OwnCount; ++index) {
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            resourceCtx.ccuKernels[layer0KernelIndex], layer0OwnArgs[index].data(),
            static_cast<uint32_t>(layer0OwnArgs[index].size())));
    }

    CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread,
        resourceCtx.ccuKernels[layer1KernelIndex], layer1PrefixArgs.data(),
        static_cast<uint32_t>(layer1PrefixArgs.size())));
    CHK_RET(QueueThreadRecord(
        slaveThread, mainThread, A84_PREFIX_READY_NOTIFY_INDEX));

    for (uint32_t index = 0; index < layer1SuffixCount; ++index) {
        CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread,
            resourceCtx.ccuKernels[layer1KernelIndex], layer1SuffixArgs[index].data(),
            static_cast<uint32_t>(layer1SuffixArgs[index].size())));
    }
    CHK_RET(QueueThreadRecord(
        slaveThread, mainThread, A84_L1_DONE_NOTIFY_INDEX));

    CHK_RET(QueueThreadWait(mainThread, A84_PREFIX_READY_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
        resourceCtx.ccuKernels[layer0KernelIndex], layer0RelayArgs.data(),
        static_cast<uint32_t>(layer0RelayArgs.size())));
    CHK_RET(QueueThreadWait(mainThread, A84_L1_DONE_NOTIFY_INDEX));
    return HCCL_SUCCESS;
}

HcclResult ExecuteA84PrefixFirst(const OpParam &param,
    const AlgResourceCtx &resourceCtx, uint64_t dataSize, uint64_t outputSize,
    uint64_t inputAddress, uint64_t outputAddress,
    uint64_t inputToken, uint64_t outputToken)
{
    CHK_PRT_RET(resourceCtx.localGroupSize != 4 && resourceCtx.localGroupSize != 8,
        HCCL_ERROR("Invalid A84 local group size %u", resourceCtx.localGroupSize),
        HCCL_E_INTERNAL);
    const uint32_t remoteGroupSize = 12U - resourceCtx.localGroupSize;
    const uint64_t outgoingPrefixSize =
        ComputeA84PrefixBytes(dataSize, remoteGroupSize);
    const uint64_t incomingPrefixSize =
        ComputeA84PrefixBytes(dataSize, resourceCtx.localGroupSize);
    CHK_PRT_RET(outgoingPrefixSize == 0 || incomingPrefixSize == 0 ||
            outgoingPrefixSize >= dataSize || incomingPrefixSize >= dataSize ||
            outgoingPrefixSize > MAX_DATA_SIZE || incomingPrefixSize > MAX_DATA_SIZE ||
            (outgoingPrefixSize & (FP32_ALIGNMENT - 1ULL)) != 0 ||
            (incomingPrefixSize & (FP32_ALIGNMENT - 1ULL)) != 0,
        HCCL_ERROR("Invalid A84 prefix sizes: outgoing=%llu incoming=%llu data=%llu",
            static_cast<unsigned long long>(outgoingPrefixSize),
            static_cast<unsigned long long>(incomingPrefixSize),
            static_cast<unsigned long long>(dataSize)),
        HCCL_E_INTERNAL);

    const uint64_t rankOffset = dataSize * static_cast<uint64_t>(param.myRank);
    CHK_PRT_RET(rankOffset >= outputSize,
        HCCL_ERROR("A84 rank offset is outside the output buffer"), HCCL_E_INTERNAL);

    std::array<PhysicalSlice, 2> ownSlices{};
    uint32_t ownSliceCount = 0;
    CHK_RET(BuildAtMostTwoSlices(0, dataSize, ownSlices, ownSliceCount));

    const uint64_t suffixSize = dataSize - outgoingPrefixSize;
    std::array<PhysicalSlice, 2> suffixSlices{};
    uint32_t suffixSliceCount = 0;
    CHK_RET(BuildAtMostTwoSlices(
        outgoingPrefixSize, suffixSize, suffixSlices, suffixSliceCount));

    std::array<A84TaskArgs, 2> layer0OwnArgs{};
    for (uint32_t index = 0; index < ownSliceCount; ++index) {
        layer0OwnArgs[index] = MakeA84TaskArgs(inputAddress, outputAddress,
            inputToken, outputToken, dataSize, rankOffset,
            ownSlices[index].offset, ownSlices[index].size, incomingPrefixSize,
            index == 0, false, A84PrefixKernelMode::LAYER0_OWN);
    }
    const A84TaskArgs layer0RelayArgs = MakeA84TaskArgs(
        inputAddress, outputAddress, inputToken, outputToken, dataSize,
        rankOffset, 0, 0, incomingPrefixSize, false, true,
        A84PrefixKernelMode::LAYER0_RELAY);

    const A84TaskArgs layer1PrefixArgs = MakeA84TaskArgs(
        inputAddress, outputAddress, inputToken, outputToken, dataSize,
        rankOffset, 0, 0, outgoingPrefixSize, true, false,
        A84PrefixKernelMode::LAYER1_PREFIX);

    std::array<A84TaskArgs, 2> layer1SuffixArgs{};
    for (uint32_t index = 0; index < suffixSliceCount; ++index) {
        layer1SuffixArgs[index] = MakeA84TaskArgs(inputAddress, outputAddress,
            inputToken, outputToken, dataSize, rankOffset,
            suffixSlices[index].offset, suffixSlices[index].size,
            outgoingPrefixSize, false, index + 1 == suffixSliceCount,
            A84PrefixKernelMode::LAYER1_SUFFIX);
    }

    return LaunchA84PrefixFirst(resourceCtx, layer0OwnArgs, ownSliceCount,
        layer0RelayArgs, layer1PrefixArgs, layer1SuffixArgs, suffixSliceCount);
}

HcclResult ExecuteTwoStage2x8(const OpParam &param, const AlgResourceCtx &resourceCtx,
    uint64_t dataSize, uint64_t outputSize, uint64_t inputAddress,
    uint64_t outputAddress, uint64_t inputToken, uint64_t outputToken)
{
    const uint64_t rankOffset =
        dataSize * static_cast<uint64_t>(param.myRank);
    CHK_PRT_RET(rankOffset >= outputSize,
        HCCL_ERROR("Rank offset is outside the output buffer"),
        HCCL_E_INTERNAL);

    if (dataSize != 0 && dataSize <= SMALL_DIRECT_THRESHOLD) {
        const SmallPullTaskArgs layer0Args = MakeSmallPullTaskArgs(
            inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, dataSize);
        const SmallPullTaskArgs layer1Args = layer0Args;
        const SmallPullWarmTaskArgs warmArgs = MakeSmallPullWarmTaskArgs(
            outputAddress, outputToken, dataSize);
        if (param.root == 1U) {
            return LaunchTwoStageSmallPullWarm(
                resourceCtx, warmArgs, warmArgs);
        }
        if (param.root == 2U) {
            CHK_RET(LaunchTwoStageSmallPull(
                resourceCtx, layer0Args, layer1Args));
            return LaunchTwoStageSmallPullWarm(
                resourceCtx, warmArgs, warmArgs);
        }
        return LaunchTwoStageSmallPull(resourceCtx, layer0Args, layer1Args);
    }

    CHK_PRT_RET(resourceCtx.partnerRank == INVALID_VALUE_RANKID ||
            resourceCtx.partnerRank >= param.rankSize,
        HCCL_ERROR("Invalid 2x8 partner rank %u", resourceCtx.partnerRank),
        HCCL_E_INTERNAL);

    const uint64_t partnerOffset =
        dataSize * static_cast<uint64_t>(resourceCtx.partnerRank);
    CHK_PRT_RET(partnerOffset >= outputSize,
        HCCL_ERROR("Partner offset is outside the output buffer"),
        HCCL_E_INTERNAL);

    // Prefix-first is used for all competition-sized large messages that fit
    // in two physical CCU transfers.  Larger, non-competition inputs retain
    // the proven D06 Split-K4 fallback below.
    if (dataSize <= 2ULL * MAX_DATA_SIZE) {
        const uint64_t relaySize =
            ComputePrefixFirstRelayBytes(dataSize);
        const uint64_t directOffset = relaySize;
        const uint64_t directSize = dataSize - relaySize;
        CHK_PRT_RET(relaySize == 0 || relaySize > MAX_DATA_SIZE ||
                directSize == 0 ||
                (relaySize & (FP32_ALIGNMENT - 1ULL)) != 0 ||
                (directSize & (FP32_ALIGNMENT - 1ULL)) != 0 ||
                relaySize + directSize != dataSize,
            HCCL_ERROR("Invalid Prefix-first R/D split: data=%llu relay=%llu direct=%llu",
                static_cast<unsigned long long>(dataSize),
                static_cast<unsigned long long>(relaySize),
                static_cast<unsigned long long>(directSize)),
            HCCL_E_INTERNAL);

        std::array<PhysicalSlice, 2> ownSlices{};
        uint32_t ownSliceCount = 0;
        CHK_RET(BuildAtMostTwoSlices(
            0, dataSize, ownSlices, ownSliceCount));

        std::array<PhysicalSlice, 2> suffixSlices{};
        uint32_t suffixSliceCount = 0;
        CHK_RET(BuildAtMostTwoSlices(
            directOffset, directSize, suffixSlices, suffixSliceCount));

        std::array<StageTaskArgs, 2> layer0OwnArgs{};
        for (uint32_t index = 0; index < ownSliceCount; ++index) {
            layer0OwnArgs[index] = MakeLayer0StageArgs(
                inputAddress, outputAddress, inputToken, outputToken,
                rankOffset, partnerOffset, ownSlices[index].offset,
                ownSlices[index].size, 0, 0, index == 0, false);
        }
        const StageTaskArgs layer0RelayArgs = MakeLayer0StageArgs(
            inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, partnerOffset, 0, 0, 0, relaySize,
            false, false);

        // Prefix: only the symmetric proxy and the local output receive R.
        // directSize==0 selects the 0x0101 Event mask in the layer-1 kernel.
        const StageTaskArgs layer1PrefixArgs = MakeLayer1StageArgs(
            inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, 0, relaySize, 0, 0, true, true, false);

        std::array<StageTaskArgs, 2> layer1SuffixArgs{};
        for (uint32_t index = 0; index < suffixSliceCount; ++index) {
            const PhysicalSlice &slice = suffixSlices[index];
            layer1SuffixArgs[index] = MakeLayer1StageArgs(
                inputAddress, outputAddress, inputToken, outputToken,
                rankOffset, slice.offset, slice.size, slice.offset,
                slice.size, false, false, false);
        }

        return LaunchTwoStagePrefixFirst(resourceCtx, layer0OwnArgs,
            ownSliceCount, layer0RelayArgs, layer1PrefixArgs,
            layer1SuffixArgs, suffixSliceCount);
    }

    std::array<MixedChunk, 4> chunks{};
    CHK_RET(BuildSplitK4Chunks(dataSize, chunks));

    const std::array<StageTaskArgs, 5> layer0Args = {
        // L0[0]: Own0-only -> OWN_MASK (0x007F).
        MakeLayer0StageArgs(inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, partnerOffset, chunks[0].offset, chunks[0].size,
            0, 0, true, false),
        // L0[1]: Relay0 + Own1 -> BOTH_MASK (0x3FFF).
        MakeLayer0StageArgs(inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, partnerOffset, chunks[1].offset, chunks[1].size,
            chunks[0].offset, chunks[0].relaySize, false, false),
        // L0[2]: Relay1 + Own2 -> BOTH_MASK (0x3FFF).
        MakeLayer0StageArgs(inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, partnerOffset, chunks[2].offset, chunks[2].size,
            chunks[1].offset, chunks[1].relaySize, false, false),
        // L0[3]: Relay2 + Own3 -> BOTH_MASK (0x3FFF).
        MakeLayer0StageArgs(inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, partnerOffset, chunks[3].offset, chunks[3].size,
            chunks[2].offset, chunks[2].relaySize, false, false),
        // L0[4]: Relay3-only plus final layer-0 PostSync -> RELAY_MASK (0x3F80).
        MakeLayer0StageArgs(inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, partnerOffset, 0, 0,
            chunks[3].offset, chunks[3].relaySize, false, true),
    };

    const std::array<StageTaskArgs, 5> layer1Args = {
        MakeLayer1StageArgs(inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, chunks[0].offset, chunks[0].size,
            chunks[0].directOffset, chunks[0].directSize, true, true, false),
        MakeLayer1StageArgs(inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, chunks[1].offset, chunks[1].size,
            chunks[1].directOffset, chunks[1].directSize, false, true, false),
        MakeLayer1StageArgs(inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, chunks[2].offset, chunks[2].size,
            chunks[2].directOffset, chunks[2].directSize, false, true, false),
        MakeLayer1StageArgs(inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, chunks[3].offset, chunks[3].size,
            chunks[3].directOffset, chunks[3].directSize, false, true, false),
        MakeLayer1StageArgs(inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, 0, 0, 0, 0, false, false, true),
    };

    return LaunchTwoStageSplitK4(resourceCtx, layer0Args, layer1Args);
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PTR_NULL(param.resCtx);
    char *context = static_cast<char *>(param.resCtx);
    std::vector<char> sequence(context, context + param.ctxSize);
    AlgResourceCtx resourceCtx;
    resourceCtx.DeSerialize(sequence);

    const auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type: %d", static_cast<int32_t>(param.dataType)),
        HCCL_E_NOT_SUPPORT);
    const uint64_t dataTypeSize = sizeIter->second;

    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("Input data size overflows uint64_t"), HCCL_E_PARA);
    const uint64_t dataSize = param.count * dataTypeSize;
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(param.rankSize == 0 ||
            dataSize > std::numeric_limits<uint64_t>::max() /
                static_cast<uint64_t>(param.rankSize),
        HCCL_ERROR("Output data size overflows uint64_t"), HCCL_E_PARA);
    const uint64_t outputSize = dataSize * static_cast<uint64_t>(param.rankSize);

    const bool hasLayer0 = resourceCtx.hasLayer0Kernel != 0;
    const bool useTwoStage2x8 = resourceCtx.useTwoStage2x8 != 0;
    const bool useAsymmetricPrefix8Plus4 =
        resourceCtx.useAsymmetricPrefix8Plus4 != 0;
    const uint32_t expectedThreadCount = hasLayer0 ? 2U : 1U;
    const bool useDedicatedSmallKernel =
        dataSize <= SMALL_DIRECT_THRESHOLD;
    const uint32_t expectedKernelCount = useDedicatedSmallKernel ?
        2U * expectedThreadCount :
        (useAsymmetricPrefix8Plus4 ? 4U : expectedThreadCount);
    CHK_PRT_RET(resourceCtx.threads.size() != expectedThreadCount ||
            resourceCtx.ccuKernels.size() != expectedKernelCount,
        HCCL_ERROR("Invalid resource count: threads=%zu kernels=%zu hasLayer0=%u prefix=%u",
            resourceCtx.threads.size(), resourceCtx.ccuKernels.size(),
            resourceCtx.hasLayer0Kernel, resourceCtx.useAsymmetricPrefix8Plus4),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(useAsymmetricPrefix8Plus4 && (!hasLayer0 || param.rankSize != 12),
        HCCL_ERROR("A84 Prefix-first context is inconsistent with rankSize=%u",
            param.rankSize), HCCL_E_INTERNAL);
    CHK_PRT_RET(useTwoStage2x8 && (!hasLayer0 || param.rankSize != 16),
        HCCL_ERROR("Two-Stage resource context is inconsistent with rankSize=%u",
            param.rankSize),
        HCCL_E_INTERNAL);

    const uint64_t inputAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddress = reinterpret_cast<uint64_t>(param.outputPtr);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_PRT_RET(useDedicatedSmallKernel && param.root > 2U,
        HCCL_ERROR("Invalid small-pull execution mode %u", param.root),
        HCCL_E_PARA);
    if (!useDedicatedSmallKernel || param.root != 1U) {
        CHK_RET_CCU(HcommCcuGetMemToken(inputAddress, dataSize, &inputToken));
    }
    CHK_RET_CCU(HcommCcuGetMemToken(outputAddress, outputSize, &outputToken));

    if (useTwoStage2x8) {
        return ExecuteTwoStage2x8(param, resourceCtx, dataSize, outputSize,
            inputAddress, outputAddress, inputToken, outputToken);
    }

    if (useAsymmetricPrefix8Plus4 && dataSize > SMALL_DIRECT_THRESHOLD) {
        return ExecuteA84PrefixFirst(param, resourceCtx, dataSize, outputSize,
            inputAddress, outputAddress, inputToken, outputToken);
    }

    const uint64_t rankOffset = dataSize * static_cast<uint64_t>(param.myRank);
    if (dataSize != 0 && dataSize <= SMALL_DIRECT_THRESHOLD) {
        const SmallPullTaskArgs taskArgs = MakeSmallPullTaskArgs(
            inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, dataSize);
        const SmallPullWarmTaskArgs warmArgs = MakeSmallPullWarmTaskArgs(
            outputAddress, outputToken, dataSize);
        if (hasLayer0) {
            if (param.root == 1U) {
                return LaunchTwoStageSmallPullWarm(
                    resourceCtx, warmArgs, warmArgs);
            }
            if (param.root == 2U) {
                CHK_RET(LaunchTwoStageSmallPull(
                    resourceCtx, taskArgs, taskArgs));
                return LaunchTwoStageSmallPullWarm(
                    resourceCtx, warmArgs, warmArgs);
            }
            return LaunchTwoStageSmallPull(resourceCtx, taskArgs, taskArgs);
        }
        if (param.root == 1U) {
            CHK_RET_CCU(HcommCcuKernelLaunch(resourceCtx.threads[0],
                resourceCtx.ccuKernels[1], warmArgs.data(),
                static_cast<uint32_t>(warmArgs.size())));
            return HCCL_SUCCESS;
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(resourceCtx.threads[0],
            resourceCtx.ccuKernels[0], taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size())));
        if (param.root == 2U) {
            CHK_RET_CCU(HcommCcuKernelLaunch(resourceCtx.threads[0],
                resourceCtx.ccuKernels[1], warmArgs.data(),
                static_cast<uint32_t>(warmArgs.size())));
        }
        return HCCL_SUCCESS;
    }

    const bool useRd4 = param.rankSize == 4 &&
        dataSize > SMALL_DIRECT_THRESHOLD &&
        dataSize <= MAX_DATA_SIZE / 2ULL;
    if (useRd4) {
        const uint64_t groupRankBase =
            static_cast<uint64_t>(param.myRank & ~1U);
        const uint64_t groupByteBase = groupRankBase * dataSize;
        const uint64_t groupBytes = 2ULL * dataSize;
        CHK_PRT_RET(groupBytes > MAX_DATA_SIZE ||
                rankOffset >= outputSize ||
                groupByteBase > outputSize - groupBytes,
            HCCL_ERROR("Invalid RD4 range: rank=%u data=%llu rankOffset=%llu "
                "groupBase=%llu groupBytes=%llu output=%llu",
                param.myRank,
                static_cast<unsigned long long>(dataSize),
                static_cast<unsigned long long>(rankOffset),
                static_cast<unsigned long long>(groupByteBase),
                static_cast<unsigned long long>(groupBytes),
                static_cast<unsigned long long>(outputSize)),
            HCCL_E_INTERNAL);
        const DirectTaskArgs taskArgs = MakeRd4TaskArgs(
            inputAddress, outputAddress, inputToken, outputToken,
            rankOffset, groupByteBase, dataSize);
        CHK_RET_CCU(HcommCcuKernelLaunch(resourceCtx.threads[0],
            resourceCtx.ccuKernels[0], taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size())));
        return HCCL_SUCCESS;
    }
    for (uint64_t windowOffset = 0; windowOffset < dataSize;) {
        const uint64_t windowSize =
            std::min<uint64_t>(DIRECT_WINDOW_SIZE, dataSize - windowOffset);
        const DirectTaskArgs taskArgs = MakeDirectTaskArgs(inputAddress, outputAddress,
            inputToken, outputToken, rankOffset, windowOffset, windowSize);

        if (hasLayer0) {
            CHK_RET(LaunchParallelDirect(resourceCtx, taskArgs));
        } else {
            CHK_RET_CCU(HcommCcuKernelLaunch(resourceCtx.threads[0],
                resourceCtx.ccuKernels[0], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size())));
        }
        windowOffset += windowSize;
    }

    return HCCL_SUCCESS;
}

} // namespace ops_hccl
