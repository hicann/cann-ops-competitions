/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <array>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle handle, const uint64_t *args, uint32_t argCount)
{
    const CcuResult result = HcommCcuKernelLaunch(thread, handle, args, argCount);
    CHK_PRT_RET(result != CCU_SUCCESS, HCCL_ERROR("CCU kernel launch failed: %d", result), ConvertCcuToHccl(result));
    return HCCL_SUCCESS;
}

bool IsKernelPresent(const AlgResourceCtx &resource, uint32_t index)
{
    return index != INVALID_VALUE_RANKID && index < resource.ccuKernels.size();
}

uint64_t SetBits(uint16_t bitCount)
{
    return (uint64_t{1} << bitCount) - 1;
}

uint64_t MakeParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    return ((repeatNum & SetBits(7)) << 55) | ((repeatLoopIndex & SetBits(7)) << 48) |
        ((totalLoopNum & SetBits(7)) << 41);
}

struct GroupReduceLoopArgs {
    uint64_t fullOffset;
    uint64_t fullLoopCount;
    uint64_t parallelParam;
    uint64_t residualBytes;
};

GroupReduceLoopArgs MakeGroupReduceLoopArgs(uint64_t bytes)
{
    constexpr uint64_t pieceBytes = 4096;
    constexpr uint64_t parallelPieces = 16;
    constexpr uint64_t loopBytes = pieceBytes * parallelPieces;
    const uint64_t fullLoopCount = bytes / loopBytes;
    const uint64_t fullOffset = fullLoopCount * loopBytes;
    const uint64_t remainder = bytes - fullOffset;
    const uint64_t completePieces = remainder / pieceBytes;
    const uint64_t residualBytes = remainder - completePieces * pieceBytes;
    uint64_t parallelParam = 0;
    uint64_t loopResidual = 0;
    if (completePieces != 0 && residualBytes == 0) {
        parallelParam = MakeParallelParam(completePieces - 1, 0, 1);
        loopResidual = pieceBytes;
    } else if (completePieces == 0 && residualBytes != 0) {
        parallelParam = MakeParallelParam(0, 0, 1);
        loopResidual = residualBytes;
    } else if (completePieces != 0) {
        parallelParam = MakeParallelParam(completePieces - 1, 1, 2);
        loopResidual = residualBytes;
    }
    return {fullOffset, fullLoopCount, parallelParam, loopResidual};
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    std::vector<char> serialized(static_cast<char *>(param.resCtx), static_cast<char *>(param.resCtx) + param.ctxSize);
    AlgResourceCtx resource;
    resource.DeSerialize(serialized);
    CHK_PRT_RET(resource.threads.empty() || (!resource.useHierarchicalTwoByEight && !resource.useCcuBufferGroupReduce &&
            !resource.useLayerPartialReduce &&
            !resource.useSmallNhr &&
            !resource.useSmallGroupFanIn &&
            !resource.useStripedClosDirect &&
            !resource.useStripedTwoLayerDirect &&
            !IsKernelPresent(resource, resource.finalReduceKernelIndex)),
        HCCL_ERROR("invalid CCU resource context"), HCCL_E_INTERNAL);

    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("unsupported data type"), HCCL_E_NOT_SUPPORT);
    const uint64_t elementSize = sizeIt->second;
    const uint64_t outputBytes = param.count * elementSize;
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(resource.threads[0], param.outputPtr, param.inputPtr,
            outputBytes));
    }
    CHK_PRT_RET(outputBytes == 0, HCCL_ERROR("zero-size ReduceScatter is not supported by CCU tokens"), HCCL_E_PARA);

    const uint64_t slotBytes = resource.localBuffer.size / param.rankSize;
    const uint64_t groupScratchBytes = resource.useCcuBufferGroupReduce ? resource.localBuffer.size : slotBytes;
    const uint64_t maxChunkBytes = std::min<uint64_t>(MAX_DATA_SIZE, groupScratchBytes - groupScratchBytes % elementSize);
    CHK_PRT_RET(maxChunkBytes == 0, HCCL_ERROR("CCL buffer is too small for one rank slot"), HCCL_E_INTERNAL);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), outputBytes * param.rankSize,
        &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), outputBytes, &outputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(resource.localBuffer.addr), resource.localBuffer.size,
        &scratchToken));

    const bool hasMeshKernel = IsKernelPresent(resource, resource.gatherKernelIndex[0]);
    const bool hasClosKernel = IsKernelPresent(resource, resource.gatherKernelIndex[1]);
    CHK_PRT_RET(!hasMeshKernel && !hasClosKernel, HCCL_ERROR("no gather CCU kernel was registered"), HCCL_E_INTERNAL);
    CHK_PRT_RET(hasClosKernel && resource.threads.size() < 2,
        HCCL_ERROR("Clos kernel requires the second CCU thread"), HCCL_E_INTERNAL);

    if (resource.useStripedTwoLayerDirect) {
        CHK_PRT_RET(!hasMeshKernel || !hasClosKernel || param.rankSize != 12 ||
            resource.directStripeCounts[0] == 0 || resource.directStripeCounts[1] == 0,
            HCCL_ERROR("striped two-layer direct resource context is invalid"), HCCL_E_INTERNAL);
        auto makeLayerArgs = [&](uint32_t layer) {
            const uint64_t stripeCount = resource.directStripeCounts[layer];
            const uint64_t elementCount = outputBytes / elementSize;
            const uint64_t regularBytes = (elementCount / stripeCount) * elementSize;
            const uint64_t lastBytes = outputBytes - regularBytes * (stripeCount - 1);
            return std::array<uint64_t, 7>{
                reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr),
                inputToken, outputToken, static_cast<uint64_t>(param.myRank) * outputBytes,
                regularBytes, lastBytes};
        };
        const auto meshArgs = makeLayerArgs(0);
        const auto closArgs = makeLayerArgs(1);
        CHK_RET(LaunchKernel(resource.threads[0], resource.ccuKernels[resource.gatherKernelIndex[0]],
            meshArgs.data(), meshArgs.size()));
        // The Clos kernel updates the user output produced by Mesh, so its
        // wait is queued after the Mesh kernel rather than relying on launch
        // order across the two CCU threads.
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[0], resource.threads[1], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[1], 0, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchKernel(resource.threads[1], resource.ccuKernels[resource.gatherKernelIndex[1]],
            closArgs.data(), closArgs.size()));
        // Keep the slave thread's final task as a local record and make the
        // direct Clos updates visible before the collective returns.
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[1], resource.threads[0], 0)));
        return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT));
    }

    if (resource.useStripedClosDirect) {
        CHK_PRT_RET(hasMeshKernel || !hasClosKernel || param.rankSize != 4,
            HCCL_ERROR("striped Clos direct resource context is invalid"), HCCL_E_INTERNAL);
        const uint64_t elementCount = outputBytes / elementSize;
        const uint64_t baseElements = elementCount / 3;
        const uint64_t remainderElements = elementCount % 3;
        const uint64_t stripe0Bytes = (baseElements + (remainderElements > 0 ? 1 : 0)) * elementSize;
        const uint64_t stripe1Bytes = (baseElements + (remainderElements > 1 ? 1 : 0)) * elementSize;
        const uint64_t stripe2Bytes = outputBytes - stripe0Bytes - stripe1Bytes;
        CHK_PRT_RET(stripe0Bytes == 0 || stripe1Bytes == 0 || stripe2Bytes == 0,
            HCCL_ERROR("striped Clos direct requires three non-empty stripes"), HCCL_E_PARA);
        const std::array<uint64_t, 11> directArgs = {
            reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr),
            inputToken, outputToken, static_cast<uint64_t>(param.myRank) * outputBytes,
            0, stripe0Bytes, stripe0Bytes, stripe1Bytes, stripe0Bytes + stripe1Bytes, stripe2Bytes};
        return LaunchKernel(resource.threads[1], resource.ccuKernels[resource.gatherKernelIndex[1]],
            directArgs.data(), directArgs.size());
    }

    if (resource.useSmallNhr) {
        CHK_PRT_RET(!hasMeshKernel || !hasClosKernel,
            HCCL_ERROR("small NHR requires both Mesh and Clos kernels"), HCCL_E_INTERNAL);
        const std::array<uint64_t, 5> nhrArgs = {
            reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr), inputToken,
            outputToken, outputBytes};
        // The four rounds have a true data dependency.  The host notification
        // only orders the two CCUs; peer-facing step notifications stay inside
        // CcuNhrKernel and are consumed exactly once by their matching wait.
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[0], resource.threads[1], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[1], 0, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchKernel(resource.threads[1], resource.ccuKernels[resource.gatherKernelIndex[1]],
            nhrArgs.data(), nhrArgs.size()));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[1], resource.threads[0], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT)));
        return LaunchKernel(resource.threads[0], resource.ccuKernels[resource.gatherKernelIndex[0]],
            nhrArgs.data(), nhrArgs.size());
    }

    if (resource.useSmallGroupFanIn) {
        CHK_PRT_RET(!hasMeshKernel || !hasClosKernel || !IsKernelPresent(resource, resource.finalPairReduceKernelIndex),
            HCCL_ERROR("small group fan-in resource context is incomplete"), HCCL_E_INTERNAL);
        const GroupReduceLoopArgs loopArgs = MakeGroupReduceLoopArgs(outputBytes);
        const std::array<uint64_t, 12> groupArgs = {
            reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(resource.localBuffer.addr),
            inputToken, scratchToken, static_cast<uint64_t>(param.myRank) * outputBytes,
            0, outputBytes, slotBytes, loopArgs.fullOffset, loopArgs.fullLoopCount,
            loopArgs.parallelParam, loopArgs.residualBytes};
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[0], resource.threads[1], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[1], 0, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchKernel(resource.threads[1], resource.ccuKernels[resource.gatherKernelIndex[1]],
            groupArgs.data(), groupArgs.size()));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[1], resource.threads[0], 0)));
        CHK_RET(LaunchKernel(resource.threads[0], resource.ccuKernels[resource.gatherKernelIndex[0]],
            groupArgs.data(), groupArgs.size()));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT)));
        const std::array<uint64_t, 7> finalArgs = {
            reinterpret_cast<uint64_t>(param.outputPtr), reinterpret_cast<uint64_t>(resource.localBuffer.addr),
            outputToken, scratchToken, 0, outputBytes, slotBytes};
        return LaunchKernel(resource.threads[0], resource.ccuKernels[resource.finalPairReduceKernelIndex],
            finalArgs.data(), finalArgs.size());
    }

    if (resource.useCcuBufferGroupReduce) {
        CHK_PRT_RET(!hasMeshKernel || !hasClosKernel, HCCL_ERROR("incomplete group-reduce CCU resource context"),
            HCCL_E_INTERNAL);
        for (uint64_t chunkOffset = 0; chunkOffset < outputBytes;) {
            const uint64_t chunkBytes = std::min(maxChunkBytes, outputBytes - chunkOffset);
            const GroupReduceLoopArgs loopArgs = MakeGroupReduceLoopArgs(chunkBytes);
            const std::array<uint64_t, 12> meshArgs = {
                reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr) + chunkOffset,
                inputToken, outputToken, static_cast<uint64_t>(param.myRank) * outputBytes,
                chunkOffset, chunkBytes, 0, loopArgs.fullOffset, loopArgs.fullLoopCount,
                loopArgs.parallelParam, loopArgs.residualBytes};
            const std::array<uint64_t, 12> closArgs = {
                reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(resource.localBuffer.addr),
                inputToken, scratchToken, static_cast<uint64_t>(param.myRank) * outputBytes,
                chunkOffset, chunkBytes, 0, loopArgs.fullOffset, loopArgs.fullLoopCount,
                loopArgs.parallelParam, loopArgs.residualBytes};
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[0], resource.threads[1], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[1], 0, CUSTOM_TIMEOUT)));
            CHK_RET(LaunchKernel(resource.threads[1], resource.ccuKernels[resource.gatherKernelIndex[1]],
                closArgs.data(), closArgs.size()));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[1], resource.threads[0], 0)));
            CHK_RET(LaunchKernel(resource.threads[0], resource.ccuKernels[resource.gatherKernelIndex[0]],
                meshArgs.data(), meshArgs.size()));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resource.threads[0],
                reinterpret_cast<void *>(reinterpret_cast<uint64_t>(param.outputPtr) + chunkOffset),
                resource.localBuffer.addr, chunkBytes / elementSize,
                static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
            chunkOffset += chunkBytes;
        }
        return HCCL_SUCCESS;
    }

    const bool useHierarchicalTwoByEight = resource.useHierarchicalTwoByEight && outputBytes > 128 * 1024;
    if (useHierarchicalTwoByEight) {
        CHK_PRT_RET(!hasMeshKernel || !IsKernelPresent(resource, resource.closPairGatherKernelIndex) ||
                !IsKernelPresent(resource, resource.finalPairReduceKernelIndex),
            HCCL_ERROR("incomplete hierarchical 2x8 CCU resource context"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resource.threads.size() < 2, HCCL_ERROR("hierarchical 2x8 requires two CCU threads"),
            HCCL_E_INTERNAL);

        // The hierarchy needs eight local source slots plus one partial from
        // each server.  Ten compact slots fit a complete 32MiB output block
        // in the 400MiB CCL buffer, unlike the baseline's 16-rank layout.
        constexpr uint64_t hierarchicalSlotCount = 11;
        const uint64_t hierarchicalChunkCapacity = resource.localBuffer.size / hierarchicalSlotCount;
        const uint64_t hierarchicalMaxChunkBytes = std::min<uint64_t>(MAX_DATA_SIZE,
            hierarchicalChunkCapacity - hierarchicalChunkCapacity % elementSize);
        CHK_PRT_RET(hierarchicalMaxChunkBytes == 0, HCCL_ERROR("CCL buffer is too small for 2x8 hierarchy"),
            HCCL_E_INTERNAL);
        for (uint64_t chunkOffset = 0; chunkOffset < outputBytes;) {
            const uint64_t chunkBytes = std::min(hierarchicalMaxChunkBytes, outputBytes - chunkOffset);
            const std::array<uint64_t, 9> gatherArgs = {
                reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(resource.localBuffer.addr),
                inputToken, scratchToken, static_cast<uint64_t>(param.myRank) * outputBytes,
                chunkOffset, chunkBytes, chunkBytes, outputBytes};
            const std::array<uint64_t, 4> closPairArgs = {
                reinterpret_cast<uint64_t>(resource.localBuffer.addr), scratchToken, chunkBytes, chunkBytes};
            const std::array<uint64_t, 7> finalPairArgs = {
                reinterpret_cast<uint64_t>(param.outputPtr), reinterpret_cast<uint64_t>(resource.localBuffer.addr),
                outputToken, scratchToken, chunkOffset, chunkBytes, chunkBytes};

            CHK_RET(LaunchKernel(resource.threads[0], resource.ccuKernels[resource.gatherKernelIndex[0]],
                gatherArgs.data(), gatherArgs.size()));
            // The record is queued after the mesh gather and its in-kernel local tree on thread 0.
            // The Clos kernel cannot publish or read a partial sum before it.
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[0], resource.threads[1], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[1], 0, CUSTOM_TIMEOUT)));
            CHK_RET(LaunchKernel(resource.threads[1], resource.ccuKernels[resource.closPairGatherKernelIndex],
                closPairArgs.data(), closPairArgs.size()));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[1], resource.threads[0], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT)));
            CHK_RET(LaunchKernel(resource.threads[0], resource.ccuKernels[resource.finalPairReduceKernelIndex],
                finalPairArgs.data(), finalPairArgs.size()));
            chunkOffset += chunkBytes;
        }
        return HCCL_SUCCESS;
    }

    if (resource.useLayerPartialReduce && outputBytes > 128 * 1024) {
        CHK_PRT_RET(!IsKernelPresent(resource, resource.finalPairReduceKernelIndex),
            HCCL_ERROR("layer-partial final kernel is missing"), HCCL_E_INTERNAL);
        for (uint64_t chunkOffset = 0; chunkOffset < outputBytes;) {
            const uint64_t chunkBytes = std::min(maxChunkBytes, outputBytes - chunkOffset);
            const std::array<uint64_t, 8> gatherArgs = {
                reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(resource.localBuffer.addr), inputToken,
                scratchToken, static_cast<uint64_t>(param.myRank) * outputBytes, chunkOffset, chunkBytes, slotBytes};
            if (hasClosKernel) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[0], resource.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[1], 0, CUSTOM_TIMEOUT)));
                CHK_RET(LaunchKernel(resource.threads[1], resource.ccuKernels[resource.gatherKernelIndex[1]],
                    gatherArgs.data(), gatherArgs.size()));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[1], resource.threads[0], 0)));
            }
            if (hasMeshKernel) {
                CHK_RET(LaunchKernel(resource.threads[0], resource.ccuKernels[resource.gatherKernelIndex[0]],
                    gatherArgs.data(), gatherArgs.size()));
            }
            if (hasClosKernel) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT)));
            }
            const std::array<uint64_t, 7> finalArgs = {
                reinterpret_cast<uint64_t>(param.outputPtr), reinterpret_cast<uint64_t>(resource.localBuffer.addr),
                outputToken, scratchToken, chunkOffset, chunkBytes, slotBytes};
            CHK_RET(LaunchKernel(resource.threads[0], resource.ccuKernels[resource.finalPairReduceKernelIndex],
                finalArgs.data(), finalArgs.size()));
            chunkOffset += chunkBytes;
        }
        return HCCL_SUCCESS;
    }

    const bool useTwoChunkPipeline = param.rankSize == 16 && hasMeshKernel && hasClosKernel &&
        resource.finalReduceThreadIndex == 1 && outputBytes > slotBytes;
    if (useTwoChunkPipeline) {
        const uint64_t bufferBytes = resource.localBuffer.size / 2;
        const uint64_t bufferSlotBytes = bufferBytes / param.rankSize;
        const uint64_t pipelineChunkBytes = std::min<uint64_t>(MAX_DATA_SIZE,
            bufferSlotBytes - bufferSlotBytes % elementSize);
        CHK_PRT_RET(pipelineChunkBytes == 0, HCCL_ERROR("CCL buffer is too small for pipeline"), HCCL_E_INTERNAL);

        auto gatherArgsFor = [&](uint64_t chunkOffset, uint64_t chunkBytes, uint32_t bufferIndex) {
            return std::array<uint64_t, 8>{
                reinterpret_cast<uint64_t>(param.inputPtr),
                reinterpret_cast<uint64_t>(resource.localBuffer.addr) + bufferIndex * bufferBytes,
                inputToken, scratchToken, static_cast<uint64_t>(param.myRank) * outputBytes,
                chunkOffset, chunkBytes, bufferSlotBytes};
        };
        auto reduceArgsFor = [&](uint64_t chunkOffset, uint64_t chunkBytes, uint32_t bufferIndex) {
            return std::array<uint64_t, 11>{
                reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr),
                reinterpret_cast<uint64_t>(resource.localBuffer.addr) + bufferIndex * bufferBytes,
                inputToken, outputToken, scratchToken, static_cast<uint64_t>(param.myRank) * outputBytes,
                chunkOffset, chunkBytes, bufferSlotBytes, 0};
        };
        auto launchClos = [&](const std::array<uint64_t, 8> &args) -> HcclResult {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[0], resource.threads[1], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[1], 0, CUSTOM_TIMEOUT)));
            CHK_RET(LaunchKernel(resource.threads[1], resource.ccuKernels[resource.gatherKernelIndex[1]], args.data(), args.size()));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[1], resource.threads[0], 0)));
            return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT));
        };
        auto launchMesh = [&](const std::array<uint64_t, 8> &args) -> HcclResult {
            return LaunchKernel(resource.threads[0], resource.ccuKernels[resource.gatherKernelIndex[0]], args.data(), args.size());
        };
        auto recordMeshReady = [&]() -> HcclResult {
            return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[0], resource.threads[1], 0));
        };
        auto launchFinal = [&](const std::array<uint64_t, 11> &args) -> HcclResult {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[1], 0, CUSTOM_TIMEOUT)));
            CHK_RET(LaunchKernel(resource.threads[1], resource.ccuKernels[resource.finalReduceKernelIndex], args.data(), args.size()));
            return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[1], resource.threads[0], 0));
        };

        std::array<uint64_t, 2> offsets{};
        std::array<uint64_t, 2> sizes{};
        uint32_t preloadCount = 0;
        uint64_t nextOffset = 0;
        while (preloadCount < 2 && nextOffset < outputBytes) {
            const uint64_t bytes = std::min(pipelineChunkBytes, outputBytes - nextOffset);
            offsets[preloadCount] = nextOffset;
            sizes[preloadCount] = bytes;
            const auto args = gatherArgsFor(nextOffset, bytes, preloadCount);
            CHK_RET(launchClos(args));
            nextOffset += bytes;
            ++preloadCount;
        }

        const auto mesh0 = gatherArgsFor(offsets[0], sizes[0], 0);
        const auto reduce0 = reduceArgsFor(offsets[0], sizes[0], 0);
        CHK_RET(launchMesh(mesh0));
        CHK_RET(recordMeshReady());
        CHK_RET(launchFinal(reduce0));

        if (preloadCount == 2) {
            const auto mesh1 = gatherArgsFor(offsets[1], sizes[1], 1);
            const auto reduce1 = reduceArgsFor(offsets[1], sizes[1], 1);
            CHK_RET(launchMesh(mesh1));
            CHK_RET(recordMeshReady());
            // The first completion is consumed before the second final kernel
            // records on the same host-notify resource.
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT)));
            CHK_RET(launchFinal(reduce1));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT)));
        } else {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT)));
        }

        for (; nextOffset < outputBytes;) {
            const uint64_t bytes = std::min(pipelineChunkBytes, outputBytes - nextOffset);
            const auto gatherArgs = gatherArgsFor(nextOffset, bytes, 0);
            const auto reduceArgs = reduceArgsFor(nextOffset, bytes, 0);
            CHK_RET(launchClos(gatherArgs));
            CHK_RET(launchMesh(gatherArgs));
            CHK_RET(recordMeshReady());
            CHK_RET(launchFinal(reduceArgs));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT)));
            nextOffset += bytes;
        }
        return HCCL_SUCCESS;
    }

    for (uint64_t chunkOffset = 0; chunkOffset < outputBytes;) {
        const uint64_t chunkBytes = std::min(maxChunkBytes, outputBytes - chunkOffset);
        // Online data shows packed reduction hurts 16-rank 2x8 large messages.
        // Keep it isolated to the independently verified 12-rank candidate.
        const bool usePackedTree = param.rankSize == 12 && outputBytes > 128 * 1024;
        const uint64_t scratchStride = usePackedTree ? chunkBytes : slotBytes;
        const std::array<uint64_t, 8> gatherArgs = {
            reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(resource.localBuffer.addr), inputToken,
            scratchToken, static_cast<uint64_t>(param.myRank) * outputBytes, chunkOffset, chunkBytes, scratchStride};

        if (hasClosKernel) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[0], resource.threads[1], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[1], 0, CUSTOM_TIMEOUT)));
            CHK_RET(LaunchKernel(resource.threads[1], resource.ccuKernels[resource.gatherKernelIndex[1]],
                gatherArgs.data(), gatherArgs.size()));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[1], resource.threads[0], 0)));
        }
        if (hasMeshKernel) {
            CHK_RET(LaunchKernel(resource.threads[0], resource.ccuKernels[resource.gatherKernelIndex[0]],
                gatherArgs.data(), gatherArgs.size()));
        }
        if (hasClosKernel) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT)));
        }

        const std::array<uint64_t, 11> reduceArgs = {
            reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr),
            reinterpret_cast<uint64_t>(resource.localBuffer.addr), inputToken, outputToken, scratchToken,
            static_cast<uint64_t>(param.myRank) * outputBytes, chunkOffset, chunkBytes, scratchStride,
            usePackedTree ? 1ULL : 0ULL};
        const uint32_t finalThreadIndex = resource.finalReduceThreadIndex;
        if (finalThreadIndex != 0) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[0],
                resource.threads[finalThreadIndex], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[finalThreadIndex], 0,
                CUSTOM_TIMEOUT)));
        }
        CHK_RET(LaunchKernel(resource.threads[finalThreadIndex], resource.ccuKernels[resource.finalReduceKernelIndex],
            reduceArgs.data(), reduceArgs.size()));
        if (finalThreadIndex != 0) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resource.threads[finalThreadIndex],
                resource.threads[0], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resource.threads[0], 0, CUSTOM_TIMEOUT)));
        }
        chunkOffset += chunkBytes;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
