/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Host-side AllReduce orchestration.
//
// Messages up to 1 MiB use the aligned WriteReduce tree. Four-rank large
// messages use bidirectional recursive halving/doubling. Twelve-rank and
// sixteen-rank large messages use a dual-die direct sharded mesh.
//
// Each peer exchange is split into low-to-high and high-to-low launches. Only
// one side performs a data operation in each launch, and PairSync completes the
// direction before the reverse direction starts. This avoids simultaneous
// bidirectional WriteReduce on one channel.
//
// The sharded body is a multiple of shardCount * 32 bytes, so every slice offset
// and transfer size is 32-byte aligned. Any remaining bytes use the proven
// pull-reduce tree path.

#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>
#include <hcomm/hcomm_primitives.h>

#include <algorithm>
#include <limits>

#include "custom.h"
#include "ccu_kernel.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {

constexpr uint64_t WRITE_REDUCE_ALIGNMENT = 32;
constexpr uint64_t SMALL_MESSAGE_BYTES = 1024ULL * 1024ULL;
constexpr uint64_t FAST_MESH_MIN_BYTES = 512ULL * 1024ULL;
constexpr uint64_t LARGE_MESSAGE_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t LARGE_MESSAGE_CHUNK_BYTES = 32ULL * 1024ULL * 1024ULL;

bool IsPowerOfTwo(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

bool UseShardedAlgorithm(uint32_t rankSize, uint64_t totalSize)
{
    const bool usePowerOfTwoRhd =
        (rankSize == 4 || rankSize == 16) && IsPowerOfTwo(rankSize);
    return (rankSize == 12 || usePowerOfTwoRhd) &&
        totalSize > SMALL_MESSAGE_BYTES;
}

bool UseFastMeshAlgorithm(uint32_t rankSize, uint64_t totalSize,
    const void *input, const void *output)
{
    const bool useRank12SmallMesh = rankSize == 12 &&
        totalSize >= FAST_MESH_MIN_BYTES && totalSize <= SMALL_MESSAGE_BYTES;
    const bool useRank16SmallMesh = rankSize == 16 &&
        totalSize >= FAST_MESH_MIN_BYTES && totalSize <= SMALL_MESSAGE_BYTES;
    return (useRank12SmallMesh || useRank16SmallMesh) && input != output;
}

bool UseHierarchicalAlgorithm(uint32_t rankSize, uint64_t totalSize)
{
    return (rankSize == 12 || rankSize == 16) &&
        totalSize > SMALL_MESSAGE_BYTES;
}

uint64_t AlignedBodySize(uint64_t totalSize, uint32_t shardCount)
{
    const uint64_t alignment = static_cast<uint64_t>(shardCount) * WRITE_REDUCE_ALIGNMENT;
    return totalSize - totalSize % alignment;
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    const auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type: %d", static_cast<int>(param.dataType)),
        HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = sizeIter->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("Input size overflow"), HCCL_E_PARA);
    const uint64_t totalSize = param.count * typeSize;
    if (totalSize == 0) {
        return HCCL_SUCCESS;
    }

    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, totalSize));
    }

    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < typeSize,
        HCCL_ERROR("HCCL scratch buffer is too small: %llu",
            static_cast<unsigned long long>(resCtx.localBuffer.size)), HCCL_E_INTERNAL);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.inputPtr), totalSize, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.outputPtr), totalSize, &outputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
        resCtx.localBuffer.size, &scratchToken));

    const uint64_t scratchAddress = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    const bool outOfPlace = inputBase != outputBase;

    if (UseFastMeshAlgorithm(param.rankSize, totalSize, param.inputPtr, param.outputPtr)) {
        CHK_PRT_RET(resCtx.ccuKernels.size() != 4 || resCtx.threads.size() != 2,
            HCCL_ERROR("Invalid fast-mesh resources: kernels=%zu threads=%zu",
                resCtx.ccuKernels.size(), resCtx.threads.size()), HCCL_E_INTERNAL);
        const bool secondaryActive =
            resCtx.ccuKernels[1] != 0 && resCtx.ccuKernels[3] != 0;
        CHK_PRT_RET((resCtx.ccuKernels[1] == 0) !=
                (resCtx.ccuKernels[3] == 0),
            HCCL_ERROR("Inconsistent secondary fast-mesh resources"),
            HCCL_E_INTERNAL);
        const uint32_t scratchLaneCount = param.rankSize - 1;
        uint64_t chunkSize = resCtx.localBuffer.size / scratchLaneCount;
        chunkSize -= chunkSize % WRITE_REDUCE_ALIGNMENT;
        CHK_PRT_RET(chunkSize < typeSize ||
                chunkSize * scratchLaneCount > resCtx.localBuffer.size,
            HCCL_ERROR("Invalid fast-mesh chunk size: %llu",
                static_cast<unsigned long long>(chunkSize)), HCCL_E_INTERNAL);

        const uint64_t bootstrapArgs[] = {
            inputBase,
            inputToken,
            outputBase,
            outputToken,
        };
        if (secondaryActive) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                param.cpuThread, resCtx.threads[1], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, resCtx.ccuKernels[0],
            bootstrapArgs, 4));
        if (secondaryActive) {
            CHK_RET_CCU(HcommCcuKernelLaunch(
                resCtx.threads[1], resCtx.ccuKernels[1], bootstrapArgs, 4));
        }

        uint64_t processedSize = 0;
        while (processedSize < totalSize) {
            const uint64_t operationSize =
                std::min<uint64_t>(chunkSize, totalSize - processedSize);
            std::vector<uint64_t> meshArgs = {
                inputBase + processedSize,
                outputBase + processedSize,
                inputToken,
                outputToken,
                scratchAddress,
                scratchToken,
                chunkSize,
                0,
                operationSize,
                processedSize,
            };

            CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, resCtx.ccuKernels[2],
                meshArgs.data(), static_cast<uint32_t>(meshArgs.size())));
            if (secondaryActive) {
                CHK_RET_CCU(HcommCcuKernelLaunch(
                    resCtx.threads[1], resCtx.ccuKernels[3],
                    meshArgs.data(), static_cast<uint32_t>(meshArgs.size())));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resCtx.threads[1], param.cpuThread, 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    param.cpuThread, 0, CUSTOM_TIMEOUT)));

                meshArgs[7] = 1;
                CHK_RET_CCU(HcommCcuKernelLaunch(
                    param.cpuThread, resCtx.ccuKernels[2],
                    meshArgs.data(), static_cast<uint32_t>(meshArgs.size())));
            }

            processedSize += operationSize;
            if (secondaryActive && processedSize < totalSize) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    param.cpuThread, resCtx.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            }
        }
        return HCCL_SUCCESS;
    }

    const bool useHierarchicalAlgorithm =
        UseHierarchicalAlgorithm(param.rankSize, totalSize);
    constexpr uint32_t HIERARCHICAL_MESH_KERNEL_COUNT = 4;
    const uint32_t pairKernelOffset =
        useHierarchicalAlgorithm ? HIERARCHICAL_MESH_KERNEL_COUNT : 0;
    const size_t expectedKernelCount =
        static_cast<size_t>(pairKernelOffset + param.rankSize);
    CHK_PRT_RET(resCtx.ccuKernels.size() != expectedKernelCount,
        HCCL_ERROR("Unexpected CCU kernel table size: %zu", resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);
    const size_t expectedHierarchicalThreadCount =
        param.rankSize == 16 ? 3 : 2;
    CHK_PRT_RET(useHierarchicalAlgorithm &&
            resCtx.threads.size() != expectedHierarchicalThreadCount,
        HCCL_ERROR("Unexpected hierarchical thread count: %zu",
            resCtx.threads.size()),
        HCCL_E_INTERNAL);
    const bool hierarchicalSecondaryActive = useHierarchicalAlgorithm &&
        resCtx.ccuKernels[1] != 0 && resCtx.ccuKernels[3] != 0;
    CHK_PRT_RET(useHierarchicalAlgorithm &&
            ((resCtx.ccuKernels[1] == 0) != (resCtx.ccuKernels[3] == 0)),
        HCCL_ERROR("Inconsistent secondary hierarchical-mesh resources"),
        HCCL_E_INTERNAL);

    std::vector<uint64_t> taskArgs = {
        inputBase,
        outputBase,
        inputToken,
        outputToken,
        scratchAddress,
        scratchToken,
        totalSize,
        0,
        0,
        0,
        0,
        0,
    };

    auto launchPairOnThread = [&](ThreadHandle thread, uint32_t peerRank,
                                  PairPhase phase) -> HcclResult {
        CHK_PRT_RET(peerRank >= param.rankSize ||
                resCtx.ccuKernels[pairKernelOffset + peerRank] == 0,
            HCCL_ERROR("Missing pair kernel from rank %u to rank %u", param.myRank, peerRank),
            HCCL_E_INTERNAL);
        taskArgs[7] = static_cast<uint64_t>(phase);
        CHK_RET_CCU(HcommCcuKernelLaunch(thread,
            resCtx.ccuKernels[pairKernelOffset + peerRank],
            taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
        return HCCL_SUCCESS;
    };
    auto launchPair = [&](uint32_t peerRank, PairPhase phase) -> HcclResult {
        return launchPairOnThread(param.cpuThread, peerRank, phase);
    };

    auto executeTreeRange = [&](uint64_t rangeOffset, uint64_t rangeSize) -> HcclResult {
        const uint64_t scratchChunkSize = std::min<uint64_t>(MAX_DATA_SIZE, resCtx.localBuffer.size);
        CHK_PRT_RET(scratchChunkSize < typeSize,
            HCCL_ERROR("No usable CCU chunk size"), HCCL_E_INTERNAL);

        uint32_t highestStride = 1;
        while ((highestStride << 1) < param.rankSize) {
            highestStride <<= 1;
        }

        uint64_t processedSize = 0;
        while (processedSize < rangeSize) {
            uint64_t chunkSize = std::min<uint64_t>(MAX_DATA_SIZE, rangeSize - processedSize);
            chunkSize -= chunkSize % typeSize;
            CHK_PRT_RET(chunkSize == 0,
                HCCL_ERROR("Invalid CCU chunk size"), HCCL_E_INTERNAL);

            bool useWriteReduce = false;
            if (chunkSize >= WRITE_REDUCE_ALIGNMENT) {
                const uint64_t alignedSize = chunkSize - chunkSize % WRITE_REDUCE_ALIGNMENT;
                if (alignedSize != 0) {
                    chunkSize = alignedSize;
                    useWriteReduce = true;
                }
            }
            if (!useWriteReduce) {
                chunkSize = std::min(scratchChunkSize, rangeSize - processedSize);
                chunkSize -= chunkSize % typeSize;
                CHK_PRT_RET(chunkSize == 0,
                    HCCL_ERROR("Invalid scratch-backed chunk size"), HCCL_E_INTERNAL);
            }

            taskArgs[0] = inputBase + rangeOffset + processedSize;
            taskArgs[1] = outputBase + rangeOffset + processedSize;
            taskArgs[6] = chunkSize;
            taskArgs[8] = 0;
            taskArgs[9] = 0;
            taskArgs[10] = 0;
            taskArgs[11] = 0;

            bool initialized = false;
            const bool chunkOutOfPlace = taskArgs[0] != taskArgs[1];
            for (uint32_t stride = 1; stride < param.rankSize; stride <<= 1) {
                const uint32_t position = param.myRank % (stride << 1);
                if (position == 0) {
                    const uint32_t child = param.myRank + stride;
                    if (child < param.rankSize) {
                        PairPhase phase;
                        if (useWriteReduce) {
                            phase = (!initialized && chunkOutOfPlace)
                                ? PairPhase::REDUCE_WR_RECV_INIT
                                : PairPhase::REDUCE_WR_RECV;
                        } else {
                            phase = (!initialized && chunkOutOfPlace)
                                ? PairPhase::REDUCE_RECEIVE_INIT
                                : PairPhase::REDUCE_RECEIVE;
                        }
                        CHK_RET(launchPair(child, phase));
                        initialized = true;
                    }
                } else if (position == stride) {
                    PairPhase phase;
                    if (useWriteReduce) {
                        phase = (!initialized && chunkOutOfPlace)
                            ? PairPhase::REDUCE_WR_SEND_INIT
                            : PairPhase::REDUCE_WR_SEND;
                    } else {
                        phase = (!initialized && chunkOutOfPlace)
                            ? PairPhase::REDUCE_SEND_INIT
                            : PairPhase::REDUCE_SEND;
                    }
                    CHK_RET(launchPair(param.myRank - stride, phase));
                    initialized = true;
                    break;
                }
            }

            CHK_PRT_RET(!initialized,
                HCCL_ERROR("Rank %u did not enter the reduction tree", param.myRank),
                HCCL_E_INTERNAL);

            for (uint32_t stride = highestStride;; stride >>= 1) {
                const uint32_t position = param.myRank % (stride << 1);
                if (position == 0) {
                    const uint32_t child = param.myRank + stride;
                    if (child < param.rankSize) {
                        CHK_RET(launchPair(child, PairPhase::BROADCAST_SEND));
                    }
                } else if (position == stride) {
                    CHK_RET(launchPair(param.myRank - stride, PairPhase::BROADCAST_RECEIVE));
                }
                if (stride == 1) {
                    break;
                }
            }

            processedSize += chunkSize;
        }
        return HCCL_SUCCESS;
    };

    if (!UseShardedAlgorithm(param.rankSize, totalSize)) {
        return executeTreeRange(0, totalSize);
    }

    const uint32_t shardCount = param.rankSize == 12 ? 8 : param.rankSize;
    const uint64_t bodySize = AlignedBodySize(totalSize, shardCount);
    CHK_PRT_RET(bodySize == 0 || bodySize % shardCount != 0,
        HCCL_ERROR("Invalid sharded body size"), HCCL_E_INTERNAL);
    const uint64_t sliceSize = bodySize / shardCount;
    CHK_PRT_RET(sliceSize % WRITE_REDUCE_ALIGNMENT != 0,
        HCCL_ERROR("Unaligned sharded slice size"), HCCL_E_INTERNAL);

    taskArgs[0] = inputBase;
    taskArgs[1] = outputBase;
    taskArgs[6] = bodySize;

    auto setRange = [&](uint32_t begin, uint32_t end) {
        taskArgs[8] = static_cast<uint64_t>(begin) * sliceSize;
        taskArgs[9] = static_cast<uint64_t>(end - begin) * sliceSize;
    };

    auto setCopyRange = [&](uint32_t begin, uint32_t end) {
        taskArgs[10] = static_cast<uint64_t>(begin) * sliceSize;
        taskArgs[11] = static_cast<uint64_t>(end - begin) * sliceSize;
    };

    if (useHierarchicalAlgorithm) {
        constexpr uint32_t LARGE_GROUP_SIZE = 8;
        constexpr uint32_t SMALL_GROUP_SIZE = 4;
        std::vector<uint32_t> localRanks = resCtx.localServerRanks;
        CHK_PRT_RET(localRanks.size() != LARGE_GROUP_SIZE &&
                localRanks.size() != SMALL_GROUP_SIZE,
            HCCL_ERROR("Expected a four-rank or eight-rank local group, found %zu",
                localRanks.size()), HCCL_E_INTERNAL);
        std::sort(localRanks.begin(), localRanks.end());
        const auto localIter =
            std::find(localRanks.begin(), localRanks.end(), param.myRank);
        CHK_PRT_RET(localIter == localRanks.end(),
            HCCL_ERROR("Local rank is missing from the local topology order"),
            HCCL_E_INTERNAL);
        const uint32_t localRank =
            static_cast<uint32_t>(localIter - localRanks.begin());

        std::vector<bool> localRankSeen(param.rankSize, false);
        for (const uint32_t rank : localRanks) {
            CHK_PRT_RET(rank >= param.rankSize || localRankSeen[rank],
                HCCL_ERROR("Invalid rank in local topology order: %u", rank),
                HCCL_E_INTERNAL);
            localRankSeen[rank] = true;
        }
        std::vector<uint32_t> remoteRanks;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (!localRankSeen[rank]) {
                remoteRanks.push_back(rank);
            }
        }

        CHK_PRT_RET(shardCount % static_cast<uint32_t>(localRanks.size()) != 0,
            HCCL_ERROR("Shard count %u is not divisible by local group size %zu",
                shardCount, localRanks.size()),
            HCCL_E_INTERNAL);
        const uint32_t ownedShardCount =
            shardCount / static_cast<uint32_t>(localRanks.size());
        const uint32_t ownedBegin = localRank * ownedShardCount;
        const uint32_t ownedEnd = ownedBegin + ownedShardCount;
        CHK_PRT_RET(ownedEnd > shardCount,
            HCCL_ERROR("Owned shard range [%u, %u) exceeds shard count %u",
                ownedBegin, ownedEnd, shardCount),
            HCCL_E_INTERNAL);
        const uint64_t ownedOffset =
            static_cast<uint64_t>(ownedBegin) * sliceSize;
        const uint64_t ownedSize =
            static_cast<uint64_t>(ownedShardCount) * sliceSize;
        const uint32_t scratchLaneCount =
            static_cast<uint32_t>(localRanks.size()) - 1;
        uint64_t scratchStride = resCtx.localBuffer.size / scratchLaneCount;
        scratchStride -= scratchStride % WRITE_REDUCE_ALIGNMENT;
        uint64_t operationLimit =
            resCtx.localBuffer.size / (LARGE_GROUP_SIZE - 1);
        operationLimit -= operationLimit % WRITE_REDUCE_ALIGNMENT;
        if (totalSize == LARGE_MESSAGE_BYTES) {
            operationLimit =
                std::min(operationLimit, LARGE_MESSAGE_CHUNK_BYTES);
        }
        CHK_PRT_RET(ownedSize == 0 ||
                scratchStride < WRITE_REDUCE_ALIGNMENT ||
                operationLimit < WRITE_REDUCE_ALIGNMENT,
            HCCL_ERROR(
                "Invalid owned mesh range %llu, scratch stride %llu, or operation limit %llu",
                static_cast<unsigned long long>(ownedSize),
                static_cast<unsigned long long>(scratchStride),
                static_cast<unsigned long long>(operationLimit)),
            HCCL_E_INTERNAL);

        const uint64_t bootstrapArgs[] = {
            inputBase,
            inputToken,
            outputBase,
            outputToken,
        };
        if (hierarchicalSecondaryActive) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                param.cpuThread, resCtx.threads[1], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
        }
        if (param.rankSize == 16) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                param.cpuThread, resCtx.threads[2], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[2], 0, CUSTOM_TIMEOUT)));
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, resCtx.ccuKernels[0],
            bootstrapArgs, 4));
        if (hierarchicalSecondaryActive) {
            CHK_RET_CCU(HcommCcuKernelLaunch(
                resCtx.threads[1], resCtx.ccuKernels[1], bootstrapArgs, 4));
        }

        struct HierarchicalSegment {
            uint64_t offset;
            uint64_t size;
            uint32_t peer;
            bool receiveReduction;
        };
        std::vector<HierarchicalSegment> segments;

        if (param.rankSize == 16) {
            CHK_PRT_RET(remoteRanks.size() != LARGE_GROUP_SIZE,
                HCCL_ERROR("Invalid remote group size: %zu", remoteRanks.size()),
                HCCL_E_INTERNAL);
            const uint32_t crossPeer = remoteRanks[localRank];
            segments.push_back({
                ownedOffset,
                ownedSize,
                crossPeer,
                param.myRank < crossPeer,
            });
        } else {
            const bool inLargeGroup = localRanks.size() == LARGE_GROUP_SIZE;
            const std::vector<uint32_t> &largeRanks =
                inLargeGroup ? localRanks : remoteRanks;
            const std::vector<uint32_t> &smallRanks =
                inLargeGroup ? remoteRanks : localRanks;
            CHK_PRT_RET(largeRanks.size() != LARGE_GROUP_SIZE ||
                    smallRanks.size() != SMALL_GROUP_SIZE,
                HCCL_ERROR("Invalid asymmetric group sizes: %zu/%zu",
                    largeRanks.size(), smallRanks.size()),
                HCCL_E_INTERNAL);
            if (inLargeGroup) {
                segments.push_back({
                    ownedOffset,
                    ownedSize,
                    smallRanks[localRank / 2],
                    true,
                });
            } else {
                for (uint32_t shardOffset = 0;
                     shardOffset < ownedShardCount; ++shardOffset) {
                    const uint32_t shard = ownedBegin + shardOffset;
                    segments.push_back({
                        static_cast<uint64_t>(shard) * sliceSize,
                        sliceSize,
                        largeRanks[shard],
                        false,
                    });
                }
            }
        }

        CHK_PRT_RET(segments.empty(),
            HCCL_ERROR("Hierarchical segment table is empty"), HCCL_E_INTERNAL);
        uint64_t localMeshSize = 0;
        for (const HierarchicalSegment &segment : segments) {
            CHK_PRT_RET(segment.offset != ownedOffset + localMeshSize,
                HCCL_ERROR("Non-contiguous hierarchical segment at offset %llu",
                    static_cast<unsigned long long>(segment.offset)),
                HCCL_E_INTERNAL);
            localMeshSize += segment.size;
        }
        CHK_PRT_RET(localMeshSize != ownedSize,
            HCCL_ERROR("Hierarchical segments cover %llu bytes, expected %llu",
                static_cast<unsigned long long>(localMeshSize),
                static_cast<unsigned long long>(ownedSize)),
            HCCL_E_INTERNAL);

        auto launchLocalReduceMode = [&](uint64_t rangeOffset,
                                         uint64_t rangeSize,
                                         uint64_t operationMode) -> HcclResult {
            uint64_t processedSize = 0;
            while (processedSize < rangeSize) {
                const uint64_t operationSize =
                    std::min(scratchStride, rangeSize - processedSize);
                const uint64_t operationOffset = rangeOffset + processedSize;
                CHK_PRT_RET(operationSize == 0 ||
                        operationSize % WRITE_REDUCE_ALIGNMENT != 0,
                    HCCL_ERROR("Invalid local-reduce chunk size: %llu",
                        static_cast<unsigned long long>(operationSize)),
                    HCCL_E_INTERNAL);
                std::vector<uint64_t> meshArgs = {
                    operationMode == 3
                        ? outputBase + operationOffset
                        : inputBase + operationOffset,
                    outputBase + operationOffset,
                    operationMode == 3 ? outputToken : inputToken,
                    outputToken,
                    scratchAddress,
                    scratchToken,
                    scratchStride,
                    operationMode,
                    operationSize,
                    operationOffset,
                };
                CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
                    resCtx.ccuKernels[2], meshArgs.data(),
                    static_cast<uint32_t>(meshArgs.size())));
                if (hierarchicalSecondaryActive) {
                    CHK_RET_CCU(HcommCcuKernelLaunch(
                        resCtx.threads[1], resCtx.ccuKernels[3],
                        meshArgs.data(), static_cast<uint32_t>(meshArgs.size())));
                    CHK_RET(static_cast<HcclResult>(
                        HcommThreadNotifyRecordOnThread(
                            resCtx.threads[1], param.cpuThread, 0)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommThreadNotifyWaitOnThread(
                            param.cpuThread, 0, CUSTOM_TIMEOUT)));
                    meshArgs[7] = 1;
                    CHK_RET_CCU(HcommCcuKernelLaunch(
                        param.cpuThread, resCtx.ccuKernels[2],
                        meshArgs.data(), static_cast<uint32_t>(meshArgs.size())));
                }
                processedSize += operationSize;
            }
            return HCCL_SUCCESS;
        };
        auto launchLocalReduce = [&](uint64_t rangeOffset,
                                     uint64_t rangeSize) -> HcclResult {
            return launchLocalReduceMode(rangeOffset, rangeSize, 0);
        };
        auto launchLocalReduceInPlace = [&](uint64_t rangeOffset,
                                            uint64_t rangeSize) -> HcclResult {
            return launchLocalReduceMode(rangeOffset, rangeSize, 3);
        };

        auto launchLocalBroadcast = [&](uint64_t rangeOffset,
                                        uint64_t rangeSize) -> HcclResult {
            uint64_t processedSize = 0;
            while (processedSize < rangeSize) {
                const uint64_t operationSize =
                    std::min(scratchStride, rangeSize - processedSize);
                const uint64_t operationOffset = rangeOffset + processedSize;
                CHK_PRT_RET(operationSize == 0 ||
                        operationSize % WRITE_REDUCE_ALIGNMENT != 0,
                    HCCL_ERROR("Invalid local-broadcast chunk size: %llu",
                        static_cast<unsigned long long>(operationSize)),
                    HCCL_E_INTERNAL);
                if (hierarchicalSecondaryActive) {
                    CHK_RET(static_cast<HcclResult>(
                        HcommThreadNotifyRecordOnThread(
                            param.cpuThread, resCtx.threads[1], 0)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommThreadNotifyWaitOnThread(
                            resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
                }
                std::vector<uint64_t> meshArgs = {
                    outputBase + operationOffset,
                    outputBase + operationOffset,
                    inputToken,
                    outputToken,
                    scratchAddress,
                    scratchToken,
                    scratchStride,
                    2,
                    operationSize,
                    operationOffset,
                };
                CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
                    resCtx.ccuKernels[2], meshArgs.data(),
                    static_cast<uint32_t>(meshArgs.size())));
                if (hierarchicalSecondaryActive) {
                    CHK_RET_CCU(HcommCcuKernelLaunch(
                        resCtx.threads[1], resCtx.ccuKernels[3],
                        meshArgs.data(), static_cast<uint32_t>(meshArgs.size())));
                    CHK_RET(static_cast<HcclResult>(
                        HcommThreadNotifyRecordOnThread(
                            resCtx.threads[1], param.cpuThread, 0)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommThreadNotifyWaitOnThread(
                            param.cpuThread, 0, CUSTOM_TIMEOUT)));
                }
                processedSize += operationSize;
            }
            return HCCL_SUCCESS;
        };

        if (param.rankSize == 16) {
            constexpr uint64_t PARALLEL_SPLIT_DENOMINATOR = 3;
            constexpr uint64_t PARALLEL_MESH_PARTS = 1;
            const uint64_t splitAlignment =
                static_cast<uint64_t>(param.rankSize) *
                WRITE_REDUCE_ALIGNMENT;
            uint64_t meshFirstSize =
                bodySize * PARALLEL_MESH_PARTS /
                PARALLEL_SPLIT_DENOMINATOR;
            meshFirstSize -= meshFirstSize % splitAlignment;
            const uint64_t closFirstSize = bodySize - meshFirstSize;
            CHK_PRT_RET(meshFirstSize == 0 || closFirstSize == 0 ||
                    meshFirstSize % splitAlignment != 0 ||
                    closFirstSize % splitAlignment != 0,
                HCCL_ERROR("Invalid parallel split: %llu/%llu",
                    static_cast<unsigned long long>(meshFirstSize),
                    static_cast<unsigned long long>(closFirstSize)),
                HCCL_E_INTERNAL);

            const uint32_t crossPeer = remoteRanks[localRank];
            const bool lowerServer = param.myRank < crossPeer;
            const uint64_t meshOwnedSize =
                meshFirstSize / LARGE_GROUP_SIZE;
            const uint64_t meshOwnedOffset =
                static_cast<uint64_t>(localRank) * meshOwnedSize;
            const uint64_t meshCrossSize = meshOwnedSize / 2;
            const uint64_t meshKeepOffset = meshOwnedOffset +
                (lowerServer ? 0 : meshCrossSize);
            const uint64_t meshSendOffset = meshOwnedOffset +
                (lowerServer ? meshCrossSize : 0);

            const uint64_t closServerSize = closFirstSize / 2;
            const uint64_t closKeepOffset = meshFirstSize +
                (lowerServer ? 0 : closServerSize);
            const uint64_t closSendOffset = meshFirstSize +
                (lowerServer ? closServerSize : 0);
            const uint64_t closOwnedSize =
                closServerSize / LARGE_GROUP_SIZE;
            const uint64_t closOwnedOffset = closKeepOffset +
                static_cast<uint64_t>(localRank) * closOwnedSize;
            CHK_PRT_RET(meshCrossSize == 0 || closOwnedSize == 0 ||
                    meshCrossSize % WRITE_REDUCE_ALIGNMENT != 0 ||
                    closOwnedSize % WRITE_REDUCE_ALIGNMENT != 0,
                HCCL_ERROR("Invalid parallel shard sizes: %llu/%llu",
                    static_cast<unsigned long long>(meshCrossSize),
                    static_cast<unsigned long long>(closOwnedSize)),
                HCCL_E_INTERNAL);

            const ThreadHandle closThread = resCtx.threads[2];
            auto startClosThread = [&]() -> HcclResult {
                constexpr uint32_t CLOS_START_NOTIFY_INDEX = 0;
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(
                        param.cpuThread, closThread,
                        CLOS_START_NOTIFY_INDEX)));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(
                        closThread, CLOS_START_NOTIFY_INDEX,
                        CUSTOM_TIMEOUT)));
                return HCCL_SUCCESS;
            };
            auto joinClosThread = [&]() -> HcclResult {
                constexpr uint32_t CLOS_JOIN_NOTIFY_INDEX = 1;
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(
                        closThread, param.cpuThread,
                        CLOS_JOIN_NOTIFY_INDEX)));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(
                        param.cpuThread, CLOS_JOIN_NOTIFY_INDEX,
                        CUSTOM_TIMEOUT)));
                return HCCL_SUCCESS;
            };
            auto launchLocalGroupBarrier = [&]() -> HcclResult {
                taskArgs[8] = 0;
                taskArgs[9] = 0;
                taskArgs[10] = 0;
                taskArgs[11] = 0;
                const uint32_t localGroupSize =
                    static_cast<uint32_t>(localRanks.size());
                for (uint32_t distance = 1;
                     distance < localGroupSize;
                     distance <<= 1) {
                    const uint32_t peerLocalRank =
                        localRank ^ distance;
                    CHK_RET(launchPair(
                        localRanks[peerLocalRank],
                        PairPhase::BROADCAST_RECEIVE));
                }
                return HCCL_SUCCESS;
            };

            auto launchCrossRange = [&](uint64_t sendOffset,
                                        uint64_t sendSize,
                                        uint64_t keepOffset,
                                        PairPhase phase,
                                        bool initializeKeep) -> HcclResult {
                uint64_t processedSize = 0;
                while (processedSize < sendSize) {
                    const uint64_t operationSize =
                        std::min(operationLimit,
                            sendSize - processedSize);
                    CHK_PRT_RET(operationSize == 0 ||
                            operationSize %
                                WRITE_REDUCE_ALIGNMENT != 0,
                        HCCL_ERROR(
                            "Invalid parallel cross chunk size: %llu",
                            static_cast<unsigned long long>(
                                operationSize)),
                        HCCL_E_INTERNAL);
                    taskArgs[8] = sendOffset + processedSize;
                    taskArgs[9] = operationSize;
                    if (initializeKeep) {
                        taskArgs[10] = keepOffset + processedSize;
                        taskArgs[11] = operationSize;
                    } else {
                        taskArgs[10] = 0;
                        taskArgs[11] = 0;
                    }
                    CHK_RET(launchPairOnThread(
                        closThread, crossPeer, phase));
                    processedSize += operationSize;
                }
                return HCCL_SUCCESS;
            };

            CHK_RET(startClosThread());
            CHK_RET(launchLocalReduce(
                meshOwnedOffset, meshOwnedSize));
            CHK_RET(launchCrossRange(
                closSendOffset, closServerSize, closKeepOffset,
                PairPhase::RHD_REDUCE_BIDIR_FIRST, true));
            CHK_RET(joinClosThread());
            CHK_RET(launchLocalGroupBarrier());

            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(
                    param.cpuThread, resCtx.threads[1], 0)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(
                    resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            CHK_RET(startClosThread());
            CHK_RET(launchLocalReduceInPlace(
                closOwnedOffset, closOwnedSize));
            CHK_RET(launchCrossRange(
                meshSendOffset, meshCrossSize, 0,
                PairPhase::RHD_REDUCE_BIDIR, false));
            CHK_RET(joinClosThread());
            CHK_RET(launchLocalGroupBarrier());

            CHK_RET(startClosThread());
            CHK_RET(launchLocalBroadcast(
                closOwnedOffset, closOwnedSize));
            CHK_RET(launchCrossRange(
                meshKeepOffset, meshCrossSize, 0,
                PairPhase::RHD_GATHER_BIDIR, false));
            CHK_RET(joinClosThread());
            CHK_RET(launchLocalGroupBarrier());

            CHK_RET(startClosThread());
            CHK_RET(launchLocalBroadcast(
                meshOwnedOffset, meshOwnedSize));
            CHK_RET(launchCrossRange(
                closKeepOffset, closServerSize, 0,
                PairPhase::RHD_GATHER_BIDIR, false));
            CHK_RET(joinClosThread());
            CHK_RET(launchLocalGroupBarrier());

            const uint64_t tailSize = totalSize - bodySize;
            if (tailSize != 0) {
                CHK_RET(executeTreeRange(bodySize, tailSize));
            }
            return HCCL_SUCCESS;
        }

        CHK_RET(launchLocalReduce(ownedOffset, localMeshSize));

        for (const HierarchicalSegment &segment : segments) {
            uint64_t processedSegmentSize = 0;
            while (processedSegmentSize < segment.size) {
                const uint64_t operationSize =
                    std::min(operationLimit,
                        segment.size - processedSegmentSize);
                const uint64_t operationOffset =
                    segment.offset + processedSegmentSize;
                CHK_PRT_RET(operationSize == 0 ||
                        operationSize % WRITE_REDUCE_ALIGNMENT != 0,
                    HCCL_ERROR("Invalid cross-server chunk size: %llu",
                        static_cast<unsigned long long>(operationSize)),
                    HCCL_E_INTERNAL);
                taskArgs[8] = operationOffset;
                taskArgs[9] = operationSize;
                taskArgs[10] = 0;
                taskArgs[11] = 0;
                CHK_RET(launchPair(
                    segment.peer, PairPhase::RHD_REDUCE_BIDIR_SCRATCH));
                processedSegmentSize += operationSize;
            }
        }

        CHK_RET(launchLocalBroadcast(ownedOffset, localMeshSize));

        const uint64_t tailSize = totalSize - bodySize;
        if (tailSize != 0) {
            CHK_RET(executeTreeRange(bodySize, tailSize));
        }
        return HCCL_SUCCESS;
    }

    if (param.rankSize == 4 || param.rankSize == 16) {
        uint32_t groupBegin = 0;
        uint32_t groupEnd = shardCount;
        bool firstLevel = true;
        for (uint32_t half = shardCount >> 1; half != 0; half >>= 1) {
            const uint32_t middle = groupBegin + half;
            const bool inLowerHalf = param.myRank < middle;
            const uint32_t peer = param.myRank ^ half;
            const uint32_t keepBegin = inLowerHalf ? groupBegin : middle;
            const uint32_t keepEnd = inLowerHalf ? middle : groupEnd;
            const uint32_t sendBegin = inLowerHalf ? middle : groupBegin;
            const uint32_t sendEnd = inLowerHalf ? groupEnd : middle;

            setRange(sendBegin, sendEnd);
            taskArgs[10] = 0;
            taskArgs[11] = 0;
            if (firstLevel && outOfPlace) {
                setCopyRange(keepBegin, keepEnd);
            }
            CHK_RET(launchPair(peer, firstLevel
                ? PairPhase::RHD_REDUCE_BIDIR_FIRST
                : PairPhase::RHD_REDUCE_BIDIR));

            groupBegin = keepBegin;
            groupEnd = keepEnd;
            firstLevel = false;
        }

        uint32_t ownedBegin = param.myRank;
        uint32_t ownedEnd = param.myRank + 1;
        for (uint32_t half = 1; half < shardCount; half <<= 1) {
            setRange(ownedBegin, ownedEnd);
            taskArgs[10] = 0;
            taskArgs[11] = 0;
            CHK_RET(launchPair(param.myRank ^ half, PairPhase::RHD_GATHER_BIDIR));
            if ((param.myRank & half) == 0) {
                ownedEnd += half;
            } else {
                ownedBegin -= half;
            }
        }
    } else {
        constexpr uint32_t LARGE_GROUP_SIZE = 8;
        constexpr uint32_t SMALL_GROUP_SIZE = 4;
        std::vector<uint32_t> localRanks = resCtx.localServerRanks;
        CHK_PRT_RET(localRanks.size() != LARGE_GROUP_SIZE &&
                localRanks.size() != SMALL_GROUP_SIZE,
            HCCL_ERROR("Expected a four-rank or eight-rank local group, found %zu",
                localRanks.size()),
            HCCL_E_INTERNAL);

        std::vector<bool> localRankSeen(param.rankSize, false);
        for (const uint32_t rank : localRanks) {
            CHK_PRT_RET(rank >= param.rankSize || localRankSeen[rank],
                HCCL_ERROR("Invalid rank in local topology order: %u", rank),
                HCCL_E_INTERNAL);
            localRankSeen[rank] = true;
        }
        std::sort(localRanks.begin(), localRanks.end());
        const auto localIter = std::find(localRanks.begin(), localRanks.end(), param.myRank);
        CHK_PRT_RET(localIter == localRanks.end(),
            HCCL_ERROR("Local rank is missing from the local topology order"),
            HCCL_E_INTERNAL);
        const uint32_t localRank =
            static_cast<uint32_t>(localIter - localRanks.begin());

        std::vector<uint32_t> remoteRanks;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (!localRankSeen[rank]) {
                remoteRanks.push_back(rank);
            }
        }

        const bool inLargeGroup = localRanks.size() == LARGE_GROUP_SIZE;
        const std::vector<uint32_t> &largeRanks =
            inLargeGroup ? localRanks : remoteRanks;
        const std::vector<uint32_t> &smallRanks =
            inLargeGroup ? remoteRanks : localRanks;
        CHK_PRT_RET(largeRanks.size() != LARGE_GROUP_SIZE ||
                smallRanks.size() != SMALL_GROUP_SIZE,
            HCCL_ERROR("Invalid asymmetric group sizes: %zu/%zu",
                largeRanks.size(), smallRanks.size()),
            HCCL_E_INTERNAL);

        uint32_t groupBegin = 0;
        uint32_t groupEnd = LARGE_GROUP_SIZE;
        bool firstLevel = true;
        for (uint32_t distance = static_cast<uint32_t>(localRanks.size()) >> 1;
             distance != 0; distance >>= 1) {
            const uint32_t middle = groupBegin + (groupEnd - groupBegin) / 2;
            const bool inLowerHalf = (localRank & distance) == 0;
            const uint32_t peer = localRanks[localRank ^ distance];
            const uint32_t keepBegin = inLowerHalf ? groupBegin : middle;
            const uint32_t keepEnd = inLowerHalf ? middle : groupEnd;
            const uint32_t sendBegin = inLowerHalf ? middle : groupBegin;
            const uint32_t sendEnd = inLowerHalf ? groupEnd : middle;

            setRange(sendBegin, sendEnd);
            taskArgs[10] = 0;
            taskArgs[11] = 0;
            if (firstLevel && outOfPlace) {
                setCopyRange(keepBegin, keepEnd);
            }
            CHK_RET(launchPair(peer, firstLevel
                ? PairPhase::RHD_REDUCE_BIDIR_FIRST
                : PairPhase::RHD_REDUCE_BIDIR));

            groupBegin = keepBegin;
            groupEnd = keepEnd;
            firstLevel = false;
        }

        taskArgs[10] = 0;
        taskArgs[11] = 0;
        if (inLargeGroup) {
            const uint32_t shard = localRank;
            const uint32_t crossPeer = smallRanks[shard / 2];
            setRange(shard, shard + 1);
            CHK_RET(launchPair(crossPeer, PairPhase::RHD_REDUCE_RECV));
            CHK_RET(launchPair(crossPeer, PairPhase::RHD_GATHER_SEND));
        } else {
            for (uint32_t shardOffset = 0; shardOffset < 2; ++shardOffset) {
                const uint32_t shard = localRank * 2 + shardOffset;
                const uint32_t crossPeer = largeRanks[shard];
                setRange(shard, shard + 1);
                CHK_RET(launchPair(crossPeer, PairPhase::RHD_REDUCE_SEND));
                CHK_RET(launchPair(crossPeer, PairPhase::RHD_GATHER_RECV));
            }
        }

        const uint32_t ownedShardCount =
            LARGE_GROUP_SIZE / static_cast<uint32_t>(localRanks.size());
        uint32_t ownedBegin = localRank * ownedShardCount;
        uint32_t ownedEnd = ownedBegin + ownedShardCount;
        for (uint32_t distance = 1;
             distance < static_cast<uint32_t>(localRanks.size());
             distance <<= 1) {
            setRange(ownedBegin, ownedEnd);
            CHK_RET(launchPair(
                localRanks[localRank ^ distance], PairPhase::RHD_GATHER_BIDIR));
            const uint32_t exchangedShardCount = distance * ownedShardCount;
            if ((localRank & distance) == 0) {
                ownedEnd += exchangedShardCount;
            } else {
                ownedBegin -= exchangedShardCount;
            }
        }
    }

    const uint64_t tailSize = totalSize - bodySize;
    if (tailSize != 0) {
        CHK_RET(executeTreeRange(bodySize, tailSize));
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
