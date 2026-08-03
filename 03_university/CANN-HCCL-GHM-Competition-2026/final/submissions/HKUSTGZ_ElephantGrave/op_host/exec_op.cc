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
#include <cstdint>
#include <cstring>
#include <limits>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {

namespace {

    constexpr uint32_t MAX_ASYNC_STRIPES = 4;
    constexpr uint32_t MAX_COMBINE_STRIPES = 8;
    constexpr uint32_t WIDE_LANE_MAX_SOURCES = 8;
    constexpr uint32_t WIDE_LANE_TILE_COUNT = 3;
    constexpr uint32_t WIDE_LANE_WORKSPACE_BANKS = 2;
    constexpr uint32_t WIDE_LANE_METADATA_COUNT = 5;
    constexpr uint32_t MAX_RUNTIME_KERNELS = 2U * MATE_CHUNK_COUNT;
    constexpr uint32_t HYBRID_MS_KERNEL_MODE = MAX_ASYNC_STRIPES + 1;
    constexpr uint32_t HIERARCHICAL_KERNEL_MODE = HYBRID_MS_KERNEL_MODE + 1;
    constexpr uint32_t WIDE_LANE_KERNEL_MODE_BASE = HIERARCHICAL_KERNEL_MODE;
    constexpr uint32_t WORKSPACE_SLOT_COUNT = 3;
    constexpr uint64_t SMALL_OUTPUT_THRESHOLD = 1U * 1024U * 1024U;
    constexpr uint64_t MS_SLICE_BYTES = 4096U;
    // Checker work is proportional to prefixSlices * rankSize^2.  Keeping
    // that product at the already verified 2x8 limit gives every topology
    // the largest safe on-chip prefix instead of under-filling 8+4.
    constexpr uint64_t MS_CHECKER_BUCKET_CAP = 640U * 1024U;
    // CCU_SCHED owns 16 block loop engines and 128 MS buffers per IO die.
    // Eight buffers are reserved by every loop for the maximum eight-source
    // reduce, so 16 is the largest legal parallel dimension in this mode.
    constexpr uint64_t MS_PARALLEL_LOOPS = 16U;
    constexpr uint64_t MS_LOOP_BYTES = MS_SLICE_BYTES * MS_PARALLEL_LOOPS;

    constexpr uint64_t LowBits(uint32_t bitCount)
    {
        return (uint64_t{1} << bitCount) - 1U;
    }

    uint64_t EncodeParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
    {
        constexpr uint32_t REPEAT_NUM_SHIFT = 55;
        constexpr uint32_t REPEAT_LOOP_SHIFT = 48;
        constexpr uint32_t TOTAL_LOOP_SHIFT = 41;
        return ((repeatNum & LowBits(7)) << REPEAT_NUM_SHIFT)
            | ((repeatLoopIndex & LowBits(7)) << REPEAT_LOOP_SHIFT)
            | ((totalLoopNum & LowBits(7)) << TOTAL_LOOP_SHIFT);
    }

    uint64_t CalculateMsPrefix(uint64_t outputBytes, uint32_t rankSize)
    {
        if (rankSize == 0) {
            return 0;
        }
        const uint64_t rankSquare = static_cast<uint64_t>(rankSize) * rankSize;
        const uint64_t alignedCap = (MS_CHECKER_BUCKET_CAP / rankSquare) * MS_SLICE_BYTES;
        return std::min(outputBytes, alignedCap);
    }

    std::array<uint64_t, 4> BuildMsGoSize(uint64_t size)
    {
        const uint64_t mainLoopCount = size / MS_LOOP_BYTES;
        const uint64_t mainBytes = mainLoopCount * MS_LOOP_BYTES;
        const uint64_t tailBytes = size - mainBytes;
        const uint64_t fullTailSlices = tailBytes / MS_SLICE_BYTES;
        const uint64_t residual = tailBytes - fullTailSlices * MS_SLICE_BYTES;

        uint64_t parallelParam = 0;
        uint64_t tailSliceBytes = 0;
        if (fullTailSlices != 0 && residual == 0) {
            parallelParam = EncodeParallelParam(fullTailSlices - 1U, 0, 1);
            tailSliceBytes = MS_SLICE_BYTES;
        } else if (fullTailSlices == 0 && residual != 0) {
            parallelParam = EncodeParallelParam(0, 0, 1);
            tailSliceBytes = residual;
        } else if (fullTailSlices != 0 && residual != 0) {
            parallelParam = EncodeParallelParam(fullTailSlices - 1U, 1, 2);
            tailSliceBytes = residual;
        }
        return {mainBytes, mainLoopCount, parallelParam, tailSliceBytes};
    }

    struct RuntimeResourceCtx {
        CommBuffer localBuffer{};
        std::array<ThreadHandle, 2> threads{};
        size_t threadCount{0};
        std::array<CcuKernelHandle, MAX_RUNTIME_KERNELS> ccuKernels{};
        size_t kernelCount{0};
        std::array<uint32_t, MAX_RUNTIME_KERNELS> stripeCounts{};
        size_t stripeCountCount{0};
        CcuKernelHandle combineKernel{0};
        uint64_t localBufferToken{0};
        uint64_t cachedInputAddress{0};
        uint64_t cachedInputSize{0};
        uint64_t cachedInputToken{0};
        uint64_t cachedOutputAddress{0};
        uint64_t cachedOutputSize{0};
        uint64_t cachedOutputToken{0};
        uint64_t commIdCacheMagic{0};
        std::array<char, COMM_INDENTIFIER_MAX_LENGTH> cachedCommId{};
    };

    template <typename T> bool ReadContextValue(const char *&cursor, const char *end, T &value)
    {
        if (cursor > end || static_cast<size_t>(end - cursor) < sizeof(T)) {
            return false;
        }
        std::memcpy(&value, cursor, sizeof(T));
        cursor += sizeof(T);
        return true;
    }

    template <typename T, size_t N>
    bool ReadContextVector(const char *&cursor, const char *end, std::array<T, N> &values, size_t &count)
    {
        if (!ReadContextValue(cursor, end, count) || count > N) {
            return false;
        }
        for (size_t i = 0; i < count; ++i) {
            if (!ReadContextValue(cursor, end, values[i])) {
                return false;
            }
        }
        return true;
    }

    bool DeserializeRuntimeContext(const void *data, uint64_t size, RuntimeResourceCtx &ctx)
    {
        if (data == nullptr || size > std::numeric_limits<size_t>::max()) {
            return false;
        }
        const char *cursor = static_cast<const char *>(data);
        const char *end = cursor + static_cast<size_t>(size);
        ThreadHandle compatibilityThread{};
        const bool baseValid = ReadContextValue(cursor, end, compatibilityThread)
            && ReadContextValue(cursor, end, ctx.localBuffer)
            && ReadContextVector(cursor, end, ctx.threads, ctx.threadCount)
            && ReadContextVector(cursor, end, ctx.ccuKernels, ctx.kernelCount)
            && ReadContextVector(cursor, end, ctx.stripeCounts, ctx.stripeCountCount)
            && ReadContextValue(cursor, end, ctx.combineKernel)
            && ReadContextValue(cursor, end, ctx.localBufferToken)
            && ReadContextValue(cursor, end, ctx.cachedInputAddress)
            && ReadContextValue(cursor, end, ctx.cachedInputSize)
            && ReadContextValue(cursor, end, ctx.cachedInputToken)
            && ReadContextValue(cursor, end, ctx.cachedOutputAddress)
            && ReadContextValue(cursor, end, ctx.cachedOutputSize)
            && ReadContextValue(cursor, end, ctx.cachedOutputToken);
        if (!baseValid) {
            return false;
        }
        // An exact pre-V12t context remains executable; the host path simply
        // falls back to HcclGetCommName for that invocation.
        if (cursor == end) {
            return true;
        }
        return ReadContextValue(cursor, end, ctx.commIdCacheMagic)
            && ReadContextValue(cursor, end, ctx.cachedCommId)
            && ctx.commIdCacheMagic == COMM_ID_CACHE_MAGIC
            && ctx.cachedCommId[0] != '\0'
            && std::memchr(ctx.cachedCommId.data(), '\0', ctx.cachedCommId.size()) != nullptr
            && cursor == end;
    }

    bool IsWideLaneMode(uint32_t mode)
    {
        return mode > WIDE_LANE_KERNEL_MODE_BASE
            && mode <= WIDE_LANE_KERNEL_MODE_BASE + WIDE_LANE_MAX_SOURCES;
    }

    uint32_t WideLaneSourceCount(uint32_t mode)
    {
        return mode - WIDE_LANE_KERNEL_MODE_BASE;
    }

    bool IsCompactMsMode(uint32_t mode)
    {
        return mode == COMPACT_MS_OUTPUT_MODE || mode == COMPACT_MS_SCRATCH_MODE;
    }

    template <size_t N>
    bool BuildStripes(uint64_t outputBytes, uint32_t stripeCount,
        std::array<uint64_t, N> &offsets, std::array<uint64_t, N> &lengths)
    {
        constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
        if (stripeCount == 0 || stripeCount > N || outputBytes % DATA_TYPE_SIZE != 0) {
            return false;
        }
        const uint64_t elementCount = outputBytes / DATA_TYPE_SIZE;
        if (elementCount < stripeCount) {
            return false;
        }

        const uint64_t baseElements = elementCount / stripeCount;
        const uint64_t extraElements = elementCount % stripeCount;
        uint64_t nextOffset = 0;
        for (uint32_t stripe = 0; stripe < stripeCount; ++stripe) {
            offsets[stripe] = nextOffset;
            lengths[stripe] = (baseElements + static_cast<uint64_t>(stripe < extraElements)) * DATA_TYPE_SIZE;
            nextOffset += lengths[stripe];
        }
        for (uint32_t stripe = stripeCount; stripe < N; ++stripe) {
            offsets[stripe] = nextOffset;
            lengths[stripe] = 0;
        }
        return nextOffset == outputBytes;
    }

    bool BuildWideLaneTiles(uint64_t outputBytes,
        std::array<uint64_t, WIDE_LANE_TILE_COUNT> &lengths, uint64_t &workspaceStride)
    {
        constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
        constexpr uint64_t TILE_ALIGNMENT = 4096U;
        if (outputBytes % DATA_TYPE_SIZE != 0 || outputBytes < 3U * TILE_ALIGNMENT) {
            return false;
        }
        const uint64_t tileBytes = (outputBytes / WIDE_LANE_TILE_COUNT / TILE_ALIGNMENT) * TILE_ALIGNMENT;
        if (tileBytes == 0 || tileBytes > outputBytes / 2U) {
            return false;
        }
        lengths = {tileBytes, tileBytes, outputBytes - 2U * tileBytes};
        const uint64_t maxLength = *std::max_element(lengths.begin(), lengths.end());
        if (maxLength > std::numeric_limits<uint64_t>::max() - (TILE_ALIGNMENT - 1U)) {
            return false;
        }
        workspaceStride = (maxLength + TILE_ALIGNMENT - 1U) & ~(TILE_ALIGNMENT - 1U);
        return lengths[2] != 0 && workspaceStride != 0;
    }

    HcclResult PreSyncThreads(const RuntimeResourceCtx &ctx)
    {
        if (ctx.threadCount < 2) {
            return HCCL_SUCCESS;
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(ctx.threads[0], ctx.threads[1], 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(ctx.threads[1], 0, CUSTOM_TIMEOUT));
        return HCCL_SUCCESS;
    }

    HcclResult PostSyncThreads(const RuntimeResourceCtx &ctx)
    {
        if (ctx.threadCount < 2) {
            return HCCL_SUCCESS;
        }
        CHK_RET(HcommThreadNotifyWaitOnThread(ctx.threads[0], 0, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyRecordOnThread(ctx.threads[1], ctx.threads[0], 0));
        return HCCL_SUCCESS;
    }

    bool IsHybridSchedule(const RuntimeResourceCtx &ctx)
    {
        if (ctx.threadCount != 2 || ctx.kernelCount != MAX_RUNTIME_KERNELS
            || ctx.stripeCountCount != MAX_RUNTIME_KERNELS || ctx.combineKernel != 0) {
            return false;
        }
        return ctx.stripeCounts[0] == HYBRID_MATE_LOCAL_MODE
            && ctx.stripeCounts[1] == HYBRID_DIRECT_CROSS_MODE
            && ctx.stripeCounts[2] == HYBRID_DIRECT_LOCAL_MODE
            && ctx.stripeCounts[3] == HYBRID_MATE_CROSS_MODE;
    }

    bool IsWriteReduce16Schedule(const RuntimeResourceCtx &ctx)
    {
        if (ctx.threadCount != 2 || ctx.kernelCount != MAX_RUNTIME_KERNELS
            || ctx.stripeCountCount != MAX_RUNTIME_KERNELS
            || ctx.combineKernel != 0) {
            return false;
        }
        return ctx.stripeCounts[0] == WRITE_REDUCE_16_LOCAL_A_MODE
            && ctx.stripeCounts[1] == WRITE_REDUCE_16_CROSS_B_MODE
            && ctx.stripeCounts[2] == WRITE_REDUCE_16_LOCAL_B_MODE
            && ctx.stripeCounts[3] == WRITE_REDUCE_16_CROSS_A_MODE;
    }

    bool IsWriteReduce12Schedule(const RuntimeResourceCtx &ctx)
    {
        if (ctx.threadCount != 2 || ctx.kernelCount != MAX_RUNTIME_KERNELS
            || ctx.stripeCountCount != MAX_RUNTIME_KERNELS
            || ctx.combineKernel != 0) {
            return false;
        }
        return ctx.stripeCounts[0] == WRITE_REDUCE_12_LOCAL_A_MODE
            && ctx.stripeCounts[1] == WRITE_REDUCE_12_CROSS_B_MODE
            && ctx.stripeCounts[2] == WRITE_REDUCE_12_LOCAL_B_MODE
            && ctx.stripeCounts[3] == WRITE_REDUCE_12_CROSS_A_MODE;
    }

    enum class Hetero12Schedule {
        NONE,
        H8,
        H4,
    };

    Hetero12Schedule GetHetero12Schedule(const RuntimeResourceCtx &ctx)
    {
        if (ctx.threadCount != 2 || ctx.kernelCount != MAX_RUNTIME_KERNELS
            || ctx.stripeCountCount != MAX_RUNTIME_KERNELS
            || ctx.combineKernel != 0) {
            return Hetero12Schedule::NONE;
        }
        const bool h8 = ctx.stripeCounts[0] == HETERO12_H8_MATE_LOCAL_MODE
            && ctx.stripeCounts[1] == HETERO12_H8_DIRECT_CROSS_MODE
            && ctx.stripeCounts[2] == HETERO12_H8_COMBINE_MODE
            && ctx.stripeCounts[3] == HETERO12_H8_PUBLISH_MODE;
        if (h8) {
            return Hetero12Schedule::H8;
        }
        const bool h4 = ctx.stripeCounts[0] == HETERO12_H4_LOCAL_PREFIX_MODE
            && ctx.stripeCounts[1] == HETERO12_H4_DIRECT_CROSS_MODE
            && ctx.stripeCounts[2] == HETERO12_H4_LOCAL_SUFFIX_MODE
            && ctx.stripeCounts[3] == HETERO12_H4_MATE_CROSS_MODE;
        return h4 ? Hetero12Schedule::H4 : Hetero12Schedule::NONE;
    }

    HcclResult HybridStageBarrier(const RuntimeResourceCtx &ctx)
    {
        CHK_RET(HcommThreadNotifyRecordOnThread(ctx.threads[0], ctx.threads[1], 0));
        CHK_RET(HcommThreadNotifyRecordOnThread(ctx.threads[1], ctx.threads[0], 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(ctx.threads[1], 0, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyWaitOnThread(ctx.threads[0], 0, CUSTOM_TIMEOUT));
        return HCCL_SUCCESS;
    }

    HcclResult ExecuteHybrid(const RuntimeResourceCtx &ctx, uint64_t inputAddress,
        uint64_t outputAddress, uint64_t inputToken, uint64_t outputToken)
    {
        CHK_PRT_RET(!IsHybridSchedule(ctx), HCCL_ERROR("[ExecuteHybrid] invalid hybrid resource layout"),
            HCCL_E_INTERNAL);
        const std::array<uint64_t, 6> taskArgs = {
            inputAddress,
            outputAddress,
            reinterpret_cast<uint64_t>(ctx.localBuffer.addr),
            inputToken,
            outputToken,
            ctx.localBufferToken,
        };
        auto launch = [&ctx, &taskArgs](uint32_t kernelIndex, uint32_t threadIndex) {
            return HcommCcuKernelLaunch(ctx.threads[threadIndex], ctx.ccuKernels[kernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
        };

        // Stage 1 runs opposite directions on the two dies.  Both streams
        // record completion before either waits, so the rendezvous cannot form
        // a cyclic wait.  Stage 2 reverses the directions on disjoint output.
        CHK_RET(PreSyncThreads(ctx));
        CHK_RET_CCU(launch(0, 0));
        CHK_RET_CCU(launch(1, 1));
        CHK_RET(HybridStageBarrier(ctx));
        CHK_RET_CCU(launch(2, 0));
        CHK_RET_CCU(launch(3, 1));
        CHK_RET(PostSyncThreads(ctx));
        return HCCL_SUCCESS;
    }

    HcclResult ExecuteWriteReduce16(const RuntimeResourceCtx &ctx,
        uint64_t inputAddress, uint64_t outputAddress,
        uint64_t inputToken, uint64_t outputToken)
    {
        CHK_PRT_RET(!IsWriteReduce16Schedule(ctx),
            HCCL_ERROR(
                "[ExecuteWriteReduce16] invalid dual-push resource layout"),
            HCCL_E_INTERNAL);
        const std::array<uint64_t, 4> taskArgs = {
            inputAddress,
            outputAddress,
            inputToken,
            outputToken,
        };
        auto launch = [&ctx, &taskArgs](
                          uint32_t kernelIndex, uint32_t threadIndex) {
            return HcommCcuKernelLaunch(ctx.threads[threadIndex],
                ctx.ccuKernels[kernelIndex], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size()));
        };

        // Every stage writes disjoint output halves on the two IO dies.
        // The host-thread rendezvous makes both first-half partials visible
        // before the topology roles reverse in stage 2.
        CHK_RET(PreSyncThreads(ctx));
        CHK_RET_CCU(launch(0, 0));
        CHK_RET_CCU(launch(1, 1));
        CHK_RET(HybridStageBarrier(ctx));
        CHK_RET_CCU(launch(2, 0));
        CHK_RET_CCU(launch(3, 1));
        CHK_RET(PostSyncThreads(ctx));
        return HCCL_SUCCESS;
    }

    HcclResult ExecuteWriteReduce12(const RuntimeResourceCtx &ctx,
        uint64_t inputAddress, uint64_t outputAddress,
        uint64_t inputToken, uint64_t outputToken)
    {
        CHK_PRT_RET(!IsWriteReduce12Schedule(ctx),
            HCCL_ERROR(
                "[ExecuteWriteReduce12] invalid asymmetric dual-push resource layout"),
            HCCL_E_INTERNAL);
        const std::array<uint64_t, 4> taskArgs = {
            inputAddress,
            outputAddress,
            inputToken,
            outputToken,
        };
        auto launch = [&ctx, &taskArgs](
                          uint32_t kernelIndex, uint32_t threadIndex) {
            return HcommCcuKernelLaunch(ctx.threads[threadIndex],
                ctx.ccuKernels[kernelIndex], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size()));
        };

        // Both sides use K=8 for cross missions even though the H8 side owns
        // four cross channels.  The identical round count keeps every
        // bipartite channel's DATA/ACK epoch aligned.
        CHK_RET(PreSyncThreads(ctx));
        CHK_RET_CCU(launch(0, 0));
        CHK_RET_CCU(launch(1, 1));
        CHK_RET(HybridStageBarrier(ctx));
        CHK_RET_CCU(launch(2, 0));
        CHK_RET_CCU(launch(3, 1));
        CHK_RET(PostSyncThreads(ctx));
        return HCCL_SUCCESS;
    }

    HcclResult ExecuteHetero12(const OpParam &param,
        const RuntimeResourceCtx &ctx, Hetero12Schedule schedule,
        uint64_t inputAddress, uint64_t outputAddress,
        uint64_t inputToken, uint64_t outputToken)
    {
        const bool rankIsH8 = param.myRank < 8U;
        CHK_PRT_RET(param.rankSize != 12
                || schedule == Hetero12Schedule::NONE
                || rankIsH8 != (schedule == Hetero12Schedule::H8),
            HCCL_ERROR("[ExecuteHetero12] rank and resource schedule disagree"),
            HCCL_E_INTERNAL);
        const uint64_t scratchAddress
            = reinterpret_cast<uint64_t>(ctx.localBuffer.addr);
        const std::array<uint64_t, 6> taskArgs = {
            inputAddress,
            outputAddress,
            scratchAddress,
            inputToken,
            outputToken,
            ctx.localBufferToken,
        };
        const std::array<uint64_t, 4> combineArgs = {
            outputAddress,
            scratchAddress,
            outputToken,
            ctx.localBufferToken,
        };
        auto launch = [&ctx, &taskArgs](
                          uint32_t kernelIndex, uint32_t threadIndex) {
            return HcommCcuKernelLaunch(ctx.threads[threadIndex],
                ctx.ccuKernels[kernelIndex], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size()));
        };

        CHK_RET(PreSyncThreads(ctx));
        CHK_RET_CCU(launch(0, 0));
        CHK_RET_CCU(launch(1, 1));
        CHK_RET(HybridStageBarrier(ctx));
        if (schedule == Hetero12Schedule::H8) {
            CHK_RET_CCU(HcommCcuKernelLaunch(ctx.threads[0],
                ctx.ccuKernels[2], combineArgs.data(),
                static_cast<uint32_t>(combineArgs.size())));
        } else {
            CHK_RET_CCU(launch(2, 0));
        }
        CHK_RET_CCU(launch(3, 1));
        CHK_RET(PostSyncThreads(ctx));
        return HCCL_SUCCESS;
    }

    HcclResult ExecuteHierarchy(const OpParam &param, const RuntimeResourceCtx &ctx, uint64_t inputAddress,
        uint64_t outputAddress, uint64_t inputToken, uint64_t outputToken)
    {
        HierarchyRankPlan plan;
        CHK_PRT_RET(!BuildHierarchyRankPlan(param.rankSize, param.myRank, plan),
            HCCL_ERROR("[ExecuteHierarchy] unsupported hierarchy rank layout"), HCCL_E_INTERNAL);
        CHK_PRT_RET(ctx.threadCount != 2 || ctx.kernelCount != 2,
            HCCL_ERROR("[ExecuteHierarchy] expected two kernels and threads"), HCCL_E_INTERNAL);

        constexpr uint64_t SLOT_ALIGNMENT = 4096U;
        const uint64_t outputBytes = param.count * sizeof(float);
        CHK_PRT_RET(outputBytes > std::numeric_limits<uint64_t>::max() - (SLOT_ALIGNMENT - 1U),
            HCCL_ERROR("[ExecuteHierarchy] aligned output size overflows"), HCCL_E_PARA);
        const uint64_t alignedStride = (outputBytes + SLOT_ALIGNMENT - 1U) & ~(SLOT_ALIGNMENT - 1U);
        CHK_PRT_RET(alignedStride > ctx.localBuffer.size / (plan.targetCount + 2U),
            HCCL_ERROR("[ExecuteHierarchy] CCL buffer is too small for %u targets", plan.targetCount),
            HCCL_E_INTERNAL);

        const uint64_t scratchAddress = reinterpret_cast<uint64_t>(ctx.localBuffer.addr);
        const uint64_t scratchToken = ctx.localBufferToken;
        const uint64_t firstElements = (param.count + 1U) / 2U;
        const std::array<uint64_t, 2> chunkOffsets = {0, firstElements * sizeof(float)};
        const std::array<uint64_t, 2> chunkLengths = {
            chunkOffsets[1],
            outputBytes - chunkOffsets[1],
        };

        for (uint32_t chunk = 0; chunk < 2; ++chunk) {
            std::array<uint64_t, 8 + 2 * HIERARCHY_MAX_TARGETS> localArgs = {
                inputAddress,
                outputAddress,
                scratchAddress,
                inputToken,
                outputToken,
                scratchToken,
                chunkOffsets[chunk],
                alignedStride,
            };
            for (uint32_t target = 0; target < plan.targetCount; ++target) {
                localArgs[8 + target] = static_cast<uint64_t>(plan.targetRanks[target]) * outputBytes
                    + chunkOffsets[chunk];
                localArgs[8 + HIERARCHY_MAX_TARGETS + target]
                    = plan.targetActiveChunk[target] == HIERARCHY_ALL_CHUNKS
                        || plan.targetActiveChunk[target] == chunk
                    ? chunkLengths[chunk]
                    : 0;
            }
            CHK_RET_CCU(HcommCcuKernelLaunch(
                ctx.threads[0], ctx.ccuKernels[0], localArgs.data(), static_cast<uint32_t>(localArgs.size())));

            CHK_RET(HcommThreadNotifyRecordOnThread(ctx.threads[0], ctx.threads[1], 0));
            CHK_RET(HcommThreadNotifyWaitOnThread(ctx.threads[1], 0, CUSTOM_TIMEOUT));

            const std::array<uint64_t, 8> crossArgs = {
                outputAddress,
                scratchAddress,
                outputToken,
                scratchToken,
                chunkOffsets[chunk],
                chunkLengths[chunk],
                alignedStride,
                chunk,
            };
            CHK_RET_CCU(HcommCcuKernelLaunch(
                ctx.threads[1], ctx.ccuKernels[1], crossArgs.data(), static_cast<uint32_t>(crossArgs.size())));
        }
        CHK_RET(PostSyncThreads(ctx));
        return HCCL_SUCCESS;
    }

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    RuntimeResourceCtx resCtx;
    CHK_PRT_RET(!DeserializeRuntimeContext(param.resCtx, param.ctxSize, resCtx),
        HCCL_ERROR("[ExecOp] invalid serialized resource context"), HCCL_E_INTERNAL);

    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[ExecOp] invalid rank size %u", param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("[ExecOp] only FP32 SUM is supported"), HCCL_E_NOT_SUPPORT);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / DATA_TYPE_SIZE,
        HCCL_ERROR("[ExecOp] output byte size overflows"), HCCL_E_PARA);
    const uint64_t outputBytes = param.count * DATA_TYPE_SIZE;
    CHK_PRT_RET(outputBytes > MAX_DATA_SIZE,
        HCCL_ERROR("[ExecOp] output size %llu exceeds one CCU communication limit %u",
            static_cast<unsigned long long>(outputBytes), MAX_DATA_SIZE),
        HCCL_E_PARA);
    CHK_PRT_RET(outputBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("[ExecOp] input byte size overflows"), HCCL_E_PARA);
    const uint64_t inputBytes = outputBytes * param.rankSize;

    // The stream-bound main thread is acquired on every invocation.  Replace
    // the serialized creation-time handle so context reuse remains stream safe.
    CHK_PRT_RET(resCtx.threadCount == 0, HCCL_ERROR("[ExecOp] no CCU thread in resource context"), HCCL_E_INTERNAL);
    resCtx.threads[0] = param.cpuThread;

    if (param.rankSize == 1) {
        CHK_RET(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, outputBytes));
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(resCtx.kernelCount == 0 || resCtx.kernelCount > MAX_RUNTIME_KERNELS
                    || resCtx.stripeCountCount != resCtx.kernelCount,
        HCCL_ERROR("[ExecOp] inconsistent kernel/thread/stripe resources: %zu kernels, %zu threads, %zu stripes",
            resCtx.kernelCount, resCtx.threadCount, resCtx.stripeCountCount),
        HCCL_E_INTERNAL);
    const bool hybridSchedule = IsHybridSchedule(resCtx);
    const bool writeReduce16Schedule = param.rankSize == 16
        && IsWriteReduce16Schedule(resCtx);
    const bool writeReduce12Schedule = param.rankSize == 12
        && IsWriteReduce12Schedule(resCtx);
    const Hetero12Schedule hetero12Schedule = param.rankSize == 12
        ? GetHetero12Schedule(resCtx)
        : Hetero12Schedule::NONE;
    const bool stagedSchedule
        = hybridSchedule || writeReduce16Schedule || writeReduce12Schedule
        || hetero12Schedule != Hetero12Schedule::NONE;
    CHK_PRT_RET(!stagedSchedule
            && (resCtx.kernelCount > 2
                || resCtx.kernelCount != resCtx.threadCount),
        HCCL_ERROR("[ExecOp] non-MATE kernels must map one-to-one to threads"), HCCL_E_INTERNAL);
    const bool dualKernel = resCtx.kernelCount == 2;
    const bool hierarchical = dualKernel && resCtx.combineKernel == 0
        && resCtx.stripeCounts[0] == HIERARCHICAL_KERNEL_MODE
        && resCtx.stripeCounts[1] == HIERARCHICAL_KERNEL_MODE;
    if (hierarchical) {
        CHK_PRT_RET(resCtx.threadCount != 2,
            HCCL_ERROR("[ExecOp] hierarchy requires two CCU threads"), HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(dualKernel != (resCtx.combineKernel != 0),
            HCCL_ERROR("[ExecOp] combine kernel state does not match topology"), HCCL_E_INTERNAL);
        CHK_PRT_RET(dualKernel && outputBytes > resCtx.localBuffer.size,
            HCCL_ERROR("[ExecOp] scratch buffer is too small: need %llu, have %llu",
                static_cast<unsigned long long>(outputBytes), static_cast<unsigned long long>(resCtx.localBuffer.size)),
            HCCL_E_INTERNAL);
    }

    const uint64_t inputAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t scratchAddress = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    if (inputAddress == resCtx.cachedInputAddress && inputBytes == resCtx.cachedInputSize) {
        inputToken = resCtx.cachedInputToken;
    } else {
        CHK_RET_CCU(HcommCcuGetMemToken(inputAddress, inputBytes, &inputToken));
    }
    if (outputAddress == resCtx.cachedOutputAddress && outputBytes == resCtx.cachedOutputSize) {
        outputToken = resCtx.cachedOutputToken;
    } else {
        CHK_RET_CCU(HcommCcuGetMemToken(outputAddress, outputBytes, &outputToken));
    }
    const uint64_t scratchToken = resCtx.localBufferToken;

    const bool singleWriteReduce4x1 = param.rankSize == 4U
        && resCtx.kernelCount == 1U && resCtx.threadCount == 1U
        && resCtx.stripeCountCount == 1U
        && resCtx.stripeCounts[0] == WRITE_REDUCE_4X1_MODE;
    if (singleWriteReduce4x1) {
        // V12i initializes recvBuf with the local contribution and publishes
        // recvBuf itself as the endpoint-reduction accumulator.  The rank-4
        // fast path therefore has no CCL-buffer address/token ABI dependency.
        const std::array<uint64_t, 4> writeReduceArgs = {
            inputAddress,
            outputAddress,
            inputToken,
            outputToken,
        };
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0],
            resCtx.ccuKernels[0], writeReduceArgs.data(),
            static_cast<uint32_t>(writeReduceArgs.size())));
        return HCCL_SUCCESS;
    }

    if (writeReduce16Schedule) {
        return ExecuteWriteReduce16(
            resCtx, inputAddress, outputAddress, inputToken, outputToken);
    }
    if (writeReduce12Schedule) {
        return ExecuteWriteReduce12(
            resCtx, inputAddress, outputAddress, inputToken, outputToken);
    }

    const bool singleCompact = resCtx.kernelCount == 1 && resCtx.threadCount == 1
        && resCtx.stripeCountCount == 1 && IsCompactMsMode(resCtx.stripeCounts[0]);
    if (singleCompact) {
        CHK_PRT_RET(resCtx.stripeCounts[0] != COMPACT_MS_OUTPUT_MODE,
            HCCL_ERROR("[ExecOp] a single compact kernel must write the final output"),
            HCCL_E_INTERNAL);
        const std::array<uint64_t, 4> compactArgs = {
            inputAddress,
            outputAddress,
            inputToken,
            outputToken,
        };
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0],
            compactArgs.data(), static_cast<uint32_t>(compactArgs.size())));
        return HCCL_SUCCESS;
    }

    if (hybridSchedule) {
        return ExecuteHybrid(resCtx, inputAddress, outputAddress, inputToken, outputToken);
    }
    if (hetero12Schedule != Hetero12Schedule::NONE) {
        return ExecuteHetero12(param, resCtx, hetero12Schedule,
            inputAddress, outputAddress, inputToken, outputToken);
    }
    if (hierarchical) {
        return ExecuteHierarchy(
            param, resCtx, inputAddress, outputAddress, inputToken, outputToken);
    }

    const uint64_t sliceOffset = outputBytes * param.myRank;
    std::array<std::array<uint64_t, WORKSPACE_SLOT_COUNT>, 2> workspaceAddresses{};
    std::array<uint64_t, 2> workspaceBytes{};
    std::array<uint64_t, 2> msPrefixBytes{};
    std::array<uint64_t, 2> wideWorkspaceAddresses{};
    std::array<uint32_t, 2> wideSourceCounts{};
    std::array<uint64_t, WIDE_LANE_TILE_COUNT> wideTileLengths{};
    uint64_t wideWorkspaceStride = 0;
    bool hasWideLane = false;
    for (size_t i = 0; i < resCtx.kernelCount; ++i) {
        if (IsWideLaneMode(resCtx.stripeCounts[i])) {
            hasWideLane = true;
            wideSourceCounts[i] = WideLaneSourceCount(resCtx.stripeCounts[i]);
        } else if (IsCompactMsMode(resCtx.stripeCounts[i])) {
            workspaceBytes[i] = 0;
        } else if (resCtx.stripeCounts[i] == 0) {
            workspaceBytes[i] = outputBytes;
        } else if (resCtx.stripeCounts[i] == HYBRID_MS_KERNEL_MODE) {
            msPrefixBytes[i] = CalculateMsPrefix(outputBytes, param.rankSize);
            workspaceBytes[i] = outputBytes - msPrefixBytes[i];
        }
    }
    CHK_PRT_RET(hasWideLane
            && !BuildWideLaneTiles(outputBytes, wideTileLengths, wideWorkspaceStride),
        HCCL_ERROR("[ExecOp] could not build three aligned wide-lane tiles"), HCCL_E_INTERNAL);
    const uint64_t scratchReserve = dualKernel ? outputBytes : 0;
    CHK_PRT_RET(scratchReserve > resCtx.localBuffer.size,
        HCCL_ERROR("[ExecOp] CCL buffer is too small for the second partial"), HCCL_E_INTERNAL);
    uint64_t workspaceRequired = 0;
    uint64_t nextWorkspaceAddress = scratchAddress + scratchReserve;
    for (size_t i = 0; i < resCtx.kernelCount; ++i) {
        if (IsCompactMsMode(resCtx.stripeCounts[i])) {
            continue;
        }
        if (wideSourceCounts[i] != 0) {
            CHK_PRT_RET(wideSourceCounts[i] < 2 || wideSourceCounts[i] > WIDE_LANE_MAX_SOURCES,
                HCCL_ERROR("[ExecOp] invalid wide-lane source count %u", wideSourceCounts[i]),
                HCCL_E_INTERNAL);
            const uint32_t slotCount
                = WIDE_LANE_WORKSPACE_BANKS * (wideSourceCounts[i] - 1U);
            CHK_PRT_RET(wideWorkspaceStride
                    > (resCtx.localBuffer.size - scratchReserve - workspaceRequired) / slotCount,
                HCCL_ERROR("[ExecOp] CCL buffer is too small for wide-lane double buffers"),
                HCCL_E_INTERNAL);
            wideWorkspaceAddresses[i] = nextWorkspaceAddress;
            const uint64_t bytes = wideWorkspaceStride * slotCount;
            workspaceRequired += bytes;
            nextWorkspaceAddress += bytes;
            continue;
        }
        CHK_PRT_RET(workspaceBytes[i] > (resCtx.localBuffer.size - scratchReserve - workspaceRequired)
                / WORKSPACE_SLOT_COUNT,
            HCCL_ERROR("[ExecOp] CCL buffer is too small for bounded lane workspaces"), HCCL_E_INTERNAL);
        if (workspaceBytes[i] == 0) {
            continue;
        }
        for (uint32_t slot = 0; slot < WORKSPACE_SLOT_COUNT; ++slot) {
            workspaceAddresses[i][slot] = nextWorkspaceAddress;
            nextWorkspaceAddress += workspaceBytes[i];
        }
        workspaceRequired += workspaceBytes[i] * WORKSPACE_SLOT_COUNT;
    }

    // For a two-die topology this is a single record/wait pair in each
    // direction, never a one-to-many notify.
    CHK_RET(PreSyncThreads(resCtx));
    for (size_t i = 0; i < resCtx.kernelCount; ++i) {
        if (IsCompactMsMode(resCtx.stripeCounts[i])) {
            const bool writesOutput = resCtx.stripeCounts[i] == COMPACT_MS_OUTPUT_MODE;
            const std::array<uint64_t, 4> compactArgs = {
                inputAddress,
                writesOutput ? outputAddress : scratchAddress,
                inputToken,
                writesOutput ? outputToken : scratchToken,
            };
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[i], resCtx.ccuKernels[i],
                compactArgs.data(), static_cast<uint32_t>(compactArgs.size())));
            continue;
        }
        if (wideSourceCounts[i] != 0) {
            const std::array<uint64_t, 8 + WIDE_LANE_METADATA_COUNT> wideArgs = {
                inputAddress,
                outputAddress,
                scratchAddress,
                inputToken,
                outputToken,
                scratchToken,
                sliceOffset,
                outputBytes,
                wideWorkspaceAddresses[i],
                wideWorkspaceStride,
                wideTileLengths[0],
                wideTileLengths[1],
                wideTileLengths[2],
            };
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[i], resCtx.ccuKernels[i],
                wideArgs.data(), static_cast<uint32_t>(wideArgs.size())));
            continue;
        }
        std::array<uint64_t, MAX_ASYNC_STRIPES> stripeOffsets{};
        std::array<uint64_t, MAX_ASYNC_STRIPES> stripeLengths{};
        std::array<uint64_t, 4> msGoSize{};
        if (resCtx.stripeCounts[i] == 0) {
            for (uint32_t slot = 0; slot < WORKSPACE_SLOT_COUNT; ++slot) {
                stripeOffsets[slot] = workspaceAddresses[i][slot];
                stripeLengths[slot] = scratchToken;
            }
        } else if (resCtx.stripeCounts[i] == HYBRID_MS_KERNEL_MODE) {
            for (uint32_t slot = 0; slot < WORKSPACE_SLOT_COUNT; ++slot) {
                stripeOffsets[slot] = workspaceAddresses[i][slot];
                stripeLengths[slot] = scratchToken;
            }
            stripeOffsets[MAX_ASYNC_STRIPES - 1] = msPrefixBytes[i];
            stripeLengths[MAX_ASYNC_STRIPES - 1] = outputBytes - msPrefixBytes[i];
            msGoSize = BuildMsGoSize(msPrefixBytes[i]);
            const std::array<uint64_t, 12> msArgs = {
                inputAddress,
                outputAddress,
                scratchAddress,
                inputToken,
                outputToken,
                scratchToken,
                sliceOffset,
                outputBytes,
                msGoSize[0],
                msGoSize[1],
                msGoSize[2],
                msGoSize[3],
            };
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[i], resCtx.ccuKernels[i],
                msArgs.data(), static_cast<uint32_t>(msArgs.size())));
            continue;
        } else {
            CHK_PRT_RET(!BuildStripes(outputBytes, resCtx.stripeCounts[i], stripeOffsets, stripeLengths),
                HCCL_ERROR("[ExecOp] invalid %u-way stripe partition for %llu bytes", resCtx.stripeCounts[i],
                    static_cast<unsigned long long>(outputBytes)),
                HCCL_E_INTERNAL);
        }
        std::array<uint64_t, 8 + 2 * MAX_ASYNC_STRIPES> taskArgs = {
            inputAddress,
            outputAddress,
            scratchAddress,
            inputToken,
            outputToken,
            scratchToken,
            sliceOffset,
            outputBytes,
        };
        std::copy(stripeOffsets.begin(), stripeOffsets.end(), taskArgs.begin() + 8);
        std::copy(
            stripeLengths.begin(), stripeLengths.end(), taskArgs.begin() + 8 + MAX_ASYNC_STRIPES);
        CHK_RET_CCU(
            HcommCcuKernelLaunch(resCtx.threads[i], resCtx.ccuKernels[i], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size())));
    }
    CHK_RET(PostSyncThreads(resCtx));

    if (dualKernel) {
        const bool compactCombine = IsCompactMsMode(resCtx.stripeCounts[0])
            && IsCompactMsMode(resCtx.stripeCounts[1]);
        if (compactCombine) {
            const std::array<uint64_t, 4> combineArgs = {
                outputAddress,
                scratchAddress,
                outputToken,
                scratchToken,
            };
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.combineKernel,
                combineArgs.data(), static_cast<uint32_t>(combineArgs.size())));
            return HCCL_SUCCESS;
        }
        const uint32_t combineStripeCount = outputBytes <= SMALL_OUTPUT_THRESHOLD
            ? 1
            : static_cast<uint32_t>(std::min<uint64_t>(MAX_COMBINE_STRIPES, param.count));
        std::array<uint64_t, MAX_COMBINE_STRIPES> stripeOffsets{};
        std::array<uint64_t, MAX_COMBINE_STRIPES> stripeLengths{};
        CHK_PRT_RET(!BuildStripes(outputBytes, combineStripeCount, stripeOffsets, stripeLengths),
            HCCL_ERROR("[ExecOp] invalid combine stripe partition"), HCCL_E_INTERNAL);
        std::array<uint64_t, 4 + 2 * MAX_COMBINE_STRIPES> combineArgs = {
            outputAddress,
            scratchAddress,
            outputToken,
            scratchToken,
        };
        std::copy_n(stripeOffsets.begin(), combineStripeCount, combineArgs.begin() + 4);
        std::copy_n(
            stripeLengths.begin(), combineStripeCount, combineArgs.begin() + 4 + combineStripeCount);
        const uint32_t combineArgCount = 4U + 2U * combineStripeCount;
        CHK_RET_CCU(HcommCcuKernelLaunch(
            resCtx.threads[0], resCtx.combineKernel, combineArgs.data(), combineArgCount));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
