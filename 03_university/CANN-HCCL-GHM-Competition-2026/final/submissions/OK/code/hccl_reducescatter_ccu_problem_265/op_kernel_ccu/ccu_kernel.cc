/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <limits>
#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "common.h"
#include "custom.h"
#include "log.h"
#include "ccu_kernel.h"

namespace ops_hccl {
namespace {
    namespace ccu = ::AscendC::ccu;

    constexpr uint32_t INPUT_XN_ID = 0;
    constexpr uint32_t INPUT_TOKEN_XN_ID = 1;
    constexpr uint32_t POST_SYNC_ID = 2;
    constexpr uint32_t CKE_INDEX = 0;

#define CCU_CHECK(call) \
    do { \
        CcuResult ccuRet = (call); \
        if (ccuRet != CCU_SUCCESS) { \
            HCCL_ERROR("CCU call failed, ret[%d]", static_cast<int32_t>(ccuRet)); \
            return ccuRet; \
        } \
    } while (0)

    struct ReduceScatterContext {
        const CcuKernelArgReduceScatter *arg = nullptr;

        std::vector<ccu::Variable> peerInput;
        std::vector<ccu::Variable> peerInputToken;
        ccu::Variable inputBase;
        ccu::Variable inputToken;
        ccu::Variable outputBase;
        ccu::Variable outputToken;
        ccu::Variable scratchBase;
        ccu::Variable scratchToken;
        ccu::Variable rankSliceOffset;
        ccu::Variable chunkCapacity;
        ccu::Variable launchChunkOffset;
        ccu::Variable launchDataSize;
        ccu::Variable chunkOffset;

        ccu::Event transferEvent;
        ccu::Event reduceEvent;
    };

    CcuResult InitResources(ReduceScatterContext &ctx)
    {
        const uint32_t rankSize = ctx.arg->rankSize;
        if (rankSize < 2 || rankSize > MAX_RANK_SIZE || ctx.arg->rankId >= rankSize || ctx.arg->channelCount == 0
            || ctx.arg->channelCount >= rankSize || ctx.arg->includeSelf > 1 || ctx.arg->writeOutput > 1) {
            HCCL_ERROR("Invalid kernel resources, rank[%u/%u], channelCount[%u]", ctx.arg->rankId, rankSize,
                ctx.arg->channelCount);
            return CCU_E_PARA;
        }
        const uint32_t sourceCount = ctx.arg->channelCount + ctx.arg->includeSelf;
        if (sourceCount == 0 || ctx.arg->scratchStartSlot + sourceCount > rankSize) {
            HCCL_ERROR("Invalid scratch slots, start[%u], sourceCount[%u], rankSize[%u]", ctx.arg->scratchStartSlot,
                sourceCount, rankSize);
            return CCU_E_PARA;
        }
        // GetResByChannel返回不占用新寄存器的Variable，避免resize后赋值造成无效寄存器分配。
        ctx.peerInput.reserve(ctx.arg->channelCount);
        ctx.peerInputToken.reserve(ctx.arg->channelCount);

        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const uint32_t peerRank = ctx.arg->peerRanks[channelIndex];
            if (peerRank >= rankSize || peerRank == ctx.arg->rankId) {
                HCCL_ERROR("Invalid peer rank[%u]", peerRank);
                return CCU_E_PARA;
            }
            for (uint32_t previous = 0; previous < channelIndex; ++previous) {
                if (ctx.arg->peerRanks[previous] == peerRank) {
                    HCCL_ERROR("Duplicated peer rank[%u]", peerRank);
                    return CCU_E_PARA;
                }
            }
            ctx.peerInput.push_back(ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], INPUT_XN_ID));
            ctx.peerInputToken.push_back(
                ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], INPUT_TOKEN_XN_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult LoadArguments(ReduceScatterContext &ctx)
    {
        uint32_t argumentIndex = 0;
        CCU_CHECK(ccu::LoadArg(ctx.inputBase, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.outputBase, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.inputToken, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.outputToken, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.scratchBase, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.scratchToken, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.rankSliceOffset, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.chunkCapacity, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.launchChunkOffset, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.launchDataSize, argumentIndex++));
        return CCU_SUCCESS;
    }

    CcuResult PreSync(ReduceScatterContext &ctx)
    {
        const uint16_t inputMask = static_cast<uint16_t>(1U << INPUT_XN_ID);
        const uint16_t inputTokenMask = static_cast<uint16_t>(1U << INPUT_TOKEN_XN_ID);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHECK(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIndex], ctx.inputBase, INPUT_XN_ID, CKE_INDEX, inputMask));
            CCU_CHECK(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIndex], ctx.inputToken, INPUT_TOKEN_XN_ID, CKE_INDEX, inputTokenMask));
        }

        const uint16_t allMask = static_cast<uint16_t>(inputMask | inputTokenMask);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHECK(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, allMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult PostSync(ReduceScatterContext &ctx)
    {
        const uint16_t syncMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHECK(ccu::NotifyRecord(ctx.arg->channels[channelIndex], CKE_INDEX, syncMask));
        }
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHECK(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, syncMask));
        }
        return CCU_SUCCESS;
    }

    std::vector<ccu::LocalAddr> BuildScratchAddresses(ReduceScatterContext &ctx, ccu::Variable dataSize)
    {
        const uint32_t sourceCount = ctx.arg->channelCount + ctx.arg->includeSelf;
        std::vector<ccu::LocalAddr> scratch(sourceCount);
        ccu::Variable scratchOffset;
        scratchOffset = 0;
        for (uint32_t slot = 0; slot < ctx.arg->scratchStartSlot; ++slot) {
            scratchOffset += ctx.chunkCapacity;
        }
        for (uint32_t slot = 0; slot < sourceCount; ++slot) {
            scratch[slot].addr = ctx.scratchBase;
            scratch[slot].addr += scratchOffset;
            scratch[slot].token = ctx.scratchToken;
            scratchOffset += dataSize;
        }
        return scratch;
    }

    ccu::LocalAddr GetLocalInput(ReduceScatterContext &ctx, ccu::Variable inputOffset)
    {
        ccu::LocalAddr input;
        input.addr = ctx.inputBase;
        input.addr += inputOffset;
        input.token = ctx.inputToken;
        return input;
    }

    ccu::RemoteAddr GetRemoteInput(ReduceScatterContext &ctx, uint32_t channelIndex, ccu::Variable inputOffset)
    {
        ccu::RemoteAddr input;
        input.addr = ctx.peerInput[channelIndex];
        input.addr += inputOffset;
        input.token = ctx.peerInputToken[channelIndex];
        return input;
    }

    ccu::LocalAddr GetOutput(ReduceScatterContext &ctx)
    {
        ccu::LocalAddr output;
        output.addr = ctx.outputBase;
        output.addr += ctx.chunkOffset;
        output.token = ctx.outputToken;
        return output;
    }

    uint16_t GetSignalMask(uint32_t signalCount)
    {
        if (signalCount == MAX_RANK_SIZE) {
            return std::numeric_limits<uint16_t>::max();
        }
        return static_cast<uint16_t>((1U << signalCount) - 1U);
    }

    CcuResult ContiguousFixedReduce(ReduceScatterContext &ctx, std::vector<ccu::LocalAddr> &scratch, uint32_t firstSlot,
        uint32_t pieceCount, ccu::Variable dataSize)
    {
        // 连续前后半区逐轮归约；命令数为ceil(log2(pieceCount))，每轮和浮点括号化顺序均固定。
        uint32_t remainingPieces = pieceCount;
        while (remainingPieces > 1) {
            const uint32_t reducePieces = remainingPieces / 2;
            const uint32_t sourceSlot = firstSlot + remainingPieces - reducePieces;
            ccu::Variable reduceSize;
            reduceSize = 0;
            for (uint32_t piece = 0; piece < reducePieces; ++piece) {
                reduceSize += dataSize;
            }
            CCU_CHECK(ccu::LocalReduce(scratch[firstSlot], scratch[sourceSlot], reduceSize, ctx.arg->dataType,
                ctx.arg->reduceOp, ctx.reduceEvent));
            CCU_CHECK(ccu::EventWait(ctx.reduceEvent));
            remainingPieces -= reducePieces;
        }
        return CCU_SUCCESS;
    }

    CcuResult PullContributions(ReduceScatterContext &ctx, std::vector<ccu::LocalAddr> &scratch, ccu::Variable dataSize)
    {
        ccu::Variable inputOffset;
        inputOffset = ctx.rankSliceOffset;
        inputOffset += ctx.chunkOffset;

        uint32_t scratchIndex = 0;
        if (ctx.arg->includeSelf != 0) {
            CCU_CHECK(ccu::LocalCopy(scratch[scratchIndex], GetLocalInput(ctx, inputOffset), dataSize,
                ctx.transferEvent, static_cast<uint16_t>(1U << scratchIndex)));
            ++scratchIndex;
        }
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const uint16_t sourceMask = static_cast<uint16_t>(1U << scratchIndex);
            CCU_CHECK(ccu::Read(ctx.arg->channels[channelIndex], scratch[scratchIndex],
                GetRemoteInput(ctx, channelIndex, inputOffset), dataSize, ctx.transferEvent, sourceMask));
            ++scratchIndex;
        }
        CCU_CHECK(ccu::EventWait(ctx.transferEvent, GetSignalMask(scratchIndex)));
        return CCU_SUCCESS;
    }

    CcuResult ProcessPartialChunk(
        ReduceScatterContext &ctx, std::vector<ccu::LocalAddr> &scratch, ccu::Variable dataSize)
    {
        const uint32_t sourceCount = ctx.arg->channelCount + ctx.arg->includeSelf;
        CCU_CHECK(PullContributions(ctx, scratch, dataSize));
        CCU_CHECK(ContiguousFixedReduce(ctx, scratch, 0, sourceCount, dataSize));
        if (ctx.arg->writeOutput != 0) {
            CCU_CHECK(ccu::LocalCopy(GetOutput(ctx), scratch[0], dataSize, ctx.reduceEvent));
            CCU_CHECK(ccu::EventWait(ctx.reduceEvent));
        }
        return CCU_SUCCESS;
    }

    CcuResult ProcessChunk(ReduceScatterContext &ctx, ccu::Variable dataSize)
    {
        std::vector<ccu::LocalAddr> scratch = BuildScratchAddresses(ctx, dataSize);
        return ProcessPartialChunk(ctx, scratch, dataSize);
    }
} // namespace

CcuResult CcuReduceScatterDeterministicKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatter *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    ReduceScatterContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK(InitResources(ctx));
    CCU_CHECK(LoadArguments(ctx));
    CCU_CHECK(PreSync(ctx));

    ctx.chunkOffset = ctx.launchChunkOffset;
    CCU_CHECK(ProcessChunk(ctx, ctx.launchDataSize));

    CCU_CHECK(PostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterMergeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterMerge *>(arg);
    if (kernelArg == nullptr || kernelArg->scratchStartSlot == 0 || kernelArg->scratchStartSlot >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    // Keep the same ten-argument ABI as the communication kernels. Loading
    // every argument sequentially also keeps task generation deterministic.
    ccu::Variable inputBase;
    ccu::Variable outputBase;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchBase;
    ccu::Variable scratchToken;
    ccu::Variable rankSliceOffset;
    ccu::Variable chunkCapacity;
    ccu::Variable chunkOffset;
    ccu::Variable dataSize;
    uint32_t argumentIndex = 0;
    CCU_CHECK(ccu::LoadArg(inputBase, argumentIndex++));
    CCU_CHECK(ccu::LoadArg(outputBase, argumentIndex++));
    CCU_CHECK(ccu::LoadArg(inputToken, argumentIndex++));
    CCU_CHECK(ccu::LoadArg(outputToken, argumentIndex++));
    CCU_CHECK(ccu::LoadArg(scratchBase, argumentIndex++));
    CCU_CHECK(ccu::LoadArg(scratchToken, argumentIndex++));
    CCU_CHECK(ccu::LoadArg(rankSliceOffset, argumentIndex++));
    CCU_CHECK(ccu::LoadArg(chunkCapacity, argumentIndex++));
    CCU_CHECK(ccu::LoadArg(chunkOffset, argumentIndex++));
    CCU_CHECK(ccu::LoadArg(dataSize, argumentIndex++));

    ccu::LocalAddr output;
    output.addr = outputBase;
    output.addr += chunkOffset;
    output.token = outputToken;

    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t slot = 0; slot < kernelArg->scratchStartSlot; ++slot) {
        scratchOffset += chunkCapacity;
    }
    ccu::LocalAddr secondaryPartial;
    secondaryPartial.addr = scratchBase;
    secondaryPartial.addr += scratchOffset;
    secondaryPartial.token = scratchToken;

    ccu::Event mergeEvent;
    CCU_CHECK(
        ccu::LocalReduce(output, secondaryPartial, dataSize, kernelArg->dataType, kernelArg->reduceOp, mergeEvent));
    CCU_CHECK(ccu::EventWait(mergeEvent));
    return CCU_SUCCESS;
}

namespace {
    struct HierLocalContext {
        const CcuKernelArgReduceScatterHierLocal *arg = nullptr;
        std::vector<ccu::Variable> peerInput;
        std::vector<ccu::Variable> peerInputToken;
        ccu::Variable inputBase;
        ccu::Variable outputBase;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable scratchBase;
        ccu::Variable scratchToken;
        ccu::Variable recvSize;
        ccu::Event transferEvent;
        ccu::Event reduceEvent;
    };

    struct HierCrossContext {
        const CcuKernelArgReduceScatterHierCross *arg = nullptr;
        std::vector<ccu::Variable> peerScratch;
        std::vector<ccu::Variable> peerScratchToken;
        ccu::Variable inputBase;
        ccu::Variable outputBase;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable scratchBase;
        ccu::Variable scratchToken;
        ccu::Variable recvSize;
        ccu::Event transferEvent;
    };

    template <typename Context> CcuResult LoadHierArguments(Context &ctx)
    {
        uint32_t argumentIndex = 0;
        CCU_CHECK(ccu::LoadArg(ctx.inputBase, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.outputBase, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.inputToken, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.outputToken, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.scratchBase, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.scratchToken, argumentIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.recvSize, argumentIndex++));
        return CCU_SUCCESS;
    }

    template <typename KernelArg> CcuResult HierPostSync(const KernelArg *arg)
    {
        const uint16_t syncMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
        for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
            CCU_CHECK(ccu::NotifyRecord(arg->channels[channelIndex], CKE_INDEX, syncMask));
        }
        for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
            CCU_CHECK(ccu::NotifyWait(arg->channels[channelIndex], CKE_INDEX, syncMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult InitHierLocal(HierLocalContext &ctx)
    {
        const auto *arg = ctx.arg;
        if (arg->rankSize != 12 && arg->rankSize != 16) {
            return CCU_E_PARA;
        }
        if (arg->serverSize != 4 && arg->serverSize != 8) {
            return CCU_E_PARA;
        }
        if (arg->rankId < arg->serverStart || arg->rankId >= arg->serverStart + arg->serverSize
            || arg->channelCount + 1 != arg->serverSize || arg->assignedOutputCount == 0
            || arg->assignedOutputCount > 3) {
            return CCU_E_PARA;
        }

        ctx.peerInput.reserve(arg->channelCount);
        ctx.peerInputToken.reserve(arg->channelCount);
        for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
            const uint32_t peerRank = arg->peerRanks[channelIndex];
            if (peerRank < arg->serverStart || peerRank >= arg->serverStart + arg->serverSize
                || peerRank == arg->rankId) {
                return CCU_E_PARA;
            }
            ctx.peerInput.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIndex], INPUT_XN_ID));
            ctx.peerInputToken.push_back(
                ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIndex], INPUT_TOKEN_XN_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult HierLocalPreSync(HierLocalContext &ctx)
    {
        const uint16_t inputMask = static_cast<uint16_t>(1U << INPUT_XN_ID);
        const uint16_t tokenMask = static_cast<uint16_t>(1U << INPUT_TOKEN_XN_ID);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHECK(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIndex], ctx.inputBase, INPUT_XN_ID, CKE_INDEX, inputMask));
            CCU_CHECK(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIndex], ctx.inputToken, INPUT_TOKEN_XN_ID, CKE_INDEX, tokenMask));
        }
        const uint16_t allMask = static_cast<uint16_t>(inputMask | tokenMask);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHECK(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, allMask));
        }
        return CCU_SUCCESS;
    }

    ccu::Variable RepeatSize(ccu::Variable size, uint32_t count)
    {
        ccu::Variable result;
        result = 0;
        for (uint32_t index = 0; index < count; ++index) {
            result += size;
        }
        return result;
    }

    CcuResult HierLocalReduce(HierLocalContext &ctx, std::vector<ccu::LocalAddr> &scratch)
    {
        uint32_t remainingPieces = static_cast<uint32_t>(scratch.size());
        while (remainingPieces > 1) {
            const uint32_t reducePieces = remainingPieces / 2;
            const uint32_t sourceSlot = remainingPieces - reducePieces;
            const ccu::Variable reduceSize = RepeatSize(ctx.recvSize, reducePieces);
            CCU_CHECK(ccu::LocalReduce(
                scratch[0], scratch[sourceSlot], reduceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.reduceEvent));
            CCU_CHECK(ccu::EventWait(ctx.reduceEvent));
            remainingPieces -= reducePieces;
        }
        return CCU_SUCCESS;
    }

    CcuResult RunHierLocal(HierLocalContext &ctx)
    {
        const uint32_t localIndex = ctx.arg->rankId - ctx.arg->serverStart;
        const uint32_t totalSourceCount = ctx.arg->assignedOutputCount * ctx.arg->serverSize;
        if (totalSourceCount > MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }
        std::vector<std::vector<ccu::LocalAddr>> allScratch(ctx.arg->assignedOutputCount);

        // Fire every independent Server-local read first.  At most 16 sources are
        // active (2x8 or 3x4), so one Event can wait for the whole batch.
        for (uint32_t assignmentIndex = 0; assignmentIndex < ctx.arg->assignedOutputCount; ++assignmentIndex) {
            const uint32_t outputRank = localIndex + assignmentIndex * ctx.arg->serverSize;
            if (outputRank >= ctx.arg->rankSize) {
                return CCU_E_PARA;
            }

            const ccu::Variable inputOffset = RepeatSize(ctx.recvSize, outputRank);
            const uint32_t blockStartSlot = assignmentIndex * ctx.arg->serverSize;
            ccu::Variable scratchOffset = RepeatSize(ctx.recvSize, blockStartSlot);
            std::vector<ccu::LocalAddr> &scratch = allScratch[assignmentIndex];
            scratch.resize(ctx.arg->serverSize);
            for (uint32_t slot = 0; slot < ctx.arg->serverSize; ++slot) {
                scratch[slot].addr = ctx.scratchBase;
                scratch[slot].addr += scratchOffset;
                scratch[slot].token = ctx.scratchToken;
                scratchOffset += ctx.recvSize;
            }

            ccu::LocalAddr localInput;
            localInput.addr = ctx.inputBase;
            localInput.addr += inputOffset;
            localInput.token = ctx.inputToken;
            const uint32_t firstSignalIndex = assignmentIndex * ctx.arg->serverSize;
            CCU_CHECK(ccu::LocalCopy(scratch[0], localInput, ctx.recvSize, ctx.transferEvent,
                static_cast<uint16_t>(1U << firstSignalIndex)));

            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
                ccu::RemoteAddr remoteInput;
                remoteInput.addr = ctx.peerInput[channelIndex];
                remoteInput.addr += inputOffset;
                remoteInput.token = ctx.peerInputToken[channelIndex];
                const uint32_t signalIndex = firstSignalIndex + channelIndex + 1;
                const uint16_t signalMask = static_cast<uint16_t>(1U << signalIndex);
                CCU_CHECK(ccu::Read(ctx.arg->channels[channelIndex], scratch[channelIndex + 1], remoteInput,
                    ctx.recvSize, ctx.transferEvent, signalMask));
            }
        }
        CCU_CHECK(ccu::EventWait(ctx.transferEvent, GetSignalMask(totalSourceCount)));
        for (uint32_t assignmentIndex = 0; assignmentIndex < ctx.arg->assignedOutputCount; ++assignmentIndex) {
            CCU_CHECK(HierLocalReduce(ctx, allScratch[assignmentIndex]));
        }
        return CCU_SUCCESS;
    }

    CcuResult InitHierCross(HierCrossContext &ctx)
    {
        const auto *arg = ctx.arg;
        if ((arg->rankSize != 12 && arg->rankSize != 16) || (arg->localServerSize != 4 && arg->localServerSize != 8)
            || (arg->remoteServerSize != 4 && arg->remoteServerSize != 8) || arg->channelCount == 0
            || arg->channelCount > 2 || arg->remoteReadChannelIndex >= arg->channelCount) {
            return CCU_E_PARA;
        }

        ctx.peerScratch.reserve(arg->channelCount);
        ctx.peerScratchToken.reserve(arg->channelCount);
        for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
            const uint32_t peerRank = arg->peerRanks[channelIndex];
            if (peerRank >= arg->rankSize || peerRank == arg->rankId) {
                return CCU_E_PARA;
            }
            ctx.peerScratch.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIndex], INPUT_XN_ID));
            ctx.peerScratchToken.push_back(
                ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIndex], INPUT_TOKEN_XN_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult HierCrossPreSync(HierCrossContext &ctx)
    {
        const uint16_t scratchMask = static_cast<uint16_t>(1U << INPUT_XN_ID);
        const uint16_t tokenMask = static_cast<uint16_t>(1U << INPUT_TOKEN_XN_ID);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHECK(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIndex], ctx.scratchBase, INPUT_XN_ID, CKE_INDEX, scratchMask));
            CCU_CHECK(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIndex], ctx.scratchToken, INPUT_TOKEN_XN_ID, CKE_INDEX, tokenMask));
        }
        const uint16_t allMask = static_cast<uint16_t>(scratchMask | tokenMask);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHECK(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, allMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunHierCross(HierCrossContext &ctx)
    {
        const uint32_t localAssignmentIndex = ctx.arg->rankId / ctx.arg->localServerSize;
        const uint32_t remoteAssignmentIndex = ctx.arg->rankId / ctx.arg->remoteServerSize;
        const uint32_t localPartialSlot = localAssignmentIndex * ctx.arg->localServerSize;
        const uint32_t remotePartialSlot = remoteAssignmentIndex * ctx.arg->remoteServerSize;

        ccu::LocalAddr output;
        output.addr = ctx.outputBase;
        output.token = ctx.outputToken;

        ccu::LocalAddr localPartial;
        localPartial.addr = ctx.scratchBase;
        localPartial.addr += RepeatSize(ctx.recvSize, localPartialSlot);
        localPartial.token = ctx.scratchToken;

        ccu::RemoteAddr remotePartial;
        remotePartial.addr = ctx.peerScratch[ctx.arg->remoteReadChannelIndex];
        remotePartial.addr += RepeatSize(ctx.recvSize, remotePartialSlot);
        remotePartial.token = ctx.peerScratchToken[ctx.arg->remoteReadChannelIndex];

        CCU_CHECK(ccu::LocalCopy(output, localPartial, ctx.recvSize, ctx.transferEvent, 1));
        CCU_CHECK(ccu::EventWait(ctx.transferEvent, 1));
        CCU_CHECK(ccu::ReadReduce(ctx.arg->channels[ctx.arg->remoteReadChannelIndex], output, remotePartial,
            ctx.recvSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.transferEvent, 1));
        CCU_CHECK(ccu::EventWait(ctx.transferEvent, 1));
        return CCU_SUCCESS;
    }
} // namespace

CcuResult CcuReduceScatterHierLocalKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterHierLocal *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    HierLocalContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK(InitHierLocal(ctx));
    CCU_CHECK(LoadHierArguments(ctx));
    CCU_CHECK(HierLocalPreSync(ctx));
    CCU_CHECK(RunHierLocal(ctx));
    CCU_CHECK(HierPostSync(ctx.arg));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterHierCrossKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterHierCross *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    HierCrossContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK(InitHierCross(ctx));
    CCU_CHECK(LoadHierArguments(ctx));
    CCU_CHECK(HierCrossPreSync(ctx));
    CCU_CHECK(RunHierCross(ctx));
    CCU_CHECK(HierPostSync(ctx.arg));
    return CCU_SUCCESS;
}

#undef CCU_CHECK

} // namespace ops_hccl
