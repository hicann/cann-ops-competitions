/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <hcomm/hcomm_primitives.h>

 #include "ccu_kernel.h"
 
 namespace ops_hccl {
 namespace {
     constexpr uint32_t OUTPUT_XN_ID = 1;
     constexpr uint32_t OUTPUT_TOKEN_XN_ID = 2;
     constexpr uint32_t CHANNEL_NOTIFY_IDX = 0;
     constexpr uint16_t OUTPUT_ADDR_MASK = 1U << OUTPUT_XN_ID;
     constexpr uint16_t OUTPUT_TOKEN_MASK = 1U << OUTPUT_TOKEN_XN_ID;
     constexpr uint16_t POST_SYNC_MASK = 1U << 3;
     constexpr uint32_t GROUP_COPY_RANK_SIZE = 4;
 
 #define CCU_KERNEL_CHECK(call) \
     do { \
         CcuResult ccuRet = (call); \
         if (ccuRet != CCU_SUCCESS) { \
             return ccuRet; \
         } \
     } while (0)
 
     CcuResult InitPeerResources(const CcuKernelArgAllGather *arg, std::vector<ccu::Variable> &peerOutputs,
         std::vector<ccu::Variable> &peerOutputTokens, bool allowDuplicatePeers)
     {
         if (arg == nullptr || arg->rankSize < 2 || arg->rankSize > MAX_RANK_SIZE || arg->rankId >= arg->rankSize
             || arg->channelCount == 0 || arg->channelCount >= arg->rankSize || arg->handleLocalCopy > 1) {
             return CCU_E_PARA;
         }
 
         peerOutputs.resize(arg->channelCount);
         peerOutputTokens.resize(arg->channelCount);
 
         uint16_t peerMask = 0;
         for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
             const uint32_t peerRank = arg->peerRanks[channelIdx];
             if (peerRank >= arg->rankSize || peerRank == arg->rankId) {
                 return CCU_E_PARA;
             }
             const uint16_t peerBit = static_cast<uint16_t>(uint32_t{1} << peerRank);
             if (!allowDuplicatePeers && (peerMask & peerBit) != 0) {
                 return CCU_E_PARA;
             }
             peerMask = static_cast<uint16_t>(peerMask | peerBit);
             peerOutputs[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
             peerOutputTokens[channelIdx]
                 = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_TOKEN_XN_ID);
         }
         return CCU_SUCCESS;
     }
 
     CcuResult InitResources(CcuAllGatherContext &ctx)
     {
         return InitPeerResources(ctx.arg, ctx.peerOutputs, ctx.peerOutputTokens, false);
     }
 
     CcuResult InitParallelRepeatResources(CcuParallelRepeatContext &ctx)
     {
         CCU_KERNEL_CHECK(InitPeerResources(ctx.arg, ctx.peerOutputs, ctx.peerOutputTokens, false));
         if (ctx.arg->handleLocalCopy != 0 || ctx.arg->blockCount == 0
             || ctx.arg->blockCount > ALLGATHER_PARALLEL_LOCAL_RANK_NUM) {
             return CCU_E_PARA;
         }
         ctx.blockOffsets.resize(ctx.arg->blockCount);
         ctx.completionEvents.resize(ctx.arg->blockCount);
         return CCU_SUCCESS;
     }
 
     CcuResult ValidateStripedResources(const CcuKernelArgAllGather *arg)
     {
         if (arg->stripeCount == 0 || arg->stripeCount > ALLGATHER_PARALLEL_MAX_INTER_CHANNEL_NUM
             || arg->stripeBaseIndex >= arg->stripeCount
             || arg->channelCount > arg->stripeCount - arg->stripeBaseIndex) {
             return CCU_E_PARA;
         }
         const uint32_t peerRank = arg->peerRanks[0];
         for (uint32_t channelIdx = 1; channelIdx < arg->channelCount; ++channelIdx) {
             if (arg->peerRanks[channelIdx] != peerRank) {
                 return CCU_E_PARA;
             }
         }
         return CCU_SUCCESS;
     }
 
     CcuResult InitStripedAllGatherResources(CcuStripedAllGatherContext &ctx)
     {
         CCU_KERNEL_CHECK(InitPeerResources(ctx.arg, ctx.peerOutputs, ctx.peerOutputTokens, true));
         CCU_KERNEL_CHECK(ValidateStripedResources(ctx.arg));
         if (ctx.arg->handleLocalCopy != 0 || ctx.arg->blockCount != 0) {
             return CCU_E_PARA;
         }
         return CCU_SUCCESS;
     }
 
     CcuResult InitParallelStripedRepeatResources(CcuParallelStripedRepeatContext &ctx)
     {
         CCU_KERNEL_CHECK(InitPeerResources(ctx.arg, ctx.peerOutputs, ctx.peerOutputTokens, true));
         CCU_KERNEL_CHECK(ValidateStripedResources(ctx.arg));
         if (ctx.arg->handleLocalCopy != 0 || ctx.arg->blockCount == 0
             || ctx.arg->blockCount > ALLGATHER_PARALLEL_LOCAL_RANK_NUM) {
             return CCU_E_PARA;
         }
         ctx.blockOffsets.resize(ctx.arg->blockCount);
         ctx.completionEvents.resize(ctx.arg->blockCount);
         return CCU_SUCCESS;
     }
 
     CcuResult InitAsym8x4Stage0InterResources(CcuAsym8x4Stage0InterContext &ctx)
     {
         CCU_KERNEL_CHECK(InitPeerResources(ctx.arg, ctx.peerOutputs, ctx.peerOutputTokens, false));
         if (ctx.arg->rankSize != ALLGATHER_ASYM_8X4_RANK_NUM || ctx.arg->handleLocalCopy != 0
             || ctx.arg->blockCount != 0) {
             return CCU_E_PARA;
         }
         return CCU_SUCCESS;
     }
 
     CcuResult InitAsym8x4Stage1IntraResources(CcuAsym8x4Stage1IntraContext &ctx)
     {
         CCU_KERNEL_CHECK(InitPeerResources(ctx.arg, ctx.peerOutputs, ctx.peerOutputTokens, false));
         const uint32_t expectedBlockCount = ctx.arg->rankId < ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM ? 1U : 2U;
         if (ctx.arg->rankSize != ALLGATHER_ASYM_8X4_RANK_NUM || ctx.arg->blockCount != expectedBlockCount) {
             return CCU_E_PARA;
         }
         ctx.forwardBlockOffsets.resize(ctx.arg->blockCount);
         ctx.forwardCompletionEvents.resize(ctx.arg->blockCount);
         return CCU_SUCCESS;
     }
 
     CcuResult LoadTaskArgs(CcuAllGatherContext &ctx)
     {
         uint32_t argIdx = 0;
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.input, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutput, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.inputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.currentRankOutputOffset, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.sliceSize, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.skipLocalCopy, argIdx++));
         return CCU_SUCCESS;
     }

     CcuResult LoadGroupCopyTaskArgs(CcuGroupCopyAllGatherContext &ctx)
     {
         CCU_KERNEL_CHECK(LoadTaskArgs(ctx));
         uint32_t argIdx = 7;
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.groupCopySize.addrOffset, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.groupCopySize.loopParam, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.groupCopySize.parallelParam, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.groupCopySize.residual, argIdx++));
         return CCU_SUCCESS;
     }
 
     CcuResult LoadDualSliceTaskArgs(CcuAllGatherDualSliceContext &ctx)
     {
         uint32_t argIdx = 0;
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.input, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutput, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.inputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.currentRankOutputOffset, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.sliceSize, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.secondSliceSize, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.skipLocalCopy, argIdx++));
         return CCU_SUCCESS;
     }
 
     CcuResult LoadParallelRepeatTaskArgs(CcuParallelRepeatContext &ctx)
     {
         uint32_t argIdx = 0;
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutput, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.sliceSize, argIdx++));
         for (uint32_t blockIdx = 0; blockIdx < ctx.arg->blockCount; ++blockIdx) {
             CCU_KERNEL_CHECK(ccu::LoadArg(ctx.blockOffsets[blockIdx], argIdx++));
         }
         return CCU_SUCCESS;
     }
 
     CcuResult LoadStripedAllGatherTaskArgs(CcuStripedAllGatherContext &ctx)
     {
         uint32_t argIdx = 0;
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.input, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutput, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.inputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.currentRankOutputOffset, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.stripeSize, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.lastStripeSize, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.skipLocalCopy, argIdx++));
         return CCU_SUCCESS;
     }
 
     CcuResult LoadParallelStripedRepeatTaskArgs(CcuParallelStripedRepeatContext &ctx)
     {
         uint32_t argIdx = 0;
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutput, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.stripeSize, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.lastStripeSize, argIdx++));
         for (uint32_t blockIdx = 0; blockIdx < ctx.arg->blockCount; ++blockIdx) {
             CCU_KERNEL_CHECK(ccu::LoadArg(ctx.blockOffsets[blockIdx], argIdx++));
         }
         return CCU_SUCCESS;
     }
 
     CcuResult LoadAsym8x4Stage0InterTaskArgs(CcuAsym8x4Stage0InterContext &ctx)
     {
         uint32_t argIdx = 0;
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.input, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutput, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.inputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.currentRankOutputOffset, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.hierarchySize, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.halfSize, argIdx++));
         return CCU_SUCCESS;
     }
 
     CcuResult LoadAsym8x4Stage1IntraTaskArgs(CcuAsym8x4Stage1IntraContext &ctx)
     {
         uint32_t argIdx = 0;
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.input, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutput, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.inputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localOutputToken, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.currentRankOutputOffset, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localLeadSize, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.localRemainingSize, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.totalSize, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.skipLocalCopy, argIdx++));
         CCU_KERNEL_CHECK(ccu::LoadArg(ctx.forwardSize, argIdx++));
         for (uint32_t blockIdx = 0; blockIdx < ctx.arg->blockCount; ++blockIdx) {
             CCU_KERNEL_CHECK(ccu::LoadArg(ctx.forwardBlockOffsets[blockIdx], argIdx++));
         }
         return CCU_SUCCESS;
     }
 
     CcuResult ExchangeOutputInfo(
         const CcuKernelArgAllGather *arg, ccu::Variable &localOutput, ccu::Variable &localOutputToken)
     {
         for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
             CCU_KERNEL_CHECK(ccu::WriteVariableWithNotify(
                 arg->channels[channelIdx], localOutput, OUTPUT_XN_ID, CHANNEL_NOTIFY_IDX, OUTPUT_ADDR_MASK));
             CCU_KERNEL_CHECK(ccu::WriteVariableWithNotify(arg->channels[channelIdx], localOutputToken,
                 OUTPUT_TOKEN_XN_ID, CHANNEL_NOTIFY_IDX, OUTPUT_TOKEN_MASK));
         }
 
         constexpr uint16_t outputInfoMask = OUTPUT_ADDR_MASK | OUTPUT_TOKEN_MASK;
         for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
             CCU_KERNEL_CHECK(ccu::NotifyWait(arg->channels[channelIdx], CHANNEL_NOTIFY_IDX, outputInfoMask));
         }
         return CCU_SUCCESS;
     }

     constexpr uint64_t GroupCopyBitMask(uint16_t end)
     {
         return (uint64_t{1} << (end + 1)) - uint64_t{1};
     }

     constexpr uint64_t GroupCopyLoopParam(uint64_t loopContextId, uint64_t addressOffset, uint64_t loopCount)
     {
         constexpr uint16_t contextBitNum = 8;
         constexpr uint16_t contextShift = 45;
         constexpr uint16_t addressBitNum = 32;
         constexpr uint16_t addressShift = 13;
         constexpr uint16_t loopCountBitNum = 13;
         return ((loopContextId & GroupCopyBitMask(contextBitNum)) << contextShift)
                | ((addressOffset & GroupCopyBitMask(addressBitNum)) << addressShift)
                | (loopCount & GroupCopyBitMask(loopCountBitNum));
     }

     constexpr uint64_t GroupCopyParallelParam(
         uint64_t repeatCount, uint64_t repeatLoopIndex, uint64_t totalLoopCount)
     {
         constexpr uint16_t repeatBitNum = 7;
         constexpr uint16_t repeatShift = 55;
         constexpr uint16_t repeatLoopBitNum = 7;
         constexpr uint16_t repeatLoopShift = 48;
         constexpr uint16_t totalLoopBitNum = 7;
         constexpr uint16_t totalLoopShift = 41;
         return ((repeatCount & GroupCopyBitMask(repeatBitNum)) << repeatShift)
                | ((repeatLoopIndex & GroupCopyBitMask(repeatLoopBitNum)) << repeatLoopShift)
                | ((totalLoopCount & GroupCopyBitMask(totalLoopBitNum)) << totalLoopShift);
     }

     constexpr uint64_t GroupCopyOffsetParam(uint64_t addressOffset, uint64_t memoryOffset, uint64_t eventOffset)
     {
         constexpr uint16_t addressBitNum = 32;
         constexpr uint16_t addressShift = 21;
         constexpr uint16_t memoryBitNum = 11;
         constexpr uint16_t memoryShift = 10;
         constexpr uint16_t eventBitNum = 10;
         return ((addressOffset & GroupCopyBitMask(addressBitNum)) << addressShift)
                | ((memoryOffset & GroupCopyBitMask(memoryBitNum)) << memoryShift)
                | (eventOffset & GroupCopyBitMask(eventBitNum));
     }

     void InitGroupCopyResources(CcuGroupCopyAllGatherContext &ctx, ccu::LocalAddr *loopSrc,
         ccu::LocalAddr *loopDst, ccu::Variable *loopLength)
     {
         if (!ctx.resourceAllocated) {
             ctx.loopConfig.msInterleave = GROUP_COPY_MS_INTERLEAVE;
             ctx.loopConfig.loopCount = GROUP_COPY_LOOP_COUNT;
             ctx.loopConfig.memSlice = GROUP_COPY_MS_PER_LOOP * GROUP_COPY_MS_SIZE;
             ctx.loopResource.eventCount = ctx.loopConfig.loopCount;
             ctx.loopResource.completedEvents = ccu::Array<ccu::Event>(ctx.loopResource.eventCount);
             ctx.loopResource.bufferCount = ctx.loopConfig.loopCount * ctx.loopConfig.msInterleave;
             ctx.loopResource.buffers = ccu::Array<ccu::CcuBuffer>(ctx.loopResource.bufferCount);
             ctx.resourceAllocated = true;
         }

         const std::string loopType = "localcopy";
         if (ctx.loopEntities.count(loopType) != 0) {
             return;
         }
         ctx.loopEntities.emplace(loopType, GroupCopyLoopEntity{});
         GroupCopyLoopEntity &entity = ctx.loopEntities[loopType];
         for (uint32_t index = 0; index < 2; ++index) {
             const uint32_t bufferBase = index * ctx.loopConfig.msInterleave;
             const ccu::Event loopEvent = ctx.loopResource.completedEvents[index];
             entity.bodies[index].reset(new ccu::Func(
                 [&ctx, index, bufferBase, loopEvent, loopSrc, loopDst, loopLength]() {
                     ccu::LocalCopy(
                         ctx.loopResource.buffers[bufferBase], loopSrc[index], loopLength[index], loopEvent, 1);
                     ccu::EventWait(loopEvent, 1);
                     ccu::LocalCopy(
                         loopDst[index], ctx.loopResource.buffers[bufferBase], loopLength[index], loopEvent, 1);
                     ccu::EventWait(loopEvent, 1);
                 }));
             entity.loops[index].reset(new ccu::Loop(entity.loopParams[index], *entity.bodies[index]));
         }
     }

     CcuResult GroupCopy(
         CcuGroupCopyAllGatherContext &ctx, ccu::LocalAddr dst, ccu::LocalAddr src, GroupCopySizeVars &copySize)
     {
         ccu::LocalAddr loopSrc[2];
         ccu::LocalAddr loopDst[2];
         ccu::Variable loopLength[2];
         InitGroupCopyResources(ctx, loopSrc, loopDst, loopLength);
         GroupCopyLoopEntity &loops = ctx.loopEntities["localcopy"];

         CCU_IF(copySize.addrOffset != 0)
         {
             ccu::Variable loopParam;
             loopParam = GroupCopyLoopParam(0, ctx.loopConfig.memSlice * ctx.loopConfig.loopCount, 0);
             loopParam += copySize.loopParam;
             ccu::Variable sliceSize;
             sliceSize = ctx.loopConfig.memSlice;

             loopSrc[0].addr = src.addr;
             loopSrc[0].token = src.token;
             loopDst[0].addr = dst.addr;
             loopDst[0].token = dst.token;
             loopLength[0] = sliceSize;

             loops.loopParams[0] = loopParam;
             ccu::Variable parallelConfig;
             parallelConfig = GroupCopyParallelParam(ctx.loopConfig.loopCount - 1, 0, 1);
             ccu::Variable offsetConfig;
             offsetConfig = GroupCopyOffsetParam(ctx.loopConfig.memSlice, ctx.loopConfig.msInterleave, 1);
             std::vector<ccu::Loop> groupLoops{*loops.loops[0]};
             ccu::LoopGroup group(parallelConfig, offsetConfig, ctx.loopConfig.loopCount, groupLoops);
         }

         CCU_IF(copySize.parallelParam != 0)
         {
             src.addr += copySize.addrOffset;
             dst.addr += copySize.addrOffset;

             loopSrc[0].addr = src.addr;
             loopSrc[0].token = src.token;
             loopDst[0].addr = dst.addr;
             loopDst[0].token = dst.token;
             loopLength[0] = copySize.residual;

             src.addr += copySize.residual;
             dst.addr += copySize.residual;
             ccu::Variable sliceSize;
             sliceSize = ctx.loopConfig.memSlice;
             loopSrc[1].addr = src.addr;
             loopSrc[1].token = src.token;
             loopDst[1].addr = dst.addr;
             loopDst[1].token = dst.token;
             loopLength[1] = sliceSize;

             loops.loopParams[0] = GroupCopyLoopParam(0, 0, 1);
             loops.loopParams[1] = GroupCopyLoopParam(0, 0, 1);
             ccu::Variable offsetConfig;
             offsetConfig = GroupCopyOffsetParam(ctx.loopConfig.memSlice, ctx.loopConfig.msInterleave, 1);
             std::vector<ccu::Loop> groupLoops{*loops.loops[0], *loops.loops[1]};
             ccu::LoopGroup group(
                 copySize.parallelParam, offsetConfig, ctx.loopConfig.loopCount, groupLoops);
         }
         return CCU_SUCCESS;
     }
 
     CcuResult ExecuteAllGather(CcuAllGatherContext &ctx)
     {
         ccu::LocalAddr src;
         src.addr = ctx.input;
         src.token = ctx.inputToken;
 
         std::vector<ccu::RemoteAddr> remoteDst(ctx.arg->channelCount);
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             remoteDst[channelIdx].addr = ctx.peerOutputs[channelIdx];
             remoteDst[channelIdx].addr += ctx.currentRankOutputOffset;
             remoteDst[channelIdx].token = ctx.peerOutputTokens[channelIdx];
         }
 
         uint16_t completionWaitMask = 0;
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             const uint32_t peerRank = ctx.arg->peerRanks[channelIdx];
             const uint16_t completionMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
             CCU_KERNEL_CHECK(ccu::Write(ctx.arg->channels[channelIdx], remoteDst[channelIdx], src, ctx.sliceSize,
                 ctx.completionEvent, completionMask));
             completionWaitMask = static_cast<uint16_t>(completionWaitMask | completionMask);
         }
 
         if (ctx.arg->handleLocalCopy != 0) {
             ccu::LocalAddr localDst;
             localDst.addr = ctx.localOutput;
             localDst.addr += ctx.currentRankOutputOffset;
             localDst.token = ctx.localOutputToken;
             const uint16_t localCompletionMask = static_cast<uint16_t>(uint32_t{1} << ctx.arg->rankId);
             CCU_IF(ctx.skipLocalCopy == 0)
             {
                 CCU_KERNEL_CHECK(
                     ccu::LocalCopy(localDst, src, ctx.sliceSize, ctx.completionEvent, localCompletionMask));
             }
             CCU_IF(ctx.skipLocalCopy != 0)
             {
                 CCU_KERNEL_CHECK(ccu::EventRecord(ctx.completionEvent, localCompletionMask));
             }
             completionWaitMask = static_cast<uint16_t>(completionWaitMask | localCompletionMask);
         }
 
         CCU_KERNEL_CHECK(ccu::EventWait(ctx.completionEvent, completionWaitMask));
         return CCU_SUCCESS;
     }

     CcuResult ExecuteGroupCopyAllGather(CcuGroupCopyAllGatherContext &ctx)
     {
         ccu::LocalAddr src;
         src.addr = ctx.input;
         src.token = ctx.inputToken;

         std::vector<ccu::RemoteAddr> remoteDst(ctx.arg->channelCount);
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             remoteDst[channelIdx].addr = ctx.peerOutputs[channelIdx];
             remoteDst[channelIdx].addr += ctx.currentRankOutputOffset;
             remoteDst[channelIdx].token = ctx.peerOutputTokens[channelIdx];
         }

         uint16_t completionWaitMask = 0;
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             const uint32_t peerRank = ctx.arg->peerRanks[channelIdx];
             const uint16_t completionMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
             CCU_KERNEL_CHECK(ccu::Write(ctx.arg->channels[channelIdx], remoteDst[channelIdx], src, ctx.sliceSize,
                 ctx.completionEvent, completionMask));
             completionWaitMask = static_cast<uint16_t>(completionWaitMask | completionMask);
         }

         if (ctx.arg->handleLocalCopy != 0) {
             ccu::LocalAddr localDst;
             localDst.addr = ctx.localOutput;
             localDst.addr += ctx.currentRankOutputOffset;
             localDst.token = ctx.localOutputToken;
             const uint16_t localCompletionMask = static_cast<uint16_t>(uint32_t{1} << ctx.arg->rankId);
             CCU_IF(ctx.skipLocalCopy == 0)
             {
                 CCU_KERNEL_CHECK(GroupCopy(ctx, localDst, src, ctx.groupCopySize));
                 CCU_KERNEL_CHECK(ccu::EventRecord(ctx.completionEvent, localCompletionMask));
             }
             CCU_IF(ctx.skipLocalCopy != 0)
             {
                 CCU_KERNEL_CHECK(ccu::EventRecord(ctx.completionEvent, localCompletionMask));
             }
             completionWaitMask = static_cast<uint16_t>(completionWaitMask | localCompletionMask);
         }

         CCU_KERNEL_CHECK(ccu::EventWait(ctx.completionEvent, completionWaitMask));
         return CCU_SUCCESS;
     }
 
     CcuResult ExecuteDualSliceAllGather(CcuAllGatherDualSliceContext &ctx)
     {
         ccu::LocalAddr firstSrc;
         firstSrc.addr = ctx.input;
         firstSrc.token = ctx.inputToken;
 
         ccu::LocalAddr secondSrc;
         secondSrc.addr = ctx.input;
         secondSrc.addr += ctx.sliceSize;
         secondSrc.token = ctx.inputToken;
 
         std::vector<ccu::RemoteAddr> remoteDst(ctx.arg->channelCount);
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             remoteDst[channelIdx].addr = ctx.peerOutputs[channelIdx];
             remoteDst[channelIdx].addr += ctx.currentRankOutputOffset;
             remoteDst[channelIdx].token = ctx.peerOutputTokens[channelIdx];
         }
 
         uint16_t completionWaitMask = 0;
         // 两批Write使用不同Event；先全部提交，再统一等待，减少切片间流水气泡。
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             const uint32_t peerRank = ctx.arg->peerRanks[channelIdx];
             const uint16_t completionMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
             CCU_KERNEL_CHECK(ccu::Write(ctx.arg->channels[channelIdx], remoteDst[channelIdx], firstSrc, ctx.sliceSize,
                 ctx.completionEvent, completionMask));
             completionWaitMask = static_cast<uint16_t>(completionWaitMask | completionMask);
         }
 
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             remoteDst[channelIdx].addr += ctx.sliceSize;
             const uint32_t peerRank = ctx.arg->peerRanks[channelIdx];
             const uint16_t completionMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
             CCU_KERNEL_CHECK(ccu::Write(ctx.arg->channels[channelIdx], remoteDst[channelIdx], secondSrc,
                 ctx.secondSliceSize, ctx.secondCompletionEvent, completionMask));
         }
 
         if (ctx.arg->handleLocalCopy != 0) {
             ccu::LocalAddr firstLocalDst;
             firstLocalDst.addr = ctx.localOutput;
             firstLocalDst.addr += ctx.currentRankOutputOffset;
             firstLocalDst.token = ctx.localOutputToken;
 
             ccu::LocalAddr secondLocalDst;
             secondLocalDst.addr = ctx.localOutput;
             secondLocalDst.addr += ctx.currentRankOutputOffset;
             secondLocalDst.addr += ctx.sliceSize;
             secondLocalDst.token = ctx.localOutputToken;
 
             const uint16_t localCompletionMask = static_cast<uint16_t>(uint32_t{1} << ctx.arg->rankId);
             CCU_IF(ctx.skipLocalCopy == 0)
             {
                 CCU_KERNEL_CHECK(
                     ccu::LocalCopy(firstLocalDst, firstSrc, ctx.sliceSize, ctx.completionEvent, localCompletionMask));
                 CCU_KERNEL_CHECK(ccu::LocalCopy(
                     secondLocalDst, secondSrc, ctx.secondSliceSize, ctx.secondCompletionEvent, localCompletionMask));
             }
             CCU_IF(ctx.skipLocalCopy != 0)
             {
                 CCU_KERNEL_CHECK(ccu::EventRecord(ctx.completionEvent, localCompletionMask));
                 CCU_KERNEL_CHECK(ccu::EventRecord(ctx.secondCompletionEvent, localCompletionMask));
             }
             completionWaitMask = static_cast<uint16_t>(completionWaitMask | localCompletionMask);
         }
 
         CCU_KERNEL_CHECK(ccu::EventWait(ctx.completionEvent, completionWaitMask));
         CCU_KERNEL_CHECK(ccu::EventWait(ctx.secondCompletionEvent, completionWaitMask));
         return CCU_SUCCESS;
     }
 
     CcuResult ExecuteParallelRepeat(CcuParallelRepeatContext &ctx)
     {
         uint16_t completionWaitMask = 0;
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             completionWaitMask = static_cast<uint16_t>(
                 completionWaitMask | static_cast<uint16_t>(uint32_t{1} << ctx.arg->peerRanks[channelIdx]));
         }
 
         // 所有数据块、所有Peer的Write全部提交后再等待，
         // 保持不同对端和不同数据块之间的并行。
         for (uint32_t blockIdx = 0; blockIdx < ctx.arg->blockCount; ++blockIdx) {
             ccu::LocalAddr src;
             src.addr = ctx.localOutput;
             src.addr += ctx.blockOffsets[blockIdx];
             src.token = ctx.localOutputToken;
             for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                 ccu::RemoteAddr remoteDst;
                 remoteDst.addr = ctx.peerOutputs[channelIdx];
                 remoteDst.addr += ctx.blockOffsets[blockIdx];
                 remoteDst.token = ctx.peerOutputTokens[channelIdx];
 
                 const uint16_t completionMask = static_cast<uint16_t>(uint32_t{1} << ctx.arg->peerRanks[channelIdx]);
                 CCU_KERNEL_CHECK(ccu::Write(ctx.arg->channels[channelIdx], remoteDst, src, ctx.sliceSize,
                     ctx.completionEvents[blockIdx], completionMask));
             }
         }
 
         for (uint32_t blockIdx = 0; blockIdx < ctx.arg->blockCount; ++blockIdx) {
             CCU_KERNEL_CHECK(ccu::EventWait(ctx.completionEvents[blockIdx], completionWaitMask));
         }
         return CCU_SUCCESS;
     }
 
     CcuResult ExecuteStripedAllGather(CcuStripedAllGatherContext &ctx)
     {
         uint16_t completionWaitMask = 0;
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             const uint32_t globalStripeIdx = ctx.arg->stripeBaseIndex + channelIdx;
             ccu::Variable stripeOffset;
             stripeOffset = 0;
             for (uint32_t stripeIdx = 0; stripeIdx < globalStripeIdx; ++stripeIdx) {
                 stripeOffset += ctx.stripeSize;
             }
 
             ccu::LocalAddr src;
             src.addr = ctx.input;
             src.addr += stripeOffset;
             src.token = ctx.inputToken;
 
             ccu::RemoteAddr remoteDst;
             remoteDst.addr = ctx.peerOutputs[channelIdx];
             remoteDst.addr += ctx.currentRankOutputOffset;
             remoteDst.addr += stripeOffset;
             remoteDst.token = ctx.peerOutputTokens[channelIdx];
 
             ccu::LocalAddr localDst;
             localDst.addr = ctx.localOutput;
             localDst.addr += ctx.currentRankOutputOffset;
             localDst.addr += stripeOffset;
             localDst.token = ctx.localOutputToken;
 
             ccu::Variable &transferSize
                 = globalStripeIdx + 1 == ctx.arg->stripeCount ? ctx.lastStripeSize : ctx.stripeSize;
             const uint16_t completionMask = static_cast<uint16_t>(uint32_t{1} << channelIdx);
             CCU_IF(transferSize != 0)
             {
                 CCU_KERNEL_CHECK(ccu::Write(
                     ctx.arg->channels[channelIdx], remoteDst, src, transferSize, ctx.completionEvent, completionMask));
                 CCU_IF(ctx.skipLocalCopy == 0)
                 {
                     CCU_KERNEL_CHECK(
                         ccu::LocalCopy(localDst, src, transferSize, ctx.localCompletionEvent, completionMask));
                 }
                 CCU_IF(ctx.skipLocalCopy != 0)
                 {
                     CCU_KERNEL_CHECK(ccu::EventRecord(ctx.localCompletionEvent, completionMask));
                 }
             }
             CCU_IF(transferSize == 0)
             {
                 CCU_KERNEL_CHECK(ccu::EventRecord(ctx.completionEvent, completionMask));
                 CCU_KERNEL_CHECK(ccu::EventRecord(ctx.localCompletionEvent, completionMask));
             }
             completionWaitMask = static_cast<uint16_t>(completionWaitMask | completionMask);
         }
 
         CCU_KERNEL_CHECK(ccu::EventWait(ctx.completionEvent, completionWaitMask));
         CCU_KERNEL_CHECK(ccu::EventWait(ctx.localCompletionEvent, completionWaitMask));
         return CCU_SUCCESS;
     }
 
     CcuResult ExecuteParallelStripedRepeat(CcuParallelStripedRepeatContext &ctx)
     {
         uint16_t completionWaitMask = 0;
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             completionWaitMask
                 = static_cast<uint16_t>(completionWaitMask | static_cast<uint16_t>(uint32_t{1} << channelIdx));
         }
 
         // 每个物理Channel只发送各数据块中与其编号对应的条带；
         // 所有块、所有Channel全部提交后再统一等待。
         for (uint32_t blockIdx = 0; blockIdx < ctx.arg->blockCount; ++blockIdx) {
             for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                 const uint32_t globalStripeIdx = ctx.arg->stripeBaseIndex + channelIdx;
                 ccu::Variable stripeOffset;
                 stripeOffset = 0;
                 for (uint32_t stripeIdx = 0; stripeIdx < globalStripeIdx; ++stripeIdx) {
                     stripeOffset += ctx.stripeSize;
                 }
 
                 ccu::LocalAddr src;
                 src.addr = ctx.localOutput;
                 src.addr += ctx.blockOffsets[blockIdx];
                 src.addr += stripeOffset;
                 src.token = ctx.localOutputToken;
 
                 ccu::RemoteAddr remoteDst;
                 remoteDst.addr = ctx.peerOutputs[channelIdx];
                 remoteDst.addr += ctx.blockOffsets[blockIdx];
                 remoteDst.addr += stripeOffset;
                 remoteDst.token = ctx.peerOutputTokens[channelIdx];
 
                 ccu::Variable &transferSize
                     = globalStripeIdx + 1 == ctx.arg->stripeCount ? ctx.lastStripeSize : ctx.stripeSize;
                 const uint16_t completionMask = static_cast<uint16_t>(uint32_t{1} << channelIdx);
                 CCU_IF(transferSize != 0)
                 {
                     CCU_KERNEL_CHECK(ccu::Write(ctx.arg->channels[channelIdx], remoteDst, src, transferSize,
                         ctx.completionEvents[blockIdx], completionMask));
                 }
                 CCU_IF(transferSize == 0)
                 {
                     CCU_KERNEL_CHECK(ccu::EventRecord(ctx.completionEvents[blockIdx], completionMask));
                 }
             }
         }
 
         for (uint32_t blockIdx = 0; blockIdx < ctx.arg->blockCount; ++blockIdx) {
             CCU_KERNEL_CHECK(ccu::EventWait(ctx.completionEvents[blockIdx], completionWaitMask));
         }
         return CCU_SUCCESS;
     }
 
     CcuResult ExecuteAsym8x4Stage0Inter(CcuAsym8x4Stage0InterContext &ctx)
     {
         const bool isLargeServerRank = ctx.arg->rankId < ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM;
         uint16_t completionWaitMask = 0;
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             ccu::LocalAddr src;
             src.addr = ctx.input;
             src.token = ctx.inputToken;
 
             ccu::RemoteAddr remoteDst;
             remoteDst.addr = ctx.peerOutputs[channelIdx];
             remoteDst.addr += ctx.currentRankOutputOffset;
             remoteDst.token = ctx.peerOutputTokens[channelIdx];
 
             const uint32_t peerRank = ctx.arg->peerRanks[channelIdx];
             const uint16_t completionMask = static_cast<uint16_t>(uint32_t{1} << peerRank);
             if (isLargeServerRank) {
                 CCU_KERNEL_CHECK(ccu::Write(ctx.arg->channels[channelIdx], remoteDst, src, ctx.hierarchySize,
                     ctx.completionEvent, completionMask));
             } else {
                 if ((peerRank & 1U) != 0) {
                     src.addr += ctx.halfSize;
                     remoteDst.addr += ctx.halfSize;
                 }
                 CCU_KERNEL_CHECK(ccu::Write(
                     ctx.arg->channels[channelIdx], remoteDst, src, ctx.halfSize, ctx.completionEvent, completionMask));
             }
             completionWaitMask = static_cast<uint16_t>(completionWaitMask | completionMask);
         }
         CCU_KERNEL_CHECK(ccu::EventWait(ctx.completionEvent, completionWaitMask));
         return CCU_SUCCESS;
     }
 
     CcuResult ExecuteAsym8x4Stage1Intra(CcuAsym8x4Stage1IntraContext &ctx)
     {
         uint16_t peerCompletionMask = 0;
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             peerCompletionMask = static_cast<uint16_t>(
                 peerCompletionMask | static_cast<uint16_t>(uint32_t{1} << ctx.arg->peerRanks[channelIdx]));
         }
 
         ccu::LocalAddr ownSrc;
         ownSrc.addr = ctx.input;
         ownSrc.addr += ctx.localLeadSize;
         ownSrc.token = ctx.inputToken;
         for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
             ccu::RemoteAddr remoteDst;
             remoteDst.addr = ctx.peerOutputs[channelIdx];
             remoteDst.addr += ctx.currentRankOutputOffset;
             remoteDst.addr += ctx.localLeadSize;
             remoteDst.token = ctx.peerOutputTokens[channelIdx];
             const uint16_t completionMask = static_cast<uint16_t>(uint32_t{1} << ctx.arg->peerRanks[channelIdx]);
             CCU_KERNEL_CHECK(ccu::Write(ctx.arg->channels[channelIdx], remoteDst, ownSrc, ctx.localRemainingSize,
                 ctx.localDataCompletionEvent, completionMask));
         }
 
         for (uint32_t blockIdx = 0; blockIdx < ctx.arg->blockCount; ++blockIdx) {
             ccu::LocalAddr forwardSrc;
             forwardSrc.addr = ctx.localOutput;
             forwardSrc.addr += ctx.forwardBlockOffsets[blockIdx];
             forwardSrc.token = ctx.localOutputToken;
             for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                 ccu::RemoteAddr remoteDst;
                 remoteDst.addr = ctx.peerOutputs[channelIdx];
                 remoteDst.addr += ctx.forwardBlockOffsets[blockIdx];
                 remoteDst.token = ctx.peerOutputTokens[channelIdx];
                 const uint16_t completionMask = static_cast<uint16_t>(uint32_t{1} << ctx.arg->peerRanks[channelIdx]);
                 CCU_KERNEL_CHECK(ccu::Write(ctx.arg->channels[channelIdx], remoteDst, forwardSrc, ctx.forwardSize,
                     ctx.forwardCompletionEvents[blockIdx], completionMask));
             }
         }
 
         uint16_t localDataCompletionMask = peerCompletionMask;
         if (ctx.arg->handleLocalCopy != 0) {
             ccu::LocalAddr localCopySrc;
             localCopySrc.addr = ctx.input;
             localCopySrc.token = ctx.inputToken;
             ccu::LocalAddr localDst;
             localDst.addr = ctx.localOutput;
             localDst.addr += ctx.currentRankOutputOffset;
             localDst.token = ctx.localOutputToken;
             const uint16_t localCompletionMask = static_cast<uint16_t>(uint32_t{1} << ctx.arg->rankId);
             CCU_IF(ctx.skipLocalCopy == 0)
             {
                 CCU_KERNEL_CHECK(ccu::LocalCopy(
                     localDst, localCopySrc, ctx.totalSize, ctx.localDataCompletionEvent, localCompletionMask));
             }
             CCU_IF(ctx.skipLocalCopy != 0)
             {
                 CCU_KERNEL_CHECK(ccu::EventRecord(ctx.localDataCompletionEvent, localCompletionMask));
             }
             localDataCompletionMask = static_cast<uint16_t>(localDataCompletionMask | localCompletionMask);
         }
 
         CCU_KERNEL_CHECK(ccu::EventWait(ctx.localDataCompletionEvent, localDataCompletionMask));
         for (uint32_t blockIdx = 0; blockIdx < ctx.arg->blockCount; ++blockIdx) {
             CCU_KERNEL_CHECK(ccu::EventWait(ctx.forwardCompletionEvents[blockIdx], peerCompletionMask));
         }
         return CCU_SUCCESS;
     }
 
     CcuResult SynchronizePeers(const CcuKernelArgAllGather *arg)
     {
         for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
             CCU_KERNEL_CHECK(ccu::NotifyRecord(arg->channels[channelIdx], CHANNEL_NOTIFY_IDX, POST_SYNC_MASK));
         }
         for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
             CCU_KERNEL_CHECK(ccu::NotifyWait(arg->channels[channelIdx], CHANNEL_NOTIFY_IDX, POST_SYNC_MASK));
         }
         return CCU_SUCCESS;
     }
 } // namespace
 
 CcuResult CcuKernel(CcuKernelArg arg)
 {
     auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
     CcuAllGatherContext ctx;
     ctx.arg = kernelArg;
 
     CCU_KERNEL_CHECK(InitResources(ctx));
     CCU_KERNEL_CHECK(LoadTaskArgs(ctx));
     CCU_KERNEL_CHECK(ExchangeOutputInfo(ctx.arg, ctx.localOutput, ctx.localOutputToken));
     CCU_IF(ctx.sliceSize != 0)
     {
         CCU_KERNEL_CHECK(ExecuteAllGather(ctx));
     }
     CCU_KERNEL_CHECK(SynchronizePeers(ctx.arg));
 
     return CCU_SUCCESS;
 }

 CcuResult CcuGroupCopyKernel(CcuKernelArg arg)
 {
     auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
     if (kernelArg == nullptr || kernelArg->rankSize != GROUP_COPY_RANK_SIZE) {
         return CCU_E_PARA;
     }

     CcuGroupCopyAllGatherContext ctx;
     ctx.arg = kernelArg;

     CCU_KERNEL_CHECK(InitResources(ctx));
     CCU_KERNEL_CHECK(LoadGroupCopyTaskArgs(ctx));
     CCU_KERNEL_CHECK(ExchangeOutputInfo(ctx.arg, ctx.localOutput, ctx.localOutputToken));
     CCU_IF(ctx.sliceSize != 0)
     {
         CCU_KERNEL_CHECK(ExecuteGroupCopyAllGather(ctx));
     }
     CCU_KERNEL_CHECK(SynchronizePeers(ctx.arg));

     return CCU_SUCCESS;
 }

 CcuResult CcuDualSliceKernel(CcuKernelArg arg)
 {
     auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
     CcuAllGatherDualSliceContext ctx;
     ctx.arg = kernelArg;
 
     CCU_KERNEL_CHECK(InitResources(ctx));
     CCU_KERNEL_CHECK(LoadDualSliceTaskArgs(ctx));
     CCU_KERNEL_CHECK(ExchangeOutputInfo(ctx.arg, ctx.localOutput, ctx.localOutputToken));
     CCU_IF(ctx.sliceSize != 0)
     {
         CCU_KERNEL_CHECK(ExecuteDualSliceAllGather(ctx));
     }
     CCU_KERNEL_CHECK(SynchronizePeers(ctx.arg));
 
     return CCU_SUCCESS;
 }
 
 CcuResult CcuParallelRepeatKernel(CcuKernelArg arg)
 {
     auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
     CcuParallelRepeatContext ctx;
     ctx.arg = kernelArg;
 
     CCU_KERNEL_CHECK(InitParallelRepeatResources(ctx));
     CCU_KERNEL_CHECK(LoadParallelRepeatTaskArgs(ctx));
     CCU_KERNEL_CHECK(ExchangeOutputInfo(ctx.arg, ctx.localOutput, ctx.localOutputToken));
     CCU_IF(ctx.sliceSize != 0)
     {
         CCU_KERNEL_CHECK(ExecuteParallelRepeat(ctx));
     }
     CCU_KERNEL_CHECK(SynchronizePeers(ctx.arg));
 
     return CCU_SUCCESS;
 }
 
 CcuResult CcuStripedAllGatherKernel(CcuKernelArg arg)
 {
     auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
     CcuStripedAllGatherContext ctx;
     ctx.arg = kernelArg;
 
     CCU_KERNEL_CHECK(InitStripedAllGatherResources(ctx));
     CCU_KERNEL_CHECK(LoadStripedAllGatherTaskArgs(ctx));
     CCU_KERNEL_CHECK(ExchangeOutputInfo(ctx.arg, ctx.localOutput, ctx.localOutputToken));
     CCU_KERNEL_CHECK(ExecuteStripedAllGather(ctx));
     CCU_KERNEL_CHECK(SynchronizePeers(ctx.arg));
 
     return CCU_SUCCESS;
 }
 
 CcuResult CcuParallelStripedRepeatKernel(CcuKernelArg arg)
 {
     auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
     CcuParallelStripedRepeatContext ctx;
     ctx.arg = kernelArg;
 
     CCU_KERNEL_CHECK(InitParallelStripedRepeatResources(ctx));
     CCU_KERNEL_CHECK(LoadParallelStripedRepeatTaskArgs(ctx));
     CCU_KERNEL_CHECK(ExchangeOutputInfo(ctx.arg, ctx.localOutput, ctx.localOutputToken));
     CCU_KERNEL_CHECK(ExecuteParallelStripedRepeat(ctx));
     CCU_KERNEL_CHECK(SynchronizePeers(ctx.arg));
 
     return CCU_SUCCESS;
 }
 
 CcuResult CcuAsym8x4Stage0InterKernel(CcuKernelArg arg)
 {
     auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
     CcuAsym8x4Stage0InterContext ctx;
     ctx.arg = kernelArg;
 
     CCU_KERNEL_CHECK(InitAsym8x4Stage0InterResources(ctx));
     CCU_KERNEL_CHECK(LoadAsym8x4Stage0InterTaskArgs(ctx));
     CCU_KERNEL_CHECK(ExchangeOutputInfo(ctx.arg, ctx.localOutput, ctx.localOutputToken));
     CCU_KERNEL_CHECK(ExecuteAsym8x4Stage0Inter(ctx));
     CCU_KERNEL_CHECK(SynchronizePeers(ctx.arg));
 
     return CCU_SUCCESS;
 }
 
 CcuResult CcuAsym8x4Stage1IntraKernel(CcuKernelArg arg)
 {
     auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
     CcuAsym8x4Stage1IntraContext ctx;
     ctx.arg = kernelArg;
 
     CCU_KERNEL_CHECK(InitAsym8x4Stage1IntraResources(ctx));
     CCU_KERNEL_CHECK(LoadAsym8x4Stage1IntraTaskArgs(ctx));
     CCU_KERNEL_CHECK(ExchangeOutputInfo(ctx.arg, ctx.localOutput, ctx.localOutputToken));
     CCU_KERNEL_CHECK(ExecuteAsym8x4Stage1Intra(ctx));
     CCU_KERNEL_CHECK(SynchronizePeers(ctx.arg));
 
     return CCU_SUCCESS;
 }
 
 #undef CCU_KERNEL_CHECK
 } // namespace ops_hccl
