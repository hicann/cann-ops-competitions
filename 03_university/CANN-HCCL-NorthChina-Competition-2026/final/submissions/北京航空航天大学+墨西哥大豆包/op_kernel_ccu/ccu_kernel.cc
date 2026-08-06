/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel.h"

#define ALLGATHER_CCU_CHK_RET(call) \
    do { \
        CcuResult ccuRet = static_cast<CcuResult>(call); \
        if (UNLIKELY(ccuRet != CCU_SUCCESS)) { \
            HCCL_ERROR("[%s] call trace: ccuRet -> %d", __func__, ccuRet); \
            return ccuRet; \
        } \
    } while (0)

namespace ops_hccl {
namespace {

    constexpr uint32_t OUTPUT_XN_ID = 1;
    constexpr uint32_t TOKEN_XN_ID = 2;
    constexpr uint32_t CKE_IDX_0 = 0;
    constexpr uint32_t POST_SYNC_ID = 3;
    constexpr uint64_t SMALL_512_DATA_SIZE = 512ULL * 1024;
    constexpr uint32_t NHR_STEP_SYNC_ID = 4;
    constexpr uint32_t NHR_RANK_SIZE = 4;
    constexpr uint32_t NHR_CHANNEL_COUNT = 3;
    constexpr uint32_t NHR_NET_LAYER = 1;
    constexpr uint32_t NHR_2X8_RANK_SIZE = 16;
    constexpr uint32_t NHR_2X8_SERVER_RANK_SIZE = 8;
    constexpr uint32_t NHR_2X8_MESH_CHANNEL_COUNT = 7;
    constexpr uint32_t NHR_2X8_CLOS_CHANNEL_COUNT = 8;
    constexpr uint32_t NHR_2X8_MESH_LAYER = 0;
    constexpr uint32_t NHR_2X8_CLOS_LAYER = 1;
    constexpr uint32_t NHR_2X8_PHASE0_CLOS_TARGET_COUNT = 3;
    constexpr uint32_t NHR_2X8_PHASE1_CLOS_TARGET_COUNT = 5;
    constexpr uint32_t NHR_2X8_PHASE0_CLOS_OFFSETS[NHR_2X8_PHASE0_CLOS_TARGET_COUNT] = {0, 7, 1};
    constexpr uint32_t NHR_2X8_PHASE1_CLOS_OFFSETS[NHR_2X8_PHASE1_CLOS_TARGET_COUNT] = {2, 3, 4, 5, 6};
    constexpr uint16_t NHR_2X8_CLOS_PHASE0_PART0_MASK =
        (1U << NHR_2X8_PHASE0_CLOS_TARGET_COUNT) - 1;
    constexpr uint16_t NHR_2X8_CLOS_PHASE0_PART1_REMOTE_BIT =
        1U << NHR_2X8_PHASE0_CLOS_TARGET_COUNT;
    constexpr uint16_t NHR_2X8_CLOS_PHASE0_PART1_LOCAL_BIT =
        1U << (NHR_2X8_PHASE0_CLOS_TARGET_COUNT + 1);
    constexpr uint16_t NHR_2X8_CLOS_PHASE0_PART1_MASK =
        NHR_2X8_CLOS_PHASE0_PART1_REMOTE_BIT | NHR_2X8_CLOS_PHASE0_PART1_LOCAL_BIT;
    constexpr uint16_t NHR_2X8_MESH_PHASE0_LOCAL_BIT = 1U << NHR_2X8_MESH_CHANNEL_COUNT;
    constexpr uint16_t NHR_2X8_MESH_PHASE0_MASK =
        (1U << (NHR_2X8_MESH_CHANNEL_COUNT + 1)) - 1;
    constexpr uint16_t NHR_2X8_CLOS_FORWARD_MASK =
        (1U << NHR_2X8_PHASE1_CLOS_TARGET_COUNT) - 1;
    constexpr uint16_t NHR_2X8_MESH_FORWARD_MASK =
        (1U << (2 * NHR_2X8_MESH_CHANNEL_COUNT)) - 1;
    constexpr uint16_t NHR_DATA_BIT_0 = 1U << 0;
    constexpr uint16_t NHR_DATA_BIT_1 = 1U << 1;
    constexpr uint16_t NHR_LOCAL_BIT = 1U << 2;
    constexpr uint32_t EIGHT_PLUS_FOUR_RANK_SIZE = 12;
    constexpr uint32_t EIGHT_PLUS_FOUR_SERVER0_SIZE = 8;
    constexpr uint32_t EIGHT_PLUS_FOUR_SERVER1_SIZE = 4;
    constexpr uint32_t EIGHT_PLUS_FOUR_INTRA_LAYER = 0;
    constexpr uint32_t EIGHT_PLUS_FOUR_INTER_LAYER = 1;
    constexpr uint32_t SMALL_512_FUSED_LAYER = 2;

    constexpr uint64_t EIGHT_PLUS_FOUR_PHASE0 = 0;
    constexpr uint64_t EIGHT_PLUS_FOUR_PHASE1 = 1;

    struct Nhr4x1Plan {
        uint32_t oppositeRank;
        uint32_t nextRank;
        uint32_t prevRank;
        uint32_t oppositeChannel;
        uint32_t nextChannel;
        uint32_t prevChannel;
    };

    struct MeshNhr2x8Plan {
        uint32_t pairRank;
        uint32_t pairChannel;
        uint32_t serverBase;
        uint32_t remoteServerBase;
        uint32_t closChannelByLocalRank[NHR_2X8_SERVER_RANK_SIZE];
    };

    constexpr uint64_t SetBits(uint16_t end)
    {
        return (uint64_t(1) << (end + 1)) - uint64_t(1);
    }

    constexpr uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
    {
        constexpr uint16_t ctxIdBitNum = 8;
        constexpr uint16_t ctxIdShiftBit = 45;
        constexpr uint16_t gsaBitNum = 32;
        constexpr uint16_t gsaShiftBit = 13;
        constexpr uint16_t loopNumBitNum = 13;
        return ((loopCtxId & SetBits(ctxIdBitNum)) << ctxIdShiftBit) | ((gsaOffset & SetBits(gsaBitNum)) << gsaShiftBit)
               | (loopIterNum & SetBits(loopNumBitNum));
    }

    constexpr uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
    {
        constexpr uint16_t repeatBitNum = 7;
        constexpr uint16_t repeatNumShiftBit = 55;
        constexpr uint16_t repeatLoopBitNum = 7;
        constexpr uint16_t repeatLoopShiftBit = 48;
        constexpr uint16_t totalLoopBitNum = 7;
        constexpr uint16_t totalLoopShiftBit = 41;
        return ((repeatNum & SetBits(repeatBitNum)) << repeatNumShiftBit)
               | ((repeatLoopIndex & SetBits(repeatLoopBitNum)) << repeatLoopShiftBit)
               | ((totalLoopNum & SetBits(totalLoopBitNum)) << totalLoopShiftBit);
    }

    constexpr uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
    {
        constexpr uint16_t gsaBitNum = 32;
        constexpr uint16_t gsaShiftBit = 21;
        constexpr uint16_t msBitNum = 11;
        constexpr uint16_t msShiftBit = 10;
        constexpr uint16_t ckeBitNum = 10;
        return ((gsaOffset & SetBits(gsaBitNum)) << gsaShiftBit) | ((msOffset & SetBits(msBitNum)) << msShiftBit)
               | (ckeOffset & SetBits(ckeBitNum));
    }

    void InitGroupCopyResources(AllGatherContext &ctx)
    {
        if (!ctx.resourceAllocated) {
            const bool isNhr2x8 = ctx.arg->topology == AllGatherTopology::TWO_SERVER_EIGHT_NPU;
            ctx.loopConfig.msInterleave = isNhr2x8 ? NHR_2X8_MS_INTERLEAVE : CCU_MS_INTERLEAVE;
            ctx.loopConfig.loopCount =
                isNhr2x8 ? NHR_2X8_LOCAL_COPY_LOOP_COUNT : CCU_MS_LOCAL_COPY_LOOP_COUNT;
            ctx.loopConfig.memSlice =
                (isNhr2x8 ? NHR_2X8_LOCAL_COPY_MS_PER_LOOP : CCU_LOCAL_COPY_MS_PER_LOOP) * CCU_MS_SIZE;

            ctx.loopResource.eventCount = ctx.loopConfig.loopCount;
            ctx.loopResource.completedEvent = ccu::Array<ccu::Event>(ctx.loopResource.eventCount);
            ctx.loopResource.bufCount = ctx.loopConfig.loopCount * ctx.loopConfig.msInterleave;
            ctx.loopResource.ccuBuf = ccu::Array<ccu::CcuBuffer>(ctx.loopResource.bufCount);
            ctx.resourceAllocated = true;
        }

        const std::string loopType = "localcopy";
        if (!ctx.IsLoopEntityRegistered(loopType)) {
            ctx.CreateLoopEntity(loopType);
            auto &entity = ctx.loopMap[loopType];
            for (uint32_t index = 0; index < 2; ++index) {
                const uint32_t bufBase = index * ctx.loopConfig.msInterleave;
                ccu::Event loopEvent = ctx.loopResource.completedEvent[index];
                entity.body[index].reset(new ccu::Func([&ctx, index, bufBase, loopEvent]() {
                    ccu::LocalCopy(ctx.loopResource.ccuBuf[bufBase], ctx.groupCopyVars.loopSrc[index],
                        ctx.groupCopyVars.loopLen[index], loopEvent, 1);
                    ccu::EventWait(loopEvent, 1);
                    ccu::LocalCopy(ctx.groupCopyVars.loopDst[index], ctx.loopResource.ccuBuf[bufBase],
                        ctx.groupCopyVars.loopLen[index], loopEvent, 1);
                    ccu::EventWait(loopEvent, 1);
                }));
                entity.loops[index].reset(new ccu::Loop(entity.loopParam[index], *entity.body[index]));
            }
        }
    }

    CcuResult GroupCopy(AllGatherContext &ctx, ccu::LocalAddr dst, ccu::LocalAddr src, GroupOpSizeVars goSize)
    {
        InitGroupCopyResources(ctx);

        auto &loops = ctx.loopMap["localcopy"];
        CCU_IF(goSize.addrOffset != 0)
        {
            ccu::Variable loopParam;
            loopParam = GetLoopParam(0, ctx.loopConfig.memSlice * ctx.loopConfig.loopCount, 0);
            loopParam += goSize.loopParam;

            ccu::Variable sliceSize;
            sliceSize = ctx.loopConfig.memSlice;
            ctx.groupCopyVars.loopSrc[0].addr = src.addr;
            ctx.groupCopyVars.loopSrc[0].token = src.token;
            ctx.groupCopyVars.loopDst[0].addr = dst.addr;
            ctx.groupCopyVars.loopDst[0].token = dst.token;
            ctx.groupCopyVars.loopLen[0] = sliceSize;

            loops.loopParam[0] = loopParam;
            ccu::Variable parallelConfig;
            parallelConfig = GetParallelParam(ctx.loopConfig.loopCount - 1, 0, 1);
            ccu::Variable offsetConfig;
            offsetConfig = GetOffsetParam(ctx.loopConfig.memSlice, ctx.loopConfig.msInterleave, 1);
            std::vector<ccu::Loop> groupLoops{*loops.loops[0]};
            ccu::LoopGroup group(parallelConfig, offsetConfig, ctx.loopConfig.loopCount, groupLoops);
        }

        CCU_IF(goSize.parallelParam != 0)
        {
            src.addr += goSize.addrOffset;
            dst.addr += goSize.addrOffset;

            ctx.groupCopyVars.loopSrc[0].addr = src.addr;
            ctx.groupCopyVars.loopSrc[0].token = src.token;
            ctx.groupCopyVars.loopDst[0].addr = dst.addr;
            ctx.groupCopyVars.loopDst[0].token = dst.token;
            ctx.groupCopyVars.loopLen[0] = goSize.residual;

            src.addr += goSize.residual;
            dst.addr += goSize.residual;

            ccu::Variable sliceSize;
            sliceSize = ctx.loopConfig.memSlice;
            ctx.groupCopyVars.loopSrc[1].addr = src.addr;
            ctx.groupCopyVars.loopSrc[1].token = src.token;
            ctx.groupCopyVars.loopDst[1].addr = dst.addr;
            ctx.groupCopyVars.loopDst[1].token = dst.token;
            ctx.groupCopyVars.loopLen[1] = sliceSize;

            ccu::Variable loopConfig0;
            ccu::Variable loopConfig1;
            loopConfig0 = GetLoopParam(0, 0, 1);
            loopConfig1 = GetLoopParam(0, 0, 1);
            ccu::Variable offsetConfig;
            offsetConfig = GetOffsetParam(ctx.loopConfig.memSlice, ctx.loopConfig.msInterleave, 1);
            loops.loopParam[0] = loopConfig0;
            loops.loopParam[1] = loopConfig1;
            std::vector<ccu::Loop> groupLoops{*loops.loops[0], *loops.loops[1]};
            ccu::LoopGroup group(goSize.parallelParam, offsetConfig, ctx.loopConfig.loopCount, groupLoops);
        }

        return CCU_SUCCESS;
    }

    void InitGroupBroadcastResources(AllGatherContext &ctx)
    {
        if (!ctx.resourceAllocated) {
            if (ctx.arg->topology == AllGatherTopology::TWO_SERVER_EIGHT_NPU) {
                ctx.loopConfig.msInterleave = NHR_2X8_MS_INTERLEAVE;
                ctx.loopConfig.loopCount = NHR_2X8_LOCAL_COPY_LOOP_COUNT;
                ctx.loopConfig.memSlice = NHR_2X8_LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
            } else if (ctx.loopConfig.msInterleave == 0 || ctx.loopConfig.loopCount == 0
                || ctx.loopConfig.memSlice == 0) {
                ctx.loopConfig.msInterleave = CCU_MS_INTERLEAVE;
                ctx.loopConfig.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
                ctx.loopConfig.memSlice = CCU_LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
            }
            ctx.loopResource.eventCount = ctx.loopConfig.loopCount;
            ctx.loopResource.completedEvent = ccu::Array<ccu::Event>(ctx.loopResource.eventCount);
            ctx.loopResource.bufCount = ctx.loopConfig.loopCount * ctx.loopConfig.msInterleave;
            ctx.loopResource.ccuBuf = ccu::Array<ccu::CcuBuffer>(ctx.loopResource.bufCount);
            for (uint32_t index = 0; index < 2; ++index) {
                ctx.groupBroadcastVars.loopRemoteDst[index].resize(ctx.arg->channelCount);
            }
            ctx.resourceAllocated = true;
        }

        const std::string loopType = "broadcast";
        if (!ctx.IsLoopEntityRegistered(loopType)) {
            ctx.CreateLoopEntity(loopType);
            auto &entity = ctx.loopMap[loopType];
            for (uint32_t index = 0; index < 2; ++index) {
                const uint32_t bufBase = index * ctx.loopConfig.msInterleave;
                ccu::Event loopEvent = ctx.loopResource.completedEvent[index];
                entity.body[index].reset(new ccu::Func([&ctx, index, bufBase, loopEvent]() {
                    ccu::LocalCopy(ctx.loopResource.ccuBuf[bufBase], ctx.groupBroadcastVars.loopSrc[index],
                        ctx.groupBroadcastVars.loopLen[index], loopEvent, 1);
                    ccu::EventWait(loopEvent, 1);
                    uint16_t eventMask = 0;
                    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                        const uint16_t remoteMask = static_cast<uint16_t>(1U << channelIdx);
                        ccu::Write(ctx.arg->channels[channelIdx],
                            ctx.groupBroadcastVars.loopRemoteDst[index][channelIdx],
                            ctx.loopResource.ccuBuf[bufBase], ctx.groupBroadcastVars.loopLen[index],
                            loopEvent, remoteMask);
                        eventMask = static_cast<uint16_t>(eventMask | remoteMask);
                    }
                    if (ctx.arg->handleSelfRank != 0) {
                        const uint16_t selfMask = static_cast<uint16_t>(1U << ctx.arg->channelCount);
                        ccu::LocalCopy(ctx.groupBroadcastVars.loopLocalDst[index],
                            ctx.loopResource.ccuBuf[bufBase], ctx.groupBroadcastVars.loopLen[index],
                            loopEvent, selfMask);
                        eventMask = static_cast<uint16_t>(eventMask | selfMask);
                    }
                    ccu::EventWait(loopEvent, eventMask);
                }));
                entity.loops[index].reset(new ccu::Loop(entity.loopParam[index], *entity.body[index]));
            }
        }
    }

    CcuResult GroupBroadcast(AllGatherContext &ctx, ccu::LocalAddr localDst,
        std::vector<ccu::RemoteAddr> remoteDst, ccu::LocalAddr src, GroupOpSizeVars goSize)
    {
        InitGroupBroadcastResources(ctx);
        auto &loops = ctx.loopMap["broadcast"];
        CCU_IF(goSize.addrOffset != 0)
        {
            ccu::Variable loopParam;
            loopParam = GetLoopParam(0, ctx.loopConfig.memSlice * ctx.loopConfig.loopCount, 0);
            loopParam += goSize.loopParam;
            ctx.groupBroadcastVars.loopSrc[0].addr = src.addr;
            ctx.groupBroadcastVars.loopSrc[0].token = src.token;
            ctx.groupBroadcastVars.loopLocalDst[0].addr = localDst.addr;
            ctx.groupBroadcastVars.loopLocalDst[0].token = localDst.token;
            for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                ctx.groupBroadcastVars.loopRemoteDst[0][channelIdx].addr = remoteDst[channelIdx].addr;
                ctx.groupBroadcastVars.loopRemoteDst[0][channelIdx].token = remoteDst[channelIdx].token;
            }
            ctx.groupBroadcastVars.loopLen[0] = ctx.loopConfig.memSlice;
            loops.loopParam[0] = loopParam;
            ccu::Variable parallelConfig;
            parallelConfig = GetParallelParam(ctx.loopConfig.loopCount - 1, 0, 1);
            ccu::Variable offsetConfig;
            offsetConfig = GetOffsetParam(ctx.loopConfig.memSlice, ctx.loopConfig.msInterleave, 1);
            std::vector<ccu::Loop> groupLoops{*loops.loops[0]};
            ccu::LoopGroup group(parallelConfig, offsetConfig, ctx.loopConfig.loopCount, groupLoops);
        }

        CCU_IF(goSize.parallelParam != 0)
        {
            src.addr += goSize.addrOffset;
            localDst.addr += goSize.addrOffset;
            for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                remoteDst[channelIdx].addr += goSize.addrOffset;
            }
            ctx.groupBroadcastVars.loopSrc[0].addr = src.addr;
            ctx.groupBroadcastVars.loopSrc[0].token = src.token;
            ctx.groupBroadcastVars.loopLocalDst[0].addr = localDst.addr;
            ctx.groupBroadcastVars.loopLocalDst[0].token = localDst.token;
            for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                ctx.groupBroadcastVars.loopRemoteDst[0][channelIdx].addr = remoteDst[channelIdx].addr;
                ctx.groupBroadcastVars.loopRemoteDst[0][channelIdx].token = remoteDst[channelIdx].token;
            }
            ctx.groupBroadcastVars.loopLen[0] = goSize.residual;
            src.addr += goSize.residual;
            localDst.addr += goSize.residual;
            for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                remoteDst[channelIdx].addr += goSize.residual;
            }
            ctx.groupBroadcastVars.loopSrc[1].addr = src.addr;
            ctx.groupBroadcastVars.loopSrc[1].token = src.token;
            ctx.groupBroadcastVars.loopLocalDst[1].addr = localDst.addr;
            ctx.groupBroadcastVars.loopLocalDst[1].token = localDst.token;
            for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                ctx.groupBroadcastVars.loopRemoteDst[1][channelIdx].addr = remoteDst[channelIdx].addr;
                ctx.groupBroadcastVars.loopRemoteDst[1][channelIdx].token = remoteDst[channelIdx].token;
            }
            ctx.groupBroadcastVars.loopLen[1] = ctx.loopConfig.memSlice;
            loops.loopParam[0] = GetLoopParam(0, 0, 1);
            loops.loopParam[1] = GetLoopParam(0, 0, 1);
            ccu::Variable offsetConfig;
            offsetConfig = GetOffsetParam(ctx.loopConfig.memSlice, ctx.loopConfig.msInterleave, 1);
            std::vector<ccu::Loop> groupLoops{*loops.loops[0], *loops.loops[1]};
            ccu::LoopGroup group(goSize.parallelParam, offsetConfig, ctx.loopConfig.loopCount, groupLoops);
        }
        return CCU_SUCCESS;
    }

    CcuResult InitResource(AllGatherBaseContext &ctx, uint32_t eventCount)
    {
        if (ctx.arg->rankId >= ctx.arg->rankSize || ctx.arg->channelCount == 0
            || ctx.arg->channelCount >= ctx.arg->rankSize) {
            HCCL_ERROR("[CcuAllGather] invalid rank %u or channel count %u for rank size %llu", ctx.arg->rankId,
                ctx.arg->channelCount, static_cast<unsigned long long>(ctx.arg->rankSize));
            return CcuResult::CCU_E_INTERNAL;
        }

        ctx.output.resize(ctx.arg->rankSize);
        ctx.token.resize(ctx.arg->rankSize);
        std::vector<bool> rankSeen(ctx.arg->rankSize, false);
        rankSeen[ctx.arg->rankId] = true;

        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            const uint32_t remoteRank = ctx.arg->remoteRanks[channelIdx];
            if (remoteRank >= ctx.arg->rankSize || rankSeen[remoteRank]) {
                HCCL_ERROR("[CcuAllGather] invalid or duplicate remote rank %u", remoteRank);
                return CcuResult::CCU_E_INTERNAL;
            }
            rankSeen[remoteRank] = true;
            ctx.output[remoteRank] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[remoteRank] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
        }
        ctx.events.resize(eventCount);
        return CCU_SUCCESS;
    }

    CcuResult BuildNhr4x1Plan(const AllGatherBaseContext &ctx, Nhr4x1Plan &plan)
    {
        if (ctx.arg->topology != AllGatherTopology::FOUR_SERVER_ONE_NPU || ctx.arg->rankSize != NHR_RANK_SIZE
            || ctx.arg->channelCount != NHR_CHANNEL_COUNT || ctx.arg->netLayer != NHR_NET_LAYER
            || ctx.arg->handleSelfRank == 0) {
            HCCL_ERROR("[CcuAllGatherNhr4x1] invalid topology resources");
            return CcuResult::CCU_E_INTERNAL;
        }

        plan.oppositeRank = (ctx.arg->rankId + 2) % NHR_RANK_SIZE;
        plan.nextRank = (ctx.arg->rankId + 1) % NHR_RANK_SIZE;
        plan.prevRank = (ctx.arg->rankId + 3) % NHR_RANK_SIZE;
        plan.oppositeChannel = NHR_CHANNEL_COUNT;
        plan.nextChannel = NHR_CHANNEL_COUNT;
        plan.prevChannel = NHR_CHANNEL_COUNT;
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            const uint32_t remoteRank = ctx.arg->remoteRanks[channelIdx];
            if (remoteRank == plan.oppositeRank) {
                plan.oppositeChannel = channelIdx;
            } else if (remoteRank == plan.nextRank) {
                plan.nextChannel = channelIdx;
            } else if (remoteRank == plan.prevRank) {
                plan.prevChannel = channelIdx;
            }
        }
        if (plan.oppositeChannel == NHR_CHANNEL_COUNT || plan.nextChannel == NHR_CHANNEL_COUNT
            || plan.prevChannel == NHR_CHANNEL_COUNT) {
            HCCL_ERROR("[CcuAllGatherNhr4x1] NHR partner channel is missing");
            return CcuResult::CCU_E_INTERNAL;
        }
        return CCU_SUCCESS;
    }

    CcuResult BuildMeshNhr2x8Plan(const AllGatherBaseContext &ctx, MeshNhr2x8Plan &plan)
    {
        plan.pairRank = ctx.arg->rankId ^ NHR_2X8_SERVER_RANK_SIZE;
        plan.serverBase =
            (ctx.arg->rankId / NHR_2X8_SERVER_RANK_SIZE) * NHR_2X8_SERVER_RANK_SIZE;
        plan.remoteServerBase =
            (plan.pairRank / NHR_2X8_SERVER_RANK_SIZE) * NHR_2X8_SERVER_RANK_SIZE;
        plan.pairChannel = ctx.arg->channelCount;

        if (ctx.arg->topology != AllGatherTopology::TWO_SERVER_EIGHT_NPU
            || ctx.arg->rankSize != NHR_2X8_RANK_SIZE || ctx.arg->rankId >= ctx.arg->rankSize) {
            HCCL_ERROR("[CcuAllGatherMeshNhr2x8] invalid topology resources");
            return CcuResult::CCU_E_INTERNAL;
        }

        if (ctx.arg->netLayer == NHR_2X8_MESH_LAYER) {
            if (ctx.arg->channelCount != NHR_2X8_MESH_CHANNEL_COUNT || ctx.arg->handleSelfRank == 0) {
                HCCL_ERROR("[CcuAllGatherMeshNhr2x8] invalid mesh resources");
                return CcuResult::CCU_E_INTERNAL;
            }

            const uint32_t serverIdx = ctx.arg->rankId / NHR_2X8_SERVER_RANK_SIZE;
            bool localRankSeen[NHR_2X8_SERVER_RANK_SIZE] = {};
            for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                const uint32_t remoteRank = ctx.arg->remoteRanks[channelIdx];
                if (remoteRank >= ctx.arg->rankSize || remoteRank == ctx.arg->rankId
                    || remoteRank / NHR_2X8_SERVER_RANK_SIZE != serverIdx) {
                    HCCL_ERROR("[CcuAllGatherMeshNhr2x8] invalid local remote rank %u", remoteRank);
                    return CcuResult::CCU_E_INTERNAL;
                }
                const uint32_t localRank = remoteRank - plan.serverBase;
                if (localRankSeen[localRank]) {
                    HCCL_ERROR("[CcuAllGatherMeshNhr2x8] duplicate local remote rank %u", remoteRank);
                    return CcuResult::CCU_E_INTERNAL;
                }
                localRankSeen[localRank] = true;
            }

            for (uint32_t localRank = 0; localRank < NHR_2X8_SERVER_RANK_SIZE; ++localRank) {
                const uint32_t globalRank = plan.serverBase + localRank;
                if (globalRank != ctx.arg->rankId && !localRankSeen[localRank]) {
                    HCCL_ERROR("[CcuAllGatherMeshNhr2x8] local remote rank %u is missing", globalRank);
                    return CcuResult::CCU_E_INTERNAL;
                }
            }
            return CCU_SUCCESS;
        }

        if (ctx.arg->netLayer == NHR_2X8_CLOS_LAYER) {
            if (ctx.arg->channelCount != NHR_2X8_CLOS_CHANNEL_COUNT) {
                HCCL_ERROR("[CcuAllGatherMeshNhr2x8] invalid Clos resources");
                return CcuResult::CCU_E_INTERNAL;
            }

            bool remoteLocalRankSeen[NHR_2X8_SERVER_RANK_SIZE] = {};
            for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                const uint32_t remoteRank = ctx.arg->remoteRanks[channelIdx];
                if (remoteRank < plan.remoteServerBase
                    || remoteRank >= plan.remoteServerBase + NHR_2X8_SERVER_RANK_SIZE) {
                    HCCL_ERROR("[CcuAllGatherMeshNhr2x8] invalid remote rank %u", remoteRank);
                    return CcuResult::CCU_E_INTERNAL;
                }
                const uint32_t remoteLocalRank = remoteRank - plan.remoteServerBase;
                if (remoteLocalRankSeen[remoteLocalRank]) {
                    HCCL_ERROR("[CcuAllGatherMeshNhr2x8] duplicate remote rank %u", remoteRank);
                    return CcuResult::CCU_E_INTERNAL;
                }
                remoteLocalRankSeen[remoteLocalRank] = true;
                plan.closChannelByLocalRank[remoteLocalRank] = channelIdx;
            }

            for (uint32_t remoteLocalRank = 0;
                remoteLocalRank < NHR_2X8_SERVER_RANK_SIZE; ++remoteLocalRank) {
                if (!remoteLocalRankSeen[remoteLocalRank]) {
                    HCCL_ERROR("[CcuAllGatherMeshNhr2x8] remote rank %u is missing",
                        plan.remoteServerBase + remoteLocalRank);
                    return CcuResult::CCU_E_INTERNAL;
                }
            }
            plan.pairChannel =
                plan.closChannelByLocalRank[plan.pairRank - plan.remoteServerBase];
            return CCU_SUCCESS;
        }

        HCCL_ERROR("[CcuAllGatherMeshNhr2x8] invalid net layer %u", ctx.arg->netLayer);
        return CcuResult::CCU_E_INTERNAL;
    }

    CcuResult LoadSmallArgs(SmallAllGatherContext &ctx)
    {
        uint32_t argId = 0;
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
        return CCU_SUCCESS;
    }

    CcuResult LoadSmall512FastArgs(Small512FastContext &ctx)
    {
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.input, 0));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], 1));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], 2));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.refreshResources, 3));
        ctx.sliceSize = SMALL_512_DATA_SIZE;
        return CCU_SUCCESS;
    }

    CcuResult LoadLargeArgs(AllGatherContext &ctx, uint32_t &argId)
    {
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.repeatCount, argId++));
        for (uint32_t repeatIdx = 0; repeatIdx < CCU_ALLGATHER_REPEAT_NUM; ++repeatIdx) {
            ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset[repeatIdx], argId++));
            ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset[repeatIdx], argId++));
            ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize[repeatIdx], argId++));
            ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.goSize[repeatIdx].addrOffset, argId++));
            ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.goSize[repeatIdx].loopParam, argId++));
            ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.goSize[repeatIdx].parallelParam, argId++));
            ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.goSize[repeatIdx].residual, argId++));
        }
        return CCU_SUCCESS;
    }

    CcuResult LoadNhr4x1LargeArgs(Nhr4x1LargeContext &ctx)
    {
        uint32_t argId = 0;
        ALLGATHER_CCU_CHK_RET(LoadLargeArgs(ctx, argId));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.rankDataSize, argId++));
        return CCU_SUCCESS;
    }

    CcuResult LoadMeshNhr2x8LargeArgs(MeshNhr2x8LargeContext &ctx)
    {
        uint32_t argId = 0;
        ALLGATHER_CCU_CHK_RET(LoadLargeArgs(ctx, argId));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.phase, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.rankDataSize, argId++));
        return CCU_SUCCESS;
    }

    CcuResult Load8Plus4LargeArgs(EightPlusFourLargeContext &ctx)
    {
        uint32_t argId = 0;
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.rankDataSize, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.batchOffset, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.half0Size, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.half1Size, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.lSize, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.sSize, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.qSize, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.half0GoSize.addrOffset, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.half0GoSize.loopParam, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.half0GoSize.parallelParam, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.half0GoSize.residual, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.half1GoSize.addrOffset, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.half1GoSize.loopParam, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.half1GoSize.parallelParam, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.half1GoSize.residual, argId++));
        ALLGATHER_CCU_CHK_RET(ccu::LoadArg(ctx.phase, argId++));
        return CCU_SUCCESS;
    }

    uint32_t GetSmallChannelIndex(const AllGatherBaseContext &ctx, uint32_t peerStep)
    {
        if (ctx.arg->small512FastMode != 0) {
            return peerStep;
        }
        uint32_t start = ctx.arg->rankId % ctx.arg->channelCount;
        if (ctx.arg->topology == AllGatherTopology::EIGHT_PLUS_FOUR
            && ctx.arg->netLayer == EIGHT_PLUS_FOUR_INTER_LAYER
            && ctx.arg->rankId >= EIGHT_PLUS_FOUR_SERVER0_SIZE) {
            start = 2 * (ctx.arg->rankId - EIGHT_PLUS_FOUR_SERVER0_SIZE);
        }
        return (start + peerStep) % ctx.arg->channelCount;
    }

    CcuResult PublishSmallPeerResources(AllGatherBaseContext &ctx)
    {
        for (uint32_t peerStep = 0; peerStep < ctx.arg->channelCount; ++peerStep) {
            const uint32_t channelIdx = GetSmallChannelIndex(ctx, peerStep);
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
                ctx.output[ctx.arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
                ctx.token[ctx.arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult PublishSmallPullPeerResources(AllGatherBaseContext &ctx)
    {
        for (uint32_t peerStep = 0; peerStep < ctx.arg->channelCount; ++peerStep) {
            const uint32_t channelIdx = GetSmallChannelIndex(ctx, peerStep);
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
                ctx.input, OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
                ctx.token[ctx.arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult PreSync(AllGatherBaseContext &ctx)
    {
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
                ctx.output[ctx.arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIdx], ctx.token[ctx.arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
        }

        constexpr uint32_t allBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            ALLGATHER_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX_0, allBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult PreSyncNhr4x1(AllGatherBaseContext &ctx, const Nhr4x1Plan &plan)
    {
        const uint32_t publishChannels[] = {plan.oppositeChannel, plan.prevChannel};
        for (uint32_t idx = 0; idx < 2; ++idx) {
            const uint32_t channelIdx = publishChannels[idx];
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
                ctx.output[ctx.arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIdx], ctx.token[ctx.arg->rankId], TOKEN_XN_ID, CKE_IDX_0,
                1U << TOKEN_XN_ID));
        }

        constexpr uint32_t allBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
        const uint32_t waitChannels[] = {plan.oppositeChannel, plan.nextChannel};
        for (uint32_t idx = 0; idx < 2; ++idx) {
            ALLGATHER_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[waitChannels[idx]], CKE_IDX_0, allBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult PreSyncMeshNhr2x8Clos(AllGatherBaseContext &ctx, const MeshNhr2x8Plan &plan,
        const uint32_t *targetOffsets, uint32_t targetCount)
    {
        const uint32_t localRank = ctx.arg->rankId - plan.serverBase;
        for (uint32_t targetIdx = 0; targetIdx < targetCount; ++targetIdx) {
            const uint32_t remoteLocalRank =
                (localRank + targetOffsets[targetIdx]) % NHR_2X8_SERVER_RANK_SIZE;
            const uint32_t channelIdx = plan.closChannelByLocalRank[remoteLocalRank];
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
                ctx.output[ctx.arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
            ALLGATHER_CCU_CHK_RET(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIdx], ctx.token[ctx.arg->rankId],
                TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
        }

        constexpr uint32_t allBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t targetIdx = 0; targetIdx < targetCount; ++targetIdx) {
            const uint32_t remoteLocalRank =
                (localRank + targetOffsets[targetIdx]) % NHR_2X8_SERVER_RANK_SIZE;
            const uint32_t channelIdx = plan.closChannelByLocalRank[remoteLocalRank];
            ALLGATHER_CCU_CHK_RET(
                ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX_0, allBits));
        }
        return CCU_SUCCESS;
    }

    ccu::Variable BuildNhrOutputOffset(
        const ccu::Variable &rankDataSize, uint32_t sourceRank, const ccu::Variable &inputOffset)
    {
        ccu::Variable outputOffset;
        outputOffset = inputOffset;
        for (uint32_t rankIdx = 0; rankIdx < sourceRank; ++rankIdx) {
            outputOffset += rankDataSize;
        }
        return outputOffset;
    }

    CcuResult Nhr4x1StageRecord(const AllGatherBaseContext &ctx, const Nhr4x1Plan &plan)
    {
        const ChannelHandle channel = ctx.arg->channels[plan.oppositeChannel];
        ALLGATHER_CCU_CHK_RET(ccu::NotifyRecord(channel, CKE_IDX_0, 1U << NHR_STEP_SYNC_ID));
        return CCU_SUCCESS;
    }

    CcuResult Nhr4x1StageWait(const AllGatherBaseContext &ctx, const Nhr4x1Plan &plan)
    {
        const ChannelHandle channel = ctx.arg->channels[plan.oppositeChannel];
        ALLGATHER_CCU_CHK_RET(ccu::NotifyWait(channel, CKE_IDX_0, 1U << NHR_STEP_SYNC_ID));
        return CCU_SUCCESS;
    }

    uint16_t GetTransferEventMask(const AllGatherBaseContext &ctx)
    {
        uint16_t eventMask = 0;
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            const uint32_t remoteRank = ctx.arg->remoteRanks[channelIdx];
            eventMask = static_cast<uint16_t>(eventMask | (1U << remoteRank));
        }
        if (ctx.arg->handleSelfRank != 0) {
            eventMask = static_cast<uint16_t>(eventMask | (1U << ctx.arg->rankId));
        }
        return eventMask;
    }

    ccu::Variable BuildSmall512RankOffset(uint32_t rank)
    {
        ccu::Variable offset;
        offset = static_cast<uint64_t>(rank) * SMALL_512_DATA_SIZE;
        return offset;
    }

    CcuResult IssueSmallLocalCopy(SmallAllGatherContext &ctx)
    {
        if (ctx.arg->handleSelfRank != 0) {
            ccu::LocalAddr src;
            src.addr = ctx.input;
            src.addr += ctx.currentRankSliceInputOffset;
            src.token = ctx.token[ctx.arg->rankId];

            ccu::LocalAddr localDst;
            localDst.addr = ctx.output[ctx.arg->rankId];
            localDst.addr += ctx.currentRankSliceOutputOffset;
            localDst.token = ctx.token[ctx.arg->rankId];
            const uint16_t selfMask = static_cast<uint16_t>(1U << ctx.arg->rankId);
            ALLGATHER_CCU_CHK_RET(
                ccu::LocalCopy(localDst, src, ctx.sliceSize, ctx.events[0], selfMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult IssueSmallReadyWrites(SmallAllGatherContext &ctx)
    {
        ccu::LocalAddr src;
        src.addr = ctx.input;
        src.addr += ctx.currentRankSliceInputOffset;
        src.token = ctx.token[ctx.arg->rankId];

        constexpr uint32_t allBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t peerStep = 0; peerStep < ctx.arg->channelCount; ++peerStep) {
            const uint32_t channelIdx = GetSmallChannelIndex(ctx, peerStep);
            ALLGATHER_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX_0, allBits));
            const uint32_t remoteRank = ctx.arg->remoteRanks[channelIdx];
            ccu::RemoteAddr remoteDst;
            remoteDst.addr = ctx.output[remoteRank];
            remoteDst.addr += ctx.currentRankSliceOutputOffset;
            remoteDst.token = ctx.token[remoteRank];
            const uint16_t mask = static_cast<uint16_t>(1U << remoteRank);
            ALLGATHER_CCU_CHK_RET(
                ccu::Write(ctx.arg->channels[channelIdx], remoteDst, src, ctx.sliceSize, ctx.events[0], mask));
        }
        return CCU_SUCCESS;
    }

    ccu::Variable BuildSmallOutputOffset(const SmallAllGatherContext &ctx, uint32_t sourceRank)
    {
        ccu::Variable outputOffset;
        outputOffset = ctx.currentRankSliceInputOffset;
        for (uint32_t rankIdx = 0; rankIdx < sourceRank; ++rankIdx) {
            outputOffset += ctx.sliceSize;
        }
        return outputOffset;
    }

    CcuResult IssueSmallReadyReads(SmallAllGatherContext &ctx)
    {
        constexpr uint32_t allBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t peerStep = 0; peerStep < ctx.arg->channelCount; ++peerStep) {
            const uint32_t channelIdx = GetSmallChannelIndex(ctx, peerStep);
            ALLGATHER_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX_0, allBits));
            const uint32_t remoteRank = ctx.arg->remoteRanks[channelIdx];

            ccu::RemoteAddr remoteSrc;
            remoteSrc.addr = ctx.output[remoteRank];
            remoteSrc.addr += ctx.currentRankSliceInputOffset;
            remoteSrc.token = ctx.token[remoteRank];

            ccu::LocalAddr localDst;
            localDst.addr = ctx.output[ctx.arg->rankId];
            localDst.addr += BuildSmallOutputOffset(ctx, remoteRank);
            localDst.token = ctx.token[ctx.arg->rankId];

            const uint16_t mask = static_cast<uint16_t>(1U << remoteRank);
            ALLGATHER_CCU_CHK_RET(
                ccu::Read(ctx.arg->channels[channelIdx], localDst, remoteSrc, ctx.sliceSize, ctx.events[0], mask));
        }
        return CCU_SUCCESS;
    }

    CcuResult IssueSmall512LocalCopy(SmallAllGatherContext &ctx)
    {
        if (ctx.arg->handleSelfRank != 0) {
            ccu::LocalAddr src;
            src.addr = ctx.input;
            src.token = ctx.token[ctx.arg->rankId];

            ccu::LocalAddr localDst;
            localDst.addr = ctx.output[ctx.arg->rankId];
            const ccu::Variable outputOffset =
                BuildSmall512RankOffset(ctx.arg->rankId);
            localDst.addr += outputOffset;
            localDst.token = ctx.token[ctx.arg->rankId];

            const uint16_t selfMask = static_cast<uint16_t>(1U << ctx.arg->rankId);
            ALLGATHER_CCU_CHK_RET(
                ccu::LocalCopy(localDst, src, ctx.sliceSize, ctx.events[0], selfMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult WaitSmallPeerResources(AllGatherBaseContext &ctx)
    {
        constexpr uint32_t allBits =
            (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t peerStep = 0;
            peerStep < ctx.arg->channelCount; ++peerStep) {
            const uint32_t channelIdx =
                GetSmallChannelIndex(ctx, peerStep);
            ALLGATHER_CCU_CHK_RET(ccu::NotifyWait(
                ctx.arg->channels[channelIdx], CKE_IDX_0, allBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult IssueSmall512Writes(SmallAllGatherContext &ctx)
    {
        ccu::LocalAddr src;
        src.addr = ctx.input;
        src.token = ctx.token[ctx.arg->rankId];

        const ccu::Variable outputOffset =
            BuildSmall512RankOffset(ctx.arg->rankId);
        for (uint32_t peerStep = 0; peerStep < ctx.arg->channelCount; ++peerStep) {
            const uint32_t channelIdx = GetSmallChannelIndex(ctx, peerStep);
            const uint32_t remoteRank = ctx.arg->remoteRanks[channelIdx];

            ccu::RemoteAddr remoteDst;
            remoteDst.addr = ctx.output[remoteRank];
            remoteDst.addr += outputOffset;
            remoteDst.token = ctx.token[remoteRank];

            const uint16_t mask =
                static_cast<uint16_t>(1U << remoteRank);
            ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx],
                remoteDst, src, ctx.sliceSize, ctx.events[0], mask));
        }
        return CCU_SUCCESS;
    }

    CcuResult WaitSmallTransfers(SmallAllGatherContext &ctx)
    {
        const uint16_t eventMask = GetTransferEventMask(ctx);
        ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.events[0], eventMask));
        return CCU_SUCCESS;
    }

    CcuResult ExecuteLarge2x8MeshPhase0(MeshNhr2x8LargeContext &ctx)
    {
        CCU_IF(ctx.sliceSize[0] != 0)
        {
            const ccu::Variable selfOutputOffset = BuildNhrOutputOffset(
                ctx.rankDataSize, ctx.arg->rankId, ctx.currentRankSliceInputOffset[0]);

            ccu::LocalAddr src;
            src.addr = ctx.input;
            src.addr += ctx.currentRankSliceInputOffset[0];
            src.token = ctx.token[ctx.arg->rankId];

            for (uint32_t channelIdx = 0; channelIdx < NHR_2X8_MESH_CHANNEL_COUNT; ++channelIdx) {
                const uint32_t remoteRank = ctx.arg->remoteRanks[channelIdx];
                const uint16_t remoteBit = static_cast<uint16_t>(1U << channelIdx);
                ccu::RemoteAddr remoteDst;
                remoteDst.addr = ctx.output[remoteRank];
                remoteDst.addr += selfOutputOffset;
                remoteDst.token = ctx.token[remoteRank];
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDst, src,
                    ctx.sliceSize[0], ctx.events[0], remoteBit));
            }

            ccu::LocalAddr localDst;
            localDst.addr = ctx.output[ctx.arg->rankId];
            localDst.addr += selfOutputOffset;
            localDst.token = ctx.token[ctx.arg->rankId];
            ALLGATHER_CCU_CHK_RET(GroupCopy(ctx, localDst, src, ctx.goSize[0]));
            ALLGATHER_CCU_CHK_RET(ccu::EventRecord(ctx.events[0], NHR_2X8_MESH_PHASE0_LOCAL_BIT));
            ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.events[0], NHR_2X8_MESH_PHASE0_MASK));
        }
        return CCU_SUCCESS;
    }

    CcuResult ExecuteLarge2x8ClosPhase0(
        MeshNhr2x8LargeContext &ctx, const MeshNhr2x8Plan &plan)
    {
        const uint32_t localRank = ctx.arg->rankId - plan.serverBase;
        CCU_IF(ctx.sliceSize[0] != 0)
        {
            const ccu::Variable selfPart0OutputOffset = BuildNhrOutputOffset(
                ctx.rankDataSize, ctx.arg->rankId, ctx.currentRankSliceInputOffset[0]);

            ccu::LocalAddr part0Src;
            part0Src.addr = ctx.input;
            part0Src.addr += ctx.currentRankSliceInputOffset[0];
            part0Src.token = ctx.token[ctx.arg->rankId];

            for (uint32_t targetIdx = 0;
                targetIdx < NHR_2X8_PHASE0_CLOS_TARGET_COUNT; ++targetIdx) {
                const uint32_t remoteLocalRank =
                    (localRank + NHR_2X8_PHASE0_CLOS_OFFSETS[targetIdx])
                    % NHR_2X8_SERVER_RANK_SIZE;
                const uint32_t remoteRank = plan.remoteServerBase + remoteLocalRank;
                const uint32_t channelIdx = plan.closChannelByLocalRank[remoteLocalRank];
                const uint16_t targetBit = static_cast<uint16_t>(1U << targetIdx);

                ccu::RemoteAddr remoteDst;
                remoteDst.addr = ctx.output[remoteRank];
                remoteDst.addr += selfPart0OutputOffset;
                remoteDst.token = ctx.token[remoteRank];
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDst,
                    part0Src, ctx.sliceSize[0], ctx.events[0], targetBit));
            }
        }

        CCU_IF(ctx.sliceSize[1] != 0)
        {
            const ccu::Variable selfPart1OutputOffset = BuildNhrOutputOffset(
                ctx.rankDataSize, ctx.arg->rankId, ctx.currentRankSliceInputOffset[1]);

            ccu::LocalAddr part1Src;
            part1Src.addr = ctx.input;
            part1Src.addr += ctx.currentRankSliceInputOffset[1];
            part1Src.token = ctx.token[ctx.arg->rankId];

            ccu::RemoteAddr pairRemoteDst;
            pairRemoteDst.addr = ctx.output[plan.pairRank];
            pairRemoteDst.addr += selfPart1OutputOffset;
            pairRemoteDst.token = ctx.token[plan.pairRank];
            ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[plan.pairChannel], pairRemoteDst,
                part1Src, ctx.sliceSize[1], ctx.events[0], NHR_2X8_CLOS_PHASE0_PART1_REMOTE_BIT));

            ccu::LocalAddr localDst;
            localDst.addr = ctx.output[ctx.arg->rankId];
            localDst.addr += selfPart1OutputOffset;
            localDst.token = ctx.token[ctx.arg->rankId];
            ALLGATHER_CCU_CHK_RET(GroupCopy(ctx, localDst, part1Src, ctx.goSize[1]));
            ALLGATHER_CCU_CHK_RET(
                ccu::EventRecord(ctx.events[0], NHR_2X8_CLOS_PHASE0_PART1_LOCAL_BIT));
        }

        CCU_IF(ctx.sliceSize[0] != 0)
        {
            ALLGATHER_CCU_CHK_RET(
                ccu::EventWait(ctx.events[0], NHR_2X8_CLOS_PHASE0_PART0_MASK));
        }
        CCU_IF(ctx.sliceSize[1] != 0)
        {
            ALLGATHER_CCU_CHK_RET(
                ccu::EventWait(ctx.events[0], NHR_2X8_CLOS_PHASE0_PART1_MASK));
        }
        return CCU_SUCCESS;
    }

    CcuResult ExecuteLarge2x8ClosPhase1(
        MeshNhr2x8LargeContext &ctx, const MeshNhr2x8Plan &plan)
    {
        CCU_IF(ctx.sliceSize[0] != 0)
        {
            const uint32_t localRank = ctx.arg->rankId - plan.serverBase;
            const ccu::Variable selfOutputOffset = BuildNhrOutputOffset(
                ctx.rankDataSize, ctx.arg->rankId, ctx.currentRankSliceInputOffset[0]);

            ccu::LocalAddr src;
            src.addr = ctx.input;
            src.addr += ctx.currentRankSliceInputOffset[0];
            src.token = ctx.token[ctx.arg->rankId];

            for (uint32_t targetIdx = 0;
                targetIdx < NHR_2X8_PHASE1_CLOS_TARGET_COUNT; ++targetIdx) {
                const uint32_t remoteLocalRank =
                    (localRank + NHR_2X8_PHASE1_CLOS_OFFSETS[targetIdx])
                    % NHR_2X8_SERVER_RANK_SIZE;
                const uint32_t remoteRank = plan.remoteServerBase + remoteLocalRank;
                const uint32_t channelIdx = plan.closChannelByLocalRank[remoteLocalRank];
                const uint16_t targetBit = static_cast<uint16_t>(1U << targetIdx);
                ccu::RemoteAddr remoteDst;
                remoteDst.addr = ctx.output[remoteRank];
                remoteDst.addr += selfOutputOffset;
                remoteDst.token = ctx.token[remoteRank];
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDst, src,
                    ctx.sliceSize[0], ctx.events[0], targetBit));
            }

            ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.events[0], NHR_2X8_CLOS_FORWARD_MASK));
        }
        return CCU_SUCCESS;
    }

    CcuResult ExecuteLarge2x8MeshPhase1(
        MeshNhr2x8LargeContext &ctx, const MeshNhr2x8Plan &plan)
    {
        CCU_IF(ctx.sliceSize[1] != 0)
        {
            const ccu::Variable selfOutputOffset = BuildNhrOutputOffset(
                ctx.rankDataSize, ctx.arg->rankId, ctx.currentRankSliceInputOffset[1]);
            const ccu::Variable pairOutputOffset = BuildNhrOutputOffset(
                ctx.rankDataSize, plan.pairRank, ctx.currentRankSliceInputOffset[1]);

            ccu::LocalAddr selfSrc;
            selfSrc.addr = ctx.output[ctx.arg->rankId];
            selfSrc.addr += selfOutputOffset;
            selfSrc.token = ctx.token[ctx.arg->rankId];

            ccu::LocalAddr pairSrc;
            pairSrc.addr = ctx.output[ctx.arg->rankId];
            pairSrc.addr += pairOutputOffset;
            pairSrc.token = ctx.token[ctx.arg->rankId];

            for (uint32_t channelIdx = 0; channelIdx < NHR_2X8_MESH_CHANNEL_COUNT; ++channelIdx) {
                const uint32_t remoteRank = ctx.arg->remoteRanks[channelIdx];
                const uint16_t selfMask = static_cast<uint16_t>(1U << (2 * channelIdx));
                const uint16_t pairMask = static_cast<uint16_t>(1U << (2 * channelIdx + 1));

                ccu::RemoteAddr selfRemoteDst;
                selfRemoteDst.addr = ctx.output[remoteRank];
                selfRemoteDst.addr += selfOutputOffset;
                selfRemoteDst.token = ctx.token[remoteRank];
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], selfRemoteDst, selfSrc,
                    ctx.sliceSize[1], ctx.events[0], selfMask));

                ccu::RemoteAddr pairRemoteDst;
                pairRemoteDst.addr = ctx.output[remoteRank];
                pairRemoteDst.addr += pairOutputOffset;
                pairRemoteDst.token = ctx.token[remoteRank];
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], pairRemoteDst, pairSrc,
                    ctx.sliceSize[1], ctx.events[0], pairMask));
            }

            ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.events[0], NHR_2X8_MESH_FORWARD_MASK));
        }
        return CCU_SUCCESS;
    }

    CcuResult ExecuteLargeMeshNhr2x8(
        MeshNhr2x8LargeContext &ctx, const MeshNhr2x8Plan &plan)
    {
        CCU_IF(ctx.phase == 0)
        {
            if (ctx.arg->netLayer == NHR_2X8_MESH_LAYER) {
                ALLGATHER_CCU_CHK_RET(ExecuteLarge2x8MeshPhase0(ctx));
            } else {
                ALLGATHER_CCU_CHK_RET(ExecuteLarge2x8ClosPhase0(ctx, plan));
            }
        }
        CCU_ELSE
        {
            if (ctx.arg->netLayer == NHR_2X8_MESH_LAYER) {
                ALLGATHER_CCU_CHK_RET(ExecuteLarge2x8MeshPhase1(ctx, plan));
            } else {
                ALLGATHER_CCU_CHK_RET(ExecuteLarge2x8ClosPhase1(ctx, plan));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult Validate8Plus4Mesh(const AllGatherBaseContext &ctx)
    {
        if (ctx.arg->rankSize != EIGHT_PLUS_FOUR_RANK_SIZE
            || ctx.arg->topology != AllGatherTopology::EIGHT_PLUS_FOUR
            || (ctx.arg->netLayer != EIGHT_PLUS_FOUR_INTRA_LAYER
                && ctx.arg->netLayer != EIGHT_PLUS_FOUR_INTER_LAYER)) {
            HCCL_ERROR("[CcuAllGather8Plus4] invalid mesh topology or layer");
            return CcuResult::CCU_E_PARA;
        }
        return CCU_SUCCESS;
    }

    ccu::Variable Build8Plus4PipelineOffset(const EightPlusFourLargeContext &ctx,
        uint32_t sourceRank, const ccu::Variable &partOffset)
    {
        ccu::Variable offset;
        offset = ctx.batchOffset;
        offset += partOffset;
        for (uint32_t rank = 0; rank < sourceRank; ++rank) {
            offset += ctx.rankDataSize;
        }
        return offset;
    }

    std::vector<ccu::RemoteAddr> Build8Plus4RemoteDst(const AllGatherBaseContext &ctx,
        const ccu::Variable &outputOffset)
    {
        std::vector<ccu::RemoteAddr> remoteDst(ctx.arg->channelCount);
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            const uint32_t remoteRank = ctx.arg->remoteRanks[channelIdx];
            remoteDst[channelIdx].addr = ctx.output[remoteRank];
            remoteDst[channelIdx].addr += outputOffset;
            remoteDst[channelIdx].token = ctx.token[remoteRank];
        }
        return remoteDst;
    }

    uint32_t Find8Plus4Channel(const AllGatherBaseContext &ctx, uint32_t remoteRank)
    {
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            if (ctx.arg->remoteRanks[channelIdx] == remoteRank) {
                return channelIdx;
            }
        }
        return ctx.arg->channelCount;
    }

    CcuResult Allocate8Plus4EventBit(uint16_t &nextEventBit, uint16_t &issuedMask, uint16_t &eventBit)
    {
        if (nextEventBit == 0) {
            HCCL_ERROR("[CcuAllGather8Plus4] exhausted transfer event bits");
            return CcuResult::CCU_E_INTERNAL;
        }
        eventBit = nextEventBit;
        issuedMask = static_cast<uint16_t>(issuedMask | eventBit);
        nextEventBit = static_cast<uint16_t>(nextEventBit << 1);
        return CCU_SUCCESS;
    }

    CcuResult Issue8Plus4Write(AllGatherBaseContext &ctx, uint32_t destinationRank, ccu::LocalAddr src,
        const ccu::Variable &outputOffset, const ccu::Variable &sliceSize, ccu::Event event, uint16_t eventBit)
    {
        const uint32_t channelIdx = Find8Plus4Channel(ctx, destinationRank);
        if (channelIdx == ctx.arg->channelCount) {
            HCCL_ERROR("[CcuAllGather8Plus4] channel to rank %u is missing", destinationRank);
            return CcuResult::CCU_E_INTERNAL;
        }
        ccu::RemoteAddr remoteDst;
        remoteDst.addr = ctx.output[destinationRank];
        remoteDst.addr += outputOffset;
        remoteDst.token = ctx.token[destinationRank];
        ALLGATHER_CCU_CHK_RET(
            ccu::Write(ctx.arg->channels[channelIdx], remoteDst, src, sliceSize, event, eventBit));
        return CCU_SUCCESS;
    }

    uint32_t Get8Plus4Seed4(uint32_t sourceRank)
    {
        return EIGHT_PLUS_FOUR_SERVER0_SIZE + sourceRank / 2;
    }

    uint32_t Get8Plus4LMeshTarget(uint32_t sourceRank)
    {
        const uint32_t pairIndex = sourceRank / 2;
        const uint32_t step = sourceRank % 2 == 0 ? 1 : 2;
        return EIGHT_PLUS_FOUR_SERVER0_SIZE
               + (pairIndex + step) % EIGHT_PLUS_FOUR_SERVER1_SIZE;
    }

    uint32_t Get8Plus4SMeshTarget(uint32_t sourceRank)
    {
        return EIGHT_PLUS_FOUR_SERVER0_SIZE
               + (sourceRank / 2 + 3) % EIGHT_PLUS_FOUR_SERVER1_SIZE;
    }

    CcuResult Issue8Plus4WritesToLocalPeers(AllGatherBaseContext &ctx, ccu::LocalAddr src,
        const ccu::Variable &outputOffset, const ccu::Variable &sliceSize, ccu::Event event,
        uint32_t firstPeer, uint32_t peerCount, uint32_t excludedPeer, uint16_t &nextEventBit,
        uint16_t &issuedMask)
    {
        for (uint32_t peer = firstPeer; peer < firstPeer + peerCount; ++peer) {
            if (peer != ctx.arg->rankId && peer != excludedPeer) {
                uint16_t eventBit = 0;
                ALLGATHER_CCU_CHK_RET(Allocate8Plus4EventBit(nextEventBit, issuedMask, eventBit));
                ALLGATHER_CCU_CHK_RET(
                    Issue8Plus4Write(ctx, peer, src, outputOffset, sliceSize, event, eventBit));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult Execute8Plus4Phase0Intra(EightPlusFourLargeContext &ctx)
    {
        CCU_IF(ctx.half0Size != 0)
        {
            ccu::LocalAddr ownSrc;
            ownSrc.addr = ctx.input;
            ownSrc.addr += ctx.batchOffset;
            ownSrc.token = ctx.token[ctx.arg->rankId];
            ccu::Variable partOffset;
            partOffset = 0;
            const ccu::Variable outputOffset =
                Build8Plus4PipelineOffset(ctx, ctx.arg->rankId, partOffset);
            ccu::LocalAddr localDst;
            localDst.addr = ctx.output[ctx.arg->rankId];
            localDst.addr += outputOffset;
            localDst.token = ctx.token[ctx.arg->rankId];
            const std::vector<ccu::RemoteAddr> remoteDst = Build8Plus4RemoteDst(ctx, outputOffset);
            ALLGATHER_CCU_CHK_RET(
                GroupBroadcast(ctx, localDst, remoteDst, ownSrc, ctx.half0GoSize));
        }
        return CCU_SUCCESS;
    }

    CcuResult Execute8Plus4Phase0Inter(EightPlusFourLargeContext &ctx)
    {
        CCU_IF(ctx.half0Size != 0)
        {
            uint16_t nextEventBit = 1U;
            uint16_t issuedMask = 0;
            if (ctx.arg->rankId < EIGHT_PLUS_FOUR_SERVER0_SIZE) {
                ccu::LocalAddr batchSrc;
                batchSrc.addr = ctx.input;
                batchSrc.addr += ctx.batchOffset;
                batchSrc.token = ctx.token[ctx.arg->rankId];
                ccu::Variable partOffset;
                partOffset = 0;
                const ccu::Variable batchOutputOffset =
                    Build8Plus4PipelineOffset(ctx, ctx.arg->rankId, partOffset);
                ccu::Variable batchSize;
                batchSize = ctx.half0Size;
                batchSize += ctx.half1Size;
                uint16_t eventBit = 0;
                ALLGATHER_CCU_CHK_RET(Allocate8Plus4EventBit(nextEventBit, issuedMask, eventBit));
                ALLGATHER_CCU_CHK_RET(Issue8Plus4Write(
                    ctx, Get8Plus4Seed4(ctx.arg->rankId), batchSrc, batchOutputOffset,
                    batchSize, ctx.events[0], eventBit));
            } else {
                const uint32_t pairedServer0Base =
                    2 * (ctx.arg->rankId - EIGHT_PLUS_FOUR_SERVER0_SIZE);
                ccu::Variable half0Offset;
                half0Offset = 0;
                const ccu::Variable half0OutputOffset =
                    Build8Plus4PipelineOffset(ctx, ctx.arg->rankId, half0Offset);
                ccu::LocalAddr half0Src;
                half0Src.addr = ctx.input;
                half0Src.addr += ctx.batchOffset;
                half0Src.token = ctx.token[ctx.arg->rankId];
                uint16_t half0Bit = 0;
                ALLGATHER_CCU_CHK_RET(
                    Allocate8Plus4EventBit(nextEventBit, issuedMask, half0Bit));
                ALLGATHER_CCU_CHK_RET(Issue8Plus4Write(ctx, pairedServer0Base,
                    half0Src, half0OutputOffset, ctx.half0Size, ctx.events[0], half0Bit));

                const ccu::Variable half1OutputOffset =
                    Build8Plus4PipelineOffset(ctx, ctx.arg->rankId, ctx.half0Size);
                ccu::LocalAddr half1Src;
                half1Src.addr = ctx.input;
                ccu::Variable half1InputOffset;
                half1InputOffset = ctx.batchOffset;
                half1InputOffset += ctx.half0Size;
                half1Src.addr += half1InputOffset;
                half1Src.token = ctx.token[ctx.arg->rankId];
                uint16_t half1Bit = 0;
                ALLGATHER_CCU_CHK_RET(
                    Allocate8Plus4EventBit(nextEventBit, issuedMask, half1Bit));
                ALLGATHER_CCU_CHK_RET(Issue8Plus4Write(ctx, pairedServer0Base + 1,
                    half1Src, half1OutputOffset, ctx.half1Size, ctx.events[0], half1Bit));
            }
            ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.events[0], issuedMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult Execute8Plus4Phase1Inter(EightPlusFourLargeContext &ctx)
    {
        CCU_IF(ctx.lSize != 0)
        {
            uint16_t nextEventBit = 1U;
            uint16_t issuedMask = 0;
            if (ctx.arg->rankId < EIGHT_PLUS_FOUR_SERVER0_SIZE) {
                const uint32_t seedRank = Get8Plus4Seed4(ctx.arg->rankId);
                const uint32_t lMeshTarget = Get8Plus4LMeshTarget(ctx.arg->rankId);
                const uint32_t sMeshTarget = Get8Plus4SMeshTarget(ctx.arg->rankId);

                ccu::LocalAddr src;
                src.addr = ctx.input;
                src.addr += ctx.batchOffset;
                src.token = ctx.token[ctx.arg->rankId];
                ccu::Variable lPartOffset;
                lPartOffset = 0;
                const ccu::Variable lOutputOffset =
                    Build8Plus4PipelineOffset(ctx, ctx.arg->rankId, lPartOffset);
                for (uint32_t destinationRank = EIGHT_PLUS_FOUR_SERVER0_SIZE;
                     destinationRank < EIGHT_PLUS_FOUR_RANK_SIZE; ++destinationRank) {
                    if (destinationRank == seedRank || destinationRank == lMeshTarget) {
                        continue;
                    }
                    uint16_t eventBit = 0;
                    ALLGATHER_CCU_CHK_RET(
                        Allocate8Plus4EventBit(nextEventBit, issuedMask, eventBit));
                    ALLGATHER_CCU_CHK_RET(Issue8Plus4Write(ctx, destinationRank,
                        src, lOutputOffset, ctx.lSize, ctx.events[0], eventBit));
                }

                ccu::LocalAddr sSrc;
                sSrc.addr = ctx.input;
                ccu::Variable sInputOffset;
                sInputOffset = ctx.batchOffset;
                sInputOffset += ctx.lSize;
                sSrc.addr += sInputOffset;
                sSrc.token = ctx.token[ctx.arg->rankId];
                const ccu::Variable sOutputOffset =
                    Build8Plus4PipelineOffset(ctx, ctx.arg->rankId, ctx.lSize);
                for (uint32_t destinationRank = EIGHT_PLUS_FOUR_SERVER0_SIZE;
                     destinationRank < EIGHT_PLUS_FOUR_RANK_SIZE; ++destinationRank) {
                    if (destinationRank == seedRank || destinationRank == sMeshTarget) {
                        continue;
                    }
                    uint16_t eventBit = 0;
                    ALLGATHER_CCU_CHK_RET(
                        Allocate8Plus4EventBit(nextEventBit, issuedMask, eventBit));
                    ALLGATHER_CCU_CHK_RET(Issue8Plus4Write(ctx, destinationRank,
                        sSrc, sOutputOffset, ctx.sSize, ctx.events[0], eventBit));
                }

                ccu::Variable qPartOffset;
                qPartOffset = ctx.lSize;
                qPartOffset += ctx.sSize;
                ccu::LocalAddr qSrc;
                qSrc.addr = ctx.input;
                ccu::Variable qInputOffset;
                qInputOffset = ctx.batchOffset;
                qInputOffset += qPartOffset;
                qSrc.addr += qInputOffset;
                qSrc.token = ctx.token[ctx.arg->rankId];
                const ccu::Variable qOutputOffset =
                    Build8Plus4PipelineOffset(ctx, ctx.arg->rankId, qPartOffset);
                for (uint32_t destinationRank = EIGHT_PLUS_FOUR_SERVER0_SIZE;
                     destinationRank < EIGHT_PLUS_FOUR_RANK_SIZE; ++destinationRank) {
                    if (destinationRank == seedRank) {
                        continue;
                    }
                    uint16_t eventBit = 0;
                    ALLGATHER_CCU_CHK_RET(
                        Allocate8Plus4EventBit(nextEventBit, issuedMask, eventBit));
                    ALLGATHER_CCU_CHK_RET(Issue8Plus4Write(ctx, destinationRank,
                        qSrc, qOutputOffset, ctx.qSize, ctx.events[0], eventBit));
                }
            }
            if (issuedMask != 0) {
                ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.events[0], issuedMask));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult Execute8Plus4Phase1Intra(EightPlusFourLargeContext &ctx)
    {
        CCU_IF(ctx.half1Size != 0)
        {
            uint16_t nextEventBit = 1U;
            uint16_t issuedMask = 0;
            if (ctx.arg->rankId < EIGHT_PLUS_FOUR_SERVER0_SIZE) {
                const uint32_t remoteRank = Get8Plus4Seed4(ctx.arg->rankId);
                ccu::Variable seedPartOffset;
                ccu::Variable seedSize;
                if (ctx.arg->rankId % 2 == 0) {
                    seedPartOffset = 0;
                    seedSize = ctx.half0Size;
                } else {
                    seedPartOffset = ctx.half0Size;
                    seedSize = ctx.half1Size;
                }
                const ccu::Variable remoteOutputOffset =
                    Build8Plus4PipelineOffset(ctx, remoteRank, seedPartOffset);
                ccu::LocalAddr remoteSrc;
                remoteSrc.addr = ctx.output[ctx.arg->rankId];
                remoteSrc.addr += remoteOutputOffset;
                remoteSrc.token = ctx.token[ctx.arg->rankId];
                ALLGATHER_CCU_CHK_RET(Issue8Plus4WritesToLocalPeers(ctx, remoteSrc,
                    remoteOutputOffset, seedSize, ctx.events[0], 0,
                    EIGHT_PLUS_FOUR_SERVER0_SIZE, EIGHT_PLUS_FOUR_RANK_SIZE,
                    nextEventBit, issuedMask));
            } else if (ctx.arg->rankId >= EIGHT_PLUS_FOUR_SERVER0_SIZE) {
                const uint32_t pairedServer0Base =
                    2 * (ctx.arg->rankId - EIGHT_PLUS_FOUR_SERVER0_SIZE);
                for (uint32_t sourceRank = pairedServer0Base;
                     sourceRank < pairedServer0Base + 2; ++sourceRank) {
                    ccu::Variable lPartOffset;
                    lPartOffset = 0;
                    const ccu::Variable lOutputOffset =
                        Build8Plus4PipelineOffset(ctx, sourceRank, lPartOffset);
                    ccu::LocalAddr lSrc;
                    lSrc.addr = ctx.output[ctx.arg->rankId];
                    lSrc.addr += lOutputOffset;
                    lSrc.token = ctx.token[ctx.arg->rankId];
                    uint16_t lEventBit = 0;
                    ALLGATHER_CCU_CHK_RET(
                        Allocate8Plus4EventBit(nextEventBit, issuedMask, lEventBit));
                    ALLGATHER_CCU_CHK_RET(Issue8Plus4Write(ctx,
                        Get8Plus4LMeshTarget(sourceRank), lSrc, lOutputOffset,
                        ctx.lSize, ctx.events[0], lEventBit));

                    const ccu::Variable sOutputOffset =
                        Build8Plus4PipelineOffset(ctx, sourceRank, ctx.lSize);
                    ccu::LocalAddr sSrc;
                    sSrc.addr = ctx.output[ctx.arg->rankId];
                    sSrc.addr += sOutputOffset;
                    sSrc.token = ctx.token[ctx.arg->rankId];
                    uint16_t sEventBit = 0;
                    ALLGATHER_CCU_CHK_RET(
                        Allocate8Plus4EventBit(nextEventBit, issuedMask, sEventBit));
                    ALLGATHER_CCU_CHK_RET(Issue8Plus4Write(ctx,
                        Get8Plus4SMeshTarget(sourceRank), sSrc, sOutputOffset,
                        ctx.sSize, ctx.events[0], sEventBit));
                }
            }

            ccu::LocalAddr ownSrc;
            ownSrc.addr = ctx.input;
            ccu::Variable inputOffset;
            inputOffset = ctx.batchOffset;
            inputOffset += ctx.half0Size;
            ownSrc.addr += inputOffset;
            ownSrc.token = ctx.token[ctx.arg->rankId];
            const ccu::Variable ownOutputOffset =
                Build8Plus4PipelineOffset(ctx, ctx.arg->rankId, ctx.half0Size);
            ccu::LocalAddr localDst;
            localDst.addr = ctx.output[ctx.arg->rankId];
            localDst.addr += ownOutputOffset;
            localDst.token = ctx.token[ctx.arg->rankId];
            const std::vector<ccu::RemoteAddr> remoteDst =
                Build8Plus4RemoteDst(ctx, ownOutputOffset);
            ALLGATHER_CCU_CHK_RET(
                GroupBroadcast(ctx, localDst, remoteDst, ownSrc, ctx.half1GoSize));

            if (issuedMask != 0) {
                ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.events[0], issuedMask));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult IssueLargeNhrStep0Writes(Nhr4x1LargeContext &ctx, const Nhr4x1Plan &plan)
    {
        for (uint32_t repeatIdx = 0; repeatIdx < CCU_ALLGATHER_REPEAT_NUM; ++repeatIdx) {
            CCU_IF(ctx.repeatCount != repeatIdx)
            {
                ccu::LocalAddr src;
                src.addr = ctx.input;
                src.addr += ctx.currentRankSliceInputOffset[repeatIdx];
                src.token = ctx.token[ctx.arg->rankId];

                ccu::RemoteAddr remoteDst;
                remoteDst.addr = ctx.output[plan.oppositeRank];
                remoteDst.addr += ctx.currentRankSliceOutputOffset[repeatIdx];
                remoteDst.token = ctx.token[plan.oppositeRank];
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[plan.oppositeChannel], remoteDst, src,
                    ctx.sliceSize[repeatIdx], ctx.events[repeatIdx], NHR_DATA_BIT_0));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult ExecuteLargeNhrLocalCopies(Nhr4x1LargeContext &ctx)
    {
        for (uint32_t repeatIdx = 0; repeatIdx < CCU_ALLGATHER_REPEAT_NUM; ++repeatIdx) {
            CCU_IF(ctx.repeatCount != repeatIdx)
            {
                ccu::LocalAddr src;
                src.addr = ctx.input;
                src.addr += ctx.currentRankSliceInputOffset[repeatIdx];
                src.token = ctx.token[ctx.arg->rankId];

                ccu::LocalAddr localDst;
                localDst.addr = ctx.output[ctx.arg->rankId];
                localDst.addr += ctx.currentRankSliceOutputOffset[repeatIdx];
                localDst.token = ctx.token[ctx.arg->rankId];
                ALLGATHER_CCU_CHK_RET(GroupCopy(ctx, localDst, src, ctx.goSize[repeatIdx]));
                ALLGATHER_CCU_CHK_RET(ccu::EventRecord(ctx.events[repeatIdx], NHR_LOCAL_BIT));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult WaitLargeNhrStep0(Nhr4x1LargeContext &ctx)
    {
        constexpr uint16_t eventMask = NHR_DATA_BIT_0 | NHR_LOCAL_BIT;
        for (uint32_t repeatIdx = 0; repeatIdx < CCU_ALLGATHER_REPEAT_NUM; ++repeatIdx) {
            CCU_IF(ctx.repeatCount != repeatIdx)
            {
                ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.events[repeatIdx], eventMask));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult IssueLargeNhrStep1OwnWrites(Nhr4x1LargeContext &ctx, const Nhr4x1Plan &plan)
    {
        for (uint32_t repeatIdx = 0; repeatIdx < CCU_ALLGATHER_REPEAT_NUM; ++repeatIdx) {
            CCU_IF(ctx.repeatCount != repeatIdx)
            {
                ccu::LocalAddr src;
                src.addr = ctx.input;
                src.addr += ctx.currentRankSliceInputOffset[repeatIdx];
                src.token = ctx.token[ctx.arg->rankId];

                ccu::RemoteAddr remoteDst;
                remoteDst.addr = ctx.output[plan.nextRank];
                remoteDst.addr += ctx.currentRankSliceOutputOffset[repeatIdx];
                remoteDst.token = ctx.token[plan.nextRank];
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[plan.nextChannel], remoteDst, src,
                    ctx.sliceSize[repeatIdx], ctx.events[repeatIdx], NHR_DATA_BIT_0));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult IssueLargeNhrStep1ForwardWrites(Nhr4x1LargeContext &ctx, const Nhr4x1Plan &plan)
    {
        for (uint32_t repeatIdx = 0; repeatIdx < CCU_ALLGATHER_REPEAT_NUM; ++repeatIdx) {
            CCU_IF(ctx.repeatCount != repeatIdx)
            {
                const ccu::Variable oppositeOutputOffset = BuildNhrOutputOffset(
                    ctx.rankDataSize, plan.oppositeRank, ctx.currentRankSliceInputOffset[repeatIdx]);
                ccu::LocalAddr oppositeSrc;
                oppositeSrc.addr = ctx.output[ctx.arg->rankId];
                oppositeSrc.addr += oppositeOutputOffset;
                oppositeSrc.token = ctx.token[ctx.arg->rankId];

                ccu::RemoteAddr oppositeRemoteDst;
                oppositeRemoteDst.addr = ctx.output[plan.nextRank];
                oppositeRemoteDst.addr += oppositeOutputOffset;
                oppositeRemoteDst.token = ctx.token[plan.nextRank];
                ALLGATHER_CCU_CHK_RET(ccu::Write(ctx.arg->channels[plan.nextChannel], oppositeRemoteDst,
                    oppositeSrc, ctx.sliceSize[repeatIdx], ctx.events[repeatIdx], NHR_DATA_BIT_1));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult WaitLargeNhrStep1(Nhr4x1LargeContext &ctx)
    {
        constexpr uint16_t eventMask = NHR_DATA_BIT_0 | NHR_DATA_BIT_1;
        for (uint32_t repeatIdx = 0; repeatIdx < CCU_ALLGATHER_REPEAT_NUM; ++repeatIdx) {
            CCU_IF(ctx.repeatCount != repeatIdx)
            {
                ALLGATHER_CCU_CHK_RET(ccu::EventWait(ctx.events[repeatIdx], eventMask));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult ExecuteLargeNhr4x1(Nhr4x1LargeContext &ctx, const Nhr4x1Plan &plan)
    {
        ALLGATHER_CCU_CHK_RET(IssueLargeNhrStep0Writes(ctx, plan));
        ALLGATHER_CCU_CHK_RET(ExecuteLargeNhrLocalCopies(ctx));
        ALLGATHER_CCU_CHK_RET(WaitLargeNhrStep0(ctx));
        ALLGATHER_CCU_CHK_RET(Nhr4x1StageRecord(ctx, plan));
        ALLGATHER_CCU_CHK_RET(IssueLargeNhrStep1OwnWrites(ctx, plan));
        ALLGATHER_CCU_CHK_RET(Nhr4x1StageWait(ctx, plan));
        ALLGATHER_CCU_CHK_RET(IssueLargeNhrStep1ForwardWrites(ctx, plan));
        ALLGATHER_CCU_CHK_RET(WaitLargeNhrStep1(ctx));
        return CCU_SUCCESS;
    }

    CcuResult PostSyncSmall(AllGatherBaseContext &ctx)
    {
        for (uint32_t peerStep = 0; peerStep < ctx.arg->channelCount; ++peerStep) {
            const uint32_t channelIdx = GetSmallChannelIndex(ctx, peerStep);
            ALLGATHER_CCU_CHK_RET(
                ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        for (uint32_t peerStep = 0; peerStep < ctx.arg->channelCount; ++peerStep) {
            const uint32_t channelIdx = GetSmallChannelIndex(ctx, peerStep);
            ALLGATHER_CCU_CHK_RET(
                ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    uint32_t FindSmallChannelByRemoteRank(
        const AllGatherBaseContext &ctx, uint32_t remoteRank)
    {
        for (uint32_t channelIdx = 0;
            channelIdx < ctx.arg->channelCount; ++channelIdx) {
            if (ctx.arg->remoteRanks[channelIdx] == remoteRank) {
                return channelIdx;
            }
        }
        return ctx.arg->channelCount;
    }

    CcuResult PostSyncSmallDissemination(AllGatherBaseContext &ctx)
    {
        for (uint32_t distance = 1;
            distance < ctx.arg->rankSize; distance <<= 1) {
            const uint32_t sendRank =
                (ctx.arg->rankId + distance) % ctx.arg->rankSize;
            const uint32_t receiveRank =
                (ctx.arg->rankId + ctx.arg->rankSize - distance)
                % ctx.arg->rankSize;
            const uint32_t sendChannel =
                FindSmallChannelByRemoteRank(ctx, sendRank);
            const uint32_t receiveChannel =
                FindSmallChannelByRemoteRank(ctx, receiveRank);
            if (sendChannel == ctx.arg->channelCount
                || receiveChannel == ctx.arg->channelCount) {
                HCCL_ERROR(
                    "[CcuAllGatherSmall512Fast] missing dissemination channel");
                return CcuResult::CCU_E_INTERNAL;
            }

            ALLGATHER_CCU_CHK_RET(ccu::NotifyRecord(
                ctx.arg->channels[sendChannel], CKE_IDX_0,
                1U << POST_SYNC_ID));
            ALLGATHER_CCU_CHK_RET(ccu::NotifyWait(
                ctx.arg->channels[receiveChannel], CKE_IDX_0,
                1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult PostSync(AllGatherBaseContext &ctx)
    {
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            ALLGATHER_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            ALLGATHER_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult PostSyncNhr4x1(AllGatherBaseContext &ctx, const Nhr4x1Plan &plan)
    {
        ALLGATHER_CCU_CHK_RET(
            ccu::NotifyRecord(ctx.arg->channels[plan.nextChannel], CKE_IDX_0, 1U << POST_SYNC_ID));
        ALLGATHER_CCU_CHK_RET(
            ccu::NotifyWait(ctx.arg->channels[plan.prevChannel], CKE_IDX_0, 1U << POST_SYNC_ID));
        return CCU_SUCCESS;
    }

    CcuResult PostSyncMeshNhr2x8Clos(AllGatherBaseContext &ctx, const MeshNhr2x8Plan &plan,
        const uint32_t *targetOffsets, uint32_t targetCount)
    {
        const uint32_t localRank = ctx.arg->rankId - plan.serverBase;
        for (uint32_t targetIdx = 0; targetIdx < targetCount; ++targetIdx) {
            const uint32_t remoteLocalRank =
                (localRank + targetOffsets[targetIdx]) % NHR_2X8_SERVER_RANK_SIZE;
            const uint32_t channelIdx = plan.closChannelByLocalRank[remoteLocalRank];
            ALLGATHER_CCU_CHK_RET(
                ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        for (uint32_t targetIdx = 0; targetIdx < targetCount; ++targetIdx) {
            const uint32_t remoteLocalRank =
                (localRank + targetOffsets[targetIdx]) % NHR_2X8_SERVER_RANK_SIZE;
            const uint32_t channelIdx = plan.closChannelByLocalRank[remoteLocalRank];
            ALLGATHER_CCU_CHK_RET(
                ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult PreSyncMeshNhr2x8(MeshNhr2x8LargeContext &ctx, const MeshNhr2x8Plan &plan)
    {
        if (ctx.arg->netLayer == NHR_2X8_MESH_LAYER) {
            return PreSync(ctx);
        }
        CCU_IF(ctx.phase == 0)
        {
            ALLGATHER_CCU_CHK_RET(PreSyncMeshNhr2x8Clos(ctx, plan,
                NHR_2X8_PHASE0_CLOS_OFFSETS, NHR_2X8_PHASE0_CLOS_TARGET_COUNT));
        }
        CCU_ELSE
        {
            ALLGATHER_CCU_CHK_RET(PreSyncMeshNhr2x8Clos(ctx, plan,
                NHR_2X8_PHASE1_CLOS_OFFSETS, NHR_2X8_PHASE1_CLOS_TARGET_COUNT));
        }
        return CCU_SUCCESS;
    }

    CcuResult PostSyncMeshNhr2x8(MeshNhr2x8LargeContext &ctx, const MeshNhr2x8Plan &plan)
    {
        if (ctx.arg->netLayer == NHR_2X8_MESH_LAYER) {
            return PostSync(ctx);
        }
        CCU_IF(ctx.phase == 0)
        {
            ALLGATHER_CCU_CHK_RET(PostSyncMeshNhr2x8Clos(ctx, plan,
                NHR_2X8_PHASE0_CLOS_OFFSETS, NHR_2X8_PHASE0_CLOS_TARGET_COUNT));
        }
        CCU_ELSE
        {
            ALLGATHER_CCU_CHK_RET(PostSyncMeshNhr2x8Clos(ctx, plan,
                NHR_2X8_PHASE1_CLOS_OFFSETS, NHR_2X8_PHASE1_CLOS_TARGET_COUNT));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunLarge8Plus4MeshKernel(CcuKernelArg arg)
    {
        auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
        if (kernelArg == nullptr || kernelArg->sizeClass != AllGatherSizeClass::LARGE) {
            HCCL_ERROR("[CcuAllGather8Plus4] invalid registered large kernel argument");
            return CcuResult::CCU_E_PARA;
        }
        EightPlusFourLargeContext ctx;
        ctx.arg = kernelArg;
        ctx.loopConfig.msInterleave = EIGHT_PLUS_FOUR_CCU_MS_INTERLEAVE;
        ctx.loopConfig.loopCount = EIGHT_PLUS_FOUR_CCU_LOOP_COUNT;
        ctx.loopConfig.memSlice = EIGHT_PLUS_FOUR_CCU_LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
        ALLGATHER_CCU_CHK_RET(InitResource(ctx, 1));
        ALLGATHER_CCU_CHK_RET(Load8Plus4LargeArgs(ctx));
        ALLGATHER_CCU_CHK_RET(Validate8Plus4Mesh(ctx));
        ALLGATHER_CCU_CHK_RET(PreSync(ctx));
        CCU_IF(ctx.phase == EIGHT_PLUS_FOUR_PHASE0)
        {
            if (ctx.arg->netLayer == EIGHT_PLUS_FOUR_INTER_LAYER) {
                ALLGATHER_CCU_CHK_RET(Execute8Plus4Phase0Inter(ctx));
            } else {
                ALLGATHER_CCU_CHK_RET(Execute8Plus4Phase0Intra(ctx));
            }
        } CCU_ELSE {
            CCU_IF(ctx.phase == EIGHT_PLUS_FOUR_PHASE1)
            {
                if (ctx.arg->netLayer == EIGHT_PLUS_FOUR_INTER_LAYER) {
                    ALLGATHER_CCU_CHK_RET(Execute8Plus4Phase1Inter(ctx));
                } else {
                    ALLGATHER_CCU_CHK_RET(Execute8Plus4Phase1Intra(ctx));
                }
            }
        }
        ALLGATHER_CCU_CHK_RET(PostSync(ctx));
        return CCU_SUCCESS;
    }

    CcuResult RunSmallAllGatherKernel(CcuKernelArg arg, AllGatherTopology expectedTopology)
    {
        auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
        if (kernelArg == nullptr || kernelArg->sizeClass != AllGatherSizeClass::SMALL
            || kernelArg->topology != expectedTopology) {
            HCCL_ERROR("[CcuAllGather] invalid registered kernel argument");
            return CcuResult::CCU_E_PARA;
        }

        SmallAllGatherContext ctx;
        ctx.arg = kernelArg;
        ALLGATHER_CCU_CHK_RET(InitResource(ctx, 1));
        ALLGATHER_CCU_CHK_RET(LoadSmallArgs(ctx));
        ALLGATHER_CCU_CHK_RET(PublishSmallPeerResources(ctx));
        ALLGATHER_CCU_CHK_RET(IssueSmallReadyWrites(ctx));
        ALLGATHER_CCU_CHK_RET(IssueSmallLocalCopy(ctx));
        ALLGATHER_CCU_CHK_RET(WaitSmallTransfers(ctx));
        ALLGATHER_CCU_CHK_RET(PostSyncSmall(ctx));
        return CCU_SUCCESS;
    }

    CcuResult RunSmallPull4x1Kernel(CcuKernelArg arg)
    {
        auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
        if (kernelArg == nullptr || kernelArg->sizeClass != AllGatherSizeClass::SMALL
            || kernelArg->topology != AllGatherTopology::FOUR_SERVER_ONE_NPU
            || kernelArg->rankSize != NHR_RANK_SIZE || kernelArg->channelCount != NHR_CHANNEL_COUNT
            || kernelArg->netLayer != NHR_NET_LAYER || kernelArg->handleSelfRank == 0) {
            HCCL_ERROR("[CcuAllGatherSmallPull4x1] invalid topology resources");
            return CcuResult::CCU_E_PARA;
        }

        SmallAllGatherContext ctx;
        ctx.arg = kernelArg;
        ALLGATHER_CCU_CHK_RET(InitResource(ctx, 1));
        ALLGATHER_CCU_CHK_RET(LoadSmallArgs(ctx));
        ALLGATHER_CCU_CHK_RET(PublishSmallPullPeerResources(ctx));
        ALLGATHER_CCU_CHK_RET(IssueSmallReadyReads(ctx));
        ALLGATHER_CCU_CHK_RET(IssueSmallLocalCopy(ctx));
        ALLGATHER_CCU_CHK_RET(WaitSmallTransfers(ctx));
        return CCU_SUCCESS;
    }

    CcuResult ValidateSmallPull8Plus4(const CcuKernelArgAllGather &kernelArg)
    {
        if (kernelArg.sizeClass != AllGatherSizeClass::SMALL
            || kernelArg.topology != AllGatherTopology::EIGHT_PLUS_FOUR
            || kernelArg.rankSize != EIGHT_PLUS_FOUR_RANK_SIZE
            || kernelArg.rankId >= EIGHT_PLUS_FOUR_RANK_SIZE) {
            HCCL_ERROR("[CcuAllGatherSmallPull8Plus4] invalid topology resources");
            return CcuResult::CCU_E_PARA;
        }

        uint32_t expectedChannelCount = 0;
        uint32_t expectedHandleSelf = 0;
        if (kernelArg.netLayer == EIGHT_PLUS_FOUR_INTRA_LAYER) {
            expectedChannelCount = kernelArg.rankId < EIGHT_PLUS_FOUR_SERVER0_SIZE
                ? EIGHT_PLUS_FOUR_SERVER0_SIZE - 1
                : EIGHT_PLUS_FOUR_SERVER1_SIZE - 1;
            expectedHandleSelf = 1;
        } else if (kernelArg.netLayer == EIGHT_PLUS_FOUR_INTER_LAYER) {
            expectedChannelCount = kernelArg.rankId < EIGHT_PLUS_FOUR_SERVER0_SIZE
                ? EIGHT_PLUS_FOUR_SERVER1_SIZE
                : EIGHT_PLUS_FOUR_SERVER0_SIZE;
        } else {
            HCCL_ERROR("[CcuAllGatherSmallPull8Plus4] invalid net layer %u", kernelArg.netLayer);
            return CcuResult::CCU_E_PARA;
        }

        if (kernelArg.channelCount != expectedChannelCount
            || kernelArg.handleSelfRank != expectedHandleSelf) {
            HCCL_ERROR("[CcuAllGatherSmallPull8Plus4] invalid layer resources");
            return CcuResult::CCU_E_PARA;
        }
        return CCU_SUCCESS;
    }

    CcuResult RunSmallPull8Plus4Kernel(CcuKernelArg arg)
    {
        auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
        if (kernelArg == nullptr) {
            HCCL_ERROR("[CcuAllGatherSmallPull8Plus4] null kernel argument");
            return CcuResult::CCU_E_PARA;
        }
        ALLGATHER_CCU_CHK_RET(ValidateSmallPull8Plus4(*kernelArg));

        SmallAllGatherContext ctx;
        ctx.arg = kernelArg;
        ALLGATHER_CCU_CHK_RET(InitResource(ctx, 1));
        ALLGATHER_CCU_CHK_RET(LoadSmallArgs(ctx));
        ALLGATHER_CCU_CHK_RET(PublishSmallPullPeerResources(ctx));
        ALLGATHER_CCU_CHK_RET(IssueSmallReadyReads(ctx));
        ALLGATHER_CCU_CHK_RET(IssueSmallLocalCopy(ctx));
        ALLGATHER_CCU_CHK_RET(WaitSmallTransfers(ctx));
        return CCU_SUCCESS;
    }

    CcuResult RunSmall512FastWriteKernel(
        CcuKernelArg arg, AllGatherTopology expectedTopology)
    {
        auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
        const uint32_t expectedRankSize =
            expectedTopology == AllGatherTopology::FOUR_SERVER_ONE_NPU ? 4U : 12U;
        if (kernelArg == nullptr || kernelArg->sizeClass != AllGatherSizeClass::SMALL
            || kernelArg->topology != expectedTopology
            || kernelArg->rankSize != expectedRankSize
            || kernelArg->channelCount != expectedRankSize - 1
            || kernelArg->netLayer != SMALL_512_FUSED_LAYER
            || kernelArg->handleSelfRank == 0
            || kernelArg->small512FastMode != SMALL_512_FAST_MODE) {
            HCCL_ERROR("[CcuAllGatherSmall512Fast] invalid registered kernel argument");
            return CcuResult::CCU_E_PARA;
        }

        Small512FastContext ctx;
        ctx.arg = kernelArg;
        ALLGATHER_CCU_CHK_RET(InitResource(ctx, 1));
        ALLGATHER_CCU_CHK_RET(LoadSmall512FastArgs(ctx));
        CCU_IF(ctx.refreshResources != 0)
        {
            ALLGATHER_CCU_CHK_RET(PublishSmallPeerResources(ctx));
            ALLGATHER_CCU_CHK_RET(WaitSmallPeerResources(ctx));
        }
        if (expectedTopology == AllGatherTopology::FOUR_SERVER_ONE_NPU) {
            ALLGATHER_CCU_CHK_RET(IssueSmall512LocalCopy(ctx));
            ALLGATHER_CCU_CHK_RET(IssueSmall512Writes(ctx));
        } else {
            ALLGATHER_CCU_CHK_RET(IssueSmall512Writes(ctx));
            ALLGATHER_CCU_CHK_RET(IssueSmall512LocalCopy(ctx));
        }
        ALLGATHER_CCU_CHK_RET(WaitSmallTransfers(ctx));
        ALLGATHER_CCU_CHK_RET(PostSyncSmallDissemination(ctx));
        return CCU_SUCCESS;
    }

    CcuResult RunLargeMeshNhr2x8Kernel(CcuKernelArg arg)
    {
        auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
        if (kernelArg == nullptr || kernelArg->topology != AllGatherTopology::TWO_SERVER_EIGHT_NPU
            || kernelArg->sizeClass != AllGatherSizeClass::LARGE) {
            HCCL_ERROR("[CcuAllGatherMeshNhr2x8] invalid registered large kernel argument");
            return CcuResult::CCU_E_PARA;
        }

        MeshNhr2x8LargeContext ctx;
        ctx.arg = kernelArg;
        MeshNhr2x8Plan plan{};
        ALLGATHER_CCU_CHK_RET(InitResource(ctx, 1));
        ALLGATHER_CCU_CHK_RET(BuildMeshNhr2x8Plan(ctx, plan));
        ALLGATHER_CCU_CHK_RET(LoadMeshNhr2x8LargeArgs(ctx));
        ALLGATHER_CCU_CHK_RET(PreSyncMeshNhr2x8(ctx, plan));
        ALLGATHER_CCU_CHK_RET(ExecuteLargeMeshNhr2x8(ctx, plan));
        ALLGATHER_CCU_CHK_RET(PostSyncMeshNhr2x8(ctx, plan));
        return CCU_SUCCESS;
    }

    CcuResult RunLargeNhr4x1Kernel(CcuKernelArg arg)
    {
        auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
        if (kernelArg == nullptr || kernelArg->sizeClass != AllGatherSizeClass::LARGE
            || kernelArg->topology != AllGatherTopology::FOUR_SERVER_ONE_NPU) {
            HCCL_ERROR("[CcuAllGatherNhr4x1] invalid registered large kernel argument");
            return CcuResult::CCU_E_PARA;
        }

        Nhr4x1LargeContext ctx;
        ctx.arg = kernelArg;
        Nhr4x1Plan plan{};
        ALLGATHER_CCU_CHK_RET(InitResource(ctx, CCU_ALLGATHER_REPEAT_NUM));
        ALLGATHER_CCU_CHK_RET(BuildNhr4x1Plan(ctx, plan));
        ALLGATHER_CCU_CHK_RET(LoadNhr4x1LargeArgs(ctx));
        ALLGATHER_CCU_CHK_RET(PreSyncNhr4x1(ctx, plan));
        ALLGATHER_CCU_CHK_RET(ExecuteLargeNhr4x1(ctx, plan));
        ALLGATHER_CCU_CHK_RET(PostSyncNhr4x1(ctx, plan));
        return CCU_SUCCESS;
    }

} // namespace

namespace topo_2x8 {
    CcuResult CcuAllGatherSmallKernel(CcuKernelArg arg)
    {
        return RunSmallAllGatherKernel(arg, AllGatherTopology::TWO_SERVER_EIGHT_NPU);
    }

    CcuResult CcuAllGatherLargeKernel(CcuKernelArg arg)
    {
        return RunLargeMeshNhr2x8Kernel(arg);
    }
} // namespace topo_2x8

namespace topo_4x1 {
    CcuResult CcuAllGatherSmallKernel(CcuKernelArg arg)
    {
        return RunSmallPull4x1Kernel(arg);
    }

    CcuResult CcuAllGatherLargeKernel(CcuKernelArg arg)
    {
        return RunLargeNhr4x1Kernel(arg);
    }
} // namespace topo_4x1

namespace topo_8plus4 {
    CcuResult CcuAllGatherSmallKernel(CcuKernelArg arg)
    {
        auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
        if (kernelArg != nullptr && kernelArg->small512FastMode != 0) {
            return RunSmall512FastWriteKernel(
                arg, AllGatherTopology::EIGHT_PLUS_FOUR);
        }
        return RunSmallPull8Plus4Kernel(arg);
    }

    CcuResult CcuAllGatherLargeKernel(CcuKernelArg arg)
    {
        return RunLarge8Plus4MeshKernel(arg);
    }
} // namespace topo_8plus4

} // namespace ops_hccl
