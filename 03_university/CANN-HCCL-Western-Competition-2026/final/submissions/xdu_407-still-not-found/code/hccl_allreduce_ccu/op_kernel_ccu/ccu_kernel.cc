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
#define PHASE0_CCU_CHK(call) \
do { \
    const CcuResult phase0CcuResult = (call); \
    if (phase0CcuResult != CCU_SUCCESS) { \
        return phase0CcuResult; \
    } \
} while (0)
namespace ccu = AscendC::ccu;
    constexpr uint32_t DIRECT12_INPUT_ADDR_VARIABLE = 0;
    constexpr uint32_t DIRECT12_INPUT_TOKEN_VARIABLE = 1;
    constexpr uint32_t DIRECT12_OUTPUT_ADDR_VARIABLE = 2;
    constexpr uint32_t DIRECT12_OUTPUT_TOKEN_VARIABLE = 3;
    constexpr uint16_t DIRECT12_INPUT_ADDR_MASK = 1U << DIRECT12_INPUT_ADDR_VARIABLE;
    constexpr uint16_t DIRECT12_INPUT_TOKEN_MASK = 1U << DIRECT12_INPUT_TOKEN_VARIABLE;
    constexpr uint16_t DIRECT12_OUTPUT_ADDR_MASK = 1U << DIRECT12_OUTPUT_ADDR_VARIABLE;
    constexpr uint16_t DIRECT12_OUTPUT_TOKEN_MASK = 1U << DIRECT12_OUTPUT_TOKEN_VARIABLE;
    constexpr uint16_t DIRECT12_ADDRESS_TOKEN_MASK = DIRECT12_INPUT_ADDR_MASK | DIRECT12_INPUT_TOKEN_MASK
                                                     | DIRECT12_OUTPUT_ADDR_MASK | DIRECT12_OUTPUT_TOKEN_MASK;

    struct Direct12TaskVariables {
        ccu::Variable inputAddr;
        ccu::Variable inputToken;
        ccu::Variable outputAddr;
        ccu::Variable outputToken;
        ccu::Variable scratchAddr;
        ccu::Variable scratchToken;
        ccu::Variable dataOffsetBytes;
        ccu::Variable elementCount;
        ccu::Variable tileOffsetBytes;
        ccu::Variable tileElementCount;
        ccu::Variable generation;
        ccu::Variable flags;
    };

    struct Direct12Context {
        Direct12TaskVariables taskArgs;
        ccu::Variable tileBytes;
        ccu::Event event;
        uint64_t scratchStrideBytes = 0;
    };

    CcuResult LoadDirect12TaskArgs(Direct12Context &ctx)
    {
        PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.inputAddr, static_cast<uint32_t>(TaskArgIndex::INPUT_ADDR)));
        PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.inputToken, static_cast<uint32_t>(TaskArgIndex::INPUT_TOKEN)));
        PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.outputAddr, static_cast<uint32_t>(TaskArgIndex::OUTPUT_ADDR)));
        PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.outputToken, static_cast<uint32_t>(TaskArgIndex::OUTPUT_TOKEN)));
        PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.scratchAddr, static_cast<uint32_t>(TaskArgIndex::SCRATCH_ADDR)));
        PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.scratchToken, static_cast<uint32_t>(TaskArgIndex::SCRATCH_TOKEN)));
        PHASE0_CCU_CHK(
            ccu::LoadArg(ctx.taskArgs.dataOffsetBytes, static_cast<uint32_t>(TaskArgIndex::DATA_OFFSET_BYTES)));
        PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.elementCount, static_cast<uint32_t>(TaskArgIndex::ELEMENT_COUNT)));
        PHASE0_CCU_CHK(
            ccu::LoadArg(ctx.taskArgs.tileOffsetBytes, static_cast<uint32_t>(TaskArgIndex::TILE_OFFSET_BYTES)));
        PHASE0_CCU_CHK(
            ccu::LoadArg(ctx.taskArgs.tileElementCount, static_cast<uint32_t>(TaskArgIndex::TILE_ELEMENT_COUNT)));
        PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.generation, static_cast<uint32_t>(TaskArgIndex::GENERATION)));
        PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.flags, static_cast<uint32_t>(TaskArgIndex::FLAGS)));
        ccu::Variable twice;
        twice = ctx.taskArgs.tileElementCount + ctx.taskArgs.tileElementCount;
        ctx.tileBytes = twice + twice;
        return CCU_SUCCESS;
    }

    bool ValidDirect12CanonicalMetadata(
        uint32_t canonicalRank, uint32_t canonicalGroup, uint32_t canonicalGroupRank)
    {
        return canonicalRank < DIRECT12_CANONICAL_LEAF_COUNT && canonicalGroup < 2
               && ((canonicalGroup == 0 && canonicalRank < 8 && canonicalGroupRank == canonicalRank)
                   || (canonicalGroup == 1 && canonicalRank >= 8 && canonicalGroupRank == canonicalRank - 8));
    }

    CcuResult ValidateDirect12Runtime(uint32_t worker, uint32_t generation, uint64_t elementCount,
        uint64_t tileElementCount, uint64_t scratchStrideBytes, uint32_t channelCount)
    {
        if (worker >= ACTIVE_WORKERS || generation >= PIPELINE_GENERATIONS || elementCount == 0
            || tileElementCount == 0 || scratchStrideBytes == 0
            || scratchStrideBytes > PACKAGE_A_DIRECT12_MAX_TILE_BYTES
            || tileElementCount > scratchStrideBytes / sizeof(float)
            || (tileElementCount * sizeof(float)) % sizeof(float) != 0 || channelCount == 0
            || channelCount > MAX_PEER_CHANNELS) {
            return CCU_E_PARA;
        }
        return CCU_SUCCESS;
    }

    template <typename StaticArg, typename Channels, typename Peers>
    CcuResult ValidateDirect12LayerStaticArg(
        const StaticArg &staticArg, const Channels &channels, const Peers &peerCanonicalRanks,
        uint32_t channelCount, FabricLayer layer)
    {
        if (!ValidDirect12CanonicalMetadata(
                staticArg.canonicalRank, staticArg.canonicalGroup, staticArg.canonicalGroupRank)
            || (layer != FabricLayer::L0 && layer != FabricLayer::L1)
            || channelCount == 0 || channelCount > MAX_PEER_CHANNELS
            || (layer == FabricLayer::L0 && staticArg.copyLocalInput != 1)
            || (layer == FabricLayer::L1 && staticArg.copyLocalInput != 0)) {
            return CCU_E_PARA;
        }
        const uint32_t group = layer == FabricLayer::L0
                                   ? staticArg.canonicalGroup
                                   : 1U - staticArg.canonicalGroup;
        const uint32_t groupSize = group == 0 ? 8 : 4;
        const uint32_t expectedChannelCount = layer == FabricLayer::L0 ? groupSize - 1 : groupSize;
        if (channelCount != expectedChannelCount) {
            return CCU_E_PARA;
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            if (channels[index] == 0 || peerCanonicalRanks[index] >= DIRECT12_CANONICAL_LEAF_COUNT
                || peerCanonicalRanks[index] == staticArg.canonicalRank
                || (peerCanonicalRanks[index] < 8 ? 0U : 1U) != group) {
                return CCU_E_PARA;
            }
            for (uint32_t prior = 0; prior < index; ++prior) {
                if (peerCanonicalRanks[prior] == peerCanonicalRanks[index]) {
                    return CCU_E_PARA;
                }
            }
        }
        const ccu::Variable dieAnchor
            = ccu::GetResByChannel<ccu::Variable>(channels[0], DIRECT12_INPUT_ADDR_VARIABLE);
        (void)dieAnchor;
        return CCU_SUCCESS;
    }

    ccu::LocalAddr Direct12LocalAddress(
        const ccu::Variable &base, const ccu::Variable &token, const ccu::Variable &firstOffset)
    {
        ccu::LocalAddr address;
        address.addr = base;
        address.addr += firstOffset;
        address.token = token;
        return address;
    }

    ccu::LocalAddr Direct12ScratchLeaf(
        const Direct12Context &ctx, uint32_t canonicalLeaf)
    {
        ccu::Variable leafOffset;
        leafOffset = canonicalLeaf * ctx.scratchStrideBytes;
        return Direct12LocalAddress(ctx.taskArgs.scratchAddr, ctx.taskArgs.scratchToken, leafOffset);
    }

    ccu::LocalAddr Direct12LocalTile(
        const ccu::Variable &base, const ccu::Variable &token, const Direct12Context &ctx)
    {
        ccu::LocalAddr address = Direct12LocalAddress(base, token, ctx.taskArgs.dataOffsetBytes);
        address.addr += ctx.taskArgs.tileOffsetBytes;
        return address;
    }

    ccu::RemoteAddr Direct12RemoteTile(ccu::Variable base, ccu::Variable token, const Direct12Context &ctx)
    {
        ccu::RemoteAddr address;
        address.addr = base;
        address.addr += ctx.taskArgs.dataOffsetBytes;
        address.addr += ctx.taskArgs.tileOffsetBytes;
        address.token = token;
        return address;
    }

    template <uint32_t ReadyNotify, uint32_t AckNotify, typename Channels>
    CcuResult PublishDirect12Addresses(
        Direct12Context &ctx, const Channels &channels, uint32_t channelCount)
    {
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(channels[index], ctx.taskArgs.inputAddr,
                DIRECT12_INPUT_ADDR_VARIABLE, ReadyNotify, DIRECT12_INPUT_ADDR_MASK));
            PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(channels[index], ctx.taskArgs.inputToken,
                DIRECT12_INPUT_TOKEN_VARIABLE, ReadyNotify, DIRECT12_INPUT_TOKEN_MASK));
            PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(channels[index], ctx.taskArgs.outputAddr,
                DIRECT12_OUTPUT_ADDR_VARIABLE, ReadyNotify, DIRECT12_OUTPUT_ADDR_MASK));
            PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(channels[index], ctx.taskArgs.outputToken,
                DIRECT12_OUTPUT_TOKEN_VARIABLE, ReadyNotify, DIRECT12_OUTPUT_TOKEN_MASK));
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], ReadyNotify, DIRECT12_ADDRESS_TOKEN_MASK));
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], AckNotify));
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], AckNotify));
        }
        return CCU_SUCCESS;
    }

    template <uint32_t ReadyNotify, uint32_t AckNotify, typename Channels, typename Operation>
    CcuResult RunDirect12RemotePhase(const Channels &channels, uint32_t channelCount, Operation operation)
    {
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], ReadyNotify));
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], ReadyNotify));
        }
        PHASE0_CCU_CHK(operation());
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], AckNotify));
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], AckNotify));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunDirect12PartialTree(Direct12Context &ctx, uint32_t canonicalGroup);

    template <uint32_t ReadyNotify, uint32_t AckNotify, typename Channels,
        typename ReadOperation, typename PartialOperation>
    CcuResult RunDirect12GatherPhase(const Channels &channels, uint32_t channelCount,
        ReadOperation readOperation, PartialOperation partialOperation)
    {
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], ReadyNotify));
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], ReadyNotify));
        }
        PHASE0_CCU_CHK(readOperation());
        // Peers may acknowledge completed reads while the two disjoint canonical subtrees reduce in parallel.
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], AckNotify));
        }
        PHASE0_CCU_CHK(partialOperation());
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], AckNotify));
        }
        return CCU_SUCCESS;
    }

    template <uint32_t ReadyNotify, uint32_t AckNotify, typename StaticArg>
    CcuResult RunDirect12GatherL0(Direct12Context &ctx, const StaticArg &staticArg, bool copyLocalInput)
    {
        return RunDirect12GatherPhase<ReadyNotify, AckNotify>(
            staticArg.l0Channels, staticArg.l0ChannelCount, [&]() {
            uint16_t waitMask = 0;
            if (copyLocalInput) {
                waitMask = 1;
                PHASE0_CCU_CHK(ccu::LocalCopy(Direct12ScratchLeaf(ctx, staticArg.canonicalRank),
                    Direct12LocalTile(ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, ctx), ctx.tileBytes,
                    ctx.event, 1));
            }
            for (uint32_t index = 0; index < staticArg.l0ChannelCount; ++index) {
                const uint16_t mask = static_cast<uint16_t>(1U << (index + (copyLocalInput ? 1 : 0)));
                waitMask = static_cast<uint16_t>(waitMask | mask);
                const ccu::Variable remoteInputAddr = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l0Channels[index], DIRECT12_INPUT_ADDR_VARIABLE);
                const ccu::Variable remoteInputToken = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l0Channels[index], DIRECT12_INPUT_TOKEN_VARIABLE);
                PHASE0_CCU_CHK(ccu::Read(staticArg.l0Channels[index],
                    Direct12ScratchLeaf(ctx, staticArg.l0PeerCanonicalRanks[index]),
                    Direct12RemoteTile(remoteInputAddr, remoteInputToken, ctx), ctx.tileBytes, ctx.event, mask));
            }
            return ccu::EventWait(ctx.event, waitMask);
        }, [&]() { return RunDirect12PartialTree(ctx, staticArg.canonicalGroup); });
    }

    template <uint32_t ReadyNotify, uint32_t AckNotify, typename StaticArg>
    CcuResult RunDirect12GatherL1(Direct12Context &ctx, const StaticArg &staticArg)
    {
        return RunDirect12GatherPhase<ReadyNotify, AckNotify>(
            staticArg.l1Channels, staticArg.l1ChannelCount, [&]() {
            uint16_t waitMask = 0;
            for (uint32_t index = 0; index < staticArg.l1ChannelCount; ++index) {
                const uint16_t mask = static_cast<uint16_t>(1U << index);
                waitMask = static_cast<uint16_t>(waitMask | mask);
                const ccu::Variable remoteInputAddr = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l1Channels[index], DIRECT12_INPUT_ADDR_VARIABLE);
                const ccu::Variable remoteInputToken = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l1Channels[index], DIRECT12_INPUT_TOKEN_VARIABLE);
                PHASE0_CCU_CHK(ccu::Read(staticArg.l1Channels[index],
                    Direct12ScratchLeaf(ctx, staticArg.l1PeerCanonicalRanks[index]),
                    Direct12RemoteTile(remoteInputAddr, remoteInputToken, ctx), ctx.tileBytes, ctx.event, mask));
            }
            return ccu::EventWait(ctx.event, waitMask);
        }, [&]() { return RunDirect12PartialTree(ctx, 1U - staticArg.canonicalGroup); });
    }

    CcuResult ReducePair(Direct12Context &ctx, uint32_t destinationLeaf, uint32_t sourceLeaf, uint16_t mask)
    {
        return ccu::LocalReduce(Direct12ScratchLeaf(ctx, destinationLeaf), Direct12ScratchLeaf(ctx, sourceLeaf),
            ctx.tileBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, mask);
    }

    CcuResult ReduceDirect12Group0(Direct12Context &ctx)
    {
        PHASE0_CCU_CHK(ReducePair(ctx, 0, 1, 1U << 0));
        PHASE0_CCU_CHK(ReducePair(ctx, 2, 3, 1U << 1));
        PHASE0_CCU_CHK(ReducePair(ctx, 4, 5, 1U << 2));
        PHASE0_CCU_CHK(ReducePair(ctx, 6, 7, 1U << 3));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 4) - 1));
        PHASE0_CCU_CHK(ReducePair(ctx, 0, 2, 1U << 0));
        PHASE0_CCU_CHK(ReducePair(ctx, 4, 6, 1U << 1));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 2) - 1));
        PHASE0_CCU_CHK(ReducePair(ctx, 0, 4, 1U));
        return ccu::EventWait(ctx.event, 1U);
    }

    CcuResult ReduceDirect12Group1(Direct12Context &ctx)
    {
        PHASE0_CCU_CHK(ReducePair(ctx, 8, 9, 1U << 0));
        PHASE0_CCU_CHK(ReducePair(ctx, 10, 11, 1U << 1));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 2) - 1));
        PHASE0_CCU_CHK(ReducePair(ctx, 8, 10, 1U));
        return ccu::EventWait(ctx.event, 1U);
    }

    CcuResult RunDirect12PartialTree(Direct12Context &ctx, uint32_t canonicalGroup)
    {
        if (canonicalGroup == 0) {
            return ReduceDirect12Group0(ctx);
        }
        return ReduceDirect12Group1(ctx);
    }

    CcuResult ReduceDirect12Group0Frontier(
        Direct12Context &ctx, const std::array<uint16_t, DIRECT12_CANONICAL_LEAF_COUNT> &readyMasks)
    {
        constexpr uint16_t REDUCE_0 = 1U << 8;
        constexpr uint16_t REDUCE_1 = 1U << 9;
        constexpr uint16_t REDUCE_2 = 1U << 10;
        constexpr uint16_t REDUCE_3 = 1U << 11;
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, readyMasks[0] | readyMasks[1]));
        PHASE0_CCU_CHK(ReducePair(ctx, 0, 1, REDUCE_0));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, readyMasks[2] | readyMasks[3]));
        PHASE0_CCU_CHK(ReducePair(ctx, 2, 3, REDUCE_1));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, readyMasks[4] | readyMasks[5]));
        PHASE0_CCU_CHK(ReducePair(ctx, 4, 5, REDUCE_2));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, readyMasks[6] | readyMasks[7]));
        PHASE0_CCU_CHK(ReducePair(ctx, 6, 7, REDUCE_3));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, REDUCE_0 | REDUCE_1 | REDUCE_2 | REDUCE_3));
        PHASE0_CCU_CHK(ReducePair(ctx, 0, 2, REDUCE_0));
        PHASE0_CCU_CHK(ReducePair(ctx, 4, 6, REDUCE_1));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, REDUCE_0));
        constexpr uint16_t OUTPUT_SEED = REDUCE_2;
        PHASE0_CCU_CHK(ccu::LocalCopy(
            Direct12LocalTile(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, ctx),
            Direct12ScratchLeaf(ctx, 0), ctx.tileBytes, ctx.event, OUTPUT_SEED));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, REDUCE_1 | OUTPUT_SEED));
        PHASE0_CCU_CHK(ccu::LocalReduce(
            Direct12LocalTile(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, ctx),
            Direct12ScratchLeaf(ctx, 4), ctx.tileBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, REDUCE_0));
        return ccu::EventWait(ctx.event, REDUCE_0);
    }

    CcuResult ReduceDirect12Group1Frontier(
        Direct12Context &ctx, const std::array<uint16_t, DIRECT12_CANONICAL_LEAF_COUNT> &readyMasks)
    {
        constexpr uint16_t REDUCE_0 = 1U << 8;
        constexpr uint16_t REDUCE_1 = 1U << 9;
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, readyMasks[8] | readyMasks[9]));
        PHASE0_CCU_CHK(ReducePair(ctx, 8, 9, REDUCE_0));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, readyMasks[10] | readyMasks[11]));
        PHASE0_CCU_CHK(ReducePair(ctx, 10, 11, REDUCE_1));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, REDUCE_0 | REDUCE_1));
        PHASE0_CCU_CHK(ReducePair(ctx, 8, 10, REDUCE_0));
        return ccu::EventWait(ctx.event, REDUCE_0);
    }

    template <uint32_t ReadyNotify, uint32_t AckNotify, typename StaticArg>
    CcuResult RunDirect12GatherL0Frontier(
        Direct12Context &ctx, const StaticArg &staticArg, bool copyLocalInput)
    {
        std::array<uint16_t, DIRECT12_CANONICAL_LEAF_COUNT> readyMasks{};
        return RunDirect12GatherPhase<ReadyNotify, AckNotify>(
            staticArg.l0Channels, staticArg.l0ChannelCount, [&]() {
            uint32_t eventIndex = 0;
            if (copyLocalInput) {
                const uint16_t mask = 1U << eventIndex++;
                readyMasks[staticArg.canonicalRank] = mask;
                PHASE0_CCU_CHK(ccu::LocalCopy(Direct12ScratchLeaf(ctx, staticArg.canonicalRank),
                    Direct12LocalTile(ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, ctx), ctx.tileBytes,
                    ctx.event, mask));
            }
            for (uint32_t index = 0; index < staticArg.l0ChannelCount; ++index) {
                const uint16_t mask = static_cast<uint16_t>(1U << eventIndex++);
                readyMasks[staticArg.l0PeerCanonicalRanks[index]] = mask;
                const ccu::Variable remoteInputAddr = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l0Channels[index], DIRECT12_INPUT_ADDR_VARIABLE);
                const ccu::Variable remoteInputToken = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l0Channels[index], DIRECT12_INPUT_TOKEN_VARIABLE);
                PHASE0_CCU_CHK(ccu::Read(staticArg.l0Channels[index],
                    Direct12ScratchLeaf(ctx, staticArg.l0PeerCanonicalRanks[index]),
                    Direct12RemoteTile(remoteInputAddr, remoteInputToken, ctx), ctx.tileBytes, ctx.event, mask));
            }
            return CCU_SUCCESS;
        }, [&]() {
            return staticArg.canonicalGroup == 0
                       ? ReduceDirect12Group0Frontier(ctx, readyMasks)
                       : ReduceDirect12Group1Frontier(ctx, readyMasks);
        });
    }

    template <uint32_t ReadyNotify, uint32_t AckNotify, typename StaticArg>
    CcuResult RunDirect12GatherL1Frontier(Direct12Context &ctx, const StaticArg &staticArg)
    {
        std::array<uint16_t, DIRECT12_CANONICAL_LEAF_COUNT> readyMasks{};
        return RunDirect12GatherPhase<ReadyNotify, AckNotify>(
            staticArg.l1Channels, staticArg.l1ChannelCount, [&]() {
            for (uint32_t index = 0; index < staticArg.l1ChannelCount; ++index) {
                const uint16_t mask = static_cast<uint16_t>(1U << index);
                readyMasks[staticArg.l1PeerCanonicalRanks[index]] = mask;
                const ccu::Variable remoteInputAddr = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l1Channels[index], DIRECT12_INPUT_ADDR_VARIABLE);
                const ccu::Variable remoteInputToken = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l1Channels[index], DIRECT12_INPUT_TOKEN_VARIABLE);
                PHASE0_CCU_CHK(ccu::Read(staticArg.l1Channels[index],
                    Direct12ScratchLeaf(ctx, staticArg.l1PeerCanonicalRanks[index]),
                    Direct12RemoteTile(remoteInputAddr, remoteInputToken, ctx), ctx.tileBytes, ctx.event, mask));
            }
            return CCU_SUCCESS;
        }, [&]() {
            return staticArg.canonicalGroup == 0
                       ? ReduceDirect12Group1Frontier(ctx, readyMasks)
                       : ReduceDirect12Group0Frontier(ctx, readyMasks);
        });
    }

    CcuResult RunDirect12Join(Direct12Context &ctx)
    {
        PHASE0_CCU_CHK(ccu::LocalReduce(
            Direct12LocalTile(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, ctx),
            Direct12ScratchLeaf(ctx, 8), ctx.tileBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 1U));
        return ccu::EventWait(ctx.event, 1U);
    }

    template <uint32_t ReadyNotify, uint32_t AckNotify, typename StaticArg>
    CcuResult RunDirect12BroadcastL1(Direct12Context &ctx, const StaticArg &staticArg)
    {
        return RunDirect12RemotePhase<ReadyNotify, AckNotify>(staticArg.l1Channels, staticArg.l1ChannelCount, [&]() {
            uint16_t waitMask = 0;
            const ccu::LocalAddr localOutput
                = Direct12LocalTile(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, ctx);
            for (uint32_t index = 0; index < staticArg.l1ChannelCount; ++index) {
                const uint16_t mask = static_cast<uint16_t>(1U << index);
                waitMask = static_cast<uint16_t>(waitMask | mask);
                const ccu::Variable remoteOutputAddr = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l1Channels[index], DIRECT12_OUTPUT_ADDR_VARIABLE);
                const ccu::Variable remoteOutputToken = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l1Channels[index], DIRECT12_OUTPUT_TOKEN_VARIABLE);
                PHASE0_CCU_CHK(ccu::Write(staticArg.l1Channels[index],
                    Direct12RemoteTile(remoteOutputAddr, remoteOutputToken, ctx), localOutput, ctx.tileBytes,
                    ctx.event, mask));
            }
            return ccu::EventWait(ctx.event, waitMask);
        });
    }

    template <uint32_t ReadyNotify, uint32_t AckNotify, typename StaticArg>
    CcuResult RunDirect12BroadcastL0(Direct12Context &ctx, const StaticArg &staticArg)
    {
        return RunDirect12RemotePhase<ReadyNotify, AckNotify>(staticArg.l0Channels, staticArg.l0ChannelCount, [&]() {
            uint16_t waitMask = 0;
            const ccu::LocalAddr localOutput
                = Direct12LocalTile(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, ctx);
            for (uint32_t index = 0; index < staticArg.l0ChannelCount; ++index) {
                const uint16_t mask = static_cast<uint16_t>(1U << index);
                waitMask = static_cast<uint16_t>(waitMask | mask);
                const ccu::Variable remoteOutputAddr = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l0Channels[index], DIRECT12_OUTPUT_ADDR_VARIABLE);
                const ccu::Variable remoteOutputToken = ccu::GetResByChannel<ccu::Variable>(
                    staticArg.l0Channels[index], DIRECT12_OUTPUT_TOKEN_VARIABLE);
                PHASE0_CCU_CHK(ccu::Write(staticArg.l0Channels[index],
                    Direct12RemoteTile(remoteOutputAddr, remoteOutputToken, ctx), localOutput, ctx.tileBytes,
                    ctx.event, mask));
            }
            return ccu::EventWait(ctx.event, waitMask);
        });
    }

    template <uint32_t ReadyNotify, uint32_t AckNotify, typename StaticArg>
    CcuResult RunDirect12GatherAll(Direct12Context &ctx, const StaticArg &staticArg, bool publish)
    {
        if (publish && staticArg.l0ChannelCount != 0) {
            PHASE0_CCU_CHK((PublishDirect12Addresses<ReadyNotify, AckNotify>(
                ctx, staticArg.l0Channels, staticArg.l0ChannelCount)));
        }
        if (publish && staticArg.l1ChannelCount != 0) {
            PHASE0_CCU_CHK((PublishDirect12Addresses<ReadyNotify, AckNotify>(
                ctx, staticArg.l1Channels, staticArg.l1ChannelCount)));
        }
        if (staticArg.l0ChannelCount != 0 || staticArg.copyLocalInput != 0) {
            if (staticArg.mode == DIRECT12_MODE_FRONTIER_512
                || staticArg.mode == DIRECT12_MODE_PREFIX_SCALAR_TAIL) {
                PHASE0_CCU_CHK((RunDirect12GatherL0Frontier<ReadyNotify, AckNotify>(
                    ctx, staticArg, staticArg.copyLocalInput != 0)));
            } else {
                PHASE0_CCU_CHK((RunDirect12GatherL0<ReadyNotify, AckNotify>(
                    ctx, staticArg, staticArg.copyLocalInput != 0)));
            }
        }
        if (staticArg.l1ChannelCount != 0) {
            if (staticArg.mode == DIRECT12_MODE_FRONTIER_512
                || staticArg.mode == DIRECT12_MODE_PREFIX_SCALAR_TAIL) {
                PHASE0_CCU_CHK((RunDirect12GatherL1Frontier<ReadyNotify, AckNotify>(ctx, staticArg)));
            } else {
                PHASE0_CCU_CHK((RunDirect12GatherL1<ReadyNotify, AckNotify>(ctx, staticArg)));
            }
        }
        return CCU_SUCCESS;
    }

    template <uint32_t ReadyNotify, uint32_t AckNotify, typename StaticArg>
    CcuResult RunDirect12BroadcastAll(Direct12Context &ctx, const StaticArg &staticArg)
    {
        if (staticArg.l0ChannelCount != 0) {
            PHASE0_CCU_CHK((RunDirect12BroadcastL0<ReadyNotify, AckNotify>(ctx, staticArg)));
        }
        if (staticArg.l1ChannelCount != 0) {
            PHASE0_CCU_CHK((RunDirect12BroadcastL1<ReadyNotify, AckNotify>(ctx, staticArg)));
        }
        return CCU_SUCCESS;
    }

    void SelectFirstDirect12PairTile(Direct12Context &ctx, ccu::Variable &secondTileBytes)
    {
        secondTileBytes = ctx.tileBytes;
        ctx.tileBytes = ctx.scratchStrideBytes;
    }

    void SelectSecondDirect12PairTile(Direct12Context &ctx, const ccu::Variable &secondTileBytes)
    {
        ccu::Variable scratchBankBytes;
        scratchBankBytes = Direct12ScratchBankBytes(ctx.scratchStrideBytes);
        ctx.taskArgs.scratchAddr += scratchBankBytes;
        ccu::Variable tileStrideBytes;
        tileStrideBytes = ctx.scratchStrideBytes;
        ctx.taskArgs.tileOffsetBytes += tileStrideBytes;
        ctx.tileBytes = secondTileBytes;
    }

    template <bool Publish, typename StaticArg>
    CcuResult RunDirect12PairGather(Direct12Context &ctx, const StaticArg &staticArg)
    {
        ccu::Variable secondTileBytes;
        SelectFirstDirect12PairTile(ctx, secondTileBytes);
        PHASE0_CCU_CHK((RunDirect12GatherAll<0, 1>(ctx, staticArg, Publish)));
        SelectSecondDirect12PairTile(ctx, secondTileBytes);
        return RunDirect12GatherAll<4, 5>(ctx, staticArg, false);
    }

    template <typename Channels>
    CcuResult RunDirect12PairBroadcastLayer(
        Direct12Context &ctx, const Channels &channels, uint32_t channelCount)
    {
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], 0));
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], 4));
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], 0));
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], 4));
        }
        ccu::Variable secondTileBytes;
        SelectFirstDirect12PairTile(ctx, secondTileBytes);
        uint16_t waitMask = 0;
        for (uint32_t index = 0; index < channelCount; ++index) {
            const uint16_t mask = static_cast<uint16_t>(1U << index);
            waitMask = static_cast<uint16_t>(waitMask | mask);
            const ccu::Variable addr = ccu::GetResByChannel<ccu::Variable>(
                channels[index], DIRECT12_OUTPUT_ADDR_VARIABLE);
            const ccu::Variable token = ccu::GetResByChannel<ccu::Variable>(
                channels[index], DIRECT12_OUTPUT_TOKEN_VARIABLE);
            PHASE0_CCU_CHK(ccu::Write(channels[index], Direct12RemoteTile(addr, token, ctx),
                Direct12LocalTile(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, ctx),
                ctx.tileBytes, ctx.event, mask));
        }
        SelectSecondDirect12PairTile(ctx, secondTileBytes);
        for (uint32_t index = 0; index < channelCount; ++index) {
            const uint16_t mask = static_cast<uint16_t>(1U << (index + 8));
            waitMask = static_cast<uint16_t>(waitMask | mask);
            const ccu::Variable addr = ccu::GetResByChannel<ccu::Variable>(
                channels[index], DIRECT12_OUTPUT_ADDR_VARIABLE);
            const ccu::Variable token = ccu::GetResByChannel<ccu::Variable>(
                channels[index], DIRECT12_OUTPUT_TOKEN_VARIABLE);
            PHASE0_CCU_CHK(ccu::Write(channels[index], Direct12RemoteTile(addr, token, ctx),
                Direct12LocalTile(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, ctx),
                ctx.tileBytes, ctx.event, mask));
        }
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], 1));
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], 5));
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], 1));
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], 5));
        }
        return CCU_SUCCESS;
    }

    template <typename StaticArg>
    CcuResult RunDirect12PairBroadcast(Direct12Context &ctx, const StaticArg &staticArg)
    {
        if (staticArg.l0ChannelCount != 0) {
            return RunDirect12PairBroadcastLayer(ctx, staticArg.l0Channels, staticArg.l0ChannelCount);
        }
        return RunDirect12PairBroadcastLayer(ctx, staticArg.l1Channels, staticArg.l1ChannelCount);
    }

    template <typename StaticArg>
    CcuResult RunDirect12PairTransition(Direct12Context &ctx, const StaticArg &staticArg)
    {
        ccu::Variable nextFirstTileBytes;
        nextFirstTileBytes = ctx.tileBytes;
        ccu::Variable firstScratchBank;
        firstScratchBank = ctx.taskArgs.scratchAddr;
        ccu::Variable nextFirstTileOffset;
        nextFirstTileOffset = ctx.taskArgs.tileOffsetBytes;
        ccu::Variable tileStrideBytes;
        tileStrideBytes = ctx.scratchStrideBytes;
        nextFirstTileOffset += tileStrideBytes;
        nextFirstTileOffset += tileStrideBytes;

        ccu::Variable fullTileBytes;
        fullTileBytes = ctx.scratchStrideBytes;
        ctx.tileBytes = fullTileBytes;
        PHASE0_CCU_CHK(RunDirect12PairBroadcast(ctx, staticArg));

        ctx.taskArgs.scratchAddr = firstScratchBank;
        ctx.taskArgs.tileOffsetBytes = nextFirstTileOffset;
        ctx.tileBytes = nextFirstTileBytes;
        PHASE0_CCU_CHK((RunDirect12GatherAll<0, 1>(ctx, staticArg, false)));
        CCU_IF(ctx.taskArgs.generation != 0)
        {
            ccu::Variable twice;
            twice = ctx.taskArgs.generation + ctx.taskArgs.generation;
            ccu::Variable nextSecondTileBytes;
            nextSecondTileBytes = twice + twice;
            SelectSecondDirect12PairTile(ctx, nextSecondTileBytes);
            PHASE0_CCU_CHK((RunDirect12GatherAll<4, 5>(ctx, staticArg, false)));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunDirect12PairStage(Direct12Context &ctx, const Direct12DieStaticArg &staticArg)
    {
        CCU_IF(ctx.taskArgs.flags == DIRECT12_FLAG_PAIR_GATHER_ALL_PUBLISH)
        {
            PHASE0_CCU_CHK((RunDirect12PairGather<true>(ctx, staticArg)));
        }
        CCU_IF(ctx.taskArgs.flags == DIRECT12_FLAG_PAIR_BROADCAST_ALL)
        {
            PHASE0_CCU_CHK(RunDirect12PairBroadcast(ctx, staticArg));
        }
        CCU_IF(ctx.taskArgs.flags == DIRECT12_FLAG_PAIR_BROADCAST_GATHER_NEXT)
        {
            PHASE0_CCU_CHK(RunDirect12PairTransition(ctx, staticArg));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunDirect12Stage(Direct12Context &ctx, const Direct12DieStaticArg &staticArg)
    {
        CCU_IF(ctx.taskArgs.flags == DIRECT12_FLAG_GATHER_ALL)
        {
            PHASE0_CCU_CHK((RunDirect12GatherAll<0, 1>(ctx, staticArg, false)));
        }
        CCU_IF(ctx.taskArgs.flags == DIRECT12_FLAG_BROADCAST_ALL)
        {
            PHASE0_CCU_CHK((RunDirect12BroadcastAll<0, 1>(ctx, staticArg)));
        }
        CCU_IF(ctx.taskArgs.flags == DIRECT12_FLAG_JOIN)
        {
            PHASE0_CCU_CHK(RunDirect12Join(ctx));
        }
        return CCU_SUCCESS;
    }

    template <typename StaticArg> CcuResult DispatchDirect12Stage(Direct12Context &ctx, const StaticArg &staticArg)
    {
        PHASE0_CCU_CHK(RunDirect12PairStage(ctx, staticArg));
        return RunDirect12Stage(ctx, staticArg);
    }

    template <typename Channels, typename Peers>
    CcuResult RunDirect12BroadcastGatherScalarLayer(Direct12Context &ctx,
        const Channels &channels, const Peers &peers, uint32_t channelCount,
        uint32_t canonicalRank, bool copyLocalInput)
    {
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], 0));
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], 4));
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], 0));
        }
        uint16_t waitMask = 0;
        uint32_t eventIndex = 0;
        const ccu::LocalAddr localOutput
            = Direct12LocalTile(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, ctx);
        for (uint32_t index = 0; index < channelCount; ++index, ++eventIndex) {
            const uint16_t mask = static_cast<uint16_t>(1U << eventIndex);
            waitMask = static_cast<uint16_t>(waitMask | mask);
            const ccu::Variable addr = ccu::GetResByChannel<ccu::Variable>(
                channels[index], DIRECT12_OUTPUT_ADDR_VARIABLE);
            const ccu::Variable token = ccu::GetResByChannel<ccu::Variable>(
                channels[index], DIRECT12_OUTPUT_TOKEN_VARIABLE);
            PHASE0_CCU_CHK(ccu::Write(channels[index], Direct12RemoteTile(addr, token, ctx),
                localOutput, ctx.tileBytes, ctx.event, mask));
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], 4));
        }
        ctx.taskArgs.dataOffsetBytes = PACKAGE_A_DIRECT12_TAIL_PREFIX_BYTES;
        ctx.taskArgs.tileOffsetBytes = 0;
        ctx.tileBytes = PACKAGE_A_DIRECT12_TAIL_SUFFIX_BYTES;
        if (copyLocalInput) {
            const uint16_t mask = static_cast<uint16_t>(1U << eventIndex++);
            waitMask = static_cast<uint16_t>(waitMask | mask);
            PHASE0_CCU_CHK(ccu::LocalCopy(Direct12ScratchLeaf(ctx, canonicalRank),
                Direct12LocalTile(ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, ctx),
                ctx.tileBytes, ctx.event, mask));
        }
        for (uint32_t index = 0; index < channelCount; ++index, ++eventIndex) {
            const uint16_t mask = static_cast<uint16_t>(1U << eventIndex);
            waitMask = static_cast<uint16_t>(waitMask | mask);
            const ccu::Variable addr = ccu::GetResByChannel<ccu::Variable>(
                channels[index], DIRECT12_INPUT_ADDR_VARIABLE);
            const ccu::Variable token = ccu::GetResByChannel<ccu::Variable>(
                channels[index], DIRECT12_INPUT_TOKEN_VARIABLE);
            PHASE0_CCU_CHK(ccu::Read(channels[index], Direct12ScratchLeaf(ctx, peers[index]),
                Direct12RemoteTile(addr, token, ctx), ctx.tileBytes, ctx.event, mask));
        }
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], 1));
            PHASE0_CCU_CHK(ccu::NotifyRecord(channels[index], 5));
        }
        for (uint32_t index = 0; index < channelCount; ++index) {
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], 1));
            PHASE0_CCU_CHK(ccu::NotifyWait(channels[index], 5));
        }
        return CCU_SUCCESS;
    }

    template <typename StaticArg>
    CcuResult RunDirect12BroadcastGatherScalar(Direct12Context &ctx, const StaticArg &staticArg)
    {
        if (staticArg.l0ChannelCount != 0) {
            return RunDirect12BroadcastGatherScalarLayer(ctx, staticArg.l0Channels,
                staticArg.l0PeerCanonicalRanks, staticArg.l0ChannelCount,
                staticArg.canonicalRank, staticArg.copyLocalInput != 0);
        }
        return RunDirect12BroadcastGatherScalarLayer(ctx, staticArg.l1Channels,
            staticArg.l1PeerCanonicalRanks, staticArg.l1ChannelCount, staticArg.canonicalRank, false);
    }

    CcuResult ReduceDirect12PrefixScalarSuffix(Direct12Context &ctx)
    {
        PHASE0_CCU_CHK(ReducePair(ctx, 0, 1, 1U << 0));
        PHASE0_CCU_CHK(ReducePair(ctx, 2, 3, 1U << 1));
        PHASE0_CCU_CHK(ReducePair(ctx, 4, 5, 1U << 2));
        PHASE0_CCU_CHK(ReducePair(ctx, 6, 7, 1U << 3));
        PHASE0_CCU_CHK(ReducePair(ctx, 8, 9, 1U << 4));
        PHASE0_CCU_CHK(ReducePair(ctx, 10, 11, 1U << 5));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 6) - 1));
        PHASE0_CCU_CHK(ReducePair(ctx, 0, 2, 1U << 0));
        PHASE0_CCU_CHK(ReducePair(ctx, 4, 6, 1U << 1));
        PHASE0_CCU_CHK(ReducePair(ctx, 8, 10, 1U << 2));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 3) - 1));
        PHASE0_CCU_CHK(ReducePair(ctx, 0, 4, 1U));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, 1U));
        PHASE0_CCU_CHK(ReducePair(ctx, 0, 8, 1U));
        return ccu::EventWait(ctx.event, 1U);
    }

    CcuResult JoinDirect12PrefixScalarSuffix(Direct12Context &ctx)
    {
        PHASE0_CCU_CHK(ReduceDirect12PrefixScalarSuffix(ctx));
        PHASE0_CCU_CHK(ccu::LocalCopy(
            Direct12LocalTile(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, ctx),
            Direct12ScratchLeaf(ctx, 0), ctx.tileBytes, ctx.event, 1U));
        return ccu::EventWait(ctx.event, 1U);
    }

    CcuResult DispatchDirect12PrefixScalarStage(
        Direct12Context &ctx, const Direct12DieStaticArg &staticArg)
    {
        CCU_IF(ctx.taskArgs.flags == DIRECT12_FLAG_PAIR_GATHER_ALL_PUBLISH)
        {
            PHASE0_CCU_CHK((RunDirect12PairGather<true>(ctx, staticArg)));
        }
        CCU_IF(ctx.taskArgs.flags == DIRECT12_FLAG_PAIR_BROADCAST_GATHER_NEXT)
        {
            PHASE0_CCU_CHK(RunDirect12PairTransition(ctx, staticArg));
        }
        CCU_IF(ctx.taskArgs.flags == DIRECT12_FLAG_JOIN)
        {
            PHASE0_CCU_CHK(RunDirect12Join(ctx));
        }
        CCU_IF(ctx.taskArgs.flags == DIRECT12_FLAG_BROADCAST_GATHER_SCALAR)
        {
            PHASE0_CCU_CHK(RunDirect12BroadcastGatherScalar(ctx, staticArg));
        }
        CCU_IF(ctx.taskArgs.flags == DIRECT12_FLAG_SCALAR_JOIN)
        {
            PHASE0_CCU_CHK(JoinDirect12PrefixScalarSuffix(ctx));
        }
        return CCU_SUCCESS;
    }

constexpr uint32_t FULL_EXCHANGE_INPUT_ADDR_VARIABLE = 0;
constexpr uint32_t FULL_EXCHANGE_INPUT_TOKEN_VARIABLE = 1;
constexpr uint16_t FULL_EXCHANGE_INPUT_ADDR_MASK = 1U << FULL_EXCHANGE_INPUT_ADDR_VARIABLE;
constexpr uint16_t FULL_EXCHANGE_INPUT_TOKEN_MASK = 1U << FULL_EXCHANGE_INPUT_TOKEN_VARIABLE;
constexpr uint16_t FULL_EXCHANGE_ADDRESS_TOKEN_MASK
    = FULL_EXCHANGE_INPUT_ADDR_MASK | FULL_EXCHANGE_INPUT_TOKEN_MASK;
struct FullExchangeTaskVariables {
    ccu::Variable inputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable dataOffsetBytes;
    ccu::Variable elementCount;
    ccu::Variable tileOffsetBytes;
    ccu::Variable tileElementCount;
    ccu::Variable generation;
    ccu::Variable flags;
};
struct FullExchangeContext {
    FullExchangeTaskVariables taskArgs;
    ccu::Variable tileBytes;
    ccu::Event event;
};
CcuResult ValidateFullExchangeStaticArg(const FullExchangeDieStaticArg &staticArg)
{
    if (!ValidDirect12CanonicalMetadata(staticArg.canonicalRank, staticArg.canonicalGroup,
            staticArg.canonicalGroup == 0 ? staticArg.canonicalRank : staticArg.canonicalRank - 8)
        || (staticArg.layer != FabricLayer::L0 && staticArg.layer != FabricLayer::L1)
        || staticArg.channelCount == 0 || staticArg.channelCount > MAX_PEER_CHANNELS
        || staticArg.copyLocalInput > 1
        || (staticArg.layer == FabricLayer::L0 && staticArg.copyLocalInput != 1)
        || (staticArg.layer == FabricLayer::L1 && staticArg.copyLocalInput != 0)) {
        return CCU_E_PARA;
    }
    const uint32_t group = staticArg.layer == FabricLayer::L0
                               ? staticArg.canonicalGroup
                               : 1U - staticArg.canonicalGroup;
    const uint32_t groupSize = group == 0 ? 8 : 4;
    const uint32_t expectedChannelCount = staticArg.layer == FabricLayer::L0 ? groupSize - 1 : groupSize;
    if (staticArg.channelCount != expectedChannelCount) {
        return CCU_E_PARA;
    }
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        if (staticArg.channels[index] == 0
            || staticArg.peerCanonicalRanks[index] >= DIRECT12_CANONICAL_LEAF_COUNT
            || staticArg.peerCanonicalRanks[index] == staticArg.canonicalRank
            || (staticArg.peerCanonicalRanks[index] < 8 ? 0U : 1U) != group) {
            return CCU_E_PARA;
        }
        for (uint32_t prior = 0; prior < index; ++prior) {
            if (staticArg.peerCanonicalRanks[prior] == staticArg.peerCanonicalRanks[index]) {
                return CCU_E_PARA;
            }
        }
    }
    const ccu::Variable dieAnchor
        = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[0], FULL_EXCHANGE_INPUT_ADDR_VARIABLE);
    (void)dieAnchor;
    return CCU_SUCCESS;
}
CcuResult LoadFullExchangeTaskArgs(FullExchangeContext &ctx)
{
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.inputAddr, static_cast<uint32_t>(TaskArgIndex::INPUT_ADDR)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.inputToken, static_cast<uint32_t>(TaskArgIndex::INPUT_TOKEN)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.outputAddr, static_cast<uint32_t>(TaskArgIndex::OUTPUT_ADDR)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.outputToken, static_cast<uint32_t>(TaskArgIndex::OUTPUT_TOKEN)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.scratchAddr, static_cast<uint32_t>(TaskArgIndex::SCRATCH_ADDR)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.scratchToken, static_cast<uint32_t>(TaskArgIndex::SCRATCH_TOKEN)));
    PHASE0_CCU_CHK(
        ccu::LoadArg(ctx.taskArgs.dataOffsetBytes, static_cast<uint32_t>(TaskArgIndex::DATA_OFFSET_BYTES)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.elementCount, static_cast<uint32_t>(TaskArgIndex::ELEMENT_COUNT)));
    PHASE0_CCU_CHK(
        ccu::LoadArg(ctx.taskArgs.tileOffsetBytes, static_cast<uint32_t>(TaskArgIndex::TILE_OFFSET_BYTES)));
    PHASE0_CCU_CHK(
        ccu::LoadArg(ctx.taskArgs.tileElementCount, static_cast<uint32_t>(TaskArgIndex::TILE_ELEMENT_COUNT)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.generation, static_cast<uint32_t>(TaskArgIndex::GENERATION)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.flags, static_cast<uint32_t>(TaskArgIndex::FLAGS)));
    ccu::Variable twice;
    twice = ctx.taskArgs.tileElementCount + ctx.taskArgs.tileElementCount;
    ctx.tileBytes = twice + twice;
    return CCU_SUCCESS;
}
ccu::LocalAddr FullExchangeLocalAddress(
    const ccu::Variable &base, const ccu::Variable &token, uint64_t offset)
{
    ccu::LocalAddr address;
    address.addr = base;
    ccu::Variable ccuOffset;
    ccuOffset = offset;
    address.addr += ccuOffset;
    address.token = token;
    return address;
}
ccu::LocalAddr FullExchangeScratchLeaf(const FullExchangeContext &ctx, uint32_t canonicalLeaf)
{
    return FullExchangeLocalAddress(
        ctx.taskArgs.scratchAddr, ctx.taskArgs.scratchToken, FullExchangeScratchLeafOffset(canonicalLeaf));
}
ccu::RemoteAddr FullExchangeRemoteInput(ccu::Variable base, ccu::Variable token)
{
    ccu::RemoteAddr address;
    address.addr = base;
    address.token = token;
    return address;
}
CcuResult PublishFullExchangeInput(
    FullExchangeContext &ctx, const FullExchangeDieStaticArg &staticArg)
{
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(staticArg.channels[index], ctx.taskArgs.inputAddr,
            FULL_EXCHANGE_INPUT_ADDR_VARIABLE, 0, FULL_EXCHANGE_INPUT_ADDR_MASK));
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(staticArg.channels[index], ctx.taskArgs.inputToken,
            FULL_EXCHANGE_INPUT_TOKEN_VARIABLE, 0, FULL_EXCHANGE_INPUT_TOKEN_MASK));
    }
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::NotifyWait(staticArg.channels[index], 0, FULL_EXCHANGE_ADDRESS_TOKEN_MASK));
    }
    return CCU_SUCCESS;
}
CcuResult RunFullExchangePartialTree(
    FullExchangeContext &ctx, const FullExchangeDieStaticArg &staticArg);
CcuResult RunFullExchangeGather(FullExchangeContext &ctx, const FullExchangeDieStaticArg &staticArg)
{
    uint16_t waitMask = 0;
    uint32_t eventIndex = 0;
    if (staticArg.copyLocalInput != 0) {
        waitMask = 1U;
        PHASE0_CCU_CHK(ccu::LocalCopy(FullExchangeScratchLeaf(ctx, staticArg.canonicalRank),
            FullExchangeLocalAddress(ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, 0), ctx.tileBytes,
            ctx.event, 1U));
        eventIndex = 1;
    }
    for (uint32_t index = 0; index < staticArg.channelCount; ++index, ++eventIndex) {
        const uint16_t eventMask = static_cast<uint16_t>(1U << eventIndex);
        waitMask = static_cast<uint16_t>(waitMask | eventMask);
        const ccu::Variable remoteInputAddr = ccu::GetResByChannel<ccu::Variable>(
            staticArg.channels[index], FULL_EXCHANGE_INPUT_ADDR_VARIABLE);
        const ccu::Variable remoteInputToken = ccu::GetResByChannel<ccu::Variable>(
            staticArg.channels[index], FULL_EXCHANGE_INPUT_TOKEN_VARIABLE);
        PHASE0_CCU_CHK(ccu::Read(staticArg.channels[index],
            FullExchangeScratchLeaf(ctx, staticArg.peerCanonicalRanks[index]),
            FullExchangeRemoteInput(remoteInputAddr, remoteInputToken), ctx.tileBytes, ctx.event, eventMask));
    }
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
    // ACK is recorded only after every input read on this physical link has completed. The join owner's stream
    // ordering plus the host's directed handoff from the other Die closes both Gathers before Join may safely
    // overwrite an in-place output buffer.
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::NotifyRecord(staticArg.channels[index], 1));
    }
    PHASE0_CCU_CHK(RunFullExchangePartialTree(ctx, staticArg));
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::NotifyWait(staticArg.channels[index], 1));
    }
    return CCU_SUCCESS;
}
CcuResult FullExchangeReducePair(
    FullExchangeContext &ctx, uint32_t destinationLeaf, uint32_t sourceLeaf, uint16_t eventMask)
{
    return ccu::LocalReduce(FullExchangeScratchLeaf(ctx, destinationLeaf),
        FullExchangeScratchLeaf(ctx, sourceLeaf), ctx.tileBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        ctx.event, eventMask);
}
CcuResult ReduceFullExchangeGroup0(FullExchangeContext &ctx)
{
    PHASE0_CCU_CHK(FullExchangeReducePair(ctx, 0, 1, 1U << 0));
    PHASE0_CCU_CHK(FullExchangeReducePair(ctx, 2, 3, 1U << 1));
    PHASE0_CCU_CHK(FullExchangeReducePair(ctx, 4, 5, 1U << 2));
    PHASE0_CCU_CHK(FullExchangeReducePair(ctx, 6, 7, 1U << 3));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 4) - 1));
    PHASE0_CCU_CHK(FullExchangeReducePair(ctx, 0, 2, 1U << 0));
    PHASE0_CCU_CHK(FullExchangeReducePair(ctx, 4, 6, 1U << 1));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 2) - 1));
    PHASE0_CCU_CHK(FullExchangeReducePair(ctx, 0, 4, 1U));
    return ccu::EventWait(ctx.event, 1U);
}
CcuResult ReduceFullExchangeGroup1(FullExchangeContext &ctx)
{
    PHASE0_CCU_CHK(FullExchangeReducePair(ctx, 8, 9, 1U << 0));
    PHASE0_CCU_CHK(FullExchangeReducePair(ctx, 10, 11, 1U << 1));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 2) - 1));
    PHASE0_CCU_CHK(FullExchangeReducePair(ctx, 8, 10, 1U));
    return ccu::EventWait(ctx.event, 1U);
}
CcuResult RunFullExchangePartialTree(
    FullExchangeContext &ctx, const FullExchangeDieStaticArg &staticArg)
{
    const uint32_t group = staticArg.layer == FabricLayer::L0
                               ? staticArg.canonicalGroup
                               : 1U - staticArg.canonicalGroup;
    if (group == 0) {
        return ReduceFullExchangeGroup0(ctx);
    }
    return ReduceFullExchangeGroup1(ctx);
}
CcuResult JoinFullExchangePartialTrees(FullExchangeContext &ctx)
{
    PHASE0_CCU_CHK(FullExchangeReducePair(ctx, 0, 8, 1U));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, 1U));
    PHASE0_CCU_CHK(ccu::LocalCopy(
        FullExchangeLocalAddress(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, 0),
        FullExchangeScratchLeaf(ctx, 0), ctx.tileBytes, ctx.event, 1U));
    return ccu::EventWait(ctx.event, 1U);
}
CcuResult DispatchFullExchangeStage(
    FullExchangeContext &ctx, const FullExchangeDieStaticArg &staticArg)
{
    CCU_IF(ctx.taskArgs.flags == FULL_EXCHANGE_FLAG_GATHER)
    {
        PHASE0_CCU_CHK(PublishFullExchangeInput(ctx, staticArg));
        PHASE0_CCU_CHK(RunFullExchangeGather(ctx, staticArg));
    }
    CCU_IF(ctx.taskArgs.flags == FULL_EXCHANGE_FLAG_JOIN)
    {
        PHASE0_CCU_CHK(JoinFullExchangePartialTrees(ctx));
    }
    return CCU_SUCCESS;
}
constexpr uint32_t FOUR_X_ONE_INPUT_ADDR_VARIABLE = 0;
constexpr uint32_t FOUR_X_ONE_INPUT_TOKEN_VARIABLE = 1;
constexpr uint32_t FOUR_X_ONE_OUTPUT_ADDR_VARIABLE = 2;
constexpr uint32_t FOUR_X_ONE_OUTPUT_TOKEN_VARIABLE = 3;
constexpr uint16_t FOUR_X_ONE_INPUT_ADDR_MASK = 1U << FOUR_X_ONE_INPUT_ADDR_VARIABLE;
constexpr uint16_t FOUR_X_ONE_INPUT_TOKEN_MASK = 1U << FOUR_X_ONE_INPUT_TOKEN_VARIABLE;
constexpr uint16_t FOUR_X_ONE_OUTPUT_ADDR_MASK = 1U << FOUR_X_ONE_OUTPUT_ADDR_VARIABLE;
constexpr uint16_t FOUR_X_ONE_OUTPUT_TOKEN_MASK = 1U << FOUR_X_ONE_OUTPUT_TOKEN_VARIABLE;
constexpr uint16_t FOUR_X_ONE_INPUT_ADDRESS_TOKEN_MASK
    = FOUR_X_ONE_INPUT_ADDR_MASK | FOUR_X_ONE_INPUT_TOKEN_MASK;
constexpr uint16_t FOUR_X_ONE_OUTPUT_ADDRESS_TOKEN_MASK
    = FOUR_X_ONE_OUTPUT_ADDR_MASK | FOUR_X_ONE_OUTPUT_TOKEN_MASK;
constexpr uint16_t FOUR_X_ONE_RING_MATERIALIZE_EVENT = 2U;
struct FourXOneTaskVariables {
    ccu::Variable inputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable dataOffsetBytes;
    ccu::Variable elementCount;
    ccu::Variable tileOffsetBytes;
    ccu::Variable tileElementCount;
    ccu::Variable generation;
    ccu::Variable flags;
};
struct FourXOneContext {
    FourXOneTaskVariables taskArgs;
    ccu::Variable bytes;
    ccu::Variable tailOffsetBytes;
    ccu::Event event;
};
template <typename StaticArg> CcuResult ValidateFourXOnePeerMetadata(const StaticArg &staticArg)
{
    if (staticArg.canonicalRank >= FOUR_X_ONE_RANK_COUNT || staticArg.channelCount == 0
        || staticArg.channelCount > FOUR_X_ONE_RANK_COUNT - 1) {
        return CCU_E_PARA;
    }
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        if (staticArg.channels[index] == 0 || staticArg.peerCanonicalRanks[index] >= FOUR_X_ONE_RANK_COUNT
            || staticArg.peerCanonicalRanks[index] == staticArg.canonicalRank) {
            return CCU_E_PARA;
        }
        for (uint32_t prior = 0; prior < index; ++prior) {
            if (staticArg.peerCanonicalRanks[prior] == staticArg.peerCanonicalRanks[index]) {
                return CCU_E_PARA;
            }
        }
    }
    const ccu::Variable dieAnchor
        = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[0], FOUR_X_ONE_INPUT_ADDR_VARIABLE);
    (void)dieAnchor;
    return CCU_SUCCESS;
}
CcuResult ValidateFourXOneRingStaticArg(const FourXOneRingStaticArg &staticArg)
{
    if (staticArg.channelCount != 2
        || staticArg.peerCanonicalRanks[0]
               != (staticArg.canonicalRank + 1) % FOUR_X_ONE_RANK_COUNT
        || staticArg.peerCanonicalRanks[1]
               != (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT - 1) % FOUR_X_ONE_RANK_COUNT) {
        return CCU_E_PARA;
    }
    return ValidateFourXOnePeerMetadata(staticArg);
}
template <typename Context> CcuResult LoadFourXOneTaskArgs(Context &ctx)
{
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.inputAddr, static_cast<uint32_t>(TaskArgIndex::INPUT_ADDR)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.inputToken, static_cast<uint32_t>(TaskArgIndex::INPUT_TOKEN)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.outputAddr, static_cast<uint32_t>(TaskArgIndex::OUTPUT_ADDR)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.outputToken, static_cast<uint32_t>(TaskArgIndex::OUTPUT_TOKEN)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.scratchAddr, static_cast<uint32_t>(TaskArgIndex::SCRATCH_ADDR)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.scratchToken, static_cast<uint32_t>(TaskArgIndex::SCRATCH_TOKEN)));
    PHASE0_CCU_CHK(
        ccu::LoadArg(ctx.taskArgs.dataOffsetBytes, static_cast<uint32_t>(TaskArgIndex::DATA_OFFSET_BYTES)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.elementCount, static_cast<uint32_t>(TaskArgIndex::ELEMENT_COUNT)));
    PHASE0_CCU_CHK(
        ccu::LoadArg(ctx.taskArgs.tileOffsetBytes, static_cast<uint32_t>(TaskArgIndex::TILE_OFFSET_BYTES)));
    PHASE0_CCU_CHK(
        ccu::LoadArg(ctx.taskArgs.tileElementCount, static_cast<uint32_t>(TaskArgIndex::TILE_ELEMENT_COUNT)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.generation, static_cast<uint32_t>(TaskArgIndex::GENERATION)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.flags, static_cast<uint32_t>(TaskArgIndex::FLAGS)));
    ccu::Variable twice;
    twice = ctx.taskArgs.tileElementCount + ctx.taskArgs.tileElementCount;
    ctx.bytes = twice + twice;
    return CCU_SUCCESS;
}
ccu::LocalAddr FourXOneLocal(const ccu::Variable &base, const ccu::Variable &token)
{
    ccu::LocalAddr address;
    address.addr = base;
    address.token = token;
    return address;
}
ccu::RemoteAddr FourXOneRemote(ccu::Variable base, ccu::Variable token)
{
    ccu::RemoteAddr address;
    address.addr = base;
    address.token = token;
    return address;
}
ccu::LocalAddr FourXOneRingLocalSlice(
    const FourXOneContext &ctx, const ccu::Variable &base, const ccu::Variable &token, uint32_t chunk)
{
    ccu::LocalAddr address = FourXOneLocal(base, token);
    ccu::Variable chunkOffset;
    chunkOffset = 0;
    for (uint32_t index = 0; index < chunk; ++index) {
        chunkOffset += ctx.taskArgs.dataOffsetBytes;
    }
    address.addr += chunkOffset;
    address.addr += ctx.taskArgs.tileOffsetBytes;
    return address;
}
ccu::RemoteAddr FourXOneRingRemoteSlice(
    const FourXOneContext &ctx, ccu::Variable base, ccu::Variable token, uint32_t chunk)
{
    ccu::RemoteAddr address = FourXOneRemote(base, token);
    ccu::Variable chunkOffset;
    chunkOffset = 0;
    for (uint32_t index = 0; index < chunk; ++index) {
        chunkOffset += ctx.taskArgs.dataOffsetBytes;
    }
    address.addr += chunkOffset;
    address.addr += ctx.taskArgs.tileOffsetBytes;
    return address;
}
const ccu::Variable &FourXOneTailRingTransferBytes(const FourXOneContext &ctx,
    const ccu::Variable &unevenBytes, uint32_t chunk)
{
    return chunk == FOUR_X_ONE_RING_UNEVEN_OWNER_CHUNK ? unevenBytes : ctx.bytes;
}
template <uint32_t ReadyNotify, uint32_t VariableBase>
CcuResult PublishFourXOneRingOutput(FourXOneContext &ctx, const FourXOneRingStaticArg &staticArg)
{
    constexpr uint16_t outputAddressTokenMask
        = static_cast<uint16_t>(FOUR_X_ONE_OUTPUT_ADDRESS_TOKEN_MASK << VariableBase);
    const ChannelHandle nextChannel = staticArg.channels[0];
    const ChannelHandle previousChannel = staticArg.channels[1];
    PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(previousChannel, ctx.taskArgs.outputAddr,
        VariableBase + FOUR_X_ONE_OUTPUT_ADDR_VARIABLE, ReadyNotify,
        static_cast<uint16_t>(FOUR_X_ONE_OUTPUT_ADDR_MASK << VariableBase)));
    PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(previousChannel, ctx.taskArgs.outputToken,
        VariableBase + FOUR_X_ONE_OUTPUT_TOKEN_VARIABLE, ReadyNotify,
        static_cast<uint16_t>(FOUR_X_ONE_OUTPUT_TOKEN_MASK << VariableBase)));
    return ccu::NotifyWait(nextChannel, ReadyNotify, outputAddressTokenMask);
}
template <uint32_t ReadyNotify, uint32_t VariableBase>
CcuResult FourXOneRingInitialize(FourXOneContext &ctx, const FourXOneRingStaticArg &staticArg)
{
    uint16_t waitMask = 0;
    // The first outgoing chunk is sourced directly from input. Only chunks that
    // receive WriteReduce traffic need to be materialized in output first.
    for (uint32_t step = 0; step < FOUR_X_ONE_RANK_COUNT - 1; ++step) {
        const uint32_t chunk = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT - 1 - step)
                               % FOUR_X_ONE_RANK_COUNT;
        const uint16_t eventMask = static_cast<uint16_t>(1U << chunk);
        waitMask = static_cast<uint16_t>(waitMask | eventMask);
        PHASE0_CCU_CHK(ccu::LocalCopy(
            FourXOneRingLocalSlice(ctx, ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, chunk),
            FourXOneRingLocalSlice(ctx, ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, chunk),
            ctx.bytes, ctx.event, eventMask));
    }
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
    // Publication remains behind all destination copies and therefore preserves
    // the original in-place source-read proof.
    return PublishFourXOneRingOutput<ReadyNotify, VariableBase>(ctx, staticArg);
}
template <uint32_t ReadyNotify, uint32_t VariableBase>
CcuResult FourXOneTailRingInitialize(FourXOneContext &ctx,
    const FourXOneRingStaticArg &staticArg, const ccu::Variable &unevenBytes)
{
    // The first outgoing chunk is sourced directly from input. Only chunks that
    // receive the first WriteReduce traffic must be materialized before publish.
    const uint32_t firstDestinationChunk
        = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT - 1) % FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK(ccu::LocalCopy(FourXOneRingLocalSlice(
        ctx, ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, firstDestinationChunk),
        FourXOneRingLocalSlice(
            ctx, ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, firstDestinationChunk),
        FourXOneTailRingTransferBytes(ctx, unevenBytes, firstDestinationChunk), ctx.event,
        FOUR_X_ONE_RING_MATERIALIZE_EVENT));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, FOUR_X_ONE_RING_MATERIALIZE_EVENT));
    return PublishFourXOneRingOutput<ReadyNotify, VariableBase>(ctx, staticArg);
}
template <uint32_t DataNotify, uint16_t ReadyMask, uint16_t AckMask,
    uint32_t VariableBase, uint16_t NextReadyMask = 0,
    bool RecordCurrentReady = true, bool PublishNextReady = false,
    bool SourceInput = false>
CcuResult RunFourXOneRingWritePhase(
    FourXOneContext &ctx, const FourXOneRingStaticArg &staticArg, uint32_t chunk, bool reduce)
{
    const ChannelHandle nextChannel = staticArg.channels[0];
    const ChannelHandle previousChannel = staticArg.channels[1];
    if (RecordCurrentReady) {
        PHASE0_CCU_CHK(ccu::NotifyRecord(nextChannel, DataNotify, ReadyMask));
    }
    PHASE0_CCU_CHK(ccu::NotifyWait(previousChannel, DataNotify, ReadyMask));
    const ccu::Variable remoteOutputAddr
        = ccu::GetResByChannel<ccu::Variable>(
            nextChannel, VariableBase + FOUR_X_ONE_OUTPUT_ADDR_VARIABLE);
    const ccu::Variable remoteOutputToken
        = ccu::GetResByChannel<ccu::Variable>(
            nextChannel, VariableBase + FOUR_X_ONE_OUTPUT_TOKEN_VARIABLE);
    const ccu::LocalAddr localOutput = SourceInput
        ? FourXOneRingLocalSlice(ctx, ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, chunk)
        : FourXOneRingLocalSlice(ctx, ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, chunk);
    const ccu::RemoteAddr remoteOutput
        = FourXOneRingRemoteSlice(ctx, remoteOutputAddr, remoteOutputToken, chunk);
    if (reduce) {
        PHASE0_CCU_CHK(ccu::WriteReduce(nextChannel, remoteOutput, localOutput, ctx.bytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 1U));
    } else {
        PHASE0_CCU_CHK(ccu::Write(nextChannel, remoteOutput, localOutput, ctx.bytes, ctx.event, 1U));
    }
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, 1U));
    PHASE0_CCU_CHK(ccu::NotifyRecord(nextChannel, DataNotify, AckMask));
    if (PublishNextReady) {
        PHASE0_CCU_CHK(ccu::NotifyRecord(nextChannel, DataNotify, NextReadyMask));
    }
    return ccu::NotifyWait(previousChannel, DataNotify, AckMask);
}
template <uint32_t DataNotify, uint16_t ReadyMask, uint16_t AckMask,
    uint32_t VariableBase, uint16_t NextReadyMask = 0,
    bool RecordCurrentReady = true, bool PublishNextReady = false,
    bool SourceInput = false>
CcuResult RunFourXOneTailRingWritePhase(FourXOneContext &ctx,
    const FourXOneRingStaticArg &staticArg, const ccu::Variable &unevenBytes,
    uint32_t chunk, bool reduce)
{
    const ChannelHandle nextChannel = staticArg.channels[0];
    const ChannelHandle previousChannel = staticArg.channels[1];
    if (RecordCurrentReady) {
        PHASE0_CCU_CHK(ccu::NotifyRecord(nextChannel, DataNotify, ReadyMask));
    }
    PHASE0_CCU_CHK(ccu::NotifyWait(previousChannel, DataNotify, ReadyMask));
    const ccu::Variable remoteOutputAddr
        = ccu::GetResByChannel<ccu::Variable>(
            nextChannel, VariableBase + FOUR_X_ONE_OUTPUT_ADDR_VARIABLE);
    const ccu::Variable remoteOutputToken
        = ccu::GetResByChannel<ccu::Variable>(
            nextChannel, VariableBase + FOUR_X_ONE_OUTPUT_TOKEN_VARIABLE);
    const ccu::LocalAddr localOutput = SourceInput
        ? FourXOneRingLocalSlice(ctx, ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, chunk)
        : FourXOneRingLocalSlice(ctx, ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, chunk);
    const ccu::RemoteAddr remoteOutput
        = FourXOneRingRemoteSlice(ctx, remoteOutputAddr, remoteOutputToken, chunk);
    const ccu::Variable transferBytes
        = FourXOneTailRingTransferBytes(ctx, unevenBytes, chunk);
    if (reduce) {
        PHASE0_CCU_CHK(ccu::WriteReduce(nextChannel, remoteOutput, localOutput, transferBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 1U));
    } else {
        PHASE0_CCU_CHK(ccu::Write(
            nextChannel, remoteOutput, localOutput, transferBytes, ctx.event, 1U));
    }
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, 1U));
    PHASE0_CCU_CHK(ccu::NotifyRecord(nextChannel, DataNotify, AckMask));
    if (PublishNextReady) {
        PHASE0_CCU_CHK(ccu::NotifyRecord(nextChannel, DataNotify, NextReadyMask));
    }
    return ccu::NotifyWait(previousChannel, DataNotify, AckMask);
}
template <uint32_t Worker, uint32_t VariableBase>
CcuResult RunFourXOneRingReduceScatter(
    FourXOneContext &ctx, const FourXOneRingStaticArg &staticArg)
{
    const uint32_t firstChunk = staticArg.canonicalRank;
    PHASE0_CCU_CHK((RunFourXOneRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(0), FourXOneRingAckMask(0), VariableBase,
        FourXOneRingReadyMask(1), true, true, true>(ctx, staticArg, firstChunk, true)));
    const uint32_t secondChunk
        = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT - 1) % FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK((RunFourXOneRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(1), FourXOneRingAckMask(1), VariableBase,
        FourXOneRingReadyMask(2), false, true>(ctx, staticArg, secondChunk, true)));
    const uint32_t thirdChunk
        = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT - 2) % FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK((RunFourXOneRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(2), FourXOneRingAckMask(2), VariableBase,
        FourXOneRingReadyMask(3), false, true>(ctx, staticArg, thirdChunk, true)));
    return CCU_SUCCESS;
}
template <uint32_t Worker>
CcuResult FourXOneTailDestinationReady(const FourXOneRingStaticArg &staticArg)
{
    const ChannelHandle nextChannel = staticArg.channels[0];
    const ChannelHandle previousChannel = staticArg.channels[1];
    PHASE0_CCU_CHK(ccu::NotifyRecord(previousChannel, FourXOneRingDataNotify(Worker),
        FOUR_X_ONE_RING_TAIL_READY_MASK));
    return ccu::NotifyWait(nextChannel, FourXOneRingDataNotify(Worker),
        FOUR_X_ONE_RING_TAIL_READY_MASK);
}
template <uint32_t Worker, uint32_t VariableBase>
CcuResult RunFourXOneTailRingReduceScatter(
    FourXOneContext &ctx, const FourXOneRingStaticArg &staticArg,
    const ccu::Variable &unevenBytes)
{
    const uint32_t secondDestinationChunk
        = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT - 2) % FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK(ccu::LocalCopy(FourXOneRingLocalSlice(
        ctx, ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, secondDestinationChunk),
        FourXOneRingLocalSlice(
            ctx, ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, secondDestinationChunk),
        FourXOneTailRingTransferBytes(ctx, unevenBytes, secondDestinationChunk), ctx.event,
        FOUR_X_ONE_RING_MATERIALIZE_EVENT));
    const uint32_t firstChunk = staticArg.canonicalRank;
    PHASE0_CCU_CHK((RunFourXOneTailRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(0), FourXOneRingAckMask(0), VariableBase,
        FourXOneRingReadyMask(1), true, true, true>(
            ctx, staticArg, unevenBytes, firstChunk, true)));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, FOUR_X_ONE_RING_MATERIALIZE_EVENT));
    PHASE0_CCU_CHK((FourXOneTailDestinationReady<Worker>(staticArg)));
    const uint32_t thirdDestinationChunk
        = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT - 3) % FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK(ccu::LocalCopy(FourXOneRingLocalSlice(
        ctx, ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, thirdDestinationChunk),
        FourXOneRingLocalSlice(
            ctx, ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, thirdDestinationChunk),
        FourXOneTailRingTransferBytes(ctx, unevenBytes, thirdDestinationChunk), ctx.event,
        FOUR_X_ONE_RING_MATERIALIZE_EVENT));
    const uint32_t secondChunk
        = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT - 1) % FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK((RunFourXOneTailRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(1), FourXOneRingAckMask(1), VariableBase,
        FourXOneRingReadyMask(2), false, true>(
            ctx, staticArg, unevenBytes, secondChunk, true)));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, FOUR_X_ONE_RING_MATERIALIZE_EVENT));
    PHASE0_CCU_CHK((FourXOneTailDestinationReady<Worker>(staticArg)));
    const uint32_t thirdChunk
        = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT - 2) % FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK((RunFourXOneTailRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(2), FourXOneRingAckMask(2), VariableBase,
        FourXOneRingReadyMask(3), false, true>(
            ctx, staticArg, unevenBytes, thirdChunk, true)));
    return CCU_SUCCESS;
}
template <uint32_t Worker, uint32_t VariableBase>
CcuResult RunFourXOneRingAllGather(
    FourXOneContext &ctx, const FourXOneRingStaticArg &staticArg)
{
    const uint32_t firstChunk
        = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT + 1) % FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK((RunFourXOneRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(3), FourXOneRingAckMask(3), VariableBase,
        FourXOneRingReadyMask(4), false, true>(ctx, staticArg, firstChunk, false)));
    const uint32_t secondChunk = staticArg.canonicalRank;
    PHASE0_CCU_CHK((RunFourXOneRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(4), FourXOneRingAckMask(4), VariableBase,
        FourXOneRingReadyMask(5), false, true>(ctx, staticArg, secondChunk, false)));
    const uint32_t thirdChunk
        = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT - 1) % FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK((RunFourXOneRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(5), FourXOneRingAckMask(5), VariableBase,
        0, false, false>(ctx, staticArg, thirdChunk, false)));
    return CCU_SUCCESS;
}
template <uint32_t Worker, uint32_t VariableBase>
CcuResult RunFourXOneTailRingAllGather(FourXOneContext &ctx,
    const FourXOneRingStaticArg &staticArg, const ccu::Variable &unevenBytes)
{
    const uint32_t firstChunk
        = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT + 1) % FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK((RunFourXOneTailRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(3), FourXOneRingAckMask(3), VariableBase,
        FourXOneRingReadyMask(4), false, true>(
            ctx, staticArg, unevenBytes, firstChunk, false)));
    const uint32_t secondChunk = staticArg.canonicalRank;
    PHASE0_CCU_CHK((RunFourXOneTailRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(4), FourXOneRingAckMask(4), VariableBase,
        FourXOneRingReadyMask(5), false, true>(
            ctx, staticArg, unevenBytes, secondChunk, false)));
    const uint32_t thirdChunk
        = (staticArg.canonicalRank + FOUR_X_ONE_RANK_COUNT - 1) % FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK((RunFourXOneTailRingWritePhase<FourXOneRingDataNotify(Worker),
        FourXOneRingReadyMask(5), FourXOneRingAckMask(5), VariableBase,
        0, false, false>(ctx, staticArg, unevenBytes, thirdChunk, false)));
    return CCU_SUCCESS;
}
template <uint32_t Worker, uint32_t VariableBase>
CcuResult RunFourXOneRingStage(FourXOneContext &ctx, const FourXOneRingStaticArg &staticArg)
{
    PHASE0_CCU_CHK((FourXOneRingInitialize<FourXOneRingAddressNotify(Worker),
        VariableBase>(ctx, staticArg)));
    PHASE0_CCU_CHK((RunFourXOneRingReduceScatter<Worker, VariableBase>(ctx, staticArg)));
    return RunFourXOneRingAllGather<Worker, VariableBase>(ctx, staticArg);
}
template <uint32_t Worker, uint32_t VariableBase>
CcuResult RunFourXOneTailRingStage(FourXOneContext &ctx,
    const FourXOneRingStaticArg &staticArg, const ccu::Variable &unevenBytes)
{
    PHASE0_CCU_CHK((FourXOneTailRingInitialize<FourXOneRingAddressNotify(Worker),
        VariableBase>(ctx, staticArg, unevenBytes)));
    PHASE0_CCU_CHK((RunFourXOneTailRingReduceScatter<Worker, VariableBase>(
        ctx, staticArg, unevenBytes)));
    return RunFourXOneTailRingAllGather<Worker, VariableBase>(
        ctx, staticArg, unevenBytes);
}
constexpr uint16_t FOUR_X_ONE_PULL_GEN0_ACQUIRE_MASK = 0x000f;
constexpr uint16_t FOUR_X_ONE_PULL_LEFT01_EVENT = 0x4000;
constexpr uint16_t FOUR_X_ONE_PULL_LEFT23_EVENT = 0x8000;
struct FourXOnePullContext {
    FourXOneTaskVariables taskArgs;
    ccu::Variable bytes;
    ccu::Variable tailOffsetBytes;
    ccu::Variable ownerOffsetBytes;
    std::array<ccu::Variable, FOUR_X_ONE_PULL_GENERATIONS> tileOffsetBytes;
    std::array<ccu::Variable, FOUR_X_ONE_PEER_COUNT> remoteInputAddr;
    std::array<ccu::Variable, FOUR_X_ONE_PEER_COUNT> remoteInputToken;
    std::array<ccu::Variable, FOUR_X_ONE_PEER_COUNT> remoteOutputAddr;
    std::array<ccu::Variable, FOUR_X_ONE_PEER_COUNT> remoteOutputToken;
    ccu::Event event;
};
CcuResult ValidateFourXOnePullStaticArg(
    const FourXOnePullStaticArg &staticArg, bool fullExchange)
{
    if (staticArg.worker >= ACTIVE_WORKERS || staticArg.hasScalarTail > 1
        || staticArg.channelCount != FOUR_X_ONE_PEER_COUNT
        || (fullExchange && (staticArg.worker != 0 || staticArg.hasScalarTail != 0))) {
        return CCU_E_PARA;
    }
    PHASE0_CCU_CHK(ValidateFourXOnePeerMetadata(staticArg));
    uint32_t expectedPeer = 0;
    for (uint32_t slot = 0; slot < staticArg.channelCount; ++slot) {
        if (expectedPeer == staticArg.canonicalRank) {
            ++expectedPeer;
        }
        if (staticArg.peerCanonicalRanks[slot] != expectedPeer) {
            return CCU_E_PARA;
        }
        ++expectedPeer;
    }
    return CCU_SUCCESS;
}
template <uint32_t Worker>
CcuResult BindFourXOnePullInputResources(
    FourXOnePullContext &ctx, const FourXOnePullStaticArg &staticArg)
{
    constexpr uint32_t variableBase = FourXOneVariableBase(Worker);
    for (uint32_t slot = 0; slot < FOUR_X_ONE_PEER_COUNT; ++slot) {
        const ChannelHandle channel = staticArg.channels[slot];
        ctx.remoteInputAddr[slot] = ccu::GetResByChannel<ccu::Variable>(
            channel, variableBase + FOUR_X_ONE_INPUT_ADDR_FIELD);
        ctx.remoteInputToken[slot] = ccu::GetResByChannel<ccu::Variable>(
            channel, variableBase + FOUR_X_ONE_INPUT_TOKEN_FIELD);
    }
    return CCU_SUCCESS;
}
template <uint32_t Worker, uint32_t AddressNotify>
CcuResult PublishFourXOnePullInputAddresses(
    FourXOnePullContext &ctx, const FourXOnePullStaticArg &staticArg)
{
    constexpr uint32_t variableBase = FourXOneVariableBase(Worker);
    constexpr uint16_t inputAddrMask
        = FourXOneAddressFieldMask(FOUR_X_ONE_INPUT_ADDR_FIELD);
    constexpr uint16_t inputTokenMask
        = FourXOneAddressFieldMask(FOUR_X_ONE_INPUT_TOKEN_FIELD);
    constexpr uint16_t inputMask
        = static_cast<uint16_t>(inputAddrMask | inputTokenMask);
    for (uint32_t slot = 0; slot < FOUR_X_ONE_PEER_COUNT; ++slot) {
        const ChannelHandle channel = staticArg.channels[slot];
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(channel, ctx.taskArgs.inputAddr,
            variableBase + FOUR_X_ONE_INPUT_ADDR_FIELD, AddressNotify, inputAddrMask));
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(channel, ctx.taskArgs.inputToken,
            variableBase + FOUR_X_ONE_INPUT_TOKEN_FIELD, AddressNotify, inputTokenMask));
    }
    for (uint32_t slot = 0; slot < FOUR_X_ONE_PEER_COUNT; ++slot) {
        PHASE0_CCU_CHK(ccu::NotifyWait(
            staticArg.channels[slot], AddressNotify, inputMask));
    }
    return CCU_SUCCESS;
}
template <uint32_t Notify, uint16_t RoleMask>
CcuResult FourXOnePullPeerProof(const FourXOnePullStaticArg &staticArg)
{
    for (uint32_t slot = 0; slot < FOUR_X_ONE_PEER_COUNT; ++slot) {
        PHASE0_CCU_CHK(ccu::NotifyRecord(staticArg.channels[slot], Notify, RoleMask));
    }
    for (uint32_t slot = 0; slot < FOUR_X_ONE_PEER_COUNT; ++slot) {
        PHASE0_CCU_CHK(ccu::NotifyWait(staticArg.channels[slot], Notify, RoleMask));
    }
    return CCU_SUCCESS;
}
ccu::LocalAddr FourXOnePullLocalTile(FourXOnePullContext &ctx,
    const ccu::Variable &base, const ccu::Variable &token, uint32_t generation)
{
    ccu::LocalAddr address = FourXOneLocal(base, token);
    address.addr += ctx.ownerOffsetBytes;
    address.addr += ctx.tileOffsetBytes[generation];
    return address;
}
ccu::RemoteAddr FourXOnePullRemoteTile(FourXOnePullContext &ctx,
    ccu::Variable base, ccu::Variable token, uint32_t generation)
{
    ccu::RemoteAddr address = FourXOneRemote(base, token);
    address.addr += ctx.ownerOffsetBytes;
    address.addr += ctx.tileOffsetBytes[generation];
    return address;
}
ccu::LocalAddr FourXOnePullScratchSlot(
    FourXOnePullContext &ctx, uint32_t generation, uint32_t slot)
{
    ccu::LocalAddr address
        = FourXOneLocal(ctx.taskArgs.scratchAddr, ctx.taskArgs.scratchToken);
    const uint32_t linearSlot = generation * FOUR_X_ONE_PEER_COUNT + slot;
    for (uint32_t index = 0; index < linearSlot; ++index) {
        address.addr += ctx.bytes;
    }
    return address;
}
ccu::LocalAddr FourXOnePullOperand(FourXOnePullContext &ctx,
    const FourXOnePullStaticArg &staticArg, uint32_t generation, uint32_t source)
{
    if (source == staticArg.canonicalRank) {
        return FourXOnePullLocalTile(
            ctx, ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, generation);
    }
    const uint32_t slot = source < staticArg.canonicalRank ? source : source - 1;
    return FourXOnePullScratchSlot(ctx, generation, slot);
}
template <uint32_t Generation, uint32_t Source, uint16_t EventMask>
CcuResult StartFourXOnePullOperand(FourXOnePullContext &ctx,
    const FourXOnePullStaticArg &staticArg)
{
    const ccu::LocalAddr destination
        = FourXOnePullOperand(ctx, staticArg, Generation, Source);
    if (Source == staticArg.canonicalRank) {
        return ccu::LocalCopy(destination,
            FourXOnePullLocalTile(
                ctx, ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, Generation),
            ctx.bytes, ctx.event, EventMask);
    }
    const uint32_t slot = Source < staticArg.canonicalRank ? Source : Source - 1;
    return ccu::Read(staticArg.channels[slot], destination,
        FourXOnePullRemoteTile(ctx, ctx.remoteInputAddr[slot],
            ctx.remoteInputToken[slot], Generation),
        ctx.bytes, ctx.event, EventMask);
}
template <uint32_t Generation>
CcuResult StartFourXOnePullGeneration(FourXOnePullContext &ctx,
    const FourXOnePullStaticArg &staticArg)
{
    constexpr uint32_t eventBase = Generation * FOUR_X_ONE_RANK_COUNT;
    PHASE0_CCU_CHK((StartFourXOnePullOperand<Generation, 0,
        static_cast<uint16_t>(1U << (eventBase + 0))>(ctx, staticArg)));
    PHASE0_CCU_CHK((StartFourXOnePullOperand<Generation, 1,
        static_cast<uint16_t>(1U << (eventBase + 1))>(ctx, staticArg)));
    PHASE0_CCU_CHK((StartFourXOnePullOperand<Generation, 2,
        static_cast<uint16_t>(1U << (eventBase + 2))>(ctx, staticArg)));
    return StartFourXOnePullOperand<Generation, 3,
        static_cast<uint16_t>(1U << (eventBase + 3))>(ctx, staticArg);
}
CcuResult ReduceFourXOnePullFixedTree(FourXOnePullContext &ctx,
    const FourXOnePullStaticArg &staticArg, uint32_t generation)
{
    const ccu::LocalAddr operand0 = FourXOnePullOperand(ctx, staticArg, generation, 0);
    const ccu::LocalAddr operand1 = FourXOnePullOperand(ctx, staticArg, generation, 1);
    const ccu::LocalAddr operand2 = FourXOnePullOperand(ctx, staticArg, generation, 2);
    const ccu::LocalAddr operand3 = FourXOnePullOperand(ctx, staticArg, generation, 3);
    PHASE0_CCU_CHK(ccu::LocalReduce(operand0, operand1, ctx.bytes,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, FOUR_X_ONE_PULL_LEFT01_EVENT));
    PHASE0_CCU_CHK(ccu::LocalReduce(operand2, operand3, ctx.bytes,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, FOUR_X_ONE_PULL_LEFT23_EVENT));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event,
        FOUR_X_ONE_PULL_LEFT01_EVENT | FOUR_X_ONE_PULL_LEFT23_EVENT));
    PHASE0_CCU_CHK(ccu::LocalReduce(operand0, operand2, ctx.bytes,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, FOUR_X_ONE_PULL_LEFT01_EVENT));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, FOUR_X_ONE_PULL_LEFT01_EVENT));
    if (staticArg.canonicalRank != 0) {
        PHASE0_CCU_CHK(ccu::LocalCopy(
            FourXOnePullLocalTile(ctx, ctx.taskArgs.outputAddr,
                ctx.taskArgs.outputToken, generation),
            operand0, ctx.bytes, ctx.event, FOUR_X_ONE_PULL_LEFT01_EVENT));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, FOUR_X_ONE_PULL_LEFT01_EVENT));
    }
    return CCU_SUCCESS;
}
CcuResult RunFourXOnePullFullExchange(FourXOnePullContext &ctx,
    const FourXOnePullStaticArg &staticArg)
{
    PHASE0_CCU_CHK((BindFourXOnePullInputResources<0>(ctx, staticArg)));
    PHASE0_CCU_CHK((PublishFourXOnePullInputAddresses<0, 0>(ctx, staticArg)));
    PHASE0_CCU_CHK((StartFourXOnePullGeneration<0>(ctx, staticArg)));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, FOUR_X_ONE_PULL_GEN0_ACQUIRE_MASK));
    CCU_IF(ctx.taskArgs.flags != FOUR_X_ONE_PULL_FLAG_DISJOINT)
    {
        PHASE0_CCU_CHK((FourXOnePullPeerProof<1,
            FOUR_X_ONE_SNAPSHOT0_MASK>(staticArg)));
    }
    return ReduceFourXOnePullFixedTree(ctx, staticArg, 0);
}
struct TwoXEightMultiRootTaskVariables {
    ccu::Variable inputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable dataOffsetBytes;
    ccu::Variable elementCount;
    ccu::Variable tileOffsetBytes;
    ccu::Variable tileElementCount;
    ccu::Variable generation;
    ccu::Variable flags;
};
struct TwoXEightMultiRootContext {
    TwoXEightMultiRootTaskVariables taskArgs;
    ccu::Variable shardBytes;
    ccu::Event event;
};
template <typename Context>
CcuResult LoadTwoXEightMultiRootTaskArgs(Context &ctx)
{
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.inputAddr, static_cast<uint32_t>(TaskArgIndex::INPUT_ADDR)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.inputToken, static_cast<uint32_t>(TaskArgIndex::INPUT_TOKEN)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.outputAddr, static_cast<uint32_t>(TaskArgIndex::OUTPUT_ADDR)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.outputToken, static_cast<uint32_t>(TaskArgIndex::OUTPUT_TOKEN)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.scratchAddr, static_cast<uint32_t>(TaskArgIndex::SCRATCH_ADDR)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.scratchToken, static_cast<uint32_t>(TaskArgIndex::SCRATCH_TOKEN)));
    PHASE0_CCU_CHK(ccu::LoadArg(
        ctx.taskArgs.dataOffsetBytes, static_cast<uint32_t>(TaskArgIndex::DATA_OFFSET_BYTES)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.elementCount, static_cast<uint32_t>(TaskArgIndex::ELEMENT_COUNT)));
    PHASE0_CCU_CHK(ccu::LoadArg(
        ctx.taskArgs.tileOffsetBytes, static_cast<uint32_t>(TaskArgIndex::TILE_OFFSET_BYTES)));
    PHASE0_CCU_CHK(ccu::LoadArg(
        ctx.taskArgs.tileElementCount, static_cast<uint32_t>(TaskArgIndex::TILE_ELEMENT_COUNT)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.generation, static_cast<uint32_t>(TaskArgIndex::GENERATION)));
    PHASE0_CCU_CHK(ccu::LoadArg(ctx.taskArgs.flags, static_cast<uint32_t>(TaskArgIndex::FLAGS)));
    ccu::Variable twice;
    twice = ctx.taskArgs.tileElementCount + ctx.taskArgs.tileElementCount;
    ctx.shardBytes = twice + twice;
    return CCU_SUCCESS;
}
CcuResult ValidateTwoXEightMultiRootStaticArg(const TwoXEightMultiRootStaticArg &staticArg)
{
    const uint32_t expectedChannels = staticArg.layer == FabricLayer::L0 ? 7 : 8;
    if (staticArg.canonicalRank >= TWO_X_EIGHT_RANK_COUNT
        || (staticArg.layer != FabricLayer::L0 && staticArg.layer != FabricLayer::L1)
        || staticArg.channelCount != expectedChannels) {
        return CCU_E_PARA;
    }
    const uint32_t localGroup = staticArg.canonicalRank / 8;
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        const uint32_t peer = staticArg.peerCanonicalRanks[index];
        if (staticArg.channels[index] == 0 || peer >= TWO_X_EIGHT_RANK_COUNT
            || peer == staticArg.canonicalRank) {
            return CCU_E_PARA;
        }
        const bool sameGroup = peer / 8 == localGroup;
        if ((staticArg.layer == FabricLayer::L0) != sameGroup) {
            return CCU_E_PARA;
        }
        for (uint32_t prior = 0; prior < index; ++prior) {
            if (staticArg.peerCanonicalRanks[prior] == peer) {
                return CCU_E_PARA;
            }
        }
    }
    const uint32_t variableBase = staticArg.layer == FabricLayer::L0 ? 0 : 4;
    const ccu::Variable dieAnchor
        = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[0], variableBase);
    (void)dieAnchor;
    return CCU_SUCCESS;
}
ccu::LocalAddr TwoXEightLocalAddress(
    const ccu::Variable &base, const ccu::Variable &token, uint64_t offset)
{
    ccu::LocalAddr address;
    address.addr = base;
    ccu::Variable ccuOffset;
    ccuOffset = offset;
    address.addr += ccuOffset;
    address.token = token;
    return address;
}
ccu::RemoteAddr TwoXEightRemoteAddress(
    ccu::Variable base, ccu::Variable token, uint64_t offset)
{
    ccu::RemoteAddr address;
    address.addr = base;
    ccu::Variable ccuOffset;
    ccuOffset = offset;
    address.addr += ccuOffset;
    address.token = token;
    return address;
}
ccu::LocalAddr TwoXEightScratchLeaf(
    const TwoXEightMultiRootContext &ctx, uint32_t canonicalLeaf)
{
    return TwoXEightLocalAddress(ctx.taskArgs.scratchAddr, ctx.taskArgs.scratchToken,
        TwoXEightMultiRootScratchLeafOffset(canonicalLeaf));
}
ccu::LocalAddr TwoXEightCopyFreeLeaf(
    const TwoXEightMultiRootContext &ctx, uint32_t canonicalLeaf, uint32_t owner)
{
    if (canonicalLeaf == 0) {
        return TwoXEightLocalAddress(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken,
            TwoXEightMultiRootShardOffset(TwoXEightMultiRootIndex(owner)));
    }
    return TwoXEightScratchLeaf(ctx, canonicalLeaf);
}
ccu::LocalAddr TwoXEightCopyFreeGatherLeaf(
    const TwoXEightMultiRootContext &ctx, uint32_t canonicalLeaf, uint32_t owner)
{
    const uint32_t storageLeaf
        = ((owner & 1U) == 0 && canonicalLeaf == owner + 1) ? owner : canonicalLeaf;
    return TwoXEightCopyFreeLeaf(ctx, storageLeaf, owner);
}
template <uint32_t CompletionNotify>
CcuResult TwoXEightPeerBarrier(const TwoXEightMultiRootStaticArg &staticArg)
{
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::NotifyRecord(staticArg.channels[index], CompletionNotify));
    }
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::NotifyWait(staticArg.channels[index], CompletionNotify));
    }
    return CCU_SUCCESS;
}
template <uint32_t AddressReady, uint32_t VariableBase>
CcuResult PublishTwoXEightAddresses(
    TwoXEightMultiRootContext &ctx, const TwoXEightMultiRootStaticArg &staticArg)
{
    constexpr uint16_t inputAddrMask = static_cast<uint16_t>(1U << (VariableBase + 0));
    constexpr uint16_t inputTokenMask = static_cast<uint16_t>(1U << (VariableBase + 1));
    constexpr uint16_t outputAddrMask = static_cast<uint16_t>(1U << (VariableBase + 2));
    constexpr uint16_t outputTokenMask = static_cast<uint16_t>(1U << (VariableBase + 3));
    constexpr uint16_t addressMask
        = inputAddrMask | inputTokenMask | outputAddrMask | outputTokenMask;
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(staticArg.channels[index], ctx.taskArgs.inputAddr,
            VariableBase + 0, AddressReady, inputAddrMask));
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(staticArg.channels[index], ctx.taskArgs.inputToken,
            VariableBase + 1, AddressReady, inputTokenMask));
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(staticArg.channels[index], ctx.taskArgs.outputAddr,
            VariableBase + 2, AddressReady, outputAddrMask));
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(staticArg.channels[index], ctx.taskArgs.outputToken,
            VariableBase + 3, AddressReady, outputTokenMask));
    }
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::NotifyWait(staticArg.channels[index], AddressReady, addressMask));
    }
    return CCU_SUCCESS;
}
CcuResult TwoXEightReducePair(TwoXEightMultiRootContext &ctx, uint32_t owner,
    uint32_t destinationLeaf, uint32_t sourceLeaf, uint16_t eventMask)
{
    return ccu::LocalReduce(TwoXEightCopyFreeLeaf(ctx, destinationLeaf, owner),
        TwoXEightCopyFreeLeaf(ctx, sourceLeaf, owner), ctx.shardBytes, HCCL_DATA_TYPE_FP32,
        HCCL_REDUCE_SUM, ctx.event, eventMask);
}
CcuResult TwoXEightCopyFreeFirstPair(TwoXEightMultiRootContext &ctx, uint32_t owner,
    uint32_t pairBase, uint16_t eventMask)
{
    if (owner == pairBase || owner == pairBase + 1) {
        return ccu::LocalReduce(TwoXEightCopyFreeLeaf(ctx, pairBase, owner),
            TwoXEightLocalAddress(ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken,
                TwoXEightMultiRootShardOffset(TwoXEightMultiRootIndex(owner))),
            ctx.shardBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, eventMask);
    }
    return TwoXEightReducePair(ctx, owner, pairBase, pairBase + 1, eventMask);
}
CcuResult FixedTreeTwoXEightHalfReduce(
    TwoXEightMultiRootContext &ctx, uint32_t owner, uint32_t groupBase)
{
    PHASE0_CCU_CHK(TwoXEightCopyFreeFirstPair(ctx, owner, groupBase + 0, 1U << 0));
    PHASE0_CCU_CHK(TwoXEightCopyFreeFirstPair(ctx, owner, groupBase + 2, 1U << 1));
    PHASE0_CCU_CHK(TwoXEightCopyFreeFirstPair(ctx, owner, groupBase + 4, 1U << 2));
    PHASE0_CCU_CHK(TwoXEightCopyFreeFirstPair(ctx, owner, groupBase + 6, 1U << 3));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 4) - 1));
    PHASE0_CCU_CHK(TwoXEightReducePair(ctx, owner, groupBase + 0, groupBase + 2, 1U << 0));
    PHASE0_CCU_CHK(TwoXEightReducePair(ctx, owner, groupBase + 4, groupBase + 6, 1U << 1));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 2) - 1));
    PHASE0_CCU_CHK(TwoXEightReducePair(ctx, owner, groupBase + 0, groupBase + 4, 1U));
    return ccu::EventWait(ctx.event, 1U);
}
template <uint32_t VariableBase>
CcuResult RunTwoXEightGather(
    TwoXEightMultiRootContext &ctx, const TwoXEightMultiRootStaticArg &staticArg)
{
    if (IsTwoXEightMultiRoot(staticArg.canonicalRank)) {
        const uint32_t rootIndex = TwoXEightMultiRootIndex(staticArg.canonicalRank);
        const uint64_t shardOffset = TwoXEightMultiRootShardOffset(rootIndex);
        uint16_t waitMask = 0;
        uint32_t eventIndex = 0;
        for (uint32_t index = 0; index < staticArg.channelCount; ++index, ++eventIndex) {
            const uint16_t eventMask = static_cast<uint16_t>(1U << eventIndex);
            waitMask = static_cast<uint16_t>(waitMask | eventMask);
            const ccu::Variable remoteInputAddr
                = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 0);
            const ccu::Variable remoteInputToken
                = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 1);
            PHASE0_CCU_CHK(ccu::Read(staticArg.channels[index],
                TwoXEightCopyFreeGatherLeaf(
                    ctx, staticArg.peerCanonicalRanks[index], staticArg.canonicalRank),
                TwoXEightRemoteAddress(remoteInputAddr, remoteInputToken, shardOffset),
                ctx.shardBytes, ctx.event, eventMask));
        }
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
        const uint32_t groupBase = TwoXEightLayerGroupBase(
            staticArg.canonicalRank, staticArg.layer == FabricLayer::L0);
        PHASE0_CCU_CHK(FixedTreeTwoXEightHalfReduce(ctx, staticArg.canonicalRank, groupBase));
    }
    return CCU_SUCCESS;
}
ccu::LocalAddr TwoXEightDualReplicaGatherLeaf(
    const TwoXEightMultiRootContext &ctx, uint32_t canonicalLeaf, uint32_t owner)
{
    if ((owner & 1U) == 0 && canonicalLeaf == owner + 1) {
        return TwoXEightScratchLeaf(ctx, owner);
    }
    return TwoXEightScratchLeaf(ctx, canonicalLeaf);
}
CcuResult TwoXEightDualReplicaReducePair(TwoXEightMultiRootContext &ctx,
    uint32_t destinationLeaf, uint32_t sourceLeaf, uint16_t eventMask)
{
    return ccu::LocalReduce(TwoXEightScratchLeaf(ctx, destinationLeaf),
        TwoXEightScratchLeaf(ctx, sourceLeaf), ctx.shardBytes, HCCL_DATA_TYPE_FP32,
        HCCL_REDUCE_SUM, ctx.event, eventMask);
}
CcuResult TwoXEightDualReplicaFirstPair(TwoXEightMultiRootContext &ctx, uint32_t owner,
    uint32_t pairBase, uint16_t eventMask)
{
    if (owner == pairBase || owner == pairBase + 1) {
        return ccu::LocalReduce(TwoXEightScratchLeaf(ctx, pairBase),
            TwoXEightLocalAddress(ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken,
                TwoXEightMultiRootShardOffset(TwoXEightMultiRootIndex(owner))),
            ctx.shardBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, eventMask);
    }
    return TwoXEightDualReplicaReducePair(ctx, pairBase, pairBase + 1, eventMask);
}
CcuResult FixedTreeTwoXEightDualReplicaHalfReduce(
    TwoXEightMultiRootContext &ctx, uint32_t owner, uint32_t groupBase)
{
    PHASE0_CCU_CHK(TwoXEightDualReplicaFirstPair(ctx, owner, groupBase + 0, 1U << 0));
    PHASE0_CCU_CHK(TwoXEightDualReplicaFirstPair(ctx, owner, groupBase + 2, 1U << 1));
    PHASE0_CCU_CHK(TwoXEightDualReplicaFirstPair(ctx, owner, groupBase + 4, 1U << 2));
    PHASE0_CCU_CHK(TwoXEightDualReplicaFirstPair(ctx, owner, groupBase + 6, 1U << 3));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 4) - 1));
    PHASE0_CCU_CHK(TwoXEightDualReplicaReducePair(
        ctx, groupBase + 0, groupBase + 2, 1U << 0));
    PHASE0_CCU_CHK(TwoXEightDualReplicaReducePair(
        ctx, groupBase + 4, groupBase + 6, 1U << 1));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 2) - 1));
    PHASE0_CCU_CHK(TwoXEightDualReplicaReducePair(
        ctx, groupBase + 0, groupBase + 4, 1U));
    return ccu::EventWait(ctx.event, 1U);
}
template <uint32_t VariableBase>
CcuResult RunTwoXEightDualReplicaGather(
    TwoXEightMultiRootContext &ctx, const TwoXEightMultiRootStaticArg &staticArg)
{
    if (!IsTwoXEightMultiRoot(staticArg.canonicalRank)) {
        return CCU_SUCCESS;
    }
    const uint32_t rootIndex = TwoXEightMultiRootIndex(staticArg.canonicalRank);
    const uint64_t shardOffset = TwoXEightMultiRootShardOffset(rootIndex);
    uint16_t waitMask = 0;
    uint32_t eventIndex = 0;
    for (uint32_t index = 0; index < staticArg.channelCount; ++index, ++eventIndex) {
        const uint16_t eventMask = static_cast<uint16_t>(1U << eventIndex);
        waitMask = static_cast<uint16_t>(waitMask | eventMask);
        const ccu::Variable remoteInputAddr
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 0);
        const ccu::Variable remoteInputToken
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 1);
        PHASE0_CCU_CHK(ccu::Read(staticArg.channels[index],
            TwoXEightDualReplicaGatherLeaf(
                ctx, staticArg.peerCanonicalRanks[index], staticArg.canonicalRank),
            TwoXEightRemoteAddress(remoteInputAddr, remoteInputToken, shardOffset),
            ctx.shardBytes, ctx.event, eventMask));
    }
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
    const uint32_t groupBase = TwoXEightLayerGroupBase(
        staticArg.canonicalRank, staticArg.layer == FabricLayer::L0);
    return FixedTreeTwoXEightDualReplicaHalfReduce(
        ctx, staticArg.canonicalRank, groupBase);
}
CcuResult FixedTreeTwoXEightReduce(TwoXEightMultiRootContext &ctx, uint32_t owner)
{
    PHASE0_CCU_CHK(TwoXEightReducePair(ctx, owner, 0, 8, 1U));
    return ccu::EventWait(ctx.event, 1U);
}
CcuResult RunTwoXEightJoin(
    TwoXEightMultiRootContext &ctx, const TwoXEightMultiRootStaticArg &staticArg)
{
    if (!IsTwoXEightMultiRoot(staticArg.canonicalRank)) {
        return CCU_SUCCESS;
    }
    return FixedTreeTwoXEightReduce(ctx, staticArg.canonicalRank);
}
template <uint32_t CompletionNotify, uint32_t VariableBase>
CcuResult RunTwoXEightBroadcast(
    TwoXEightMultiRootContext &ctx, const TwoXEightMultiRootStaticArg &staticArg)
{
    for (uint32_t rootIndex = 0; rootIndex < TWO_X_EIGHT_MULTI_ROOT_COUNT; ++rootIndex) {
        if (staticArg.canonicalRank == TWO_X_EIGHT_MULTI_ROOT_ROOTS[rootIndex]) {
            const uint64_t shardOffset = TwoXEightMultiRootShardOffset(rootIndex);
            const ccu::LocalAddr localOutput
                = TwoXEightLocalAddress(ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, shardOffset);
            uint16_t waitMask = 0;
            for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
                const uint16_t eventMask = static_cast<uint16_t>(1U << index);
                waitMask = static_cast<uint16_t>(waitMask | eventMask);
                const ccu::Variable remoteOutputAddr
                    = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 2);
                const ccu::Variable remoteOutputToken
                    = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 3);
                PHASE0_CCU_CHK(ccu::Write(staticArg.channels[index],
                    TwoXEightRemoteAddress(remoteOutputAddr, remoteOutputToken, shardOffset),
                    localOutput, ctx.shardBytes, ctx.event, eventMask));
            }
            PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
        }
    }
    return TwoXEightPeerBarrier<CompletionNotify>(staticArg);
}
template <uint32_t CompletionNotify, uint32_t VariableBase>
CcuResult RunTwoXEightDualReplicaJoinBroadcast(
    TwoXEightMultiRootContext &ctx, const TwoXEightMultiRootStaticArg &staticArg)
{
    for (uint32_t rootIndex = 0; rootIndex < TWO_X_EIGHT_MULTI_ROOT_COUNT; ++rootIndex) {
        if (staticArg.canonicalRank != TWO_X_EIGHT_MULTI_ROOT_ROOTS[rootIndex]) {
            continue;
        }
        const uint64_t shardOffset = TwoXEightMultiRootShardOffset(rootIndex);
        const ccu::LocalAddr output = TwoXEightLocalAddress(
            ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken, shardOffset);
        const ccu::LocalAddr joined = staticArg.layer == FabricLayer::L0
            ? output : TwoXEightScratchLeaf(ctx, 1);
        PHASE0_CCU_CHK(ccu::LocalCopy(
            joined, TwoXEightScratchLeaf(ctx, 0), ctx.shardBytes, ctx.event, 1U));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, 1U));
        PHASE0_CCU_CHK(ccu::LocalReduce(joined, TwoXEightScratchLeaf(ctx, 8),
            ctx.shardBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 1U));
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, 1U));
        uint16_t waitMask = 0;
        for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
            const uint16_t eventMask = static_cast<uint16_t>(1U << index);
            waitMask = static_cast<uint16_t>(waitMask | eventMask);
            const ccu::Variable remoteOutputAddr
                = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 2);
            const ccu::Variable remoteOutputToken
                = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 3);
            PHASE0_CCU_CHK(ccu::Write(staticArg.channels[index],
                TwoXEightRemoteAddress(remoteOutputAddr, remoteOutputToken, shardOffset),
                joined, ctx.shardBytes, ctx.event, eventMask));
        }
        PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
    }
    return TwoXEightPeerBarrier<CompletionNotify>(staticArg);
}
template <uint32_t AddressReady, uint32_t BroadcastCompletion, uint32_t VariableBase>
CcuResult DispatchTwoXEightMultiRootStage(
    TwoXEightMultiRootContext &ctx, const TwoXEightMultiRootStaticArg &staticArg)
{
    CCU_IF(ctx.taskArgs.flags == TWO_X_EIGHT_MULTI_ROOT_FLAG_GATHER)
    {
        PHASE0_CCU_CHK((PublishTwoXEightAddresses<AddressReady, VariableBase>(ctx, staticArg)));
        PHASE0_CCU_CHK((RunTwoXEightGather<VariableBase>(ctx, staticArg)));
    }
    CCU_IF(ctx.taskArgs.flags == TWO_X_EIGHT_MULTI_ROOT_FLAG_DUAL_REPLICA_GATHER)
    {
        PHASE0_CCU_CHK((PublishTwoXEightAddresses<AddressReady, VariableBase>(ctx, staticArg)));
        PHASE0_CCU_CHK((RunTwoXEightDualReplicaGather<VariableBase>(ctx, staticArg)));
    }
    CCU_IF(ctx.taskArgs.flags == TWO_X_EIGHT_MULTI_ROOT_FLAG_JOIN)
    {
        PHASE0_CCU_CHK(RunTwoXEightJoin(ctx, staticArg));
    }
    CCU_IF(ctx.taskArgs.flags == TWO_X_EIGHT_MULTI_ROOT_FLAG_BROADCAST)
    {
        PHASE0_CCU_CHK((RunTwoXEightBroadcast<BroadcastCompletion, VariableBase>(ctx, staticArg)));
    }
    CCU_IF(ctx.taskArgs.flags == TWO_X_EIGHT_MULTI_ROOT_FLAG_DUAL_REPLICA_JOIN_BROADCAST)
    {
        PHASE0_CCU_CHK((RunTwoXEightDualReplicaJoinBroadcast<BroadcastCompletion,
            VariableBase>(ctx, staticArg)));
    }
    return CCU_SUCCESS;
}
struct TwoXEightBulkContext {
    TwoXEightMultiRootTaskVariables taskArgs;
    ccu::Variable shardBytes;
    ccu::Variable ownerOffset;
    ccu::Variable ownerTransferBytes;
    ccu::Variable oneWaveLeafOffset;
    ccu::Event event;
};
const ccu::Variable &TwoXEightBulkOwnerOffset(
    TwoXEightBulkContext &ctx, uint32_t owner)
{
    ctx.ownerOffset = static_cast<uint64_t>(owner) * TWO_X_EIGHT_BULK_OWNER_SHARD_BYTES;
    CCU_IF(ctx.taskArgs.generation == TWO_X_EIGHT_BULK_LAYOUT_UNEVEN)
    {
        ctx.ownerOffset = TwoXEightBulkTailOwnerOffset(owner);
    }
    CCU_IF(ctx.taskArgs.generation == TWO_X_EIGHT_BULK_LAYOUT_UNEVEN_REMAINDER)
    {
        ctx.ownerOffset = TwoXEightBulkTailOwnerOffset(owner);
    }
    CCU_IF(ctx.taskArgs.generation == TWO_X_EIGHT_BULK_LAYOUT_ONE_WAVE_TAIL)
    {
        ctx.ownerOffset = TwoXEightOneWaveOwnerOffset(owner, true);
    }
    return ctx.ownerOffset;
}
const ccu::Variable &TwoXEightBulkOwnerTransferBytes(
    TwoXEightBulkContext &ctx, uint32_t owner)
{
    ctx.ownerTransferBytes = ctx.shardBytes;
    CCU_IF(ctx.taskArgs.generation == TWO_X_EIGHT_BULK_LAYOUT_UNEVEN_REMAINDER)
    {
        ctx.ownerTransferBytes = TwoXEightBulkTailTileBytes(owner, 1);
    }
    CCU_IF(ctx.taskArgs.generation == TWO_X_EIGHT_BULK_LAYOUT_ONE_WAVE_TAIL)
    {
        ctx.ownerTransferBytes = TwoXEightOneWaveOwnerBytes(owner, true);
    }
    return ctx.ownerTransferBytes;
}
ccu::LocalAddr TwoXEightBulkLocalData(
    TwoXEightBulkContext &ctx, const ccu::Variable &base, const ccu::Variable &token,
    uint32_t owner)
{
    ccu::LocalAddr address;
    address.addr = base;
    address.addr += ctx.taskArgs.dataOffsetBytes;
    address.addr += TwoXEightBulkOwnerOffset(ctx, owner);
    address.token = token;
    return address;
}
ccu::RemoteAddr TwoXEightBulkRemoteOutput(TwoXEightBulkContext &ctx,
    ccu::Variable base, ccu::Variable token, uint32_t owner)
{
    ccu::RemoteAddr address;
    address.addr = base;
    address.addr += ctx.taskArgs.dataOffsetBytes;
    address.addr += TwoXEightBulkOwnerOffset(ctx, owner);
    address.token = token;
    return address;
}
ccu::Variable TwoXEightBulkScratchLeafOffset(
    TwoXEightBulkContext &ctx, uint32_t canonicalLeaf)
{
    ccu::Variable offset;
    offset = canonicalLeaf * PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES;
    CCU_IF(ctx.taskArgs.generation == TWO_X_EIGHT_BULK_LAYOUT_UNEVEN)
    {
        offset = canonicalLeaf * PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_TILE_BYTES;
    }
    return offset;
}
ccu::LocalAddr TwoXEightBulkScratchLeaf(
    TwoXEightBulkContext &ctx, uint32_t canonicalLeaf)
{
    ccu::LocalAddr address;
    address.addr = ctx.taskArgs.scratchAddr;
    address.addr += TwoXEightBulkScratchLeafOffset(ctx, canonicalLeaf);
    address.token = ctx.taskArgs.scratchToken;
    return address;
}
ccu::RemoteAddr TwoXEightBulkRemoteScratch(TwoXEightBulkContext &ctx,
    ccu::Variable base, ccu::Variable token, uint32_t canonicalLeaf)
{
    ccu::RemoteAddr address;
    address.addr = base;
    address.addr += TwoXEightBulkScratchLeafOffset(ctx, canonicalLeaf);
    address.token = token;
    return address;
}
const ccu::Variable &TwoXEightBulkOneWaveLeafOffset(
    TwoXEightBulkContext &ctx, uint32_t canonicalLeaf, uint32_t owner)
{
    const uint32_t storageLeaf
        = ((owner & 1U) == 0 && canonicalLeaf == owner + 1) ? owner : canonicalLeaf;
    if (storageLeaf == 0) {
        ctx.oneWaveLeafOffset = TwoXEightOneWaveOwnerOffset(owner, false);
    } else if (storageLeaf >= TWO_X_EIGHT_REDUCE_GROUP_SIZE) {
        ctx.oneWaveLeafOffset
            = (owner < TWO_X_EIGHT_REDUCE_GROUP_SIZE
                      ? TwoXEightOneWaveOwnerOffset(owner, false)
                            + TwoXEightOneWaveOwnerBytes(owner, false)
                      : 0)
              + (storageLeaf - TWO_X_EIGHT_REDUCE_GROUP_SIZE)
                    * TwoXEightOneWaveOwnerBytes(owner, false);
    } else {
        ctx.oneWaveLeafOffset = TwoXEightOneWaveLeafOffset(storageLeaf, owner, false);
    }
    CCU_IF(ctx.taskArgs.generation == TWO_X_EIGHT_BULK_LAYOUT_ONE_WAVE_TAIL)
    {
        if (storageLeaf == 0) {
            ctx.oneWaveLeafOffset = TwoXEightOneWaveOwnerOffset(owner, true);
        } else if (storageLeaf >= TWO_X_EIGHT_REDUCE_GROUP_SIZE) {
            ctx.oneWaveLeafOffset
                = (owner < TWO_X_EIGHT_REDUCE_GROUP_SIZE
                          ? TwoXEightOneWaveOwnerOffset(owner, true)
                                + TwoXEightOneWaveOwnerBytes(owner, true)
                          : 0)
                  + (storageLeaf - TWO_X_EIGHT_REDUCE_GROUP_SIZE)
                        * TwoXEightOneWaveOwnerBytes(owner, true);
        } else {
            ctx.oneWaveLeafOffset = TwoXEightOneWaveLeafOffset(storageLeaf, owner, true);
        }
    }
    return ctx.oneWaveLeafOffset;
}
ccu::LocalAddr TwoXEightBulkOneWaveLocalLeaf(
    TwoXEightBulkContext &ctx, uint32_t canonicalLeaf, uint32_t owner)
{
    ccu::LocalAddr address;
    const uint32_t storageLeaf
        = ((owner & 1U) == 0 && canonicalLeaf == owner + 1) ? owner : canonicalLeaf;
    if (storageLeaf != 0 && storageLeaf < TWO_X_EIGHT_REDUCE_GROUP_SIZE) {
        address.addr = ctx.taskArgs.scratchAddr;
        address.token = ctx.taskArgs.scratchToken;
    } else {
        address.addr = ctx.taskArgs.outputAddr;
        address.token = ctx.taskArgs.outputToken;
    }
    address.addr += TwoXEightBulkOneWaveLeafOffset(ctx, canonicalLeaf, owner);
    return address;
}
ccu::RemoteAddr TwoXEightBulkOneWaveRemoteLeaf(TwoXEightBulkContext &ctx,
    ccu::Variable scratchAddr, ccu::Variable scratchToken,
    ccu::Variable outputAddr, ccu::Variable outputToken,
    uint32_t canonicalLeaf, uint32_t owner)
{
    ccu::RemoteAddr address;
    const uint32_t storageLeaf
        = ((owner & 1U) == 0 && canonicalLeaf == owner + 1) ? owner : canonicalLeaf;
    if (storageLeaf != 0 && storageLeaf < TWO_X_EIGHT_REDUCE_GROUP_SIZE) {
        address.addr = scratchAddr;
        address.token = scratchToken;
    } else {
        address.addr = outputAddr;
        address.token = outputToken;
    }
    address.addr += TwoXEightBulkOneWaveLeafOffset(ctx, canonicalLeaf, owner);
    return address;
}
template <uint32_t AddressReady, uint32_t VariableBase>
CcuResult PublishTwoXEightBulkAddresses(
    TwoXEightBulkContext &ctx, const TwoXEightBulkStaticArg &staticArg)
{
    constexpr uint16_t scratchAddrMask = static_cast<uint16_t>(1U << (VariableBase + 0));
    constexpr uint16_t scratchTokenMask = static_cast<uint16_t>(1U << (VariableBase + 1));
    constexpr uint16_t outputAddrMask = static_cast<uint16_t>(1U << (VariableBase + 2));
    constexpr uint16_t outputTokenMask = static_cast<uint16_t>(1U << (VariableBase + 3));
    constexpr uint16_t addressMask
        = scratchAddrMask | scratchTokenMask | outputAddrMask | outputTokenMask;
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(staticArg.channels[index], ctx.taskArgs.scratchAddr,
            VariableBase + 0, AddressReady, scratchAddrMask));
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(staticArg.channels[index], ctx.taskArgs.scratchToken,
            VariableBase + 1, AddressReady, scratchTokenMask));
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(staticArg.channels[index], ctx.taskArgs.outputAddr,
            VariableBase + 2, AddressReady, outputAddrMask));
        PHASE0_CCU_CHK(ccu::WriteVariableWithNotify(staticArg.channels[index], ctx.taskArgs.outputToken,
            VariableBase + 3, AddressReady, outputTokenMask));
    }
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::NotifyWait(staticArg.channels[index], AddressReady, addressMask));
    }
    return CCU_SUCCESS;
}
template <uint32_t PushCompletion, uint32_t VariableBase>
CcuResult RunTwoXEightBulkPush(
    TwoXEightBulkContext &ctx, const TwoXEightBulkStaticArg &staticArg)
{
    uint16_t waitMask = 0;
    uint32_t eventIndex = 0;
    if (staticArg.layer == FabricLayer::L0) {
        waitMask = 1U;
        PHASE0_CCU_CHK(ccu::LocalCopy(
            TwoXEightBulkScratchLeaf(ctx, staticArg.canonicalRank),
            TwoXEightBulkLocalData(ctx, ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken,
                staticArg.canonicalRank),
            ctx.shardBytes, ctx.event, 1U));
        eventIndex = 1;
    }
    for (uint32_t index = 0; index < staticArg.channelCount; ++index, ++eventIndex) {
        const uint16_t eventMask = static_cast<uint16_t>(1U << eventIndex);
        waitMask = static_cast<uint16_t>(waitMask | eventMask);
        const ccu::Variable remoteScratchAddr
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 0);
        const ccu::Variable remoteScratchToken
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 1);
        PHASE0_CCU_CHK(ccu::Write(staticArg.channels[index],
            TwoXEightBulkRemoteScratch(
                ctx, remoteScratchAddr, remoteScratchToken, staticArg.canonicalRank),
            TwoXEightBulkLocalData(ctx, ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken,
                staticArg.peerCanonicalRanks[index]),
            TwoXEightBulkOwnerTransferBytes(ctx, staticArg.peerCanonicalRanks[index]),
            ctx.event, eventMask));
    }
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
    return TwoXEightPeerBarrier<PushCompletion>(staticArg);
}
template <uint32_t PushCompletion, uint32_t VariableBase>
CcuResult RunTwoXEightBulkOneWavePush(
    TwoXEightBulkContext &ctx, const TwoXEightBulkStaticArg &staticArg)
{
    uint16_t waitMask = 0;
    uint32_t eventIndex = 0;
    for (uint32_t index = 0; index < staticArg.channelCount; ++index, ++eventIndex) {
        const uint16_t eventMask = static_cast<uint16_t>(1U << eventIndex);
        waitMask = static_cast<uint16_t>(waitMask | eventMask);
        const ccu::Variable remoteScratchAddr
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 0);
        const ccu::Variable remoteScratchToken
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 1);
        const ccu::Variable remoteOutputAddr
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 2);
        const ccu::Variable remoteOutputToken
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 3);
        PHASE0_CCU_CHK(ccu::Write(staticArg.channels[index],
            TwoXEightBulkOneWaveRemoteLeaf(ctx,
                remoteScratchAddr, remoteScratchToken, remoteOutputAddr, remoteOutputToken,
                staticArg.canonicalRank, staticArg.peerCanonicalRanks[index]),
            TwoXEightBulkLocalData(ctx, ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken,
                staticArg.peerCanonicalRanks[index]),
            TwoXEightBulkOwnerTransferBytes(ctx, staticArg.peerCanonicalRanks[index]),
            ctx.event, eventMask));
    }
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
    return TwoXEightPeerBarrier<PushCompletion>(staticArg);
}
CcuResult TwoXEightBulkReducePair(TwoXEightBulkContext &ctx,
    uint32_t destinationLeaf, uint32_t sourceLeaf, uint16_t eventMask)
{
    return ccu::LocalReduce(TwoXEightBulkScratchLeaf(ctx, destinationLeaf),
        TwoXEightBulkScratchLeaf(ctx, sourceLeaf), ctx.shardBytes, HCCL_DATA_TYPE_FP32,
        HCCL_REDUCE_SUM, ctx.event, eventMask);
}
CcuResult FixedTreeTwoXEightBulkReduce(TwoXEightBulkContext &ctx)
{
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 0, 8, 1U << 0));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 1, 9, 1U << 1));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 2, 10, 1U << 2));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 3, 11, 1U << 3));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 4, 12, 1U << 4));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 5, 13, 1U << 5));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 6, 14, 1U << 6));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 7, 15, 1U << 7));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 8) - 1));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 0, 1, 1U << 0));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 2, 3, 1U << 1));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 4, 5, 1U << 2));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 6, 7, 1U << 3));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 4) - 1));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 0, 2, 1U << 0));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 4, 6, 1U << 1));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 2) - 1));
    PHASE0_CCU_CHK(TwoXEightBulkReducePair(ctx, 0, 4, 1U));
    return ccu::EventWait(ctx.event, 1U);
}
CcuResult RunTwoXEightBulkJoin(
    TwoXEightBulkContext &ctx, const TwoXEightBulkStaticArg &staticArg)
{
    PHASE0_CCU_CHK(FixedTreeTwoXEightBulkReduce(ctx));
    PHASE0_CCU_CHK(ccu::LocalCopy(
        TwoXEightBulkLocalData(ctx, ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken,
            staticArg.canonicalRank),
        TwoXEightBulkScratchLeaf(ctx, 0), ctx.shardBytes, ctx.event, 1U));
    return ccu::EventWait(ctx.event, 1U);
}
CcuResult TwoXEightBulkOneWaveReducePair(TwoXEightBulkContext &ctx, uint32_t owner,
    uint32_t destinationLeaf, uint32_t sourceLeaf, uint16_t eventMask)
{
    return ccu::LocalReduce(TwoXEightBulkOneWaveLocalLeaf(ctx, destinationLeaf, owner),
        TwoXEightBulkOneWaveLocalLeaf(ctx, sourceLeaf, owner), ctx.shardBytes,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, eventMask);
}
CcuResult TwoXEightBulkOneWaveCopyFreeFirstPair(TwoXEightBulkContext &ctx, uint32_t owner,
    uint32_t pairBase, uint16_t eventMask)
{
    if (owner == pairBase || owner == pairBase + 1) {
        return ccu::LocalReduce(TwoXEightBulkOneWaveLocalLeaf(ctx, pairBase, owner),
            TwoXEightBulkLocalData(ctx, ctx.taskArgs.inputAddr, ctx.taskArgs.inputToken, owner),
            ctx.shardBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, eventMask);
    }
    return TwoXEightBulkOneWaveReducePair(ctx, owner, pairBase, pairBase + 1, eventMask);
}
CcuResult FixedTreeTwoXEightBulkOneWaveReduce(
    TwoXEightBulkContext &ctx, const TwoXEightBulkStaticArg &staticArg)
{
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveCopyFreeFirstPair(
        ctx, staticArg.canonicalRank, 0, 1U << 0));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveCopyFreeFirstPair(
        ctx, staticArg.canonicalRank, 2, 1U << 1));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveCopyFreeFirstPair(
        ctx, staticArg.canonicalRank, 4, 1U << 2));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveCopyFreeFirstPair(
        ctx, staticArg.canonicalRank, 6, 1U << 3));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveCopyFreeFirstPair(
        ctx, staticArg.canonicalRank, 8, 1U << 4));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveCopyFreeFirstPair(
        ctx, staticArg.canonicalRank, 10, 1U << 5));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveCopyFreeFirstPair(
        ctx, staticArg.canonicalRank, 12, 1U << 6));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveCopyFreeFirstPair(
        ctx, staticArg.canonicalRank, 14, 1U << 7));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 8) - 1));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveReducePair(ctx, staticArg.canonicalRank, 0, 2, 1U << 0));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveReducePair(ctx, staticArg.canonicalRank, 4, 6, 1U << 1));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveReducePair(ctx, staticArg.canonicalRank, 8, 10, 1U << 2));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveReducePair(ctx, staticArg.canonicalRank, 12, 14, 1U << 3));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 4) - 1));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveReducePair(ctx, staticArg.canonicalRank, 0, 4, 1U << 0));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveReducePair(ctx, staticArg.canonicalRank, 8, 12, 1U << 1));
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, (1U << 2) - 1));
    PHASE0_CCU_CHK(TwoXEightBulkOneWaveReducePair(ctx, staticArg.canonicalRank, 0, 8, 1U));
    return ccu::EventWait(ctx.event, 1U);
}
CcuResult RunTwoXEightBulkOneWaveJoin(
    TwoXEightBulkContext &ctx, const TwoXEightBulkStaticArg &staticArg)
{
    return FixedTreeTwoXEightBulkOneWaveReduce(ctx, staticArg);
}
template <uint32_t ReadyNotify>
CcuResult TwoXEightBulkOneWavePeerReady(const TwoXEightBulkStaticArg &staticArg)
{
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::NotifyRecord(staticArg.channels[index], ReadyNotify));
    }
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        PHASE0_CCU_CHK(ccu::NotifyWait(staticArg.channels[index], ReadyNotify));
    }
    return CCU_SUCCESS;
}
template <uint32_t ReadyNotify, uint32_t AllGatherCompletion, uint32_t VariableBase>
CcuResult RunTwoXEightBulkOneWaveAllGather(
    TwoXEightBulkContext &ctx, const TwoXEightBulkStaticArg &staticArg)
{
    PHASE0_CCU_CHK((TwoXEightBulkOneWavePeerReady<ReadyNotify>(staticArg)));
    const ccu::LocalAddr root
        = TwoXEightBulkOneWaveLocalLeaf(ctx, 0, staticArg.canonicalRank);
    uint16_t waitMask = 0;
    uint32_t eventIndex = 0;
    for (uint32_t index = 0; index < staticArg.channelCount; ++index, ++eventIndex) {
        const uint16_t eventMask = static_cast<uint16_t>(1U << eventIndex);
        waitMask = static_cast<uint16_t>(waitMask | eventMask);
        const ccu::Variable remoteOutputAddr
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 2);
        const ccu::Variable remoteOutputToken
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 3);
        PHASE0_CCU_CHK(ccu::Write(staticArg.channels[index],
            TwoXEightBulkRemoteOutput(
                ctx, remoteOutputAddr, remoteOutputToken, staticArg.canonicalRank),
            root, ctx.shardBytes, ctx.event, eventMask));
    }
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
    return TwoXEightPeerBarrier<AllGatherCompletion>(staticArg);
}
template <uint32_t AllGatherCompletion, uint32_t VariableBase>
CcuResult RunTwoXEightBulkAllGather(
    TwoXEightBulkContext &ctx, const TwoXEightBulkStaticArg &staticArg)
{
    const ccu::LocalAddr localOutput
        = TwoXEightBulkLocalData(ctx, ctx.taskArgs.outputAddr, ctx.taskArgs.outputToken,
            staticArg.canonicalRank);
    uint16_t waitMask = 0;
    for (uint32_t index = 0; index < staticArg.channelCount; ++index) {
        const uint16_t eventMask = static_cast<uint16_t>(1U << index);
        waitMask = static_cast<uint16_t>(waitMask | eventMask);
        const ccu::Variable remoteOutputAddr
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 2);
        const ccu::Variable remoteOutputToken
            = ccu::GetResByChannel<ccu::Variable>(staticArg.channels[index], VariableBase + 3);
        PHASE0_CCU_CHK(ccu::Write(staticArg.channels[index],
            TwoXEightBulkRemoteOutput(
                ctx, remoteOutputAddr, remoteOutputToken, staticArg.canonicalRank),
            localOutput, ctx.shardBytes, ctx.event, eventMask));
    }
    PHASE0_CCU_CHK(ccu::EventWait(ctx.event, waitMask));
    return TwoXEightPeerBarrier<AllGatherCompletion>(staticArg);
}
template <uint32_t AddressReady, uint32_t PushCompletion, uint32_t JoinReady,
    uint32_t AllGatherCompletion, uint32_t VariableBase>
CcuResult DispatchTwoXEightBulkStage(
    TwoXEightBulkContext &ctx, const TwoXEightBulkStaticArg &staticArg)
{
    CCU_IF(ctx.taskArgs.flags == TWO_X_EIGHT_BULK_FLAG_PUSH)
    {
        PHASE0_CCU_CHK((PublishTwoXEightBulkAddresses<AddressReady, VariableBase>(ctx, staticArg)));
        PHASE0_CCU_CHK((RunTwoXEightBulkPush<PushCompletion, VariableBase>(ctx, staticArg)));
    }
    CCU_IF(ctx.taskArgs.flags == TWO_X_EIGHT_BULK_FLAG_JOIN)
    {
        PHASE0_CCU_CHK(RunTwoXEightBulkJoin(ctx, staticArg));
    }
    CCU_IF(ctx.taskArgs.flags == TWO_X_EIGHT_BULK_FLAG_ALL_GATHER)
    {
        PHASE0_CCU_CHK((RunTwoXEightBulkAllGather<AllGatherCompletion, VariableBase>(ctx, staticArg)));
    }
    CCU_IF(ctx.taskArgs.flags == TWO_X_EIGHT_BULK_FLAG_ONE_WAVE_PUSH)
    {
        PHASE0_CCU_CHK((PublishTwoXEightBulkAddresses<AddressReady, VariableBase>(ctx, staticArg)));
        PHASE0_CCU_CHK((RunTwoXEightBulkOneWavePush<PushCompletion, VariableBase>(ctx, staticArg)));
    }
    CCU_IF(ctx.taskArgs.flags == TWO_X_EIGHT_BULK_FLAG_ONE_WAVE_JOIN)
    {
        PHASE0_CCU_CHK(RunTwoXEightBulkOneWaveJoin(ctx, staticArg));
    }
    CCU_IF(ctx.taskArgs.flags == TWO_X_EIGHT_BULK_FLAG_ONE_WAVE_ALL_GATHER)
    {
        PHASE0_CCU_CHK((RunTwoXEightBulkOneWaveAllGather<JoinReady,
            AllGatherCompletion, VariableBase>(ctx, staticArg)));
    }
    return CCU_SUCCESS;
}
} // namespace
CcuResult Direct12DieKernel(CcuKernelArg arg)
{
auto *staticArg = static_cast<Direct12DieStaticArg *>(arg);
if (staticArg == nullptr
    || (staticArg->mode != DIRECT12_MODE_STANDARD && staticArg->mode != DIRECT12_MODE_FRONTIER_512)) {
    return CCU_E_PARA;
}
if ((staticArg->l0ChannelCount == 0) == (staticArg->l1ChannelCount == 0)) {
    return CCU_E_PARA;
}
if (staticArg->l0ChannelCount != 0) {
    PHASE0_CCU_CHK(ValidateDirect12LayerStaticArg(
        *staticArg, staticArg->l0Channels, staticArg->l0PeerCanonicalRanks,
        staticArg->l0ChannelCount, FabricLayer::L0));
}
if (staticArg->l1ChannelCount != 0) {
    PHASE0_CCU_CHK(ValidateDirect12LayerStaticArg(
        *staticArg, staticArg->l1Channels, staticArg->l1PeerCanonicalRanks,
        staticArg->l1ChannelCount, FabricLayer::L1));
}
PHASE0_CCU_CHK(ValidateDirect12Runtime(0, 0, sizeof(float), sizeof(float), staticArg->scratchStrideBytes,
    staticArg->l0ChannelCount + staticArg->l1ChannelCount));
Direct12Context ctx;
ctx.scratchStrideBytes = staticArg->scratchStrideBytes;
PHASE0_CCU_CHK(LoadDirect12TaskArgs(ctx));
return DispatchDirect12Stage(ctx, *staticArg);
}
CcuResult Direct12PrefixScalarDieKernel(CcuKernelArg arg)
{
auto *staticArg = static_cast<Direct12DieStaticArg *>(arg);
if (staticArg == nullptr || staticArg->mode != DIRECT12_MODE_PREFIX_SCALAR_TAIL) {
    return CCU_E_PARA;
}
if ((staticArg->l0ChannelCount == 0) == (staticArg->l1ChannelCount == 0)) {
    return CCU_E_PARA;
}
if (staticArg->l0ChannelCount != 0) {
    PHASE0_CCU_CHK(ValidateDirect12LayerStaticArg(
        *staticArg, staticArg->l0Channels, staticArg->l0PeerCanonicalRanks,
        staticArg->l0ChannelCount, FabricLayer::L0));
}
if (staticArg->l1ChannelCount != 0) {
    PHASE0_CCU_CHK(ValidateDirect12LayerStaticArg(
        *staticArg, staticArg->l1Channels, staticArg->l1PeerCanonicalRanks,
        staticArg->l1ChannelCount, FabricLayer::L1));
}
PHASE0_CCU_CHK(ValidateDirect12Runtime(0, 0, sizeof(float), sizeof(float), staticArg->scratchStrideBytes,
    staticArg->l0ChannelCount + staticArg->l1ChannelCount));
Direct12Context ctx;
ctx.scratchStrideBytes = staticArg->scratchStrideBytes;
PHASE0_CCU_CHK(LoadDirect12TaskArgs(ctx));
return DispatchDirect12PrefixScalarStage(ctx, *staticArg);
}
CcuResult FullExchangeDieKernel(CcuKernelArg arg)
{
auto *staticArg = static_cast<FullExchangeDieStaticArg *>(arg);
if (staticArg == nullptr) {
    return CCU_E_PARA;
}
PHASE0_CCU_CHK(ValidateFullExchangeStaticArg(*staticArg));
FullExchangeContext ctx;
PHASE0_CCU_CHK(LoadFullExchangeTaskArgs(ctx));
return DispatchFullExchangeStage(ctx, *staticArg);
}
CcuResult FourXOneRingKernel(CcuKernelArg arg)
{
auto *staticArg = static_cast<FourXOneRingStaticArg *>(arg);
if (staticArg == nullptr) {
    return CCU_E_PARA;
}
PHASE0_CCU_CHK(ValidateFourXOneRingStaticArg(*staticArg));
FourXOneContext ctx;
PHASE0_CCU_CHK(LoadFourXOneTaskArgs(ctx));
ccu::Variable ringNotifyBranch;
ringNotifyBranch = ctx.taskArgs.generation;
CCU_IF(ctx.taskArgs.flags == FOUR_X_ONE_RING_FLAG_EXECUTE)
{
    CCU_IF(ringNotifyBranch == 0)
    {
        PHASE0_CCU_CHK((RunFourXOneRingStage<0, 0>(ctx, *staticArg)));
    }
    CCU_IF(ringNotifyBranch == 2)
    {
        PHASE0_CCU_CHK((RunFourXOneRingStage<1, 4>(ctx, *staticArg)));
    }
}
CCU_IF(ctx.taskArgs.flags == FOUR_X_ONE_RING_FLAG_EXECUTE_UNEVEN_TAIL)
{
    ccu::Variable unevenBytes;
    unevenBytes = ctx.bytes;
    ccu::Variable scalarBytes;
    scalarBytes = sizeof(float);
    unevenBytes += scalarBytes;
    CCU_IF(ringNotifyBranch == 0)
    {
        PHASE0_CCU_CHK((RunFourXOneTailRingStage<0, 0>(ctx, *staticArg, unevenBytes)));
    }
    CCU_IF(ringNotifyBranch == 2)
    {
        PHASE0_CCU_CHK((RunFourXOneTailRingStage<1, 4>(ctx, *staticArg, unevenBytes)));
    }
}
return CCU_SUCCESS;
}
CcuResult FourXOnePullFullExchangeKernel(CcuKernelArg arg)
{
auto *staticArg = static_cast<FourXOnePullStaticArg *>(arg);
if (staticArg == nullptr) {
    return CCU_E_PARA;
}
PHASE0_CCU_CHK(ValidateFourXOnePullStaticArg(*staticArg, true));
FourXOnePullContext ctx;
PHASE0_CCU_CHK(LoadFourXOneTaskArgs(ctx));
ctx.ownerOffsetBytes = 0;
ctx.tileOffsetBytes[0] = 0;
ctx.tileOffsetBytes[1] = 0;
return RunFourXOnePullFullExchange(ctx, *staticArg);
}
CcuResult TwoXEightMultiRootKernel(CcuKernelArg arg)
{
auto *staticArg = static_cast<TwoXEightMultiRootStaticArg *>(arg);
if (staticArg == nullptr) {
    return CCU_E_PARA;
}
PHASE0_CCU_CHK(ValidateTwoXEightMultiRootStaticArg(*staticArg));
TwoXEightMultiRootContext ctx;
PHASE0_CCU_CHK(LoadTwoXEightMultiRootTaskArgs(ctx));
if (staticArg->layer == FabricLayer::L0) {
    return DispatchTwoXEightMultiRootStage<0, 3, 0>(ctx, *staticArg);
}
return DispatchTwoXEightMultiRootStage<4, 7, 4>(ctx, *staticArg);
}
CcuResult TwoXEightBulkKernel(CcuKernelArg arg)
{
auto *staticArg = static_cast<TwoXEightBulkStaticArg *>(arg);
if (staticArg == nullptr) {
    return CCU_E_PARA;
}
PHASE0_CCU_CHK(ValidateTwoXEightMultiRootStaticArg(*staticArg));
TwoXEightBulkContext ctx;
PHASE0_CCU_CHK(LoadTwoXEightMultiRootTaskArgs(ctx));
if (staticArg->layer == FabricLayer::L0) {
    return DispatchTwoXEightBulkStage<0, 1, 2, 3, 0>(ctx, *staticArg);
}
return DispatchTwoXEightBulkStage<4, 5, 6, 7, 4>(ctx, *staticArg);
}
#undef PHASE0_CCU_CHK
} // namespace ops_hccl
