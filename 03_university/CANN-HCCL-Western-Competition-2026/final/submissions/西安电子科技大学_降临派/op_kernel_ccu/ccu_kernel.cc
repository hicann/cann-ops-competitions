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

#include <ccu/ccu_control_flow_macro.h>
#include <ccu/ccu_primitives.hpp>

#include <vector>

#include "custom.h"

namespace ops_hccl {
namespace ccu = AscendC::ccu;

namespace {
    // Notify 0 用于交换任务地址，Notify 1 用于可复用的数据阶段握手。
    constexpr uint32_t INPUT_VAR = 0;
    constexpr uint32_t INPUT_TOKEN_VAR = 1;
    constexpr uint32_t OUTPUT_VAR = 2;
    constexpr uint32_t OUTPUT_TOKEN_VAR = 3;
    constexpr uint32_t ADDRESS_NOTIFY = 0;
    constexpr uint32_t DATA_NOTIFY = 1;
    constexpr uint16_t POST_MASK = 1U << 0;
    constexpr uint16_t ACK_MASK = 1U << 1;
    constexpr uint16_t INPUT_MASK = 1U << 0;
    constexpr uint16_t INPUT_TOKEN_MASK = 1U << 1;
    constexpr uint16_t OUTPUT_MASK = 1U << 2;
    constexpr uint16_t OUTPUT_TOKEN_MASK = 1U << 3;
    constexpr uint32_t DIE_NUM = 2;

#define RETURN_IF_CCU_ERROR(expr) \
    do { \
        CcuResult result = (expr); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

    ccu::LocalAddr LocalAddress(const ccu::Variable &base, const ccu::Variable &token, const ccu::Variable &offset)
    {
        ccu::LocalAddr address;
        address.addr = base;
        address.addr += offset;
        address.token = token;
        return address;
    }

    ccu::RemoteAddr RemoteAddress(const ccu::Variable &base, const ccu::Variable &token, const ccu::Variable &offset)
    {
        ccu::RemoteAddr address;
        address.addr = base;
        address.addr += offset;
        address.token = token;
        return address;
    }

    ccu::Variable Scale(const ccu::Variable &value, uint32_t multiplier)
    {
        // 只保留一个临时 XN；二进制分解会耗尽 Packed 八通道 Kernel 的寄存器预算。
        ccu::Variable result;
        result = 0;
        for (uint32_t index = 0; index < multiplier; ++index) {
            result += value;
        }
        return result;
    }

    struct SmallContext {
        const SmallKernelArg *arg = nullptr;
        ccu::Variable input;
        ccu::Variable inputToken;
        ccu::Variable output;
        ccu::Variable outputToken;
        ccu::Variable scratchToken;
        ccu::Event event;
    };

    CcuResult LoadSmallContext(SmallContext &ctx, CcuKernelArg arg)
    {
        ctx.arg = static_cast<const SmallKernelArg *>(arg);
        if (ctx.arg == nullptr || ctx.arg->rankSize < 2U || ctx.arg->rankSize > MAX_RANK_SIZE
            || ctx.arg->channelCount + 1U != ctx.arg->rankSize) {
            return CCU_E_PARA;
        }
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.input, 0U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.inputToken, 1U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.output, 2U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputToken, 3U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.scratchToken, 4U));
        return CCU_SUCCESS;
    }

    ccu::LocalAddr SmallScratch(const SmallContext &ctx, uint32_t slot)
    {
        ccu::LocalAddr address;
        address.addr = ctx.arg->localScratch[slot];
        address.token = ctx.scratchToken;
        return address;
    }

    CcuResult ShareOwnerAddresses(SmallContext &ctx)
    {
        constexpr uint16_t addressMask = INPUT_MASK | INPUT_TOKEN_MASK | OUTPUT_MASK | OUTPUT_TOKEN_MASK;
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            const ChannelHandle channel = ctx.arg->channels[index];
            RETURN_IF_CCU_ERROR(
                ccu::WriteVariableWithNotify(channel, ctx.input, INPUT_VAR, ADDRESS_NOTIFY, INPUT_MASK));
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
                channel, ctx.inputToken, INPUT_TOKEN_VAR, ADDRESS_NOTIFY, INPUT_TOKEN_MASK));
            RETURN_IF_CCU_ERROR(
                ccu::WriteVariableWithNotify(channel, ctx.output, OUTPUT_VAR, ADDRESS_NOTIFY, OUTPUT_MASK));
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
                channel, ctx.outputToken, OUTPUT_TOKEN_VAR, ADDRESS_NOTIFY, OUTPUT_TOKEN_MASK));
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[index], ADDRESS_NOTIFY, addressMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult PullOwnerInputs(SmallContext &ctx)
    {
        uint16_t readMask = 0;
        ccu::Variable offset;
        offset = ctx.arg->ownerOffset;
        ccu::Variable bytes;
        bytes = ctx.arg->ownerBytes;
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            const uint32_t source = ctx.arg->peers[index];
            const uint16_t bit = static_cast<uint16_t>(1U << source);
            readMask = static_cast<uint16_t>(readMask | bit);
            const ChannelHandle channel = ctx.arg->channels[index];
            const ccu::Variable remoteInput = ccu::GetResByChannel<ccu::Variable>(channel, INPUT_VAR);
            const ccu::Variable remoteToken = ccu::GetResByChannel<ccu::Variable>(channel, INPUT_TOKEN_VAR);
            RETURN_IF_CCU_ERROR(ccu::Read(channel, SmallScratch(ctx, source),
                RemoteAddress(remoteInput, remoteToken, offset), bytes, ctx.event, bit));
        }
        return ccu::EventWait(ctx.event, readMask);
    }

    CcuResult ReduceOwnerTree(SmallContext &ctx, ccu::LocalAddr &root)
    {
        ccu::Variable offset;
        offset = ctx.arg->ownerOffset;
        ccu::Variable bytes;
        bytes = ctx.arg->ownerBytes;
        ccu::LocalAddr roots[MAX_RANK_SIZE];
        for (uint32_t source = 0; source < ctx.arg->rankSize; ++source) {
            roots[source] = source == ctx.arg->myRank ? LocalAddress(ctx.input, ctx.inputToken, offset)
                                                      : SmallScratch(ctx, source);
        }
        for (uint32_t stride = 1U; stride < ctx.arg->rankSize; stride *= 2U) {
            uint16_t mask = 0;
            uint32_t operation = 0;
            for (uint32_t base = 0; base + stride < ctx.arg->rankSize; base += stride * 2U) {
                const uint16_t bit = static_cast<uint16_t>(1U << operation++);
                mask = static_cast<uint16_t>(mask | bit);
                const uint32_t right = base + stride;
                // 用户输入区不能作为原地规约的写入目标。
                const bool preserveInput = stride == 1U && base == ctx.arg->myRank;
                const ccu::LocalAddr destination = preserveInput ? roots[right] : roots[base];
                const ccu::LocalAddr source = preserveInput ? roots[base] : roots[right];
                RETURN_IF_CCU_ERROR(
                    ccu::LocalReduce(destination, source, bytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, bit));
                if (preserveInput) {
                    roots[base] = destination;
                }
            }
            RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, mask));
        }
        root = roots[0];
        return CCU_SUCCESS;
    }

    CcuResult PublishOwner(SmallContext &ctx, const ccu::LocalAddr &root)
    {
        ccu::Variable offset;
        offset = ctx.arg->ownerOffset;
        ccu::Variable bytes;
        bytes = ctx.arg->ownerBytes;
        uint16_t mask = 1U;
        RETURN_IF_CCU_ERROR(
            ccu::LocalCopy(LocalAddress(ctx.output, ctx.outputToken, offset), root, bytes, ctx.event, 1U));
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            const uint16_t bit = static_cast<uint16_t>(1U << (index + 1U));
            mask = static_cast<uint16_t>(mask | bit);
            const ChannelHandle channel = ctx.arg->channels[index];
            const ccu::Variable remoteOutput = ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_VAR);
            const ccu::Variable remoteToken = ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_VAR);
            RETURN_IF_CCU_ERROR(
                ccu::Write(channel, RemoteAddress(remoteOutput, remoteToken, offset), root, bytes, ctx.event, bit));
        }
        return ccu::EventWait(ctx.event, mask);
    }

    CcuResult RunOwnerPullSmall(SmallContext &ctx)
    {
        // 每个 Rank 在本地规约一个归属分片，再将该分片发布给所有 Peer。
        RETURN_IF_CCU_ERROR(ShareOwnerAddresses(ctx));
        RETURN_IF_CCU_ERROR(PullOwnerInputs(ctx));
        ccu::LocalAddr root;
        RETURN_IF_CCU_ERROR(ReduceOwnerTree(ctx, root));
        return PublishOwner(ctx, root);
    }

    int32_t SmallChannelForPeer(const SmallKernelArg &arg, uint32_t peer)
    {
        for (uint32_t index = 0; index < arg.channelCount; ++index) {
            if (arg.peers[index] == peer) {
                return static_cast<int32_t>(index);
            }
        }
        return -1;
    }

    CcuResult Run4x1OneShot(SmallContext &ctx)
    {
        if (ctx.arg->rankSize != fused_4x1::RANK_SIZE) {
            return CCU_E_PARA;
        }
        ccu::Variable bytes;
        bytes = ctx.arg->smallBytes;
        ccu::Variable zero;
        zero = 0;
        const ccu::LocalAddr input = LocalAddress(ctx.input, ctx.inputToken, zero);
        const ccu::LocalAddr output = LocalAddress(ctx.output, ctx.outputToken, zero);
        // 本地规约树使用 Peer 暂存区前，先发布不可变的输入副本。
        uint32_t channels[fused_4x1::RANK_SIZE - 1U];
        for (uint32_t step = 0; step < fused_4x1::RANK_SIZE - 1U; ++step) {
            const int32_t channel = SmallChannelForPeer(*ctx.arg, ctx.arg->myRank ^ (step + 1U));
            if (channel < 0) {
                return CCU_E_INTERNAL;
            }
            channels[step] = static_cast<uint32_t>(channel);
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channel], ctx.scratchToken, INPUT_VAR, ADDRESS_NOTIFY, INPUT_MASK));
        }
        RETURN_IF_CCU_ERROR(ccu::LocalCopy(output, input, bytes, ctx.event, 1U << 3));
        for (uint32_t step = 0; step < fused_4x1::RANK_SIZE - 1U; ++step) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[channels[step]], ADDRESS_NOTIFY, INPUT_MASK));
        }
        uint16_t writeMask = 0;
        for (uint32_t step = 0; step < fused_4x1::RANK_SIZE - 1U; ++step) {
            const uint16_t bit = static_cast<uint16_t>(1U << step);
            writeMask = static_cast<uint16_t>(writeMask | bit);
            ccu::RemoteAddr destination;
            destination.addr = ctx.arg->remoteScratch[channels[step]];
            destination.token = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channels[step]], INPUT_VAR);
            RETURN_IF_CCU_ERROR(
                ccu::Write(ctx.arg->channels[channels[step]], destination, input, bytes, ctx.event, bit));
        }
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, writeMask | (1U << 3)));
        for (uint32_t step = 0; step < fused_4x1::RANK_SIZE - 1U; ++step) {
            RETURN_IF_CCU_ERROR(ccu::NotifyRecord(ctx.arg->channels[channels[step]], ADDRESS_NOTIFY, OUTPUT_MASK));
        }
        for (uint32_t step = 0; step < fused_4x1::RANK_SIZE - 1U; ++step) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[channels[step]], ADDRESS_NOTIFY, OUTPUT_MASK));
        }
        const uint32_t pairPeer = ctx.arg->myRank ^ 1U;
        const uint32_t otherPair = (ctx.arg->myRank & ~1U) ^ 2U;
        RETURN_IF_CCU_ERROR(ccu::LocalReduce(
            output, SmallScratch(ctx, pairPeer), bytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 1U));
        RETURN_IF_CCU_ERROR(ccu::LocalReduce(SmallScratch(ctx, otherPair), SmallScratch(ctx, otherPair + 1U), bytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 2U));
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, 3U));
        RETURN_IF_CCU_ERROR(ccu::LocalReduce(
            output, SmallScratch(ctx, otherPair), bytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 1U));
        return ccu::EventWait(ctx.event, 1U);
    }

    struct LargeContext {
        const LargeKernelArg *arg = nullptr;
        ccu::Variable input;
        ccu::Variable inputToken;
        ccu::Variable output;
        ccu::Variable outputToken;
        ccu::Variable inplace;
        ccu::Variable normalBytes;
        ccu::Variable lastBytes;
        ccu::Variable phase;
        ccu::Variable tailRound;
        std::vector<ccu::Variable> offsets;
        std::vector<ccu::Variable> remoteOutput;
        std::vector<ccu::Variable> remoteOutputToken;
        ccu::Event event;
    };

    CcuResult LoadLargeContext(LargeContext &ctx, CcuKernelArg arg)
    {
        ctx.arg = static_cast<const LargeKernelArg *>(arg);
        if (ctx.arg == nullptr || ctx.arg->dieId >= DIE_NUM || ctx.arg->channelCount == 0U
            || ctx.arg->channelCount > MAX_RANK_SIZE || ctx.arg->rankSize < 2U) {
            return CCU_E_PARA;
        }
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.input, 0U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.inputToken, 1U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.output, 2U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputToken, 3U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.inplace, 4U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.normalBytes, 5U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.lastBytes, 6U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.phase, 7U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.tailRound, 8U));
        ctx.offsets.reserve(ctx.arg->rankSize);
        for (uint32_t rank = 0; rank < ctx.arg->rankSize; ++rank) {
            ccu::Variable offset;
            RETURN_IF_CCU_ERROR(ccu::LoadArg(offset, latin::OFFSET_ARG + rank));
            ctx.offsets.push_back(offset);
        }
        ctx.remoteOutput.reserve(ctx.arg->channelCount);
        ctx.remoteOutputToken.reserve(ctx.arg->channelCount);
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            ChannelHandle channel = ctx.arg->channels[index];
            ctx.remoteOutput.push_back(ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_VAR));
            ctx.remoteOutputToken.push_back(ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_VAR));
        }
        return CCU_SUCCESS;
    }

    CcuResult ExchangeLargeOutputs(LargeContext &ctx)
    {
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            ChannelHandle channel = ctx.arg->channels[index];
            RETURN_IF_CCU_ERROR(
                ccu::WriteVariableWithNotify(channel, ctx.output, OUTPUT_VAR, ADDRESS_NOTIFY, OUTPUT_MASK));
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
                channel, ctx.outputToken, OUTPUT_TOKEN_VAR, ADDRESS_NOTIFY, OUTPUT_TOKEN_MASK));
        }
        uint16_t mask = static_cast<uint16_t>(OUTPUT_MASK | OUTPUT_TOKEN_MASK);
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[index], ADDRESS_NOTIFY, mask));
        }
        return CCU_SUCCESS;
    }

    CcuResult SynchronizeLarge(LargeContext &ctx)
    {
        // 按 Rank 定序的 Post/Ack 交换使每条 Channel 上的对称 Barrier 不会死锁。
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            if (ctx.arg->myRank < ctx.arg->peers[index]) {
                RETURN_IF_CCU_ERROR(ccu::NotifyRecord(ctx.arg->channels[index], DATA_NOTIFY, POST_MASK));
            }
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            if (ctx.arg->myRank > ctx.arg->peers[index]) {
                RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[index], DATA_NOTIFY, POST_MASK));
            }
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            if (ctx.arg->myRank > ctx.arg->peers[index]) {
                RETURN_IF_CCU_ERROR(ccu::NotifyRecord(ctx.arg->channels[index], DATA_NOTIFY, ACK_MASK));
            }
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            if (ctx.arg->myRank < ctx.arg->peers[index]) {
                RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[index], DATA_NOTIFY, ACK_MASK));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult InitializeLatinOwner(LargeContext &ctx)
    {
        if (ctx.arg->dieId != ctx.arg->primaryDie) {
            return CCU_SUCCESS;
        }
        uint32_t stripsPerOwner = ctx.arg->rankSize - 1U;
        ccu::Variable ownerOffset = Scale(ctx.normalBytes, ctx.arg->myRank * stripsPerOwner);
        ccu::Variable ownerBytes = Scale(ctx.normalBytes, stripsPerOwner);
        if (ctx.arg->myRank + 1U == ctx.arg->rankSize) {
            ownerBytes = Scale(ctx.normalBytes, stripsPerOwner - 1U);
            ownerBytes += ctx.lastBytes;
        }
        CCU_IF(ctx.inplace == 0ULL)
        {
            ccu::LocalAddr source = LocalAddress(ctx.input, ctx.inputToken, ownerOffset);
            ccu::LocalAddr destination = LocalAddress(ctx.output, ctx.outputToken, ownerOffset);
            RETURN_IF_CCU_ERROR(ccu::LocalCopy(destination, source, ownerBytes, ctx.event, 1U));
            RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, 1U));
        }
        return CCU_SUCCESS;
    }

    CcuResult LatinReduceRound(LargeContext &ctx)
    {
        uint32_t lastRank = ctx.arg->rankSize - 1U;
        uint16_t writeMask = 0;
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            uint32_t owner = ctx.arg->peers[index];
            const ccu::Variable &offset = ctx.offsets[owner];
            ccu::LocalAddr source = LocalAddress(ctx.input, ctx.inputToken, offset);
            ccu::RemoteAddr destination = RemoteAddress(ctx.remoteOutput[index], ctx.remoteOutputToken[index], offset);
            ccu::Variable bytes;
            bytes = ctx.normalBytes;
            if (owner == lastRank) {
                CCU_IF(ctx.tailRound == 1ULL)
                {
                    bytes = ctx.lastBytes;
                }
            }
            uint16_t mask = static_cast<uint16_t>(1U << index);
            writeMask = static_cast<uint16_t>(writeMask | mask);
            RETURN_IF_CCU_ERROR(ccu::WriteReduce(ctx.arg->channels[index], destination, source, bytes,
                HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, mask));
        }
        return ccu::EventWait(ctx.event, writeMask);
    }

    CcuResult LatinGather(LargeContext &ctx)
    {
        uint32_t stripsPerOwner = ctx.arg->rankSize - 1U;
        ccu::Variable normalOwnerBytes = Scale(ctx.normalBytes, stripsPerOwner);
        ccu::Variable lastOwnerBytes = Scale(ctx.normalBytes, stripsPerOwner - 1U);
        lastOwnerBytes += ctx.lastBytes;
        uint16_t readMask = 0;
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            uint32_t owner = ctx.arg->peers[index];
            ccu::Variable ownerOffset = Scale(ctx.normalBytes, owner * stripsPerOwner);
            const ccu::Variable &bytes = owner + 1U == ctx.arg->rankSize ? lastOwnerBytes : normalOwnerBytes;
            ccu::LocalAddr destination = LocalAddress(ctx.output, ctx.outputToken, ownerOffset);
            ccu::RemoteAddr source = RemoteAddress(ctx.remoteOutput[index], ctx.remoteOutputToken[index], ownerOffset);
            uint16_t mask = static_cast<uint16_t>(1U << index);
            readMask = static_cast<uint16_t>(readMask | mask);
            RETURN_IF_CCU_ERROR(ccu::Read(ctx.arg->channels[index], destination, source, bytes, ctx.event, mask));
        }
        return ccu::EventWait(ctx.event, readMask);
    }

    CcuResult RunLatinLarge(LargeContext &ctx)
    {
        CCU_IF(ctx.phase == latin::INIT_PHASE)
        {
            RETURN_IF_CCU_ERROR(InitializeLatinOwner(ctx));
            RETURN_IF_CCU_ERROR(ExchangeLargeOutputs(ctx));
        }
        CCU_IF(ctx.phase == latin::REDUCE_PHASE)
        {
            RETURN_IF_CCU_ERROR(LatinReduceRound(ctx));
        }
        CCU_IF(ctx.phase == latin::GATHER_PHASE)
        {
            RETURN_IF_CCU_ERROR(LatinGather(ctx));
        }
        CCU_IF(ctx.phase == latin::BARRIER_PHASE)
        {
            // Host 先汇合两个 Die，再执行跨 Rank Barrier，以保持轮次边界。
            RETURN_IF_CCU_ERROR(SynchronizeLarge(ctx));
        }
        return CCU_SUCCESS;
    }

    struct PackedContext {
        const LargeKernelArg *arg = nullptr;
        ccu::Variable input;
        ccu::Variable inputToken;
        ccu::Variable output;
        ccu::Variable outputToken;
        ccu::Variable scratch;
        ccu::Variable scratchToken;
        ccu::Variable normalBytes;
        ccu::Variable lastBytes;
        ccu::Variable cellBytes;
        ccu::Variable tailBytes;
        ccu::Variable otherCellBytes;
        ccu::Variable otherTailBytes;
        ccu::Variable phase;
        std::vector<ccu::Variable> remoteInput;
        std::vector<ccu::Variable> remoteInputToken;
        std::vector<ccu::Variable> remoteOutput;
        std::vector<ccu::Variable> remoteOutputToken;
        ccu::Event event;
    };

    CcuResult LoadPackedContext(PackedContext &ctx, CcuKernelArg arg)
    {
        ctx.arg = static_cast<const LargeKernelArg *>(arg);
        if (ctx.arg == nullptr || ctx.arg->mode != KernelMode::LARGE_2X8_PACKED_PULL || ctx.arg->rankSize != 16U
            || ctx.arg->channelCount == 0U || ctx.arg->channelCount > MAX_RANK_SIZE || ctx.arg->dieId >= DIE_NUM
            || ctx.arg->primaryDie >= DIE_NUM) {
            return CCU_E_PARA;
        }
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.input, 0U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.inputToken, 1U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.output, 2U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputToken, 3U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.scratch, 4U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.scratchToken, 5U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.normalBytes, 6U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.lastBytes, 7U));
        // 两个 Die Kernel 使用相同的 13 参数 ABI，仅本地 Cell 参数对的位置不同。
        if (ctx.arg->dieId == 0U) {
            RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.cellBytes, 8U));
            RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.tailBytes, 9U));
            RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.otherCellBytes, 10U));
            RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.otherTailBytes, 11U));
        } else {
            RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.otherCellBytes, 8U));
            RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.otherTailBytes, 9U));
            RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.cellBytes, 10U));
            RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.tailBytes, 11U));
        }
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.phase, 12U));
        ctx.remoteInput.reserve(ctx.arg->channelCount);
        ctx.remoteInputToken.reserve(ctx.arg->channelCount);
        ctx.remoteOutput.reserve(ctx.arg->channelCount);
        ctx.remoteOutputToken.reserve(ctx.arg->channelCount);
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            const ChannelHandle channel = ctx.arg->channels[index];
            ctx.remoteInput.push_back(ccu::GetResByChannel<ccu::Variable>(channel, INPUT_VAR));
            ctx.remoteInputToken.push_back(ccu::GetResByChannel<ccu::Variable>(channel, INPUT_TOKEN_VAR));
            ctx.remoteOutput.push_back(ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_VAR));
            ctx.remoteOutputToken.push_back(ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_VAR));
        }
        return CCU_SUCCESS;
    }

    ccu::Variable PackedOwnerOffset(PackedContext &ctx, uint32_t owner)
    {
        return Scale(ctx.normalBytes, owner);
    }

    ccu::Variable PackedOwnerBytes(PackedContext &ctx, uint32_t owner)
    {
        return owner + 1U == ctx.arg->rankSize ? ctx.lastBytes : ctx.normalBytes;
    }

    ccu::LocalAddr PackedLocalInput(PackedContext &ctx, uint32_t owner, const ccu::Variable &offset)
    {
        ccu::Variable totalOffset = PackedOwnerOffset(ctx, owner);
        totalOffset += offset;
        return LocalAddress(ctx.input, ctx.inputToken, totalOffset);
    }

    ccu::LocalAddr PackedLocalOutput(PackedContext &ctx, uint32_t owner, const ccu::Variable &offset)
    {
        ccu::Variable totalOffset = PackedOwnerOffset(ctx, owner);
        totalOffset += offset;
        return LocalAddress(ctx.output, ctx.outputToken, totalOffset);
    }

    ccu::LocalAddr PackedPartial(PackedContext &ctx, const ccu::Variable &offset)
    {
        // 主 Die 直接累加到最终 Owner 分片，从 Die 使用 Scratch 保存局部结果。
        if (ctx.arg->dieId == ctx.arg->primaryDie) {
            return PackedLocalOutput(ctx, ctx.arg->myRank, offset);
        }
        return LocalAddress(ctx.scratch, ctx.scratchToken, offset);
    }

    ccu::RemoteAddr PackedRemoteInput(PackedContext &ctx, uint32_t channel, const ccu::Variable &offset)
    {
        ccu::Variable totalOffset = PackedOwnerOffset(ctx, ctx.arg->myRank);
        totalOffset += offset;
        return RemoteAddress(ctx.remoteInput[channel], ctx.remoteInputToken[channel], totalOffset);
    }

    CcuResult PackedSync(PackedContext &ctx, uint16_t postMask, uint16_t ackMask)
    {
        // 有序 Post/Ack 协议允许后续 Packed 阶段安全复用同一个 Notify 索引。
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            if (ctx.arg->myRank < ctx.arg->peers[index]) {
                RETURN_IF_CCU_ERROR(ccu::NotifyRecord(ctx.arg->channels[index], DATA_NOTIFY, postMask));
            }
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            if (ctx.arg->myRank > ctx.arg->peers[index]) {
                RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[index], DATA_NOTIFY, postMask));
            }
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            if (ctx.arg->myRank > ctx.arg->peers[index]) {
                RETURN_IF_CCU_ERROR(ccu::NotifyRecord(ctx.arg->channels[index], DATA_NOTIFY, ackMask));
            }
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            if (ctx.arg->myRank < ctx.arg->peers[index]) {
                RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[index], DATA_NOTIFY, ackMask));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult InitializePackedOwner(PackedContext &ctx)
    {
        uint16_t copyMask = 0;
        ccu::Variable zero;
        zero = 0;
        if (ctx.arg->dieId == ctx.arg->primaryDie) {
            RETURN_IF_CCU_ERROR(ccu::LocalCopy(PackedLocalOutput(ctx, ctx.arg->myRank, zero),
                PackedLocalInput(ctx, ctx.arg->myRank, zero), PackedOwnerBytes(ctx, ctx.arg->myRank), ctx.event, 1U));
            copyMask = 1U;
        }
        constexpr uint16_t addressMask = INPUT_MASK | INPUT_TOKEN_MASK | OUTPUT_MASK | OUTPUT_TOKEN_MASK;
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            const ChannelHandle channel = ctx.arg->channels[index];
            RETURN_IF_CCU_ERROR(
                ccu::WriteVariableWithNotify(channel, ctx.input, INPUT_VAR, ADDRESS_NOTIFY, INPUT_MASK));
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
                channel, ctx.inputToken, INPUT_TOKEN_VAR, ADDRESS_NOTIFY, INPUT_TOKEN_MASK));
            RETURN_IF_CCU_ERROR(
                ccu::WriteVariableWithNotify(channel, ctx.output, OUTPUT_VAR, ADDRESS_NOTIFY, OUTPUT_MASK));
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
                channel, ctx.outputToken, OUTPUT_TOKEN_VAR, ADDRESS_NOTIFY, OUTPUT_TOKEN_MASK));
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[index], ADDRESS_NOTIFY, addressMask));
        }
        return copyMask == 0U ? CCU_SUCCESS : ccu::EventWait(ctx.event, copyMask);
    }

    CcuResult PullPackedPartial(PackedContext &ctx)
    {
        const uint16_t eventMask = static_cast<uint16_t>((1U << ctx.arg->channelCount) - 1U);
        // 轮转各 Channel 的首个 Cell，使并发读取始终落到互斥的本地 Cell。
        std::vector<ccu::Variable> offsets(ctx.arg->channelCount);
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            ccu::Variable offset = Scale(ctx.cellBytes, index);
            const uint16_t bit = static_cast<uint16_t>(1U << index);
            const CcuResult result = ctx.arg->dieId == ctx.arg->primaryDie
                                         ? ccu::ReadReduce(ctx.arg->channels[index], PackedPartial(ctx, offset),
                                               PackedRemoteInput(ctx, index, offset), ctx.cellBytes,
                                               HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, bit)
                                         : ccu::Read(ctx.arg->channels[index], PackedPartial(ctx, offset),
                                               PackedRemoteInput(ctx, index, offset), ctx.cellBytes, ctx.event, bit);
            if (result != CCU_SUCCESS) {
                return result;
            }
            offsets[index] = Scale(ctx.cellBytes, (index + 1U) % ctx.arg->channelCount);
        }
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, eventMask));

        ccu::Variable round;
        ccu::Variable one;
        round = 1ULL;
        one = 1ULL;
        CCU_WHILE(round != static_cast<uint64_t>(ctx.arg->channelCount))
        {
            for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
                const uint16_t bit = static_cast<uint16_t>(1U << index);
                RETURN_IF_CCU_ERROR(ccu::ReadReduce(ctx.arg->channels[index], PackedPartial(ctx, offsets[index]),
                    PackedRemoteInput(ctx, index, offsets[index]), ctx.cellBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
                    ctx.event, bit));
            }
            RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, eventMask));
            for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
                offsets[index] += ctx.cellBytes;
                const uint32_t wrapRound = ctx.arg->channelCount - 1U - index;
                if (wrapRound != 0U) {
                    CCU_IF(round == static_cast<uint64_t>(wrapRound))
                    {
                        offsets[index] = 0ULL;
                    }
                }
            }
            round += one;
        }

        ccu::Variable tailOffset = Scale(ctx.cellBytes, ctx.arg->channelCount);
        CCU_IF(ctx.tailBytes != 0ULL)
        {
            for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
                const CcuResult result
                    = ctx.arg->dieId != ctx.arg->primaryDie && index == 0U
                          ? ccu::Read(ctx.arg->channels[index], PackedPartial(ctx, tailOffset),
                                PackedRemoteInput(ctx, index, tailOffset), ctx.tailBytes, ctx.event, 1U)
                          : ccu::ReadReduce(ctx.arg->channels[index], PackedPartial(ctx, tailOffset),
                                PackedRemoteInput(ctx, index, tailOffset), ctx.tailBytes, HCCL_DATA_TYPE_FP32,
                                HCCL_REDUCE_SUM, ctx.event, 1U);
                if (result != CCU_SUCCESS) {
                    return result;
                }
                RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, 1U));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult MergePackedOwner(PackedContext &ctx)
    {
        if (ctx.arg->dieId != ctx.arg->primaryDie) {
            return CCU_SUCCESS;
        }
        ccu::Variable zero;
        zero = 0;
        // Host 汇合两个 Init/Pull Kernel 后，才在主 Die 下发合并阶段。
        RETURN_IF_CCU_ERROR(ccu::LocalReduce(PackedLocalOutput(ctx, ctx.arg->myRank, zero),
            LocalAddress(ctx.scratch, ctx.scratchToken, zero), PackedOwnerBytes(ctx, ctx.arg->myRank),
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 1U));
        return ccu::EventWait(ctx.event, 1U);
    }

    CcuResult GatherPackedOwners(PackedContext &ctx)
    {
        RETURN_IF_CCU_ERROR(PackedSync(ctx, 1U << 0, 1U << 1));
        ccu::Variable zero;
        zero = 0;
        uint16_t readMask = 0;
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            const uint32_t owner = ctx.arg->peers[index];
            const uint16_t bit = static_cast<uint16_t>(1U << index);
            readMask = static_cast<uint16_t>(readMask | bit);
            RETURN_IF_CCU_ERROR(ccu::Read(ctx.arg->channels[index], PackedLocalOutput(ctx, owner, zero),
                RemoteAddress(ctx.remoteOutput[index], ctx.remoteOutputToken[index], PackedOwnerOffset(ctx, owner)),
                PackedOwnerBytes(ctx, owner), ctx.event, bit));
        }
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, readMask));
        return PackedSync(ctx, 1U << 2, 1U << 3);
    }

    CcuResult RunPackedLarge(PackedContext &ctx)
    {
        CCU_IF(ctx.phase == packed_pull::INIT_PHASE)
        {
            RETURN_IF_CCU_ERROR(InitializePackedOwner(ctx));
            RETURN_IF_CCU_ERROR(PullPackedPartial(ctx));
        }
        CCU_IF(ctx.phase == packed_pull::MERGE_PHASE)
        {
            RETURN_IF_CCU_ERROR(MergePackedOwner(ctx));
        }
        CCU_IF(ctx.phase == packed_pull::GATHER_PHASE)
        {
            RETURN_IF_CCU_ERROR(GatherPackedOwners(ctx));
        }
        return CCU_SUCCESS;
    }

    struct Fused4x1Context {
        const LargeKernelArg *arg = nullptr;
        ccu::Variable input;
        ccu::Variable output;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable offsets[fused_4x1::RANK_SIZE];
        ccu::Variable bytes[fused_4x1::RANK_SIZE];
        ccu::Variable remoteOutput[fused_4x1::RANK_SIZE - 1U];
        ccu::Variable remoteToken[fused_4x1::RANK_SIZE - 1U];
        ccu::Event event;
    };

    int32_t LargeChannelForPeer(const LargeKernelArg &arg, uint32_t peer)
    {
        for (uint32_t index = 0; index < arg.channelCount; ++index) {
            if (arg.peers[index] == peer) {
                return static_cast<int32_t>(index);
            }
        }
        return -1;
    }

    CcuResult LoadFused4x1Context(Fused4x1Context &ctx, CcuKernelArg arg)
    {
        ctx.arg = static_cast<const LargeKernelArg *>(arg);
        if (ctx.arg == nullptr || ctx.arg->mode != KernelMode::LARGE_4X1_FUSED_RSAG
            || ctx.arg->rankSize != fused_4x1::RANK_SIZE || ctx.arg->channelCount != fused_4x1::RANK_SIZE - 1U) {
            return CCU_E_PARA;
        }
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.input, 0U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.output, 1U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.inputToken, 2U));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputToken, 3U));
        for (uint32_t rank = 0; rank < fused_4x1::RANK_SIZE; ++rank) {
            RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.offsets[rank], 4U + rank * 2U));
            RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.bytes[rank], 5U + rank * 2U));
        }
        constexpr uint16_t addressMask = OUTPUT_MASK | OUTPUT_TOKEN_MASK;
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            const ChannelHandle channel = ctx.arg->channels[index];
            ctx.remoteOutput[index] = ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_VAR);
            ctx.remoteToken[index] = ccu::GetResByChannel<ccu::Variable>(channel, OUTPUT_TOKEN_VAR);
            RETURN_IF_CCU_ERROR(
                ccu::WriteVariableWithNotify(channel, ctx.output, OUTPUT_VAR, ADDRESS_NOTIFY, OUTPUT_MASK));
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
                channel, ctx.outputToken, OUTPUT_TOKEN_VAR, ADDRESS_NOTIFY, OUTPUT_TOKEN_MASK));
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[index], ADDRESS_NOTIFY, addressMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunFused4x1(Fused4x1Context &ctx)
    {
        const uint32_t rank = ctx.arg->myRank;
        RETURN_IF_CCU_ERROR(ccu::LocalCopy(LocalAddress(ctx.output, ctx.outputToken, ctx.offsets[rank]),
            LocalAddress(ctx.input, ctx.inputToken, ctx.offsets[rank]), ctx.bytes[rank], ctx.event));
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event));
        for (uint32_t step = 0; step < fused_4x1::RANK_SIZE - 1U; ++step) {
            const uint32_t peer = rank ^ (step + 1U);
            const int32_t channelIndex = LargeChannelForPeer(*ctx.arg, peer);
            if (channelIndex < 0) {
                return CCU_E_INTERNAL;
            }
            const ChannelHandle channel = ctx.arg->channels[channelIndex];
            // Owner 分片全部初始化后，通过 Ready/Done 握手串行化 Peer 规约。
            RETURN_IF_CCU_ERROR(ccu::NotifyRecord(channel, ADDRESS_NOTIFY, 1U << 4));
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(channel, ADDRESS_NOTIFY, 1U << 4));
            RETURN_IF_CCU_ERROR(ccu::WriteReduce(channel,
                RemoteAddress(ctx.remoteOutput[channelIndex], ctx.remoteToken[channelIndex], ctx.offsets[peer]),
                LocalAddress(ctx.input, ctx.inputToken, ctx.offsets[peer]), ctx.bytes[peer], HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, ctx.event));
            RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event));
            RETURN_IF_CCU_ERROR(ccu::NotifyRecord(channel, ADDRESS_NOTIFY, 1U << 5));
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(channel, ADDRESS_NOTIFY, 1U << 5));
        }
        const ccu::LocalAddr owner = LocalAddress(ctx.output, ctx.outputToken, ctx.offsets[rank]);
        uint16_t writeMask = 0;
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            const uint16_t bit = static_cast<uint16_t>(1U << index);
            writeMask = static_cast<uint16_t>(writeMask | bit);
            RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[index],
                RemoteAddress(ctx.remoteOutput[index], ctx.remoteToken[index], ctx.offsets[rank]), owner,
                ctx.bytes[rank], ctx.event, bit));
        }
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, writeMask));
        // 两级距离 Ring Barrier 保证 Kernel 返回前所有远端 Owner 写入均已可见。
        for (uint32_t distance = 1U; distance < fused_4x1::RANK_SIZE; distance *= 2U) {
            const uint32_t sendPeer = (rank + distance) % fused_4x1::RANK_SIZE;
            const uint32_t receivePeer = (rank + fused_4x1::RANK_SIZE - distance) % fused_4x1::RANK_SIZE;
            const int32_t sendChannel = LargeChannelForPeer(*ctx.arg, sendPeer);
            const int32_t receiveChannel = LargeChannelForPeer(*ctx.arg, receivePeer);
            if (sendChannel < 0 || receiveChannel < 0) {
                return CCU_E_INTERNAL;
            }
            RETURN_IF_CCU_ERROR(ccu::NotifyRecord(ctx.arg->channels[sendChannel], ADDRESS_NOTIFY, 1U << 6));
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[receiveChannel], ADDRESS_NOTIFY, 1U << 6));
        }
        return CCU_SUCCESS;
    }
} // namespace

CcuResult SmallAllReduceKernel(CcuKernelArg arg)
{
    SmallContext ctx;
    RETURN_IF_CCU_ERROR(LoadSmallContext(ctx, arg));
    if (ctx.arg->mode == KernelMode::SMALL_OWNER_PULL) {
        return RunOwnerPullSmall(ctx);
    }
    if (ctx.arg->mode == KernelMode::SMALL_4X1_ONESHOT) {
        return Run4x1OneShot(ctx);
    }
    return CCU_E_NOT_SUPPORT;
}

CcuResult LargeAllReduceKernel(CcuKernelArg arg)
{
    const auto *kernelArg = static_cast<const LargeKernelArg *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PARA;
    }
    if (kernelArg->mode == KernelMode::LARGE_2X8_PACKED_PULL) {
        PackedContext ctx;
        RETURN_IF_CCU_ERROR(LoadPackedContext(ctx, arg));
        return RunPackedLarge(ctx);
    }
    if (kernelArg->mode == KernelMode::LARGE_4X1_FUSED_RSAG) {
        Fused4x1Context ctx;
        RETURN_IF_CCU_ERROR(LoadFused4x1Context(ctx, arg));
        return RunFused4x1(ctx);
    }
    LargeContext ctx;
    RETURN_IF_CCU_ERROR(LoadLargeContext(ctx, arg));
    return RunLatinLarge(ctx);
}
} // namespace ops_hccl
