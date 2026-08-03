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
#include "log.h"

#define CCU_RETURN_IF_ERROR(call) \
    do { \
        const CcuResult ccuCallRet = (call); \
        if (ccuCallRet != CCU_SUCCESS) { \
            return ccuCallRet; \
        } \
    } while (0)

namespace ops_hccl {

namespace {

constexpr uint32_t ADDR_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t POST_SYNC_ID = 3;
constexpr uint32_t SEED_READY_ID = 4;
constexpr uint32_t CKE_INDEX = 0;
constexpr uint32_t FIXED_TASK_ARG_NUM = 5;
constexpr uint32_t MAX_CHANNELS_PER_DIE = 8;
constexpr uint32_t MAX_CCU_TASK_ARGS = 13;
constexpr uint32_t CLOS_4X1_CHANNEL_NUM = 3;
constexpr uint32_t CLOS_4X1_MAX_CHUNKS = 4;
constexpr uint16_t ADDR_MASK = static_cast<uint16_t>(1U << ADDR_XN_ID);
constexpr uint16_t TOKEN_MASK = static_cast<uint16_t>(1U << TOKEN_XN_ID);
constexpr uint16_t POST_SYNC_MASK = static_cast<uint16_t>(1U << POST_SYNC_ID);
constexpr uint16_t SEED_READY_MASK = static_cast<uint16_t>(1U << SEED_READY_ID);
constexpr uint16_t ADDRESS_READY_MASK = static_cast<uint16_t>(ADDR_MASK | TOKEN_MASK);

bool IsArgValid(const CcuDirectKernelArg *arg)
{
    return arg != nullptr && arg->rankSize >= 2 && arg->rankSize <= MAX_RANK_SIZE
        && arg->rankId < arg->rankSize && arg->channelCount > 0
        && arg->channelCount <= MAX_CHANNELS_PER_DIE
        && FIXED_TASK_ARG_NUM + arg->channelCount <= MAX_CCU_TASK_ARGS;
}

CcuResult ValidatePeerMap(const CcuDirectKernelArg *arg)
{
    bool seenRank[MAX_RANK_SIZE]{};
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const uint32_t peerRank = arg->peerRanks[channelIndex];
        if (peerRank >= arg->rankSize || peerRank == arg->rankId || seenRank[peerRank]) {
            return CCU_E_PARA;
        }
        seenRank[peerRank] = true;
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuPeerLanePullKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuDirectKernelArg *>(arg);
    if (!IsArgValid(kernelArg) || ValidatePeerMap(kernelArg) != CCU_SUCCESS) {
        HCCL_ERROR("[CcuPeerLanePullKernel] Invalid kernel argument");
        return CCU_E_PARA;
    }

    CcuPeerLanePullContext ctx{};
    ctx.arg = kernelArg;
    ctx.inputAddrs.resize(ctx.arg->channelCount + 1U);
    ctx.inputTokens.resize(ctx.arg->channelCount + 1U);
    ctx.destinationAddrs.resize(ctx.arg->channelCount);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ctx.inputAddrs[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], ADDR_XN_ID);
        ctx.inputTokens[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], TOKEN_XN_ID);
    }
    const uint32_t localIndex = ctx.arg->channelCount;

    // 参数布局：
    // [0] sourceAddr [1] sourceToken [2] outputToken [3] firstBytes [4] secondBytes
    // [5..12] 每个Channel对应的绝对目标地址。每Die最多8条Channel，总参数不超过13。
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputAddrs[localIndex], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputTokens[localIndex], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.firstSliceSize, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.secondSliceSize, argId++));
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.destinationAddrs[channelIndex], argId++));
    }

    // 先广播本Rank输入地址。随后每条Channel独立等待并尽早下发Read，
    // 去掉V8“等齐整个Die才开始传输”的前置屏障。
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.inputAddrs[localIndex], ADDR_XN_ID, CKE_INDEX, ADDR_MASK));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.inputTokens[localIndex], TOKEN_XN_ID, CKE_INDEX, TOKEN_MASK));
    }

    uint16_t laneMask = 0;
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, ADDRESS_READY_MASK));

        ccu::RemoteAddr source;
        source.addr = ctx.inputAddrs[channelIndex];
        source.token = ctx.inputTokens[channelIndex];
        ccu::LocalAddr destination;
        destination.addr = ctx.destinationAddrs[channelIndex];
        destination.token = ctx.outputToken;
        const uint16_t eventMask = static_cast<uint16_t>(1U << channelIndex);
        CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[channelIndex], destination, source,
            ctx.firstSliceSize, ctx.firstEvent, eventMask));
        laneMask = static_cast<uint16_t>(laneMask | eventMask);
    }

    CCU_IF(ctx.secondSliceSize != 0)
    {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ccu::RemoteAddr source;
            source.addr = ctx.inputAddrs[channelIndex];
            source.addr += ctx.firstSliceSize;
            source.token = ctx.inputTokens[channelIndex];
            ccu::LocalAddr destination;
            destination.addr = ctx.destinationAddrs[channelIndex];
            destination.addr += ctx.firstSliceSize;
            destination.token = ctx.outputToken;
            const uint16_t eventMask = static_cast<uint16_t>(1U << channelIndex);
            CCU_RETURN_IF_ERROR(ccu::Read(ctx.arg->channels[channelIndex], destination, source,
                ctx.secondSliceSize, ctx.secondEvent, eventMask));
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.secondEvent, laneMask));
    }
    CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.firstEvent, laneMask));

    // Pull完成本地输出后仍做逐Peer生命周期确认，保证本Rank返回前对端不再读取sendBuf。
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyRecord(ctx.arg->channels[channelIndex], CKE_INDEX, POST_SYNC_MASK));
    }
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

namespace {

struct RotatingPushContext {
    const CcuRotatingPushKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankOutputOffset;
    ccu::Variable firstBytes;
    ccu::Variable secondBytes;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event remoteEvent;
    ccu::Event localCopyEvent;
};

struct LatencyPushContext {
    const CcuLatencyPushKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankOutputOffset;
    ccu::Variable transferBytes;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event remoteEvent;
    ccu::Event localCopyEvent;
};

struct Clos4x1PushContext {
    const CcuClos4x1PushKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankOutputOffset;
    ccu::Variable chunkBytes[CLOS_4X1_MAX_CHUNKS];
    ccu::Variable remoteOutputs[CLOS_4X1_CHANNEL_NUM];
    ccu::Variable remoteTokens[CLOS_4X1_CHANNEL_NUM];
    ccu::Event networkEvent;
    ccu::Event localCopyEvent;
};

struct DualSeedCrossContext {
    const CcuDualSeedCrossKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankOutputOffset;
    ccu::Variable smallDirectBytes;
    ccu::Variable smallStripe0Bytes;
    ccu::Variable smallStripe1Bytes;
    ccu::Variable largeDirectBytes;
    ccu::Variable largeTailBytes;
    ccu::Variable phaseMode;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event remoteEvent;
};

struct WideDualSeedCrossContext {
    const CcuWideDualSeedCrossKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankOutputOffset;
    ccu::Variable smallDirectFirstBytes;
    ccu::Variable smallDirectSecondBytes;
    ccu::Variable smallStripe0Bytes;
    ccu::Variable smallStripe1Bytes;
    ccu::Variable largeDirectFirstBytes;
    ccu::Variable largeDirectSecondBytes;
    ccu::Variable largeTailBytes;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event remoteEvent;
};

struct DualSeedRelayContext {
    const CcuDualSeedRelayKernelArg *arg = nullptr;
    ccu::Variable output;
    ccu::Variable outputToken;
    ccu::Variable sourceBlockOffsets[2];
    ccu::Variable smallDirectBytes;
    ccu::Variable smallStripe0Bytes;
    ccu::Variable smallStripe1Bytes;
    ccu::Variable largeDirectBytes;
    ccu::Variable largeTailBytes;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event relayEvent;
};

bool ValidateStaticPeerMap(uint32_t rankSize, uint32_t rankId, uint32_t channelCount,
    const uint32_t *peerRanks)
{
    if (rankSize < 2U || rankSize > MAX_RANK_SIZE || rankId >= rankSize
        || channelCount == 0U || channelCount > MAX_CHANNELS_PER_DIE || peerRanks == nullptr) {
        return false;
    }
    bool seenRank[MAX_RANK_SIZE]{};
    for (uint32_t i = 0; i < channelCount; ++i) {
        const uint32_t peerRank = peerRanks[i];
        if (peerRank >= rankSize || peerRank == rankId || seenRank[peerRank]) {
            return false;
        }
        seenRank[peerRank] = true;
    }
    return true;
}

uint32_t FindPeerChannel(const uint32_t *peerRanks, uint32_t channelCount, uint32_t targetRank)
{
    for (uint32_t i = 0; i < channelCount; ++i) {
        if (peerRanks[i] == targetRank) {
            return i;
        }
    }
    return channelCount;
}

CcuResult ExchangeOutputResources(const CcuKernelArgBase *arg, ccu::Variable &localOutput,
    ccu::Variable &localOutputToken, std::vector<ccu::Variable> &remoteOutputs,
    std::vector<ccu::Variable> &remoteTokens)
{
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            arg->channels[i], localOutput, ADDR_XN_ID, CKE_INDEX, ADDR_MASK));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            arg->channels[i], localOutputToken, TOKEN_XN_ID, CKE_INDEX, TOKEN_MASK));
    }
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(arg->channels[i], CKE_INDEX, ADDRESS_READY_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult PublishOutputResources(const CcuKernelArgBase *arg,
    ccu::Variable &localOutput, ccu::Variable &localOutputToken)
{
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            arg->channels[i], localOutput, ADDR_XN_ID, CKE_INDEX, ADDR_MASK));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            arg->channels[i], localOutputToken, TOKEN_XN_ID, CKE_INDEX, TOKEN_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult WaitPostSync(const CcuKernelArgBase *arg)
{
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(arg->channels[i], CKE_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult FullPostSync(const CcuKernelArgBase *arg)
{
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyRecord(arg->channels[i], CKE_INDEX, POST_SYNC_MASK));
    }
    return WaitPostSync(arg);
}

// 512KB时数据面很短，先做一次组合EventWait再开始全部PostSync会把
// “最慢数据完成”和“最慢生命周期确认”串成两段。这里让每条Channel一完成
// 就立即发出自己的PostSync，最后再统一等待对端确认。事件位0..N-1必须与
// Channel一一对应，因此该函数只由单Slice专用静态Kernel调用。
CcuResult LatencyCompleteAndPostSync(
    const CcuKernelArgBase *arg, ccu::Event &event)
{
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        const uint16_t eventBit = static_cast<uint16_t>(1U << i);
        CCU_RETURN_IF_ERROR(ccu::EventWait(event, eventBit));
        CCU_RETURN_IF_ERROR(
            ccu::NotifyRecord(arg->channels[i], CKE_INDEX, POST_SYNC_MASK));
    }
    return WaitPostSync(arg);
}

} // namespace

CcuResult CcuRotatingDirectPushKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuRotatingPushKernelArg *>(arg);
    if (kernelArg == nullptr
        || !ValidateStaticPeerMap(kernelArg->rankSize, kernelArg->rankId,
            kernelArg->channelCount, kernelArg->peerRanks)
        || kernelArg->channelCount > 7U) {
        HCCL_ERROR("[CcuRotatingDirectPushKernel] Invalid static peer map");
        return CCU_E_PARA;
    }

    RotatingPushContext ctx{};
    ctx.arg = kernelArg;
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        ctx.remoteOutputs[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], ADDR_XN_ID);
        ctx.remoteTokens[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
    }

    // [0]input [1]output [2]inputToken [3]outputToken [4]rankOffset
    // [5]firstBytes [6]secondBytes，共7个动态参数。
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.rankOutputOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.firstBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.secondBytes, argId++));

    if (ctx.arg->latencyPipeline) {
        CCU_RETURN_IF_ERROR(
            PublishOutputResources(ctx.arg, ctx.output, ctx.outputToken));
    } else {
        CCU_RETURN_IF_ERROR(ExchangeOutputResources(
            ctx.arg, ctx.output, ctx.outputToken, ctx.remoteOutputs, ctx.remoteTokens));
    }

    uint16_t remoteMask = 0U;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        if (ctx.arg->latencyPipeline) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                ctx.arg->channels[i], CKE_INDEX, ADDRESS_READY_MASK));
        }
        ccu::LocalAddr source;
        source.addr = ctx.input;
        source.token = ctx.inputToken;
        ccu::RemoteAddr destination;
        destination.addr = ctx.remoteOutputs[i];
        destination.addr += ctx.rankOutputOffset;
        destination.token = ctx.remoteTokens[i];
        const uint16_t eventBit = static_cast<uint16_t>(1U << i);
        remoteMask = static_cast<uint16_t>(remoteMask | eventBit);
        CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[i], destination, source,
            ctx.firstBytes, ctx.remoteEvent, eventBit));
    }

    if (ctx.arg->handleLocalCopy) {
        ccu::LocalAddr localSource;
        localSource.addr = ctx.input;
        localSource.token = ctx.inputToken;
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.output;
        localDestination.addr += ctx.rankOutputOffset;
        localDestination.token = ctx.outputToken;
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDestination, localSource,
            ctx.firstBytes, ctx.localCopyEvent, 1U));
    }

    // 第二条带把Channel提交顺序循环移动一位，避免两个大条带都由同一物理链路先发。
    uint16_t secondRemoteMask = 0U;
    CCU_IF(ctx.secondBytes != 0U)
    {
        for (uint32_t issueIndex = 0; issueIndex < ctx.arg->channelCount; ++issueIndex) {
            const uint32_t channelIndex = (issueIndex + 1U) % ctx.arg->channelCount;
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.addr += ctx.firstBytes;
            source.token = ctx.inputToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.rankOutputOffset;
            destination.addr += ctx.firstBytes;
            destination.token = ctx.remoteTokens[channelIndex];
            const uint16_t eventBit =
                static_cast<uint16_t>(1U << (ctx.arg->channelCount + issueIndex));
            secondRemoteMask = static_cast<uint16_t>(secondRemoteMask | eventBit);
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                ctx.secondBytes, ctx.remoteEvent, eventBit));
        }
        if (ctx.arg->handleLocalCopy) {
            ccu::LocalAddr localSource;
            localSource.addr = ctx.input;
            localSource.addr += ctx.firstBytes;
            localSource.token = ctx.inputToken;
            ccu::LocalAddr localDestination;
            localDestination.addr = ctx.output;
            localDestination.addr += ctx.rankOutputOffset;
            localDestination.addr += ctx.firstBytes;
            localDestination.token = ctx.outputToken;
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDestination, localSource,
                ctx.secondBytes, ctx.localCopyEvent, 2U));
        }
    }

    if (ctx.arg->latencyPipeline) {
        // latency专用Engine上下文只承载512KB，secondBytes恒为0。
        CCU_RETURN_IF_ERROR(
            LatencyCompleteAndPostSync(ctx.arg, ctx.remoteEvent));
        if (ctx.arg->handleLocalCopy) {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.localCopyEvent, 1U));
        }
    } else {
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, remoteMask));
        CCU_IF(ctx.secondBytes != 0U)
        {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, secondRemoteMask));
        }
        if (ctx.arg->handleLocalCopy) {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.localCopyEvent, 1U));
            CCU_IF(ctx.secondBytes != 0U)
            {
                CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.localCopyEvent, 2U));
            }
        }
        CCU_RETURN_IF_ERROR(FullPostSync(ctx.arg));
    }
    return CCU_SUCCESS;
}

CcuResult CcuWideRotatingDirectPushKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuRotatingPushKernelArg *>(arg);
    if (kernelArg == nullptr
        || !ValidateStaticPeerMap(kernelArg->rankSize, kernelArg->rankId,
            kernelArg->channelCount, kernelArg->peerRanks)
        || kernelArg->channelCount > 7U
        || (kernelArg->reuseNetworkEventForLocalCopy
            && (!kernelArg->handleLocalCopy || !kernelArg->combineSliceWaits))) {
        HCCL_ERROR("[CcuWideRotatingDirectPushKernel] Invalid static peer map");
        return CCU_E_PARA;
    }

    RotatingPushContext ctx{};
    ctx.arg = kernelArg;
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        ctx.remoteOutputs[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], ADDR_XN_ID);
        ctx.remoteTokens[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
    }

    // [0]input [1]output [2]inputToken [3]outputToken [4]rankOffset
    // [5]firstBytes [6]secondBytes，共7个动态参数。
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    if (ctx.arg->useStaticGeometry) {
        ctx.rankOutputOffset = ctx.arg->staticRankOutputOffset;
        ctx.firstBytes = ctx.arg->staticFirstBytes;
        ctx.secondBytes = ctx.arg->staticSecondBytes;
    } else {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.rankOutputOffset, argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.firstBytes, argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.secondBytes, argId++));
    }

    if (ctx.arg->exchangeResources) {
        if (ctx.arg->progressiveAddressReady) {
            // 先向全部Peer发布资源；数据面按Channel逐个等待，避免最慢Peer
            // 阻塞已经完成地址交换的链路。
            CCU_RETURN_IF_ERROR(
                PublishOutputResources(ctx.arg, ctx.output, ctx.outputToken));
        } else {
            CCU_RETURN_IF_ERROR(ExchangeOutputResources(
                ctx.arg, ctx.output, ctx.outputToken, ctx.remoteOutputs, ctx.remoteTokens));
        }
    }

    uint16_t remoteMask = 0U;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        if (ctx.arg->exchangeResources && ctx.arg->progressiveAddressReady) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                ctx.arg->channels[i], CKE_INDEX, ADDRESS_READY_MASK));
        }
        ccu::LocalAddr source;
        source.addr = ctx.input;
        source.token = ctx.inputToken;
        ccu::RemoteAddr destination;
        destination.addr = ctx.remoteOutputs[i];
        destination.addr += ctx.rankOutputOffset;
        destination.token = ctx.remoteTokens[i];
        const uint16_t eventBit = static_cast<uint16_t>(1U << i);
        remoteMask = static_cast<uint16_t>(remoteMask | eventBit);
        CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[i], destination, source,
            ctx.firstBytes, ctx.remoteEvent, eventBit));
    }

    uint16_t localCopyMask = 0U;
    if (ctx.arg->handleLocalCopy) {
        ccu::LocalAddr localSource;
        localSource.addr = ctx.input;
        localSource.token = ctx.inputToken;
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.output;
        localDestination.addr += ctx.rankOutputOffset;
        localDestination.token = ctx.outputToken;
        const uint16_t localCopyBit = ctx.arg->reuseNetworkEventForLocalCopy
            ? static_cast<uint16_t>(1U << 14U) : static_cast<uint16_t>(1U);
        localCopyMask = static_cast<uint16_t>(localCopyMask | localCopyBit);
        if (ctx.arg->reuseNetworkEventForLocalCopy) {
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDestination, localSource,
                ctx.firstBytes, ctx.remoteEvent, localCopyBit));
        } else {
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDestination, localSource,
                ctx.firstBytes, ctx.localCopyEvent, localCopyBit));
        }
    }

    // 第二条带把Channel提交顺序循环移动一位，避免两个大条带都由同一物理链路先发。
    uint16_t secondRemoteMask = 0U;
    CCU_IF(ctx.secondBytes != 0U)
    {
        for (uint32_t issueIndex = 0; issueIndex < ctx.arg->channelCount; ++issueIndex) {
            const uint32_t channelIndex = (issueIndex + 1U) % ctx.arg->channelCount;
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.addr += ctx.firstBytes;
            source.token = ctx.inputToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.rankOutputOffset;
            destination.addr += ctx.firstBytes;
            destination.token = ctx.remoteTokens[channelIndex];
            const uint16_t eventBit =
                static_cast<uint16_t>(1U << (ctx.arg->channelCount + issueIndex));
            secondRemoteMask = static_cast<uint16_t>(secondRemoteMask | eventBit);
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                ctx.secondBytes, ctx.remoteEvent, eventBit));
        }
        if (ctx.arg->handleLocalCopy) {
            ccu::LocalAddr localSource;
            localSource.addr = ctx.input;
            localSource.addr += ctx.firstBytes;
            localSource.token = ctx.inputToken;
            ccu::LocalAddr localDestination;
            localDestination.addr = ctx.output;
            localDestination.addr += ctx.rankOutputOffset;
            localDestination.addr += ctx.firstBytes;
            localDestination.token = ctx.outputToken;
            const uint16_t localCopyBit = ctx.arg->reuseNetworkEventForLocalCopy
                ? static_cast<uint16_t>(1U << 15U) : static_cast<uint16_t>(2U);
            localCopyMask = static_cast<uint16_t>(localCopyMask | localCopyBit);
            if (ctx.arg->reuseNetworkEventForLocalCopy) {
                CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDestination, localSource,
                    ctx.secondBytes, ctx.remoteEvent, localCopyBit));
            } else {
                CCU_RETURN_IF_ERROR(ccu::LocalCopy(localDestination, localSource,
                    ctx.secondBytes, ctx.localCopyEvent, localCopyBit));
            }
        }
    }

    if (ctx.arg->combineSliceWaits) {
        CCU_IF(ctx.secondBytes != 0U)
        {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent,
                static_cast<uint16_t>(remoteMask | secondRemoteMask)));
        }
        CCU_ELSE
        {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, remoteMask));
        }
    } else {
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, remoteMask));
        CCU_IF(ctx.secondBytes != 0U)
        {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, secondRemoteMask));
        }
    }
    if (ctx.arg->finalSync && ctx.arg->reuseNetworkEventForLocalCopy) {
        CCU_RETURN_IF_ERROR(FullPostSync(ctx.arg));
    }
    if (ctx.arg->handleLocalCopy) {
        if (ctx.arg->reuseNetworkEventForLocalCopy) {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, localCopyMask));
        } else {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.localCopyEvent, 1U));
            CCU_IF(ctx.secondBytes != 0U)
            {
                CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.localCopyEvent, 2U));
            }
        }
    }
    if (ctx.arg->finalSync && !ctx.arg->reuseNetworkEventForLocalCopy) {
        CCU_RETURN_IF_ERROR(FullPostSync(ctx.arg));
    }
    return CCU_SUCCESS;
}

CcuResult CcuLatencyDirectPushKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuLatencyPushKernelArg *>(arg);
    if (kernelArg == nullptr
        || !ValidateStaticPeerMap(kernelArg->rankSize, kernelArg->rankId,
            kernelArg->channelCount, kernelArg->peerRanks)
        || kernelArg->channelCount > MAX_CHANNELS_PER_DIE) {
        HCCL_ERROR("[CcuLatencyDirectPushKernel] Invalid latency peer map");
        return CCU_E_PARA;
    }

    LatencyPushContext ctx{};
    ctx.arg = kernelArg;
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        ctx.remoteOutputs[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], ADDR_XN_ID);
        ctx.remoteTokens[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
    }

    // [0]input [1]output [2]inputToken [3]outputToken
    // [4]rankOutputOffset [5]transferBytes，共6个动态参数。
    uint32_t argId = 0U;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.rankOutputOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.transferBytes, argId++));

    CCU_RETURN_IF_ERROR(
        PublishOutputResources(ctx.arg, ctx.output, ctx.outputToken));

    // 与V21测试点10保持同一关键顺序：先让所有网络Write进入两个IO Die
    // 的数据面，再提交不与网络目标重叠的OwnCopy。
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            ctx.arg->channels[i], CKE_INDEX, ADDRESS_READY_MASK));
        ccu::LocalAddr source;
        source.addr = ctx.input;
        source.token = ctx.inputToken;
        ccu::RemoteAddr destination;
        destination.addr = ctx.remoteOutputs[i];
        destination.addr += ctx.rankOutputOffset;
        destination.token = ctx.remoteTokens[i];
        const uint16_t eventBit = static_cast<uint16_t>(1U << i);
        CCU_RETURN_IF_ERROR(ccu::Write(
            ctx.arg->channels[i], destination, source,
            ctx.transferBytes, ctx.remoteEvent, eventBit));
    }

    if (ctx.arg->handleLocalCopy) {
        ccu::LocalAddr source;
        source.addr = ctx.input;
        source.token = ctx.inputToken;
        ccu::LocalAddr destination;
        destination.addr = ctx.output;
        destination.addr += ctx.rankOutputOffset;
        destination.token = ctx.outputToken;
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(
            destination, source, ctx.transferBytes, ctx.localCopyEvent, 1U));
    }

    // 512KB冲刺路径：Write事件已经覆盖所有远端DMA完成。
    // 不再为每个Peer追加Record+Wait往返，直接等待组合数据事件。
    const uint16_t remoteMask = static_cast<uint16_t>(
        (1U << ctx.arg->channelCount) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, remoteMask));
    if (ctx.arg->handleLocalCopy) {
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.localCopyEvent, 1U));
    }
    return CCU_SUCCESS;
}


CcuResult CcuRegistered4x1DirectPushKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuRegistered4x1PushKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize != 4U
        || kernelArg->rankId >= 4U || kernelArg->channelCount != 3U
        || kernelArg->transferBytes != 512ULL * 1024ULL
        || !ValidateStaticPeerMap(kernelArg->rankSize, kernelArg->rankId,
            kernelArg->channelCount, kernelArg->peerRanks)) {
        HCCL_ERROR("[CcuRegistered4x1DirectPushKernel] Invalid registered 4x1 map");
        return CCU_E_PARA;
    }

    ccu::Variable remoteOutputs[3];
    ccu::Variable remoteTokens[3];
    for (uint32_t i = 0U; i < 3U; ++i) {
        remoteOutputs[i] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], ADDR_XN_ID);
        remoteTokens[i] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], TOKEN_XN_ID);
    }

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankOutputOffset;
    ccu::Variable transferBytes;
    input = kernelArg->inputAddr;
    inputToken = kernelArg->inputToken;
    transferBytes = kernelArg->transferBytes;
    if (kernelArg->publishResources || kernelArg->handleLocalCopy) {
        output = kernelArg->outputAddr;
        outputToken = kernelArg->outputToken;
    }
    if (kernelArg->handleLocalCopy) {
        rankOutputOffset = kernelArg->rankOutputOffset;
    }

    if (kernelArg->publishResources) {
        CCU_RETURN_IF_ERROR(PublishOutputResources(kernelArg, output, outputToken));
        for (uint32_t i = 0U; i < 3U; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                kernelArg->channels[i], CKE_INDEX, ADDRESS_READY_MASK));
        }
    }

    ccu::Event transferEvent;
    for (uint32_t i = 0U; i < 3U; ++i) {
        ccu::LocalAddr source;
        source.addr = input;
        source.token = inputToken;
        ccu::RemoteAddr destination;
        destination.addr = remoteOutputs[i];
        destination.addr += rankOutputOffset;
        destination.token = remoteTokens[i];
        const uint16_t eventBit = static_cast<uint16_t>(1U << i);
        CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[i], destination,
            source, transferBytes, transferEvent, eventBit));
    }

    uint16_t completionMask = 0x7U;
    if (kernelArg->handleLocalCopy) {
        ccu::LocalAddr source;
        source.addr = input;
        source.token = inputToken;
        ccu::LocalAddr destination;
        destination.addr = output;
        destination.addr += rankOutputOffset;
        destination.token = outputToken;
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(destination, source,
            transferBytes, transferEvent, 0x8U));
        completionMask = 0xFU;
    }

    // The V27 path already proved that the Write completion events are a
    // sufficient return boundary for this scored payload.  The registered
    // graph removes six LoadArg tasks and, on warm launches, the six resource
    // publication tasks plus three address-ready waits.
    CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, completionMask));
    return CCU_SUCCESS;
}

CcuResult CcuOutput512StaticDirectPushKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuOutput512StaticPushKernelArg *>(arg);
    if (kernelArg == nullptr
        || (kernelArg->rankSize != 4U && kernelArg->rankSize != 12U
            && kernelArg->rankSize != 16U)
        || kernelArg->rankId >= kernelArg->rankSize
        || kernelArg->transferBytes == 0U
        || kernelArg->channelCount == 0U
        || kernelArg->channelCount > MAX_CHANNELS_PER_DIE
        || !ValidateStaticPeerMap(kernelArg->rankSize, kernelArg->rankId,
            kernelArg->channelCount, kernelArg->peerRanks)) {
        HCCL_ERROR("[CcuOutput512StaticDirectPushKernel] Invalid resident map");
        return CCU_E_PARA;
    }
    const uint64_t outputBytes =
        kernelArg->transferBytes * kernelArg->rankSize;
    constexpr uint64_t targetBytes = 512ULL * 1024ULL;
    const uint64_t delta = outputBytes >= targetBytes
        ? outputBytes - targetBytes : targetBytes - outputBytes;
    if (delta >= static_cast<uint64_t>(kernelArg->rankSize) * sizeof(float)) {
        HCCL_ERROR("[CcuOutput512StaticDirectPushKernel] Invalid output geometry");
        return CCU_E_PARA;
    }

    ccu::Variable remoteOutputs[MAX_CHANNELS_PER_DIE];
    ccu::Variable remoteTokens[MAX_CHANNELS_PER_DIE];
    for (uint32_t i = 0U; i < kernelArg->channelCount; ++i) {
        remoteOutputs[i] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[i], ADDR_XN_ID);
        remoteTokens[i] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[i], TOKEN_XN_ID);
    }

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankOutputOffset;
    ccu::Variable transferBytes;
    input = kernelArg->inputAddr;
    inputToken = kernelArg->inputToken;
    transferBytes = kernelArg->transferBytes;
    if (kernelArg->publishResources || kernelArg->handleLocalCopy) {
        output = kernelArg->outputAddr;
        outputToken = kernelArg->outputToken;
    }
    if (kernelArg->handleLocalCopy) {
        rankOutputOffset = kernelArg->rankOutputOffset;
    }

    if (kernelArg->publishResources) {
        // Publish the slot expected by the peer on this channel rather than
        // the raw recv base.  The warm graph can therefore issue every Write
        // without a destination-offset instruction.
        for (uint32_t i = 0U; i < kernelArg->channelCount; ++i) {
            ccu::Variable peerSlot;
            peerSlot = kernelArg->outputAddr
                + static_cast<uint64_t>(kernelArg->peerRanks[i])
                    * kernelArg->transferBytes;
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                kernelArg->channels[i], peerSlot, ADDR_XN_ID,
                CKE_INDEX, ADDR_MASK));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                kernelArg->channels[i], outputToken, TOKEN_XN_ID,
                CKE_INDEX, TOKEN_MASK));
        }
        for (uint32_t i = 0U; i < kernelArg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                kernelArg->channels[i], CKE_INDEX, ADDRESS_READY_MASK));
        }
    }

    ccu::Event transferEvent;
    uint16_t remoteMask = 0U;
    ccu::LocalAddr sharedSource;
    sharedSource.addr = input;
    sharedSource.token = inputToken;
    for (uint32_t i = 0U; i < kernelArg->channelCount; ++i) {
        ccu::RemoteAddr destination;
        destination.addr = remoteOutputs[i];
        destination.token = remoteTokens[i];
        const uint16_t eventBit = static_cast<uint16_t>(1U << i);
        remoteMask = static_cast<uint16_t>(remoteMask | eventBit);
        CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[i], destination,
            sharedSource, transferBytes, transferEvent, eventBit));
    }

    uint16_t completionMask = remoteMask;
    if (kernelArg->handleLocalCopy) {
        ccu::LocalAddr destination;
        destination.addr = output;
        destination.addr += rankOutputOffset;
        destination.token = outputToken;
        const uint16_t localCopyBit =
            static_cast<uint16_t>(1U << kernelArg->channelCount);
        completionMask =
            static_cast<uint16_t>(completionMask | localCopyBit);
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(destination, sharedSource,
            transferBytes, transferEvent, localCopyBit));
    }

    CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, completionMask));
    return CCU_SUCCESS;
}

CcuResult CcuClos4x1DirectPushKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuClos4x1PushKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize != 4U
        || kernelArg->channelCount != CLOS_4X1_CHANNEL_NUM
        || (kernelArg->chunkCount != 1U && kernelArg->chunkCount != CLOS_4X1_MAX_CHUNKS)
        || !ValidateStaticPeerMap(kernelArg->rankSize, kernelArg->rankId,
            kernelArg->channelCount, kernelArg->peerRanks)) {
        HCCL_ERROR("[CcuClos4x1DirectPushKernel] Invalid common-Die peer map");
        return CCU_E_PARA;
    }

    Clos4x1PushContext ctx{};
    ctx.arg = kernelArg;
    for (uint32_t channelIndex = 0; channelIndex < CLOS_4X1_CHANNEL_NUM; ++channelIndex) {
        ctx.remoteOutputs[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], ADDR_XN_ID);
        ctx.remoteTokens[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], TOKEN_XN_ID);
    }

    // [0]input [1]output [2]inputToken [3]outputToken [4]rankOffset
    // [5..8]四个Chunk字节数。latency静态实例只加载[5]，共6项；
    // bandwidth静态实例加载全部9项，均低于13项上限。
    uint32_t argId = 0U;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.rankOutputOffset, argId++));
    for (uint32_t chunkIndex = 0; chunkIndex < ctx.arg->chunkCount; ++chunkIndex) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.chunkBytes[chunkIndex], argId++));
    }

    // 先一次广播输出基址/Token。随后每个Peer地址一到即提交首Chunk，
    // 不再等待三条Channel全部完成地址交换后才启动数据面。
    for (uint32_t channelIndex = 0; channelIndex < CLOS_4X1_CHANNEL_NUM; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            ctx.arg->channels[channelIndex], ctx.output, ADDR_XN_ID, CKE_INDEX, ADDR_MASK));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            ctx.arg->channels[channelIndex], ctx.outputToken, TOKEN_XN_ID, CKE_INDEX, TOKEN_MASK));
    }

    uint16_t localCopyMask = 0U;
    if (ctx.arg->handleLocalCopy) {
        for (uint32_t chunkIndex = 0; chunkIndex < ctx.arg->chunkCount; ++chunkIndex) {
            ccu::LocalAddr source;
            source.addr = ctx.input;
            ccu::LocalAddr destination;
            destination.addr = ctx.output;
            destination.addr += ctx.rankOutputOffset;
            for (uint32_t priorChunk = 0; priorChunk < chunkIndex; ++priorChunk) {
                source.addr += ctx.chunkBytes[priorChunk];
                destination.addr += ctx.chunkBytes[priorChunk];
            }
            source.token = ctx.inputToken;
            destination.token = ctx.outputToken;
            const uint16_t eventBit = static_cast<uint16_t>(1U << chunkIndex);
            localCopyMask = static_cast<uint16_t>(localCopyMask | eventBit);
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(destination, source,
                ctx.chunkBytes[chunkIndex], ctx.localCopyEvent, eventBit));
        }
    }

    uint16_t networkMask = 0U;
    for (uint32_t channelIndex = 0; channelIndex < CLOS_4X1_CHANNEL_NUM; ++channelIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            ctx.arg->channels[channelIndex], CKE_INDEX, ADDRESS_READY_MASK));
        ccu::LocalAddr source;
        source.addr = ctx.input;
        source.token = ctx.inputToken;
        ccu::RemoteAddr destination;
        destination.addr = ctx.remoteOutputs[channelIndex];
        destination.addr += ctx.rankOutputOffset;
        destination.token = ctx.remoteTokens[channelIndex];
        const uint16_t eventBit = static_cast<uint16_t>(1U << channelIndex);
        networkMask = static_cast<uint16_t>(networkMask | eventBit);
        CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
            ctx.chunkBytes[0], ctx.networkEvent, eventBit));
    }

    // 大消息静态实例使用4个粗粒度Chunk。每个Chunk轮换首发Channel，
    // 3 peer × 4 chunk正好使用12个Event位，最后仅做一次组合EventWait。
    for (uint32_t chunkIndex = 1U; chunkIndex < ctx.arg->chunkCount; ++chunkIndex) {
        for (uint32_t issueIndex = 0; issueIndex < CLOS_4X1_CHANNEL_NUM; ++issueIndex) {
            const uint32_t channelIndex =
                (issueIndex + chunkIndex) % CLOS_4X1_CHANNEL_NUM;
            ccu::LocalAddr source;
            source.addr = ctx.input;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.rankOutputOffset;
            for (uint32_t priorChunk = 0; priorChunk < chunkIndex; ++priorChunk) {
                source.addr += ctx.chunkBytes[priorChunk];
                destination.addr += ctx.chunkBytes[priorChunk];
            }
            source.token = ctx.inputToken;
            destination.token = ctx.remoteTokens[channelIndex];
            const uint32_t eventIndex =
                chunkIndex * CLOS_4X1_CHANNEL_NUM + issueIndex;
            const uint16_t eventBit = static_cast<uint16_t>(1U << eventIndex);
            networkMask = static_cast<uint16_t>(networkMask | eventBit);
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                ctx.chunkBytes[chunkIndex], ctx.networkEvent, eventBit));
        }
    }

    if (ctx.arg->latencyPipeline) {
        // 4x1 512KB冲刺路径：仅等待三个网络Write事件，省去逐Peer尾部握手。
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.networkEvent, networkMask));
    } else {
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.networkEvent, networkMask));
        CCU_RETURN_IF_ERROR(FullPostSync(ctx.arg));
    }
    if (ctx.arg->handleLocalCopy) {
        // OwnCopy写入本Rank块，与三路网络写无重叠；放到PostSync之后再等待，
        // 让其完成时间与全局生命周期确认重叠。
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.localCopyEvent, localCopyMask));
    }
    return CCU_SUCCESS;
}

CcuResult CcuDualSeedCrossKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuDualSeedCrossKernelArg *>(arg);
    const uint32_t expectedChannels =
        kernelArg != nullptr && kernelArg->sourceIsSmallServer ? 8U : 4U;
    if (kernelArg == nullptr || kernelArg->rankSize != 12U
        || !ValidateStaticPeerMap(kernelArg->rankSize, kernelArg->rankId,
            kernelArg->channelCount, kernelArg->peerRanks)
        || kernelArg->channelCount != expectedChannels) {
        HCCL_ERROR("[CcuDualSeedCrossKernel] Invalid asymmetric cross map");
        return CCU_E_PARA;
    }

    const uint32_t seed0Channel = FindPeerChannel(
        kernelArg->peerRanks, kernelArg->channelCount, kernelArg->seedRanks[0]);
    const uint32_t seed1Channel = kernelArg->sourceIsSmallServer
        ? FindPeerChannel(kernelArg->peerRanks, kernelArg->channelCount, kernelArg->seedRanks[1])
        : seed0Channel;
    if (seed0Channel >= kernelArg->channelCount || seed1Channel >= kernelArg->channelCount
        || (kernelArg->sourceIsSmallServer && seed0Channel == seed1Channel)) {
        HCCL_ERROR("[CcuDualSeedCrossKernel] Seed rank is absent or duplicated");
        return CCU_E_PARA;
    }

    DualSeedCrossContext ctx{};
    ctx.arg = kernelArg;
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        ctx.remoteOutputs[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], ADDR_XN_ID);
        ctx.remoteTokens[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
    }

    // 共11个动态参数；最后一项phaseMode由同一Kernel句柄选择Seed-only或Direct-only。
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.rankOutputOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallDirectBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallStripe0Bytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallStripe1Bytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.largeDirectBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.largeTailBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.phaseMode, argId++));

    if (ctx.arg->latencyPipeline) {
        CCU_RETURN_IF_ERROR(
            PublishOutputResources(ctx.arg, ctx.output, ctx.outputToken));
    } else {
        CCU_RETURN_IF_ERROR(ExchangeOutputResources(
            ctx.arg, ctx.output, ctx.outputToken, ctx.remoteOutputs, ctx.remoteTokens));
    }

    // phaseMode=0：仅下发Seed条带；两个Write先全部提交，再一次组合EventWait。
    CCU_IF(ctx.phaseMode != 1U)
    {
        uint16_t phaseMask = 0U;
        if (ctx.arg->sourceIsSmallServer) {
            if (ctx.arg->latencyPipeline) {
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                    ctx.arg->channels[seed0Channel], CKE_INDEX, ADDRESS_READY_MASK));
            }
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.token = ctx.inputToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[seed0Channel];
            destination.addr += ctx.rankOutputOffset;
            destination.token = ctx.remoteTokens[seed0Channel];
            constexpr uint16_t stripe0Bit = 1U;
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[seed0Channel], destination, source,
                ctx.smallStripe0Bytes, ctx.remoteEvent, stripe0Bit));
            phaseMask = static_cast<uint16_t>(phaseMask | stripe0Bit);

            if (ctx.arg->latencyPipeline) {
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                    ctx.arg->channels[seed1Channel], CKE_INDEX, ADDRESS_READY_MASK));
            }
            ccu::LocalAddr secondSource;
            secondSource.addr = ctx.input;
            secondSource.addr += ctx.smallStripe0Bytes;
            secondSource.token = ctx.inputToken;
            ccu::RemoteAddr secondDestination;
            secondDestination.addr = ctx.remoteOutputs[seed1Channel];
            secondDestination.addr += ctx.rankOutputOffset;
            secondDestination.addr += ctx.smallStripe0Bytes;
            secondDestination.token = ctx.remoteTokens[seed1Channel];
            constexpr uint16_t stripe1Bit = 2U;
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[seed1Channel],
                secondDestination, secondSource,
                ctx.smallStripe1Bytes, ctx.remoteEvent, stripe1Bit));
            phaseMask = static_cast<uint16_t>(phaseMask | stripe1Bit);
        } else {
            if (ctx.arg->latencyPipeline) {
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                    ctx.arg->channels[seed0Channel], CKE_INDEX, ADDRESS_READY_MASK));
            }
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.token = ctx.inputToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[seed0Channel];
            destination.addr += ctx.rankOutputOffset;
            destination.token = ctx.remoteTokens[seed0Channel];
            constexpr uint16_t largeTailBit = 1U;
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[seed0Channel], destination, source,
                ctx.largeTailBytes, ctx.remoteEvent, largeTailBit));
            phaseMask = static_cast<uint16_t>(phaseMask | largeTailBit);
        }
        if (!ctx.arg->latencyPipeline) {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, phaseMask));
        }
    }

    // phaseMode=1：仅下发direct后缀；该Launch与机内Relay并行。
    CCU_IF(ctx.phaseMode != 0U)
    {
        uint16_t phaseMask = 0U;
        ccu::Variable directOffset;
        ccu::Variable *directBytes = nullptr;
        if (ctx.arg->sourceIsSmallServer) {
            directOffset = ctx.smallStripe0Bytes;
            directOffset += ctx.smallStripe1Bytes;
            directBytes = &ctx.smallDirectBytes;
        } else {
            directOffset = ctx.largeTailBytes;
            directBytes = &ctx.largeDirectBytes;
        }
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            if (ctx.arg->latencyPipeline) {
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                    ctx.arg->channels[i], CKE_INDEX, ADDRESS_READY_MASK));
            }
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.addr += directOffset;
            source.token = ctx.inputToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[i];
            destination.addr += ctx.rankOutputOffset;
            destination.addr += directOffset;
            destination.token = ctx.remoteTokens[i];
            const uint16_t eventBit = static_cast<uint16_t>(1U << i);
            phaseMask = static_cast<uint16_t>(phaseMask | eventBit);
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[i], destination, source,
                *directBytes, ctx.remoteEvent, eventBit));
        }
        if (!ctx.arg->latencyPipeline) {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, phaseMask));
        }
    }

    // Seed-only的PostSync完成后Host记录READY；Direct-only随后与Relay并行。
    if (ctx.arg->latencyPipeline) {
        // latency上下文只执行Direct-only且覆盖全部Cross Channel。
        CCU_RETURN_IF_ERROR(
            LatencyCompleteAndPostSync(ctx.arg, ctx.remoteEvent));
    } else {
        CCU_RETURN_IF_ERROR(FullPostSync(ctx.arg));
    }
    return CCU_SUCCESS;
}

CcuResult CcuWideDualSeedCrossKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuWideDualSeedCrossKernelArg *>(arg);
    const uint32_t expectedChannels =
        kernelArg != nullptr && kernelArg->sourceIsSmallServer ? 8U : 4U;
    const uint32_t expectedIncomingSeeds =
        kernelArg != nullptr && kernelArg->sourceIsSmallServer ? 2U : 1U;
    if (kernelArg == nullptr || kernelArg->rankSize != 12U
        || !ValidateStaticPeerMap(kernelArg->rankSize, kernelArg->rankId,
            kernelArg->channelCount, kernelArg->peerRanks)
        || kernelArg->channelCount != expectedChannels
        || kernelArg->incomingSeedRankCount != expectedIncomingSeeds) {
        HCCL_ERROR("[CcuWideDualSeedCrossKernel] Invalid asymmetric cross map");
        return CCU_E_PARA;
    }

    const uint32_t seed0Channel = FindPeerChannel(
        kernelArg->peerRanks, kernelArg->channelCount, kernelArg->seedRanks[0]);
    const uint32_t seed1Channel = kernelArg->sourceIsSmallServer
        ? FindPeerChannel(kernelArg->peerRanks, kernelArg->channelCount, kernelArg->seedRanks[1])
        : seed0Channel;
    if (seed0Channel >= kernelArg->channelCount || seed1Channel >= kernelArg->channelCount
        || (kernelArg->sourceIsSmallServer && seed0Channel == seed1Channel)) {
        HCCL_ERROR("[CcuWideDualSeedCrossKernel] Invalid outbound Seed channels");
        return CCU_E_PARA;
    }

    uint32_t incomingSeedChannels[2]{};
    for (uint32_t index = 0; index < kernelArg->incomingSeedRankCount; ++index) {
        incomingSeedChannels[index] = FindPeerChannel(
            kernelArg->peerRanks, kernelArg->channelCount, kernelArg->incomingSeedRanks[index]);
        if (incomingSeedChannels[index] >= kernelArg->channelCount
            || (index != 0U && incomingSeedChannels[index] == incomingSeedChannels[0])) {
            HCCL_ERROR("[CcuWideDualSeedCrossKernel] Invalid incoming Seed channels");
            return CCU_E_PARA;
        }
    }

    WideDualSeedCrossContext ctx{};
    ctx.arg = kernelArg;
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        ctx.remoteOutputs[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], ADDR_XN_ID);
        ctx.remoteTokens[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
    }

    // [0..4]地址、Token与本Rank输出偏移；[5..11]两侧Direct双切片及Relay前缀。
    uint32_t argId = 0U;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.rankOutputOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallDirectFirstBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallDirectSecondBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallStripe0Bytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallStripe1Bytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.largeDirectFirstBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.largeDirectSecondBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.largeTailBytes, argId++));

    if (ctx.arg->exchangeResources) {
        if (ctx.arg->phaseKind == CcuDualSeedPhaseKind::SEED) {
            // Seed先等待真正使用的1/2条Channel；其余地址交换与Seed DMA重叠，
            // Direct静态Kernel随后复用这些XN资源。
            CCU_RETURN_IF_ERROR(
                PublishOutputResources(ctx.arg, ctx.output, ctx.outputToken));
        } else {
            CCU_RETURN_IF_ERROR(ExchangeOutputResources(
                ctx.arg, ctx.output, ctx.outputToken, ctx.remoteOutputs, ctx.remoteTokens));
        }
    }

    if (ctx.arg->phaseKind == CcuDualSeedPhaseKind::SEED) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            ctx.arg->channels[seed0Channel], CKE_INDEX, ADDRESS_READY_MASK));
        if (ctx.arg->sourceIsSmallServer) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                ctx.arg->channels[seed1Channel], CKE_INDEX, ADDRESS_READY_MASK));
        }

        uint16_t seedEventMask = 0U;
        if (ctx.arg->sourceIsSmallServer) {
            ccu::LocalAddr firstSource;
            firstSource.addr = ctx.input;
            firstSource.token = ctx.inputToken;
            ccu::RemoteAddr firstDestination;
            firstDestination.addr = ctx.remoteOutputs[seed0Channel];
            firstDestination.addr += ctx.rankOutputOffset;
            firstDestination.token = ctx.remoteTokens[seed0Channel];
            constexpr uint16_t firstSeedBit = 1U;
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[seed0Channel],
                firstDestination, firstSource, ctx.smallStripe0Bytes,
                ctx.remoteEvent, firstSeedBit));
            seedEventMask = static_cast<uint16_t>(seedEventMask | firstSeedBit);

            ccu::LocalAddr secondSource;
            secondSource.addr = ctx.input;
            secondSource.addr += ctx.smallStripe0Bytes;
            secondSource.token = ctx.inputToken;
            ccu::RemoteAddr secondDestination;
            secondDestination.addr = ctx.remoteOutputs[seed1Channel];
            secondDestination.addr += ctx.rankOutputOffset;
            secondDestination.addr += ctx.smallStripe0Bytes;
            secondDestination.token = ctx.remoteTokens[seed1Channel];
            constexpr uint16_t secondSeedBit = 2U;
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[seed1Channel],
                secondDestination, secondSource, ctx.smallStripe1Bytes,
                ctx.remoteEvent, secondSeedBit));
            seedEventMask = static_cast<uint16_t>(seedEventMask | secondSeedBit);
        } else {
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.token = ctx.inputToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[seed0Channel];
            destination.addr += ctx.rankOutputOffset;
            destination.token = ctx.remoteTokens[seed0Channel];
            constexpr uint16_t seedBit = 1U;
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[seed0Channel],
                destination, source, ctx.largeTailBytes, ctx.remoteEvent, seedBit));
            seedEventMask = static_cast<uint16_t>(seedEventMask | seedBit);
        }

        // 把未使用Channel的地址交换等待隐藏在Seed数据传输之后。
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (channelIndex == seed0Channel
                || (ctx.arg->sourceIsSmallServer && channelIndex == seed1Channel)) {
                continue;
            }
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                ctx.arg->channels[channelIndex], CKE_INDEX, ADDRESS_READY_MASK));
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, seedEventMask));

        // ID4只承担Seed-ready，绝不与ID3最终屏障复用，避免位通知合并。
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            ctx.arg->channels[seed0Channel], CKE_INDEX, SEED_READY_MASK));
        if (ctx.arg->sourceIsSmallServer) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
                ctx.arg->channels[seed1Channel], CKE_INDEX, SEED_READY_MASK));
        }
        for (uint32_t index = 0; index < ctx.arg->incomingSeedRankCount; ++index) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                ctx.arg->channels[incomingSeedChannels[index]], CKE_INDEX, SEED_READY_MASK));
        }
    } else {
        ccu::Variable directOffset;
        ccu::Variable directFirstBytes;
        ccu::Variable directSecondBytes;
        if (ctx.arg->sourceIsSmallServer) {
            directOffset = ctx.smallStripe0Bytes;
            directOffset += ctx.smallStripe1Bytes;
            directFirstBytes = ctx.smallDirectFirstBytes;
            directSecondBytes = ctx.smallDirectSecondBytes;
        } else {
            directOffset = ctx.largeTailBytes;
            directFirstBytes = ctx.largeDirectFirstBytes;
            directSecondBytes = ctx.largeDirectSecondBytes;
        }

        uint16_t firstMask = 0U;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ccu::LocalAddr source;
            source.addr = ctx.input;
            source.addr += directOffset;
            source.token = ctx.inputToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.rankOutputOffset;
            destination.addr += directOffset;
            destination.token = ctx.remoteTokens[channelIndex];
            const uint16_t eventBit = static_cast<uint16_t>(1U << channelIndex);
            firstMask = static_cast<uint16_t>(firstMask | eventBit);
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[channelIndex],
                destination, source, directFirstBytes, ctx.remoteEvent, eventBit));
        }

        uint16_t secondMask = 0U;
        CCU_IF(directSecondBytes != 0U)
        {
            // 第二切片循环移动一个首发Channel；8 Channel时正好用满16个Event位。
            for (uint32_t issueIndex = 0; issueIndex < ctx.arg->channelCount; ++issueIndex) {
                const uint32_t channelIndex =
                    (issueIndex + 1U) % ctx.arg->channelCount;
                ccu::LocalAddr source;
                source.addr = ctx.input;
                source.addr += directOffset;
                source.addr += directFirstBytes;
                source.token = ctx.inputToken;
                ccu::RemoteAddr destination;
                destination.addr = ctx.remoteOutputs[channelIndex];
                destination.addr += ctx.rankOutputOffset;
                destination.addr += directOffset;
                destination.addr += directFirstBytes;
                destination.token = ctx.remoteTokens[channelIndex];
                const uint16_t eventBit = static_cast<uint16_t>(
                    1U << (ctx.arg->channelCount + issueIndex));
                secondMask = static_cast<uint16_t>(secondMask | eventBit);
                CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[channelIndex],
                    destination, source, directSecondBytes, ctx.remoteEvent, eventBit));
            }
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, firstMask));
        CCU_IF(directSecondBytes != 0U)
        {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent, secondMask));
        }
    }

    if (ctx.arg->finalSync) {
        CCU_RETURN_IF_ERROR(FullPostSync(ctx.arg));
    }
    return CCU_SUCCESS;
}

CcuResult CcuDualSeedRelayKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuDualSeedRelayKernelArg *>(arg);
    const uint32_t expectedChannels =
        kernelArg != nullptr && kernelArg->relayOnLargeServer ? 7U : 3U;
    if (kernelArg == nullptr || !kernelArg->enabled || kernelArg->rankSize != 12U
        || !ValidateStaticPeerMap(kernelArg->rankSize, kernelArg->rankId,
            kernelArg->channelCount, kernelArg->peerRanks)
        || kernelArg->channelCount != expectedChannels
        || kernelArg->sourceRankCount == 0U || kernelArg->sourceRankCount > 2U) {
        HCCL_ERROR("[CcuDualSeedRelayKernel] Invalid relay map");
        return CCU_E_PARA;
    }

    DualSeedRelayContext ctx{};
    ctx.arg = kernelArg;
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        ctx.remoteOutputs[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], ADDR_XN_ID);
        ctx.remoteTokens[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
    }

    // [0]output [1]outputToken [2..3]两个静态源对应的输出块偏移，
    // [4..6]小源分区，[7..8]大源分区，共9项。
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sourceBlockOffsets[0], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sourceBlockOffsets[1], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallDirectBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallStripe0Bytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallStripe1Bytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.largeDirectBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.largeTailBytes, argId++));

    uint16_t relayMask = 0U;
    uint32_t operationIndex = 0U;
    for (uint32_t sourceIndex = 0; sourceIndex < ctx.arg->sourceRankCount; ++sourceIndex) {
        const uint32_t sourceRank = ctx.arg->sourceRanks[sourceIndex];
        if (sourceRank >= ctx.arg->rankSize) {
            return CCU_E_PARA;
        }

        ccu::Variable stripeOffset;
        ccu::Variable *stripeSize = nullptr;
        if (ctx.arg->relayOnLargeServer) {
            if (sourceRank < 8U || ctx.arg->stripeIndices[sourceIndex] > 1U) {
                return CCU_E_PARA;
            }
            if (ctx.arg->stripeIndices[sourceIndex] == 0U) {
                stripeOffset = 0U;
                stripeSize = &ctx.smallStripe0Bytes;
            } else {
                stripeOffset = ctx.smallStripe0Bytes;
                stripeSize = &ctx.smallStripe1Bytes;
            }
        } else {
            if (sourceRank >= 8U) {
                return CCU_E_PARA;
            }
            stripeOffset = 0U;
            stripeSize = &ctx.largeTailBytes;
        }

        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ccu::LocalAddr source;
            source.addr = ctx.output;
            source.addr += ctx.sourceBlockOffsets[sourceIndex];
            source.addr += stripeOffset;
            source.token = ctx.outputToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.sourceBlockOffsets[sourceIndex];
            destination.addr += stripeOffset;
            destination.token = ctx.remoteTokens[channelIndex];
            const uint16_t eventBit = static_cast<uint16_t>(1U << operationIndex);
            ++operationIndex;
            relayMask = static_cast<uint16_t>(relayMask | eventBit);
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                *stripeSize, ctx.relayEvent, eventBit));
        }
    }
    if (operationIndex > 7U) {
        return CCU_E_PARA;
    }
    CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.relayEvent, relayMask));
    CCU_RETURN_IF_ERROR(FullPostSync(ctx.arg));
    return CCU_SUCCESS;
}

CcuResult CcuWideDualSeedRelayKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuDualSeedRelayKernelArg *>(arg);
    const uint32_t expectedChannels =
        kernelArg != nullptr && kernelArg->relayOnLargeServer ? 7U : 3U;
    if (kernelArg == nullptr || !kernelArg->enabled || kernelArg->rankSize != 12U
        || !ValidateStaticPeerMap(kernelArg->rankSize, kernelArg->rankId,
            kernelArg->channelCount, kernelArg->peerRanks)
        || kernelArg->channelCount != expectedChannels
        || kernelArg->sourceRankCount == 0U || kernelArg->sourceRankCount > 2U) {
        HCCL_ERROR("[CcuWideDualSeedRelayKernel] Invalid relay map");
        return CCU_E_PARA;
    }

    DualSeedRelayContext ctx{};
    ctx.arg = kernelArg;
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        ctx.remoteOutputs[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], ADDR_XN_ID);
        ctx.remoteTokens[i] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
    }

    // [0]output [1]outputToken [2..3]两个静态源对应的输出块偏移，
    // [4..6]小源分区，[7..8]大源分区，共9项。
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sourceBlockOffsets[0], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sourceBlockOffsets[1], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallDirectBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallStripe0Bytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.smallStripe1Bytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.largeDirectBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.largeTailBytes, argId++));

    uint16_t relayMask = 0U;
    uint32_t operationIndex = 0U;
    for (uint32_t sourceIndex = 0; sourceIndex < ctx.arg->sourceRankCount; ++sourceIndex) {
        const uint32_t sourceRank = ctx.arg->sourceRanks[sourceIndex];
        if (sourceRank >= ctx.arg->rankSize) {
            return CCU_E_PARA;
        }

        ccu::Variable stripeOffset;
        ccu::Variable *stripeSize = nullptr;
        if (ctx.arg->relayOnLargeServer) {
            if (sourceRank < 8U || ctx.arg->stripeIndices[sourceIndex] > 1U) {
                return CCU_E_PARA;
            }
            if (ctx.arg->stripeIndices[sourceIndex] == 0U) {
                stripeOffset = 0U;
                stripeSize = &ctx.smallStripe0Bytes;
            } else {
                stripeOffset = ctx.smallStripe0Bytes;
                stripeSize = &ctx.smallStripe1Bytes;
            }
        } else {
            if (sourceRank >= 8U) {
                return CCU_E_PARA;
            }
            stripeOffset = 0U;
            stripeSize = &ctx.largeTailBytes;
        }

        for (uint32_t issueIndex = 0; issueIndex < ctx.arg->channelCount; ++issueIndex) {
            // 小Server会连续Relay两个大Server源块；第二个源循环移动首发
            // Channel，避免两轮都让同一物理链路先排队。
            const uint32_t channelIndex =
                (issueIndex + sourceIndex) % ctx.arg->channelCount;
            ccu::LocalAddr source;
            source.addr = ctx.output;
            source.addr += ctx.sourceBlockOffsets[sourceIndex];
            source.addr += stripeOffset;
            source.token = ctx.outputToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.sourceBlockOffsets[sourceIndex];
            destination.addr += stripeOffset;
            destination.token = ctx.remoteTokens[channelIndex];
            const uint16_t eventBit = static_cast<uint16_t>(1U << operationIndex);
            ++operationIndex;
            relayMask = static_cast<uint16_t>(relayMask | eventBit);
            CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                *stripeSize, ctx.relayEvent, eventBit));
        }
    }
    if (operationIndex > 7U) {
        return CCU_E_PARA;
    }
    CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.relayEvent, relayMask));
    CCU_RETURN_IF_ERROR(FullPostSync(ctx.arg));
    return CCU_SUCCESS;
}

namespace v22_rank16 {

constexpr uint32_t V22_OUTPUT_XN_ID = 1;
constexpr uint32_t V22_TOKEN_XN_ID = 2;
constexpr uint32_t V22_CKE_INDEX = 0;
constexpr uint32_t V22_POST_SYNC_BASE_ID = 3;
constexpr uint32_t V22_FULL_SYNC_ID = 6;
constexpr uint32_t V22_RANKS_PER_SERVER = 8;
constexpr uint32_t V22_MAX_PHASE_CHUNK_NUM = 7;
constexpr uint32_t V22_EVENT_BIT_NUM = 16;

struct V22AllGatherGroupContext {
    const V22CcuKernelArgAllGatherGroup *arg;
    ccu::Variable input;
    ccu::Variable localOutput;
    ccu::Variable localToken;
    ccu::Variable rankOutputOffset;
    ccu::Variable firstSliceSize;
    ccu::Variable secondSliceSize;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event remoteEvent;
    ccu::Event localEvent;
};

struct V22PhaseContext {
    const V22CcuKernelArgAllGatherPhase *arg;
    ccu::Variable input;
    ccu::Variable localOutput;
    ccu::Variable localToken;
    ccu::Variable rankOutputOffset;
    ccu::Variable batchOffset;
    ccu::Variable baseChunkSize;
    ccu::Variable lastChunkSize;
    ccu::Variable chunkOffsets[V22_MAX_PHASE_CHUNK_NUM];
    ccu::Variable remoteBlockOffsets[V22_RANKS_PER_SERVER];
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event remoteEvent;
    ccu::Event localEvent;
};

template <typename Context>
CcuResult V22PublishRemoteResources(Context &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            ctx.arg->channels[i],
            ctx.localOutput,
            V22_OUTPUT_XN_ID,
            V22_CKE_INDEX,
            1U << V22_OUTPUT_XN_ID));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            ctx.arg->channels[i],
            ctx.localToken,
            V22_TOKEN_XN_ID,
            V22_CKE_INDEX,
            1U << V22_TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

template <typename Context>
CcuResult V22WaitRemoteResources(Context &ctx)
{
    const uint32_t resourceBits =
        (1U << V22_OUTPUT_XN_ID) | (1U << V22_TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            ctx.arg->channels[i], V22_CKE_INDEX, resourceBits));
    }
    return CCU_SUCCESS;
}

template <typename Context>
CcuResult V22ExchangeRemoteResources(Context &ctx)
{
    CCU_RETURN_IF_ERROR(V22PublishRemoteResources(ctx));
    return V22WaitRemoteResources(ctx);
}

template <typename Context>
CcuResult V22FullPostSync(Context &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            ctx.arg->channels[i],
            V22_CKE_INDEX,
            1U << V22_FULL_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            ctx.arg->channels[i],
            V22_CKE_INDEX,
            1U << V22_FULL_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult V22TransferSlicesAsync(V22AllGatherGroupContext &ctx)
{
    ccu::LocalAddr firstSrc;
    firstSrc.addr = ctx.input;
    firstSrc.token = ctx.localToken;

    uint16_t firstRemoteMask = 0U;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        if (ctx.arg->exchangeResources && (ctx.arg->latencyPipeline || ctx.arg->progressiveAddressReady)) {
            const uint32_t resourceBits =
                (1U << V22_OUTPUT_XN_ID) | (1U << V22_TOKEN_XN_ID);
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                ctx.arg->channels[i], V22_CKE_INDEX, resourceBits));
        }
        ccu::RemoteAddr firstDst;
        firstDst.addr = ctx.remoteOutputs[i];
        firstDst.addr += ctx.rankOutputOffset;
        firstDst.token = ctx.remoteTokens[i];

        const uint16_t eventBit = static_cast<uint16_t>(1U << i);
        firstRemoteMask =
            static_cast<uint16_t>(firstRemoteMask | eventBit);
        CCU_RETURN_IF_ERROR(ccu::Write(
            ctx.arg->channels[i],
            firstDst,
            firstSrc,
            ctx.firstSliceSize,
            ctx.remoteEvent,
            eventBit));
    }

    uint16_t secondRemoteMask = 0U;
    CCU_IF(ctx.secondSliceSize != 0U)
    {
        ccu::LocalAddr secondSrc;
        secondSrc.addr = ctx.input;
        secondSrc.addr += ctx.firstSliceSize;
        secondSrc.token = ctx.localToken;

        for (uint32_t issueIndex = 0; issueIndex < ctx.arg->channelCount; ++issueIndex) {
            const uint32_t i = ctx.arg->rotateSecondSlice
                ? (issueIndex + 1U) % ctx.arg->channelCount
                : issueIndex;
            ccu::RemoteAddr secondDst;
            secondDst.addr = ctx.remoteOutputs[i];
            secondDst.addr += ctx.rankOutputOffset;
            secondDst.addr += ctx.firstSliceSize;
            secondDst.token = ctx.remoteTokens[i];

            const uint16_t eventBit = static_cast<uint16_t>(
                1U << (ctx.arg->channelCount + issueIndex));
            secondRemoteMask =
                static_cast<uint16_t>(secondRemoteMask | eventBit);
            CCU_RETURN_IF_ERROR(ccu::Write(
                ctx.arg->channels[i],
                secondDst,
                secondSrc,
                ctx.secondSliceSize,
                ctx.remoteEvent,
                eventBit));
        }
    }

    if (ctx.arg->handleLocalCopy) {
        ccu::LocalAddr firstLocalDst;
        firstLocalDst.addr = ctx.localOutput;
        firstLocalDst.addr += ctx.rankOutputOffset;
        firstLocalDst.token = ctx.localToken;
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(
            firstLocalDst,
            firstSrc,
            ctx.firstSliceSize,
            ctx.localEvent,
            1U));

        CCU_IF(ctx.secondSliceSize != 0U)
        {
            ccu::LocalAddr secondSrc;
            secondSrc.addr = ctx.input;
            secondSrc.addr += ctx.firstSliceSize;
            secondSrc.token = ctx.localToken;

            ccu::LocalAddr secondLocalDst;
            secondLocalDst.addr = ctx.localOutput;
            secondLocalDst.addr += ctx.rankOutputOffset;
            secondLocalDst.addr += ctx.firstSliceSize;
            secondLocalDst.token = ctx.localToken;
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(
                secondLocalDst,
                secondSrc,
                ctx.secondSliceSize,
                ctx.localEvent,
                2U));
        }
    }

    if (ctx.arg->combineSliceWaits) {
        CCU_IF(ctx.secondSliceSize != 0U)
        {
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.remoteEvent,
                static_cast<uint16_t>(firstRemoteMask | secondRemoteMask)));
        }
        CCU_ELSE
        {
            CCU_RETURN_IF_ERROR(ccu::EventWait(
                ctx.remoteEvent, firstRemoteMask));
        }
    } else {
        CCU_RETURN_IF_ERROR(ccu::EventWait(
            ctx.remoteEvent, firstRemoteMask));
        CCU_IF(ctx.secondSliceSize != 0U)
        {
            CCU_RETURN_IF_ERROR(ccu::EventWait(
                ctx.remoteEvent, secondRemoteMask));
        }
    }

    if (ctx.arg->handleLocalCopy) {
        if (ctx.arg->combineSliceWaits) {
            CCU_IF(ctx.secondSliceSize != 0U)
            {
                CCU_RETURN_IF_ERROR(ccu::EventWait(
                    ctx.localEvent, 3U));
            }
            CCU_ELSE
            {
                CCU_RETURN_IF_ERROR(ccu::EventWait(
                    ctx.localEvent, 1U));
            }
        } else {
            CCU_RETURN_IF_ERROR(ccu::EventWait(
                ctx.localEvent, 1U));
            CCU_IF(ctx.secondSliceSize != 0U)
            {
                CCU_RETURN_IF_ERROR(ccu::EventWait(
                    ctx.localEvent, 2U));
            }
        }
    }
    return CCU_SUCCESS;
}

bool V22IsDirectSeed(uint32_t destinationLocalRank,
    uint32_t sourceLocalRank,
    uint32_t seedCount,
    uint32_t patternOffset,
    uint32_t patternSize)
{
    const uint32_t start =
        (sourceLocalRank + patternOffset) % patternSize;
    const uint32_t distance =
        (destinationLocalRank + patternSize - start) % patternSize;
    return distance < seedCount;
}

uint32_t V22FindChannelByLocalRank(
    const V22CcuKernelArgAllGatherPhase *arg,
    uint32_t targetLocalRank,
    uint32_t patternSize)
{
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        if (arg->peerRanks[i] % patternSize == targetLocalRank) {
            return i;
        }
    }
    return arg->channelCount;
}

CcuResult V22RunCrossDirectBatch(V22PhaseContext &ctx)
{
    if (ctx.arg->patternSize == 0U ||
        ctx.arg->patternSize > V22_RANKS_PER_SERVER ||
        ctx.arg->incomingPatternSize == 0U ||
        ctx.arg->incomingPatternSize > V22_RANKS_PER_SERVER ||
        ctx.arg->channelCount != ctx.arg->patternSize ||
        ctx.arg->seedCount == 0U ||
        ctx.arg->seedCount > ctx.arg->patternSize ||
        ctx.arg->incomingSeedCount == 0U ||
        ctx.arg->incomingSeedCount > ctx.arg->incomingPatternSize ||
        ctx.arg->chunkCount == 0U ||
        ctx.arg->chunkCount > V22_MAX_PHASE_CHUNK_NUM ||
        ctx.arg->operationCount == 0U ||
        ctx.arg->operationCount > V22_MAX_BATCH_PLAN_SIZE ||
        ctx.arg->operationCount *
            (ctx.arg->seedCount < ctx.arg->channelCount ?
                ctx.arg->seedCount :
                ctx.arg->channelCount) >
            V22_EVENT_BIT_NUM) {
        return CCU_E_PARA;
    }

    const uint32_t sourceLocalRank =
        ctx.arg->rankId % ctx.arg->patternSize;

    uint16_t remoteMask = 0U;
    uint32_t operationIndex = 0U;
    bool resourceReady[V22_RANKS_PER_SERVER]{};
    for (uint32_t planIndex = 0;
         planIndex < ctx.arg->operationCount;
         ++planIndex) {
        const V22CcuTransferPlan &plan =
            ctx.arg->plans[planIndex];
        if (plan.chunkIndex >= ctx.arg->chunkCount) {
            return CCU_E_PARA;
        }

        ccu::LocalAddr src;
        src.addr = ctx.input;
        src.addr += ctx.batchOffset;
        src.addr += ctx.chunkOffsets[plan.chunkIndex];
        src.token = ctx.localToken;
        ccu::Variable &chunkSize =
            plan.chunkIndex + 1U == ctx.arg->chunkCount ?
            ctx.lastChunkSize :
            ctx.baseChunkSize;

        for (uint32_t issueIndex = 0; issueIndex < ctx.arg->channelCount; ++issueIndex) {
            const uint32_t i =
                ctx.arg->rotateSecondChunk && plan.chunkIndex != 0U
                ? (issueIndex + 1U) % ctx.arg->channelCount
                : issueIndex;
            const uint32_t destinationLocalRank =
                ctx.arg->peerRanks[i] % ctx.arg->patternSize;
            if (!V22IsDirectSeed(
                    destinationLocalRank,
                    sourceLocalRank,
                    ctx.arg->seedCount,
                    plan.chunkIndex % ctx.arg->patternSize,
                    ctx.arg->patternSize)) {
                continue;
            }

            if (ctx.arg->exchangeResources && ctx.arg->progressiveAddressReady
                && !resourceReady[i]) {
                const uint32_t resourceBits =
                    (1U << V22_OUTPUT_XN_ID) | (1U << V22_TOKEN_XN_ID);
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                    ctx.arg->channels[i], V22_CKE_INDEX, resourceBits));
                resourceReady[i] = true;
            }

            ccu::RemoteAddr dst;
            dst.addr = ctx.remoteOutputs[i];
            dst.addr += ctx.rankOutputOffset;
            dst.addr += ctx.batchOffset;
            dst.addr += ctx.chunkOffsets[plan.chunkIndex];
            dst.token = ctx.remoteTokens[i];

            const uint16_t eventBit =
                static_cast<uint16_t>(1U << operationIndex);
            ++operationIndex;
            remoteMask =
                static_cast<uint16_t>(remoteMask | eventBit);
            CCU_RETURN_IF_ERROR(ccu::Write(
                ctx.arg->channels[i],
                dst,
                src,
                chunkSize,
                ctx.remoteEvent,
                eventBit));
        }
    }

    // Seed DMA执行时继续收齐未使用Channel的资源，供后续exchange=false的
    // Direct静态Kernel直接复用。
    if (ctx.arg->exchangeResources && ctx.arg->progressiveAddressReady) {
        const uint32_t resourceBits =
            (1U << V22_OUTPUT_XN_ID) | (1U << V22_TOKEN_XN_ID);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (!resourceReady[channelIndex]) {
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                    ctx.arg->channels[channelIndex], V22_CKE_INDEX, resourceBits));
            }
        }
    }

    CCU_RETURN_IF_ERROR(ccu::EventWait(
        ctx.remoteEvent, remoteMask));
    return CCU_SUCCESS;
}

CcuResult V22SeedPostSync(V22PhaseContext &ctx)
{
    const uint32_t outgoingLocalRank =
        ctx.arg->rankId % ctx.arg->patternSize;
    const uint32_t incomingLocalRank =
        ctx.arg->rankId % ctx.arg->incomingPatternSize;
    const uint32_t seedSyncId =
        V22_POST_SYNC_BASE_ID +
        ctx.arg->plans[0].chunkIndex / 3U;

    if (ctx.arg->seedRecord) {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            const uint32_t destinationLocalRank =
                ctx.arg->peerRanks[i] % ctx.arg->patternSize;
            bool selectedInBatch = false;
            for (uint32_t planIndex = 0;
                 planIndex < ctx.arg->operationCount;
                 ++planIndex) {
                const V22CcuTransferPlan &plan =
                    ctx.arg->plans[planIndex];
                if (V22IsDirectSeed(
                        destinationLocalRank,
                        outgoingLocalRank,
                        ctx.arg->seedCount,
                        plan.chunkIndex % ctx.arg->patternSize,
                        ctx.arg->patternSize)) {
                    selectedInBatch = true;
                    break;
                }
            }
            if (selectedInBatch) {
                CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
                    ctx.arg->channels[i],
                    V22_CKE_INDEX,
                    1U << seedSyncId));
            }
        }
    }

    if (ctx.arg->seedWait) {
        for (uint32_t channelIndex = 0;
             channelIndex < ctx.arg->channelCount;
             ++channelIndex) {
            const uint32_t sourceLocalRank =
                ctx.arg->peerRanks[channelIndex] %
                ctx.arg->incomingPatternSize;
            bool selectedInBatch = false;
            for (uint32_t planIndex = 0;
                 planIndex < ctx.arg->operationCount;
                 ++planIndex) {
                const V22CcuTransferPlan &plan =
                    ctx.arg->plans[planIndex];
                if (V22IsDirectSeed(
                        incomingLocalRank,
                        sourceLocalRank,
                        ctx.arg->incomingSeedCount,
                        plan.chunkIndex %
                            ctx.arg->incomingPatternSize,
                        ctx.arg->incomingPatternSize)) {
                    selectedInBatch = true;
                    break;
                }
            }
            if (selectedInBatch) {
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                    ctx.arg->channels[channelIndex],
                    V22_CKE_INDEX,
                    1U << seedSyncId));
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult V22RunIntraRelayBatch(V22PhaseContext &ctx)
{
    if (ctx.arg->patternSize < 2U ||
        ctx.arg->patternSize > V22_RANKS_PER_SERVER ||
        ctx.arg->channelCount != ctx.arg->patternSize - 1U ||
        ctx.arg->seedCount == 0U ||
        ctx.arg->seedCount >= ctx.arg->patternSize ||
        ctx.arg->chunkCount == 0U ||
        ctx.arg->chunkCount > V22_MAX_PHASE_CHUNK_NUM ||
        ctx.arg->operationCount > V22_MAX_BATCH_PLAN_SIZE ||
        ctx.arg->operationCount > V22_EVENT_BIT_NUM) {
        return CCU_E_PARA;
    }
    if (ctx.arg->operationCount == 0U) {
        return CCU_SUCCESS;
    }

    const uint32_t myLocalRank =
        ctx.arg->rankId % ctx.arg->patternSize;

    uint16_t remoteMask = 0U;
    for (uint32_t operationIndex = 0;
         operationIndex < ctx.arg->operationCount;
         ++operationIndex) {
        const V22CcuTransferPlan &plan =
            ctx.arg->plans[operationIndex];
        const uint32_t missingCount =
            ctx.arg->patternSize - ctx.arg->seedCount;
        if (plan.chunkIndex >= ctx.arg->chunkCount ||
            plan.patternOffset >= ctx.arg->patternSize ||
            plan.missingIndex >= missingCount ||
            plan.relayOwnerOffset >= ctx.arg->seedCount ||
            plan.sourceIndex >= V22_RANKS_PER_SERVER) {
            return CCU_E_PARA;
        }

        const uint32_t sourceLocalRank =
            plan.sourceIndex % ctx.arg->patternSize;
        const uint32_t start =
            (sourceLocalRank + plan.patternOffset) %
            ctx.arg->patternSize;
        const uint32_t ownerLocalRank =
            (start + plan.relayOwnerOffset) %
            ctx.arg->patternSize;
        if (ownerLocalRank != myLocalRank) {
            return CCU_E_PARA;
        }

        const uint32_t destinationLocalRank =
            (start + ctx.arg->seedCount + plan.missingIndex) %
            ctx.arg->patternSize;
        const uint32_t channelIndex =
            V22FindChannelByLocalRank(
                ctx.arg,
                destinationLocalRank,
                ctx.arg->patternSize);
        if (channelIndex >= ctx.arg->channelCount) {
            return CCU_E_PARA;
        }

        ccu::LocalAddr src;
        src.addr = ctx.localOutput;
        src.addr += ctx.remoteBlockOffsets[plan.sourceIndex];
        src.addr += ctx.batchOffset;
        src.addr += ctx.chunkOffsets[plan.chunkIndex];
        src.token = ctx.localToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteOutputs[channelIndex];
        dst.addr += ctx.remoteBlockOffsets[plan.sourceIndex];
        dst.addr += ctx.batchOffset;
        dst.addr += ctx.chunkOffsets[plan.chunkIndex];
        dst.token = ctx.remoteTokens[channelIndex];

        ccu::Variable &chunkSize =
            plan.chunkIndex + 1U == ctx.arg->chunkCount ?
            ctx.lastChunkSize :
            ctx.baseChunkSize;
        const uint16_t eventBit =
            static_cast<uint16_t>(1U << operationIndex);
        remoteMask =
            static_cast<uint16_t>(remoteMask | eventBit);
        CCU_RETURN_IF_ERROR(ccu::Write(
            ctx.arg->channels[channelIndex],
            dst,
            src,
            chunkSize,
            ctx.remoteEvent,
            eventBit));
    }

    CCU_RETURN_IF_ERROR(ccu::EventWait(
        ctx.remoteEvent, remoteMask));
    return CCU_SUCCESS;
}

} // namespace v22_rank16

CcuResult CcuV22AllGatherGroupKernel(CcuKernelArg arg)
{
    auto *kernelArg =
        static_cast<V22CcuKernelArgAllGatherGroup *>(arg);
    if (kernelArg == nullptr ||
        kernelArg->channelCount == 0U ||
        kernelArg->channelCount >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    v22_rank16::V22AllGatherGroupContext ctx;
    ctx.arg = kernelArg;
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        ctx.remoteOutputs[i] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[i],
                v22_rank16::V22_OUTPUT_XN_ID);
        ctx.remoteTokens[i] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[i],
                v22_rank16::V22_TOKEN_XN_ID);
    }

    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.localOutput, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.localToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(
        ctx.rankOutputOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(
        ctx.firstSliceSize, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(
        ctx.secondSliceSize, argId++));

    if (ctx.arg->exchangeResources) {
        if (ctx.arg->latencyPipeline || ctx.arg->progressiveAddressReady) {
            CCU_RETURN_IF_ERROR(
                v22_rank16::V22PublishRemoteResources(ctx));
        } else {
            CCU_RETURN_IF_ERROR(
                v22_rank16::V22ExchangeRemoteResources(ctx));
        }
    }
    CCU_RETURN_IF_ERROR(
        v22_rank16::V22TransferSlicesAsync(ctx));
    if (ctx.arg->finalSync) {
        if (ctx.arg->latencyPipeline) {
            for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
                CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                    ctx.arg->channels[i], v22_rank16::V22_CKE_INDEX,
                    1U << v22_rank16::V22_FULL_SYNC_ID));
            }
        } else {
            CCU_RETURN_IF_ERROR(
                v22_rank16::V22FullPostSync(ctx));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuV22AllGatherPhaseKernel(CcuKernelArg arg)
{
    auto *kernelArg =
        static_cast<V22CcuKernelArgAllGatherPhase *>(arg);
    if (kernelArg == nullptr ||
        kernelArg->channelCount == 0U ||
        kernelArg->channelCount >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    v22_rank16::V22PhaseContext ctx;
    ctx.arg = kernelArg;
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        ctx.remoteOutputs[i] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[i],
                v22_rank16::V22_OUTPUT_XN_ID);
        ctx.remoteTokens[i] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[i],
                v22_rank16::V22_TOKEN_XN_ID);
    }

    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.localOutput, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.localToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(
        ctx.rankOutputOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(
        ctx.batchOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(
        ctx.baseChunkSize, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(
        ctx.lastChunkSize, argId++));
    for (uint32_t i = 0;
         i < v22_rank16::V22_RANKS_PER_SERVER;
         ++i) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(
            ctx.remoteBlockOffsets[i], argId++));
    }
    ctx.chunkOffsets[0] = 0U;
    for (uint32_t i = 1;
         i < v22_rank16::V22_MAX_PHASE_CHUNK_NUM;
         ++i) {
        ctx.chunkOffsets[i] = ctx.chunkOffsets[i - 1U];
        ctx.chunkOffsets[i] += ctx.baseChunkSize;
    }

    if (ctx.arg->exchangeResources) {
        if (ctx.arg->progressiveAddressReady) {
            CCU_RETURN_IF_ERROR(
                v22_rank16::V22PublishRemoteResources(ctx));
        } else {
            CCU_RETURN_IF_ERROR(
                v22_rank16::V22ExchangeRemoteResources(ctx));
        }
    }

    if (ctx.arg->phaseKind ==
        V22CcuPhaseKind::INTRA_RELAY_BATCH) {
        CCU_RETURN_IF_ERROR(
            v22_rank16::V22RunIntraRelayBatch(ctx));
    } else if (ctx.arg->phaseKind ==
        V22CcuPhaseKind::CROSS_DIRECT_BATCH) {
        CCU_RETURN_IF_ERROR(
            v22_rank16::V22RunCrossDirectBatch(ctx));
        if (ctx.arg->seedRecord || ctx.arg->seedWait) {
            CCU_RETURN_IF_ERROR(
                v22_rank16::V22SeedPostSync(ctx));
        }
    } else {
        return CCU_E_PARA;
    }

    if (ctx.arg->finalSync) {
        CCU_RETURN_IF_ERROR(
            v22_rank16::V22FullPostSync(ctx));
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl

#undef CCU_RETURN_IF_ERROR
