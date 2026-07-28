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

#define CCU_KERNEL_CHK_RET(call) \
    do { \
        CcuResult ccuRet = (call); \
        if (UNLIKELY(ccuRet != CCU_SUCCESS)) { \
            HCCL_ERROR("[%s] call trace: ccuRet -> %d", __func__, ccuRet); \
            return ccuRet; \
        } \
    } while (0)

namespace ops_hccl {
namespace {

    constexpr uint16_t OUTPUT_XN_ID = 0;
    constexpr uint16_t TOKEN_XN_ID = 1;
    constexpr uint16_t DATA_SIGNAL_ID = 2;
    constexpr uint16_t ACK_SIGNAL_ID = 3;
    constexpr uint16_t POST_SYNC_ID = 4;
    constexpr uint16_t CKE_IDX = 0;

    struct PeerKernelContext {
        const CcuKernelArgPeers *arg = nullptr;
        ccu::Variable output;
        ccu::Variable token;
        ccu::Variable dataSize;
        ccu::Variable normalSliceSize;
        ccu::Variable lastSliceSize;
        std::vector<ccu::Variable> remoteOutput;
        std::vector<ccu::Variable> remoteToken;
        ccu::Event event;
    };

    uint32_t FindPeerIndex(const CcuKernelArgPeers *arg, uint32_t peerRank)
    {
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            if (arg->peerRanks[i] == peerRank) {
                return i;
            }
        }
        return MAX_RANK_SIZE;
    }

    CcuResult InitPeerContext(PeerKernelContext &ctx, const CcuKernelArgPeers *arg)
    {
        if (arg == nullptr || arg->channelCount == 0 || arg->channelCount >= MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }
        ctx.arg = arg;
        ctx.remoteOutput.resize(arg->channelCount);
        ctx.remoteToken.resize(arg->channelCount);
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            ctx.remoteOutput[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], OUTPUT_XN_ID);
            ctx.remoteToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], TOKEN_XN_ID);
        }

        uint32_t argId = 0;
        CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.output, argId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.token, argId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.dataSize, argId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
        CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++));
        return CCU_SUCCESS;
    }

    CcuResult ExchangeMemoryInfo(PeerKernelContext &ctx)
    {
        const uint32_t allBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(
                ctx.arg->channels[i], ctx.output, OUTPUT_XN_ID, CKE_IDX, 1U << OUTPUT_XN_ID));
            CCU_KERNEL_CHK_RET(
                ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token, TOKEN_XN_ID, CKE_IDX, 1U << TOKEN_XN_ID));
        }
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_KERNEL_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX, allBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult PostSync(PeerKernelContext &ctx)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_KERNEL_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX, 1U << POST_SYNC_ID));
        }
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_KERNEL_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX, 1U << POST_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunFlatBroadcast(CcuKernelArg arg)
    {
        auto *kernelArg = static_cast<CcuKernelArgPeers *>(arg);
        PeerKernelContext ctx;
        CCU_KERNEL_CHK_RET(InitPeerContext(ctx, kernelArg));
        CCU_KERNEL_CHK_RET(ExchangeMemoryInfo(ctx));

        if (kernelArg->rankId == kernelArg->rootId) {
            ccu::LocalAddr src;
            src.addr = ctx.output;
            src.token = ctx.token;
            CCU_IF(ctx.dataSize != 0)
            {
                for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                    ccu::RemoteAddr dst;
                    dst.addr = ctx.remoteOutput[i];
                    dst.token = ctx.remoteToken[i];
                    CCU_KERNEL_CHK_RET(ccu::Write(kernelArg->channels[i], dst, src, ctx.dataSize, ctx.event, 1U << i));
                }
                CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, (1U << kernelArg->channelCount) - 1U));
            }
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                CCU_KERNEL_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX, 1U << DATA_SIGNAL_ID));
            }
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                CCU_KERNEL_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX, 1U << ACK_SIGNAL_ID));
            }
        } else {
            const uint32_t rootIdx = FindPeerIndex(kernelArg, kernelArg->rootId);
            if (rootIdx >= kernelArg->channelCount) {
                return CCU_E_NOT_FOUND;
            }
            CCU_KERNEL_CHK_RET(ccu::NotifyWait(kernelArg->channels[rootIdx], CKE_IDX, 1U << DATA_SIGNAL_ID));
            CCU_KERNEL_CHK_RET(ccu::NotifyRecord(kernelArg->channels[rootIdx], CKE_IDX, 1U << ACK_SIGNAL_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunScatter(CcuKernelArg arg)
    {
        auto *kernelArg = static_cast<CcuKernelArgPeers *>(arg);
        PeerKernelContext ctx;
        CCU_KERNEL_CHK_RET(InitPeerContext(ctx, kernelArg));
        CCU_KERNEL_CHK_RET(ExchangeMemoryInfo(ctx));

        if (kernelArg->rankId == kernelArg->rootId) {
            CCU_IF(ctx.dataSize != 0)
            {
                for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                    const uint32_t dstRank = kernelArg->peerRanks[i];
                    ccu::Variable sliceSize;
                    if (dstRank + 1 == kernelArg->rankSize) {
                        sliceSize = ctx.lastSliceSize;
                    } else {
                        sliceSize = ctx.normalSliceSize;
                    }
                    ccu::Variable sliceOffset;
                    sliceOffset = 0;
                    for (uint32_t offsetIdx = 0; offsetIdx < dstRank; ++offsetIdx) {
                        sliceOffset += ctx.normalSliceSize;
                    }

                    ccu::LocalAddr src;
                    src.addr = ctx.output;
                    src.addr += sliceOffset;
                    src.token = ctx.token;
                    ccu::RemoteAddr dst;
                    dst.addr = ctx.remoteOutput[i];
                    dst.addr += sliceOffset;
                    dst.token = ctx.remoteToken[i];
                    CCU_IF(sliceSize != 0)
                    {
                        CCU_KERNEL_CHK_RET(ccu::Write(kernelArg->channels[i], dst, src, sliceSize, ctx.event, 1U << i));
                    }
                    CCU_IF(sliceSize == 0)
                    {
                        CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, 1U << i));
                    }
                }
                CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, (1U << kernelArg->channelCount) - 1U));
            }
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                CCU_KERNEL_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX, 1U << DATA_SIGNAL_ID));
            }
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                CCU_KERNEL_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX, 1U << ACK_SIGNAL_ID));
            }
        } else {
            const uint32_t rootIdx = FindPeerIndex(kernelArg, kernelArg->rootId);
            if (rootIdx >= kernelArg->channelCount) {
                return CCU_E_NOT_FOUND;
            }
            CCU_KERNEL_CHK_RET(ccu::NotifyWait(kernelArg->channels[rootIdx], CKE_IDX, 1U << DATA_SIGNAL_ID));
            CCU_KERNEL_CHK_RET(ccu::NotifyRecord(kernelArg->channels[rootIdx], CKE_IDX, 1U << ACK_SIGNAL_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunAllGather(CcuKernelArg arg)
    {
        auto *kernelArg = static_cast<CcuKernelArgPeers *>(arg);
        PeerKernelContext ctx;
        CCU_KERNEL_CHK_RET(InitPeerContext(ctx, kernelArg));
        CCU_KERNEL_CHK_RET(ExchangeMemoryInfo(ctx));

        ccu::Variable ownSliceSize;
        if (kernelArg->rankId + 1 == kernelArg->rankSize) {
            ownSliceSize = ctx.lastSliceSize;
        } else {
            ownSliceSize = ctx.normalSliceSize;
        }
        ccu::Variable ownSliceOffset;
        ownSliceOffset = 0;
        for (uint32_t offsetIdx = 0; offsetIdx < kernelArg->rankId; ++offsetIdx) {
            ownSliceOffset += ctx.normalSliceSize;
        }

        ccu::LocalAddr src;
        src.addr = ctx.output;
        src.addr += ownSliceOffset;
        src.token = ctx.token;

        uint32_t sendMask = 0;
        CCU_IF(ownSliceSize != 0)
        {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                if (kernelArg->peerRanks[i] == kernelArg->rootId) {
                    continue;
                }
                ccu::RemoteAddr dst;
                dst.addr = ctx.remoteOutput[i];
                dst.addr += ownSliceOffset;
                dst.token = ctx.remoteToken[i];
                CCU_KERNEL_CHK_RET(ccu::Write(kernelArg->channels[i], dst, src, ownSliceSize, ctx.event, 1U << i));
                sendMask |= 1U << i;
            }
            if (sendMask != 0) {
                CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, sendMask));
            }
        }
        CCU_KERNEL_CHK_RET(PostSync(ctx));
        return CCU_SUCCESS;
    }

    CcuResult WriteNhrSlices(PeerKernelContext &ctx, const BroadcastNhrStep &step, uint32_t peerIdx, uint32_t signalId)
    {
        uint32_t eventMask = 0;
        for (uint32_t i = 0; i < step.txSliceCount; ++i) {
            const uint32_t sliceIdx = step.txSliceIdxs[i];
            ccu::Variable sliceSize;
            if (sliceIdx + 1 == ctx.arg->rankSize) {
                sliceSize = ctx.lastSliceSize;
            } else {
                sliceSize = ctx.normalSliceSize;
            }
            ccu::Variable sliceOffset;
            sliceOffset = 0;
            for (uint32_t offsetIdx = 0; offsetIdx < sliceIdx; ++offsetIdx) {
                sliceOffset += ctx.normalSliceSize;
            }
            ccu::LocalAddr src;
            src.addr = ctx.output;
            src.addr += sliceOffset;
            src.token = ctx.token;
            ccu::RemoteAddr dst;
            dst.addr = ctx.remoteOutput[peerIdx];
            dst.addr += sliceOffset;
            dst.token = ctx.remoteToken[peerIdx];
            CCU_IF(sliceSize != 0)
            {
                CCU_KERNEL_CHK_RET(ccu::Write(
                    ctx.arg->channels[peerIdx], dst, src, sliceSize, ctx.event, static_cast<uint16_t>(1U << i)));
            }
            CCU_IF(sliceSize == 0)
            {
                CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, static_cast<uint16_t>(1U << i)));
            }
            eventMask |= 1U << i;
        }
        if (eventMask != 0) {
            CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, eventMask));
        }
        CCU_KERNEL_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[peerIdx], CKE_IDX, 1U << signalId));
        return CCU_SUCCESS;
    }

    CcuResult RunNhrBroadcast(CcuKernelArg arg)
    {
        auto *kernelArg = static_cast<CcuKernelArgNhr *>(arg);
        PeerKernelContext ctx;
        CCU_KERNEL_CHK_RET(InitPeerContext(ctx, kernelArg));
        CCU_KERNEL_CHK_RET(ExchangeMemoryInfo(ctx));

        for (uint32_t stepIdx = 0; stepIdx < kernelArg->scatterStepCount; ++stepIdx) {
            const BroadcastNhrStep &step = kernelArg->steps[stepIdx];
            if (step.txSliceCount != 0) {
                const uint32_t peerIdx = FindPeerIndex(kernelArg, step.toRank);
                if (peerIdx >= kernelArg->channelCount) {
                    return CCU_E_NOT_FOUND;
                }
                CCU_KERNEL_CHK_RET(WriteNhrSlices(ctx, step, peerIdx, DATA_SIGNAL_ID));
            }
            if (step.rxSliceCount != 0) {
                const uint32_t peerIdx = FindPeerIndex(kernelArg, step.fromRank);
                if (peerIdx >= kernelArg->channelCount) {
                    return CCU_E_NOT_FOUND;
                }
                CCU_KERNEL_CHK_RET(ccu::NotifyWait(kernelArg->channels[peerIdx], CKE_IDX, 1U << DATA_SIGNAL_ID));
            }
        }

        for (uint32_t stepIdx = kernelArg->scatterStepCount; stepIdx < kernelArg->totalStepCount; ++stepIdx) {
            const BroadcastNhrStep &step = kernelArg->steps[stepIdx];
            const uint32_t toPeerIdx = FindPeerIndex(kernelArg, step.toRank);
            const uint32_t fromPeerIdx = FindPeerIndex(kernelArg, step.fromRank);
            if (toPeerIdx >= kernelArg->channelCount || fromPeerIdx >= kernelArg->channelCount) {
                return CCU_E_NOT_FOUND;
            }
            CCU_KERNEL_CHK_RET(WriteNhrSlices(ctx, step, toPeerIdx, ACK_SIGNAL_ID));
            CCU_KERNEL_CHK_RET(ccu::NotifyWait(kernelArg->channels[fromPeerIdx], CKE_IDX, 1U << ACK_SIGNAL_ID));
        }
        CCU_KERNEL_CHK_RET(PostSync(ctx));
        return CCU_SUCCESS;
    }

} // namespace

CcuResult CcuFlatBroadcastLayer0Kernel(CcuKernelArg arg)
{
    return RunFlatBroadcast(arg);
}

CcuResult CcuFlatBroadcastLayer1Kernel(CcuKernelArg arg)
{
    return RunFlatBroadcast(arg);
}

CcuResult CcuScatterLayer0Kernel(CcuKernelArg arg)
{
    return RunScatter(arg);
}

CcuResult CcuScatterLayer1Kernel(CcuKernelArg arg)
{
    return RunScatter(arg);
}

CcuResult CcuAllGatherLayer0Kernel(CcuKernelArg arg)
{
    return RunAllGather(arg);
}

CcuResult CcuAllGatherLayer1Kernel(CcuKernelArg arg)
{
    return RunAllGather(arg);
}

CcuResult CcuBroadcastNhrLayer1Kernel(CcuKernelArg arg)
{
    return RunNhrBroadcast(arg);
}

} // namespace ops_hccl

#undef CCU_KERNEL_CHK_RET
