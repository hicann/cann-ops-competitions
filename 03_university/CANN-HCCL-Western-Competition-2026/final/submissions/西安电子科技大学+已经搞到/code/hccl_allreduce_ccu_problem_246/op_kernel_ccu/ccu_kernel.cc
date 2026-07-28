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

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include <hcomm/ccu/ccu_primitives.hpp>

namespace ops_hccl {
namespace {
constexpr uint32_t kInputId = 0;
constexpr uint32_t kOutputId = 1;
constexpr uint32_t kInputTokenId = 2;
constexpr uint32_t kOutputTokenId = 3;
constexpr uint32_t kPostSyncId = 4;
constexpr uint32_t kChannelCkeId = 0;
constexpr uint32_t kPipeParallelDim = 16;
// 16 个 MS buffer 可覆盖拓扑三每个 die 的最多 8 条远端通道及本地源，
// 同时保留拓扑一/二的并行度。
constexpr uint32_t kPipeInterleave = 16;
constexpr uint64_t kPipeMemSlice = 4096;
// 512 KiB 专用图固定使用 16 路 4 KiB 并行块。每批至多 7 个新远端源，
// 第 8 个 MS slot 放本地输入或上一批 partial，资源占用固定为 128 MS。
constexpr uint32_t kSmallTiledParallelDim = 16;
constexpr uint32_t kSmallTiledInterleave = 8;
constexpr uint32_t kSmallTiledMaxRemote = kSmallTiledInterleave - 1;
constexpr uint64_t kSmallTiledMemSlice = 4096;

#define CCU_CHK_RET(expr) \
    do { \
        CcuResult _ret = (expr); \
        if (_ret != CCU_SUCCESS) { \
            return _ret; \
        } \
    } while (0)

} // namespace

namespace {
uint64_t PipeBits(uint16_t end)
{
    return (uint64_t{1} << (end + 1)) - 1;
}

uint64_t PipeLoopParam(uint64_t ctx, uint64_t offset, uint64_t iter)
{
    return ((ctx & PipeBits(8)) << 45) |
           ((offset & PipeBits(32)) << 13) |
           (iter & PipeBits(13));
}

uint64_t PipeParallelParam(uint64_t repeat, uint64_t index, uint64_t total)
{
    return ((repeat & PipeBits(7)) << 55) |
           ((index & PipeBits(7)) << 48) |
           ((total & PipeBits(7)) << 41);
}

uint64_t PipeOffsetParam(uint64_t gsa, uint64_t ms, uint64_t cke)
{
    return ((gsa & PipeBits(32)) << 21) |
           ((ms & PipeBits(11)) << 10) |
           (cke & PipeBits(10));
}

struct SmallTiledReduceVars {
    ccu::LocalAddr dst;
    std::vector<ccu::RemoteAddr> remoteSrc;
    ccu::LocalAddr localSrc;
    ccu::Variable len;
};

// 从官方 GroupReduce 图裁剪出的固定小包 primitive。LoopGroup 并行处理
// 16 个 4 KiB block，远端读、本地读和 LocalReduce 在 MS 内完成。
CcuResult SmallTiledBatchReduce(
    const ChannelHandle *channels, uint32_t channelCount,
    ccu::LocalAddr dst, const std::vector<ccu::RemoteAddr> &src,
    ccu::LocalAddr localSrc, ccu::Variable loopIter,
    HcclDataType dataType, HcclReduceOp reduceOp,
    ccu::Array<ccu::Event> &events,
    ccu::Array<ccu::CcuBuffer> &buffers)
{
    if (channelCount == 0 || channelCount > kSmallTiledMaxRemote ||
        src.size() != channelCount) {
        return CCU_E_PARA;
    }

    const uint32_t sourceCount = channelCount + 1;
    SmallTiledReduceVars vars;
    vars.remoteSrc.resize(channelCount);
    ccu::Variable loopCfg;
    const ccu::Event groupEvent = events[0];
    std::unique_ptr<ccu::Func> body(new ccu::Func(
        [&, groupEvent, channelCount, sourceCount]() {
            for (uint32_t i = 0; i < channelCount; ++i) {
                ccu::Read(channels[i], buffers[i], vars.remoteSrc[i],
                          vars.len, groupEvent,
                          static_cast<uint16_t>(1U << i));
            }
            ccu::LocalCopy(buffers[channelCount], vars.localSrc, vars.len,
                           groupEvent,
                           static_cast<uint16_t>(1U << channelCount));
            ccu::EventWait(groupEvent,
                           static_cast<uint16_t>((1U << sourceCount) - 1U));
            ccu::LocalReduce(&buffers[0], sourceCount, dataType, dataType,
                             reduceOp, vars.len, groupEvent, 1);
            ccu::EventWait(groupEvent, 1);
            ccu::LocalCopy(vars.dst, buffers[0], vars.len, groupEvent, 1);
            ccu::EventWait(groupEvent, 1);
        }));
    std::unique_ptr<ccu::Loop> loop(new ccu::Loop(loopCfg, *body));

    for (uint32_t i = 0; i < channelCount; ++i) {
        vars.remoteSrc[i] = src[i];
    }
    vars.localSrc = localSrc;
    vars.dst = dst;
    vars.len = kSmallTiledMemSlice;

    ccu::Variable parallelCfg;
    ccu::Variable offsetCfg;
    parallelCfg = PipeParallelParam(kSmallTiledParallelDim - 1, 0, 1);
    offsetCfg = PipeOffsetParam(kSmallTiledMemSlice, kSmallTiledInterleave, 1);
    loopCfg = PipeLoopParam(0, kSmallTiledMemSlice * kSmallTiledParallelDim, 0);
    loopCfg = loopCfg + loopIter;
    std::vector<ccu::Loop> groupLoops{*loop};
    ccu::LoopGroup group(parallelCfg, offsetCfg, kSmallTiledParallelDim,
                         groupLoops);
    (void)group;
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize == 0 ||
        kernelArg->rankId >= kernelArg->rankSize || kernelArg->channelCount == 0 ||
        kernelArg->channelCount >= kernelArg->rankSize) {
        return CCU_E_PARA;
    }

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable sliceOffset;
    ccu::Variable sliceSize;
    ccu::Variable phase;
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(input, argId++));
    CCU_CHK_RET(ccu::LoadArg(output, argId++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(sliceOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(phase, argId++));

    if (kernelArg->combineOnly) {
        // 仅用于让 HCOMM 根据通道选择 combine kernel 所属 die。
        (void)ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[0], kInputId);
        ccu::LocalAddr dst;
        dst.addr = output;
        dst.token = outputToken;
        ccu::LocalAddr src;
        src.addr = scratch;
        src.token = scratchToken;
        ccu::Event combineEvent;
        dst.addr = dst.addr + sliceOffset;
        src.addr = src.addr + sliceOffset;
        CCU_IF(sliceSize != 0) {
            CCU_CHK_RET(ccu::LocalReduce(dst, src, sliceSize, kernelArg->dataType,
                                          kernelArg->reduceType, combineEvent, 1));
            CCU_CHK_RET(ccu::EventWait(combineEvent, 1));
        }
        return CCU_SUCCESS;
    }

    std::vector<ccu::Variable> remoteInput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteOutput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteOutputToken(kernelArg->channelCount);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        remoteInput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kInputId);
        remoteOutput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kOutputId);
        remoteInputToken[i] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kInputTokenId);
        remoteOutputToken[i] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kOutputTokenId);
    }

    CCU_IF(phase == 0) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], input, kInputId,
                                                      kChannelCkeId, 1U << kInputId));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], inputToken, kInputTokenId,
                                                      kChannelCkeId, 1U << kInputTokenId));
        }
        constexpr uint32_t inputBits = (1U << kInputId) | (1U << kInputTokenId);
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kChannelCkeId, inputBits));
        }
    } CCU_ELSE {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], output, kOutputId,
                                                      kChannelCkeId, 1U << kOutputId));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], outputToken, kOutputTokenId,
                                                      kChannelCkeId, 1U << kOutputTokenId));
        }
        constexpr uint32_t outputBits = (1U << kOutputId) | (1U << kOutputTokenId);
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kChannelCkeId, outputBits));
        }
    }

    ccu::LocalAddr inputAddr;
    inputAddr.addr = input;
    inputAddr.addr = inputAddr.addr + sliceOffset;
    inputAddr.token = inputToken;
    ccu::LocalAddr outputAddr;
    outputAddr.addr = output;
    outputAddr.addr = outputAddr.addr + sliceOffset;
    outputAddr.token = outputToken;
    ccu::LocalAddr scratchAddr;
    scratchAddr.addr = scratch;
    scratchAddr.addr = scratchAddr.addr + sliceOffset;
    scratchAddr.token = scratchToken;
    ccu::LocalAddr partial = kernelArg->includeLocalInput ? outputAddr : scratchAddr;
    ccu::Event event;

    CCU_IF(sliceSize != 0) {
        CCU_IF(phase != 0) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                ccu::RemoteAddr remote;
                remote.addr = remoteOutput[i];
                remote.addr += sliceOffset;
                remote.token = remoteOutputToken[i];
                CCU_CHK_RET(ccu::Write(kernelArg->channels[i], remote, outputAddr, sliceSize,
                                        event, 1U << i));
            }
            const uint32_t allWrites = (1U << kernelArg->channelCount) - 1U;
            CCU_CHK_RET(ccu::EventWait(event, allWrites));
        } CCU_ELSE {
            uint32_t firstRemote = 0;
            if (kernelArg->includeLocalInput) {
                CCU_CHK_RET(ccu::LocalCopy(partial, inputAddr, sliceSize, event, 1));
                CCU_CHK_RET(ccu::EventWait(event, 1));
            } else {
                ccu::RemoteAddr remote;
                remote.addr = remoteInput[0];
                remote.addr += sliceOffset;
                remote.token = remoteInputToken[0];
                CCU_CHK_RET(ccu::Read(kernelArg->channels[0], partial, remote, sliceSize, event, 1));
                CCU_CHK_RET(ccu::EventWait(event, 1));
                firstRemote = 1;
            }

            for (uint32_t i = firstRemote; i < kernelArg->channelCount; ++i) {
                ccu::RemoteAddr remote;
                remote.addr = remoteInput[i];
                remote.addr += sliceOffset;
                remote.token = remoteInputToken[i];
                CCU_CHK_RET(ccu::ReadReduce(kernelArg->channels[i], partial, remote, sliceSize,
                                             kernelArg->dataType, kernelArg->reduceType, event, 1));
                CCU_CHK_RET(ccu::EventWait(event, 1));
            }
        }
    }

    // 所有对端读完后才允许下一次 launch 复用 input。
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }

    return CCU_SUCCESS;
}

namespace {
bool IsTopologyKernelArgValid(const CcuKernelArgAllReduce *kernelArg, uint32_t expectedRankSize)
{
    return kernelArg != nullptr && kernelArg->rankSize == expectedRankSize &&
        kernelArg->rankId < kernelArg->rankSize && kernelArg->channelCount > 0 &&
        kernelArg->channelCount < kernelArg->rankSize;
}

CcuResult CcuSmallKernel(CcuKernelArg arg, uint32_t expectedRankSize)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if (!IsTopologyKernelArgValid(kernelArg, expectedRankSize)) {
        return CCU_E_PARA;
    }

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable sliceOffset;
    ccu::Variable sliceSize;
    ccu::Variable unusedPhase;
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(input, argId++));
    CCU_CHK_RET(ccu::LoadArg(output, argId++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(sliceOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(sliceSize, argId++));
    // V2.3 兼容 one-shot 图保留第9项 phase 参数布局；图本身不使用该值。
    CCU_CHK_RET(ccu::LoadArg(unusedPhase, argId++));

    std::vector<ccu::Variable> remoteInput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->channelCount);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        remoteInput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kInputId);
        remoteInputToken[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], kInputTokenId);
        CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], input, kInputId,
                                                  kChannelCkeId, 1U << kInputId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], inputToken, kInputTokenId,
                                                  kChannelCkeId, 1U << kInputTokenId));
    }
    constexpr uint32_t inputBits = (1U << kInputId) | (1U << kInputTokenId);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kChannelCkeId, inputBits));
    }

    ccu::LocalAddr inputAddr;
    inputAddr.addr = input;
    inputAddr.addr += sliceOffset;
    inputAddr.token = inputToken;
    ccu::LocalAddr outputAddr;
    outputAddr.addr = output;
    outputAddr.addr += sliceOffset;
    outputAddr.token = outputToken;
    ccu::LocalAddr scratchAddr;
    scratchAddr.addr = scratch;
    scratchAddr.addr += sliceOffset;
    scratchAddr.token = scratchToken;
    ccu::LocalAddr partial = kernelArg->includeLocalInput ? outputAddr : scratchAddr;
    ccu::Event event;

    CCU_IF(sliceSize != 0) {
        uint32_t firstRemote = 0;
        if (kernelArg->includeLocalInput) {
            CCU_CHK_RET(ccu::LocalCopy(partial, inputAddr, sliceSize, event, 1));
            CCU_CHK_RET(ccu::EventWait(event, 1));
        } else {
            ccu::RemoteAddr remote;
            remote.addr = remoteInput[0];
            remote.addr += sliceOffset;
            remote.token = remoteInputToken[0];
            CCU_CHK_RET(ccu::Read(kernelArg->channels[0], partial, remote, sliceSize, event, 1));
            CCU_CHK_RET(ccu::EventWait(event, 1));
            firstRemote = 1;
        }
        for (uint32_t i = firstRemote; i < kernelArg->channelCount; ++i) {
            ccu::RemoteAddr remote;
            remote.addr = remoteInput[i];
            remote.addr += sliceOffset;
            remote.token = remoteInputToken[i];
            CCU_CHK_RET(ccu::ReadReduce(kernelArg->channels[i], partial, remote, sliceSize,
                                         kernelArg->dataType, kernelArg->reduceType, event, 1));
            CCU_CHK_RET(ccu::EventWait(event, 1));
        }
    }

    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    return CCU_SUCCESS;
}

CcuResult CcuSmallTiledKernel(CcuKernelArg arg, uint32_t expectedRankSize,
                              bool fusedGather)
{
    auto *a = static_cast<CcuKernelArgAllReduce *>(arg);
    if (!IsTopologyKernelArgValid(a, expectedRankSize) ||
        (expectedRankSize != 4 && expectedRankSize != 12 &&
         expectedRankSize != 16) ||
        (fusedGather && (expectedRankSize != 4 || !a->includeLocalInput ||
                         a->channelCount != 3))) {
        return CCU_E_PARA;
    }

    ccu::Variable input, output, inputToken, outputToken;
    ccu::Variable scratch, scratchToken, sliceOffset, sliceSize;
    CCU_CHK_RET(ccu::LoadArg(input, 0));
    CCU_CHK_RET(ccu::LoadArg(output, 1));
    CCU_CHK_RET(ccu::LoadArg(inputToken, 2));
    CCU_CHK_RET(ccu::LoadArg(outputToken, 3));
    CCU_CHK_RET(ccu::LoadArg(scratch, 4));
    CCU_CHK_RET(ccu::LoadArg(scratchToken, 5));
    CCU_CHK_RET(ccu::LoadArg(sliceOffset, 6));
    CCU_CHK_RET(ccu::LoadArg(sliceSize, 7));

    std::vector<ccu::Variable> remoteInput(a->channelCount);
    std::vector<ccu::Variable> remoteInputToken(a->channelCount);
    std::vector<ccu::Variable> remoteOutput;
    std::vector<ccu::Variable> remoteOutputToken;
    if (fusedGather) {
        remoteOutput.resize(a->channelCount);
        remoteOutputToken.resize(a->channelCount);
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        remoteInput[i] =
            ccu::GetResByChannel<ccu::Variable>(a->channels[i], kInputId);
        remoteInputToken[i] =
            ccu::GetResByChannel<ccu::Variable>(a->channels[i], kInputTokenId);
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            a->channels[i], input, kInputId, kChannelCkeId, 1U << kInputId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            a->channels[i], inputToken, kInputTokenId, kChannelCkeId,
            1U << kInputTokenId));
        if (fusedGather) {
            remoteOutput[i] =
                ccu::GetResByChannel<ccu::Variable>(a->channels[i], kOutputId);
            remoteOutputToken[i] =
                ccu::GetResByChannel<ccu::Variable>(
                    a->channels[i], kOutputTokenId);
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                a->channels[i], output, kOutputId, kChannelCkeId,
                1U << kOutputId));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                a->channels[i], outputToken, kOutputTokenId, kChannelCkeId,
                1U << kOutputTokenId));
        }
    }
    const uint32_t readyBits =
        (1U << kInputId) | (1U << kInputTokenId) |
        (fusedGather ? ((1U << kOutputId) | (1U << kOutputTokenId)) : 0U);
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(
            ccu::NotifyWait(a->channels[i], kChannelCkeId, readyBits));
    }

    ccu::LocalAddr inputAddr;
    inputAddr.addr = input;
    inputAddr.addr += sliceOffset;
    inputAddr.token = inputToken;
    ccu::LocalAddr outputAddr;
    outputAddr.addr = output;
    outputAddr.addr += sliceOffset;
    outputAddr.token = outputToken;
    ccu::LocalAddr scratchAddr;
    scratchAddr.addr = scratch;
    scratchAddr.addr += sliceOffset;
    scratchAddr.token = scratchToken;
    ccu::LocalAddr partial =
        a->includeLocalInput ? outputAddr : scratchAddr;

    // 512 KiB full-reduce 为 8 个 64 KiB LoopGroup；拓扑一每 rank
    // 只归约 128 KiB owner slice，因此固定为 2 个 LoopGroup。
    ccu::Variable loopIter;
    loopIter = fusedGather ? 2U : 8U;
    ccu::Array<ccu::Event> events(kSmallTiledParallelDim);
    ccu::Array<ccu::CcuBuffer> buffers(
        kSmallTiledParallelDim * kSmallTiledInterleave);

    uint32_t begin = 0;
    if (!a->includeLocalInput) {
        ccu::RemoteAddr firstRemote;
        firstRemote.addr = remoteInput[0];
        firstRemote.addr += sliceOffset;
        firstRemote.token = remoteInputToken[0];
        ccu::Event seedEvent;
        CCU_IF(sliceSize != 0) {
            CCU_CHK_RET(ccu::Read(a->channels[0], partial, firstRemote,
                                  sliceSize, seedEvent, 1));
            CCU_CHK_RET(ccu::EventWait(seedEvent, 1));
        }
        begin = 1;
    }

    bool firstBatch = true;
    for (; begin < a->channelCount; begin += kSmallTiledMaxRemote) {
        const uint32_t batchSize = std::min<uint32_t>(
            kSmallTiledMaxRemote, a->channelCount - begin);
        std::vector<ccu::RemoteAddr> remoteSrc(batchSize);
        for (uint32_t i = 0; i < batchSize; ++i) {
            remoteSrc[i].addr = remoteInput[begin + i];
            remoteSrc[i].addr += sliceOffset;
            remoteSrc[i].token = remoteInputToken[begin + i];
        }
        const ccu::LocalAddr localSrc =
            (a->includeLocalInput && firstBatch) ? inputAddr : partial;
        CCU_CHK_RET(SmallTiledBatchReduce(
            &a->channels[begin], batchSize, partial, remoteSrc, localSrc,
            loopIter, a->dataType, a->reduceType, events, buffers));
        firstBatch = false;
    }

    // 拓扑一所有 channel 位于同一 die。每 rank 只计算自己的 128 KiB
    // slice，并在同一个 mission 内直接写到三个 peer，省掉第二次 Host launch。
    if (fusedGather) {
        ccu::Event gatherEvent;
        uint32_t gatherMask = 0;
        for (uint32_t i = 0; i < a->channelCount; ++i) {
            ccu::RemoteAddr remote;
            remote.addr = remoteOutput[i];
            remote.addr += sliceOffset;
            remote.token = remoteOutputToken[i];
            CCU_CHK_RET(ccu::Write(a->channels[i], remote, outputAddr,
                                   sliceSize, gatherEvent, 1U << i));
            gatherMask |= 1U << i;
        }
        CCU_CHK_RET(ccu::EventWait(gatherEvent, gatherMask));
    }

    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyRecord(
            a->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(
            a->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    return CCU_SUCCESS;
}

CcuResult CcuLargeKernel(CcuKernelArg arg, uint32_t expectedRankSize)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if (!IsTopologyKernelArgValid(kernelArg, expectedRankSize)) {
        return CCU_E_PARA;
    }
    return CcuKernel(arg);
}

CcuResult CcuFixedGatherKernelImpl(CcuKernelArg arg, uint32_t expectedRankSize)
{
    auto *a = static_cast<CcuKernelArgAllReduce *>(arg);
    if (!IsTopologyKernelArgValid(a, expectedRankSize)) {
        return CCU_E_PARA;
    }

    ccu::Variable unusedInput, output, unusedInputToken, outputToken;
    ccu::Variable unusedScratch, unusedScratchToken, off, size, unusedPhase;
    uint32_t n = 0;
    CCU_CHK_RET(ccu::LoadArg(unusedInput, n++));
    CCU_CHK_RET(ccu::LoadArg(output, n++));
    CCU_CHK_RET(ccu::LoadArg(unusedInputToken, n++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, n++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratch, n++));
    CCU_CHK_RET(ccu::LoadArg(unusedScratchToken, n++));
    CCU_CHK_RET(ccu::LoadArg(off, n++));
    CCU_CHK_RET(ccu::LoadArg(size, n++));
    CCU_CHK_RET(ccu::LoadArg(unusedPhase, n++));

    std::vector<ccu::Variable> remoteOutput(a->channelCount);
    std::vector<ccu::Variable> remoteOutputToken(a->channelCount);
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        remoteOutput[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kOutputId);
        remoteOutputToken[i] =
            ccu::GetResByChannel<ccu::Variable>(a->channels[i], kOutputTokenId);
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], output, kOutputId,
                                                  kChannelCkeId, 1U << kOutputId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], outputToken, kOutputTokenId,
                                                  kChannelCkeId, 1U << kOutputTokenId));
    }
    constexpr uint32_t outputBits = (1U << kOutputId) | (1U << kOutputTokenId);
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(a->channels[i], kChannelCkeId, outputBits));
    }

    ccu::LocalAddr outputAddr;
    outputAddr.addr = output;
    outputAddr.addr += off;
    outputAddr.token = outputToken;
    ccu::Event event;
    CCU_IF(size != 0) {
        for (uint32_t i = 0; i < a->channelCount; ++i) {
            ccu::RemoteAddr remote;
            remote.addr = remoteOutput[i];
            remote.addr += off;
            remote.token = remoteOutputToken[i];
            CCU_CHK_RET(ccu::Write(a->channels[i], remote, outputAddr, size, event,
                                    static_cast<uint16_t>(1U << i)));
        }
        const uint16_t writeMask =
            static_cast<uint16_t>((1U << a->channelCount) - 1U);
        CCU_CHK_RET(ccu::EventWait(event, writeMask));
    }

    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyRecord(a->channels[i], kChannelCkeId,
                                      1U << kPostSyncId));
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(a->channels[i], kChannelCkeId,
                                     1U << kPostSyncId));
    }
    return CCU_SUCCESS;
}

// 拓扑三 8+8 专用图：同一 die 上的远端读先并行落到 CCU buffer，
// 再做一次本地归约，避免 V2.3 图逐 channel ReadReduce 的串行链。
// 跨 die 的 combine/phase1 仍沿用已有 Host barrier 和 output 发布协议。
[[maybe_unused]] CcuResult CcuTopo3KernelImpl(CcuKernelArg arg, bool withAllGather)
{
    auto *a = static_cast<CcuKernelArgAllReduce *>(arg);
    if (!IsTopologyKernelArgValid(a, 16) || a->channelCount >= 15) {
        return CCU_E_PARA;
    }

    ccu::Variable input, output, inputToken, outputToken, scratch, scratchToken;
    ccu::Variable off, size, phase;
    uint32_t n = 0;
    CCU_CHK_RET(ccu::LoadArg(input, n++));
    CCU_CHK_RET(ccu::LoadArg(output, n++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, n++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, n++));
    CCU_CHK_RET(ccu::LoadArg(scratch, n++));
    CCU_CHK_RET(ccu::LoadArg(scratchToken, n++));
    CCU_CHK_RET(ccu::LoadArg(off, n++));
    CCU_CHK_RET(ccu::LoadArg(size, n++));
    CCU_CHK_RET(ccu::LoadArg(phase, n++));

    std::vector<ccu::Variable> remoteInput(a->channelCount);
    std::vector<ccu::Variable> remoteOutput(a->channelCount);
    std::vector<ccu::Variable> remoteInputToken(a->channelCount);
    std::vector<ccu::Variable> remoteOutputToken(a->channelCount);
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        remoteInput[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kInputId);
        remoteOutput[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kOutputId);
        remoteInputToken[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kInputTokenId);
        remoteOutputToken[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kOutputTokenId);
    }

    CCU_IF(phase == 0) {
        for (uint32_t i = 0; i < a->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], input, kInputId,
                                                      kChannelCkeId, 1U << kInputId));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], inputToken, kInputTokenId,
                                                      kChannelCkeId, 1U << kInputTokenId));
        }
        constexpr uint32_t inputBits = (1U << kInputId) | (1U << kInputTokenId);
        for (uint32_t i = 0; i < a->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(a->channels[i], kChannelCkeId, inputBits));
        }
    } CCU_ELSE {
        for (uint32_t i = 0; i < a->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], output, kOutputId,
                                                      kChannelCkeId, 1U << kOutputId));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], outputToken, kOutputTokenId,
                                                      kChannelCkeId, 1U << kOutputTokenId));
        }
        constexpr uint32_t outputBits = (1U << kOutputId) | (1U << kOutputTokenId);
        for (uint32_t i = 0; i < a->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(a->channels[i], kChannelCkeId, outputBits));
        }
    }

    ccu::LocalAddr inputAddr;
    inputAddr.addr = input;
    inputAddr.addr += off;
    inputAddr.token = inputToken;
    ccu::LocalAddr outputAddr;
    outputAddr.addr = output;
    outputAddr.addr += off;
    outputAddr.token = outputToken;
    ccu::LocalAddr scratchAddr;
    scratchAddr.addr = scratch;
    scratchAddr.addr += off;
    scratchAddr.token = scratchToken;
    ccu::LocalAddr partial = a->includeLocalInput ? outputAddr : scratchAddr;
    ccu::Event event;

    CCU_IF(size != 0) {
        CCU_IF(phase == 0) {
            const uint32_t sourceCount = a->channelCount + (a->includeLocalInput ? 1U : 0U);
            ccu::Array<ccu::CcuBuffer> buffers(sourceCount);
            for (uint32_t i = 0; i < a->channelCount; ++i) {
                ccu::RemoteAddr remote;
                remote.addr = remoteInput[i];
                remote.addr += off;
                remote.token = remoteInputToken[i];
                CCU_CHK_RET(ccu::Read(a->channels[i], buffers[i], remote, size, event,
                                       static_cast<uint16_t>(1U << i)));
            }
            if (a->includeLocalInput) {
                CCU_CHK_RET(ccu::LocalCopy(buffers[a->channelCount], inputAddr, size, event,
                                           static_cast<uint16_t>(1U << a->channelCount)));
            }
            const uint16_t sourceMask = static_cast<uint16_t>((1U << sourceCount) - 1U);
            CCU_CHK_RET(ccu::EventWait(event, sourceMask));
            if (sourceCount > 1) {
                CCU_CHK_RET(ccu::LocalReduce(&buffers[0], sourceCount, a->dataType,
                                              a->dataType, a->reduceType, size, event, 1));
                CCU_CHK_RET(ccu::EventWait(event, 1));
            }
            CCU_CHK_RET(ccu::LocalCopy(partial, buffers[0], size, event, 1));
            CCU_CHK_RET(ccu::EventWait(event, 1));
        } CCU_ELSE {
            if (withAllGather) {
                for (uint32_t i = 0; i < a->channelCount; ++i) {
                    ccu::RemoteAddr remote;
                    remote.addr = remoteOutput[i];
                    remote.addr += off;
                    remote.token = remoteOutputToken[i];
                    CCU_CHK_RET(ccu::Write(a->channels[i], remote, outputAddr, size, event,
                                            static_cast<uint16_t>(1U << i)));
                }
                const uint16_t writeMask = static_cast<uint16_t>((1U << a->channelCount) - 1U);
                CCU_CHK_RET(ccu::EventWait(event, writeMask));
            }
        }
    }

    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyRecord(a->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(a->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    return CCU_SUCCESS;
}

CcuResult CcuLargePipeKernelImpl(CcuKernelArg arg)
{
    auto *a = static_cast<CcuKernelArgAllReduce *>(arg);
    if (a == nullptr || (a->rankSize != 4 && a->rankSize != 12 && a->rankSize != 16) ||
        a->rankId >= a->rankSize || a->channelCount == 0 ||
        a->channelCount >= kPipeInterleave) {
        return CCU_E_PARA;
    }

    ccu::Variable input, output, inputToken, outputToken, scratch, scratchToken;
    ccu::Variable off, size, phase, goOffset, goLoop, goParallel, goResidual;
    uint32_t n = 0;
    CCU_CHK_RET(ccu::LoadArg(input, n++));
    CCU_CHK_RET(ccu::LoadArg(output, n++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, n++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, n++));
    CCU_CHK_RET(ccu::LoadArg(scratch, n++));
    CCU_CHK_RET(ccu::LoadArg(scratchToken, n++));
    CCU_CHK_RET(ccu::LoadArg(off, n++));
    CCU_CHK_RET(ccu::LoadArg(size, n++));
    CCU_CHK_RET(ccu::LoadArg(phase, n++));
    CCU_CHK_RET(ccu::LoadArg(goOffset, n++));
    CCU_CHK_RET(ccu::LoadArg(goLoop, n++));
    CCU_CHK_RET(ccu::LoadArg(goParallel, n++));
    CCU_CHK_RET(ccu::LoadArg(goResidual, n++));

    std::vector<ccu::Variable> rin(a->channelCount), rtok(a->channelCount);
    std::vector<ccu::Variable> rout(a->channelCount), rtokout(a->channelCount);
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        rin[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kInputId);
        rtok[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kInputTokenId);
        rout[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kOutputId);
        rtokout[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kOutputTokenId);
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], input, kInputId, kChannelCkeId,
                                                   1U << kInputId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], inputToken, kInputTokenId,
                                                   kChannelCkeId, 1U << kInputTokenId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], output, kOutputId, kChannelCkeId,
                                                   1U << kOutputId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], outputToken, kOutputTokenId,
                                                   kChannelCkeId, 1U << kOutputTokenId));
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(a->channels[i], kChannelCkeId,
            (1U << kInputId) | (1U << kInputTokenId) |
            (1U << kOutputId) | (1U << kOutputTokenId)));
    }

    ccu::LocalAddr inAddr;
    inAddr.addr = input;
    inAddr.token = inputToken;
    ccu::LocalAddr outAddr;
    outAddr.addr = output;
    outAddr.token = outputToken;
    ccu::LocalAddr scratchAddr;
    scratchAddr.addr = scratch;
    scratchAddr.token = scratchToken;
    inAddr.addr += off;
    outAddr.addr += off;
    scratchAddr.addr += off;
    ccu::LocalAddr partial = a->includeLocalInput ? outAddr : scratchAddr;

    CCU_IF(size != 0) {
        CCU_IF(phase == 0) {
            ccu::Array<ccu::Event> events(kPipeParallelDim);
            ccu::Array<ccu::CcuBuffer> buffers(kPipeParallelDim * kPipeInterleave);
            std::vector<ccu::RemoteAddr> rem(a->channelCount);
            for (uint32_t i = 0; i < a->channelCount; ++i) {
                rem[i].addr = rin[i];
                rem[i].addr += off;
                rem[i].token = rtok[i];
            }
            const ccu::LocalAddr local = a->includeLocalInput ? inAddr : inAddr;
            std::unique_ptr<ccu::Func> body(new ccu::Func([&, local]() {
                for (uint32_t i = 0; i < a->channelCount; ++i) {
                    ccu::Read(a->channels[i], buffers[i], rem[i], size, events[0], 1U << i);
                }
                if (a->includeLocalInput) {
                    ccu::LocalCopy(buffers[a->channelCount], local, size, events[0],
                                   1U << a->channelCount);
                }
                const uint32_t count = a->channelCount + (a->includeLocalInput ? 1U : 0U);
                ccu::EventWait(events[0], static_cast<uint16_t>((1U << count) - 1));
                if (count > 1) {
                    ccu::LocalReduce(&buffers[0], count, a->dataType, a->dataType,
                                     a->reduceType, size, events[0], 1);
                    ccu::EventWait(events[0], 1);
                }
                ccu::LocalCopy(partial, buffers[0], size, events[0], 1);
                ccu::EventWait(events[0], 1);
            }));
            ccu::Variable loopCfg;
            loopCfg = PipeLoopParam(0, kPipeMemSlice * kPipeParallelDim, 0);
            loopCfg = loopCfg + goLoop;
            ccu::Variable parallelCfg;
            parallelCfg = PipeParallelParam(kPipeParallelDim - 1, 0, 1);
            ccu::Variable offsetCfg;
            offsetCfg = PipeOffsetParam(kPipeMemSlice, kPipeInterleave, 1);
            ccu::Loop loop(loopCfg, *body);
            CCU_IF(goLoop != 0) {
                std::vector<ccu::Loop> loops{loop};
                ccu::LoopGroup group(parallelCfg, offsetCfg, kPipeParallelDim, loops);
                (void)group;
            } CCU_ELSE {
                // Non-loop tail remains a correctness-safe serial reduction.
                ccu::Event event;
                uint32_t first = 0;
                if (a->includeLocalInput) {
                    CCU_CHK_RET(ccu::LocalCopy(partial, inAddr, size, event, 1));
                    CCU_CHK_RET(ccu::EventWait(event, 1));
                } else {
                    ccu::RemoteAddr r;
                    r.addr = rin[0];
                    r.token = rtok[0];
                    r.addr += off;
                    CCU_CHK_RET(ccu::Read(a->channels[0], partial, r, size, event, 1));
                    CCU_CHK_RET(ccu::EventWait(event, 1));
                    first = 1;
                }
                for (uint32_t i = first; i < a->channelCount; ++i) {
                    ccu::RemoteAddr r;
                    r.addr = rin[i];
                    r.token = rtok[i];
                    r.addr += off;
                    CCU_CHK_RET(ccu::ReadReduce(a->channels[i], partial, r, size,
                                                 a->dataType, a->reduceType, event, 1));
                    CCU_CHK_RET(ccu::EventWait(event, 1));
                }
            }
        } CCU_ELSE {
            ccu::Event event;
            for (uint32_t i = 0; i < a->channelCount; ++i) {
                ccu::RemoteAddr r;
                r.addr = rout[i];
                r.token = rtokout[i];
                r.addr += off;
                CCU_CHK_RET(ccu::Write(a->channels[i], r, outAddr, size, event, 1U << i));
            }
            CCU_CHK_RET(ccu::EventWait(event, static_cast<uint16_t>((1U << a->channelCount) - 1)));
        }
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyRecord(a->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(a->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuLargePipeKernel(CcuKernelArg arg)
{
    return CcuLargePipeKernelImpl(arg);
}

CcuResult CcuCombineKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CCU_E_PARA;
    }
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable sliceOffset;
    ccu::Variable sliceSize;
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(input, argId++));
    CCU_CHK_RET(ccu::LoadArg(output, argId++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(sliceOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(sliceSize, argId++));
    // Combine 图同样不依赖 runtime phase；省去第 9 个参数的图内加载。

    (void)ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[0], kInputId);
    ccu::LocalAddr dst;
    dst.addr = output;
    dst.addr += sliceOffset;
    dst.token = outputToken;
    ccu::LocalAddr src;
    src.addr = scratch;
    src.addr += sliceOffset;
    src.token = scratchToken;
    ccu::Event event;
    CCU_IF(sliceSize != 0) {
        if (kernelArg->combineScratch) {
            CCU_CHK_RET(ccu::LocalReduce(dst, src, sliceSize, kernelArg->dataType,
                                          kernelArg->reduceType, event, 1));
            CCU_CHK_RET(ccu::EventWait(event, 1));
        }
        if (kernelArg->combineInput) {
            ccu::LocalAddr localInput;
            localInput.addr = input;
            localInput.addr += sliceOffset;
            localInput.token = inputToken;
            CCU_CHK_RET(ccu::LocalReduce(dst, localInput, sliceSize, kernelArg->dataType,
                                          kernelArg->reduceType, event, 1));
            CCU_CHK_RET(ccu::EventWait(event, 1));
        }
    }
    return CCU_SUCCESS;
}

template <uint32_t kTopologyKind>
CcuResult CcuTopologyLaneKernel(CcuKernelArg arg)
{
    static_assert(kTopologyKind == 4 || kTopologyKind == 16,
                  "Only contest 4-rank and 16-rank lane graphs are supported");
    auto *a = static_cast<CcuKernelArgAllReduce *>(arg);
    if (a == nullptr || a->channelCount == 0 || a->rankId >= a->rankSize ||
        a->rankSize != kTopologyKind || a->topologyKind != kTopologyKind) {
        return CCU_E_PARA;
    }

    ccu::Variable in, out, inTok, outTok, scratch, scratchTok, off, size, phase;
    uint32_t n = 0;
    CCU_CHK_RET(ccu::LoadArg(in, n++));
    CCU_CHK_RET(ccu::LoadArg(out, n++));
    CCU_CHK_RET(ccu::LoadArg(inTok, n++));
    CCU_CHK_RET(ccu::LoadArg(outTok, n++));
    CCU_CHK_RET(ccu::LoadArg(scratch, n++));
    CCU_CHK_RET(ccu::LoadArg(scratchTok, n++));
    CCU_CHK_RET(ccu::LoadArg(off, n++));
    CCU_CHK_RET(ccu::LoadArg(size, n++));
    CCU_CHK_RET(ccu::LoadArg(phase, n++));

    std::vector<ccu::Variable> remoteIn(a->channelCount), remoteInToken(a->channelCount);
    std::vector<ccu::Variable> remoteOut(a->channelCount), remoteOutToken(a->channelCount);
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        remoteIn[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kInputId);
        remoteInToken[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kInputTokenId);
        remoteOut[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kOutputId);
        remoteOutToken[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kOutputTokenId);
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], in, kInputId,
            kChannelCkeId, 1U << kInputId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], inTok, kInputTokenId,
            kChannelCkeId, 1U << kInputTokenId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], out, kOutputId,
            kChannelCkeId, 1U << kOutputId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], outTok, kOutputTokenId,
            kChannelCkeId, 1U << kOutputTokenId));
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(a->channels[i], kChannelCkeId,
            (1U << kInputId) | (1U << kInputTokenId) |
            (1U << kOutputId) | (1U << kOutputTokenId)));
    }

    const uint32_t groupBase = a->rankId < 8 ? 0 : 8;
    constexpr uint32_t groupSize = 8;
    ccu::LocalAddr input; input.addr = in; input.addr += off; input.token = inTok;
    ccu::LocalAddr output; output.addr = out; output.addr += off; output.token = outTok;
    // LocalAddr/Variable 是 CCU 图节点句柄，不能先复制 output 再改 partial.addr，
    // 否则 non-owner die 会把后续 phase 的 output 一并重定向到 scratch。
    ccu::LocalAddr partial;
    if (a->includeLocalInput) {
        partial.addr = out; partial.addr += off; partial.token = outTok;
    } else {
        partial.addr = scratch; partial.addr += off; partial.token = scratchTok;
    }
    ccu::Event event;

    CCU_IF(size != 0) {
        CCU_IF(phase == 0) {
            uint32_t firstRemote = 0;
            if (a->includeLocalInput && !a->deferLocalInput) {
                CCU_CHK_RET(ccu::LocalCopy(partial, input, size, event, 1));
                CCU_CHK_RET(ccu::EventWait(event, 1));
            } else {
                firstRemote = a->channelCount;
                for (uint32_t i = 0; i < a->channelCount; ++i) {
                    const uint32_t peer = a->peerRanks[i];
                    const bool selected = kTopologyKind == 4 ||
                        (peer >= groupBase && peer < groupBase + groupSize);
                    if (!selected) continue;
                    ccu::RemoteAddr remote; remote.addr = remoteIn[i]; remote.addr += off;
                    remote.token = remoteInToken[i];
                    CCU_CHK_RET(ccu::Read(a->channels[i], partial, remote, size, event, 1));
                    CCU_CHK_RET(ccu::EventWait(event, 1));
                    firstRemote = i;
                    break;
                }
            }
            for (uint32_t i = 0; i < a->channelCount; ++i) {
                const uint32_t peer = a->peerRanks[i];
                const bool selected = kTopologyKind == 4 ||
                    (peer >= groupBase && peer < groupBase + groupSize);
                if (!selected || ((a->deferLocalInput || !a->includeLocalInput) &&
                                  i <= firstRemote)) continue;
                ccu::RemoteAddr remote; remote.addr = remoteIn[i]; remote.addr += off;
                remote.token = remoteInToken[i];
                CCU_CHK_RET(ccu::ReadReduce(a->channels[i], partial, remote, size,
                    a->dataType, a->reduceType, event, 1));
                CCU_CHK_RET(ccu::EventWait(event, 1));
            }
        } CCU_ELSE {
            CCU_IF(phase == 1) {
                if constexpr (kTopologyKind == 16) {
                    if (groupBase == 0) {
                    const uint32_t counterpart = a->rankId + 8;
                    for (uint32_t i = 0; i < a->channelCount; ++i) {
                        if (a->peerRanks[i] != counterpart) continue;
                        ccu::RemoteAddr remote; remote.addr = remoteOut[i]; remote.addr += off;
                        remote.token = remoteOutToken[i];
                        CCU_CHK_RET(ccu::ReadReduce(a->channels[i], output, remote, size,
                            a->dataType, a->reduceType, event, 1));
                        CCU_CHK_RET(ccu::EventWait(event, 1));
                        CCU_CHK_RET(ccu::Write(a->channels[i], remote, output, size, event, 1));
                        CCU_CHK_RET(ccu::EventWait(event, 1));
                    }
                    }
                }
            } CCU_ELSE {
                uint32_t writeMask = 0;
                for (uint32_t i = 0; i < a->channelCount; ++i) {
                    const uint32_t peer = a->peerRanks[i];
                    const bool selected = kTopologyKind == 4 ||
                        (peer >= groupBase && peer < groupBase + groupSize);
                    if (!selected) continue;
                    ccu::RemoteAddr remote; remote.addr = remoteOut[i]; remote.addr += off;
                    remote.token = remoteOutToken[i];
                    CCU_CHK_RET(ccu::Write(a->channels[i], remote, output, size,
                        event, 1U << i));
                    writeMask |= 1U << i;
                }
                if (writeMask != 0) CCU_CHK_RET(ccu::EventWait(event, writeMask));
            }
        }
    }

    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyRecord(a->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(a->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    return CCU_SUCCESS;
}

// 分层大包图：阶段 0 在本地组内归约，阶段 1 仅由两个组根交换，
// 阶段 3 由组根向组内成员广播。该入口暂不接入默认调度，先独立做 VM。
template <uint32_t kRankSize>
CcuResult CcuHierarchicalLargeKernel(CcuKernelArg arg)
{
    static_assert(kRankSize == 12 || kRankSize == 16, "hierarchical graph supports 12/16 ranks");
    auto *a = static_cast<CcuKernelArgAllReduce *>(arg);
    if (!IsTopologyKernelArgValid(a, kRankSize)) return CCU_E_PARA;

    ccu::Variable in, out, inTok, outTok, scratch, scratchTok, off, size, phase;
    uint32_t n = 0;
    CCU_CHK_RET(ccu::LoadArg(in, n++));
    CCU_CHK_RET(ccu::LoadArg(out, n++));
    CCU_CHK_RET(ccu::LoadArg(inTok, n++));
    CCU_CHK_RET(ccu::LoadArg(outTok, n++));
    CCU_CHK_RET(ccu::LoadArg(scratch, n++));
    CCU_CHK_RET(ccu::LoadArg(scratchTok, n++));
    CCU_CHK_RET(ccu::LoadArg(off, n++));
    CCU_CHK_RET(ccu::LoadArg(size, n++));
    CCU_CHK_RET(ccu::LoadArg(phase, n++));

    std::vector<ccu::Variable> rin(a->channelCount), rtin(a->channelCount);
    std::vector<ccu::Variable> rout(a->channelCount), rtout(a->channelCount);
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        rin[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kInputId);
        rtin[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kInputTokenId);
        rout[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kOutputId);
        rtout[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], kOutputTokenId);
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], in, kInputId, kChannelCkeId,
                                                  1U << kInputId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], inTok, kInputTokenId,
                                                  kChannelCkeId, 1U << kInputTokenId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], out, kOutputId, kChannelCkeId,
                                                  1U << kOutputId));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(a->channels[i], outTok, kOutputTokenId,
                                                  kChannelCkeId, 1U << kOutputTokenId));
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(a->channels[i], kChannelCkeId,
            (1U << kInputId) | (1U << kInputTokenId) |
            (1U << kOutputId) | (1U << kOutputTokenId)));
    }

    const uint32_t groupBase = a->localGroupSize != 0 ? a->localGroupBase :
        (a->rankId < 8 ? 0 : 8);
    const uint32_t root = groupBase;
    ccu::LocalAddr input, output, scratchAddr;
    input.addr = in; input.addr += off; input.token = inTok;
    output.addr = out; output.addr += off; output.token = outTok;
    scratchAddr.addr = scratch; scratchAddr.addr += off; scratchAddr.token = scratchTok;
    // direct-hier 模式沿用 V1：所有 die 都原地累积到 output，第二个
    // die 由 Host barrier 保证读取到第一个 die 的 partial。
    ccu::LocalAddr partial = output;
    ccu::Event event;

    CCU_IF(size != 0) {
        CCU_IF(phase == 0) {
            // HCCL Mesh 2-die 模板按最多 7 条远端通道分批，避免把
            // 拓扑二/三的全部 channel 放入同一个 mission。
            constexpr uint32_t kBatchRemote = 7;
            bool firstBatch = true;
            for (uint32_t begin = 0; begin < a->channelCount; begin += kBatchRemote) {
                uint32_t batchCount = std::min<uint32_t>(kBatchRemote, a->channelCount - begin);
                std::vector<uint32_t> selected;
                for (uint32_t i = begin; i < begin + batchCount; ++i) {
                    selected.push_back(i);
                }
                const bool localBatch = true;
                const uint32_t sourceCount = static_cast<uint32_t>(selected.size()) +
                    (localBatch ? 1U : 0U);
                if (sourceCount == 0) {
                    firstBatch = false;
                    continue;
                }
                ccu::Array<ccu::CcuBuffer> buffers(sourceCount);
                for (uint32_t j = 0; j < selected.size(); ++j) {
                    uint32_t i = selected[j];
                    ccu::RemoteAddr remote;
                    remote.addr = rin[i]; remote.addr += off; remote.token = rtin[i];
                    CCU_CHK_RET(ccu::Read(a->channels[i], buffers[j], remote, size, event,
                                           static_cast<uint16_t>(1U << j)));
                }
                if (localBatch) {
                    ccu::LocalAddr local = firstBatch && a->initOutput ? input : partial;
                    CCU_CHK_RET(ccu::LocalCopy(buffers[sourceCount - 1], local, size, event,
                                               static_cast<uint16_t>(1U << (sourceCount - 1))));
                }
                CCU_CHK_RET(ccu::EventWait(event, static_cast<uint16_t>((1U << sourceCount) - 1)));
                if (sourceCount > 1) {
                    CCU_CHK_RET(ccu::LocalReduce(&buffers[0], sourceCount, a->dataType,
                                                 a->dataType, a->reduceType, size, event, 1));
                    CCU_CHK_RET(ccu::EventWait(event, 1));
                }
                CCU_CHK_RET(ccu::LocalCopy(partial, buffers[0], size, event, 1));
                CCU_CHK_RET(ccu::EventWait(event, 1));
                firstBatch = false;
            }
        } CCU_ELSE {
            CCU_IF(phase == 1) {
                // phase0 完成 die 内归约，Host combine 随后合并两个 die；
                // phase1 只负责将合并后的组根结果广播到本地 peer。
                if (a->rankId != root) {
                    for (uint32_t i = 0; i < a->channelCount; ++i) {
                        if (a->peerRanks[i] != root) continue;
                        ccu::RemoteAddr remote;
                        remote.addr = rout[i]; remote.addr += off; remote.token = rtout[i];
                        CCU_CHK_RET(ccu::Read(a->channels[i], output, remote, size, event, 1));
                        CCU_CHK_RET(ccu::EventWait(event, 1));
                        break;
                    }
                }
            }
        }
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyRecord(a->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(a->channels[i], kChannelCkeId, 1U << kPostSyncId));
    }
    return CCU_SUCCESS;
}

CcuResult CcuSmall4Kernel(CcuKernelArg arg) { return CcuSmallKernel(arg, 4); }
CcuResult CcuSmall12Kernel(CcuKernelArg arg) { return CcuSmallKernel(arg, 12); }
CcuResult CcuSmall16Kernel(CcuKernelArg arg) { return CcuSmallKernel(arg, 16); }
CcuResult CcuSmall4TiledFusedKernel(CcuKernelArg arg)
{
    return CcuSmallTiledKernel(arg, 4, true);
}
CcuResult CcuSmall12TiledKernel(CcuKernelArg arg)
{
    return CcuSmallTiledKernel(arg, 12, false);
}
CcuResult CcuSmall16TiledKernel(CcuKernelArg arg)
{
    return CcuSmallTiledKernel(arg, 16, false);
}
CcuResult CcuSmall4LaneKernel(CcuKernelArg arg)
{
    return CcuTopologyLaneKernel<4>(arg);
}
CcuResult CcuSmall16HierKernel(CcuKernelArg arg)
{
    return CcuTopologyLaneKernel<16>(arg);
}
CcuResult CcuHierarchicalLarge12Kernel(CcuKernelArg arg)
{
    return CcuHierarchicalLargeKernel<12>(arg);
}
CcuResult CcuHierarchicalLarge16Kernel(CcuKernelArg arg)
{
    return CcuHierarchicalLargeKernel<16>(arg);
}
CcuResult CcuLarge4Kernel(CcuKernelArg arg) { return CcuLargeKernel(arg, 4); }
CcuResult CcuLarge12Kernel(CcuKernelArg arg) { return CcuLargeKernel(arg, 12); }
CcuResult CcuLarge16Kernel(CcuKernelArg arg) { return CcuLargeKernel(arg, 16); }
CcuResult CcuTopo1LargeReduce4Kernel(CcuKernelArg arg)
{
    return CcuSmallKernel(arg, 4);
}
CcuResult CcuTopo1LargeGather4Kernel(CcuKernelArg arg)
{
    return CcuFixedGatherKernelImpl(arg, 4);
}
CcuResult CcuTopo2LargeReduce12Kernel(CcuKernelArg arg)
{
    return CcuSmallKernel(arg, 12);
}
CcuResult CcuTopo2LargeGather12Kernel(CcuKernelArg arg)
{
    return CcuFixedGatherKernelImpl(arg, 12);
}
CcuResult CcuTopo3SmallReduce8x8Kernel(CcuKernelArg arg)
{
    return CcuSmallKernel(arg, 16);
}
CcuResult CcuTopo3LargeReduce8x8Kernel(CcuKernelArg arg)
{
    // 固定 ReduceScatter 图：Host 传入 rank-owned slice offset/size，图中
    // 只执行 phase0 partial，不再保留 runtime phase 分支。
    return CcuSmallKernel(arg, 16);
}
CcuResult CcuTopo3LargeGather8x8Kernel(CcuKernelArg arg)
{
    // 固定 AllGather 图：每个 rank 只发布自己已经归约完成的 owner slice；
    // 所有 rank 的不同 offset 合起来构成完整输出。
    return CcuFixedGatherKernelImpl(arg, 16);
}

} // namespace ops_hccl
