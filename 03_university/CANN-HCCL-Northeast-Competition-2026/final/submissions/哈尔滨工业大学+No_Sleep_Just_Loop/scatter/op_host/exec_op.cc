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
#include <limits>
#include <vector>

#include <ccu/ccu_res.h>

#include "log.h"
#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {

HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle handle, const void *args, uint32_t argCount)
{
    CcuResult result = HcommCcuKernelLaunch(thread, handle, args, argCount);
    CHK_PRT_RET(result != CCU_SUCCESS,
        HCCL_ERROR("[ExecOp] CCU kernel launch failed, ccuRet -> %d", result), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

bool IsKernelPresent(const AlgResourceCtx &resCtx, uint32_t index)
{
    return index != INVALID_VALUE_RANKID && index < resCtx.ccuKernels.size();
}

// 代理内部切分点(proxyBytes / closDirectBytes 的分界)的对齐粒度。
// 512 同时是 CCL buffer 与各类元素大小的整数倍, 所以对齐之后
// "root 的 Clos 直发段长度"、"代转者的落点偏移"、"scratch 槽位步长"三者在
// 长度和偏移上都是对齐的, 两段仍精确覆盖 sliceBytes 而不多写一个字节。
constexpr uint64_t PROXY_SPLIT_ALIGN = 512;

// 与 scatter.cc WAVE_MIN_SEND_BYTES 同一常量, 禁止另写 512*1024。
constexpr uint64_t WAVE_MIN_SEND_BYTES = 1ULL * 1024 * 1024;

inline bool IsSmallScatter(uint64_t sendBytes)
{
    return sendBytes < WAVE_MIN_SEND_BYTES;
}

// ===========================================================================
// 路径 1: RootStar (512KB)
//   taskArgs 按角色与 doSelfCopy 精简；sliceBytes/selfOffset/peerOffsets 已在注册期固化:
//     root + 自留块 : {inputPtr, inputToken, outputPtr, outputToken}   4 项
//     root 无自留块 : {inputPtr, inputToken}                           2 项
//     peer          : {outputPtr, outputToken}                         2 项
// ===========================================================================
HcclResult ExecRootStar(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t sliceBytes,
    uint64_t sendBytes)
{
    const bool isRootRank = (param.myRank == param.root);
    uint64_t inputToken = 0;
    if (isRootRank) {
        // 非 root 允许传 null sendBuf, 不查询它的 token
        CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), sendBytes, &inputToken));
    }
    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), sliceBytes, &outputToken));

    const uint64_t inPtr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outPtr = reinterpret_cast<uint64_t>(param.outputPtr);

    if (isRootRank) {
        const std::array<uint64_t, 4> rootSelfArgs = {inPtr, inputToken, outPtr, outputToken};
        const std::array<uint64_t, 2> rootNoSelfArgs = {inPtr, inputToken};

        const bool hasMesh = IsKernelPresent(resCtx, resCtx.meshKernelIndex);
        const bool hasClos = IsKernelPresent(resCtx, resCtx.closKernelIndex);

        // E4: 仅 SMALL Root 在 Launch 前 Host 拷自留块。LARGE 零 diff。
        if (IsSmallScatter(sendBytes)) {
            const uint64_t maxUint64 = std::numeric_limits<uint64_t>::max();
            CHK_PRT_RET(param.rankSize == 0 || sliceBytes > maxUint64 / param.rankSize,
                HCCL_ERROR("[ExecOp] SMALL HostSelfCopy size overflows uint64_t"), HCCL_E_PARA);
            CHK_PRT_RET(param.root >= param.rankSize,
                HCCL_ERROR("[ExecOp] SMALL HostSelfCopy root %u >= rankSize %u",
                    param.root, param.rankSize), HCCL_E_PARA);
            const uint64_t selfOffset = static_cast<uint64_t>(param.root) * sliceBytes;
            CHK_PRT_RET(inPtr > maxUint64 - selfOffset,
                HCCL_ERROR("[ExecOp] SMALL HostSelfCopy src address overflows"), HCCL_E_PARA);
            CHK_PRT_RET(resCtx.threads.empty(),
                HCCL_ERROR("[ExecOp] SMALL HostSelfCopy needs a CCU thread"), HCCL_E_INTERNAL);
            void *src = reinterpret_cast<void *>(inPtr + selfOffset);
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, src, sliceBytes)));
            HCCL_INFO("[ExecOp] SMALL HostSelfCopy root=%u off=%llu len=%llu",
                param.root, static_cast<unsigned long long>(selfOffset),
                static_cast<unsigned long long>(sliceBytes));
        }

        if (hasMesh && hasClos) {
            CHK_PRT_RET(resCtx.threads.size() < 2,
                HCCL_ERROR("[ExecOp] RootStar mixed path requires two CCU threads"), HCCL_E_INTERNAL);
            // CheckerV3 要求从属 thread 以 WAIT 开始。thread0 先放行 thread1，
            // 两条 thread 分别下发 Mesh/Clos，最后由 thread1 回通知 thread0 会合。
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[resCtx.meshKernelIndex],
                rootSelfArgs.data(), static_cast<uint32_t>(rootSelfArgs.size())));
            CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[resCtx.closKernelIndex],
                rootNoSelfArgs.data(), static_cast<uint32_t>(rootNoSelfArgs.size())));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
            return HCCL_SUCCESS;
        }

        // 单 kernel(纯 Mesh 或纯 Clos): 自留块归它
        const uint32_t kernelIndex = hasMesh ? resCtx.meshKernelIndex : resCtx.closKernelIndex;
        return LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[kernelIndex],
            rootSelfArgs.data(), static_cast<uint32_t>(rootSelfArgs.size()));
    }

    // 非 root: 只有一条到 Root 的 channel, 一个 kernel
    const std::array<uint64_t, 2> peerArgs = {outPtr, outputToken};
    const uint32_t kernelIndex = IsKernelPresent(resCtx, resCtx.meshKernelIndex)
        ? resCtx.meshKernelIndex : resCtx.closKernelIndex;
    return LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[kernelIndex],
        peerArgs.data(), static_cast<uint32_t>(peerArgs.size()));
}

// ===========================================================================
// 路径 2: Wave (400M+4B)
//   taskArgs 布局由 ScatterKernelArgId 给出, 共 SCATTER_KERNEL_ARG_COUNT 项。
// ===========================================================================
HcclResult ExecWave(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t sliceBytes,
    uint64_t elementSize, uint64_t sendBytes)
{
    const uint64_t maxUint64 = std::numeric_limits<uint64_t>::max();
    CHK_PRT_RET(elementSize == 0 || param.count > maxUint64 / elementSize,
        HCCL_ERROR("[ExecOp] recvCount byte size overflows uint64_t"), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize == 0 || sliceBytes > maxUint64 / param.rankSize,
        HCCL_ERROR("[ExecOp] Scatter send size overflows uint64_t"), HCCL_E_PARA);

    // CCU 单次传输上限 256 MiB; 两段 wave 覆盖全部计分大包, 且余量保持 FP32 对齐。
    const uint64_t maxChunkBytes = (static_cast<uint64_t>(MAX_DATA_SIZE) / elementSize) * elementSize;
    CHK_PRT_RET(maxChunkBytes == 0 || sliceBytes > maxChunkBytes * 2,
        HCCL_ERROR("[ExecOp] sliceBytes %llu needs more than two CCU waves",
            static_cast<unsigned long long>(sliceBytes)), HCCL_E_PARA);
    const uint64_t chunk0Bytes = std::min(sliceBytes, maxChunkBytes);
    const uint64_t chunk1Bytes = sliceBytes - chunk0Bytes;

    uint64_t inputToken = 0;
    if (param.myRank == param.root) {
        // 非 root 允许传 null sendBuf, 不查询它的 token
        CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), sendBytes, &inputToken));
    }
    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), sliceBytes, &outputToken));

    const std::array<uint64_t, SCATTER_KERNEL_ARG_COUNT> taskArgs = {
        reinterpret_cast<uint64_t>(param.inputPtr),   // SCATTER_ARG_INPUT
        inputToken,                                   // SCATTER_ARG_INPUT_TOKEN
        reinterpret_cast<uint64_t>(param.outputPtr),  // SCATTER_ARG_OUTPUT
        outputToken,                                  // SCATTER_ARG_OUTPUT_TOKEN
        sliceBytes,                                   // SCATTER_ARG_SLICE_BYTES
        chunk0Bytes,                                  // SCATTER_ARG_CHUNK0_BYTES
        chunk1Bytes,                                  // SCATTER_ARG_CHUNK1_BYTES
    };

    const bool hasMesh = IsKernelPresent(resCtx, resCtx.meshKernelIndex);
    const bool hasClos = IsKernelPresent(resCtx, resCtx.closKernelIndex);
    CHK_PRT_RET(!hasMesh && !hasClos, HCCL_ERROR("[ExecOp] no Wave kernel registered"), HCCL_E_INTERNAL);

    if (hasMesh && hasClos) {
        CHK_PRT_RET(resCtx.threads.size() < 2,
            HCCL_ERROR("[ExecOp] two kernels require two CCU threads"), HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[resCtx.meshKernelIndex],
            taskArgs.data(), SCATTER_KERNEL_ARG_COUNT));
        CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[resCtx.closKernelIndex],
            taskArgs.data(), SCATTER_KERNEL_ARG_COUNT));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    const uint32_t kernelIndex = hasMesh ? resCtx.meshKernelIndex : resCtx.closKernelIndex;
    return LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[kernelIndex],
        taskArgs.data(), SCATTER_KERNEL_ARG_COUNT);
}

// ===========================================================================
// 路径 4: Wave-Star
//   Root mixed: thread0 Record -> thread1 Wait; Mesh->t0, Clos->t1; thread1 Record -> thread0 Wait
//   Peer: 1 Thread / 1 Kernel, 只下发 recvBuf 地址/token
//   Root taskArgs 仍为 7 项(与 WAVESTAR_* 对齐, 本轮不缩短);
//   Kernel Root 始终 LoadArg 这 7 项(含无 SelfCopy / Single-Wave 不用的 arg);
//   Peer 仍为 2 项, 与 Host peerArgs 一致。
//   sliceBytes / selfOffset / peerOffsets 已在注册期写入 KernelArg。
// ===========================================================================
HcclResult ExecWaveStar(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t sliceBytes,
    uint64_t elementSize, uint64_t sendBytes)
{
    const uint64_t maxUint64 = std::numeric_limits<uint64_t>::max();
    CHK_PRT_RET(elementSize == 0 || param.count > maxUint64 / elementSize,
        HCCL_ERROR("[ExecOp] recvCount byte size overflows uint64_t"), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize == 0 || sliceBytes > maxUint64 / param.rankSize,
        HCCL_ERROR("[ExecOp] Scatter send size overflows uint64_t"), HCCL_E_PARA);

    const uint64_t maxChunkBytes = (static_cast<uint64_t>(MAX_DATA_SIZE) / elementSize) * elementSize;
    CHK_PRT_RET(maxChunkBytes == 0 || sliceBytes > maxChunkBytes * 2,
        HCCL_ERROR("[ExecOp] sliceBytes %llu needs more than two CCU waves",
            static_cast<unsigned long long>(sliceBytes)), HCCL_E_PARA);
    const uint64_t chunk0Bytes = std::min(sliceBytes, maxChunkBytes);
    const uint64_t chunk1Bytes = sliceBytes - chunk0Bytes;

    const bool isRootRank = (param.myRank == param.root);
    const bool hasMesh = IsKernelPresent(resCtx, resCtx.meshKernelIndex);
    const bool hasClos = IsKernelPresent(resCtx, resCtx.closKernelIndex);
    CHK_PRT_RET(!hasMesh && !hasClos, HCCL_ERROR("[ExecOp] no WaveStar kernel registered"), HCCL_E_INTERNAL);

    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), sliceBytes, &outputToken));
    const uint64_t outPtr = reinterpret_cast<uint64_t>(param.outputPtr);

    if (!isRootRank) {
        const std::array<uint64_t, 2> peerArgs = {outPtr, outputToken};
        CHK_PRT_RET(resCtx.threads.empty(),
            HCCL_ERROR("[ExecOp] WaveStar peer requires one CCU thread"), HCCL_E_INTERNAL);
        const uint32_t kernelIndex = hasMesh ? resCtx.meshKernelIndex : resCtx.closKernelIndex;
        return LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[kernelIndex],
            peerArgs.data(), static_cast<uint32_t>(peerArgs.size()));
    }

    uint64_t inputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), sendBytes, &inputToken));
    const std::array<uint64_t, WAVESTAR_KERNEL_ARG_COUNT> rootArgs = {
        reinterpret_cast<uint64_t>(param.inputPtr),  // WAVESTAR_ARG_INPUT
        inputToken,                                  // WAVESTAR_ARG_INPUT_TOKEN
        outPtr,                                      // WAVESTAR_ARG_OUTPUT
        outputToken,                                 // WAVESTAR_ARG_OUTPUT_TOKEN
        sliceBytes,                                  // WAVESTAR_ARG_SLICE_BYTES
        chunk0Bytes,                                 // WAVESTAR_ARG_CHUNK0_BYTES
        chunk1Bytes,                                 // WAVESTAR_ARG_CHUNK1_BYTES
    };

    if (hasMesh && hasClos) {
        CHK_PRT_RET(resCtx.threads.size() < 2,
            HCCL_ERROR("[ExecOp] WaveStar mixed Root requires two CCU threads"), HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[resCtx.meshKernelIndex],
            rootArgs.data(), WAVESTAR_KERNEL_ARG_COUNT));
        CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[resCtx.closKernelIndex],
            rootArgs.data(), WAVESTAR_KERNEL_ARG_COUNT));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    const uint32_t kernelIndex = hasMesh ? resCtx.meshKernelIndex : resCtx.closKernelIndex;
    return LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[kernelIndex],
        rootArgs.data(), WAVESTAR_KERNEL_ARG_COUNT);
}

// ===========================================================================
// 路径 3: Proxy (512MB)
//   taskArgs 9 项; 代转者的两条流由 host 侧串行下发(Mesh 完成后才放行 Clos)。
// ===========================================================================
HcclResult ExecProxy(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t sliceBytes,
    uint64_t elementSize, uint64_t sendBytes)
{
    // 本路径不做分段, 单个 Write 的字节数即 sliceBytes, 必须落在 CCU 单次传输上限内。
    // (路径 2 用 two-wave 覆盖超过上限的情形, 那是它自己的机制, 不在这里共用。)
    CHK_PRT_RET(sliceBytes > MAX_DATA_SIZE,
        HCCL_ERROR("[ExecOp] sliceBytes %llu exceeds MAX_DATA_SIZE", static_cast<unsigned long long>(sliceBytes)),
        HCCL_E_PARA);

    uint64_t proxyBytes = 0;
    if (resCtx.proxyEnabled && resCtx.proxyDenominator != 0) {
        proxyBytes = (sliceBytes / resCtx.proxyDenominator) * resCtx.proxyNumerator;
        // 多目标覆盖(仅 4x3 中档): 每个代转者代转 k 个目标, 每目标只代转 1/k 尾段。
        // 总代理量 F·k·(p/k)·S 与 k 无关; k==1 时本行不执行。
        if (resCtx.proxyTargetsPerForwarder > 1) {
            proxyBytes /= resCtx.proxyTargetsPerForwarder;
        }
        proxyBytes = (proxyBytes / elementSize) * elementSize; // 元素对齐
        // 内部切分点也对齐到 512B。总块长对齐不代表切分点对齐 —— 按比例算出的
        // proxyBytes 对 512 取余往往是几百字节(2x8 上曾为 340), 于是 root 的
        // Clos 直发段长度、代转者的落点偏移、scratch 槽位步长全都落在非对齐地址上。
        // 512 是元素大小的整数倍, 所以这不会破坏元素边界; 两段仍精确覆盖 sliceBytes。
        proxyBytes = (proxyBytes / PROXY_SPLIT_ALIGN) * PROXY_SPLIT_ALIGN;
        if (proxyBytes == 0 || proxyBytes >= sliceBytes) {
            proxyBytes = 0;
        }
    }
    const uint64_t closDirectBytes = sliceBytes - proxyBytes;

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), sendBytes, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), sliceBytes, &outputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
        resCtx.localBuffer.size, &scratchToken));

    // 分块流水(仅 4x3): 尾段均分两块, 元素对齐。非分块路径传占位值, 内核不会读它们。
    uint64_t stageChunk0Bytes = 0;
    uint64_t stageChunk1Bytes = 0;
    if (resCtx.multiStageRelay && proxyBytes > 0) {
        stageChunk0Bytes = ((proxyBytes / 2) / elementSize) * elementSize;
        stageChunk1Bytes = proxyBytes - stageChunk0Bytes;
        if (stageChunk0Bytes == 0 || stageChunk1Bytes == 0) {
            stageChunk0Bytes = proxyBytes;
            stageChunk1Bytes = 0;
        }
    }

    const std::array<uint64_t, 12> taskArgs = {
        reinterpret_cast<uint64_t>(param.inputPtr),   // arg0: sendBuf 基址
        inputToken,                                   // arg1: sendBuf token
        reinterpret_cast<uint64_t>(param.outputPtr),  // arg2: recvBuf 基址
        outputToken,                                  // arg3: recvBuf token
        sliceBytes,                                   // arg4: 每 rank 块字节数
        closDirectBytes,                              // arg5: 跨服块由 root 的 Clos 直发的字节数
        proxyBytes,                                   // arg6: 跨服块尾段字节数(0 = 不启用代理)
        reinterpret_cast<uint64_t>(resCtx.localBuffer.addr), // arg7: 本端 CCL buffer
        scratchToken,                                 // arg8: CCL buffer token
        0,                                            // arg9: 分块流水阶段号(逐阶段改写)
        stageChunk0Bytes,                             // arg10: 第 0 块字节数
        stageChunk1Bytes,                             // arg11: 第 1 块字节数
    };

    const bool hasMesh = IsKernelPresent(resCtx, resCtx.meshKernelIndex);
    const bool hasClos = IsKernelPresent(resCtx, resCtx.closKernelIndex);
    CHK_PRT_RET(!hasMesh && !hasClos,
        HCCL_ERROR("[ExecOp] no Proxy kernel was registered"), HCCL_E_INTERNAL);

    if (hasMesh && hasClos) {
        CHK_PRT_RET(resCtx.threads.size() < 2,
            HCCL_ERROR("[ExecOp] two kernels require two CCU threads"), HCCL_E_INTERNAL);
        // CheckerV3 要求从属流(threads[1])的第一个任务必须是 local WAIT
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[1], 0, CUSTOM_TIMEOUT)));

        const bool hasMeshB = IsKernelPresent(resCtx, resCtx.meshKernelBIndex);
        if (resCtx.multiStageRelay && resCtx.isForwarder) {
            // 分块流水: 阶段 0 = Mesh 收第 0 块(+ Clos 只做前同步);
            //           阶段 1 = Mesh 收第 1 块  ‖  Clos 转第 0 块   <- 重叠就在这里;
            //           阶段 2 = Clos 转第 1 块并收尾。
            // 阶段顺序由 host 的跨线程收敛保证, 所以不需要任何跨 Die 会合。
            for (uint32_t stage = 0; stage < 3; ++stage) {
                std::vector<uint64_t> stageArgs(taskArgs.begin(), taskArgs.end());
                stageArgs[9] = stage; // arg9 = 阶段号
                if (stage <= 1) {
                    CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[resCtx.meshKernelIndex],
                        stageArgs.data(), static_cast<uint32_t>(stageArgs.size())));
                }
                CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[resCtx.closKernelIndex],
                    stageArgs.data(), static_cast<uint32_t>(stageArgs.size())));
                // 本阶段两条流都跑完才开下一阶段。
                // ⚠️ 每个阶段必须用**不同的** notify idx: 复用同一个 idx 会形成
                // "两个 RECORD 供一个 WAIT", CheckerV3 直接判 many-to-one
                // (实测: resource=AICPU_NOTIFY, previousProducer/nextProducer 同 notifyId)。
                // 故 idx = 1 + stage, 与 CreateProxyResources 申请的 4 个 notify 对应。
                const uint32_t stageNotifyIdx = 1 + stage;
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], stageNotifyIdx)));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(resCtx.threads[1], stageNotifyIdx, CUSTOM_TIMEOUT)));
            }
        } else if (resCtx.isForwarder && hasMeshB) {
            // 去串行化(仅 400M 档): 代转者的 Mesh 组拆成
            //   TAIL(前同步 + 只等尾段就绪, 结束得早) -> SYNC(只做后同步栅栏, 结束得晚)。
            // 跨线程握手放在 TAIL 之后而不是整条 Mesh 流之后 —— TAIL 一结束 Clos kernel
            // 立刻开跑, 转发于是落在 root 的直发块**旁边**而不是后面。
            // 保护没有减少: SYNC 的后同步栅栏仍然保证本 rank 自己那块数据已落位,
            // 所以算子的完成时刻仍是两者较晚者, 没有提前返回。
            // 注: 不能合成一个 kernel —— CCU 要求一个 kernel 的 channel 全在同一 Die,
            //     Mesh 与 Clos 跨 Die, kernel 内做不了这个会合(v7 实测注册即失败)。
            CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[resCtx.meshKernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
            // 注意用 idx1: 开头的"放行"已占用 idx0, 复用会让一个 WAIT 被两个 RECORD 供给
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 1)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[1], 1, CUSTOM_TIMEOUT)));
            CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[resCtx.closKernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
            CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[resCtx.meshKernelBIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
        } else if (resCtx.isForwarder) {
            // 代转者: Clos kernel 必须等本 rank 的 Mesh kernel 把中转数据收完才能转发,
            // 故两条流串行。这不是额外开销: Clos 本来就要等那份数据。
            CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[resCtx.meshKernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
            // 注意用 idx1: 开头的"放行"已占用 idx0, 复用会让一个 WAIT 被两个 RECORD 供给
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 1)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[1], 1, CUSTOM_TIMEOUT)));
            CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[resCtx.closKernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
        } else if (hasMeshB) {
            // 去串行化(仅 400M 档): root 的 Mesh 组拆成
            //   SEED (前同步 + 只投递代理尾段, 结束得早) -> LOCAL(直发块 + 自留块 + 后同步)。
            // 两者同挂 threads[0] 并按 SEED -> LOCAL 顺序下发, kernel 边界(SEED 内的
            // EventWait 必须完成才结束)保证代理尾段**抢在直发块之前**占用 Mesh 链路。
            // 这是同一 kernel 内调换语句顺序做不到的 —— 平台实测那样改毫无效果。
            // Clos kernel 照旧挂 threads[1] 与它们并行。
            CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[resCtx.meshKernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
            CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[resCtx.meshKernelBIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
            CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[resCtx.closKernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
        } else {
            // 其余 rank: 两个 kernel 落在不同 IO Die, 各挂一条 thread 并行执行
            CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[resCtx.meshKernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
            CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[resCtx.closKernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
        }

        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    const uint32_t kernelIndex = hasMesh ? resCtx.meshKernelIndex : resCtx.closKernelIndex;
    return LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[kernelIndex],
        taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("[ExecOp] invalid CCU resource context"), HCCL_E_INTERNAL);

    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(),
        HCCL_ERROR("[ExecOp] unsupported dataType %d", static_cast<int>(param.dataType)), HCCL_E_NOT_SUPPORT);
    const uint64_t elementSize = sizeIt->second;
    const uint64_t sliceBytes = param.count * elementSize;
    CHK_PRT_RET(sliceBytes == 0, HCCL_ERROR("[ExecOp] zero-size Scatter is not supported"), HCCL_E_PARA);
    const uint64_t sendBytes = sliceBytes * param.rankSize;

    // 直接按选路时写入 ctx 的路径分发。路径决定 kernel 函数与 taskArgs 布局,
    // 必须与注册期一致 —— 因此不再在这里用 sendBytes 二次推导(那需要重新探测
    // 拓扑, 两处判据一旦不同步就会把 taskArgs 下发给布局不匹配的 kernel)。
    switch (static_cast<ScatterPath>(resCtx.path)) {
        case ScatterPath::PROXY:
            return ExecProxy(param, resCtx, sliceBytes, elementSize, sendBytes);
        case ScatterPath::WAVE:
            return ExecWave(param, resCtx, sliceBytes, elementSize, sendBytes);
        case ScatterPath::WAVE_STAR:
            return ExecWaveStar(param, resCtx, sliceBytes, elementSize, sendBytes);
        case ScatterPath::ROOT_STAR:
            return ExecRootStar(param, resCtx, sliceBytes, sendBytes);
        case ScatterPath::ROOT_STAR_PULL:
            HCCL_ERROR("[ExecOp] ROOT_STAR_PULL is disabled");
            return HCCL_E_NOT_SUPPORT;
        default:
            break;
    }
    HCCL_ERROR("[ExecOp] unknown scatter path %u in resource context", resCtx.path);
    return HCCL_E_INTERNAL;
}
} // namespace ops_hccl
