/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>
#include <algorithm>
#include <array>
#include <limits>

#include "custom.h"
#include "exec_op.h"

namespace {
constexpr uint64_t kPipeMemSlice = 4096;
constexpr uint32_t kPipeParallelDim = 16;
uint64_t PipeBits(uint16_t end) { return (uint64_t{1} << (end + 1)) - 1; }
uint64_t PipeParallelParam(uint64_t repeat, uint64_t index, uint64_t total)
{
    return ((repeat & PipeBits(7)) << 55) | ((index & PipeBits(7)) << 48) |
           ((total & PipeBits(7)) << 41);
}
void FillPipeArgs(uint64_t bytes, uint64_t *args)
{
    const uint64_t loopBytes = kPipeMemSlice * kPipeParallelDim;
    const uint64_t full = bytes / loopBytes;
    const uint64_t rem = bytes % loopBytes;
    // 当前实验 kernel 先只启用“整块 LoopGroup”；带尾部时回退到
    // 现有串行 ReadReduce，避免遗漏尾部数据。
    args[0] = 0;
    args[1] = rem == 0 ? full : 0;
    args[2] = 0;
    args[3] = 0;
    if (rem == 0) {
        return;
    }
    const uint64_t whole = rem / kPipeMemSlice;
    const uint64_t tail = rem % kPipeMemSlice;
    if (whole != 0 && tail == 0) {
        args[2] = PipeParallelParam(whole - 1, 0, 1);
        args[3] = kPipeMemSlice;
    } else if (whole == 0 && tail != 0) {
        args[2] = PipeParallelParam(0, 0, 1);
        args[3] = tail;
    } else if (whole != 0) {
        args[2] = PipeParallelParam(whole - 1, 1, 2);
        args[3] = tail;
    }
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    auto typeIt = SIZE_TABLE.find(param.dataType);
    if (typeIt == SIZE_TABLE.end() || param.rankSize == 0 ||
        param.rankSize > CcuKernelArgBase::kMaxRankSize) {
        return HCCL_E_NOT_SUPPORT;
    }

    const uint64_t elementSize = typeIt->second;
    if (param.count > std::numeric_limits<uint64_t>::max() / elementSize) {
        HCCL_ERROR("AllReduce count overflows byte size");
        return HCCL_E_PARA;
    }
    const uint64_t totalBytes = param.count * elementSize;
    if (param.rankSize == 1) {
        int32_t copyRet = HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr,
                                                  param.inputPtr, totalBytes);
        return static_cast<HcclResult>(copyRet);
    }
    const size_t dieCount = resCtx.threads.size();
    const bool directHierPath = resCtx.algorithmKind == 5;
    const bool topology3SmallPath = resCtx.algorithmKind == 6;
    const bool topology3LargePath = resCtx.algorithmKind == 7;
    const bool topology1LargePath = resCtx.algorithmKind == 8;
    const bool topology2LargePath = resCtx.algorithmKind == 9;
    const bool topology1Small512 = resCtx.algorithmKind == 10;
    const bool topology2Small512 = resCtx.algorithmKind == 11;
    const bool topology3Small512 = resCtx.algorithmKind == 12;
    const bool tiledSmall512 =
        topology1Small512 || topology2Small512 || topology3Small512;
    const bool fixedLargePath =
        topology1LargePath || topology2LargePath || topology3LargePath;
    const size_t combineCount = (dieCount == 2 && !directHierPath) ? 1 : 0;
    const size_t gatherCount = fixedLargePath ? dieCount : 0;
    const size_t largeStageCount = resCtx.topologyKind == 4 ? 2 : 3;
    const size_t combineKernelIndex = dieCount;
    const size_t gatherKernelOffset = dieCount + combineCount;
    const size_t expectedKernelCount = dieCount + combineCount + gatherCount;
    if (dieCount == 0 || dieCount > 2 || resCtx.ccuKernels.size() != expectedKernelCount ||
        resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0) {
        return HCCL_E_INTERNAL;
    }
    if (fixedLargePath &&
        (resCtx.reduceKernelCount != dieCount + combineCount ||
         resCtx.gatherKernelCount != gatherCount ||
         resCtx.reduceKernelCount + resCtx.gatherKernelCount != resCtx.ccuKernels.size())) {
        HCCL_ERROR("Fixed large CCU stage layout is invalid: kind=%u reduce=%u gather=%u "
            "expected=%zu/%zu", resCtx.algorithmKind, resCtx.reduceKernelCount,
            resCtx.gatherKernelCount, dieCount + combineCount, gatherCount);
        return HCCL_E_INTERNAL;
    }
    const bool fixedSmallPath = resCtx.algorithmKind == 0 ||
        resCtx.algorithmKind == 2 || topology3SmallPath || tiledSmall512;
    if (fixedSmallPath &&
        (resCtx.reduceKernelCount != dieCount + combineCount ||
         resCtx.gatherKernelCount != 0 ||
         resCtx.reduceKernelCount != resCtx.ccuKernels.size())) {
        HCCL_ERROR("Fixed small CCU stage layout is invalid: kind=%u reduce=%u gather=%u "
            "expected=%zu/0", resCtx.algorithmKind, resCtx.reduceKernelCount,
            resCtx.gatherKernelCount, dieCount + combineCount);
        return HCCL_E_INTERNAL;
    }
    if (resCtx.algorithmKind > 12) {
        HCCL_ERROR("Invalid v4 algorithm kind %u", resCtx.algorithmKind);
        return HCCL_E_INTERNAL;
    }

    uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    const uint64_t scratchBase = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    uint64_t scratchToken = 0;
    CHK_RET(HcommCcuGetMemToken(inputBase, totalBytes, &inputToken));
    CHK_RET(HcommCcuGetMemToken(outputBase, totalBytes, &outputToken));
    CHK_RET(HcommCcuGetMemToken(scratchBase, resCtx.localBuffer.size, &scratchToken));

    // kind 10/11/12 是 V4.8 三拓扑 512 KiB 专用图；kind 7/8/9
    // 是 V4.4 三种固定阶段大包图。其余 kind 保留为功能点或回退路径。
    const uint64_t maxChunkBytes = std::min<uint64_t>(MAX_DATA_SIZE, resCtx.localBuffer.size);
    for (uint64_t processed = 0; processed < totalBytes; processed += maxChunkBytes) {
        const uint64_t bytes = std::min<uint64_t>(maxChunkBytes, totalBytes - processed);
        const bool oneShotSmallPath = resCtx.algorithmKind == 0 || resCtx.algorithmKind == 6;
        const bool topologyLanePath = resCtx.algorithmKind == 2;
        const bool topology1SlicePath = topology1Small512;
        const bool fullSliceSmallPath =
            oneShotSmallPath || topology2Small512 || topology3Small512;
        const uint32_t laneCount = (topology1SlicePath ? 4U :
            (topologyLanePath ? (param.rankSize == 16 ? 8U : 4U) :
            (oneShotSmallPath || directHierPath ? 1U : param.rankSize)));
        const uint32_t lane = topology1SlicePath ? param.myRank :
            (topologyLanePath ?
            (param.rankSize == 16 ? param.myRank % 8U : param.myRank) :
            param.myRank);
        const uint64_t chunkElements = bytes / elementSize;
        const uint64_t baseElements = chunkElements / laneCount;
        const uint64_t extraElements = chunkElements % laneCount;
        const uint64_t laneElements = baseElements + (lane < extraElements ? 1 : 0);
        const uint64_t laneOffsetElements = lane * baseElements + std::min<uint64_t>(lane, extraElements);
        const uint64_t myOffsetBytes = (fullSliceSmallPath || directHierPath) ? 0 :
            laneOffsetElements * elementSize;
        const uint64_t myBytes = (fullSliceSmallPath || directHierPath) ? bytes :
            laneElements * elementSize;
        uint64_t args[13] = {
            inputBase + processed,
            outputBase + processed,
            inputToken,
            outputToken,
            scratchBase,
            scratchToken,
            myOffsetBytes,
            myBytes,
            0,
            0,
            0,
            0,
            0,
        };
        constexpr uint32_t combineKernelArgCount = 8U;

        const bool pipeLargePath = resCtx.algorithmKind == 3;
        if (pipeLargePath) {
            FillPipeArgs(bytes, args + 9);
        }
        if (tiledSmall512) {
            // topology1 单 die 只需一次 fused mission；topology2/3 保留
            // V4.4 已闭环的双 die 并发与 Host combine，只替换 reduce 图。
            if (topology1Small512 && dieCount != 1) {
                return HCCL_E_INTERNAL;
            }
            if (dieCount == 2) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resCtx.threads[0], resCtx.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            }
            for (size_t die = 0; die < dieCount; ++die) {
                CcuResult ret = HcommCcuKernelLaunch(
                    resCtx.threads[die], resCtx.ccuKernels[die], args, 8U);
                if (ret != CCU_SUCCESS) return ConvertCcuToHccl(ret);
            }
            if (dieCount == 2) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resCtx.threads[1], resCtx.threads[0], 0)));
                CcuResult ret = HcommCcuKernelLaunch(
                    resCtx.threads[0], resCtx.ccuKernels[combineKernelIndex],
                    args, combineKernelArgCount);
                if (ret != CCU_SUCCESS) return ConvertCcuToHccl(ret);
            }
            continue;
        }
        if (directHierPath) {
            // V1 兼容的两 die 顺序归约：第一个 die 生成 output partial，
            // 第二个 die 在 Host barrier 后以 output 作为本地源继续归约。
            for (size_t die = 0; die < dieCount; ++die) {
                if (die != 0) {
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                        resCtx.threads[die - 1], resCtx.threads[die], 0)));
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                        resCtx.threads[die], 0, CUSTOM_TIMEOUT)));
                }
                CcuResult ret = HcommCcuKernelLaunch(resCtx.threads[die],
                    resCtx.ccuKernels[die], args, 9U);
                if (ret != CCU_SUCCESS) return ConvertCcuToHccl(ret);
            }
            if (dieCount > 1) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resCtx.threads.back(), resCtx.threads[0], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
            }
            continue;
        }
        if (!topologyLanePath) {
            // 固定图路径：一次 partial，必要时一次 combine；大包随后从
            // 独立资源组启动 owned-slice Gather。旧回退 kernel 仍兼容 phase。
            if (dieCount == 2) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resCtx.threads[0], resCtx.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            }
            const uint32_t partialArgCount = pipeLargePath ? 13U : 9U;
            for (size_t die = 0; die < dieCount; ++die) {
                CcuResult ret = HcommCcuKernelLaunch(resCtx.threads[die],
                    resCtx.ccuKernels[die], args, partialArgCount);
                if (ret != CCU_SUCCESS) return ConvertCcuToHccl(ret);
            }
            if (dieCount == 2) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resCtx.threads[1], resCtx.threads[0], 0)));
                CcuResult ret = HcommCcuKernelLaunch(resCtx.threads[0],
                    resCtx.ccuKernels[combineKernelIndex], args, combineKernelArgCount);
                if (ret != CCU_SUCCESS) return ConvertCcuToHccl(ret);
            }
            if (!oneShotSmallPath) {
                args[8] = 1; // 旧 V2.3 phase1，或固定阶段图的独立 Gather kernel。
                if (dieCount == 2) {
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                        resCtx.threads[0], resCtx.threads[1], 0)));
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                        resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
                }
                const size_t firstGatherKernel =
                    fixedLargePath ? gatherKernelOffset : 0;
                CcuResult ret = HcommCcuKernelLaunch(resCtx.threads[0],
                    resCtx.ccuKernels[firstGatherKernel], args,
                    pipeLargePath ? 13U : 9U);
                if (ret != CCU_SUCCESS) return ConvertCcuToHccl(ret);
                if (dieCount == 2) {
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                        resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
                    ret = HcommCcuKernelLaunch(resCtx.threads[1],
                        resCtx.ccuKernels[firstGatherKernel + 1], args,
                        pipeLargePath ? 13U : 9U);
                    if (ret != CCU_SUCCESS) return ConvertCcuToHccl(ret);
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                        resCtx.threads[1], resCtx.threads[0], 0)));
                }
            }
            continue;
        }

        // 4/16-rank 512KiB topology lane path. 该路径仍需通过平台数据验证，
        // 与已恢复的 V2.3 大包路径完全隔离。
        for (uint32_t stageSlot = 0; stageSlot < static_cast<uint32_t>(largeStageCount); ++stageSlot) {
            args[8] = ((resCtx.topologyKind == 4 && stageSlot == 1) ||
                       (resCtx.topologyKind != 4 && stageSlot == 2)) ? 3 : stageSlot;
            if (dieCount == 2) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resCtx.threads[0], resCtx.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            }
            for (size_t die = 0; die < dieCount; ++die) {
                CcuResult ret = HcommCcuKernelLaunch(resCtx.threads[die],
                    resCtx.ccuKernels[die], args, 9U);
                if (ret != CCU_SUCCESS) return ConvertCcuToHccl(ret);
            }
            if (dieCount == 2) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resCtx.threads[1], resCtx.threads[0], 0)));
                if (stageSlot == 0 && (resCtx.reduceCombineMask & 3U) != 0) {
                    CcuResult ret = HcommCcuKernelLaunch(resCtx.threads[0],
                        resCtx.ccuKernels[combineKernelIndex], args, combineKernelArgCount);
                    if (ret != CCU_SUCCESS) return ConvertCcuToHccl(ret);
                }
            }
        }
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
