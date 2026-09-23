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
#include <vector>

#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>
#include <hcomm/hcomm_primitives.h>

#include "log.h"
#include "../include/custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint32_t GATE_NOTIFY_IDX = 0; 
constexpr uint32_t DONE_NOTIFY_IDX = 1;     
constexpr uint32_t INVALID_GROUP = 0xFFFFFFFFu;

constexpr uint64_t RELAY_MIN_CHUNK = 1024 * 1024;

HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel, const std::vector<uint64_t> &taskArgs)
{
    CcuResult ccuRet = HcommCcuKernelLaunch(thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[Scatter][ExecOp] ccu kernel launch failed: ccuRet -> %d", ccuRet), ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("Missing main thread"), HCCL_E_INTERNAL);
    resCtx.threads[0] = param.cpuThread;

    const uint64_t dataTypeSize = SIZE_TABLE.at(param.dataType);
    const uint64_t chunkBytes = param.count * dataTypeSize;

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, chunkBytes));
    }

    const bool amRoot = (param.myRank == param.root);
    CHK_PRT_RET(resCtx.ccuKernels.size() < static_cast<size_t>(resCtx.groupCount) * KERNEL_SLOTS_PER_GROUP
            || resCtx.threads.size() < static_cast<size_t>(resCtx.groupCount),
        HCCL_ERROR("[Scatter][ExecOp] resource mismatch: kernels=%zu threads=%zu groups=%u",
            resCtx.ccuKernels.size(), resCtx.threads.size(), resCtx.groupCount), HCCL_E_INTERNAL);
    const ScatterTopoView topo = DeriveTopo(resCtx.groupOfRank, resCtx.groupCount, param.rankSize, param.myRank,
        param.root);
    const bool twoGroups = (resCtx.groupCount > 1);
    const uint32_t meshCount = topo.meshCount;
    const uint32_t crossCount = topo.crossCount;
    const uint32_t k = topo.relayCount;
    const bool relayTopo = (topo.relayTopo != 0);
    const bool relayData = relayTopo && (crossCount > 4);

    const bool pullMode = (chunkBytes < RELAY_MIN_CHUNK) || (chunkBytes > MAX_DATA_SIZE) || (meshCount == 0);
    const uint32_t slot = pullMode ? 0u : 1u;
    const bool useRelay = (!pullMode && relayData);

    uint64_t beta = 0;
    uint64_t alpha = chunkBytes;
    if (useRelay) {
        beta = chunkBytes * (crossCount - 4) / (k + 4);
        beta = beta / dataTypeSize * dataTypeSize;
        alpha = chunkBytes - beta;
    }

    const bool roleRelay = (topo.amRelay != 0);
    const bool amRelay = useRelay && roleRelay;
    const uint32_t targetIdx = topo.relayTargetRank;

    uint64_t sendToken = 0;
    uint64_t recvToken = 0;
    uint64_t cclToken = 0;
    CcuResult ccuRet = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), chunkBytes, &recvToken);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("[Scatter][ExecOp] get recv token failed: ccuRet -> %d", ccuRet),
        ConvertCcuToHccl(ccuRet));
    if (amRoot) {
        ccuRet = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), chunkBytes * param.rankSize, &sendToken);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS,
            HCCL_ERROR("[Scatter][ExecOp] get send token failed: ccuRet -> %d", ccuRet), ConvertCcuToHccl(ccuRet));
    }
    if (amRelay && resCtx.localBuffer.addr != nullptr && resCtx.localBuffer.size != 0) {
        ccuRet = HcommCcuGetMemToken(
            reinterpret_cast<uint64_t>(resCtx.localBuffer.addr), resCtx.localBuffer.size, &cclToken);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS,
            HCCL_ERROR("[Scatter][ExecOp] get ccl token failed: ccuRet -> %d", ccuRet), ConvertCcuToHccl(ccuRet));
    }

    HCCL_INFO("[Scatter][ExecOp] rank=%u root=%u rankSize=%u chunk=%llu M=%u C=%u k=%u relayTopo=%d pullMode=%d "
              "useRelay=%d amRelay=%d target=%u alpha=%llu beta=%llu",
        param.myRank, param.root, param.rankSize, static_cast<unsigned long long>(chunkBytes), meshCount, crossCount, k,
        static_cast<int>(relayTopo), static_cast<int>(pullMode), static_cast<int>(useRelay), static_cast<int>(amRelay),
        targetIdx, static_cast<unsigned long long>(alpha), static_cast<unsigned long long>(beta));

    auto PickKernel = [&](uint32_t g, uint32_t slot) -> CcuKernelHandle {
        if (g == INVALID_GROUP || g >= resCtx.groupCount) {
            return static_cast<CcuKernelHandle>(0);
        }
        const uint32_t idx = g * KERNEL_SLOTS_PER_GROUP + slot;
        CHK_PRT_RET(idx >= resCtx.ccuKernels.size(),
            HCCL_ERROR("[Scatter][ExecOp] kernel slot out of range: group=%u slot=%u", g, slot),
            static_cast<CcuKernelHandle>(0));
        return resCtx.ccuKernels[idx];
    };

    const uint64_t sliceStep = pullMode ? MAX_DATA_SIZE : chunkBytes;

    for (uint64_t sliceOff = 0; sliceOff < chunkBytes; sliceOff += sliceStep) {
        const uint64_t sliceBytes = std::min<uint64_t>(sliceStep, chunkBytes - sliceOff);
        auto MakeArgs = [&](uint32_t g) {
            const uint64_t firstPeerRank = (twoGroups && g == 0) ? topo.firstMeshPeerRank : topo.firstCrossRank;
            return std::vector<uint64_t>{
                reinterpret_cast<uint64_t>(param.outputPtr) + sliceOff,
                recvToken,
                static_cast<uint64_t>(param.root),
                static_cast<uint64_t>(param.myRank) * chunkBytes + sliceOff,
                sliceBytes,
                sliceOff,
                reinterpret_cast<uint64_t>(param.inputPtr),
                sendToken,
                chunkBytes,
                alpha,
                beta,
                reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
                cclToken,
                static_cast<uint64_t>(targetIdx),
                static_cast<uint64_t>(targetIdx) * chunkBytes + alpha + sliceOff,
                firstPeerRank * chunkBytes,
                useRelay ? 1ULL : 0ULL,
            };
        };
        auto EntryOf = [&](uint32_t g, uint32_t slot) -> ScatterEntry {
            const bool isMesh = twoGroups && (g == 0);
            return PickScatterEntry(amRoot, roleRelay, isMesh, relayTopo, slot);
        };
        auto ArgsOf = [&](uint32_t g, uint32_t slot) {
            const auto full = MakeArgs(g);
            const auto &ids = ScatterEntryArgs(EntryOf(g, slot));
            std::vector<uint64_t> packed;
            packed.reserve(ids.size());
            for (auto id : ids) packed.push_back(full[id]);
            return packed;
        };

        HcclResult ret = HCCL_SUCCESS;
        if (amRoot) {
            if (twoGroups && pullMode) {
                CHK_RET(LaunchKernel(resCtx.threads[0], PickKernel(1u, slot), ArgsOf(1u, slot)));
                CHK_RET(LaunchKernel(resCtx.threads[0], PickKernel(0u, slot), ArgsOf(0u, slot)));
            } else if (twoGroups) {
                ret = static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], GATE_NOTIFY_IDX));
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Scatter][ExecOp] root gate record failed: ret -> %d", static_cast<int32_t>(ret)), ret);
                CHK_RET(LaunchKernel(resCtx.threads[0], PickKernel(0u, slot), ArgsOf(0u, slot)));
                ret = static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(resCtx.threads[1], GATE_NOTIFY_IDX, CUSTOM_TIMEOUT));
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Scatter][ExecOp] slave gate wait failed: ret -> %d", static_cast<int32_t>(ret)), ret);
                CHK_RET(LaunchKernel(resCtx.threads[1], PickKernel(1u, slot), ArgsOf(1u, slot)));
                ret = static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], DONE_NOTIFY_IDX));
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Scatter][ExecOp] slave finish record failed: ret -> %d", static_cast<int32_t>(ret)),
                    ret);
                ret = static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(resCtx.threads[0], DONE_NOTIFY_IDX, CUSTOM_TIMEOUT));
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Scatter][ExecOp] master finish wait failed: ret -> %d", static_cast<int32_t>(ret)),
                    ret);
            } else {
                CHK_RET(LaunchKernel(resCtx.threads[0], PickKernel(0u, slot), ArgsOf(0u, slot)));
            }
            continue;
        }

        if (amRelay) {
            CHK_RET(LaunchKernel(resCtx.threads[0], PickKernel(0u, 1u), ArgsOf(0u, 1u)));
            ret = static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], GATE_NOTIFY_IDX));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][ExecOp] relay gate record failed: ret -> %d", static_cast<int32_t>(ret)), ret);
            CHK_RET(LaunchKernel(resCtx.threads[0], PickKernel(0u, 0u), ArgsOf(0u, 0u)));
            ret = static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[1], GATE_NOTIFY_IDX, CUSTOM_TIMEOUT));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][ExecOp] relay slave gate wait failed: ret -> %d", static_cast<int32_t>(ret)), ret);
            CHK_RET(LaunchKernel(resCtx.threads[1], PickKernel(1u, 1u), ArgsOf(1u, 1u)));
            ret = static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], DONE_NOTIFY_IDX));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][ExecOp] relay slave finish record failed: ret -> %d", static_cast<int32_t>(ret)),
                ret);
            ret = static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[0], DONE_NOTIFY_IDX, CUSTOM_TIMEOUT));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][ExecOp] master finish wait failed: ret -> %d", static_cast<int32_t>(ret)), ret);
            continue;
        }
        const uint32_t ownSlot = roleRelay ? 0u : slot;
        const uint32_t groupOfRoot = resCtx.groupOfRank[param.root];
        CHK_PRT_RET(groupOfRoot >= resCtx.groupCount,
            HCCL_ERROR("[Scatter][ExecOp] root[%u] not in any channel group of rank[%u]", param.root, param.myRank),
            HCCL_E_INTERNAL);
        CHK_RET(LaunchKernel(resCtx.threads[0], PickKernel(groupOfRoot, ownSlot), ArgsOf(groupOfRoot, ownSlot)));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
