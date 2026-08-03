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
#include <limits>
#include <map>
#include <mutex>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {

namespace {

    constexpr uint32_t THREAD_NOTIFY_IDX = 0;
    constexpr uint32_t TASK_ARG_NUM = 13;
    constexpr uint32_t FLAT_TASK_ARG_NUM = 10;
    constexpr uint64_t HIERARCHICAL_PATH_THRESHOLD = 1024 * 1024;
    constexpr uint64_t HIERARCHICAL_SPLIT_ALIGN = 4096;
    constexpr uint64_t RECURSIVE_PATH_THRESHOLD = 1024 * 1024;

    std::mutex g_callCountMtx;
    std::map<const void *, uint64_t> g_callCount;

    HcclResult ConvertCcuResult(CcuResult result)
    {
        switch (result) {
            case CCU_SUCCESS:
                return HCCL_SUCCESS;
            case CCU_E_PARA:
                return HCCL_E_PARA;
            case CCU_E_PTR:
                return HCCL_E_PTR;
            case CCU_E_INTERNAL:
                return HCCL_E_INTERNAL;
            case CCU_E_NOT_SUPPORT:
                return HCCL_E_NOT_SUPPORT;
            case CCU_E_NOT_FOUND:
                return HCCL_E_NOT_FOUND;
            case CCU_E_UNAVAIL:
                return HCCL_E_UNAVAIL;
            default:
                return HCCL_E_INTERNAL;
        }
    }

    HcclResult LaunchKernel(
        ThreadHandle thread, CcuKernelHandle kernel, const std::vector<uint64_t> &taskArgs, uint32_t argNum)
    {
        CcuResult result = HcommCcuKernelLaunch(thread, kernel, taskArgs.data(), argNum);
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("[AllGather] CCU kernel launch failed: %d", result);
            return ConvertCcuResult(result);
        }
        return HCCL_SUCCESS;
    }

    // 每次 launch 取一个全局递增序号：taskArgs[11]=是否首轮（显式交换地址/token），
    // taskArgs[12]=序号奇偶（PostSync 的 notify 位随之轮换，防多打一坍缩）
    uint64_t NextLaunchSeq(const OpParam &param)
    {
        std::lock_guard<std::mutex> lock(g_callCountMtx);
        return g_callCount[param.resCtx]++;
    }

    HcclResult PreSyncThreads(ThreadHandle mainThread, ThreadHandle slaveThread)
    {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(mainThread, slaveThread, THREAD_NOTIFY_IDX)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(slaveThread, THREAD_NOTIFY_IDX, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    HcclResult PostSyncThreads(ThreadHandle mainThread, ThreadHandle slaveThread)
    {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(mainThread, THREAD_NOTIFY_IDX, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(slaveThread, mainThread, THREAD_NOTIFY_IDX)));
        return HCCL_SUCCESS;
    }

    HcclResult LaunchAllKernels(const OpParam &param, const AlgResourceCtx &resCtx,
        const std::vector<uint64_t> &taskArgs, uint32_t argNum, const std::vector<CcuKernelHandle> &kernels)
    {
        if (kernels.size() == 1) {
            return LaunchKernel(param.cpuThread, kernels[0], taskArgs, argNum);
        }

        const ThreadHandle slaveThread = resCtx.threads[1];
        CHK_RET(PreSyncThreads(param.cpuThread, slaveThread));
        CHK_RET(LaunchKernel(param.cpuThread, kernels[0], taskArgs, argNum));
        CHK_RET(LaunchKernel(slaveThread, kernels[1], taskArgs, argNum));
        CHK_RET(PostSyncThreads(param.cpuThread, slaveThread));
        return HCCL_SUCCESS;
    }

    // 反序列化结果缓存：引擎上下文一次构建后不变，避免每 call 重复拷贝解析
    std::mutex g_resCtxMtx;
    std::map<const void *, AlgResourceCtx> g_resCtxCache;

    const AlgResourceCtx &GetResCtx(const OpParam &param)
    {
        std::lock_guard<std::mutex> lock(g_resCtxMtx);
        const auto it = g_resCtxCache.find(param.resCtx);
        if (it != g_resCtxCache.end()) {
            return it->second;
        }
        char *ctx = static_cast<char *>(param.resCtx);
        std::vector<char> seq(ctx, ctx + param.ctxSize);
        AlgResourceCtx resCtx;
        resCtx.DeSerialize(seq);
        const auto result = g_resCtxCache.emplace(param.resCtx, std::move(resCtx));
        return result.first->second;
    }

    // 内存 token 缓存：harness buffer 同址复用，避免每 call 两次驱动查询
    std::mutex g_tokenMtx;
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> g_tokenCache;

    CcuResult GetMemTokenCached(uint64_t va, uint64_t size, uint64_t &token)
    {
        const auto key = std::make_pair(va, size);
        {
            std::lock_guard<std::mutex> lock(g_tokenMtx);
            const auto it = g_tokenCache.find(key);
            if (it != g_tokenCache.end()) {
                token = it->second;
                return CCU_SUCCESS;
            }
        }
        uint64_t tokenInfo = 0;
        const CcuResult ccuRet = HcommCcuGetMemToken(va, size, &tokenInfo);
        if (ccuRet != CCU_SUCCESS) {
            return ccuRet;
        }
        std::lock_guard<std::mutex> lock(g_tokenMtx);
        g_tokenCache[key] = tokenInfo;
        token = tokenInfo;
        return CCU_SUCCESS;
    }

    // 地址/token 交换判定：全通信域首次或输入/输出 VA 变化时，下一次 launch 先显式交换
    std::mutex g_vaMtx;
    std::map<const void *, std::pair<uint64_t, uint64_t>> g_vaCache;

    bool ConsumeExchangeFlag(const OpParam &param, uint64_t inputAddr, uint64_t outputAddr)
    {
        std::lock_guard<std::mutex> lock(g_vaMtx);
        const auto va = std::make_pair(inputAddr, outputAddr);
        const auto it = g_vaCache.find(param.resCtx);
        if (it == g_vaCache.end() || it->second != va) {
            g_vaCache[param.resCtx] = va;
            return true;
        }
        return false;
    }

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    const AlgResourceCtx &resCtx = GetResCtx(param);

    const auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(), HCCL_ERROR("[AllGather] Unsupported data type %d", param.dataType),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeIter->second,
        HCCL_ERROR("[AllGather] Input size overflow"), HCCL_E_PARA);
    const uint64_t dataSize = param.count * sizeIter->second;
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.rankSize == 0 || dataSize > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("[AllGather] Output size overflow"), HCCL_E_PARA);
    const uint64_t outputSize = dataSize * param.rankSize;

    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("[AllGather] No CCU thread in resource context"), HCCL_E_INTERNAL);
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, dataSize));
    }
    // 分层拓扑用 ccuKernels，非分层拓扑用 flatKernels，二者必其一与 threads 对应
    const bool useFlatKernels = !resCtx.flatKernels.empty();
    const std::vector<CcuKernelHandle> &activeKernels = useFlatKernels ? resCtx.flatKernels : resCtx.ccuKernels;
    CHK_PRT_RET(activeKernels.empty() || activeKernels.size() != resCtx.threads.size() || activeKernels.size() > 2,
        HCCL_ERROR("[AllGather] Invalid resources: %zu hier kernels, %zu flat kernels, %zu threads",
            resCtx.ccuKernels.size(), resCtx.flatKernels.size(), resCtx.threads.size()),
        HCCL_E_INTERNAL);

    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CcuResult ccuRet = GetMemTokenCached(inputAddr, dataSize, inputToken);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("[AllGather] Failed to get input token: %d", ccuRet);
        return ConvertCcuResult(ccuRet);
    }
    ccuRet = GetMemTokenCached(outputAddr, outputSize, outputToken);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("[AllGather] Failed to get output token: %d", ccuRet);
        return ConvertCcuResult(ccuRet);
    }

    uint64_t processedSize = 0;
    bool exchangePending = ConsumeExchangeFlag(param, inputAddr, outputAddr);
    while (processedSize < dataSize) {
        const uint64_t sliceSize = std::min<uint64_t>(MAX_DATA_SIZE, dataSize - processedSize);
        const uint64_t outputOffset = dataSize * param.myRank;
        const uint64_t localBlockOffset = (static_cast<uint64_t>(param.myRank) / 8) * 8 * dataSize;
        bool useHierarchicalPath = (param.rankSize == 16 || param.rankSize == 12) && resCtx.ccuKernels.size() == 2
                                   && sliceSize >= HIERARCHICAL_PATH_THRESHOLD;
        uint64_t collectPartSize = 0;
        uint64_t directPartSize = 0;
        if (useHierarchicalPath) {
            // 三路拆分（LP 最优）：2*8 = A4:D3:S2 4（15/11 S/B）；8+4 = A3:D2:S2 2（11/7 S/B）
            const uint64_t collectNumer = (param.rankSize == 16) ? 4 : 3;
            const uint64_t directNumer = (param.rankSize == 16) ? 3 : 2;
            const uint64_t denom = (param.rankSize == 16) ? 11 : 7;
            collectPartSize = ((sliceSize * collectNumer / denom) / HIERARCHICAL_SPLIT_ALIGN)
                              * HIERARCHICAL_SPLIT_ALIGN;
            directPartSize = ((sliceSize * directNumer / denom) / HIERARCHICAL_SPLIT_ALIGN)
                             * HIERARCHICAL_SPLIT_ALIGN;
            if (collectPartSize == 0 || directPartSize == 0
                || collectPartSize + directPartSize >= sliceSize) {
                collectPartSize = 0;
                directPartSize = 0;
                useHierarchicalPath = false;
            }
        }
        const uint64_t relayPartSize = sliceSize - collectPartSize - directPartSize;

        const bool useRecursivePath = !useHierarchicalPath && param.rankSize == 4
                                      && resCtx.flatKernels.size() == 1 && sliceSize >= RECURSIVE_PATH_THRESHOLD;
        if (useRecursivePath) {
            // 4*1 大消息递归倍增（单 launch 融合两轮，内部双 barrier 分用 even/odd 位）：
            // 占两个 launch 序号保持全局奇偶纪律
            const uint64_t seq = NextLaunchSeq(param);
            (void)NextLaunchSeq(param);
            std::vector<uint64_t> taskArgs = {
                inputAddr + processedSize,
                outputAddr + processedSize,
                inputToken,
                outputToken,
                sliceSize,
                outputOffset,
                TRANSFER_ROUND_FUSED,
                (static_cast<uint64_t>(param.myRank) & ~1ULL) * dataSize, // pairBaseOffset
                exchangePending ? 1ULL : 0ULL,
                seq & 1,
            };
            exchangePending = false;
            CHK_RET(LaunchAllKernels(param, resCtx, taskArgs, FLAT_TASK_ARG_NUM, resCtx.flatKernels));
        } else if (!useHierarchicalPath && useFlatKernels) {
            // 极简扁平 kernel（10 个 taskArgs），arg7 保留为 0
            const uint64_t seq = NextLaunchSeq(param);
            std::vector<uint64_t> taskArgs = {
                inputAddr + processedSize,
                outputAddr + processedSize,
                inputToken,
                outputToken,
                sliceSize,
                outputOffset,
                TRANSFER_FLAT,
                0,
                exchangePending ? 1ULL : 0ULL,
                seq & 1,
            };
            exchangePending = false;
            CHK_RET(LaunchAllKernels(param, resCtx, taskArgs, FLAT_TASK_ARG_NUM, resCtx.flatKernels));
        } else if (!useHierarchicalPath) {
            // 分层 kernel 的扁平整写分支（13 个 taskArgs，mode=TRANSFER_FLAT）
            const uint64_t seq = NextLaunchSeq(param);
            std::vector<uint64_t> taskArgs = {
                inputAddr + processedSize,
                outputAddr + processedSize,
                inputToken,
                outputToken,
                sliceSize,
                outputOffset,
                TRANSFER_FLAT,
                0,
                0,
                0,
                localBlockOffset,
                exchangePending ? 1ULL : 0ULL,
                seq & 1,
            };
            exchangePending = false;
            CHK_RET(LaunchAllKernels(param, resCtx, taskArgs, TASK_ARG_NUM, resCtx.ccuKernels));
        } else {
            std::vector<uint64_t> taskArgs = {
                inputAddr + processedSize,
                outputAddr + processedSize,
                inputToken,
                outputToken,
                sliceSize,
                outputOffset,
                TRANSFER_FLAT,
                collectPartSize,
                directPartSize,
                relayPartSize,
                localBlockOffset,
                0, // exchangeFlag：仅本轮首个 launch 按需置 1
                0, // syncParity：notify 位按 launch 奇偶轮换
            };
            const uint64_t seqOne = NextLaunchSeq(param);
            taskArgs[6] = TRANSFER_PHASE_ONE;
            taskArgs[11] = exchangePending ? 1 : 0;
            taskArgs[12] = seqOne & 1;
            exchangePending = false;
            CHK_RET(LaunchAllKernels(param, resCtx, taskArgs, TASK_ARG_NUM, resCtx.ccuKernels));
            const uint64_t seqTwo = NextLaunchSeq(param);
            taskArgs[6] = TRANSFER_PHASE_TWO;
            taskArgs[12] = seqTwo & 1;
            CHK_RET(LaunchAllKernels(param, resCtx, taskArgs, TASK_ARG_NUM, resCtx.ccuKernels));
        }
        processedSize += sliceSize;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
