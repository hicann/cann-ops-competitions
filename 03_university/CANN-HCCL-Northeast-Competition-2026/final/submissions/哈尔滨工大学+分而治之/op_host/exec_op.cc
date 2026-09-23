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

#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
    // E4 全 helper 中继：每 serving helper 的份额 align4((n-k-4)*m/(k+3))，恒小于 m（本赛拓扑 n < 2k+7）
    uint64_t RelayShare(uint32_t k, uint32_t n, uint64_t m)
    {
        if (k < 2 || n < k + 5) {
            return 0;
        }
        const uint64_t share = static_cast<uint64_t>(n - k - 4) * m / (k + 3) & ~3ULL;
        const uint64_t maxY = static_cast<uint64_t>(n - k) * m;
        return std::min(share, maxY);
    }

    // 中继收益足以覆盖 P2 额外同步开销的最小 chunk 尺寸（小消息保持纯 E2 行为）
    constexpr uint64_t kRelayMinChunkBytes = 8ULL * 1024 * 1024;
} // namespace


HcclResult ExecOp(const OpParam &param, aclrtStream userStream)
{
    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    // 执行规划：以 MAX_DATA_SIZE=256MiB 为单个 CCU launch 的上限切 chunk；
    // 相位1 kernel 每 die 组一个（中继行为经 kernelArg 常量与 CCU_IF 表达），
    // 前缀独立 launch 后由其实际 stream 释放 P2，与本 chunk 剩余 P1 以及下一 chunk 重叠。
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("[P2Probe] unsupported data type"), HCCL_E_PARA);
    const uint64_t sliceBytes = param.count * sizeIt->second;
    CHK_PRT_RET(sliceBytes == 0, HCCL_ERROR("[P2Probe] slice bytes is zero"), HCCL_E_PARA);

    const uint32_t groupCount = static_cast<uint32_t>(resCtx.ccuKernels.size());
    CHK_PRT_RET(groupCount == 0 || groupCount > 2,
        HCCL_ERROR("[P2Probe] unexpected P1 kernel count %zu", resCtx.ccuKernels.size()), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.relayKernels.size() > 1, HCCL_ERROR("[P2Probe] unexpected P2 kernel count"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.prefixGroups.size() != groupCount
                    || (resCtx.helperPrefixGroup != UINT32_MAX && resCtx.helperPrefixGroup >= groupCount),
        HCCL_ERROR("[PrefixStage] invalid prefix group context"), HCCL_E_INTERNAL);
    // E4.8 紧凑接收参数：非 root 的相位1 kernel 只声明 7 个 LoadArg（见 ccu_kernel.cc），
    // launch 参数数必须与 kernel 发射的 LoadArg 数一致；角色 rank 全局（root/非 root），逐组相同。
    const bool isRootRank = param.myRank == param.root;
    constexpr uint32_t kReceiverTaskArgNum = 7;
    const uint32_t p1TaskArgNum = isRootRank ? 12U : kReceiverTaskArgNum;
    const bool splitStreams = groupCount > 1;
    CHK_PRT_RET(splitStreams
                    && (resCtx.threads.size() != 2 || resCtx.auxiliaryStream == nullptr || resCtx.startNotify == nullptr
                        || resCtx.doneNotify == nullptr),
        HCCL_ERROR("[P2Probe] incomplete fork/join context"), HCCL_E_INTERNAL);
    const bool hasP2 = !resCtx.relayKernels.empty();
    CHK_PRT_RET(hasP2
                    && (resCtx.relayStream == nullptr || resCtx.relayForkParity0 == nullptr
                        || resCtx.relayForkParity1 == nullptr || resCtx.relayDoneNotify == nullptr),
        HCCL_ERROR("[P2Probe] incomplete relay stream context"), HCCL_E_INTERNAL);

    uint64_t inputToken = 0;
    if (param.myRank == param.root) {
        const uint64_t inputBytes = sliceBytes * param.rankSize;
        const CcuResult tokenResult
            = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, &inputToken);
        CHK_PRT_RET(tokenResult != CCU_SUCCESS, HCCL_ERROR("[P2Probe] input token acquisition failed: %d", tokenResult),
            HCCL_E_INTERNAL);
    }

    // E5.0 小消息短握手 Push（stageMode=4）：peer 发布 output 描述符 -> root 推写 ->
    // root 发 DONE -> peer 等 DONE，无尾 ACK（协议证明见 e50/p1-protocol-proof.md）。
    // 回退：stageMode 常量改回 2 即恢复 E4.5 pull（kernel 两分支共存）。
    // 所有 rank 用同一公式选路（B <= 1MiB）；复用相位1 注册，不新增 per-die mission；
    // 非 root 只 launch 含 root channel 的 die 组，root launch 全部组（双 die 组复用 fork/join）。
    constexpr uint64_t kSmallReadMaxBytes = 1ULL << 20;
    if (sliceBytes <= kSmallReadMaxBytes) {
        CHK_PRT_RET(!isRootRank && resCtx.smallRootGroup >= groupCount,
            HCCL_ERROR("[SmallRead] invalid small root group %u", resCtx.smallRootGroup), HCCL_E_INTERNAL);
        uint64_t smallOutputToken = 0;
        const CcuResult smallTokenResult
            = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), sliceBytes, &smallOutputToken);
        CHK_PRT_RET(smallTokenResult != CCU_SUCCESS,
            HCCL_ERROR("[SmallRead] output token acquisition failed: %d", smallTokenResult), HCCL_E_INTERNAL);
        // root 12 参；非 root 紧凑 7 参（布局见 ccu_kernel.cc 的 compactP1Receiver）
        uint64_t smallTaskArgs[]
            = {reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr), inputToken,
                smallOutputToken, sliceBytes, 0, sliceBytes, 0, 0, 0, 0, 4};
        uint64_t smallReceiverArgs[]
            = {reinterpret_cast<uint64_t>(param.outputPtr), smallOutputToken, 0, 0, 0, sliceBytes, 4};
        const uint64_t *smallArgs = isRootRank ? smallTaskArgs : smallReceiverArgs;
        const uint32_t launchGroups = isRootRank ? groupCount : 1;
        HCCL_INFO("[SmallRead] rank %u recvCount=%llu sliceBytes=%llu groups=%u", param.myRank,
            static_cast<unsigned long long>(param.count), static_cast<unsigned long long>(sliceBytes), launchGroups);
        CHK_PRT_RET(resCtx.smallEntryVersion != 1 || resCtx.smallKernels.size() != groupCount
                        || resCtx.smallDedicated.size() != groupCount,
            HCCL_ERROR("[SmallEntry] invalid serialized context"), HCCL_E_INTERNAL);
        const auto launchSmall = [&](uint32_t group, ThreadHandle thread) -> CcuResult {
            const bool dedicated = resCtx.smallDedicated[group] != 0;
            const auto handle = dedicated ? resCtx.smallKernels[group] : resCtx.ccuKernels[group];
            const uint32_t argc = dedicated ? (isRootRank ? 5U : 2U) : p1TaskArgNum;
            HCCL_INFO("[SmallEntryLaunch] rank=%u root=%u group=%u dedicated=%u args=%u", param.myRank, param.root,
                group, dedicated ? 1U : 0U, argc);
            return HcommCcuKernelLaunch(thread, handle, smallArgs, argc);
        };
        const bool splitSmall = isRootRank && splitStreams;
        if (splitSmall) {
            CHK_PRT_RET(aclrtRecordNotify(resCtx.startNotify, userStream) != ACL_SUCCESS
                            || aclrtWaitAndResetNotify(resCtx.startNotify, resCtx.auxiliaryStream, 60) != ACL_SUCCESS,
                HCCL_ERROR("[SmallRead] fork failed"), HCCL_E_INTERNAL);
        }
        if (isRootRank) {
            for (uint32_t k = 0; k < groupCount; ++k) {
                const ThreadHandle thread = splitSmall && k == 1 ? resCtx.threads[1] : param.cpuThread;
                const CcuResult smallResult = launchSmall(k, thread);
                CHK_PRT_RET(smallResult != CCU_SUCCESS,
                    HCCL_ERROR("[SmallRead] small kernel launch failed: %d", smallResult), HCCL_E_INTERNAL);
            }
        } else {
            const CcuResult smallResult = launchSmall(resCtx.smallRootGroup, param.cpuThread);
            CHK_PRT_RET(smallResult != CCU_SUCCESS,
                HCCL_ERROR("[SmallRead] small kernel launch failed: %d", smallResult), HCCL_E_INTERNAL);
        }
        if (splitSmall) {
            CHK_PRT_RET(aclrtRecordNotify(resCtx.doneNotify, resCtx.auxiliaryStream) != ACL_SUCCESS
                            || aclrtWaitAndResetNotify(resCtx.doneNotify, userStream, 60) != ACL_SUCCESS,
                HCCL_ERROR("[SmallRead] join failed"), HCCL_E_INTERNAL);
        }
        
        return HCCL_SUCCESS;
    }

    // E4 门控（全 rank 同公式、同输入 -> 决策一致）：
    // 拓扑门控 relayEligible 之外，还要求 chunk 数 <= 2（奇偶双缓冲/scratch 区间/epoch bit 均按两块设计）、
    // chunk 足够大、中继份额经本端 CCL buffer 一半封顶后仍小于 chunk。
    // E4.7：passive 路由不再均衡拆两块——官方尺寸（中 512MiB/大 400MiB 为总输入，143 slice≤42.67MiB）
    // 单 chunk 即可；两块流水会多付一轮 postsync barrier + 握手（实测 T0 ≈ +12μs/chunk）。
    // 仅当 slice > 256MiB（单次 launch 上限）才切第二块，此时两块各自中继逻辑不变。
    const uint64_t chunkLimit = MAX_DATA_SIZE;
    const uint64_t numChunks = (sliceBytes + chunkLimit - 1) / chunkLimit;
    const bool relayCapable = resCtx.relayEligible != 0 && numChunks <= 2;
    const uint64_t bufferCap = (resCtx.localBuffer.size / 2) & ~3ULL;
    const auto chunkRelayY = [&](uint64_t chunkBytes) -> uint64_t {
        if (!relayCapable || chunkBytes < kRelayMinChunkBytes) {
            return 0;
        }
        uint64_t y = std::min(RelayShare(resCtx.kRoot, param.rankSize, chunkBytes), bufferCap);
        return y < chunkBytes ? y : 0;
    };
    const uint64_t chunk0Bytes = std::min<uint64_t>(chunkLimit, sliceBytes);
    const uint64_t relayY0 = chunkRelayY(chunk0Bytes);
    const uint64_t relayY1 = numChunks > 1 ? chunkRelayY(sliceBytes - chunk0Bytes) : 0;
    // E4.6 大消息纯直连星形（stageMode=3）：整次调用无中继（relayY 全 0）时启用。
    // 门控输入全部为拓扑+统一尺寸的纯函数，全 rank 决策一致（契约见 .syncheck/tests/final_ccu/e46/）；
    // 非 root 只 launch 持有 root channel 的 die 组（smallRootGroup），root launch 全部组。
    const bool starDirect = relayY0 == 0 && relayY1 == 0;
    const bool starPeer = starDirect && param.myRank != param.root;
    CHK_PRT_RET(starPeer && resCtx.smallRootGroup >= groupCount,
        HCCL_ERROR("[StarDirect] invalid root group %u", resCtx.smallRootGroup), HCCL_E_INTERNAL);
    HCCL_INFO("[P2Probe] rank %u recvCount=%llu sliceBytes=%llu numChunks=%llu relayY0=%llu relayY1=%llu star=%u",
        param.myRank, static_cast<unsigned long long>(param.count), static_cast<unsigned long long>(sliceBytes),
        static_cast<unsigned long long>(numChunks), static_cast<unsigned long long>(relayY0),
        static_cast<unsigned long long>(relayY1), starDirect ? 1U : 0U);

    // scratch（本端 CCL 头）双缓冲：chunk0 用 [0, relayY0)，chunk1 用 [relayY0, relayY0+relayY1)；
    // 同一次调用内区间不复用，无需 CREDIT；跨调用由 op 末尾的流汇合保证不提前复用
    const uint64_t scratchAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    uint64_t scratchToken = 0;
    if (relayY0 + relayY1 > 0) {
        const CcuResult scratchResult = HcommCcuGetMemToken(scratchAddr, relayY0 + relayY1, &scratchToken);
        CHK_PRT_RET(scratchResult != CCU_SUCCESS,
            HCCL_ERROR("[P2Probe] scratch token acquisition failed: %d", scratchResult), HCCL_E_INTERNAL);
    }

    bool anyRelay = false;
    uint32_t parity = 0;
    for (uint64_t offset = 0; offset < sliceBytes;) {
        const uint64_t chunkBytes = std::min<uint64_t>(chunkLimit, sliceBytes - offset);
        uint64_t outputToken = 0;
        const CcuResult outputTokenResult
            = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr) + offset, chunkBytes, &outputToken);
        CHK_PRT_RET(outputTokenResult != CCU_SUCCESS,
            HCCL_ERROR("[P2Probe] output token acquisition failed: %d", outputTokenResult), HCCL_E_INTERNAL);

        const uint64_t relayY = parity == 0 ? relayY0 : relayY1;
        const bool useRelay = relayY > 0;
        const uint64_t relaySuffix = chunkBytes - relayY;
        const uint64_t chunkScratchAddr = scratchAddr + (parity != 0 ? relayY0 : 0);
        uint64_t taskArgs[]
            = {reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr), inputToken,
                outputToken, sliceBytes, offset, chunkBytes, chunkScratchAddr, scratchToken, relayY, relaySuffix, 0};
        constexpr uint32_t taskArgNum = sizeof(taskArgs) / sizeof(taskArgs[0]);
        // E4.8 紧凑接收布局（与 ccu_kernel.cc compactP1Receiver 对齐）：
        // [0]=outputAddr [1]=outputToken [2]=scratchAddr [3]=scratchToken [4]=relayY
        // [5]=totalSliceBytes [6]=stageMode
        uint64_t receiverTaskArgs[] = {reinterpret_cast<uint64_t>(param.outputPtr), outputToken, chunkScratchAddr,
            scratchToken, relayY, sliceBytes, 0};
        const uint64_t *p1TaskArgs = isRootRank ? taskArgs : receiverTaskArgs;

        // L2：非 root 仅持有 root channel 的组有 regular/prefix 工作（publishRootIdx 组必持 root
        // channel），单流 launch 即可；root 保持双 die 组 fork/join。星形路径同此条件。
        const bool splitCall = splitStreams && isRootRank;
        if (splitCall) {
            CHK_PRT_RET(aclrtRecordNotify(resCtx.startNotify, userStream) != ACL_SUCCESS
                            || aclrtWaitAndResetNotify(resCtx.startNotify, resCtx.auxiliaryStream, 60) != ACL_SUCCESS,
                HCCL_ERROR("[P2Probe] fork failed"), HCCL_E_INTERNAL);
        }
        const auto groupStream = [&](uint32_t k) -> aclrtStream {
            return splitCall && k == 1 ? resCtx.auxiliaryStream : userStream;
        };
        const auto forkRelay = [&](aclrtStream producer) -> HcclResult {
            const aclrtNotify notify = parity != 0 ? resCtx.relayForkParity1 : resCtx.relayForkParity0;
            if (aclrtRecordNotify(notify, producer) != ACL_SUCCESS
                || aclrtWaitAndResetNotify(notify, resCtx.relayStream, 60) != ACL_SUCCESS) {
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        };
        if (starDirect) {
            HCCL_INFO("[StarDirect] rank %u sliceBytes=%llu chunkOffset=%llu launches=%u", param.myRank,
                static_cast<unsigned long long>(sliceBytes), static_cast<unsigned long long>(offset),
                starPeer ? 1U : groupCount);
        }
        if (useRelay && hasP2 && resCtx.helperPrefixGroup == UINT32_MAX) {
            CHK_RET(forkRelay(userStream));
        }
        for (uint32_t k = 0; k < resCtx.ccuKernels.size(); ++k) {
            // L2/star：非 root 只 launch 持有 root channel 的组（其余组 regular 为空且必无 prefix）
            if (!isRootRank && k != resCtx.smallRootGroup) {
                continue;
            }
            const ThreadHandle thread = splitCall && k == 1 ? resCtx.threads[1] : param.cpuThread;
            if (useRelay && resCtx.prefixGroups[k] != 0) {
                HCCL_INFO("[PrefixStage] rank=%u group=%u chunkOffset=%llu helperGroup=%u bytes=%llu", param.myRank, k,
                    static_cast<unsigned long long>(offset), resCtx.helperPrefixGroup,
                    static_cast<unsigned long long>(relayY));
                taskArgs[11] = 1;
                receiverTaskArgs[6] = 1;
                const CcuResult prefixResult
                    = HcommCcuKernelLaunch(thread, resCtx.ccuKernels[k], p1TaskArgs, p1TaskArgNum);
                CHK_PRT_RET(prefixResult != CCU_SUCCESS,
                    HCCL_ERROR("[PrefixStage] prefix launch failed: %d", prefixResult), HCCL_E_INTERNAL);
                if (hasP2 && resCtx.helperPrefixGroup == k) {
                    CHK_RET(forkRelay(groupStream(k)));
                }
            }
            taskArgs[11] = starDirect ? 3 : 0;
            receiverTaskArgs[6] = starDirect ? 3 : 0;
            const CcuResult launchResult = HcommCcuKernelLaunch(thread, resCtx.ccuKernels[k], p1TaskArgs, p1TaskArgNum);
            CHK_PRT_RET(launchResult != CCU_SUCCESS, HCCL_ERROR("[P2Probe] CCU kernel launch failed: %d", launchResult),
                HCCL_E_INTERNAL);
        }
        if (useRelay && hasP2) {
            const CcuResult relayResult
                = HcommCcuKernelLaunch(resCtx.relayThread, resCtx.relayKernels[0], taskArgs, taskArgNum - 1);
            CHK_PRT_RET(relayResult != CCU_SUCCESS, HCCL_ERROR("[P2Probe] relay kernel launch failed: %d", relayResult),
                HCCL_E_INTERNAL);
            anyRelay = true;
        }
        if (splitCall) {
            CHK_PRT_RET(aclrtRecordNotify(resCtx.doneNotify, resCtx.auxiliaryStream) != ACL_SUCCESS
                            || aclrtWaitAndResetNotify(resCtx.doneNotify, userStream, 60) != ACL_SUCCESS,
                HCCL_ERROR("[P2Probe] join failed"), HCCL_E_INTERNAL);
        }
        parity ^= 1;
        offset += chunkBytes;
    }

    // op 末尾一次性汇合 P2 流：保证全部转发落地后本 rank 才完成（dest 的输出前缀来自 P2）
    if (anyRelay) {
        CHK_PRT_RET(aclrtRecordNotify(resCtx.relayDoneNotify, resCtx.relayStream) != ACL_SUCCESS
                        || aclrtWaitAndResetNotify(resCtx.relayDoneNotify, userStream, 60) != ACL_SUCCESS,
            HCCL_ERROR("[P2Probe] relay join failed"), HCCL_E_INTERNAL);
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
