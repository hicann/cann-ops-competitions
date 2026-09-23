/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <algorithm>

#include "log.h"
#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint32_t SCATTER_MAX_LAUNCH = 4;

struct ScatterLaunchSet {
    CcuKernelHandle kernel[SCATTER_MAX_LAUNCH] = {0, 0, 0, 0};
    std::vector<uint64_t> args[SCATTER_MAX_LAUNCH];
    uint32_t num = 0;
};

struct ScatterExecCache {
    const void *resCtx = nullptr;
    uint64_t ctxSize = 0;
    AlgResourceCtx ctx;
    std::vector<std::vector<uint32_t>> dieChannelIdx;
    uint32_t ownCopyDie = 0;
};

thread_local ScatterExecCache g_execCache;
thread_local ScatterLaunchSet g_launch;

struct ScatterPlanCache {
    const void *resCtx = nullptr;
    uint32_t root = INVALID_VALUE_RANKID;
    uint64_t shareBytes = 0;
    uint32_t allowRelay = 0;
    ScatterPlan plan;
};

thread_local ScatterPlanCache g_planCache;

struct ScatterTokenCache {
    static constexpr uint32_t ENTRY_NUM = 8;
    uint64_t va[ENTRY_NUM] = {0};
    uint64_t size[ENTRY_NUM] = {0};
    uint64_t token[ENTRY_NUM] = {0};
    uint32_t next = 0;
};

thread_local ScatterTokenCache g_tokenCache;

HcclResult ScatterGetMemToken(uint64_t va, uint64_t size, uint64_t &token)
{
    for (uint32_t i = 0; i < ScatterTokenCache::ENTRY_NUM; i++) {
        if (g_tokenCache.va[i] == va && g_tokenCache.size[i] == size) {
            token = g_tokenCache.token[i];
            return HCCL_SUCCESS;
        }
    }
    CHK_RET_CCU(HcommCcuGetMemToken(va, size, &token));
    const uint32_t slot = g_tokenCache.next;
    g_tokenCache.va[slot] = va;
    g_tokenCache.size[slot] = size;
    g_tokenCache.token[slot] = token;
    g_tokenCache.next = (slot + 1U) % ScatterTokenCache::ENTRY_NUM;
    return HCCL_SUCCESS;
}

uint32_t ScatterArgNumOf(uint32_t kind, uint32_t channelCount, bool takeOwnCopy)
{
    if (kind == SCATTER_KIND_PULL_ROOT) {
        return static_cast<uint32_t>(SC_PR_ARG_FIXED_NUM)
            + (takeOwnCopy ? static_cast<uint32_t>(SC_PR_OWN_ARG_NUM) : 0U);
    }
    return static_cast<uint32_t>(SC_PUSH_ARG_FIXED_NUM) + SCATTER_PUSH_ARG_GROUPS * channelCount;
}

HcclResult PrepareCache(const OpParam &param)
{
    if (g_execCache.resCtx == param.resCtx && g_execCache.ctxSize == param.ctxSize) {
        return HCCL_SUCCESS;
    }

    char *ctx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(ctx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    g_execCache.ctx.DeSerialize(seq);

    const AlgResourceCtx &resCtx = g_execCache.ctx;

    ScatterGroupChannelsByDie(resCtx.myRank, resCtx.channelDie, resCtx.serverId, g_execCache.dieChannelIdx);
    uint64_t grouped = 0;
    for (uint32_t die = 0; die < SCATTER_DIE_NUM; die++) {
        grouped += g_execCache.dieChannelIdx[die].size();
    }
    if (grouped != resCtx.channelDie.size()) {
        HCCL_ERROR("[PrepareCache] grouped[%llu] != channelNum[%llu], die id out of range?",
            static_cast<unsigned long long>(grouped), static_cast<unsigned long long>(resCtx.channelDie.size()));
        return HCCL_E_INTERNAL;
    }

    g_execCache.ownCopyDie = ScatterOwnCopyDie(g_execCache.dieChannelIdx);

    g_execCache.resCtx = param.resCtx;
    g_execCache.ctxSize = param.ctxSize;
    return HCCL_SUCCESS;
}

std::vector<uint64_t> &BeginLaunch(uint32_t idx, uint32_t kind, uint32_t die, uint32_t chCnt,
    bool takeOwnCopy = false)
{
    g_launch.kernel[idx] = g_execCache.ctx.ccuKernels[static_cast<size_t>(kind) * SCATTER_DIE_NUM + die];
    std::vector<uint64_t> &args = g_launch.args[idx];
    args.resize(static_cast<size_t>(ScatterArgNumOf(kind, chCnt, takeOwnCopy)));
    return args;
}

uint64_t RecvBytesFrom(const OpParam &param, const ScatterPlan &plan, uint32_t peer, uint64_t pieceBytes)
{
    const uint32_t parent = plan.relayOn ? plan.relayParent[param.myRank] : INVALID_VALUE_RANKID;
    if (peer == param.root) {
        return (parent == INVALID_VALUE_RANKID) ? pieceBytes : plan.directBytes;
    }
    if (peer == parent) {
        return plan.relayBytes;
    }
    return 0;
}

void FillPullRootArgs(const OpParam &param, const ScatterPlan &plan, uint64_t inToken, uint64_t outToken,
    uint64_t pieceOff, uint64_t pieceBytes, bool takeOwnCopy, std::vector<uint64_t> &args)
{
    const uint64_t inBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outBase = reinterpret_cast<uint64_t>(param.outputPtr);

    args[SC_PR_ARG_IN_BASE] = inBase;
    args[SC_PR_ARG_IN_TOKEN] = inToken;

    if (takeOwnCopy) {
        const uint64_t ownDst = outBase + pieceOff;
        const uint64_t ownSrc = inBase + static_cast<uint64_t>(param.myRank) * plan.shareBytes + pieceOff;
        const uint32_t ownBase = static_cast<uint32_t>(SC_PR_ARG_FIXED_NUM);
        args[ownBase + SC_PR_OWN_OFF_DST_BASE] = ownDst;
        args[ownBase + SC_PR_OWN_OFF_DST_TOKEN] = outToken;
        args[ownBase + SC_PR_OWN_OFF_SRC_ADDR] = ownSrc;

        args[ownBase + SC_PR_OWN_OFF_SIZE] = (ownSrc == ownDst) ? 0ULL : pieceBytes;
    }

}

void FillPullNodeArgs(const OpParam &param, const ScatterPlan &plan, uint32_t pullCh, uint64_t outToken,
    uint64_t pieceOff, uint64_t pieceBytes, uint32_t chCnt, std::vector<uint64_t> &args)
{
    args[SC_PUSH_ARG_ROLE] = SC_PUSH_ROLE_NODE_PULL;
    args[SC_PUSH_ARG_HAS_PUB] = 0ULL;
    args[SC_PUSH_ARG_Y1] = reinterpret_cast<uint64_t>(param.outputPtr) + pieceOff;
    args[SC_PUSH_ARG_Y2] = outToken;
    args[SC_PUSH_ARG_Y3] = pullCh;
    args[SC_PUSH_ARG_Y4] = 0ULL;

    args[SC_PUSH_ARG_Y5] = static_cast<uint64_t>(param.myRank) * plan.shareBytes + pieceOff;
    args[SC_PUSH_ARG_Y6] = 0ULL;
    args[SC_PUSH_ARG_Y7] = pieceBytes;
    args[SC_PUSH_ARG_Y8] = 0ULL;
    const uint32_t pcBase = static_cast<uint32_t>(SC_PUSH_ARG_FIXED_NUM);
    for (uint32_t c = 0; c < chCnt; c++) {
        args[pcBase + c] = 0ULL;
    }
}

void FillPushRootArgs(const OpParam &param, const ScatterPlan &plan, const std::vector<uint32_t> &dieChannelIdx,
    uint64_t inToken, uint64_t outToken, uint64_t pieceOff, uint64_t pieceBytes, bool takeOwnCopy,
    std::vector<uint64_t> &args)
{
    const uint32_t myRank = param.myRank;
    const uint64_t inBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outBase = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t share = plan.shareBytes;
    const uint32_t chCnt = static_cast<uint32_t>(dieChannelIdx.size());

    args[SC_PUSH_ARG_ROLE] = SC_PUSH_ROLE_ROOT;

    args[SC_PUSH_ARG_HAS_PUB] = plan.relayOn ? 1ULL : 0ULL;
    args[SC_PUSH_ARG_Y1] = inBase;
    args[SC_PUSH_ARG_Y2] = inToken;

    args[SC_PUSH_ARG_Y3] = pieceBytes;
    args[SC_PUSH_ARG_Y4] = plan.relayOn ? plan.directBytes : pieceBytes;

    if (takeOwnCopy) {
        const uint64_t ownDst = outBase + pieceOff;
        const uint64_t ownSrc = inBase + static_cast<uint64_t>(myRank) * share + pieceOff;
        args[SC_PUSH_ARG_Y5] = ownDst;
        args[SC_PUSH_ARG_Y6] = outToken;
        args[SC_PUSH_ARG_Y7] = ownSrc;
        args[SC_PUSH_ARG_Y8] = (ownSrc == ownDst) ? 0ULL : pieceBytes;
    } else {
        args[SC_PUSH_ARG_Y5] = 0ULL;
        args[SC_PUSH_ARG_Y6] = 0ULL;
        args[SC_PUSH_ARG_Y7] = 0ULL;
        args[SC_PUSH_ARG_Y8] = 0ULL;
    }

    const uint32_t pcBase = static_cast<uint32_t>(SC_PUSH_ARG_FIXED_NUM);
    for (uint32_t c = 0; c < chCnt; c++) {
        const uint32_t peer = ScatterPeerRankOf(dieChannelIdx[c], myRank);
        args[pcBase + c] = inBase + static_cast<uint64_t>(peer) * share + pieceOff;
    }
}

bool FillPushNodeArgs(const OpParam &param, const ScatterPlan &plan, const AlgResourceCtx &resCtx,
    const std::vector<uint32_t> &dieChannelIdx, uint64_t outToken, uint64_t pieceOff, uint64_t pieceBytes, bool doPull,
    bool doFwd, bool doWait, std::vector<uint64_t> &args)
{
    const uint32_t myRank = param.myRank;
    const uint64_t share = plan.shareBytes;
    const uint32_t target = plan.myRelayTarget;
    const bool isRelay = plan.IsRelay();
    const uint32_t chCnt = static_cast<uint32_t>(dieChannelIdx.size());

    uint64_t phase = SC_NODE_PHASE_NONE;
    if (doPull && doFwd) {
        phase = SC_NODE_PHASE_BOTH;
    } else if (doPull) {
        phase = SC_NODE_PHASE_PULL;
    } else if (doFwd) {
        phase = SC_NODE_PHASE_FWD;
    } else if (isRelay) {
        phase = SC_NODE_PHASE_WAIT;
    }

    args[SC_PUSH_ARG_ROLE] = SC_PUSH_ROLE_NODE;
    args[SC_PUSH_ARG_HAS_PUB] = 0ULL;

    args[SC_PUSH_ARG_Y1] = reinterpret_cast<uint64_t>(param.outputPtr) + pieceOff;
    args[SC_PUSH_ARG_Y2] = outToken;
    args[SC_PUSH_ARG_Y3] = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    args[SC_PUSH_ARG_Y4] = resCtx.localBufferToken;

    args[SC_PUSH_ARG_Y5]
        = isRelay ? (static_cast<uint64_t>(target) * share + pieceOff + plan.directBytes) : 0ULL;

    args[SC_PUSH_ARG_Y6] = isRelay ? plan.directBytes : 0ULL;
    args[SC_PUSH_ARG_Y7] = isRelay ? plan.relayBytes : 0ULL;
    args[SC_PUSH_ARG_Y8] = phase;

    const uint32_t pcBase = static_cast<uint32_t>(SC_PUSH_ARG_FIXED_NUM);
    bool hasWork = false;
    for (uint32_t c = 0; c < chCnt; c++) {
        const uint32_t peer = ScatterPeerRankOf(dieChannelIdx[c], myRank);
        uint64_t mode = SC_NODE_MODE_IDLE;
        if (doPull && peer == param.root) {
            mode = SC_NODE_MODE_PULL;
        } else if (doFwd && peer == target) {
            mode = SC_NODE_MODE_FWD;
        } else if (doWait && RecvBytesFrom(param, plan, peer, pieceBytes) != 0) {
            mode = isRelay ? SC_NODE_MODE_WAIT : SC_NODE_MODE_RECV;
        }
        args[pcBase + c] = mode;
        if (mode != SC_NODE_MODE_IDLE) {
            hasWork = true;
        }
    }
    return hasWork;
}

bool ScatterFindRootChannel(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<std::vector<uint32_t>> &dieChannelIdx, uint32_t &dieOut, uint32_t &idxOut)
{
    const uint32_t rootCh = ScatterChannelOf(param.root, param.myRank);
    if (rootCh >= resCtx.channelDie.size()) {
        return false;
    }
    const uint32_t die = resCtx.channelDie[rootCh];
    if (die >= dieChannelIdx.size()) {
        return false;
    }
    const std::vector<uint32_t> &list = dieChannelIdx[die];
    for (uint32_t i = 0; i < list.size(); i++) {
        if (list[i] == rootCh) {
            dieOut = die;
            idxOut = i;
            return true;
        }
    }
    return false;
}

HcclResult LaunchParallel(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const uint32_t num = g_launch.num;
    if (num == 0) {
        return HCCL_SUCCESS;
    }

    ThreadHandle threads[SCATTER_MAX_LAUNCH];
    threads[0] = param.cpuThread;
    for (uint32_t i = 1; i < num; i++) {
        if (i >= resCtx.threads.size() || resCtx.threads[i] == 0) {
            return HCCL_E_INTERNAL;
        }
        threads[i] = resCtx.threads[i];
    }

    for (uint32_t i = 1; i < num; i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[0], threads[i], SCATTER_INTER_THREAD_NOTIFY_IDX)));
    }
    for (uint32_t i = 1; i < num; i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[i], SCATTER_INTER_THREAD_NOTIFY_IDX, CUSTOM_TIMEOUT)));
    }

    for (uint32_t i = 0; i < num; i++) {
        if (g_launch.kernel[i] == 0) {
            HCCL_ERROR("[LaunchParallel] kernel[%u] not registered", i);
            return HCCL_E_INTERNAL;
        }
        const CcuResult ret = HcommCcuKernelLaunch(
            threads[i], g_launch.kernel[i], g_launch.args[i].data(), static_cast<uint32_t>(g_launch.args[i].size()));
        if (ret != CCU_SUCCESS) {
            return ConvertCcuToHccl(ret);
        }
    }

    for (uint32_t i = 1; i < num; i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[0], SCATTER_INTER_THREAD_NOTIFY_IDX + i - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t i = 1; i < num; i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[i], threads[0], SCATTER_INTER_THREAD_NOTIFY_IDX + i - 1)));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSequential(const OpParam &param)
{
    for (uint32_t i = 0; i < g_launch.num; i++) {
        if (g_launch.kernel[i] == 0) {
            HCCL_ERROR("[LaunchSequential] kernel[%u] not registered", i);
            return HCCL_E_INTERNAL;
        }
        const CcuResult ret = HcommCcuKernelLaunch(param.cpuThread, g_launch.kernel[i], g_launch.args[i].data(),
            static_cast<uint32_t>(g_launch.args[i].size()));
        if (ret != CCU_SUCCESS) {
            return ConvertCcuToHccl(ret);
        }
    }
    return HCCL_SUCCESS;
}

} // namespace

bool TryGetCachedRankInfo(const void *resCtx, uint64_t ctxSize, uint32_t &myRank, uint32_t &rankSize)
{
    if (resCtx == nullptr || g_execCache.resCtx != resCtx || g_execCache.ctxSize != ctxSize) {
        return false;
    }
    if (g_execCache.ctx.rankSize == 0 || g_execCache.ctx.myRank == INVALID_VALUE_RANKID) {
        return false;
    }
    myRank = g_execCache.ctx.myRank;
    rankSize = g_execCache.ctx.rankSize;
    return true;
}

HcclResult ExecOp(const OpParam &param)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_RET(PrepareCache(param));

    const AlgResourceCtx &resCtx = g_execCache.ctx;
    const std::vector<std::vector<uint32_t>> &dieChannelIdx = g_execCache.dieChannelIdx;
    const uint32_t rankSize = param.rankSize;

    const uint64_t dtSize = ScatterDataTypeSize(param.dataType);
    const uint64_t shareBytes = param.count * dtSize;

    const uint64_t maxCountPerPiece = static_cast<uint64_t>(MAX_DATA_SIZE) / dtSize;
    const uint64_t pieceNum = (param.count + maxCountPerPiece - 1) / maxCountPerPiece;

    const uint32_t allowRelay = (pieceNum == 1) ? 1U : 0U;
    if (g_planCache.resCtx != param.resCtx || g_planCache.root != param.root
        || g_planCache.shareBytes != shareBytes || g_planCache.allowRelay != allowRelay) {
        g_planCache.plan = ScatterPlan();
        ScatterBuildPlan(rankSize, param.root, param.myRank, resCtx.serverId, shareBytes, resCtx.localBuffer.size,
            allowRelay != 0, g_planCache.plan);
        g_planCache.resCtx = param.resCtx;
        g_planCache.root = param.root;
        g_planCache.shareBytes = shareBytes;
        g_planCache.allowRelay = allowRelay;
    }
    const ScatterPlan &plan = g_planCache.plan;

    if (plan.IsRelay() && resCtx.localBufferToken == 0) {
        HCCL_ERROR("[ExecOp] relay planned but hccl buffer token is unavailable");
        return HCCL_E_UNAVAIL;
    }

    uint64_t inToken = 0;
    uint64_t outToken = 0;
    if (plan.role == SCATTER_ROLE_ROOT) {
        CHK_RET(ScatterGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), shareBytes * rankSize, inToken));
    }
    CHK_RET(ScatterGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), shareBytes, outToken));

    uint64_t processed = 0;
    for (uint64_t piece = 0; piece < pieceNum; piece++) {
        const uint64_t pieceCount = std::min(maxCountPerPiece, param.count - processed);
        const uint64_t pieceOff = processed * dtSize;
        const uint64_t pieceBytes = pieceCount * dtSize;
        processed += pieceCount;
        g_launch.num = 0;

        if (ScatterPullMode(plan.shareBytes, pieceBytes)) {
            if (plan.role == SCATTER_ROLE_ROOT) {
                for (uint32_t die = 0; die < SCATTER_DIE_NUM; die++) {
                    if (dieChannelIdx[die].empty()) {
                        continue;
                    }
                    const uint32_t chCnt = static_cast<uint32_t>(dieChannelIdx[die].size());
                    const bool takeOwnCopy = (die == g_execCache.ownCopyDie);
                    FillPullRootArgs(param, plan, inToken, outToken, pieceOff, pieceBytes, takeOwnCopy,
                        BeginLaunch(g_launch.num, SCATTER_KIND_PULL_ROOT, die, chCnt, takeOwnCopy));
                    g_launch.num++;
                }
            } else {
                uint32_t rootDie = 0;
                uint32_t pullCh = 0;
                if (!ScatterFindRootChannel(param, resCtx, dieChannelIdx, rootDie, pullCh)) {
                    HCCL_ERROR("[ExecOp] cannot locate root[%u] channel for rank[%u]", param.root, param.myRank);
                    return HCCL_E_INTERNAL;
                }
                const uint32_t chCnt = static_cast<uint32_t>(dieChannelIdx[rootDie].size());
                FillPullNodeArgs(param, plan, pullCh, outToken, pieceOff, pieceBytes, chCnt,
                    BeginLaunch(g_launch.num, SCATTER_KIND_PUSH, rootDie, chCnt));
                g_launch.num++;
            }
            CHK_RET(LaunchParallel(param, resCtx));
            continue;
        }

        if (plan.IsRelay()) {
            const uint32_t dieRoot = resCtx.channelDie[ScatterChannelOf(param.root, param.myRank)];
            const uint32_t dieTgt = resCtx.channelDie[ScatterChannelOf(plan.myRelayTarget, param.myRank)];
            const uint32_t rootChCnt = static_cast<uint32_t>(dieChannelIdx[dieRoot].size());

            if (dieRoot == dieTgt) {
                (void)FillPushNodeArgs(param, plan, resCtx, dieChannelIdx[dieRoot], outToken, pieceOff, pieceBytes,
                    true, true, false, BeginLaunch(g_launch.num++, SCATTER_KIND_PUSH, dieRoot, rootChCnt));
            } else {
                (void)FillPushNodeArgs(param, plan, resCtx, dieChannelIdx[dieRoot], outToken, pieceOff, pieceBytes,
                    true, false, false, BeginLaunch(g_launch.num++, SCATTER_KIND_PUSH, dieRoot, rootChCnt));
                const uint32_t tgtChCnt = static_cast<uint32_t>(dieChannelIdx[dieTgt].size());
                (void)FillPushNodeArgs(param, plan, resCtx, dieChannelIdx[dieTgt], outToken, pieceOff, pieceBytes,
                    false, true, false, BeginLaunch(g_launch.num++, SCATTER_KIND_PUSH, dieTgt, tgtChCnt));
            }
            (void)FillPushNodeArgs(param, plan, resCtx, dieChannelIdx[dieRoot], outToken, pieceOff, pieceBytes, false,
                false, true, BeginLaunch(g_launch.num++, SCATTER_KIND_PUSH, dieRoot, rootChCnt));

            CHK_RET(LaunchSequential(param));
            continue;
        }

        for (uint32_t die = 0; die < SCATTER_DIE_NUM; die++) {
            if (dieChannelIdx[die].empty()) {
                continue;
            }
            const uint32_t chCnt = static_cast<uint32_t>(dieChannelIdx[die].size());
            std::vector<uint64_t> &args = BeginLaunch(g_launch.num, SCATTER_KIND_PUSH, die, chCnt);
            if (plan.role == SCATTER_ROLE_ROOT) {
                const bool takeOwnCopy = (die == g_execCache.ownCopyDie);
                FillPushRootArgs(
                    param, plan, dieChannelIdx[die], inToken, outToken, pieceOff, pieceBytes, takeOwnCopy, args);
                g_launch.num++;
            } else if (FillPushNodeArgs(param, plan, resCtx, dieChannelIdx[die], outToken, pieceOff, pieceBytes, false,
                           false, true, args)) {
                g_launch.num++;
            }
        }
        CHK_RET(LaunchParallel(param, resCtx));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
