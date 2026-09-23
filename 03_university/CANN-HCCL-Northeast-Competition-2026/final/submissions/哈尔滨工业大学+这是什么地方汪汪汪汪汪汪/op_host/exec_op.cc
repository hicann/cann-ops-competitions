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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "ccu_kernel.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

constexpr uint32_t TASK_ARG_NUM = 10;
constexpr uint32_t SMALL_RECEIVER_TASK_ARG_NUM = 4;
constexpr uint32_t RANK12_SMALL_ROOT_TASK_ARG_NUM = 6;
constexpr uint32_t HELPER_TASK_ARG_NUM = 12;
constexpr uint32_t WIDE_PUSH_TASK_ARG_NUM = 8;
constexpr uint64_t HELPER_THRESHOLD_BYTES = 1024ULL * 1024ULL;
constexpr uint64_t TARGETED_PULL_TOTAL_BYTES = 512ULL * 1024ULL;
constexpr uint64_t RANK16_TARGET_BLOCK_BYTES = 32ULL * 1024ULL;
constexpr uint64_t DMA_BOUNDARY_ALIGNMENT_BYTES = 32ULL;
constexpr uint32_t CLOS_CAPACITY = 4;
constexpr uint32_t DESCRIPTOR_CACHE_NONE = 0;
constexpr uint32_t DESCRIPTOR_CACHE_PUSH_RECEIVER = 1;
constexpr uint32_t DESCRIPTOR_CACHE_PULL_ROOT = 2;
constexpr uint32_t DESCRIPTOR_PATH_GENERIC_RECEIVER = 1;
constexpr uint32_t DESCRIPTOR_PATH_SMALL_RECEIVER = 2;
constexpr uint32_t DESCRIPTOR_PATH_PULL_ROOT = 3;

struct HelperRoute {
    bool enabled{false};
    uint32_t helperTarget{INVALID_VALUE_RANKID};
    uint32_t senderRank{INVALID_VALUE_RANKID};
    uint32_t localPeerCount{0};
    uint32_t remotePeerCount{0};
};

bool AddOverflows(uint64_t base, uint64_t length)
{
    return base > std::numeric_limits<uint64_t>::max() - length;
}

uint64_t AlignHelperBoundary(uint64_t bytes)
{
    return bytes / DMA_BOUNDARY_ALIGNMENT_BYTES *
        DMA_BOUNDARY_ALIGNMENT_BYTES;
}

bool IsEightPlusFourTopology(const AlgResourceCtx &resCtx)
{
    if (resCtx.rankSize != 12U || resCtx.instanceCount != 2U) {
        return false;
    }
    const uint32_t first = resCtx.instanceSizes[0];
    const uint32_t second = resCtx.instanceSizes[1];
    return (first == 8U && second == 4U) ||
        (first == 4U && second == 8U);
}

HelperRoute BuildHelperRoute(const AlgResourceCtx &resCtx, const OpParam &param)
{
    HelperRoute route;
    if (resCtx.helperTopologyValid == 0 || resCtx.rankSize != param.rankSize) {
        return route;
    }
    const uint32_t myGroup = resCtx.localGroupByRank[param.myRank];
    const bool rootIsLocal = resCtx.localGroupByRank[param.root] == myGroup;
    uint32_t myInstanceSize = 0;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        myInstanceSize += resCtx.localGroupByRank[rank] == myGroup ? 1U : 0U;
    }
    uint32_t rootInstanceSize = 0;
    if (rootIsLocal) {
        rootInstanceSize = myInstanceSize;
    } else if (resCtx.instanceCount == 2) {
        rootInstanceSize = param.rankSize - myInstanceSize;
    } else if (resCtx.instanceCount != 0) {
        rootInstanceSize = resCtx.instanceSizes[0];
        for (uint32_t i = 0; i < resCtx.instanceCount; ++i) {
            const uint32_t size = resCtx.instanceSizes[i];
            if (size != rootInstanceSize) {
                return route;
            }
        }
    }
    if (rootInstanceSize == 0 || rootInstanceSize > param.rankSize) {
        return route;
    }
    route.localPeerCount = rootInstanceSize - 1U;
    route.remotePeerCount = param.rankSize - rootInstanceSize;
    if (route.localPeerCount == 0 || route.remotePeerCount <= CLOS_CAPACITY ||
        route.localPeerCount > route.remotePeerCount) {
        return route;
    }
    route.enabled = true;
    route.senderRank = param.root;
    if (!rootIsLocal) {
        // The root publishes the chosen sender as a dynamic CCU variable.  A
        // remote receiver must not guess membership of the root's L0 instance.
        return route;
    }

    const uint32_t rootGroup = myGroup;
    std::vector<uint32_t> helpers;
    std::vector<uint32_t> remoteTargets;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.root) {
            continue;
        }
        if (resCtx.localGroupByRank[rank] == rootGroup) {
            helpers.push_back(rank);
        } else {
            remoteTargets.push_back(rank);
        }
    }
    if (helpers.size() != route.localPeerCount ||
        remoteTargets.size() != route.remotePeerCount) {
        route.enabled = false;
        return route;
    }
    for (uint32_t i = 0; i < helpers.size(); ++i) {
        if (param.myRank == helpers[i]) {
            route.helperTarget = remoteTargets[i];
        }
        if (param.myRank == remoteTargets[i]) {
            route.senderRank = helpers[i];
        }
    }
    return route;
}

HcclResult LaunchGroups(const OpParam &param, const AlgResourceCtx &resCtx,
    const CcuKernelHandle *kernels, const uint64_t *taskArgs,
    uint32_t taskArgNum, bool useMain, bool useSub)
{
    if (useSub) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            param.cpuThread, resCtx.threads[1], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
    }
    if (useMain) {
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, kernels[0],
            taskArgs, taskArgNum));
    }
    if (useSub) {
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[1], kernels[1],
            taskArgs, taskArgNum));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            param.cpuThread, 0, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[1], param.cpuThread, 0)));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchOneGroup(const OpParam &param, const AlgResourceCtx &resCtx,
    const CcuKernelHandle *kernels, uint32_t kernelCount, const uint64_t *taskArgs,
    uint32_t taskArgNum, uint32_t group)
{
    CHK_PRT_RET(group >= kernelCount,
        HCCL_ERROR("Invalid CCU kernel group %u", group), HCCL_E_INTERNAL);
    return LaunchGroups(param, resCtx, kernels, taskArgs, taskArgNum,
        group == 0U, group == 1U);
}

bool DescriptorMatches(const DescriptorCacheEntry &entry, uint64_t addr,
    uint64_t token)
{
    return entry.addr == addr && entry.token == token && addr != 0 && token != 0;
}

HcclResult UpdateDescriptorCache(HcclComm comm, const OpParam &param,
    uint64_t offset, const DescriptorCacheEntry &entry)
{
    return HcclEngineCtxCopy(comm, COMM_ENGINE_CCU, param.tag, &entry,
        sizeof(entry), offset);
}

HcclResult UpdateDescriptorCacheState(HcclComm comm, const OpParam &param,
    const DescriptorCacheState &state)
{
    return HcclEngineCtxCopy(comm, COMM_ENGINE_CCU, param.tag, &state,
        sizeof(state), offsetof(AlgResourceCtx, lastDescriptorCache));
}

} // namespace

HcclResult ExecOp(HcclComm comm, const OpParam &param, uint64_t blockBytes)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(param.resCtx);
    CHK_PRT_RET(param.ctxSize != sizeof(AlgResourceCtx),
        HCCL_ERROR("Invalid fixed resource context size"), HCCL_E_INTERNAL);
    const auto &resCtx = *static_cast<const AlgResourceCtx *>(param.resCtx);

    CHK_PRT_RET(resCtx.magic != AlgResourceCtx::MAGIC ||
        resCtx.version != AlgResourceCtx::VERSION ||
        resCtx.myRank != param.myRank || resCtx.rankSize != param.rankSize ||
        resCtx.ccuKernelCount == 0 || resCtx.ccuKernelCount > 2 ||
        resCtx.smallReceiverKernelCount > resCtx.ccuKernelCount ||
        ((resCtx.rankSize == 4U || resCtx.rank12SmallSpecialized != 0) &&
            resCtx.smallReceiverKernelCount != resCtx.ccuKernelCount) ||
        (resCtx.rankSize != 4U && resCtx.rank12SmallSpecialized == 0 &&
            resCtx.smallReceiverKernelCount != 0) ||
        (resCtx.helperTopologyValid != 0 &&
            resCtx.rank12SmallSpecialized == 0 &&
            !IsEightPlusFourTopology(resCtx) &&
            resCtx.helperKernelCount != resCtx.ccuKernelCount) ||
        ((resCtx.helperTopologyValid == 0 ||
            resCtx.rank12SmallSpecialized != 0 ||
            IsEightPlusFourTopology(resCtx)) &&
            resCtx.helperKernelCount != 0) ||
        (IsEightPlusFourTopology(resCtx) &&
            resCtx.rank12SmallSpecialized == 0 &&
            resCtx.widePushKernelCount != resCtx.ccuKernelCount) ||
        ((!IsEightPlusFourTopology(resCtx) ||
            resCtx.rank12SmallSpecialized != 0) &&
            resCtx.widePushKernelCount != 0) ||
        (resCtx.rank4SmallSpecialized != 0 && resCtx.rankSize != 4U) ||
        resCtx.threadCount != resCtx.ccuKernelCount,
        HCCL_ERROR("Invalid CCU resource counts: primary=%u smallReceiver=%u helper=%u widePush=%u threads=%u rank12Small=%u rank4Small=%u",
            resCtx.ccuKernelCount, resCtx.smallReceiverKernelCount,
            resCtx.helperKernelCount, resCtx.widePushKernelCount,
            resCtx.threadCount, resCtx.rank12SmallSpecialized,
            resCtx.rank4SmallSpecialized),
        HCCL_E_INTERNAL);
    if (blockBytes == 0) {
        if (resCtx.lastDescriptorCache.kind != DESCRIPTOR_CACHE_NONE) {
            const DescriptorCacheState empty{};
            CHK_RET(UpdateDescriptorCacheState(comm, param, empty));
        }
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(blockBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("Scatter total input byte calculation overflows"), HCCL_E_PARA);
    const uint64_t totalBytes = blockBytes * param.rankSize;

    const uint64_t recvBase = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t sendBase = reinterpret_cast<uint64_t>(param.inputPtr);
    CHK_PRT_RET(AddOverflows(recvBase, blockBytes),
        HCCL_ERROR("Receive address range overflows"), HCCL_E_PARA);
    if (param.myRank == param.root) {
        CHK_PRT_RET(AddOverflows(sendBase, totalBytes),
            HCCL_ERROR("Send address range overflows"), HCCL_E_PARA);
    }

    uint64_t recvToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(recvBase, blockBytes, &recvToken));
    uint64_t sendToken = 0;
    if (param.myRank == param.root) {
        CHK_RET_CCU(HcommCcuGetMemToken(sendBase, totalBytes, &sendToken));
    }

    uint64_t rootSource = 0;
    if (param.myRank == param.root) {
        rootSource = sendBase + static_cast<uint64_t>(param.root) * blockBytes;
    }
    const uint64_t rootSliceInPlace =
        (param.myRank == param.root && rootSource == recvBase) ? 1U : 0U;
    const bool isRoot = param.myRank == param.root;
    const uint32_t activeGroup = isRoot ? 0U : resCtx.peerKernelGroups[param.root];
    CHK_PRT_RET(!isRoot && activeGroup >= resCtx.ccuKernelCount,
        HCCL_ERROR("No CCU kernel group for root %u", param.root), HCCL_E_INTERNAL);
    const bool helperSizeEligible = blockBytes >= HELPER_THRESHOLD_BYTES;
    const bool smallCall = !helperSizeEligible;
    const bool rank16PullCall = smallCall && param.rankSize == 16U &&
        blockBytes == RANK16_TARGET_BLOCK_BYTES;
    const bool rank4SmallCall = param.rankSize == 4U &&
        totalBytes <= TARGETED_PULL_TOTAL_BYTES;
    const bool rank12SmallCall = resCtx.rank12SmallSpecialized != 0;
    const bool specializedSmallRootCall = rank12SmallCall ||
        resCtx.rank4SmallSpecialized != 0;
    const bool smallReceiverKernel = !isRoot &&
        (specializedSmallRootCall || rank4SmallCall);
    const uint32_t pushDescriptorPath = smallReceiverKernel ?
        DESCRIPTOR_PATH_SMALL_RECEIVER : DESCRIPTOR_PATH_GENERIC_RECEIVER;
    const uint32_t descriptorPath = rank16PullCall ? DESCRIPTOR_PATH_PULL_ROOT :
        pushDescriptorPath;
    // Cache reuse is limited to the immediately preceding call on the stream
    // which built the resource context.  Its queue (including the two-die
    // join) orders a cache-miss publication before the next cache-hit READY.
    // Any intervening call, including a different root/path or another stream,
    // clears lastDescriptorCache below before a later call can reuse an XN.
    const bool cacheStreamOrdered = param.cpuThread == resCtx.ccuThread;
    const bool pushDescriptorReuse = smallCall && !rank16PullCall && !isRoot &&
        cacheStreamOrdered &&
        resCtx.lastDescriptorCache.kind == DESCRIPTOR_CACHE_PUSH_RECEIVER &&
        resCtx.lastDescriptorCache.root == param.root &&
        resCtx.lastDescriptorCache.path == pushDescriptorPath &&
        DescriptorMatches(resCtx.pushRecvCache[param.root],
            recvBase, recvToken);
    const bool pullDescriptorReuse = rank16PullCall && isRoot &&
        cacheStreamOrdered &&
        resCtx.lastDescriptorCache.kind == DESCRIPTOR_CACHE_PULL_ROOT &&
        resCtx.lastDescriptorCache.root == param.root &&
        resCtx.lastDescriptorCache.path == DESCRIPTOR_PATH_PULL_ROOT &&
        DescriptorMatches(resCtx.pullSendCache,
            sendBase, sendToken);
    const uint64_t metadataReuse =
        (pushDescriptorReuse || pullDescriptorReuse) ? 1U : 0U;

    HelperRoute route;
    if (helperSizeEligible) {
        route = BuildHelperRoute(resCtx, param);
    }
    const bool helperCall = helperSizeEligible && route.enabled &&
        resCtx.helperKernelCount == resCtx.ccuKernelCount;
    const bool widePushCall = helperSizeEligible &&
        IsEightPlusFourTopology(resCtx) &&
        route.localPeerCount > route.remotePeerCount &&
        resCtx.widePushKernelCount == resCtx.ccuKernelCount;
    const uint64_t helperNumerator = route.remotePeerCount > CLOS_CAPACITY ?
        route.remotePeerCount - CLOS_CAPACITY : 0;
    const uint64_t helperDenominator = route.localPeerCount + CLOS_CAPACITY;
    const uint64_t maxChunkBytes = std::min<uint64_t>(MAX_DATA_SIZE, blockBytes);
    const uint64_t maxDelegatedBytes = helperCall && helperDenominator != 0 ?
        AlignHelperBoundary(maxChunkBytes * helperNumerator /
            helperDenominator) : 0;
    uint64_t stageBase = 0;
    uint64_t stageToken = 0;
    if (maxDelegatedBytes != 0) {
        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr ||
            resCtx.localBuffer.size < maxDelegatedBytes,
            HCCL_ERROR("HCCL staging buffer is too small: need=%llu available=%llu",
                static_cast<unsigned long long>(maxDelegatedBytes),
                static_cast<unsigned long long>(resCtx.localBuffer.size)),
            HCCL_E_NOT_SUPPORT);
        if (route.helperTarget != INVALID_VALUE_RANKID) {
            stageBase = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
            CHK_RET_CCU(HcommCcuGetMemToken(stageBase, maxDelegatedBytes, &stageToken));
        }
    }

    for (uint64_t chunkOffset = 0; chunkOffset < blockBytes;) {
        const uint64_t chunkBytes = std::min<uint64_t>(MAX_DATA_SIZE, blockBytes - chunkOffset);
        uint64_t delegatedBytes = 0;
        if (helperCall && helperDenominator != 0) {
            delegatedBytes = AlignHelperBoundary(chunkBytes * helperNumerator /
                helperDenominator);
        }
        const bool useHelperChunk = delegatedBytes != 0 && delegatedBytes < chunkBytes;
        if (widePushCall) {
            const uint64_t sendChunkBase = sendBase + chunkOffset;
            const uint64_t recvChunkBase = recvBase + chunkOffset;
            const uint64_t widePushArgs[WIDE_PUSH_TASK_ARG_NUM] = {
                sendChunkBase,
                recvChunkBase,
                sendToken,
                recvToken,
                blockBytes,
                chunkBytes,
                param.root,
                rootSliceInPlace,
            };
            const bool useMain = isRoot || activeGroup == 0U;
            const bool useSub = resCtx.widePushKernelCount > 1 &&
                (isRoot || activeGroup == 1U);
            CHK_RET(LaunchGroups(param, resCtx, resCtx.widePushKernels,
                widePushArgs, WIDE_PUSH_TASK_ARG_NUM, useMain, useSub));
            chunkOffset += chunkBytes;
            continue;
        }
        if (useHelperChunk) {
            const uint64_t sendChunkBase = sendBase + chunkOffset;
            const uint64_t recvChunkBase = recvBase + chunkOffset;
            uint64_t helperArgs[HELPER_TASK_ARG_NUM] = {
                sendChunkBase,
                recvChunkBase,
                sendToken,
                recvToken,
                stageBase,
                stageToken,
                blockBytes,
                chunkBytes - delegatedBytes,
                param.root,
                rootSliceInPlace,
                delegatedBytes,
                SCATTER_HELPER_NORMAL,
            };
            if (isRoot) {
                CHK_RET(LaunchGroups(param, resCtx, resCtx.helperKernels,
                    helperArgs, HELPER_TASK_ARG_NUM, true,
                    resCtx.helperKernelCount > 1));
            } else if (route.helperTarget != INVALID_VALUE_RANKID) {
                const uint32_t rootGroup = resCtx.peerKernelGroups[param.root];
                const uint32_t targetGroup = resCtx.peerKernelGroups[route.helperTarget];
                helperArgs[11] = SCATTER_HELPER_STAGE;
                CHK_RET(LaunchOneGroup(param, resCtx, resCtx.helperKernels,
                    resCtx.helperKernelCount, helperArgs, HELPER_TASK_ARG_NUM, rootGroup));
                helperArgs[11] = SCATTER_HELPER_FORWARD;
                CHK_RET(LaunchOneGroup(param, resCtx, resCtx.helperKernels,
                    resCtx.helperKernelCount, helperArgs, HELPER_TASK_ARG_NUM, targetGroup));
                helperArgs[11] = SCATTER_HELPER_FINISH;
                CHK_RET(LaunchOneGroup(param, resCtx, resCtx.helperKernels,
                    resCtx.helperKernelCount, helperArgs, HELPER_TASK_ARG_NUM, rootGroup));
            } else {
                const uint32_t rootGroup = resCtx.peerKernelGroups[param.root];
                const uint32_t senderGroup = resCtx.peerKernelGroups[route.senderRank];
                CHK_RET(LaunchGroups(param, resCtx, resCtx.helperKernels,
                    helperArgs, HELPER_TASK_ARG_NUM,
                    rootGroup == 0U || senderGroup == 0U,
                    resCtx.helperKernelCount > 1 &&
                        (rootGroup == 1U || senderGroup == 1U)));
            }
            chunkOffset += chunkBytes;
            continue;
        }
        const uint64_t taskArgs[TASK_ARG_NUM] = {
            sendBase,
            recvBase,
            sendToken,
            recvToken,
            blockBytes,
            chunkOffset,
            chunkBytes,
            param.root,
            rootSliceInPlace,
            metadataReuse,
        };
        const bool useMain = isRoot || activeGroup == 0U;
        const bool useSub = resCtx.ccuKernelCount > 1 && (isRoot || activeGroup == 1U);
        if (specializedSmallRootCall) {
            if (isRoot) {
                const uint64_t rootArgs[RANK12_SMALL_ROOT_TASK_ARG_NUM] = {
                    sendBase,
                    recvBase,
                    sendToken,
                    recvToken,
                    blockBytes,
                    rootSliceInPlace,
                };
                CHK_RET(LaunchGroups(param, resCtx, resCtx.ccuKernels,
                    rootArgs, RANK12_SMALL_ROOT_TASK_ARG_NUM, useMain, useSub));
            } else {
                const uint64_t receiverArgs[SMALL_RECEIVER_TASK_ARG_NUM] = {
                    recvBase,
                    recvToken,
                    param.root,
                    metadataReuse,
                };
                CHK_RET(LaunchGroups(param, resCtx, resCtx.smallReceiverKernels,
                    receiverArgs, SMALL_RECEIVER_TASK_ARG_NUM, useMain, useSub));
            }
            chunkOffset += chunkBytes;
            continue;
        }
        if (rank4SmallCall && !isRoot) {
            const uint64_t receiverArgs[SMALL_RECEIVER_TASK_ARG_NUM] = {
                recvBase,
                recvToken,
                param.root,
                metadataReuse,
            };
            CHK_RET(LaunchGroups(param, resCtx, resCtx.smallReceiverKernels,
                receiverArgs, SMALL_RECEIVER_TASK_ARG_NUM, useMain, useSub));
            chunkOffset += chunkBytes;
            continue;
        }
        CHK_RET(LaunchGroups(param, resCtx, resCtx.ccuKernels, taskArgs,
            TASK_ARG_NUM, useMain, useSub));
        chunkOffset += chunkBytes;
    }

    DescriptorCacheState nextState{};
    if (cacheStreamOrdered && smallCall && rank16PullCall && isRoot) {
        const DescriptorCacheEntry next{sendBase, sendToken};
        if (resCtx.pullSendCache.addr != next.addr ||
            resCtx.pullSendCache.token != next.token) {
            CHK_RET(UpdateDescriptorCache(comm, param,
                offsetof(AlgResourceCtx, pullSendCache), next));
        }
        nextState = DescriptorCacheState{DESCRIPTOR_CACHE_PULL_ROOT, param.root,
            descriptorPath};
    } else if (cacheStreamOrdered && smallCall && !rank16PullCall && !isRoot) {
        const DescriptorCacheEntry next{recvBase, recvToken};
        const DescriptorCacheEntry &current = resCtx.pushRecvCache[param.root];
        if (current.addr != next.addr || current.token != next.token) {
            const uint64_t offset = offsetof(AlgResourceCtx, pushRecvCache) +
                static_cast<uint64_t>(param.root) * sizeof(next);
            CHK_RET(UpdateDescriptorCache(comm, param, offset, next));
        }
        nextState = DescriptorCacheState{DESCRIPTOR_CACHE_PUSH_RECEIVER, param.root,
            descriptorPath};
    }
    if (resCtx.lastDescriptorCache.kind != nextState.kind ||
        resCtx.lastDescriptorCache.root != nextState.root ||
        resCtx.lastDescriptorCache.path != nextState.path) {
        CHK_RET(UpdateDescriptorCacheState(comm, param, nextState));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
