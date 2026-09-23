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
#include <hcomm_primitives.h>
#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>
#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
    HcclResult LaunchRelay(const OpParam &param, ThreadHandle thread, CcuKernelHandle kernel, uint64_t *args)
    {
        if (param.count <= SCATTER_CCU_CHUNK / sizeof(float)) {
            // A single block has no offset; its length is also the rank stride.
            if (args[6] != 0 || args[7] != args[11] || args[9] != args[10]) {
                return HCCL_E_INTERNAL;
            }
            const bool isRoot = param.myRank == param.root;
            uint64_t compact[SCATTER_COMPACT_TASK_ARGS] = {isRoot ? args[0] : args[2], args[1],
                isRoot ? args[3] : args[5], args[4], args[7], args[8], args[9], isRoot ? args[13] : args[12]};
            return ConvertCcuToHccl(HcommCcuKernelLaunch(thread, kernel, compact, SCATTER_COMPACT_TASK_ARGS));
        }
        return ConvertCcuToHccl(HcommCcuKernelLaunch(thread, kernel, args, SCATTER_RELAY_TASK_ARGS));
    }

    HcclResult LaunchOverlapChunk(const OpParam &param, const AlgResourceCtx &resources, uint64_t *args)
    {
        // This path is selected only for the validated contiguous 2x8 or 4x3 topology.
        if ((param.rankSize != 12 && param.rankSize != 16) || !resources.relayEnabled) {
            return HCCL_E_PARA;
        }
        uint32_t role = SCATTER_ROLE_DIRECT, partner = INVALID_VALUE_RANKID;
        const uint32_t groupSize = param.rankSize == 16 ? 8 : 3;
        const uint32_t localBegin = param.root / groupSize * groupSize;
        std::vector<uint32_t> helpers, targets;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank >= localBegin && rank < localBegin + groupSize) {
                if (rank != param.root) {
                    helpers.push_back(rank);
                }
            } else if (targets.size() < groupSize - 1) {
                targets.push_back(rank);
            }
        }
        if (helpers.size() != groupSize - 1 || targets.size() != helpers.size()) {
            return HCCL_E_INTERNAL;
        }
        if (param.myRank == param.root) {
            role = SCATTER_ROLE_ROOT;
        }
        for (uint32_t i = 0; i < helpers.size(); ++i) {
            if (param.myRank == helpers[i]) {
                role = SCATTER_ROLE_HELPER;
                partner = targets[i];
            } else if (param.myRank == targets[i]) {
                role = SCATTER_ROLE_TARGET;
                partner = helpers[i];
            }
        }
        uint32_t rootGroup = INVALID_VALUE_RANKID, partnerGroup = INVALID_VALUE_RANKID;
        for (uint32_t i = 0; i < resources.groupMasks.size(); ++i) {
            if ((resources.groupMasks[i] & (1ULL << param.root)) != 0) {
                rootGroup = i;
            }
            if (partner != INVALID_VALUE_RANKID && (resources.groupMasks[i] & (1ULL << partner)) != 0) {
                partnerGroup = i;
            }
        }
        if (role != SCATTER_ROLE_ROOT && rootGroup == INVALID_VALUE_RANKID) {
            return HCCL_E_INTERNAL;
        }
        if ((role == SCATTER_ROLE_HELPER || role == SCATTER_ROLE_TARGET) && partnerGroup == INVALID_VALUE_RANKID) {
            return HCCL_E_INTERNAL;
        }
        const auto threadFor = [&](uint32_t group) {
            return group == 0 ? param.cpuThread : resources.workers[group - 1];
        };
        const auto launch = [&](uint32_t group, uint64_t mode) -> HcclResult {
            args[12] = mode;
            return LaunchRelay(param, threadFor(group), resources.ccuKernels[group], args);
        };
        // A non-root rank whose entire work belongs to one group needs no worker
        // handoff. The user thread orders both completion and scratch reuse.
        // Kernel handles still select their original channel die, as in ExecOp.
        if (role == SCATTER_ROLE_DIRECT
            || ((role == SCATTER_ROLE_HELPER || role == SCATTER_ROLE_TARGET) && rootGroup == partnerGroup)) {
            const auto launchMain = [&](uint64_t mode) -> HcclResult {
                args[12] = mode;
                return LaunchRelay(param, param.cpuThread, resources.ccuKernels[rootGroup], args);
            };
            if (role == SCATTER_ROLE_HELPER) {
                // A single-block helper with both channels on this die receives
                // and forwards in one mission. Its kernel retains all barriers.
                if (args[8] != 0 && param.count > SCATTER_CCU_CHUNK / sizeof(float)) {
                    CHK_RET(launchMain(2));
                }
                CHK_RET(launchMain(3));
            } else {
                CHK_RET(launchMain(role == SCATTER_ROLE_TARGET ? 4 : 0));
            }
            args[12] = 0;
            return HCCL_SUCCESS;
        }
        // Only workers with a real mission participate in startup and completion.
        std::vector<bool> activeGroups(resources.groupMasks.size(), role == SCATTER_ROLE_ROOT);
        if (role == SCATTER_ROLE_HELPER) {
            activeGroups[rootGroup] = true;
            activeGroups[partnerGroup] = args[8] != 0;
        } else if (role == SCATTER_ROLE_TARGET) {
            activeGroups[rootGroup] = args[9] != 0;
            activeGroups[partnerGroup] = args[8] != 0;
        }
        // Separate startup, tail-ready and completion slots prevent notification reuse races.
        for (uint32_t i = 0; i < resources.workers.size(); ++i) {
            if (!activeGroups[i + 1]) {
                continue;
            }
            const ThreadHandle worker = resources.workers[i];
            CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, worker, 0));
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        }
        if (role == SCATTER_ROLE_ROOT) {
            for (uint32_t group = 0; group < resources.groupMasks.size(); ++group) {
                CHK_RET(launch(group, 0));
            }
        } else if (role == SCATTER_ROLE_HELPER) {
            if (args[8] != 0) {
                CHK_RET(launch(rootGroup, 2));
                if (partnerGroup != rootGroup) {
                    const uint32_t readySlot = partnerGroup == 0 ? MAX_RANK_SIZE - 1 : 1;
                    CHK_RET(HcommThreadNotifyRecordOnThread(threadFor(rootGroup), threadFor(partnerGroup), readySlot));
                    CHK_RET(HcommThreadNotifyWaitOnThread(threadFor(partnerGroup), readySlot, CUSTOM_TIMEOUT));
                    CHK_RET(launch(partnerGroup, 1));
                }
            }
            CHK_RET(launch(rootGroup, 3));
        } else if (role == SCATTER_ROLE_TARGET) {
            if (rootGroup == partnerGroup) {
                CHK_RET(launch(rootGroup, 4));
            } else {
                if (args[9] != 0) {
                    CHK_RET(launch(rootGroup, 0));
                }
                if (args[8] != 0) {
                    CHK_RET(launch(partnerGroup, 1));
                }
            }
        } else {
            CHK_RET(launch(rootGroup, 0));
        }
        // A helper cannot reuse its scratch until its forwarding mission has finished.
        for (uint32_t i = 0; i < resources.workers.size(); ++i) {
            if (!activeGroups[i + 1]) {
                continue;
            }
            CHK_RET(HcommThreadNotifyRecordOnThread(resources.workers[i], param.cpuThread, i));
            CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, i, CUSTOM_TIMEOUT));
        }
        args[12] = 0;
        return HCCL_SUCCESS;
    }

    HcclResult CopyDeferred(const OpParam &param, const AlgResourceCtx &resources, uint64_t ownInput, uint64_t output,
        uint64_t inputToken, uint64_t outputToken, uint64_t bytes)
    {
        const uint64_t scratch = reinterpret_cast<uint64_t>(resources.localBuffer.addr);
        const uint64_t capacity = std::min(SCATTER_CCU_CHUNK, resources.localBuffer.size) / 4 * 4;
        if (scratch == 0 || capacity == 0) {
            return HCCL_E_PARA;
        }
        uint64_t scratchToken = 0;
        CHK_RET(ConvertCcuToHccl(HcommCcuGetMemToken(scratch, capacity, &scratchToken)));
        const bool backwards = output > ownInput && output < ownInput + bytes;
        for (uint64_t done = 0; done < bytes;) {
            const uint64_t length = std::min(capacity, bytes - done);
            const uint64_t offset = backwards ? bytes - done - length : done;
            uint64_t args[SCATTER_RELAY_TASK_ARGS]
                = {ownInput + offset, output + offset, scratch, inputToken, outputToken, scratchToken, length};
            args[12] = ~uint64_t{0};
            if (resources.relayEnabled && param.count <= SCATTER_CCU_CHUNK / sizeof(float)) {
                uint64_t compact[SCATTER_COMPACT_TASK_ARGS]
                    = {args[0], args[1], args[2], args[3], args[4], args[5], args[6], ~uint64_t{0}};
                CHK_RET(ConvertCcuToHccl(HcommCcuKernelLaunch(
                    param.cpuThread, resources.ccuKernels.back(), compact, SCATTER_COMPACT_TASK_ARGS)));
            } else {
                const uint32_t argCount = resources.relayEnabled ? SCATTER_RELAY_TASK_ARGS : 7;
                CHK_RET(ConvertCcuToHccl(
                    HcommCcuKernelLaunch(param.cpuThread, resources.ccuKernels.back(), args, argCount)));
            }
            done += length;
        }
        return HCCL_SUCCESS;
    }

    HcclResult ExecRelay(const OpParam &param, const AlgResourceCtx &resources)
    {
        const uint64_t bytes = param.count * sizeof(float);
        const uint64_t input = reinterpret_cast<uint64_t>(param.inputPtr);
        const uint64_t output = reinterpret_cast<uint64_t>(param.outputPtr);
        const uint64_t scratch = reinterpret_cast<uint64_t>(resources.localBuffer.addr);
        if (scratch == 0 || resources.localBuffer.size < SCATTER_CCU_CHUNK) {
            return HCCL_E_PARA;
        }
        uint64_t inputToken = 0, outputToken = 0, scratchToken = 0;
        CHK_RET(ConvertCcuToHccl(HcommCcuGetMemToken(output, bytes, &outputToken)));
        CHK_RET(ConvertCcuToHccl(HcommCcuGetMemToken(scratch, SCATTER_CCU_CHUNK, &scratchToken)));
        bool copyOwn = false, deferred = false;
        uint64_t ownInput = 0;
        if (param.myRank == param.root) {
            CHK_RET(ConvertCcuToHccl(HcommCcuGetMemToken(input, bytes * param.rankSize, &inputToken)));
            ownInput = input + bytes * param.root;
            const bool overlap = output < input + bytes * param.rankSize && input < output + bytes;
            deferred = overlap && output != ownInput;
            copyOwn = !overlap;
        }
        uint64_t relayBytes = 0;
        if (param.rankSize == 16) {
            relayBytes = bytes * 4 / 11;
        } else if (param.rankSize == 12) {
            relayBytes = bytes * 5 / 6;
        } else {
            return HCCL_E_INTERNAL;
        }
        relayBytes = relayBytes / sizeof(float) * sizeof(float);
        if (relayBytes == 0 || relayBytes >= bytes) {
            return HCCL_E_INTERNAL;
        }
        const uint64_t relayStart = bytes - relayBytes;
        for (uint64_t offset = 0; offset < bytes; offset += SCATTER_CCU_CHUNK) {
            const uint64_t normalLen = std::min(SCATTER_CCU_CHUNK, bytes - offset);
            const uint64_t relayLen = offset < relayBytes ? std::min(SCATTER_CCU_CHUNK, relayBytes - offset) : 0;
            const uint64_t prefixLen = offset < relayStart ? std::min(SCATTER_CCU_CHUNK, relayStart - offset) : 0;
            uint64_t args[SCATTER_RELAY_TASK_ARGS]
                = {param.myRank == param.root ? input : 0, output, scratch, inputToken, outputToken, scratchToken,
                    offset, normalLen, relayLen, prefixLen, relayStart, bytes, 0, copyOwn ? 1ULL : 0ULL};
            CHK_RET(LaunchOverlapChunk(param, resources, args));
        }
        if (deferred) {
            CHK_RET(CopyDeferred(param, resources, ownInput, output, inputToken, outputToken, bytes));
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    auto *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resources;
    resources.DeSerialize(seq);
    if (resources.groupMasks.empty() || resources.ccuKernels.size() != resources.groupMasks.size() + 1
        || resources.workers.size() + 1 != resources.groupMasks.size()) {
        return HCCL_E_INTERNAL;
    }
    if (resources.relayEnabled) {
        return ExecRelay(param, resources);
    }
    const uint64_t bytes = param.count * sizeof(float);
    const uint64_t input = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t output = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0, outputToken = 0;
    CHK_RET(ConvertCcuToHccl(HcommCcuGetMemToken(output, bytes, &outputToken)));
    bool copyOwn = false, deferred = false;
    uint64_t ownInput = 0;
    if (param.myRank == param.root) {
        CHK_RET(ConvertCcuToHccl(HcommCcuGetMemToken(input, bytes * param.rankSize, &inputToken)));
        ownInput = input + bytes * param.root;
        const bool overlap = output < input + bytes * param.rankSize && input < output + bytes;
        deferred = overlap && output != ownInput;
        copyOwn = !overlap;
    }
    const uint64_t directChunk = param.rankSize == 4 ? SCATTER_FOUR_RANK_DIRECT_CHUNK : SCATTER_CCU_CHUNK;
    const bool parallel = param.myRank == param.root && !resources.workers.empty();
    if (parallel) {
        // Wake every worker before the main stream can block on any group.
        for (ThreadHandle worker : resources.workers) {
            CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, worker, 0));
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        }
    }
    for (size_t group = 0; group < resources.groupMasks.size(); ++group) {
        if (param.myRank != param.root && (resources.groupMasks[group] & (1ULL << param.root)) == 0) {
            continue;
        }
        const ThreadHandle thread = parallel && group != 0 ? resources.workers[group - 1] : param.cpuThread;
        uint32_t firstAddressRank = 0;
        if (param.myRank == param.root && param.rankSize == 12 && bytes <= SCATTER_CCU_CHUNK) {
            uint64_t mask = resources.groupMasks[group];
            if (group == 0) {
                mask |= 1ULL << param.root;
            }
            if (mask == 0) {
                return HCCL_E_INTERNAL;
            }
            while ((mask & (1ULL << firstAddressRank)) == 0) {
                ++firstAddressRank;
            }
        }
        for (uint64_t offset = 0; offset < bytes;) {
            const uint64_t length = std::min(directChunk, bytes - offset);
            if (param.myRank == param.root) {
                uint64_t args[SCATTER_CCU_TASK_ARGS] = {input + offset + bytes * firstAddressRank, output + offset,
                    inputToken, outputToken, length, bytes, copyOwn && group == 0 ? 1ULL : 0ULL};
                CHK_RET(ConvertCcuToHccl(
                    HcommCcuKernelLaunch(thread, resources.ccuKernels[group], args, SCATTER_CCU_TASK_ARGS)));
            } else {
                uint64_t args[SCATTER_RECEIVE_TASK_ARGS] = {output + offset, outputToken};
                CHK_RET(ConvertCcuToHccl(
                    HcommCcuKernelLaunch(thread, resources.ccuKernels[group], args, SCATTER_RECEIVE_TASK_ARGS)));
            }
            offset += length;
        }
    }
    if (parallel) {
        // Each worker owns a distinct completion slot on this call's user stream.
        for (uint32_t i = 0; i < resources.workers.size(); ++i) {
            CHK_RET(HcommThreadNotifyRecordOnThread(resources.workers[i], param.cpuThread, i));
            CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, i, CUSTOM_TIMEOUT));
        }
    }
    if (deferred) {
        CHK_RET(CopyDeferred(param, resources, ownInput, output, inputToken, outputToken, bytes));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
