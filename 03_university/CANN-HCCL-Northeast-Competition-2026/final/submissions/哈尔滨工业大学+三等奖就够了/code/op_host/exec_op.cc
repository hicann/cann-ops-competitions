/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <algorithm>
#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>
#include "log.h"
#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize != sizeof(AlgResourceCtx),
        HCCL_ERROR("Missing Scatter resource context"), HCCL_E_INTERNAL);
    AlgResourceCtx resource{};
    std::memcpy(&resource, param.resCtx, sizeof(resource));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > SCATTER_MAX_RANKS || param.root >= param.rankSize
            || param.myRank >= param.rankSize || param.count > UINT64_MAX / sizeof(float) / param.rankSize,
        HCCL_ERROR("Invalid Scatter launch parameters"), HCCL_E_PARA);
    CHK_PRT_RET(resource.failed || resource.kernelCount[param.root] == 0 || resource.kernelCount[param.root] > SCATTER_MAX_RANKS
            || resource.rankId != param.myRank || resource.rankSize != param.rankSize,
        HCCL_ERROR("Scatter resource context does not match communicator"), HCCL_E_INTERNAL);
    const uint64_t bytes = param.count * sizeof(float);
    const bool directPath = bytes < SCATTER_OVERLAP_BYTES || bytes > SCATTER_TRANSFER_BYTES
        || param.rankSize <= 5;
    const uint32_t kernelCount = resource.kernelCount[param.root];
    const CcuKernelHandle *kernels = resource.kernels[param.root];
    const uint32_t rootGroup = resource.rootChannelGroup[param.root];
    CHK_PRT_RET(rootGroup >= kernelCount, HCCL_ERROR("Invalid root channel group"), HCCL_E_INTERNAL);
    const uint32_t localMask = resource.localMask;
    CHK_PRT_RET(!ScatterIsLocal(localMask, param.myRank),
        HCCL_ERROR("Missing local rank membership"), HCCL_E_INTERNAL);
    const bool helper = param.myRank != param.root && ScatterIsLocal(localMask, param.root);
    uint64_t fragment = 0, directBytes = 0, remoteCount = 0;
    // Remote ranks know only their own group, never the root's L/q/D.
    if (param.myRank == param.root || helper) {
        const uint32_t localCount = ScatterLocalCount(localMask, param.rankSize);
        remoteCount = param.rankSize - localCount;
        if (localCount > 1 && remoteCount > 4 && bytes >= SCATTER_OVERLAP_BYTES
            && bytes <= SCATTER_TRANSFER_BYTES) {
            fragment = std::min(bytes * (remoteCount - 4) / (remoteCount * (localCount + 3)),
                SCATTER_STAGING_BYTES / remoteCount) & ~uint64_t(3);
        }
        directBytes = bytes - (localCount - 1) * fragment;
    }
    uint64_t input = 0;
    uint64_t inputToken = 0;
    const uint64_t output = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t outputToken = 0;
    uint64_t scratch = 0, scratchToken = 0;
    if (helper && fragment != 0) {
        const uint64_t scratchBytes = remoteCount * fragment;
        CHK_PRT_RET(resource.stagingBase == 0 || resource.stagingSize < scratchBytes
            || resource.stagingBase > UINT64_MAX - resource.stagingSize || output > UINT64_MAX - bytes,
            HCCL_ERROR("Invalid helper staging capacity or address range"), HCCL_E_PARA);
        const uint64_t first = resource.stagingBase;
        const uint64_t last = first + resource.stagingSize - scratchBytes;
        auto safe = [&](uint64_t address) { return address + scratchBytes <= output || address >= output + bytes; };
        CHK_PRT_RET(!safe(first) && !safe(last), HCCL_ERROR("Helper staging overlaps output at both buffer ends"), HCCL_E_PARA);
        scratch = safe(first) ? first : last;
        CHK_RET_CCU(HcommCcuGetMemToken(scratch, scratchBytes, &scratchToken));
    }
    ScatterCopyMode copyMode = ScatterCopyMode::NONE;
    if (param.myRank == param.root) {
        input = reinterpret_cast<uint64_t>(param.inputPtr);
        CHK_RET_CCU(HcommCcuGetMemToken(input, bytes * param.rankSize, &inputToken));
        if (output != input + bytes * param.root) {
            // ValidateBuffers has already checked both complete address ranges.
            const bool disjoint = output + bytes <= input || output >= input + bytes * param.rankSize;
            copyMode = disjoint ? ScatterCopyMode::WITH_READS : ScatterCopyMode::AFTER_READS;
        }
    }
    CHK_RET_CCU(HcommCcuGetMemToken(output, bytes, &outputToken));
    // Each registered role keeps one fixed contiguous layout for every launch and die.
    uint64_t rootArgs[] = {input, output, inputToken, outputToken, 0, bytes * param.myRank, 0,
        bytes / SCATTER_TRANSFER_BYTES, bytes % SCATTER_TRANSFER_BYTES, 0, 0, directBytes};
    uint64_t helperArgs[] = {output, outputToken, 0, bytes * param.myRank,
        bytes / SCATTER_TRANSFER_BYTES, bytes % SCATTER_TRANSFER_BYTES,
        scratch, scratchToken, fragment, directBytes, 0};
    uint64_t remoteArgs[] = {output, outputToken, 0, bytes * param.myRank,
        bytes / SCATTER_TRANSFER_BYTES, bytes % SCATTER_TRANSFER_BYTES, 0};
    auto launch = [&](uint32_t group, ScatterPhase phase, bool isLast) -> CcuResult {
        if (param.myRank == param.root) {
            // Slot 4 uses zero only for direct phases; relay q=0 still has a nonzero phase.
            rootArgs[4] = directPath ? 0 : static_cast<uint64_t>(phase);
            rootArgs[6] = static_cast<uint64_t>(ScatterCopyMode::NONE);
            rootArgs[9] = static_cast<uint64_t>(phase == ScatterPhase::SMALL_FULL
                || phase == ScatterPhase::SMALL_PUBLISH);
            rootArgs[10] = static_cast<uint64_t>(phase == ScatterPhase::SMALL_FULL
                || phase == ScatterPhase::SMALL_FINISH || phase == ScatterPhase::ROOT_WAIT);
            const bool copyWithReads = copyMode == ScatterCopyMode::WITH_READS
                && ((directPath && phase == ScatterPhase::SMALL_FULL)
                    || (!directPath && phase == ScatterPhase::ROOT_WAIT && group == 0));
            const bool copyAfterReads = copyMode == ScatterCopyMode::AFTER_READS && isLast
                && ((!directPath && phase == ScatterPhase::ROOT_WAIT) || directPath);
            rootArgs[6] = static_cast<uint64_t>(copyWithReads ? ScatterCopyMode::WITH_READS
                : copyAfterReads ? ScatterCopyMode::AFTER_READS : ScatterCopyMode::NONE);
            if (param.rankSize <= 5) {
                // N<=5 is direct at every size; registration omits relay-only fields.
                uint64_t directArgs[] = {rootArgs[0], rootArgs[1], rootArgs[2], rootArgs[3],
                    rootArgs[5], rootArgs[6], rootArgs[7], rootArgs[8], rootArgs[9], rootArgs[10]};
                return HcommCcuKernelLaunch(param.cpuThread, kernels[group], directArgs,
                    sizeof(directArgs) / sizeof(directArgs[0]));
            }
            return HcommCcuKernelLaunch(param.cpuThread, kernels[group], rootArgs,
                sizeof(rootArgs) / sizeof(rootArgs[0]));
        }
        if (param.rankSize <= 5) {
            uint64_t directArgs[] = {output, outputToken, bytes * param.myRank,
                bytes / SCATTER_TRANSFER_BYTES, bytes % SCATTER_TRANSFER_BYTES};
            return HcommCcuKernelLaunch(param.cpuThread, kernels[group], directArgs,
                sizeof(directArgs) / sizeof(directArgs[0]));
        }
        if (helper) {
            helperArgs[2] = static_cast<uint64_t>(phase);
            helperArgs[10] = static_cast<uint64_t>(phase == ScatterPhase::SMALL_FULL
                || (helper && phase == ScatterPhase::READ_OWN));
            return HcommCcuKernelLaunch(param.cpuThread, kernels[group], helperArgs,
                sizeof(helperArgs) / sizeof(helperArgs[0]));
        }
        remoteArgs[2] = static_cast<uint64_t>(phase);
        remoteArgs[6] = static_cast<uint64_t>(phase == ScatterPhase::SMALL_FULL);
        return HcommCcuKernelLaunch(param.cpuThread, kernels[group], remoteArgs,
            sizeof(remoteArgs) / sizeof(remoteArgs[0]));
    };
    if (directPath) {
        if (param.myRank != param.root) {
            CHK_RET_CCU(launch(rootGroup, ScatterPhase::SMALL_FULL, true));
        } else if (kernelCount > 1) {
            auto groupAt = [&](uint32_t position) {
                return bytes < SCATTER_OVERLAP_BYTES ? resource.smallGroupOrder[position] : position;
            };
            for (uint32_t group = 0; group + 1 < kernelCount; ++group) {
                CHK_RET_CCU(launch(groupAt(group), ScatterPhase::SMALL_PUBLISH, false));
            }
            CHK_RET_CCU(launch(groupAt(kernelCount - 1), ScatterPhase::SMALL_FULL, false));
            for (uint32_t group = 0; group + 1 < kernelCount; ++group) {
                CHK_RET_CCU(launch(groupAt(group), ScatterPhase::SMALL_FINISH, group + 2 == kernelCount));
            }
        } else {
            CHK_RET_CCU(launch(0, ScatterPhase::SMALL_FULL, true));
        }
        return HCCL_SUCCESS;
    }
    if (helper) {
        CHK_RET_CCU(launch(rootGroup, ScatterPhase::STAGE, false));
    }
    for (uint32_t group = 0; group < kernelCount; ++group) {
        CHK_RET_CCU(launch(group, ScatterPhase::PUBLISH, false));
    }
    if (param.myRank == param.root) {
        for (uint32_t group = 0; group < kernelCount; ++group) {
            CHK_RET_CCU(launch(group, ScatterPhase::ROOT_WAIT, group + 1 == kernelCount));
        }
        for (uint32_t group = 0; group < kernelCount; ++group) {
            CHK_RET_CCU(launch(group, ScatterPhase::ROOT_RELEASE, false));
        }
    } else {
        if (helper) {
            CHK_RET_CCU(launch(rootGroup, ScatterPhase::READ_OWN, false));
            for (uint32_t group = 0; group < kernelCount; ++group) {
                CHK_RET_CCU(launch(group, ScatterPhase::DRAIN, false));
            }
        } else {
            CHK_RET_CCU(launch(rootGroup, ScatterPhase::READ_PROVIDERS, false));
            for (uint32_t group = 0; group < kernelCount; ++group) {
                if (group != rootGroup) {
                    CHK_RET_CCU(launch(group, ScatterPhase::READ_PROVIDERS, false));
                }
            }
        }
        CHK_RET_CCU(launch(rootGroup, ScatterPhase::FINISH, false));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
