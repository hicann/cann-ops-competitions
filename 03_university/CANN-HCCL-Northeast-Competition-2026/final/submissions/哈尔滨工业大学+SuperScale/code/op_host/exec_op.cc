/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of this repository.
 */
#include <map>
#include <mutex>
#include <utility>
#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include "log.h"
#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
// Small calls alternate between two generations of variable slots and notify
// bits. Every rank counts the small calls of a given root on
// this communicator, and all ranks see the same call sequence, so they derive
// the same generation without exchanging anything.
std::mutex g_stageMutex;
std::map<std::pair<uint64_t, uint32_t>, uint64_t> g_stageCalls;
}
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resources)
{
    // The caller passes the already deserialized kernel context; the hot path
    // no longer copies and reparses the engine buffer on every invocation.
    const uint32_t dataKernels = resources.dataKernelCount;
    const uint32_t totalKernels = static_cast<uint32_t>(resources.ccuKernels.size());
    if (dataKernels == 0 || dataKernels > 3 || totalKernels != dataKernels) return HCCL_E_INTERNAL;
    const uint64_t bytes = param.count * 4;
    uint64_t input = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t output = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t scratch = reinterpret_cast<uint64_t>(resources.localBuffer.addr);
    uint64_t inputToken = 0, outputToken = 0, scratchToken = 0;
    if (param.myRank == param.root && HcommCcuGetMemToken(input, bytes * param.rankSize, &inputToken) != CCU_SUCCESS)
        return HCCL_E_INTERNAL;
    if (HcommCcuGetMemToken(output, bytes, &outputToken) != CCU_SUCCESS) return HCCL_E_INTERNAL;
    const bool staged = CcuStaged(bytes, param.rankSize, std::min<uint64_t>(resources.minCclSize, MAX_DATA_SIZE));
    if (bytes > CCU_SMALL_BYTES && scratch != 0 && resources.localBuffer.size != 0 &&
        HcommCcuGetMemToken(scratch, resources.localBuffer.size, &scratchToken) != CCU_SUCCESS) return HCCL_E_INTERNAL;
    uint64_t generation = 0;
    if (staged) {
        const uint64_t key = resources.threads.empty() ? 0 : static_cast<uint64_t>(resources.threads[0]);
        const std::lock_guard<std::mutex> guard(g_stageMutex);
        uint64_t &calls = g_stageCalls[{key, param.root}];
        generation = calls++ & 1U;
    }
    // A small call carries its generation in place of the scratch pair.
    uint64_t args[] = {input, output, inputToken, outputToken, staged ? generation : scratch, scratchToken};
    const uint32_t argCount = staged ? 5 : (bytes <= CCU_SMALL_BYTES ? 4 : 6);
    if (param.myRank != param.root) {
        // Helper phases run in order on one host thread: root-die buffers and
        // tail-ready, far-die scratch service, then root-die output completion.
        for (CcuKernelHandle kernel : resources.ccuKernels)
            if (HcommCcuKernelLaunch(param.cpuThread, kernel, args, argCount) != CCU_SUCCESS)
                return HCCL_E_INTERNAL;
        return HCCL_SUCCESS;
    }
    if (dataKernels > 2) return HCCL_E_INTERNAL;
    auto runPair = [&](uint32_t begin, uint32_t count) -> HcclResult {
        if (count == 0) return HCCL_SUCCESS;
        if (count == 2) {
            if (resources.threads.empty()) return HCCL_E_INTERNAL;
            ThreadHandle worker = resources.threads[0];
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, worker, 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT)));
            if (HcommCcuKernelLaunch(worker, resources.ccuKernels[begin + 1], args, argCount) != CCU_SUCCESS)
                return HCCL_E_INTERNAL;
        }
        if (HcommCcuKernelLaunch(param.cpuThread, resources.ccuKernels[begin], args, argCount) != CCU_SUCCESS)
            return HCCL_E_INTERNAL;
        if (count == 2) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resources.threads[0], param.cpuThread, 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, 0, CUSTOM_TIMEOUT)));
        }
        return HCCL_SUCCESS;
    };
    // One kernel per root die, forked and joined. Each die waits for its own
    // writes and for every reader's ACK before returning, so the join also
    // protects the input. Each reused thread notify is consumed before reuse.
    return runPair(0, dataKernels);
}
}
