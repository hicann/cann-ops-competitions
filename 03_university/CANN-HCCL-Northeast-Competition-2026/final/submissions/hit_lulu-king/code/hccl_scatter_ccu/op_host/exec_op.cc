#include <algorithm>
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hcomm_primitives.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {
constexpr uint32_t WORKER_START_NOTIFY = 0;

HcclResult LaunchRelayStage(const AlgResourceCtx &resource, uint32_t thread, uint32_t kind,
    const std::vector<uint64_t> &args)
{
    const uint32_t registeredKind = (kind >= RELAY_INIT_INPUT && kind <= RELAY_FINISH_INPUT) ||
        kind == RELAY_INPUT_ONESHOT ? RELAY_INIT_INPUT :
        ((kind >= RELAY_INIT_OUTPUT && kind <= RELAY_FINISH_OUTPUT) || kind == RELAY_OUTPUT_ONESHOT
            ? RELAY_INIT_OUTPUT : kind);
    std::vector<uint64_t> launchArgs = args;
    launchArgs.push_back(kind);
    for (const auto &kernel : resource.kernels) {
        if (kernel.threadIndex == thread && kernel.kind == registeredKind) {
            CHK_RET_CCU(HcommCcuKernelLaunch(resource.threads[thread], kernel.handle, launchArgs.data(), launchArgs.size()));
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("Missing relay kernel thread=%u kind=%u", thread, kind);
    return HCCL_E_INTERNAL;
}

HcclResult LaunchRelay(const AlgResourceCtx &resource, std::vector<uint64_t> args)
{
    const ThreadHandle main = resource.threads[0];
    // Metadata-only channels execute independently of the ingress/egress pipeline.
    for (uint32_t t = resource.activeThreads; t < resource.threads.size(); ++t) {
        CHK_RET(HcommThreadNotifyRecordOnThread(main, resource.threads[t], 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(resource.threads[t], 0, 0));
        CHK_RET(LaunchRelayStage(resource, t, RELAY_RECEIVER, args));
        CHK_RET(HcommThreadNotifyRecordOnThread(resource.threads[t], main, 24 + t));
    }
    auto joinControls = [&]() -> HcclResult {
        for (uint32_t t = resource.activeThreads; t < resource.threads.size(); ++t) {
            CHK_RET(HcommThreadNotifyWaitOnThread(main, 24 + t, 0));
        }
        return HCCL_SUCCESS;
    };
    if (resource.relayRole == 2 && resource.activeThreads == 1 &&
        resource.kernels[0].kind == RELAY_RESIDENT_HELPER) {
        CHK_RET(LaunchRelayStage(resource, 0, RELAY_RESIDENT_HELPER, args));
        return joinControls();
    }
    if (resource.relayRole != 2) {
        const uint32_t kind = resource.relayRole == 1 ? RELAY_ROOT : RELAY_RECEIVER;
        for (uint32_t t = 1; t < resource.activeThreads; ++t) {
            CHK_RET(HcommThreadNotifyRecordOnThread(main, resource.threads[t], 0));
            CHK_RET(HcommThreadNotifyWaitOnThread(resource.threads[t], 0, 0));
            CHK_RET(LaunchRelayStage(resource, t, kind, args));
            CHK_RET(HcommThreadNotifyRecordOnThread(resource.threads[t], main, t));
        }
        CHK_RET(LaunchRelayStage(resource, 0, kind, args));
        for (uint32_t t = 1; t < resource.activeThreads; ++t) {
            CHK_RET(HcommThreadNotifyWaitOnThread(main, t, 0));
        }
        return HCCL_SUCCESS;
    }

    if (resource.relayBytes == resource.tileBytes) {
        // Publish output metadata independently before ingress starts waiting.
        // Four launches replace INIT_INPUT/INIT_OUTPUT/WAIT/FORWARD/FINISH_OUTPUT/FINISH_INPUT.
        for (uint32_t t = 1; t < resource.activeThreads; ++t) {
            CHK_RET(HcommThreadNotifyRecordOnThread(main, resource.threads[t], 2));
            CHK_RET(HcommThreadNotifyWaitOnThread(resource.threads[t], 2, 0));
            CHK_RET(LaunchRelayStage(resource, t, RELAY_INIT_OUTPUT, args));
        }
        CHK_RET(LaunchRelayStage(resource, 0, RELAY_INPUT_ONESHOT, args));
        args[6] = 0;
        args[7] = resource.relayBytes;
        args[8] = 0;
        for (uint32_t t = 1; t < resource.activeThreads; ++t) {
            CHK_RET(HcommThreadNotifyRecordOnThread(main, resource.threads[t], 0));
            CHK_RET(HcommThreadNotifyWaitOnThread(resource.threads[t], 0, 0));
            CHK_RET(LaunchRelayStage(resource, t, RELAY_OUTPUT_ONESHOT, args));
            CHK_RET(HcommThreadNotifyRecordOnThread(resource.threads[t], main, 2 * t));
        }
        for (uint32_t t = 1; t < resource.activeThreads; ++t) {
            CHK_RET(HcommThreadNotifyWaitOnThread(main, 2 * t, 0));
        }
        CHK_RET(LaunchRelayStage(resource, 0, RELAY_FINISH_INPUT, args));
        return joinControls();
    }

    // The input thread owns the root channel. Output threads own their Die's
    // destination channels. Two banks permit ingress(r+1) / egress(r) overlap.
    CHK_RET(LaunchRelayStage(resource, 0, RELAY_INIT_INPUT, args));
    for (uint32_t t = 1; t < resource.activeThreads; ++t) {
        CHK_RET(HcommThreadNotifyRecordOnThread(main, resource.threads[t], 2));
        CHK_RET(HcommThreadNotifyWaitOnThread(resource.threads[t], 2, 0));
        CHK_RET(LaunchRelayStage(resource, t, RELAY_INIT_OUTPUT, args));
    }
    const uint64_t rounds = (resource.relayBytes + resource.tileBytes - 1) / resource.tileBytes;
    for (uint64_t r = 0; r < rounds; ++r) {
        const uint32_t bank = r % 2;
        if (r >= 2) {
            for (uint32_t t = 1; t < resource.activeThreads; ++t) {
                CHK_RET(HcommThreadNotifyWaitOnThread(main, 2 * t + bank, 0));
            }
        }
        const uint32_t kind = r < 2 ? (bank == 0 ? RELAY_WAIT_0 : RELAY_WAIT_1)
                                   : (bank == 0 ? RELAY_REUSE_0 : RELAY_REUSE_1);
        CHK_RET(LaunchRelayStage(resource, 0, kind, args));
        args[6] = r * resource.tileBytes;
        args[7] = std::min(resource.tileBytes, resource.relayBytes - args[6]);
        args[8] = bank * resource.targetCount * resource.tileBytes;
        for (uint32_t t = 1; t < resource.activeThreads; ++t) {
            CHK_RET(HcommThreadNotifyRecordOnThread(main, resource.threads[t], bank));
            CHK_RET(HcommThreadNotifyWaitOnThread(resource.threads[t], bank, 0));
            CHK_RET(LaunchRelayStage(resource, t, RELAY_FORWARD, args));
            if (r + 1 == rounds) {
                CHK_RET(LaunchRelayStage(resource, t, RELAY_FINISH_OUTPUT, args));
            }
            CHK_RET(HcommThreadNotifyRecordOnThread(resource.threads[t], main, 2 * t + bank));
        }
    }
    for (uint32_t bank = 0; bank < std::min<uint64_t>(rounds, 2); ++bank) {
        for (uint32_t t = 1; t < resource.activeThreads; ++t) {
            CHK_RET(HcommThreadNotifyWaitOnThread(main, 2 * t + bank, 0));
        }
    }
    CHK_RET(LaunchRelayStage(resource, 0, RELAY_FINISH_INPUT, args));
    return joinControls();
}

HcclResult LaunchKernels(const OpParam &param, const AlgResourceCtx &resource,
    const std::vector<uint64_t> &taskArgs, uint64_t recvBytes)
{
    CHK_PRT_RET(resource.kernels.empty() || resource.kernels.size() != resource.threads.size(),
        HCCL_ERROR("Kernel/thread resource mismatch"), HCCL_E_INTERNAL);

    if (resource.kernels.size() == 1) {
        CHK_RET_CCU(HcommCcuKernelLaunch(resource.threads[0], resource.kernels[0].handle,
            taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
        return HCCL_SUCCESS;
    }

    const ThreadHandle mainThread = resource.threads[0];
    for (uint32_t index = 1; index < resource.threads.size(); ++index) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, resource.threads[index], WORKER_START_NOTIFY));
        CHK_RET(HcommThreadNotifyWaitOnThread(resource.threads[index], WORKER_START_NOTIFY, 0));
        CHK_RET_CCU(HcommCcuKernelLaunch(resource.threads[index], resource.kernels[index].handle,
            taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
        CHK_RET(HcommThreadNotifyRecordOnThread(resource.threads[index], mainThread, index));
    }

    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resource.kernels[0].handle,
        taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
    for (uint32_t index = 1; index < resource.threads.size(); ++index) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, index, 0));
    }
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0,
        HCCL_ERROR("Invalid resource context"), HCCL_E_INTERNAL);

    char *context = static_cast<char *>(param.resCtx);
    std::vector<char> sequence(context, context + param.ctxSize);
    AlgResourceCtx resource;
    resource.DeSerialize(sequence);

    const auto typeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeIter == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type"), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeIter->second,
        HCCL_ERROR("Receive byte size overflow"), HCCL_E_PARA);
    const uint64_t recvBytes = param.count * typeIter->second;

    if (resource.relayRole != 0) {
        std::vector<uint64_t> args(9, 0);
        args[2] = reinterpret_cast<uint64_t>(param.outputPtr);
        CHK_RET_CCU(HcommCcuGetMemToken(args[2], recvBytes, &args[3]));
        if (param.myRank == param.root) {
            CHK_PRT_RET(recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
                HCCL_ERROR("Send byte size overflow"), HCCL_E_PARA);
            args[0] = reinterpret_cast<uint64_t>(param.inputPtr);
            CHK_RET_CCU(HcommCcuGetMemToken(args[0], recvBytes * param.rankSize, &args[1]));
        }
        if (resource.relayRole == 2) {
            args[4] = reinterpret_cast<uint64_t>(resource.scratch.addr);
            const uint64_t banks = std::min<uint64_t>(2,
                (resource.relayBytes + resource.tileBytes - 1) / resource.tileBytes);
            CHK_RET_CCU(HcommCcuGetMemToken(args[4], banks * resource.targetCount * resource.tileBytes, &args[5]));
        }
        return LaunchRelay(resource, args);
    }

    uint64_t recvToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, &recvToken));
    std::vector<uint64_t> taskArgs;
    if (param.myRank == param.root) {
        CHK_PRT_RET(recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
            HCCL_ERROR("Send byte size overflow"), HCCL_E_PARA);
        const uint64_t sendBytes = recvBytes * param.rankSize;
        uint64_t sendToken = 0;
        CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), sendBytes, &sendToken));
        taskArgs = {reinterpret_cast<uint64_t>(param.inputPtr), sendToken};
        taskArgs.push_back(reinterpret_cast<uint64_t>(param.outputPtr));
        taskArgs.push_back(recvToken);
    } else {
        taskArgs = {reinterpret_cast<uint64_t>(param.outputPtr), recvToken};
    }
    return LaunchKernels(param, resource, taskArgs, recvBytes);
}
} // namespace ops_hccl
