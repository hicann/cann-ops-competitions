/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <algorithm>
#include <array>
#include <initializer_list>
#include <ccu/ccu_res.h>

#include "ccu_launch.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
    // At most 6 fixed arguments, MAX_RANK_SIZE gather offsets, 2 tile
    // flags and 3 fused-receiver arguments. Never allocate on the launch path.
    struct TaskArgs {
        static constexpr size_t CAPACITY = MAX_RANK_SIZE + 11U;
        std::array<uint64_t, CAPACITY> values{};
        uint32_t count = 0;
        TaskArgs() = default;
        TaskArgs(std::initializer_list<uint64_t> initial) { Append(initial); }
        TaskArgs &operator=(std::initializer_list<uint64_t> initial)
        {
            count = 0;
            Append(initial);
            return *this;
        }
        void push_back(uint64_t value) { values[count++] = value; }
        void Append(std::initializer_list<uint64_t> initial)
        {
            for (uint64_t value : initial) { push_back(value); }
        }
        uint64_t *data() { return values.data(); }
        uint32_t size() const { return count; }
        uint64_t &operator[](size_t i) { return values[i]; }
    };
    constexpr uint32_t SCATTER_TASK_ARG_NUM = 5;
    constexpr uint32_t THREAD_NOTIFY_INDEX = 0;
    constexpr uint32_t THREAD_NOTIFY_TIMEOUT_MS = 0;

    HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel, const uint64_t *taskArgs, uint32_t argCount)
    {
        CcuResult ret = HcommCcuKernelLaunch(thread, kernel, taskArgs, argCount);
        CHK_PRT_RET(ret != CCU_SUCCESS,
            HCCL_ERROR("[ScatterExecOp] kernel launch failed, ret %d", static_cast<int32_t>(ret)),
            ConvertCcuToHccl(ret));
        return HCCL_SUCCESS;
    }

    HcclResult RecordThreadNotify(ThreadHandle source, ThreadHandle destination)
    {
        int32_t ret = HcommThreadNotifyRecordOnThread(source, destination, THREAD_NOTIFY_INDEX);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterExecOp] thread notify record failed, ret %d", ret),
            static_cast<HcclResult>(ret));
        return HCCL_SUCCESS;
    }

    HcclResult WaitThreadNotify(ThreadHandle thread)
    {
        int32_t ret = HcommThreadNotifyWaitOnThread(thread, THREAD_NOTIFY_INDEX, THREAD_NOTIFY_TIMEOUT_MS);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterExecOp] thread notify wait failed, ret %d", ret),
            static_cast<HcclResult>(ret));
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOpLegacy16(const OpParam &param)
{
    FrozenResourceCtx resCtx;
    CHK_PRT_RET(!resCtx.Load(param.resCtx, param.ctxSize),
        HCCL_ERROR("[ScatterExecOp] invalid cached resource layout"), HCCL_E_INTERNAL);

    const uint64_t sliceBytes = param.count * sizeof(float);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1U) {
        int32_t ret = HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, sliceBytes);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterExecOp] local copy failed, ret %d", ret),
            static_cast<HcclResult>(ret));
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.ccuKernels.empty(), HCCL_ERROR("[ScatterExecOp] CCU kernels are missing"), HCCL_E_INTERNAL);

    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    if (param.myRank == param.root) {
        CcuResult ret = HcommCcuGetMemToken(inputBase, sliceBytes * param.rankSize, &inputToken);
        CHK_PRT_RET(ret != CCU_SUCCESS,
            HCCL_ERROR("[ScatterExecOp] input token failed, ret %d", static_cast<int32_t>(ret)), ConvertCcuToHccl(ret));
        // CCU memory tokens are process scoped; the same token authorizes the
        // root's receive buffer without another query on the critical path.
        outputToken = inputToken;
    } else {
        CcuResult ret = HcommCcuGetMemToken(outputBase, sliceBytes, &outputToken);
        CHK_PRT_RET(ret != CCU_SUCCESS,
            HCCL_ERROR("[ScatterExecOp] output token failed, ret %d", static_cast<int32_t>(ret)),
            ConvertCcuToHccl(ret));
    }

    const auto &relay = resCtx.relay;
    if (resCtx.directPush != 0U) {
        const bool dual = resCtx.ccuKernels.size() == IO_DIE_NUM;
        if (dual) {
            CHK_RET(RecordThreadNotify(param.cpuThread, resCtx.workerThread));
            CHK_RET(WaitThreadNotify(resCtx.workerThread));
        }
        // Keep the restored topology's exact schedule. Other direct routes
        // balance in elements to avoid a separate four-byte final mission.
        const uint64_t chunks = (sliceBytes + MAX_DATA_SIZE - 1U) / MAX_DATA_SIZE;
        const uint64_t directTile = param.rankSize == 16U ? MAX_DATA_SIZE
            : ((param.count + chunks - 1U) / chunks) * sizeof(float);
        for (uint64_t offset = 0; offset < sliceBytes; offset += directTile) {
            const uint64_t bytes = std::min<uint64_t>(directTile, sliceBytes - offset);
            for (size_t order = 0; order < resCtx.ccuKernels.size(); ++order) {
                const size_t k = dual ? (order == 0U ? 1U - resCtx.rootMainIndex : resCtx.rootMainIndex) : order;
                TaskArgs args;
                if ((resCtx.directPullMask & (1U << k)) != 0U) {
                    if (param.myRank == param.root) {
                        args = {inputBase + offset, inputToken};
                        const uint32_t writes = resCtx.directWriteMasks[k];
                        if (writes != 0U || k == 0U) {
                            args.push_back(bytes);
                        }
                        if (writes != 0U) {
                            for (size_t i = 0; i < resCtx.directPeers[k].size(); ++i) {
                                if ((writes & (1U << i)) != 0U) {
                                    args.push_back(bytes);
                                    args.push_back(inputBase + sliceBytes * resCtx.directPeers[k][i] + offset);
                                    break;
                                }
                            }
                        }
                        if (k == 0U) {
                            args.push_back(outputBase + offset);
                            args.push_back(inputBase + sliceBytes * param.root + offset);
                        }
                    } else {
                        args = {outputBase + offset, outputToken, bytes, sliceBytes * param.myRank};
                    }
                } else if (param.myRank == param.root) {
                    args = {inputToken, bytes};
                    if (k == 0U) {
                        args.push_back(outputBase + offset);
                        args.push_back(inputBase + sliceBytes * param.root + offset);
                    }
                    for (uint32_t peer : resCtx.directPeers[k]) {
                        args.push_back(inputBase + sliceBytes * peer + offset);
                    }
                } else {
                    args = {outputBase + offset, outputToken};
                }
                const auto thread = dual && k != resCtx.rootMainIndex ? resCtx.workerThread : param.cpuThread;
                CHK_RET(LaunchKernel(thread, resCtx.ccuKernels[k], args.data(), args.size()));
            }
        }
        if (dual) {
            CHK_RET(RecordThreadNotify(resCtx.workerThread, param.cpuThread));
            CHK_RET(WaitThreadNotify(param.cpuThread));
        }
        return HCCL_SUCCESS;
    }
    if (relay.coarse != 0U && param.myRank == param.root) {
        const uint64_t tail = sliceBytes * (relay.remotes.size() - 4U)
            / (relay.remotes.size() * (relay.relays.size() + 4U)) / sizeof(float) * sizeof(float);
        const uint64_t direct = sliceBytes - relay.relays.size() * tail;
        const bool dual = resCtx.ccuKernels.size() == IO_DIE_NUM;
        const size_t main = resCtx.rootMainIndex;
        CHK_PRT_RET(main >= resCtx.ccuKernels.size(), HCCL_ERROR("invalid root main index"), HCCL_E_INTERNAL);
        const auto launch = [&](size_t index, ThreadHandle thread) -> HcclResult {
            uint64_t firstSource = inputBase;
            bool hasPush = false;
            for (uint32_t peer : resCtx.directPeers[index]) {
                if (std::find(relay.remotes.begin(), relay.remotes.end(), peer) != relay.remotes.end()) {
                    firstSource += sliceBytes * peer;
                    hasPush = true;
                    break;
                }
            }
            TaskArgs args = {inputBase, inputToken};
            if (param.rankSize == 16U || hasPush || index == 0U) {
                args.push_back(sliceBytes);
            }
            if (hasPush) {
                args.push_back(direct);
                args.push_back(firstSource);
            }
            if (index == 0U) {
                args.push_back(outputBase);
                args.push_back(inputBase + sliceBytes * param.root);
            }
            return LaunchKernel(thread, resCtx.ccuKernels[index], args.data(), args.size());
        };
        if (dual) {
            CHK_RET(RecordThreadNotify(param.cpuThread, resCtx.workerThread));
            CHK_RET(WaitThreadNotify(resCtx.workerThread));
            CHK_RET(launch(1U - main, resCtx.workerThread));
            CHK_RET(RecordThreadNotify(resCtx.workerThread, param.cpuThread));
        }
        CHK_RET(launch(main, param.cpuThread));
        if (dual) {
            CHK_RET(WaitThreadNotify(param.cpuThread));
        }
        return HCCL_SUCCESS;
    }
    if (relay.persistent != 0U && param.myRank == param.root) {
        // Publish the full input only once. Both root dies can now stay active
        // while receivers independently drain their own tile queues.
        uint64_t rootArgs[] = {inputBase, outputBase, inputToken, sliceBytes,
            inputBase + sliceBytes * param.root};
        uint64_t publishArgs[] = {inputBase, inputToken};
        if (resCtx.ccuKernels.size() == 1U) {
            return LaunchKernel(param.cpuThread, resCtx.ccuKernels[0], rootArgs, SCATTER_TASK_ARG_NUM);
        }
        CHK_PRT_RET(resCtx.ccuKernels.size() != IO_DIE_NUM,
            HCCL_ERROR("[ScatterExecOp] invalid root kernel count"), HCCL_E_INTERNAL);
        CHK_RET(RecordThreadNotify(param.cpuThread, resCtx.workerThread));
        CHK_RET(WaitThreadNotify(resCtx.workerThread));
        CHK_RET(LaunchKernel(resCtx.workerThread, resCtx.ccuKernels[0], rootArgs, SCATTER_TASK_ARG_NUM));
        CHK_RET(RecordThreadNotify(resCtx.workerThread, param.cpuThread));
        CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[1], publishArgs, 2U));
        CHK_RET(WaitThreadNotify(param.cpuThread));
        return HCCL_SUCCESS;
    }
    if (relay.coarse != 0U && relay.role == 1U) {
        const uint64_t relayCount = relay.relays.size();
        const uint64_t remoteCount = relay.remotes.size();
        const uint64_t tail = sliceBytes * (remoteCount - 4U) / (remoteCount * (relayCount + 4U))
            / sizeof(float) * sizeof(float);
        const uint64_t direct = sliceBytes - relayCount * tail;
        TaskArgs args = {0U, outputBase, outputToken, relay.scratch, sliceBytes, tail,
            sliceBytes * param.myRank, sliceBytes * relay.remotes.front() + direct + relay.index * tail};
        // Gather only the outbound payload first. Its forwarding overlaps the
        // much larger own-slice read on the receiving queue.
        CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[0], args.data(), args.size()));
        const uint64_t publish[] = {relay.scratch, outputToken};
        for (size_t k = 0; k < relay.forwardKernels.size(); ++k) {
            const auto thread = relay.forwardThreads[k];
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, thread, 0U)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, 0U, 0)));
            CHK_RET(LaunchKernel(thread, relay.forwardKernels[k], publish, 2U));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread, param.cpuThread, 2U * k)));
        }
        args[0] = 1U;
        CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[0], args.data(), args.size()));
        for (size_t k = 0; k < relay.forwardKernels.size(); ++k) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, 2U * k, 0)));
        }
        return HCCL_SUCCESS;
    }
    // Balance all tiles in elements so 400 MiB + 4 B cannot produce a
    // four-byte final transfer. Direct-only paths keep their large messages.
    const uint64_t tileLimit = relay.coarse != 0U ? sliceBytes :
        (relay.enabled != 0U ? RELAY_TILE_BYTES : MAX_DATA_SIZE);
    const uint64_t tileCount = (sliceBytes + tileLimit - 1U) / tileLimit;
    const uint64_t tileElements = (param.count + tileCount - 1U) / tileCount;
    const uint64_t tileBytes = tileElements * sizeof(float);
    if (relay.residentReceiver != 0U) {
        const uint64_t relayCount = relay.relays.size();
        const uint64_t remoteCount = relay.remotes.size();
        const auto tailSize = [&](uint64_t bytes) {
            return bytes * (remoteCount - 4U) / (remoteCount * (relayCount + 4U)) / sizeof(float) * sizeof(float);
        };
        const uint64_t tail = tailSize(tileBytes);
        const uint64_t lastBytes = sliceBytes - (tileCount - 1U) * tileBytes;
        const uint64_t lastTail = tailSize(lastBytes);
        const uint64_t args[] = {outputBase, outputToken, sliceBytes * param.myRank,
            tileBytes, tileBytes - relayCount * tail, tail, uint64_t{0} - tileCount,
            lastBytes - relayCount * lastTail, lastTail};
        return LaunchKernel(param.cpuThread, resCtx.ccuKernels[0], args, 9U);
    }
    // v16: small pulls gain nothing from dual-die parallelism; serialising
    // both die launches on the main stream drops the fork-join notify pairs.
    const bool smallPull = relay.enabled == 0U && resCtx.directPush == 0U && sliceBytes < 1024U * 1024U;
    const bool rootDualDie = resCtx.ccuKernels.size() == IO_DIE_NUM && !smallPull;
    if (rootDualDie) {
        CHK_RET(RecordThreadNotify(param.cpuThread, resCtx.workerThread));
        CHK_RET(WaitThreadNotify(resCtx.workerThread));
    }
    uint64_t processedBytes = 0;
    uint64_t round = 0;
    while (processedBytes < sliceBytes) {
        const uint32_t slot = round % 2U;
        if (relay.pipeline != 0U && round >= 2U) {
            for (size_t k = 0; k < relay.forwardThreads.size(); ++k) {
                if (relay.forwardOnMain[k] != 0U) {
                    continue;
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(param.cpuThread, 2U * k + slot, 0)));
            }
        }
        const uint64_t scratch = relay.scratch + slot * RELAY_BANK_BYTES;
        const uint64_t chunkBytes = std::min<uint64_t>(tileBytes, sliceBytes - processedBytes);
        const uint64_t relayCount = relay.relays.size();
        const uint64_t remoteCount = relay.remotes.size();
        // Balance the root Clos port (4 links) and each root-to-relay link.
        const uint64_t tailBytes
            = relay.enabled != 0U
                  ? (chunkBytes * (remoteCount - 4U) / (remoteCount * (relayCount + 4U)) / sizeof(float))
                        * sizeof(float)
                  : 0U;
        const uint64_t directBytes = chunkBytes - relayCount * tailBytes;
        const uint64_t inputChunk = inputBase + processedBytes;
        const uint64_t outputChunk = outputBase + processedBytes;
        uint64_t taskArgs[SCATTER_TASK_ARG_NUM] = {
            inputChunk,
            outputChunk,
            inputToken,
            chunkBytes,
            inputChunk + sliceBytes * param.myRank,
        };
        TaskArgs recvArgs = {outputChunk, outputToken,
            relay.enabled != 0U && relay.role == 2U ? directBytes : chunkBytes,
            sliceBytes * param.myRank + (relay.persistent != 0U ? processedBytes : 0U)};
        if (relay.enabled != 0U && relay.role == 1U) {
            recvArgs.push_back(scratch);
            recvArgs.push_back(tailBytes);
            for (uint32_t peer : relay.remotes) {
                recvArgs.push_back(sliceBytes * peer + directBytes + relay.index * tailBytes
                    + (relay.persistent != 0U ? processedBytes : 0U));
            }
        }
        if (relay.persistent != 0U) {
            recvArgs.push_back(round == 0U ? 1U : 0U);
            recvArgs.push_back(processedBytes + chunkBytes == sliceBytes ? 1U : 0U);
        }
        if (!relay.fusedIndices.empty()) {
            recvArgs.push_back(tailBytes);
            recvArgs.push_back(relay.index * tailBytes);
            recvArgs.push_back(outputChunk + directBytes);
        }
        if (relay.coarse != 0U && relay.role == 2U) {
            recvArgs = {outputChunk, outputToken};
            if (!relay.fusedIndices.empty()) {
                recvArgs.Append({tailBytes, relay.index * tailBytes, outputChunk + directBytes});
            }
        }
        // Remote receivers do not produce staging data. Start their other-die
        // reads before the direct read, rather than making them depend on it.
        if (relay.enabled != 0U && relay.role == 2U) {
            for (size_t k = 0; k < relay.forwardKernels.size(); ++k) {
                if (relay.forwardOnMain[k] != 0U) {
                    continue;
                }
                TaskArgs args = {outputToken, tailBytes, relay.index * tailBytes};
                for (uint32_t index : relay.forwardIndices[k]) {
                    args.push_back(outputChunk + directBytes + index * tailBytes);
                }
                const auto thread = relay.forwardThreads[k];
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, thread, slot)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, slot, 0)));
                CHK_RET(LaunchKernel(thread, relay.forwardKernels[k], args.data(), args.size()));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(thread, param.cpuThread, 2U * k + slot)));
            }
        }
        uint64_t publishArgs[] = {inputChunk, inputToken};
        if (resCtx.ccuKernels.size() == 1U) {
            CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[0],
                param.myRank == param.root ? taskArgs : recvArgs.data(),
                param.myRank == param.root ? SCATTER_TASK_ARG_NUM : static_cast<uint32_t>(recvArgs.size())));
        } else {
            CHK_PRT_RET(resCtx.ccuKernels.size() != IO_DIE_NUM,
                HCCL_ERROR("[ScatterExecOp] invalid kernel count %zu", resCtx.ccuKernels.size()), HCCL_E_INTERNAL);
            // Independent die queues: one stream fork before all tiles and
            // one join afterwards, without a cross-die barrier on each tile.
            // v16: on a small pull both die launches go to the main stream.
            const ThreadHandle firstThread = smallPull ? param.cpuThread : resCtx.workerThread;
            CHK_RET(LaunchKernel(firstThread, resCtx.ccuKernels[0], taskArgs, SCATTER_TASK_ARG_NUM));
            CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[1], publishArgs, 2U));
        }
        // The relay may reuse scratch only after all consumers acknowledge it.
        // Root can publish the following chunk while forwarding is in flight.
        if (relay.enabled != 0U && relay.role == 1U) {
            std::array<TaskArgs, IO_DIE_NUM> args{};
            for (size_t k = 0; k < relay.forwardKernels.size(); ++k) {
                args[k] = {scratch, outputToken};
            }
            for (size_t k = 0; k < relay.forwardThreads.size(); ++k) {
                if (relay.forwardOnMain[k] != 0U) {
                    continue;
                }
                const ThreadHandle forwardThread = relay.forwardThreads[k];
                // READY and CONSUMED use different bits for the two banks.
                // The main stream reclaims a bank only after EVERY consumer
                // die has completed, which also covers mixed-die topologies.
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(param.cpuThread, forwardThread, slot)));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(forwardThread, slot, 0)));
                CHK_RET(LaunchKernel(forwardThread, relay.forwardKernels[k], args[k].data(), args[k].size()));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(forwardThread, param.cpuThread, 2U * k + slot)));
            }
            // Launch the other die first, then serialize work that shares
            // this rank's receiving die on its existing main queue.
            for (size_t k = 0; k < relay.forwardKernels.size(); ++k) {
                if (relay.forwardOnMain[k] != 0U) {
                    CHK_RET(LaunchKernel(param.cpuThread, relay.forwardKernels[k], args[k].data(), args[k].size()));
                }
            }
        }
        processedBytes += chunkBytes;
        ++round;
    }
    if (resCtx.relay.pipeline != 0U) {
        for (uint64_t i = round > 2U ? round - 2U : 0U; i < round; ++i) {
            for (size_t k = 0; k < resCtx.relay.forwardThreads.size(); ++k) {
                if (relay.forwardOnMain[k] != 0U) {
                    continue;
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(param.cpuThread, 2U * k + i % 2U, 0)));
            }
        }
    }
    if (rootDualDie) {
        CHK_RET(RecordThreadNotify(resCtx.workerThread, param.cpuThread));
        CHK_RET(WaitThreadNotify(param.cpuThread));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param)
{
    FrozenResourceCtx resCtx;
    CHK_PRT_RET(!resCtx.Load(param.resCtx, param.ctxSize),
        HCCL_ERROR("[ScatterExecOp] invalid cached resource layout"), HCCL_E_INTERNAL);

    const uint64_t sliceBytes = param.count * sizeof(float);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1U) {
        int32_t ret = HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, sliceBytes);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterExecOp] local copy failed, ret %d", ret),
            static_cast<HcclResult>(ret));
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.ccuKernels.empty(), HCCL_ERROR("[ScatterExecOp] CCU kernels are missing"), HCCL_E_INTERNAL);

    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    if (param.myRank == param.root) {
        CcuResult ret = HcommCcuGetMemToken(inputBase, sliceBytes * param.rankSize, &inputToken);
        CHK_PRT_RET(ret != CCU_SUCCESS,
            HCCL_ERROR("[ScatterExecOp] input token failed, ret %d", static_cast<int32_t>(ret)), ConvertCcuToHccl(ret));
        // CCU memory tokens are process scoped; the same token authorizes the
        // root's receive buffer without another query on the critical path.
        outputToken = inputToken;
    } else {
        CcuResult ret = HcommCcuGetMemToken(outputBase, sliceBytes, &outputToken);
        CHK_PRT_RET(ret != CCU_SUCCESS,
            HCCL_ERROR("[ScatterExecOp] output token failed, ret %d", static_cast<int32_t>(ret)),
            ConvertCcuToHccl(ret));
    }

    const auto &relay = resCtx.relay;
    if (resCtx.directPush != 0U) {
        const bool dual = resCtx.ccuKernels.size() == IO_DIE_NUM;
        if (dual) {
            CHK_RET(RecordThreadNotify(param.cpuThread, resCtx.workerThread));
            CHK_RET(WaitThreadNotify(resCtx.workerThread));
        }
        // Keep the restored topology's exact schedule. Other direct routes
        // balance in elements to avoid a separate four-byte final mission.
        const uint64_t chunks = (sliceBytes + MAX_DATA_SIZE - 1U) / MAX_DATA_SIZE;
        const uint64_t directTile = param.rankSize == 16U ? MAX_DATA_SIZE
            : ((param.count + chunks - 1U) / chunks) * sizeof(float);
        for (uint64_t offset = 0; offset < sliceBytes; offset += directTile) {
            const uint64_t bytes = std::min<uint64_t>(directTile, sliceBytes - offset);
            for (size_t order = 0; order < resCtx.ccuKernels.size(); ++order) {
                const size_t k = dual ? (order == 0U ? 1U - resCtx.rootMainIndex : resCtx.rootMainIndex) : order;
                TaskArgs args;
                if ((resCtx.directPullMask & (1U << k)) != 0U) {
                    if (param.myRank == param.root) {
                        args = {inputBase + offset, inputToken};
                        const uint32_t writes = resCtx.directWriteMasks[k];
                        if (writes != 0U || k == ((resCtx.directPullMask >> ROOT_COPY_INDEX_SHIFT) & 1U)) {
                            args.push_back(bytes);
                        }
                        if (writes != 0U) {
                            for (size_t i = 0; i < resCtx.directPeers[k].size(); ++i) {
                                if ((writes & (1U << i)) != 0U) {
                                    args.push_back(bytes);
                                    args.push_back(inputBase + sliceBytes * resCtx.directPeers[k][i] + offset);
                                    break;
                                }
                            }
                        }
                        if (k == ((resCtx.directPullMask >> ROOT_COPY_INDEX_SHIFT) & 1U)) {
                            args.push_back(outputBase + offset);
                            args.push_back(inputBase + sliceBytes * param.root + offset);
                        }
                    } else {
                        args = {outputBase + offset, outputToken, bytes, sliceBytes * param.myRank};
                    }
                } else if (param.myRank == param.root) {
                    args = {inputToken, bytes};
                    if (k == ((resCtx.directPullMask >> ROOT_COPY_INDEX_SHIFT) & 1U)) {
                        args.push_back(outputBase + offset);
                        args.push_back(inputBase + sliceBytes * param.root + offset);
                    }
                    for (uint32_t peer : resCtx.directPeers[k]) {
                        args.push_back(inputBase + sliceBytes * peer + offset);
                    }
                } else {
                    args = {outputBase + offset, outputToken};
                }
                const auto thread = dual && k != resCtx.rootMainIndex ? resCtx.workerThread : param.cpuThread;
                CHK_RET(LaunchKernel(thread, resCtx.ccuKernels[k], args.data(), args.size()));
            }
        }
        if (dual) {
            CHK_RET(RecordThreadNotify(resCtx.workerThread, param.cpuThread));
            CHK_RET(WaitThreadNotify(param.cpuThread));
        }
        return HCCL_SUCCESS;
    }
    if (relay.coarse != 0U && param.myRank == param.root) {
        const uint64_t tail = sliceBytes * (relay.remotes.size() - 4U)
            / (relay.remotes.size() * (relay.relays.size() + 4U)) / sizeof(float) * sizeof(float);
        const uint64_t direct = sliceBytes - relay.relays.size() * tail;
        const bool dual = resCtx.ccuKernels.size() == IO_DIE_NUM;
        const size_t main = resCtx.rootMainIndex;
        CHK_PRT_RET(main >= resCtx.ccuKernels.size(), HCCL_ERROR("invalid root main index"), HCCL_E_INTERNAL);
        const auto launch = [&](size_t index, ThreadHandle thread) -> HcclResult {
            uint64_t firstSource = inputBase;
            bool hasPush = false;
            for (uint32_t peer : resCtx.directPeers[index]) {
                if (std::find(relay.remotes.begin(), relay.remotes.end(), peer) != relay.remotes.end()) {
                    firstSource += sliceBytes * peer;
                    hasPush = true;
                    break;
                }
            }
            TaskArgs args = {inputBase, inputToken};
            if (param.rankSize == 16U || hasPush || index == ((resCtx.directPullMask >> ROOT_COPY_INDEX_SHIFT) & 1U)) {
                args.push_back(sliceBytes);
            }
            if (hasPush) {
                args.push_back(direct);
                args.push_back(firstSource);
            }
            if (index == ((resCtx.directPullMask >> ROOT_COPY_INDEX_SHIFT) & 1U)) {
                args.push_back(outputBase);
                args.push_back(inputBase + sliceBytes * param.root);
            }
            return LaunchKernel(thread, resCtx.ccuKernels[index], args.data(), args.size());
        };
        if (dual) {
            CHK_RET(RecordThreadNotify(param.cpuThread, resCtx.workerThread));
            CHK_RET(WaitThreadNotify(resCtx.workerThread));
            CHK_RET(launch(1U - main, resCtx.workerThread));
            CHK_RET(RecordThreadNotify(resCtx.workerThread, param.cpuThread));
        }
        CHK_RET(launch(main, param.cpuThread));
        if (dual) {
            CHK_RET(WaitThreadNotify(param.cpuThread));
        }
        return HCCL_SUCCESS;
    }
    if (relay.persistent != 0U && param.myRank == param.root) {
        // Publish the full input only once. Both root dies can now stay active
        // while receivers independently drain their own tile queues.
        uint64_t rootArgs[] = {inputBase, outputBase, inputToken, sliceBytes,
            inputBase + sliceBytes * param.root};
        uint64_t publishArgs[] = {inputBase, inputToken};
        if (resCtx.ccuKernels.size() == 1U) {
            return LaunchKernel(param.cpuThread, resCtx.ccuKernels[0], rootArgs, SCATTER_TASK_ARG_NUM);
        }
        CHK_PRT_RET(resCtx.ccuKernels.size() != IO_DIE_NUM,
            HCCL_ERROR("[ScatterExecOp] invalid root kernel count"), HCCL_E_INTERNAL);
        CHK_RET(RecordThreadNotify(param.cpuThread, resCtx.workerThread));
        CHK_RET(WaitThreadNotify(resCtx.workerThread));
        CHK_RET(LaunchKernel(resCtx.workerThread, resCtx.ccuKernels[0], rootArgs, SCATTER_TASK_ARG_NUM));
        CHK_RET(RecordThreadNotify(resCtx.workerThread, param.cpuThread));
        CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[1], publishArgs, 2U));
        CHK_RET(WaitThreadNotify(param.cpuThread));
        return HCCL_SUCCESS;
    }
    if (relay.coarse != 0U && relay.role == 1U) {
        const uint64_t relayCount = relay.relays.size();
        const uint64_t remoteCount = relay.remotes.size();
        const uint64_t tail = sliceBytes * (remoteCount - 4U) / (remoteCount * (relayCount + 4U))
            / sizeof(float) * sizeof(float);
        const uint64_t direct = sliceBytes - relayCount * tail;
        TaskArgs args = {0U, outputBase, outputToken, relay.scratch, sliceBytes, tail,
            sliceBytes * param.myRank, sliceBytes * relay.remotes.front() + direct + relay.index * tail};
        // Gather only the outbound payload first. Its forwarding overlaps the
        // much larger own-slice read on the receiving queue.
        CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[0], args.data(), args.size()));
        const uint64_t publish[] = {relay.scratch, outputToken};
        for (size_t k = 0; k < relay.forwardKernels.size(); ++k) {
            const auto thread = relay.forwardThreads[k];
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, thread, 0U)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, 0U, 0)));
            CHK_RET(LaunchKernel(thread, relay.forwardKernels[k], publish, 2U));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread, param.cpuThread, 2U * k)));
        }
        args[0] = 1U;
        CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[0], args.data(), args.size()));
        for (size_t k = 0; k < relay.forwardKernels.size(); ++k) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, 2U * k, 0)));
        }
        return HCCL_SUCCESS;
    }
    // Balance all tiles in elements so 400 MiB + 4 B cannot produce a
    // four-byte final transfer. Direct-only paths keep their large messages.
    const uint64_t tileLimit = relay.coarse != 0U ? sliceBytes :
        (relay.enabled != 0U ? RELAY_TILE_BYTES : MAX_DATA_SIZE);
    const uint64_t tileCount = (sliceBytes + tileLimit - 1U) / tileLimit;
    const uint64_t tileElements = (param.count + tileCount - 1U) / tileCount;
    const uint64_t tileBytes = tileElements * sizeof(float);
    if (relay.residentReceiver != 0U) {
        const uint64_t relayCount = relay.relays.size();
        const uint64_t remoteCount = relay.remotes.size();
        const auto tailSize = [&](uint64_t bytes) {
            return bytes * (remoteCount - 4U) / (remoteCount * (relayCount + 4U)) / sizeof(float) * sizeof(float);
        };
        const uint64_t tail = tailSize(tileBytes);
        const uint64_t lastBytes = sliceBytes - (tileCount - 1U) * tileBytes;
        const uint64_t lastTail = tailSize(lastBytes);
        const uint64_t args[] = {outputBase, outputToken, sliceBytes * param.myRank,
            tileBytes, tileBytes - relayCount * tail, tail, uint64_t{0} - tileCount,
            lastBytes - relayCount * lastTail, lastTail};
        return LaunchKernel(param.cpuThread, resCtx.ccuKernels[0], args, 9U);
    }
    // v16: small pulls gain nothing from dual-die parallelism; serialising
    // both die launches on the main stream drops the fork-join notify pairs.
    const bool smallPull = relay.enabled == 0U && resCtx.directPush == 0U && sliceBytes < 1024U * 1024U;
    const bool rootDualDie = resCtx.ccuKernels.size() == IO_DIE_NUM && !smallPull;
    if (rootDualDie) {
        CHK_RET(RecordThreadNotify(param.cpuThread, resCtx.workerThread));
        CHK_RET(WaitThreadNotify(resCtx.workerThread));
    }
    uint64_t processedBytes = 0;
    uint64_t round = 0;
    while (processedBytes < sliceBytes) {
        const uint32_t slot = round % 2U;
        if (relay.pipeline != 0U && round >= 2U) {
            for (size_t k = 0; k < relay.forwardThreads.size(); ++k) {
                if (relay.forwardOnMain[k] != 0U) {
                    continue;
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(param.cpuThread, 2U * k + slot, 0)));
            }
        }
        const uint64_t scratch = relay.scratch + slot * RELAY_BANK_BYTES;
        const uint64_t chunkBytes = std::min<uint64_t>(tileBytes, sliceBytes - processedBytes);
        const uint64_t relayCount = relay.relays.size();
        const uint64_t remoteCount = relay.remotes.size();
        // Balance the root Clos port (4 links) and each root-to-relay link.
        const uint64_t tailBytes
            = relay.enabled != 0U
                  ? (chunkBytes * (remoteCount - 4U) / (remoteCount * (relayCount + 4U)) / sizeof(float))
                        * sizeof(float)
                  : 0U;
        const uint64_t directBytes = chunkBytes - relayCount * tailBytes;
        const uint64_t inputChunk = inputBase + processedBytes;
        const uint64_t outputChunk = outputBase + processedBytes;
        uint64_t taskArgs[SCATTER_TASK_ARG_NUM] = {
            inputChunk,
            outputChunk,
            inputToken,
            chunkBytes,
            inputChunk + sliceBytes * param.myRank,
        };
        TaskArgs recvArgs = {outputChunk, outputToken,
            relay.enabled != 0U && relay.role == 2U ? directBytes : chunkBytes,
            sliceBytes * param.myRank + (relay.persistent != 0U ? processedBytes : 0U)};
        if (relay.enabled != 0U && relay.role == 1U) {
            recvArgs.push_back(scratch);
            recvArgs.push_back(tailBytes);
            for (uint32_t peer : relay.remotes) {
                recvArgs.push_back(sliceBytes * peer + directBytes + relay.index * tailBytes
                    + (relay.persistent != 0U ? processedBytes : 0U));
            }
        }
        if (relay.persistent != 0U) {
            recvArgs.push_back(round == 0U ? 1U : 0U);
            recvArgs.push_back(processedBytes + chunkBytes == sliceBytes ? 1U : 0U);
        }
        if (!relay.fusedIndices.empty()) {
            recvArgs.push_back(tailBytes);
            recvArgs.push_back(relay.index * tailBytes);
            recvArgs.push_back(outputChunk + directBytes);
        }
        if (relay.coarse != 0U && relay.role == 2U) {
            recvArgs = {outputChunk, outputToken};
            if (!relay.fusedIndices.empty()) {
                recvArgs.Append({tailBytes, relay.index * tailBytes, outputChunk + directBytes});
            }
        }
        // Remote receivers do not produce staging data. Start their other-die
        // reads before the direct read, rather than making them depend on it.
        if (relay.enabled != 0U && relay.role == 2U) {
            for (size_t k = 0; k < relay.forwardKernels.size(); ++k) {
                if (relay.forwardOnMain[k] != 0U) {
                    continue;
                }
                TaskArgs args = {outputToken, tailBytes, relay.index * tailBytes};
                for (uint32_t index : relay.forwardIndices[k]) {
                    args.push_back(outputChunk + directBytes + index * tailBytes);
                }
                const auto thread = relay.forwardThreads[k];
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, thread, slot)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, slot, 0)));
                CHK_RET(LaunchKernel(thread, relay.forwardKernels[k], args.data(), args.size()));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(thread, param.cpuThread, 2U * k + slot)));
            }
        }
        uint64_t publishArgs[] = {inputChunk, inputToken};
        if (resCtx.ccuKernels.size() == 1U) {
            CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[0],
                param.myRank == param.root ? taskArgs : recvArgs.data(),
                param.myRank == param.root ? SCATTER_TASK_ARG_NUM : static_cast<uint32_t>(recvArgs.size())));
        } else {
            CHK_PRT_RET(resCtx.ccuKernels.size() != IO_DIE_NUM,
                HCCL_ERROR("[ScatterExecOp] invalid kernel count %zu", resCtx.ccuKernels.size()), HCCL_E_INTERNAL);
            // Independent die queues: one stream fork before all tiles and
            // one join afterwards, without a cross-die barrier on each tile.
            // v16: on a small pull both die launches go to the main stream.
            const ThreadHandle firstThread = smallPull ? param.cpuThread : resCtx.workerThread;
            CHK_RET(LaunchKernel(firstThread, resCtx.ccuKernels[0], taskArgs, SCATTER_TASK_ARG_NUM));
            CHK_RET(LaunchKernel(param.cpuThread, resCtx.ccuKernels[1], publishArgs, 2U));
        }
        // The relay may reuse scratch only after all consumers acknowledge it.
        // Root can publish the following chunk while forwarding is in flight.
        if (relay.enabled != 0U && relay.role == 1U) {
            std::array<TaskArgs, IO_DIE_NUM> args{};
            for (size_t k = 0; k < relay.forwardKernels.size(); ++k) {
                args[k] = {scratch, outputToken};
            }
            for (size_t k = 0; k < relay.forwardThreads.size(); ++k) {
                if (relay.forwardOnMain[k] != 0U) {
                    continue;
                }
                const ThreadHandle forwardThread = relay.forwardThreads[k];
                // READY and CONSUMED use different bits for the two banks.
                // The main stream reclaims a bank only after EVERY consumer
                // die has completed, which also covers mixed-die topologies.
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(param.cpuThread, forwardThread, slot)));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(forwardThread, slot, 0)));
                CHK_RET(LaunchKernel(forwardThread, relay.forwardKernels[k], args[k].data(), args[k].size()));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(forwardThread, param.cpuThread, 2U * k + slot)));
            }
            // Launch the other die first, then serialize work that shares
            // this rank's receiving die on its existing main queue.
            for (size_t k = 0; k < relay.forwardKernels.size(); ++k) {
                if (relay.forwardOnMain[k] != 0U) {
                    CHK_RET(LaunchKernel(param.cpuThread, relay.forwardKernels[k], args[k].data(), args[k].size()));
                }
            }
        }
        processedBytes += chunkBytes;
        ++round;
    }
    if (resCtx.relay.pipeline != 0U) {
        for (uint64_t i = round > 2U ? round - 2U : 0U; i < round; ++i) {
            for (size_t k = 0; k < resCtx.relay.forwardThreads.size(); ++k) {
                if (relay.forwardOnMain[k] != 0U) {
                    continue;
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(param.cpuThread, 2U * k + i % 2U, 0)));
            }
        }
    }
    if (rootDualDie) {
        CHK_RET(RecordThreadNotify(resCtx.workerThread, param.cpuThread));
        CHK_RET(WaitThreadNotify(param.cpuThread));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
