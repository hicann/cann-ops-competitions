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

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "log.h"
#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
    // 多 IO Die 拓扑（12/16 rank）下，主线程与从线程在启动 Kernel 前后做本地同步，保证同一算子内
    // 两个 Die 上的 Kernel 归属同一次调用。
    HcclResult PreSyncThreads(const std::vector<ThreadHandle> &threads)
    {
        if (threads.size() <= 1) {
            return HCCL_SUCCESS;
        }

        for (uint32_t threadIdx = 1; threadIdx < threads.size(); threadIdx++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[threadIdx], 0)));
        }
        for (uint32_t threadIdx = 1; threadIdx < threads.size(); threadIdx++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[threadIdx], 0, CUSTOM_TIMEOUT)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult PostSyncThreads(const std::vector<ThreadHandle> &threads)
    {
        if (threads.size() <= 1) {
            return HCCL_SUCCESS;
        }

        for (uint32_t threadIdx = 1; threadIdx < threads.size(); threadIdx++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], threadIdx - 1, CUSTOM_TIMEOUT)));
        }
        for (uint32_t threadIdx = 1; threadIdx < threads.size(); threadIdx++) {
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(threads[threadIdx], threads[0], threadIdx - 1)));
        }
        return HCCL_SUCCESS;
    }

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    CHK_PRT_RET(
        resCtx.threads.empty() || resCtx.ccuKernels.empty(), HCCL_ERROR("CCU resource is empty"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.threads.size() != resCtx.ccuKernels.size(),
        HCCL_ERROR("Thread count %zu does not match kernel count %zu", resCtx.threads.size(), resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.peersPerKernel.size() != resCtx.ccuKernels.size(),
        HCCL_ERROR("Peer group count %zu does not match kernel count %zu", resCtx.peersPerKernel.size(),
            resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);

    // Engine Context 可跨调用复用，但主 Thread 必须绑定本次传入的 stream。
    resCtx.threads[0] = param.cpuThread;

    const auto typeSizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(
        typeSizeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type %d", param.dataType), HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = typeSizeIt->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("Scatter data size overflows uint64"), HCCL_E_PARA);
    const uint64_t chunkBytes = param.count * typeSize;
    if (chunkBytes == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    const uint64_t sendBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t recvBase = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t sendToken = 0;
    uint64_t recvToken = 0;
    // 只有 root 读取 sendBuf，非 root 的 sendBuf 不参与通信，无需计算 Token。
    if (param.myRank == param.root) {
        CHK_RET_CCU(HcommCcuGetMemToken(sendBase, chunkBytes * param.rankSize, &sendToken));
    }
    CHK_RET_CCU(HcommCcuGetMemToken(recvBase, chunkBytes, &recvToken));

    // root 自留 chunk 的源偏移（其余对端 chunk 偏移按 peerRank * chunkBytes 逐一计算）。
    const uint64_t rootOffset = static_cast<uint64_t>(param.root) * chunkBytes;
    // 本 rank 的 chunk 偏移：root 时为 root*chunkBytes（自留块），非 root 时为 myRank*chunkBytes（自己该读的块）。
    const uint64_t myRankOffset = static_cast<uint64_t>(param.myRank) * chunkBytes;

    // 预计算每个 Kernel 各对端 chunk 的源偏移（与分片无关，只随 chunkBytes 变化）。
    std::vector<std::vector<uint64_t>> peerOffsetsPerKernel(resCtx.ccuKernels.size());
    for (uint32_t kernelIdx = 0; kernelIdx < resCtx.ccuKernels.size(); kernelIdx++) {
        const std::vector<uint32_t> &peers = resCtx.peersPerKernel[kernelIdx];
        peerOffsetsPerKernel[kernelIdx].reserve(peers.size());
        for (const uint32_t peer : peers) {
            peerOffsetsPerKernel[kernelIdx].push_back(static_cast<uint64_t>(peer) * chunkBytes);
        }
    }

    // 拉模型（小消息，≤1MB）：root 发布 sendBuf、非 root 主动读自己的 chunk，单次下发，无分片。
    // 单线程化：所有 kernel 都在 threads[0] 串行 launch（先 mesh 组发布、后 clos 组发布），
    // 去掉 PreSync/PostSync 两对 thread-notify（GATE/DONE 握手），省掉小消息固定开销里的整轮线程同步。
    if (resCtx.usePull != 0) {
        for (uint32_t kernelIdx = 0; kernelIdx < resCtx.ccuKernels.size(); kernelIdx++) {
            // 任务参数布局：recvBuf 地址/Token、sendBuf 地址/Token、整块大小、本 rank chunk 偏移。
            std::vector<uint64_t> taskArgs;
            taskArgs.reserve(6);
            taskArgs.push_back(recvBase);
            taskArgs.push_back(recvToken);
            taskArgs.push_back(sendBase);
            taskArgs.push_back(sendToken);
            taskArgs.push_back(chunkBytes);
            taskArgs.push_back(myRankOffset);
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[kernelIdx], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size())));
        }
        return HCCL_SUCCESS;
    }

    // 中继（2×8 / 4×3，>1MB）：root 把跨机块前 relayBytes 交给本地 relay 转发，摊薄 Clos 上行。
    if (resCtx.useRelay != 0) {
        const uint64_t relayBytes = resCtx.relayBytes;
        const uint64_t relaySliceBytes = resCtx.relaySliceBytes; // 单次中转上限（= staging 容量）
        const uint64_t tailSize = chunkBytes - relayBytes; // 尾块大小（root 直发部分）

        uint32_t relayRole = resCtx.relayRole;
        uint32_t relayPeerRank = resCtx.relayPeerRank;

        // relay 的 Clos(forward) kernel 单独 defer：等 mesh(gather) 完成后用线程 notify 串起来再 launch，
        // 规避 device 跨 die 同步（环回不通）。gather 先 launch，forward 在 gather 完成后 launch、与 own 读并行。
        // 组序按 (layer,die) 排序：Clos(die0) 排在 mesh(die1) 之前，故 forwardKernelIdx 在遍历 mesh 前已确定。
        uint32_t forwardKernelIdx = UINT32_MAX;
        uint32_t gatherKernelIdx = UINT32_MAX;
        uint64_t stagingToken = 0;
        if (relayRole == 1) {
            CHK_RET_CCU(HcommCcuGetMemToken(resCtx.relayStagingAddr, relaySliceBytes, &stagingToken));
        }

        CHK_RET(PreSyncThreads(resCtx.threads));
        for (uint32_t kernelIdx = 0; kernelIdx < resCtx.ccuKernels.size(); kernelIdx++) {
            const std::vector<uint64_t> &peerOffsets = peerOffsetsPerKernel[kernelIdx];
            std::vector<uint64_t> taskArgs;
            if (param.myRank == param.root) {
                // root：recvBuf/Token、sendBuf/Token、chunkBytes、root 自留偏移、relayBytes、尾块大小，
                // 每 channel 的 (peerOffset, payloadOffset)——mesh 组 payloadOffset=转发目标偏移，clos 组恒 0。
                taskArgs.reserve(8 + peerOffsets.size() * 2);
                taskArgs.push_back(recvBase);
                taskArgs.push_back(recvToken);
                taskArgs.push_back(sendBase);
                taskArgs.push_back(sendToken);
                taskArgs.push_back(chunkBytes);
                taskArgs.push_back(rootOffset);
                taskArgs.push_back(relayBytes);
                taskArgs.push_back(tailSize);
                const std::vector<uint32_t> &peers = resCtx.peersPerKernel[kernelIdx];
                for (size_t c = 0; c < peerOffsets.size(); c++) {
                    taskArgs.push_back(peerOffsets[c]);
                    const uint32_t peer = peers[c];
                    const uint32_t target = (peer < resCtx.relayTargetOf.size()) ? resCtx.relayTargetOf[peer]
                                                                                : INVALID_VALUE_RANKID;
                    const uint64_t payloadOffset =
                        (target != INVALID_VALUE_RANKID) ? static_cast<uint64_t>(target) * chunkBytes : 0;
                    taskArgs.push_back(payloadOffset);
                }
            } else if (relayRole == 1) {
                // relay 拆 mesh(die1, 对端=root)/Clos(die0, 对端=target)。mesh 组再拆 gather+own 两 kernel：
                // gather 读 payload→staging（4 args）、own 读自己块→recvBuf（4 args），
                // forward(Clos) 读 staging→target（3 args）在 gather 完成后启动、与 own 读并行。
                const std::vector<uint32_t> &relayPeers = resCtx.peersPerKernel[kernelIdx];
                const bool isMeshKernel = !relayPeers.empty() && relayPeers[0] == param.root;
                if (isMeshKernel) {
                    gatherKernelIdx = kernelIdx;
                    // gather：staging/Token、目标块偏移、本片大小(relaySliceBytes)、本片偏移(0)
                    taskArgs.reserve(5);
                    taskArgs.push_back(resCtx.relayStagingAddr);
                    taskArgs.push_back(stagingToken);
                    taskArgs.push_back(static_cast<uint64_t>(relayPeerRank) * chunkBytes);
                    taskArgs.push_back(relaySliceBytes);
                    taskArgs.push_back(0);
                    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx], resCtx.ccuKernels[kernelIdx],
                        taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));

                    // gather 完成即通知 forward 线程（排在 gather 后、own 前的 stream 位置），
                    // 让 forward(Clos) 与 own(自己块读) 并行。
                    if (forwardKernelIdx != UINT32_MAX) {
                        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                            resCtx.threads[gatherKernelIdx], resCtx.threads[forwardKernelIdx], 1)));
                    }

                    // own：recvBuf/Token、chunkBytes、自己块偏移
                    taskArgs.clear();
                    taskArgs.reserve(4);
                    taskArgs.push_back(recvBase);
                    taskArgs.push_back(recvToken);
                    taskArgs.push_back(chunkBytes);
                    taskArgs.push_back(myRankOffset);
                    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx], resCtx.relayOwnKernel, taskArgs.data(),
                        static_cast<uint32_t>(taskArgs.size())));
                    continue; // gather+own 已 launch，跳过底部统一 launch
                }
                // Clos(forward)：参数在 post-sync 里按切片构造（staging/Token、本片大小、本片偏移、是否最后一片）
                forwardKernelIdx = kernelIdx;
                continue; // 跳过本次 launch，等 gather 完成后再单独 launch
            } else if (relayRole == 2) {
                // target：recvBuf/Token
                taskArgs.reserve(2);
                taskArgs.push_back(recvBase);
                taskArgs.push_back(recvToken);
            } else {
                // 直收整块的 target（2×8 的 rank15 / 4×3 的 rank5..11），走 direct non-root 布局。
                taskArgs.reserve(7 + peerOffsets.size());
                taskArgs.push_back(recvBase);
                taskArgs.push_back(recvToken);
                taskArgs.push_back(sendBase);
                taskArgs.push_back(sendToken);
                taskArgs.push_back(0);          // sliceOffset（单写，恒 0）
                taskArgs.push_back(chunkBytes); // sliceSize
                taskArgs.push_back(rootOffset);
                for (const uint64_t peerOffset : peerOffsets) {
                    taskArgs.push_back(peerOffset);
                }
            }
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx], resCtx.ccuKernels[kernelIdx], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size())));
        }

        if (relayRole == 1) {
            // 自定义 post-sync（替代 PostSyncThreads）：按 relaySliceBytes 切片做 gather↔forward 循环。
            // 每片 forward 在该片 gather 完成后启动；下一片 gather 在上一片 forward 完成后启动（staging 复用）。
            if (forwardKernelIdx != UINT32_MAX && gatherKernelIdx != UINT32_MAX) {
                uint64_t sliceOffset = 0;
                while (sliceOffset < relayBytes) {
                    const uint64_t sliceSize = std::min<uint64_t>(relaySliceBytes, relayBytes - sliceOffset);
                    const bool isLast = (sliceOffset + sliceSize >= relayBytes);

                    // 等 gather 本片完成（slot 1），启动 forward 本片。
                    CHK_RET(static_cast<HcclResult>(
                        HcommThreadNotifyWaitOnThread(resCtx.threads[forwardKernelIdx], 1, CUSTOM_TIMEOUT)));
                    std::vector<uint64_t> fwdArgs;
                    fwdArgs.reserve(4);
                    fwdArgs.push_back(resCtx.relayStagingAddr);
                    fwdArgs.push_back(stagingToken);
                    fwdArgs.push_back(sliceSize);
                    fwdArgs.push_back(sliceOffset);
                    const CcuKernelHandle fwdKernel =
                        isLast ? resCtx.ccuKernels[forwardKernelIdx] : resCtx.relayClosNoDoneKernel;
                    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[forwardKernelIdx], fwdKernel, fwdArgs.data(),
                        static_cast<uint32_t>(fwdArgs.size())));

                    sliceOffset += sliceSize;
                    if (sliceOffset < relayBytes) {
                        // forward 本片完成（staging 可复用）→ 通知 gather 线程（slot 0），启动下一片 gather。
                        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                            resCtx.threads[forwardKernelIdx], resCtx.threads[gatherKernelIdx], 0)));
                        CHK_RET(static_cast<HcclResult>(
                            HcommThreadNotifyWaitOnThread(resCtx.threads[gatherKernelIdx], 0, CUSTOM_TIMEOUT)));
                        const uint64_t nextSize = std::min<uint64_t>(relaySliceBytes, relayBytes - sliceOffset);
                        std::vector<uint64_t> gathArgs;
                        gathArgs.reserve(5);
                        gathArgs.push_back(resCtx.relayStagingAddr);
                        gathArgs.push_back(stagingToken);
                        gathArgs.push_back(static_cast<uint64_t>(relayPeerRank) * chunkBytes);
                        gathArgs.push_back(nextSize);
                        gathArgs.push_back(sliceOffset);
                        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[gatherKernelIdx],
                            resCtx.ccuKernels[gatherKernelIdx], gathArgs.data(),
                            static_cast<uint32_t>(gathArgs.size())));
                        // 下一片 gather 完成 → 通知 forward 线程（slot 1）。
                        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                            resCtx.threads[gatherKernelIdx], resCtx.threads[forwardKernelIdx], 1)));
                    }
                }
            }
            if (gatherKernelIdx != UINT32_MAX) {
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(resCtx.threads[gatherKernelIdx], resCtx.threads[0], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
            }
        } else {
            CHK_RET(PostSyncThreads(resCtx.threads));
        }
        return HCCL_SUCCESS;
    }

    // 大消息（>256MB）：分片下放 kernel 内，单次 launch 背靠背发 2 片，地址交换/DONE 只做一次。
    if (resCtx.useLarge != 0) {
        const uint64_t slice1Size = MAX_DATA_SIZE;
        const uint64_t slice2Size = chunkBytes - MAX_DATA_SIZE;
        CHK_RET(PreSyncThreads(resCtx.threads));
        for (uint32_t kernelIdx = 0; kernelIdx < resCtx.ccuKernels.size(); kernelIdx++) {
            const std::vector<uint64_t> &peerOffsets = peerOffsetsPerKernel[kernelIdx];
            // 任务参数布局：recvBuf 地址/Token、sendBuf 地址/Token、两片大小、root 自留偏移、各对端 chunk 偏移。
            std::vector<uint64_t> taskArgs;
            taskArgs.reserve(7 + peerOffsets.size());
            taskArgs.push_back(recvBase);
            taskArgs.push_back(recvToken);
            taskArgs.push_back(sendBase);
            taskArgs.push_back(sendToken);
            taskArgs.push_back(slice1Size);
            taskArgs.push_back(slice2Size);
            taskArgs.push_back(rootOffset);
            for (const uint64_t peerOffset : peerOffsets) {
                taskArgs.push_back(peerOffset);
            }
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx], resCtx.ccuKernels[kernelIdx], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size())));
        }
        CHK_RET(PostSyncThreads(resCtx.threads));
        return HCCL_SUCCESS;
    }

    // 单次通信量上限为 256MB（MAX_DATA_SIZE），512MB / 400M+4B 等大消息按分片循环下发。
    uint64_t processedSize = 0;
    while (processedSize < chunkBytes) {
        const uint64_t sliceSize = std::min<uint64_t>(MAX_DATA_SIZE, chunkBytes - processedSize);
        const uint64_t sliceOffset = processedSize;

        CHK_RET(PreSyncThreads(resCtx.threads));
        for (uint32_t kernelIdx = 0; kernelIdx < resCtx.ccuKernels.size(); kernelIdx++) {
            const std::vector<uint64_t> &peerOffsets = peerOffsetsPerKernel[kernelIdx];
            // 任务参数布局：recvBuf 地址/Token、sendBuf 地址/Token、分片偏移/大小、root 自留偏移，
            // 之后依次为每个对端 chunk 的源偏移。
            std::vector<uint64_t> taskArgs;
            taskArgs.reserve(7 + peerOffsets.size());
            taskArgs.push_back(recvBase);
            taskArgs.push_back(recvToken);
            taskArgs.push_back(sendBase);
            taskArgs.push_back(sendToken);
            taskArgs.push_back(sliceOffset);
            taskArgs.push_back(sliceSize);
            taskArgs.push_back(rootOffset);
            for (const uint64_t peerOffset : peerOffsets) {
                taskArgs.push_back(peerOffset);
            }

            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx], resCtx.ccuKernels[kernelIdx], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size())));
        }
        CHK_RET(PostSyncThreads(resCtx.threads));

        processedSize += sliceSize;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
