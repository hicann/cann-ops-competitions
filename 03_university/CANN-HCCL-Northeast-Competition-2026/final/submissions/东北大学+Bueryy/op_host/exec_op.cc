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
#include <type_traits>
#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
using namespace scatter_plan;

HcclResult Token(void *pointer, uint64_t size, uint64_t &token)
{
    token = 0;
    if (size == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(pointer);
    CHK_PRT_RET(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(pointer), size, &token) != CCU_SUCCESS,
                HCCL_ERROR("获取通信内存 Token 失败"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

template<class Threads>
HcclResult SyncBefore(const Threads &threads)
{
    if (threads.size() == 2) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[1], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[1], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

template<class Threads>
HcclResult SyncAfter(const Threads &threads)
{
    if (threads.size() == 2) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], 0, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[1], threads[0], 0)));
    }
    return HCCL_SUCCESS;
}

HcclResult SyncPhaseBarrier(const std::vector<ThreadHandle> &threads)
{
    if (threads.size() == 2) {
        // 中间屏障独占槽1，防止较快线程的首尾通知覆盖尚未消费的阶段通知。
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[1], threads[0], 1)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[1], 1)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], 1, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[1], 1, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

template<class Context>
ReadMetadataMode GetReadMetadataMode(const OpParam &param, const Context &ctx,
                                     uint64_t inputAddress)
{
    const uint64_t rankBytes = param.count * sizeof(float);
    if (ctx.registeredRoot == param.root &&
        reinterpret_cast<uint64_t>(ctx.registeredInput.addr) == inputAddress &&
        ctx.registeredInput.size >= rankBytes * param.rankSize) {
        return ReadMetadataMode::RegisteredAddress;
    }
    return ReadMetadataMode::Refresh;
}

template<class Context>
HcclResult LaunchRead(const OpParam &param, const Context &ctx, uint64_t offset,
                      uint64_t bytes, ReadMetadataMode mode,
                      const std::array<uint64_t, 3> &addresses, const std::array<uint64_t, 3> &tokens,
                      bool noReady, const uint32_t *rootGroups = nullptr)
{
    const uint64_t rankBytes = param.count * sizeof(float);
    const uint64_t totalBytes = rankBytes * param.rankSize;
    const bool largeRefresh = UseLargeReadFastPath(ctx.topology, param.root, totalBytes);
    const bool refresh = largeRefresh || UseSmallRefreshRead(ctx.topology, totalBytes);
    const bool linkedClosRead = UseLinkedClosRead(ctx.topology, param.root, ctx.registeredRoot,
        totalBytes, ctx.linkedInput.bytes, SCATTER_CANDIDATE);
    const bool pullAll = UseUnevenPull(ctx.topology, param.root, ctx.registeredRoot,
        totalBytes, ctx.linkedInput.bytes, SCATTER_CANDIDATE);
    if ((linkedClosRead || pullAll) && param.myRank == param.root) {
        // 叶子会直接读预交换输入；root 不得因私有地址变化单方面切换协议。
        CHK_PRT_RET(addresses[0] != ctx.linkedInput.address || tokens[0] != ctx.linkedInput.token,
            HCCL_ERROR("预交换 Read 的输入地址或 Token 已改变"), HCCL_E_PARA);
    }
    auto launch = [&](uint32_t group, ThreadHandle thread) -> HcclResult {
        const auto &arg = ctx.kernelArgs[group];
        const bool mesh = SameServer(ctx.topology, param.myRank, arg.ranks[0]);
        if (pullAll) {
            const uint64_t source = ctx.linkedInput.address + param.myRank * rankBytes + offset;
            const uint64_t output = addresses[1] + offset;
            const std::array<uint64_t, 11> args = {param.root, UNCONFIRMED_READ_MODE,
                source, ctx.linkedInput.token, output, tokens[1], 0, bytes, 0,
                param.myRank == param.root && source != output ? bytes : 0, 0};
            CHK_PRT_RET(HcommCcuKernelLaunch(thread, ctx.readKernels[group], args.data(), args.size()) != CCU_SUCCESS,
                HCCL_ERROR("8+4 大机预交换 Read 下发失败"), HCCL_E_INTERNAL);
            return HCCL_SUCCESS;
        }
        const bool linkedRead = linkedClosRead && !mesh;
        const bool linkedWrite = !linkedRead && noReady && UseLinkedWrite(ctx.topology, param.root, totalBytes, mesh);
        const bool directedWrite = linkedWrite || UseDirectedWrite(ctx.topology, param.root, totalBytes, mesh);
        const auto launchMode = directedWrite ? ReadMetadataMode::DirectedWrite :
            (refresh ? ReadMetadataMode::LargeRefresh : mode);
        // 预交换 Clos 只等待完成；8+4 大机的小消息自拷贝也必须由 Mesh 承担。
        const bool copyGroup = linkedClosRead ? mesh : UseReadCopyGroup(ctx.topology, param.root, totalBytes,
            mesh, group == ctx.selfGroup);
        const bool mixedMesh = !linkedWrite && UseMixedMesh(ctx.topology, param.root, totalBytes, mesh, SCATTER_CANDIDATE);
        const auto args = linkedRead ? PackLinkedClosReadArguments(param.myRank, param.root,
            rankBytes, offset, bytes, ctx.linkedInput.address, ctx.linkedInput.token, addresses[1], tokens[1]) :
            mixedMesh ? PackMixedMeshArguments(param.myRank, param.root,
            rankBytes, offset, bytes, addresses, tokens, noReady) :
            PackReadArguments(param.myRank, param.root, arg.ranks[0], arg.selfCopy && copyGroup,
                rankBytes, offset, bytes, linkedWrite ? LINK_WRITE_MODE : static_cast<uint64_t>(launchMode), directedWrite,
                addresses, tokens, linkedWrite ? offset : reinterpret_cast<uint64_t>(ctx.registeredInput.addr));
        CHK_PRT_RET(HcommCcuKernelLaunch(thread, ctx.readKernels[group], args.data(),
                                        args.size()) != CCU_SUCCESS,
                    HCCL_ERROR("Scatter 用户缓冲区 Read 下发失败，组 %u", group), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    };
    if (param.myRank != param.root && (totalBytes <= SMALL_TOTAL_BYTES || largeRefresh)) {
        if (rootGroups != nullptr) {
            const uint32_t group = rootGroups[param.root];
            CHK_PRT_RET(group >= ctx.kernelArgs.size(), HCCL_ERROR("固定上下文的 root 分组无效"), HCCL_E_INTERNAL);
            return launch(group, ctx.threads[0]);
        }
        for (uint32_t group = 0; group < ctx.kernelArgs.size(); ++group) {
            const auto &arg = ctx.kernelArgs[group];
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                if (arg.ranks[i] == param.root) {
                    // 前一轮从线程已汇合到主线程；本轮只有这一条入边，无需启动空 die 或交接从线程。
                    return launch(group, ctx.threads[0]);
                }
            }
        }
        HCCL_ERROR("Scatter Read 未找到连接 root 的 channel");
        return HCCL_E_NOT_FOUND;
    }
    if (pullAll && param.myRank == param.root) {
        for (uint32_t group = 0; group < ctx.kernelArgs.size(); ++group) {
            if (SameServer(ctx.topology, param.myRank, ctx.kernelArgs[group].ranks[0])) {
                return launch(group, ctx.threads[0]);
            }
        }
        return HCCL_E_NOT_FOUND;
    }
    if (linkedClosRead && param.myRank == param.root) {
        uint32_t mesh = UINT32_MAX, clos = UINT32_MAX;
        for (uint32_t group = 0; group < ctx.kernelArgs.size(); ++group) {
            (SameServer(ctx.topology, param.myRank, ctx.kernelArgs[group].ranks[0]) ? mesh : clos) = group;
        }
        CHK_PRT_RET(mesh == UINT32_MAX || clos == UINT32_MAX,
            HCCL_ERROR("预交换 Read 缺少完整的 Mesh 和 Clos 分组"), HCCL_E_INTERNAL);
        // 远端自主 Read 与 Mesh 并行；root 主线程最后消费所有读完成通知。
        // 两个 kernel 都在主线程，不创建缺少入口 WAIT 的从线程任务队列。
        CHK_RET(launch(mesh, ctx.threads[0]));
        if (ctx.topology == Topology::FourByThree) return HCCL_SUCCESS;
        return launch(clos, ctx.threads[0]);
    }
    if (SCATTER_CANDIDATE == 0 && largeRefresh && param.myRank == param.root) {
        uint32_t mesh = UINT32_MAX, clos = UINT32_MAX;
        for (uint32_t group = 0; group < ctx.kernelArgs.size(); ++group) {
            (SameServer(ctx.topology, param.myRank, ctx.kernelArgs[group].ranks[0]) ? mesh : clos) = group;
        }
        CHK_PRT_RET(ctx.threads.size() != 2 || mesh == UINT32_MAX || clos == UINT32_MAX,
            HCCL_ERROR("8+4 大机 root 缺少完整分组"), HCCL_E_INTERNAL);
        CHK_RET(SyncBefore(ctx.threads));
        // 物理 die 仍由 channel 决定；先安排从线程 Clos，再由稳定主线程执行 Mesh。
        CHK_RET(launch(clos, ctx.threads[1]));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(ctx.threads[1], ctx.threads[0], 0)));
        CHK_RET(launch(mesh, ctx.threads[0]));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(ctx.threads[0], 0, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }
    CHK_RET(SyncBefore(ctx.threads));
    for (uint32_t group = 0; group < ctx.kernelArgs.size(); ++group) {
        CHK_RET(launch(group, ctx.threads[group]));
    }
    CHK_RET(SyncAfter(ctx.threads));
    return HCCL_SUCCESS;
}

HcclResult LaunchPhase(const OpParam &param, const AlgResourceCtx &ctx, const Plan &plan,
                       uint32_t phase, const std::array<uint64_t, 3> &addresses,
                       const std::array<uint64_t, 3> &tokens, bool noReady,
                       uint32_t onlyGroup = UINT32_MAX, bool mainOnly = false)
{
    for (uint32_t group = 0; group < ctx.kernelArgs.size(); ++group) {
        if (onlyGroup != UINT32_MAX && group != onlyGroup) {
            continue;
        }
        const auto &kernelArg = ctx.kernelArgs[group];
        // 中继拓扑的单阶段直写也共用带活动槽位参数的写 Kernel。
        const bool relayLayout = ctx.relayKernels[group] != 0;
        // 本组接收地址的用途由本轮入边决定，所有 channel 只发布一个地址和 Token。
        Buffer published = Buffer::Output;
        for (const auto &edge : plan.transfers) {
            if (edge.phase == phase && edge.target == param.myRank && edge.source != param.myRank) {
                for (uint32_t i = 0; i < kernelArg.channelCount; ++i) {
                    if (kernelArg.ranks[i] == edge.source) {
                        published = edge.targetBuffer;
                    }
                }
            }
        }
        const size_t publishedIndex = static_cast<size_t>(published);
        std::vector<uint64_t> args = {addresses[publishedIndex], tokens[publishedIndex], 0, 0, 0, 0, 0};
        if (kernelArg.selfCopy && group == ctx.selfGroup) {
            for (const auto &edge : plan.transfers) {
                if (edge.phase == phase && edge.source == param.myRank && edge.target == param.myRank) {
                    const size_t sourceBuffer = static_cast<size_t>(edge.sourceBuffer);
                    const size_t targetBuffer = static_cast<size_t>(edge.targetBuffer);
                    args[2] = addresses[sourceBuffer] + edge.sourceOffset;
                    args[3] = tokens[sourceBuffer];
                    args[4] = addresses[targetBuffer] + edge.targetOffset;
                    args[5] = tokens[targetBuffer];
                    args[6] = args[2] == args[4] ? 0 : edge.bytes;
                }
            }
        }
        std::vector<std::vector<const Transfer *>> outgoing(kernelArg.channelCount);
        for (const auto &edge : plan.transfers) {
            if (edge.phase != phase || edge.source == edge.target) {
                continue;
            }
            for (uint32_t i = 0; i < kernelArg.channelCount; ++i) {
                if (edge.source == param.myRank && edge.target == kernelArg.ranks[i]) {
                    outgoing[i].push_back(&edge);
                }
            }
        }
        const uint32_t slots = kernelArg.slots;
        for (const auto &edges : outgoing) {
            CHK_PRT_RET(edges.size() > slots, HCCL_ERROR("单 channel 分片数超过已注册槽位"), HCCL_E_INTERNAL);
        }
        // 2×8 Mesh 保留两阶段缓存，Clos 的十一参数包含本轮 root 与中继模式。
        if (kernelArg.optimization == WriteOptimization::FusedMetadata) {
            const bool mesh = SameServer(Topology::TwoByEight, param.myRank, kernelArg.ranks[0]);
            if (mesh) {
                const auto mode = plan.phases == 1 ? WriteMetadataMode::Refresh :
                    (phase == 0 ? WriteMetadataMode::PublishBoth : WriteMetadataMode::ReuseOutput);
                args = PackPairMeshArguments(args, outgoing, static_cast<uint64_t>(mode), addresses, tokens);
            } else {
                CHK_PRT_RET(plan.phases != 1 || kernelArg.selfCopy,
                    HCCL_ERROR("2×8 Clos 必须使用无本地拷贝的单阶段计划"), HCCL_E_INTERNAL);
                args = PackPairClosArguments(args, outgoing, param.myRank, param.root,
                    param.count * sizeof(float), addresses, tokens, plan.algorithm == Algorithm::PairRelay);
            }
            CHK_PRT_RET(args.size() != (mesh ? PAIR_MESH_WRITE_ARGS : PAIR_CLOS_WRITE_ARGS),
                HCCL_ERROR("2×8 紧凑参数数量不符"), HCCL_E_INTERNAL);
        } else if (kernelArg.optimization == WriteOptimization::Compact) {
            const bool mesh = SameServer(Topology::FourByThree, param.myRank, kernelArg.ranks[0]);
            const bool fusedRoot = SCATTER_CANDIDATE == 0 && mesh &&
                plan.algorithm == Algorithm::TriRelay && param.myRank == param.root && phase == 0;
            const auto mode = fusedRoot ? WriteMetadataMode::RootFusedBoth :
                (plan.phases == 1 ? WriteMetadataMode::Refresh :
                    (phase == 0 ? WriteMetadataMode::PublishBoth : WriteMetadataMode::ReuseOutput));
            args = PackTriWriteArguments(args, outgoing, kernelArg.ranks, param.myRank,
                param.count * sizeof(float), addresses, tokens, param.root / 3, static_cast<uint64_t>(mode));
            if (fusedRoot) PackTriRootFinal(args, plan, param.root, kernelArg.ranks, addresses);
            CHK_PRT_RET(args.size() != (mesh ? TRI_MESH_WRITE_ARGS : TRI_CLOS_WRITE_ARGS),
                HCCL_ERROR("4×3 紧凑参数数量不符"), HCCL_E_INTERNAL);
        } else {
            // 其余拓扑沿用公共七项、可选槽位标记和逐通道搬运参数。
            if (relayLayout) {
                for (uint32_t slot = 0; slot < slots; ++slot) {
                    bool active = slot == 0 && kernelArg.selfCopy && args[6] != 0;
                    for (const auto &edges : outgoing) {
                        active = active || slot < edges.size();
                    }
                    args.push_back(active ? 1 : 0);
                }
            }
            for (uint32_t slot = 0; slot < slots; ++slot) {
                for (uint32_t i = 0; i < kernelArg.channelCount; ++i) {
                    if (slot >= outgoing[i].size()) {
                        args.insert(args.end(), {0, 0, 0, 0});
                        continue;
                    }
                    const auto &edge = *outgoing[i][slot];
                    const size_t localBuffer = static_cast<size_t>(edge.sourceBuffer);
                    args.push_back(addresses[localBuffer] + edge.sourceOffset);
                    args.push_back(tokens[localBuffer]);
                    args.push_back(edge.targetOffset);
                    args.push_back(edge.bytes);
                }
            }
        }
        const uint64_t linkMode = LinkedTargetMode(outgoing, noReady);
        CHK_PRT_RET(linkMode > 2, HCCL_ERROR("单个写组混合了不同建链目标缓冲区"), HCCL_E_INTERNAL);
        args.push_back(linkMode);
        const auto kernel = ctx.writeKernels[group];
        CHK_PRT_RET(HcommCcuKernelLaunch(ctx.threads[mainOnly ? 0 : group],
                                        kernel, args.data(), args.size()) != CCU_SUCCESS,
                    HCCL_ERROR("Scatter CCU 下发失败，阶段 %u，组 %u", phase, group), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchRelayContinuous(const OpParam &param, const AlgResourceCtx &ctx, const Plan &plan,
                                const std::array<uint64_t, 3> &addresses,
                                const std::array<uint64_t, 3> &tokens, bool noReady)
{
    uint32_t mesh = UINT32_MAX;
    uint32_t clos = UINT32_MAX;
    for (uint32_t group = 0; group < ctx.kernelArgs.size(); ++group) {
        if (SameServer(ctx.topology, param.myRank, ctx.kernelArgs[group].ranks[0])) {
            mesh = group;
        } else {
            clos = group;
        }
    }
    CHK_PRT_RET(ctx.kernelArgs.size() != 2 || mesh == UINT32_MAX || clos == UINT32_MAX,
                HCCL_ERROR("中继连续发送缺少 Mesh 或 Clos 分组"), HCCL_E_INTERNAL);
    const bool helper = param.myRank != param.root &&
        SameServer(ctx.topology, param.myRank, param.root);
    const uint64_t rankBytes = param.count * sizeof(float);
    const bool triPull = UseTriPull(ctx.topology, param.root, ctx.registeredRoot, rankBytes,
        ctx.linkedInput.bytes, ctx.linkedOutput.bytes, ctx.linkedOutput.scratchBytes, SCATTER_CANDIDATE);
    if (triPull && param.myRank == param.root) {
        CHK_PRT_RET(addresses[0] != ctx.linkedInput.address || tokens[0] != ctx.linkedInput.token,
            HCCL_ERROR("4×3 大消息预交换输入已改变"), HCCL_E_PARA);
        // root 不下发 Clos；融合 Mesh 仍完整生产 Scratch、同机 Output 和本地分片。
        return LaunchPhase(param, ctx, plan, 0, addresses, tokens, noReady, mesh, true);
    }
    if (triPull && !helper) {
        const uint64_t bytes = Fraction(rankBytes / 4, TRI_DIRECT_NUM, TRI_DIRECT_DEN) * 4;
        const std::array<uint64_t, 11> args = {param.root, TRI_PULL_READ_MODE,
            ctx.linkedInput.address + param.myRank * rankBytes, ctx.linkedInput.token,
            addresses[1], tokens[1], 0, bytes, 0, 0, 0};
        CHK_PRT_RET(HcommCcuKernelLaunch(ctx.threads[0], ctx.readKernels[clos],
            args.data(), args.size()) != CCU_SUCCESS,
            HCCL_ERROR("4×3 直达 Read 与中继 Write 接收下发失败"), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }
    const Plan closPlan = RelayClosPlan(plan, ctx.topology);
    CHK_RET(SyncBefore(ctx.threads));
    // root 与远端接收者沿原 die 顺序启动；中继 Clos 必须等本地 Mesh 阶段0完成。
    for (uint32_t group = 0; group < ctx.kernelArgs.size(); ++group) {
        if (group == mesh) {
            CHK_RET(LaunchPhase(param, ctx, plan, 0, addresses, tokens, noReady, mesh));
        } else if (!helper) {
            CHK_RET(LaunchPhase(param, ctx, closPlan, 0, addresses, tokens, noReady, clos));
        }
    }
    if (helper) {
        // 槽1只承载本切片暂存区就绪；无反向等待，让 Mesh 阶段1与 Clos 代发重叠。
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(ctx.threads[mesh], ctx.threads[clos], 1)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(ctx.threads[clos], 1, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchPhase(param, ctx, closPlan, 0, addresses, tokens, noReady, clos));
    }
    if (!(SCATTER_CANDIDATE == 0 && ctx.topology == Topology::FourByThree && param.myRank == param.root)) {
        CHK_RET(LaunchPhase(param, ctx, plan, 1, addresses, tokens, noReady, mesh));
    }
    // 本切片两个 die 均结束后才允许下一切片刷新元数据、复用暂存区或返回用户 stream。
    CHK_RET(SyncAfter(ctx.threads));
    return HCCL_SUCCESS;
}
HcclResult BridgeStream(ThreadHandle main, ThreadHandle user)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(main, user, 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(user, 0, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(user, main, 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(main, 0, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ExecRole(const OpParam &param, HcclComm comm, RoleResourceCtx &ctx)
{
    const bool bridge = ctx.mainThread != param.cpuThread;
    if (bridge) CHK_RET(BridgeStream(ctx.mainThread, param.cpuThread));
    const uint64_t rankBytes = param.count * sizeof(float);
    const std::array<uint64_t, 3> addresses = {reinterpret_cast<uint64_t>(param.inputPtr),
        reinterpret_cast<uint64_t>(param.outputPtr), 0};
    const bool linkedRead = ctx.linkedOutput.bytes != 0 && UseRoleLinkedRead(param.root,
        ctx.registeredRoot, rankBytes * param.rankSize, ctx.linkedInput.bytes);
    if (linkedRead) {
        CHK_PRT_RET(param.myRank == param.root && addresses[0] != ctx.linkedInput.address,
            HCCL_ERROR("4×1 预交换输入地址已改变"), HCCL_E_PARA);
        // 用户授权复用建链 token；仅本地输出地址或容量变化才重新查询。
        uint64_t outputToken = ctx.linkedOutput.token;
        if (addresses[1] != ctx.linkedOutput.address || rankBytes > ctx.linkedOutput.bytes) {
            CHK_RET(Token(param.outputPtr, rankBytes, outputToken));
        }
        const uint64_t source = ctx.linkedInput.address + param.myRank * rankBytes;
        const std::array<uint64_t, ROLE_FAST_READ_ARGS> args = {source, addresses[1],
            ctx.linkedInput.token, outputToken,
            param.myRank == param.root && source == addresses[1] ? 0 : rankBytes};
        CHK_PRT_RET(HcommCcuKernelLaunch(ctx.mainThread, ctx.receiver, args.data(), args.size()) != CCU_SUCCESS,
            HCCL_ERROR("4×1 专用 Read 下发失败"), HCCL_E_INTERNAL);
        // 快路径没有修改 channel 元数据，旧 Write 地址缓存仍代表真实发布过的内容。
        if (bridge) CHK_RET(BridgeStream(ctx.mainThread, param.cpuThread));
        return HCCL_SUCCESS;
    }
    std::array<uint64_t, 3> tokens{};
    CHK_RET(Token(param.outputPtr, rankBytes, tokens[1]));
    if (param.myRank == param.root) CHK_RET(Token(param.inputPtr, rankBytes * param.rankSize, tokens[0]));
    const bool linkedKernel = ctx.linkedOutput.bytes != 0;
    const bool noReady = UseRoleLinkWrite(ctx.linkedOutput.bytes, rankBytes);
    if (noReady && param.myRank != param.root) {
        CHK_PRT_RET(addresses[1] != ctx.linkedOutput.address || tokens[1] != ctx.linkedOutput.token,
            HCCL_ERROR("4×1 免 READY 输出描述已改变"), HCCL_E_PARA);
    }
    const auto previousRoot = ctx.publishedRoot;
    const auto previousOutput = ctx.publishedOutput;
    if (ctx.publishedRoot != param.root) {
        ctx.publishedRoot = param.root;
        ctx.publishedOutput = 0;
    }
    if (noReady) ctx.publishedOutput = 0;
    for (uint64_t offset = 0; offset < rankBytes;) {
        const uint64_t bytes = std::min(TRANSFER_LIMIT, rankBytes - offset);
        std::array<uint64_t, ROLE_UNIFIED_WRITE_ARGS> unified{};
        unified[10] = noReady;
        if (param.myRank == param.root) {
            const auto args = PackRoleSendArguments(param.root, ctx.peers, rankBytes,
                offset, bytes, addresses, tokens);
            unified[0] = 3;
            std::copy(args.begin(), args.end(), unified.begin() + 1);
            const auto result = linkedKernel ?
                HcommCcuKernelLaunch(ctx.mainThread, ctx.sender, unified.data(), unified.size()) :
                HcommCcuKernelLaunch(ctx.mainThread, ctx.sender, args.data(), args.size());
            CHK_PRT_RET(result != CCU_SUCCESS, HCCL_ERROR("4×1 写发送下发失败"), HCCL_E_INTERNAL);
        } else {
            const uint64_t output = addresses[1] + offset;
            const auto args = PackRoleReceiveArguments(param.root, ctx.peers, output, tokens[1],
                ctx.publishedOutput != output);
            CHK_PRT_RET(args[0] >= 3, HCCL_ERROR("4×1 未找到 root 通道"), HCCL_E_INTERNAL);
            std::copy(args.begin(), args.end(), unified.begin());
            const auto result = linkedKernel ?
                HcommCcuKernelLaunch(ctx.mainThread, ctx.sender, unified.data(), unified.size()) :
                HcommCcuKernelLaunch(ctx.mainThread, ctx.receiver, args.data(), args.size());
            CHK_PRT_RET(result != CCU_SUCCESS, HCCL_ERROR("4×1 写接收下发失败"), HCCL_E_INTERNAL);
            if (!noReady) ctx.publishedOutput = output;
        }
        offset += bytes;
    }
    if (previousRoot != ctx.publishedRoot || previousOutput != ctx.publishedOutput) {
        CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag, &ctx, sizeof(ctx), 0));
    }
    if (bridge) CHK_RET(BridgeStream(ctx.mainThread, param.cpuThread));
    return HCCL_SUCCESS;
}

HcclResult ExecFastRead(const OpParam &param, const FastReadResourceCtx &ctx)
{
    const bool bridge = ctx.threads[0] != param.cpuThread;
    if (bridge) CHK_RET(BridgeStream(ctx.threads[0], param.cpuThread));
    const uint64_t rankBytes = param.count * sizeof(float);
    const std::array<uint64_t, 3> addresses = {reinterpret_cast<uint64_t>(param.inputPtr),
        reinterpret_cast<uint64_t>(param.outputPtr), 0};
    std::array<uint64_t, 3> tokens{};
    CHK_RET(Token(param.outputPtr, rankBytes, tokens[1]));
    if (param.myRank == param.root) CHK_RET(Token(param.inputPtr, rankBytes * param.rankSize, tokens[0]));
    const bool noReady = UseRoleLinkWrite(ctx.linkedOutput.bytes, rankBytes);
    const bool linkedClosReceiver = UseUnevenPull(ctx.topology, param.root, ctx.registeredRoot,
        rankBytes * param.rankSize, ctx.linkedInput.bytes, SCATTER_CANDIDATE) ||
        (UseLinkedClosRead(ctx.topology, param.root, ctx.registeredRoot,
            rankBytes * param.rankSize, ctx.linkedInput.bytes, SCATTER_CANDIDATE) &&
            !SameServer(ctx.topology, param.myRank, param.root));
    if (noReady && param.myRank != param.root && !linkedClosReceiver) {
        CHK_PRT_RET(addresses[1] != ctx.linkedOutput.address || tokens[1] != ctx.linkedOutput.token,
                    HCCL_ERROR("免 READY 的输出地址或 Token 已改变"), HCCL_E_PARA);
    }
    const auto mode = param.myRank == param.root ?
        GetReadMetadataMode(param, ctx, addresses[0]) : ReadMetadataMode::Refresh;
    for (uint64_t offset = 0; offset < rankBytes;) {
        const uint64_t bytes = std::min(TRANSFER_LIMIT, rankBytes - offset);
        CHK_RET(LaunchRead(param, ctx, offset, bytes, mode, addresses, tokens, noReady, ctx.rootGroups.data()));
        offset += bytes;
    }
    if (bridge) CHK_RET(BridgeStream(ctx.threads[0], param.cpuThread));
    return HCCL_SUCCESS;
}

}

HcclResult ExecOp(const OpParam &param, HcclComm comm)
{
    static_assert(std::is_trivially_copyable<RoleResourceCtx>::value, "固定上下文必须可按字节复制");
    if (SCATTER_CANDIDATE == 0 && param.ctxSize == sizeof(RoleResourceCtx)) {
        RoleResourceCtx fixed;
        std::memcpy(&fixed, param.resCtx, sizeof(fixed));
        if (fixed.magic == ROLE_CTX_MAGIC) return ExecRole(param, comm, fixed);
    }
    char *raw = static_cast<char *>(param.resCtx);
    uint64_t serializedBytes = param.ctxSize;
    static_assert(std::is_trivially_copyable<FastReadResourceCtx>::value, "直接路径前缀必须可按字节复制");
    if (SCATTER_CANDIDATE == 0 && param.ctxSize >= sizeof(FastReadResourceCtx)) {
        uint64_t magic = 0;
        std::memcpy(&magic, raw, sizeof(magic));
        if (magic == FAST_CTX_MAGIC) {
            FastReadResourceCtx fixed;
            std::memcpy(&fixed, raw, sizeof(fixed));
            CHK_PRT_RET(fixed.fallbackBytes != param.ctxSize - sizeof(fixed),
                HCCL_ERROR("固定上下文的回退长度无效"), HCCL_E_INTERNAL);
            if (UseFastReadContext(fixed.topology, param.root,
                param.count * sizeof(float) * param.rankSize, SCATTER_CANDIDATE)) {
                return ExecFastRead(param, fixed);
            }
            raw += sizeof(fixed);
            serializedBytes = fixed.fallbackBytes;
        }
    }
    std::vector<char> serialized(raw, raw + serializedBytes);
    AlgResourceCtx ctx;
    ctx.DeSerialize(serialized);
    // 资源保留稳定的编排主线程。切换用户 stream 时建立双向交接，既保证
    // 本次输入已就绪，也让 Checker 能从通信域的主线程到达本次 CCU 图。
    const bool bridgeStream = ctx.threads[0] != param.cpuThread;
    if (bridgeStream) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(ctx.threads[0], param.cpuThread, 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, 0, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, ctx.threads[0], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(ctx.threads[0], 0, CUSTOM_TIMEOUT)));
    }
    constexpr uint32_t elementBytes = sizeof(float);
    const uint64_t rankBytes = param.count * elementBytes;
    const uint64_t totalBytes = rankBytes * param.rankSize;
    const auto algorithm = Select(ctx.topology, param.root, totalBytes, SCATTER_CANDIDATE);
    const bool relay = algorithm == Algorithm::PairRelay || algorithm == Algorithm::UnevenRelay ||
        algorithm == Algorithm::TriRelay;
    const std::array<uint64_t, 3> addresses = {reinterpret_cast<uint64_t>(param.inputPtr),
        reinterpret_cast<uint64_t>(param.outputPtr), reinterpret_cast<uint64_t>(ctx.scratch.addr)};
    std::array<uint64_t, 3> tokens{};
    CHK_RET(Token(param.outputPtr, rankBytes, tokens[1]));
    if (param.myRank == param.root) {
        CHK_RET(Token(param.inputPtr, totalBytes, tokens[0]));
    }
    if (relay) {
        CHK_RET(Token(ctx.scratch.addr, ctx.scratch.size, tokens[2]));
    }
    const bool noReady = UseRoleLinkWrite(ctx.linkedOutput.bytes, rankBytes);
    const bool linkedClosReceiver = UseUnevenPull(ctx.topology, param.root, ctx.registeredRoot,
        totalBytes, ctx.linkedInput.bytes, SCATTER_CANDIDATE) ||
        (UseLinkedClosRead(ctx.topology, param.root, ctx.registeredRoot,
            totalBytes, ctx.linkedInput.bytes, SCATTER_CANDIDATE) &&
            !SameServer(ctx.topology, param.myRank, param.root));
    if (noReady) {
        CHK_PRT_RET(param.myRank != param.root && !linkedClosReceiver &&
                    (addresses[1] != ctx.linkedOutput.address || tokens[1] != ctx.linkedOutput.token),
                    HCCL_ERROR("免 READY 的输出地址或 Token 已改变"), HCCL_E_PARA);
        CHK_PRT_RET(relay && (addresses[2] != ctx.linkedOutput.scratchAddress ||
                    ctx.scratch.size != ctx.linkedOutput.scratchBytes || tokens[2] != ctx.linkedOutput.scratchToken),
                    HCCL_ERROR("免 READY 的暂存区描述已改变"), HCCL_E_PARA);
    }
    const bool read = algorithm == Algorithm::DirectRead;
    const auto mode = read && param.myRank == param.root ?
        GetReadMetadataMode(param, ctx, addresses[0]) : ReadMetadataMode::Refresh;
    HCCL_INFO("Scatter 预研：候选=%u，路径=%u，拓扑=%u，root=%u，rank=%u，recvCount=%llu",
        SCATTER_CANDIDATE, static_cast<uint32_t>(algorithm), static_cast<uint32_t>(ctx.topology),
        param.root, param.myRank, static_cast<unsigned long long>(param.count));
    for (uint64_t offset = 0; offset < rankBytes;) {
        uint64_t sliceBytes = std::min(TRANSFER_LIMIT, rankBytes - offset);
        if (read) {
            CHK_RET(LaunchRead(param, ctx, offset, sliceBytes, mode, addresses, tokens, noReady));
            offset += sliceBytes;
            continue;
        }
        Plan plan;
        while (true) {
            plan = Build(param.rankSize, param.root, algorithm, rankBytes, offset, sliceBytes, elementBytes);
            const uint64_t scratchNeeded = *std::max_element(plan.scratchBytes.begin(), plan.scratchBytes.end());
            if (!relay || scratchNeeded <= ctx.scratch.size) {
                break;
            }
            sliceBytes = (sliceBytes / elementBytes / 2) * elementBytes;
            CHK_PRT_RET(sliceBytes == 0, HCCL_ERROR("暂存区无法容纳最小中继分片"), HCCL_E_MEMORY);
        }
        // 后续切片仍握手，确保中继已读完上一片 Scratch，再允许 root 覆盖。
        const bool sliceNoReady = noReady && offset == 0;
        if (algorithm == Algorithm::PairRelay || algorithm == Algorithm::TriRelay) {
            CHK_RET(LaunchRelayContinuous(param, ctx, plan, addresses, tokens, sliceNoReady));
            offset += sliceBytes;
            continue;
        }
        CHK_RET(SyncBefore(ctx.threads));
        for (uint32_t phase = 0; phase < plan.phases; ++phase) {
            if (phase != 0) {
                CHK_RET(SyncPhaseBarrier(ctx.threads));
            }
            CHK_RET(LaunchPhase(param, ctx, plan, phase, addresses, tokens, sliceNoReady));
        }
        CHK_RET(SyncAfter(ctx.threads));
        offset += sliceBytes;
    }
    if (bridgeStream) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(ctx.threads[0], param.cpuThread, 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, 0, CUSTOM_TIMEOUT)));
        // 用户线程完成等待后向编排线程确认，构成完整的从线程收尾边界。
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, ctx.threads[0], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(ctx.threads[0], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}
}
