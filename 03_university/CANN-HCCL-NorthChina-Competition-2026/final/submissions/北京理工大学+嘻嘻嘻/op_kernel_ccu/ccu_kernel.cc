/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>

#include "ccu_kernel.h"

#define CCU_RETURN_IF_ERROR(call) \
    do { \
        const CcuResult ccuResult = (call); \
        if (ccuResult != CCU_SUCCESS) { \
            return ccuResult; \
        } \
    } while (0)

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {
namespace {

constexpr uint64_t CCU_MS_INTERLEAVE = 8;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint32_t LOCAL_COPY_MS_PER_LOOP = 8;
constexpr uint32_t LOCAL_COPY_LOOP_COUNT = 8;
constexpr uint32_t COPY_LOOP_ENTITY_COUNT = 2;

struct LoopGroupConfig {
    uint32_t msInterleave;
    uint32_t loopCount;
    uint64_t memSlice;
};

struct LoopGroupResource {
    ccu::Array<ccu::Event> completedEvent{0};
    ccu::Array<ccu::CcuBuffer> ccuBuffer{0};
    uint32_t eventCount;
    uint32_t bufferCount;
};

struct GroupOpSizeVars {
    ccu::Variable addrOffset;
    ccu::Variable loopParam;
    ccu::Variable parallelParam;
    ccu::Variable residual;
};

struct GroupCopyVariables {
    ccu::LocalAddr loopSource[COPY_LOOP_ENTITY_COUNT];
    ccu::LocalAddr loopDestination[COPY_LOOP_ENTITY_COUNT];
    ccu::Variable loopLength[COPY_LOOP_ENTITY_COUNT];
};

struct GroupCopyContext {
    LoopGroupConfig config{};
    LoopGroupResource resource{};
    GroupCopyVariables variables{};
    std::unique_ptr<ccu::Func> body[COPY_LOOP_ENTITY_COUNT];
    std::unique_ptr<ccu::Loop> loop[COPY_LOOP_ENTITY_COUNT];
    ccu::Variable loopParam[COPY_LOOP_ENTITY_COUNT];
    bool resourceAllocated{false};
    bool loopEntityCreated{false};
};

struct AllGatherKernelContext {
    const CcuKernelArgAllGather *arg{nullptr};
    ccu::Variable sendBase;
    ccu::Variable inputToken;
    ccu::Variable recvBase;
    ccu::Variable outputToken;
    ccu::Variable rankDataBytes;
    ccu::Variable sliceOffset;
    ccu::Variable sliceBytes;
    GroupOpSizeVars localGoSize;
    std::vector<ccu::Variable> peerRecvBase;
    std::vector<ccu::Variable> peerOutputToken;
    ccu::Event completionEvent;
    std::unique_ptr<GroupCopyContext> groupCopy;
};

/**
 * @brief 生成从最低位开始且包含指定最高位的连续位掩码
 * @param endBit 连续位掩码包含的最高位编号
 * @return 返回生成的无符号 64 位掩码
 */
constexpr uint64_t BuildLowBitMask(uint16_t endBit)
{
    return (uint64_t{1} << (endBit + 1)) - uint64_t{1};
}

/**
 * @brief 按 CCU Loop ABI 编码循环上下文、地址步长和迭代次数
 * @param loopContextId Loop 上下文编号
 * @param gsaOffset 每次循环的全局地址偏移
 * @param loopIterationCount Loop 串行迭代次数
 * @return 返回可直接赋给 CCU Variable 的 Loop 参数
 */
constexpr uint64_t BuildLoopParam(uint64_t loopContextId, uint64_t gsaOffset, uint64_t loopIterationCount)
{
    constexpr uint16_t contextIdBitCount = 8;
    constexpr uint16_t contextIdShift = 45;
    constexpr uint16_t gsaBitCount = 32;
    constexpr uint16_t gsaShift = 13;
    constexpr uint16_t loopCountBitCount = 13;
    return ((loopContextId & BuildLowBitMask(contextIdBitCount)) << contextIdShift)
        | ((gsaOffset & BuildLowBitMask(gsaBitCount)) << gsaShift)
        | (loopIterationCount & BuildLowBitMask(loopCountBitCount));
}

/**
 * @brief 按 CCU LoopGroup ABI 编码并行展开参数
 * @param repeatCount 并行重复次数减一后的编码值
 * @param repeatLoopIndex 从哪个 Loop 实体开始重复
 * @param totalLoopCount 本次 LoopGroup 包含的 Loop 实体数量
 * @return 返回可直接赋给 CCU Variable 的并行展开参数
 */
constexpr uint64_t BuildParallelParam(uint64_t repeatCount, uint64_t repeatLoopIndex, uint64_t totalLoopCount)
{
    constexpr uint16_t repeatBitCount = 7;
    constexpr uint16_t repeatShift = 55;
    constexpr uint16_t repeatLoopBitCount = 7;
    constexpr uint16_t repeatLoopShift = 48;
    constexpr uint16_t totalLoopBitCount = 7;
    constexpr uint16_t totalLoopShift = 41;
    return ((repeatCount & BuildLowBitMask(repeatBitCount)) << repeatShift)
        | ((repeatLoopIndex & BuildLowBitMask(repeatLoopBitCount)) << repeatLoopShift)
        | ((totalLoopCount & BuildLowBitMask(totalLoopBitCount)) << totalLoopShift);
}

/**
 * @brief 按 CCU LoopGroup ABI 编码全局地址、MS 和事件资源偏移
 * @param gsaOffset 每个并行实体的全局地址偏移
 * @param msOffset 每个并行实体的 MS 资源偏移
 * @param eventOffset 每个并行实体的事件资源偏移
 * @return 返回可直接赋给 CCU Variable 的资源偏移参数
 */
constexpr uint64_t BuildOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t eventOffset)
{
    constexpr uint16_t gsaBitCount = 32;
    constexpr uint16_t gsaShift = 21;
    constexpr uint16_t msBitCount = 11;
    constexpr uint16_t msShift = 10;
    constexpr uint16_t eventBitCount = 10;
    return ((gsaOffset & BuildLowBitMask(gsaBitCount)) << gsaShift)
        | ((msOffset & BuildLowBitMask(msBitCount)) << msShift)
        | (eventOffset & BuildLowBitMask(eventBitCount));
}

/**
 * @brief 校验 Kernel 固定参数中的容量、映射和事件掩码关系
 * @param kernelArg 待校验的 AllGather Kernel 固定参数
 * @return 校验通过返回 CCU_SUCCESS，参数非法返回 CCU_E_PARA
 */
CcuResult ValidateKernelArg(const CcuKernelArgAllGather &kernelArg)
{
    if (kernelArg.rankSize == 0 || kernelArg.rankSize > MAX_RANK_SIZE || kernelArg.myRank >= kernelArg.rankSize
        || kernelArg.channelCount > MAX_PEER_COUNT || kernelArg.segmentCount > MAX_SEGMENT_COUNT
        || kernelArg.operationCount > MAX_OPERATION_COUNT
        || kernelArg.operationCount != kernelArg.channelCount * kernelArg.segmentCount
        || kernelArg.variantType > static_cast<uint32_t>(KernelVariantType::RECEIVE_ONLY)
        || kernelArg.phaseId > static_cast<uint32_t>(KernelPhaseId::FAST_LOCAL_FORWARD)
        || kernelArg.sourceKind > static_cast<uint32_t>(SourceKind::OUTPUT) || kernelArg.handleSelf > 1) {
        return CCU_E_PARA;
    }
    if ((kernelArg.variantType == static_cast<uint32_t>(KernelVariantType::RECEIVE_ONLY))
        != (kernelArg.segmentCount == 0)) {
        return CCU_E_PARA;
    }
    if (kernelArg.sourceKind == static_cast<uint32_t>(SourceKind::INPUT)
        && (kernelArg.segmentCount != 1 || kernelArg.segmentRanks[0] != kernelArg.myRank)) {
        return CCU_E_PARA;
    }
    if (kernelArg.handleSelf != 0 && kernelArg.sourceKind != static_cast<uint32_t>(SourceKind::INPUT)) {
        return CCU_E_PARA;
    }

    bool peerSeen[MAX_RANK_SIZE] = {false};
    for (uint32_t peerIndex = 0; peerIndex < kernelArg.channelCount; ++peerIndex) {
        const uint32_t peerRank = kernelArg.peerRanks[peerIndex];
        if (kernelArg.channels[peerIndex] == 0 || peerRank >= kernelArg.rankSize || peerRank == kernelArg.myRank
            || peerSeen[peerRank]) {
            return CCU_E_PARA;
        }
        peerSeen[peerRank] = true;
    }

    for (uint32_t segmentIndex = 0; segmentIndex < kernelArg.segmentCount; ++segmentIndex) {
        if (kernelArg.segmentRanks[segmentIndex] >= kernelArg.rankSize) {
            return CCU_E_PARA;
        }
    }

    bool eventBitSeen[MAX_OPERATION_COUNT] = {false};
    uint16_t expectedMask = 0;
    for (uint32_t operationIndex = 0; operationIndex < kernelArg.operationCount; ++operationIndex) {
        const uint32_t eventBit = kernelArg.eventBits[operationIndex];
        if (eventBit >= MAX_OPERATION_COUNT || eventBitSeen[eventBit]) {
            return CCU_E_PARA;
        }
        eventBitSeen[eventBit] = true;
        expectedMask |= static_cast<uint16_t>(1U << eventBit);
    }
    if (kernelArg.handleSelf != 0) {
        if (kernelArg.selfEventBit >= MAX_OPERATION_COUNT || eventBitSeen[kernelArg.selfEventBit]) {
            return CCU_E_PARA;
        }
        expectedMask |= static_cast<uint16_t>(1U << kernelArg.selfEventBit);
    } else if (kernelArg.selfEventBit != INVALID_EVENT_BIT) {
        return CCU_E_PARA;
    }
    return expectedMask == kernelArg.completionMask ? CCU_SUCCESS : CCU_E_PARA;
}

/**
 * @brief 为官方 mem2mem GroupCopy 分配 LoopGroup 所需的 MS、事件和 Loop 实体
 * @param context 待初始化的本地拷贝上下文
 * @return 初始化成功返回 CCU_SUCCESS
 */
CcuResult InitializeGroupCopyContext(GroupCopyContext &context)
{
    if (!context.resourceAllocated) {
        context.config.msInterleave = CCU_MS_INTERLEAVE;
        context.config.loopCount = LOCAL_COPY_LOOP_COUNT;
        context.config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
        context.resource.eventCount = context.config.loopCount;
        context.resource.completedEvent = ccu::Array<ccu::Event>(context.resource.eventCount);
        context.resource.bufferCount = context.config.loopCount * context.config.msInterleave;
        context.resource.ccuBuffer = ccu::Array<ccu::CcuBuffer>(context.resource.bufferCount);
        context.resourceAllocated = true;
    }
    if (context.loopEntityCreated) {
        return CCU_SUCCESS;
    }

    for (uint32_t index = 0; index < COPY_LOOP_ENTITY_COUNT; ++index) {
        const uint32_t bufferBase = index * context.config.msInterleave;
        ccu::Event loopEvent = context.resource.completedEvent[index];
        context.body[index].reset(new ccu::Func([&context, index, bufferBase, loopEvent]() {
            ccu::LocalCopy(context.resource.ccuBuffer[bufferBase], context.variables.loopSource[index],
                context.variables.loopLength[index], loopEvent, 1);
            ccu::EventWait(loopEvent, 1);
            ccu::LocalCopy(context.variables.loopDestination[index], context.resource.ccuBuffer[bufferBase],
                context.variables.loopLength[index], loopEvent, 1);
            ccu::EventWait(loopEvent, 1);
        }));
        context.loop[index].reset(new ccu::Loop(context.loopParam[index], *context.body[index]));
    }
    context.loopEntityCreated = true;
    return CCU_SUCCESS;
}

/**
 * @brief 使用官方 mem2mem LoopGroup 结构完成一片本地输入到输出的拷贝
 * @param context 本地拷贝使用的 LoopGroup 上下文
 * @param destination 本地输出目标地址及其 Token
 * @param source 本地输入源地址及其 Token
 * @param goSize Host 侧 CalGoSize 生成的四项运行时参数
 * @return 指令生成成功返回 CCU_SUCCESS，资源初始化失败返回对应错误码
 */
CcuResult GroupCopy(GroupCopyContext &context, ccu::LocalAddr &destination, ccu::LocalAddr &source,
    GroupOpSizeVars &goSize)
{
    CCU_RETURN_IF_ERROR(InitializeGroupCopyContext(context));

    ccu::Variable loopParam;
    ccu::Variable sliceSize;
    ccu::Variable parallelConfig;
    ccu::Variable offsetConfig;
    ccu::Variable firstLoopConfig;
    ccu::Variable secondLoopConfig;

    CCU_IF(goSize.addrOffset != 0)
    {
        loopParam = BuildLoopParam(0, context.config.memSlice * context.config.loopCount, 0);
        loopParam += goSize.loopParam;
        sliceSize = context.config.memSlice;

        context.variables.loopSource[0].addr = source.addr;
        context.variables.loopSource[0].token = source.token;
        context.variables.loopDestination[0].addr = destination.addr;
        context.variables.loopDestination[0].token = destination.token;
        context.variables.loopLength[0] = sliceSize;

        context.loopParam[0] = loopParam;
        parallelConfig = BuildParallelParam(context.config.loopCount - 1, 0, 1);
        offsetConfig = BuildOffsetParam(context.config.memSlice, context.config.msInterleave, 1);
        std::vector<ccu::Loop> groupLoops{*context.loop[0]};
        ccu::LoopGroup loopGroup(parallelConfig, offsetConfig, context.config.loopCount, groupLoops);
    }

    CCU_IF(goSize.parallelParam != 0)
    {
        ccu::LocalAddr workingSource;
        ccu::LocalAddr workingDestination;
        workingSource.addr = source.addr;
        workingSource.token = source.token;
        workingDestination.addr = destination.addr;
        workingDestination.token = destination.token;
        workingSource.addr += goSize.addrOffset;
        workingDestination.addr += goSize.addrOffset;

        context.variables.loopSource[0].addr = workingSource.addr;
        context.variables.loopSource[0].token = workingSource.token;
        context.variables.loopDestination[0].addr = workingDestination.addr;
        context.variables.loopDestination[0].token = workingDestination.token;
        context.variables.loopLength[0] = goSize.residual;

        workingSource.addr += goSize.residual;
        workingDestination.addr += goSize.residual;
        sliceSize = context.config.memSlice;

        context.variables.loopSource[1].addr = workingSource.addr;
        context.variables.loopSource[1].token = workingSource.token;
        context.variables.loopDestination[1].addr = workingDestination.addr;
        context.variables.loopDestination[1].token = workingDestination.token;
        context.variables.loopLength[1] = sliceSize;

        firstLoopConfig = BuildLoopParam(0, 0, 1);
        secondLoopConfig = BuildLoopParam(0, 0, 1);
        offsetConfig = BuildOffsetParam(context.config.memSlice, context.config.msInterleave, 1);
        context.loopParam[0] = firstLoopConfig;
        context.loopParam[1] = secondLoopConfig;
        std::vector<ccu::Loop> groupLoops{*context.loop[0], *context.loop[1]};
        ccu::LoopGroup loopGroup(goSize.parallelParam, offsetConfig, context.config.loopCount, groupLoops);
    }
    return CCU_SUCCESS;
}

/**
 * @brief 无条件加载固定 11 项任务参数以保持所有 Kernel 变体 ABI 一致
 * @param context 接收任务参数 Variable 的 Kernel 上下文
 * @return 全部参数加载成功返回 CCU_SUCCESS，加载失败返回对应错误码
 */
CcuResult LoadTaskArgs(AllGatherKernelContext &context)
{
    CCU_RETURN_IF_ERROR(ccu::LoadArg(context.sendBase, TASK_ARG_SEND_BASE));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(context.inputToken, TASK_ARG_INPUT_TOKEN));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(context.recvBase, TASK_ARG_RECV_BASE));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(context.outputToken, TASK_ARG_OUTPUT_TOKEN));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(context.rankDataBytes, TASK_ARG_RANK_DATA_BYTES));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(context.sliceOffset, TASK_ARG_SLICE_OFFSET));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(context.sliceBytes, TASK_ARG_SLICE_BYTES));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(context.localGoSize.addrOffset, TASK_ARG_LOCAL_GO_ADDR_OFFSET));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(context.localGoSize.loopParam, TASK_ARG_LOCAL_GO_LOOP_PARAM));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(context.localGoSize.parallelParam, TASK_ARG_LOCAL_GO_PARALLEL_PARAM));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(context.localGoSize.residual, TASK_ARG_LOCAL_GO_RESIDUAL));
    return CCU_SUCCESS;
}

/**
 * @brief 按 channels 与 peerRanks 平行映射取得每个对端的输出地址和 Token 资源
 * @param context 保存固定参数并接收对端资源的 Kernel 上下文
 * @return 资源获取完成返回 CCU_SUCCESS
 */
CcuResult InitializePeerResources(AllGatherKernelContext &context)
{
    context.peerRecvBase.reserve(context.arg->channelCount);
    context.peerOutputToken.reserve(context.arg->channelCount);
    for (uint32_t peerIndex = 0; peerIndex < context.arg->channelCount; ++peerIndex) {
        context.peerRecvBase.emplace_back(
            ccu::GetResByChannel<ccu::Variable>(context.arg->channels[peerIndex], OUTPUT_XN_ID));
        context.peerOutputToken.emplace_back(
            ccu::GetResByChannel<ccu::Variable>(context.arg->channels[peerIndex], TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

/**
 * @brief 向所有对端交换本端输出基址和输出 Token 并等待双方完成交换
 * @param context 包含本端任务参数和对端 Channel 的 Kernel 上下文
 * @return 前同步成功返回 CCU_SUCCESS，任一通知操作失败返回对应错误码
 */
CcuResult PreSync(AllGatherKernelContext &context)
{
    constexpr uint16_t addressMask = static_cast<uint16_t>(1U << OUTPUT_XN_ID);
    constexpr uint16_t tokenMask = static_cast<uint16_t>(1U << TOKEN_XN_ID);
    constexpr uint16_t allMask = addressMask | tokenMask;
    for (uint32_t peerIndex = 0; peerIndex < context.arg->channelCount; ++peerIndex) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            context.arg->channels[peerIndex], context.recvBase, OUTPUT_XN_ID, CKE_IDX, addressMask));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            context.arg->channels[peerIndex], context.outputToken, TOKEN_XN_ID, CKE_IDX, tokenMask));
    }
    for (uint32_t peerIndex = 0; peerIndex < context.arg->channelCount; ++peerIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(context.arg->channels[peerIndex], CKE_IDX, allMask));
    }
    return CCU_SUCCESS;
}

/**
 * @brief 按 peer 外层和 segment 内层顺序生成远端 AllGather 写操作
 * @param context 包含源类型、静态映射、运行时地址和事件资源的 Kernel 上下文
 * @return 全部写操作生成成功返回 CCU_SUCCESS，CCU 指令生成失败返回对应错误码
 */
CcuResult ExecuteTransfers(AllGatherKernelContext &context)
{
    std::vector<ccu::Variable> segmentOffsets(context.arg->segmentCount);
    std::vector<ccu::LocalAddr> segmentSources(context.arg->segmentCount);
    for (uint32_t segmentIndex = 0; segmentIndex < context.arg->segmentCount; ++segmentIndex) {
        segmentOffsets[segmentIndex] = 0;
        for (uint32_t rankIndex = 0; rankIndex < context.arg->segmentRanks[segmentIndex]; ++rankIndex) {
            segmentOffsets[segmentIndex] += context.rankDataBytes;
        }

        if (context.arg->sourceKind == static_cast<uint32_t>(SourceKind::INPUT)) {
            segmentSources[segmentIndex].addr = context.sendBase;
            segmentSources[segmentIndex].token = context.inputToken;
        } else {
            segmentSources[segmentIndex].addr = context.recvBase;
            segmentSources[segmentIndex].addr += segmentOffsets[segmentIndex];
            segmentSources[segmentIndex].token = context.outputToken;
        }
        segmentSources[segmentIndex].addr += context.sliceOffset;
    }

    uint32_t operationIndex = 0;
    for (uint32_t peerIndex = 0; peerIndex < context.arg->channelCount; ++peerIndex) {
        for (uint32_t segmentIndex = 0; segmentIndex < context.arg->segmentCount;
             ++segmentIndex, ++operationIndex) {
            ccu::RemoteAddr remoteDestination;
            remoteDestination.addr = context.peerRecvBase[peerIndex];
            remoteDestination.addr += segmentOffsets[segmentIndex];
            remoteDestination.addr += context.sliceOffset;
            remoteDestination.token = context.peerOutputToken[peerIndex];
            const uint16_t eventMask =
                static_cast<uint16_t>(1U << context.arg->eventBits[operationIndex]);

            CCU_IF(context.sliceBytes != 0)
            {
                CCU_RETURN_IF_ERROR(ccu::Write(context.arg->channels[peerIndex], remoteDestination,
                    segmentSources[segmentIndex], context.sliceBytes, context.completionEvent, eventMask));
            }
            CCU_IF(context.sliceBytes == 0)
            {
                CCU_RETURN_IF_ERROR(ccu::EventRecord(context.completionEvent, eventMask));
            }
        }
    }
    return CCU_SUCCESS;
}

/**
 * @brief 在唯一 handleSelf 变体中把本端输入片复制到本端 AllGather 输出槽位
 * @param context 包含本端地址、Token、CalGoSize 参数和完成事件的 Kernel 上下文
 * @return 不负责本地拷贝或拷贝成功返回 CCU_SUCCESS，拷贝失败返回对应错误码
 */
CcuResult HandleSelfCopy(AllGatherKernelContext &context)
{
    if (context.arg->handleSelf == 0) {
        return CCU_SUCCESS;
    }

    ccu::Variable selfRankOffset;
    selfRankOffset = 0;
    for (uint32_t rankIndex = 0; rankIndex < context.arg->myRank; ++rankIndex) {
        selfRankOffset += context.rankDataBytes;
    }

    ccu::LocalAddr localSource;
    localSource.addr = context.sendBase;
    localSource.addr += context.sliceOffset;
    localSource.token = context.inputToken;

    ccu::LocalAddr localDestination;
    localDestination.addr = context.recvBase;
    localDestination.addr += selfRankOffset;
    localDestination.addr += context.sliceOffset;
    localDestination.token = context.outputToken;

    context.groupCopy.reset(new GroupCopyContext());
    CCU_RETURN_IF_ERROR(GroupCopy(*context.groupCopy, localDestination, localSource, context.localGoSize));
    const uint16_t selfMask = static_cast<uint16_t>(1U << context.arg->selfEventBit);
    CCU_RETURN_IF_ERROR(ccu::EventRecord(context.completionEvent, selfMask));
    return CCU_SUCCESS;
}

/**
 * @brief 在所有本地数据操作完成后与每个对端执行一次后同步
 * @param context 包含静态 Channel 列表的 Kernel 上下文
 * @return 后同步成功返回 CCU_SUCCESS，任一通知操作失败返回对应错误码
 */
CcuResult PostSync(AllGatherKernelContext &context)
{
    constexpr uint16_t postSyncMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t peerIndex = 0; peerIndex < context.arg->channelCount; ++peerIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(context.arg->channels[peerIndex], CKE_IDX, postSyncMask));
    }
    for (uint32_t peerIndex = 0; peerIndex < context.arg->channelCount; ++peerIndex) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(context.arg->channels[peerIndex], CKE_IDX, postSyncMask));
    }
    return CCU_SUCCESS;
}

}

CcuResult CcuKernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }
    auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
    AllGatherKernelContext context;
    context.arg = kernelArg;

    CCU_RETURN_IF_ERROR(LoadTaskArgs(context));
    CCU_RETURN_IF_ERROR(ValidateKernelArg(*kernelArg));
    CCU_RETURN_IF_ERROR(InitializePeerResources(context));
    // 2×8 forward 复用前一 gather 在相同本地 Channel 写入的 XN
    if (kernelArg->rankSize != 16
        || kernelArg->phaseId != static_cast<uint32_t>(KernelPhaseId::FAST_LOCAL_FORWARD)) {
        // 仅首片交换稳定的输出地址和 Token，后续片复用 Channel XN
        CCU_IF(context.sliceOffset == 0)
        {
            CCU_RETURN_IF_ERROR(PreSync(context));
        }
    }
    CCU_RETURN_IF_ERROR(ExecuteTransfers(context));
    CCU_RETURN_IF_ERROR(HandleSelfCopy(context));
    if (kernelArg->completionMask != 0) {
        CCU_RETURN_IF_ERROR(ccu::EventWait(context.completionEvent, kernelArg->completionMask));
    }
    CCU_RETURN_IF_ERROR(PostSync(context));
    return CCU_SUCCESS;
}

}

#undef CCU_RETURN_IF_ERROR
