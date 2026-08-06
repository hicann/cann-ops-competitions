/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

// 北京科技大学南风的队伍：决赛通信算法编排实现。

#include <algorithm>
#include <vector>
#include <hccl/hcomm_primitives.h>

#include "log.h"
#include "ccu_launch.h"
#include "ccu_res.h"
#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
// 超过该阈值的大消息在评测用 Rank-12/Rank-16 拓扑上使用专用 Gateway 调度。
constexpr uint64_t LARGE_THRESHOLD = 1024 * 1024;

// 将逻辑输入范围切分为固定数量的 Gateway 分片。
void FillPipelineSlice(uint64_t count, uint32_t chunk,
    uint64_t &offset, uint64_t &sliceCount)
{
    const uint64_t base = count / G2_PIPELINE_CHUNK_COUNT;
    const uint64_t remainder = count % G2_PIPELINE_CHUNK_COUNT;
    sliceCount = base + (chunk < remainder ? 1U : 0U);
    offset = chunk * base + std::min<uint64_t>(chunk, remainder);
}

// 将逻辑转发段换算为元素偏移和长度。
void FillForwardTransfers(uint64_t sliceOffset, uint64_t sliceCount,
    const GatewayPlan &plan,
    uint64_t (&offsets)[G2_MAX_FORWARD_TRANSFERS],
    uint64_t (&counts)[G2_MAX_FORWARD_TRANSFERS])
{
    const uint64_t base = sliceCount / plan.forwardSegmentCount;
    const uint64_t remainder = sliceCount % plan.forwardSegmentCount;
    for (uint32_t transfer = 0;
        transfer < plan.forwardChannelIndices.size(); ++transfer) {
        const uint32_t begin = plan.forwardSegmentBegins[transfer];
        const uint32_t span = plan.forwardSegmentSpans[transfer];
        offsets[transfer] = sliceOffset + begin * base +
            std::min<uint64_t>(begin, remainder);
        const uint64_t remainderInSpan = remainder > begin ?
            std::min<uint64_t>(remainder - begin, span) : 0;
        counts[transfer] = span * base + remainderInSpan;
    }
}

// 填充 CcuKernel 消费的参数 ABI 结构。
void FillAllGatherArg(const OpParam &param, const AlgResourceCtx &context,
    const std::vector<ChannelHandle> &channels, const std::vector<uint32_t> &ranks,
    uint64_t sliceOffset, uint64_t sliceCount, uint64_t inputToken,
    uint64_t outputToken, uint32_t copyLocal, uint32_t kernelMode,
    AllGatherKernelArg &arg)
{
    arg.sendBuf = param.inputPtr;
    arg.recvBuf = param.outputPtr;
    arg.sendCount = param.count;
    arg.sliceOffset = sliceOffset;
    arg.sliceCount = sliceCount;
    arg.inputToken = inputToken;
    arg.outputToken = outputToken;
    arg.dataTypeSize = context.dataTypeSize;
    arg.myRank = context.myRank;
    arg.rankSize = context.rankSize;
    arg.copyLocalOutput = copyLocal;
    arg.kernelMode = kernelMode;
    arg.channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t i = 0; i < channels.size(); ++i) {
        arg.channels[i] = channels[i];
        arg.channelIndexToRank[i] = ranks[i];
    }
}

// 执行两阶段 Gateway 算法：先跨组交换，再本地组交换，
// 最后通过本地链路转发缺失的远端数据分片。
HcclResult RunGateway(const OpParam &param,
    const AlgResourceCtx &context,
    uint64_t inputToken, uint64_t outputToken)
{
    const GatewayPlan &plan = context.gateway;
    const uint32_t forwardCount =
        static_cast<uint32_t>(plan.forwardChannelIndices.size());
    const bool rank16Shape = context.rankSize == 16 &&
        plan.localChannels.size() == 7 &&
        plan.crossChannels.size() == 5 &&
        plan.forwardSegmentCount == 7 &&
        forwardCount == 15;
    const bool rank12Shape = context.rankSize == 12 && (
        (plan.localChannels.size() == 7 &&
         plan.crossChannels.size() == 3 &&
         plan.forwardSegmentCount == 3 &&
         forwardCount == 3) ||
        (plan.localChannels.size() == 3 && plan.crossChannels.size() == 6 &&
         plan.forwardSegmentCount == 3 &&
         forwardCount == 6));
    CHK_PRT_RET(context.threads.size() != 2 ||
            (!rank16Shape && !rank12Shape) ||
            plan.localRanks.size() != plan.localChannels.size() ||
            plan.crossRanks.size() != plan.crossChannels.size() ||
            plan.forwardSourceRanks.size() != forwardCount ||
            plan.forwardSegmentBegins.size() != forwardCount ||
            plan.forwardSegmentSpans.size() != forwardCount ||
            forwardCount > G2_MAX_FORWARD_TRANSFERS ||
            plan.crossKernel == 0 || plan.localKernel == 0 ||
            plan.forwardKernel == 0,
        HCCL_ERROR("G2 context is incomplete"), HCCL_E_INTERNAL);
    CHK_PRT_RET(param.count < G2_PIPELINE_CHUNK_COUNT *
            plan.forwardSegmentCount,
        HCCL_ERROR("G2 input is too small for non-zero pipeline slices"),
        HCCL_E_NOT_SUPPORT);
    for (uint32_t transfer = 0; transfer < forwardCount; ++transfer) {
        CHK_PRT_RET(
            plan.forwardChannelIndices[transfer] >=
                    plan.localChannels.size() ||
                plan.forwardSourceRanks[transfer] >= context.rankSize ||
                plan.forwardSegmentSpans[transfer] == 0 ||
                plan.forwardSegmentBegins[transfer] +
                    plan.forwardSegmentSpans[transfer] >
                    plan.forwardSegmentCount,
            HCCL_ERROR("G2 serialized transfer is invalid"),
            HCCL_E_INTERNAL);
    }

    // 启动第一组 Kernel 前，先同步两个 Host 线程。
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(
            context.threads[0], context.threads[1], 0)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(
            context.threads[1], 0, CUSTOM_TIMEOUT)));

    // 每个流水分片复用已注册的 Kernel，仅更新偏移和传输长度参数。
    for (uint32_t chunk = 0;
        chunk < G2_PIPELINE_CHUNK_COUNT; ++chunk) {
        uint64_t sliceOffset = 0;
        uint64_t sliceCount = 0;
        FillPipelineSlice(param.count, chunk, sliceOffset, sliceCount);

        AllGatherKernelArg cross {};
        FillAllGatherArg(param, context, plan.crossChannels,
            plan.crossRanks, sliceOffset, sliceCount,
            inputToken, outputToken, 0, ALLGATHER_KERNEL_DIRECT_WRITE, cross);

        AllGatherKernelArg local {};
        FillAllGatherArg(param, context, plan.localChannels,
            plan.localRanks, sliceOffset, sliceCount,
            inputToken, outputToken, 1, ALLGATHER_KERNEL_DIRECT_WRITE, local);

        G2ForwardKernelArg forward {};
        forward.recvBuf = param.outputPtr;
        forward.sendCount = param.count;
        forward.outputToken = outputToken;
        forward.dataTypeSize = context.dataTypeSize;
        forward.forwardTransferCount = forwardCount;
        forward.channelCount =
            static_cast<uint32_t>(plan.localChannels.size());
        std::copy(plan.localChannels.begin(), plan.localChannels.end(),
            forward.channels);
        FillForwardTransfers(sliceOffset, sliceCount, plan,
            forward.forwardSliceOffsets, forward.forwardSliceCounts);
        for (uint32_t transfer = 0;
            transfer < forwardCount; ++transfer) {
            forward.forwardChannelIndices[transfer] =
                plan.forwardChannelIndices[transfer];
            forward.forwardSourceRanks[transfer] =
                plan.forwardSourceRanks[transfer];
        }

        // 先由跨组 Kernel 和本地组 Kernel 填充中间结果。
        CHK_RET(ConvertCcuToHccl(HcommCcuKernelLaunch(
            context.threads[0], plan.crossKernel, &cross, 1)));
        // 再由转发 Kernel 补齐缺失的远端 Rank 数据。
        CHK_RET(ConvertCcuToHccl(HcommCcuKernelLaunch(
            context.threads[1], plan.localKernel, &local, 1)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(
                context.threads[1], context.threads[0], 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(
                context.threads[0], 0, CUSTOM_TIMEOUT)));

        CHK_RET(ConvertCcuToHccl(HcommCcuKernelLaunch(
            context.threads[0], plan.forwardKernel, &forward, 1)));
    }
    return HCCL_SUCCESS;
}

// 执行扁平路径，将大输入切分为不超过 MAX_DATA_SIZE 的分片。
HcclResult RunFlat(const OpParam &param, const AlgResourceCtx &context,
    uint64_t inputToken, uint64_t outputToken)
{
    CHK_PRT_RET(context.threads.empty() || context.flat.kernel == 0 ||
            context.flat.channels.empty() ||
            context.flat.channels.size() != context.flat.remoteRanks.size(),
        HCCL_ERROR("Flat context is incomplete"), HCCL_E_INTERNAL);
    const uint64_t maxCount = MAX_DATA_SIZE / context.dataTypeSize;
    CHK_PRT_RET(maxCount == 0, HCCL_ERROR("Invalid flat slice size"), HCCL_E_INTERNAL);
    uint64_t offset = 0;
    while (offset < param.count) {
        // 确保每次 CCU 调用都不超过已注册的缓冲区上限。
        const uint64_t count = std::min(maxCount, param.count - offset);
        AllGatherKernelArg arg {};
        FillAllGatherArg(param, context, context.flat.channels,
            context.flat.remoteRanks, offset, count,
            inputToken, outputToken, 1, context.flat.kernelMode, arg);
        CHK_RET(ConvertCcuToHccl(HcommCcuKernelLaunch(context.threads[0],
            context.flat.kernel, &arg, 1)));
        offset += count;
    }
    return HCCL_SUCCESS;
}
} // 匿名命名空间

HcclResult ExecOp(const OpParam &param)
{
    char *raw = static_cast<char *>(param.resCtx);
    std::vector<char> serialized(raw, raw + param.ctxSize);
    AlgResourceCtx context;
    context.DeSerialize(serialized);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    const uint64_t inputBytes = param.count * context.dataTypeSize;
    const bool expectG2 = (context.rankSize == 16 ||
        context.rankSize == 12) && inputBytes > LARGE_THRESHOLD;
    CHK_PRT_RET(expectG2 && context.gateway.valid == 0,
        HCCL_ERROR("G2 was selected but not compiled"), HCCL_E_NOT_SUPPORT);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr),
        inputBytes, &inputToken);
    HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr),
        inputBytes * context.rankSize, &outputToken);
    if (expectG2) {
        return RunGateway(
            param, context, inputToken, outputToken);
    }
    return RunFlat(param, context, inputToken, outputToken);
}
} // namespace ops_hccl
