/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>

#include <algorithm>
#include <vector>

#include "log.h"
#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint32_t THREAD_NOTIFY_IDX = 0;
// 512KB 小数据继续走单阶段直写，避免两次 Kernel 启动和阶段屏障反而放大延迟。
constexpr uint64_t TWO_AXIS_MIN_TOTAL_SIZE = 8ULL * 1024 * 1024;
// 仅为 512KB 性能档启用极简拉取路径。上限留到 1MB，以覆盖 8+4
// 拓扑因 float32/rank 数不可整除产生的少量向上对齐。
constexpr uint64_t SMALL_READ_MAX_TOTAL_SIZE = 1ULL * 1024 * 1024;
constexpr uint32_t STRUCTURED_8X4_RANK_SIZE = 12;

struct GroupBroadcastGoSize {
    uint64_t addrOffset;
    uint64_t loopParam;
    uint64_t parallelParam;
    uint64_t residual;
};

constexpr uint64_t MakeBitMask(uint16_t highBit)
{
    return (uint64_t{1} << (highBit + 1)) - 1;
}

constexpr uint64_t PackParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t repeatBitHigh = 7;
    constexpr uint16_t repeatNumShift = 55;
    constexpr uint16_t repeatLoopShift = 48;
    constexpr uint16_t totalLoopShift = 41;
    return ((repeatNum & MakeBitMask(repeatBitHigh)) << repeatNumShift) |
           ((repeatLoopIndex & MakeBitMask(repeatBitHigh)) << repeatLoopShift) |
           ((totalLoopNum & MakeBitMask(repeatBitHigh)) << totalLoopShift);
}

// 将广播长度拆为：完整 LoopGroup、并行尾块和不足 32KB 的残块
GroupBroadcastGoSize CalcGroupBroadcastGoSize(uint64_t size)
{
    constexpr uint64_t loopSize =
        CUSTOM_GROUP_BROADCAST_BLOCK_SIZE * CUSTOM_GROUP_BROADCAST_LOOP_COUNT;
    constexpr uint64_t maxLoopIterNum = MakeBitMask(12);
    constexpr uint64_t maxSize = loopSize * (maxLoopIterNum + 1);

    uint64_t loopIterNum = size / loopSize;
    uint64_t parallelBlockNum = (size % loopSize) / CUSTOM_GROUP_BROADCAST_BLOCK_SIZE;
    uint64_t residual = size % CUSTOM_GROUP_BROADCAST_BLOCK_SIZE;
    if (size == maxSize) {
        loopIterNum = maxLoopIterNum;
        parallelBlockNum = CUSTOM_GROUP_BROADCAST_LOOP_COUNT - 1;
        residual = CUSTOM_GROUP_BROADCAST_BLOCK_SIZE;
    }

    uint64_t parallelParam = 0;
    if (parallelBlockNum != 0 && residual == 0) {
        parallelParam = PackParallelParam(parallelBlockNum - 1, 0, 1);
        residual = CUSTOM_GROUP_BROADCAST_BLOCK_SIZE;
    } else if (parallelBlockNum == 0 && residual != 0) {
        parallelParam = PackParallelParam(0, 0, 1);
    } else if (parallelBlockNum != 0) {
        parallelParam = PackParallelParam(parallelBlockNum - 1, 1, 2);
    }

    return GroupBroadcastGoSize{
        loopSize * loopIterNum,
        loopIterNum,
        parallelParam,
        residual,
    };
}

void AppendGroupBroadcastGoSize(std::vector<uint64_t> &taskArgs,
                                const GroupBroadcastGoSize &goSize)
{
    taskArgs.push_back(goSize.addrOffset);
    taskArgs.push_back(goSize.loopParam);
    taskArgs.push_back(goSize.parallelParam);
    taskArgs.push_back(goSize.residual);
}

// 主 thread 通知从 thread：用户 stream 上此前下发的数据准备完成
HcclResult ThreadSyncBeforeCcuKernels(const std::vector<ThreadHandle> &threads)
{
    if (threads.size() == 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(threads.size() != 2,
        HCCL_ERROR("[ThreadSyncBeforeCcuKernels] invalid thread num [%zu]", threads.size()),
        HCCL_E_INTERNAL);

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(threads[0], threads[1], THREAD_NOTIFY_IDX)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(threads[1], THREAD_NOTIFY_IDX, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

// 主 thread 等待从 thread：两个 die 上的 CCU Kernel 均完成后才允许用户 stream 继续
HcclResult ThreadSyncAfterCcuKernels(const std::vector<ThreadHandle> &threads)
{
    if (threads.size() == 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(threads.size() != 2,
        HCCL_ERROR("[ThreadSyncAfterCcuKernels] invalid thread num [%zu]", threads.size()),
        HCCL_E_INTERNAL);

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(threads[0], THREAD_NOTIFY_IDX, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(threads[1], threads[0], THREAD_NOTIFY_IDX)));
    return HCCL_SUCCESS;
}

HcclResult LaunchCcuKernels(const AlgResourceCtx &resCtx,
                            const std::vector<CcuKernelHandle> &kernels,
                            const std::vector<uint64_t> &taskArgs)
{
    CHK_PRT_RET(kernels.empty() || resCtx.threads.size() != kernels.size(),
        HCCL_ERROR("[LaunchCcuKernels] thread num [%zu] does not match kernel num [%zu]",
                   resCtx.threads.size(), kernels.size()),
        HCCL_E_INTERNAL);

    CHK_RET(ThreadSyncBeforeCcuKernels(resCtx.threads));

    // 每个 die 的 Kernel 下发到独立 thread，使两个 IO Die 上的 CCU 并行执行。
    for (uint32_t i = 0; i < kernels.size(); i++) {
        CcuResult launchRet = HcommCcuKernelLaunch(
            resCtx.threads[i], kernels[i], taskArgs.data(), taskArgs.size());
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[LaunchCcuKernels] kernel[%u] launch failed, ccuRet -> %d", i, launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    CHK_RET(ThreadSyncAfterCcuKernels(resCtx.threads));
    return HCCL_SUCCESS;
}

// 小数据由每个 rank 主动读取所有 peer 的 input 到本地 output。
// Kernel 内只需等待本地 Read 完成，因此本 rank 的 output 在返回时已经就绪。
HcclResult LaunchSmallReadKernelSlice(const AlgResourceCtx &resCtx,
                                      uint64_t inputAddr, uint64_t outputAddr,
                                      uint64_t token, uint64_t dataSize,
                                      uint64_t sliceCount,
                                      uint64_t dataTypeSize)
{
    CHK_PRT_RET(
        resCtx.smallReadCcuKernels.empty() ||
            resCtx.smallReadCcuKernels.size() != resCtx.threads.size(),
        HCCL_ERROR(
            "[LaunchSmallReadKernelSlice] thread num [%zu] does not match "
            "small kernel num [%zu]",
            resCtx.threads.size(), resCtx.smallReadCcuKernels.size()),
        HCCL_E_INTERNAL);

    const uint64_t sliceSize = sliceCount * dataTypeSize;
    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        sliceSize,
        dataSize,
    };
    return LaunchCcuKernels(
        resCtx, resCtx.smallReadCcuKernels, taskArgs);
}

// 单阶段直写路径：保持当前小数据和其他拓扑的行为不变。
HcclResult LaunchDirectKernelSlice(const AlgResourceCtx &resCtx, uint64_t inputAddr,
                                   uint64_t outputAddr, uint64_t token, uint64_t dataSize,
                                   uint32_t myRank, uint64_t sliceCount,
                                   uint64_t dataTypeSize)
{
    uint64_t sliceSize = sliceCount * dataTypeSize;
    GroupBroadcastGoSize groupBroadcastGoSize = CalcGroupBroadcastGoSize(sliceSize);

    uint64_t currentRankSliceInputOffset = 0;
    uint64_t currentRankSliceOutputOffset = dataSize * myRank;

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        currentRankSliceInputOffset,
        currentRankSliceOutputOffset,
        sliceSize,
        groupBroadcastGoSize.addrOffset,
        groupBroadcastGoSize.loopParam,
        groupBroadcastGoSize.parallelParam,
        groupBroadcastGoSize.residual,
    };

    return LaunchCcuKernels(resCtx, resCtx.ccuKernels, taskArgs);
}

bool Has2x8AxisKernels(const AlgResourceCtx &resCtx)
{
    if (resCtx.threads.size() != 2 ||
        resCtx.kernelArgs.size() != resCtx.threads.size() ||
        resCtx.ccuKernels.size() != resCtx.threads.size()) {
        return false;
    }

    bool hasMeshAxis = false;
    bool hasClosAxis = false;
    for (const auto &kernelArg : resCtx.kernelArgs) {
        if (kernelArg.axisId == CUSTOM_AXIS_MESH && kernelArg.channelCount == 7) {
            hasMeshAxis = true;
        } else if (kernelArg.axisId == CUSTOM_AXIS_CLOS && kernelArg.channelCount == 8) {
            hasClosAxis = true;
        } else {
            return false;
        }
    }
    return hasMeshAxis && hasClosAxis;
}

bool Has8x4AxisKernels(const AlgResourceCtx &resCtx)
{
    if (resCtx.threads.size() != 2 ||
        resCtx.kernelArgs.size() != resCtx.threads.size() ||
        resCtx.ccuKernels.size() != resCtx.threads.size()) {
        return false;
    }

    bool hasMeshAxis = false;
    bool hasClosAxis = false;
    for (const auto &kernelArg : resCtx.kernelArgs) {
        if (kernelArg.rankSize != STRUCTURED_8X4_RANK_SIZE) {
            return false;
        }
        const bool inLargeServer = kernelArg.rankId < 8;
        const uint32_t meshChannelCount = inLargeServer ? 7U : 3U;
        const uint32_t closChannelCount = inLargeServer ? 4U : 8U;
        if (kernelArg.axisId == CUSTOM_AXIS_MESH &&
            kernelArg.channelCount == meshChannelCount) {
            hasMeshAxis = true;
        } else if (kernelArg.axisId == CUSTOM_AXIS_CLOS &&
                   kernelArg.channelCount == closChannelCount) {
            hasClosAxis = true;
        } else {
            return false;
        }
    }
    return hasMeshAxis && hasClosAxis;
}

// 2×8 小数据继续单阶段直写，但复用双轴算法的统一 Kernel Handle。
HcclResult Launch2x8DirectKernelSlice(const AlgResourceCtx &resCtx, uint64_t inputAddr,
                                      uint64_t outputAddr, uint64_t token, uint64_t dataSize,
                                      uint32_t myRank, uint64_t sliceCount,
                                      uint64_t dataTypeSize)
{
    uint64_t sliceSize = sliceCount * dataTypeSize;
    const GroupBroadcastGoSize emptyGoSize = CalcGroupBroadcastGoSize(0);
    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        CUSTOM_AXIS_STAGE_DIRECT,
        sliceSize,
        0,
        0,
        0,
        0,
        dataSize * myRank,
        0,
    };
    for (uint32_t i = 0; i < 4; i++) {
        AppendGroupBroadcastGoSize(taskArgs, emptyGoSize);
    }

    return LaunchCcuKernels(resCtx, resCtx.ccuKernels, taskArgs);
}

// 2×8 完整 Clos 两阶段路径。将每个 slice 切为 8/22、3/22、11/22：
//   stage 1：Mesh 广播前 8/22；Clos 将前 8/22 送到远端根，并将中间 3/22 全发；
//   stage 2：Mesh 广播本端后 14/22 并转发远端根的前 8/22；
//            Clos 将最后 11/22 全发。
// 该切分与原 4/11 + 1 理论时长相同（BusBW=550），但去掉了两次四目标
// 直写，所有多目标块都只从 HBM 读取一次。
HcclResult Launch2x8AxisKernelSlice(const AlgResourceCtx &resCtx, uint64_t inputAddr,
                                    uint64_t outputAddr, uint64_t token, uint64_t dataSize,
                                    uint32_t myRank, uint64_t sliceCount,
                                    uint64_t dataTypeSize)
{
    uint64_t sliceSize = sliceCount * dataTypeSize;
    // 前 11 份按 2 个基础单元合并，不能整除 22 的元素全部并入最后一块。
    uint64_t unitSize = (sliceCount / 22) * dataTypeSize;
    uint64_t prefixSize = 8 * unitSize;
    uint64_t directSize = 3 * unitSize;
    uint64_t suffixSize = sliceSize - prefixSize;
    uint64_t bulkSize = sliceSize - prefixSize - directSize;

    GroupBroadcastGoSize prefixGoSize = CalcGroupBroadcastGoSize(prefixSize);
    GroupBroadcastGoSize suffixGoSize = CalcGroupBroadcastGoSize(suffixSize);
    GroupBroadcastGoSize directGoSize = CalcGroupBroadcastGoSize(directSize);
    GroupBroadcastGoSize bulkGoSize = CalcGroupBroadcastGoSize(bulkSize);

    constexpr uint32_t ranksPerServer = 8;
    uint32_t relayRank =
        (myRank + ranksPerServer) % (2 * ranksPerServer);

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        CUSTOM_AXIS_STAGE_1,
        sliceSize,
        directSize,
        prefixSize,
        suffixSize,
        bulkSize,
        dataSize * myRank,
        dataSize * relayRank,
    };
    AppendGroupBroadcastGoSize(taskArgs, prefixGoSize);
    AppendGroupBroadcastGoSize(taskArgs, suffixGoSize);
    AppendGroupBroadcastGoSize(taskArgs, directGoSize);
    AppendGroupBroadcastGoSize(taskArgs, bulkGoSize);

    CHK_RET(LaunchCcuKernels(resCtx, resCtx.ccuKernels, taskArgs));
    // 上一次 Launch 的主/从 thread 汇合后再启动第二阶段，HBM 是两个 CCU 间的交接面。
    taskArgs[3] = CUSTOM_AXIS_STAGE_2;
    CHK_RET(LaunchCcuKernels(resCtx, resCtx.ccuKernels, taskArgs));
    return HCCL_SUCCESS;
}

std::vector<uint64_t> Build8x4TaskArgs(uint64_t inputAddr, uint64_t outputAddr,
                                      uint64_t token, uint64_t stage,
                                      uint64_t dataSize, uint32_t myRank,
                                      uint64_t sliceCount, uint64_t dataTypeSize)
{
    uint64_t sliceSize = sliceCount * dataTypeSize;
    // CCU_SCHED AllGather 的有效 Mesh/Clos 带宽取 47/180。令 r = 180/47，
    // a 为第一阶段 Mesh 分片占比，x 为大机 rank 4~7 提前经 Clos 广播的
    // 数据占比，则两阶段临界负载分别为：
    //   T1 = max(a, (4 - 3a + 4x) / r)
    //   T2 = max(1, 4(1 - x) / r, 8(1 - a) / r)
    // 取 x = 2/47、a = 196/321 后，各阶段的 Mesh/Clos 临界负载重新平衡。
    // 不能整除的元素并入后一块，保证完整覆盖且地址连续。
    uint64_t piece0Size = ((sliceCount / 321) * 196) * dataTypeSize;
    uint64_t piece1Size = sliceSize - piece0Size;
    uint64_t earlyClosSize = ((sliceCount / 47) * 2) * dataTypeSize;
    uint64_t lateClosSize = sliceSize - earlyClosSize;
    // Direct 与两阶段路径不会在同一次调用中出现，复用第 0 组 GoSize：
    // Direct 保存完整 slice，两阶段保存 rank 4~7 的第二阶段后缀。
    GroupBroadcastGoSize directOrLateClosGoSize =
        CalcGroupBroadcastGoSize(
            stage == CUSTOM_AXIS_STAGE_DIRECT ? sliceSize : lateClosSize);
    GroupBroadcastGoSize piece0GoSize = CalcGroupBroadcastGoSize(piece0Size);
    GroupBroadcastGoSize piece1GoSize = CalcGroupBroadcastGoSize(piece1Size);
    GroupBroadcastGoSize earlyClosGoSize =
        CalcGroupBroadcastGoSize(earlyClosSize);

    uint32_t relayRank = myRank;
    if (myRank < 4) {
        relayRank = 8 + myRank;
    } else if (myRank >= 8) {
        relayRank = myRank - 8;
    }

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        stage,
        sliceSize,
        piece0Size,
        earlyClosSize,
        dataSize * myRank,
        dataSize * relayRank,
    };
    AppendGroupBroadcastGoSize(taskArgs, directOrLateClosGoSize);
    AppendGroupBroadcastGoSize(taskArgs, piece0GoSize);
    AppendGroupBroadcastGoSize(taskArgs, piece1GoSize);
    AppendGroupBroadcastGoSize(taskArgs, earlyClosGoSize);
    return taskArgs;
}

// 8+4 小数据继续单阶段直写，避免为了半块调度增加一次 Kernel 启动。
HcclResult Launch8x4DirectKernelSlice(const AlgResourceCtx &resCtx, uint64_t inputAddr,
                                      uint64_t outputAddr, uint64_t token,
                                      uint64_t dataSize, uint32_t myRank,
                                      uint64_t sliceCount, uint64_t dataTypeSize)
{
    std::vector<uint64_t> taskArgs =
        Build8x4TaskArgs(inputAddr, outputAddr, token, CUSTOM_AXIS_STAGE_DIRECT,
                         dataSize, myRank, sliceCount, dataTypeSize);
    return LaunchCcuKernels(resCtx, resCtx.ccuKernels, taskArgs);
}

// 8+4 结构化两阶段调度：
//   step 1 = 196/321：每卡 Mesh 广播 piece0；0~3 号大机卡将 piece0
//                     中继、piece1 直发；小机卡将 piece0 中继；
//                     4~7 号大机卡提前广播 2/47 的本地数据。
//   step 2 = 1：每卡 Mesh 广播 piece1，中继卡再广播收到的 piece0；
//                4~7 号大机卡广播剩余 45/47，小机卡直发 piece1。
// 该切分按 CCU_SCHED 的 47/180 有效带宽配平，不增加 Kernel 启动或
// 跨 Die 同步；每段全组发送仍只从 HBM 读入 MS 一次。
HcclResult Launch8x4StructuredKernelSlice(const AlgResourceCtx &resCtx,
                                          uint64_t inputAddr, uint64_t outputAddr,
                                          uint64_t token, uint64_t dataSize,
                                          uint32_t myRank, uint64_t sliceCount,
                                          uint64_t dataTypeSize)
{
    std::vector<uint64_t> taskArgs =
        Build8x4TaskArgs(inputAddr, outputAddr, token, CUSTOM_AXIS_STAGE_1,
                         dataSize, myRank, sliceCount, dataTypeSize);
    CHK_RET(LaunchCcuKernels(resCtx, resCtx.ccuKernels, taskArgs));
    // 两个 CCU 经 Host thread 汇合后再进入第二阶段，确保跨 Die 中继块已写回 HBM。
    taskArgs[3] = CUSTOM_AXIS_STAGE_2;
    CHK_RET(LaunchCcuKernels(resCtx, resCtx.ccuKernels, taskArgs));
    return HCCL_SUCCESS;
}

enum class KernelPath {
    SMALL_READ,
    TWO_BY_EIGHT_AXIS,
    EIGHT_BY_FOUR_STRUCTURED,
    TWO_BY_EIGHT_DIRECT,
    EIGHT_BY_FOUR_DIRECT,
    GENERIC_DIRECT,
};

KernelPath SelectKernelPath(const OpParam &param,
                            const AlgResourceCtx &resCtx,
                            uint64_t totalDataSize)
{
    if (totalDataSize <= SMALL_READ_MAX_TOTAL_SIZE &&
        resCtx.smallReadCcuKernels.size() == resCtx.threads.size()) {
        return KernelPath::SMALL_READ;
    }
    const bool has2x8Axis =
        param.rankSize == 16 && Has2x8AxisKernels(resCtx);
    if (has2x8Axis && totalDataSize >= TWO_AXIS_MIN_TOTAL_SIZE) {
        return KernelPath::TWO_BY_EIGHT_AXIS;
    }
    const bool has8x4Axis =
        param.rankSize == STRUCTURED_8X4_RANK_SIZE &&
        Has8x4AxisKernels(resCtx);
    if (has8x4Axis && totalDataSize >= TWO_AXIS_MIN_TOTAL_SIZE) {
        return KernelPath::EIGHT_BY_FOUR_STRUCTURED;
    }
    if (has2x8Axis) {
        return KernelPath::TWO_BY_EIGHT_DIRECT;
    }
    if (has8x4Axis) {
        return KernelPath::EIGHT_BY_FOUR_DIRECT;
    }
    return KernelPath::GENERIC_DIRECT;
}

HcclResult LaunchKernelSlice(
    KernelPath path, const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
    uint64_t dataSize, uint64_t sliceCount, uint64_t dataTypeSize)
{
    switch (path) {
        case KernelPath::SMALL_READ:
            return LaunchSmallReadKernelSlice(
                resCtx, inputAddr, outputAddr, token, dataSize,
                sliceCount, dataTypeSize);
        case KernelPath::TWO_BY_EIGHT_AXIS:
            return Launch2x8AxisKernelSlice(
                resCtx, inputAddr, outputAddr, token, dataSize,
                param.myRank, sliceCount, dataTypeSize);
        case KernelPath::EIGHT_BY_FOUR_STRUCTURED:
            return Launch8x4StructuredKernelSlice(
                resCtx, inputAddr, outputAddr, token, dataSize,
                param.myRank, sliceCount, dataTypeSize);
        case KernelPath::TWO_BY_EIGHT_DIRECT:
            return Launch2x8DirectKernelSlice(
                resCtx, inputAddr, outputAddr, token, dataSize,
                param.myRank, sliceCount, dataTypeSize);
        case KernelPath::EIGHT_BY_FOUR_DIRECT:
            return Launch8x4DirectKernelSlice(
                resCtx, inputAddr, outputAddr, token, dataSize,
                param.myRank, sliceCount, dataTypeSize);
        default:
            return LaunchDirectKernelSlice(
                resCtx, inputAddr, outputAddr, token, dataSize,
                param.myRank, sliceCount, dataTypeSize);
    }
}

HcclResult GetInputMemoryToken(
    const OpParam &param, uint64_t dataSize,
    uint64_t baseInputAddr, uint64_t baseOutputAddr,
    uint64_t &token)
{
    if (param.inputPtr != nullptr) {
        CHK_RET_CCU(HcommCcuGetMemToken(
            baseInputAddr, dataSize, &token));
    } else if (param.outputPtr != nullptr) {
        CHK_RET_CCU(HcommCcuGetMemToken(
            baseOutputAddr, dataSize, &token));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecuteSlices(
    const OpParam &param, const AlgResourceCtx &resCtx,
    KernelPath path, uint64_t dataSize, uint64_t dataTypeSize,
    uint64_t token, uint64_t baseInputAddr, uint64_t baseOutputAddr)
{
    const uint64_t maxDataCountPerLoop = MAX_DATA_SIZE / dataTypeSize;
    const uint64_t loopCount =
        param.count / maxDataCountPerLoop +
        static_cast<uint64_t>(param.count % maxDataCountPerLoop != 0);
    uint64_t processedDataCount = 0;
    for (uint64_t loop = 0; loop < loopCount; loop++) {
        const uint64_t sliceCount = std::min(
            maxDataCountPerLoop,
            param.count - loop * maxDataCountPerLoop);
        const uint64_t byteOffset = processedDataCount * dataTypeSize;
        CHK_RET(LaunchKernelSlice(
            path, param, resCtx, baseInputAddr + byteOffset,
            baseOutputAddr + byteOffset, token, dataSize,
            sliceCount, dataTypeSize));
        processedDataCount += sliceCount;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *context = static_cast<char *>(param.resCtx);
    std::vector<char> sequence(context, context + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(sequence);
    const uint64_t dataTypeSize = SIZE_TABLE.at(param.dataType);
    const uint64_t dataSize = param.count * dataTypeSize;
    if (param.count == 0) {
        HCCL_INFO("[ExecOp] DataCount == 0, ExecOp Run Ends.");
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(
                resCtx.threads[0], param.outputPtr,
                param.inputPtr, dataSize)));
        HCCL_INFO("[ExecOp] RankSize == 1, ExecOp Run Ends.");
        return HCCL_SUCCESS;
    }
    uint64_t token = 0;
    const uint64_t baseInputAddr =
        reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t baseOutputAddr =
        reinterpret_cast<uint64_t>(param.outputPtr);
    CHK_RET(GetInputMemoryToken(
        param, dataSize, baseInputAddr, baseOutputAddr, token));
    const KernelPath path = SelectKernelPath(
        param, resCtx, dataSize * param.rankSize);
    return ExecuteSlices(
        param, resCtx, path, dataSize, dataTypeSize, token,
        baseInputAddr, baseOutputAddr);
}
} // namespace ops_hccl
