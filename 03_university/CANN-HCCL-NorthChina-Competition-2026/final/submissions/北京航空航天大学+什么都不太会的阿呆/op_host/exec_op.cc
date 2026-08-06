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
 #include <array>
 #include <limits>
 #include <vector>
 
 #include <ccu/ccu_launch.h>
 #include <ccu/ccu_res.h>
 
 #include "custom.h"
 #include "exec_op.h"
 #include "log.h"
 
 namespace ops_hccl {
 namespace {
     constexpr uint64_t OMNIPIPE_2X8_MIN_DATA_SIZE = 16ULL * 1024 * 1024;
     constexpr uint64_t OMNIPIPE_8X4_MIN_DATA_SIZE = 16ULL * 1024 * 1024;
     constexpr uint64_t GROUP_COPY_4X1_MIN_DATA_SIZE = 16ULL * 1024 * 1024;
     constexpr uint64_t OMNIPIPE_SLICE_ALIGNMENT = 128;
     constexpr uint64_t OMNIPIPE_MESH_BANDWIDTH = 47;
     constexpr uint64_t OMNIPIPE_CLOS_BANDWIDTH = 180;
     constexpr uint32_t OMNIPIPE_STAGE_NUM = 4;
     constexpr uint64_t GROUP_COPY_MEM_SLICE = 8ULL * 4096;
     constexpr uint64_t GROUP_COPY_LOOP_COUNT = 8;
 
     struct KernelLaunchTask {
         CcuKernelHandle kernel{};
         const uint64_t *taskArgs = nullptr;
         uint32_t taskArgNum = 0;
     };
 
     struct OmniPipe2x8Slices {
         std::array<uint64_t, OMNIPIPE_STAGE_NUM> meshOffsets{};
         std::array<uint64_t, OMNIPIPE_STAGE_NUM> meshSizes{};
         std::array<uint64_t, OMNIPIPE_STAGE_NUM> closOffsets{};
         std::array<uint64_t, OMNIPIPE_STAGE_NUM> closSizes{};
     };
 
     struct OmniPipe8x4Slices {
         uint64_t localLeadSize = 0;
         uint64_t hierarchySize = 0;
         uint64_t directSize = 0;
     };
 
     uint64_t AlignDownOmniPipe(uint64_t value)
     {
         return value / OMNIPIPE_SLICE_ALIGNMENT * OMNIPIPE_SLICE_ALIGNMENT;
     }
 
     uint64_t AlignUpOmniPipe(uint64_t value)
     {
         return (value + OMNIPIPE_SLICE_ALIGNMENT - 1) / OMNIPIPE_SLICE_ALIGNMENT * OMNIPIPE_SLICE_ALIGNMENT;
     }

     constexpr uint64_t GroupCopyBitMask(uint16_t end)
     {
         return (uint64_t{1} << (end + 1)) - uint64_t{1};
     }

     constexpr uint64_t GroupCopyParallelParam(
         uint64_t repeatCount, uint64_t repeatLoopIndex, uint64_t totalLoopCount)
     {
         constexpr uint16_t repeatBitNum = 7;
         constexpr uint16_t repeatShift = 55;
         constexpr uint16_t repeatLoopBitNum = 7;
         constexpr uint16_t repeatLoopShift = 48;
         constexpr uint16_t totalLoopBitNum = 7;
         constexpr uint16_t totalLoopShift = 41;
         return ((repeatCount & GroupCopyBitMask(repeatBitNum)) << repeatShift)
                | ((repeatLoopIndex & GroupCopyBitMask(repeatLoopBitNum)) << repeatLoopShift)
                | ((totalLoopCount & GroupCopyBitMask(totalLoopBitNum)) << totalLoopShift);
     }

     std::array<uint64_t, 4> CalcGroupCopySize(uint64_t size)
     {
         const uint64_t loopSize = GROUP_COPY_LOOP_COUNT * GROUP_COPY_MEM_SLICE;
         uint64_t fullLoopCount = size / loopSize;
         uint64_t parallelLoopCount = (size - fullLoopCount * loopSize) / GROUP_COPY_MEM_SLICE;
         uint64_t residual
             = size - fullLoopCount * loopSize - parallelLoopCount * GROUP_COPY_MEM_SLICE;

         constexpr uint16_t maxLoopCountBitNum = 12;
         const uint64_t maxLoopCount = GroupCopyBitMask(maxLoopCountBitNum);
         if (size == loopSize * (maxLoopCount + 1)) {
             fullLoopCount = maxLoopCount;
             parallelLoopCount = GROUP_COPY_LOOP_COUNT - 1;
             residual = GROUP_COPY_MEM_SLICE;
         }

         uint64_t parallelParam = 0;
         uint64_t tailSize = 0;
         if (parallelLoopCount != 0 && residual == 0) {
             parallelParam = GroupCopyParallelParam(parallelLoopCount - 1, 0, 1);
             tailSize = GROUP_COPY_MEM_SLICE;
         } else if (parallelLoopCount == 0 && residual != 0) {
             parallelParam = GroupCopyParallelParam(0, 0, 1);
             tailSize = residual;
         } else if (parallelLoopCount != 0 && residual != 0) {
             parallelParam = GroupCopyParallelParam(parallelLoopCount - 1, 1, 2);
             tailSize = residual;
         }

         return {GROUP_COPY_MEM_SLICE * GROUP_COPY_LOOP_COUNT * fullLoopCount, fullLoopCount, parallelParam,
             tailSize};
     }
 
     HcclResult CalcOmniPipe2x8Slices(uint64_t loopSize, OmniPipe2x8Slices &slices)
     {
         CHK_PRT_RET(loopSize == 0 || loopSize > MAX_DATA_SIZE,
             HCCL_ERROR("Invalid 2x8 OmniPipe loop size %llu", static_cast<unsigned long long>(loopSize)), HCCL_E_PARA);
 
         // 与官方CCU OmniPipe的8x2带宽模型一致：Mesh=47，Clos=180，切成4个Stage。
         const uint64_t meshSize0 = AlignDownOmniPipe(loopSize * OMNIPIPE_MESH_BANDWIDTH / OMNIPIPE_CLOS_BANDWIDTH);
         const uint64_t closSize1 = meshSize0;
         const uint64_t meshSize1 = AlignDownOmniPipe(
             closSize1 * (ALLGATHER_PARALLEL_LOCAL_RANK_NUM - 1) * OMNIPIPE_MESH_BANDWIDTH / OMNIPIPE_CLOS_BANDWIDTH);
         CHK_PRT_RET(meshSize0 == 0 || meshSize0 + meshSize1 >= loopSize,
             HCCL_ERROR("Unable to split 2x8 OmniPipe loop size %llu", static_cast<unsigned long long>(loopSize)),
             HCCL_E_PARA);
 
         const uint64_t meshSize2 = loopSize - meshSize0 - meshSize1;
         const uint64_t closSize2 = AlignDownOmniPipe(std::min<uint64_t>(meshSize1,
             meshSize2 * OMNIPIPE_CLOS_BANDWIDTH / (OMNIPIPE_MESH_BANDWIDTH * (ALLGATHER_PARALLEL_LOCAL_RANK_NUM - 1))));
         CHK_PRT_RET(closSize2 == 0 || closSize1 + closSize2 >= loopSize,
             HCCL_ERROR("Invalid 2x8 OmniPipe middle slices"), HCCL_E_PARA);
 
         const uint64_t forwardedSize = closSize1 + closSize2;
         const uint64_t meshSize3 = AlignDownOmniPipe(
             (loopSize - forwardedSize) * (ALLGATHER_PARALLEL_LOCAL_RANK_NUM - 1) * OMNIPIPE_MESH_BANDWIDTH
             / ((ALLGATHER_PARALLEL_LOCAL_RANK_NUM - 1) * OMNIPIPE_MESH_BANDWIDTH + OMNIPIPE_CLOS_BANDWIDTH));
         const uint64_t closSize3 = loopSize - forwardedSize - meshSize3;
         CHK_PRT_RET(meshSize2 == 0 || meshSize3 == 0 || closSize3 == 0, HCCL_ERROR("Invalid 2x8 OmniPipe tail slices"),
             HCCL_E_PARA);
 
         slices.meshOffsets = {0, meshSize0, meshSize0 + meshSize1, forwardedSize};
         slices.meshSizes = {meshSize0, meshSize1, meshSize2, meshSize3};
         // Stage 0在Clos轴发送本Rank整段；后3个Stage转发其余7个本Server Rank的数据。
         slices.closOffsets = {0, 0, closSize1, forwardedSize + meshSize3};
         slices.closSizes = {loopSize, closSize1, closSize2, closSize3};
         return HCCL_SUCCESS;
     }
 
     HcclResult CalcOmniPipe8x4Slices(uint64_t dataSize, OmniPipe8x4Slices &slices)
     {
         CHK_PRT_RET(dataSize == 0 || dataSize > MAX_DATA_SIZE,
             HCCL_ERROR("Invalid 8+4 OmniPipe data size %llu", static_cast<unsigned long long>(dataSize)), HCCL_E_PARA);
 
         // 4卡Server的CLOS入口与Mesh扩散平衡点：
         // 8(1-h)+2h / 4 = 1+2h，得到h=2/7；首阶段Mesh领先量为h/2=1/7。
         slices.localLeadSize = AlignDownOmniPipe(dataSize / 7);
         slices.hierarchySize = 2 * slices.localLeadSize;
         slices.directSize = dataSize - slices.hierarchySize;
         CHK_PRT_RET(slices.localLeadSize == 0 || slices.hierarchySize >= dataSize,
             HCCL_ERROR("Unable to split 8+4 OmniPipe data size %llu", static_cast<unsigned long long>(dataSize)),
             HCCL_E_PARA);
         return HCCL_SUCCESS;
     }
 
     HcclResult LaunchCcuKernels(const OpParam &param, const AlgResourceCtx &resCtx,
         const std::vector<CcuKernelHandle> &kernels, const std::vector<uint64_t> &taskArgs)
     {
         const uint32_t taskArgNum = static_cast<uint32_t>(taskArgs.size());
         if (kernels.size() == 1) {
             CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, kernels[0], taskArgs.data(), taskArgNum));
             return HCCL_SUCCESS;
         }
 
         const ThreadHandle slaveThread = resCtx.threads[1];
         CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, slaveThread, NOTIFY_IDX_ACK)));
         CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(slaveThread, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
 
         CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, kernels[0], taskArgs.data(), taskArgNum));
         CHK_RET_CCU(HcommCcuKernelLaunch(slaveThread, kernels[1], taskArgs.data(), taskArgNum));
 
         // 主Thread等待从Thread完成，使双Die任务仍服从用户Stream的先后顺序。
         CHK_RET(
             static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
         CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(slaveThread, param.cpuThread, NOTIFY_IDX_ACK)));
         return HCCL_SUCCESS;
     }
 
     HcclResult LaunchConcurrentCcuKernels(const OpParam &param, const AlgResourceCtx &resCtx,
         const std::array<KernelLaunchTask, ALLGATHER_PARALLEL_MAX_KERNEL_NUM> &launchTasks, size_t taskCount)
     {
         CHK_PRT_RET(
             taskCount == 0 || taskCount > ALLGATHER_PARALLEL_MAX_KERNEL_NUM || resCtx.threads.size() < taskCount,
             HCCL_ERROR("Invalid concurrent launch task count %zu, thread count %zu", taskCount, resCtx.threads.size()),
             HCCL_E_PARA);
         for (size_t taskIdx = 0; taskIdx < taskCount; ++taskIdx) {
             CHK_PRT_RET(launchTasks[taskIdx].taskArgs == nullptr || launchTasks[taskIdx].taskArgNum == 0,
                 HCCL_ERROR("Invalid concurrent launch task %zu", taskIdx), HCCL_E_PARA);
         }
 
         for (size_t taskIdx = 1; taskIdx < taskCount; ++taskIdx) {
             CHK_RET(static_cast<HcclResult>(
                 HcommThreadNotifyRecordOnThread(param.cpuThread, resCtx.threads[taskIdx], NOTIFY_IDX_ACK)));
         }
         for (size_t taskIdx = 1; taskIdx < taskCount; ++taskIdx) {
             CHK_RET(static_cast<HcclResult>(
                 HcommThreadNotifyWaitOnThread(resCtx.threads[taskIdx], NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
         }
 
         CHK_RET_CCU(HcommCcuKernelLaunch(
             param.cpuThread, launchTasks[0].kernel, launchTasks[0].taskArgs, launchTasks[0].taskArgNum));
         for (size_t taskIdx = 1; taskIdx < taskCount; ++taskIdx) {
             CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[taskIdx], launchTasks[taskIdx].kernel,
                 launchTasks[taskIdx].taskArgs, launchTasks[taskIdx].taskArgNum));
         }
 
         // 每个从Thread使用主Thread上的独立Notify槽，避免多个完成信号相互覆盖。
         for (size_t taskIdx = 1; taskIdx < taskCount; ++taskIdx) {
             CHK_RET(static_cast<HcclResult>(
                 HcommThreadNotifyWaitOnThread(param.cpuThread, static_cast<uint32_t>(taskIdx - 1), CUSTOM_TIMEOUT)));
         }
         for (size_t taskIdx = 1; taskIdx < taskCount; ++taskIdx) {
             CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                 resCtx.threads[taskIdx], param.cpuThread, static_cast<uint32_t>(taskIdx - 1))));
         }
         return HCCL_SUCCESS;
     }
 
     HcclResult LaunchOmniPipe2x8Loop(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize,
         uint64_t loopOffset, uint64_t loopSize, uint64_t inputBase, uint64_t outputBase, uint64_t inputToken,
         uint64_t outputToken)
     {
         OmniPipe2x8Slices slices;
         CHK_RET(CalcOmniPipe2x8Slices(loopSize, slices));
 
         const uint64_t ownBlockOffset = dataSize * param.myRank + loopOffset;
         const uint64_t loopInput = inputBase + loopOffset;
         const uint64_t skipLocalCopy = loopInput == outputBase + ownBlockOffset ? uint64_t{1} : uint64_t{0};
 
         // Stage 0：Clos发送本Rank整段并完成本地拷贝；Mesh同时发送首个均衡片。
         const std::array<uint64_t, 7> stage0IntraArgs
             = {loopInput, outputBase, inputToken, outputToken, ownBlockOffset, slices.meshSizes[0], uint64_t{1}};
         const std::array<uint64_t, 7> stage0InterArgs
             = {loopInput, outputBase, inputToken, outputToken, ownBlockOffset, loopSize, skipLocalCopy};
         std::array<KernelLaunchTask, ALLGATHER_PARALLEL_MAX_KERNEL_NUM> stage0Tasks{};
         size_t stage0TaskCount = 0;
         for (CcuKernelHandle kernel : resCtx.parallelPhase1IntraCcuKernels) {
             stage0Tasks[stage0TaskCount++]
                 = KernelLaunchTask{kernel, stage0IntraArgs.data(), static_cast<uint32_t>(stage0IntraArgs.size())};
         }
         for (CcuKernelHandle kernel : resCtx.parallelPhase1InterCcuKernels) {
             stage0Tasks[stage0TaskCount++]
                 = KernelLaunchTask{kernel, stage0InterArgs.data(), static_cast<uint32_t>(stage0InterArgs.size())};
         }
         CHK_RET(LaunchConcurrentCcuKernels(param, resCtx, stage0Tasks, stage0TaskCount));
 
         // Stage 1/2：Mesh继续发送本Rank后续片，Clos转发已由Mesh收齐的7个块。
         // Stage 3：Mesh扩散Stage 0收到的对端Rank中间片，Clos转发剩余7个尾片。
         for (uint32_t stage = 1; stage < OMNIPIPE_STAGE_NUM; ++stage) {
             uint64_t meshInput = inputBase + loopOffset + slices.meshOffsets[stage];
             uint64_t meshInputToken = inputToken;
             uint64_t meshOutputOffset = ownBlockOffset + slices.meshOffsets[stage];
             if (stage + 1 == OMNIPIPE_STAGE_NUM) {
                 meshOutputOffset = dataSize * resCtx.parallelCounterpartRank + loopOffset + slices.meshOffsets[stage];
                 meshInput = outputBase + meshOutputOffset;
                 meshInputToken = outputToken;
             }
             const std::array<uint64_t, 7> intraArgs = {meshInput, outputBase, meshInputToken, outputToken,
                 meshOutputOffset, slices.meshSizes[stage], uint64_t{1}};
 
             std::array<uint64_t, 3 + ALLGATHER_PARALLEL_LOCAL_RANK_NUM - 1> interArgs{};
             interArgs[0] = outputBase;
             interArgs[1] = outputToken;
             interArgs[2] = slices.closSizes[stage];
             size_t blockIdx = 0;
             for (uint32_t localRank : resCtx.parallelLocalRanks) {
                 if (localRank == param.myRank) {
                     continue;
                 }
                 interArgs[3 + blockIdx++] = dataSize * localRank + loopOffset + slices.closOffsets[stage];
             }
             CHK_PRT_RET(blockIdx != ALLGATHER_PARALLEL_LOCAL_RANK_NUM - 1,
                 HCCL_ERROR("Invalid 2x8 OmniPipe local block count %zu", blockIdx), HCCL_E_PARA);
 
             std::array<KernelLaunchTask, ALLGATHER_PARALLEL_MAX_KERNEL_NUM> stageTasks{};
             size_t stageTaskCount = 0;
             // 慢轴优先下发，给Mesh一个很小的启动领先量。
             for (CcuKernelHandle kernel : resCtx.parallelPhase2IntraCcuKernels) {
                 stageTasks[stageTaskCount++]
                     = KernelLaunchTask{kernel, intraArgs.data(), static_cast<uint32_t>(intraArgs.size())};
             }
             for (CcuKernelHandle kernel : resCtx.parallelPhase2InterCcuKernels) {
                 stageTasks[stageTaskCount++]
                     = KernelLaunchTask{kernel, interArgs.data(), static_cast<uint32_t>(interArgs.size())};
             }
             CHK_RET(LaunchConcurrentCcuKernels(param, resCtx, stageTasks, stageTaskCount));
         }
         return HCCL_SUCCESS;
     }
 
     HcclResult LaunchOmniPipe2x8(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize,
         uint64_t inputBase, uint64_t outputBase, uint64_t inputToken, uint64_t outputToken)
     {
         const uint64_t loopCount = (dataSize + MAX_DATA_SIZE - 1) / MAX_DATA_SIZE;
         CHK_PRT_RET(loopCount == 0 || loopCount > 2,
             HCCL_ERROR("Invalid 2x8 OmniPipe loop count %llu", static_cast<unsigned long long>(loopCount)),
             HCCL_E_PARA);
         const uint64_t regularLoopSize
             = loopCount == 1 ? dataSize : AlignUpOmniPipe((dataSize + loopCount - 1) / loopCount);
         CHK_PRT_RET(regularLoopSize == 0 || regularLoopSize > MAX_DATA_SIZE,
             HCCL_ERROR("Invalid 2x8 OmniPipe regular loop size %llu", static_cast<unsigned long long>(regularLoopSize)),
             HCCL_E_PARA);
 
         uint64_t loopOffset = 0;
         for (uint64_t loop = 0; loop < loopCount; ++loop) {
             const uint64_t loopSize = loop + 1 == loopCount ? dataSize - loopOffset : regularLoopSize;
             CHK_RET(LaunchOmniPipe2x8Loop(
                 param, resCtx, dataSize, loopOffset, loopSize, inputBase, outputBase, inputToken, outputToken));
             loopOffset += loopSize;
         }
         return HCCL_SUCCESS;
     }
 
     HcclResult LaunchOmniPipe8x4(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize,
         uint64_t inputBase, uint64_t outputBase, uint64_t inputToken, uint64_t outputToken)
     {
         OmniPipe8x4Slices slices;
         CHK_RET(CalcOmniPipe8x4Slices(dataSize, slices));
 
         const uint64_t ownBlockOffset = dataSize * param.myRank;
         const uint64_t skipLocalCopy = inputBase == outputBase + ownBlockOffset ? uint64_t{1} : uint64_t{0};
 
         // Stage 0：独立Kernel并发执行1/7本Server扩散与2/7非对称网关传输。
         const std::array<uint64_t, 7> stage0IntraArgs
             = {inputBase, outputBase, inputToken, outputToken, ownBlockOffset, slices.localLeadSize, uint64_t{1}};
         const std::array<uint64_t, 7> stage0InterArgs = {
             inputBase, outputBase, inputToken, outputToken, ownBlockOffset, slices.hierarchySize, slices.localLeadSize};
         std::array<KernelLaunchTask, ALLGATHER_PARALLEL_MAX_KERNEL_NUM> stage0Tasks{};
         size_t stage0TaskCount = 0;
         for (CcuKernelHandle kernel : resCtx.parallelPhase1IntraCcuKernels) {
             stage0Tasks[stage0TaskCount++]
                 = KernelLaunchTask{kernel, stage0IntraArgs.data(), static_cast<uint32_t>(stage0IntraArgs.size())};
         }
         for (CcuKernelHandle kernel : resCtx.parallelPhase1InterCcuKernels) {
             stage0Tasks[stage0TaskCount++]
                 = KernelLaunchTask{kernel, stage0InterArgs.data(), static_cast<uint32_t>(stage0InterArgs.size())};
         }
         CHK_RET(LaunchConcurrentCcuKernels(param, resCtx, stage0Tasks, stage0TaskCount));
 
         // Stage 1 Mesh参数：本Rank剩余6/7从input读取，网关块从output读取，
         // 本Rank全量本地拷贝也在同一Kernel内与网络Write并行。
         std::array<uint64_t, 12> stage1IntraArgs{};
         stage1IntraArgs[0] = inputBase;
         stage1IntraArgs[1] = outputBase;
         stage1IntraArgs[2] = inputToken;
         stage1IntraArgs[3] = outputToken;
         stage1IntraArgs[4] = ownBlockOffset;
         stage1IntraArgs[5] = slices.localLeadSize;
         stage1IntraArgs[6] = dataSize - slices.localLeadSize;
         stage1IntraArgs[7] = dataSize;
         stage1IntraArgs[8] = skipLocalCopy;
 
         uint32_t forwardBlockCount = 0;
         if (param.myRank < ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM) {
             const uint32_t pairedSmallRank = ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM + param.myRank / 2;
             stage1IntraArgs[9] = slices.localLeadSize;
             stage1IntraArgs[10] = dataSize * pairedSmallRank + (param.myRank & 1U) * slices.localLeadSize;
             forwardBlockCount = 1;
         } else {
             const uint32_t firstLargeRank = 2 * (param.myRank - ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM);
             stage1IntraArgs[9] = slices.hierarchySize;
             stage1IntraArgs[10] = dataSize * firstLargeRank;
             stage1IntraArgs[11] = dataSize * (firstLargeRank + 1);
             forwardBlockCount = 2;
         }
 
         const std::array<uint64_t, 7> stage1InterArgs = {inputBase + slices.hierarchySize, outputBase, inputToken,
             outputToken, ownBlockOffset + slices.hierarchySize, slices.directSize, uint64_t{1}};
         std::array<KernelLaunchTask, ALLGATHER_PARALLEL_MAX_KERNEL_NUM> stage1Tasks{};
         size_t stage1TaskCount = 0;
         for (CcuKernelHandle kernel : resCtx.parallelPhase2IntraCcuKernels) {
             stage1Tasks[stage1TaskCount++]
                 = KernelLaunchTask{kernel, stage1IntraArgs.data(), static_cast<uint32_t>(10 + forwardBlockCount)};
         }
         for (CcuKernelHandle kernel : resCtx.parallelPhase2InterCcuKernels) {
             stage1Tasks[stage1TaskCount++]
                 = KernelLaunchTask{kernel, stage1InterArgs.data(), static_cast<uint32_t>(stage1InterArgs.size())};
         }
         return LaunchConcurrentCcuKernels(param, resCtx, stage1Tasks, stage1TaskCount);
     }
 } // namespace
 
 HcclResult ExecOp(const OpParam &param)
 {
     CHK_PTR_NULL(param.resCtx);
     CHK_PRT_RET(param.ctxSize == 0 || param.ctxSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
         HCCL_ERROR("Invalid resource context size %llu", static_cast<unsigned long long>(param.ctxSize)), HCCL_E_PARA);
 
     // 反序列化
     char *ctx = static_cast<char *>(param.resCtx);
     std::vector<char> seq(ctx, ctx + param.ctxSize);
     AlgResourceCtx resCtx;
     resCtx.DeSerialize(seq);
 
     auto typeSize = SIZE_TABLE.find(param.dataType);
     CHK_PRT_RET(typeSize == SIZE_TABLE.end(),
         HCCL_ERROR("Unsupported data type %u", static_cast<uint32_t>(param.dataType)), HCCL_E_NOT_SUPPORT);
     CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize->second,
         HCCL_ERROR("Input size overflow"), HCCL_E_PARA);
 
     const uint64_t dataSize = param.count * typeSize->second;
     if (dataSize == 0) {
         return HCCL_SUCCESS;
     }
 
     if (param.rankSize == 1) {
         if (param.inputPtr == param.outputPtr) {
             return HCCL_SUCCESS;
         }
         CHK_RET(HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, dataSize));
         return HCCL_SUCCESS;
     }
 
     const size_t parallelPhase1KernelCount
         = resCtx.parallelPhase1IntraCcuKernels.size() + resCtx.parallelPhase1InterCcuKernels.size();
     const size_t parallelPhase2KernelCount
         = resCtx.parallelPhase2IntraCcuKernels.size() + resCtx.parallelPhase2InterCcuKernels.size();
     const bool parallelResourcesPresent
         = !resCtx.parallelPhase1IntraCcuKernels.empty() || !resCtx.parallelPhase1InterCcuKernels.empty()
           || !resCtx.parallelPhase2IntraCcuKernels.empty() || !resCtx.parallelPhase2InterCcuKernels.empty();
     const bool invalidParallelResources
         = resCtx.parallel2x8Enabled > 1 || resCtx.asym8x4Enabled > 1
           || (resCtx.parallel2x8Enabled != 0 && resCtx.asym8x4Enabled != 0)
           || (resCtx.parallel2x8Enabled == 0 && resCtx.asym8x4Enabled == 0
               && (parallelResourcesPresent || resCtx.parallelInterChannelCount != 0
                   || !resCtx.parallelLocalRanks.empty()))
           || (resCtx.parallel2x8Enabled != 0
               && (param.rankSize != 2 * ALLGATHER_PARALLEL_LOCAL_RANK_NUM
                   || resCtx.parallelCounterpartRank >= param.rankSize || resCtx.parallelInterChannelCount != 1
                   || resCtx.parallelLocalRanks.size() != ALLGATHER_PARALLEL_LOCAL_RANK_NUM
                   || resCtx.parallelPhase1IntraCcuKernels.empty() || resCtx.parallelPhase1IntraCcuKernels.size() > 2
                   || resCtx.parallelPhase1InterCcuKernels.size() != 1
                   || resCtx.parallelPhase2IntraCcuKernels.size() != resCtx.parallelPhase1IntraCcuKernels.size()
                   || resCtx.parallelPhase2InterCcuKernels.size() != 1
                   || parallelPhase1KernelCount > ALLGATHER_PARALLEL_MAX_KERNEL_NUM
                   || parallelPhase2KernelCount > ALLGATHER_PARALLEL_MAX_KERNEL_NUM
                   || resCtx.threads.size() < std::max(parallelPhase1KernelCount, parallelPhase2KernelCount)))
           || (resCtx.asym8x4Enabled != 0
               && (param.rankSize != ALLGATHER_ASYM_8X4_RANK_NUM || resCtx.parallelInterChannelCount != 0
                   || !resCtx.parallelLocalRanks.empty() || resCtx.parallelPhase1IntraCcuKernels.empty()
                   || resCtx.parallelPhase1InterCcuKernels.empty() || resCtx.parallelPhase2IntraCcuKernels.empty()
                   || resCtx.parallelPhase2InterCcuKernels.empty() || resCtx.parallelPhase1IntraCcuKernels.size() > 2
                   || resCtx.parallelPhase1InterCcuKernels.size() > 2 || resCtx.parallelPhase2IntraCcuKernels.size() > 2
                   || resCtx.parallelPhase2InterCcuKernels.size() > 2
                   || parallelPhase1KernelCount > ALLGATHER_PARALLEL_MAX_KERNEL_NUM
                   || parallelPhase2KernelCount > ALLGATHER_PARALLEL_MAX_KERNEL_NUM
                   || resCtx.threads.size() < std::max(parallelPhase1KernelCount, parallelPhase2KernelCount)));
 
     CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize
                     || resCtx.ccuKernels.empty() || resCtx.ccuKernels.size() > 2
                     || resCtx.dualSliceCcuKernels.size() != resCtx.ccuKernels.size()
                     || resCtx.threads.size() < resCtx.ccuKernels.size() || invalidParallelResources,
         HCCL_ERROR("Invalid launch resources: rank %u/%u, kernel count %zu", param.myRank, param.rankSize,
             resCtx.ccuKernels.size()),
         HCCL_E_PARA);
     if (resCtx.parallel2x8Enabled != 0) {
         uint16_t localRankMask = 0;
         for (uint32_t localRank : resCtx.parallelLocalRanks) {
             CHK_PRT_RET(localRank >= param.rankSize || localRank == resCtx.parallelCounterpartRank,
                 HCCL_ERROR("Invalid 2x8 local rank %u", localRank), HCCL_E_PARA);
             const uint16_t localRankBit = static_cast<uint16_t>(uint32_t{1} << localRank);
             CHK_PRT_RET(
                 (localRankMask & localRankBit) != 0, HCCL_ERROR("Duplicate 2x8 local rank %u", localRank), HCCL_E_PARA);
             localRankMask = static_cast<uint16_t>(localRankMask | localRankBit);
         }
         CHK_PRT_RET((localRankMask & static_cast<uint16_t>(uint32_t{1} << param.myRank)) == 0,
             HCCL_ERROR("2x8 local rank set does not contain rank %u", param.myRank), HCCL_E_PARA);
     }
     CHK_PRT_RET(dataSize > std::numeric_limits<uint64_t>::max() / param.rankSize, HCCL_ERROR("Output size overflow"),
         HCCL_E_PARA);
 
     const uint64_t outputSize = dataSize * param.rankSize;
     const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
     const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
 
     // 输入和输出可能来自不同的注册内存区域，分别获取Token。
     uint64_t inputToken = 0;
     uint64_t outputToken = 0;
     CHK_RET_CCU(HcommCcuGetMemToken(inputBase, dataSize, &inputToken));
     CHK_RET_CCU(HcommCcuGetMemToken(outputBase, outputSize, &outputToken));
 
     // 拓扑专用OmniPipe仅用于对应大包；小包继续走已验证的Direct路径。
     if (resCtx.parallel2x8Enabled != 0 && dataSize >= OMNIPIPE_2X8_MIN_DATA_SIZE && dataSize <= 2ULL * MAX_DATA_SIZE) {
         return LaunchOmniPipe2x8(param, resCtx, dataSize, inputBase, outputBase, inputToken, outputToken);
     }
     if (resCtx.asym8x4Enabled != 0 && dataSize >= OMNIPIPE_8X4_MIN_DATA_SIZE && dataSize <= MAX_DATA_SIZE) {
         return LaunchOmniPipe8x4(param, resCtx, dataSize, inputBase, outputBase, inputToken, outputToken);
     }

     if (param.rankSize == 4 && dataSize >= GROUP_COPY_4X1_MIN_DATA_SIZE && dataSize <= MAX_DATA_SIZE) {
         const uint64_t currentRankOutputOffset = dataSize * param.myRank;
         const uint64_t skipLocalCopy
             = inputBase == outputBase + currentRankOutputOffset ? uint64_t{1} : uint64_t{0};
         const std::array<uint64_t, 4> groupCopySize = CalcGroupCopySize(dataSize);
         const std::vector<uint64_t> taskArgs = {inputBase, outputBase, inputToken, outputToken,
             currentRankOutputOffset, dataSize, skipLocalCopy, groupCopySize[0], groupCopySize[1],
             groupCopySize[2], groupCopySize[3]};
         return LaunchCcuKernels(param, resCtx, resCtx.groupCopyCcuKernels, taskArgs);
     }

     const uint64_t currentRankOutputOffset = dataSize * param.myRank;
     uint64_t processedSize = 0;
 
     // 每次融合两个不超过256MiB的切片。
     // 小于等于256MiB的数据继续使用原Kernel，不增加额外参数或分支。
     while (dataSize - processedSize > MAX_DATA_SIZE) {
         const uint64_t firstSliceSize = MAX_DATA_SIZE;
         const uint64_t secondSliceSize = std::min<uint64_t>(MAX_DATA_SIZE, dataSize - processedSize - firstSliceSize);
         const uint64_t inputAddr = inputBase + processedSize;
         const uint64_t outputAddr = outputBase + processedSize;
         const uint64_t skipLocalCopy = inputAddr == outputAddr + currentRankOutputOffset ? uint64_t{1} : uint64_t{0};
 
         // 顺序必须与CcuDualSliceKernel中的LoadArg顺序保持一致。
         std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, inputToken, outputToken, currentRankOutputOffset,
             firstSliceSize, secondSliceSize, skipLocalCopy};
         CHK_RET(LaunchCcuKernels(param, resCtx, resCtx.dualSliceCcuKernels, taskArgs));
 
         processedSize += firstSliceSize + secondSliceSize;
     }
 
     if (processedSize < dataSize) {
         const uint64_t sliceSize = dataSize - processedSize;
         const uint64_t inputAddr = inputBase + processedSize;
         const uint64_t outputAddr = outputBase + processedSize;
         const uint64_t skipLocalCopy = inputAddr == outputAddr + currentRankOutputOffset ? uint64_t{1} : uint64_t{0};
 
         // 顺序必须与CcuKernel中的LoadArg顺序保持一致。
         std::vector<uint64_t> taskArgs
             = {inputAddr, outputAddr, inputToken, outputToken, currentRankOutputOffset, sliceSize, skipLocalCopy};
         CHK_RET(LaunchCcuKernels(param, resCtx, resCtx.ccuKernels, taskArgs));
     }
 
     return HCCL_SUCCESS;
 }
 } // namespace ops_hccl
