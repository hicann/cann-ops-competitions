/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef OPS_HCCL_CCU_KERNEL_H
 #define OPS_HCCL_CCU_KERNEL_H
 
 #include <map>
 #include <memory>
 #include <string>
 #include <vector>
 
 #include <ccu/ccu_types.h>
 
 #include "custom.h"
 
 namespace ccu = ::AscendC::ccu;
 
 namespace ops_hccl {
 
 // 注册Kernel时固定的拓扑参数。同一个Kernel中的channels全部属于同一个Die。
 struct CcuKernelArgAllGather : public CcuKernelArgBase {
     uint32_t rankSize = 0;
     uint32_t rankId = 0;
     uint32_t peerRanks[MAX_RANK_SIZE] = {};
     uint32_t handleLocalCopy = 0;
     uint32_t blockCount = 0;
     uint32_t stripeBaseIndex = 0;
     uint32_t stripeCount = 0;
 };
 
 // Kernel运行期参数和CCU资源。
 struct CcuAllGatherContext {
     const CcuKernelArgAllGather *arg = nullptr;
 
     ccu::Variable input;
     ccu::Variable inputToken;
     ccu::Variable localOutput;
     ccu::Variable localOutputToken;
     std::vector<ccu::Variable> peerOutputs;
     std::vector<ccu::Variable> peerOutputTokens;
     ccu::Variable currentRankOutputOffset;
     ccu::Variable sliceSize;
     ccu::Variable skipLocalCopy;
     ccu::Event completionEvent;
 };

 // 4x1大包专用：使用官方Mesh1D Direct示例的LoopGroup完成本地搬运。
 constexpr uint64_t GROUP_COPY_MS_INTERLEAVE = 8;
 constexpr uint64_t GROUP_COPY_MS_SIZE = 4096;
 constexpr uint32_t GROUP_COPY_MS_PER_LOOP = 8;
 constexpr uint32_t GROUP_COPY_LOOP_COUNT = 8;

 struct GroupCopyLoopConfig {
     uint32_t msInterleave = 0;
     uint32_t loopCount = 0;
     uint64_t memSlice = 0;
 };

 struct GroupCopyLoopResource {
     ccu::Array<ccu::Event> completedEvents{0};
     ccu::Array<ccu::CcuBuffer> buffers{0};
     uint32_t eventCount = 0;
     uint32_t bufferCount = 0;
 };

 struct GroupCopySizeVars {
     ccu::Variable addrOffset;
     ccu::Variable loopParam;
     ccu::Variable parallelParam;
     ccu::Variable residual;
 };

 struct GroupCopyLoopEntity {
     std::unique_ptr<ccu::Func> bodies[2];
     std::unique_ptr<ccu::Loop> loops[2];
     ccu::Variable loopParams[2];
 };

 struct CcuGroupCopyAllGatherContext : public CcuAllGatherContext {
     GroupCopySizeVars groupCopySize;
     GroupCopyLoopConfig loopConfig;
     GroupCopyLoopResource loopResource;
     bool resourceAllocated = false;
     std::map<std::string, GroupCopyLoopEntity> loopEntities;
 };
 
 // 大包融合路径在同一次Kernel中下发两个不超过MAX_DATA_SIZE的切片。
 struct CcuAllGatherDualSliceContext : public CcuAllGatherContext {
     ccu::Variable secondSliceSize;
     ccu::Event secondCompletionEvent;
 };
 
 // 2x8并行算法第二阶段：把本端已经持有的多个数据块发送给同一通信范围内的所有Peer。
 struct CcuParallelRepeatContext {
     const CcuKernelArgAllGather *arg = nullptr;
 
     ccu::Variable localOutput;
     ccu::Variable localOutputToken;
     std::vector<ccu::Variable> peerOutputs;
     std::vector<ccu::Variable> peerOutputTokens;
     ccu::Variable sliceSize;
     std::vector<ccu::Variable> blockOffsets;
     std::vector<ccu::Event> completionEvents;
 };
 
 // 2x8并行算法的跨Server第一阶段：同一对端的多个物理Channel各发送一个不重叠的数据条带。
 struct CcuStripedAllGatherContext {
     const CcuKernelArgAllGather *arg = nullptr;
 
     ccu::Variable input;
     ccu::Variable inputToken;
     ccu::Variable localOutput;
     ccu::Variable localOutputToken;
     std::vector<ccu::Variable> peerOutputs;
     std::vector<ccu::Variable> peerOutputTokens;
     ccu::Variable currentRankOutputOffset;
     ccu::Variable stripeSize;
     ccu::Variable lastStripeSize;
     ccu::Variable skipLocalCopy;
     ccu::Event completionEvent;
     ccu::Event localCompletionEvent;
 };
 
 // 2x8并行算法的跨Server第二阶段：多个物理Channel并行转发每个已聚合数据块的不同条带。
 struct CcuParallelStripedRepeatContext {
     const CcuKernelArgAllGather *arg = nullptr;
 
     ccu::Variable localOutput;
     ccu::Variable localOutputToken;
     std::vector<ccu::Variable> peerOutputs;
     std::vector<ccu::Variable> peerOutputTokens;
     ccu::Variable stripeSize;
     ccu::Variable lastStripeSize;
     std::vector<ccu::Variable> blockOffsets;
     std::vector<ccu::Event> completionEvents;
 };
 
 // 8+4阶段0：大Server Rank发送2/7前缀到一个小Server网关；
 // 小Server Rank把2/7前缀的两个1/7半片分别发送到两个大Server网关。
 struct CcuAsym8x4Stage0InterContext {
     const CcuKernelArgAllGather *arg = nullptr;
 
     ccu::Variable input;
     ccu::Variable inputToken;
     ccu::Variable localOutput;
     ccu::Variable localOutputToken;
     std::vector<ccu::Variable> peerOutputs;
     std::vector<ccu::Variable> peerOutputTokens;
     ccu::Variable currentRankOutputOffset;
     ccu::Variable hierarchySize;
     ccu::Variable halfSize;
     ccu::Event completionEvent;
 };
 
 // 8+4阶段1：发送本Rank剩余数据、扩散阶段0收到的网关块，并并行完成本地拷贝。
 struct CcuAsym8x4Stage1IntraContext {
     const CcuKernelArgAllGather *arg = nullptr;
 
     ccu::Variable input;
     ccu::Variable inputToken;
     ccu::Variable localOutput;
     ccu::Variable localOutputToken;
     std::vector<ccu::Variable> peerOutputs;
     std::vector<ccu::Variable> peerOutputTokens;
     ccu::Variable currentRankOutputOffset;
     ccu::Variable localLeadSize;
     ccu::Variable localRemainingSize;
     ccu::Variable totalSize;
     ccu::Variable skipLocalCopy;
     ccu::Variable forwardSize;
     std::vector<ccu::Variable> forwardBlockOffsets;
     ccu::Event localDataCompletionEvent;
     std::vector<ccu::Event> forwardCompletionEvents;
 };
 
 // CCU Kernel 函数
 CcuResult CcuKernel(CcuKernelArg arg);
 CcuResult CcuGroupCopyKernel(CcuKernelArg arg);
 CcuResult CcuDualSliceKernel(CcuKernelArg arg);
 CcuResult CcuParallelRepeatKernel(CcuKernelArg arg);
 CcuResult CcuStripedAllGatherKernel(CcuKernelArg arg);
 CcuResult CcuParallelStripedRepeatKernel(CcuKernelArg arg);
 CcuResult CcuAsym8x4Stage0InterKernel(CcuKernelArg arg);
 CcuResult CcuAsym8x4Stage1IntraKernel(CcuKernelArg arg);
 
 } // namespace ops_hccl
 
 #endif // OPS_HCCL_CCU_KERNEL_H
