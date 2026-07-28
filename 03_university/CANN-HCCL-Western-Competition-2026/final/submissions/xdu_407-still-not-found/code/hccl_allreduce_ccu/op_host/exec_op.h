/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_EXEC_OP_H
#define OPS_HCCL_CCU_EXEC_OP_H

#include <array>
#include <vector>

#include <hccl/hcomm_primitives.h>
#include "common.h"
#include "custom.h"

namespace ops_hccl {
HcclResult ClassifyTopology(uint32_t rankSize, TopologyKind &topology);
HcclResult ValidateTopology(uint32_t rankSize, TopologyKind topology, const CanonicalRankGroupMap &rankMap);
HcclResult BuildCanonicalRankGroupMap(TopologyKind topology, uint32_t rankSize, CanonicalRankGroupMap &rankMap);
HcclResult SelectExactPlan(
    AlgorithmPackage package, TopologyKind topology, MessageSizeKind messageSize, PlanSpec &plan);
HcclResult BuildDeterministicShards(const PlanSpec &plan, uint64_t elementCount,
    std::array<DeterministicShard, MAX_LOGICAL_SHARDS> &shards, uint32_t &shardCount);
HcclResult BuildScratchLayout(const PlanSpec &plan, const std::array<DeterministicShard, MAX_LOGICAL_SHARDS> &shards,
    uint32_t shardCount, std::array<ScratchLayoutRecord, MAX_SCRATCH_LAYOUT_RECORDS> &scratchRecords,
    uint32_t &scratchRecordCount);
HcclResult BuildStageSpec(const PlanSpec &plan, uint32_t stageOffset, uint32_t worker, uint32_t generation,
    FabricLayer layer, DieEndpoint die, const CanonicalRankGroupMap &rankMap,
    const std::array<PeerChannelResource, MAX_PEER_CHANNELS> &peerChannels, uint32_t peerChannelCount,
    StageSpec &stage);
HcclResult AuditStageSpec(const StageSpec &stage);
HcclResult BuildTaskArgs(
    const InvocationMemory &memory, const InvocationSlice &slice, const InvocationControl &control, TaskArgs &taskArgs);
HcclResult ScheduleDirect12(const OpParam &param, const AlgResourceCtx &resourceContext);
HcclResult ScheduleFullExchange(const OpParam &param, const AlgResourceCtx &resourceContext);
HcclResult ScheduleFourXOnePullFullExchange(const OpParam &param, const AlgResourceCtx &resourceContext);
HcclResult ScheduleFourXOneRing(const OpParam &param, const AlgResourceCtx &resourceContext);
HcclResult ScheduleFourXOneRingTail(const OpParam &param, const AlgResourceCtx &resourceContext);
HcclResult ScheduleTwoXEightMultiRoot(const OpParam &param, const AlgResourceCtx &resourceContext);
HcclResult ScheduleTwoXEightBulk(const OpParam &param, const AlgResourceCtx &resourceContext);
HcclResult ScheduleTwoXEightBulkTail(const OpParam &param, const AlgResourceCtx &resourceContext);

// Temporary declarations consumed by the current monolithic scaffold sources.
HcclResult SelectAlgorithmPlan(uint32_t rankSize, uint64_t dataBytes, AlgorithmPlan &plan);
HcclResult GetResourceProfile(ResourceProfileId profileId, ResourceProfile &profile);
HcclResult BuildExecutionSlices(uint64_t dataBytes, uint64_t maxSliceBytes, std::vector<ExecutionSlice> &slices);
const char *GetTopologyName(TopologyKind topology);
const char *GetPlanName(PlanId planId);
const char *GetKernelName(CcuKernelId kernelId);

// 执行算法任务编排
HcclResult ExecOp(const OpParam &param);
} // namespace ops_hccl
#endif // OPS_HCCL_CCU_EXEC_OP_H
