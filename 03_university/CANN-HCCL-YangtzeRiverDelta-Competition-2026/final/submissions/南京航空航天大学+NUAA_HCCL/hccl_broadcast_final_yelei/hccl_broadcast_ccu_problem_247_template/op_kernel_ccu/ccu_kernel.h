/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

#include "custom.h"

namespace ops_hccl {

struct CcuKernelArgBroadcast : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t root = 0;
    uint32_t dieId = 0;
    uint32_t notifyIndex = 0;
    uint32_t topologyMode = F2_TOPO_FLAT_FOUR;
    uint32_t peerRanks[MAX_RANK_SIZE] = {};

    uint64_t offset = 0;
    uint64_t dataSize = 0;

    uint32_t ownerCount = 0;
    uint32_t sourceOwners[MAX_RANK_SIZE] = {};
    uint32_t destinationOwners[MAX_RANK_SIZE] = {};
    uint64_t stripeOffsets[MAX_RANK_SIZE] = {};
    uint64_t stripeSizes[MAX_RANK_SIZE] = {};
    uint64_t chunkOffsets[F2_PATH_COUNT][F2_CHUNK_COUNT] = {};
    uint64_t chunkSizes[F2_PATH_COUNT][F2_CHUNK_COUNT] = {};

    uint32_t rootServerSize = 0;
    uint32_t rootServerRanks[MAX_RANK_SIZE] = {};
    uint32_t remoteServerSize = 0;
    uint32_t remoteServerRanks[MAX_RANK_SIZE] = {};

    uint32_t pipelineRootServerSize = 0;
    uint32_t pipelineRootServerRanks[MAX_RANK_SIZE] = {};
    uint32_t pipelineRelayServerSize = 0;
    uint32_t pipelineRelayServerRanks[MAX_RANK_SIZE] = {};
    uint64_t pipelineOffsets[F2_PIPELINE_MAX_RELAY_COUNT][F2_PIPELINE_CHUNK_COUNT] = {};
    uint64_t pipelineSizes[F2_PIPELINE_MAX_RELAY_COUNT][F2_PIPELINE_CHUNK_COUNT] = {};

    uint32_t forestTreeCount = 0;
    uint32_t forestActive[F4_FOREST_MAX_TREE_COUNT] = {};
    uint32_t forestParents[F4_FOREST_MAX_TREE_COUNT] = {};
    uint32_t forestChildCounts[F4_FOREST_MAX_TREE_COUNT] = {};
    uint32_t forestChildren[F4_FOREST_MAX_TREE_COUNT][MAX_RANK_SIZE] = {};
    uint32_t forestChildDieMasks[F4_FOREST_MAX_TREE_COUNT] = {};
    uint64_t forestOffsets[F4_FOREST_MAX_TREE_COUNT][F4_FOREST_CHUNK_COUNT] = {};
    uint64_t forestSizes[F4_FOREST_MAX_TREE_COUNT][F4_FOREST_CHUNK_COUNT] = {};
};

CcuResult CcuPrepareKernel(CcuKernelArg arg);
CcuResult CcuDirectKernel0(CcuKernelArg arg);
CcuResult CcuDirectKernel1(CcuKernelArg arg);
CcuResult CcuScatterKernel(CcuKernelArg arg);
CcuResult CcuCrossKernel(CcuKernelArg arg);
CcuResult CcuLocalAllgatherKernel(CcuKernelArg arg);
CcuResult CcuPackedThreePathKernel(CcuKernelArg arg);
CcuResult CcuClosPipelineKernel(CcuKernelArg arg);
CcuResult CcuMixedForestKernel(CcuKernelArg arg);
CcuResult CcuMeshKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
