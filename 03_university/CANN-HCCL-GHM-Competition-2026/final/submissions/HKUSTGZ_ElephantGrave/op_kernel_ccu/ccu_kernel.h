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

#include <ccu/ccu_types.h>

#include "common.h"
#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

/**
 * A rank owns exactly one channel for every peer.  The host partitions those
 * channels by network layer (and therefore by IO die) and registers one copy
 * of this kernel per non-empty layer.
 */
struct CcuKernelArgPartialReduce : public CcuKernelArgBase {
    HcclDataType dataType{HCCL_DATA_TYPE_FP32};
    HcclDataType outputDataType{HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HCCL_REDUCE_SUM};
    bool includeSelf{true};
    bool enableMsPrefix{false};
    bool enableWideLane{false};
    bool enableRank4PackedMs{false};
    bool compactArgs{false};
    uint32_t stripeCount{1};
    uint64_t staticSliceOffset{0};
    std::array<uint64_t, 4> staticMsGoSize{};
};

struct CcuKernelArgCombine : public CcuKernelArgBase {
    HcclDataType dataType{HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HCCL_REDUCE_SUM};
    uint32_t stripeCount{1};
    bool compactArgs{false};
    uint64_t staticStripeLength{0};
};

struct CcuKernelArgNodeLocal : public CcuKernelArgBase {
    HcclDataType dataType{HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HCCL_REDUCE_SUM};
    uint32_t targetCount{1};
};

struct CcuKernelArgCrossFinalize : public CcuKernelArgBase {
    HcclDataType dataType{HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HCCL_REDUCE_SUM};
    uint32_t ownerChannelIndex0{0};
    uint32_t ownerChannelIndex1{0};
    uint32_t remoteSlot{0};
};

constexpr uint32_t MATE_MAX_PIECES = 2U * MATE_MAX_PEERS;

struct CcuKernelArgMateLocal : public CcuKernelArgBase {
    HcclDataType dataType{HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HCCL_REDUCE_SUM};
    uint32_t pieceCount{0};
    uint32_t ownPieceCount{0};
    std::array<uint64_t, MATE_MAX_PIECES> inputOffsets{};
    std::array<uint64_t, MATE_MAX_PIECES> outputOffsets{};
    std::array<uint64_t, MATE_MAX_PIECES> pieceLengths{};
};

struct CcuKernelArgMateCross : public CcuKernelArgBase {
    HcclDataType dataType{HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HCCL_REDUCE_SUM};
    uint64_t remotePartialOffset{0};
    std::array<uint64_t, MATE_MAX_PEERS> outputOffsets{};
    std::array<uint64_t, MATE_MAX_PEERS> shardLengths{};
};

struct CcuKernelArgDirect : public CcuKernelArgBase {
    HcclDataType dataType{HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HCCL_REDUCE_SUM};
    uint64_t inputOffset{0};
    uint64_t outputOffset{0};
    uint64_t length{0};
    uint64_t workspaceOffset{0};
    uint64_t workspaceStride{0};
    std::array<uint64_t, DIRECT_TILE_COUNT> tileLengths{};
};

constexpr uint32_t WRITE_REDUCE_4X1_STRIPES = 3U;

struct CcuKernelArgWriteReduce4x1 : public CcuKernelArgBase {
    HcclDataType dataType{HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HCCL_REDUCE_SUM};
    uint32_t myRank{0};
    uint64_t outputBytes{0};
    std::array<uint32_t, WRITE_REDUCE_4X1_STRIPES> targetRanks{};
    std::array<uint32_t, WRITE_REDUCE_4X1_STRIPES> sourceIndicesAtTarget{};
    std::array<uint64_t, WRITE_REDUCE_4X1_STRIPES> stripeOffsets{};
    std::array<uint64_t, WRITE_REDUCE_4X1_STRIPES> stripeLengths{};
};

constexpr uint32_t WRITE_REDUCE_DUAL_MAX_CHANNELS = 8U;

struct CcuKernelArgWriteReduceDual : public CcuKernelArgBase {
    HcclDataType dataType{HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HCCL_REDUCE_SUM};
    uint32_t myRank{0};
    uint32_t rankSize{0};
    uint32_t stripeCount{0};
    bool initializeRange{false};
    uint64_t outputBytes{0};
    uint64_t phaseOffset{0};
    uint64_t phaseLength{0};
    std::array<uint32_t, WRITE_REDUCE_DUAL_MAX_CHANNELS> targetRanks{};
    std::array<uint32_t, WRITE_REDUCE_DUAL_MAX_CHANNELS> sourceIndicesAtTarget{};
    std::array<uint64_t, WRITE_REDUCE_DUAL_MAX_CHANNELS> stripeOffsets{};
    std::array<uint64_t, WRITE_REDUCE_DUAL_MAX_CHANNELS> stripeLengths{};
};

struct CcuKernelArgPublisher : public CcuKernelArgBase {};

// Reduce one topology layer directly from peer inputs into output/scratch.
CcuResult CcuReduceScatterPartialKernel(CcuKernelArg arg);

// Four cross-only ranks: initialize recvBuf from the local contribution, push
// three Latin stripes directly into each peer's recvBuf, and synchronize the
// three waves with CKE bits 3/1/3. The direct-output ABI remains four words.
CcuResult CcuReduceScatterWriteReduce4x1Kernel(CcuKernelArg arg);

// Dual-die rank-12/rank-16 push phase: initialize or extend one output range,
// then send Latin stripes to all peers on exactly one topology/IO die.
// Consecutive waves share one rank-agnostic alternating bit3/bit1 DATA epoch.
CcuResult CcuReduceScatterWriteReduceDualKernel(CcuKernelArg arg);

// Exact official rank-16 large target-owned variant. Stage-1 kernels publish
// the immutable input base/token once; stage-2 kernels reuse those XNs and
// close source lifetime with one final PostSync. The runtime ABI remains the
// same four words used by the measured V12w schedule.
CcuResult CcuReduceScatterOwnerPullDualKernel(CcuKernelArg arg);

// Reduce local-node inputs into the owned output chunk and compact remote
// destination partials.
CcuResult CcuReduceScatterMateLocalKernel(CcuKernelArg arg);

// Rank-12 H8 variant: allow more owned pieces than local ranks so the full
// local result and four H4 helper shards have nearly identical piece sizes.
CcuResult CcuReduceScatterMateLocalExpandedKernel(CcuKernelArg arg);

// Pull all disjoint remote-helper shards directly into the destination chunk.
CcuResult CcuReduceScatterMateCrossKernel(CcuKernelArg arg);

// Build a remote-node-only partial directly in the destination output.
CcuResult CcuReduceScatterDirectCrossKernel(CcuKernelArg arg);

// Rank-12 H8 variant: build the remote-node-only partial at scratch offset 0.
CcuResult CcuReduceScatterDirectCrossScratchKernel(CcuKernelArg arg);

// Add all local-node sources in place to the remote partial.
CcuResult CcuReduceScatterDirectLocalKernel(CcuKernelArg arg);

// Rank-12 H4 variant: initialize output from the local source before reducing
// local peers, rather than accumulating into an existing remote partial.
CcuResult CcuReduceScatterDirectLocalInitKernel(CcuKernelArg arg);

// Rank-12 H8 stage-2 peer for MateCross.  It publishes the helper mailbox and
// participates in the symmetric completion handshake without reading data.
CcuResult CcuReduceScatterPublisherKernel(CcuKernelArg arg);

// Rank-12 H8 local combine with an intra-node completion handshake.  The
// channel operations pin this otherwise local-only mission to the intra IO die.
CcuResult CcuReduceScatterPinnedCombineKernel(CcuKernelArg arg);

// Build node partials for own and remote-node destinations.
CcuResult CcuReduceScatterNodeLocalKernel(CcuKernelArg arg);

// Merge exactly one remote node partial into the destination owner.
CcuResult CcuReduceScatterCrossFinalizeKernel(CcuKernelArg arg);

// Deterministically merge the two IO-die partials after both kernels finish.
CcuResult CcuReduceScatterCombineKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
