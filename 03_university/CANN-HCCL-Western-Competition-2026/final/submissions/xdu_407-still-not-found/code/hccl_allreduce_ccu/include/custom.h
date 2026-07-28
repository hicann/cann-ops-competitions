/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "common.h"
#include "log.h"

enum class AlgorithmPackage : uint32_t { PACKAGE_A = 0 };

constexpr AlgorithmPackage ACTIVE_PACKAGE = AlgorithmPackage::PACKAGE_A;

enum class TopologyKind : uint32_t { TOPOLOGY_4X1 = 0, TOPOLOGY_8_PLUS_4, TOPOLOGY_2X8, UNSUPPORTED };

enum class MessageSizeKind : uint32_t { SIZE_512KB = 0, SIZE_512MB, SIZE_400MB_PLUS_4B, UNSUPPORTED };

enum class PlanId : uint32_t {
    PLAN_4X1_512KB = 0,
    PLAN_4X1_512MB,
    PLAN_4X1_400MB_PLUS_4B,
    PLAN_8_PLUS_4_512KB,
    PLAN_8_PLUS_4_512MB,
    PLAN_8_PLUS_4_400MB_PLUS_4B,
    PLAN_2X8_512KB,
    PLAN_2X8_512MB,
    PLAN_2X8_400MB_PLUS_4B,
    INVALID
};

enum class OptimizationTarget : uint32_t { LATENCY = 0, BANDWIDTH };

enum class DeterminismPolicy : uint32_t { FIXED_RANK_AND_SLICE_ORDER = 0 };

enum class AlgorithmFamily : uint32_t {
    RING_RSAG = 1,
    FULL_EXCHANGE_FIXED_TREE = 2,
    DIRECT12_RSAG = 3,
    TOPOLOGY_BALANCED_DIRECT_RSAG = 4,
    BULK_PUSH_STAGED_REDUCE = 5,
    INVALID = 15
};

enum class TailPolicy : uint32_t {
    NONE = 0,
    SPLIT_SCALAR = 1,
    UNEVEN_SHARDS = 2,
    FAMILY_NATIVE = 4,
    INVALID = 5
};

enum class ResourceProfileId : uint32_t { PROFILE_4X1 = 0, PROFILE_8_PLUS_4, PROFILE_2X8, INVALID };

// Temporary monolithic kernel ID domain used by the current scaffold host sources.
enum class CcuKernelId : uint32_t {
    KERNEL_4X1_512KB = 0,
    KERNEL_4X1_512MB,
    KERNEL_4X1_400MB_PLUS_4B,
    KERNEL_8_PLUS_4_512KB,
    KERNEL_8_PLUS_4_512MB,
    KERNEL_8_PLUS_4_400MB_PLUS_4B,
    KERNEL_2X8_512KB,
    KERNEL_2X8_512MB,
    KERNEL_2X8_400MB_PLUS_4B,
    INVALID
};

// Rewrite-only stage kernel domain. These values are never aliases of CcuKernelId.
enum class CcuStageKernelId : uint32_t {
    EIGHT_PLUS_FOUR_ALL_EXCHANGE_L0 = 11,
    EIGHT_PLUS_FOUR_ALL_EXCHANGE_L1 = 12,
    EIGHT_PLUS_FOUR_DIRECT12_RS_L0 = 13,
    TWO_X_EIGHT_MULTI_ROOT_LOCAL_REDUCE_L0 = 29,
    TWO_X_EIGHT_MULTI_ROOT_CROSS_REDUCE_L1 = 31,
    EIGHT_PLUS_FOUR_DIRECT12_RS_L0_DIE1 = 55,
    FOUR_X_ONE_RING_FUSED_WORKER0_L1 = 57,
    FOUR_X_ONE_RING_FUSED_WORKER1_L1 = 58,
    TWO_X_EIGHT_BULK_PUSH_L0 = 59,
    TWO_X_EIGHT_BULK_PUSH_L1 = 60,
    FOUR_X_ONE_PULL_FULL_EXCHANGE_L1 = 61,
    COUNT = 64,
    INVALID = UINT32_MAX
};

static_assert(!std::is_same<CcuKernelId, CcuStageKernelId>::value, "kernel ID domains must remain distinct");

constexpr std::array<CcuStageKernelId, 2> DIRECT12_STAGE_ORDER = {
    CcuStageKernelId::EIGHT_PLUS_FOUR_DIRECT12_RS_L0,
    CcuStageKernelId::EIGHT_PLUS_FOUR_DIRECT12_RS_L0_DIE1,
};

constexpr std::array<CcuStageKernelId, 2> FULL_EXCHANGE_STAGE_ORDER = {
    CcuStageKernelId::EIGHT_PLUS_FOUR_ALL_EXCHANGE_L0,
    CcuStageKernelId::EIGHT_PLUS_FOUR_ALL_EXCHANGE_L1,
};

constexpr std::array<CcuStageKernelId, 2> FOUR_X_ONE_RING_STAGE_ORDER = {
    CcuStageKernelId::FOUR_X_ONE_RING_FUSED_WORKER0_L1,
    CcuStageKernelId::FOUR_X_ONE_RING_FUSED_WORKER1_L1,
};

constexpr std::array<CcuStageKernelId, 1> FOUR_X_ONE_PULL_FULL_EXCHANGE_STAGE_ORDER = {
    CcuStageKernelId::FOUR_X_ONE_PULL_FULL_EXCHANGE_L1,
};

constexpr std::array<CcuStageKernelId, 2> TWO_X_EIGHT_MULTI_ROOT_STAGE_ORDER = {
    CcuStageKernelId::TWO_X_EIGHT_MULTI_ROOT_LOCAL_REDUCE_L0,
    CcuStageKernelId::TWO_X_EIGHT_MULTI_ROOT_CROSS_REDUCE_L1,
};

constexpr std::array<CcuStageKernelId, 2> TWO_X_EIGHT_BULK_STAGE_ORDER = {
    CcuStageKernelId::TWO_X_EIGHT_BULK_PUSH_L0,
    CcuStageKernelId::TWO_X_EIGHT_BULK_PUSH_L1,
};

enum class TaskArgIndex : uint32_t {
    INPUT_ADDR = 0,
    INPUT_TOKEN,
    OUTPUT_ADDR,
    OUTPUT_TOKEN,
    SCRATCH_ADDR,
    SCRATCH_TOKEN,
    DATA_OFFSET_BYTES,
    ELEMENT_COUNT,
    TILE_OFFSET_BYTES,
    TILE_ELEMENT_COUNT,
    GENERATION,
    FLAGS,
    COUNT
};

constexpr uint32_t TASK_ARG_COUNT = static_cast<uint32_t>(TaskArgIndex::COUNT);
static_assert(TASK_ARG_COUNT == 12);
static_assert(TASK_ARG_COUNT <= 13);

struct TaskArgs {
    uint64_t inputAddr = 0;
    uint64_t inputToken = 0;
    uint64_t outputAddr = 0;
    uint64_t outputToken = 0;
    uint64_t scratchAddr = 0;
    uint64_t scratchToken = 0;
    uint64_t dataOffsetBytes = 0;
    uint64_t elementCount = 0;
    uint64_t tileOffsetBytes = 0;
    uint64_t tileElementCount = 0;
    uint64_t generation = 0;
    uint64_t flags = 0;

    uint64_t *Data()
    {
        return &inputAddr;
    }

    const uint64_t *Data() const
    {
        return &inputAddr;
    }

    static constexpr uint32_t Size()
    {
        return TASK_ARG_COUNT;
    }

};

static_assert(sizeof(TaskArgs) == 12 * sizeof(uint64_t));
static_assert(std::is_standard_layout<TaskArgs>::value);
static_assert(std::is_trivially_copyable<TaskArgs>::value);
static_assert(offsetof(TaskArgs, inputAddr) == static_cast<uint32_t>(TaskArgIndex::INPUT_ADDR) * sizeof(uint64_t));
static_assert(offsetof(TaskArgs, inputToken) == static_cast<uint32_t>(TaskArgIndex::INPUT_TOKEN) * sizeof(uint64_t));
static_assert(offsetof(TaskArgs, outputAddr) == static_cast<uint32_t>(TaskArgIndex::OUTPUT_ADDR) * sizeof(uint64_t));
static_assert(offsetof(TaskArgs, outputToken) == static_cast<uint32_t>(TaskArgIndex::OUTPUT_TOKEN) * sizeof(uint64_t));
static_assert(offsetof(TaskArgs, scratchAddr) == static_cast<uint32_t>(TaskArgIndex::SCRATCH_ADDR) * sizeof(uint64_t));
static_assert(
    offsetof(TaskArgs, scratchToken) == static_cast<uint32_t>(TaskArgIndex::SCRATCH_TOKEN) * sizeof(uint64_t));
static_assert(
    offsetof(TaskArgs, dataOffsetBytes) == static_cast<uint32_t>(TaskArgIndex::DATA_OFFSET_BYTES) * sizeof(uint64_t));
static_assert(
    offsetof(TaskArgs, elementCount) == static_cast<uint32_t>(TaskArgIndex::ELEMENT_COUNT) * sizeof(uint64_t));
static_assert(
    offsetof(TaskArgs, tileOffsetBytes) == static_cast<uint32_t>(TaskArgIndex::TILE_OFFSET_BYTES) * sizeof(uint64_t));
static_assert(
    offsetof(TaskArgs, tileElementCount) == static_cast<uint32_t>(TaskArgIndex::TILE_ELEMENT_COUNT) * sizeof(uint64_t));
static_assert(offsetof(TaskArgs, generation) == static_cast<uint32_t>(TaskArgIndex::GENERATION) * sizeof(uint64_t));
static_assert(offsetof(TaskArgs, flags) == static_cast<uint32_t>(TaskArgIndex::FLAGS) * sizeof(uint64_t));

constexpr uint32_t ACTIVE_WORKERS = 2;
constexpr uint32_t MAX_WORKERS = 4;
constexpr uint32_t PIPELINE_GENERATIONS = 2;
constexpr uint32_t CHANNEL_NOTIFY_COUNT = 8;
constexpr uint32_t CELL_ENTRY_NOTIFY_INDEX = 0;
constexpr uint32_t CELL_EXIT_NOTIFY_INDEX = 1;
constexpr uint32_t CELL_INTERNAL_NOTIFY_BASE = 2;
constexpr uint32_t CELL_INTERNAL_NOTIFY_GENERATIONS = 2;
constexpr uint32_t CELL_THREAD_NOTIFY_COUNT = 4;

constexpr uint32_t CellInternalNotifyIndex(uint32_t phase)
{
    return CELL_INTERNAL_NOTIFY_BASE + phase % CELL_INTERNAL_NOTIFY_GENERATIONS;
}

static_assert(CELL_ENTRY_NOTIFY_INDEX < CELL_THREAD_NOTIFY_COUNT);
static_assert(CELL_EXIT_NOTIFY_INDEX < CELL_THREAD_NOTIFY_COUNT);
static_assert(CELL_INTERNAL_NOTIFY_BASE + CELL_INTERNAL_NOTIFY_GENERATIONS == CELL_THREAD_NOTIFY_COUNT);
static_assert(CellInternalNotifyIndex(0) == 2);
static_assert(CellInternalNotifyIndex(1) == 3);
static_assert(CellInternalNotifyIndex(2) == 2);
static_assert(CellInternalNotifyIndex(CELL_INTERNAL_NOTIFY_GENERATIONS - 1) < CELL_THREAD_NOTIFY_COUNT);
constexpr uint64_t MEBIBYTE = 1024ULL * 1024ULL;
constexpr uint32_t DIRECT12_SCRATCH_BANK_COUNT = 2;
constexpr uint32_t DIRECT12_CANONICAL_LEAF_COUNT = 12;
constexpr uint32_t DIRECT12_MAX_TILE_PAIRS = 3;
constexpr uint64_t DIRECT12_TILE_BYTES = 8ULL * MEBIBYTE;
constexpr uint64_t PACKAGE_A_DIRECT12_512_TILE_BYTES = 12ULL * MEBIBYTE;
constexpr uint64_t PACKAGE_A_DIRECT12_TAIL_TILE_BYTES = PACKAGE_A_DIRECT12_512_TILE_BYTES;
constexpr uint64_t PACKAGE_A_DIRECT12_MAX_TILE_BYTES = PACKAGE_A_DIRECT12_512_TILE_BYTES;
constexpr uint64_t DIRECT12_SCRATCH_BANK_BYTES = DIRECT12_CANONICAL_LEAF_COUNT * DIRECT12_TILE_BYTES;
constexpr uint64_t DIRECT12_SCRATCH_BYTES = DIRECT12_SCRATCH_BANK_COUNT * DIRECT12_SCRATCH_BANK_BYTES;
constexpr uint64_t PACKAGE_A_DIRECT12_512_SCRATCH_BYTES = 288ULL * MEBIBYTE;
constexpr uint64_t PACKAGE_A_DIRECT12_TAIL_SCRATCH_BYTES = PACKAGE_A_DIRECT12_512_SCRATCH_BYTES;
constexpr uint64_t PACKAGE_A_DIRECT12_TAIL_MESSAGE_ELEMENTS = 104857601ULL;
constexpr uint64_t PACKAGE_A_DIRECT12_TAIL_PREFIX_ELEMENTS = 104857596ULL;
constexpr uint64_t PACKAGE_A_DIRECT12_TAIL_SUFFIX_ELEMENTS = 5ULL;
constexpr uint64_t PACKAGE_A_DIRECT12_TAIL_PREFIX_BYTES = 419430384ULL;
constexpr uint64_t PACKAGE_A_DIRECT12_TAIL_SUFFIX_BYTES = 20ULL;
constexpr uint32_t DIRECT12_MODE_STANDARD = 0;
constexpr uint32_t DIRECT12_MODE_FRONTIER_512 = 1;
constexpr uint32_t DIRECT12_MODE_PREFIX_SCALAR_TAIL = 3;
constexpr uint64_t FULL_EXCHANGE_TILE_BYTES = MEBIBYTE / 2;
constexpr uint64_t FULL_EXCHANGE_SCRATCH_BYTES = DIRECT12_CANONICAL_LEAF_COUNT * FULL_EXCHANGE_TILE_BYTES;
constexpr uint32_t FOUR_X_ONE_RANK_COUNT = 4;
constexpr uint32_t FOUR_X_ONE_PEER_COUNT = FOUR_X_ONE_RANK_COUNT - 1;
constexpr uint64_t FOUR_X_ONE_SMALL_TILE_BYTES = MEBIBYTE / 2;
constexpr uint64_t FOUR_X_ONE_SMALL_SCRATCH_BYTES
    = FOUR_X_ONE_PEER_COUNT * FOUR_X_ONE_SMALL_TILE_BYTES;
constexpr uint32_t FOUR_X_ONE_PULL_GENERATIONS = 2;
constexpr uint64_t FOUR_X_ONE_PULL_FLAG_CONSERVATIVE = 0;
constexpr uint64_t FOUR_X_ONE_PULL_FLAG_DISJOINT = 1;
constexpr uint64_t PACKAGE_A_FOUR_X_ONE_RING_512_TILE_BYTES = 64ULL * MEBIBYTE;
constexpr uint64_t PACKAGE_A_FOUR_X_ONE_RING_TAIL_TILE_BYTES = 50ULL * MEBIBYTE;
constexpr uint32_t FOUR_X_ONE_RING_UNEVEN_OWNER_CHUNK = FOUR_X_ONE_RANK_COUNT - 1;
constexpr uint32_t TWO_X_EIGHT_RANK_COUNT = 16;
constexpr uint32_t TWO_X_EIGHT_MULTI_ROOT_COUNT = TWO_X_EIGHT_RANK_COUNT;
constexpr uint64_t TWO_X_EIGHT_MULTI_ROOT_MESSAGE_BYTES = MEBIBYTE / 2;
constexpr uint64_t TWO_X_EIGHT_MULTI_ROOT_SHARD_BYTES
    = TWO_X_EIGHT_MULTI_ROOT_MESSAGE_BYTES / TWO_X_EIGHT_MULTI_ROOT_COUNT;
constexpr uint64_t TWO_X_EIGHT_MULTI_ROOT_SCRATCH_BYTES
    = TWO_X_EIGHT_RANK_COUNT * TWO_X_EIGHT_MULTI_ROOT_SHARD_BYTES;
constexpr std::array<uint32_t, TWO_X_EIGHT_MULTI_ROOT_COUNT> TWO_X_EIGHT_MULTI_ROOT_ROOTS
    = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
constexpr uint64_t TWO_X_EIGHT_BULK_MESSAGE_BYTES = 512ULL * MEBIBYTE;
constexpr uint64_t TWO_X_EIGHT_BULK_OUTER_SLICE_BYTES = TWO_X_EIGHT_BULK_MESSAGE_BYTES;
constexpr uint32_t TWO_X_EIGHT_BULK_OUTER_SLICE_COUNT
    = TWO_X_EIGHT_BULK_MESSAGE_BYTES / TWO_X_EIGHT_BULK_OUTER_SLICE_BYTES;
constexpr uint64_t TWO_X_EIGHT_BULK_OWNER_SHARD_BYTES
    = TWO_X_EIGHT_BULK_OUTER_SLICE_BYTES / TWO_X_EIGHT_RANK_COUNT;
constexpr uint64_t TWO_X_EIGHT_BULK_TILE_BYTES = 8ULL * MEBIBYTE;
constexpr uint64_t PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES = 25ULL * MEBIBYTE - sizeof(float);
constexpr uint64_t PACKAGE_A_TWO_X_EIGHT_BULK_REMAINDER_TILE_BYTES
    = TWO_X_EIGHT_BULK_OWNER_SHARD_BYTES - PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES;
constexpr uint32_t TWO_X_EIGHT_BULK_TILES_PER_SHARD
    = TWO_X_EIGHT_BULK_OWNER_SHARD_BYTES / TWO_X_EIGHT_BULK_TILE_BYTES;
constexpr uint32_t PACKAGE_A_TWO_X_EIGHT_BULK_TILES_PER_SHARD = 2;
constexpr uint64_t TWO_X_EIGHT_BULK_SCRATCH_BYTES
    = TWO_X_EIGHT_RANK_COUNT * TWO_X_EIGHT_BULK_TILE_BYTES;
constexpr uint64_t PACKAGE_A_TWO_X_EIGHT_BULK_SCRATCH_BYTES
    = TWO_X_EIGHT_RANK_COUNT * PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES;
constexpr uint64_t TWO_X_EIGHT_BULK_TAIL_MESSAGE_BYTES = 400ULL * MEBIBYTE + sizeof(float);
constexpr uint64_t TWO_X_EIGHT_BULK_TAIL_UNEVEN_SLICE_OFFSET = 0;
constexpr uint64_t TWO_X_EIGHT_BULK_TAIL_OWNER_BASE_BYTES = 25ULL * MEBIBYTE;
constexpr uint64_t PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_TILE_BYTES = PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES;
constexpr uint64_t PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_SCRATCH_BYTES
    = TWO_X_EIGHT_RANK_COUNT * PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_TILE_BYTES;
constexpr uint64_t TWO_X_EIGHT_BULK_LAYOUT_UNIFORM = 0;
constexpr uint64_t TWO_X_EIGHT_BULK_LAYOUT_UNEVEN = 1;
constexpr uint64_t TWO_X_EIGHT_BULK_LAYOUT_UNEVEN_REMAINDER = 2;
constexpr uint64_t TWO_X_EIGHT_BULK_LAYOUT_ONE_WAVE_UNIFORM = 3;
constexpr uint64_t TWO_X_EIGHT_BULK_LAYOUT_ONE_WAVE_TAIL = 4;
constexpr uint64_t PIPELINE_SLOT_BYTES = 32ULL * MEBIBYTE;
constexpr uint64_t PIPELINE_SLOT_ALIGNMENT_BYTES = 2ULL * MEBIBYTE;
constexpr uint16_t CHANNEL_NOTIFY_MASK = 1;
constexpr uint64_t DIRECT12_FLAG_PUBLISH = 1ULL << 0;
constexpr uint64_t DIRECT12_FLAG_JOIN = 2;
constexpr uint64_t DIRECT12_FLAG_BROADCAST = 3;
constexpr uint64_t DIRECT12_FLAG_L1_GATHER = 4;
constexpr uint64_t DIRECT12_FLAG_L1_PUBLISH = 5;
constexpr uint64_t DIRECT12_FLAG_L1_BROADCAST = 6;
constexpr uint64_t DIRECT12_FLAG_GATHER_ALL_PUBLISH = 7;
constexpr uint64_t DIRECT12_FLAG_GATHER_ALL = 8;
constexpr uint64_t DIRECT12_FLAG_BROADCAST_ALL = 9;
constexpr uint64_t DIRECT12_FLAG_PAIR_GATHER_ALL_PUBLISH = 10;
constexpr uint64_t DIRECT12_FLAG_PAIR_GATHER_ALL = 11;
constexpr uint64_t DIRECT12_FLAG_PAIR_BROADCAST_ALL = 12;
constexpr uint64_t DIRECT12_FLAG_PAIR_BROADCAST_GATHER_NEXT = 13;
constexpr uint64_t DIRECT12_FLAG_BROADCAST_GATHER_SCALAR = 14;
constexpr uint64_t DIRECT12_FLAG_SCALAR_JOIN = 15;
constexpr uint64_t FULL_EXCHANGE_FLAG_GATHER = 1;
constexpr uint64_t FULL_EXCHANGE_FLAG_JOIN = 2;
constexpr uint64_t FOUR_X_ONE_RING_FLAG_EXECUTE = 1;
constexpr uint64_t FOUR_X_ONE_RING_FLAG_EXECUTE_UNEVEN_TAIL = 2;
constexpr uint64_t TWO_X_EIGHT_MULTI_ROOT_FLAG_GATHER = 1;
constexpr uint64_t TWO_X_EIGHT_MULTI_ROOT_FLAG_JOIN = 2;
constexpr uint64_t TWO_X_EIGHT_MULTI_ROOT_FLAG_BROADCAST = 3;
constexpr uint64_t TWO_X_EIGHT_MULTI_ROOT_FLAG_DUAL_REPLICA_GATHER = 4;
constexpr uint64_t TWO_X_EIGHT_MULTI_ROOT_FLAG_DUAL_REPLICA_JOIN_BROADCAST = 5;
constexpr uint64_t TWO_X_EIGHT_BULK_FLAG_PUSH = 1;
constexpr uint64_t TWO_X_EIGHT_BULK_FLAG_JOIN = 2;
constexpr uint64_t TWO_X_EIGHT_BULK_FLAG_ALL_GATHER = 3;
constexpr uint64_t TWO_X_EIGHT_BULK_FLAG_ONE_WAVE_PUSH = 4;
constexpr uint64_t TWO_X_EIGHT_BULK_FLAG_ONE_WAVE_JOIN = 5;
constexpr uint64_t TWO_X_EIGHT_BULK_FLAG_ONE_WAVE_ALL_GATHER = 6;

struct CellKey {
    AlgorithmPackage package = AlgorithmPackage::PACKAGE_A;
    TopologyKind topology = TopologyKind::UNSUPPORTED;
    MessageSizeKind messageSize = MessageSizeKind::UNSUPPORTED;
};

struct CellDescriptor {
    CellKey key{};
    PlanId planId = PlanId::INVALID;
    AlgorithmFamily family = AlgorithmFamily::INVALID;
    OptimizationTarget target = OptimizationTarget::LATENCY;
    ResourceProfileId resourceProfile = ResourceProfileId::INVALID;
    TailPolicy tailPolicy = TailPolicy::INVALID;
    uint32_t streamCount = 0;
    uint32_t kernelHandleCount = 0;
    uint64_t tileBytes = 0;
    uint64_t scratchBytes = 0;
    bool implemented = false;
};

constexpr uint32_t CELL_DESCRIPTOR_COUNT = 9;
constexpr CellDescriptor MakeCell(AlgorithmPackage package, TopologyKind topology, MessageSizeKind messageSize,
    PlanId planId, AlgorithmFamily family, OptimizationTarget target, ResourceProfileId resourceProfile,
    TailPolicy tailPolicy, uint32_t streamCount, uint32_t kernelHandleCount, uint64_t tileBytes,
    uint64_t scratchBytes = 0, bool implemented = false)
{
    return {{package, topology, messageSize}, planId, family, target, resourceProfile, tailPolicy, streamCount,
        kernelHandleCount, tileBytes, scratchBytes, implemented};
}

constexpr std::array<CellDescriptor, CELL_DESCRIPTOR_COUNT> CELL_DESCRIPTORS = {{
    MakeCell(AlgorithmPackage::PACKAGE_A, TopologyKind::TOPOLOGY_4X1, MessageSizeKind::SIZE_512KB,
        PlanId::PLAN_4X1_512KB, AlgorithmFamily::FULL_EXCHANGE_FIXED_TREE, OptimizationTarget::LATENCY,
        ResourceProfileId::PROFILE_4X1, TailPolicy::NONE, 1, 1, FOUR_X_ONE_SMALL_TILE_BYTES,
        FOUR_X_ONE_SMALL_SCRATCH_BYTES, true),
    MakeCell(AlgorithmPackage::PACKAGE_A, TopologyKind::TOPOLOGY_4X1, MessageSizeKind::SIZE_512MB,
        PlanId::PLAN_4X1_512MB, AlgorithmFamily::RING_RSAG, OptimizationTarget::BANDWIDTH,
        ResourceProfileId::PROFILE_4X1, TailPolicy::NONE, 2, 2,
        PACKAGE_A_FOUR_X_ONE_RING_512_TILE_BYTES, 0, true),
    MakeCell(AlgorithmPackage::PACKAGE_A, TopologyKind::TOPOLOGY_4X1, MessageSizeKind::SIZE_400MB_PLUS_4B,
        PlanId::PLAN_4X1_400MB_PLUS_4B, AlgorithmFamily::RING_RSAG, OptimizationTarget::BANDWIDTH,
        ResourceProfileId::PROFILE_4X1, TailPolicy::UNEVEN_SHARDS, 2, 2,
        PACKAGE_A_FOUR_X_ONE_RING_TAIL_TILE_BYTES, 0, true),
    MakeCell(AlgorithmPackage::PACKAGE_A, TopologyKind::TOPOLOGY_8_PLUS_4, MessageSizeKind::SIZE_512KB,
        PlanId::PLAN_8_PLUS_4_512KB, AlgorithmFamily::FULL_EXCHANGE_FIXED_TREE, OptimizationTarget::LATENCY,
        ResourceProfileId::PROFILE_8_PLUS_4, TailPolicy::NONE, 2, 2, FULL_EXCHANGE_TILE_BYTES,
        FULL_EXCHANGE_SCRATCH_BYTES, true),
    MakeCell(AlgorithmPackage::PACKAGE_A, TopologyKind::TOPOLOGY_8_PLUS_4, MessageSizeKind::SIZE_512MB,
        PlanId::PLAN_8_PLUS_4_512MB, AlgorithmFamily::DIRECT12_RSAG, OptimizationTarget::BANDWIDTH,
        ResourceProfileId::PROFILE_8_PLUS_4, TailPolicy::NONE, 2, 2, PACKAGE_A_DIRECT12_512_TILE_BYTES,
        PACKAGE_A_DIRECT12_512_SCRATCH_BYTES, true),
    MakeCell(AlgorithmPackage::PACKAGE_A, TopologyKind::TOPOLOGY_8_PLUS_4, MessageSizeKind::SIZE_400MB_PLUS_4B,
        PlanId::PLAN_8_PLUS_4_400MB_PLUS_4B, AlgorithmFamily::DIRECT12_RSAG, OptimizationTarget::BANDWIDTH,
        ResourceProfileId::PROFILE_8_PLUS_4, TailPolicy::UNEVEN_SHARDS, 2, 2,
        PACKAGE_A_DIRECT12_TAIL_TILE_BYTES, PACKAGE_A_DIRECT12_TAIL_SCRATCH_BYTES, true),
    MakeCell(AlgorithmPackage::PACKAGE_A, TopologyKind::TOPOLOGY_2X8, MessageSizeKind::SIZE_512KB,
        PlanId::PLAN_2X8_512KB, AlgorithmFamily::TOPOLOGY_BALANCED_DIRECT_RSAG, OptimizationTarget::LATENCY,
        ResourceProfileId::PROFILE_2X8, TailPolicy::NONE, 2, 2, TWO_X_EIGHT_MULTI_ROOT_MESSAGE_BYTES,
        TWO_X_EIGHT_MULTI_ROOT_SCRATCH_BYTES, true),
    MakeCell(AlgorithmPackage::PACKAGE_A, TopologyKind::TOPOLOGY_2X8, MessageSizeKind::SIZE_512MB,
        PlanId::PLAN_2X8_512MB, AlgorithmFamily::BULK_PUSH_STAGED_REDUCE, OptimizationTarget::BANDWIDTH,
        ResourceProfileId::PROFILE_2X8, TailPolicy::NONE, 2, 2, PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES,
        PACKAGE_A_TWO_X_EIGHT_BULK_SCRATCH_BYTES, true),
    MakeCell(AlgorithmPackage::PACKAGE_A, TopologyKind::TOPOLOGY_2X8, MessageSizeKind::SIZE_400MB_PLUS_4B,
        PlanId::PLAN_2X8_400MB_PLUS_4B, AlgorithmFamily::BULK_PUSH_STAGED_REDUCE, OptimizationTarget::BANDWIDTH,
        ResourceProfileId::PROFILE_2X8, TailPolicy::FAMILY_NATIVE, 2, 2,
        PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_TILE_BYTES, PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_SCRATCH_BYTES, true),
}};

constexpr bool SameCellKey(const CellKey &left, const CellKey &right)
{
    return left.package == right.package && left.topology == right.topology && left.messageSize == right.messageSize;
}

constexpr bool ValidateCellDescriptorTable()
{
    for (uint32_t index = 0; index < CELL_DESCRIPTORS.size(); ++index) {
        const CellDescriptor &cell = CELL_DESCRIPTORS[index];
        if (cell.key.package != AlgorithmPackage::PACKAGE_A || cell.key.topology == TopologyKind::UNSUPPORTED
            || cell.key.messageSize == MessageSizeKind::UNSUPPORTED
            || cell.planId == PlanId::INVALID || cell.family == AlgorithmFamily::INVALID
            || cell.resourceProfile == ResourceProfileId::INVALID || cell.tailPolicy == TailPolicy::INVALID
            || cell.streamCount == 0 || cell.streamCount > 2 || cell.kernelHandleCount == 0
            || cell.kernelHandleCount > 2 || cell.tileBytes == 0) {
            return false;
        }
        for (uint32_t prior = 0; prior < index; ++prior) {
            if (SameCellKey(cell.key, CELL_DESCRIPTORS[prior].key)) {
                return false;
            }
        }
    }
    return true;
}

constexpr const CellDescriptor *FindCellDescriptor(
    AlgorithmPackage package, TopologyKind topology, MessageSizeKind messageSize)
{
    for (const CellDescriptor &cell : CELL_DESCRIPTORS) {
        if (cell.key.package == package && cell.key.topology == topology && cell.key.messageSize == messageSize) {
            return &cell;
        }
    }
    return nullptr;
}

static_assert(ValidateCellDescriptorTable(), "the fixed nine-cell matrix must be complete and unique");

constexpr uint32_t Direct12NotifySlot(uint32_t worker, uint32_t generation)
{
    return worker * PIPELINE_GENERATIONS + generation;
}

constexpr uint32_t Direct12CreditSlot(uint32_t worker, uint32_t pairGeneration)
{
    return Direct12NotifySlot(worker, pairGeneration);
}

constexpr uint64_t Direct12ShardElementCount(uint64_t totalElements, uint32_t canonicalRank)
{
    return totalElements / DIRECT12_CANONICAL_LEAF_COUNT
           + static_cast<uint64_t>(canonicalRank < totalElements % DIRECT12_CANONICAL_LEAF_COUNT);
}

constexpr uint64_t Direct12ShardElementOffset(uint64_t totalElements, uint32_t canonicalRank)
{
    return (totalElements / DIRECT12_CANONICAL_LEAF_COUNT) * canonicalRank
           + (canonicalRank < totalElements % DIRECT12_CANONICAL_LEAF_COUNT
                   ? canonicalRank
                   : totalElements % DIRECT12_CANONICAL_LEAF_COUNT);
}

constexpr uint64_t Direct12TileElements(uint64_t tileBytes)
{
    return tileBytes / sizeof(float);
}

constexpr uint32_t Direct12TileCount(uint64_t shardElements, uint64_t tileBytes)
{
    return static_cast<uint32_t>((shardElements + Direct12TileElements(tileBytes) - 1)
                                 / Direct12TileElements(tileBytes));
}

constexpr uint64_t Direct12TileElementCount(uint64_t shardElements, uint32_t tile, uint64_t tileBytes)
{
    const uint64_t tileElements = Direct12TileElements(tileBytes);
    const uint64_t offset = static_cast<uint64_t>(tile) * tileElements;
    return offset >= shardElements
               ? 0
               : (shardElements - offset < tileElements ? shardElements - offset : tileElements);
}

constexpr uint64_t Direct12ScratchBankBytes(uint64_t scratchStrideBytes)
{
    return DIRECT12_CANONICAL_LEAF_COUNT * scratchStrideBytes;
}

constexpr uint64_t Direct12ScratchBytes(uint64_t scratchStrideBytes)
{
    return DIRECT12_SCRATCH_BANK_COUNT * Direct12ScratchBankBytes(scratchStrideBytes);
}

constexpr uint32_t Direct12ReadyNotify(uint32_t worker, uint32_t generation)
{
    return Direct12NotifySlot(worker, generation) * 2;
}

constexpr uint32_t Direct12AckNotify(uint32_t worker, uint32_t generation)
{
    return Direct12ReadyNotify(worker, generation) + 1;
}

constexpr uint32_t NotifyIndex(uint32_t worker, uint32_t generation)
{
    return worker * PIPELINE_GENERATIONS + generation;
}

constexpr uint32_t FOUR_X_ONE_VARIABLES_PER_WORKER = 4;
constexpr uint32_t FOUR_X_ONE_INPUT_ADDR_FIELD = 0;
constexpr uint32_t FOUR_X_ONE_INPUT_TOKEN_FIELD = 1;
constexpr uint32_t FOUR_X_ONE_OUTPUT_ADDR_FIELD = 2;
constexpr uint32_t FOUR_X_ONE_OUTPUT_TOKEN_FIELD = 3;

constexpr uint32_t FourXOneVariableBase(uint32_t worker)
{
    return worker * FOUR_X_ONE_VARIABLES_PER_WORKER;
}

constexpr uint16_t FourXOneAddressFieldMask(uint32_t field)
{
    return static_cast<uint16_t>(1U << field);
}

constexpr uint32_t FourXOneRingDataNotify(uint32_t worker)
{
    return worker * 4;
}

constexpr uint32_t FourXOneRingAddressNotify(uint32_t worker)
{
    return FourXOneRingDataNotify(worker) + 1;
}

constexpr uint16_t FourXOneRingReadyMask(uint32_t phase)
{
    return static_cast<uint16_t>(1U << (phase * 2));
}

constexpr uint16_t FourXOneRingAckMask(uint32_t phase)
{
    return static_cast<uint16_t>(1U << (phase * 2 + 1));
}

constexpr uint16_t FOUR_X_ONE_RING_TAIL_READY_MASK = 0x4000;
constexpr uint16_t FOUR_X_ONE_RING_TAIL_ACK_MASK = 0x8000;

constexpr uint16_t FOUR_X_ONE_SNAPSHOT0_MASK = 0x0001;

constexpr uint32_t Direct12WorkerForTile(uint32_t tile)
{
    return tile % ACTIVE_WORKERS;
}

constexpr uint32_t Direct12GenerationForTile(uint32_t tile)
{
    return (tile / ACTIVE_WORKERS) % PIPELINE_GENERATIONS;
}

constexpr uint64_t Direct12ScratchLeafOffset(
    uint32_t worker, uint32_t canonicalLeaf, uint64_t scratchStrideBytes)
{
    return worker * Direct12ScratchBankBytes(scratchStrideBytes)
           + canonicalLeaf * scratchStrideBytes;
}

constexpr uint64_t FullExchangeScratchLeafOffset(uint32_t canonicalLeaf)
{
    return canonicalLeaf * FULL_EXCHANGE_TILE_BYTES;
}

constexpr uint64_t TwoXEightMultiRootScratchLeafOffset(uint32_t canonicalLeaf)
{
    return canonicalLeaf * TWO_X_EIGHT_MULTI_ROOT_SHARD_BYTES;
}

constexpr bool IsTwoXEightMultiRoot(uint32_t canonicalRank)
{
    return canonicalRank < TWO_X_EIGHT_RANK_COUNT;
}

constexpr uint32_t TwoXEightMultiRootIndex(uint32_t canonicalRank)
{
    return canonicalRank;
}

constexpr uint64_t TwoXEightMultiRootShardOffset(uint32_t rootIndex)
{
    return rootIndex * TWO_X_EIGHT_MULTI_ROOT_SHARD_BYTES;
}

constexpr uint32_t TWO_X_EIGHT_REDUCE_GROUP_SIZE = TWO_X_EIGHT_RANK_COUNT / 2;

constexpr uint32_t TwoXEightLayerGroupBase(uint32_t canonicalRank, bool isLocalLayer)
{
    const uint32_t localBase
        = canonicalRank / TWO_X_EIGHT_REDUCE_GROUP_SIZE * TWO_X_EIGHT_REDUCE_GROUP_SIZE;
    return isLocalLayer ? localBase : localBase ^ TWO_X_EIGHT_REDUCE_GROUP_SIZE;
}

constexpr uint64_t TwoXEightBulkScratchLeafOffset(uint32_t canonicalLeaf)
{
    return canonicalLeaf * PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES;
}

constexpr uint64_t TwoXEightBulkTailScratchLeafOffset(uint32_t canonicalLeaf)
{
    return canonicalLeaf * PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_TILE_BYTES;
}

constexpr uint64_t TwoXEightBulkTileBytes(uint32_t tile)
{
    return tile == 0 ? PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES
                     : PACKAGE_A_TWO_X_EIGHT_BULK_REMAINDER_TILE_BYTES;
}

constexpr uint64_t TwoXEightBulkTileOffset(uint32_t owner, uint32_t tile)
{
    return static_cast<uint64_t>(owner) * TWO_X_EIGHT_BULK_OWNER_SHARD_BYTES
           + (tile == 0 ? 0 : PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES);
}

constexpr uint64_t TwoXEightBulkTailOwnerBytes(uint32_t owner)
{
    return TWO_X_EIGHT_BULK_TAIL_OWNER_BASE_BYTES + (owner == 0 ? sizeof(float) : 0);
}

constexpr uint64_t TwoXEightBulkTailOwnerOffset(uint32_t owner)
{
    return static_cast<uint64_t>(owner) * TWO_X_EIGHT_BULK_TAIL_OWNER_BASE_BYTES
           + (owner == 0 ? 0 : sizeof(float));
}

constexpr uint64_t TwoXEightBulkTailTileBytes(uint32_t owner, uint32_t tile)
{
    return tile == 0 ? PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_TILE_BYTES
                     : TwoXEightBulkTailOwnerBytes(owner) - PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_TILE_BYTES;
}

constexpr uint64_t TwoXEightOneWaveOwnerBytes(uint32_t owner, bool tail)
{
    return tail ? 25ULL * MEBIBYTE + (owner == 0 ? sizeof(float) : 0)
                : 32ULL * MEBIBYTE;
}

constexpr uint64_t TwoXEightOneWaveOwnerOffset(uint32_t owner, bool tail)
{
    return tail ? static_cast<uint64_t>(owner) * 25ULL * MEBIBYTE
                      + (owner == 0 ? 0 : sizeof(float))
                : static_cast<uint64_t>(owner) * 32ULL * MEBIBYTE;
}

constexpr uint64_t TwoXEightOneWaveHalfBytes(uint32_t owner, bool tail)
{
    return TWO_X_EIGHT_REDUCE_GROUP_SIZE * TwoXEightOneWaveOwnerBytes(owner, tail);
}

constexpr uint64_t TwoXEightOneWaveLeafOffset(uint32_t canonicalLeaf, uint32_t owner, bool tail)
{
    return (canonicalLeaf % TWO_X_EIGHT_REDUCE_GROUP_SIZE)
           * TwoXEightOneWaveOwnerBytes(owner, tail);
}

constexpr bool EqualSizedRangesDisjoint(uint64_t first, uint64_t second, uint64_t bytes)
{
    return first < second ? second - first >= bytes : first - second >= bytes;
}

constexpr uint64_t FourXOneRingChunkElements(uint64_t totalElements)
{
    return totalElements / FOUR_X_ONE_RANK_COUNT;
}

constexpr uint64_t FourXOneRingTileElements(uint64_t tileBytes)
{
    return tileBytes / sizeof(float);
}

constexpr uint32_t FourXOneRingTileCount(uint64_t chunkElements, uint64_t tileBytes)
{
    return static_cast<uint32_t>(
        (chunkElements + FourXOneRingTileElements(tileBytes) - 1) / FourXOneRingTileElements(tileBytes));
}

constexpr uint64_t FourXOneRingTileElementCount(uint64_t chunkElements, uint64_t tileBytes, uint32_t tile)
{
    const uint64_t tileElements = FourXOneRingTileElements(tileBytes);
    const uint64_t offset = static_cast<uint64_t>(tile) * tileElements;
    return offset >= chunkElements
               ? 0
               : (chunkElements - offset < tileElements
                         ? chunkElements - offset
                         : tileElements);
}

constexpr uint64_t FourXOneTailBulkElements(uint64_t totalElements)
{
    return totalElements - totalElements % FOUR_X_ONE_RANK_COUNT;
}

constexpr uint64_t FourXOneTailElementCount(uint64_t totalElements)
{
    return totalElements - FourXOneTailBulkElements(totalElements);
}

static_assert(ACTIVE_WORKERS <= MAX_WORKERS);
static_assert(ACTIVE_WORKERS == DIRECT12_SCRATCH_BANK_COUNT);
static_assert(Direct12ScratchBytes(PACKAGE_A_DIRECT12_512_TILE_BYTES)
              == PACKAGE_A_DIRECT12_512_SCRATCH_BYTES);
static_assert(Direct12ScratchBytes(PACKAGE_A_DIRECT12_TAIL_TILE_BYTES)
              == PACKAGE_A_DIRECT12_TAIL_SCRATCH_BYTES);
static_assert(PACKAGE_A_DIRECT12_TAIL_MESSAGE_ELEMENTS * sizeof(float)
              == 400ULL * MEBIBYTE + sizeof(float));
static_assert(PACKAGE_A_DIRECT12_TAIL_PREFIX_ELEMENTS % DIRECT12_CANONICAL_LEAF_COUNT == 0);
static_assert(PACKAGE_A_DIRECT12_TAIL_PREFIX_ELEMENTS + PACKAGE_A_DIRECT12_TAIL_SUFFIX_ELEMENTS
              == PACKAGE_A_DIRECT12_TAIL_MESSAGE_ELEMENTS);
static_assert(PACKAGE_A_DIRECT12_TAIL_PREFIX_BYTES
              == PACKAGE_A_DIRECT12_TAIL_PREFIX_ELEMENTS * sizeof(float));
static_assert(PACKAGE_A_DIRECT12_TAIL_PREFIX_BYTES + PACKAGE_A_DIRECT12_TAIL_SUFFIX_BYTES
              == 400ULL * MEBIBYTE + sizeof(float));
static_assert(PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES % sizeof(float) == 0);
static_assert(PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES < TWO_X_EIGHT_BULK_OWNER_SHARD_BYTES);
static_assert(TwoXEightLayerGroupBase(0, true) == 0);
static_assert(TwoXEightLayerGroupBase(0, false) == 8);
static_assert(TwoXEightLayerGroupBase(8, true) == 8);
static_assert(TwoXEightLayerGroupBase(8, false) == 0);
static_assert(TWO_X_EIGHT_BULK_OUTER_SLICE_COUNT == 1);
static_assert(PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES == 25ULL * MEBIBYTE - sizeof(float));
static_assert(PACKAGE_A_TWO_X_EIGHT_BULK_REMAINDER_TILE_BYTES == 7ULL * MEBIBYTE + sizeof(float));
static_assert(PACKAGE_A_TWO_X_EIGHT_BULK_SCRATCH_BYTES == 400ULL * MEBIBYTE - 64);
static_assert(PACKAGE_A_TWO_X_EIGHT_BULK_SCRATCH_BYTES < 400ULL * MEBIBYTE);
static_assert(TWO_X_EIGHT_BULK_TAIL_OWNER_BASE_BYTES == 25ULL * MEBIBYTE);
static_assert(PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_SCRATCH_BYTES == 400ULL * MEBIBYTE - 64);
static_assert(NotifyIndex(1, 0) == 2);
static_assert(NotifyIndex(1, 1) == 3);
static_assert(Direct12ReadyNotify(0, 0) == 0 && Direct12AckNotify(0, 0) == 1);
static_assert(Direct12ReadyNotify(0, 1) == 2 && Direct12AckNotify(0, 1) == 3);
static_assert(Direct12ReadyNotify(1, 0) == 4 && Direct12AckNotify(1, 0) == 5);
static_assert(Direct12ReadyNotify(1, 1) == 6 && Direct12AckNotify(1, 1) == 7);
static_assert(Direct12AckNotify(ACTIVE_WORKERS - 1, PIPELINE_GENERATIONS - 1) + 1 == CHANNEL_NOTIFY_COUNT);
static_assert(FOUR_X_ONE_SMALL_SCRATCH_BYTES == 3ULL * MEBIBYTE / 2);
static_assert(FourXOneVariableBase(0) == 0 && FourXOneVariableBase(1) == 4);
static_assert(FourXOneVariableBase(ACTIVE_WORKERS - 1) + FOUR_X_ONE_VARIABLES_PER_WORKER
              == CHANNEL_NOTIFY_COUNT);
static_assert(FourXOneAddressFieldMask(FOUR_X_ONE_INPUT_ADDR_FIELD) == 0x0001
              && FourXOneAddressFieldMask(FOUR_X_ONE_INPUT_TOKEN_FIELD) == 0x0002
              && FourXOneAddressFieldMask(FOUR_X_ONE_OUTPUT_ADDR_FIELD) == 0x0004
              && FourXOneAddressFieldMask(FOUR_X_ONE_OUTPUT_TOKEN_FIELD) == 0x0008);
static_assert(FourXOneRingDataNotify(0) == 0 && FourXOneRingAddressNotify(0) == 1
              && FourXOneRingDataNotify(1) == 4 && FourXOneRingAddressNotify(1) == 5);
static_assert(FourXOneRingReadyMask(0) == 0x0001 && FourXOneRingAckMask(0) == 0x0002
              && FourXOneRingReadyMask(5) == 0x0400 && FourXOneRingAckMask(5) == 0x0800);
static_assert((FourXOneRingReadyMask(0) | FourXOneRingAckMask(0)
                  | FourXOneRingReadyMask(1) | FourXOneRingAckMask(1)
                  | FourXOneRingReadyMask(2) | FourXOneRingAckMask(2)
                  | FourXOneRingReadyMask(3) | FourXOneRingAckMask(3)
                  | FourXOneRingReadyMask(4) | FourXOneRingAckMask(4)
                  | FourXOneRingReadyMask(5) | FourXOneRingAckMask(5)
                  | FOUR_X_ONE_RING_TAIL_READY_MASK | FOUR_X_ONE_RING_TAIL_ACK_MASK)
              == 0xcfff);
static_assert(Direct12ScratchLeafOffset(DIRECT12_SCRATCH_BANK_COUNT - 1,
                  DIRECT12_CANONICAL_LEAF_COUNT - 1, PACKAGE_A_DIRECT12_512_TILE_BYTES)
                  + PACKAGE_A_DIRECT12_512_TILE_BYTES
              == PACKAGE_A_DIRECT12_512_SCRATCH_BYTES);
static_assert(Direct12ScratchLeafOffset(DIRECT12_SCRATCH_BANK_COUNT - 1,
                  DIRECT12_CANONICAL_LEAF_COUNT - 1, PACKAGE_A_DIRECT12_TAIL_TILE_BYTES)
                  + PACKAGE_A_DIRECT12_TAIL_TILE_BYTES
              == PACKAGE_A_DIRECT12_TAIL_SCRATCH_BYTES);
static_assert(FullExchangeScratchLeafOffset(DIRECT12_CANONICAL_LEAF_COUNT - 1) + FULL_EXCHANGE_TILE_BYTES
    == FULL_EXCHANGE_SCRATCH_BYTES);
static_assert(TWO_X_EIGHT_MULTI_ROOT_COUNT == TWO_X_EIGHT_RANK_COUNT);
static_assert(TWO_X_EIGHT_MULTI_ROOT_SHARD_BYTES == 32ULL * 1024);
static_assert(TWO_X_EIGHT_MULTI_ROOT_SCRATCH_BYTES == TWO_X_EIGHT_MULTI_ROOT_MESSAGE_BYTES);
static_assert(TWO_X_EIGHT_MULTI_ROOT_ROOTS.front() == 0
              && TWO_X_EIGHT_MULTI_ROOT_ROOTS.back() == TWO_X_EIGHT_RANK_COUNT - 1);
static_assert(TwoXEightMultiRootShardOffset(TWO_X_EIGHT_MULTI_ROOT_COUNT - 1)
                  + TWO_X_EIGHT_MULTI_ROOT_SHARD_BYTES
              == TWO_X_EIGHT_MULTI_ROOT_MESSAGE_BYTES);
static_assert(TwoXEightMultiRootScratchLeafOffset(TWO_X_EIGHT_RANK_COUNT - 1)
                  + TWO_X_EIGHT_MULTI_ROOT_SHARD_BYTES
              == TWO_X_EIGHT_MULTI_ROOT_SCRATCH_BYTES);
static_assert(TWO_X_EIGHT_BULK_OUTER_SLICE_COUNT == 1);
static_assert(TWO_X_EIGHT_BULK_TILES_PER_SHARD == 4);
static_assert(PACKAGE_A_TWO_X_EIGHT_BULK_TILES_PER_SHARD == 2);
static_assert(TwoXEightBulkTileBytes(0) + TwoXEightBulkTileBytes(1)
              == TWO_X_EIGHT_BULK_OWNER_SHARD_BYTES);
static_assert(TwoXEightBulkTileOffset(
                  TWO_X_EIGHT_RANK_COUNT - 1, PACKAGE_A_TWO_X_EIGHT_BULK_TILES_PER_SHARD - 1)
                  + TwoXEightBulkTileBytes(PACKAGE_A_TWO_X_EIGHT_BULK_TILES_PER_SHARD - 1)
              == TWO_X_EIGHT_BULK_MESSAGE_BYTES);
static_assert(TwoXEightBulkScratchLeafOffset(TWO_X_EIGHT_RANK_COUNT - 1)
                  + PACKAGE_A_TWO_X_EIGHT_BULK_TILE_BYTES
              == PACKAGE_A_TWO_X_EIGHT_BULK_SCRATCH_BYTES);
static_assert(TwoXEightBulkTailScratchLeafOffset(TWO_X_EIGHT_RANK_COUNT - 1)
                  + PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_TILE_BYTES
              == PACKAGE_A_TWO_X_EIGHT_BULK_TAIL_SCRATCH_BYTES);
static_assert(TWO_X_EIGHT_BULK_TAIL_UNEVEN_SLICE_OFFSET
                  + TwoXEightBulkTailOwnerOffset(TWO_X_EIGHT_RANK_COUNT - 1)
                  + TwoXEightBulkTailOwnerBytes(TWO_X_EIGHT_RANK_COUNT - 1)
              == TWO_X_EIGHT_BULK_TAIL_MESSAGE_BYTES);
static_assert(TwoXEightBulkTailTileBytes(0, 0) + TwoXEightBulkTailTileBytes(0, 1)
              == TwoXEightBulkTailOwnerBytes(0));
static_assert(TwoXEightBulkTailTileBytes(1, 0) + TwoXEightBulkTailTileBytes(1, 1)
              == TwoXEightBulkTailOwnerBytes(1));
static_assert(TwoXEightBulkTailTileBytes(0, 1) == 2 * sizeof(float));
static_assert(TwoXEightBulkTailTileBytes(1, 1) == sizeof(float));
static_assert(TwoXEightOneWaveOwnerBytes(0, false) == 32ULL * MEBIBYTE);
static_assert(TwoXEightOneWaveOwnerBytes(0, false) * TWO_X_EIGHT_RANK_COUNT
              == TWO_X_EIGHT_BULK_MESSAGE_BYTES);
static_assert(TwoXEightOneWaveOwnerOffset(TWO_X_EIGHT_RANK_COUNT - 1, false)
                  + TwoXEightOneWaveOwnerBytes(TWO_X_EIGHT_RANK_COUNT - 1, false)
              == TWO_X_EIGHT_BULK_MESSAGE_BYTES);
static_assert(TwoXEightOneWaveOwnerOffset(TWO_X_EIGHT_RANK_COUNT - 1, true)
                  + TwoXEightOneWaveOwnerBytes(TWO_X_EIGHT_RANK_COUNT - 1, true)
              == TWO_X_EIGHT_BULK_TAIL_MESSAGE_BYTES);
static_assert(TwoXEightOneWaveOwnerBytes(0, true) == 25ULL * MEBIBYTE + sizeof(float));
static_assert(TwoXEightOneWaveOwnerBytes(1, true) == 25ULL * MEBIBYTE);
static_assert(TwoXEightOneWaveHalfBytes(0, false) == 256ULL * MEBIBYTE);
static_assert(TwoXEightOneWaveHalfBytes(0, true) == 200ULL * MEBIBYTE + 32);
static_assert(TwoXEightOneWaveLeafOffset(TWO_X_EIGHT_REDUCE_GROUP_SIZE - 1, 0, false)
                  + TwoXEightOneWaveOwnerBytes(0, false)
              == 256ULL * MEBIBYTE);
static_assert(TwoXEightOneWaveLeafOffset(TWO_X_EIGHT_RANK_COUNT - 1, 0, true)
                  + TwoXEightOneWaveOwnerBytes(0, true)
              == 200ULL * MEBIBYTE + 32);
static_assert(!EqualSizedRangesDisjoint(0x1000, 0x1000, 0x100));
static_assert(!EqualSizedRangesDisjoint(0x1000, 0x1080, 0x100));
static_assert(!EqualSizedRangesDisjoint(0x1080, 0x1000, 0x100));
static_assert(EqualSizedRangesDisjoint(0x1000, 0x1100, 0x100));
static_assert(EqualSizedRangesDisjoint(0x1100, 0x1000, 0x100));
static_assert(CHANNEL_NOTIFY_MASK == 1);
static_assert(PIPELINE_SLOT_BYTES % PIPELINE_SLOT_ALIGNMENT_BYTES == 0);

constexpr uint32_t MAX_CANONICAL_RANKS = static_cast<uint32_t>(MAX_RANK_SIZE);
constexpr uint32_t MAX_CANONICAL_GROUPS = 4;
constexpr uint32_t MAX_PEER_CHANNELS = MAX_CANONICAL_RANKS - 1;
constexpr uint32_t MAX_PEER_OPERATIONS_PER_CHANNEL = 2;
constexpr uint32_t MAX_STAGE_PEER_OPERATIONS = MAX_PEER_CHANNELS * MAX_PEER_OPERATIONS_PER_CHANNEL;
constexpr uint32_t MAX_STAGE_SCRATCH_SUBRANGES = 16;
constexpr uint32_t MAX_STAGE_SYNC_ACTIONS = 64;
constexpr uint32_t MAX_LOGICAL_SHARDS = MAX_CANONICAL_RANKS;
constexpr uint32_t MAX_SCRATCH_LAYOUT_RECORDS = 32;
constexpr uint32_t MAX_REGISTERED_STAGES = 64;
constexpr uint32_t MAX_SLAVE_THREADS = MAX_WORKERS - 1;
constexpr uint32_t INVALID_SCRATCH_SUBRANGE_INDEX = UINT32_MAX;
constexpr uint64_t RESOURCE_CONTEXT_MAGIC = 0x4343555231435458ULL; // "CCUR1CTX"
constexpr uint32_t RESOURCE_CONTEXT_SCHEMA_VERSION = 6;
constexpr uint32_t RESOURCE_CONTEXT_FRAMEWORK_VERSION = 1;
constexpr uint32_t INVALID_RESOURCE_CONTEXT_FRAMEWORK_VERSION = 0;
constexpr uint64_t MAX_SERIALIZED_RESOURCE_CONTEXT_BYTES = MEBIBYTE;

static_assert(static_cast<uint32_t>(CcuStageKernelId::COUNT) <= MAX_REGISTERED_STAGES);
static_assert(MAX_CANONICAL_GROUPS >= 4);
static_assert(MAX_PEER_CHANNELS == MAX_CANONICAL_RANKS - 1);
static_assert(MAX_STAGE_PEER_OPERATIONS >= MAX_PEER_CHANNELS);
static_assert(MAX_STAGE_SCRATCH_SUBRANGES <= MAX_CANONICAL_RANKS);
static_assert(MAX_STAGE_SYNC_ACTIONS >= MAX_STAGE_PEER_OPERATIONS * 2);
static_assert(MAX_STAGE_SCRATCH_SUBRANGES >= ACTIVE_WORKERS * PIPELINE_GENERATIONS);
static_assert(MAX_LOGICAL_SHARDS == MAX_CANONICAL_RANKS);

enum class FabricLayer : uint32_t { L0 = 0, L1, INVALID };

enum class DieEndpoint : uint32_t { DIE_0 = 0, DIE_1, CROSS_DIE, NOT_APPLICABLE };

struct StageLaunchSpec {
    CcuStageKernelId stageId = CcuStageKernelId::INVALID;
    DieEndpoint die = DieEndpoint::NOT_APPLICABLE;
    uint32_t worker = 0;
    uint32_t creditSlot = 0;
    uint32_t tile = 0;
    uint64_t flags = 0;
};

enum class PeerOperationKind : uint32_t {
    READ = 0,
    WRITE,
    READ_REDUCE,
    WRITE_REDUCE,
    LOCAL_COPY,
    LOCAL_REDUCE,
    INVALID
};

enum class MemoryRegion : uint32_t {
    INPUT = 0,
    OUTPUT,
    SCRATCH,
    HCCL_BUFFER,
    REMOTE_INPUT,
    REMOTE_OUTPUT,
    REMOTE_SCRATCH,
    INVALID
};

enum class OperationOwner : uint32_t { LOCAL_RANK = 0, CANONICAL_PEER, LOGICAL_SHARD, INVALID };

enum class ScratchUse : uint32_t { REDUCE_STAGING = 0, EXCHANGE_STAGING, PIPELINE_SLOT, FINALIZATION, INVALID };

enum class SyncActionKind : uint32_t { RECORD = 0, WAIT, INVALID };

enum class StateMachinePhase : uint32_t {
    PROBE = 0,
    REDUCE_SCATTER,
    REDUCE,
    EXCHANGE,
    ALL_GATHER,
    BROADCAST,
    COMPLETE,
    INVALID
};

struct CanonicalRankGroupMap {
    uint32_t rankCount = 0;
    uint32_t groupCount = 0;
    std::array<uint32_t, MAX_CANONICAL_RANKS> canonicalToPhysical{};
    std::array<uint32_t, MAX_CANONICAL_RANKS> physicalToCanonical{};
    std::array<uint32_t, MAX_CANONICAL_RANKS> groupByCanonicalRank{};
    std::array<uint32_t, MAX_CANONICAL_RANKS> groupLocalRank{};
};

struct CanonicalPeerSet {
    std::array<uint8_t, MAX_CANONICAL_RANKS> mask{};
    uint32_t count = 0;
};

struct PeerChannelResource {
    uint32_t canonicalPeer = INVALID_VALUE_RANKID;
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    FabricLayer layer = FabricLayer::INVALID;
    CommProtocol protocol = COMM_PROTOCOL_RESERVED;
    EndpointDesc localEndpoint{};
    EndpointDesc remoteEndpoint{};
    EndpointAttrDieId localEndpointDieId = UINT32_MAX;
    EndpointAttrDieId remoteEndpointDieId = UINT32_MAX;
    EndpointAttrLocation localEndpointLocation = UINT32_MAX;
    EndpointAttrLocation remoteEndpointLocation = UINT32_MAX;
    DieEndpoint localDie = DieEndpoint::NOT_APPLICABLE;
    DieEndpoint remoteDie = DieEndpoint::NOT_APPLICABLE;
    ChannelHandle handle = 0;
};

struct PeerOperation {
    PeerOperationKind kind = PeerOperationKind::INVALID;
    uint32_t channelSlot = 0;
    uint32_t canonicalPeer = INVALID_VALUE_RANKID;
    MemoryRegion sourceRegion = MemoryRegion::INVALID;
    MemoryRegion destinationRegion = MemoryRegion::INVALID;
    uint32_t sourceScratchSubrange = INVALID_SCRATCH_SUBRANGE_INDEX;
    uint32_t destinationScratchSubrange = INVALID_SCRATCH_SUBRANGE_INDEX;
    uint64_t sourceOffsetBytes = 0;
    uint64_t destinationOffsetBytes = 0;
    uint64_t transferBytes = 0;
    uint64_t elementCount = 0;
    OperationOwner owner = OperationOwner::INVALID;
    uint32_t ownerIndex = 0;
    uint32_t fixedReductionOrder = 0;
};

struct ScratchSubrange {
    uint64_t offsetBytes = 0;
    uint64_t sizeBytes = 0;
    ScratchUse use = ScratchUse::INVALID;
    uint32_t ownerLogicalShard = 0;
    uint32_t firstUseSequence = 0;
    uint32_t lastUseSequence = 0;
};

struct SyncAction {
    SyncActionKind kind = SyncActionKind::INVALID;
    uint32_t canonicalPeer = INVALID_VALUE_RANKID;
    uint32_t channelSlot = 0;
    uint32_t notifyIndex = 0;
    StateMachinePhase phase = StateMachinePhase::INVALID;
    uint32_t generation = 0;
    uint32_t sequenceOrder = 0;
};

struct StageSpec {
    CcuStageKernelId stageId = CcuStageKernelId::INVALID;
    uint32_t worker = 0;
    uint32_t generation = 0;
    FabricLayer layer = FabricLayer::INVALID;
    DieEndpoint die = DieEndpoint::NOT_APPLICABLE;
    CanonicalRankGroupMap rankMap{};
    std::array<PeerChannelResource, MAX_PEER_CHANNELS> peerChannels{};
    uint32_t peerChannelCount = 0;
    std::array<PeerOperation, MAX_STAGE_PEER_OPERATIONS> peerOperations{};
    uint32_t peerOperationCount = 0;
    std::array<ScratchSubrange, MAX_STAGE_SCRATCH_SUBRANGES> scratchSubranges{};
    uint32_t scratchSubrangeCount = 0;
    std::array<SyncAction, MAX_STAGE_SYNC_ACTIONS> syncActions{};
    uint32_t syncActionCount = 0;
};

struct PlanSpec {
    PlanId id = PlanId::INVALID;
    AlgorithmPackage package = ACTIVE_PACKAGE;
    TopologyKind topology = TopologyKind::UNSUPPORTED;
    MessageSizeKind messageSize = MessageSizeKind::UNSUPPORTED;
    uint64_t tileBytes = 0;
    uint64_t tileElementCount = 0;
    uint32_t logicalShardCount = 0;
    CcuStageKernelId firstStage = CcuStageKernelId::INVALID;
    uint32_t stageCount = 0;
};

struct DeterministicShard {
    uint32_t logicalShard = 0;
    uint32_t canonicalOwner = INVALID_VALUE_RANKID;
    uint64_t elementOffset = 0;
    uint64_t elementCount = 0;
    uint32_t fixedReductionOrder = 0;
};

struct ScratchLayoutRecord {
    uint32_t logicalShard = 0;
    ScratchUse use = ScratchUse::INVALID;
    uint32_t generation = 0;
    uint32_t slot = 0;
    uint64_t offsetBytes = 0;
    uint64_t sizeBytes = 0;
    uint64_t alignmentBytes = PIPELINE_SLOT_ALIGNMENT_BYTES;
    uint32_t firstUseStage = 0;
    uint32_t lastUseStage = 0;
};

struct RegisteredStageResource {
    CcuStageKernelId stageId = CcuStageKernelId::INVALID;
    uint32_t registrationOrder = 0;
    CcuKernelHandle handle = 0;
};

struct AddressToken {
    uint64_t address = 0;
    uint64_t token = 0;
};

struct InvocationMemory {
    AddressToken input{};
    AddressToken output{};
    AddressToken scratch{};
};

struct InvocationSlice {
    uint64_t dataOffsetBytes = 0;
    uint64_t elementCount = 0;
    uint64_t tileOffsetBytes = 0;
    uint64_t tileElementCount = 0;
};

struct InvocationControl {
    uint64_t generation = 0;
    uint64_t flags = 0;
};

#define ASSERT_POD_RECORD(type) \
    static_assert(std::is_standard_layout<type>::value, #type " must be standard-layout"); \
    static_assert(std::is_trivially_copyable<type>::value, #type " must be trivially copyable")

ASSERT_POD_RECORD(CanonicalRankGroupMap);
ASSERT_POD_RECORD(CanonicalPeerSet);
ASSERT_POD_RECORD(StageLaunchSpec);
ASSERT_POD_RECORD(PeerChannelResource);
ASSERT_POD_RECORD(PeerOperation);
ASSERT_POD_RECORD(ScratchSubrange);
ASSERT_POD_RECORD(SyncAction);
ASSERT_POD_RECORD(StageSpec);
ASSERT_POD_RECORD(PlanSpec);
ASSERT_POD_RECORD(DeterministicShard);
ASSERT_POD_RECORD(ScratchLayoutRecord);
ASSERT_POD_RECORD(RegisteredStageResource);
ASSERT_POD_RECORD(AddressToken);
ASSERT_POD_RECORD(InvocationMemory);
ASSERT_POD_RECORD(InvocationSlice);
ASSERT_POD_RECORD(InvocationControl);
#undef ASSERT_POD_RECORD

struct AlgorithmPlan {
    AlgorithmPackage package = ACTIVE_PACKAGE;
    PlanId id = PlanId::INVALID;
    TopologyKind topology = TopologyKind::UNSUPPORTED;
    MessageSizeKind messageSize = MessageSizeKind::UNSUPPORTED;
    AlgorithmFamily family = AlgorithmFamily::INVALID;
    TailPolicy tailPolicy = TailPolicy::INVALID;
    OptimizationTarget target = OptimizationTarget::LATENCY;
    ResourceProfileId resourceProfile = ResourceProfileId::INVALID;
    CcuKernelId kernelId = CcuKernelId::INVALID;
    uint32_t kernelSlot = 0;
    uint32_t streamCount = 0;
    uint32_t kernelHandleCount = 0;
    uint64_t tileBytes = 0;
    uint64_t scratchBytes = 0;
    bool implemented = false;
    uint64_t maxSliceBytes = MAX_DATA_SIZE;
    DeterminismPolicy determinism = DeterminismPolicy::FIXED_RANK_AND_SLICE_ORDER;
};

struct ResourceProfile {
    ResourceProfileId id = ResourceProfileId::INVALID;
    uint32_t threadNum = 1;
    uint32_t notifyNumPerThread = 1;
    uint32_t kernelCount = 0;
    bool acquireAllPeerChannels = true;
    CommEngine engine = CommEngine::COMM_ENGINE_CCU;
};

constexpr ResourceProfile MakeResourceProfile(ResourceProfileId profileId)
{
    switch (profileId) {
        case ResourceProfileId::PROFILE_4X1:
        case ResourceProfileId::PROFILE_8_PLUS_4:
        case ResourceProfileId::PROFILE_2X8:
            return ResourceProfile{
                profileId, 1, CELL_THREAD_NOTIFY_COUNT, 3, true, CommEngine::COMM_ENGINE_CCU};
        default:
            return ResourceProfile{};
    }
}

static_assert(MakeResourceProfile(ResourceProfileId::PROFILE_4X1).notifyNumPerThread == CELL_THREAD_NOTIFY_COUNT);
static_assert(MakeResourceProfile(ResourceProfileId::PROFILE_8_PLUS_4).notifyNumPerThread
              == CELL_THREAD_NOTIFY_COUNT);
static_assert(MakeResourceProfile(ResourceProfileId::PROFILE_2X8).notifyNumPerThread == CELL_THREAD_NOTIFY_COUNT);
static_assert(MakeResourceProfile(ResourceProfileId::INVALID).id == ResourceProfileId::INVALID);

struct ExecutionSlice {
    uint64_t offset = 0;
    uint64_t size = 0;
};

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64];
    // kernel函数
    void *kernelFunc;
    // KernelArg实例指针
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

namespace custom_detail {
template <typename T> void AppendScalar(std::vector<char> &data, const T &value)
{
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value, "only scalars may be serialized directly");
    const auto *begin = reinterpret_cast<const char *>(&value);
    data.insert(data.end(), begin, begin + sizeof(T));
}

template <typename T> bool ReadScalar(const std::vector<char> &data, size_t &offset, T &value)
{
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value, "only scalars may be deserialized directly");
    if (offset > data.size() || sizeof(T) > data.size() - offset) {
        return false;
    }
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

template <typename T> void AppendRecord(std::vector<char> &data, const T &value)
{
    static_assert(std::is_trivially_copyable<T>::value, "only trivially copyable records may be serialized");
    const auto *begin = reinterpret_cast<const char *>(&value);
    data.insert(data.end(), begin, begin + sizeof(T));
}

template <typename T> bool ReadRecord(const std::vector<char> &data, size_t &offset, T &value)
{
    static_assert(std::is_trivially_copyable<T>::value, "only trivially copyable records may be deserialized");
    if (offset > data.size() || sizeof(T) > data.size() - offset) {
        return false;
    }
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

inline uint32_t RankCountForTopology(TopologyKind topology)
{
    switch (topology) {
        case TopologyKind::TOPOLOGY_4X1:
            return 4;
        case TopologyKind::TOPOLOGY_8_PLUS_4:
            return 12;
        case TopologyKind::TOPOLOGY_2X8:
            return 16;
        default:
            return 0;
    }
}

inline uint32_t GroupCountForTopology(TopologyKind topology)
{
    switch (topology) {
        case TopologyKind::TOPOLOGY_4X1:
            return 4;
        case TopologyKind::TOPOLOGY_8_PLUS_4:
        case TopologyKind::TOPOLOGY_2X8:
            return 2;
        default:
            return 0;
    }
}

inline bool IsValidPackage(AlgorithmPackage package)
{
    return package == AlgorithmPackage::PACKAGE_A;
}

inline bool IsValidProfile(TopologyKind topology, ResourceProfileId profile)
{
    switch (topology) {
        case TopologyKind::TOPOLOGY_4X1:
            return profile == ResourceProfileId::PROFILE_4X1;
        case TopologyKind::TOPOLOGY_8_PLUS_4:
            return profile == ResourceProfileId::PROFILE_8_PLUS_4;
        case TopologyKind::TOPOLOGY_2X8:
            return profile == ResourceProfileId::PROFILE_2X8;
        default:
            return false;
    }
}

inline bool IsValidLayer(FabricLayer layer)
{
    return layer == FabricLayer::L0 || layer == FabricLayer::L1;
}

inline bool IsValidDie(DieEndpoint die)
{
    return die == DieEndpoint::DIE_0 || die == DieEndpoint::DIE_1 || die == DieEndpoint::CROSS_DIE;
}

inline bool IsSupportedCcuProtocol(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBC_TP;
}

inline bool EndpointDieMatches(EndpointAttrDieId dieId, DieEndpoint die)
{
    return (dieId == 0 && die == DieEndpoint::DIE_0) || (dieId == 1 && die == DieEndpoint::DIE_1);
}

inline FabricLayer ExpectedPeerLayer(TopologyKind topology, const CanonicalRankGroupMap &rankMap,
    uint32_t localCanonicalRank, uint32_t peerCanonicalRank)
{
    if (localCanonicalRank >= rankMap.rankCount || peerCanonicalRank >= rankMap.rankCount) {
        return FabricLayer::INVALID;
    }
    if (topology == TopologyKind::TOPOLOGY_4X1) {
        return FabricLayer::L1;
    }
    return rankMap.groupByCanonicalRank[localCanonicalRank] == rankMap.groupByCanonicalRank[peerCanonicalRank]
               ? FabricLayer::L0
               : FabricLayer::L1;
}

inline bool IsValidRankMap(const CanonicalRankGroupMap &rankMap, TopologyKind topology)
{
    const uint32_t expectedRankCount = RankCountForTopology(topology);
    const uint32_t expectedGroupCount = GroupCountForTopology(topology);
    if (rankMap.rankCount != expectedRankCount || rankMap.groupCount != expectedGroupCount) {
        return false;
    }

    std::array<bool, MAX_CANONICAL_RANKS> physicalSeen{};
    std::array<bool, MAX_CANONICAL_RANKS * MAX_CANONICAL_GROUPS> groupLocalSeen{};
    std::array<bool, MAX_CANONICAL_GROUPS> groupSeen{};
    for (uint32_t canonicalRank = 0; canonicalRank < rankMap.rankCount; ++canonicalRank) {
        const uint32_t physicalRank = rankMap.canonicalToPhysical[canonicalRank];
        const uint32_t group = rankMap.groupByCanonicalRank[canonicalRank];
        const uint32_t groupLocalRank = rankMap.groupLocalRank[canonicalRank];
        const uint32_t expectedGroupSize = topology == TopologyKind::TOPOLOGY_8_PLUS_4
                                               ? (group == 0 ? 8 : 4)
                                               : expectedRankCount / expectedGroupCount;
        if (physicalRank >= rankMap.rankCount || physicalSeen[physicalRank] || group >= rankMap.groupCount
            || groupLocalRank >= expectedGroupSize) {
            return false;
        }
        const uint32_t groupLocalKey = group * MAX_CANONICAL_RANKS + groupLocalRank;
        if (groupLocalKey >= groupLocalSeen.size() || groupLocalSeen[groupLocalKey]) {
            return false;
        }
        physicalSeen[physicalRank] = true;
        groupLocalSeen[groupLocalKey] = true;
        groupSeen[group] = true;
        if (rankMap.physicalToCanonical[physicalRank] != canonicalRank) {
            return false;
        }
    }
    for (uint32_t group = 0; group < rankMap.groupCount; ++group) {
        if (!groupSeen[group]) {
            return false;
        }
    }
    return true;
}

inline bool BuildCompatibilityRankMap(TopologyKind topology, CanonicalRankGroupMap &rankMap)
{
    rankMap = CanonicalRankGroupMap{};
    rankMap.rankCount = RankCountForTopology(topology);
    rankMap.groupCount = GroupCountForTopology(topology);
    if (rankMap.rankCount == 0 || rankMap.groupCount == 0) {
        return false;
    }
    for (uint32_t rank = 0; rank < rankMap.rankCount; ++rank) {
        uint32_t group = 0;
        uint32_t groupLocalRank = rank;
        if (topology == TopologyKind::TOPOLOGY_4X1) {
            group = rank;
            groupLocalRank = 0;
        } else if (rank >= 8) {
            group = 1;
            groupLocalRank = rank - 8;
        }
        rankMap.canonicalToPhysical[rank] = rank;
        rankMap.physicalToCanonical[rank] = rank;
        rankMap.groupByCanonicalRank[rank] = group;
        rankMap.groupLocalRank[rank] = groupLocalRank;
    }
    return true;
}

inline void AppendCommBuffer(std::vector<char> &data, const CommBuffer &buffer)
{
    static_assert(sizeof(uintptr_t) <= sizeof(uint64_t), "serialized addresses require at most 64 bits");
    const uint64_t address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buffer.addr));
    AppendScalar(data, address);
    AppendScalar(data, buffer.size);
}

inline bool ReadCommBuffer(const std::vector<char> &data, size_t &offset, CommBuffer &buffer)
{
    uint64_t address = 0;
    if (!ReadScalar(data, offset, address) || !ReadScalar(data, offset, buffer.size)) {
        return false;
    }
    buffer.addr = reinterpret_cast<void *>(static_cast<uintptr_t>(address));
    return true;
}

inline void AppendRankMap(std::vector<char> &data, const CanonicalRankGroupMap &rankMap)
{
    AppendScalar(data, rankMap.rankCount);
    AppendScalar(data, rankMap.groupCount);
    for (uint32_t rank = 0; rank < rankMap.rankCount; ++rank) {
        AppendScalar(data, rankMap.canonicalToPhysical[rank]);
        AppendScalar(data, rankMap.physicalToCanonical[rank]);
        AppendScalar(data, rankMap.groupByCanonicalRank[rank]);
        AppendScalar(data, rankMap.groupLocalRank[rank]);
    }
}

inline bool ReadRankMap(const std::vector<char> &data, size_t &offset, CanonicalRankGroupMap &rankMap)
{
    if (!ReadScalar(data, offset, rankMap.rankCount) || !ReadScalar(data, offset, rankMap.groupCount)
        || rankMap.rankCount > MAX_CANONICAL_RANKS || rankMap.groupCount > MAX_CANONICAL_GROUPS) {
        return false;
    }
    for (uint32_t rank = 0; rank < rankMap.rankCount; ++rank) {
        if (!ReadScalar(data, offset, rankMap.canonicalToPhysical[rank])
            || !ReadScalar(data, offset, rankMap.physicalToCanonical[rank])
            || !ReadScalar(data, offset, rankMap.groupByCanonicalRank[rank])
            || !ReadScalar(data, offset, rankMap.groupLocalRank[rank])) {
            return false;
        }
    }
    return true;
}

inline void AppendPeerChannel(std::vector<char> &data, const PeerChannelResource &peer)
{
    AppendScalar(data, peer.canonicalPeer);
    AppendScalar(data, peer.remoteRank);
    AppendScalar(data, peer.layer);
    AppendScalar(data, peer.protocol);
    AppendRecord(data, peer.localEndpoint);
    AppendRecord(data, peer.remoteEndpoint);
    AppendScalar(data, peer.localEndpointDieId);
    AppendScalar(data, peer.remoteEndpointDieId);
    AppendScalar(data, peer.localEndpointLocation);
    AppendScalar(data, peer.remoteEndpointLocation);
    AppendScalar(data, peer.localDie);
    AppendScalar(data, peer.remoteDie);
    AppendScalar(data, peer.handle);
}

inline bool ReadPeerChannel(const std::vector<char> &data, size_t &offset, PeerChannelResource &peer)
{
    return ReadScalar(data, offset, peer.canonicalPeer) && ReadScalar(data, offset, peer.remoteRank)
           && ReadScalar(data, offset, peer.layer) && ReadScalar(data, offset, peer.protocol)
           && ReadRecord(data, offset, peer.localEndpoint) && ReadRecord(data, offset, peer.remoteEndpoint)
           && ReadScalar(data, offset, peer.localEndpointDieId) && ReadScalar(data, offset, peer.remoteEndpointDieId)
           && ReadScalar(data, offset, peer.localEndpointLocation)
           && ReadScalar(data, offset, peer.remoteEndpointLocation) && ReadScalar(data, offset, peer.localDie)
           && ReadScalar(data, offset, peer.remoteDie) && ReadScalar(data, offset, peer.handle);
}

inline void AppendRegisteredStage(std::vector<char> &data, const RegisteredStageResource &stage)
{
    AppendScalar(data, stage.stageId);
    AppendScalar(data, stage.registrationOrder);
    AppendScalar(data, stage.handle);
}

inline bool ReadRegisteredStage(const std::vector<char> &data, size_t &offset, RegisteredStageResource &stage)
{
    return ReadScalar(data, offset, stage.stageId) && ReadScalar(data, offset, stage.registrationOrder)
           && ReadScalar(data, offset, stage.handle);
}

} // namespace custom_detail

struct AlgResourceCtx {
    uint64_t magic = RESOURCE_CONTEXT_MAGIC;
    uint32_t schemaVersion = RESOURCE_CONTEXT_SCHEMA_VERSION;
    uint32_t frameworkVersion = RESOURCE_CONTEXT_FRAMEWORK_VERSION;
    AlgorithmPackage package = ACTIVE_PACKAGE;
    TopologyKind topology = TopologyKind::UNSUPPORTED;
    MessageSizeKind messageSize = MessageSizeKind::UNSUPPORTED;
    AlgorithmFamily family = AlgorithmFamily::INVALID;
    TailPolicy tailPolicy = TailPolicy::INVALID;
    ResourceProfileId resourceProfile = ResourceProfileId::INVALID;
    uint32_t workerCount = 0;
    uint32_t requiredPeerCount = 0;
    std::array<uint8_t, MAX_CANONICAL_RANKS> requiredPeerMask{};
    uint32_t kernelHandleCount = 0;
    uint64_t tileBytes = 0;
    uint64_t scratchBytes = 0;
    uint64_t streamKey = 0;
    uint64_t invocationSequence = 0;
    ThreadHandle mainThread = 0;
    uint32_t localPhysicalRank = INVALID_VALUE_RANKID;
    uint32_t localCanonicalRank = INVALID_VALUE_RANKID;
    uint32_t localGroup = INVALID_VALUE_RANKID;
    uint32_t localGroupRank = INVALID_VALUE_RANKID;
    CommBuffer hcclBuffer{};
    CanonicalRankGroupMap rankMap{};
    std::array<ThreadHandle, MAX_SLAVE_THREADS> slaveThreads{};
    uint32_t slaveThreadCount = 0;
    std::array<PeerChannelResource, MAX_PEER_CHANNELS> peerChannels{};
    uint32_t peerChannelCount = 0;
    std::array<RegisteredStageResource, MAX_REGISTERED_STAGES> registeredStages{};
    uint32_t registeredStageCount = 0;

    // Temporary, non-serialized scaffold views. Canonical state is held only in the fixed fields above.
    ThreadHandle ccuThread = 0;        ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer{};          ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;

    HcclResult SerializeChecked(std::vector<char> &out) const
    {
        if (ValidateLaunchReady() != HCCL_SUCCESS || !LegacyViewsMatchCanonical()) {
            out.clear();
            return HCCL_E_PARA;
        }

        std::vector<char> encoded;
        custom_detail::AppendScalar(encoded, magic);
        custom_detail::AppendScalar(encoded, schemaVersion);
        custom_detail::AppendScalar(encoded, frameworkVersion);
        custom_detail::AppendScalar(encoded, package);
        custom_detail::AppendScalar(encoded, topology);
        custom_detail::AppendScalar(encoded, messageSize);
        custom_detail::AppendScalar(encoded, family);
        custom_detail::AppendScalar(encoded, tailPolicy);
        custom_detail::AppendScalar(encoded, resourceProfile);
        custom_detail::AppendScalar(encoded, workerCount);
        custom_detail::AppendScalar(encoded, requiredPeerCount);
        for (uint8_t required : requiredPeerMask) {
            custom_detail::AppendScalar(encoded, required);
        }
        custom_detail::AppendScalar(encoded, kernelHandleCount);
        custom_detail::AppendScalar(encoded, tileBytes);
        custom_detail::AppendScalar(encoded, scratchBytes);
        custom_detail::AppendScalar(encoded, streamKey);
        custom_detail::AppendScalar(encoded, invocationSequence);
        custom_detail::AppendScalar(encoded, mainThread);
        custom_detail::AppendScalar(encoded, localPhysicalRank);
        custom_detail::AppendScalar(encoded, localCanonicalRank);
        custom_detail::AppendScalar(encoded, localGroup);
        custom_detail::AppendScalar(encoded, localGroupRank);
        custom_detail::AppendCommBuffer(encoded, hcclBuffer);
        custom_detail::AppendRankMap(encoded, rankMap);
        custom_detail::AppendScalar(encoded, slaveThreadCount);
        for (uint32_t index = 0; index < slaveThreadCount; ++index) {
            custom_detail::AppendScalar(encoded, slaveThreads[index]);
        }
        custom_detail::AppendScalar(encoded, peerChannelCount);
        for (uint32_t index = 0; index < peerChannelCount; ++index) {
            custom_detail::AppendPeerChannel(encoded, peerChannels[index]);
        }
        custom_detail::AppendScalar(encoded, registeredStageCount);
        for (uint32_t index = 0; index < registeredStageCount; ++index) {
            custom_detail::AppendRegisteredStage(encoded, registeredStages[index]);
        }
        if (encoded.size() > MAX_SERIALIZED_RESOURCE_CONTEXT_BYTES) {
            out.clear();
            return HCCL_E_INTERNAL;
        }
        out.swap(encoded);
        return HCCL_SUCCESS;
    }

    // Temporary scaffold wrapper. Canonical state is the only serialization source.
    std::vector<char> Serialize() const
    {
        std::vector<char> result;
        if (SerializeChecked(result) != HCCL_SUCCESS) {
            result.clear();
        }
        return result;
    }

    HcclResult DeSerialize(const std::vector<char> &data)
    {
        if (data.empty() || data.size() > MAX_SERIALIZED_RESOURCE_CONTEXT_BYTES) {
            return InvalidateAndReturn(HCCL_E_PARA);
        }
        AlgResourceCtx decoded;
        size_t offset = 0;
        if (!custom_detail::ReadScalar(data, offset, decoded.magic) || decoded.magic != RESOURCE_CONTEXT_MAGIC
            || !custom_detail::ReadScalar(data, offset, decoded.schemaVersion)
            || decoded.schemaVersion != RESOURCE_CONTEXT_SCHEMA_VERSION
            || !custom_detail::ReadScalar(data, offset, decoded.frameworkVersion)
            || decoded.frameworkVersion != RESOURCE_CONTEXT_FRAMEWORK_VERSION) {
            return InvalidateAndReturn(HCCL_E_NOT_SUPPORT);
        }
        if (!custom_detail::ReadScalar(data, offset, decoded.package)
            || !custom_detail::ReadScalar(data, offset, decoded.topology)
            || !custom_detail::ReadScalar(data, offset, decoded.messageSize)
            || !custom_detail::ReadScalar(data, offset, decoded.family)
            || !custom_detail::ReadScalar(data, offset, decoded.tailPolicy)
            || !custom_detail::ReadScalar(data, offset, decoded.resourceProfile)
            || !custom_detail::ReadScalar(data, offset, decoded.workerCount)
            || !custom_detail::ReadScalar(data, offset, decoded.requiredPeerCount)
            || !ReadRequiredPeerMask(data, offset, decoded)
            || !custom_detail::ReadScalar(data, offset, decoded.kernelHandleCount)
            || !custom_detail::ReadScalar(data, offset, decoded.tileBytes)
            || !custom_detail::ReadScalar(data, offset, decoded.scratchBytes)
            || !custom_detail::ReadScalar(data, offset, decoded.streamKey)
            || !custom_detail::ReadScalar(data, offset, decoded.invocationSequence)
            || !custom_detail::ReadScalar(data, offset, decoded.mainThread)
            || !custom_detail::ReadScalar(data, offset, decoded.localPhysicalRank)
            || !custom_detail::ReadScalar(data, offset, decoded.localCanonicalRank)
            || !custom_detail::ReadScalar(data, offset, decoded.localGroup)
            || !custom_detail::ReadScalar(data, offset, decoded.localGroupRank)
            || !custom_detail::ReadCommBuffer(data, offset, decoded.hcclBuffer)
            || !custom_detail::ReadRankMap(data, offset, decoded.rankMap)
            || !ReadCanonicalResources(data, offset, decoded) || offset != data.size()
            || decoded.ValidateCanonicalStructure() != HCCL_SUCCESS || decoded.ValidateLaunchReady() != HCCL_SUCCESS
            || decoded.RebuildLegacyViewsUnchecked() != HCCL_SUCCESS) {
            return InvalidateAndReturn(HCCL_E_PARA);
        }
        *this = decoded;
        return HCCL_SUCCESS;
    }

    HcclResult ImportLegacyViews()
    {
        return RebuildLegacyViews();
    }

    HcclResult RebuildLegacyViews()
    {
        return ValidateCanonicalStructure() == HCCL_SUCCESS ? RebuildLegacyViewsUnchecked() : HCCL_E_PARA;
    }

    HcclResult ValidateCanonicalStructure() const
    {
        return HasStructurallyValidCanonicalState() ? HCCL_SUCCESS : HCCL_E_PARA;
    }

    HcclResult ValidateLaunchReady() const
    {
        return IsLaunchReadyCanonicalState() ? HCCL_SUCCESS : HCCL_E_PARA;
    }

private:
    static bool ReadRequiredPeerMask(const std::vector<char> &data, size_t &offset, AlgResourceCtx &decoded)
    {
        for (uint8_t &required : decoded.requiredPeerMask) {
            if (!custom_detail::ReadScalar(data, offset, required)) {
                return false;
            }
        }
        return true;
    }

    static bool ReadCanonicalResources(const std::vector<char> &data, size_t &offset, AlgResourceCtx &decoded)
    {
        if (!custom_detail::ReadScalar(data, offset, decoded.slaveThreadCount)
            || decoded.slaveThreadCount > decoded.slaveThreads.size()) {
            return false;
        }
        for (uint32_t index = 0; index < decoded.slaveThreadCount; ++index) {
            if (!custom_detail::ReadScalar(data, offset, decoded.slaveThreads[index])) {
                return false;
            }
        }
        if (!custom_detail::ReadScalar(data, offset, decoded.peerChannelCount)
            || decoded.peerChannelCount > decoded.peerChannels.size()) {
            return false;
        }
        for (uint32_t index = 0; index < decoded.peerChannelCount; ++index) {
            if (!custom_detail::ReadPeerChannel(data, offset, decoded.peerChannels[index])) {
                return false;
            }
        }
        if (!custom_detail::ReadScalar(data, offset, decoded.registeredStageCount)
            || decoded.registeredStageCount > decoded.registeredStages.size()) {
            return false;
        }
        for (uint32_t index = 0; index < decoded.registeredStageCount; ++index) {
            if (!custom_detail::ReadRegisteredStage(data, offset, decoded.registeredStages[index])) {
                return false;
            }
        }
        return true;
    }

    bool LegacyKernelsMatchCanonicalStages() const
    {
        if (ccuKernels.empty()) {
            return true;
        }
        if (registeredStageCount == 0 || ccuKernels.size() != registeredStageCount) {
            return false;
        }
        for (uint32_t index = 0; index < registeredStageCount; ++index) {
            if (ccuKernels[index] != registeredStages[index].handle) {
                return false;
            }
        }
        return true;
    }

    bool LegacyViewsMatchCanonical() const
    {
        if ((ccuThread != 0 && ccuThread != mainThread)
            || ((localBuffer.addr != nullptr || localBuffer.size != 0)
                && (localBuffer.addr != hcclBuffer.addr || localBuffer.size != hcclBuffer.size))) {
            return false;
        }
        if (!threads.empty()) {
            if (threads.size() != slaveThreadCount + 1 || threads.front() != mainThread) {
                return false;
            }
            for (uint32_t index = 0; index < slaveThreadCount; ++index) {
                if (threads[index + 1] != slaveThreads[index]) {
                    return false;
                }
            }
        }
        return LegacyKernelsMatchCanonicalStages();
    }

    bool HasStructurallyValidCanonicalState() const
    {
        const CellDescriptor *cell = FindCellDescriptor(package, topology, messageSize);
        if (magic != RESOURCE_CONTEXT_MAGIC || schemaVersion != RESOURCE_CONTEXT_SCHEMA_VERSION
            || frameworkVersion != RESOURCE_CONTEXT_FRAMEWORK_VERSION || !custom_detail::IsValidPackage(package)
            || !custom_detail::IsValidProfile(topology, resourceProfile) || streamKey == 0 || mainThread == 0
            || hcclBuffer.addr == nullptr || hcclBuffer.size == 0 || !custom_detail::IsValidRankMap(rankMap, topology)
            || slaveThreadCount > slaveThreads.size() || slaveThreadCount + 1 > MAX_WORKERS
            || peerChannelCount > peerChannels.size() || registeredStageCount > registeredStages.size()
            || cell == nullptr || !cell->implemented || family != cell->family || tailPolicy != cell->tailPolicy
            || resourceProfile != cell->resourceProfile || workerCount != cell->streamCount
            || kernelHandleCount != cell->kernelHandleCount || tileBytes != cell->tileBytes
            || scratchBytes != cell->scratchBytes || hcclBuffer.size < scratchBytes
            || requiredPeerCount == 0 || requiredPeerCount > rankMap.rankCount - 1 || !HasValidRequiredPeers()
            || !HasValidLocalIdentity()) {
            return false;
        }
        return HasValidRuntimeHandles() && HasValidThreads() && HasValidPeerChannels() && HasValidRegisteredStages();
    }

    bool IsLaunchReadyCanonicalState() const
    {
        return HasStructurallyValidCanonicalState() && mainThread != 0 && slaveThreadCount + 1 == workerCount
               && peerChannelCount == requiredPeerCount && registeredStageCount == kernelHandleCount;
    }

    bool HasValidLocalIdentity() const
    {
        return localPhysicalRank < rankMap.rankCount && localCanonicalRank < rankMap.rankCount
               && rankMap.physicalToCanonical[localPhysicalRank] == localCanonicalRank
               && rankMap.canonicalToPhysical[localCanonicalRank] == localPhysicalRank
               && localGroup == rankMap.groupByCanonicalRank[localCanonicalRank]
               && localGroupRank == rankMap.groupLocalRank[localCanonicalRank];
    }

    bool HasValidThreads() const
    {
        for (uint32_t index = 0; index < slaveThreadCount; ++index) {
            if (slaveThreads[index] == 0 || slaveThreads[index] == mainThread) {
                return false;
            }
            for (uint32_t prior = 0; prior < index; ++prior) {
                if (slaveThreads[index] == slaveThreads[prior]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool HasValidRequiredPeers() const
    {
        uint32_t count = 0;
        for (uint32_t canonical = 0; canonical < requiredPeerMask.size(); ++canonical) {
            const uint8_t required = requiredPeerMask[canonical];
            if (required > 1 || (canonical >= rankMap.rankCount && required != 0)
                || (canonical == localCanonicalRank && required != 0)) {
                return false;
            }
            count += required;
        }
        return count == requiredPeerCount;
    }

    bool HasValidRuntimeHandles() const
    {
        return workerCount == 1 ? mainThread != 0
                                : workerCount == 2 && mainThread != 0 && slaveThreads[0] != 0
                                      && slaveThreads[0] != mainThread;
    }

    bool HasValidPeerChannels() const
    {
        if (peerChannelCount != requiredPeerCount) {
            return false;
        }
        std::array<bool, MAX_CANONICAL_RANKS> peerSeen{};
        for (uint32_t index = 0; index < peerChannelCount; ++index) {
            const PeerChannelResource &peer = peerChannels[index];
            if (peer.handle == 0 || peer.canonicalPeer >= rankMap.rankCount || peer.remoteRank >= rankMap.rankCount
                || peer.canonicalPeer == localCanonicalRank || peer.remoteRank == localPhysicalRank
                || rankMap.physicalToCanonical[peer.remoteRank] != peer.canonicalPeer
                || !custom_detail::IsValidLayer(peer.layer) || !custom_detail::IsValidDie(peer.localDie)
                || peer.remoteDie != DieEndpoint::NOT_APPLICABLE || peer.remoteEndpointDieId != UINT32_MAX
                || !custom_detail::IsSupportedCcuProtocol(peer.protocol)
                || peer.localEndpoint.protocol != peer.protocol || peer.remoteEndpoint.protocol != peer.protocol
                || peer.localEndpoint.commAddr.type == COMM_ADDR_TYPE_RESERVED
                || peer.remoteEndpoint.commAddr.type == COMM_ADDR_TYPE_RESERVED
                || peer.localEndpoint.loc.locType != ENDPOINT_LOC_TYPE_DEVICE
                || peer.remoteEndpoint.loc.locType != ENDPOINT_LOC_TYPE_DEVICE
                || peer.localEndpointLocation == UINT32_MAX || peer.remoteEndpointLocation == UINT32_MAX
                || !custom_detail::EndpointDieMatches(peer.localEndpointDieId, peer.localDie)
                || peer.layer
                       != custom_detail::ExpectedPeerLayer(topology, rankMap, localCanonicalRank, peer.canonicalPeer)
                || requiredPeerMask[peer.canonicalPeer] == 0 || peerSeen[peer.canonicalPeer]) {
                return false;
            }
            peerSeen[peer.canonicalPeer] = true;
            for (uint32_t prior = 0; prior < index; ++prior) {
                if (peer.canonicalPeer == peerChannels[prior].canonicalPeer
                    || peer.remoteRank == peerChannels[prior].remoteRank || peer.handle == peerChannels[prior].handle) {
                    return false;
                }
            }
        }
        for (uint32_t canonical = 0; canonical < rankMap.rankCount; ++canonical) {
            if ((requiredPeerMask[canonical] != 0) != peerSeen[canonical]) {
                return false;
            }
        }
        return true;
    }

    bool HasValidRegisteredStages() const
    {
        for (uint32_t index = 0; index < registeredStageCount; ++index) {
            const RegisteredStageResource &stage = registeredStages[index];
            if (stage.handle == 0 || stage.registrationOrder != index
                || static_cast<uint32_t>(stage.stageId) >= static_cast<uint32_t>(CcuStageKernelId::COUNT)) {
                return false;
            }
            for (uint32_t prior = 0; prior < index; ++prior) {
                if (stage.stageId == registeredStages[prior].stageId
                    || stage.handle == registeredStages[prior].handle) {
                    return false;
                }
            }
        }
        return true;
    }

    HcclResult RebuildLegacyViewsUnchecked()
    {
        ccuThread = mainThread;
        localBuffer = hcclBuffer;
        threads.clear();
        threads.reserve(slaveThreadCount + 1);
        threads.push_back(ccuThread);
        for (uint32_t index = 0; index < slaveThreadCount; ++index) {
            threads.push_back(slaveThreads[index]);
        }
        ccuKernels.clear();
        ccuKernels.reserve(registeredStageCount);
        for (uint32_t index = 0; index < registeredStageCount; ++index) {
            ccuKernels.push_back(registeredStages[index].handle);
        }
        return HCCL_SUCCESS;
    }

    HcclResult InvalidateAndReturn(HcclResult result)
    {
        *this = AlgResourceCtx{};
        magic = 0;
        schemaVersion = 0;
        frameworkVersion = INVALID_RESOURCE_CONTEXT_FRAMEWORK_VERSION;
        return result;
    }
};

#endif // OPS_HCCL_CUSTOM_H
