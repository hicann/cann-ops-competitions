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

#include <array>
#include <cstdint>
#include <type_traits>

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "common.h"

constexpr uint32_t SCATTER_SERVER_NUM = 2;
constexpr uint32_t SCATTER_RANKS_PER_SERVER = 8;
constexpr uint32_t SCATTER_EXPECTED_RANK_SIZE = SCATTER_SERVER_NUM * SCATTER_RANKS_PER_SERVER;
constexpr uint32_t SCATTER_MESH_WORKER_COUNT = SCATTER_RANKS_PER_SERVER - 1;
constexpr uint32_t SCATTER_SMALL_MESH_WORKER_COUNT = 4;
constexpr uint32_t SCATTER_THREAD_COUNT = SCATTER_MESH_WORKER_COUNT + 1;
constexpr uint32_t SCATTER_CHANNEL_COUNT = SCATTER_EXPECTED_RANK_SIZE - 1;
constexpr uint32_t SCATTER_CHANNEL_NOTIFY_NUM = 2;
// The scored large cases have at most a 32 MiB block per rank. One wave keeps
// helper forwarding overlapped with the helper's local Mesh receive while
// avoiding a second set of transfer and channel-notify tasks.
constexpr uint64_t SCATTER_PIPELINE_CHUNK_BYTES = 32ULL * 1024 * 1024;
// Slot 0 holds one complete chunk. Slot 1 starts at 32 MiB and holds either a
// 4/11 helper prefix or a 7/11 root suffix, so 64 MiB is a simple safe bound.
constexpr uint64_t SCATTER_REQUIRED_CCL_BYTES = 64ULL * 1024 * 1024;
// The decision is based on the root's complete send buffer, not one rank's block.
constexpr uint64_t SCATTER_SMALL_TOTAL_BYTES = 1ULL * 1024 * 1024;
// A dedicated graph for the scored 512 KiB aggregate case coalesces the
// remote half into four parallel two-block CLOS transfers, then relays the
// second block of each pair over remote Mesh.
constexpr uint64_t SCATTER_BULK_RELAY_TOTAL_BYTES = 512ULL * 1024;
// Helper traffic ratio. 4/11 balances (1+x)B/M against (8-7x)B/(4M).
constexpr uint64_t SCATTER_HELPER_RATIO_NUMERATOR = 4;
constexpr uint64_t SCATTER_HELPER_RATIO_DENOMINATOR = 11;
constexpr uint32_t SCATTER_RESOURCE_MAGIC = 0x53435436U; // "SCT6"
constexpr uint32_t SCATTER_RESOURCE_VERSION = 1;
constexpr uint32_t SCATTER_HOST_RESOURCE_MAGIC = 0x53434836U; // "SCH6"

typedef struct {
    void *addr = nullptr;
    uint64_t size = 0;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    uint32_t magic = SCATTER_RESOURCE_MAGIC;
    uint32_t version = SCATTER_RESOURCE_VERSION;
    ThreadHandle aicpuThread = 0; ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer;       ///< 本端HCCL通信内存
    std::array<ThreadHandle, SCATTER_THREAD_COUNT> threads{};
    // Direct rank indexing removes the per-invocation channel-table rebuild.
    // The self entry remains invalid because no self channel is acquired.
    std::array<ChannelInfo, SCATTER_EXPECTED_RANK_SIZE> channels{};

};

// CPU EngineCtx is scoped by (communicator, tag). It is the warm-path index
// for immutable rank/resource metadata and for exports whose source handles
// remain valid for the communicator lifetime.
struct HostResourceCtx {
    uint32_t magic = SCATTER_HOST_RESOURCE_MAGIC;
    uint32_t version = SCATTER_RESOURCE_VERSION;
    void *deviceResCtx = nullptr;
    uint64_t deviceResCtxSize = 0;
    ThreadHandle aicpuThreadOnCpu = 0;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
};

static_assert(std::is_trivially_copyable<CommBuffer>::value, "CommBuffer must be byte-copyable");
static_assert(std::is_trivially_copyable<ChannelInfo>::value, "ChannelInfo must be byte-copyable");
static_assert(std::is_trivially_copyable<AlgResourceCtx>::value, "AlgResourceCtx must be byte-copyable");
static_assert(std::is_standard_layout<AlgResourceCtx>::value, "AlgResourceCtx must have a stable layout");
static_assert(std::is_trivially_copyable<HostResourceCtx>::value, "HostResourceCtx must be byte-copyable");
static_assert(std::is_standard_layout<HostResourceCtx>::value, "HostResourceCtx must have a stable layout");
static_assert(sizeof(void *) == sizeof(uint64_t), "Scatter resource ABI requires LP64");
static_assert(sizeof(CommBuffer) == 16, "Unexpected CommBuffer ABI");
static_assert(sizeof(ChannelInfo) == 32, "Unexpected ChannelInfo ABI");
static_assert(sizeof(AlgResourceCtx) == 608, "Unexpected AlgResourceCtx ABI");
static_assert(sizeof(HostResourceCtx) == 40, "Unexpected HostResourceCtx ABI");

#endif // OPS_HCCL_CUSTOM_H
