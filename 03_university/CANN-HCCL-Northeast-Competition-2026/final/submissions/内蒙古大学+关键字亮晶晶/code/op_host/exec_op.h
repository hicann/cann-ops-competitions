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

#include <vector>

#include "common.h"

struct AlgResourceCtx;

namespace ops_hccl {
// This is a capacity-model starting point, not a proof about the simulator.
// h=2, R=9, k=4 => total relay/helper=5/6 S. With fanout=2,
// each of four targets is relayed by 5/12 S.
constexpr uint64_t SCATTER_WINDOW_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t SCATTER_PARALLEL_THRESHOLD = 64ULL * 1024ULL * 1024ULL;
// V23 safe small-message path is deliberately narrow: it covers the 512 KiB performance class
// without changing medium/large-message scheduling.
constexpr uint64_t SCATTER_SMALL_FAST_THRESHOLD = 1ULL * 1024ULL * 1024ULL;
constexpr uint32_t SCATTER_SMALL_FAST_THREADS = 2U;
constexpr uint64_t SCATTER_RELAY_NUM = 5ULL;
constexpr uint64_t SCATTER_RELAY_DEN = 12ULL;
constexpr uint32_t SCATTER_RELAY_FANOUT = 2U;
constexpr uint32_t SCATTER_RELAY_STAGES = 4U;
constexpr uint32_t SCATTER_MAX_THREADS = 4U;
// P4 4x3 root only: each path-grouped RelayDirect kernel and each local helper
// may own an independent TS lane.  Keep ordinary Direct paths on the proven
// four-lane limit; the existing 2x8 Hub already proves that up to eight TS
// threads are a supported resource shape in this submission.
constexpr uint32_t SCATTER_RELAY_ROOT_MAX_THREADS = 8U;
// A2 4x3 streaming: four relay stages reuse two ping-pong scratch slots.
// Direct/non-Relay schedules keep the proven join layout starting at notify 2.
constexpr uint32_t SCATTER_JOIN_BASE = 2U;
// Relay needs one completion notify per (pair, ping-pong slot). With fanout=2
// and two slots this is notify [0,3]. Relay joins therefore start at 4.
// Reusing only one notify per pair would enqueue stage0+stage1 RECORDs before
// the first stage2 WAIT and is rejected by CheckerV3 as a many-to-one conflict.
constexpr uint32_t SCATTER_RELAY_SLOT_COUNT = 2U;
constexpr uint32_t SCATTER_RELAY_JOIN_BASE = SCATTER_RELAY_FANOUT * SCATTER_RELAY_SLOT_COUNT;

// D6 correctness-first Hub geometry: never choose modes from rank-local
// buffer size.  Use at least four rounds, adding rounds only when required by
// the fixed 240 MiB global two-slot budget.  D7 may retune this after D6 is
// functionally stable.
constexpr uint64_t SCATTER_HUB_WINDOW_BYTES = SCATTER_WINDOW_BYTES;
constexpr uint64_t SCATTER_HUB_SCRATCH_BUDGET = 240ULL * 1024ULL * 1024ULL;
constexpr uint32_t SCATTER_HUB_ROUNDS = 4U; // minimum, not a fixed execution count
constexpr uint32_t SCATTER_HUB_PEERS = 7U;
constexpr uint32_t SCATTER_HUB_JOIN_BASE = 16U;
constexpr uint32_t SCATTER_HUB_MAX_THREADS = 8U;

struct HubWindowGeometry {
    uint64_t bytes = 0;
    uint64_t relay = 0;
    uint64_t prefix = 0;
    uint64_t firstChunk = 0;
    uint64_t slotPitch = 0;
    uint64_t scratchBytes = 0;
    uint32_t rounds = 0U;
};
bool BuildHubWindowGeometry(uint64_t bytes, HubWindowGeometry &geometry);

struct ScatterExecutionPlan {
    uint64_t sliceBytes = 0;
    uint64_t totalBytes = 0;
    uint64_t windowBytes = 0;
    uint64_t windowCount = 0;
    uint64_t scratchBytes = 0;
    uint32_t threadCount = 1U;
    bool useRelay = false;
    bool useHub = false;
    bool batchDirect = false;
    bool smallFastDirect = false;
    bool legacy16 = false;
};

// The host resource scheduler and executor MUST use this same decision.
HcclResult BuildExecutionPlan(const OpParam &param, const AlgResourceCtx &resCtx,
    ScatterExecutionPlan &plan);
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<ThreadHandle> &parallelThreads, const ScatterExecutionPlan &plan);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_EXEC_OP_H
