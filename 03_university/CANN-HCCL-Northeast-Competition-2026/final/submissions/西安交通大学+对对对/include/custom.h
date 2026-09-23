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

#include <cstdint>
#include <type_traits>

#include "common.h"

constexpr uint64_t SCATTER_CTX_MAGIC = 0x5343415454455231ULL;
constexpr uint32_t SCATTER_CTX_VERSION = 2500;
// Submission-time ablation switches: 1=small event mask, 2=compact relay,
// 4=relay receive overlap. Every rank must use the same compiled mask.
#ifndef SCATTER_V016_FEATURE_MASK
#define SCATTER_V016_FEATURE_MASK 7
#endif
static_assert(SCATTER_V016_FEATURE_MASK >= 0 && SCATTER_V016_FEATURE_MASK <= 7,
    "Invalid V016 feature mask");
constexpr uint32_t SCATTER_FEATURES = SCATTER_V016_FEATURE_MASK;
// Independent V016 options: 1=small READY/Write, 2=4x3 relay Copy,
// 4=2x8 relay Copy, 8=per-tail release with same-die helper forwarding.
#ifndef SCATTER_V016_OPT_MASK
#define SCATTER_V016_OPT_MASK 15
#endif
static_assert(SCATTER_V016_OPT_MASK >= 0 && SCATTER_V016_OPT_MASK <= 15,
    "Invalid V016 optimization mask");
constexpr uint32_t SCATTER_OPTIMIZATIONS = SCATTER_V016_OPT_MASK;
// V017: 1=interleaved helper-tail release, 2=target small DIRECT early Copy.
// These affect disjoint paths. Keep the proven V016 switches independently.
#ifndef SCATTER_V017_TUNE_MASK
#define SCATTER_V017_TUNE_MASK 3
#endif
static_assert(SCATTER_V017_TUNE_MASK >= 0 && SCATTER_V017_TUNE_MASK <= 3,
    "Invalid V017 tuning mask");
constexpr uint32_t SCATTER_TUNING = SCATTER_V017_TUNE_MASK;
// V018: 1=2x8 whole-helper tail overlap, 2=2x8 helper ACK_REQUEST fusion,
// 4=4x1/4x3 small receiver-pull. All ranks use the same compiled mask.
#ifndef SCATTER_V018_OPT_MASK
#define SCATTER_V018_OPT_MASK 7
#endif
static_assert(SCATTER_V018_OPT_MASK >= 0 && SCATTER_V018_OPT_MASK <= 7,
    "Invalid V018 optimization mask");
constexpr uint32_t SCATTER_V018_OPTIONS = SCATTER_V018_OPT_MASK;
// V019 fix1 keeps only B: 2x8 per-peer READY/Write. The failed 4x3
// bridge and request-fusion extensions are absent from this submission.
// Reject accidental re-enabling through external compiler flags.
#ifndef SCATTER_V019_OPT_MASK
#define SCATTER_V019_OPT_MASK 2
#endif
static_assert(SCATTER_V019_OPT_MASK == 0 || SCATTER_V019_OPT_MASK == 2,
    "V019 fix1 supports only the isolated 2x8 READY/Write option");
constexpr uint32_t SCATTER_V019_OPTIONS = SCATTER_V019_OPT_MASK;
// V020: 1=4x3 tail-before-helper-output, 2=2x8 equivalent scheduling,
// 4=8+4 fused single-wave DIRECT, 8=4x1 small pull early Copy.
// Independent registration-time options, identical across ranks.
#ifndef SCATTER_V020_OPT_MASK
#define SCATTER_V020_OPT_MASK 15
#endif
static_assert(SCATTER_V020_OPT_MASK >= 0 && SCATTER_V020_OPT_MASK <= 15,
    "Invalid V020 optimization mask");
constexpr uint32_t SCATTER_V020_OPTIONS = SCATTER_V020_OPT_MASK;
// V021: 1=4x1 small Pull with three Host-computed source addresses,
// 2=4x3 small Pull with common input base and receiver offset.
// Keep the proven V020 paths and all completion handshakes unchanged.
#ifndef SCATTER_V021_OPT_MASK
#define SCATTER_V021_OPT_MASK 3
#endif
static_assert(SCATTER_V021_OPT_MASK >= 0 && SCATTER_V021_OPT_MASK <= 3,
    "Invalid V021 optimization mask");
constexpr uint32_t SCATTER_V021_OPTIONS = SCATTER_V021_OPT_MASK;
// V022: 1=8+4 small READY/Write, 2=its merged Write events,
// 4=its Copy wait after DONE/ACK, 8=main-heavy small Pull groups,
// 16=4x3 small Pull publication in topology priority order.
// Each option is admitted independently; all ranks use the same build.
#ifndef SCATTER_V022_OPT_MASK
#define SCATTER_V022_OPT_MASK 31
#endif
static_assert(SCATTER_V022_OPT_MASK >= 0 && SCATTER_V022_OPT_MASK <= 31,
    "Invalid V022 optimization mask");
constexpr uint32_t SCATTER_V022_OPTIONS = SCATTER_V022_OPT_MASK;
// V023: 1=2x8 small Pull, 2=8+4 small Pull, 4=4x1 receiver MS,
// 8=4x3 receiver MS, 16=already-admitted 8+4 large READY/Write.
// Independent, registration-time options; every rank uses the same build.
#ifndef SCATTER_V023_OPT_MASK
#define SCATTER_V023_OPT_MASK 31
#endif
static_assert(SCATTER_V023_OPT_MASK >= 0 && SCATTER_V023_OPT_MASK <= 31,
    "Invalid V023 optimization mask");
constexpr uint32_t SCATTER_V023_OPTIONS = SCATTER_V023_OPT_MASK;
// V024: 1=2x8 full-MS receiver, 2=8+4 full-MS receiver,
// 4=4x3 direct prefix with a private 16-KiB MS tail. Root protocols are unchanged.
#ifndef SCATTER_V024_OPT_MASK
#define SCATTER_V024_OPT_MASK 7
#endif
static_assert(SCATTER_V024_OPT_MASK >= 0 && SCATTER_V024_OPT_MASK <= 7,
    "Invalid V024 optimization mask");
constexpr uint32_t SCATTER_V024_OPTIONS = SCATTER_V024_OPT_MASK;

// V025: credit-gated short Push, independently admitted for 2x8 / 8+4 / 4x3.
// The next receiver READY protects reuse of the previous DONE notification.
#ifndef SCATTER_V025_OPT_MASK
#define SCATTER_V025_OPT_MASK 7
#endif
static_assert(SCATTER_V025_OPT_MASK >= 0 && SCATTER_V025_OPT_MASK <= 7,
    "Invalid V025 optimization mask");
constexpr uint32_t SCATTER_V025_OPTIONS = SCATTER_V025_OPT_MASK;

enum class ScatterPullAddressMode : uint32_t {
    SLICE = 0, THREE_SOURCES = 1, BASE_OFFSET = 2
};
constexpr uint32_t SCATTER_MAX_DIES = 2;
constexpr uint32_t SCATTER_UNKNOWN_DIE = 0xFFFFFFFFU;
constexpr uint32_t SCATTER_THREAD_START = 0;
constexpr uint32_t SCATTER_THREAD_DONE = 0;
constexpr uint32_t SCATTER_MAX_BATCHES = 6;
constexpr uint32_t SCATTER_CHANNEL_NOTIFY_COUNT = 4;
constexpr uint64_t SCATTER_SMALL_LIMIT = 1024ULL * 1024;

enum class ScatterAlgorithm : uint32_t { DIRECT = 0, TREE = 1, RELAY = 2 };
enum class ScatterTopology : uint32_t {
    UNKNOWN = 0, TWO_EIGHT = 1, FOUR_ONE = 2, EIGHT_FOUR = 3, FOUR_THREE = 4
};
enum class ScatterTopologyStatus : uint32_t {
    NOT_QUERIED = 0, FOUND = 1, NO_MESH_INSTANCE = 2, QUERY_FAILED = 3,
    INVALID_MEMBERS = 4, CONFLICTING_MESH = 5, SINGLE_RANK = 6
};
enum class ScatterRouteReason : uint32_t {
    MATCHED = 0, UNKNOWN_MEMBERS = 1, INVALID_PARTITION = 2,
    UNSUPPORTED_TOPOLOGY = 3, UNMATCHED_SIZE = 4, TREE_SCRATCH_TOO_SMALL = 5,
    NO_LOCAL_HELPER = 6, REMOTE_FANOUT_AT_MOST_FOUR = 7,
    RELAY_SCRATCH_TOO_SMALL = 8, DIRECT_POLICY = 9, SINGLE_RANK = 10
};

// One all-peer channel set per communicator. Topology and capacity are shared
// using registration tags; the small registered prefix contains no payload.
struct ScatterCommResources {
    uint64_t magic = SCATTER_CTX_MAGIC;
    uint32_t version = SCATTER_CTX_VERSION;
    uint32_t initStage = 0;
    uint32_t myRank = 0;
    uint32_t rankSize = 0;
    uint32_t topologyKnown = 0;
    uint32_t localRankMask = 0;
    ScatterTopologyStatus topologyStatus = ScatterTopologyStatus::NOT_QUERIED;
    uint64_t scratchAddress = 0;
    uint64_t scratchBytes = 0;
    HcclMemHandle metadataHandle = nullptr;
    ThreadHandle worker = 0;
    ChannelHandle channels[MAX_RANK_SIZE] = {};
    uint32_t dies[MAX_RANK_SIZE] = {};
    uint32_t serverMasks[MAX_RANK_SIZE] = {};
    uint64_t capacities[MAX_RANK_SIZE] = {};
};

struct ScatterBatch {
    uint32_t kernelCount = 0;
    uint32_t parallel = 0;
    CcuKernelHandle kernels[MAX_RANK_SIZE] = {};
};
struct ScatterShape {
    uint32_t batchCount = 0;
    uint32_t reserved = 0;
    ScatterBatch batches[SCATTER_MAX_BATCHES] = {};
};

// Specialize the instruction graph at the actual CCU transfer limit, without
// introducing an unmeasured small-message threshold. Both modes are direct.
enum class ScatterDataPath : uint32_t {
    SINGLE_WAVE = 0,
    CHUNKED = 1
};

enum class ScatterSchedule : uint32_t {
    PARALLEL = 0,
    SEQUENTIAL = 1
};

struct ScatterKernelGroup {
    CcuKernelHandle kernel = 0;
    uint32_t peerMask = 0;
    uint32_t dieId = SCATTER_UNKNOWN_DIE;
    uint32_t localPeerCount = 0;
    uint32_t remotePeerCount = 0;
};

// Only fixed communication resources are cached. The auxiliary thread belongs
// to the communicator's pool, not this root exclusively: invocations must stay
// ordered and join it before the user stream continues. Buffers, tokens and the
// main thread are obtained for every invocation.
struct AlgResourceCtx {
    uint64_t magic = SCATTER_CTX_MAGIC;
    uint32_t version = SCATTER_CTX_VERSION;
    uint32_t root = 0;
    uint32_t myRank = 0;
    uint32_t rankSize = 0;
    ScatterDataPath dataPath = ScatterDataPath::SINGLE_WAVE;
    ScatterSchedule schedule = ScatterSchedule::PARALLEL;
    uint32_t groupCount = 0;
    uint32_t localRankMask = 0;
    uint32_t topologyKnown = 0;
    ThreadHandle worker = 0;
    CcuKernelHandle copyKernel = 0;
    ScatterKernelGroup groups[MAX_RANK_SIZE] = {};
    ScatterAlgorithm algorithm = ScatterAlgorithm::DIRECT;
    ScatterTopology topology = ScatterTopology::UNKNOWN;
    uint32_t sceneId = 0;
    uint32_t directFusion = 0;
    uint32_t rootAckFusion = 0;
    uint32_t compactScratch = 0;
    uint32_t relayOverlap = 0;
    uint32_t relayCopy = 0;
    uint64_t recvBytes = 0;
    uint64_t chunkBytes = 0;
    uint64_t fullSteps = 0;
    uint64_t tailBytes = 0;
    uint64_t scratchAddress = 0;
    uint64_t scratchBytes = 0;
    ScatterShape fullShape{};
    ScatterShape tailShape{};
    uint64_t scratchUsedBytes = 0;
    ScatterRouteReason routeReason = ScatterRouteReason::MATCHED;
};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value, "Context must be byte-copyable");
static_assert(std::is_standard_layout<AlgResourceCtx>::value, "Context must have a fixed layout");
static_assert(std::is_trivially_copyable<ScatterCommResources>::value, "Resources must be byte-copyable");

#endif // OPS_HCCL_CUSTOM_H
