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

#include <memory>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

// v0.5B contest feature switches.  They are macros so every combination can
// be compiled without editing the frozen source.  The platform build uses the
// default value 1 for all three switches.
#ifndef HCCL_V05B_ENABLE_RANK16_LEAN_SAG
#define HCCL_V05B_ENABLE_RANK16_LEAN_SAG 1
#endif

#ifndef HCCL_V05B_ENABLE_SKIP_ROOT_DMA
#define HCCL_V05B_ENABLE_SKIP_ROOT_DMA 1
#endif

#ifndef HCCL_V05B_ENABLE_DIRECT_STAR
#define HCCL_V05B_ENABLE_DIRECT_STAR 1
#endif

static_assert(HCCL_V05B_ENABLE_RANK16_LEAN_SAG == 0 || HCCL_V05B_ENABLE_RANK16_LEAN_SAG == 1,
    "HCCL_V05B_ENABLE_RANK16_LEAN_SAG must be 0 or 1");
static_assert(HCCL_V05B_ENABLE_SKIP_ROOT_DMA == 0 || HCCL_V05B_ENABLE_SKIP_ROOT_DMA == 1,
    "HCCL_V05B_ENABLE_SKIP_ROOT_DMA must be 0 or 1");
static_assert(HCCL_V05B_ENABLE_DIRECT_STAR == 0 || HCCL_V05B_ENABLE_DIRECT_STAR == 1,
    "HCCL_V05B_ENABLE_DIRECT_STAR must be 0 or 1");

constexpr bool V05B_ENABLE_RANK16_LEAN_SAG = HCCL_V05B_ENABLE_RANK16_LEAN_SAG != 0;
constexpr bool V05B_ENABLE_SKIP_ROOT_DMA = HCCL_V05B_ENABLE_SKIP_ROOT_DMA != 0;
constexpr bool V05B_ENABLE_DIRECT_STAR = HCCL_V05B_ENABLE_DIRECT_STAR != 0;
constexpr uint32_t V05B_FEATURE_MASK =
    (V05B_ENABLE_RANK16_LEAN_SAG ? 1U : 0U) |
    (V05B_ENABLE_SKIP_ROOT_DMA ? 2U : 0U) |
    (V05B_ENABLE_DIRECT_STAR ? 4U : 0U);

// v0.7A keeps the two new mechanisms independently switchable so their
// platform effects can be measured without maintaining divergent sources.
#ifndef HCCL_V07A_ENABLE_ASYM_PUSH_DIRECT
#define HCCL_V07A_ENABLE_ASYM_PUSH_DIRECT 1
#endif

#ifndef HCCL_V07A_ENABLE_OWNER_READY_SAG
#define HCCL_V07A_ENABLE_OWNER_READY_SAG 1
#endif

static_assert(HCCL_V07A_ENABLE_ASYM_PUSH_DIRECT == 0 || HCCL_V07A_ENABLE_ASYM_PUSH_DIRECT == 1,
    "HCCL_V07A_ENABLE_ASYM_PUSH_DIRECT must be 0 or 1");
static_assert(HCCL_V07A_ENABLE_OWNER_READY_SAG == 0 || HCCL_V07A_ENABLE_OWNER_READY_SAG == 1,
    "HCCL_V07A_ENABLE_OWNER_READY_SAG must be 0 or 1");

constexpr bool V07A_ENABLE_ASYM_PUSH_DIRECT = HCCL_V07A_ENABLE_ASYM_PUSH_DIRECT != 0;
constexpr bool V07A_ENABLE_OWNER_READY_SAG = HCCL_V07A_ENABLE_OWNER_READY_SAG != 0;
constexpr uint32_t V07A_FEATURE_MASK =
    (V07A_ENABLE_ASYM_PUSH_DIRECT ? 1U : 0U) |
    (V07A_ENABLE_OWNER_READY_SAG ? 2U : 0U);

#ifndef HCCL_V12A_ENABLE_FINAL5_DRAIN_FUSION
#define HCCL_V12A_ENABLE_FINAL5_DRAIN_FUSION 1
#endif

static_assert(HCCL_V12A_ENABLE_FINAL5_DRAIN_FUSION == 0 ||
    HCCL_V12A_ENABLE_FINAL5_DRAIN_FUSION == 1,
    "HCCL_V12A_ENABLE_FINAL5_DRAIN_FUSION must be 0 or 1");

constexpr bool V12A_ENABLE_FINAL5_DRAIN_FUSION = HCCL_V12A_ENABLE_FINAL5_DRAIN_FUSION != 0;

// v0.13 isolates the new cell geometry to the 4-rank 400 MiB + 4 B case.
// Disabling the switch restores the fully validated v0.12A selection path.
#ifndef HCCL_V13_ENABLE_PERSISTENT_ROLLING2_400
#define HCCL_V13_ENABLE_PERSISTENT_ROLLING2_400 1
#endif

static_assert(HCCL_V13_ENABLE_PERSISTENT_ROLLING2_400 == 0 ||
    HCCL_V13_ENABLE_PERSISTENT_ROLLING2_400 == 1,
    "HCCL_V13_ENABLE_PERSISTENT_ROLLING2_400 must be 0 or 1");

constexpr bool V13_ENABLE_PERSISTENT_ROLLING2_400 =
    HCCL_V13_ENABLE_PERSISTENT_ROLLING2_400 != 0;

// v0.18A extends the proven 8 MiB Persistent Rolling-2 protocol only to the
// exact 4-rank 512 MiB contest case.  The independent switch and constants
// keep the v0.13B 400 MiB + 4 B graph byte-for-byte isolated.
#ifndef HCCL_V18A_ENABLE_PERSISTENT_ROLLING2_512
#define HCCL_V18A_ENABLE_PERSISTENT_ROLLING2_512 1
#endif

static_assert(HCCL_V18A_ENABLE_PERSISTENT_ROLLING2_512 == 0 ||
    HCCL_V18A_ENABLE_PERSISTENT_ROLLING2_512 == 1,
    "HCCL_V18A_ENABLE_PERSISTENT_ROLLING2_512 must be 0 or 1");

constexpr bool V18A_ENABLE_PERSISTENT_ROLLING2_512 =
    HCCL_V18A_ENABLE_PERSISTENT_ROLLING2_512 != 0;

// v0.18 keeps one fast-context tag per communicator while selecting the
// measured best 512 KiB direction by topology: Pull for rank4, Registered
// Push for rank12/rank16.
constexpr uint64_t V18_SMALL_FAST_CONTEXT_MAGIC = 0x4843434C56313848ULL; // "HCCLV18H"
constexpr uint32_t V18_SMALL_FAST_CONTEXT_VERSION = 1;

struct SmallFastContextHeader {
    uint64_t magic{0};
    uint32_t version{0};
    uint32_t algorithm{0};
    uint32_t rankId{INVALID_VALUE_RANKID};
    uint32_t rankSize{0};
    uint64_t streamKey{0};
    ThreadHandle mainThread{0};
};

static_assert(sizeof(SmallFastContextHeader) == 40,
    "SmallFastContextHeader must remain a stable 40-byte prefix");

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

struct CcuKernelArgBroadcast : CcuKernelArgBase {
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t rootId;
    // v0.14B captures the first 512 KiB invocation's memory identity in a
    // second registered graph.  Runtime execution uses it only after an exact
    // address+token match; otherwise the two-argument v0.14A graph is used.
    uint64_t registeredBaseAddr;
    uint64_t registeredToken;
    uint32_t hasPrev;
    uint32_t hasNext;
    uint32_t prevChannelIdx;
    uint32_t nextChannelIdx;
    uint32_t chainIsFirst;
    uint32_t chainIsLast;
    // Pair-chain variants keep the two segment roles in the registered kernel
    // argument.  They are deliberately not runtime task arguments, so the
    // CCU graph remains static just like the v0.3B2 single-segment variants.
    uint32_t chainPairIsFirst;
    uint32_t chainPairIsLast;
    // Registered per static chain kernel.  A unique bit is used by exactly one
    // pair (or the contest tail), so READY never needs an ACK before reuse.
    uint32_t chainReadyMask;
    // v0.3F assigns one immutable notify location to each of the two segments
    // carried by a static pair kernel.  Separate locations restore a 16 MiB
    // forwarding cell without reintroducing ACK or notify-bit reuse.
    uint32_t cutReadyNotifyIdx0;
    uint32_t cutReadyMask0;
    uint32_t cutReadyNotifyIdx1;
    uint32_t cutReadyMask1;
    // Rolling2 carries four consecutive segments in one static CCU kernel.
    // Each segment retains its own immutable READY location while two event
    // slots are drained and reused as a sliding DMA window.
    uint32_t cutReadyNotifyIdx2;
    uint32_t cutReadyMask2;
    uint32_t cutReadyNotifyIdx3;
    uint32_t cutReadyMask3;
    // v0.12A's 400 MiB + 4 B final kernel carries segment24 in addition
    // to the preceding four segments, then drains FINAL in the same graph.
    uint32_t cutReadyNotifyIdx4;
    uint32_t cutReadyMask4;
    // Lean-SAG: only the kernel group containing root waits for the scatter
    // READY signal.  The same immutable index lets v0.5B omit the redundant
    // owner-to-root AllGather DMA while retaining final synchronization.
    uint32_t leanHasRootChannel;
    uint32_t leanRootChannelIdx;
};

enum class BroadcastAlgorithm : uint32_t {
    DIRECT = 0,
    SCATTER_ALLGATHER = 1,
    PIPELINED_CHAIN = 2,
    SEGMENT_CUT_THROUGH_CHAIN = 3,
    LEAN_SCATTER_ALLGATHER = 4,
    ASYMMETRIC_DIRECT = 5,
    OWNER_READY_SCATTER_ALLGATHER = 6,
    QUAD4_ROLLING2_FINAL5_CHAIN = 7,
    PERSISTENT_ROLLING2_400 = 8,
    // v0.14A: the contest's fixed 512 KiB float32 case carries only the
    // invocation-specific buffer address and memory token.  The byte count
    // is embedded in the registered CCU graph.
    ASYMMETRIC_DIRECT_512K_2ARG = 9,
    ASYMMETRIC_DIRECT_512K_REGISTERED = 10,
    // v0.16B reverses only the fixed 512 KiB star data direction.  Root
    // publishes its source identity, receivers Read in parallel and ACK only
    // after their local DMA Event has completed.
    ASYMMETRIC_PULL_512K_2ARG = 11,
    // v0.18A: one static 64 x 8 MiB Rolling-2 graph for the exact 4-rank
    // 512 MiB case.  This is deliberately distinct from the 50-cell + 4 B
    // v0.13B graph so neither its tail geometry nor Engine Context is reused.
    PERSISTENT_ROLLING2_512 = 12,
};

enum class ChainKernelVariant : uint32_t {
    FIRST = 0,
    MIDDLE = 1,
    LAST = 2,
    SINGLE = 3,
    UNIQUE_PAIR_SLOT_BASE = 4,
    UNIQUE_TAIL_SLOT12 = 20,
    UNIQUE_DRAIN = 21,
    COUNT = 22,
};

constexpr uint32_t CHAIN_UNIQUE_PAIR_SLOT_COUNT = 16;
constexpr uint32_t CHAIN_UNIQUE_TAIL_SLOT = 12;
constexpr uint32_t CUT_THROUGH_PAIR_COUNT = 16;
constexpr uint32_t CUT_THROUGH_TAIL_KERNEL = CUT_THROUGH_PAIR_COUNT;
constexpr uint32_t CUT_THROUGH_DRAIN_KERNEL = CUT_THROUGH_TAIL_KERNEL + 1;
constexpr uint32_t CUT_THROUGH_KERNEL_COUNT = CUT_THROUGH_DRAIN_KERNEL + 1;
constexpr uint32_t CUT_THROUGH_MAX_SEGMENTS = 2 * CUT_THROUGH_PAIR_COUNT;
constexpr uint32_t CUT_THROUGH_TAIL_SEGMENT = 24;
constexpr uint32_t QUAD4_SEGMENTS_PER_KERNEL = 4;
constexpr uint32_t FINAL5_SEGMENTS_PER_KERNEL = 5;
constexpr uint32_t QUAD4_KERNEL_COUNT = CUT_THROUGH_MAX_SEGMENTS / QUAD4_SEGMENTS_PER_KERNEL;
constexpr uint32_t FINAL5_DRAIN_KERNEL = QUAD4_KERNEL_COUNT;
constexpr uint32_t QUAD4_DRAIN_KERNEL = FINAL5_DRAIN_KERNEL + 1;
constexpr uint32_t QUAD4_RESOURCE_COUNT = QUAD4_DRAIN_KERNEL + 1;
constexpr uint64_t V13_PERSISTENT_SEGMENT_SIZE = 8ULL * 1024ULL * 1024ULL;
constexpr uint64_t V13_PERSISTENT_TAIL_SIZE = V13_PERSISTENT_SEGMENT_SIZE + 4ULL;
constexpr uint32_t V13_PERSISTENT_SEGMENT_COUNT = 50;
constexpr uint32_t V13_PERSISTENT_READY_NOTIFY_COUNT =
    (V13_PERSISTENT_SEGMENT_COUNT + 15U) / 16U;
constexpr uint32_t V13_PERSISTENT_CHANNEL_NOTIFY_NUM =
    1U + V13_PERSISTENT_READY_NOTIFY_COUNT;
static_assert(V13_PERSISTENT_SEGMENT_SIZE * V13_PERSISTENT_SEGMENT_COUNT ==
    400ULL * 1024ULL * 1024ULL,
    "v0.13 persistent cells must cover exactly 400 MiB before the final 4 B");
static_assert(V13_PERSISTENT_SEGMENT_COUNT <= V13_PERSISTENT_READY_NOTIFY_COUNT * 16U,
    "v0.13 persistent READY locations must cover every segment");
static_assert(V13_PERSISTENT_SEGMENT_COUNT % 2U == 0,
    "v0.13 persistent Rolling-2 geometry requires an even segment count");
constexpr uint64_t V18A_PERSISTENT_512_SEGMENT_SIZE = 8ULL * 1024ULL * 1024ULL;
constexpr uint32_t V18A_PERSISTENT_512_SEGMENT_COUNT = 64;
constexpr uint32_t V18A_PERSISTENT_512_READY_NOTIFY_COUNT =
    (V18A_PERSISTENT_512_SEGMENT_COUNT + 15U) / 16U;
constexpr uint32_t V18A_PERSISTENT_512_CHANNEL_NOTIFY_NUM =
    1U + V18A_PERSISTENT_512_READY_NOTIFY_COUNT;
static_assert(V18A_PERSISTENT_512_SEGMENT_SIZE * V18A_PERSISTENT_512_SEGMENT_COUNT ==
    512ULL * 1024ULL * 1024ULL,
    "v0.18A persistent cells must cover exactly 512 MiB");
static_assert(V18A_PERSISTENT_512_SEGMENT_COUNT <=
    V18A_PERSISTENT_512_READY_NOTIFY_COUNT * 16U,
    "v0.18A persistent READY locations must cover every segment");
static_assert(V18A_PERSISTENT_512_SEGMENT_COUNT % 2U == 0,
    "v0.18A persistent Rolling-2 geometry requires an even segment count");
static_assert(V18A_PERSISTENT_512_CHANNEL_NOTIFY_NUM == V13_PERSISTENT_CHANNEL_NOTIFY_NUM,
    "v0.18A must retain the proven five-notify channel resource shape");
static_assert(QUAD4_KERNEL_COUNT * QUAD4_SEGMENTS_PER_KERNEL == CUT_THROUGH_MAX_SEGMENTS,
    "Quad4 kernels must cover all static cut-through segment slots");
static_assert(static_cast<uint32_t>(ChainKernelVariant::UNIQUE_TAIL_SLOT12) ==
    static_cast<uint32_t>(ChainKernelVariant::UNIQUE_PAIR_SLOT_BASE) + CHAIN_UNIQUE_PAIR_SLOT_COUNT,
    "Unique pair handles must be contiguous");

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

struct AlgResourceCtx {
    SmallFastContextHeader smallFastHeader;
    BroadcastAlgorithm algorithm{BroadcastAlgorithm::DIRECT};
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<CcuKernelHandle> allGatherKernels;
    std::vector<std::vector<uint32_t>> peersPerKernel;
    std::vector<CcuKernelHandle> registeredFastKernels;
    uint64_t registeredBaseAddr{0};
    uint64_t registeredToken{0};

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << smallFastHeader;
        binaryStream << algorithm;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << allGatherKernels;
        binaryStream << peersPerKernel;
        binaryStream << registeredFastKernels;
        binaryStream << registeredBaseAddr;
        binaryStream << registeredToken;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> smallFastHeader;
        binaryStream >> algorithm;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> allGatherKernels;
        binaryStream >> peersPerKernel;
        binaryStream >> registeredFastKernels;
        binaryStream >> registeredBaseAddr;
        binaryStream >> registeredToken;
    }
};

#endif // OPS_HCCL_CUSTOM_H
