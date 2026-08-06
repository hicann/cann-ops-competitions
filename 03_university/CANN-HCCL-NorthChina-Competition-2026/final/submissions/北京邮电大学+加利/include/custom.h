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
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_types.h>

#include "common.h"

struct CommBuffer {
    void *addr{nullptr};
    uint64_t size{0};
};

enum class FinalTopology : uint32_t {
    TOPOLOGY_4X1 = 0,
    TOPOLOGY_8X4 = 1,
    TOPOLOGY_2X8 = 2,
};

enum class AllGatherProfile : uint32_t {
    GENERIC = 0,
    P2X8_512K = 1,
    P2X8_512M = 2,
    P2X8_400M4B = 3,
    P4X1_512K = 4,
    P4X1_512M = 5,
    P4X1_400M4B = 6,
    P8P4_512K = 7,
    P8P4_512M = 8,
    P8P4_400M4B = 9,
};

enum class AllGatherAlgorithm : uint32_t {
    // 未命中九类固定 Profile 时使用的 Direct 正确性基线；支持任意合法 Slice 大小，
    // 由 Host 按安全上限分块下发，通信层随实际拓扑调整。
    GENERIC_DIRECT = 0,
    // 512 KiB 标称 Output 的固定 one-shot Direct 图；通信层由 Profile 单独决定。
    FIXED_DIRECT_SMALL = 1,
    // 4x1 大消息只使用 Clos，在一个 Kernel 内执行 Rolling-2。
    CLOS_DIRECT_ROLLING2 = 2,
    // 2x8/8+4 大消息将本地 Mesh 与远端 Clos 拆分并按实际 IO Die 并行执行。
    LAYERED_DIRECT_ROLLING2 = 3,
};

// 数据路径必须进入 Engine Context tag。大消息 Profile 的 algorithm 描述通信图，
// 不能区分 Rolling-2 与 single-DMA；若只依赖 algorithm，切换 Kernel 后可能复用旧 Context。
enum class AllGatherDataPath : uint32_t {
    GENERIC_DIRECT = 0,
    FIXED_SMALL = 1,
    ROLLING2 = 2,
    SINGLE_DMA = 3,
    // 分层大消息：Host 用 CCU Thread Notify 编排四个阶段 Kernel，避免依赖
    // 当前 HCCL-VM 无法建模的跨 Die shared CKE。2x8 和 8+4 使用各自路由表。
    HYBRID_RELAY = 4,
    // 保留原两个长驻 Kernel + shared CKE 入口，用于真实硬件 A/B。
    HYBRID_RELAY_CKE = 5,
    // 2x8 分布式 Root：每个远端源由同 slot Rank relay 前缀，其余后缀经 Clos
    // 直达。允许单 Peer 动态共享 4B Clos 聚合带宽，使用最少的 22 个网络 WQE。
    DISTRIBUTED_ROOT = 6,
    // 保留四 Peer Phase-B 路由；单 Peer 限制为 B 时仍达到 15S/(11B) 下界。
    DISTRIBUTED_ROOT_CAPPED = 7,
    // 8+4 双向 Half-Root：4->8 只跨 Clos 发送两个半片，8->4 在 4 卡侧
    // 使用 Root 前缀 relay，同时压低双向 Clos 流量和网络 WQE 数。
    BIDIRECTIONAL_HALF_ROOT = 8,
    // 保留单 Peer Phase-B 传输不超过 relay 前缀的路由，用于真机限速 A/B。
    BIDIRECTIONAL_HALF_ROOT_CAPPED = 9,
    // 4x1 完整 Slice Direct：K4 完美匹配顺序提交，并将逐 Peer READY 等待
    // 与对应 Write 流水化。原 SINGLE_DMA 保留用于真机 A/B。
    MATCHED_READY = 10,
    // 8+4 Half-Root 保持动态 Clos 带宽路由，仅按真实读依赖缩减同步扇出，
    // 并以相对槽位提交 Phase-A/Phase-B。原 Half-Root 路径保留用于真机 A/B。
    BIDIRECTIONAL_HALF_ROOT_MATCHED = 11,
    // 8+4 Half-Root 的 Shared-Clos 实验路径；保持独立 Phase-A/Phase-B 入口，
    // 原 Half-Root、capped 与 matched 路径均保留用于静态 A/B。
    BIDIRECTIONAL_HALF_ROOT_SHARED = 12,
    // 8+4 Shared-Clos 的 5/8 relay 比例实验；用于验证 Clos 收发各自拥有
    // 4B 带宽时的平衡点，数据路由与同步保持和 Shared 5/7 路径一致。
    BIDIRECTIONAL_HALF_ROOT_SHARED_5_8 = 13,
    // 8+4 Full-Seed 4/7：低位 A 在 Phase A 向匹配 U 发布完整 Slice，
    // Phase B 仅向其余三个 U 发送 relay tail，填满 Phase-A Clos RX 空档。
    BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7 = 14,
    // 2x8 Distributed Root 的单 Peer 串行实验：两台 Server 反向解释相同
    // offset，令每一轮收发落在同一 Rank Pair，并让该 Peer 独占 4B Clos。
    DISTRIBUTED_ROOT_MATCHED_SERIAL = 15,
    // 4x1 K4 Peer 串行、Peer 内 Rolling-2：同一时刻仅一个匹配 Peer 使用 Clos，
    // 每个 Peer 以两个 16 MiB DMA slot 流水完整 Slice。
    MATCHED_SERIAL_ROLLING2 = 16,
    INVALID = 17,
};

// A/B 时只修改对应固定用例；Host 和执行期使用同一配置，避免派发判断漂移。
constexpr AllGatherDataPath P2X8_512M_DATA_PATH
    = AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL;
constexpr AllGatherDataPath P2X8_400M4B_DATA_PATH
    = AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL;
constexpr AllGatherDataPath P8P4_512M_DATA_PATH
    = AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7;
constexpr AllGatherDataPath P8P4_400M4B_DATA_PATH
    = AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7;
constexpr AllGatherDataPath P4X1_512M_DATA_PATH = AllGatherDataPath::MATCHED_SERIAL_ROLLING2;
constexpr AllGatherDataPath P4X1_400M4B_DATA_PATH = AllGatherDataPath::MATCHED_READY;

inline AllGatherDataPath GetConfiguredAllGatherDataPath(AllGatherProfile profile)
{
    switch (profile) {
        case AllGatherProfile::GENERIC:
            return AllGatherDataPath::GENERIC_DIRECT;
        case AllGatherProfile::P2X8_512K:
        case AllGatherProfile::P4X1_512K:
        case AllGatherProfile::P8P4_512K:
            return AllGatherDataPath::FIXED_SMALL;
        case AllGatherProfile::P2X8_512M:
            return P2X8_512M_DATA_PATH;
        case AllGatherProfile::P2X8_400M4B:
            return P2X8_400M4B_DATA_PATH;
        case AllGatherProfile::P4X1_512M:
            return P4X1_512M_DATA_PATH;
        case AllGatherProfile::P4X1_400M4B:
            return P4X1_400M4B_DATA_PATH;
        case AllGatherProfile::P8P4_512M:
            return P8P4_512M_DATA_PATH;
        case AllGatherProfile::P8P4_400M4B:
            return P8P4_400M4B_DATA_PATH;
        default:
            return AllGatherDataPath::INVALID;
    }
}

enum class AllGatherLayerPolicy : uint32_t {
    CLOS_ONLY = 0,
    LOCAL_MESH_REMOTE_CLOS = 1,
};

enum class AllGatherLayerRole : uint32_t {
    CLOS = 0,
    LOCAL_MESH = 1,
};

constexpr uint64_t ALLGATHER_TARGET_OUTPUT_512K_BYTES = 512ULL * 1024ULL;
constexpr uint64_t ALLGATHER_TARGET_OUTPUT_512M_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t ALLGATHER_TARGET_OUTPUT_400M4B_BYTES = 400ULL * 1024ULL * 1024ULL + 4ULL;
constexpr uint64_t ALLGATHER_PLATFORM_SLICE_ALIGNMENT_BYTES = 512ULL;
constexpr uint64_t ALLGATHER_ROLLING2_CELL_BYTES = 8ULL * 1024ULL * 1024ULL;
constexpr uint64_t ALLGATHER_P4X1_MATCHED_ROLLING2_CELL_BYTES = 16ULL * 1024ULL * 1024ULL;
constexpr uint32_t ALLGATHER_P2X8_HYBRID_STRIPE_COUNT = 11U;
constexpr uint32_t ALLGATHER_P8P4_HYBRID_STRIPE_COUNT = 9U;
constexpr uint32_t ALLGATHER_INVALID_LAYER = std::numeric_limits<uint32_t>::max();
constexpr uint32_t ALLGATHER_P4X1_RANK_COUNT = 4U;
constexpr uint32_t ALLGATHER_P4X1_PEER_COUNT = 3U;
// 每列构成 K4 的一组双向完美匹配，避免升序提交在低 Rank 形成首发 incast。
constexpr uint32_t ALLGATHER_P4X1_MATCHING_ORDER
    [ALLGATHER_P4X1_RANK_COUNT][ALLGATHER_P4X1_PEER_COUNT] = {
        {1U, 2U, 3U},
        {0U, 3U, 2U},
        {3U, 0U, 1U},
        {2U, 1U, 0U},
    };

struct AllGatherProfileSpec {
    AllGatherProfile profile{AllGatherProfile::GENERIC};
    FinalTopology topology{FinalTopology::TOPOLOGY_4X1};
    uint32_t expectedRankSize{0};
    uint64_t targetOutputBytes{0};
    AllGatherAlgorithm algorithm{AllGatherAlgorithm::GENERIC_DIRECT};
    AllGatherLayerPolicy layerPolicy{AllGatherLayerPolicy::CLOS_ONLY};
    // 0 表示运行时使用完整 Slice（小消息）或 Host 动态块（通用回退）。
    uint64_t cellBytes{0};
    uint32_t eventSlots{1};
    // 每层网络设备只属于一个 IO Die：分层 Profile 固定一个 Mesh 和一个 Clos Kernel。
    uint32_t minKernelCount{1};
    uint32_t maxKernelCount{1};
};

inline bool GetAllGatherProfileSpec(AllGatherProfile profile, AllGatherProfileSpec &spec)
{
    switch (profile) {
        case AllGatherProfile::P2X8_512K:
            spec = {profile, FinalTopology::TOPOLOGY_2X8, 16U, ALLGATHER_TARGET_OUTPUT_512K_BYTES,
                AllGatherAlgorithm::FIXED_DIRECT_SMALL,
                AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS, 0U, 1U, 2U, 2U};
            return true;
        case AllGatherProfile::P2X8_512M:
            spec = {profile, FinalTopology::TOPOLOGY_2X8, 16U, ALLGATHER_TARGET_OUTPUT_512M_BYTES,
                AllGatherAlgorithm::LAYERED_DIRECT_ROLLING2,
                AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS,
                ALLGATHER_ROLLING2_CELL_BYTES, 2U, 2U, 2U};
            return true;
        case AllGatherProfile::P2X8_400M4B:
            spec = {profile, FinalTopology::TOPOLOGY_2X8, 16U, ALLGATHER_TARGET_OUTPUT_400M4B_BYTES,
                AllGatherAlgorithm::LAYERED_DIRECT_ROLLING2,
                AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS,
                ALLGATHER_ROLLING2_CELL_BYTES, 2U, 2U, 2U};
            return true;
        case AllGatherProfile::P4X1_512K:
            spec = {profile, FinalTopology::TOPOLOGY_4X1, 4U, ALLGATHER_TARGET_OUTPUT_512K_BYTES,
                AllGatherAlgorithm::FIXED_DIRECT_SMALL,
                AllGatherLayerPolicy::CLOS_ONLY, 0U, 1U, 1U, 1U};
            return true;
        case AllGatherProfile::P4X1_512M:
            spec = {profile, FinalTopology::TOPOLOGY_4X1, 4U, ALLGATHER_TARGET_OUTPUT_512M_BYTES,
                AllGatherAlgorithm::CLOS_DIRECT_ROLLING2, AllGatherLayerPolicy::CLOS_ONLY,
                ALLGATHER_ROLLING2_CELL_BYTES, 2U, 1U, 1U};
            return true;
        case AllGatherProfile::P4X1_400M4B:
            spec = {profile, FinalTopology::TOPOLOGY_4X1, 4U, ALLGATHER_TARGET_OUTPUT_400M4B_BYTES,
                AllGatherAlgorithm::CLOS_DIRECT_ROLLING2, AllGatherLayerPolicy::CLOS_ONLY,
                ALLGATHER_ROLLING2_CELL_BYTES, 2U, 1U, 1U};
            return true;
        case AllGatherProfile::P8P4_512K:
            spec = {profile, FinalTopology::TOPOLOGY_8X4, 12U, ALLGATHER_TARGET_OUTPUT_512K_BYTES,
                AllGatherAlgorithm::FIXED_DIRECT_SMALL,
                AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS, 0U, 1U, 2U, 2U};
            return true;
        case AllGatherProfile::P8P4_512M:
            spec = {profile, FinalTopology::TOPOLOGY_8X4, 12U, ALLGATHER_TARGET_OUTPUT_512M_BYTES,
                AllGatherAlgorithm::LAYERED_DIRECT_ROLLING2,
                AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS,
                ALLGATHER_ROLLING2_CELL_BYTES, 2U, 2U, 2U};
            return true;
        case AllGatherProfile::P8P4_400M4B:
            spec = {profile, FinalTopology::TOPOLOGY_8X4, 12U, ALLGATHER_TARGET_OUTPUT_400M4B_BYTES,
                AllGatherAlgorithm::LAYERED_DIRECT_ROLLING2,
                AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS,
                ALLGATHER_ROLLING2_CELL_BYTES, 2U, 2U, 2U};
            return true;
        case AllGatherProfile::GENERIC:
            spec = {profile, FinalTopology::TOPOLOGY_4X1, 0U, 0U,
                AllGatherAlgorithm::GENERIC_DIRECT,
                AllGatherLayerPolicy::CLOS_ONLY, 0U, 1U, 1U, 1U};
            return true;
        default:
            return false;
    }
}

inline void ApplyAllGatherTopologyPolicy(
    FinalTopology topology, uint32_t rankSize, AllGatherProfileSpec &spec)
{
    if (spec.profile != AllGatherProfile::GENERIC) {
        return;
    }
    spec.topology = topology;
    spec.expectedRankSize = rankSize;
    // 12/16 Rank 的通用大 Slice同样使用本地 Mesh、远端 Clos，避免安全回退把全部
    // 流量重新压回 Clos；各 Context 仍保存并复核实际 Layer、Die 和 Peer 集。
    if (topology != FinalTopology::TOPOLOGY_4X1) {
        spec.layerPolicy = AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS;
        spec.minKernelCount = 2U;
        spec.maxKernelCount = 2U;
    }
}

inline bool MatchesAllGatherOutputProfile(
    uint64_t sendCount, uint64_t targetOutputBytes, uint32_t rankSize, uint64_t dataTypeSize)
{
    uint64_t actualOutputBytes = 0U;
    bool matched = false;
    if (rankSize != 0U && dataTypeSize != 0U
        && sendCount <= std::numeric_limits<uint64_t>::max() / dataTypeSize) {
        const uint64_t sliceBytes = sendCount * dataTypeSize;
        if (sliceBytes <= std::numeric_limits<uint64_t>::max() / rankSize) {
            actualOutputBytes = sliceBytes * rankSize;
            const uint64_t allocationQuantum = static_cast<uint64_t>(rankSize) * dataTypeSize;
            const uint64_t platformDownwardWindow
                = static_cast<uint64_t>(rankSize) * ALLGATHER_PLATFORM_SLICE_ALIGNMENT_BYTES;
            // 官方用例按 dtype、rankSize 向上补齐，增量小于一个分配量子；当前测试平台
            // 还存在每 Rank Slice 向下对齐 512B 的输入，两种口径都映射回同一标称 Profile。
            matched = actualOutputBytes >= targetOutputBytes
                ? actualOutputBytes - targetOutputBytes < allocationQuantum
                : targetOutputBytes - actualOutputBytes < platformDownwardWindow;
        }
    }
    return matched;
}

inline AllGatherProfile SelectAllGatherProfile(
    uint64_t sendCount, uint32_t rankSize, uint64_t dataTypeSize)
{
    const bool is512K = MatchesAllGatherOutputProfile(
        sendCount, ALLGATHER_TARGET_OUTPUT_512K_BYTES, rankSize, dataTypeSize);
    const bool is512M = MatchesAllGatherOutputProfile(
        sendCount, ALLGATHER_TARGET_OUTPUT_512M_BYTES, rankSize, dataTypeSize);
    const bool is400M4B = MatchesAllGatherOutputProfile(
        sendCount, ALLGATHER_TARGET_OUTPUT_400M4B_BYTES, rankSize, dataTypeSize);
    if (rankSize == 16U) {
        return is512K ? AllGatherProfile::P2X8_512K
                      : (is512M ? AllGatherProfile::P2X8_512M
                                : (is400M4B ? AllGatherProfile::P2X8_400M4B : AllGatherProfile::GENERIC));
    }
    if (rankSize == 4U) {
        return is512K ? AllGatherProfile::P4X1_512K
                      : (is512M ? AllGatherProfile::P4X1_512M
                                : (is400M4B ? AllGatherProfile::P4X1_400M4B : AllGatherProfile::GENERIC));
    }
    if (rankSize == 12U) {
        return is512K ? AllGatherProfile::P8P4_512K
                      : (is512M ? AllGatherProfile::P8P4_512M
                                : (is400M4B ? AllGatherProfile::P8P4_400M4B : AllGatherProfile::GENERIC));
    }
    return AllGatherProfile::GENERIC;
}

// DIRECT_INPUT 表示 Input 与完整 Output 不重叠，可直接从 Input 发送；
// CANONICAL_OUTPUT 表示标准 in-place，源数据已经位于 output + rank * sliceBytes；
// STAGED_OUTPUT 覆盖其余重叠形式，先经 HCCL Buffer 按 memmove 顺序搬到标准位置。
enum class AllGatherSourceMode : uint32_t {
    DIRECT_INPUT = 0,
    CANONICAL_OUTPUT = 1,
    STAGED_OUTPUT = 2,
};

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount{0};
};

struct CcuKernelArgAllGather final : CcuKernelArgBase {
    uint32_t rankId{0};
    uint32_t rankSize{0};
    // 主 Transfer Kernel 置 1 时为 out-of-place 生成本 Rank 的本地输出。
    // canonical in-place 的数据本已在最终 Offset，不提交无效的同址 Copy。
    uint32_t copyLocalSlice{0};
    AllGatherSourceMode sourceMode{AllGatherSourceMode::DIRECT_INPUT};
    AllGatherProfile profile{AllGatherProfile::GENERIC};
    AllGatherDataPath dataPath{AllGatherDataPath::GENERIC_DIRECT};
    AllGatherLayerRole layerRole{AllGatherLayerRole::CLOS};
    // 固定图在注册时捕获精确字节数；通用回退图使用 0，并从动态参数取得分块范围。
    uint64_t fixedSliceBytes{0};
    // 固定 Profile 的 dtype 字节数用于 Kernel 侧复核官方向上补齐量子。
    uint64_t dataTypeSize{0};
    // 小消息等于实际 Slice；大消息为 Profile 指定的 Rolling-2 cell。
    uint64_t cellBytes{0};
    // Channel 与 Rank 的关系在注册期固定。Direct 路径不依赖该映射，2x8/8+4 relay
    // 用它定位 Clos Seed、Mesh 目的端以及本地 Output 中的远端源 Slice。
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t localRankMask{0};
};

struct CcuKernelArgPrepare final : CcuKernelArgBase {};

struct KernelLaunchMeta {
    AllGatherLayerRole layerRole{AllGatherLayerRole::CLOS};
    CommTopo topoType{COMM_TOPO_RESERVED};
    uint32_t layer{ALLGATHER_INVALID_LAYER};
    // Kernel 所在 Die 由 Channel 本地 Endpoint 推导；同组还必须具有相同 layer。
    uint32_t actualDieId{0};
    uint32_t channelCount{0};
    uint32_t peerMask{0};
    uint32_t isPrimary{0};
    uint32_t copyLocalSlice{0};
};

struct AlgResourceCtx {
    static constexpr uint32_t MAGIC = 0x41474444U; // "AGDD": dual-Die strict dispatch
    static constexpr uint32_t VERSION = 17U;
    static constexpr uint64_t MAX_SERIALIZED_SIZE = 1024U;
    static constexpr uint32_t MAX_EXTRA_THREAD_COUNT = 1U;
    static constexpr uint32_t MAX_TRANSFER_KERNEL_COUNT = 3U;

    uint32_t magic{MAGIC};
    uint32_t version{VERSION};
    uint32_t rankSize{0};
    FinalTopology topology{FinalTopology::TOPOLOGY_4X1};
    AllGatherProfile profile{AllGatherProfile::GENERIC};
    AllGatherAlgorithm algorithm{AllGatherAlgorithm::GENERIC_DIRECT};
    AllGatherDataPath dataPath{AllGatherDataPath::GENERIC_DIRECT};
    AllGatherLayerPolicy layerPolicy{AllGatherLayerPolicy::CLOS_ONLY};
    AllGatherSourceMode sourceMode{AllGatherSourceMode::DIRECT_INPUT};
    // 固定图记录本次调用实际的每 Rank Slice 字节数；通用图保持为 0。
    uint64_t fixedSliceBytes{0};
    uint64_t dataTypeSize{0};
    uint64_t cellBytes{0};
    uint32_t localRankCount{0};
    uint32_t localRankMask{0};
    uint32_t meshLayer{ALLGATHER_INVALID_LAYER};
    // Server 内 Full-Mesh 在不同 RankGraph 中可能标记为 1DMESH 或 CUSTOM，
    // 必须保存实际类型，执行期不能把 layer 身份重新猜成固定枚举值。
    CommTopo meshTopoType{COMM_TOPO_RESERVED};
    uint32_t closLayer{ALLGATHER_INVALID_LAYER};
    CommBuffer localBuffer{};
    std::vector<ThreadHandle> extraThreads;
    // Thread-phased 路径中 Transfer 保存 Phase-A，下面的向量保存同索引的 Phase-B。
    std::vector<CcuKernelHandle> transferKernels;
    std::vector<CcuKernelHandle> hybridPhaseBKernels;
    std::vector<KernelLaunchMeta> kernelMeta;
    CcuKernelHandle prepareKernel{};

    // Engine Context 只保存可序列化的资源快照。magic/version 和各 vector 上界用于拒绝
    // 旧版本或损坏的 Context，避免把无效 Handle 带入执行路径。
    std::vector<char> Serialize() const
    {
        std::vector<char> result;
        result.reserve(MAX_SERIALIZED_SIZE);
        AppendPod(result, magic);
        AppendPod(result, version);
        AppendPod(result, rankSize);
        AppendPod(result, topology);
        AppendPod(result, profile);
        AppendPod(result, algorithm);
        AppendPod(result, dataPath);
        AppendPod(result, layerPolicy);
        AppendPod(result, sourceMode);
        AppendPod(result, fixedSliceBytes);
        AppendPod(result, dataTypeSize);
        AppendPod(result, cellBytes);
        AppendPod(result, localRankCount);
        AppendPod(result, localRankMask);
        AppendPod(result, meshLayer);
        AppendPod(result, meshTopoType);
        AppendPod(result, closLayer);
        AppendPod(result, localBuffer);
        AppendVector(result, extraThreads);
        AppendVector(result, transferKernels);
        AppendVector(result, hybridPhaseBKernels);
        AppendVector(result, kernelMeta);
        AppendPod(result, prepareKernel);
        return result;
    }

    bool DeSerialize(const std::vector<char> &data)
    {
        if (data.empty() || data.size() > MAX_SERIALIZED_SIZE) {
            return false;
        }

        size_t offset = 0;
        uint32_t parsedMagic = 0;
        uint32_t parsedVersion = 0;
        uint32_t parsedRankSize = 0;
        FinalTopology parsedTopology = FinalTopology::TOPOLOGY_4X1;
        AllGatherProfile parsedProfile = AllGatherProfile::GENERIC;
        AllGatherAlgorithm parsedAlgorithm = AllGatherAlgorithm::GENERIC_DIRECT;
        AllGatherDataPath parsedDataPath = AllGatherDataPath::GENERIC_DIRECT;
        AllGatherLayerPolicy parsedLayerPolicy = AllGatherLayerPolicy::CLOS_ONLY;
        AllGatherSourceMode parsedSourceMode = AllGatherSourceMode::DIRECT_INPUT;
        uint64_t parsedFixedSliceBytes = 0;
        uint64_t parsedDataTypeSize = 0;
        uint64_t parsedCellBytes = 0;
        uint32_t parsedLocalRankCount = 0;
        uint32_t parsedLocalRankMask = 0;
        uint32_t parsedMeshLayer = ALLGATHER_INVALID_LAYER;
        CommTopo parsedMeshTopoType = COMM_TOPO_RESERVED;
        uint32_t parsedClosLayer = ALLGATHER_INVALID_LAYER;
        CommBuffer parsedBuffer{};
        std::vector<ThreadHandle> parsedThreads;
        std::vector<CcuKernelHandle> parsedKernels;
        std::vector<CcuKernelHandle> parsedHybridPhaseBKernels;
        std::vector<KernelLaunchMeta> parsedMeta;
        CcuKernelHandle parsedPrepare{};
        if (!ReadPod(data, offset, parsedMagic) || !ReadPod(data, offset, parsedVersion)
            || !ReadPod(data, offset, parsedRankSize) || !ReadPod(data, offset, parsedTopology)
            || !ReadPod(data, offset, parsedProfile) || !ReadPod(data, offset, parsedAlgorithm)
            || !ReadPod(data, offset, parsedDataPath)
            || !ReadPod(data, offset, parsedLayerPolicy) || !ReadPod(data, offset, parsedSourceMode)
            || !ReadPod(data, offset, parsedFixedSliceBytes)
            || !ReadPod(data, offset, parsedDataTypeSize) || !ReadPod(data, offset, parsedCellBytes)
            || !ReadPod(data, offset, parsedLocalRankCount) || !ReadPod(data, offset, parsedLocalRankMask)
            || !ReadPod(data, offset, parsedMeshLayer)
            || !ReadPod(data, offset, parsedMeshTopoType)
            || !ReadPod(data, offset, parsedClosLayer)
            || !ReadPod(data, offset, parsedBuffer)
            || !ReadVector(data, offset, MAX_EXTRA_THREAD_COUNT, parsedThreads)
            || !ReadVector(data, offset, MAX_TRANSFER_KERNEL_COUNT, parsedKernels)
            || !ReadVector(data, offset, MAX_TRANSFER_KERNEL_COUNT, parsedHybridPhaseBKernels)
            || !ReadVector(data, offset, MAX_TRANSFER_KERNEL_COUNT, parsedMeta)
            || !ReadPod(data, offset, parsedPrepare) || offset != data.size()) {
            return false;
        }

        magic = parsedMagic;
        version = parsedVersion;
        rankSize = parsedRankSize;
        topology = parsedTopology;
        profile = parsedProfile;
        algorithm = parsedAlgorithm;
        dataPath = parsedDataPath;
        layerPolicy = parsedLayerPolicy;
        sourceMode = parsedSourceMode;
        fixedSliceBytes = parsedFixedSliceBytes;
        dataTypeSize = parsedDataTypeSize;
        cellBytes = parsedCellBytes;
        localRankCount = parsedLocalRankCount;
        localRankMask = parsedLocalRankMask;
        meshLayer = parsedMeshLayer;
        meshTopoType = parsedMeshTopoType;
        closLayer = parsedClosLayer;
        localBuffer = parsedBuffer;
        extraThreads = std::move(parsedThreads);
        transferKernels = std::move(parsedKernels);
        hybridPhaseBKernels = std::move(parsedHybridPhaseBKernels);
        kernelMeta = std::move(parsedMeta);
        prepareKernel = parsedPrepare;
        return true;
    }

private:
    template <typename T> static void AppendPod(std::vector<char> &data, const T &value)
    {
        static_assert(std::is_trivially_copyable<T>::value, "EngineCtx fields must be trivially copyable");
        const char *begin = reinterpret_cast<const char *>(&value);
        data.insert(data.end(), begin, begin + sizeof(T));
    }

    template <typename T> static void AppendVector(std::vector<char> &data, const std::vector<T> &values)
    {
        static_assert(std::is_trivially_copyable<T>::value, "EngineCtx fields must be trivially copyable");
        const uint32_t count = static_cast<uint32_t>(values.size());
        AppendPod(data, count);
        if (!values.empty()) {
            const char *begin = reinterpret_cast<const char *>(values.data());
            data.insert(data.end(), begin, begin + values.size() * sizeof(T));
        }
    }

    template <typename T> static bool ReadPod(const std::vector<char> &data, size_t &offset, T &value)
    {
        static_assert(std::is_trivially_copyable<T>::value, "EngineCtx fields must be trivially copyable");
        if (offset > data.size() || sizeof(T) > data.size() - offset) {
            return false;
        }
        std::memcpy(&value, data.data() + offset, sizeof(T));
        offset += sizeof(T);
        return true;
    }

    template <typename T>
    static bool ReadVector(const std::vector<char> &data, size_t &offset, uint32_t maxCount, std::vector<T> &values)
    {
        static_assert(std::is_trivially_copyable<T>::value, "EngineCtx fields must be trivially copyable");
        uint32_t count = 0;
        if (!ReadPod(data, offset, count) || count > maxCount || offset > data.size()
            || static_cast<size_t>(count) > (data.size() - offset) / sizeof(T)) {
            return false;
        }
        values.resize(count);
        if (count != 0) {
            const size_t byteCount = static_cast<size_t>(count) * sizeof(T);
            std::memcpy(values.data(), data.data() + offset, byteCount);
            offset += byteCount;
        }
        return true;
    }
};

#endif // OPS_HCCL_CUSTOM_H
