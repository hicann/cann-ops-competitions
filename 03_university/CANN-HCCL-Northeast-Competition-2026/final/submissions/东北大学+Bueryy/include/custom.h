/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_FINAL_CUSTOM_H
#define SCATTER_FINAL_CUSTOM_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

// 预研候选：0 自动选择，1 全量直写，2 全量拉取，3 强制 2×8 中继，4 强制 8+4 小机中继，5 强制 4×3 中继。
// 正式候选采用自动选择，按角色分工、实际数据边与阶段依赖生成执行序列。（候选版本：v25）
#ifndef SCATTER_CANDIDATE
#define SCATTER_CANDIDATE 0
#endif

namespace scatter_plan {
constexpr uint32_t MAX_RANKS = 16;
constexpr uint64_t TRANSFER_LIMIT = 256ULL * 1024 * 1024;
constexpr uint64_t SMALL_TOTAL_BYTES = 1ULL * 1024 * 1024;
constexpr uint64_t RELAY_TOTAL_BYTES = 8ULL * 1024 * 1024;
// 2×8 使用已测较好比例：较大档直达 63.68%，较小档 63.8%。
// 按完整 rankBytes 分档，尾片沿用本次调用的比例。
constexpr uint64_t PAIR_RATIO_SPLIT_RANK_BYTES = 28ULL * 1024 * 1024;
constexpr uint32_t PAIR_DIRECT_NUM = 398;
constexpr uint32_t PAIR_DIRECT_DEN = 625;
constexpr uint32_t PAIR_DIRECT_NUM_LOW = 319;
constexpr uint32_t PAIR_DIRECT_DEN_LOW = 500;
constexpr uint32_t UNEVEN_DIRECT_NUM = 11;
constexpr uint32_t UNEVEN_DIRECT_DEN = 14;
constexpr uint32_t TRI_DIRECT_NUM = 813;
constexpr uint32_t TRI_DIRECT_DEN = 1000;
constexpr uint64_t MIXED_MESH_MODE = 5;
constexpr uint64_t LINKED_MIXED_MESH_MODE = 7;
constexpr uint64_t LINKED_CLOS_READ_MODE = 8;
constexpr uint64_t UNCONFIRMED_READ_MODE = 9;
constexpr uint64_t TRI_PULL_READ_MODE = 10;
constexpr uint32_t ROLE_FAST_READ_ARGS = 5;
constexpr uint32_t ROLE_UNIFIED_WRITE_ARGS = 11;
constexpr uint32_t MIXED_MESH_READS = 5;
constexpr uint32_t MIXED_MESH_WRITES = 2;
enum class Topology : uint32_t { Generic, TwoByEight, FourByOne, EightPlusFour, FourByThree };
enum class Algorithm : uint32_t { DirectWrite, DirectRead, PairRelay, UnevenRelay, TriRelay };
enum class Buffer : uint32_t { Input, Output, Scratch };

// 所有 rank 仅按共同的 root、长度和建链范围选择协议；不按本地指针单方面回退。
inline bool UseLinkedClosRead(Topology topology, uint32_t root, uint32_t registeredRoot,
    uint64_t totalBytes, uint64_t linkedInputBytes, uint32_t candidate)
{
    return candidate == 0 && (topology == Topology::TwoByEight || topology == Topology::FourByThree ||
        topology == Topology::EightPlusFour) &&
        root == registeredRoot && totalBytes != 0 && totalBytes <= SMALL_TOTAL_BYTES &&
        totalBytes <= linkedInputBytes;
}

// 4×1 单独实验：只在共同的初始 root 与建链输入范围内取消读完成通知。
inline bool UseRoleLinkedRead(uint32_t root, uint32_t registeredRoot,
    uint64_t totalBytes, uint64_t linkedInputBytes)
{
    return root == registeredRoot && totalBytes != 0 && totalBytes <= SMALL_TOTAL_BYTES &&
        totalBytes <= linkedInputBytes;
}

// 各 rank 根据公共调用参数选择同一种大消息刷新协议，不依赖本地缓存判断远端地址。
inline bool UseLargeReadFastPath(Topology topology, uint32_t root, uint64_t totalBytes)
{
    return topology == Topology::EightPlusFour && root < 8 && totalBytes >= RELAY_TOTAL_BYTES;
}

// 初始输入容量由所有 rank 共同持有，不能按单端地址变化改变通信协议。
inline bool UseUnevenPull(Topology topology, uint32_t root, uint32_t registeredRoot,
    uint64_t totalBytes, uint64_t linkedInputBytes, uint32_t candidate)
{
    return candidate == 0 && topology == Topology::EightPlusFour && root < 8 &&
        root == registeredRoot && totalBytes >= RELAY_TOTAL_BYTES &&
        totalBytes <= linkedInputBytes && totalBytes / 12 <= TRANSFER_LIMIT;
}

inline bool UseSmallRefreshRead(Topology topology, uint64_t totalBytes)
{
    return totalBytes <= SMALL_TOTAL_BYTES &&
        (topology == Topology::TwoByEight || topology == Topology::EightPlusFour);
}

// V13 把大机 Mesh 也改成定向写后两个大档退回 924/725（与 V10 相同），故恢复机内 Read。
inline bool UseDirectedWrite(Topology topology, uint32_t root, uint64_t totalBytes, bool mesh)
{
    return (topology == Topology::FourByOne && totalBytes <= SMALL_TOTAL_BYTES) ||
        (UseLargeReadFastPath(topology, root, totalBytes) && !mesh);
}

// 大机 Mesh 的正式 Read mission 保留小消息刷新 Read，并承载大消息五读两写。
inline bool UseMeshReadWriteOnly(Topology topology, uint32_t rank, bool mesh, uint32_t candidate)
{
    return candidate == 0 && topology == Topology::EightPlusFour && rank < 8 && mesh;
}

inline bool UseMixedMesh(Topology topology, uint32_t root, uint64_t totalBytes, bool mesh,
                         uint32_t candidate)
{
    return candidate == 0 && mesh && UseLargeReadFastPath(topology, root, totalBytes);
}

// 选本轮 root 七个升序机内对端的最后两个，所有 rank 可独立得到相同结果。
inline std::array<uint32_t, MIXED_MESH_WRITES> MixedMeshWritePeers(uint32_t root)
{
    std::array<uint32_t, MIXED_MESH_WRITES> peers{UINT32_MAX, UINT32_MAX};
    if (root >= 8) return peers;
    uint32_t position = 0;
    for (uint32_t rank = 0; rank < 8; ++rank) {
        if (rank == root) continue;
        if (position >= MIXED_MESH_READS) peers[position - MIXED_MESH_READS] = rank;
        ++position;
    }
    return peers;
}

inline bool IsMixedMeshWritePeer(uint32_t rank, uint32_t root)
{
    const auto peers = MixedMeshWritePeers(root);
    return rank == peers[0] || rank == peers[1];
}

inline bool UseRoleWritePath(Topology topology, uint32_t candidate)
{
    return candidate == 0 && topology == Topology::FourByOne;
}

// 固定前缀只服务已有直接路径；中继与强制候选仍读取完整上下文。
inline bool UseFastReadContext(Topology topology, uint32_t root, uint64_t totalBytes, uint32_t candidate)
{
    return candidate == 0 && (topology == Topology::TwoByEight ||
        topology == Topology::EightPlusFour || topology == Topology::FourByThree) &&
        (totalBytes <= SMALL_TOTAL_BYTES || UseLargeReadFastPath(topology, root, totalBytes));
}

// 仅大机 root 的大消息把自拷贝迁到 Mesh；其他调用保留默认复制组。
inline bool UseReadCopyGroup(Topology topology, uint32_t root, uint64_t totalBytes,
                             bool mesh, bool defaultCopy)
{
    return UseLargeReadFastPath(topology, root, totalBytes) ? mesh : defaultCopy;
}

constexpr uint32_t ROLE_SEND_ARGS = 9;
constexpr uint32_t ROLE_RECEIVE_ARGS = 4;
constexpr uint32_t ROLE_LINK_SEND_ARGS = ROLE_SEND_ARGS + 1;
constexpr uint32_t ROLE_LINK_RECEIVE_ARGS = ROLE_RECEIVE_ARGS + 4;
constexpr uint64_t ROLE_LINK_READ_MODE = 2;
static_assert(ROLE_LINK_SEND_ARGS <= 13 && ROLE_LINK_RECEIVE_ARGS <= 13,
              "建链免 READY 实验保持单 SQE 参数容量");

// 免 READY 选择只依赖通信域一致的初始长度和本轮长度，不按某个 rank 的地址单独回退。
inline bool UseRoleLinkWrite(uint64_t linkedBytes, uint64_t rankBytes)
{
    return linkedBytes != 0 && rankBytes != 0 && rankBytes <= linkedBytes &&
        rankBytes <= TRANSFER_LIMIT;
}

// 0 沿用动态元数据；1 使用建链 Output；2 使用建链 Scratch。
// 只编码缓冲区用途，不改变原有中继阶段及完成协议。
constexpr uint64_t LINK_WRITE_MODE = 6;
inline bool UseLinkedWrite(Topology topology, uint32_t root, uint64_t totalBytes, bool mesh)
{
    return topology != Topology::Generic &&
        (totalBytes <= SMALL_TOTAL_BYTES || (UseLargeReadFastPath(topology, root, totalBytes) && !mesh));
}

template<size_t N>
inline std::array<uint64_t, N + 1> AppendRoleLinkMode(const std::array<uint64_t, N> &args,
                                                    uint64_t mode)
{
    std::array<uint64_t, N + 1> linked{};
    std::copy(args.begin(), args.end(), linked.begin());
    linked[N] = mode;
    return linked;
}
// 通用 Write 回退使用原发送字段，正式 auto 在前面追加角色槽。
inline std::array<uint64_t, ROLE_SEND_ARGS> PackRoleSendArguments(uint32_t root,
    const uint32_t *peers, uint64_t rankBytes, uint64_t offset, uint64_t bytes,
    const std::array<uint64_t, 3> &addresses, const std::array<uint64_t, 3> &tokens)
{
    const uint64_t source = addresses[0] + root * rankBytes + offset;
    const uint64_t output = addresses[1] + offset;
    return {tokens[0], addresses[0] + peers[0] * rankBytes + offset,
        addresses[0] + peers[1] * rankBytes + offset,
        addresses[0] + peers[2] * rankBytes + offset, bytes,
        source, output, tokens[1], source == output ? 0 : bytes};
}

inline std::array<uint64_t, ROLE_RECEIVE_ARGS> PackRoleReceiveArguments(uint32_t root,
    const uint32_t *peers, uint64_t output, uint64_t token, bool publishAddress)
{
    uint64_t role = 3;
    for (uint32_t i = 0; i < 3; ++i) {
        if (peers[i] == root) role = i;
    }
    return {role, output, token, publishAddress ? 1ULL : 0ULL};
}

// 接收 mission 共用八槽布局；Write 分支忽略最后三槽，Read 分支使用本轮输出。
inline std::array<uint64_t, ROLE_LINK_RECEIVE_ARGS> PackRoleLinkReceiveArguments(
    const std::array<uint64_t, ROLE_RECEIVE_ARGS> &args, uint64_t mode,
    uint64_t sourceAddress = 0, uint64_t sourceToken = 0, uint64_t bytes = 0)
{
    return {args[0], args[1], args[2], args[3], mode, sourceAddress, sourceToken, bytes};
}

// 读写定向协议共用十一项参数；直写的槽2、槽6分别表示首个对端源地址和 rank 步长。
inline std::array<uint64_t, 11> PackReadArguments(uint32_t rank, uint32_t root, uint32_t firstPeer,
    bool selfCopy, uint64_t rankBytes, uint64_t offset, uint64_t bytes, uint64_t mode,
    bool directedWrite, const std::array<uint64_t, 3> &addresses,
    const std::array<uint64_t, 3> &tokens, uint64_t registeredAddress)
{
    const uint64_t sourceOffset = rank * rankBytes + offset;
    const uint64_t output = addresses[1] + offset;
    const uint64_t copySource = rank == root ? addresses[0] + sourceOffset : 0;
    const uint64_t copyBytes = rank == root && selfCopy && copySource != output ? bytes : 0;
    const uint64_t input = rank == root ? addresses[0] +
        (directedWrite ? firstPeer * rankBytes + offset : 0) : 0;
    return {root, mode, input, tokens[0], output, tokens[1],
        directedWrite ? rankBytes : sourceOffset, bytes, copySource, copyBytes, registeredAddress};
}

// 复用十一槽布局；Clos 不承担自拷贝，源地址直接在 host 计算。
inline std::array<uint64_t, 11> PackLinkedClosReadArguments(uint32_t rank, uint32_t root,
    uint64_t rankBytes, uint64_t offset, uint64_t bytes, uint64_t linkedInputAddress,
    uint64_t linkedInputToken, uint64_t outputAddress, uint64_t outputToken)
{
    return {root, LINKED_CLOS_READ_MODE,
        rank == root ? 0 : linkedInputAddress + rank * rankBytes + offset,
        rank == root ? 0 : linkedInputToken, rank == root ? 0 : outputAddress + offset,
        rank == root ? 0 : outputToken, 0, bytes, 0, 0, 0};
}

// 五读两写仍用十一参数：仅 root 的槽6、槽10改为两笔 Write 的源地址。
inline std::array<uint64_t, 11> PackMixedMeshArguments(uint32_t rank, uint32_t root,
    uint64_t rankBytes, uint64_t offset, uint64_t bytes,
    const std::array<uint64_t, 3> &addresses, const std::array<uint64_t, 3> &tokens, bool linked = false)
{
    auto args = PackReadArguments(rank, root, 0, true, rankBytes, offset, bytes,
        linked ? LINKED_MIXED_MESH_MODE : MIXED_MESH_MODE, false, addresses, tokens, 0);
    if (rank == root) {
        const auto peers = MixedMeshWritePeers(root);
        args[6] = addresses[0] + peers[0] * rankBytes + offset;
        args[10] = addresses[0] + peers[1] * rankBytes + offset;
    }
    return args;
}

struct Transfer {
    uint32_t phase;
    uint32_t source;
    uint32_t target;
    Buffer sourceBuffer;
    Buffer targetBuffer;
    uint64_t sourceOffset;
    uint64_t targetOffset;
    uint64_t bytes;
};

struct Plan {
    Algorithm algorithm = Algorithm::DirectWrite;
    uint32_t phases = 1;
    std::vector<Transfer> transfers;
    std::array<uint64_t, MAX_RANKS> scratchBytes{};
};

inline uint64_t LinkedTargetMode(const std::vector<std::vector<const Transfer *>> &outgoing,
                                bool enabled)
{
    if (!enabled) return 0;
    uint64_t mode = 0;
    for (const auto &edges : outgoing) {
        for (const auto *edge : edges) {
            const uint64_t next = edge->targetBuffer == Buffer::Output ? 1 :
                (edge->targetBuffer == Buffer::Scratch ? 2 : 3);
            if (next == 3 || (mode != 0 && mode != next)) return 3;
            mode = next;
        }
    }
    return mode == 0 ? 1 : mode;
}

inline bool SameServer(Topology topology, uint32_t a, uint32_t b)
{
    if (topology == Topology::TwoByEight || topology == Topology::EightPlusFour) {
        return a / 8 == b / 8;
    }
    if (topology == Topology::FourByThree) {
        return a / 3 == b / 3;
    }
    return a == b;
}

inline Algorithm Select(Topology topology, uint32_t root, uint64_t totalBytes, uint32_t candidate)
{
    if (candidate == 1) {
        return Algorithm::DirectWrite;
    }
    if (candidate == 2 || (candidate == 0 && totalBytes <= SMALL_TOTAL_BYTES)) {
        return Algorithm::DirectRead;
    }
    if (topology == Topology::TwoByEight &&
        (candidate == 3 || (candidate == 0 && totalBytes >= RELAY_TOTAL_BYTES))) {
        return Algorithm::PairRelay;
    }
    if (candidate == 0 && topology == Topology::EightPlusFour && root < 8 &&
        totalBytes >= RELAY_TOTAL_BYTES) {
        return Algorithm::DirectRead;
    }
    if (topology == Topology::EightPlusFour && root >= 8 &&
        (candidate == 4 || (candidate == 0 && totalBytes >= RELAY_TOTAL_BYTES))) {
        return Algorithm::UnevenRelay;
    }
    if (topology == Topology::FourByThree &&
        (candidate == 5 || (candidate == 0 && totalBytes >= RELAY_TOTAL_BYTES))) {
        return Algorithm::TriRelay;
    }
    return Algorithm::DirectWrite;
}

// 先乘商再乘分子，最后处理余数；避免大整数乘法溢出和整除造成的大量偏差。
inline uint64_t Fraction(uint64_t elements, uint64_t numerator, uint64_t denominator)
{
    return (elements / denominator) * numerator + (elements % denominator) * numerator / denominator;
}

// 只替换整片能够容纳的直达段；中继尾段仍按原 Write 协议消费 Scratch。
inline bool UseTriPull(Topology topology, uint32_t root, uint32_t registeredRoot,
    uint64_t rankBytes, uint64_t linkedInputBytes, uint64_t linkedOutputBytes,
    uint64_t scratchBytes, uint32_t candidate)
{
    if (candidate != 0 || topology != Topology::FourByThree || root != registeredRoot ||
        rankBytes * 12 < RELAY_TOTAL_BYTES || rankBytes > TRANSFER_LIMIT ||
        rankBytes * 12 > linkedInputBytes || rankBytes > linkedOutputBytes) return false;
    const uint64_t elements = rankBytes / 4;
    const uint64_t tail = elements - Fraction(elements, TRI_DIRECT_NUM, TRI_DIRECT_DEN);
    return 9 * ((tail + 1) / 2) * 4 <= scratchBytes;
}

// 2×8 按单 rank 接收量选直达比例；两档各取自己已测到的最好值，不引入未测过的比例。
inline uint64_t PairDirectElements(uint64_t elements, uint64_t rankBytes)
{
    return rankBytes >= PAIR_RATIO_SPLIT_RANK_BYTES ?
        Fraction(elements, PAIR_DIRECT_NUM, PAIR_DIRECT_DEN) :
        Fraction(elements, PAIR_DIRECT_NUM_LOW, PAIR_DIRECT_DEN_LOW);
}

inline Plan Build(uint32_t ranks, uint32_t root, Algorithm algorithm, uint64_t rankBytes,
                  uint64_t offset, uint64_t sliceBytes, uint32_t elementBytes)
{
    Plan plan;
    plan.algorithm = algorithm;
    plan.phases = (algorithm == Algorithm::PairRelay || algorithm == Algorithm::UnevenRelay ||
                   algorithm == Algorithm::TriRelay) ? 2 : 1;
    const uint64_t elements = sliceBytes / elementBytes;
    auto add = [&](uint32_t phase, uint32_t source, uint32_t target, Buffer sourceBuffer,
                   Buffer targetBuffer, uint64_t sourceOffset, uint64_t targetOffset, uint64_t bytes) {
        if (bytes != 0) {
            plan.transfers.push_back({phase, source, target, sourceBuffer, targetBuffer,
                                      sourceOffset, targetOffset, bytes});
        }
    };
    // root 自己的分片总是通过本地拷贝完成；这也让 Read 候选不依赖 root 的远端自环。
    add(plan.phases - 1, root, root, Buffer::Input, Buffer::Output,
        root * rankBytes + offset, offset, sliceBytes);
    for (uint32_t target = 0; target < ranks; ++target) {
        if (target == root) {
            continue;
        }
        const uint64_t inputOffset = target * rankBytes + offset;
        if (algorithm == Algorithm::TriRelay) {
            // 4×3：root 同机的两个搭档按 1:1 分担 5/27 的代发尾段，直达部分占 22/27。
            const uint32_t p0 = root / 3 * 3 + (root + 1) % 3;
            const uint32_t p1 = root / 3 * 3 + (root + 2) % 3;
            if (target == p0 || target == p1) {
                // 同机分片整条走 Mesh 直写对方 Output，不占用暂存区。
                add(1, root, target, Buffer::Input, Buffer::Output, inputOffset, offset, sliceBytes);
                continue;
            }
            const uint64_t directElements = Fraction(elements, TRI_DIRECT_NUM, TRI_DIRECT_DEN);
            const uint64_t directBytes = directElements * elementBytes;
            // 先取完整 5/27 尾段再按 1:1 拆分，保证 直达 + 两段中继 == 整条切片。
            const uint64_t relay0Elements = Fraction(elements - directElements, 1, 2);
            const uint64_t relay1Elements = elements - directElements - relay0Elements;
            const uint64_t relay0Bytes = relay0Elements * elementBytes;
            const uint64_t relay1Bytes = relay1Elements * elementBytes;
            // 第一阶段 Mesh 负载为 5/6，第二阶段为 1：直达 Clos 按 5:6 分摊。
            const uint64_t earlyBytes = Fraction(directElements, 5, 11) * elementBytes;
            add(0, root, target, Buffer::Input, Buffer::Output, inputOffset, offset, earlyBytes);
            add(1, root, target, Buffer::Input, Buffer::Output, inputOffset + earlyBytes,
                offset + earlyBytes, directBytes - earlyBytes);
            const uint64_t scratchOffset0 = plan.scratchBytes[p0];
            add(0, root, p0, Buffer::Input, Buffer::Scratch,
                inputOffset + directBytes, scratchOffset0, relay0Bytes);
            add(1, p0, target, Buffer::Scratch, Buffer::Output,
                scratchOffset0, offset + directBytes, relay0Bytes);
            plan.scratchBytes[p0] += relay0Bytes;
            const uint64_t scratchOffset1 = plan.scratchBytes[p1];
            add(0, root, p1, Buffer::Input, Buffer::Scratch,
                inputOffset + directBytes + relay0Bytes, scratchOffset1, relay1Bytes);
            add(1, p1, target, Buffer::Scratch, Buffer::Output,
                scratchOffset1, offset + directBytes + relay0Bytes, relay1Bytes);
            plan.scratchBytes[p1] += relay1Bytes;
            continue;
        }
        if (plan.phases == 1 || (SameServer(algorithm == Algorithm::UnevenRelay ?
                Topology::EightPlusFour : Topology::TwoByEight, root, target) && target != root)) {
            // 中继阶段 0 只准备代发段；同机整份分片留到阶段 1，与代发和后续 Clos 直发并行。
            add(plan.phases - 1, root, target, Buffer::Input, Buffer::Output,
                inputOffset, offset, sliceBytes);
            continue;
        }
        if (algorithm == Algorithm::PairRelay) {
            const uint32_t relay = (target + 8) % 16;
            const uint64_t directElements = relay == root ? elements :
                PairDirectElements(elements, rankBytes);
            const uint64_t directBytes = directElements * elementBytes;
            // 保留逻辑两段格式；实际执行前由 RelayClosPlan 合并为连续 Clos，4:15 不影响总直达比例。
            const uint64_t earlyBytes = Fraction(directElements, 4, 15) * elementBytes;
            add(0, root, target, Buffer::Input, Buffer::Output, inputOffset, offset, earlyBytes);
            add(1, root, target, Buffer::Input, Buffer::Output, inputOffset + earlyBytes,
                offset + earlyBytes, directBytes - earlyBytes);
            if (relay != root) {
                const uint64_t relayBytes = sliceBytes - directBytes;
                const uint64_t scratchOffset = plan.scratchBytes[relay];
                add(0, root, relay, Buffer::Input, Buffer::Scratch,
                    inputOffset + directBytes, scratchOffset, relayBytes);
                add(1, relay, target, Buffer::Scratch, Buffer::Output,
                    scratchOffset, offset + directBytes, relayBytes);
                plan.scratchBytes[relay] += relayBytes;
            }
        } else {
            const uint64_t directElements = Fraction(elements, UNEVEN_DIRECT_NUM, UNEVEN_DIRECT_DEN);
            const uint64_t directBytes = directElements * elementBytes;
            // 小机每条 Mesh 的第一阶段负载为 4/7，第二阶段为 1：直达流量按 4:7 分摊。
            const uint64_t earlyBytes = Fraction(directElements, 4, 11) * elementBytes;
            add(0, root, target, Buffer::Input, Buffer::Output, inputOffset, offset, earlyBytes);
            add(1, root, target, Buffer::Input, Buffer::Output, inputOffset + earlyBytes,
                offset + earlyBytes, directBytes - earlyBytes);
            const uint64_t tailElements = elements - directElements;
            uint32_t helper = 0;
            for (uint32_t relay = 8; relay < 12; ++relay) {
                if (relay == root) {
                    continue;
                }
                const uint64_t begin = Fraction(tailElements, helper, 3) * elementBytes;
                const uint64_t end = Fraction(tailElements, helper + 1, 3) * elementBytes;
                const uint64_t scratchOffset = plan.scratchBytes[relay];
                add(0, root, relay, Buffer::Input, Buffer::Scratch,
                    inputOffset + directBytes + begin, scratchOffset, end - begin);
                add(1, relay, target, Buffer::Scratch, Buffer::Output,
                    scratchOffset, offset + directBytes + begin, end - begin);
                plan.scratchBytes[relay] += end - begin;
                ++helper;
            }
        }
    }
    return plan;
}
// 2×8 和 4×3 的 Clos 数据在目标端互不重叠；同源同目标的两个连续直达段合为一笔。
// 中继读取暂存区的依赖由 host 的 Mesh→Clos 单向交接保证，不能依赖目标端等待代替。
inline Plan RelayClosPlan(const Plan &original, Topology topology)
{
    Plan clos;
    clos.algorithm = original.algorithm;
    clos.transfers.reserve(topology == Topology::FourByThree ? 27 : 15);
    for (const auto &edge : original.transfers) {
        if (SameServer(topology, edge.source, edge.target)) {
            continue;
        }
        bool merged = false;
        for (auto &previous : clos.transfers) {
            if (previous.source == edge.source && previous.target == edge.target &&
                previous.sourceBuffer == edge.sourceBuffer && previous.targetBuffer == edge.targetBuffer &&
                previous.sourceOffset + previous.bytes == edge.sourceOffset &&
                previous.targetOffset + previous.bytes == edge.targetOffset) {
                previous.bytes += edge.bytes;
                merged = true;
                break;
            }
        }
        if (!merged) {
            clos.transfers.push_back(edge);
            clos.transfers.back().phase = 0;
        }
    }
    return clos;
}

// 4×3 逻辑搬运描述：活动波数、两条通道的首段四元组、源步长和跨本机跳距。
// Mesh 的两条通道分别承载两个中继尾段；Clos 的九条通道共用一个首段和地址步进。
// Mesh 增加阶段模式与本 rank 的 Output 地址/token：阶段0 一次发布 Scratch 与 Output，
// 阶段1 直接用缓存的 Output，不再 publish 也不再等 READY。
constexpr uint32_t TRI_MESH_WRITE_ARGS = 20;
constexpr uint32_t TRI_CLOS_WRITE_ARGS = 10;
constexpr uint32_t PAIR_MESH_WRITE_ARGS = 19;
constexpr uint32_t PAIR_CLOS_WRITE_ARGS = 11;
// 只裁剪非 root 服务器之间的 Clos 握手；直写和空尾段仍沿用这一对称参与集合。
inline bool TriClosPeerActive(uint32_t rank, uint32_t peer, uint32_t rootServer)
{
    return rank / 3 == rootServer || peer / 3 == rootServer;
}

inline std::array<uint64_t, 11> PackCompactTransfers(
    const std::vector<std::vector<const Transfer *>> &outgoing, const uint32_t *peers,
    uint32_t rank, uint64_t rankBytes, const std::array<uint64_t, 3> &addresses,
    const std::array<uint64_t, 3> &tokens, bool hasCopy)
{
    std::array<uint64_t, 11> result{};
    result[0] = hasCopy ? 1 : 0;
    const bool mesh = SameServer(Topology::FourByThree, rank, peers[0]);
    auto pack = [&](uint32_t lane, const Transfer &edge) {
        const auto buffer = static_cast<size_t>(edge.sourceBuffer);
        const uint32_t begin = 1 + lane * 4;
        result[begin] = addresses[buffer] + edge.sourceOffset;
        result[begin + 1] = tokens[buffer];
        result[begin + 2] = edge.targetOffset;
        result[begin + 3] = edge.bytes;
    };
    if (mesh) {
        for (uint32_t i = 0; i < outgoing.size(); ++i) {
            result[0] = std::max(result[0], static_cast<uint64_t>(outgoing[i].size()));
            if (!outgoing[i].empty()) {
                pack(i, *outgoing[i][0]);
            }
        }
        result[9] = rankBytes;
        result[10] = 3 * rankBytes;
    } else {
        const Transfer *first = nullptr;
        for (const auto &edges : outgoing) {
            if (!edges.empty() && (first == nullptr || edges[0]->target < first->target)) {
                first = edges[0];
            }
        }
        if (first != nullptr) {
            result[0] = 1;
            pack(0, *first);
            const bool input = first->sourceBuffer == Buffer::Input;
            result[9] = input ? rankBytes : first->bytes;
            result[10] = input ? 3 * rankBytes : 0;
        }
    }
    return result;
}

// 保留上述逻辑搬运描述；下发时去掉本组不使用或相同的字段。
inline std::vector<uint64_t> PackTriWriteArguments(
    const std::vector<uint64_t> &common,
    const std::vector<std::vector<const Transfer *>> &outgoing, const uint32_t *peers,
    uint32_t rank, uint64_t rankBytes, const std::array<uint64_t, 3> &addresses,
    const std::array<uint64_t, 3> &tokens, uint32_t rootServer, uint64_t mode)
{
    const auto packed = PackCompactTransfers(outgoing, peers, rank, rankBytes,
                                             addresses, tokens, common[6] != 0);
    if (!SameServer(Topology::FourByThree, rank, peers[0])) {
        // Clos 无本地拷贝，九个对端共用首段；每轮刷新 root 服务器以选择握手对端。
        return {common[0], common[1], packed[0], packed[1], packed[2], packed[3],
                packed[4], packed[9], packed[10], rootServer};
    }
    // Mesh 只有 root 从 Input 发送；两条通道首段目标偏移相同。
    const uint64_t targetOffset = packed[4] != 0 ? packed[3] : packed[7];
    return {common[0], common[1], mode, addresses[1], tokens[1],
            common[2], common[4], common[6], packed[0], tokens[0],
            targetOffset, packed[1], packed[4], packed[5], packed[8], packed[9], 0, 0, 0, 0};
}

// 在原阶段0布局后填充阶段1：先释放 Scratch，再写 Output；复制只属于后半段。
inline void PackTriRootFinal(std::vector<uint64_t> &args, const Plan &plan, uint32_t root,
    const uint32_t *peers, const std::array<uint64_t, 3> &addresses)
{
    for (const auto &edge : plan.transfers) {
        if (edge.phase != 1 || edge.source != root) continue;
        if (edge.target == root) {
            args[5] = addresses[0] + edge.sourceOffset;
            args[6] = addresses[1] + edge.targetOffset;
            args[7] = args[5] == args[6] ? 0 : edge.bytes;
        }
        for (uint32_t i = 0; i < 2; ++i) {
            if (edge.target == peers[i]) {
                args[16 + i] = addresses[0] + edge.sourceOffset;
                args[18] = edge.bytes;
                args[19] = edge.targetOffset;
            }
        }
    }
}

inline std::vector<uint64_t> PackPairMeshArguments(
    const std::vector<uint64_t> &common,
    const std::vector<std::vector<const Transfer *>> &outgoing, uint64_t mode,
    const std::array<uint64_t, 3> &addresses, const std::array<uint64_t, 3> &tokens)
{
    const Transfer *first = nullptr;
    for (const auto &edges : outgoing) {
        if (!edges.empty()) {
            first = edges[0];
            break;
        }
    }
    // 保留各对端源地址；只共享 Input token、目标偏移和发送长度。
    std::vector<uint64_t> result{common[0], common[1], common[2], common[4], common[6],
        mode, addresses[1], tokens[1], first != nullptr || common[6] != 0 ? 1ULL : 0ULL,
        tokens[0], first == nullptr ? 0 : first->targetOffset, first == nullptr ? 0 : first->bytes};
    for (const auto &edges : outgoing) {
        result.push_back(edges.empty() ? 0 : addresses[0] + edges[0]->sourceOffset);
    }
    return result;
}

inline std::vector<uint64_t> PackPairClosArguments(
    const std::vector<uint64_t> &common,
    const std::vector<std::vector<const Transfer *>> &outgoing, uint32_t rank, uint32_t root,
    uint64_t rankBytes, const std::array<uint64_t, 3> &addresses,
    const std::array<uint64_t, 3> &tokens, bool pairRelay)
{
    const Transfer *first = nullptr;
    uint64_t normalBytes = 0;
    uint64_t buddyBytes = 0;
    for (const auto &edges : outgoing) {
        if (edges.empty()) {
            continue;
        }
        const auto &edge = *edges[0];
        first = first == nullptr ? &edge : first;
        if (edge.target % 8 == rank % 8) {
            buddyBytes = edge.bytes;
        } else {
            normalBytes = edge.bytes;
        }
    }
    // 4 字节尾片可能只有 buddy 一条非零直达边；不能按边数判断 root。
    const uint64_t role = rank == root ? 1 :
        (SameServer(Topology::TwoByEight, rank, root) ? (pairRelay ? 2 : 3) : 0);
    uint64_t sourceAddress = 0;
    uint64_t sourceToken = 0;
    uint64_t targetOffset = 0;
    if (first != nullptr) {
        const auto buffer = static_cast<size_t>(first->sourceBuffer);
        sourceAddress = addresses[buffer] + first->sourceOffset;
        sourceToken = tokens[buffer];
        targetOffset = first->targetOffset;
        if (role == 1) {
            sourceAddress -= (first->target % 8) * rankBytes;
        } else {
            normalBytes = first->bytes;
        }
    }
    return {common[0], common[1], role, sourceAddress, sourceToken, targetOffset,
            role == 1 ? rankBytes : 0, normalBytes, buddyBytes, root, pairRelay ? 1ULL : 0ULL};
}
} // namespace scatter_plan

// 同一份调度代码也供无 CANN 依赖的区间来源检查使用。
#ifndef SCATTER_PLAN_ONLY
#include "binary_stream.h"
#include "common.h"

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

enum class WriteOptimization : uint32_t { Baseline, FusedMetadata, ChannelReady, ChannelFinished, Compact };
enum class WriteMetadataMode : uint64_t { Refresh, PublishBoth, ReuseOutput, RootFusedBoth };

// 正式 auto 建链交换接收缓冲区；凭据不写日志。沿用平台已通过的输出可提前覆盖实验前提。
constexpr uint64_t ROLE_LINK_MAGIC = 0x5343415456323501ULL;
struct RoleLinkDescriptor {
    uint64_t magic = ROLE_LINK_MAGIC;
    uint64_t rank = 0;
    uint64_t address = 0;
    uint64_t bytes = 0;
    uint64_t token = 0;
    uint64_t scratchAddress = 0;
    uint64_t scratchBytes = 0;
    uint64_t scratchToken = 0;
};
static_assert(sizeof(RoleLinkDescriptor) == 8 * sizeof(uint64_t), "建链描述必须保持固定字节布局");

struct InputLinkDescriptor {
    uint64_t address = 0;
    uint64_t bytes = 0;
    uint64_t token = 0;
};

// 原接收/暂存区描述保持不变，额外交换初始 root 的输入凭据。
struct LinkExchangeDescriptor {
    RoleLinkDescriptor output;
    InputLinkDescriptor input;
};
static_assert(sizeof(LinkExchangeDescriptor) == 11 * sizeof(uint64_t), "建链包必须保持固定字节布局");

struct ScatterKernelArg {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t ranks[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
    uint32_t rank = 0;
    uint32_t selfCopy = 0;
    uint32_t slots = 1;
    WriteOptimization optimization = WriteOptimization::Baseline;
    // 4×1 与 8+4 Clos 编译定向写分支。
    uint32_t directedWrite = 0;
    // 正式 auto 的 4×1 两个 mission 分别编译统一 Write 和固定角色 Read。
    uint32_t roleWriteOnly = 0;
    uint64_t roleLinkBytes = 0;
    RoleLinkDescriptor roleLinkTargets[scatter_plan::MAX_RANKS]{};
    // 正式 8+4 大机 Mesh 同时保留刷新 Read、五读两写回退及预交换 Read。
    uint32_t meshReadWriteOnly = 0;
    uint32_t registeredRoot = INVALID_VALUE_RANKID;
};

enum class ReadMetadataMode : uint64_t {
    RegisteredAddress = 1,
    Refresh = 2,
    LargeRefresh = 3,
    DirectedWrite = 4,
    MixedMesh = scatter_plan::MIXED_MESH_MODE,
};

// 4×1 热路径直接读取固定布局，避免每轮复制 vector 和反序列化通用资源。
constexpr uint64_t ROLE_CTX_MAGIC = 0x5343415456323502ULL;
struct RoleResourceCtx {
    uint64_t magic = ROLE_CTX_MAGIC;
    ThreadHandle mainThread{};
    CcuKernelHandle sender{}, receiver{};
    uint32_t peers[3]{};
    uint32_t publishedRoot = INVALID_VALUE_RANKID;
    uint64_t publishedOutput = 0;
    RoleLinkDescriptor linkedOutput;
    InputLinkDescriptor linkedInput;
    uint32_t registeredRoot = INVALID_VALUE_RANKID;
};

// 固定布局不含 channel 句柄和仅注册期使用的对端描述；完整旧布局跟在前缀后用于回退。
constexpr uint64_t FAST_CTX_MAGIC = 0x5343415456323503ULL;
struct FastReadGroup {
    uint32_t ranks[scatter_plan::MAX_RANKS]{};
    uint32_t channelCount = 0;
    uint32_t selfCopy = 0;
};

struct FastReadResourceCtx {
    uint64_t magic = FAST_CTX_MAGIC;
    uint64_t fallbackBytes = 0;
    RoleLinkDescriptor linkedOutput;
    CommBuffer registeredInput;
    InputLinkDescriptor linkedInput;
    uint32_t registeredRoot = INVALID_VALUE_RANKID;
    uint32_t selfGroup = 0;
    scatter_plan::Topology topology = scatter_plan::Topology::Generic;
    std::array<ThreadHandle, 2> threads{};
    std::array<FastReadGroup, 2> kernelArgs{};
    std::array<CcuKernelHandle, 2> readKernels{};
    std::array<uint32_t, scatter_plan::MAX_RANKS> rootGroups{};
};
static_assert(sizeof(FastReadResourceCtx) <= 384, "直接路径前缀必须保持紧凑");

struct AlgResourceCtx {
    CommBuffer scratch;
    RoleLinkDescriptor linkedOutput;
    // 建链时交换的初始 root 输入；后续更换地址时由 Read 协议刷新，继续复用原 channel。
    CommBuffer registeredInput;
    InputLinkDescriptor linkedInput;
    uint32_t registeredRoot = INVALID_VALUE_RANKID;
    uint32_t selfGroup = 0;
    scatter_plan::Topology topology = scatter_plan::Topology::Generic;
    std::vector<ThreadHandle> threads;
    std::vector<ScatterKernelArg> kernelArgs;
    std::vector<CcuKernelHandle> writeKernels;
    std::vector<CcuKernelHandle> readKernels;
    std::vector<CcuKernelHandle> relayKernels;

    std::vector<char> Serialize()
    {
        BinaryStream stream;
        stream << scratch << linkedOutput << registeredInput << linkedInput << registeredRoot << selfGroup
               << topology << threads << kernelArgs << writeKernels << readKernels << relayKernels;
        std::vector<char> data;
        stream.Dump(data);
        return data;
    }
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream stream(data);
        stream >> scratch >> linkedOutput >> registeredInput >> linkedInput >> registeredRoot >> selfGroup
               >> topology >> threads >> kernelArgs >> writeKernels >> readKernels >> relayKernels;
    }
};

// 调用者先验证两个完整分组；此处只提取每轮下发真正需要的固定字段。
inline FastReadResourceCtx MakeFastReadContext(const AlgResourceCtx &ctx, uint64_t fallbackBytes)
{
    FastReadResourceCtx fixed;
    fixed.fallbackBytes = fallbackBytes;
    fixed.linkedOutput = ctx.linkedOutput;
    fixed.registeredInput = ctx.registeredInput;
    fixed.linkedInput = ctx.linkedInput;
    fixed.registeredRoot = ctx.registeredRoot;
    fixed.selfGroup = ctx.selfGroup;
    fixed.topology = ctx.topology;
    fixed.rootGroups.fill(UINT32_MAX);
    std::copy_n(ctx.threads.begin(), 2, fixed.threads.begin());
    std::copy_n(ctx.readKernels.begin(), 2, fixed.readKernels.begin());
    for (uint32_t group = 0; group < 2; ++group) {
        const auto &from = ctx.kernelArgs[group];
        auto &to = fixed.kernelArgs[group];
        to.channelCount = from.channelCount;
        to.selfCopy = from.selfCopy;
        std::copy_n(from.ranks, from.channelCount, to.ranks);
        for (uint32_t i = 0; i < from.channelCount; ++i) fixed.rootGroups[from.ranks[i]] = group;
    }
    return fixed;
}
#endif
#endif
