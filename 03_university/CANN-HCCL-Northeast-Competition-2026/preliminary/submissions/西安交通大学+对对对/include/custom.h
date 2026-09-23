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
#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>

#include "binary_stream.h"
#include "common.h"
#include "log.h"

// Selection is based on the complete root input, not per-rank recvCount.
constexpr uint64_t SCATTER_SMALL_INPUT_BYTES = 1ULL * 1024 * 1024;
constexpr uint64_t SCATTER_CHUNK_BYTES = 8ULL * 1024 * 1024;
constexpr uint32_t SCATTER_MAX_WORKERS = 15;
// Logical small-message peer groups, including the group executed on main.
constexpr uint32_t SCATTER_SMALL_WORKERS = 4;
constexpr uint32_t SCATTER_SMALL_GROUP_RANKS = 4;
constexpr uint32_t SCATTER_SMALL_FANOUT = 0;
constexpr uint32_t SCATTER_SMALL_TREE = 1;
// Logical groups stay at four; one TS worker batches the three non-root leaders.
constexpr uint32_t SCATTER_SMALL_TREE_AUX_WORKERS = 1;
// READY/ACK pairs use 0..3. INIT must not share an ACK across invocations.
constexpr uint32_t SCATTER_CHANNEL_NOTIFIES = 5;
constexpr uint32_t SCATTER_PUSH_INIT_NOTIFY = 4;
constexpr uint32_t SCATTER_ROUTE_DIRECT = 0;
constexpr uint32_t SCATTER_ROUTE_HYBRID = 1;
constexpr uint32_t SCATTER_ROUTE_HYBRID_PUSH = 2;
constexpr uint32_t SCATTER_SERVER_RANKS = 8;
constexpr uint32_t SCATTER_RELAY_COUNT = 7;
constexpr uint32_t SCATTER_REMOTE_COUNT = 8;
constexpr uint32_t SCATTER_TAIL_DENOMINATOR = 11;

static_assert(SCATTER_RELAY_COUNT + 1 == SCATTER_SERVER_RANKS, "Every local peer must be a relay");
static_assert(SCATTER_RELAY_COUNT + 1 == SCATTER_REMOTE_COUNT, "One relay has two children");
static_assert(SCATTER_SMALL_WORKERS > 0 && SCATTER_SMALL_WORKERS <= SCATTER_MAX_WORKERS,
    "Small workers must fit the main thread's reserved joins");
static_assert(SCATTER_SMALL_WORKERS * SCATTER_SMALL_GROUP_RANKS == SCATTER_SERVER_RANKS * 2 &&
    SCATTER_SERVER_RANKS % SCATTER_SMALL_GROUP_RANKS == 0,
    "Four small groups must partition the two eight-rank servers");
static_assert(SCATTER_SMALL_TREE_AUX_WORKERS == 1,
    "The small-tree worker must schedule all non-root leaders exactly once");
static_assert(SCATTER_PUSH_INIT_NOTIFY >= 4 && SCATTER_PUSH_INIT_NOTIFY < SCATTER_CHANNEL_NOTIFIES,
    "Push INIT must have its own channel notification after the two READY/ACK pairs");

constexpr uint32_t ScatterSmallRootWorkerCount(uint32_t rankSize, uint32_t myRank, uint32_t root)
{
    return rankSize == SCATTER_SERVER_RANKS + SCATTER_REMOTE_COUNT && myRank == root ?
        SCATTER_SMALL_WORKERS : 0U;
}

constexpr uint32_t ScatterSmallAuxWorkerCount(uint32_t groupCount, uint32_t smallMode = SCATTER_SMALL_FANOUT)
{
    // A zero-group plan retains the original single-thread path.
    return groupCount == 0 ? 0U :
        (smallMode == SCATTER_SMALL_TREE ? SCATTER_SMALL_TREE_AUX_WORKERS : groupCount - 1);
}

inline uint32_t ScatterSmallGroupIndex(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    const auto found = std::find(ranks.begin(), ranks.end(), rank);
    return found == ranks.end() ? INVALID_VALUE_RANKID :
        static_cast<uint32_t>(found - ranks.begin()) / SCATTER_SMALL_GROUP_RANKS;
}

inline uint32_t ScatterSmallGroupLeader(const std::vector<uint32_t> &ranks, uint32_t group, uint32_t root)
{
    // Call only after validating the packed, four-rank group list.
    const uint32_t begin = group * SCATTER_SMALL_GROUP_RANKS;
    for (uint32_t index = begin; index < begin + SCATTER_SMALL_GROUP_RANKS; ++index) {
        if (ranks[index] == root) {
            return root;
        }
    }
    return ranks[begin];
}

// Indices refer to the sorted, validated relay/remote vectors, not rank IDs.
// Relay 0 serves remote 0/1; relay 1..6 serves remote 2..7 respectively.
constexpr uint32_t ScatterRelayChildCount(uint32_t relayIndex)
{
    return relayIndex == 0 ? 2U : 1U;
}

constexpr bool ScatterIsHybridRoute(uint32_t routeMode)
{
    return routeMode == SCATTER_ROUTE_HYBRID || routeMode == SCATTER_ROUTE_HYBRID_PUSH;
}

// PUSH reserves a separate incoming pair even on edges that retain Read.
// The layout is collective; the transfer primitive is chosen per channel.
constexpr uint32_t ScatterChildLane(uint32_t routeMode, uint32_t child)
{
    return child + (routeMode == SCATTER_ROUTE_HYBRID_PUSH ? 1U : 0U);
}

constexpr uint32_t ScatterRelayBufferSlots(uint32_t routeMode, uint32_t relayIndex)
{
    return 2U * ScatterChildLane(routeMode, ScatterRelayChildCount(relayIndex));
}

constexpr uint32_t ScatterRelayRemoteIndex(uint32_t relayIndex, uint32_t child)
{
    return relayIndex == 0 ? child : relayIndex + 1;
}

constexpr uint32_t ScatterRemoteRelayIndex(uint32_t remoteIndex)
{
    return remoteIndex < 2 ? 0U : remoteIndex - 1;
}

constexpr uint32_t ScatterRemoteChildIndex(uint32_t remoteIndex)
{
    return remoteIndex < 2 ? remoteIndex : 0U;
}

inline uint64_t ScatterTailCount(uint64_t count, uint32_t remoteIndex)
{
    const uint64_t weight = remoteIndex < 2 ? 2U : 4U;
    // floor(count * weight / 11), without multiplying the full count.
    return (count / SCATTER_TAIL_DENOMINATOR) * weight +
        ((count % SCATTER_TAIL_DENOMINATOR) * weight) / SCATTER_TAIL_DENOMINATOR;
}

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
    uint32_t channelProtocol = std::numeric_limits<uint32_t>::max();
};

inline bool ScatterUsePush(uint32_t routeMode, const ChannelInfo &channel)
{
    // Only the reviewed 9.1.0 UB path is enabled. Other established channel
    // protocols retain the existing input -> root CCL -> receiver Read path.
    return routeMode == SCATTER_ROUTE_HYBRID_PUSH &&
        channel.channelProtocol == static_cast<uint32_t>(COMM_PROTOCOL_UBC_CTP);
}

// Read the existing BinaryStream layout without an intermediate byte vector.
// Scalar bytes and each vector's size_t count retain the writer's native ABI.
class ScatterContextReader {
public:
    ScatterContextReader(const void *data, uint64_t bytes)
        : cursor_(static_cast<const char *>(data)), remaining_(0)
    {
        if (data != nullptr && bytes <= std::numeric_limits<size_t>::max()) {
            remaining_ = static_cast<size_t>(bytes);
        }
    }

    template <typename T> bool Read(T &value)
    {
        static_assert(std::is_trivially_copyable<T>::value, "Context scalars must be trivially copyable");
        if (sizeof(T) > remaining_) {
            return false;
        }
        // memcpy also supports fields that are unaligned in the byte stream.
        std::memcpy(static_cast<void *>(&value), cursor_, sizeof(T));
        cursor_ += sizeof(T);
        remaining_ -= sizeof(T);
        return true;
    }

    template <typename T> bool Read(std::vector<T> &values)
    {
        static_assert(std::is_trivially_copyable<T>::value, "Context elements must be trivially copyable");
        size_t count = 0;
        if (!Read(count) || count > remaining_ / sizeof(T) || count > values.max_size()) {
            return false;
        }
        // Validate before allocation or multiplication; remaining_ is size_t.
        values.resize(count);
        if (count != 0) {
            const size_t bytes = count * sizeof(T);
            std::memcpy(static_cast<void *>(values.data()), cursor_, bytes);
            cursor_ += bytes;
            remaining_ -= bytes;
        }
        return true;
    }

private:
    const char *cursor_;
    size_t remaining_;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread = 0;
    CommBuffer localBuffer;
    // Main plus up to 15 large-root workers, one small-tree worker, three small
    // fan-out workers, or one hybrid remote-tail worker; only main for other roles.
    std::vector<ThreadHandle> threads;
    std::vector<ChannelInfo> channels;
    uint64_t chunkBytes = 0; // Zero selects a small-input algorithm.
    // Small input: logical peer groups including main; large input: auxiliary workers.
    uint32_t workerCount = 0;

    uint32_t smallMode = SCATTER_SMALL_FANOUT;
    // Two sorted server lists, each split into two groups of four. Membership
    // is immutable per comm; the leader of root's group is computed per call.
    std::vector<uint32_t> smallRanks;

    // Hybrid plans are immutable and root-specific. The shared index helpers
    // assign weighted tails; their lengths are recomputed for each call's count.
    uint32_t routeMode = SCATTER_ROUTE_DIRECT;
    uint32_t routeRoot = INVALID_VALUE_RANKID;
    std::vector<uint32_t> relayRanks;
    std::vector<uint32_t> remoteRanks;

    std::vector<char> Serialize() const
    {
        BinaryStream stream;
        stream << aicpuThread << localBuffer << threads << channels << chunkBytes << workerCount;
        // Large contexts retain their existing wire layout. Small/base contexts
        // carry the mode and topology; all cache tags are versioned with it.
        if (chunkBytes != 0) {
            stream << routeMode << routeRoot << relayRanks << remoteRanks;
        } else {
            stream << smallMode << smallRanks;
        }
        std::vector<char> result;
        stream.Dump(result);
        return result;
    }

    bool DeSerialize(const void *data, uint64_t bytes)
    {
        ScatterContextReader reader(data, bytes);
        if (!reader.Read(aicpuThread) || !reader.Read(localBuffer) || !reader.Read(threads) ||
            !reader.Read(channels) || !reader.Read(chunkBytes) || !reader.Read(workerCount)) {
            return false;
        }
        routeMode = SCATTER_ROUTE_DIRECT;
        routeRoot = INVALID_VALUE_RANKID;
        relayRanks.clear();
        remoteRanks.clear();
        smallMode = SCATTER_SMALL_FANOUT;
        smallRanks.clear();
        if (chunkBytes != 0) {
            if (!reader.Read(routeMode) || !reader.Read(routeRoot) || !reader.Read(relayRanks) ||
                !reader.Read(remoteRanks)) {
                return false;
            }
        } else if (!reader.Read(smallMode) || !reader.Read(smallRanks)) {
            return false;
        }
        // Like the previous reader, allow unused trailing storage bytes.
        return true;
    }
};

#endif // OPS_HCCL_CUSTOM_H
