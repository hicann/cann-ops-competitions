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

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

// V26 retains V24's one ACK plus eight per-destination completion notifies for each of
// up to four rolling buffers on the layer-1 mate channel. Distinct indices
// keep the one-bit notification protocol safe while later stripes run.
constexpr uint32_t CHANNEL_NOTIFY_NUM = 33;
constexpr uint32_t THREAD_START_NOTIFY_IDX = 0;
// Main-thread notify 0 is reserved for Host/Device synchronization.
constexpr uint32_t MAIN_THREAD_FINISH_NOTIFY_BASE = 1;
constexpr uint64_t CCL_SLOT_ALIGNMENT = 128;
constexpr uint64_t SERIAL_REDUCE_THRESHOLD_BYTES = 64 * 1024;
// The 512 KiB / 16-rank performance case produces exactly 32 KiB per rank.
constexpr uint64_t V6_READ_SMALL_OUTPUT_BYTES = 32 * 1024;
// Version 15 aggressive paths: fixed competition shapes only.
constexpr uint32_t V15_RANKS = 16;
constexpr uint32_t V15_SERVER_RANKS = 8;
constexpr uint64_t V15_SMALL_INPUT_BYTES = 512 * 1024;
constexpr uint64_t V15_LARGE_OUTPUT_MIN_BYTES = 25 * 1024 * 1024;
// Version 26 retains V24's direct-input layout and K4/K3 stripe policy.  For every destination the local
// server contribution is copied once into its accumulator; the mate's
// contribution is sent directly from user input with ordinary WriteReduce.
// The fused remote reduction therefore avoids both the remote-server pack and
// the subsequent ReadReduce landing operation.
//
// Each buffer owns eight pair accumulators and seven disjoint incoming leaves.
// Capacity selects two to four active buffers: small functional environments
// retain the proven two-buffer fallback, while the 400 MiB performance
// environment fits three buffers and removes most/all reuse barriers.
// Threads 0..6 drive the seven layer-0 links, thread 7 drives the layer-1 mate
// link, and threads 8..14 initialize the accumulators. Point 6 remains K4 and
// point 7 remains K3. Each data operation plus its single completion Record is
// submitted as a two-descriptor batch. Descriptor-level data tasks, destination
// order and exact completion dependencies are unchanged.
constexpr uint32_t V19_PUBLISH_SLOT_COUNT = 0;
constexpr uint32_t V19_ACCUMULATOR_SLOT_BASE = 0;
constexpr uint32_t V19_INCOMING_SLOT_BASE = V19_ACCUMULATOR_SLOT_BASE + V15_SERVER_RANKS;
constexpr uint32_t V19_INCOMING_SLOT_COUNT = 7;
constexpr uint32_t V19_PACK_SLOT_COUNT = V19_INCOMING_SLOT_BASE;
constexpr uint32_t V19_SLOTS_PER_BUFFER = V19_INCOMING_SLOT_BASE + V19_INCOMING_SLOT_COUNT;
constexpr uint32_t V19_CROSS_THREAD_INDEX = 7;
constexpr uint32_t V19_PACK_THREAD_BASE = 8;
constexpr uint32_t V19_PACK_THREAD_COUNT = 7;
constexpr uint32_t V22_POINT6_STRIPE_COUNT = 4;
constexpr uint32_t V22_POINT7_STRIPE_COUNT = 3;
constexpr uint32_t V22_FALLBACK_BUFFER_COUNT = 2;
constexpr uint64_t V19_POINT6_OUTPUT_BYTES = 32 * 1024 * 1024;
constexpr uint32_t V22_MAX_BUFFER_COUNT = V22_POINT6_STRIPE_COUNT;
constexpr uint32_t V24_CROSS_DATA_NOTIFY_BASE = 1;
constexpr uint32_t V24_CROSS_DATA_NOTIFY_PER_BUFFER = V15_SERVER_RANKS;
// Pack completion is reduced on pack threads 8..14 before one exact token is
// sent to cross thread 7. Per-buffer token ranges are 1..4, 5..8 and 9..12;
// every waiting thread therefore remains below the proven 15-notify limit.
constexpr uint32_t V22_PACK_PAIR_NOTIFY_BASE = 1;
constexpr uint32_t V22_PACK_LEVEL2_NOTIFY_BASE = 5;
constexpr uint32_t V22_PACK_LEVEL3_NOTIFY_BASE = 9;
constexpr uint32_t V22_PACK_ROOT_NOTIFY_BASE = 1;
// Arrival-tree owners stay inside their leaf pairs: 0/1, 2/3 and 4/5.  This
// makes the owner's stream order prove that its own incoming write completed;
// the peer's exact token proves the other leaf.  Buffers 0/2 and 1/3 share an
// owner but use separate token groups.
constexpr uint32_t V22_TREE_ROOT_NOTIFY_BASE = 1;
constexpr uint32_t V22_TREE_ROOT_NOTIFY_STRIDE = 4;
constexpr uint32_t V22_TREE_PAIR23_NOTIFY_BASE = 1;
constexpr uint32_t V22_TREE_PAIR45_NOTIFY_BASE = 1;
constexpr uint32_t V22_TREE_PAIR45_NOTIFY_STRIDE = 2;
// Indices 13 and 14 are unused by the pack and arrival trees.  They close the
// lifetime of fallback buffers 0 and 1 before the same storage is repacked.
constexpr uint32_t V22_BUFFER_REUSE_NOTIFY_BASE = 13;

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    // One entry exists for every peer, in ascending remote-rank order.
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    // Communication memory exported by this peer Channel.
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;          ///< AICPU_TS主线程，始终等于threads[0]
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< 每个远端rank对应一个并行Thread
    std::vector<ChannelInfo> channels; ///< 每个远端rank对应唯一Channel

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
    }
};

#endif // OPS_HCCL_CUSTOM_H
