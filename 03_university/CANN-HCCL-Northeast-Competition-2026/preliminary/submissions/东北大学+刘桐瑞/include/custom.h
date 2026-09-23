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

#define SCATTER_CHK(call) \
    do { \
        const int32_t scatterRet = (call); \
        if (UNLIKELY(scatterRet != HCCL_SUCCESS)) { \
            return static_cast<HcclResult>(scatterRet); \
        } \
    } while (0)

inline uint32_t ScatterDataTypeSize(HcclDataType dataType)
{
    return (dataType == HCCL_DATA_TYPE_FP32) ? static_cast<uint32_t>(sizeof(float)) : 0U;
}

constexpr uint64_t SCATTER_RELAY_DISABLE = 0;
constexpr uint64_t SCATTER_RELAY_NUM = 5;
constexpr uint64_t SCATTER_RELAY_DEN = 14;

#define SCATTER_MERGE_TRANSFER 0

constexpr uint32_t SCATTER_CLOS_THREAD_NUM = 5;

#define SCATTER_USE_BATCH_MODE 0

#define SCATTER_USE_START_HANDSHAKE 1

#define SCATTER_USE_BREADTH_FIRST 1

constexpr uint32_t SCATTER_MAX_PLAN_CHANNELS = 32;

#define SCATTER_LINK_ROTATE 0

#define SCATTER_USE_FUSED_WRITE 1

#define SCATTER_RELAY_USE_READ 1

#define SCATTER_USE_TASK_CACHE 0

#define SCATTER_USE_HOST_ARGS_LAUNCH 0

#define SCATTER_USE_READ_BROADCAST 1

constexpr uint64_t SCATTER_READ_BCAST_MAX_TOTAL = 4ULL * 1024 * 1024;

#define SCATTER_CACHE_HOST_THREADS 1

#define SCATTER_CACHE_RES_CTX 1
constexpr uint32_t SCATTER_RES_CTX_KEY_BYTES = 32;

inline uint32_t ScatterBuildCacheTag(const char *base, uint64_t count, char *out, uint32_t outLen)
{
    constexpr uint32_t hexDigits = 16;
    constexpr uint32_t reserved = hexDigits + 2;
    uint32_t pos = 0;
    while (base[pos] != '\0' && (pos + reserved) < outLen) {
        out[pos] = base[pos];
        pos++;
    }
    out[pos++] = '#';
    for (int32_t shift = 60; shift >= 0; shift -= 4) {
        const uint32_t nibble = static_cast<uint32_t>((count >> static_cast<uint32_t>(shift)) & 0xFULL);
        out[pos++] = static_cast<char>((nibble < 10) ? ('0' + nibble) : ('a' + nibble - 10));
    }
    out[pos] = '\0';
    return pos;
}

constexpr uint32_t SCATTER_SLOT_NUM = 2;

constexpr uint32_t SCATTER_STREAM_NUM = 2;
constexpr uint32_t SCATTER_NOTIFY_PER_STREAM = SCATTER_SLOT_NUM * 2;
constexpr uint32_t SCATTER_CHANNEL_NOTIFY_NUM = SCATTER_NOTIFY_PER_STREAM * SCATTER_STREAM_NUM;

constexpr uint32_t SCATTER_STREAM_MAIN = 0;
constexpr uint32_t SCATTER_STREAM_RELAY = 1;

inline uint32_t ScatterAckNotifyIdx(uint32_t stream, uint64_t slot)
{
    return stream * SCATTER_NOTIFY_PER_STREAM + static_cast<uint32_t>(slot % SCATTER_SLOT_NUM);
}

inline uint32_t ScatterDataNotifyIdx(uint32_t stream, uint64_t slot)
{
    return stream * SCATTER_NOTIFY_PER_STREAM + SCATTER_SLOT_NUM + static_cast<uint32_t>(slot % SCATTER_SLOT_NUM);
}

constexpr uint32_t SCATTER_SUB_THREAD_START_NOTIFY_IDX = 0;

constexpr uint32_t SCATTER_RELAY_GO_NOTIFY_IDX = 1;

constexpr uint64_t SCATTER_SLOT_ALIGN = 512;
constexpr uint64_t SCATTER_SLOT_MAX_BYTES = 16ULL * 1024 * 1024;

constexpr uint64_t SCATTER_TAIL_BYTES = 1ULL * 1024 * 1024;

struct ScatterChunkPlan {
    uint64_t chunks = 0;
    uint64_t base = 0;
    uint64_t last = 0;
};

inline ScatterChunkPlan ScatterPlanChunks(uint64_t size, uint64_t slotBytes)
{
    ScatterChunkPlan plan;
    if (size == 0 || slotBytes == 0) {
        return plan;
    }
    if (size > SCATTER_TAIL_BYTES) {
        const uint64_t body = size - SCATTER_TAIL_BYTES;
        const uint64_t n = (body + slotBytes - 1) / slotBytes;
        const uint64_t base = (body + n - 1) / n / SCATTER_SLOT_ALIGN * SCATTER_SLOT_ALIGN;
        const uint64_t sent = base * n;

        if (base > 0 && sent < size && (size - sent) <= slotBytes) {
            plan.chunks = n + 1;
            plan.base = base;
            plan.last = size - sent;
            return plan;
        }
    }

    plan.chunks = (size + slotBytes - 1) / slotBytes;
    plan.base = slotBytes;
    plan.last = size - slotBytes * (plan.chunks - 1);
    return plan;
}

inline uint64_t ScatterChunkOffset(const ScatterChunkPlan &plan, uint64_t idx)
{
    return plan.base * idx;
}

inline uint64_t ScatterChunkBytes(const ScatterChunkPlan &plan, uint64_t idx)
{
    return (idx + 1 == plan.chunks) ? plan.last : plan.base;
}

constexpr uint32_t SCATTER_REGION_MAIN = 0;
constexpr uint32_t SCATTER_REGION_RELAY = 1;
constexpr uint32_t SCATTER_REGION_NUM = 2;

inline uint64_t ScatterSlotOffset(uint32_t region, uint64_t slotIdx, uint64_t slotBytes)
{
    return (static_cast<uint64_t>(region) * SCATTER_SLOT_NUM + slotIdx % SCATTER_SLOT_NUM) * slotBytes;
}

inline uint64_t ScatterRegionBytes(uint64_t slotBytes)
{
    return static_cast<uint64_t>(SCATTER_SLOT_NUM) * slotBytes;
}

inline uint64_t ScatterLandingBytes(uint64_t slotBytes)
{
    return static_cast<uint64_t>(SCATTER_REGION_NUM) * ScatterRegionBytes(slotBytes);
}

inline bool ScatterUseReadBroadcast(uint64_t shareSize, uint32_t rankSize, uint64_t slotBytes)
{
#if SCATTER_USE_READ_BROADCAST
    if (shareSize == 0 || rankSize <= 1) {
        return false;
    }

    const uint64_t total = shareSize * static_cast<uint64_t>(rankSize);
    return total <= SCATTER_READ_BCAST_MAX_TOTAL && total <= ScatterLandingBytes(slotBytes);
#else
    (void)shareSize;
    (void)rankSize;
    (void)slotBytes;
    return false;
#endif
}

inline bool ScatterUseMerged(uint64_t shareSize, uint64_t slotBytes)
{
#if SCATTER_MERGE_TRANSFER

    return shareSize != 0 && shareSize <= ScatterRegionBytes(slotBytes);
#else
    (void)shareSize;
    (void)slotBytes;
    return false;
#endif
}

constexpr uint64_t SCATTER_RELAY_MIN_SHARE = 1ULL * 1024 * 1024;

inline uint64_t ScatterRelayBytes(uint64_t shareSize)
{
    if (SCATTER_RELAY_DISABLE != 0 || shareSize < SCATTER_RELAY_MIN_SHARE) {
        return 0;
    }
    uint64_t relay = shareSize / SCATTER_RELAY_DEN * SCATTER_RELAY_NUM;
    relay = relay / SCATTER_SLOT_ALIGN * SCATTER_SLOT_ALIGN;
    return (relay > shareSize) ? shareSize : relay;
}

inline uint64_t ScatterDirectBytes(uint64_t shareSize)
{
    return shareSize - ScatterRelayBytes(shareSize);
}

constexpr uint32_t SCATTER_ROLE_NONE = 0;
constexpr uint32_t SCATTER_ROLE_FORWARD = 1;
constexpr uint32_t SCATTER_ROLE_RECV_RELAY = 2;

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;

    uint32_t sendFullShare = 1;
    uint32_t relayForRank = INVALID_VALUE_RANKID;

    uint32_t threadIdx = 0;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;
    CommBuffer localBuffer;
    uint64_t slotBytes = 0;
    uint32_t relayRole = SCATTER_ROLE_NONE;
    uint32_t rootChannelIdx = 0;
    uint32_t relayChannelIdx = 0;
    uint32_t recvFromRootFull = 1;
    std::vector<ThreadHandle> threads;
    std::vector<ChannelInfo> channels;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << slotBytes;
        binaryStream << relayRole;
        binaryStream << rootChannelIdx;
        binaryStream << relayChannelIdx;
        binaryStream << recvFromRootFull;
        binaryStream << threads;
        binaryStream << channels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> slotBytes;
        binaryStream >> relayRole;
        binaryStream >> rootChannelIdx;
        binaryStream >> relayChannelIdx;
        binaryStream >> recvFromRootFull;
        binaryStream >> threads;
        binaryStream >> channels;
    }
};

#endif
