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
#include <cstring>
#include <type_traits>
#include <utility>

// Channel notifications are separate from the thread notifications used by the launcher.
constexpr uint32_t SCATTER_CHANNEL_NOTIFY_NUM = 9;
constexpr uint32_t SCATTER_NOTIFY_CONSUMED = 8;
constexpr uint64_t SCATTER_SLICE_ALIGN = 128;
constexpr uint64_t SCATTER_MAX_SLICE = 64 * 1024 * 1024;
constexpr uint64_t SCATTER_LARGE_MAX_SLICE = 96 * 1024 * 1024;
constexpr uint64_t SCATTER_PACK_LIMIT = 16 * 1024 * 1024;
constexpr uint64_t SCATTER_RELAY_THRESHOLD = 8 * 1024 * 1024;

inline uint64_t ScatterSlotSize(uint64_t bufferSize, uint32_t rankSize)
{
    return rankSize > 1 ? bufferSize / (rankSize - 1) / SCATTER_SLICE_ALIGN * SCATTER_SLICE_ALIGN : 0;
}

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;          ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< AICPU_TS通信引擎上的thread资源
    std::vector<ChannelInfo> channels; ///< AICPU_TS通信引擎上的channel资源

    std::vector<uint32_t> localRanks; // Lowest layer of the README's 2x8 topology; empty means direct fallback.

    // 序列化
    std::vector<char> Serialize()
    {
        std::vector<char> result;
        Append(result, &aicpuThread, sizeof(aicpuThread));
        Append(result, &localBuffer, sizeof(localBuffer));
        AppendVector(result, threads);
        AppendVector(result, channels);
        AppendVector(result, localRanks);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        // Same native wire layout as BinaryStream, but one copy per vector instead
        // of one stream read per element. Validate lengths before allocating.
        const char *cursor = data.data();
        size_t remaining = data.size();
        AlgResourceCtx decoded{};
        if (!Read(cursor, remaining, &decoded.aicpuThread, sizeof(decoded.aicpuThread))
            || !Read(cursor, remaining, &decoded.localBuffer, sizeof(decoded.localBuffer))
            || !ReadVector(cursor, remaining, decoded.threads) || !ReadVector(cursor, remaining, decoded.channels)
            || !ReadVector(cursor, remaining, decoded.localRanks)) {
            *this = AlgResourceCtx{};
            return;
        }
        *this = std::move(decoded);
    }

private:
    static void Append(std::vector<char> &out, const void *src, size_t length)
    {
        if (length != 0) {
            const auto *bytes = static_cast<const char *>(src);
            out.insert(out.end(), bytes, bytes + length);
        }
    }

    template <typename T> static void AppendVector(std::vector<char> &out, const std::vector<T> &items)
    {
        static_assert(std::is_trivially_copyable<T>::value, "Resource vectors require a native POD layout");
        const size_t count = items.size();
        Append(out, &count, sizeof(count));
        Append(out, items.data(), count * sizeof(T));
    }

    static bool Read(const char *&cursor, size_t &remaining, void *dst, size_t length)
    {
        if (length > remaining)
            return false;
        if (length != 0) {
            std::memcpy(dst, cursor, length);
            cursor += length;
            remaining -= length;
        }
        return true;
    }

    template <typename T> static bool ReadVector(const char *&cursor, size_t &remaining, std::vector<T> &items)
    {
        static_assert(std::is_trivially_copyable<T>::value, "Resource vectors require a native POD layout");
        size_t count = 0;
        if (!Read(cursor, remaining, &count, sizeof(count)) || count > remaining / sizeof(T))
            return false;
        items.resize(count);
        return Read(cursor, remaining, items.data(), count * sizeof(T));
    }
};

#endif // OPS_HCCL_CUSTOM_H
