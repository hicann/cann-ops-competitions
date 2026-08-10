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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "common.h"

constexpr size_t MAX_CUSTOM_THREAD_NUM = 16;
constexpr size_t MAX_CUSTOM_CHANNEL_NUM = 15;

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

template <typename T, size_t Capacity>
class ContextVector {
public:
    bool empty() const { return size() == 0; }
    size_t size() const { return viewData_ == nullptr ? storage_.size() : viewSize_; }
    T *data() { return storage_.data(); }
    const T *data() const { return storage_.data(); }

    T &operator[](size_t index)
    {
        if (viewData_ == nullptr) return storage_[index];
        std::memcpy(&scratch_, viewData_ + index * sizeof(T), sizeof(T));
        return scratch_;
    }

    T operator[](size_t index) const
    {
        if (viewData_ == nullptr) return storage_[index];
        T value;
        std::memcpy(&value, viewData_ + index * sizeof(T), sizeof(T));
        return value;
    }

    void resize(size_t size)
    {
        viewData_ = nullptr;
        viewSize_ = 0;
        storage_.resize(size <= Capacity ? size : 0);
    }

    void reserve(size_t capacity) { storage_.reserve(capacity <= Capacity ? capacity : Capacity); }

    void push_back(const T &value)
    {
        if (storage_.size() < Capacity) storage_.push_back(value);
    }

    void Attach(const char *data, size_t size)
    {
        storage_.clear();
        viewData_ = data;
        viewSize_ = size;
    }

private:
    std::vector<T> storage_;
    const char *viewData_ = nullptr;
    size_t viewSize_ = 0;
    T scratch_{};
};

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

static_assert(std::is_trivially_copyable<ThreadHandle>::value,
    "ThreadHandle must support raw serialization");
static_assert(std::is_trivially_copyable<CommBuffer>::value,
    "CommBuffer must support raw serialization");
static_assert(std::is_trivially_copyable<ChannelInfo>::value,
    "ChannelInfo must support raw serialization");

struct AlgResourceCtx {
    ThreadHandle aicpuThread = 0;
    CommBuffer localBuffer;
    ContextVector<ThreadHandle, MAX_CUSTOM_THREAD_NUM> threads;
    ContextVector<ChannelInfo, MAX_CUSTOM_CHANNEL_NUM> channels;
    ContextVector<void *, MAX_CUSTOM_CHANNEL_NUM> remoteOutputs;

    std::vector<char> Serialize()
    {
        const uint32_t threadNum = static_cast<uint32_t>(threads.size());
        const uint32_t channelNum = static_cast<uint32_t>(channels.size());
        const uint32_t outputNum = static_cast<uint32_t>(remoteOutputs.size());
        const size_t serializedSize = sizeof(aicpuThread) + sizeof(localBuffer) + sizeof(threadNum) +
            sizeof(ThreadHandle) * threadNum + sizeof(channelNum) + sizeof(ChannelInfo) * channelNum +
            sizeof(outputNum) + sizeof(void *) * outputNum;
        std::vector<char> result(serializedSize);
        char *cursor = result.data();
        Write(cursor, aicpuThread);
        Write(cursor, localBuffer);
        Write(cursor, threadNum);
        WriteArray(cursor, threads.data(), threadNum);
        Write(cursor, channelNum);
        WriteArray(cursor, channels.data(), channelNum);
        Write(cursor, outputNum);
        WriteArray(cursor, remoteOutputs.data(), outputNum);
        return result;
    }

    void DeSerialize(const char *data, size_t size)
    {
        if (data == nullptr || size == 0) {
            aicpuThread = 0;
            localBuffer = CommBuffer{};
            threads.Attach(nullptr, 0);
            channels.Attach(nullptr, 0);
            remoteOutputs.Attach(nullptr, 0);
            return;
        }

        const char *cursor = data;
        const char *end = cursor + size;
        uint32_t threadNum = 0;
        uint32_t channelNum = 0;
        uint32_t outputNum = 0;
        const char *threadData = nullptr;
        const char *channelData = nullptr;
        const char *outputData = nullptr;
        if (!Read(cursor, end, aicpuThread) || !Read(cursor, end, localBuffer) ||
            !Read(cursor, end, threadNum) || threadNum == 0 || threadNum > MAX_CUSTOM_THREAD_NUM ||
            !ReadSpan(cursor, end, threadData, sizeof(ThreadHandle), threadNum) ||
            !Read(cursor, end, channelNum) || channelNum > MAX_CUSTOM_CHANNEL_NUM ||
            !ReadSpan(cursor, end, channelData, sizeof(ChannelInfo), channelNum) ||
            !Read(cursor, end, outputNum) || outputNum > MAX_CUSTOM_CHANNEL_NUM ||
            (outputNum != 0 && outputNum != channelNum) ||
            !ReadSpan(cursor, end, outputData, sizeof(void *), outputNum) || cursor != end) {
            aicpuThread = 0;
            localBuffer = CommBuffer{};
            threads.Attach(nullptr, 0);
            channels.Attach(nullptr, 0);
            remoteOutputs.Attach(nullptr, 0);
            return;
        }
        threads.Attach(threadData, threadNum);
        channels.Attach(channelData, channelNum);
        remoteOutputs.Attach(outputData, outputNum);
    }

    void DeSerialize(std::vector<char> &data)
    {
        DeSerialize(data.data(), data.size());
    }

private:
    template <typename T>
    static void Write(char *&cursor, const T &value)
    {
        std::memcpy(cursor, &value, sizeof(T));
        cursor += sizeof(T);
    }

    template <typename T>
    static void WriteArray(char *&cursor, const T *values, size_t count)
    {
        const size_t bytes = sizeof(T) * count;
        if (bytes != 0) {
            std::memcpy(cursor, values, bytes);
            cursor += bytes;
        }
    }

    template <typename T>
    static bool Read(const char *&cursor, const char *end, T &value)
    {
        if (static_cast<size_t>(end - cursor) < sizeof(T)) return false;
        std::memcpy(&value, cursor, sizeof(T));
        cursor += sizeof(T);
        return true;
    }

    static bool ReadSpan(
        const char *&cursor, const char *end, const char *&values, size_t elementSize, size_t count)
    {
        const size_t bytes = elementSize * count;
        if (static_cast<size_t>(end - cursor) < bytes) return false;
        values = bytes == 0 ? nullptr : cursor;
        cursor += bytes;
        return true;
    }
};

#endif // OPS_HCCL_CUSTOM_H
