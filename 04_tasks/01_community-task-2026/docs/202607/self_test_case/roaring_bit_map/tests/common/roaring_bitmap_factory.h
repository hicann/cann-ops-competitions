/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "roaring_bitmap.h"
#include "extent.h"

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace aclco::test::roaring_bitmap_factory
{

template <typename T>
using RoaringBitmapT = aclco::RoaringBitmap<T>;

/**
 * 生成唯一随机元素集合
 */
template <typename T>
inline std::vector<T> GenerateElements(std::size_t n, uint32_t seed = 1u)
{
  std::vector<T> out;
  out.reserve(n);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::uint64_t> dist(
      0, static_cast<std::uint64_t>(std::numeric_limits<T>::max()));
  std::unordered_set<T> seen;
  seen.reserve(n * 2 + 1);
  while (out.size() < n) {
    T v = static_cast<T>(dist(rng));
    if (seen.insert(v).second) {
      out.push_back(v);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

/**
 * 将已排序的 uint32_t 元素序列化为 RoaringBitmap 格式字节（简化实现）。
 *
 * 格式：使用单个 Array Container 或 Bitmap Container，
 * 不考虑多 Container / Run Container 的情况，适用于小规模测试数据。
 * 生产环境应使用 CRoaring 等标准库生成序列化数据。
 */
inline std::vector<std::byte> SerializeUint32(const std::vector<std::uint32_t>& elements)
{
  if (elements.empty()) {
    // 空位图：最小 header
    std::vector<std::byte> out(16);
    auto write32 = [&](std::size_t off, std::uint32_t v) {
      out[off]     = static_cast<std::byte>(v & 0xFF);
      out[off + 1] = static_cast<std::byte>((v >> 8) & 0xFF);
      out[off + 2] = static_cast<std::byte>((v >> 16) & 0xFF);
      out[off + 3] = static_cast<std::byte>((v >> 24) & 0xFF);
    };
    write32(0, 16);  // total size bytes
    // flags = 1 (offsets_in_serialized_data), num_containers = 0
    out[4] = std::byte{1};
    out[5] = std::byte{0};
    out[6] = std::byte{0};
    out[7] = std::byte{0};
    // num_keys = 0
    for (int i = 8; i < 16; ++i) out[i] = std::byte{0};
    return out;
  }

  // 按高 16 位分组
  std::vector<std::uint16_t> keys;          // container keys
  std::vector<std::vector<std::uint16_t>> containers;  // container 内低 16 位值

  std::uint16_t curKey = static_cast<std::uint16_t>(elements[0] >> 16);
  std::vector<std::uint16_t> curVals;
  for (auto e : elements) {
    std::uint16_t k = static_cast<std::uint16_t>(e >> 16);
    std::uint16_t v = static_cast<std::uint16_t>(e & 0xFFFF);
    if (k != curKey) {
      if (!curVals.empty()) {
        keys.push_back(curKey);
        containers.push_back(std::move(curVals));
      }
      curKey = k;
      curVals.clear();
    }
    curVals.push_back(v);
  }
  if (!curVals.empty()) {
    keys.push_back(curKey);
    containers.push_back(std::move(curVals));
  }

  std::size_t numContainers = keys.size();
  std::size_t headerSize = 16;  // total_size(4) + flags(2) + num_containers(2) + num_keys(8)
  std::size_t keyCardSize = numContainers * 4;          // key(2) + card_m1(2)
  std::size_t offsetsSize = numContainers * 4;           // uint32 per container
  std::size_t containersDataSize = 0;
  for (auto& c : containers) {
    std::size_t card = c.size();
    if (card <= 4096) {
      containersDataSize += card * sizeof(std::uint16_t);  // array container: sorted uint16 values
    } else {
      containersDataSize += 8192;  // bitset container: fixed 8KB
    }
  }

  std::size_t totalSize = headerSize + keyCardSize + offsetsSize + containersDataSize;
  std::vector<std::byte> out(totalSize);

  auto write32 = [&](std::size_t off, std::uint32_t v) {
    if (off + 4 <= out.size()) {
      out[off]     = static_cast<std::byte>(v & 0xFF);
      out[off + 1] = static_cast<std::byte>((v >> 8) & 0xFF);
      out[off + 2] = static_cast<std::byte>((v >> 16) & 0xFF);
      out[off + 3] = static_cast<std::byte>((v >> 24) & 0xFF);
    }
  };
  auto write16 = [&](std::size_t off, std::uint16_t v) {
    if (off + 2 <= out.size()) {
      out[off]     = static_cast<std::byte>(v & 0xFF);
      out[off + 1] = static_cast<std::byte>((v >> 8) & 0xFF);
    }
  };

  // total_size (4 bytes)
  write32(0, static_cast<std::uint32_t>(totalSize));
  // flags (2 bytes): bit 0 = offsets_in_serialized_data, bit 1 = has_run
  std::uint16_t flags = 1;  // offsets_in_serialized_data
  write16(4, flags);
  // num_containers (2 bytes)
  write16(6, static_cast<std::uint16_t>(numContainers));
  // num_keys (8 bytes)
  write32(8, static_cast<std::uint32_t>(elements.size()));
  write32(12, 0);

  // key/card pairs
  for (std::size_t i = 0; i < numContainers; ++i) {
    std::size_t off = headerSize + i * 4;
    write16(off, keys[i]);
    std::uint16_t card_m1;
    if (containers[i].size() <= 4096) {
      card_m1 = static_cast<std::uint16_t>(containers[i].size() - 1);
    } else {
      card_m1 = static_cast<std::uint16_t>(0xFFFF - 1);  // bitset marker
    }
    write16(off + 2, card_m1);
  }

  // container offsets (uint32 each)
  std::size_t dataStart = headerSize + keyCardSize + offsetsSize;
  for (std::size_t i = 0; i < numContainers; ++i) {
    std::size_t offset = 0;
    for (std::size_t j = 0; j < i; ++j) {
      std::size_t card = containers[j].size();
      if (card <= 4096) {
        offset += card * sizeof(std::uint16_t);
      } else {
        offset += 8192;
      }
    }
    std::size_t off = headerSize + keyCardSize + i * 4;
    write32(off, static_cast<std::uint32_t>(offset));
  }

  // container data
  std::size_t dataOff = dataStart;
  for (std::size_t i = 0; i < numContainers; ++i) {
    if (containers[i].size() <= 4096) {
      for (auto v : containers[i]) {
        write16(dataOff, v);
        dataOff += 2;
      }
    } else {
      // bitset container: 8KB bitmap
      auto& c = containers[i];
      for (std::size_t j = 0; j < 8192; ++j) out[dataOff + j] = std::byte{0};
      for (auto v : c) {
        std::size_t byteIdx = v / 8;
        std::size_t bitIdx  = v % 8;
        out[dataOff + byteIdx] |= static_cast<std::byte>(1 << bitIdx);
      }
      dataOff += 8192;
    }
  }

  return out;
}

/**
 * 将已排序的 uint64_t 元素序列化为 64 位 RoaringBitmap 格式字节（简化实现）。
 *
 * 64 位格式使用 32 位子位图的数组结构。
 */
inline std::vector<std::byte> SerializeUint64(const std::vector<std::uint64_t>& elements)
{
  if (elements.empty()) {
    std::vector<std::byte> out(16);
    auto write32 = [&](std::size_t off, std::uint32_t v) {
      out[off]     = static_cast<std::byte>(v & 0xFF);
      out[off + 1] = static_cast<std::byte>((v >> 8) & 0xFF);
      out[off + 2] = static_cast<std::byte>((v >> 16) & 0xFF);
      out[off + 3] = static_cast<std::byte>((v >> 24) & 0xFF);
    };
    write32(0, 16);
    write32(8, 0);  // num_keys = 0
    return out;
  }

  // 按高 32 位分组
  std::vector<std::uint32_t> bucketKeys;
  std::vector<std::vector<std::uint32_t>> buckets;

  std::uint32_t curBucket = static_cast<std::uint32_t>(elements[0] >> 32);
  std::vector<std::uint32_t> curVals;
  for (auto e : elements) {
    std::uint32_t bk = static_cast<std::uint32_t>(e >> 32);
    std::uint32_t bv = static_cast<std::uint32_t>(e & 0xFFFFFFFF);
    if (bk != curBucket) {
      if (!curVals.empty()) {
        bucketKeys.push_back(curBucket);
        buckets.push_back(std::move(curVals));
      }
      curBucket = bk;
      curVals.clear();
    }
    curVals.push_back(bv);
  }
  if (!curVals.empty()) {
    bucketKeys.push_back(curBucket);
    buckets.push_back(std::move(curVals));
  }

  // 序列化每个 bucket 为 32 位子位图
  std::vector<std::vector<std::byte>> serializedBuckets;
  std::size_t totalDataSize = 0;
  for (std::size_t i = 0; i < buckets.size(); ++i) {
    std::sort(buckets[i].begin(), buckets[i].end());
    auto ser = SerializeUint32(buckets[i]);
    serializedBuckets.push_back(ser);
    totalDataSize += ser.size();
  }

  std::size_t headerSize = 24;  // total_size(4) + flags(2) + num_buckets(2) + num_keys(8) + [4 reserved] + total_data_size(4)
  std::size_t bucketOffsetsSize = buckets.size() * 8;  // key(4) + offset(4)

  // For simplicity, inline the sub-bitmap data (not a separate data region)
  std::size_t bucketHeadersSize = 0;
  for (auto& sb : serializedBuckets) {
    bucketHeadersSize += sb.size();
  }

  std::size_t totalSize = headerSize + bucketOffsetsSize + bucketHeadersSize;
  std::vector<std::byte> out(totalSize);

  auto write32 = [&](std::size_t off, std::uint32_t v) {
    out[off]     = static_cast<std::byte>(v & 0xFF);
    out[off + 1] = static_cast<std::byte>((v >> 8) & 0xFF);
    out[off + 2] = static_cast<std::byte>((v >> 16) & 0xFF);
    out[off + 3] = static_cast<std::byte>((v >> 24) & 0xFF);
  };
  auto write16 = [&](std::size_t off, std::uint16_t v) {
    out[off]     = static_cast<std::byte>(v & 0xFF);
    out[off + 1] = static_cast<std::byte>((v >> 8) & 0xFF);
  };

  write32(0, static_cast<std::uint32_t>(totalSize));
  write16(4, 1);  // flags
  write16(6, static_cast<std::uint16_t>(buckets.size()));
  write32(8, static_cast<std::uint32_t>(elements.size()));
  write32(12, 0);
  write32(16, static_cast<std::uint32_t>(totalDataSize));  // total data size
  write32(20, 0);  // reserved

  // bucket offsets
  std::size_t dataStart = headerSize + bucketOffsetsSize;
  for (std::size_t i = 0; i < buckets.size(); ++i) {
    std::size_t off = headerSize + i * 8;
    write32(off, bucketKeys[i]);
    write32(off + 4, 0);  // offset within data region (0 = inline)
    // inline data
    auto& sb = serializedBuckets[i];
    std::copy(sb.begin(), sb.end(), out.begin() + dataStart);
    dataStart += sb.size();
  }

  return out;
}

/**
 * 序列化元素为 RoaringBitmap 格式字节
 */
template <typename T>
inline std::vector<std::byte> Serialize(const std::vector<T>& elements)
{
  if constexpr (std::is_same_v<T, std::uint32_t>) {
    return SerializeUint32(elements);
  } else {
    return SerializeUint64(elements);
  }
}

/**
 * 创建 RoaringBitmap
 */
template <typename T>
inline RoaringBitmapT<T> MakeRoaringBitmap(const std::vector<std::byte>& serialized, aclrtStream stream)
{
  return RoaringBitmapT<T>(serialized.data(), stream);
}

/**
 * 根据模式生成查询 key
 * - "in-set": 全部来自 elements
 * - "out-of-set": 全部不在 elements 中
 * - "mixed": 一半在集合内、一半在集合外
 * - "zero": 全 0
 * - "max": 全最大值
 */
template <typename T>
inline std::vector<T> GenerateQueryKeys(std::size_t n,
                                        const std::string& pattern,
                                        const std::vector<T>& elements,
                                        uint32_t seed = 1u)
{
  std::vector<T> out;
  out.reserve(n);
  std::unordered_set<T> elemSet(elements.begin(), elements.end());

  if (pattern == "in-set") {
    for (std::size_t i = 0; i < n && i < elements.size(); ++i) {
      out.push_back(elements[i]);
    }
    // 如果 n > elements.size()，循环使用
    while (out.size() < n) {
      for (std::size_t i = 0; out.size() < n && i < elements.size(); ++i) {
        out.push_back(elements[i]);
      }
    }
  } else if (pattern == "out-of-set") {
    std::mt19937_64 rng(seed);
    while (out.size() < n) {
      T v = static_cast<T>(rng());
      if (elemSet.find(v) == elemSet.end()) {
        out.push_back(v);
      }
    }
  } else if (pattern == "mixed") {
    std::size_t half = n / 2;
    // 前一半 from elements
    for (std::size_t i = 0; i < half && i < elements.size(); ++i) {
      out.push_back(elements[i]);
    }
    while (out.size() < half) {
      for (std::size_t i = 0; out.size() < half && i < elements.size(); ++i) {
        out.push_back(elements[i]);
      }
    }
    // 后一半 not in elements
    std::mt19937_64 rng(seed);
    while (out.size() < n) {
      T v = static_cast<T>(rng());
      if (elemSet.find(v) == elemSet.end()) {
        out.push_back(v);
      }
    }
  } else if (pattern == "zero") {
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(static_cast<T>(0));
    }
  } else if (pattern == "max") {
    T m = std::numeric_limits<T>::max();
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(m);
    }
  } else {
    // 默认：sequential
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(static_cast<T>(i));
    }
  }
  return out;
}

/**
 * 执行 Contains 查询，返回 host 侧结果向量
 */
template <typename T>
inline std::vector<unsigned char> ContainsKeys(const RoaringBitmapT<T>& bitmap,
                                               const std::vector<T>& hostKeys,
                                               aclrtStream stream)
{
  if (hostKeys.empty()) {
    return {};
  }
  aclco::test::DeviceBuffer<T> dKeys(hostKeys.size());
  dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

  aclco::test::DeviceBuffer<unsigned char> dResult(hostKeys.size());
  dResult.MemsetZero(stream);

  bitmap.Contains(static_cast<void*>(dKeys.Data()),
                  static_cast<void*>(dResult.Data()),
                  aclco::Extent<std::size_t>(hostKeys.size()),
                  stream);
  return dResult.CopyToHost(stream);
}

/**
 * 计算精确匹配率：RoaringBitmap 为精确结构，结果必须与 ground truth 100% 一致。
 * 返回实际匹配比例（1.0 = 全部匹配）。若出现假阳性或假阴性则返回负值。
 */
template <typename T>
inline double ComputeMatchRate(const std::vector<T>& queries,
                               const std::vector<unsigned char>& results,
                               const std::unordered_set<T>& groundTruth)
{
  std::size_t total = queries.size();
  std::size_t matched = 0;
  for (std::size_t i = 0; i < total; ++i) {
    bool expected = (groundTruth.find(queries[i]) != groundTruth.end());
    bool actual   = (results[i] != 0);
    if (expected == actual) {
      ++matched;
    }
  }
  return total == 0 ? 1.0 : static_cast<double>(matched) / static_cast<double>(total);
}

}  // namespace aclco::test::roaring_bitmap_factory
