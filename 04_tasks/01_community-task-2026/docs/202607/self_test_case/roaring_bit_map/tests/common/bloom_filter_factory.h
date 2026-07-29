/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "bloom_filter.h"
#include "extent.h"
#include "hash_functions.h"

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace aclco::test::bloom_filter_factory
{

/**
 * 默认指纹策略：xxhash_64 + uint32_t word + WordsPerBlock words/block（与 cuCollections 上游默认值一致）
 */
template <typename Key, std::uint32_t WordsPerBlock = 8>
using DefaultPolicy = aclco::DefaultFilterPolicy<aclco::xxhash_64<Key>, uint32_t, WordsPerBlock>;

template <typename Key, std::uint32_t WordsPerBlock = 8>
using BloomFilterT = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, DefaultPolicy<Key, WordsPerBlock>>;

/**
 * 根据 key 数量推荐 numBlocks，使 FPR 保持在合理范围（约 1%）。
 * 总位数 = WordsPerBlock * numBlocks * sizeof(Word) * 8 = 8 * numBlocks * 32 = 256 * numBlocks
 * 为使 FPR ~ 1%，需要 ~9.6 bits/key，故 numBlocks ≈ keyNum / 26
 */
inline std::size_t RecommendNumBlocks(std::size_t keyNum)
{
  if (keyNum <= 100) {
    return 10;
  }
  if (keyNum <= 10000) {
    return std::max<std::size_t>(100, keyNum / 26);
  }
  return std::max<std::size_t>(1000, keyNum / 26);
}

/**
 * 创建 BloomFilter
 */
template <typename Key, std::uint32_t WordsPerBlock = 8>
inline BloomFilterT<Key, WordsPerBlock> MakeBloomFilter(std::size_t numBlocks, aclrtStream stream)
{
  return BloomFilterT<Key, WordsPerBlock>(aclco::Extent<std::size_t>(numBlocks), DefaultPolicy<Key, WordsPerBlock>(), stream);
}

/**
 * 根据模式生成测试 key
 * @param pattern: "sequential" / "random" / "duplicates" / "zero" / "sparse" / "min" / "max"
 */
template <typename Key>
inline std::vector<Key> GenerateKeys(std::size_t n, const std::string& pattern, uint32_t seed = 1u)
{
  std::vector<Key> out;
  out.reserve(n);
  std::mt19937 rng(seed);

  if (pattern == "sequential") {
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(static_cast<Key>(i));
    }
  } else if (pattern == "random") {
    if constexpr (std::is_integral_v<Key>) {
      std::uniform_int_distribution<std::int64_t> dist(
        0, static_cast<std::int64_t>(std::min<std::uint64_t>(
             std::numeric_limits<Key>::max(), static_cast<std::uint64_t>(1) << 31)));
      std::unordered_set<Key> seen;
      seen.reserve(n * 2 + 1);
      while (out.size() < n) {
        Key k = static_cast<Key>(dist(rng));
        if (seen.insert(k).second) {
          out.push_back(k);
        }
      }
    } else {
      std::uniform_real_distribution<float> dist(-1.0e6f, 1.0e6f);
      for (std::size_t i = 0; i < n; ++i) {
        out.push_back(static_cast<Key>(dist(rng)));
      }
    }
  } else if (pattern == "duplicates") {
    Key v = static_cast<Key>(42);
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(v);
    }
  } else if (pattern == "zero") {
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(static_cast<Key>(0));
    }
  } else if (pattern == "sparse") {
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(static_cast<Key>(static_cast<std::uint64_t>(i) * 1000ULL));
    }
  } else if (pattern == "min") {
    Key v{};
    if constexpr (std::is_integral_v<Key>) {
      v = std::numeric_limits<Key>::min();
    } else {
      v = std::numeric_limits<Key>::lowest();
    }
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(v);
    }
  } else if (pattern == "max") {
    Key v{};
    if constexpr (std::is_integral_v<Key>) {
      v = std::numeric_limits<Key>::max();
    } else {
      v = std::numeric_limits<Key>::max();
    }
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(v);
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(static_cast<Key>(i));
    }
  }
  return out;
}

/**
 * 生成不在 addedKeys 中的 absent key（用于 FPR 测试）
 *
 * 采用从 0 开始递增候选、跳过 added set 的方式：
 * - 整型：避免 maxKey+1 在 maxKey 为类型最大值时溢出回绕导致 absent key 与 added 冲突
 * - float：从 0 附近递增，float 在该范围精度充足（ULP=1），不会产生重复位模式
 */
template <typename Key>
inline std::vector<Key> GenerateAbsentKeys(std::size_t n, const std::vector<Key>& addedKeys)
{
  std::unordered_set<Key> added(addedKeys.begin(), addedKeys.end());
  std::vector<Key> out;
  out.reserve(n);

  if constexpr (std::is_integral_v<Key>) {
    for (std::uint64_t candidate = 0; out.size() < n; ++candidate) {
      Key k = static_cast<Key>(candidate);
      if (added.find(k) == added.end()) {
        out.push_back(k);
      }
    }
  } else {
    for (std::uint64_t i = 0; out.size() < n; ++i) {
      Key k = static_cast<Key>(static_cast<float>(i));
      if (added.find(k) == added.end()) {
        out.push_back(k);
      }
    }
  }
  return out;
}

/**
 * 计算 TPR（True Positive Rate）：已添加 key 中查询命中的比例
 * BloomFilter 无假阴性，TPR 必须为 1.0
 */
template <typename Key>
inline double ComputeTPR(const std::vector<Key>& queries,
                         const std::vector<unsigned char>& results,
                         const std::unordered_set<Key>& addedSet)
{
  std::size_t added = 0;
  std::size_t hits = 0;
  for (std::size_t i = 0; i < queries.size(); ++i) {
    if (addedSet.find(queries[i]) != addedSet.end()) {
      ++added;
      if (results[i] != 0) {
        ++hits;
      }
    }
  }
  if (added == 0) {
    return 1.0;
  }
  return static_cast<double>(hits) / static_cast<double>(added);
}

/**
 * 计算 FPR（False Positive Rate）：未添加 key 中误报的比例
 * 仅作为参考指标，不作硬性失败条件
 */
template <typename Key>
inline double ComputeFPR(const std::vector<Key>& queries,
                         const std::vector<unsigned char>& results,
                         const std::unordered_set<Key>& addedSet)
{
  std::size_t absent = 0;
  std::size_t falsePositives = 0;
  for (std::size_t i = 0; i < queries.size(); ++i) {
    if (addedSet.find(queries[i]) == addedSet.end()) {
      ++absent;
      if (results[i] != 0) {
        ++falsePositives;
      }
    }
  }
  if (absent == 0) {
    return 0.0;
  }
  return static_cast<double>(falsePositives) / static_cast<double>(absent);
}

/**
 * 执行 Add 操作并同步
 */
template <typename Key>
inline void AddKeys(BloomFilterT<Key>& filter, const std::vector<Key>& hostKeys, aclrtStream stream)
{
  if (hostKeys.empty()) {
    return;
  }
  aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
  dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);
  filter.Add(static_cast<void*>(dKeys.Data()), aclco::Extent<std::size_t>(hostKeys.size()), stream);
  aclco::test::Sync(stream);
}

/**
 * 执行 Contains 查询，返回 host 侧结果向量
 *
 * 注意：API 文档中 Contains 的 output 元素类型为 bool，此处使用 unsigned char 是因为
 * std::vector<bool> 是标准库特化（按位存储），不支持 data()，无法与 DeviceBuffer 配合。
 * bool 与 unsigned char 均为 1 字节，true/1 与 false/0 的内存布局一致，可安全互换。
 */
template <typename Key>
inline std::vector<unsigned char> ContainsKeys(const BloomFilterT<Key>& filter,
                                               const std::vector<Key>& hostKeys,
                                               aclrtStream stream)
{
  if (hostKeys.empty()) {
    return {};
  }
  aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
  dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

  aclco::test::DeviceBuffer<unsigned char> dResult(hostKeys.size());
  dResult.MemsetZero(stream);

  filter.Contains(static_cast<void*>(dKeys.Data()),
                  static_cast<void*>(dResult.Data()),
                  aclco::Extent<std::size_t>(hostKeys.size()),
                  stream);
  return dResult.CopyToHost(stream);
}

/**
 * 生成 stencil 数组（unsigned char），用于 AddIf / ContainsIf。
 * 偶数索引为 1（处理），奇数索引为 0（跳过）。
 */
inline std::vector<unsigned char> GenerateStencil(std::size_t n)
{
  std::vector<unsigned char> stencil(n);
  for (std::size_t i = 0; i < n; ++i) {
    stencil[i] = static_cast<unsigned char>((i % 2 == 0) ? 1 : 0);
  }
  return stencil;
}

/**
 * 从 hostKeys 中提取 stencil[i]!=0 对应的 key 子集
 */
template <typename Key>
inline std::vector<Key> FilterKeysByStencil(const std::vector<Key>& hostKeys,
                                            const std::vector<unsigned char>& stencil)
{
  std::vector<Key> out;
  out.reserve(hostKeys.size());
  for (std::size_t i = 0; i < hostKeys.size(); ++i) {
    if (stencil[i] != 0) {
      out.push_back(hostKeys[i]);
    }
  }
  return out;
}

/**
 * 条件添加：仅对 stencil[i] != 0 的 key 执行 AddIf
 */
template <typename Key>
inline void AddIfKeys(BloomFilterT<Key>& filter,
                      const std::vector<Key>& hostKeys,
                      const std::vector<unsigned char>& stencil,
                      aclrtStream stream)
{
  if (hostKeys.empty()) {
    return;
  }
  aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
  dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

  aclco::test::DeviceBuffer<unsigned char> dStencil(hostKeys.size());
  dStencil.CopyFromHostAsync(stencil.data(), hostKeys.size(), stream);

  filter.AddIf(static_cast<void*>(dKeys.Data()),
               static_cast<void*>(dStencil.Data()),
               aclco::Extent<std::size_t>(hostKeys.size()),
               stream);
  aclco::test::Sync(stream);
}

/**
 * 条件查询：仅对 stencil[i] != 0 的 key 执行 ContainsIf，返回 host 侧结果向量
 */
template <typename Key>
inline std::vector<unsigned char> ContainsIfKeys(const BloomFilterT<Key>& filter,
                                                 const std::vector<Key>& hostKeys,
                                                 const std::vector<unsigned char>& stencil,
                                                 aclrtStream stream)
{
  if (hostKeys.empty()) {
    return {};
  }
  aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
  dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

  aclco::test::DeviceBuffer<unsigned char> dStencil(hostKeys.size());
  dStencil.CopyFromHostAsync(stencil.data(), hostKeys.size(), stream);

  aclco::test::DeviceBuffer<unsigned char> dResult(hostKeys.size());
  dResult.MemsetZero(stream);

  filter.ContainsIf(static_cast<void*>(dKeys.Data()),
                    static_cast<void*>(dStencil.Data()),
                    static_cast<void*>(dResult.Data()),
                    aclco::Extent<std::size_t>(hostKeys.size()),
                    stream);
  return dResult.CopyToHost(stream);
}

} // namespace aclco::test::bloom_filter_factory
