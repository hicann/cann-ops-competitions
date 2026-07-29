/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_factory.h"
#include "tests/common/test_print.h"

TEMPLATE_TEST_CASE(
  "bloom_filter Intersect correctness (sync)",
  "[bloom_filter][intersect]",
  int32_t,
  uint32_t,
  int64_t,
  uint64_t,
  float)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  std::size_t numKeys = 4096u;
  // 构造 A、B 使其存在交集：A = [0, numKeys), B = [numKeys/2, numKeys + numKeys/2)
  // 交集 = [numKeys/2, numKeys)，共 numKeys/2 个 key
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Intersect correctness (sync)", Key, 0, numBlocks, numKeys, params);

  std::vector<Key> keysA;
  std::vector<Key> keysB;
  keysA.reserve(numKeys);
  keysB.reserve(numKeys);
  for (std::size_t i = 0; i < numKeys; ++i) {
    keysA.push_back(static_cast<Key>(i));
  }
  for (std::size_t i = 0; i < numKeys; ++i) {
    keysB.push_back(static_cast<Key>(numKeys / 2 + i));
  }

  // 计算精确交集
  std::vector<Key> intersectionKeys;
  std::unordered_set<Key> setA(keysA.begin(), keysA.end());
  for (auto k : keysB) {
    if (setA.find(k) != setA.end()) {
      intersectionKeys.push_back(k);
    }
  }

  auto filterA = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  auto filterB = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  aclco::test::bloom_filter_factory::AddKeys<Key>(filterA, keysA, stream);
  aclco::test::bloom_filter_factory::AddKeys<Key>(filterB, keysB, stream);

  SECTION("intersect: true intersection TPR must be 1.0 (no false negative)") {
    PRINT_SECTION("intersect: true intersection TPR must be 1.0");
    filterA.Intersect(filterB, stream);
    aclco::test::Sync(stream);

    // 对真实交集 key 查询，TPR 必须为 1.0（无假阴性保证）
    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filterA, intersectionKeys, stream);
    std::unordered_set<Key> addedSet(intersectionKeys.begin(), intersectionKeys.end());

    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(intersectionKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (intersect, no false negative for A∩B, must be 1.0)");
    REQUIRE_PRINT(tpr == 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter IntersectAsync correctness",
  "[bloom_filter][intersectAsync]",
  int32_t,
  uint32_t,
  int64_t,
  uint64_t,
  float)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  std::size_t numKeys = 2048u;
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("IntersectAsync correctness", Key, 0, numBlocks, numKeys, params);

  std::vector<Key> keysA;
  std::vector<Key> keysB;
  for (std::size_t i = 0; i < numKeys; ++i) {
    keysA.push_back(static_cast<Key>(i));
  }
  for (std::size_t i = 0; i < numKeys; ++i) {
    keysB.push_back(static_cast<Key>(numKeys / 2 + i));
  }

  std::vector<Key> intersectionKeys;
  std::unordered_set<Key> setA(keysA.begin(), keysA.end());
  for (auto k : keysB) {
    if (setA.find(k) != setA.end()) {
      intersectionKeys.push_back(k);
    }
  }

  auto filterA = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  auto filterB = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  aclco::test::bloom_filter_factory::AddKeys<Key>(filterA, keysA, stream);
  aclco::test::bloom_filter_factory::AddKeys<Key>(filterB, keysB, stream);

  SECTION("intersectAsync: true intersection TPR must be 1.0") {
    PRINT_SECTION("intersectAsync: true intersection TPR must be 1.0");
    filterA.IntersectAsync(filterB, stream);
    aclco::test::Sync(stream);

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filterA, intersectionKeys, stream);
    std::unordered_set<Key> addedSet(intersectionKeys.begin(), intersectionKeys.end());

    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(intersectionKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (intersectAsync, must be 1.0)");
    REQUIRE_PRINT(tpr == 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter Intersect with empty filter",
  "[bloom_filter][intersect]",
  uint32_t,
  int64_t,
  float)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  std::size_t numKeys = 1024u;
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Intersect with empty filter", Key, 0, numBlocks, numKeys, params);

  auto keys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, "sequential");

  auto filterA = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  auto filterB = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  aclco::test::bloom_filter_factory::AddKeys<Key>(filterA, keys, stream);

  SECTION("intersect with empty: A∩∅ = ∅, capacity preserved") {
    PRINT_SECTION("intersect with empty: A∩∅ = ∅, capacity preserved");
    filterA.Intersect(filterB, stream);
    aclco::test::Sync(stream);

    // 容量保持不变
    REQUIRE_PRINT(static_cast<std::size_t>(filterA.BlockExtent()) == numBlocks);

    // 空过滤器交集后，所有原 key 查询应返回 false（A∩∅ = ∅）
    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filterA, keys, stream);
    bool allFalse = true;
    for (auto r : results) {
      if (r != 0) {
        allFalse = false;
        break;
      }
    }
    PRINT_INFO("intersect with empty should yield empty result");
    REQUIRE_PRINT(allFalse);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter Intersect mismatched numBlocks",
  "[bloom_filter][intersect]",
  uint32_t,
  int64_t,
  float)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Intersect mismatched numBlocks", Key, 0, 0, 0, "mismatched");

  auto filterA = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(100, stream);
  auto filterB = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(200, stream);
  aclco::test::Sync(stream);

  SECTION("intersect different numBlocks throws") {
    PRINT_SECTION("intersect different numBlocks throws");
    REQUIRE_THROWS(filterA.Intersect(filterB, stream));
  }
}
