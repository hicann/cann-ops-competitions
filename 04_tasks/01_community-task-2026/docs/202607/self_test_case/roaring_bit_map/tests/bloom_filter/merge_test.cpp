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

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_factory.h"
#include "tests/common/device_buffer.h"
#include "tests/common/test_print.h"

TEMPLATE_TEST_CASE(
  "bloom_filter Merge correctness (sync)",
  "[bloom_filter][merge]",
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

  auto numKeysA = GENERATE(1u, 128u, 4096u);
  auto numKeysB = GENERATE(1u, 128u, 4096u);
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeysA + numKeysB);

  std::string params = "numKeysA=" + std::to_string(numKeysA) +
                       ", numKeysB=" + std::to_string(numKeysB) +
                       ", numBlocks=" + std::to_string(numBlocks);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Merge correctness (sync)", Key, 0, numBlocks,
                                     numKeysA + numKeysB, params);

  auto keysA = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeysA, "random", 1u);
  auto keysB = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeysB, "random", 2u);

  auto filterA = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  auto filterB = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  aclco::test::bloom_filter_factory::AddKeys<Key>(filterA, keysA, stream);
  aclco::test::bloom_filter_factory::AddKeys<Key>(filterB, keysB, stream);

  SECTION("merge: union TPR must be 1.0") {
    PRINT_SECTION("merge: union TPR must be 1.0");
    filterA.Merge(filterB, stream);
    aclco::test::Sync(stream);

    // 合并后查询 A ∪ B 中所有 key，TPR 必须为 1.0
    std::vector<Key> unionKeys;
    unionKeys.reserve(keysA.size() + keysB.size());
    unionKeys.insert(unionKeys.end(), keysA.begin(), keysA.end());
    unionKeys.insert(unionKeys.end(), keysB.begin(), keysB.end());

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filterA, unionKeys, stream);
    std::unordered_set<Key> addedSet(unionKeys.begin(), unionKeys.end());

    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(unionKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (merge union, must be 1.0)");
    REQUIRE_PRINT(tpr == 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter MergeAsync correctness",
  "[bloom_filter][mergeAsync]",
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

  std::size_t numKeysA = 1024u;
  std::size_t numKeysB = 1024u;
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeysA + numKeysB);

  std::string params = "numKeysA=" + std::to_string(numKeysA) +
                       ", numKeysB=" + std::to_string(numKeysB);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("MergeAsync correctness", Key, 0, numBlocks,
                                     numKeysA + numKeysB, params);

  auto keysA = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeysA, "random", 1u);
  auto keysB = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeysB, "random", 2u);

  auto filterA = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  auto filterB = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  aclco::test::bloom_filter_factory::AddKeys<Key>(filterA, keysA, stream);
  aclco::test::bloom_filter_factory::AddKeys<Key>(filterB, keysB, stream);

  SECTION("mergeAsync: union TPR must be 1.0") {
    PRINT_SECTION("mergeAsync: union TPR must be 1.0");
    filterA.MergeAsync(filterB, stream);
    aclco::test::Sync(stream);

    std::vector<Key> unionKeys;
    unionKeys.insert(unionKeys.end(), keysA.begin(), keysA.end());
    unionKeys.insert(unionKeys.end(), keysB.begin(), keysB.end());

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filterA, unionKeys, stream);
    std::unordered_set<Key> addedSet(unionKeys.begin(), unionKeys.end());

    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(unionKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (mergeAsync union, must be 1.0)");
    REQUIRE_PRINT(tpr == 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter Merge with empty filter",
  "[bloom_filter][merge]",
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
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Merge with empty filter", Key, 0, numBlocks, numKeys, params);

  auto keys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, "random", 1u);

  auto filterA = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  auto filterB = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  aclco::test::bloom_filter_factory::AddKeys<Key>(filterA, keys, stream);

  SECTION("merge empty into non-empty, TPR stays 1.0") {
    PRINT_SECTION("merge empty into non-empty, TPR stays 1.0");
    filterA.Merge(filterB, stream);
    aclco::test::Sync(stream);

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filterA, keys, stream);
    std::unordered_set<Key> addedSet(keys.begin(), keys.end());

    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(keys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (merge empty, must be 1.0)");
    REQUIRE_PRINT(tpr == 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter Merge mismatched numBlocks",
  "[bloom_filter][merge]",
  uint32_t,
  int64_t,
  float)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Merge mismatched numBlocks", Key, 0, 0, 0, "mismatched");

  auto filterA = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(100, stream);
  auto filterB = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(200, stream);
  aclco::test::Sync(stream);

  SECTION("merge different numBlocks throws") {
    PRINT_SECTION("merge different numBlocks throws");
    REQUIRE_THROWS(filterA.Merge(filterB, stream));
  }
}
