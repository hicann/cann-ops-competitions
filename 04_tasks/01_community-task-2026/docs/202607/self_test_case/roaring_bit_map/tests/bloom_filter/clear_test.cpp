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

#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_factory.h"
#include "tests/common/test_print.h"

TEMPLATE_TEST_CASE(
  "bloom_filter Clear correctness (sync)",
  "[bloom_filter][clear]",
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

  auto numKeys = GENERATE(1u, 128u, 4096u, 65536u);
  auto pattern = GENERATE(as<std::string>{}, "sequential", "random", "duplicates");
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys) +
                       ", pattern=" + pattern;
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Clear correctness (sync)", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, pattern);
  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);
  aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);

  SECTION("clear then all queries return false") {
    PRINT_SECTION("clear then all queries return false");
    filter.Clear(stream);
    aclco::test::Sync(stream);

    // 容量保持不变
    REQUIRE_PRINT(static_cast<std::size_t>(filter.BlockExtent()) == numBlocks);

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, hostKeys, stream);
    bool allFalse = true;
    for (auto r : results) {
      if (r != 0) {
        allFalse = false;
        break;
      }
    }
    PRINT_INFO("after Clear, all queries should return false");
    REQUIRE_PRINT(allFalse);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter ClearAsync correctness",
  "[bloom_filter][clearAsync]",
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
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("ClearAsync correctness", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, "random");
  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);
  aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);

  SECTION("clearAsync then all queries return false") {
    PRINT_SECTION("clearAsync then all queries return false");
    filter.ClearAsync(stream);
    aclco::test::Sync(stream);

    REQUIRE_PRINT(static_cast<std::size_t>(filter.BlockExtent()) == numBlocks);

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, hostKeys, stream);
    bool allFalse = true;
    for (auto r : results) {
      if (r != 0) {
        allFalse = false;
        break;
      }
    }
    PRINT_INFO("after ClearAsync, all queries should return false");
    REQUIRE_PRINT(allFalse);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter Clear multiple times",
  "[bloom_filter][clear]",
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
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Clear multiple times", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, "random");
  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  SECTION("clear without add, then add, then clear") {
    PRINT_SECTION("clear without add, then add, then clear");
    // 直接 clear 空过滤器
    filter.Clear(stream);
    aclco::test::Sync(stream);
    REQUIRE_PRINT(static_cast<std::size_t>(filter.BlockExtent()) == numBlocks);

    // 添加后再 clear
    aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);
    filter.Clear(stream);
    aclco::test::Sync(stream);

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, hostKeys, stream);
    bool allFalse = true;
    for (auto r : results) {
      if (r != 0) {
        allFalse = false;
        break;
      }
    }
    REQUIRE_PRINT(allFalse);

    // 再 clear 一次
    filter.Clear(stream);
    aclco::test::Sync(stream);
    REQUIRE_PRINT(static_cast<std::size_t>(filter.BlockExtent()) == numBlocks);
  }

  SECTION("clear then re-add, TPR=1.0") {
    PRINT_SECTION("clear then re-add, TPR=1.0");
    aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);
    filter.Clear(stream);
    aclco::test::Sync(stream);

    // 重新添加后 TPR 应为 1.0
    aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);
    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, hostKeys, stream);
    std::unordered_set<Key> addedSet(hostKeys.begin(), hostKeys.end());
    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(hostKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (after clear and re-add, must be 1.0)");
    REQUIRE_PRINT(tpr == 1.0);
  }
}
