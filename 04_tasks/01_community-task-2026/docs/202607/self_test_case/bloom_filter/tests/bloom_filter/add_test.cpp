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
  "bloom_filter Add correctness (sync)",
  "[bloom_filter][add]",
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

  auto numKeys = GENERATE(1u, 8u, 128u, 4096u, 65536u);
  auto pattern = GENERATE(as<std::string>{},
                          "sequential", "random", "duplicates",
                          "zero", "sparse", "min", "max");
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys) +
                       ", pattern=" + pattern +
                       ", numBlocks=" + std::to_string(numBlocks);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Add correctness (sync)", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, pattern);
  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  SECTION("add then TPR must be 1.0") {
    PRINT_SECTION("add then TPR must be 1.0");
    aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, hostKeys, stream);

    std::unordered_set<Key> addedSet(hostKeys.begin(), hostKeys.end());
    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(hostKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (must be 1.0, no false negative)");
    REQUIRE_PRINT(tpr == 1.0);
  }

  SECTION("add empty keys") {
    PRINT_SECTION("add empty keys");
    std::vector<Key> emptyKeys;
    aclco::test::bloom_filter_factory::AddKeys<Key>(filter, emptyKeys, stream);
    // 空添加不应崩溃，BlockExtent 保持不变
    REQUIRE_PRINT(static_cast<std::size_t>(filter.BlockExtent()) == numBlocks);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter AddAsync correctness",
  "[bloom_filter][addAsync]",
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

  auto numKeys = GENERATE(1u, 1024u, 8192u);
  auto pattern = GENERATE(as<std::string>{}, "sequential", "random", "duplicates");
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys) +
                       ", pattern=" + pattern +
                       ", numBlocks=" + std::to_string(numBlocks);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("AddAsync correctness", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, pattern);
  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  SECTION("addAsync then TPR must be 1.0") {
    PRINT_SECTION("addAsync then TPR must be 1.0");
    aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
    dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);
    filter.AddAsync(static_cast<void*>(dKeys.Data()),
                    aclco::Extent<std::size_t>(hostKeys.size()), stream);
    aclco::test::Sync(stream);

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, hostKeys, stream);

    std::unordered_set<Key> addedSet(hostKeys.begin(), hostKeys.end());
    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(hostKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (async, must be 1.0)");
    REQUIRE_PRINT(tpr == 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter Add repeated (idempotent)",
  "[bloom_filter][add]",
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
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Add repeated idempotent", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, "random");
  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  SECTION("add same keys twice, TPR still 1.0") {
    PRINT_SECTION("add same keys twice, TPR still 1.0");
    aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);
    aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, hostKeys, stream);
    std::unordered_set<Key> addedSet(hostKeys.begin(), hostKeys.end());
    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(hostKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (must be 1.0 after repeated add)");
    REQUIRE_PRINT(tpr == 1.0);
  }
}
