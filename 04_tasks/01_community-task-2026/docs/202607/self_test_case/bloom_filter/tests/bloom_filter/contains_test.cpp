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
  "bloom_filter Contains TPR correctness (sync)",
  "[bloom_filter][contains]",
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

  auto numKeys = GENERATE(1u, 4u, 128u, 1024u, 8192u, 65536u);
  auto pattern = GENERATE(as<std::string>{},
                          "sequential", "random", "duplicates",
                          "zero", "sparse", "min", "max");
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys) +
                       ", pattern=" + pattern +
                       ", numBlocks=" + std::to_string(numBlocks);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Contains TPR correctness", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, pattern);
  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);
  aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);

  SECTION("all added keys must be found (TPR=1.0)") {
    PRINT_SECTION("all added keys must be found (TPR=1.0)");
    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, hostKeys, stream);

    std::unordered_set<Key> addedSet(hostKeys.begin(), hostKeys.end());
    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(hostKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " numKeys=" + std::to_string(numKeys) +
               " pattern=" + pattern);
    REQUIRE_PRINT(tpr == 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter Contains FPR reference",
  "[bloom_filter][contains]",
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

  auto numKeys = GENERATE(128u, 4096u, 65536u);
  auto pattern = GENERATE(as<std::string>{}, "sequential", "random");
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys) +
                       ", pattern=" + pattern;
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Contains FPR reference", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, pattern);
  auto absentKeys = aclco::test::bloom_filter_factory::GenerateAbsentKeys<Key>(numKeys, hostKeys);

  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);
  aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);

  SECTION("absent keys FPR recorded") {
    PRINT_SECTION("absent keys FPR recorded");
    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, absentKeys, stream);

    std::unordered_set<Key> addedSet(hostKeys.begin(), hostKeys.end());
    double fpr = aclco::test::bloom_filter_factory::ComputeFPR<Key>(absentKeys, results, addedSet);
    PRINT_INFO("FPR=" + std::to_string(fpr) + " numKeys=" + std::to_string(numKeys) +
               " pattern=" + pattern + " numBlocks=" + std::to_string(numBlocks));

    REQUIRE_PRINT(fpr < 1.0);
  }

  SECTION("mixed queries TPR=1.0 and FPR<1.0") {
    PRINT_SECTION("mixed queries TPR=1.0 and FPR<1.0");
    // 混合查询：前半 added，后半 absent
    std::vector<Key> mixed;
    mixed.reserve(hostKeys.size() + absentKeys.size());
    mixed.insert(mixed.end(), hostKeys.begin(), hostKeys.end());
    mixed.insert(mixed.end(), absentKeys.begin(), absentKeys.end());

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, mixed, stream);

    std::unordered_set<Key> addedSet(hostKeys.begin(), hostKeys.end());
    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(mixed, results, addedSet);
    double fpr = aclco::test::bloom_filter_factory::ComputeFPR<Key>(mixed, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " FPR=" + std::to_string(fpr));
    REQUIRE_PRINT(tpr == 1.0);
    REQUIRE_PRINT(fpr < 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter ContainsAsync correctness",
  "[bloom_filter][containsAsync]",
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

  auto numKeys = GENERATE(1u, 128u, 4096u);
  auto pattern = GENERATE(as<std::string>{}, "sequential", "random");
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys) +
                       ", pattern=" + pattern;
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("ContainsAsync correctness", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, pattern);
  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);
  aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);

  SECTION("containsAsync TPR=1.0") {
    PRINT_SECTION("containsAsync TPR=1.0");
    aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
    dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);
    aclco::test::DeviceBuffer<unsigned char> dResult(hostKeys.size());
    dResult.MemsetZero(stream);

    // ContainsAsync 异步启动，D2H 拷贝排队在同一流上等待 kernel 完成
    filter.ContainsAsync(static_cast<void*>(dKeys.Data()),
                         static_cast<void*>(dResult.Data()),
                         aclco::Extent<std::size_t>(hostKeys.size()), stream);
    auto results = dResult.CopyToHost(stream);

    std::unordered_set<Key> addedSet(hostKeys.begin(), hostKeys.end());
    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(hostKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (async, must be 1.0)");
    REQUIRE_PRINT(tpr == 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter Contains empty filter",
  "[bloom_filter][contains]",
  uint32_t,
  int64_t,
  float)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  std::size_t numKeys = 128u;
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Contains empty filter", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, "sequential");

  SECTION("query on empty filter returns all false") {
    PRINT_SECTION("query on empty filter returns all false");
    auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
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
  }
}
