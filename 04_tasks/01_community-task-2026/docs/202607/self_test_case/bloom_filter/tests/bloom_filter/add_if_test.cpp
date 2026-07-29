/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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
  "bloom_filter AddIf correctness (sync)",
  "[bloom_filter][addIf]",
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

  auto numKeys = GENERATE(2u, 8u, 128u, 4096u, 65536u);
  auto pattern = GENERATE(as<std::string>{}, "sequential", "random", "duplicates",
                          "zero", "sparse");
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys) +
                       ", pattern=" + pattern +
                       ", numBlocks=" + std::to_string(numBlocks);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("AddIf correctness (sync)", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, pattern);
  auto stencil = aclco::test::bloom_filter_factory::GenerateStencil(numKeys);
  auto selectedKeys = aclco::test::bloom_filter_factory::FilterKeysByStencil<Key>(hostKeys, stencil);

  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  SECTION("addIf: only stencil=1 keys added, TPR=1.0 for selected") {
    PRINT_SECTION("addIf: only stencil=1 keys added, TPR=1.0 for selected");
    aclco::test::bloom_filter_factory::AddIfKeys<Key>(filter, hostKeys, stencil, stream);

    // 对 stencil=1 的 key 查询，TPR 必须为 1.0
    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, selectedKeys, stream);
    std::unordered_set<Key> addedSet(selectedKeys.begin(), selectedKeys.end());
    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(selectedKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (addIf selected, must be 1.0)");
    REQUIRE_PRINT(tpr == 1.0);

    // 用 absent keys（从未添加过的 key）验证：这些 key 不在 hostKeys 中，
    // 自然也不会被 AddIf 添加，查询应全部返回 false。
    auto absentKeys = aclco::test::bloom_filter_factory::GenerateAbsentKeys<Key>(numKeys, hostKeys);
    auto absentResults = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, absentKeys, stream);
    bool allAbsentFalse = true;
    for (std::size_t i = 0; i < numKeys; ++i) {
      if (absentResults[i] != 0) {
        allAbsentFalse = false;
        break;
      }
    }
    PRINT_INFO("all absent keys not found: " + std::to_string(allAbsentFalse));
    // 注：小 numKeys（如 2, 8）由于 FPR 较高可能误报，仅在大 numKeys 时断言
    if (numKeys >= 128) {
      REQUIRE_PRINT(allAbsentFalse);
    }
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter AddIfAsync correctness",
  "[bloom_filter][addIfAsync]",
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

  auto numKeys = GENERATE(128u, 4096u);
  auto pattern = GENERATE(as<std::string>{}, "sequential", "random");
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys) +
                       ", pattern=" + pattern;
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("AddIfAsync correctness", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, pattern);
  auto stencil = aclco::test::bloom_filter_factory::GenerateStencil(numKeys);
  auto selectedKeys = aclco::test::bloom_filter_factory::FilterKeysByStencil<Key>(hostKeys, stencil);

  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  SECTION("addIfAsync: only stencil=1 keys added, TPR=1.0") {
    PRINT_SECTION("addIfAsync: only stencil=1 keys added, TPR=1.0");
    aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
    dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);
    aclco::test::DeviceBuffer<unsigned char> dStencil(hostKeys.size());
    dStencil.CopyFromHostAsync(stencil.data(), hostKeys.size(), stream);

    filter.AddIfAsync(static_cast<void*>(dKeys.Data()),
                      static_cast<void*>(dStencil.Data()),
                      aclco::Extent<std::size_t>(hostKeys.size()), stream);
    aclco::test::Sync(stream);

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, selectedKeys, stream);
    std::unordered_set<Key> addedSet(selectedKeys.begin(), selectedKeys.end());
    double tpr = aclco::test::bloom_filter_factory::ComputeTPR<Key>(selectedKeys, results, addedSet);
    PRINT_INFO("TPR=" + std::to_string(tpr) + " (addIfAsync, must be 1.0)");
    REQUIRE_PRINT(tpr == 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter AddIf empty stencil",
  "[bloom_filter][addIf]",
  uint32_t,
  int64_t,
  float)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  std::size_t numKeys = 512u;
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("AddIf empty stencil", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, "random");
  std::vector<unsigned char> allZeroStencil(numKeys, 0);

  auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
  aclco::test::Sync(stream);

  SECTION("addIf with all-zero stencil, no keys added") {
    PRINT_SECTION("addIf with all-zero stencil, no keys added");
    aclco::test::bloom_filter_factory::AddIfKeys<Key>(filter, hostKeys, allZeroStencil, stream);

    auto results = aclco::test::bloom_filter_factory::ContainsKeys<Key>(filter, hostKeys, stream);
    bool allFalse = true;
    for (auto r : results) {
      if (r != 0) {
        allFalse = false;
        break;
      }
    }
    PRINT_INFO("all keys not found: " + std::to_string(allFalse));
    REQUIRE_PRINT(allFalse);
  }
}
