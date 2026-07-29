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

#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_factory.h"
#include "tests/common/test_print.h"

TEMPLATE_TEST_CASE(
  "bloom_filter create correctness",
  "[bloom_filter][create]",
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
  using BF = aclco::test::bloom_filter_factory::BloomFilterT<Key>;

  auto numBlocks = GENERATE(1u, 10u, 100u, 1000u, 10000u);
  std::string params = "numBlocks=" + std::to_string(numBlocks);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("create correctness", Key, 0, numBlocks, 0, params);

  SECTION("constructor initializes and BlockExtent matches") {
    PRINT_SECTION("constructor initializes and BlockExtent matches");
    BF filter(aclco::Extent<std::size_t>(numBlocks),
              aclco::test::bloom_filter_factory::DefaultPolicy<Key>(),
              stream);
    aclco::test::Sync(stream);

    REQUIRE_PRINT(static_cast<std::size_t>(filter.BlockExtent()) == numBlocks);
  }

  SECTION("default policy constructor") {
    PRINT_SECTION("default policy constructor");
    BF filter(aclco::Extent<std::size_t>(numBlocks),
              aclco::test::bloom_filter_factory::DefaultPolicy<Key>(),
              stream);
    aclco::test::Sync(stream);

    REQUIRE_PRINT(static_cast<std::size_t>(filter.BlockExtent()) == numBlocks);
    // Data 指针非空
    REQUIRE_PRINT(filter.Data() != nullptr);
  }

  SECTION("factory MakeBloomFilter") {
    PRINT_SECTION("factory MakeBloomFilter");
    auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
    aclco::test::Sync(stream);

    REQUIRE_PRINT(static_cast<std::size_t>(filter.BlockExtent()) == numBlocks);
    REQUIRE_PRINT(filter.Data() != nullptr);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter create edge cases",
  "[bloom_filter][create]",
  uint32_t,
  int64_t,
  float)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  std::string params_edge = "numBlocks=1";
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("create edge cases", Key, 0, 1, 0, params_edge);

  SECTION("single block") {
    PRINT_SECTION("single block");
    auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(1, stream);
    aclco::test::Sync(stream);
    REQUIRE_PRINT(static_cast<std::size_t>(filter.BlockExtent()) == 1u);
    REQUIRE_PRINT(filter.Data() != nullptr);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter create with ArrowFilterPolicy",
  "[bloom_filter][create]",
  uint32_t,
  int64_t,
  float)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;
  using ArrowPolicy = aclco::ArrowFilterPolicy<Key>;
  using ArrowBF = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, ArrowPolicy>;

  auto numBlocks = 100u;
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("create ArrowFilterPolicy", Key, 0, numBlocks, 0, "ArrowPolicy");

  ArrowBF filter(aclco::Extent<std::size_t>(numBlocks), ArrowPolicy(), stream);
  aclco::test::Sync(stream);

  REQUIRE_PRINT(static_cast<std::size_t>(filter.BlockExtent()) == numBlocks);
  REQUIRE_PRINT(filter.Data() != nullptr);
}
