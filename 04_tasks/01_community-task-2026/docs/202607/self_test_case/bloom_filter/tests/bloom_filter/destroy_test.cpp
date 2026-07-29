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
#include <memory>
#include <string>

#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_factory.h"
#include "tests/common/test_print.h"

TEMPLATE_TEST_CASE(
  "bloom_filter destroy correctness",
  "[bloom_filter][destroy]",
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

  auto numBlocks = GENERATE(10u, 100u, 1000u);
  std::string params = "numBlocks=" + std::to_string(numBlocks);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("destroy correctness", Key, 0, numBlocks, 0, params);

  SECTION("construct empty then destroy, no crash") {
    PRINT_SECTION("construct empty then destroy, no crash");
    {
      auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
      aclco::test::Sync(stream);
      REQUIRE_PRINT(filter.Data() != nullptr);
      REQUIRE_PRINT(static_cast<std::size_t>(filter.BlockExtent()) == numBlocks);
    }
    aclco::test::Sync(stream);
    REQUIRE_PRINT(true);
  }
}

TEMPLATE_TEST_CASE(
  "bloom_filter destroy after add",
  "[bloom_filter][destroy]",
  uint32_t,
  int64_t,
  float)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  std::size_t numKeys = 65536u;
  auto numBlocks = aclco::test::bloom_filter_factory::RecommendNumBlocks(numKeys);

  std::string params = "numKeys=" + std::to_string(numKeys) +
                       ", numBlocks=" + std::to_string(numBlocks);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("destroy after add", Key, 0, numBlocks, numKeys, params);

  auto hostKeys = aclco::test::bloom_filter_factory::GenerateKeys<Key>(numKeys, "random");

  SECTION("add then destroy, no crash") {
    PRINT_SECTION("add then destroy, no crash");
    {
      auto filter = aclco::test::bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, stream);
      aclco::test::Sync(stream);
      aclco::test::bloom_filter_factory::AddKeys<Key>(filter, hostKeys, stream);
    }
    aclco::test::Sync(stream);
    REQUIRE_PRINT(true);
  }
}
