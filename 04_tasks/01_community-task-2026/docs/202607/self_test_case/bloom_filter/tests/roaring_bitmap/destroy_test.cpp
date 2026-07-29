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
#include <vector>

#include "tests/common/acl_env.h"
#include "tests/common/roaring_bitmap_factory.h"
#include "tests/common/test_print.h"

TEMPLATE_TEST_CASE(
  "roaring_bitmap destroy correctness",
  "[roaring_bitmap][destroy]",
  std::uint32_t,
  std::uint64_t)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  auto numElements = GENERATE(0u, 128u, 4096u);
  std::string params = "numElements=" + std::to_string(numElements);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("destroy correctness", Key, 0, 0, numElements, params);

  auto elements = aclco::test::roaring_bitmap_factory::GenerateElements<Key>(numElements);
  auto serialized = aclco::test::roaring_bitmap_factory::Serialize<Key>(elements);

  SECTION("construct then destroy, no crash") {
    PRINT_SECTION("construct then destroy, no crash");
    {
      auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
      aclco::test::Sync(stream);
      REQUIRE_PRINT(bitmap.Data() != nullptr);
    }
    aclco::test::Sync(stream);
    REQUIRE_PRINT(true);
  }
}

TEMPLATE_TEST_CASE(
  "roaring_bitmap destroy after query",
  "[roaring_bitmap][destroy]",
  std::uint32_t,
  std::uint64_t)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  std::size_t numElements = 65536u;
  std::string params = "numElements=" + std::to_string(numElements);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("destroy after query", Key, 0, 0, numElements, params);

  auto elements = aclco::test::roaring_bitmap_factory::GenerateElements<Key>(numElements);
  auto serialized = aclco::test::roaring_bitmap_factory::Serialize<Key>(elements);

  auto queryKeys = aclco::test::roaring_bitmap_factory::GenerateQueryKeys<Key>(
      1024u, "in-set", elements);

  SECTION("query then destroy, no crash") {
    PRINT_SECTION("query then destroy, no crash");
    {
      auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
      aclco::test::Sync(stream);
      auto results = aclco::test::roaring_bitmap_factory::ContainsKeys<Key>(bitmap, queryKeys, stream);
      REQUIRE_PRINT(!results.empty());
    }
    aclco::test::Sync(stream);
    REQUIRE_PRINT(true);
  }
}
