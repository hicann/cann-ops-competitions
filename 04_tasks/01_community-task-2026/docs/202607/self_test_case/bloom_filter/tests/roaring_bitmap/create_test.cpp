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
#include <fstream>
#include <string>
#include <vector>

#include "tests/common/acl_env.h"
#include "tests/common/roaring_bitmap_factory.h"
#include "tests/common/test_print.h"

TEMPLATE_TEST_CASE(
  "roaring_bitmap create correctness",
  "[roaring_bitmap][create]",
  std::uint32_t,
  std::uint64_t)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  auto numElements = GENERATE(0u, 1u, 128u, 4096u, 65536u);
  std::string params = "numElements=" + std::to_string(numElements);

  // capacity=0 表示不适用，numInsert 传 numElements
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("create correctness", Key, 0, 0, numElements, params);

  auto elements = aclco::test::roaring_bitmap_factory::GenerateElements<Key>(numElements);
  auto serialized = aclco::test::roaring_bitmap_factory::Serialize<Key>(elements);

  SECTION("constructor: Data and SizeBytes consistent") {
    PRINT_SECTION("constructor: Data and SizeBytes consistent");
    auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
    aclco::test::Sync(stream);

    REQUIRE_PRINT(bitmap.Data() != nullptr);
    REQUIRE_PRINT(bitmap.SizeBytes() > 0);
    REQUIRE_PRINT(bitmap.Size() == numElements);
    REQUIRE_PRINT(bitmap.Empty() == (numElements == 0));
  }
}

TEMPLATE_TEST_CASE(
  "roaring_bitmap create edge cases",
  "[roaring_bitmap][create]",
  std::uint32_t,
  std::uint64_t)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("create edge cases", Key, 0, 0, 1, "edge");

  SECTION("single element") {
    PRINT_SECTION("single element");
    auto elements = aclco::test::roaring_bitmap_factory::GenerateElements<Key>(1);
    auto serialized = aclco::test::roaring_bitmap_factory::Serialize<Key>(elements);
    auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
    aclco::test::Sync(stream);

    REQUIRE_PRINT(bitmap.Data() != nullptr);
    REQUIRE_PRINT(bitmap.Size() == 1);
    REQUIRE_PRINT(!bitmap.Empty());
  }

  SECTION("empty bitmap") {
    PRINT_SECTION("empty bitmap");
    std::vector<Key> emptyElements;
    auto serialized = aclco::test::roaring_bitmap_factory::Serialize<Key>(emptyElements);
    auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
    aclco::test::Sync(stream);

    REQUIRE_PRINT(bitmap.Size() == 0);
    REQUIRE_PRINT(bitmap.Empty());
  }

  SECTION("SizeBytes matches serialized size") {
    PRINT_SECTION("SizeBytes matches serialized size");
    auto elements = aclco::test::roaring_bitmap_factory::GenerateElements<Key>(4096);
    auto serialized = aclco::test::roaring_bitmap_factory::Serialize<Key>(elements);
    auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
    aclco::test::Sync(stream);

    REQUIRE_PRINT(bitmap.SizeBytes() == serialized.size());
  }

  SECTION("re-creation from same serialized data consistent") {
    PRINT_SECTION("re-creation from same serialized data consistent");
    auto elements = aclco::test::roaring_bitmap_factory::GenerateElements<Key>(4096);
    auto serialized = aclco::test::roaring_bitmap_factory::Serialize<Key>(elements);

    auto bitmap1 = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
    aclco::test::Sync(stream);
    auto bitmap2 = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
    aclco::test::Sync(stream);

    REQUIRE_PRINT(bitmap1.Size() == bitmap2.Size());
    REQUIRE_PRINT(bitmap1.SizeBytes() == bitmap2.SizeBytes());
    REQUIRE_PRINT(bitmap1.Empty() == bitmap2.Empty());
  }
}

TEMPLATE_TEST_CASE(
  "roaring_bitmap create from testdata binary files",
  "[roaring_bitmap][create][testdata]",
  std::uint32_t,
  std::uint64_t)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("create from testdata", Key, 0, 0, 0, "testdata");

  if constexpr (std::is_same_v<Key, std::uint32_t>) {
    SECTION("bitmapwithruns.bin") {
      PRINT_SECTION("bitmapwithruns.bin");
      std::ifstream file("tests/testdata/bitmapwithruns.bin", std::ios::binary | std::ios::ate);
      REQUIRE_PRINT(file.is_open());
      auto fileSize = static_cast<std::size_t>(file.tellg());
      file.seekg(0, std::ios::beg);
      std::vector<std::byte> serialized(fileSize);
      file.read(reinterpret_cast<char*>(serialized.data()), static_cast<std::streamsize>(fileSize));

      auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
      aclco::test::Sync(stream);
      REQUIRE_PRINT(bitmap.Data() != nullptr);
      REQUIRE_PRINT(bitmap.Size() > 0);
      REQUIRE_PRINT(bitmap.SizeBytes() == fileSize);
      REQUIRE_PRINT(!bitmap.Empty());
    }

    SECTION("bitmapwithoutruns.bin") {
      PRINT_SECTION("bitmapwithoutruns.bin");
      std::ifstream file("tests/testdata/bitmapwithoutruns.bin", std::ios::binary | std::ios::ate);
      REQUIRE_PRINT(file.is_open());
      auto fileSize = static_cast<std::size_t>(file.tellg());
      file.seekg(0, std::ios::beg);
      std::vector<std::byte> serialized(fileSize);
      file.read(reinterpret_cast<char*>(serialized.data()), static_cast<std::streamsize>(fileSize));

      auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
      aclco::test::Sync(stream);
      REQUIRE_PRINT(bitmap.Data() != nullptr);
      REQUIRE_PRINT(bitmap.Size() > 0);
      REQUIRE_PRINT(!bitmap.Empty());
    }
  }

  if constexpr (std::is_same_v<Key, std::uint64_t>) {
    SECTION("portable_bitmap64.bin") {
      PRINT_SECTION("portable_bitmap64.bin");
      std::ifstream file("tests/testdata/portable_bitmap64.bin", std::ios::binary | std::ios::ate);
      REQUIRE_PRINT(file.is_open());
      auto fileSize = static_cast<std::size_t>(file.tellg());
      file.seekg(0, std::ios::beg);
      std::vector<std::byte> serialized(fileSize);
      file.read(reinterpret_cast<char*>(serialized.data()), static_cast<std::streamsize>(fileSize));

      auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
      aclco::test::Sync(stream);
      REQUIRE_PRINT(bitmap.Data() != nullptr);
      REQUIRE_PRINT(bitmap.Size() > 0);
      REQUIRE_PRINT(!bitmap.Empty());
    }
  }
}
